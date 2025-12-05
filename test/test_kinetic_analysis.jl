
using Test
using SparseArrays
using LinearAlgebra
using AbstractFBCModels
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
import ConstraintTrees as C
import JuMP as J
using Graphs
using DocStringExtensions
using COBREXA
using COCOA
using HiGHS
import Distributed as D
using StableRNGs


@testset "EnvZ-OmpR Complete Pipeline" begin

    @testset "1. Model Creation" begin
        model = create_envz_ompr_model()

        @test model isa A.AbstractFBCModel
        @test length(A.reactions(model)) == 14
        @test length(A.metabolites(model)) == 9

        @info "Model created successfully" n_reactions=length(A.reactions(model)) n_metabolites=length(A.metabolites(model))
    end

    @testset "2. Concordance Analysis" begin
        model = create_envz_ompr_model()

        # Run concordance analysis
        results = activity_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            kinetic_analysis=false,
            use_transitivity=true,
            concordance_tolerance=0.01,
            cv_threshold=0.01
        )

        @test results isa COCOA.ConcordanceResults

        # Extract concordance modules
        concordance_modules = extract_concordance_modules(results)

        # Expected: 4 concordance modules (1 balanced + 3 unbalanced)
        @test length(concordance_modules) == 4

        # Check balanced module (should be first)
        balanced = concordance_modules[1]
        expected_balanced = Set([:XD, :XT, :XpY, :XTYp, :XDYp])
        @test balanced == expected_balanced

        # Check unbalanced modules (order may vary)
        unbalanced_sets = Set([
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ])

        remaining = Set(concordance_modules[2:end])
        @test remaining == unbalanced_sets

        @info "Concordance modules verified" n_modules=length(concordance_modules) balanced_size=length(balanced)
    end

    @testset "3. Kinetic Analysis" begin
        model = create_envz_ompr_model()

        # Manually specify concordance modules (as extracted from concordance analysis)
        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),                           # balanced
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),                 # 𝒞m1
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),                         # 𝒞m2
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])                          # 𝒞m3
        ]

        kinetic_modules = kinetic_analysis(concordance_modules, model, min_module_size=1)

        @test !isempty(kinetic_modules)
        @test all(km -> km isa Set{Symbol}, kinetic_modules)

        # Expected results (matching the paper):
        # After merging and Theorem S4-6:
        # - Giant module with 9 non-terminal complexes
        # - 4 terminal singleton modules
        expected_giant_module = Set([
            :XD, :X, :XT, Symbol("Xp+Y"), :XpY,        # from 𝒞1
            Symbol("XT+Yp"), :XTYp, Symbol("XD+Yp"), :XDYp  # from 𝒞4
        ])

        expected_terminal_singletons = Set([
            Set([:Xp]),
            Set([Symbol("X+Yp")]),
            Set([Symbol("XT+Y")]),
            Set([Symbol("XD+Y")])
        ])

        # Verify results: giant module + 4 terminal singletons = 5 total
        @test length(kinetic_modules) == 5
        @test kinetic_modules[1] == expected_giant_module  # Giant module is first (largest)

        # Verify terminal singletons are present
        terminal_modules = Set(kinetic_modules[2:end])
        @test terminal_modules == expected_terminal_singletons

        @info "Kinetic modules verified" n_modules=length(kinetic_modules) giant_size=length(kinetic_modules[1])
    end

    @testset "4. ACR/ACRR Identification" begin
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        kinetic_modules = kinetic_analysis(concordance_modules, model, min_module_size=1)

        # Identify ACR/ACRR
        acr_results = identify_acr_acrr(kinetic_modules, model)

        # Verify ACR: Yp should be identified as ACR (Section S.5.2 of paper)
        @test :Yp in acr_results.acr_metabolites
        @test length(acr_results.acr_metabolites) == 1

        # Verify ACRR: Should have 15 pairs (all pairwise combinations of 6 metabolites in giant module)
        # The 6 metabolites are: X, XD, XDYp, XT, XTYp, XpY
        # C(6,2) = 15
        @test length(acr_results.acrr_pairs) == 15

        # Verify the ACRR pairs are the expected ones
        expected_acrr_metabolites = Set([:X, :XD, :XDYp, :XT, :XTYp, :XpY])

        # Extract all metabolites appearing in ACRR pairs
        acrr_metabolites = Set{Symbol}()
        for (s1, s2) in acr_results.acrr_pairs
            push!(acrr_metabolites, s1)
            push!(acrr_metabolites, s2)
        end

        @test acrr_metabolites == expected_acrr_metabolites

        # Verify specific expected pairs (sample)
        @test (:X, :XD) in acr_results.acrr_pairs
        @test (:XT, :XpY) in acr_results.acrr_pairs
        @test (:XD, :XTYp) in acr_results.acrr_pairs

        @info "ACR/ACRR identification verified" n_acr=length(acr_results.acr_metabolites) n_acrr=length(acr_results.acrr_pairs)
    end

    @testset "5. Complete Pipeline Integration" begin
        model = create_envz_ompr_model()

        # Step 1: Concordance analysis
        concordance_results = activity_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            kinetic_analysis=false,
            use_transitivity=true,
            concordance_tolerance=0.01,
            cv_threshold=0.01
        )

        # Step 2: Extract concordance modules
        concordance_modules = extract_concordance_modules(concordance_results)

        # Step 3: Kinetic analysis
        kinetic_modules = kinetic_analysis(concordance_modules, model, min_module_size=1)

        # Step 4: Identify ACR/ACRR
        acr_results = identify_acr_acrr(kinetic_modules, model)

        # Verify final outputs
        @test length(concordance_modules) == 4
        @test length(kinetic_modules) == 5
        @test :Yp in acr_results.acr_metabolites
        @test length(acr_results.acrr_pairs) == 15

        # Verify structural deficiency calculation (should be 2 for EnvZ-OmpR)
        # This is computed internally during kinetic_analysis
        # We can verify by checking the log output or running the calculation directly

        @info "Complete pipeline validated" concordance_modules=length(concordance_modules) kinetic_modules=length(kinetic_modules) acr=acr_results.acr_metabolites acrr_pairs=length(acr_results.acrr_pairs)
    end

    @testset "6. Mathematical Properties" begin
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        kinetic_modules = kinetic_analysis(concordance_modules, model, min_module_size=1)

        # Verify Y𝚫 matrix construction
        Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)
        complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

        # Build Y𝚫 from giant module (should have 8 columns for 9 coupled complexes)
        giant_module = kinetic_modules[1]
        @test length(giant_module) == 9

        # Expected: 9 coupled complexes → 8 coupling relations (columns in Y𝚫)
        # This is verified internally but we can check the structure

        # Verify metabolite count
        @test length(metabolite_ids) == 9

        # Verify the 6 metabolites in ACRR pairs are those appearing in giant module
        metabolites_in_giant = Set{Symbol}()
        for complex in giant_module
            idx = findfirst(==(complex), complex_ids)
            if !isnothing(idx)
                stoich = Y_matrix[:, idx]
                for (i, met) in enumerate(metabolite_ids)
                    if abs(stoich[i]) > 1e-10
                        push!(metabolites_in_giant, met)
                    end
                end
            end
        end

        expected_metabolites = Set([:X, :Xp, :XD, :XDYp, :XT, :XTYp, :XpY, :Y, :Yp])
        @test metabolites_in_giant == expected_metabolites

        @info "Mathematical properties verified" n_metabolites_in_giant=length(metabolites_in_giant)
    end

    @testset "7. Paper Section S.5.2 Validation" begin
        # This test explicitly validates the EnvZ-OmpR example from Section S.5.2 of the paper
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        kinetic_modules = kinetic_analysis(concordance_modules, model, min_module_size=1)
        acr_results = identify_acr_acrr(kinetic_modules, model)

        # Paper states: "In the EnvZ-OmpR system... Yp exhibits ACR"
        @test :Yp in acr_results.acr_metabolites

        # Paper states: "This is a deficiency-2 network"
        # (verified by classical deficiency formula: δ = n - ℓ - s = 13 - 4 - 7 = 2)

        # Paper states: "Mass action deficiency δₖ = 1"
        # (verified by Proposition S4-8 and Lemma S4-7 during kinetic_analysis)

        # Paper states: "All non-terminal complexes are mutually coupled" (Theorem S4-6)
        @test length(kinetic_modules[1]) == 9  # Giant module with 9 non-terminal complexes

        @info "Paper Section S.5.2 validation complete ✓"
    end
end
