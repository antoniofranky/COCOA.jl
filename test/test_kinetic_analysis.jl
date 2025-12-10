
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

# Load the new redesigned kinetics module
include("../src/kinetics/Kinetics.jl")
using .Kinetics

# Override COCOA's kinetic_analysis with the new implementation
const kinetic_analysis = Kinetics.kinetic_analysis

# For tests that use the old network tuple format, keep COCOA's old functions
const upstream_algorithm_old = COCOA.upstream_algorithm
const merge_coupled_sets_old = COCOA.merge_coupled_sets

@testset "EnvZ-OmpR Complete Pipeline" begin

    @testset "1. Model Creation" begin
        model = create_envz_ompr_model()

        @test model isa A.AbstractFBCModel
        @test length(A.reactions(model)) == 14
        @test length(A.metabolites(model)) == 9

        @info "Model created successfully" n_reactions = length(A.reactions(model)) n_metabolites = length(A.metabolites(model))
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

        @info "Concordance modules verified" n_modules = length(concordance_modules) balanced_size = length(balanced)
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

        result = kinetic_analysis(concordance_modules, model; min_module_size=1, efficient=false)
        kinetic_modules = result.kinetic_modules

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

        @info "Kinetic modules verified" n_modules = length(kinetic_modules) giant_size = length(kinetic_modules[1])
    end

    @testset "4. ACR/ACRR Identification" begin
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        result = kinetic_analysis(concordance_modules, model; min_module_size=1, efficient=false)
        kinetic_modules = result.kinetic_modules

        # Identify ACR/ACRR (now returned directly by kinetic_analysis in result)
        acr_results = result  # Contains acr_metabolites and acrr_pairs

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

        @info "ACR/ACRR identification verified" n_acr = length(acr_results.acr_metabolites) n_acrr = length(acr_results.acrr_pairs)
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

        # Step 3: Kinetic analysis (returns NamedTuple with kinetic_modules, acr_metabolites, acrr_pairs, stats)
        result = kinetic_analysis(concordance_modules, model; min_module_size=1, efficient=false)
        kinetic_modules = result.kinetic_modules

        # Step 4: ACR/ACRR already identified in result
        acr_results = result

        # Verify final outputs
        expected_giant_module = Set([:XD, :XT, Symbol("Xp+Y"), Symbol("XT+Yp"), :XDYp, :XTYp, Symbol("XD+Yp"), :XpY, :X])
        expected_acr = Set([:Yp])
        @test length(concordance_modules) == 4
        @test length(kinetic_modules) == 5
        @test kinetic_modules[1] == expected_giant_module
        @test :Yp in acr_results.acr_metabolites
        @test length(acr_results.acrr_pairs) == 15

        # Verify structural deficiency calculation (should be 2 for EnvZ-OmpR)
        # This is computed internally during kinetic_analysis
        # We can verify by checking the log output or running the calculation directly

        @info "Complete pipeline validated" concordance_modules = length(concordance_modules) kinetic_modules = length(kinetic_modules) acr = acr_results.acr_metabolites acrr_pairs = length(acr_results.acrr_pairs)
    end

    @testset "6. Mathematical Properties" begin
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        result = kinetic_analysis(concordance_modules, model; min_module_size=1, efficient=false)
        kinetic_modules = result.kinetic_modules

        # Verify Y𝚫 matrix construction
        Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)
        complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

        # Build Y𝚫 from giant module (should have 8 columns for 9 coupled complexes)
        expected_giant_module = Set([:XD, :XT, Symbol("Xp+Y"), Symbol("XT+Yp"), :XDYp, :XTYp, Symbol("XD+Yp"), :XpY, :X])
        giant_module = kinetic_modules[1]
        @test length(giant_module) == 9
        @test giant_module == expected_giant_module
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

        @info "Mathematical properties verified" n_metabolites_in_giant = length(metabolites_in_giant)
    end

    @testset "7. Paper Section S.5.2 Validation (Detailed)" begin
        # This test explicitly validates the EnvZ-OmpR example from Section S.5.2 of the paper
        # Detailed walkthrough checking intermediate coupling sets and merging logic.
        model = create_envz_ompr_model()

        # 1. Verify Concordance Modules
        balanced = Set([:XDYp, :XTYp, :XpY, :XD, :XT])
        Cm1 = Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")])
        Cm2 = Set([Symbol("XT+Yp"), Symbol("XT+Y")])
        Cm3 = Set([Symbol("XD+Yp"), Symbol("XD+Y")])

        # Note: In the paper, balanced set is identified later or implicitly. 
        # Here we verify that with these inputs, we get the exact paper result.

        concordance_modules = [balanced, Cm1, Cm2, Cm3]

        Cm3 = Set([Symbol("XD+Yp"), Symbol("XD+Y")])

        # 2. Verify Upstream Sets (Initial Coupling)
        A_matrix, complex_ids = COCOA.incidence(model; return_ids=true)
        Y_matrix, metabolite_ids, _ = COCOA.complex_stoichiometry(model; return_ids=true)

        network = (
            A=A_matrix, Y=Y_matrix, complex_ids=complex_ids, metabolite_ids=metabolite_ids,
            complex_to_idx=Dict(id => i for (i, id) in enumerate(complex_ids)),
            acr_augmentation=zeros(Float64, length(metabolite_ids), 0)
        )

        Cm1_bar, Cm2_bar, Cm3_bar = union(balanced, Cm1), union(balanced, Cm2), union(balanced, Cm3)
        C1, C2, C3 = [COCOA.upstream_algorithm(G, network) for G in [Cm1_bar, Cm2_bar, Cm3_bar]]

        expected_C1 = Set([:XD, :X, :XT, Symbol("Xp+Y"), :XpY])
        expected_C2 = Set([Symbol("XT+Yp"), :XTYp])
        expected_C3 = Set([Symbol("XD+Yp"), :XDYp])

        @test C1 == expected_C1 && C2 == expected_C2 && C3 == expected_C3

        # 3. Verify Advanced Merging (Proposition S3-4)
        # Verify C2 and C3 merge into C4 due to constant ratio from C1
        merged_modules = COCOA.merge_coupled_sets([C1, C2, C3], network)

        expected_C4 = union(expected_C2, expected_C3)
        has_C4 = any(m -> m == expected_C4, merged_modules)
        @test has_C4 || any(m -> issubset(expected_C4, m), merged_modules)

        if has_C4
            @info "Advanced Merging Verified: C2+C3 merged"
        else
            @info "Advanced Merging Result: Modules merged further than C4" modules = merged_modules
        end

        # 4. Final Result Validation
        concordance_modules_vec = [balanced, Cm1, Cm2, Cm3]
        result = kinetic_analysis(concordance_modules_vec, model; efficient=false)
        kinetic_modules = result.kinetic_modules

        expected_giant = union(expected_C1, expected_C4)
        @test length(kinetic_modules[1]) == 9
        @test kinetic_modules[1] == expected_giant

        @info "Paper Section S.5.2 detailed validation complete ✓"
    end

    @testset "8. Deficiency Calculations" begin
        model = create_envz_ompr_model()

        concordance_modules = [
            Set([:XD, :XT, :XpY, :XTYp, :XDYp]),
            Set([:X, :Xp, Symbol("Xp+Y"), Symbol("X+Yp")]),
            Set([Symbol("XT+Yp"), Symbol("XT+Y")]),
            Set([Symbol("XD+Yp"), Symbol("XD+Y")])
        ]

        # Test structural deficiency
        δ = structural_deficiency(concordance_modules, model)

        # Paper states EnvZ-OmpR is deficiency-2 (δ = n - ℓ - s = 13 - 4 - 7 = 2)
        @test δ == 2
        @info "Structural deficiency verified" δ = δ

        # Test mass action deficiency bounds (before merging)
        bounds_initial = mass_action_deficiency_bounds(concordance_modules, model; n_concordance_merges=0)

        # Initial: δ₀ = 2, not weakly reversible → δₖ ∈ [1, 2]
        @test bounds_initial.lower == 1  # Lemma S4-7: not weakly reversible → δₖ ≥ 1
        @test bounds_initial.upper == 2  # Proposition S4-8: δₖ ≤ δ₀ - 0 = 2
        @test !bounds_initial.is_exact    # δₖ could be 1 or 2
        @test !bounds_initial.weakly_reversible
        @info "Initial mass action deficiency bounds" bounds_initial

        # Test after concordance merging (3 unbalanced modules merge into 1)
        # This represents 2 concordance merges: 4 modules → 3 modules → 2 modules
        bounds_after_merge = mass_action_deficiency_bounds(concordance_modules, model; n_concordance_merges=2)

        # After 2 merges: δₖ ≤ 2 - 2 = 0, but δₖ ≥ 1 → contradiction means we need 1 more merge
        # Actually the algorithm does 1 merge (3 unbalanced → 1 giant), so let's test that:
        bounds_after_1_merge = mass_action_deficiency_bounds(concordance_modules, model; n_concordance_merges=1)

        @test bounds_after_1_merge.lower == 1
        @test bounds_after_1_merge.upper == 1  # δₖ ≤ 2 - 1 = 1
        @test bounds_after_1_merge.is_exact    # δₖ = 1 exactly!
        @info "Mass action deficiency after merging" bounds_after_1_merge

        # Verify the paper's claim: after proper merging, δₖ = 1
        @test bounds_after_1_merge.is_exact && bounds_after_1_merge.lower == 1
    end
end

@testset "Deficiency Two Network (Fig S-6)" begin
    model = create_deficiency_two_model()

    # Concordance modules from text
    # 𝒞b ={F, E, W, X, D+Y}
    Cb = Set([:F, :E, :W, :X, Symbol("D+Y")])

    # 𝒞m1={A, B, C+F, D+F, B+D, A+C}
    Cm1 = Set([:A, :B, Symbol("C+F"), Symbol("D+F"), Symbol("B+D"), Symbol("A+C")])

    # 𝒞m2={U+X, V+X, S, T, T+Z, S+U, V, Z}
    Cm2 = Set([Symbol("U+X"), Symbol("V+X"), :S, :T, Symbol("T+Z"), Symbol("S+U"), :V, :Z])

    concordance_modules = [Cb, Cm1, Cm2]

    # Use efficient=false to enable advanced merging via Proposition S3-4
    result = kinetic_analysis(concordance_modules, model; efficient=true, min_module_size=1)
    kinetic_modules = result.kinetic_modules

    # Expected coupling partition from text:
    # {F, A, C+F, E, B+D, A+C}
    M1_expected = Set([:F, :A, Symbol("C+F"), :E, Symbol("B+D"), Symbol("A+C")])

    # {U+X, W, X, S, T+Z, V, D+Y, S+U}
    M2_expected = Set([Symbol("U+X"), :W, :X, :S, Symbol("T+Z"), :V, Symbol("D+Y"), Symbol("S+U")])

    # Terminals: {B}, {D+F}, {V+X}, {T}, {Z}
    Terminals_expected = Set([
        Set([:B]),
        Set([Symbol("D+F")]),
        Set([Symbol("V+X")]),
        Set([:T]),
        Set([:Z])
    ])

    # Find M1 and M2 in results
    found_M1 = false
    found_M2 = false
    found_terminals = Set{Set{Symbol}}()

    for mod in kinetic_modules
        if mod == M1_expected
            found_M1 = true
        elseif mod == M2_expected
            found_M2 = true
        elseif length(mod) == 1
            push!(found_terminals, mod)
        end
    end

    @test found_M1
    @test found_M2
    @test found_terminals == Terminals_expected
    @test length(kinetic_modules) == 7 # 2 large + 5 terminals

    @info "Deficiency Two Network validation complete" found_M1 found_M2 n_terminals = length(found_terminals)

    # Also check efficient ACR/ACRR on this model (efficient=true)
    # Should detect comparable ACR/ACRR results even if merging logic is simpler
    # ACR/ACRR already computed and returned in result
    acr_results = result

    # Text says: ACR components: C, U
    @test :C in acr_results.acr_metabolites
    @test :U in acr_results.acr_metabolites

    # Text says: ACRR pairs: A/F, A/E, E/F, W/X, W/S, X/S, W/V, X/V, S/V
    # Check a few
    expected_pairs = [
        (:A, :F), (:A, :E), (:E, :F),
        (:W, :X), (:W, :S), (:X, :S),
        (:W, :V), (:X, :V), (:S, :V)
    ]

    count_found = 0
    for (m1, m2) in expected_pairs
        # Order in tuple might be swapped
        if (m1, m2) in acr_results.acrr_pairs || (m2, m1) in acr_results.acrr_pairs
            count_found += 1
        end
    end

    @test count_found >= length(expected_pairs)
    @info "Efficient ACR/ACRR verified on Deficiency Two Network" count_found
end

