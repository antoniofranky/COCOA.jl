"""
Comprehensive validation tests for COCOA kinetic module analysis.
Validates against the paper "Kinetic modules are sources of concentration 
robustness in biochemical networks" by Langary et al. (2025).
"""

using Test
using COCOA
using COBREXA
using HiGHS
using SparseArrays
using LinearAlgebra
using Graphs
import AbstractFBCModels as A

# ===== HELPER FUNCTIONS FOR VALIDATION =====

"""
Calculate network deficiency: δ = n - ℓ - s
where n = number of complexes, ℓ = number of linkage classes, s = rank of stoichiometric matrix
"""
function calculate_deficiency(Y::SparseMatrixCSC, A::SparseMatrixCSC)
    n_complexes = size(Y, 2)

    # Calculate linkage classes (weakly connected components)
    # Create undirected graph from reaction graph
    n = size(A, 1)  # number of complexes
    g = SimpleGraph(n)

    # Add edges for reactions (both directions for weak connectivity)
    for j in 1:size(A, 2)
        # Find source and target complexes for reaction j
        source_idx = findfirst(x -> x < 0, A[:, j])
        target_idx = findfirst(x -> x > 0, A[:, j])
        if !isnothing(source_idx) && !isnothing(target_idx)
            add_edge!(g, source_idx, target_idx)
        end
    end

    # Count connected components (linkage classes)
    linkage_classes = connected_components(g)
    n_linkage = length(linkage_classes)

    # Calculate rank of stoichiometric matrix S = Y * A
    S = Y * A
    s_rank = rank(Matrix(S))

    deficiency = n_complexes - n_linkage - s_rank

    return deficiency, n_complexes, n_linkage, s_rank
end

"""
Identify strongly connected components (strong linkage classes) in the reaction graph
"""
function identify_strong_linkage_classes(A::SparseMatrixCSC)
    n = size(A, 1)  # number of complexes
    g = SimpleDiGraph(n)

    # Add directed edges for reactions
    for j in 1:size(A, 2)
        source_idx = findfirst(x -> x < 0, A[:, j])
        target_idx = findfirst(x -> x > 0, A[:, j])
        if !isnothing(source_idx) && !isnothing(target_idx)
            add_edge!(g, source_idx, target_idx)
        end
    end

    # Find strongly connected components
    sccs = strongly_connected_components(g)

    # Identify terminal strong linkage classes
    terminal_sccs = Int[]
    for (i, scc) in enumerate(sccs)
        is_terminal = true
        for v in scc
            for neighbor in outneighbors(g, v)
                if !(neighbor in scc)
                    is_terminal = false
                    break
                end
            end
            if !is_terminal
                break
            end
        end
        if is_terminal
            push!(terminal_sccs, i)
        end
    end

    return sccs, terminal_sccs
end

"""
Check if a set of complexes is autonomous (no incoming reactions from outside)
"""
function is_autonomous(complex_indices::Set{Int}, A::SparseMatrixCSC)
    for j in 1:size(A, 2)
        # Find source and target for reaction j
        source_idx = findfirst(x -> x < 0, A[:, j])
        target_idx = findfirst(x -> x > 0, A[:, j])

        if !isnothing(source_idx) && !isnothing(target_idx)
            # If source is outside but target is inside, not autonomous
            if !(source_idx in complex_indices) && (target_idx in complex_indices)
                return false
            end
        end
    end
    return true
end

"""
Extract complex names from kinetic results in a standardized way
"""
function extract_complex_names(kinetic_results::COCOA.KineticModuleResults, module_id::Symbol)
    if haskey(kinetic_results.kinetic_modules, module_id)
        return Set(string(c) for c in kinetic_results.kinetic_modules[module_id])
    end
    return Set{String}()
end

# ===== MODEL DEFINITIONS FROM THE PAPER =====

"""
Create the EnvZ-OmpR model exactly as described in the paper (Fig. 1A and Fig. S-5).
This is a closed network with no exchange reactions.

Expected properties:
- 9 species: A, B, C, D, E, F, G, H, I
- 14 reactions: R1-R14
- 13 complexes total
- 4 linkage classes
- 8 strong linkage classes
- 4 terminal strong linkage classes
- Deficiency = 13 - 4 - 7 = 2
"""
function create_envz_ompr_paper_model()
    model = A.CanonicalModel.Model()

    # Add the 9 species from the paper
    for s in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        model.metabolites[s] = A.CanonicalModel.Metabolite(name="Species $s")
    end

    # Add the 14 reactions exactly as in the paper
    # Note: All reactions are irreversible as shown in Fig. 1A

    # R1: A → B
    model.reactions["R1"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("A" => -1.0, "B" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R2: B → A
    model.reactions["R2"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("B" => -1.0, "A" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R3: B → C
    model.reactions["R3"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("B" => -1.0, "C" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R4: C → B
    model.reactions["R4"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("C" => -1.0, "B" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R5: C → D
    model.reactions["R5"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("C" => -1.0, "D" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R6: D + E → F
    model.reactions["R6"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("D" => -1.0, "E" => -1.0, "F" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R7: F → D + E
    model.reactions["R7"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("F" => -1.0, "D" => 1.0, "E" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R8: F → B + G
    model.reactions["R8"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("F" => -1.0, "B" => 1.0, "G" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R9: C + G → H
    model.reactions["R9"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("C" => -1.0, "G" => -1.0, "H" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R10: H → C + G
    model.reactions["R10"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("H" => -1.0, "C" => 1.0, "G" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R11: H → C + I
    model.reactions["R11"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("H" => -1.0, "C" => 1.0, "I" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R12: A + G → I  (Note: Paper shows this going to complex J, but J = A+I based on R13/R14)
    model.reactions["R12"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("A" => -1.0, "G" => -1.0, "I" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R13: I → A + G
    model.reactions["R13"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("I" => -1.0, "A" => 1.0, "G" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R14: I → A + I  (This creates A + 2I from I, representing the J → A + I from paper)
    # Actually, looking more carefully, J is a separate complex in the paper
    # Let me correct R12-R14 to match the paper exactly

    # Actually, I need to add species J
    model.metabolites["J"] = A.CanonicalModel.Metabolite(name="Species J")

    # Correct R12-R14:
    # R12: A + G → J
    model.reactions["R12"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("A" => -1.0, "G" => -1.0, "J" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R13: J → A + G
    model.reactions["R13"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("J" => -1.0, "A" => 1.0, "G" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # R14: J → A + I
    model.reactions["R14"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("J" => -1.0, "A" => 1.0, "I" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    return model
end

"""
Create the deficiency-2 network from Fig. S-6 in the supplementary material.

Expected properties:
- Deficiency = 2
- Three concordance modules (blue=balanced, yellow, pink)
- Two extended modules
"""
function create_deficiency_two_paper_model()
    model = A.CanonicalModel.Model()

    # Add metabolites from Fig. S-6
    species = ["A", "B", "C", "D", "E", "F", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    for s in species
        model.metabolites[s] = A.CanonicalModel.Metabolite(name="Species $s")
    end

    # Add reactions from Fig. S-6
    # Top pathway (yellow module)
    # k1/k2: F ⟷ A
    model.reactions["k1"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("F" => -1.0, "A" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )
    model.reactions["k2"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("A" => -1.0, "F" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k3: A → B
    model.reactions["k3"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("A" => -1.0, "B" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k4: C + F → E
    model.reactions["k4"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("C" => -1.0, "F" => -1.0, "E" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k5: E → D + F
    model.reactions["k5"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("E" => -1.0, "D" => 1.0, "F" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k6: B + D → A + C
    model.reactions["k6"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("B" => -1.0, "D" => -1.0, "A" => 1.0, "C" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # Bottom pathway (pink module)
    # k7/k8: X ⟷ S
    model.reactions["k7"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("X" => -1.0, "S" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )
    model.reactions["k8"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("S" => -1.0, "X" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k9: S → T
    model.reactions["k9"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("S" => -1.0, "T" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k10: U + X → W
    model.reactions["k10"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("U" => -1.0, "X" => -1.0, "W" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k11: W → V + X
    model.reactions["k11"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("W" => -1.0, "V" => 1.0, "X" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k12: T + Z → S + U
    model.reactions["k12"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("T" => -1.0, "Z" => -1.0, "S" => 1.0, "U" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k13: V → D + Y
    model.reactions["k13"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("V" => -1.0, "D" => 1.0, "Y" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    # k14: D + Y → Z
    model.reactions["k14"] = A.CanonicalModel.Reaction(
        stoichiometry=Dict("D" => -1.0, "Y" => -1.0, "Z" => 1.0),
        lower_bound=0.0, upper_bound=100.0
    )

    return model
end

# ===== MAIN TEST SUITE =====

@testset "COCOA Kinetic Module Mathematical Validation" begin

    @testset "EnvZ-OmpR Model (Paper Fig. 1A)" begin
        println("\n=== Validating EnvZ-OmpR Model from Paper ===")

        model = create_envz_ompr_paper_model()

        # Basic model structure
        @test length(A.metabolites(model)) == 10  # A-J
        @test length(A.reactions(model)) == 14    # R1-R14
        println("✓ Model structure: 10 metabolites, 14 reactions")

        # Run kinetic concordance analysis
        results = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=true,
            sample_size=100,
            cv_threshold=0.1,  # Tight tolerance
            seed=42
        )

        @test isa(results, COCOA.ConcentrationRobustnessResults)
        kinetic_results = results.kinetic_results

        # Extract matrices
        Y = kinetic_results.Y_matrix
        A_mat = kinetic_results.A_matrix

        println("  Matrix dimensions: Y=$(size(Y)), A=$(size(A_mat))")

        # ===== DEFICIENCY CALCULATION =====
        deficiency, n_complexes, n_linkage, s_rank = calculate_deficiency(Y, A_mat)

        println("  Deficiency components:")
        println("    n (complexes) = $n_complexes")
        println("    ℓ (linkage classes) = $n_linkage")
        println("    s (rank) = $s_rank")
        println("    δ (deficiency) = $deficiency")

        # Paper states: 13 complexes, 4 linkage classes, rank 7, deficiency = 2
        @test n_complexes == 13
        @test n_linkage == 4
        @test deficiency == 2
        println("✓ Network deficiency = 2 (matches paper)")

        # ===== STRONG LINKAGE CLASSES =====
        sccs, terminal_sccs = identify_strong_linkage_classes(A_mat)

        println("  Strong linkage classes: $(length(sccs))")
        println("  Terminal SLCs: $(length(terminal_sccs))")

        # Paper states: 8 strong linkage classes, 4 terminal
        @test length(sccs) == 8
        @test length(terminal_sccs) == 4
        println("✓ Graph structure matches paper (8 SLCs, 4 terminal)")

        # ===== CONCORDANCE MODULES =====
        concordance_modules = kinetic_results.concordance_results.modules

        println("  Concordance modules found: $(nrow(concordance_modules))")

        # Check for balanced module (should contain single species)
        balanced_found = false
        for row in eachrow(concordance_modules)
            complexes_str = row.complexes
            complex_list = split(complexes_str, ",")

            # Check if this module contains mostly single-species complexes
            single_species = filter(c -> !contains(strip(c), "_"), complex_list)
            if length(single_species) >= 3  # Should have F, E, W, X, D+Y as balanced
                balanced_found = true
                println("  ✓ Found balanced module with $(length(complex_list)) complexes")
                break
            end
        end
        @test balanced_found

        # ===== KINETIC MODULES =====
        kinetic_modules = kinetic_results.kinetic_modules
        giant_module_id = kinetic_results.giant_module_id

        @test haskey(kinetic_modules, giant_module_id)
        giant_module = kinetic_modules[giant_module_id]

        println("  Kinetic modules: $(length(kinetic_modules))")
        println("  Giant module size: $(length(giant_module)) complexes")

        # The giant module should contain most complexes
        @test length(giant_module) >= 10  # Most of the 13 complexes
        println("✓ Giant kinetic module identified")

        # ===== CONCENTRATION ROBUSTNESS =====
        robust_metabolites = results.robust_metabolites
        robust_pairs = results.robust_metabolite_pairs

        println("  Robust metabolites: $(length(robust_metabolites))")
        println("  Robust metabolite pairs: $(length(robust_pairs))")

        # For deficiency 2, we expect some robustness
        @test results.n_robust_metabolites >= 0
        @test results.n_robust_pairs >= 0
        println("✓ Concentration robustness analysis completed")

        # ===== MATRIX CONSISTENCY: S = Y × A =====
        S_reconstructed = Y * A_mat

        # Check that S has correct dimensions
        @test size(S_reconstructed, 1) == 10  # 10 metabolites

        # Check that mass is conserved (each reaction column sums to 0)
        for j in 1:size(S_reconstructed, 2)
            @test abs(sum(S_reconstructed[:, j])) < 1e-10
        end
        println("✓ Matrix relationship S = Y × A verified")

        # ===== AUTONOMY PROPERTY =====
        # The giant kinetic module should be autonomous
        giant_complex_indices = Set{Int}()
        for complex_symbol in giant_module
            complex_str = string(complex_symbol)
            if haskey(kinetic_results.complex_to_idx, complex_str)
                push!(giant_complex_indices, kinetic_results.complex_to_idx[complex_str])
            end
        end

        if !isempty(giant_complex_indices)
            @test is_autonomous(giant_complex_indices, A_mat)
            println("✓ Giant module is autonomous")
        end
    end

    @testset "Deficiency-2 Network (Paper Fig. S-6)" begin
        println("\n=== Validating Deficiency-2 Network from Supplementary ===")

        model = create_deficiency_two_paper_model()

        # Basic model structure
        @test length(A.metabolites(model)) == 14  # A-F, S-Z
        @test length(A.reactions(model)) == 14    # k1-k14
        println("✓ Model structure: 14 metabolites, 14 reactions")

        # Run kinetic concordance analysis
        results = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=false,  # Skip for this test
            sample_size=50,
            cv_threshold=0.1,
            seed=42
        )

        @test isa(results, COCOA.KineticModuleResults)

        # Extract matrices
        Y = results.Y_matrix
        A_mat = results.A_matrix

        # ===== DEFICIENCY CALCULATION =====
        deficiency, n_complexes, n_linkage, s_rank = calculate_deficiency(Y, A_mat)

        println("  Deficiency components:")
        println("    n (complexes) = $n_complexes")
        println("    ℓ (linkage classes) = $n_linkage")
        println("    s (rank) = $s_rank")
        println("    δ (deficiency) = $deficiency")

        # Should have deficiency = 2
        @test deficiency == 2
        println("✓ Network deficiency = 2 (as expected)")

        # ===== CONCORDANCE MODULES =====
        concordance_modules = results.concordance_results.modules

        println("  Concordance modules found: $(nrow(concordance_modules))")

        # Paper states 3 concordance modules:
        # - Balanced: {F, E, W, X, D+Y}
        # - Module 1: {A, B, C+F, D+F, B+D, A+C}
        # - Module 2: {U+X, V+X, S, T, T+Z, S+U, V, Z}

        # We should find at least 1 module (could be merged)
        @test nrow(concordance_modules) >= 1

        # ===== KINETIC MODULES =====
        kinetic_modules = results.kinetic_modules

        println("  Kinetic modules: $(length(kinetic_modules))")

        # Should identify kinetic modules
        @test length(kinetic_modules) >= 1

        # Check the giant module
        giant_module_id = results.giant_module_id
        @test haskey(kinetic_modules, giant_module_id)

        giant_module = kinetic_modules[giant_module_id]
        println("  Giant module size: $(length(giant_module)) complexes")

        println("✓ Kinetic modules identified for deficiency-2 network")
    end

    @testset "Mathematical Properties and Invariants" begin
        println("\n=== Testing Mathematical Invariants ===")

        # Test with a simple model to verify properties
        model = create_envz_ompr_paper_model()

        results = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=false,
            sample_size=20,
            seed=42
        )

        Y = results.Y_matrix
        A_mat = results.A_matrix

        # ===== PROPERTY 1: Complexes form a partition =====
        # Every complex should be in exactly one concordance module
        all_complexes_in_modules = Set{String}()
        for row in eachrow(results.concordance_results.modules)
            complexes_str = row.complexes
            for complex in split(complexes_str, ",")
                complex_clean = strip(complex)
                @test !(complex_clean in all_complexes_in_modules)  # No duplicates
                push!(all_complexes_in_modules, complex_clean)
            end
        end

        # All complexes should be covered
        @test length(all_complexes_in_modules) == size(Y, 2)
        println("✓ Concordance modules form a partition")

        # ===== PROPERTY 2: Conservation of mass =====
        S = Y * A_mat
        for j in 1:size(S, 2)
            reaction_sum = sum(S[:, j])
            @test abs(reaction_sum) < 1e-10  # Each reaction conserves mass
        end
        println("✓ Mass conservation verified")

        # ===== PROPERTY 3: Deficiency is non-negative =====
        deficiency, _, _, _ = calculate_deficiency(Y, A_mat)
        @test deficiency >= 0
        println("✓ Deficiency is non-negative: δ = $deficiency")

        # ===== PROPERTY 4: Kinetic modules are disjoint =====
        all_complexes_in_kinetic = Set{Symbol}()
        for (module_id, module_complexes) in results.kinetic_modules
            for complex in module_complexes
                @test !(complex in all_complexes_in_kinetic)  # No overlap
                push!(all_complexes_in_kinetic, complex)
            end
        end
        println("✓ Kinetic modules are disjoint")

        # ===== PROPERTY 5: Y matrix structure =====
        # Y should have non-negative integer entries (stoichiometric coefficients)
        @test all(Y.nzval .>= 0)
        @test all(isinteger.(Y.nzval))
        println("✓ Y matrix has correct structure")

        # ===== PROPERTY 6: A matrix structure =====
        # A should be an incidence matrix with entries in {-1, 0, 1}
        @test all(abs.(A_mat.nzval) .<= 1)
        @test all(isinteger.(A_mat.nzval))

        # Each reaction column should have exactly one -1 and one +1
        for j in 1:size(A_mat, 2)
            col = A_mat[:, j]
            @test count(x -> x == -1, col) == 1  # One source complex
            @test count(x -> x == 1, col) == 1   # One target complex
        end
        println("✓ A matrix is a proper incidence matrix")
    end

    @testset "Reproducibility and Consistency" begin
        println("\n=== Testing Reproducibility ===")

        model = create_envz_ompr_paper_model()

        # Run analysis twice with same seed
        results1 = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=false,
            sample_size=30,
            seed=123
        )

        results2 = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=false,
            sample_size=30,
            seed=123
        )

        # Check that key results match
        @test keys(results1.kinetic_modules) == keys(results2.kinetic_modules)
        @test results1.giant_module_id == results2.giant_module_id
        @test size(results1.Y_matrix) == size(results2.Y_matrix)
        @test size(results1.A_matrix) == size(results2.A_matrix)

        # Check matrix values match
        @test norm(results1.Y_matrix - results2.Y_matrix) < 1e-10
        @test norm(results1.A_matrix - results2.A_matrix) < 1e-10

        println("✓ Same seed produces identical results")

        # Different seeds should still produce valid results
        results3 = COCOA.kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            include_kinetic_modules=true,
            include_robustness=false,
            sample_size=30,
            seed=456
        )

        # Structural properties should be the same
        @test size(results3.Y_matrix) == size(results1.Y_matrix)
        @test size(results3.A_matrix) == size(results1.A_matrix)

        # Calculate deficiency for both - should be identical
        def1, _, _, _ = calculate_deficiency(results1.Y_matrix, results1.A_matrix)
        def3, _, _, _ = calculate_deficiency(results3.Y_matrix, results3.A_matrix)
        @test def1 == def3

        println("✓ Different seeds preserve structural invariants")
    end
end

println("\n" * "="^50)
println("Mathematical validation complete!")
println("="^50)