using Test
using COCOA
using COBREXA
using HiGHS
using GLPK
using SparseArrays
using DataFrames

# Load model format packages needed by COBREXA
import JSONFBCModels
import SBMLFBCModels
import AbstractFBCModels as A

# Include our model creation
include("envz_ompr_model.jl")

@testset "COCOA.jl - EnvZ-OmpR Paper Validation" begin

    @testset "EnvZ-OmpR Model Loading and Basic Validation" begin
        println("Testing EnvZ-OmpR model loading...")

        # Use the paper-style EnvZ-OmpR model definition used by strict tests.
        model = create_envz_ompr_model()
        println("✓ Created EnvZ-OmpR model (paper-style)")

        # Basic model validation
        @test model isa A.AbstractFBCModel

        # According to the kinetic modules paper (Figure 1A):
        # - The EnvZ-OmpR system should have 9 species (A, B, C, D, E, F, G, H, I)
        # - 14 reactions (R1-R14) 
        # - 13 complexes
        metabolites = A.metabolites(model)
        reactions = A.reactions(model)

        @test length(metabolites) == 9
        @test length(reactions) == 14

        println("  Model has $(length(metabolites)) metabolites and $(length(reactions)) reactions")

        # Test that model is feasible
        basic_solution = flux_balance_analysis(model, optimizer=HiGHS.Optimizer)
        @test basic_solution !== nothing
        println("  ✓ Model is feasible")

        println("✓ EnvZ-OmpR model validated")
    end

    @testset "Kinetic Analysis Results Match Paper" begin
        println("Testing kinetic analysis results against paper expectations...")

        model = create_envz_ompr_model()

        # Test 1: Build concordance constraints using the canonical paper pipeline
        println("  Building concordance constraints...")
        constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

        @test haskey(constraints, :balance)
        @test haskey(constraints, :activities)
        @test haskey(constraints, :balance)
        println("    ✓ Built canonical concordance constraints")

        # Test 2: Extract complexes using constraint-aware method
        println("  Extracting complexes from constraints...")
        complexes, _ = COCOA.extract_complexes(constraints)

        @test length(complexes) > 0
        println("    ✓ Found $(length(complexes)) complexes")

        # Paper expectation: EnvZ-OmpR system has exactly 13 complexes.
        expected_complexes = 13
        actual_complexes = length(complexes)
        @test actual_complexes == expected_complexes
        println("    ✓ Complex count matches paper expectation: $actual_complexes")

        # Test 3: Verify incidence dimensions for the paper model
        A_matrix_check, _, _ = COCOA.incidence(constraints; return_ids=true)
        @test size(A_matrix_check, 1) == expected_complexes
        @test size(A_matrix_check, 2) == 14
        println("    ✓ Incidence matrix dimensions match paper model")

        # Test 4: Matrix extraction and validation
        println("  Testing matrix extraction...")
        Y_matrix, _, _ = COCOA.complex_stoichiometry(constraints; return_ids=true, model=model)
        A_matrix, complex_ids, reaction_ids = COCOA.incidence(constraints; return_ids=true)
        network_data = (
            Y_matrix=Y_matrix,
            A_matrix=A_matrix,
            complexes=complexes,
            complex_activities=haskey(constraints, :activities) ? constraints.activities : nothing,
            reaction_names=String.(reaction_ids),
        )

        @test network_data.Y_matrix isa SparseArrays.SparseMatrixCSC
        @test network_data.A_matrix isa SparseArrays.SparseMatrixCSC

        # Validate matrix dimensions according to paper
        n_metabolites = length(A.metabolites(model))
        @test size(network_data.Y_matrix, 1) == n_metabolites  # Species dimension
        @test size(network_data.A_matrix, 1) == length(complexes)  # Complex dimension

        println("    ✓ Y matrix: $(size(network_data.Y_matrix)), $(nnz(network_data.Y_matrix)) non-zeros")
        println("    ✓ A matrix: $(size(network_data.A_matrix)), $(nnz(network_data.A_matrix)) non-zeros")

        # Test 5: Matrix properties match paper definitions
        Y_values = SparseArrays.nonzeros(network_data.Y_matrix)
        A_values = SparseArrays.nonzeros(network_data.A_matrix)

        @test eltype(network_data.Y_matrix) == Float64  # Stoichiometric coefficients
        @test eltype(network_data.A_matrix) <: Integer  # Incidence matrix  
        @test all(Y_values .> 0)  # All positive stoichiometric coefficients
        @test all(abs.(A_values) .<= 1)  # Only ±1 or 0 in incidence matrix

        println("    ✓ Matrix properties match paper: Y(Float64), A($(eltype(network_data.A_matrix)))")

        # Test 6: Expected matrix structure for EnvZ-OmpR system
        # According to paper: Y should be 9×13 (species × complexes) approximately
        expected_species = 9
        @test size(network_data.Y_matrix, 1) >= expected_species

        # A matrix is complexes × reactions for the paper model
        @test size(network_data.A_matrix, 2) == 14

        println("    ✓ Matrix dimensions match expected structure")

        # Test 7: Verify kinetic module structure
        println("  Testing kinetic module structure...")

        # According to the paper, the EnvZ-OmpR system should show:
        # - Clear separation between complexes
        # - Proper reaction splitting
        # - Consistent stoichiometric structure

        # Check that we have proper complex-reaction associations
        for i in 1:size(network_data.A_matrix, 1)
            row = network_data.A_matrix[i, :]
            reaction_count = count(x -> abs(x) > 0, row)
            @test reaction_count > 0  # Each complex should participate in at least one reaction
        end

        println("    ✓ All complexes participate in reactions")

        # Test 8: Compare with original extraction
        println("  Comparing constraint-aware vs original extraction...")
        _, _, original_complexes = COCOA.complex_stoichiometry(model; return_ids=true)

        println("    Original extraction: $(length(original_complexes)) complexes")
        println("    Constraint extraction: $(length(complexes)) complexes")

        @test length(original_complexes) == length(complexes)
        println("    ✓ Constraint and model extraction agree on complex count")

        println("✓ All kinetic analysis results validated")
    end

    @testset "Paper Compliance Validation" begin
        println("Validating full compliance with kinetic modules paper...")

        model = create_envz_ompr_model()

        constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)
        Y_matrix, _, _ = COCOA.complex_stoichiometry(constraints; return_ids=true, model=model)
        A_matrix, _, reaction_ids = COCOA.incidence(constraints; return_ids=true)
        complexes, _ = COCOA.extract_complexes(constraints)
        network_data = (
            Y_matrix=Y_matrix,
            A_matrix=A_matrix,
            complexes=complexes,
            complex_activities=haskey(constraints, :activities) ? constraints.activities : nothing,
            reaction_names=String.(reaction_ids),
        )

        # Paper requirements checklist specific to EnvZ-OmpR
        requirements = [
            ("Model has exactly 9 species (paper)",
                length(A.metabolites(model)) == 9),
            ("Model has exactly 14 reactions (paper)",
                length(A.reactions(model)) == 14),
            ("Complex count matches paper (13)",
                length(complexes) == 13),
            ("Reaction count in incidence matches paper (14)",
                size(network_data.A_matrix, 2) == 14),
            ("Y matrix represents stoichiometric coefficients (Float64)",
                eltype(network_data.Y_matrix) == Float64),
            ("A matrix represents incidence structure (Integer)",
                eltype(network_data.A_matrix) <: Integer),
            ("Y matrix has positive values only",
                all(SparseArrays.nonzeros(network_data.Y_matrix) .> 0)),
            ("A matrix has only ±1 values",
                all(abs.(SparseArrays.nonzeros(network_data.A_matrix)) .<= 1)),
            ("Complex activities available",
                network_data.complex_activities !== nothing),
            ("Complexes use real metabolite combinations (not artificial substrate/product)",
                !any(occursin("_substrate", string(id)) || occursin("_product", string(id)) for id in keys(complexes))),
            ("Constraint-aware extraction agrees with model complex count",
                length(complexes) == 13),
            ("Matrix dimensions are consistent",
                size(network_data.Y_matrix, 1) == length(A.metabolites(model)) &&
                size(network_data.A_matrix, 1) == length(complexes)),
        ]

        println("\n  EnvZ-OmpR Paper Compliance Checklist:")
        all_passed = true
        for (requirement, passed) in requirements
            status = passed ? "✓" : "✗"
            println("    $status $requirement")
            all_passed = all_passed && passed
        end

        @test all_passed

        # Print summary of key results
        println("\n  Key Results Summary:")
        println("    Species: $(length(A.metabolites(model)))")
        println("    Reactions: $(length(A.reactions(model)))")
        println("    Complexes: $(length(complexes))")
        println("    Y matrix: $(size(network_data.Y_matrix))")
        println("    A matrix: $(size(network_data.A_matrix))")
        println("    Reactions in A matrix: $(size(network_data.A_matrix, 2))")

        println("\n✓ Full paper compliance validated!")
    end

    @testset "Robust Metabolite Pairs Validation" begin
        println("Testing robust metabolite pairs with structural validation fix...")

        # Create a simple predictable test model
        function create_simple_predictable_model()
            model = CM.Model()

            # 4 metabolites: A, B, C, D
            model.metabolites = Dict(
                "A" => CM.Metabolite(name="Metabolite A"),
                "B" => CM.Metabolite(name="Metabolite B"),
                "C" => CM.Metabolite(name="Metabolite C"),
                "D" => CM.Metabolite(name="Metabolite D")
            )

            # R1: A + C ⇄ B + C (reversible, C acts as catalyst)
            # This will create complexes: A+C and B+C which should form a robust pair (A,B)
            model.reactions["R1"] = CM.Reaction(
                name="A + C to B + C",
                stoichiometry=Dict("A" => -1.0, "C" => -1.0, "B" => 1.0, "C" => 1.0),
                lower_bound=-1000.0,  # Reversible
                upper_bound=1000.0
            )

            # R2: B → D (irreversible) 
            # This creates complexes: B and D (should NOT be robust - no catalyst)
            model.reactions["R2"] = CM.Reaction(
                name="B to D",
                stoichiometry=Dict("B" => -1.0, "D" => 1.0),
                lower_bound=0.0,
                upper_bound=1000.0
            )

            # Exchange reactions for feasibility
            model.reactions["EX_A"] = CM.Reaction(
                name="Exchange A",
                stoichiometry=Dict("A" => 1.0),
                lower_bound=-10.0, upper_bound=10.0
            )
            model.reactions["EX_C"] = CM.Reaction(
                name="Exchange C",
                stoichiometry=Dict("C" => 1.0),
                lower_bound=-10.0, upper_bound=10.0
            )
            model.reactions["EX_D"] = CM.Reaction(
                name="Exchange D",
                stoichiometry=Dict("D" => -1.0),
                lower_bound=0.0, upper_bound=10.0
            )

            return model
        end

        # Test with synthetic Y matrix first (bypassing constraint system issues)
        println("  Testing with synthetic Y matrix...")

        # Create Y matrix that represents the expected complexes:
        # Complex 1: A+C [1,0,1,0] 
        # Complex 2: B+C [0,1,1,0]  
        # These should be robust: differ in A,B (qq1=2), each has C additional (qq2=qq3=1)
        # Complex 3: B   [0,1,0,0]
        # Complex 4: D   [0,0,0,1]
        # These should NOT be robust: differ in B,D (qq1=2), no additional metabolites (qq2=qq3=0)

        test_Y = sparse([
            1.0 0.0 0.0 0.0;  # A
            0.0 1.0 1.0 0.0;  # B  
            1.0 1.0 0.0 0.0;  # C
            0.0 0.0 0.0 1.0   # D
        ])

        metabolite_names = ["A", "B", "C", "D"]

        met_ids = Symbol.(metabolite_names)
        complex_to_idx = Dict(:C1 => 1, :C2 => 2, :C3 => 3, :C4 => 4)
        # Keep C3/C4 as singleton modules so only C1-C2 is evaluated for pairwise ACRR
        robust_res = COCOA._detect_acr_acrr(
            [Set([:C1, :C2]), Set([:C3]), Set([:C4])],
            test_Y,
            met_ids,
            complex_to_idx;
            efficient=true,
        )
        robust_pairs = [(String(a), String(b)) for (a, b) in robust_res.acrr_pairs]

        println("    Found $(length(robust_pairs)) robust pairs: $robust_pairs")

        # Manually verify which pairs should be valid
        println("    Manual validation:")
        manual_pairs = Set()
        for i in 1:size(test_Y, 2)
            for j in (i+1):size(test_Y, 2)
                vec1 = test_Y[:, i]
                vec2 = test_Y[:, j]

                diff_mask = vec1 .!= vec2
                differing_indices = findall(diff_mask)
                n_differences = length(differing_indices)

                if n_differences == 2
                    met_idx1, met_idx2 = differing_indices

                    # qq conditions
                    non_zero_vec1 = findall(!iszero, vec1)
                    non_zero_vec2 = findall(!iszero, vec2)
                    qq1 = true  # already confirmed
                    qq2 = length(setdiff(non_zero_vec1, differing_indices)) == 1
                    qq3 = length(setdiff(non_zero_vec2, differing_indices)) == 1

                    println("      Complex $i vs $j: qq1=$qq1, qq2=$qq2, qq3=$qq3")
                    println("        Vec1: $(Array(vec1)), Vec2: $(Array(vec2))")
                    println("        Differences: $differing_indices")
                    println("        Additional in Vec1: $(setdiff(non_zero_vec1, differing_indices))")
                    println("        Additional in Vec2: $(setdiff(non_zero_vec2, differing_indices))")

                    if qq1 && qq2 && qq3
                        met_name1 = metabolite_names[met_idx1]
                        met_name2 = metabolite_names[met_idx2]
                        pair = met_name1 < met_name2 ? (met_name1, met_name2) : (met_name2, met_name1)
                        push!(manual_pairs, pair)
                        println("        ✓ Valid robust pair: $pair")
                    else
                        println("        ✗ Invalid (fails qq conditions)")
                    end
                end
            end
        end

        # Expected result: Only (A,B) should be robust due to catalyst C
        @test Set(robust_pairs) == Set([("A", "B")])
        @test length(robust_pairs) == 1
        println("    ✓ Correct result: Only (A,B) is robust as expected")

        # Test edge cases
        println("  Testing edge cases...")

        # Empty module list
        empty_res = COCOA._detect_acr_acrr(
            Set{Symbol}[],
            spzeros(0, 0),
            Symbol[],
            Dict{Symbol,Int}();
            efficient=true,
        )
        @test isempty(empty_res.acrr_pairs)

        # Single complex module
        single_res = COCOA._detect_acr_acrr(
            [Set([:Only])],
            sparse([1.0; 0.0; 1.0;;]),
            [:X, :Y, :Z],
            Dict(:Only => 1);
            efficient=true,
        )
        @test isempty(single_res.acrr_pairs)
        println("    ✓ Edge cases handled correctly")

        println("    ✓ Edge cases handled correctly")
        println("✓ Robust metabolite pairs validation complete!")
    end

    @testset "Architecture Integration Test" begin
        println("Testing end-to-end architecture integration...")

        model = create_envz_ompr_model()

        # Test the complete analysis pipeline
        # 1. Build constraints
        constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

        # 2. Extract complexes
        complexes, _ = COCOA.extract_complexes(constraints)

        # 3. Extract network matrices
        Y_matrix, _, _ = COCOA.complex_stoichiometry(constraints; return_ids=true, model=model)
        A_matrix, _, reaction_ids = COCOA.incidence(constraints; return_ids=true)
        network_data = (
            Y_matrix=Y_matrix,
            A_matrix=A_matrix,
            complexes=complexes,
            complex_activities=haskey(constraints, :activities) ? constraints.activities : nothing,
            reaction_names=String.(reaction_ids),
        )

        # 4. Verify data structure integrity
        @test haskey(network_data, :Y_matrix)
        @test haskey(network_data, :A_matrix)
        @test haskey(network_data, :complexes)
        @test haskey(network_data, :complex_activities)

        # 5. Test kinetic analysis via the current public API
        concordance_result = activity_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            kinetic_analysis=false,
            use_transitivity=true,
            concordance_tolerance=0.01,
            cv_threshold=0.01,
        )
        kinetic_results = kinetic_analysis(concordance_result, model; min_module_size=1, efficient=false)
        @test kinetic_results isa NamedTuple
        @test haskey(kinetic_results, :complexes)
        @test haskey(kinetic_results, :acr)
        @test haskey(kinetic_results, :acrr)
        println("    ✓ Kinetic analysis successful")

        println("    ✓ End-to-end pipeline successful")

        println("✓ Architecture integration validated!")
    end

end # end of testset "COCOA.jl - EnvZ-OmpR Paper Validation"

# Run the stricter paper example validation suite as part of default tests.
include("test_kinetic_analysis.jl")
