using Test
using Distributed
# Add processes if not already available
if nprocs() < 16
    addprocs(16 - nprocs())
end

# Load packages on all workers
@everywhere using COCOA
@everywhere using COBREXA
@everywhere using SBMLFBCModels
@everywhere using HiGHS
@everywhere using GLPK

@testset "COCOA.jl" begin

    @testset "Basic concordance analysis on E. coli core" begin
        # Download the E. coli core model if not available
        if !isfile("e_coli_core.xml")
            model_file = download_model(
                "http://bigg.ucsd.edu/static/models/e_coli_core.xml",
                "e_coli_core.xml",
                "b4db506aeed0e434c1f5f1fdd35feda0dfe5d82badcfda0e9d1342335ab31116"
            )
        end

        # Load the model
        model = load_model("e_coli_core.xml")

        # Run concordance analysis with small sample size for testing
        results = find_concordant_complexes(
            model;
            optimizer=GLPK.Optimizer,  # HiGHS is better for LP problems
            workers=workers(),          # Use all available workers
            sample_size=10,             # Small sample size for quick testing
            correlation_threshold=0.95,
            batch_size=100,             # Smaller batch size for testing
            seed=42                     # Fixed seed for reproducibility
        )

        # Test that results have the expected structure
        @test results isa ConcordanceResults
        @test hasproperty(results, :complexes)
        @test hasproperty(results, :pairs)
        @test hasproperty(results, :modules)
        @test hasproperty(results, :metadata)

        # Test that we found some complexes
        @test nrow(results.complexes) > 0
        @test results.metadata["total_complexes"] > 0

        # Test that complex DataFrame has expected columns
        @test "complex_id" in names(results.complexes)
        @test "module_id" in names(results.complexes)
        @test "min_activity" in names(results.complexes)
        @test "max_activity" in names(results.complexes)
        @test "is_balanced" in names(results.complexes)

        # Test that pairs DataFrame has expected columns (if any pairs found)
        if nrow(results.pairs) > 0
            @test "complex1" in names(results.pairs)
            @test "complex2" in names(results.pairs)
            @test "is_trivial" in names(results.pairs)
            @test "lambda_value" in names(results.pairs)
        end

        # Test that modules DataFrame has expected columns
        @test "module_id" in names(results.modules)
        @test "size" in names(results.modules)
        @test "complexes" in names(results.modules)

        # Test that balanced complexes have zero activity
        balanced_mask = results.complexes.is_balanced
        if any(balanced_mask)
            @test all(abs.(results.complexes.min_activity[balanced_mask]) .< 1e-9)
            @test all(abs.(results.complexes.max_activity[balanced_mask]) .< 1e-9)
        end

        # Test metadata
        @test results.metadata["total_complexes"] == size(results.complexes, 1)
        @test results.metadata["balanced_complexes"] >= 0
        @test results.metadata["concordant_pairs"] >= 0
        @test results.metadata["trivial_pairs"] >= 0
        @test results.metadata["modules"] == nrow(results.modules)

        @info "Concordance analysis results:" complexes = results.metadata["total_complexes"] balanced = results.metadata["balanced_complexes"] pairs = results.metadata["concordant_pairs"] modules = results.metadata["modules"]
    end

    # Clean up extra processes
    if nprocs() > 1
        rmprocs(workers())
    end
end # end of testset "COCOA.jl"