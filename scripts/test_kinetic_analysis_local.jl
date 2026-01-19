"""
Local Testing Script for Kinetic Module Analysis

This script tests the kinetic module analysis on a single model locally
before submitting to SLURM.

Usage:
    julia --threads=auto --project=. test_kinetic_analysis_local.jl

Or with specific file:
    julia --threads=auto --project=. test_kinetic_analysis_local.jl <result_file.jld2>

Note: Use --threads=auto or -t auto to use all available CPU threads.
"""

using Pkg
# Ensure we're using the COCOA project
if !isfile("Project.toml")
    cd(dirname(dirname(@__FILE__)))
end
Pkg.activate(".")

using COCOA, COBREXA, JLD2, Dates
using SBMLFBCModels, AbstractFBCModels

# Print thread configuration
println("Julia threads: $(Threads.nthreads())")
if Threads.nthreads() == 1
    @warn "Running with single thread. For better performance, restart with: julia --threads=auto"
end

# Configuration - adjust these paths for local testing
const LOCAL_CONFIG = (
    # Directory containing concordance result JLD2 files
    results_dir = raw"C:\Users\anton\master-thesis\toolbox\results\1e-10\extracted\work\schaffran1\jobresults\1e-10\random_0",

    # Directory containing original model XML files
    models_dir = raw"C:\Users\anton\master-thesis\toolbox\prpd_models\random_0\random_0",

    # Output directory for kinetic analysis results
    output_dir = raw"C:\Users\anton\master-thesis\toolbox\results\kinetic_analysis\random_0"
)

"""
List available result files for testing.
"""
function list_available_files()
    files = filter(f -> endswith(f, ".jld2") && startswith(basename(f), "kinetic_results_"),
                   readdir(LOCAL_CONFIG.results_dir, join=true))
    sort!(files)
    return files
end

"""
Run kinetic analysis on a single result file.
"""
function test_single_file(result_file::String)
    println("=" ^ 70)
    println("LOCAL TEST: Kinetic Module Analysis")
    println("=" ^ 70)
    println("Result file: $result_file")
    println("Models dir: $(LOCAL_CONFIG.models_dir)")
    println("Output dir: $(LOCAL_CONFIG.output_dir)")
    println()

    # Include the main analysis script functions
    include(joinpath(@__DIR__, "kinetic_module_analysis.jl"))
end

"""
Interactive file selection for local testing.
"""
function interactive_test()
    files = list_available_files()

    println("=" ^ 70)
    println("Kinetic Module Analysis - Local Testing")
    println("=" ^ 70)
    println()
    println("Found $(length(files)) result files in:")
    println("  $(LOCAL_CONFIG.results_dir)")
    println()

    # Show first few files
    println("Available files (showing first 10):")
    for (i, f) in enumerate(first(files, 10))
        println("  $i. $(basename(f))")
    end
    if length(files) > 10
        println("  ... and $(length(files) - 10) more")
    end
    println()

    # Test with first file by default
    println("Testing with first file: $(basename(files[1]))")
    println()

    # Run the test
    run_analysis(files[1])
end

"""
Run the kinetic analysis on a result file.
"""
function run_analysis(result_file::String)
    println("=" ^ 60)
    println("Kinetic Module Analysis")
    println("=" ^ 60)
    println("Input file: $result_file")
    println("Model directory: $(LOCAL_CONFIG.models_dir)")
    println("Output directory: $(LOCAL_CONFIG.output_dir)")
    println("Timestamp: $(Dates.now())")
    println("Julia threads: $(Threads.nthreads())")
    println("=" ^ 60)

    # Load concordance results
    println("\n1. Loading concordance results...")
    data = JLD2.load(result_file)

    results = data["results"]
    stored_model_name = get(data, "model_name", nothing)
    stored_model_file = get(data, "model_file", nothing)
    analysis_params = get(data, "analysis_parameters", Dict())

    # Extract model name
    model_name = if stored_model_name !== nothing
        stored_model_name
    else
        # Extract from filename
        basename_str = basename(result_file)
        rest = basename_str[17:end]  # After "kinetic_results_"
        m = match(r"^(.+)_\d+_\d+_cv", rest)
        m !== nothing ? m.captures[1] : error("Could not extract model name")
    end

    println("   Model name: $model_name")
    println("   Stored model file: $stored_model_file")

    # Find and load the original model
    println("\n2. Loading original model...")
    model_file = joinpath(LOCAL_CONFIG.models_dir, "$(model_name).xml")

    if !isfile(model_file)
        error("Model file not found: $model_file")
    end
    println("   Found model at: $model_file")

    model = COBREXA.load_model(model_file)
    n_reactions = length(AbstractFBCModels.reactions(model))
    n_metabolites = length(AbstractFBCModels.metabolites(model))
    println("   Loaded: $n_reactions reactions, $n_metabolites metabolites")

    # Extract concordance modules from results
    println("\n3. Extracting concordance modules...")
    concordance_modules = COCOA.extract_concordance_modules(results)

    n_balanced = length(concordance_modules[1])
    n_unbalanced_modules = length(concordance_modules) - 1
    n_total_complexes = sum(length, concordance_modules)

    println("   Total complexes: $n_total_complexes")
    println("   Balanced complexes: $n_balanced")
    println("   Unbalanced concordance modules: $n_unbalanced_modules")

    # Show concordance module sizes
    println("\n   Module size distribution:")
    sizes = [length(cm) for cm in concordance_modules[2:end]]
    if !isempty(sizes)
        println("     Min: $(minimum(sizes)), Max: $(maximum(sizes)), Mean: $(round(sum(sizes)/length(sizes), digits=1))")
    end

    # Run kinetic analysis with efficient=false
    println("\n4. Running kinetic module analysis (efficient=false)...")
    println("   This performs thorough ACR/ACRR detection using Propositions S3-5 and S3-6")
    println("   Using $(Threads.nthreads()) threads for parallel operations")

    kinetic_timing = @timed begin
        kinetic_results = COCOA.kinetic_analysis(
            concordance_modules,
            model;
            min_module_size=1,
            efficient=false
        )
    end

    kinetic_duration = kinetic_timing.time
    kinetic_memory = kinetic_timing.bytes

    println("   Analysis completed in $(round(kinetic_duration, digits=2)) seconds")
    println("   Memory allocated: $(round(kinetic_memory / 1e6, digits=2)) MB")

    # Print results summary
    println("\n5. Results Summary:")
    println("   Kinetic modules: $(length(kinetic_results.kinetic_modules))")

    sizes = [length(km) for km in kinetic_results.kinetic_modules]
    n_singletons = count(==(1), sizes)
    n_multi = count(>(1), sizes)
    if !isempty(sizes)
        max_size = maximum(sizes)
        println("   - Singleton modules: $n_singletons")
        println("   - Multi-complex modules: $n_multi")
        println("   - Largest module size: $max_size")

        # Show largest modules
        if max_size > 1
            println("\n   Largest kinetic modules:")
            sorted_modules = sort(kinetic_results.kinetic_modules, by=length, rev=true)
            for (i, km) in enumerate(first(sorted_modules, 3))
                if length(km) > 1
                    println("     $i. Size $(length(km)): $(join(first(collect(km), 5), ", "))$(length(km) > 5 ? "..." : "")")
                end
            end
        end
    end

    println("\n   ACR metabolites: $(length(kinetic_results.acr_metabolites))")
    if !isempty(kinetic_results.acr_metabolites)
        println("     $(join(first(kinetic_results.acr_metabolites, 10), ", "))$(length(kinetic_results.acr_metabolites) > 10 ? "..." : "")")
    end

    println("\n   ACRR pairs: $(length(kinetic_results.acrr_pairs))")
    if !isempty(kinetic_results.acrr_pairs)
        for (m1, m2) in first(kinetic_results.acrr_pairs, 5)
            println("     - ($m1, $m2)")
        end
        if length(kinetic_results.acrr_pairs) > 5
            println("     ... and $(length(kinetic_results.acrr_pairs) - 5) more")
        end
    end

    # Save results
    mkpath(LOCAL_CONFIG.output_dir)
    output_filename = "kinetic_modules_$(model_name)_efficient_false.jld2"
    output_path = joinpath(LOCAL_CONFIG.output_dir, output_filename)

    println("\n6. Saving results to: $output_path")

    JLD2.save(output_path,
        "kinetic_modules", kinetic_results.kinetic_modules,
        "acr_metabolites", kinetic_results.acr_metabolites,
        "acrr_pairs", kinetic_results.acrr_pairs,
        "concordance_modules", concordance_modules,
        "model_name", model_name,
        "model_file", model_file,
        "concordance_result_file", result_file,
        "statistics", Dict(
            "n_kinetic_modules" => length(kinetic_results.kinetic_modules),
            "n_singleton_modules" => n_singletons,
            "n_multi_complex_modules" => n_multi,
            "largest_module_size" => isempty(sizes) ? 0 : maximum(sizes),
            "n_acr_metabolites" => length(kinetic_results.acr_metabolites),
            "n_acrr_pairs" => length(kinetic_results.acrr_pairs),
            "n_total_complexes" => n_total_complexes,
            "n_balanced_complexes" => n_balanced,
            "n_unbalanced_modules" => n_unbalanced_modules
        ),
        "timing", Dict(
            "kinetic_analysis_seconds" => kinetic_duration,
            "kinetic_analysis_memory_bytes" => kinetic_memory,
            "n_threads" => Threads.nthreads()
        ),
        "analysis_parameters", Dict(
            "efficient" => false,
            "min_module_size" => 1
        ),
        "timestamp", Dates.now();
        compress=true
    )

    println("\n" * "=" ^ 60)
    println("✓ Kinetic module analysis completed successfully!")
    println("=" ^ 60)

    return (
        kinetic_modules=kinetic_results.kinetic_modules,
        acr_metabolites=kinetic_results.acr_metabolites,
        acrr_pairs=kinetic_results.acrr_pairs,
        output_path=output_path
    )
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        # Process specific file
        run_analysis(ARGS[1])
    else
        # Interactive mode
        interactive_test()
    end
end
