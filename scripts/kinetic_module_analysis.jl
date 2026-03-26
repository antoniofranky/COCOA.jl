"""
Kinetic Module Analysis Script

This script processes concordance analysis results (JLD2 files) and performs
kinetic module analysis with ACR/ACRR detection using efficient=true for
fast analysis.

Usage:
  Local testing (single file):
    julia kinetic_module_analysis.jl <result_file.jld2> <model_dir> <output_dir>

  SLURM array job:
    julia kinetic_module_analysis.jl <results_dir> <model_dir> <output_dir> <array_index>

Arguments:
  - result_file.jld2 or results_dir: JLD2 file or directory containing concordance results
  - model_dir: Directory containing the original model XML files
  - output_dir: Directory to save kinetic analysis results
  - array_index: (Optional) SLURM_ARRAY_TASK_ID for array jobs
"""

using COCOA, COBREXA, JLD2, Dates
using SBMLFBCModels, AbstractFBCModels

# Print thread configuration at startup
println("Julia version: $(VERSION)")
println("Julia threads: $(Threads.nthreads())")
if Threads.nthreads() == 1
    @warn "Running with single thread. For better performance, use: julia --threads=auto"
end

# Parse command line arguments
function parse_args()
    if length(ARGS) < 3
        error("""
        Usage:
          Single file: julia kinetic_module_analysis.jl <result_file.jld2> <model_dir> <output_dir>
          Array job:   julia kinetic_module_analysis.jl <results_dir> <model_dir> <output_dir> <array_index>
        """)
    end

    input_path = ARGS[1]
    model_dir = ARGS[2]
    output_dir = ARGS[3]
    array_index = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing

    return (input_path=input_path, model_dir=model_dir, output_dir=output_dir, array_index=array_index)
end

"""
Extract model name from JLD2 filename.
Expected format: kinetic_results_<model_name>_<params>.jld2
"""
function extract_model_name_from_filename(filename::String)
    # Remove directory path and extension
    basename_str = basename(filename)

    # Pattern: kinetic_results_<model_name>_42_100000_cv0p01_samples1000_transitivitytrue_tol_10.jld2
    # We need to extract <model_name> which can contain underscores

    # Remove prefix "kinetic_results_"
    if startswith(basename_str, "kinetic_results_")
        rest = basename_str[17:end]  # After "kinetic_results_"
    else
        error("Unexpected filename format: $basename_str")
    end

    # Find the pattern "_42_100000_" which marks the end of model name
    # The pattern is: _<seed>_<batch_size>_cv
    pattern_match = match(r"^(.+)_\d+_\d+_cv", rest)
    if pattern_match !== nothing
        return pattern_match.captures[1]
    else
        error("Could not extract model name from: $basename_str")
    end
end

"""
Find model file path given the model name and model directory.
"""
function find_model_file(model_name::String, model_dir::String)
    # Try different possible paths
    possible_paths = [
        joinpath(model_dir, "$(model_name).xml"),
        joinpath(model_dir, "random_0", "$(model_name).xml"),  # Nested structure
        joinpath(model_dir, model_name, "$(model_name).xml"),
    ]

    for path in possible_paths
        if isfile(path)
            return path
        end
    end

    error("Model file not found for $model_name. Searched in:\n  " * join(possible_paths, "\n  "))
end

"""
Process a single JLD2 result file and perform kinetic analysis.
"""
function process_result_file(result_file::String, model_dir::String, output_dir::String)
    println("="^60)
    println("Kinetic Module Analysis")
    println("="^60)
    println("Input file: $result_file")
    println("Model directory: $model_dir")
    println("Output directory: $output_dir")
    println("Timestamp: $(Dates.now())")
    println("Julia threads: $(Threads.nthreads())")
    println("="^60)

    # Load concordance results
    println("\n1. Loading concordance results...")
    data = JLD2.load(result_file)

    results = data["results"]
    stored_model_name = get(data, "model_name", nothing)
    stored_model_file = get(data, "model_file", nothing)
    analysis_params = get(data, "analysis_parameters", Dict())

    # Extract model name from filename if not stored
    model_name = if stored_model_name !== nothing
        stored_model_name
    else
        extract_model_name_from_filename(result_file)
    end

    println("   Model name: $model_name")
    println("   Stored model file: $stored_model_file")

    # Check if kinetic analysis was already performed
    if analysis_params !== nothing && get(analysis_params, "kinetic_analysis", false)
        println("\n   NOTE: Concordance results already include kinetic analysis.")
        println("   Re-running with efficient=false for thorough ACR/ACRR detection...")
    end

    # Find and load the original model
    println("\n2. Loading original model...")
    model_file = find_model_file(model_name, model_dir)
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

    # Run kinetic analysis with efficient=true
    println("\n4. Running kinetic module analysis (efficient=true)...")
    println("   Fast pairwise ACR/ACRR detection from stoichiometric differences")
    println("   Using $(Threads.nthreads()) threads for parallel operations")

    kinetic_timing = @timed begin
        kinetic_results = COCOA.kinetic_analysis(
            concordance_modules,
            model;
            min_module_size=1,
            efficient=true  # Fast pairwise detection, avoids matrix rank computation
        )
    end

    kinetic_duration = kinetic_timing.time
    kinetic_memory = kinetic_timing.bytes

    println("   Analysis completed in $(round(kinetic_duration, digits=2)) seconds")
    println("   Memory allocated: $(round(kinetic_memory / 1e6, digits=2)) MB")

    # Print results summary
    println("\n5. Results Summary:")
    println("   Kinetic modules: $(length(kinetic_results.kinetic_modules))")

    # Count modules by size
    sizes = [length(km) for km in kinetic_results.kinetic_modules]
    n_singletons = count(==(1), sizes)
    n_multi = count(>(1), sizes)
    if !isempty(sizes)
        max_size = maximum(sizes)
        println("   - Singleton modules: $n_singletons")
        println("   - Multi-complex modules: $n_multi")
        println("   - Largest module size: $max_size")
    end

    println("   ACR metabolites: $(length(kinetic_results.acr_metabolites))")
    if !isempty(kinetic_results.acr_metabolites)
        println("     Examples: $(join(first(kinetic_results.acr_metabolites, 5), ", "))")
    end

    println("   ACRR pairs: $(length(kinetic_results.acrr_pairs))")
    if !isempty(kinetic_results.acrr_pairs)
        example_pairs = first(kinetic_results.acrr_pairs, 3)
        for (m1, m2) in example_pairs
            println("     - ($m1, $m2)")
        end
    end

    # Construct output filename
    output_filename = "kinetic_modules_$(model_name).jld2"
    output_path = joinpath(output_dir, output_filename)

    # Create output directory if needed
    mkpath(output_dir)

    # Save results
    println("\n6. Saving results to: $output_path")

    JLD2.save(output_path,
        # Kinetic analysis results
        "kinetic_modules", kinetic_results.kinetic_modules,
        "acr_metabolites", kinetic_results.acr_metabolites,
        "acrr_pairs", kinetic_results.acrr_pairs,

        # Concordance modules (input to kinetic analysis)
        "concordance_modules", concordance_modules,

        # Metadata
        "model_name", model_name,
        "model_file", model_file,
        "concordance_result_file", result_file,

        # Statistics
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

        # Timing
        "timing", Dict(
            "kinetic_analysis_seconds" => kinetic_duration,
            "kinetic_analysis_memory_bytes" => kinetic_memory,
            "n_threads" => Threads.nthreads()
        ),

        # Analysis parameters
        "analysis_parameters", Dict(
            "efficient" => true,
            "min_module_size" => 1
        ), "timestamp", Dates.now();
        compress=true
    )

    println("\n" * "="^60)
    println("Kinetic module analysis completed successfully!")
    println("="^60)

    return output_path
end

"""
Get list of JLD2 files to process.
"""
function get_result_files(input_path::String, array_index::Union{Int,Nothing})
    if isfile(input_path)
        # Single file mode
        return [input_path]
    elseif isdir(input_path)
        # Directory mode - get all JLD2 files
        files = filter(f -> endswith(f, ".jld2") && startswith(basename(f), "kinetic_results_"),
            readdir(input_path, join=true))
        sort!(files)  # Ensure consistent ordering

        if array_index !== nothing
            # SLURM array mode - process only the file at array_index
            if array_index < 1 || array_index > length(files)
                error("Array index $array_index out of range. Total files: $(length(files))")
            end
            return [files[array_index]]
        else
            return files
        end
    else
        error("Input path does not exist: $input_path")
    end
end

# Main execution
function main()
    args = parse_args()

    println("Input path: $(args.input_path)")
    println("Model directory: $(args.model_dir)")
    println("Output directory: $(args.output_dir)")
    if args.array_index !== nothing
        println("Array index: $(args.array_index)")
    end

    # Get files to process
    result_files = get_result_files(args.input_path, args.array_index)
    println("\nFiles to process: $(length(result_files))")

    # Process each file
    for (i, result_file) in enumerate(result_files)
        println("\n" * "#"^70)
        println("Processing file $i/$(length(result_files)): $(basename(result_file))")
        println("#"^70)

        try
            output_path = process_result_file(result_file, args.model_dir, args.output_dir)
            println("✓ Successfully saved: $output_path")
        catch e
            println("\n✗ ERROR processing $(basename(result_file)):")
            println("  $e")

            # Save error log
            error_file = joinpath(args.output_dir, "error_$(basename(result_file)).txt")
            mkpath(args.output_dir)
            open(error_file, "w") do f
                println(f, "Error processing: $result_file")
                println(f, "Timestamp: $(Dates.now())")
                println(f, "Error: $e")
                println(f, "\nStacktrace:")
                for line in stacktrace(catch_backtrace())
                    println(f, "  $line")
                end
            end
            println("  Error log saved to: $error_file")

            # Re-throw in single file mode, continue in batch mode
            if length(result_files) == 1
                rethrow(e)
            end
        end
    end

    println("\n" * "="^70)
    println("All processing complete!")
    println("="^70)
end

# Run main
main()
