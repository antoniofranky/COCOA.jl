using JLD2, DataFrames, CSV, Dates

"""
Script to collect and summarize results from COCOA array job runs.
Usage: julia collect_array_results.jl <results_directory> [output_prefix]
"""

function collect_array_results(results_dir::String, output_prefix::String="cocoa_summary")
    println("Collecting COCOA array results from: $results_dir")
    
    # Find all result files
    result_files = []
    error_files = []
    
    for file in readdir(results_dir)
        if startswith(file, "concordance_results_") && endswith(file, ".jld2")
            push!(result_files, joinpath(results_dir, file))
        elseif startswith(file, "error_") && endswith(file, ".txt")
            push!(error_files, joinpath(results_dir, file))
        end
    end
    
    n_results = length(result_files)
    n_errors = length(error_files)
    
    println("Found $n_results successful results and $n_errors errors")
    
    if n_results == 0
        println("No result files found!")
        return
    end
    
    # Collect summary statistics
    summary_data = []
    detailed_stats = []
    
    for result_file in result_files
        try
            println("Processing: $(basename(result_file))")
            
            # Load result file
            data = JLD2.load(result_file)
            results = data["results"]
            model_name = get(data, "model_name", "unknown")
            analysis_duration = get(data, "analysis_duration_seconds", NaN)
            analysis_params = get(data, "analysis_parameters", Dict())
            
            stats = results.stats
            
            # Extract summary info
            summary_row = (
                model_name = model_name,
                analysis_duration_min = round(analysis_duration / 60, digits=2),
                n_complexes = stats["n_complexes"],
                n_balanced = stats["n_balanced"],
                n_trivially_balanced = stats["n_trivially_balanced"],
                n_concordant_total = stats["n_concordant_total"],
                n_concordant_opt = stats["n_concordant_opt"],
                n_concordant_inferred = stats["n_concordant_inferred"],
                n_trivial_pairs = stats["n_trivial_pairs"],
                n_non_concordant_pairs = stats["n_non_concordant_pairs"],
                n_modules = stats["n_modules"],
                n_candidate_pairs = stats["n_candidate_pairs"],
                n_candidates_skipped_by_transitivity = stats["n_candidates_skipped_by_transitivity"],
                batches_completed = stats["batches_completed"],
                n_timeout_pairs = stats["n_timeout_pairs"],
                validation_passed = stats["validation_passed"],
                sample_size = get(analysis_params, "sample_size", NaN),
                cv_threshold = get(analysis_params, "cv_threshold", NaN),
                batch_size = get(analysis_params, "batch_size", NaN),
                use_transitivity = get(analysis_params, "use_transitivity", missing),
                concordance_tolerance = get(analysis_params, "concordance_tolerance", NaN),
                balanced_tolerance = get(analysis_params, "balanced_tolerance", NaN),
                seed = get(analysis_params, "seed", NaN)
            )
            
            push!(summary_data, summary_row)
            
            # Store full stats for detailed analysis
            full_stats = merge(Dict("model_name" => model_name), stats)
            push!(detailed_stats, full_stats)
            
        catch e
            println("ERROR processing $result_file: $e")
        end
    end
    
    # Create DataFrames
    summary_df = DataFrame(summary_data)
    detailed_df = DataFrame(detailed_stats)
    
    # Sort by model name
    sort!(summary_df, :model_name)
    sort!(detailed_df, :model_name)
    
    # Generate timestamp for output files
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Save summary results
    summary_file = "$(output_prefix)_summary_$(timestamp).csv"
    detailed_file = "$(output_prefix)_detailed_$(timestamp).csv"
    
    CSV.write(joinpath(results_dir, summary_file), summary_df)
    CSV.write(joinpath(results_dir, detailed_file), detailed_df)
    
    println("\nResults Summary:")
    println("="^60)
    println("Total models processed: $(nrow(summary_df))")
    println("Total models with errors: $n_errors")
    
    if nrow(summary_df) > 0
        println("\nAnalysis Statistics:")
        println("  Average analysis time: $(round(mean(skipmissing(summary_df.analysis_duration_min)), digits=2)) minutes")
        println("  Total analysis time: $(round(sum(skipmissing(summary_df.analysis_duration_min)), digits=2)) minutes")
        println("  Average complexes per model: $(round(mean(summary_df.n_complexes), digits=0))")
        println("  Average concordant pairs per model: $(round(mean(summary_df.n_concordant_total), digits=0))")
        println("  Average modules per model: $(round(mean(summary_df.n_modules), digits=1))")
        
        # Validation summary
        n_valid = sum(summary_df.validation_passed)
        n_invalid = sum(.!summary_df.validation_passed)
        println("  Models with validation passed: $n_valid")
        println("  Models with validation failed: $n_invalid")
        
        # Top models by concordant pairs
        println("\nTop 5 models by total concordant pairs:")
        top_models = sort(summary_df, :n_concordant_total, rev=true)[1:min(5, nrow(summary_df)), :]
        for row in eachrow(top_models)
            println("  $(row.model_name): $(row.n_concordant_total) pairs")
        end
    end
    
    # Error summary
    if n_errors > 0
        println("\nError Summary:")
        println("  Error files found: $n_errors")
        println("  Check error files for details:")
        for error_file in error_files
            println("    $(basename(error_file))")
        end
    end
    
    println("\nOutput files saved:")
    println("  Summary: $summary_file")  
    println("  Detailed: $detailed_file")
    println("="^60)
    
    return summary_df, detailed_df
end

# Main execution
if length(ARGS) < 1
    error("Usage: julia collect_array_results.jl <results_directory> [output_prefix]")
end

results_dir = ARGS[1]
output_prefix = length(ARGS) >= 2 ? ARGS[2] : "cocoa_summary"

if !isdir(results_dir)
    error("Results directory does not exist: $results_dir")
end

# Collect results
summary_df, detailed_df = collect_array_results(results_dir, output_prefix)

println("\nCollection completed successfully!")