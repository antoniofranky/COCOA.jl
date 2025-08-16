using JSON

println("=== CV Threshold Optimization Results Collection ===")

# CV thresholds that were tested
cv_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# Collect results from all array jobs
results_table = []
missing_files = []

for i in 1:length(cv_thresholds)
    result_file = "cv_threshold_result_$(i).json"
    
    if isfile(result_file)
        try
            result = JSON.parsefile(result_file)
            push!(results_table, result)
            println("✓ Loaded results from task $i (CV threshold: $(result["cv_threshold"]))")
        catch e
            println("✗ Error reading $result_file: $e")
            push!(missing_files, result_file)
        end
    else
        println("✗ Missing result file: $result_file")
        push!(missing_files, result_file)
    end
end

if length(missing_files) > 0
    println("\n=== WARNING ===")
    println("Missing $(length(missing_files)) result files:")
    for file in missing_files
        println("  - $file")
    end
    println("Make sure all array jobs completed successfully.")
end

if length(results_table) == 0
    println("\n=== ERROR ===")
    println("No results found. Check that array jobs completed and generated output files.")
    exit(1)
end

# Sort results by CV threshold
sort!(results_table, by = r -> r["cv_threshold"])

# Analysis and recommendations
println("\n=== Results Summary ===")
println("CV_Threshold | Status   | Concordant_Pairs | Kinetic_Modules | Giant_Size | Runtime_sec")
println("-------------|----------|------------------|-----------------|------------|------------")

for result in results_table
    status = get(result, "status", "unknown")
    if status == "failed" || haskey(result, "error")
        error_msg = get(result, "error", "unknown error")
        # Truncate long error messages
        if length(error_msg) > 30
            error_msg = error_msg[1:27] * "..."
        end
        println("$(rpad(result["cv_threshold"], 12)) | $(rpad("FAILED", 8)) | $(rpad(error_msg, 16)) | $(rpad("-", 15)) | $(rpad("-", 10)) | $(result["runtime_sec"])")
    else
        concordant = get(result, "n_concordant_pairs", "missing")
        modules = get(result, "n_kinetic_modules", "missing")
        giant = get(result, "giant_kinetic_module_size", "missing")
        runtime = get(result, "runtime_sec", "missing")
        
        println("$(rpad(result["cv_threshold"], 12)) | $(rpad("SUCCESS", 8)) | $(rpad(concordant, 16)) | $(rpad(modules, 15)) | $(rpad(giant, 10)) | $runtime")
    end
end

# Find successful results for analysis
successful_results = filter(r -> get(r, "status", "") == "success" && !haskey(r, "error"), results_table)

if length(successful_results) >= 2
    println("\n=== Threshold Analysis ===")

    # Calculate relative changes in concordant pairs
    for i in eachindex(successful_results)[2:end]
        prev_pairs = successful_results[i-1]["n_concordant_pairs"]
        curr_pairs = successful_results[i]["n_concordant_pairs"]

        if prev_pairs > 0
            change_pct = round(100 * (curr_pairs - prev_pairs) / prev_pairs, digits=1)
            println("CV $(successful_results[i-1]["cv_threshold"]) → $(successful_results[i]["cv_threshold"]): $(change_pct)% change in concordant pairs")
        end
    end

    # Recommend threshold where change is < 5%
    println("\n=== Recommendation ===")
    println("Look for the CV threshold where:")
    println("1. Change in concordant pairs < 5% compared to stricter threshold")
    println("2. Runtime is reasonable for your 343-model analysis")
    println("3. Giant kinetic module size stabilizes")

    # Find the "elbow point"
    min_change_idx = 1
    min_change = Inf

    for i in eachindex(successful_results)[2:end]
        prev_pairs = successful_results[i-1]["n_concordant_pairs"]
        curr_pairs = successful_results[i]["n_concordant_pairs"]

        if prev_pairs > 0
            change_pct = abs(100 * (curr_pairs - prev_pairs) / prev_pairs)
            if change_pct < min_change && change_pct < 5.0
                min_change = change_pct
                min_change_idx = i
            end
        end
    end

    if min_change_idx > 1 && min_change < Inf
        optimal_result = successful_results[min_change_idx]
        println("\nOptimal CV threshold: $(optimal_result["cv_threshold"])")
        println("  - Concordant pairs: $(optimal_result["n_concordant_pairs"])")
        println("  - Runtime: $(optimal_result["runtime_sec"])s")
        println("  - Change from stricter threshold: $(round(min_change, digits=1))%")
    else
        # If no clear elbow, recommend the middle successful result
        mid_idx = div(length(successful_results), 2) + 1
        optimal_result = successful_results[min(mid_idx, end)]
        println("\nNo clear elbow point found. Recommended middle threshold: $(optimal_result["cv_threshold"])")
        println("  - Concordant pairs: $(optimal_result["n_concordant_pairs"])")
        println("  - Runtime: $(optimal_result["runtime_sec"])s")
    end
else
    println("\n=== WARNING ===")
    println("Not enough successful runs ($(length(successful_results))) to determine optimal threshold.")
    println("Consider adjusting the CV threshold range or checking for errors.")
    
    if length(successful_results) == 1
        optimal_result = successful_results[1]
        println("\nOnly successful threshold: $(optimal_result["cv_threshold"])")
        println("  - Concordant pairs: $(optimal_result["n_concordant_pairs"])")
        println("  - Runtime: $(optimal_result["runtime_sec"])s")
    end
end

println("\n=== Final Parameters for 343-Model Analysis ===")
if length(successful_results) > 0
    best_result = successful_results[min_change_idx]
    println("Recommended parameters:")
    println("  cv_threshold = $(best_result["cv_threshold"])")
    println("  coarse_cv_threshold = $(best_result["coarse_cv_threshold"])")
    println("  sample_size = 200")
    println("  max_pairs_in_memory = 1_250_000")

    # Estimate total runtime for 343 models
    total_time_hours = round(343 * best_result["runtime_sec"] / 3600, digits=1)
    println("  Estimated total runtime for 343 models: $(total_time_hours) hours")
    
    # Save recommended parameters to file
    recommended_params = Dict(
        "cv_threshold" => best_result["cv_threshold"],
        "coarse_cv_threshold" => best_result["coarse_cv_threshold"],
        "sample_size" => 200,
        "max_pairs_in_memory" => 1_250_000,
        "estimated_runtime_hours_343_models" => total_time_hours,
        "optimization_date" => string(now())
    )
    
    open("recommended_cv_parameters.json", "w") do f
        JSON.print(f, recommended_params, 2)
    end
    println("\nRecommended parameters saved to: recommended_cv_parameters.json")
end