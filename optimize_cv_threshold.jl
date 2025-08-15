using COCOA, COBREXA, HiGHS, SBMLFBCModels, Distributed, SlurmClusterManager

# Set up cluster
addprocs(SlurmManager())
@everywhere using COCOA, HiGHS

# Load test model
model = load_model("/work/schaffran1/toolbox/prpd_models/Alloascoidea_hylecoeti.xml")

println("=== CV Threshold Optimization ===")
println("Model: Alloascoidea_hylecoeti.xml")
println("Reactions: $(length(AbstractFBCModels.reactions(model)))")
println("Metabolites: $(length(AbstractFBCModels.metabolites(model)))")

# Fixed parameters
const SAMPLE_SIZE = 200
const MAX_PAIRS_IN_MEMORY = 500_000
const COARSE_SAMPLE_SIZE = 40

# CV thresholds to test (strict to loose)
cv_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# Store results
results_table = []

println("\n=== Testing CV Thresholds ===")
println("Fixed parameters:")
println("  sample_size: $SAMPLE_SIZE")
println("  max_pairs_in_memory: $MAX_PAIRS_IN_MEMORY")
println("  coarse_sample_size: $COARSE_SAMPLE_SIZE")
println()

for (i, cv_threshold) in enumerate(cv_thresholds)
    coarse_cv_threshold = cv_threshold * 10  # 10x for coarse filtering
    
    println("[$i/$(length(cv_thresholds))] Testing CV threshold: $cv_threshold (coarse: $coarse_cv_threshold)")
    
    start_time = time()
    
    try
        result = kinetic_concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            sample_size=SAMPLE_SIZE,
            coarse_sample_size=COARSE_SAMPLE_SIZE,
            coarse_cv_threshold=coarse_cv_threshold,
            cv_threshold=cv_threshold,
            max_pairs_in_memory=MAX_PAIRS_IN_MEMORY,
            include_kinetic_modules=true,
            include_robustness=true,
            min_module_size=2
        )
        
        elapsed = time() - start_time
        
        # Extract key metrics
        summary = result.summary
        metrics = Dict(
            "cv_threshold" => cv_threshold,
            "coarse_cv_threshold" => coarse_cv_threshold,
            "runtime_sec" => round(elapsed, digits=1),
            "n_concordant_pairs" => summary["n_concordant_pairs"],
            "n_balanced_complexes" => summary["n_balanced_complexes"],
            "n_concordance_modules" => summary["n_concordance_modules"],
            "n_kinetic_modules" => summary["n_kinetic_modules"],
            "giant_kinetic_module_size" => summary["giant_kinetic_module_size"],
            "n_metabolites_absolute_robust" => summary["n_metabolites_absolute_robust"],
            "n_metabolite_pairs_ratio_robust" => summary["n_metabolite_pairs_ratio_robust"]
        )
        
        push!(results_table, metrics)
        
        println("  ✓ SUCCESS in $(metrics["runtime_sec"])s")
        println("    Concordant pairs: $(metrics["n_concordant_pairs"])")
        println("    Kinetic modules: $(metrics["n_kinetic_modules"])")
        println("    Giant module size: $(metrics["giant_kinetic_module_size"])")
        
    catch e
        elapsed = time() - start_time
        println("  ✗ FAILED after $(round(elapsed, digits=1))s: $e")
        
        # Still record the failure
        push!(results_table, Dict(
            "cv_threshold" => cv_threshold,
            "coarse_cv_threshold" => coarse_cv_threshold,
            "runtime_sec" => round(elapsed, digits=1),
            "error" => string(e),
            "n_concordant_pairs" => missing,
            "n_balanced_complexes" => missing,
            "n_concordance_modules" => missing,
            "n_kinetic_modules" => missing,
            "giant_kinetic_module_size" => missing,
            "n_metabolites_absolute_robust" => missing,
            "n_metabolite_pairs_ratio_robust" => missing
        ))
    end
    
    println()
end

# Analysis and recommendations
println("=== Results Summary ===")
println("CV_Threshold | Concordant_Pairs | Kinetic_Modules | Giant_Size | Runtime_sec")
println("-------------|------------------|-----------------|------------|------------")

for result in results_table
    if haskey(result, "error")
        println("$(rpad(result["cv_threshold"], 12)) | FAILED: $(result["error"])")
    else
        println("$(rpad(result["cv_threshold"], 12)) | $(rpad(result["n_concordant_pairs"], 16)) | $(rpad(result["n_kinetic_modules"], 15)) | $(rpad(result["giant_kinetic_module_size"], 10)) | $(result["runtime_sec"])")
    end
end

# Find optimal threshold
successful_results = filter(r -> !haskey(r, "error"), results_table)

if length(successful_results) >= 2
    println("\n=== Threshold Analysis ===")
    
    # Calculate relative changes in concordant pairs
    for i in 2:length(successful_results)
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
    
    for i in 2:length(successful_results)
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
    
    optimal_result = successful_results[min_change_idx]
    println("\nOptimal CV threshold: $(optimal_result["cv_threshold"])")
    println("  - Concordant pairs: $(optimal_result["n_concordant_pairs"])")
    println("  - Runtime: $(optimal_result["runtime_sec"])s")
    println("  - Change from stricter threshold: $(round(min_change, digits=1))%")
else
    println("\n=== WARNING ===")
    println("Not enough successful runs to determine optimal threshold.")
    println("Consider adjusting the CV threshold range or checking for errors.")
end

println("\n=== Final Parameters for 343-Model Analysis ===")
if length(successful_results) > 0
    best_result = successful_results[min_change_idx]
    println("Recommended parameters:")
    println("  cv_threshold = $(best_result["cv_threshold"])")
    println("  coarse_cv_threshold = $(best_result["coarse_cv_threshold"])")
    println("  sample_size = $SAMPLE_SIZE")
    println("  max_pairs_in_memory = $MAX_PAIRS_IN_MEMORY")
    
    # Estimate total runtime for 343 models
    total_time_hours = round(343 * best_result["runtime_sec"] / 3600, digits=1)
    println("  Estimated total runtime for 343 models: $(total_time_hours) hours")
end