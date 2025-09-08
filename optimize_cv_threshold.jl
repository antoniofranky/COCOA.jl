using COCOA, COBREXA, HiGHS, SBMLFBCModels, Distributed, JSON, AbstractFBCModels
@everywhere using COCOA, HiGHS

# Get array task ID from command line argument
if length(ARGS) < 1
    error("Usage: julia optimize_cv_threshold.jl <array_task_id>")
end

array_task_id = parse(Int, ARGS[1])

# Load test model
model = load_model("/work/schaffran1/toolbox/prpd_models/Alloascoidea_hylecoeti.xml")

println("=== CV Threshold Optimization - Task $array_task_id ===")
println("Model: Alloascoidea_hylecoeti.xml")
println("Reactions: $(length(AbstractFBCModels.reactions(model)))")
println("Metabolites: $(length(AbstractFBCModels.metabolites(model)))")

# Fixed parameters
const SAMPLE_SIZE = 200
const MAX_PAIRS_IN_MEMORY = 1_250_000
const COARSE_SAMPLE_SIZE = 40

# CV thresholds to test (strict to loose)
cv_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# Select the CV threshold for this array task
if array_task_id < 1 || array_task_id > length(cv_thresholds)
    error("Array task ID $array_task_id is out of range (1-$(length(cv_thresholds)))")
end

cv_threshold = cv_thresholds[array_task_id]
coarse_cv_threshold = cv_threshold * 10

println("\n=== Testing CV Threshold $cv_threshold ===")
println("Fixed parameters:")
println("  sample_size: $SAMPLE_SIZE")
println("  max_pairs_in_memory: $MAX_PAIRS_IN_MEMORY")
println("  coarse_sample_size: $COARSE_SAMPLE_SIZE")
println("  cv_threshold: $cv_threshold")
println("  coarse_cv_threshold: $coarse_cv_threshold")
println()

start_time = time()

try
    result = activity_concordance_analysis(
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
        "array_task_id" => array_task_id,
        "cv_threshold" => cv_threshold,
        "coarse_cv_threshold" => coarse_cv_threshold,
        "runtime_sec" => round(elapsed, digits=1),
        "n_concordant_pairs" => summary["n_concordant_pairs"],
        "n_balanced_complexes" => summary["n_balanced_complexes"],
        "n_concordance_modules" => summary["n_concordance_modules"],
        "n_kinetic_modules" => summary["n_kinetic_modules"],
        "giant_kinetic_module_size" => summary["giant_kinetic_module_size"],
        "n_metabolites_absolute_robust" => summary["n_metabolites_absolute_robust"],
        "n_metabolite_pairs_ratio_robust" => summary["n_metabolite_pairs_ratio_robust"],
        "status" => "success"
    )

    println("✓ SUCCESS in $(metrics["runtime_sec"])s")
    println("  Concordant pairs: $(metrics["n_concordant_pairs"])")
    println("  Kinetic modules: $(metrics["n_kinetic_modules"])")
    println("  Giant module size: $(metrics["giant_kinetic_module_size"])")

    # Save results to JSON file
    output_file = "cv_threshold_result_$(array_task_id).json"
    open(output_file, "w") do f
        JSON.print(f, metrics, 2)
    end
    println("\nResults saved to: $output_file")

catch e
    elapsed = time() - start_time
    println("✗ FAILED after $(round(elapsed, digits=1))s: $e")

    # Record the failure
    metrics = Dict(
        "array_task_id" => array_task_id,
        "cv_threshold" => cv_threshold,
        "coarse_cv_threshold" => coarse_cv_threshold,
        "runtime_sec" => round(elapsed, digits=1),
        "error" => string(e),
        "status" => "failed",
        "n_concordant_pairs" => missing,
        "n_balanced_complexes" => missing,
        "n_concordance_modules" => missing,
        "n_kinetic_modules" => missing,
        "giant_kinetic_module_size" => missing,
        "n_metabolites_absolute_robust" => missing,
        "n_metabolite_pairs_ratio_robust" => missing
    )

    # Save error results to JSON file
    output_file = "cv_threshold_result_$(array_task_id).json"
    open(output_file, "w") do f
        JSON.print(f, metrics, 2)
    end
    println("\nError results saved to: $output_file")
end

println("\n=== Task $array_task_id Completed ===")
println("CV threshold $cv_threshold analysis finished.")
println("\nTo collect all results, run the analysis script after all array jobs complete.")