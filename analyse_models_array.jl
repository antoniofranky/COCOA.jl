using Distributed
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2
@everywhere using COCOA, HiGHS

# Parse command line arguments
if length(ARGS) < 3
    error("Usage: julia analyse_models_array.jl <model_file> <results_dir> <model_name>")
end

model_file = ARGS[1]
results_dir = ARGS[2]
model_name = ARGS[3]

# --- Analysis Parameters ---
# Modify these parameters as needed
sample_size = 1000
seed = 42
cv_threshold = 0.01
batch_size = 100_000
use_transitivity = true


# HiGHS solver settings
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("random_seed", seed),
    COBREXA.set_optimizer_attribute("time_limit", 1200.0),  # 20 minutes per optimization
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

# Construct output path
output_filename = "concordance_results_" * model_name * "_" *
                  lpad(string(seed), 2, "0") * "_" *
                  string(batch_size) * "_cv" *
                  replace(string(cv_threshold), "." => "p") * "_samples" *
                  string(sample_size) *
                  "_transitivity" * string(use_transitivity) * ".jld2"

output_path = joinpath(results_dir, output_filename)

# Log analysis start
println("="^60)
println("COCOA Concordance Analysis")
println("="^60)
println("Model: $model_name")
println("Input file: $model_file")
println("Output file: $output_path")
println("Parameters:")
println("  Sample size: $sample_size")
println("  Seed: $seed")
println("  CV threshold: $cv_threshold")
println("  Batch size: $batch_size")
println("  Use transitivity: $use_transitivity")
println("="^60)

# Check if model file exists
if !isfile(model_file)
    error("Model file not found: $model_file")
end

# Create results directory if it doesn't exist
mkpath(results_dir)

try
    # Load the model
    println("Loading model: $model_file")
    model = COBREXA.load_model(model_file)

    # Get basic model info
    n_reactions = length(AbstractFBCModels.reactions(model))
    n_metabolites = length(AbstractFBCModels.metabolites(model))
    println("Model loaded successfully:")
    println("  Reactions: $n_reactions")
    println("  Metabolites: $n_metabolites")

    # Run concordance analysis
    println("\nStarting concordance analysis...")
    analysis_start_time = time()

    results = COCOA.kinetic_concordance_analysis(
        model;
        optimizer=HiGHS.Optimizer,
        settings=highs_settings,
        sample_size=sample_size,
        seed=seed,
        cv_threshold=cv_threshold,
        batch_size=batch_size,
        use_transitivity=use_transitivity
    )

    analysis_end_time = time()
    analysis_duration = analysis_end_time - analysis_start_time

    # Save results
    println("\nSaving results to: $output_path")
    JLD2.save(output_path,
        "results", results,
        "model_name", model_name,
        "model_file", model_file,
        "analysis_parameters", Dict(
            "sample_size" => sample_size,
            "seed" => seed,
            "cv_threshold" => cv_threshold,
            "batch_size" => batch_size,
            "use_transitivity" => use_transitivity
        ),
        "analysis_duration_seconds", analysis_duration)

    # Print summary
    stats = results.stats
    println("\nAnalysis Summary:")
    println("="^60)
    println("Analysis completed successfully!")
    println("Duration: $(round(analysis_duration/60, digits=2)) minutes")
    println("\nModel Statistics:")
    println("  Complexes: $(stats["n_complexes"])")
    println("  Balanced complexes: $(stats["n_balanced"])")
    println("  Trivially balanced: $(stats["n_trivially_balanced"])")
    println("\nConcordance Results:")
    println("  Total concordant pairs: $(stats["n_concordant_total"])")
    println("  Concordant via optimization: $(stats["n_concordant_opt"])")
    println("  Concordant via inference: $(stats["n_concordant_inferred"])")
    println("  Trivial concordant pairs: $(stats["n_trivial_pairs"])")
    println("  Non-concordant pairs: $(stats["n_non_concordant_pairs"])")
    println("  Modules found: $(stats["n_modules"])")
    println("\nProcessing Statistics:")
    println("  Candidates processed: $(stats["n_candidate_pairs"])")
    println("  Candidates skipped by transitivity: $(stats["n_candidates_skipped_by_transitivity"])")
    println("  Batches completed: $(stats["batches_completed"])")
    println("  Timeout pairs: $(stats["n_timeout_pairs"])")
    println("  Validation passed: $(stats["validation_passed"])")
    println("="^60)

catch e
    println("\nERROR: Analysis failed for model $model_name")
    println("Error details: ", e)
    println("Stacktrace:")
    for line in stacktrace(catch_backtrace())
        println("  ", line)
    end

    # Save error information
    error_filename = "error_" * model_name * "_" * string(Int(round(time()))) * ".txt"
    error_path = joinpath(results_dir, error_filename)

    open(error_path, "w") do f
        println(f, "Error in analysis of model: $model_name")
        println(f, "Model file: $model_file")
        println(f, "Timestamp: $(Dates.now())")
        println(f, "Error: $e")
        println(f, "\nStacktrace:")
        for line in stacktrace(catch_backtrace())
            println(f, "  $line")
        end
    end

    println("Error details saved to: $error_path")
    rethrow(e)
end