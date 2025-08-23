using Distributed
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2, Dates
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
output_filename = "kinetic_results_" * model_name * "_" *
                  lpad(string(seed), 2, "0") * "_" *
                  string(batch_size) * "_cv" *
                  replace(string(cv_threshold), "." => "p") * "_samples" *
                  string(sample_size) *
                  "_transitivity" * string(use_transitivity) * ".jld2"

output_path = joinpath(results_dir, output_filename)

# Log analysis start
println("="^60)
println("COCOA Kinetic Concordance Analysis")
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

    # Run kinetic concordance analysis
    println("\nStarting kinetic concordance analysis...")
    
    # Run analysis with timing
    analysis_timing = @timed begin
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
    end
    
    # Extract timing information
    analysis_duration = analysis_timing.time
    gc_time = analysis_timing.gctime
    memory_allocated = analysis_timing.bytes
    
    println("Analysis completed in $(round(analysis_duration, digits=2)) seconds")
    println("Memory allocated: $(round(memory_allocated / 1e9, digits=2)) GB")
    println("GC time: $(round(gc_time, digits=2)) seconds")

    # Save results with memory statistics (optimized for size)
    println("\nSaving results to: $output_path")
    
    # Create compact summary instead of full results to reduce file size
    compact_results = Dict(
        "n_robust_metabolites" => results.n_robust_metabolites,
        "n_robust_pairs" => results.n_robust_pairs,
        "largest_robust_module_size" => results.largest_robust_module_size,
        "robust_metabolites" => results.robust_metabolites,
        "robust_metabolite_pairs" => results.robust_metabolite_pairs,
        "summary" => results.summary,
        # Save only essential concordance statistics, not full DataFrames
        "n_complexes" => results.summary !== nothing ? get(results.summary, "n_complexes", 0) : 0,
        "n_modules" => results.summary !== nothing ? get(results.summary, "n_modules", 0) : 0,
        "n_kinetic_modules" => results.summary !== nothing ? get(results.summary, "n_kinetic_modules", 0) : 0
    )
    
    JLD2.save(output_path,
        "results", compact_results,
        "model_name", model_name,
        "model_file", model_file,
        "analysis_parameters", Dict(
            "sample_size" => sample_size,
            "seed" => seed,
            "cv_threshold" => cv_threshold,
            "batch_size" => batch_size,
            "use_transitivity" => use_transitivity
        ),
        "timing_statistics", Dict(
            "analysis_duration_seconds" => analysis_duration,
            "memory_allocated_bytes" => memory_allocated,
            "memory_allocated_gb" => memory_allocated / 1e9,
            "gc_time_seconds" => gc_time,
            "gc_time_fraction" => gc_time / analysis_duration
        ),
        "timestamp", Dates.now())

    # Print summary - handle ConcentrationRobustnessResults
    println("\nKinetic Concordance Analysis Summary:")
    println("="^60)
    println("Analysis completed successfully!")
    println("Duration: $(round(analysis_duration/60, digits=2)) minutes")
    println("Memory allocated: $(round(memory_allocated / 1e9, digits=2)) GB")
    println("GC time: $(round(gc_time, digits=2))s ($(round(gc_time/analysis_duration*100, digits=1))%)")
    
    # Display kinetic and robustness results
    println("\nRobustness Results:")
    println("  Robust metabolites: $(results.n_robust_metabolites)")
    println("  Robust metabolite pairs: $(results.n_robust_pairs)")
    println("  Largest robust module size: $(results.largest_robust_module_size)")
    
    # Print general statistics from summary if available
    if results.summary !== nothing
        stats = results.summary
        if haskey(stats, "n_complexes")
            println("\nModel Statistics:")
            println("  Complexes: $(stats["n_complexes"])")
        end
        if haskey(stats, "n_balanced")
            println("  Balanced complexes: $(stats["n_balanced"])")
        end
        if haskey(stats, "n_modules")
            println("  Modules found: $(stats["n_modules"])")
        end
        if haskey(stats, "n_concordant_total")
            println("\nConcordance Results:")
            println("  Total concordant pairs: $(stats["n_concordant_total"])")
        end
        if haskey(stats, "n_kinetic_modules")
            println("\nKinetic Statistics:")
            println("  Kinetic modules: $(stats["n_kinetic_modules"])")
        end
    else
        println("\nNote: Summary statistics not available")
    end
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