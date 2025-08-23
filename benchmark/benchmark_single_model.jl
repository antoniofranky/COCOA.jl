using Distributed
using BenchmarkTools, JLD2, Dates
@everywhere using COCOA, HiGHS, SBMLFBCModels, AbstractFBCModels, COBREXA

# Get model file from command line argument
if length(ARGS) != 1
    error("Usage: julia benchmark_single_model.jl <model_file>")
end

const MODEL_FILE = ARGS[1]
const BENCHMARK_DIR = "/work/schaffran1/COCOA.jl/benchmark/models"
const RESULTS_DIR = "/work/schaffran1/results_testjobs/benchmark_results"
const TIMESTAMP = Dates.format(now(), "yyyymmdd_HHMMSS")

# Ensure results directory exists
mkpath(RESULTS_DIR)

# Optimizer settings for consistent benchmarking
const HIGHS_SETTINGS = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("random_seed", 42),
    COBREXA.set_optimizer_attribute("time_limit", 600.0),
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

# Benchmark configuration for different model sizes
function get_benchmark_config(n_reactions::Int)
    # Consistent parameters optimized for large to very large models (5000-28000+ reactions)
    base_config = (
        sample_size=200,              # Higher sample size for better statistical power in large solution spaces
        coarse_cv_threshold=0.0005,  # Stricter coarse filtering for large models with more noise
        cv_threshold=0.00005,        # Very strict final threshold for high precision
    )

    # Quadratic scaling for time limits: 48 hours for largest (28686 reactions)
    # Scale down quadratically to smaller models
    max_reactions = 28686  # Largest model size
    max_seconds = 48 * 3600  # 48 hours in seconds

    # Quadratic scaling: time = max_time * (reactions/max_reactions)²
    # With minimum of 30 minutes for small models
    time_fraction = (n_reactions / max_reactions)^2
    scaled_seconds = max(1800, Int(round(max_seconds * time_fraction)))  # Minimum 30 minutes

    if n_reactions <= 1000
        # Small/medium models: multiple samples for better statistics
        return merge(base_config, (
            samples=3,
            seconds=scaled_seconds,
            skip_precompile=false,
            skip_final_result=false
        ))
    elseif n_reactions <= 5000
        # Medium-large models: single run, but still do precompile and final result
        return merge(base_config, (
            samples=1,
            seconds=scaled_seconds,
            skip_precompile=false,
            skip_final_result=false
        ))
    else
        # Very large models: minimal runs to avoid memory/time issues
        return merge(base_config, (
            samples=1,
            seconds=scaled_seconds,
            skip_precompile=true,    # Skip precompilation for very large models
            skip_final_result=true   # Skip final result run, use benchmark result instead
        ))
    end
end

# Function to get model statistics
function get_model_stats(model)
    n_reactions = length(AbstractFBCModels.reactions(model))
    n_metabolites = length(AbstractFBCModels.metabolites(model))
    n_genes = length(AbstractFBCModels.genes(model))

    return (
        n_reactions=n_reactions,
        n_metabolites=n_metabolites,
        n_genes=n_genes,
        model_type=nameof(typeof(model))
    )
end

# Memory tracking wrapper
function track_memory_usage(func)
    # Force GC before measurement
    GC.gc()

    # Get initial memory stats
    initial_memory = Base.Sys.total_memory() - Base.Sys.free_memory()

    # Run function and measure
    result = func()

    # Force GC and measure final memory
    GC.gc()
    final_memory = Base.Sys.total_memory() - Base.Sys.free_memory()

    peak_memory_mb = (final_memory - initial_memory) / 1024^2

    return result, peak_memory_mb
end

# Main benchmarking function for single model
function benchmark_single_model(model_file::String)
    model_path = joinpath(BENCHMARK_DIR, model_file)

    if !isfile(model_path)
        error("Model file not found: $model_path")
    end

    @info "Starting benchmark for" model_file

    # Load model (assuming already split and prepared)
    @info "Loading pre-prepared model..."
    load_start = time()
    model = COBREXA.load_model(model_path)
    load_time = time() - load_start

    model_stats = get_model_stats(model)
    @info "Model loaded" model_stats... load_time_sec = round(load_time, digits=2)

    # Get benchmark configuration based on model size
    config = get_benchmark_config(model_stats.n_reactions)
    @info "Benchmark configuration" config...

    # Log the strategy being used
    if config.skip_precompile && config.skip_final_result
        @info "Using minimal run strategy for very large model (>5000 reactions)"
    elseif config.skip_precompile || config.skip_final_result
        @info "Using reduced run strategy for large model"
    else
        @info "Using full benchmark strategy for small/medium model"
    end

    # Pre-compile with a minimal run (conditional for large models)
    precompile_time = 0.0  # Initialize outside try-catch to ensure scope
    if !config.skip_precompile
        @info "Pre-compiling activity_concordance_analysis..."
        precompile_start = time()
        precomp_model = COBREXA.load_model("/work/schaffran1/COCOA.jl/test/e_coli_core.xml")
        try
            # Minimal run for precompilation
            activity_concordance_analysis(
                precomp_model;
                optimizer=HiGHS.Optimizer,
                settings=HIGHS_SETTINGS,
                sample_size=20,
                batch_size=300,
                coarse_cv_threshold=0.01,
                cv_threshold=0.001,
                seed=42
            )
            precompile_time = time() - precompile_start
            @info "Pre-compilation complete" precompile_time_sec = round(precompile_time, digits=2)
        catch e
            @warn "Pre-compilation failed, continuing anyway" error = string(e)
            precompile_time = time() - precompile_start
        end
    else
        @info "Skipping pre-compilation for very large model to save memory and time"
    end

    # Force garbage collection before benchmark
    GC.gc()

    # Create benchmark function that only measures the analysis
    analysis_func = () -> activity_concordance_analysis(
        model;
        optimizer=HiGHS.Optimizer,
        settings=HIGHS_SETTINGS,
        sample_size=config.sample_size,
        coarse_cv_threshold=config.coarse_cv_threshold,
        cv_threshold=config.cv_threshold,
        seed=42,
        use_transitivity=true
    )

    # Run benchmark with memory tracking
    @info "Running benchmark with memory tracking..."
    benchmark_start = time()

    # Use BenchmarkTools with memory tracking
    benchmark_result = @benchmark $(analysis_func)() samples = config.samples seconds = config.seconds

    benchmark_total_time = time() - benchmark_start

    # Get one final result for analysis statistics (conditional for large models)
    analysis_results = nothing
    peak_memory_mb = 0.0
    final_result_time = 0.0

    if !config.skip_final_result
        @info "Getting final analysis results..."
        final_result_start = time()
        analysis_results, peak_memory_mb = track_memory_usage(analysis_func)
        final_result_time = time() - final_result_start
    else
        @info "Skipping final result run for very large model - using benchmark result instead"
        # For very large models, we'll use the benchmark result as our analysis result
        # and estimate memory usage from the benchmark
        peak_memory_mb = benchmark_result.memory / 1024^2
    end

    # Compile comprehensive benchmark statistics
    benchmark_stats = Dict(
        # Model information
        "model_file" => model_file,
        "timestamp" => TIMESTAMP,
        "array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", "unknown"),
        "array_job_id" => get(ENV, "SLURM_ARRAY_JOB_ID", "unknown"),
        "node_name" => get(ENV, "SLURMD_NODENAME", "unknown"),

        # Model characteristics
        "n_reactions" => model_stats.n_reactions,
        "n_metabolites" => model_stats.n_metabolites,
        "n_genes" => model_stats.n_genes,
        "model_type" => string(model_stats.model_type),

        # Benchmark configuration
        "sample_size" => config.sample_size,
        "coarse_cv_threshold" => config.coarse_cv_threshold,
        "cv_threshold" => config.cv_threshold,
        "benchmark_samples" => config.samples,
        "benchmark_seconds" => config.seconds,

        # Timing breakdown
        "model_load_time_sec" => load_time,
        "precompile_time_sec" => precompile_time,
        "benchmark_total_time_sec" => benchmark_total_time,
        "final_result_time_sec" => final_result_time,

        # Pure analysis timing (excluding Julia overhead)
        "min_analysis_time_sec" => minimum(benchmark_result.times) / 1e9,
        "median_analysis_time_sec" => median(benchmark_result.times) / 1e9,
        "mean_analysis_time_sec" => mean(benchmark_result.times) / 1e9,
        "max_analysis_time_sec" => maximum(benchmark_result.times) / 1e9,
        "std_analysis_time_sec" => std(benchmark_result.times) / 1e9,

        # Memory usage (from BenchmarkTools and our tracking)
        "benchmark_memory_bytes" => benchmark_result.memory,
        "benchmark_memory_mb" => benchmark_result.memory / 1024^2,
        "peak_memory_mb" => peak_memory_mb,
        "allocs" => benchmark_result.allocs,

        # Benchmark quality metrics
        "benchmark_samples_completed" => length(benchmark_result.times),
        "gc_fraction" => sum(benchmark_result.gctimes) / sum(benchmark_result.times),
        "gc_time_sec" => sum(benchmark_result.gctimes) / 1e9,
    )

    # Add analysis statistics if available
    if analysis_results !== nothing && haskey(analysis_results, :stats)
        for (key, value) in analysis_results.stats
            benchmark_stats["analysis_$(key)"] = value
        end
    end

    @info "Benchmark complete" (
        model_file=model_file,
        median_analysis_time_sec=round(benchmark_stats["median_analysis_time_sec"], digits=2),
        peak_memory_mb=round(benchmark_stats["peak_memory_mb"], digits=1),
        samples_completed=benchmark_stats["benchmark_samples_completed"]
    )

    return (
        benchmark_stats=benchmark_stats,
        benchmark_result=benchmark_result,
        analysis_results=analysis_results
    )
end

# Main execution
function main()
    @info "Starting single model benchmark" model_file = MODEL_FILE n_workers = nworkers()

    try
        result = benchmark_single_model(MODEL_FILE)

        if result !== nothing
            # Save results to model-specific file
            model_name = splitext(MODEL_FILE)[1]
            results_file = joinpath(RESULTS_DIR, "benchmark_$(model_name)_$(TIMESTAMP).jld2")

            JLD2.save(
                results_file,
                "result", result,
                "benchmark_stats", result.benchmark_stats,
                "model_file", MODEL_FILE,
                "timestamp", TIMESTAMP,
                "system_info", Dict(
                    "n_workers" => nworkers(),
                    "julia_version" => string(VERSION),
                    "array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", "unknown"),
                    "array_job_id" => get(ENV, "SLURM_ARRAY_JOB_ID", "unknown"),
                    "node_name" => get(ENV, "SLURMD_NODENAME", "unknown"),
                    "cpus" => get(ENV, "SLURM_CPUS_PER_TASK", "unknown"),
                    "memory_mb" => get(ENV, "SLURM_MEM_PER_NODE", "unknown")
                )
            )

            @info "Results saved to" results_file

            # Print final summary
            stats = result.benchmark_stats
            println("\n=== BENCHMARK SUMMARY ===")
            println("Model: $(stats["model_file"])")
            println("Reactions: $(stats["n_reactions"])")
            println("Median Analysis Time: $(round(stats["median_analysis_time_sec"], digits=2)) seconds")
            println("Peak Memory Usage: $(round(stats["peak_memory_mb"], digits=1)) MB")
            println("Samples Completed: $(stats["benchmark_samples_completed"])")
            if haskey(stats, "analysis_n_complexes")
                println("Complexes Found: $(stats["analysis_n_complexes"])")
            end
            if haskey(stats, "analysis_n_modules")
                println("Modules Found: $(stats["analysis_n_modules"])")
            end
            println("========================")
        else
            @error "Benchmark failed"
            exit(1)
        end

    catch e
        @error "Benchmark failed with exception" error = string(e)

        # Save error information
        model_name = splitext(MODEL_FILE)[1]
        error_file = joinpath(RESULTS_DIR, "benchmark_ERROR_$(model_name)_$(TIMESTAMP).jld2")

        JLD2.save(
            error_file,
            "error", string(e),
            "model_file", MODEL_FILE,
            "timestamp", TIMESTAMP,
            "failed", true
        )

        exit(1)
    end

    @info "Single model benchmark complete"
end

# Run main function
main()
