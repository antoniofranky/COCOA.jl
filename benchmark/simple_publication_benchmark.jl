using Pkg
# Activate the COCOA.jl project environment
Pkg.activate("/work/schaffran1/COCOA.jl")
Pkg.instantiate()

using Distributed
using JLD2, Dates, Statistics, Printf
using DataFrames

# Add exactly 63 worker processes for 64 total cores (1 main + 63 workers)
# On login nodes with limited cores, adjust accordingly
target_workers = min(63, max(1, Sys.CPU_THREADS - 1))
if nworkers() < target_workers
    addprocs(target_workers - nworkers())
end

@everywhere using COCOA, HiGHS, SBMLFBCModels, AbstractFBCModels, COBREXA

# ============================================================================
# PRECOMPILATION AND WARM-UP
# ============================================================================

"""
Precompile and warm up Julia for consistent benchmarking
This eliminates compilation overhead from measurements
"""
function warmup_julia()
    @info "Warming up Julia environment..."

    # Load a small test model for warm-up
    test_model_path = "/work/schaffran1/COCOA.jl/benchmark/models/e_coli_core.xml"
    if !isfile(test_model_path)
        @warn "Test model not found, skipping warm-up"
        return
    end

    try
        # Load and run a minimal analysis to trigger compilation
        @info "Loading test model for warm-up"
        test_model = COBREXA.load_model(test_model_path)

        @info "Running warm-up analysis (small sample)"
        @time warmup_result = concordance_analysis(
            test_model;
            optimizer=HiGHS.Optimizer,
            settings=[
                COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
                COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
                COBREXA.set_optimizer_attribute("time_limit", 60.0),
                COBREXA.set_optimizer_attribute("presolve", "on"),
            ],
            sample_size=10,  # Very small sample for warm-up
            batch_size=250,
            coarse_cv_threshold=0.001,
            cv_threshold=0.0005,
            seed=42,
            use_transitivity=false,  # Disable for faster warm-up
        )

        # Log warm-up completion with proper DataFrame handling
        complexes = get(warmup_result, :complexes, [])
        n_complexes = if isa(complexes, AbstractDataFrame)
            nrow(complexes)
        else
            length(complexes)
        end
        @info "Warm-up complete" n_complexes = n_complexes

        # Force garbage collection after warm-up
        GC.gc(true)
        sleep(1)

    catch e
        @warn "Warm-up failed, proceeding anyway" error = e
    end
end

# ============================================================================
# CONFIGURATION
# ============================================================================

const BENCHMARK_DIR = "/work/schaffran1/COCOA.jl/benchmark/models"
const RESULTS_DIR = "/work/schaffran1/results_testjobs/publication_benchmarks"
const TIMESTAMP = Dates.format(now(), "yyyymmdd_HHMMSS")
const N_CORES = 64

# Ensure results directory exists
mkpath(RESULTS_DIR)

# Standard optimizer settings for reproducibility
const HIGHS_SETTINGS = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("random_seed", 42),
    COBREXA.set_optimizer_attribute("time_limit", 7200.0),  # 2 hours per run
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

# ============================================================================
# MEMORY MONITORING
# ============================================================================

"""
Get memory statistics from /proc filesystem (system-level)
"""
function get_system_memory_stats()
    pid = getpid()
    status_file = "/proc/$pid/status"

    if !isfile(status_file)
        return Dict{String,Float64}()
    end

    status = read(status_file, String)
    metrics = Dict{String,Float64}()

    for (pattern, key) in [
        (r"VmPeak:\s+(\d+)\s+kB", "system_peak_memory_mb"),
        (r"VmRSS:\s+(\d+)\s+kB", "system_rss_memory_mb"),
        (r"VmData:\s+(\d+)\s+kB", "system_data_memory_mb"),
    ]
        m = match(pattern, status)
        if m !== nothing
            metrics[key] = parse(Float64, m[1]) / 1024  # Convert to MB
        end
    end

    return metrics
end

"""
Track memory during function execution using both GC and system monitoring
"""
function track_memory_during_execution(func; sample_interval=0.5)
    # Clean GC state before measurement
    GC.gc(true)
    GC.gc(true)  # Double GC to ensure clean state

    # Measure baseline
    gc_before = Base.gc_num()
    memory_before = Base.gc_live_bytes()
    system_before = get_system_memory_stats()

    # Background system memory sampling
    memory_samples = []
    sampling = Ref(true)

    @async begin
        while sampling[]
            push!(memory_samples, get_system_memory_stats())
            sleep(sample_interval)
        end
    end

    # Execute function with timing
    start_time = time()
    result = func()
    execution_time = time() - start_time

    # Stop sampling
    sampling[] = false
    sleep(sample_interval * 1.5)

    # Measure after execution
    gc_after = Base.gc_num()
    memory_after = Base.gc_live_bytes()
    gc_diff = Base.GC_Diff(gc_after, gc_before)
    system_after = get_system_memory_stats()

    # Calculate GC-based memory statistics (computational memory)
    memory_stats = Dict{String,Float64}(
        "execution_time_sec" => execution_time,
        "gc_time_sec" => gc_diff.total_time / 1e9,
        "gc_allocd_mb" => gc_diff.allocd / 1024^2,
        "gc_live_delta_mb" => (memory_after - memory_before) / 1024^2,
        "n_memory_samples" => length(memory_samples),
    )

    # Add system memory statistics (deployment memory)
    if !isempty(memory_samples)
        for key in keys(first(memory_samples))
            values = [get(s, key, 0.0) for s in memory_samples]
            memory_stats["peak_$key"] = maximum(values)
            memory_stats["mean_$key"] = mean(values)
        end
    end

    # Add baseline and final system memory
    for (key, value) in system_before
        memory_stats["before_$key"] = value
    end
    for (key, value) in system_after
        memory_stats["after_$key"] = value
    end

    return result, memory_stats
end# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

"""
Run benchmark for a single model
"""
function benchmark_single_model(model_file::String; n_runs::Int=3, sample_size::Int=200)
    model_path = joinpath(BENCHMARK_DIR, model_file)

    @info "Benchmarking $model_file" n_cores = N_CORES n_runs = n_runs

    # Load model ONCE before benchmarking (exclude from timing)
    @info "Loading model: $model_file"
    load_start = time()
    model = COBREXA.load_model(model_path)
    load_time = time() - load_start

    # Get model statistics
    model_stats = Dict(
        "model_file" => model_file,
        "n_reactions" => length(AbstractFBCModels.reactions(model)),
        "n_metabolites" => length(AbstractFBCModels.metabolites(model)),
        "n_genes" => length(AbstractFBCModels.genes(model)),
        "model_load_time_sec" => load_time,
    )

    @info "Model loaded" (
        reactions=model_stats["n_reactions"],
        metabolites=model_stats["n_metabolites"],
        genes=model_stats["n_genes"],
        load_time=round(load_time, digits=2)
    )

    # Run multiple benchmark runs
    run_results = []

    for run_idx in 1:n_runs
        @info "Run $run_idx/$n_runs for $model_file"

        # Clean state between runs
        GC.gc(true)
        sleep(1)

        # Create analysis function (model already loaded)
        analysis_func = () -> concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            settings=HIGHS_SETTINGS,
            sample_size=sample_size,
            coarse_cv_threshold=0.0005,
            cv_threshold=0.00005,
            seed=42 + run_idx,
            use_transitivity=true
        )

        # Run with memory tracking (pure computation time)
        @info "Starting computation measurement (no startup overhead)"
        analysis_result, memory_stats = track_memory_during_execution(analysis_func)

        # Extract results
        run_data = Dict{String,Any}(
            "run_idx" => run_idx,
            "execution_time_sec" => memory_stats["execution_time_sec"],
            # Primary memory metrics for publication
            "gc_allocd_mb" => memory_stats["gc_allocd_mb"],  # Computational memory
            "peak_system_memory_mb" => get(memory_stats, "peak_system_peak_memory_mb",
                get(memory_stats, "peak_system_rss_memory_mb", NaN)),  # System memory
            # Secondary metrics
            "gc_time_sec" => memory_stats["gc_time_sec"],
            "gc_live_delta_mb" => memory_stats["gc_live_delta_mb"],
            "mean_system_memory_mb" => get(memory_stats, "mean_system_rss_memory_mb", NaN),
            "n_memory_samples" => memory_stats["n_memory_samples"],
        )

        # Extract analysis results
        if haskey(analysis_result, :complexes)
            complexes = analysis_result.complexes
            if isa(complexes, AbstractDataFrame)
                # Handle DataFrame case
                run_data["n_complexes"] = nrow(complexes)
                run_data["complex_sizes"] = [1]  # Placeholder since structure is different
            else
                # Handle array/collection case
                run_data["n_complexes"] = length(complexes)
                run_data["complex_sizes"] = [length(c) for c in complexes]
            end
        else
            run_data["n_complexes"] = 0
            run_data["complex_sizes"] = Int[]
        end

        if haskey(analysis_result, :modules)
            modules = analysis_result.modules
            if isa(modules, AbstractDataFrame)
                # Handle DataFrame case
                run_data["n_modules"] = nrow(modules)
                run_data["module_sizes"] = [1]  # Placeholder since structure is different
            else
                # Handle array/collection case
                run_data["n_modules"] = length(modules)
                run_data["module_sizes"] = [length(m) for m in modules]
            end
        else
            run_data["n_modules"] = 0
            run_data["module_sizes"] = Int[]
        end

        run_data["detailed_memory_stats"] = memory_stats

        push!(run_results, run_data)

        @info "Run $run_idx complete" (
            time_sec=round(run_data["execution_time_sec"], digits=2),
            gc_memory_mb=round(run_data["gc_allocd_mb"], digits=1),
            system_memory_mb=round(get(run_data, "peak_system_memory_mb", NaN), digits=1),
            n_complexes=run_data["n_complexes"],
            n_modules=run_data["n_modules"]
        )
    end

    # Calculate summary statistics
    summary_stats = Dict{String,Any}(
        "model_file" => model_file,
        "n_runs" => n_runs,
        "timestamp" => TIMESTAMP,
        "n_reactions" => model_stats["n_reactions"],
        "n_metabolites" => model_stats["n_metabolites"],
        "n_genes" => model_stats["n_genes"],
        "model_load_time_sec" => model_stats["model_load_time_sec"],
        "n_cores" => N_CORES,
        "n_workers" => nworkers(),
        "julia_version" => string(VERSION),
        "sample_size" => sample_size,
        "mean_execution_time_sec" => mean([r["execution_time_sec"] for r in run_results]),
        "std_execution_time_sec" => std([r["execution_time_sec"] for r in run_results]),
        # Computational memory (GC-based)
        "mean_gc_allocd_mb" => mean([r["gc_allocd_mb"] for r in run_results]),
        "std_gc_allocd_mb" => std([r["gc_allocd_mb"] for r in run_results]),
        # System memory (deployment requirements)
        "mean_peak_system_mb" => mean([r["peak_system_memory_mb"] for r in run_results if !isnan(r["peak_system_memory_mb"])]),
        "std_peak_system_mb" => std([r["peak_system_memory_mb"] for r in run_results if !isnan(r["peak_system_memory_mb"])]),
        "n_complexes" => run_results[1]["n_complexes"],
        "n_modules" => run_results[1]["n_modules"],
    )

    return Dict(
        "summary" => summary_stats,
        "runs" => run_results,
        "model_stats" => model_stats,
    )
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    @info "Starting publication benchmark with startup optimization"

    # STEP 1: Warm up Julia
    warmup_julia()

    # STEP 2: Check workers
    expected_workers = min(63, max(1, Sys.CPU_THREADS - 1))
    if nworkers() != expected_workers
        @warn "Expected $expected_workers workers for $(Sys.CPU_THREADS) total cores, got $(nworkers())"
    end

    @info "Environment ready" (
        n_cores=N_CORES,
        n_workers=nworkers(),
        julia_version=VERSION,
        hostname=gethostname()
    )

    # STEP 3: Get model from command line
    if length(ARGS) != 1
        error("Usage: julia simple_publication_benchmark.jl <model_file>")
    end

    model_file = ARGS[1]
    model_path = joinpath(BENCHMARK_DIR, model_file)

    if !isfile(model_path)
        error("Model file not found: $model_path")
    end

    @info "Benchmarking model" model = model_file

    # STEP 4: Run benchmark
    benchmark_start = time()
    result = benchmark_single_model(model_file, n_runs=3)
    total_time = time() - benchmark_start

    # STEP 5: Save results
    output_file = joinpath(RESULTS_DIR, "benchmark_$(splitext(model_file)[1])_$(TIMESTAMP).jld2")
    result["total_benchmark_time_sec"] = total_time

    JLD2.save(output_file, "result", result)

    @info "Benchmark complete" (
        model=model_file,
        output_file=output_file,
        total_time=round(total_time, digits=2),
        n_complexes=result["summary"]["n_complexes"],
        mean_time=round(result["summary"]["mean_execution_time_sec"], digits=2),
        gc_memory=round(result["summary"]["mean_gc_allocd_mb"], digits=1),
        system_memory=round(get(result["summary"], "mean_peak_system_mb", NaN), digits=1)
    )

    # Print summary
    s = result["summary"]
    println("\n" * "="^60)
    println("PUBLICATION BENCHMARK RESULT")
    println("="^60)
    println(@sprintf("Model: %s", s["model_file"]))
    println(@sprintf("Reactions: %d, Metabolites: %d, Genes: %d",
        s["n_reactions"], s["n_metabolites"], s["n_genes"]))
    println(@sprintf("Complexes: %d, Modules: %d", s["n_complexes"], s["n_modules"]))
    println(@sprintf("Execution Time: %.2f±%.2f sec",
        s["mean_execution_time_sec"], s["std_execution_time_sec"]))
    println(@sprintf("Computational Memory: %.1f±%.1f MB (GC allocations)",
        s["mean_gc_allocd_mb"], s["std_gc_allocd_mb"]))

    # Only show system memory if available
    if haskey(s, "mean_peak_system_mb") && !isnan(s["mean_peak_system_mb"])
        println(@sprintf("System Peak Memory: %.1f±%.1f MB (RSS)",
            s["mean_peak_system_mb"], s["std_peak_system_mb"]))
    end
    println("="^60)

    return result
end

# Run if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
