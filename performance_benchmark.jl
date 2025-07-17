
"""
Performance benchmark script for COCOA concordance analysis.

This script creates models of different sizes using ordered fractions between 0.1 and 1.0,
runs concordance analysis on each model, and collects performance metrics.
"""

import AbstractFBCModels.CanonicalModel
using Distributed
using BenchmarkTools
using Statistics
using Plots
using DataFrames
using Logging
using Profile
using JLD2
include("./src/COCOA.jl")
using .COCOA
# Worker counts to test
WORKER_COUNTS = [15]

# Function to configure workers
function configure_workers(target_count::Int)
    current_workers = nworkers()

    if current_workers < target_count
        addprocs(target_count - current_workers)
        println("Added workers. Total workers: $(nworkers())")
    elseif current_workers > target_count
        # Remove excess workers
        workers_to_remove = workers()[(target_count+1):end]
        if !isempty(workers_to_remove)
            rmprocs(workers_to_remove)
            println("Removed workers. Total workers: $(nworkers())")
        end
    end

    return nworkers()
end

# Start with 15 workers initially
configure_workers(15)

# Load packages on all workers
@everywhere begin
    using COBREXA
    using HiGHS
    using AbstractFBCModels
    using SBMLFBCModels

    # Load COCOA functions directly
    include("./src/COCOA.jl")

    # Import the functions we need
    using .COCOA: concordance_analysis, split_into_elementary_steps
end

# Configure logging
global_logger(ConsoleLogger(stderr, Logging.Info))

"""
Load a sample metabolic model for benchmarking.
"""
function load_sample_model()
    # Try to load a model - adjust path as needed
    model_paths = [
        "./test/e_coli_core.xml"
    ]

    for path in model_paths
        if isfile(path)
            @info "Loading model from: $path"
            return load_model(path)
        end
    end

    # If no model found, create a simple test model
    @warn "No model file found. Creating simple test model."
    return create_test_model()
end

"""
Create a simple test model for benchmarking if no model file is available.
"""
function create_test_model()


    model = Model()

    # Add some metabolites
    for i in 1:20
        model.metabolites["M$i"] = Metabolite(
            name="Metabolite $i",
            compartment="c"
        )
    end

    # Add some reactions
    for i in 1:15
        stoich = Dict()
        # Simple A -> B reactions
        stoich["M$i"] = -1.0
        stoich["M$(i+1)"] = 1.0

        model.reactions["R$i"] = Reaction(
            name="Reaction $i",
            stoichiometry=stoich,
            lower_bound=-1000.0,
            upper_bound=1000.0
        )
    end

    # Add objective reaction
    model.reactions["R_objective"] = Reaction(
        name="Objective",
        stoichiometry=Dict("M20" => -1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    model.objective = "R_objective"

    return model
end

"""
Run performance benchmark for a single ordered fraction with specified worker count.
"""
function benchmark_fraction(base_model, ordered_fraction::Float64, worker_count::Int)
    @info "Benchmarking ordered fraction: $ordered_fraction with $worker_count workers"

    # Create model with elementary steps
    model = split_into_elementary_steps(
        base_model;
        ordered_fraction=ordered_fraction,
        seed=42
    )

    # Count complexes (estimate)
    n_reactions = length(model.reactions)
    n_metabolites = length(model.metabolites)
    estimated_complexes = n_reactions * 2  # Rough estimate

    @info "Model stats" ordered_fraction n_reactions n_metabolites estimated_complexes

    # Configure optimizer
    optimizer = HiGHS.Optimizer
    settings = [
        "output_flag" => false,
        "log_to_console" => false,
        "presolve" => "on",
        "parallel" => "off"
    ]

    # Measure performance
    gc_before = Base.gc_num()

    # Run concordance analysis with performance measurement
    result = @timed begin
        concordance_analysis(
            model;
            optimizer=optimizer,
            settings=settings,
            workers=workers(),
            tolerance=1e-12,
            correlation_threshold=0.95,
            sample_size=50,  # Reduced for faster benchmarking
            stage_size=200,  # Smaller stages for benchmark
            batch_size=50,
            seed=42
        )
    end

    gc_after = Base.gc_num()

    # Calculate memory usage
    memory_allocated = (gc_after.allocd - gc_before.allocd) / 1024^2  # MB

    # Extract results
    analysis_result = result.value
    elapsed_time = result.time

    # Get actual complex count
    actual_complexes = nrow(analysis_result.complexes)

    return (
        ordered_fraction=ordered_fraction,
        worker_count=worker_count,
        n_complexes=actual_complexes,
        n_reactions=n_reactions,
        n_metabolites=n_metabolites,
        time_seconds=elapsed_time,
        memory_mb=memory_allocated,
        n_modules=length(unique(analysis_result.complexes.module)),
        n_balanced=sum(analysis_result.complexes.is_balanced),
        analysis_result=analysis_result
    )
end

"""
Run full performance benchmark across all ordered fractions and worker counts.
"""
function run_performance_benchmark()
    @info "Starting performance benchmark with worker counts: $WORKER_COUNTS"

    # Load base model
    base_model = load_sample_model()

    # Define ordered fractions
    ordered_fractions = 0.1:0.1:1.0

    # Store results
    results = []

    # Run benchmark for each worker count
    for worker_count in WORKER_COUNTS
        @info "Testing with $worker_count workers"

        # Configure workers
        configure_workers(worker_count)

        # Run benchmark for each fraction with this worker count
        for fraction in ordered_fractions
            try
                result = benchmark_fraction(base_model, fraction, worker_count)
                push!(results, result)

                @info "Completed fraction $fraction with $worker_count workers" time = result.time_seconds memory = result.memory_mb complexes = result.n_complexes

                # Force garbage collection between runs
                GC.gc()

            catch e
                @error "Failed to benchmark fraction $fraction with $worker_count workers: $e"
                # Continue with other fractions
            end
        end
    end

    return results
end

"""
Create performance plots including worker scaling analysis.
"""
function create_performance_plots(results)
    @info "Creating performance plots"

    # Extract data for plotting
    fractions = [r.ordered_fraction for r in results]
    worker_counts = [r.worker_count for r in results]
    complexes = [r.n_complexes for r in results]
    times = [r.time_seconds for r in results]
    memory = [r.memory_mb for r in results]

    # Create plots by complexity (complexes vs time/memory)
    p1 = plot(
        title="Concordance Analysis Performance: Time vs Model Size",
        xlabel="Number of Complexes",
        ylabel="Time (seconds)",
        legend=:topright
    )

    p2 = plot(
        title="Concordance Analysis Performance: Memory vs Model Size",
        xlabel="Number of Complexes",
        ylabel="Memory Usage (MB)",
        legend=:topright
    )

    # Plot data by worker count
    colors = [:blue, :red, :green, :orange]
    markers = [:circle, :square, :diamond, :triangle]

    for (i, wc) in enumerate(WORKER_COUNTS)
        worker_mask = worker_counts .== wc
        if any(worker_mask)
            plot!(p1, complexes[worker_mask], times[worker_mask],
                label="$wc workers", color=colors[i], marker=markers[i], linewidth=2)
            plot!(p2, complexes[worker_mask], memory[worker_mask],
                label="$wc workers", color=colors[i], marker=markers[i], linewidth=2)
        end
    end

    # Combined complexity plot
    p_combined = plot(p1, p2, layout=(1, 2), size=(1200, 400))
    savefig(p_combined, "performance_benchmark_results.png")

    # Create worker scaling plots
    p3 = plot(
        title="Worker Scaling Performance",
        xlabel="Number of Workers",
        ylabel="Time (seconds)",
        legend=:topright
    )

    p4 = plot(
        title="Worker Scaling Memory Usage",
        xlabel="Number of Workers",
        ylabel="Memory (MB)",
        legend=:topright
    )

    # Plot worker scaling for different fractions
    fraction_colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
    selected_fractions = [0.2, 0.5, 0.8, 1.0]  # Sample fractions to avoid clutter

    for (i, frac) in enumerate(selected_fractions)
        frac_mask = fractions .≈ frac
        if any(frac_mask)
            plot!(p3, worker_counts[frac_mask], times[frac_mask],
                label="Fraction $frac", color=fraction_colors[i], marker=:circle, linewidth=2)
            plot!(p4, worker_counts[frac_mask], memory[frac_mask],
                label="Fraction $frac", color=fraction_colors[i], marker=:square, linewidth=2)
        end
    end

    # Worker scaling plot
    p_scaling = plot(p3, p4, layout=(1, 2), size=(1200, 400))
    savefig(p_scaling, "worker_scaling_performance.png")

    @info "Plots saved: performance_benchmark_results.png, worker_scaling_performance.png"

    return p_combined, p_scaling
end

"""
Save results to JLD2 format.
"""
function save_results(results)
    @info "Saving results"

    # Create DataFrame
    df = DataFrame(
        ordered_fraction=[r.ordered_fraction for r in results],
        worker_count=[r.worker_count for r in results],
        n_complexes=[r.n_complexes for r in results],
        n_reactions=[r.n_reactions for r in results],
        n_metabolites=[r.n_metabolites for r in results],
        time_seconds=[r.time_seconds for r in results],
        memory_mb=[r.memory_mb for r in results],
        n_modules=[r.n_modules for r in results],
        n_balanced=[r.n_balanced for r in results]
    )

    # Save full results with analysis data
    @save "performance_benchmark_full.jld2" results df

    @info "Results saved: performance_benchmark_full.jld2"

    return df
end

"""
Print summary statistics including worker scaling analysis.
"""
function print_summary(results)
    @info "Performance Benchmark Summary"
    @info "=============================="

    times = [r.time_seconds for r in results]
    memory = [r.memory_mb for r in results]
    complexes = [r.n_complexes for r in results]
    worker_counts = [r.worker_count for r in results]

    println("Worker counts tested: $WORKER_COUNTS")
    println("Ordered fractions tested: $(length(unique([r.ordered_fraction for r in results])))")
    println("Total benchmark runs: $(length(results))")
    println("Complexes range: $(minimum(complexes)) - $(maximum(complexes))")
    println("Time range: $(round(minimum(times), digits=2)) - $(round(maximum(times), digits=2)) seconds")
    println("Memory range: $(round(minimum(memory), digits=1)) - $(round(maximum(memory), digits=1)) MB")
    println("Average time: $(round(mean(times), digits=2)) seconds")
    println("Average memory: $(round(mean(memory), digits=1)) MB")

    # Worker scaling analysis
    println("\nWorker Scaling Analysis:")
    for wc in WORKER_COUNTS
        worker_results = filter(r -> r.worker_count == wc, results)
        if !isempty(worker_results)
            avg_time = mean([r.time_seconds for r in worker_results])
            avg_memory = mean([r.memory_mb for r in worker_results])
            println("  $wc workers: $(round(avg_time, digits=2))s avg time, $(round(avg_memory, digits=1))MB avg memory")
        end
    end

    # Performance scaling
    if length(results) > 1
        time_scaling = maximum(times) / minimum(times)
        memory_scaling = maximum(memory) / minimum(memory)
        complexity_scaling = maximum(complexes) / minimum(complexes)

        println("\nScaling factors:")
        println("Time scaling factor: $(round(time_scaling, digits=2))x")
        println("Memory scaling factor: $(round(memory_scaling, digits=2))x")
        println("Complexity scaling factor: $(round(complexity_scaling, digits=2))x")

        # Worker efficiency analysis
        if length(WORKER_COUNTS) > 1
            single_worker_times = [r.time_seconds for r in results if r.worker_count == 1]
            max_worker_times = [r.time_seconds for r in results if r.worker_count == maximum(WORKER_COUNTS)]

            if !isempty(single_worker_times) && !isempty(max_worker_times)
                speedup = mean(single_worker_times) / mean(max_worker_times)
                efficiency = speedup / maximum(WORKER_COUNTS)
                println("Parallel speedup: $(round(speedup, digits=2))x")
                println("Parallel efficiency: $(round(efficiency * 100, digits=1))%")
            end
        end
    end
end

"""
Main function to run the complete benchmark.
"""
function main()
    @info "Starting COCOA performance benchmark"
    @info "Workers: $(nworkers())"

    # Run benchmark
    results = run_performance_benchmark()

    if isempty(results)
        @error "No results obtained from benchmark"
        return
    end

    # Create plots
    plots = create_performance_plots(results)

    # Save results
    df = save_results(results)

    # Print summary
    print_summary(results)

    @info "Benchmark completed successfully!"

    return results, df, plots
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    results, df, plots = main()
end