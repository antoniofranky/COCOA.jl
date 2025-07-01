"""
Comprehensive benchmarking script for COCOA.jl parameter optimization.

This script tests different parameter combinations to find optimal settings
for memory efficiency and computational performance.
"""

using Pkg
Pkg.activate(".")

using COCOA
using HiGHS
using COBREXA
using AbstractFBCModels
using DataFrames
using Statistics
using BenchmarkTools
using ProgressMeter
using Plots
using CSV

# Load your model (adjust path as needed)
println("Loading model...")
model_path = "c:\\Users\\anton\\master-thesis\\toolbox\\e_coli_core.xml"
if isfile(model_path)
    model = load_model(model_path)
else
    println("Model file not found at $model_path")
    println("Please adjust the model_path variable to point to your model file")
    exit(1)
end

"""
Benchmark different parameter configurations for COCOA analysis.
"""
function benchmark_parameter_combinations(model; max_tests=10)
    println("Setting up benchmark configurations...")

    # Parameter combinations to test
    configurations = [
        # Small batch sizes for memory efficiency
        ConcordanceConfig(
            sample_size=500,
            sample_batch_size=50,
            concordance_batch_size=25,
            stage_size=100,
            correlation_threshold=0.95
        ),

        # Medium batch sizes
        ConcordanceConfig(
            sample_size=1000,
            sample_batch_size=100,
            concordance_batch_size=50,
            stage_size=200,
            correlation_threshold=0.95
        ),

        # Larger batch sizes for speed
        ConcordanceConfig(
            sample_size=1000,
            sample_batch_size=200,
            concordance_batch_size=100,
            stage_size=500,
            correlation_threshold=0.95
        ),

        # Lower correlation threshold (more candidates)
        ConcordanceConfig(
            sample_size=800,
            sample_batch_size=100,
            concordance_batch_size=50,
            stage_size=200,
            correlation_threshold=0.90
        ),

        # Higher correlation threshold (fewer candidates)
        ConcordanceConfig(
            sample_size=800,
            sample_batch_size=100,
            concordance_batch_size=50,
            stage_size=200,
            correlation_threshold=0.98
        ),
    ]

    results = DataFrame(
        config_id=Int[],
        sample_size=Int[],
        sample_batch_size=Int[],
        concordance_batch_size=Int[],
        stage_size=Int[],
        correlation_threshold=Float64[],
        runtime_seconds=Float64[],
        memory_gb=Float64[],
        n_complexes=Int[],
        n_candidates=Int[],
        n_concordant=Int[],
        n_modules=Int[],
        allocation_mb=Float64[]
    )

    for (i, config) in enumerate(configurations)
        if i > max_tests
            break
        end

        println("\n" * "="^60)
        println("Testing configuration $i/$max_tests")
        println("Sample size: $(config.sample_size)")
        println("Sample batch size: $(config.sample_batch_size)")
        println("Concordance batch size: $(config.concordance_batch_size)")
        println("Stage size: $(config.stage_size)")
        println("Correlation threshold: $(config.correlation_threshold)")
        println("="^60)

        # Force garbage collection before benchmark
        GC.gc()

        # Benchmark with allocation tracking
        bench_result = @timed begin
            result = concordance_analysis(
                model;
                optimizer=HiGHS.Optimizer,
                config=config
            )
            result
        end

        runtime = bench_result.time
        allocated_bytes = bench_result.bytes
        result = bench_result.value

        # Extract metrics
        stats = result.stats

        push!(results, (
            i,
            config.sample_size,
            config.sample_batch_size,
            config.concordance_batch_size,
            config.stage_size,
            config.correlation_threshold,
            runtime,
            allocated_bytes / 1e9,  # Convert to GB
            stats["n_complexes"],
            stats["n_candidate_pairs"],
            stats["n_concordant_pairs"],
            stats["n_modules"],
            allocated_bytes / 1e6   # Convert to MB
        ))

        println("Runtime: $(round(runtime, digits=2)) seconds")
        println("Memory allocated: $(round(allocated_bytes / 1e6, digits=1)) MB")
        println("Complexes: $(stats["n_complexes"])")
        println("Candidates: $(stats["n_candidate_pairs"])")
        println("Concordant pairs: $(stats["n_concordant_pairs"])")
        println("Modules: $(stats["n_modules"])")
    end

    return results
end

"""
Test buffer efficiency with different sizes.
"""
function test_buffer_efficiency()
    println("\n" * "="^60)
    println("Testing buffer efficiency...")
    println("="^60)

    # Test different buffer sizes
    buffer_sizes = [50, 100, 200, 500]
    n_complexes = 1000
    n_reactions = 500

    buffer_results = DataFrame(
        batch_size=Int[],
        allocation_time_ms=Float64[],
        resize_time_ms=Float64[]
    )

    for batch_size in buffer_sizes
        println("Testing batch size: $batch_size")

        # Test initial allocation
        alloc_time = @elapsed begin
            buffers = AnalysisBuffers(n_complexes, n_reactions, batch_size)
        end

        # Test buffer resizing
        resize_time = @elapsed begin
            ensure_buffer_capacity!(buffers, batch_size * 2, n_reactions * 2)
        end

        push!(buffer_results, (
            batch_size,
            alloc_time * 1000,  # Convert to ms
            resize_time * 1000
        ))

        println("  Allocation: $(round(alloc_time * 1000, digits=2)) ms")
        println("  Resize: $(round(resize_time * 1000, digits=2)) ms")
    end

    return buffer_results
end

"""
Memory usage profiling for different model sizes (simulated).
"""
function profile_memory_usage()
    println("\n" * "="^60)
    println("Profiling memory usage estimates...")
    println("="^60)

    # Simulate different model sizes
    model_sizes = [
        (500, 300),    # Small model
        (1500, 1000),  # Medium model  
        (5000, 3000),  # Large model
        (15000, 10000), # Very large model
        (50000, 30000)  # Huge model
    ]

    memory_estimates = DataFrame(
        n_complexes=Int[],
        n_reactions=Int[],
        sparse_matrix_gb=Float64[],
        correlation_gb=Float64[],
        tracker_gb=Float64[],
        total_gb=Float64[],
        shared_gb=Float64[]
    )

    for (n_complexes, n_reactions) in model_sizes
        estimates = COCOA.estimate_memory_usage(n_complexes, n_reactions)

        push!(memory_estimates, (
            n_complexes,
            n_reactions,
            estimates["sparse_matrix_GB"],
            estimates["correlation_overhead_GB"],
            estimates["tracker_GB"],
            estimates["total_GB"],
            estimates["total_with_sharing_GB"]
        ))

        println("Model: $(n_complexes) complexes, $(n_reactions) reactions")
        println("  Total memory: $(round(estimates["total_GB"], digits=2)) GB")
        println("  With sharing: $(round(estimates["total_with_sharing_GB"], digits=2)) GB")
    end

    return memory_estimates
end

"""
Test streaming correlation parameters.
"""
function test_streaming_parameters(model)
    println("\n" * "="^60)
    println("Testing streaming correlation parameters...")
    println("="^60)

    # Extract complexes for testing
    complexes, A_matrix, _ = COCOA.extract_complexes_and_incidence(model)
    balanced_complexes = COCOA.find_trivially_balanced_complexes(complexes)
    trivial_pairs = COCOA.find_trivially_concordant_pairs(complexes)

    # Mock positive/negative/unrestricted sets for testing
    positive_complexes = Set{Int}(1:100)
    negative_complexes = Set{Int}(101:200)
    unrestricted_complexes = Set{Int}(201:length(complexes))

    correlation_configs = [
        (sample_size=200, batch_size=50, threshold=0.90),
        (sample_size=500, batch_size=100, threshold=0.95),
        (sample_size=800, batch_size=100, threshold=0.95),
        (sample_size=500, batch_size=200, threshold=0.95),
    ]

    streaming_results = DataFrame(
        sample_size=Int[],
        batch_size=Int[],
        threshold=Float64[],
        runtime_seconds=Float64[],
        n_candidates=Int[],
        memory_mb=Float64[]
    )

    for (sample_size, batch_size, threshold) in correlation_configs
        config = ConcordanceConfig(
            sample_size=sample_size,
            sample_batch_size=batch_size,
            correlation_threshold=threshold
        )

        println("Testing: samples=$sample_size, batch=$batch_size, threshold=$threshold")

        # Benchmark streaming correlation
        bench_result = @timed begin
            candidates = COCOA.streaming_correlation_filter(
                model, complexes, A_matrix, balanced_complexes,
                positive_complexes, negative_complexes, unrestricted_complexes,
                trivial_pairs;
                optimizer=HiGHS.Optimizer,
                config=config
            )
            candidates
        end

        runtime = bench_result.time
        allocated_bytes = bench_result.bytes
        candidates = bench_result.value

        push!(streaming_results, (
            sample_size,
            batch_size,
            threshold,
            runtime,
            length(candidates),
            allocated_bytes / 1e6
        ))

        println("  Runtime: $(round(runtime, digits=2)) s")
        println("  Candidates: $(length(candidates))")
        println("  Memory: $(round(allocated_bytes / 1e6, digits=1)) MB")
    end

    return streaming_results
end

"""
Generate optimization recommendations based on benchmark results.
"""
function generate_recommendations(benchmark_results, buffer_results, memory_estimates, streaming_results)
    println("\n" * "="^80)
    println("OPTIMIZATION RECOMMENDATIONS")
    println("="^80)

    # Find best configuration by different criteria
    fastest_config = argmin(benchmark_results.runtime_seconds)
    memory_efficient = argmin(benchmark_results.memory_gb)
    balanced_config = argmin(benchmark_results.runtime_seconds .* benchmark_results.memory_gb)

    println("\n🚀 FASTEST CONFIGURATION (Config #$fastest_config):")
    fastest_row = benchmark_results[fastest_config, :]
    println("   Sample size: $(fastest_row.sample_size)")
    println("   Sample batch size: $(fastest_row.sample_batch_size)")
    println("   Concordance batch size: $(fastest_row.concordance_batch_size)")
    println("   Stage size: $(fastest_row.stage_size)")
    println("   Correlation threshold: $(fastest_row.correlation_threshold)")
    println("   Runtime: $(round(fastest_row.runtime_seconds, digits=2)) seconds")
    println("   Memory: $(round(fastest_row.memory_gb, digits=2)) GB")

    println("\n💾 MOST MEMORY EFFICIENT (Config #$memory_efficient):")
    memory_row = benchmark_results[memory_efficient, :]
    println("   Sample size: $(memory_row.sample_size)")
    println("   Sample batch size: $(memory_row.sample_batch_size)")
    println("   Concordance batch size: $(memory_row.concordance_batch_size)")
    println("   Stage size: $(memory_row.stage_size)")
    println("   Correlation threshold: $(memory_row.correlation_threshold)")
    println("   Runtime: $(round(memory_row.runtime_seconds, digits=2)) seconds")
    println("   Memory: $(round(memory_row.memory_gb, digits=2)) GB")

    println("\n⚖️  BEST BALANCED (Config #$balanced_config):")
    balanced_row = benchmark_results[balanced_config, :]
    println("   Sample size: $(balanced_row.sample_size)")
    println("   Sample batch size: $(balanced_row.sample_batch_size)")
    println("   Concordance batch size: $(balanced_row.concordance_batch_size)")
    println("   Stage size: $(balanced_row.stage_size)")
    println("   Correlation threshold: $(balanced_row.correlation_threshold)")
    println("   Runtime: $(round(balanced_row.runtime_seconds, digits=2)) seconds")
    println("   Memory: $(round(balanced_row.memory_gb, digits=2)) GB")

    # Buffer recommendations
    optimal_buffer = argmin(buffer_results.allocation_time_ms .+ buffer_results.resize_time_ms)
    println("\n🔧 OPTIMAL BUFFER SIZE: $(buffer_results[optimal_buffer, :batch_size])")

    # Memory scaling insights
    println("\n📊 MEMORY SCALING INSIGHTS:")
    largest_model = argmax(memory_estimates.n_complexes)
    println("   For $(memory_estimates[largest_model, :n_complexes]) complexes:")
    println("   Expected memory: $(round(memory_estimates[largest_model, :total_gb], digits=1)) GB")
    println("   With sharing: $(round(memory_estimates[largest_model, :shared_gb], digits=1)) GB")

    # Streaming recommendations
    best_streaming = argmin(streaming_results.runtime_seconds ./ streaming_results.n_candidates)
    println("\n🌊 OPTIMAL STREAMING PARAMETERS:")
    streaming_row = streaming_results[best_streaming, :]
    println("   Sample size: $(streaming_row.sample_size)")
    println("   Batch size: $(streaming_row.batch_size)")
    println("   Threshold: $(streaming_row.threshold)")
    println("   Efficiency: $(round(streaming_row.n_candidates / streaming_row.runtime_seconds, digits=1)) candidates/sec")

    println("\n" * "="^80)
    println("CONFIGURATION SUMMARY FOR YOUR USE CASE:")
    println("="^80)

    # Generate final recommendation
    println("\n# Recommended configuration for your E. coli model:")
    println("config = ConcordanceConfig(")
    println("    sample_size=$(balanced_row.sample_size),")
    println("    sample_batch_size=$(balanced_row.sample_batch_size),")
    println("    concordance_batch_size=$(balanced_row.concordance_batch_size),")
    println("    stage_size=$(balanced_row.stage_size),")
    println("    correlation_threshold=$(balanced_row.correlation_threshold),")
    println("    tolerance=1e-9,")
    println("    use_shared_arrays=false,  # Single worker")
    println("    seed=42")
    println(")")
    println("\n# Then run:")
    println("result = concordance_analysis(model; optimizer=HiGHS.Optimizer, config=config)")
end

"""
Create visualization plots for the benchmark results.
"""
function create_performance_plots(benchmark_results, buffer_results, streaming_results)
    println("\n📈 Creating performance visualization plots...")

    # Runtime vs Memory trade-off
    p1 = scatter(
        benchmark_results.runtime_seconds,
        benchmark_results.memory_gb,
        xlabel="Runtime (seconds)",
        ylabel="Memory (GB)",
        title="Runtime vs Memory Trade-off",
        markersize=8,
        alpha=0.7
    )

    # Sample size impact
    p2 = scatter(
        benchmark_results.sample_size,
        benchmark_results.runtime_seconds,
        xlabel="Sample Size",
        ylabel="Runtime (seconds)",
        title="Sample Size Impact on Runtime",
        markersize=8,
        alpha=0.7
    )

    # Buffer allocation performance
    p3 = plot(
        buffer_results.batch_size,
        buffer_results.allocation_time_ms,
        xlabel="Batch Size",
        ylabel="Time (ms)",
        title="Buffer Allocation Performance",
        label="Allocation",
        linewidth=3
    )
    plot!(p3, buffer_results.batch_size, buffer_results.resize_time_ms, label="Resize", linewidth=3)

    # Streaming efficiency
    p4 = scatter(
        streaming_results.sample_size,
        streaming_results.n_candidates ./ streaming_results.runtime_seconds,
        xlabel="Sample Size",
        ylabel="Candidates per Second",
        title="Streaming Correlation Efficiency",
        markersize=8,
        alpha=0.7
    )

    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))

    # Save plot
    plot_path = "cocoa_performance_analysis.png"
    savefig(combined_plot, plot_path)
    println("Performance plots saved to: $plot_path")

    return combined_plot
end

"""
Main benchmarking function.
"""
function main()
    println("🧪 COCOA.jl Performance Benchmarking Suite")
    println("="^60)

    # Test 1: Buffer efficiency
    buffer_results = test_buffer_efficiency()

    # Test 2: Memory usage profiling
    memory_estimates = profile_memory_usage()

    # Test 3: Streaming correlation parameters
    streaming_results = test_streaming_parameters(model)

    # Test 4: Full benchmark of parameter combinations
    benchmark_results = benchmark_parameter_combinations(model; max_tests=5)

    # Generate recommendations
    generate_recommendations(benchmark_results, buffer_results, memory_estimates, streaming_results)

    # Create visualizations
    plots = create_performance_plots(benchmark_results, buffer_results, streaming_results)

    # Save results
    println("\n💾 Saving benchmark results...")
    CSV.write("cocoa_benchmark_results.csv", benchmark_results)
    CSV.write("cocoa_buffer_results.csv", buffer_results)
    CSV.write("cocoa_memory_estimates.csv", memory_estimates)
    CSV.write("cocoa_streaming_results.csv", streaming_results)

    println("\nBenchmark complete! Results saved to CSV files.")
    println("Check the plots and recommendations above for optimization guidance.")

    return (
        benchmark_results=benchmark_results,
        buffer_results=buffer_results,
        memory_estimates=memory_estimates,
        streaming_results=streaming_results,
        plots=plots
    )
end

# Run the benchmark if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
