"""
Memory profiling script to test buffer optimization effectiveness.
"""

using Pkg
Pkg.activate(".")

using COCOA
using HiGHS
using COBREXA
using AbstractFBCModels
using BenchmarkTools
using Profile

println("🔬 COCOA Buffer Optimization Profiling")
println("="^50)

# Load model
model_path = "c:\\Users\\anton\\master-thesis\\toolbox\\e_coli_core.xml"
if isfile(model_path)
    model = load_model(model_path)
    println("✅ Model loaded")
else
    println("❌ Model not found")
    exit(1)
end

"""
Compare memory allocation with and without buffer pre-allocation.
"""
function compare_buffer_strategies()
    println("\n🧪 Comparing buffer strategies...")

    # Extract complexes for testing
    complexes, A_matrix, _ = COCOA.extract_complexes_and_incidence(model)
    n_complexes = length(complexes)
    n_reactions = size(sparse(A_matrix), 2)

    println("Model stats: $n_complexes complexes, $n_reactions reactions")

    # Test different scenarios
    scenarios = [
        (batch_size=50, n_batches=10, name="Small batches"),
        (batch_size=100, n_batches=10, name="Medium batches"),
        (batch_size=200, n_batches=5, name="Large batches"),
    ]

    for scenario in scenarios
        println("\n📊 Testing: $(scenario.name)")
        println("  Batch size: $(scenario.batch_size)")
        println("  Number of batches: $(scenario.n_batches)")

        # Test WITH pre-allocated buffers
        GC.gc()  # Clean up before test

        with_buffers_time = @elapsed begin
            # Pre-allocate buffers once
            buffers = AnalysisBuffers(n_complexes, n_reactions, scenario.batch_size)

            for batch in 1:scenario.n_batches
                # Simulate buffer usage by filling with random data
                for i in 1:scenario.batch_size
                    buffers.activities[:, i] .= rand(n_complexes)
                    buffers.sample_buffer[i, :] .= rand(n_reactions)
                end

                # Simulate buffer resize if needed
                if batch == 3
                    ensure_buffer_capacity!(buffers, scenario.batch_size * 2, n_reactions)
                end
            end
        end

        # Test WITHOUT pre-allocated buffers (naive approach)
        GC.gc()  # Clean up before test

        without_buffers_time = @elapsed begin
            for batch in 1:scenario.n_batches
                # Allocate new arrays each time (naive approach)
                activities = zeros(Float64, n_complexes, scenario.batch_size)
                sample_buffer = zeros(Float64, scenario.batch_size, n_reactions)

                # Fill with data (same work as above)
                for i in 1:scenario.batch_size
                    activities[:, i] .= rand(n_complexes)
                    sample_buffer[i, :] .= rand(n_reactions)
                end
            end
        end

        # Calculate improvement
        speedup = without_buffers_time / with_buffers_time
        improvement_pct = (1 - with_buffers_time / without_buffers_time) * 100

        println("  ⏱️  With buffers: $(round(with_buffers_time * 1000, digits=2)) ms")
        println("  ⏱️  Without buffers: $(round(without_buffers_time * 1000, digits=2)) ms")
        println("  🚀 Speedup: $(round(speedup, digits=2))x")
        println("  📈 Improvement: $(round(improvement_pct, digits=1))%")

        if speedup > 1.5
            println("  ✅ Significant improvement with buffer pre-allocation")
        elseif speedup > 1.1
            println("  ✅ Moderate improvement with buffer pre-allocation")
        else
            println("  ⚠️  Minimal improvement - overhead may be dominating")
        end
    end
end

"""
Profile memory allocation patterns during streaming correlation.
"""
function profile_streaming_allocation()
    println("\n🔍 Profiling streaming correlation memory patterns...")

    # Configure for profiling
    config_small = ConcordanceConfig(
        sample_size=100,
        sample_batch_size=25,
        correlation_threshold=0.95
    )

    config_large = ConcordanceConfig(
        sample_size=400,
        sample_batch_size=100,
        correlation_threshold=0.95
    )

    # Extract necessary components
    complexes, A_matrix, _ = COCOA.extract_complexes_and_incidence(model)
    balanced_complexes = COCOA.find_trivially_balanced_complexes(complexes)
    trivial_pairs = COCOA.find_trivially_concordant_pairs(complexes)

    # Mock classification for testing
    positive_complexes = Set{Int}(1:min(50, length(complexes)))
    negative_complexes = Set{Int}(51:min(100, length(complexes)))
    unrestricted_complexes = Set{Int}(101:length(complexes))

    for (config, name) in [(config_small, "Small config"), (config_large, "Large config")]
        println("\n📊 Testing: $name")
        println("  Sample size: $(config.sample_size)")
        println("  Batch size: $(config.sample_batch_size)")

        # Profile memory allocation
        allocation_result = @timed begin
            candidates = COCOA.streaming_correlation_filter(
                model, complexes, A_matrix, balanced_complexes,
                positive_complexes, negative_complexes, unrestricted_complexes,
                trivial_pairs;
                optimizer=HiGHS.Optimizer,
                config=config
            )
            candidates
        end

        runtime = allocation_result.time
        allocated_bytes = allocation_result.bytes
        candidates = allocation_result.value

        # Calculate metrics
        mb_allocated = allocated_bytes / 1e6
        allocation_rate = allocated_bytes / runtime / 1e6  # MB/s
        candidates_per_mb = length(candidates) / mb_allocated

        println("  ⏱️  Runtime: $(round(runtime, digits=2)) seconds")
        println("  💾 Memory allocated: $(round(mb_allocated, digits=1)) MB")
        println("  📈 Allocation rate: $(round(allocation_rate, digits=1)) MB/s")
        println("  🎯 Candidates found: $(length(candidates))")
        println("  ⚡ Efficiency: $(round(candidates_per_mb, digits=1)) candidates/MB")

        # Memory efficiency assessment
        if mb_allocated < 50
            println("  ✅ Excellent memory efficiency")
        elseif mb_allocated < 200
            println("  ✅ Good memory efficiency")
        else
            println("  ⚠️  High memory usage")
        end
    end
end

"""
Benchmark buffer resizing performance.
"""
function benchmark_buffer_resize()
    println("\n📏 Benchmarking buffer resize performance...")

    initial_sizes = [50, 100, 200]
    resize_factors = [1.5, 2.0, 3.0]

    for initial_size in initial_sizes
        println("\n🔧 Initial batch size: $initial_size")

        # Create initial buffer
        buffers = AnalysisBuffers(1000, 500, initial_size)

        for factor in resize_factors
            new_size = Int(round(initial_size * factor))

            # Benchmark resize operation
            resize_time = @elapsed begin
                ensure_buffer_capacity!(buffers, new_size, 500)
            end

            # Check if resize worked
            actual_capacity = size(buffers.activities, 2)

            println("  📈 Resize to $(new_size): $(round(resize_time * 1000, digits=2)) ms")
            println("     Actual capacity: $actual_capacity")

            if actual_capacity >= new_size
                println("     ✅ Resize successful")
            else
                println("     ❌ Resize failed")
            end
        end
    end
end

"""
Test allocation patterns with different correlation thresholds.
"""
function test_correlation_threshold_impact()
    println("\n🎯 Testing correlation threshold impact on memory...")

    thresholds = [0.85, 0.90, 0.95, 0.98]

    # Setup base configuration
    base_config = ConcordanceConfig(
        sample_size=300,
        sample_batch_size=75,
        correlation_threshold=0.95  # Will be overridden
    )

    # Extract necessary components
    complexes, A_matrix, _ = COCOA.extract_complexes_and_incidence(model)
    balanced_complexes = COCOA.find_trivially_balanced_complexes(complexes)
    trivial_pairs = COCOA.find_trivially_concordant_pairs(complexes)

    positive_complexes = Set{Int}(1:min(50, length(complexes)))
    negative_complexes = Set{Int}(51:min(100, length(complexes)))
    unrestricted_complexes = Set{Int}(101:length(complexes))

    for threshold in thresholds
        println("\n📊 Testing threshold: $threshold")

        # Update config
        test_config = ConcordanceConfig(
            sample_size=base_config.sample_size,
            sample_batch_size=base_config.sample_batch_size,
            correlation_threshold=threshold,
            tolerance=base_config.tolerance
        )

        # Test streaming correlation
        result = @timed begin
            candidates = COCOA.streaming_correlation_filter(
                model, complexes, A_matrix, balanced_complexes,
                positive_complexes, negative_complexes, unrestricted_complexes,
                trivial_pairs;
                optimizer=HiGHS.Optimizer,
                config=test_config
            )
            candidates
        end

        runtime = result.time
        allocated_bytes = result.bytes
        candidates = result.value

        # Calculate candidates as percentage of all possible pairs
        n_active = length(complexes) - length(balanced_complexes)
        max_pairs = n_active * (n_active - 1) ÷ 2
        candidate_percentage = length(candidates) / max_pairs * 100

        println("  Candidates: $(length(candidates)) ($(round(candidate_percentage, digits=1))% of possible)")
        println("  Memory: $(round(allocated_bytes / 1e6, digits=1)) MB")
        println("  Runtime: $(round(runtime, digits=2)) s")

        # Memory per candidate
        if length(candidates) > 0
            memory_per_candidate = allocated_bytes / length(candidates) / 1e3  # KB
            println("  Memory/candidate: $(round(memory_per_candidate, digits=1)) KB")
        end
    end
end

"""
Main profiling function.
"""
function main()
    println("Starting comprehensive memory profiling...")

    # Test 1: Buffer strategy comparison
    compare_buffer_strategies()

    # Test 2: Streaming allocation profiling
    profile_streaming_allocation()

    # Test 3: Buffer resize benchmarking
    benchmark_buffer_resize()

    # Test 4: Correlation threshold impact
    test_correlation_threshold_impact()

    println("\n✨ Memory profiling completed!")
    println("\n💡 Key takeaways:")
    println("  1. Use pre-allocated buffers for repeated operations")
    println("  2. Choose batch sizes based on available memory")
    println("  3. Higher correlation thresholds reduce candidate pairs and memory usage")
    println("  4. Buffer resizing is fast - don't over-allocate initially")
    println("  5. Monitor allocation rates during streaming correlation")
end

# Run profiling
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
