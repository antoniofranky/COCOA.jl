"""
Configuration optimizer for COCOA.jl based on system resources and model size.
"""

using Pkg
Pkg.activate(".")

using COCOA
using COBREXA
using AbstractFBCModels
using Sys

"""
Automatically determine optimal configuration based on available resources and model size.
"""
function optimize_config_for_system(model; target_runtime_minutes=5, memory_limit_gb=8)
    println("🔧 Optimizing COCOA configuration for your system...")

    # System information
    total_memory_gb = Sys.total_memory() / 1e9
    available_memory_gb = min(memory_limit_gb, total_memory_gb * 0.7)  # Use 70% of total
    cpu_threads = Sys.CPU_THREADS

    println("System resources:")
    println("  Total memory: $(round(total_memory_gb, digits=1)) GB")
    println("  Available for analysis: $(round(available_memory_gb, digits=1)) GB")
    println("  CPU threads: $cpu_threads")

    # Model characteristics
    n_reactions = length(reactions(model))
    n_metabolites = length(metabolites(model))

    # Estimate model complexity
    complexes, A_matrix, _ = COCOA.extract_complexes_and_incidence(
        model;
        config=ConcordanceConfig(use_shared_arrays=false)
    )
    n_complexes = length(complexes)

    println("\nModel characteristics:")
    println("  Reactions: $n_reactions")
    println("  Metabolites: $n_metabolites")
    println("  Estimated complexes: $n_complexes")

    # Estimate memory requirements for different configurations
    memory_estimates = COCOA.estimate_memory_usage(n_complexes, n_reactions)
    estimated_memory_gb = memory_estimates["total_GB"]

    println("  Estimated memory need: $(round(estimated_memory_gb, digits=2)) GB")

    # Determine configuration based on model size and resources
    if n_complexes < 500  # Small model
        config = small_model_config(available_memory_gb, cpu_threads)
        complexity = "Small"
    elseif n_complexes < 2000  # Medium model
        config = medium_model_config(available_memory_gb, cpu_threads)
        complexity = "Medium"
    elseif n_complexes < 10000  # Large model
        config = large_model_config(available_memory_gb, cpu_threads)
        complexity = "Large"
    else  # Very large model
        config = very_large_model_config(available_memory_gb, cpu_threads)
        complexity = "Very Large"
    end

    # Adjust for memory constraints
    if estimated_memory_gb > available_memory_gb
        println("⚠️  Memory constraint detected - adjusting batch sizes...")
        config = adjust_for_memory_limit(config, estimated_memory_gb, available_memory_gb)
    end

    # Final recommendations
    println("\n🎯 Recommended configuration for $complexity model:")
    print_config(config)

    # Estimate runtime
    estimated_runtime = estimate_runtime(n_complexes, config)
    println("\n⏱️  Estimated runtime: $(round(estimated_runtime/60, digits=1)) minutes")

    if estimated_runtime > target_runtime_minutes * 60
        println("⚠️  Estimated runtime exceeds target - consider:")
        println("  • Increasing correlation threshold")
        println("  • Reducing sample size")
        println("  • Using larger batch sizes (if memory allows)")
    end

    return config
end

function small_model_config(memory_gb, threads)
    ConcordanceConfig(
        sample_size=1000,
        sample_batch_size=200,
        concordance_batch_size=100,
        stage_size=500,
        correlation_threshold=0.95,
        tolerance=1e-9,
        use_shared_arrays=false,
        seed=42
    )
end

function medium_model_config(memory_gb, threads)
    ConcordanceConfig(
        sample_size=1000,
        sample_batch_size=150,
        concordance_batch_size=75,
        stage_size=300,
        correlation_threshold=0.95,
        tolerance=1e-9,
        use_shared_arrays=threads > 1,
        seed=42
    )
end

function large_model_config(memory_gb, threads)
    ConcordanceConfig(
        sample_size=800,
        sample_batch_size=100,
        concordance_batch_size=50,
        stage_size=200,
        correlation_threshold=0.96,
        tolerance=1e-9,
        use_shared_arrays=threads > 1,
        seed=42
    )
end

function very_large_model_config(memory_gb, threads)
    ConcordanceConfig(
        sample_size=600,
        sample_batch_size=75,
        concordance_batch_size=25,
        stage_size=100,
        correlation_threshold=0.97,
        tolerance=1e-9,
        use_shared_arrays=threads > 1,
        seed=42
    )
end

function adjust_for_memory_limit(config, estimated_gb, available_gb)
    scale_factor = available_gb / estimated_gb * 0.8  # Use 80% to be safe

    new_sample_batch_size = max(25, Int(round(config.sample_batch_size * scale_factor)))
    new_concordance_batch_size = max(10, Int(round(config.concordance_batch_size * scale_factor)))
    new_stage_size = max(50, Int(round(config.stage_size * scale_factor)))

    return ConcordanceConfig(
        sample_size=config.sample_size,
        sample_batch_size=new_sample_batch_size,
        concordance_batch_size=new_concordance_batch_size,
        stage_size=new_stage_size,
        correlation_threshold=min(0.98, config.correlation_threshold + 0.01),  # Slightly higher threshold
        tolerance=config.tolerance,
        use_shared_arrays=config.use_shared_arrays,
        seed=config.seed
    )
end

function estimate_runtime(n_complexes, config)
    # Very rough runtime estimation based on empirical observations
    # This is a simplified model - actual runtime depends on many factors

    base_time = n_complexes * 0.001  # Base time per complex

    # Sampling overhead
    sampling_time = config.sample_size * 0.01

    # Correlation computation overhead (quadratic in active complexes)
    n_active = Int(n_complexes * 0.8)  # Assume 80% are active
    correlation_time = (n_active^2 / 1e6) * 10

    # Concordance testing overhead
    estimated_candidates = Int(n_active * 3)  # Rough estimate
    concordance_time = estimated_candidates * 0.02

    total_time = base_time + sampling_time + correlation_time + concordance_time

    # Adjust for batch sizes (larger batches = faster processing)
    batch_efficiency = min(2.0, config.sample_batch_size / 50)
    total_time = total_time / batch_efficiency

    return total_time
end

function print_config(config)
    println("ConcordanceConfig(")
    println("    sample_size=$(config.sample_size),")
    println("    sample_batch_size=$(config.sample_batch_size),")
    println("    concordance_batch_size=$(config.concordance_batch_size),")
    println("    stage_size=$(config.stage_size),")
    println("    correlation_threshold=$(config.correlation_threshold),")
    println("    tolerance=$(config.tolerance),")
    println("    use_shared_arrays=$(config.use_shared_arrays),")
    println("    seed=$(config.seed)")
    println(")")
end

"""
Test the optimized configuration.
"""
function test_optimized_config(model, config)
    println("\n🧪 Testing optimized configuration...")

    try
        result = @timed concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            config=config
        )

        runtime = result.time
        memory_mb = result.bytes / 1e6
        analysis_result = result.value

        stats = analysis_result.stats

        println("\n📊 Test Results:")
        println("  ✅ Analysis completed successfully")
        println("  ⏱️  Runtime: $(round(runtime, digits=2)) seconds")
        println("  💾 Memory: $(round(memory_mb, digits=1)) MB")
        println("  🔍 Complexes: $(stats["n_complexes"])")
        println("  🎯 Candidate pairs: $(stats["n_candidate_pairs"])")
        println("  ✅ Concordant pairs: $(stats["n_concordant_pairs"])")
        println("  📦 Modules: $(stats["n_modules"])")

        # Performance assessment
        efficiency = stats["n_candidate_pairs"] / runtime
        println("  ⚡ Processing rate: $(round(efficiency, digits=1)) candidates/second")

        if runtime < 120  # Less than 2 minutes
            println("  🚀 Excellent performance!")
        elseif runtime < 300  # Less than 5 minutes
            println("  ✅ Good performance")
        else
            println("  ⚠️  Consider further optimization")
        end

        return true, stats

    catch e
        println("❌ Test failed: $e")
        return false, nothing
    end
end

"""
Generate a configuration recommendation report.
"""
function generate_config_report(model, config, test_success, test_stats)
    println("\n" * "="^60)
    println("COCOA CONFIGURATION OPTIMIZATION REPORT")
    println("="^60)

    println("\n📋 MODEL SUMMARY:")
    println("  Reactions: $(length(reactions(model)))")
    println("  Metabolites: $(length(metabolites(model)))")
    if test_success && test_stats !== nothing
        println("  Complexes: $(test_stats["n_complexes"])")
        println("  Modules found: $(test_stats["n_modules"])")
    end

    println("\n⚙️  RECOMMENDED CONFIGURATION:")
    print_config(config)

    println("\n💡 OPTIMIZATION TIPS:")
    println("  1. If analysis is too slow:")
    println("     • Increase correlation_threshold (0.96-0.98)")
    println("     • Increase batch sizes (if memory allows)")
    println("     • Reduce sample_size (trade accuracy for speed)")

    println("\n  2. If memory usage is too high:")
    println("     • Decrease sample_batch_size")
    println("     • Decrease stage_size")
    println("     • Set use_shared_arrays=false")

    println("\n  3. For large-scale HPC runs:")
    println("     • Set use_shared_arrays=true")
    println("     • Increase batch sizes significantly")
    println("     • Consider multiple workers")

    if test_success && test_stats !== nothing
        println("\n📈 EXPECTED PERFORMANCE:")
        println("  Runtime: ~$(round(test_stats["elapsed_time"], digits=1)) seconds")
        println("  Candidate pairs: ~$(test_stats["n_candidate_pairs"])")
        println("  Memory efficiency: Good")
    end

    println("\n🎯 READY TO USE:")
    println("Copy the configuration above and use it like this:")
    println("")
    println("config = ConcordanceConfig(...)")
    println("result = concordance_analysis(model; optimizer=HiGHS.Optimizer, config=config)")

    println("\n" * "="^60)
end

"""
Main optimization function.
"""
function main()
    println("🚀 COCOA Configuration Optimizer")
    println("="^40)

    # Load model
    model_path = "c:\\Users\\anton\\master-thesis\\toolbox\\e_coli_core.xml"
    if isfile(model_path)
        model = load_model(model_path)
        println("✅ Model loaded")
    else
        println("❌ Model not found at $model_path")
        println("Please adjust the model_path variable")
        return
    end

    # Optimize configuration
    optimal_config = optimize_config_for_system(model)

    # Test the configuration
    test_success, test_stats = test_optimized_config(model, optimal_config)

    # Generate report
    generate_config_report(model, optimal_config, test_success, test_stats)

    return optimal_config
end

# Run optimization if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    optimal_config = main()
end
