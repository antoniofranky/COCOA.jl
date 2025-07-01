"""
Quick memory and performance test for COCOA.jl optimizations.
"""

using Pkg
Pkg.activate(".")

using COCOA
using HiGHS
using COBREXA
using AbstractFBCModels

println("🧪 Quick COCOA Memory & Performance Test")
println("="^50)

# Load model
model_path = "c:\\Users\\anton\\master-thesis\\toolbox\\e_coli_core.xml"
if isfile(model_path)
    model = load_model(model_path)
    println("✅ Model loaded: $(length(AbstractFBCModels.reactions(model))) reactions, $(length(AbstractFBCModels.metabolites(model))) metabolites")
else
    println("❌ Model not found at $model_path")
    exit(1)
end

println("\n🔧 Testing buffer functionality...")

# Test 1: Buffer creation and resizing
println("Testing AnalysisBuffers...")
try
    buffers = AnalysisBuffers(100, 500, 50)
    println("✅ Buffer creation successful")

    # Test resizing
    original_capacity = size(buffers.activities, 2)
    ensure_buffer_capacity!(buffers, 100, 600)
    new_capacity = size(buffers.activities, 2)

    println("✅ Buffer resizing: $original_capacity → $new_capacity")

    if new_capacity >= 100
        println("✅ Buffer resize working correctly")
    else
        println("❌ Buffer resize failed")
    end
catch e
    println("❌ Buffer test failed: $e")
end

println("\n⚡ Testing basic concordance analysis...")

# Test 2: Quick concordance analysis with small parameters
quick_config = ConcordanceConfig(
    sample_size=200,
    sample_batch_size=50,
    concordance_batch_size=25,
    stage_size=50,
    correlation_threshold=0.95,
    tolerance=1e-9
)

println("Configuration:")
println("  Sample size: $(quick_config.sample_size)")
println("  Batch size: $(quick_config.sample_batch_size)")
println("  Correlation threshold: $(quick_config.correlation_threshold)")

try
    println("\nRunning concordance analysis...")

    # Time the analysis
    start_time = time()
    result = @timed concordance_analysis(
        model;
        optimizer=HiGHS.Optimizer,
        config=quick_config
    )
    analysis_time = time() - start_time

    # Extract results
    analysis_result = result.value
    memory_used = result.bytes

    stats = analysis_result.stats

    println("\n📊 Results:")
    println("  Runtime: $(round(analysis_time, digits=2)) seconds")
    println("  Memory allocated: $(round(memory_used / 1e6, digits=1)) MB")
    println("  Complexes found: $(stats["n_complexes"])")
    println("  Candidate pairs: $(stats["n_candidate_pairs"])")
    println("  Concordant pairs: $(stats["n_concordant_pairs"])")
    println("  Modules found: $(stats["n_modules"])")
    println("  Stages completed: $(stats["stages_completed"])")

    # Calculate efficiency metrics
    candidates_per_second = stats["n_candidate_pairs"] / analysis_time
    memory_per_complex = memory_used / stats["n_complexes"]

    println("\n⚡ Efficiency Metrics:")
    println("  Candidates processed per second: $(round(candidates_per_second, digits=1))")
    println("  Memory per complex: $(round(memory_per_complex / 1e3, digits=1)) KB")

    # Memory efficiency assessment
    if memory_used < 100e6  # Less than 100 MB
        println("✅ Excellent memory efficiency")
    elseif memory_used < 500e6  # Less than 500 MB
        println("✅ Good memory efficiency")
    else
        println("⚠️  High memory usage - consider smaller batch sizes")
    end

    # Speed assessment
    if analysis_time < 60  # Less than 1 minute
        println("✅ Fast execution")
    elseif analysis_time < 300  # Less than 5 minutes
        println("✅ Reasonable execution time")
    else
        println("⚠️  Slow execution - consider optimizing parameters")
    end

catch e
    println("❌ Analysis failed: $e")
end

println("\n🧠 Memory Usage Estimates for Larger Models:")

# Test 3: Memory scaling estimates
model_sizes = [
    (1000, 800, "Small"),
    (5000, 3000, "Medium"),
    (15000, 10000, "Large"),
    (50000, 30000, "Very Large")
]

for (n_complexes, n_reactions, size_name) in model_sizes
    estimates = COCOA.estimate_memory_usage(n_complexes, n_reactions)

    println("$size_name model ($n_complexes complexes, $n_reactions reactions):")
    println("  Estimated memory: $(round(estimates["total_GB"], digits=2)) GB")
    if estimates["total_GB"] > 8
        println("  ⚠️  Consider using shared arrays or reducing batch sizes")
    end
end

println("\n🎯 Parameter Recommendations:")
println("For your current model size:")

# Generate specific recommendations based on model size
n_reactions = length(AbstractFBCModels.reactions(model))
n_metabolites = length(AbstractFBCModels.metabolites(model))

if n_reactions < 500
    println("  Recommended sample_size: 500-1000")
    println("  Recommended sample_batch_size: 100-200")
    println("  Recommended stage_size: 200-500")
elseif n_reactions < 2000
    println("  Recommended sample_size: 800-1200")
    println("  Recommended sample_batch_size: 100-150")
    println("  Recommended stage_size: 100-300")
else
    println("  Recommended sample_size: 1000-1500")
    println("  Recommended sample_batch_size: 50-100")
    println("  Recommended stage_size: 100-200")
end

println("\nGeneral optimization tips:")
println("  • Use correlation_threshold=0.95 for balanced speed/accuracy")
println("  • Increase batch sizes for more memory but faster processing")
println("  • Decrease batch sizes for less memory but slower processing")
println("  • Use shared arrays (use_shared_arrays=true) for multi-worker setups")

println("\n✨ Test completed!")
println("For comprehensive benchmarking, run: julia benchmark_parameters.jl")
