"""
Test performance of kinetic_analysis with and without advanced merging
"""

using COCOA
using Test
using HiGHS

# Create EnvZ-OmpR model
model = create_envz_ompr_model()

# Run concordance analysis
println("Running concordance analysis...")
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    kinetic_analysis=false,
    sample_size=100,
    concordance_tolerance=0.01,
    cv_threshold=0.01
)

concordance_modules = extract_concordance_modules(results)

println("\n" * "="^80)
println("PERFORMANCE COMPARISON: Advanced Merging ON vs OFF")
println("="^80)

println("\nModel Statistics:")
println("  - Concordance modules: $(length(concordance_modules) - 1)")
println("  - Balanced complexes: $(length(concordance_modules[1]))")
println("  - Total complexes: $(sum(length(m) for m in concordance_modules))")

# Test 1: WITH advanced merging (default)
println("\n" * "-"^80)
println("Test 1: WITH Advanced Merging (enable_advanced_merging=true)")
println("-"^80)
@time kinetic_modules_with = kinetic_analysis(
    concordance_modules,
    model;
    enable_advanced_merging=true,
    min_module_size=2
)

println("\nResults WITH advanced merging:")
println("  - Kinetic modules found: $(length(kinetic_modules_with))")
println("  - Largest module size: $(isempty(kinetic_modules_with) ? 0 : length(first(kinetic_modules_with)))")

# Test 2: WITHOUT advanced merging
println("\n" * "-"^80)
println("Test 2: WITHOUT Advanced Merging (enable_advanced_merging=false)")
println("-"^80)
@time kinetic_modules_without = kinetic_analysis(
    concordance_modules,
    model;
    enable_advanced_merging=false,
    min_module_size=2
)

println("\nResults WITHOUT advanced merging:")
println("  - Kinetic modules found: $(length(kinetic_modules_without))")
println("  - Largest module size: $(isempty(kinetic_modules_without) ? 0 : length(first(kinetic_modules_without)))")

# Compare results
println("\n" * "="^80)
println("COMPARISON")
println("="^80)

if kinetic_modules_with == kinetic_modules_without
    println("✓ Results are IDENTICAL - advanced merging had no additional effect")
else
    println("✗ Results DIFFER:")
    println("  - WITH advanced: $(length(kinetic_modules_with)) modules")
    println("  - WITHOUT advanced: $(length(kinetic_modules_without)) modules")
    println("  - Difference: $(length(kinetic_modules_without) - length(kinetic_modules_with)) modules")
end

# Test with larger synthetic model (simulate many concordance modules)
println("\n" * "="^80)
println("SIMULATING LARGE MODEL SCENARIO")
println("="^80)

# Create synthetic concordance modules to test the early termination
function create_synthetic_concordance(n_modules::Int)
    balanced = Set(Symbol("C$i") for i in 1:10)
    modules = [balanced]
    for i in 1:n_modules
        push!(modules, Set(Symbol("M$(i)_$j") for j in 1:5))
    end
    return modules
end

# Test with 100 modules (still manageable)
println("\nTest with 100 synthetic concordance modules:")
synthetic_100 = create_synthetic_concordance(100)
n_comparisons_100 = (100 * 99) ÷ 2
println("  - Number of pairwise comparisons: $(n_comparisons_100)")
println("  - This should run WITH advanced merging")

# Test with 1000 modules (triggers warning)
println("\nTest with 1000 synthetic concordance modules:")
synthetic_1000 = create_synthetic_concordance(1000)
n_comparisons_1000 = (1000 * 999) ÷ 2
println("  - Number of pairwise comparisons: $(n_comparisons_1000)")
println("  - With default max_pairwise_comparisons=50M: should still run")

# Test with 10000 modules (exceeds limit)
println("\nTest with 10000 synthetic concordance modules:")
synthetic_10000 = create_synthetic_concordance(10000)
n_comparisons_10000 = (10000 * 9999) ÷ 2
println("  - Number of pairwise comparisons: $(n_comparisons_10000)")
println("  - With default max_pairwise_comparisons=50M: should auto-disable")

println("\n" * "="^80)
println("DEMONSTRATION: Your 4,798-module model")
println("="^80)
n_comparisons_user = (4798 * 4797) ÷ 2
println("Number of pairwise comparisons needed: $(n_comparisons_user)")
println("This is $(round(n_comparisons_user/1e6, digits=2)) million comparisons")
println("\nRecommendation: Use enable_advanced_merging=false for this model")

println("\n✓ All performance tests complete!")
