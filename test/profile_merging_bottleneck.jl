"""
Profile the cost of individual operations in advanced merging
"""

using COCOA
using HiGHS
using LinearAlgebra

# Create model and run concordance analysis
model = create_envz_ompr_model()
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    kinetic_analysis=false,
    sample_size=100,
    concordance_tolerance=0.01,
    cv_threshold=0.01
)
concordance_modules = extract_concordance_modules(results)

# Extract network topology (mimic what kinetic_analysis does)
A_matrix, complex_ids = COCOA.incidence(model; return_ids=true)
Y_matrix, metabolite_ids, _ = COCOA.complex_stoichiometry(model; return_ids=true)
complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

println("="^80)
println("PROFILING ADVANCED MERGING OPERATIONS")
println("="^80)

# Create some upstream sets to test with
balanced = concordance_modules[1]
upstream_sets = Set{Symbol}[]
for conc_module in concordance_modules[2:end]
    extended_module = balanced ∪ conc_module
    # For profiling, just use the extended modules directly
    push!(upstream_sets, extended_module)
end

n = length(upstream_sets)
println("\nTest setup:")
println("  - Number of upstream sets: $n")
println("  - Pairwise comparisons: $((n * (n-1)) ÷ 2)")

# Step 1: Profile building Y_Delta
println("\n" * "-"^80)
println("Operation 1: Building Y_Delta matrix")
println("-"^80)
@time Y_Delta = COCOA.build_coupling_companion_matrix(upstream_sets, Y_matrix, complex_to_idx)
println("  Y_Delta size: $(size(Y_Delta))")
println("  Memory: $(sizeof(Y_Delta) / 1024) KB")

# Step 2: Profile a single pairwise check
println("\n" * "-"^80)
println("Operation 2: Single pairwise comparison")
println("-"^80)

if n >= 2
    c_alpha = first(upstream_sets[1])
    c_beta = first(upstream_sets[2])
    idx_alpha = complex_to_idx[c_alpha]
    idx_beta = complex_to_idx[c_beta]

    println("Comparing: $c_alpha vs $c_beta")

    # Profile y_diff computation
    print("  - Computing y_diff: ")
    @time y_diff = collect(Y_matrix[:, idx_alpha] - Y_matrix[:, idx_beta])

    # Profile linear solve
    print("  - Solving Y_Delta \\ y_diff: ")
    @time can_merge = COCOA.can_merge_via_proposition_s34(y_diff, Y_Delta)
    println("  - Result: $can_merge")
end

# Step 3: Profile rebuilding Y_Delta (happens after each merge)
println("\n" * "-"^80)
println("Operation 3: Rebuilding Y_Delta after merge")
println("-"^80)
# Simulate a merge by combining two sets
if n >= 2
    merged_sets = [upstream_sets[1] ∪ upstream_sets[2]]
    append!(merged_sets, upstream_sets[3:end])

    @time Y_Delta_rebuilt = COCOA.build_coupling_companion_matrix(merged_sets, Y_matrix, complex_to_idx)
    println("  New Y_Delta size: $(size(Y_Delta_rebuilt))")
end

# Step 4: Test with synthetic larger case
println("\n" * "="^80)
println("SCALING TEST: Synthetic larger model")
println("="^80)

function create_synthetic_sets(n_sets::Int, n_complexes_per_set::Int)
    sets = Set{Symbol}[]
    for i in 1:n_sets
        push!(sets, Set(Symbol("S$(i)_C$j") for j in 1:n_complexes_per_set))
    end
    return sets
end

for n_test in [10, 50, 100]
    println("\nTesting with $n_test sets:")
    synthetic_sets = create_synthetic_sets(n_test, 10)
    n_pairs = (n_test * (n_test - 1)) ÷ 2
    println("  - Pairwise comparisons: $n_pairs")

    # Build a synthetic Y_Delta (just random matrix for testing)
    n_metabolites = 100
    n_columns = sum(length(s) - 1 for s in synthetic_sets if length(s) >= 2)
    Y_Delta_synthetic = randn(n_metabolites, max(1, n_columns))

    print("  - Time for $n_pairs linear solves: ")
    @time begin
        for i in 1:n_test, j in (i+1):n_test
            y_diff_test = randn(n_metabolites)
            xi = Y_Delta_synthetic \ y_diff_test
            residual_norm = norm(Y_Delta_synthetic * xi - y_diff_test)
        end
    end
end

println("\n" * "="^80)
println("ANALYSIS")
println("="^80)

println("""
Key findings:
1. Building Y_Delta: O(total_complexes) - relatively cheap
2. Linear solve: O(m×n²) where m=metabolites, n=columns in Y_Delta
3. Rebuilding Y_Delta: Done after EVERY merge - very expensive in nested loop

The nested loop structure is the killer:
  - Outer: up to 10 iterations
  - Middle: n × (n-1) / 2 pairwise comparisons  
  - Inner: Each merge triggers Y_Delta rebuild

For 4,798 modules:
  - 11.5M potential comparisons
  - Each comparison: linear solve (~100μs estimated)
  - Each merge: rebuild Y_Delta (~ms range)
  - Total: could be hours without optimizations!
""")

println("\n✓ Profiling complete!")
