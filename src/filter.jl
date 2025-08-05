"""
filter.jl - Ultra memory-efficient CV filtering using OnlineStats.jl

Streaming three-stage pipeline focused purely on CV computation.
"""

using Base.Threads
using OnlineStats
using DataStructures

# --- Bit vector optimization helpers ---
"""Convert Set{Int} to BitVector for O(1) lookups and better cache efficiency"""
@inline function set_to_bitvector(indices_set::Set{Int}, n::Int)
    bv = falses(n)
    for idx in indices_set
        if 1 <= idx <= n
            bv[idx] = true
        end
    end
    return bv
end

# --- Numerically stable online variance computation ---
mutable struct StableVariance
    n::Int
    mean::Float64
    m2::Float64  # Sum of squares of differences from current mean

    StableVariance() = new(0, 0.0, 0.0)
end

@inline function add_sample!(sv::StableVariance, x::Float64)
    sv.n += 1
    delta = x - sv.mean
    sv.mean += delta / sv.n
    delta2 = x - sv.mean
    sv.m2 += delta * delta2
    return sv
end

@inline function get_mean(sv::StableVariance)::Float64
    return sv.mean
end

@inline function get_variance(sv::StableVariance)::Float64
    sv.n < 2 && return 0.0
    return sv.m2 / (sv.n - 1)
end

@inline function get_std(sv::StableVariance)::Float64
    return sqrt(get_variance(sv))
end

@inline function get_count(sv::StableVariance)::Int
    return sv.n
end

# --- Helper functions for CV ---
@inline function compute_cv(variance_stat::Variance, epsilon::Float64=1e-8)
    n = nobs(variance_stat)
    m = mean(variance_stat)
    s = std(variance_stat)

    n < 2 && return Inf
    abs(m) < epsilon && return Inf

    return s / abs(m)
end

# Stable version using Welford's algorithm
@inline function compute_cv_stable(sv::StableVariance, epsilon::Float64=1e-8)
    n = get_count(sv)
    m = get_mean(sv)
    s = get_std(sv)

    n < 2 && return Inf
    abs(m) < epsilon && return Inf

    return s / abs(m)
end

# Generic CV computation that chooses algorithm based on configuration
@inline function compute_cv_adaptive(
    c1_samples::Vector{Float64},
    c2_samples::Vector{Float64},
    start_idx::Int,
    end_idx::Int,
    use_stable_variance::Bool,
    cv_epsilon::Float64
)
    if use_stable_variance
        stable_var = StableVariance()
        compute_ratios_batch_stable!(stable_var, c1_samples, c2_samples, start_idx, end_idx, cv_epsilon)
        return compute_cv_stable(stable_var, cv_epsilon), get_count(stable_var)
    else
        ratio_stat = Variance()
        compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, start_idx, end_idx, cv_epsilon)
        return compute_cv(ratio_stat, cv_epsilon), nobs(ratio_stat)
    end
end


# Optimized function to compute ratios and fit to variance stat
@inline function compute_ratios_batch!(
    variance_stat::Variance,
    c1_samples::Vector{Float64},
    c2_samples::Vector{Float64},
    start_idx::Int,
    end_idx::Int,
    epsilon::Float64
)
    # Stream ratios directly - OnlineStats processes "one observation at a time" for O(1) memory
    @inbounds for k in start_idx:end_idx
        ratio = (c1_samples[k] + epsilon) / (c2_samples[k] + epsilon)
        isfinite(ratio) && OnlineStats.fit!(variance_stat, ratio)
    end
    return variance_stat
end

# Stable version using Welford's algorithm
@inline function compute_ratios_batch_stable!(
    stable_var::StableVariance,
    c1_samples::Vector{Float64},
    c2_samples::Vector{Float64},
    start_idx::Int,
    end_idx::Int,
    epsilon::Float64
)
    @inbounds for k in start_idx:end_idx
        ratio = (c1_samples[k] + epsilon) / (c2_samples[k] + epsilon)
        isfinite(ratio) && add_sample!(stable_var, ratio)
    end
    return stable_var
end

# --- Priority structure with bit flags for directions ---
struct PairPriority
    c1_idx::Int
    c2_idx::Int
    directions_bits::UInt8  # Bit flags: bit 0 = positive, bit 1 = negative
    cv::Float64
    n_samples::Int
end

# Constants for direction bit flags
const DIRECTION_POSITIVE = 0x01
const DIRECTION_NEGATIVE = 0x02

# Helper functions for direction bit manipulation
@inline function directions_to_bits(directions::Set{Symbol})::UInt8
    bits = 0x00
    :positive in directions && (bits |= DIRECTION_POSITIVE)
    :negative in directions && (bits |= DIRECTION_NEGATIVE)
    return bits
end

@inline function bits_to_directions(bits::UInt8)::Set{Symbol}
    directions = Set{Symbol}()
    (bits & DIRECTION_POSITIVE) != 0 && push!(directions, :positive)
    (bits & DIRECTION_NEGATIVE) != 0 && push!(directions, :negative)
    return directions
end

@inline function has_positive_direction(bits::UInt8)::Bool
    return (bits & DIRECTION_POSITIVE) != 0
end

@inline function has_negative_direction(bits::UInt8)::Bool
    return (bits & DIRECTION_NEGATIVE) != 0
end

# Constructor that takes Set{Symbol} and converts to bits
function PairPriority(c1_idx::Int, c2_idx::Int, directions::Set{Symbol}, cv::Float64, n_samples::Int)
    return PairPriority(c1_idx, c2_idx, directions_to_bits(directions), cv, n_samples)
end

# --- Heap-based top-k maintenance ---
"""
Heap-based priority queue for maintaining top-k candidates by CV value.
Uses a max-heap so we can efficiently remove the worst (highest CV) candidates.
"""
mutable struct TopKHeap
    heap::BinaryHeap{PairPriority}
    max_size::Int

    function TopKHeap(max_size::Int)
        # Custom ordering: higher CV = higher priority (worse candidates)
        heap = BinaryHeap(Base.By(p -> p.cv), PairPriority[])
        new(heap, max_size)
    end
end

@inline function heap_size(h::TopKHeap)::Int
    return length(h.heap)
end

@inline function is_heap_full(h::TopKHeap)::Bool
    return length(h.heap) >= h.max_size
end

function add_to_heap!(h::TopKHeap, priority::PairPriority)
    if !is_heap_full(h)
        # Heap not full, just add
        push!(h.heap, priority)
    elseif priority.cv < top(h.heap).cv
        # New candidate is better than worst in heap
        pop!(h.heap)  # Remove worst
        push!(h.heap, priority)  # Add new candidate
    end
    # If heap is full and new candidate is worse, do nothing
end

function extract_sorted_results(h::TopKHeap)::Vector{PairPriority}
    results = Vector{PairPriority}()
    sizehint!(results, length(h.heap))

    # Extract all elements (this empties the heap)
    while !isempty(h.heap)
        push!(results, pop!(h.heap))
    end

    # Reverse since we extracted from max-heap (worst first)
    reverse!(results)
    return results
end

# --- Main filtering function ---
"""
$(TYPEDSIGNATURES)

Ultra-efficient three-stage streaming pipeline.
"""
function streaming_filter(
    complexes::Vector,
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    coarse_sample_size::Int=20,
    coarse_cv_threshold::Float64=0.1,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10,
    use_threads::Bool=false,
    chunk_size::Int=10_000,
    use_stable_variance::Bool=false,
    use_heap_pruning::Bool=true,
    max_pairs_in_memory::Int=100_000
)
    @info "Starting memory-efficient CV filtering pipeline"

    # Access BitVectors directly from ConcordanceTracker for maximum efficiency
    # Ensure they are allocated first
    ensure_mask_allocated!(concordance_tracker, :balanced)
    ensure_mask_allocated!(concordance_tracker, :positive)
    ensure_mask_allocated!(concordance_tracker, :negative)

    balanced_complexes = concordance_tracker.balanced_mask
    positive_complexes = concordance_tracker.positive_mask
    negative_complexes = concordance_tracker.negative_mask

    # Stage 1: Count pairs without materializing
    stage1_start = time()
    n_pairs = count_valid_pairs(complexes, balanced_complexes, trivial_pairs, concordance_tracker)
    @info "Stage 1: Counted pairs" n_pairs time_sec = round(time() - stage1_start, digits=2)

    # Stage 2 & 3: Stream processing
    stage23_start = time()
    priorities = if n_pairs <= max_pairs_in_memory
        # Small enough to process directly
        @info "Processing all pairs directly (small dataset)"
        process_all_pairs(
            complexes, trivial_pairs, samples_tree,
            concordance_tracker, coarse_sample_size, coarse_cv_threshold,
            cv_threshold, cv_epsilon, min_valid_samples, use_threads,
            use_stable_variance, use_heap_pruning
        )
    else
        # Stream in chunks
        @info "Processing in streaming chunks (large dataset)" chunk_size = chunk_size
        process_streaming_chunks(
            complexes, trivial_pairs, samples_tree,
            concordance_tracker, coarse_sample_size, coarse_cv_threshold,
            cv_threshold, cv_epsilon, min_valid_samples, use_threads,
            chunk_size, use_stable_variance, use_heap_pruning
        )
    end
    stage23_time = time() - stage23_start

    @info "Filtering complete" total_candidates = length(priorities) time_sec = round(stage23_time, digits=2)

    return priorities
end

# --- Stage 1: Counting without allocation ---
# Optimized BitVector implementation for large models (50K+ reactions)
function count_valid_pairs(complexes, balanced::BitVector, trivial, concordance_tracker)
    count = 0
    n = length(complexes)
    @inbounds for i in 1:n
        # Fast BitVector lookup - no hash computation needed
        i_balanced = balanced[i]
        if i_balanced
            continue  # Skip entire inner loop if i is balanced
        end

        for j in (i+1):n
            if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker, i_balanced)
                count += 1
            end
        end
    end
    return count
end



# BitVector-optimized version with O(1) lookups
@inline function should_test_pair_indices(i::Int, j::Int, balanced::BitVector, trivial, concordance_tracker, i_balanced::Bool)
    # i_balanced already checked by caller - skip redundant lookup
    i_balanced && return false

    # Fast BitVector lookup for j - O(1) with better cache behavior
    balanced[j] && return false

    # Trivial pairs check - relatively fast Set lookup
    (i, j) in trivial && return false

    # Concordance checks - more expensive due to Union-Find operations
    are_concordant(concordance_tracker, i, j) && return false
    is_non_concordant(concordance_tracker, i, j) && return false

    return true
end


# --- Direct processing for smaller datasets ---
# Optimized BitVector implementation for large models
function process_all_pairs(
    complexes, trivial, samples_tree,
    concordance_tracker, coarse_sample_size, coarse_cv_threshold,
    cv_threshold, cv_epsilon, min_valid_samples, use_threads,
    use_stable_variance, use_heap_pruning
)
    # Access BitVectors directly from ConcordanceTracker for maximum efficiency
    balanced = concordance_tracker.balanced_mask
    positive = concordance_tracker.positive_mask
    negative = concordance_tracker.negative_mask

    # Generator for valid pairs with BitVector optimization
    valid_pairs = (
        (i, j) for i in 1:length(complexes)
        for j in (i+1):length(complexes)
        if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker, balanced[i])
    )

    # Process with streaming statistics using BitVectors
    priorities = process_pair_stream(
        valid_pairs, samples_tree, concordance_tracker,
        positive, negative, coarse_sample_size, coarse_cv_threshold,
        cv_threshold, cv_epsilon, min_valid_samples, use_threads,
        use_stable_variance, use_heap_pruning
    )

    return priorities
end

# --- Chunked streaming for huge datasets ---
# Optimized BitVector implementation for large models
function process_streaming_chunks(
    complexes, trivial, samples_tree,
    concordance_tracker, coarse_sample_size, coarse_cv_threshold,
    cv_threshold, cv_epsilon, min_valid_samples, use_threads,
    chunk_size, use_stable_variance, use_heap_pruning
)
    # Access BitVectors directly from ConcordanceTracker for maximum efficiency
    balanced = concordance_tracker.balanced_mask
    positive = concordance_tracker.positive_mask
    negative = concordance_tracker.negative_mask

    # Pre-allocate with type annotation for better performance
    all_priorities = Vector{PairPriority}()
    sizehint!(all_priorities, 100_000)  # Use reasonable default
    chunk_pairs = Vector{Tuple{Int,Int}}()
    sizehint!(chunk_pairs, chunk_size)

    n = length(complexes)
    @inbounds for i in 1:n
        # Fast BitVector lookup for i
        i_balanced = balanced[i]
        if i_balanced
            continue  # Skip entire inner loop if i is balanced
        end

        for j in (i+1):n
            if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker, i_balanced)
                push!(chunk_pairs, (i, j))

                if length(chunk_pairs) >= chunk_size
                    # Process chunk
                    chunk_priorities = process_pair_stream(
                        chunk_pairs, samples_tree, concordance_tracker,
                        positive, negative, coarse_sample_size, coarse_cv_threshold,
                        cv_threshold, cv_epsilon, min_valid_samples, use_threads,
                        use_stable_variance, use_heap_pruning
                    )
                    append!(all_priorities, chunk_priorities)

                    # Clear chunk
                    empty!(chunk_pairs)

                    # Keep only top candidates to save memory
                    max_pairs_limit = 100_000  # Use reasonable default
                    if length(all_priorities) > max_pairs_limit
                        if use_heap_pruning
                            # Use heap for O(n log k) pruning instead of O(n log n) sorting
                            heap = TopKHeap(max_pairs_limit ÷ 2)
                            for priority in all_priorities
                                add_to_heap!(heap, priority)
                            end
                            all_priorities = extract_sorted_results(heap)
                            @info "Memory pruning (heap)" kept_best = length(all_priorities)
                        else
                            # Fallback to sorting
                            sort!(all_priorities, by=p -> p.cv)
                            resize!(all_priorities, max_pairs_limit ÷ 2)
                            @info "Memory pruning (sort)" kept_best = length(all_priorities)
                        end
                    end
                end
            end
        end
    end

    # Process final chunk
    if !isempty(chunk_pairs)
        chunk_priorities = process_pair_stream(
            chunk_pairs, samples_tree, concordance_tracker,
            positive, negative, coarse_sample_size, coarse_cv_threshold,
            cv_threshold, cv_epsilon, min_valid_samples, use_threads,
            use_stable_variance, use_heap_pruning
        )
        append!(all_priorities, chunk_priorities)
    end

    # Final sort - only needed if not using heap (heap already maintains order)
    if !use_heap_pruning
        sort!(all_priorities, by=p -> p.cv)
    end
    return all_priorities
end

# BitVector version of core streaming processor
function process_pair_stream(
    pairs, samples_tree, concordance_tracker,
    positive::BitVector, negative::BitVector, coarse_sample_size, coarse_cv_threshold,
    cv_threshold, cv_epsilon, min_valid_samples, use_threads,
    use_stable_variance, use_heap_pruning
)
    idx_to_id = concordance_tracker.idx_to_id

    if use_threads
        @info "Using parallel processing with threads"
        process_pairs_parallel(pairs, samples_tree, idx_to_id, positive, negative,
            coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
            min_valid_samples, use_stable_variance, use_heap_pruning)
    else
        @info "Using serial processing"
        process_pairs_serial(pairs, samples_tree, idx_to_id, positive, negative,
            coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
            min_valid_samples, use_stable_variance, use_heap_pruning)
    end
end

# --- BitVector Serial processing ---
function process_pairs_serial(
    pairs, samples_tree, idx_to_id,
    positive::BitVector, negative::BitVector, coarse_sample_size,
    coarse_cv_threshold, cv_threshold, cv_epsilon, min_valid_samples,
    use_stable_variance, use_heap_pruning
)
    # Pre-allocate with type annotation
    priorities = Vector{PairPriority}()
    pair_count = 0
    stage2_passed = 0
    stage3_passed = 0


    for (i, j) in pairs
        pair_count += 1

        # Cache lookups to avoid repeated dictionary access
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        # Early exit if samples are missing
        if isnothing(c1_samples) || isnothing(c2_samples)
            continue
        end

        # Cache sample lengths to avoid repeated calls
        c1_len, c2_len = length(c1_samples), length(c2_samples)

        # Stage 2: Coarse filter - use cached lengths
        n_coarse = min(coarse_sample_size, c1_len, c2_len)
        n_coarse < 2 && continue

        # Create new variance statistic (OnlineStats doesn't support reset)
        ratio_stat = Variance()
        compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, 1, n_coarse, cv_epsilon)

        cv_coarse = compute_cv(ratio_stat, cv_epsilon)
        cv_coarse > coarse_cv_threshold && continue
        stage2_passed += 1

        # Stage 3: Full analysis - reuse and continue with same stat, use cached lengths
        max_samples = min(c1_len, c2_len)
        if max_samples > n_coarse
            compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, n_coarse + 1, max_samples, cv_epsilon)
        end

        n_samples = nobs(ratio_stat)
        n_samples < min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, cv_epsilon)
        cv_full > cv_threshold && continue
        stage3_passed += 1

        # Passed all filters
        directions_bits = determine_directions(j, positive, negative)
        push!(priorities, PairPriority(i, j, directions_bits, cv_full, n_samples))

        if pair_count % 50_000 == 0
            @info "Processing progress" pairs_tested = pair_count coarse_passed = stage2_passed cv_passed = stage3_passed candidates = length(priorities)
        end
    end

    @info "Serial processing complete" pairs_tested = pair_count coarse_passed = stage2_passed cv_passed = stage3_passed final_candidates = length(priorities)
    return priorities
end

# --- BitVector Parallel processing ---
function process_pairs_parallel(
    pairs, samples_tree, idx_to_id,
    positive::BitVector, negative::BitVector, coarse_sample_size,
    coarse_cv_threshold, cv_threshold, cv_epsilon, min_valid_samples,
    use_stable_variance, use_heap_pruning
)
    # Convert to vector only if not already a vector, avoiding copy when possible
    pairs_vec = if pairs isa AbstractVector
        pairs
    else
        collect(pairs)
    end
    n_threads = Threads.nthreads()

    # Pre-allocate thread-local results
    thread_results = Vector{Vector{PairPriority}}(undef, n_threads)
    for i in 1:n_threads
        thread_results[i] = Vector{PairPriority}()
        sizehint!(thread_results[i], length(pairs_vec) ÷ n_threads + 100)
    end

    # Process in parallel
    @threads for idx in eachindex(pairs_vec)
        tid = Threads.threadid()
        i, j = pairs_vec[idx]

        # Cache lookups and get samples
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        # Early exit if samples are missing
        if isnothing(c1_samples) || isnothing(c2_samples)
            continue
        end

        # Cache sample lengths to avoid repeated calls
        c1_len, c2_len = length(c1_samples), length(c2_samples)

        # Coarse filter - use cached lengths
        n_coarse = min(coarse_sample_size, c1_len, c2_len)
        n_coarse < 2 && continue

        # Create new variance statistic (OnlineStats doesn't support reset)
        ratio_stat = Variance()
        compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, 1, n_coarse, cv_epsilon)

        cv_coarse = compute_cv(ratio_stat, cv_epsilon)
        cv_coarse > coarse_cv_threshold && continue

        # Full analysis - use cached lengths
        max_samples = min(c1_len, c2_len)
        if max_samples > n_coarse
            compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, n_coarse + 1, max_samples, cv_epsilon)
        end

        n_samples = nobs(ratio_stat)
        n_samples < min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, cv_epsilon)
        cv_full > cv_threshold && continue

        directions_bits = determine_directions(j, positive, negative)
        push!(thread_results[tid], PairPriority(i, j, directions_bits, cv_full, n_samples))
    end

    # Merge thread results
    return vcat(thread_results...)
end

@inline function determine_directions(c2_idx::Int,
    positive::Set{Int}, negative::Set{Int}
)::Set{Symbol}
    # Pre-allocate direction set with capacity hint
    directions = Set{Symbol}()
    sizehint!(directions, 2)

    # Check constraints on both complexes to determine valid directions
    c2_positive = c2_idx in positive
    c2_negative = c2_idx in negative

    # Positive direction test: set c2 = +1
    # Only feasible if c2 can achieve positive values (not constrained to negative only)
    if !c2_negative
        push!(directions, :positive)
    end

    # Negative direction test: set c2 = -1  
    # Only feasible if c2 can achieve negative values (not constrained to positive only)
    if !c2_positive
        push!(directions, :negative)
    end

    return directions
end

# BitVector version of determine_directions with O(1) lookups, returns bit flags
@inline function determine_directions(c2_idx::Int,
    positive::BitVector, negative::BitVector
)::UInt8
    # Fast BitVector lookups - O(1) with better cache behavior
    c2_positive = positive[c2_idx]
    c2_negative = negative[c2_idx]

    bits = 0x00
    # Positive direction test: set c2 = +1
    # Only feasible if c2 can achieve positive values (not constrained to negative only)
    if !c2_negative
        bits |= DIRECTION_POSITIVE
    end

    # Negative direction test: set c2 = -1  
    # Only feasible if c2 can achieve negative values (not constrained to positive only)
    if !c2_positive
        bits |= DIRECTION_NEGATIVE
    end

    return bits
end