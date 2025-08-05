"""
filter.jl - Ultra memory-efficient CV filtering using OnlineStats.jl and BitVectors

Streaming three-stage pipeline focused purely on CV computation using OnlineStats and BitVectors.
"""

using OnlineStats
using DataStructures
using DocStringExtensions
import ConstraintTrees as C


# --- Helper functions for CV ---
@inline function compute_cv(variance_stat::Variance, epsilon::Float64=1e-8)
    n = nobs(variance_stat)
    m = mean(variance_stat)
    s = std(variance_stat)

    n < 2 && return Inf
    abs(m) < epsilon && return Inf

    return s / abs(m)
end



# Optimized function to compute ratios and fit to variance stat using views
@inline function compute_ratios_batch!(
    variance_stat::Variance,
    c1_view::AbstractVector{Float64},
    c2_view::AbstractVector{Float64},
    epsilon::Float64
)
    # Stream ratios directly - OnlineStats processes "one observation at a time" for O(1) memory
    # Views handle the range slicing efficiently without copying data
    @simd for k in eachindex(c1_view)
        @inbounds begin
            ratio = (c1_view[k] + epsilon) / (c2_view[k] + epsilon)
            isfinite(ratio) && OnlineStats.fit!(variance_stat, ratio)
        end
    end
    return variance_stat
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
    # Prevent runaway heap growth that could cause memory issues
    if length(h.heap) >= h.max_size * 2
        @warn "Heap size exceeded bounds, potential memory issue" current_size = length(h.heap) max_allowed = h.max_size
        return  # Skip this addition to prevent memory explosion
    end
    
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
    # Pre-allocate with exact size for optimal performance
    heap_size = length(h.heap)
    results = Vector{PairPriority}(undef, heap_size)

    # Extract all elements directly into pre-allocated array
    idx = heap_size
    while !isempty(h.heap)
        results[idx] = pop!(h.heap)
        idx -= 1
    end
    
    # No need to reverse - we filled backwards from max-heap (best first)
    return results
end

# --- Main filtering function ---
"""
$(TYPEDSIGNATURES)

Ultra-efficient three-stage streaming pipeline.
"""
function streaming_filter(
    complexes::Vector{Symbol},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    coarse_sample_size::Int=20,
    coarse_cv_threshold::Float64=0.1,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10,
    chunk_size::Int=10_000,
    max_pairs_in_memory::Int=1_000_000
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
            complexes, balanced_complexes, trivial_pairs, samples_tree,
            concordance_tracker, positive_complexes, negative_complexes,
            coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
            min_valid_samples, chunk_size, max_pairs_in_memory
        )
    else
        # Stream in chunks
        @info "Processing in streaming chunks (large dataset)" chunk_size = chunk_size
        process_streaming_chunks(
            complexes, balanced_complexes, trivial_pairs, samples_tree,
            concordance_tracker, positive_complexes, negative_complexes,
            coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
            min_valid_samples, chunk_size, max_pairs_in_memory
        )
    end
    stage23_time = time() - stage23_start

    @info "Filtering complete" total_candidates = length(priorities) time_sec = round(stage23_time, digits=2)

    return priorities
end

# --- Stage 1: Counting without allocation ---
# Optimized BitVector implementation for large models (50K+ reactions)
function count_valid_pairs(complexes::Vector{Symbol}, balanced::BitVector, trivial::Set{Tuple{Int,Int}}, concordance_tracker::ConcordanceTracker)
    count = 0
    n = length(complexes)
    @inbounds for i in 1:n
        # Fast BitVector lookup - no hash computation needed
        i_balanced = balanced[i]
        if i_balanced
            continue  # Skip entire inner loop if i is balanced
        end

        @simd for j in (i+1):n
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
    complexes::Vector{Symbol}, balanced::BitVector, trivial::Set{Tuple{Int,Int}}, samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker, positive::BitVector, negative::BitVector,
    coarse_sample_size::Int, coarse_cv_threshold::Float64, cv_threshold::Float64, cv_epsilon::Float64,
    min_valid_samples::Int, chunk_size::Int, max_pairs_in_memory::Int
)
    # Generator for valid pairs with BitVector optimization
    valid_pairs = (
        (i, j) for i in 1:length(complexes)
        for j in (i+1):length(complexes)
        if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker, balanced[i])
    )

    # Process with streaming statistics using BitVectors
    priorities = process_pair_stream(
        valid_pairs, samples_tree, concordance_tracker,
        positive, negative, coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
        min_valid_samples
    )

    return priorities
end

# --- Chunked streaming for huge datasets ---
# Optimized BitVector implementation for large models
function process_streaming_chunks(
    complexes::Vector{Symbol}, balanced::BitVector, trivial::Set{Tuple{Int,Int}}, samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker, positive::BitVector, negative::BitVector,
    coarse_sample_size::Int, coarse_cv_threshold::Float64, cv_threshold::Float64, cv_epsilon::Float64,
    min_valid_samples::Int, chunk_size::Int, max_pairs_in_memory::Int
)
    # Pre-allocate with type annotation for better performance
    all_priorities = Vector{PairPriority}()
    sizehint!(all_priorities, max_pairs_in_memory)
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
                        positive, negative, coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
                        min_valid_samples
                    )
                    append!(all_priorities, chunk_priorities)

                    # Clear chunk
                    empty!(chunk_pairs)

                    # Keep only top candidates to save memory
                    if length(all_priorities) > max_pairs_in_memory
                        # Use heap for O(n log k) pruning instead of O(n log n) sorting
                        heap = TopKHeap(max_pairs_in_memory ÷ 2)
                        for priority in all_priorities
                            add_to_heap!(heap, priority)
                        end
                        all_priorities = extract_sorted_results(heap)
                        @info "Memory pruning (heap)" kept_best = length(all_priorities)
                    end
                end
            end
        end
    end

    # Process final chunk
    if !isempty(chunk_pairs)
        chunk_priorities = process_pair_stream(
            chunk_pairs, samples_tree, concordance_tracker,
            positive, negative, coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon,
            min_valid_samples
        )
        append!(all_priorities, chunk_priorities)
    end

    # Final sort using direct field access for better performance
    sort!(all_priorities, by=p -> p.cv)
    return all_priorities
end

# BitVector version of core streaming processor
function process_pair_stream(
    pairs, samples_tree, concordance_tracker,
    positive::BitVector, negative::BitVector,
    coarse_sample_size::Int, coarse_cv_threshold::Float64, cv_threshold::Float64, cv_epsilon::Float64,
    min_valid_samples::Int
)
    idx_to_id = concordance_tracker.idx_to_id

    @info "Using serial processing"
    process_pairs_serial(pairs, samples_tree, idx_to_id, positive, negative, coarse_sample_size, coarse_cv_threshold, cv_threshold, cv_epsilon, min_valid_samples)
end

# --- BitVector Serial processing ---
function process_pairs_serial(
    pairs, samples_tree, idx_to_id,
    positive::BitVector, negative::BitVector,
    coarse_sample_size::Int, coarse_cv_threshold::Float64, cv_threshold::Float64, cv_epsilon::Float64,
    min_valid_samples::Int
)
    # Pre-allocate with size estimation for better performance
    # Estimate: ~1-5% of pairs typically pass all filters
    estimated_candidates = if pairs isa AbstractVector
        max(100, length(pairs) ÷ 20)  # Conservative 5% estimate
    else
        1000  # Default for generators
    end
    priorities = Vector{PairPriority}()
    sizehint!(priorities, estimated_candidates)
    
    pair_count = 0
    stage2_passed = 0
    stage3_passed = 0


    # Note: OnlineStats.Variance() doesn't support reset, so we create fresh ones per pair
    
    # Direct array access - much faster than dictionary lookups for consecutive indices
    # idx_to_id is already a Vector{Symbol}, so we can access it directly
    
    for (i, j) in pairs
        pair_count += 1

        # Direct vector access - O(1) with no hash computation
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        
        # Direct tree access - no copying, more memory efficient
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

        # Reuse variance statistic by creating fresh one (more efficient than clearing)
        ratio_stat = Variance()
        @views compute_ratios_batch!(ratio_stat, c1_samples[1:n_coarse], c2_samples[1:n_coarse], cv_epsilon)

        cv_coarse = compute_cv(ratio_stat, cv_epsilon)
        cv_coarse > coarse_cv_threshold && continue
        stage2_passed += 1

        # Stage 3: Full analysis - reuse and continue with same stat, use cached lengths
        max_samples = min(c1_len, c2_len)
        if max_samples > n_coarse
            @views compute_ratios_batch!(ratio_stat, c1_samples[n_coarse+1:max_samples], c2_samples[n_coarse+1:max_samples], cv_epsilon)
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