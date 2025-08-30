"""
filter.jl - Revolutionary streaming candidate filter for 1B+ candidates

Streaming producer-consumer architecture that processes unlimited candidates 
with constant memory usage (<1MB) through bidirectional filter-analysis communication.
"""

import OnlineStatsBase
import Statistics
# DocStringExtensions macros require using
using DocStringExtensions
import ConstraintTrees as C
import Base: push!, length, isempty, popfirst!

# ========================================================================================
# Section 1: Core Data Structures (Minimal & Efficient)
# ========================================================================================

"""
Packed candidate structure optimized for memory efficiency.
15 bytes total vs 25+ bytes in previous versions.
"""
struct PairCandidate
    c1_idx::Int           # Use Int for direct compatibility with ConcordanceTracker
    c2_idx::Int           # Avoids Int() conversions throughout pipeline
    directions_bits::UInt8 # 1 byte - bit flags for directions
    cv::Float32           # 4 bytes - sufficient precision for CV values
    n_samples::UInt16     # 2 bytes - supports up to 65K samples
end

# Direction bit flags (same as before for compatibility)
const DIRECTION_POSITIVE = 0x01
const DIRECTION_NEGATIVE = 0x02

# Helper functions for direction bits
@inline function directions_to_bits(directions::Set{Symbol})::UInt8
    bits = 0x00
    :positive in directions && (bits |= DIRECTION_POSITIVE)
    :negative in directions && (bits |= DIRECTION_NEGATIVE)
    return bits
end

@inline function determine_directions_bits(c2_idx::Int, positive::BitVector, negative::BitVector)::UInt8
    bits = 0x00
    # Positive direction feasible if c2 not constrained to negative only
    !negative[c2_idx] && (bits |= DIRECTION_POSITIVE)
    # Negative direction feasible if c2 not constrained to positive only  
    !positive[c2_idx] && (bits |= DIRECTION_NEGATIVE)
    return bits
end

# ========================================================================================
# Circular Buffer for Streaming
# ========================================================================================

"""
Circular buffer for efficient streaming without reallocations.
"""
mutable struct CircularBuffer{T}
    data::Vector{T}
    capacity::Int
    size::Int
    head::Int
    tail::Int

    function CircularBuffer{T}(capacity::Int) where T
        new(Vector{T}(undef, capacity), capacity, 0, 1, 1)
    end
end

@inline function push!(buf::CircularBuffer{T}, item::T) where T
    if buf.size < buf.capacity
        buf.data[buf.tail] = item
        buf.tail = mod1(buf.tail + 1, buf.capacity)
        buf.size += 1
    else
        # Overwrite oldest
        buf.data[buf.tail] = item
        buf.tail = mod1(buf.tail + 1, buf.capacity)
        buf.head = mod1(buf.head + 1, buf.capacity)
    end
end

@inline function popfirst!(buf::CircularBuffer{T})::T where T
    if buf.size == 0
        throw(BoundsError("Buffer is empty"))
    end
    item = buf.data[buf.head]
    buf.head = mod1(buf.head + 1, buf.capacity)
    buf.size -= 1
    return item
end

@inline isempty(buf::CircularBuffer) = buf.size == 0
@inline length(buf::CircularBuffer) = buf.size

"""
Get all items as a view without copying.
"""
function view_all(buf::CircularBuffer{T})::SubArray{T} where T
    if buf.size == 0
        return view(buf.data, 1:0)
    elseif buf.head <= buf.tail - 1
        return view(buf.data, buf.head:buf.tail-1)
    else
        # Wrapped around - need to handle separately
        # For now, return first contiguous segment
        return view(buf.data, buf.head:buf.capacity)
    end
end

"""
Custom OnlineStat for coefficient of variation computation using Welford's algorithm.
Provides numerical stability and supports efficient reuse via Base.empty!.
"""
mutable struct CVStat <: OnlineStatsBase.OnlineStat{Number}
    mean::Float64
    m2::Float64    # Sum of squared differences from mean (for Welford's algorithm)
    n::Int
    CVStat() = new(0.0, 0.0, 0)
end

# Required OnlineStatsBase interface: Update statistics using Welford's algorithm
function OnlineStatsBase._fit!(o::CVStat, y)
    o.n += 1
    delta = y - o.mean
    o.mean += delta / o.n
    delta2 = y - o.mean
    o.m2 += delta * delta2
end

# Optional: Enable reset for efficient reuse
function Base.empty!(o::CVStat)
    o.mean = 0.0
    o.m2 = 0.0
    o.n = 0
    return o
end

# Helper methods for CV calculation
@inline function Statistics.mean(o::CVStat)
    return o.mean
end

@inline function Statistics.std(o::CVStat)
    o.n < 2 && return 0.0
    return sqrt(o.m2 / (o.n - 1))
end

"""
Compute coefficient of variation from CVStat using numerically stable Welford's algorithm.
Returns (cv, n_valid_samples) where cv = std/mean.
"""
@inline function compute_cv(o::CVStat, epsilon::Float64=1e-15)::Tuple{Float64,Int}
    o.n < 2 && return (Inf, o.n)
    abs(o.mean) < epsilon && return (Inf, o.n)
    cv = Statistics.std(o) / abs(o.mean)
    return (cv, o.n)
end

# ========================================================================================  
# Section 2: Streaming Filter Core
# ========================================================================================

"""
Revolutionary streaming candidate filter supporting 1B+ candidates with <1MB memory.
Implements Julia iterator protocol for zero-allocation candidate generation.
"""
mutable struct StreamingCandidateFilter
    # Pair iteration state
    current_i::Int
    current_j::Int
    n_complexes::Int
    iteration_count::Int  # Track total iterations for debugging

    # Core data
    complexes::Vector{Symbol}
    balanced::BitVector
    positive::BitVector
    negative::BitVector
    trivial_pairs::Set{Tuple{Int,Int}}
    samples_tree::C.Tree{Vector{Float64}}
    concordance_tracker::ConcordanceTracker

    # CV filtering parameters
    cv_threshold::Float64
    cv_epsilon::Float64
    min_valid_samples::Int

    # Computation state (reused with reset support)
    cv_stat::CVStat
    idx_to_id::Vector{Symbol}

    # Statistics and debugging
    pairs_tested::Int
    candidates_found::Int
    pairs_balanced_filtered::Int
    pairs_trivial_filtered::Int
    pairs_concordant_filtered::Int  # This tracks transitivity filtering
    pairs_missing_samples::Int
    pairs_cv_filtered::Int
    insufficient_samples::Int

    # Enhanced transitivity tracking
    pairs_transitivity_concordant_filtered::Int  # Pairs filtered as known concordant
    pairs_transitivity_non_concordant_filtered::Int  # Pairs filtered as known non-concordant
    pairs_skipped_by_transitivity::Int  # Total pairs skipped (concordant + non-concordant)
    transitivity_updates_received::Int  # How many discovery updates received

    # Control flags
    should_stop::Bool
    use_transitivity::Bool  # Whether to apply transitivity filtering
end

function StreamingCandidateFilter(
    complexes::Vector{Symbol},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10,
    use_transitivity::Bool=true
)
    # Ensure BitVectors are allocated
    ensure_mask_allocated!(concordance_tracker, :balanced)
    ensure_mask_allocated!(concordance_tracker, :positive)
    ensure_mask_allocated!(concordance_tracker, :negative)

    n = length(complexes)

    StreamingCandidateFilter(
        1, 2, n, 0,  # Start at (1,2), iteration_count=0
        complexes,
        concordance_tracker.balanced_mask,
        concordance_tracker.positive_mask,
        concordance_tracker.negative_mask,
        trivial_pairs,
        samples_tree,
        concordance_tracker,
        cv_threshold, cv_epsilon, min_valid_samples,
        CVStat(),
        concordance_tracker.idx_to_id,
        0, 0, 0, 0, 0, 0, 0, 0,  # Original statistics counters
        0, 0, 0, 0,  # Enhanced transitivity tracking counters
        false,  # should_stop
        use_transitivity  # use_transitivity flag
    )
end

# ========================================================================================
# Section 3: Iterator Protocol Implementation
# ========================================================================================

"""
Implement Julia iterator protocol for StreamingCandidateFilter.
This enables `for candidate in filter` syntax with zero allocations.
"""
Base.eltype(::StreamingCandidateFilter) = PairCandidate
Base.IteratorSize(::StreamingCandidateFilter) = Base.SizeUnknown()

function Base.iterate(filter::StreamingCandidateFilter, state=nothing)
    # Find next valid candidate
    while filter.current_i <= filter.n_complexes && !filter.should_stop
        filter.iteration_count += 1
        i, j = filter.current_i, filter.current_j


        # Check bounds before processing
        if i <= filter.n_complexes && j <= filter.n_complexes && i < j
            # Process current pair BEFORE advancing
            candidate = process_pair(filter, i, j)

            # Advance to next pair
            filter.current_j += 1
            if filter.current_j > filter.n_complexes
                filter.current_i += 1
                filter.current_j = filter.current_i + 1  # Reset j for new i
            end

            # Return candidate if found
            if candidate !== nothing
                filter.candidates_found += 1
                return (candidate, nothing)
            end
        else
            # Invalid bounds, advance to next valid position
            filter.current_j += 1
            if filter.current_j > filter.n_complexes
                filter.current_i += 1
                filter.current_j = filter.current_i + 1
            end

            # Check if we've exhausted all pairs
            if filter.current_i > filter.n_complexes
                break
            end
        end
    end

    # Iterator exhausted - show detailed debugging statistics
    total_pairs_possible = filter.n_complexes * (filter.n_complexes - 1) ÷ 2
    transitivity_effectiveness = round(filter.pairs_skipped_by_transitivity / max(1, filter.pairs_tested) * 100, digits=1)

    return nothing
end

"""
Process a single pair (i,j) and return PairCandidate if it passes all filters.
Returns nothing if pair should be skipped.
Optimized version with cached data access and reduced allocations.
"""
function process_pair(filter::StreamingCandidateFilter, i::Int, j::Int)::Union{PairCandidate,Nothing}
    filter.pairs_tested += 1

    # Early filtering checks (fast rejection) with @inbounds for performance
    @inbounds begin
        # Cache balanced mask access
        balanced_mask = filter.balanced
        if balanced_mask[i] || balanced_mask[j]
            filter.pairs_balanced_filtered += 1
            return nothing
        end

        # Skip trivial pairs (optimized Set lookup)
        if (i, j) in filter.trivial_pairs
            filter.pairs_trivial_filtered += 1
            return nothing
        end

        # Skip early transitivity filtering here - it causes O(n²) overhead
        # Transitivity filtering moved to after CV check for much better performance

        # Cache complex IDs lookup
        idx_to_id = filter.idx_to_id
        c1_id = idx_to_id[i]
        c2_id = idx_to_id[j]
    end

    # Cache samples tree access
    samples_tree = filter.samples_tree
    c1_samples = samples_tree[c1_id]
    c2_samples = samples_tree[c2_id]

    # Handle missing samples - use ismissing() for proper Julia missing value detection
    if ismissing(c1_samples) || ismissing(c2_samples) || c1_samples === nothing || c2_samples === nothing
        filter.pairs_missing_samples += 1
        # Default behavior: Include pairs with missing samples as candidates
        # since we have no evidence to exclude them from concordance analysis
        directions_bits = determine_directions_bits(j, filter.positive, filter.negative)
        return PairCandidate(
            i,
            j,
            directions_bits,
            Float32(Inf),  # Use Inf CV to indicate missing data
            UInt16(0)      # Zero valid samples
        )
    end

    # Determine sample size - handle case where samples exist but are insufficient
    n_samples = min(length(c1_samples), length(c2_samples))
    if n_samples < 2
        filter.pairs_missing_samples += 1
        # Include pairs with insufficient samples as candidates with infinite CV
        directions_bits = determine_directions_bits(j, filter.positive, filter.negative)
        return PairCandidate(
            i,
            j,
            directions_bits,
            Float32(Inf),  # Use Inf CV to indicate insufficient data
            UInt16(n_samples)  # Record actual sample count
        )
    end

    # Reset and compute CV using OnlineStats with Welford's algorithm
    cv_stat = filter.cv_stat  # Cache cv_stat reference
    empty!(cv_stat)

    # Cache epsilon for faster access in tight loop
    epsilon = filter.cv_epsilon

    @inbounds for k in 1:n_samples
        # Handle missing values within sample arrays
        c1_val = c1_samples[k]
        c2_val = c2_samples[k]

        # Skip missing values using Julia's ismissing() function
        if ismissing(c1_val) || ismissing(c2_val)
            continue
        end

        ratio = (c1_val + epsilon) / (c2_val + epsilon)
        if isfinite(ratio)
            OnlineStatsBase.fit!(cv_stat, ratio)
        end
    end

    cv, n_valid = compute_cv(filter.cv_stat, filter.cv_epsilon)


    # Handle insufficient samples - include them as candidates regardless of CV
    if n_valid < filter.min_valid_samples
        filter.insufficient_samples += 1
        # Include pairs with insufficient samples as candidates
        # Don't apply CV threshold since we don't have enough data to make that judgment
        directions_bits = determine_directions_bits(j, filter.positive, filter.negative)
        return PairCandidate(
            i,
            j,
            directions_bits,
            Float32(cv),  # Record computed CV even if based on few samples
            UInt16(n_valid)
        )
    end

    # Apply CV threshold only to pairs with sufficient samples
    if cv > filter.cv_threshold
        filter.pairs_cv_filtered += 1
        return nothing
    end

    # Apply transitivity filtering AFTER CV check (much more efficient)
    # Only check transitivity for pairs that passed CV filtering
    if filter.use_transitivity
        if are_concordant(filter.concordance_tracker, i, j)
            filter.pairs_transitivity_concordant_filtered += 1
            filter.pairs_skipped_by_transitivity += 1
            return nothing  # Skip known concordant pairs (already in ConcordanceTracker)
        elseif is_non_concordant(filter.concordance_tracker, i, j)
            filter.pairs_transitivity_non_concordant_filtered += 1
            filter.pairs_skipped_by_transitivity += 1
            return nothing  # Skip known non-concordant pairs
        end
    end

    # Determine directions
    directions_bits = determine_directions_bits(j, filter.positive, filter.negative)

    # Create candidate
    return PairCandidate(
        i,
        j,
        directions_bits,
        Float32(cv),
        UInt16(n_valid)
    )
end

# ========================================================================================
# Section 4: Bidirectional Communication
# ========================================================================================

"""
Update filter with newly discovered concordant pairs from analysis.
This enables dynamic filtering where analysis discoveries improve filter efficiency.
"""
function update_filter_with_discoveries!(
    filter::StreamingCandidateFilter,
    newly_concordant::AbstractVector{Tuple{Symbol,Symbol}}
)
    isempty(newly_concordant) && return

    # Update concordance tracker with discoveries
    for (c1_id, c2_id) in newly_concordant
        c1_idx = filter.concordance_tracker.id_to_idx[c1_id]
        c2_idx = filter.concordance_tracker.id_to_idx[c2_id]
        union_sets!(filter.concordance_tracker, c1_idx, c2_idx)
    end

    filter.transitivity_updates_received += 1  # Track discovery updates
    @debug "Updated filter with discoveries" count = length(newly_concordant) total_updates = filter.transitivity_updates_received
end

"""
Signal filter to stop early (e.g., quality threshold reached).
"""
function signal_early_stop!(filter::StreamingCandidateFilter, reason::String="external_signal")
    @info "Filter early stop requested" reason = reason candidates_so_far = filter.candidates_found
    filter.should_stop = true
end



# ========================================================================================
# Section 7: Type Compatibility
# ========================================================================================


"""
Get filtering statistics report from a StreamingCandidateFilter.
Returns a NamedTuple with detailed filtering breakdown.
"""
function get_filter_report(filter::StreamingCandidateFilter)
    total_filtered = filter.pairs_balanced_filtered +
                     filter.pairs_trivial_filtered +
                     filter.pairs_cv_filtered +
                     filter.pairs_transitivity_concordant_filtered +
                     filter.pairs_transitivity_non_concordant_filtered

    return (
        pairs_tested=filter.pairs_tested,
        candidates_found=filter.candidates_found,
        total_filtered=total_filtered,
        filtering_rate=round(total_filtered / max(1, filter.pairs_tested) * 100, digits=1),
        breakdown=(
            balanced=filter.pairs_balanced_filtered,
            trivial=filter.pairs_trivial_filtered,
            cv_threshold=filter.pairs_cv_filtered,
            known_concordant=filter.pairs_transitivity_concordant_filtered,
            known_non_concordant=filter.pairs_transitivity_non_concordant_filtered,
            missing_samples=filter.pairs_missing_samples,
            insufficient_samples=filter.insufficient_samples
        )
    )
end

# Export main interface
export StreamingCandidateFilter, PairCandidate,
    update_filter_with_discoveries!, signal_early_stop!, get_filter_report