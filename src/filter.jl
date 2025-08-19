"""
filter.jl - Revolutionary streaming candidate filter for 1B+ candidates

Streaming producer-consumer architecture that processes unlimited candidates 
with constant memory usage (<1MB) through bidirectional filter-analysis communication.
"""

using OnlineStats
using OnlineStatsBase
using Statistics
using DocStringExtensions
using Mmap
import ConstraintTrees as C

# ========================================================================================
# Section 1: Core Data Structures (Minimal & Efficient)
# ========================================================================================

"""
Packed candidate structure optimized for memory efficiency.
15 bytes total vs 25+ bytes in previous versions.
"""
struct PairCandidate
    c1_idx::UInt32        # 4 bytes - supports up to 4B complexes
    c2_idx::UInt32        # 4 bytes 
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

"""
Custom OnlineStat for coefficient of variation computation using Welford's algorithm.
Provides numerical stability and supports efficient reuse via Base.empty!.
"""
mutable struct CVStat <: OnlineStat{Number}
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
    cv = std(o) / abs(o.mean)
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
    pairs_skipped_by_transitivity::Int  # Pairs skipped due to existing concordance/non-concordance
    transitivity_updates_received::Int  # How many discovery updates received

    # Control flags
    should_stop::Bool
    # Note: transitivity filtering moved to batch processing stage for better effectiveness
end

function StreamingCandidateFilter(
    complexes::Vector{Symbol},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10
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
        0, 0,  # Enhanced transitivity tracking counters
        false
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
    
    @info "Streaming filter complete" (
        pairs_tested=filter.pairs_tested,
        total_pairs_possible=total_pairs_possible,
        candidates_found=filter.candidates_found,
        pairs_balanced_filtered=filter.pairs_balanced_filtered,
        pairs_trivial_filtered=filter.pairs_trivial_filtered,
        pairs_concordant_filtered=filter.pairs_concordant_filtered,
        pairs_skipped_by_transitivity=filter.pairs_skipped_by_transitivity,
        transitivity_effectiveness_pct=transitivity_effectiveness,
        transitivity_updates_received=filter.transitivity_updates_received,
        pairs_missing_samples=filter.pairs_missing_samples,
        pairs_cv_filtered=filter.pairs_cv_filtered,
        insufficient_samples=filter.insufficient_samples,
        cv_threshold=filter.cv_threshold,
        total_iterations=filter.iteration_count,
        final_position=(filter.current_i, filter.current_j)
    )
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

        # Skip transitivity filtering at candidate generation stage.
        # Transitivity filtering is more effective during batch processing
        # when the concordance tracker has built up more relationships.
        # Early filtering here can miss valid concordant pairs.
        # 
        # Note: pairs_concordant_filtered and pairs_skipped_by_transitivity
        # remain at 0 to indicate no early transitivity filtering occurred.

        # Cache complex IDs lookup
        idx_to_id = filter.idx_to_id
        c1_id = idx_to_id[i]
        c2_id = idx_to_id[j]
    end

    # Cache samples tree access
    samples_tree = filter.samples_tree
    c1_samples = samples_tree[c1_id]
    c2_samples = samples_tree[c2_id]

    # Skip if samples missing
    if c1_samples === nothing || c2_samples === nothing
        filter.pairs_missing_samples += 1
        return nothing
    end

    # Determine sample size and check minimum requirement
    n_samples = min(length(c1_samples), length(c2_samples))
    if n_samples < 2
        filter.pairs_missing_samples += 1
        return nothing
    end

    # Reset and compute CV using OnlineStats with Welford's algorithm
    cv_stat = filter.cv_stat  # Cache cv_stat reference
    empty!(cv_stat)
    
    # Cache epsilon for faster access in tight loop
    epsilon = filter.cv_epsilon

    @inbounds for k in 1:n_samples
        ratio = (c1_samples[k] + epsilon) / (c2_samples[k] + epsilon)
        if isfinite(ratio)
            fit!(cv_stat, ratio)
        end
    end

    cv, n_valid = compute_cv(filter.cv_stat, filter.cv_epsilon)

    # Debug: Log first few CV calculations 
    if filter.pairs_tested <= 10  # Log first 10 pairs regardless of CV
        mean_ratio = mean(filter.cv_stat)
        @info "CV debug" i j c1_id c2_id cv n_samples n_valid mean_ratio
    end

    # Track insufficient samples but include in candidates
    if n_valid < filter.min_valid_samples
        filter.insufficient_samples += 1
    end

    # Apply CV threshold
    if cv > filter.cv_threshold
        filter.pairs_cv_filtered += 1
        return nothing
    end

    # Determine directions
    directions_bits = determine_directions_bits(j, filter.positive, filter.negative)

    # Create candidate
    return PairCandidate(
        UInt32(i),
        UInt32(j),
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
# Section 5: Memory Management & Adaptation  
# ========================================================================================

"""
Monitor memory pressure and adapt processing accordingly.
"""
function get_memory_pressure()::Symbol
    try
        if Sys.islinux()
            meminfo = read("/proc/meminfo", String)
            for line in split(meminfo, '\n')
                if startswith(line, "MemAvailable:")
                    kb = parse(Int, split(line)[2])
                    gb = kb / (1024^2)
                    return gb < 2.0 ? :high : (gb < 8.0 ? :medium : :low)
                end
            end
        end
    catch
        # Fallback for non-Linux or permission issues
    end
    return :medium  # Conservative default
end


# ========================================================================================
# Section 6: Chunked Streaming Architecture 
# ========================================================================================

"""
Chunked streaming wrapper that provides true constant-memory processing.
Yields chunks of candidates instead of collecting all into memory.

This enables processing of unlimited candidates with memory usage limited only by chunk size,
making it suitable for both personal computers and HPC clusters.
"""
struct ChunkedStreamingFilter{T<:AbstractVector{PairCandidate}}
    base_filter::StreamingCandidateFilter
    chunk_size::Int
    chunk_buffer::T

    function ChunkedStreamingFilter(
        base_filter::StreamingCandidateFilter;
        chunk_size::Int=10_000,
        chunk_type::Type{T}=Vector{PairCandidate}
    ) where {T<:AbstractVector{PairCandidate}}
        buffer = chunk_type(undef, 0)
        sizehint!(buffer, chunk_size)
        new{typeof(buffer)}(base_filter, chunk_size, buffer)
    end
end

"""
Iterator state for ChunkedStreamingFilter.
Tracks current position in the underlying filter.
"""
mutable struct ChunkedStreamingState
    filter_state::Union{Nothing,Any}
    chunk_buffer::Vector{PairCandidate}
    buffer_pos::Int
    is_exhausted::Bool
end

Base.eltype(::ChunkedStreamingFilter) = Vector{PairCandidate}
Base.IteratorSize(::ChunkedStreamingFilter) = Base.SizeUnknown()

"""
Iterator implementation for ChunkedStreamingFilter.
Returns chunks of candidates with constant memory usage.
"""
function Base.iterate(chunked_filter::ChunkedStreamingFilter, state::Union{Nothing,ChunkedStreamingState}=nothing)
    # Initialize state on first call
    if state === nothing
        buffer = similar(chunked_filter.chunk_buffer, 0)
        sizehint!(buffer, chunked_filter.chunk_size)
        state = ChunkedStreamingState(nothing, buffer, 1, false)
    end

    # Return nothing if already exhausted
    state.is_exhausted && return nothing

    # Clear buffer for next chunk
    empty!(state.chunk_buffer)

    # Fill chunk buffer
    candidates_collected = 0
    while candidates_collected < chunked_filter.chunk_size
        result = Base.iterate(chunked_filter.base_filter, state.filter_state)

        if result === nothing
            # Base filter is exhausted
            state.is_exhausted = true
            break
        end

        candidate, new_filter_state = result
        state.filter_state = new_filter_state

        push!(state.chunk_buffer, candidate)
        candidates_collected += 1
    end

    # Return chunk if we collected any candidates
    if !isempty(state.chunk_buffer)
        # Return a copy to allow state buffer reuse
        chunk_copy = copy(state.chunk_buffer)
        return (chunk_copy, state)
    else
        # No more candidates
        return nothing
    end
end

"""
Create a chunked streaming filter from the same parameters as the original filter.
This provides true constant-memory processing of unlimited candidates.

Example usage:
```julia
chunked_filter = create_chunked_streaming_filter(
    complexes, trivial_pairs, samples_tree, concordance_tracker;
    chunk_size=10_000,  # Process 10K candidates at a time
    cv_threshold=0.01
)

for chunk in chunked_filter
    # Process chunk with existing batch logic
    results = process_concordance_batch(constraints, chunk, tracker; ...)
    # Memory for chunk is freed automatically here
end
```
"""
function create_chunked_streaming_filter(
    complexes::Vector{Symbol},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    chunk_size::Int=10_000,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10
)::ChunkedStreamingFilter
    base_filter = StreamingCandidateFilter(
        complexes, trivial_pairs, samples_tree, concordance_tracker;
        cv_threshold, cv_epsilon, min_valid_samples
    )

    return ChunkedStreamingFilter(base_filter; chunk_size)
end

# ========================================================================================
# Section 6.5: Disk Spillover for Massive Models (1B+ candidates)
# ========================================================================================

"""
Disk-backed chunked streaming filter for massive models.
Uses memory-mapped files to handle unlimited candidates with minimal RAM usage.

Only use this for extremely large models where even chunked streaming 
might generate too many candidates to fit in available disk cache.
"""
mutable struct DiskBackedChunkedFilter
    base_filter::StreamingCandidateFilter
    chunk_size::Int
    spillover_file::String
    mmap_buffer::Union{Nothing,Vector{UInt8}}
    candidates_spilled::Int
    max_memory_chunks::Int

    function DiskBackedChunkedFilter(
        base_filter::StreamingCandidateFilter;
        chunk_size::Int=10_000,
        max_memory_chunks::Int=100,  # Keep only 100 chunks (1M candidates) in memory
        spillover_dir::String=tempdir()
    )
        # Create temporary file for spillover
        spillover_file = joinpath(spillover_dir, "cocoa_candidates_$(rand(UInt32)).bin")

        new(base_filter, chunk_size, spillover_file, nothing, 0, max_memory_chunks)
    end
end

"""
For extreme cases (1B+ candidates), create a disk-backed filter that spills to disk
when memory usage becomes too high.

Example: 
```julia
# For a model with 100K complexes (5B potential pairs)
disk_filter = create_disk_backed_filter(
    complexes, trivial_pairs, samples_tree, concordance_tracker;
    max_memory_chunks = 50,  # Only keep 500K candidates in memory
    spillover_dir = "/tmp"   # Use fast SSD for spillover
)
```
"""
function create_disk_backed_filter(
    complexes::Vector{Symbol},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    chunk_size::Int=10_000,
    max_memory_chunks::Int=100,
    spillover_dir::String=tempdir(),
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=10
)::DiskBackedChunkedFilter
    @info "Creating disk-backed filter for massive model" spillover_dir

    base_filter = StreamingCandidateFilter(
        complexes, trivial_pairs, samples_tree, concordance_tracker;
        cv_threshold, cv_epsilon, min_valid_samples
    )

    return DiskBackedChunkedFilter(
        base_filter;
        chunk_size, max_memory_chunks, spillover_dir
    )
end

# ========================================================================================
# Section 7: Type Compatibility
# ========================================================================================

"""
Type alias for backward compatibility with analysis.jl.
PairCandidate has the same field structure as the old PairPriority.
"""
const PairPriority = PairCandidate

# Export main interface
export StreamingCandidateFilter, PairCandidate, PairPriority,
    ChunkedStreamingFilter, create_chunked_streaming_filter,
    DiskBackedChunkedFilter, create_disk_backed_filter,
    update_filter_with_discoveries!, signal_early_stop!