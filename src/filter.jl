"""
filter.jl - Streaming candidate filter for concordance pair generation

Streaming producer-consumer architecture that traverses the pair space with O(1)
incremental memory per candidate through bidirectional filter-analysis communication.
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
24 bytes on 64-bit systems (two Int indices, UInt8 direction bits, Float32 CV, UInt16 sample count).
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
Streaming candidate filter that traverses the pair space with O(1) incremental
memory per candidate. Implements Julia iterator protocol; no O(n²) candidate list
is materialised.
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

    # Block cache for the coefficient of variation.
    #
    # The CV of a pair's activity ratio is a PURE function of (i, j) and the flux
    # samples — it touches no mutable state. It is also the expensive part of the
    # filter: O(sample_size) per pair over n_complexes^2/2 pairs, which for a
    # genome-scale model is 256 million pairs. The filter itself runs serially on the
    # master while every worker sits idle.
    #
    # So the CVs for a whole block of consecutive pairs are computed in parallel with
    # threads and cached; the iterator then walks the block sequentially and applies
    # only the cheap, order-dependent checks (transitivity, counters) as before. The
    # candidate set is therefore unchanged — only the order of arithmetic differs, and
    # that arithmetic is deterministic per pair.
    cv_block_start::Int          # linear index of the first pair held in the cache
    cv_block_len::Int            # number of valid entries
    cv_block_cv::Vector{Float32}
    cv_block_n::Vector{UInt16}
    cv_block_size::Int
    inherit_non_concordance::Bool

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
    cv_epsilon::Float64=1e-16,
    min_valid_samples::Int=10,
    use_transitivity::Bool=true,
    inherit_non_concordance::Bool=true
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
        # CV block cache: empty, sized so one block is a few MB regardless of model size
        0, 0, Vector{Float32}(undef, CV_BLOCK_SIZE), Vector{UInt16}(undef, CV_BLOCK_SIZE),
        CV_BLOCK_SIZE, inherit_non_concordance,
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

const CV_BLOCK_SIZE = 1 << 20   # ~1M pairs per block: 6 MB of cache, ample work per thread

"""
Linear index (1-based) of the upper-triangular pair (i, j), i < j, over n complexes.
"""
@inline function _pair_linear_index(i::Int, j::Int, n::Int)::Int
    return (i - 1) * n - (i * (i - 1)) ÷ 2 + (j - i)
end

"""
Compute the coefficient of variation of the activity ratio of one pair directly from the
samples. Pure: reads only the sample vectors, touches no filter state, so it is safe to
call from several threads at once.
"""
function _pair_cv(c1_samples, c2_samples, epsilon::Float64)::Tuple{Float64,Int}
    n_samples = min(length(c1_samples), length(c2_samples))
    n_samples < 2 && return (Inf, n_samples)
    mean = 0.0
    m2 = 0.0
    k = 0
    @inbounds for t in 1:n_samples
        a = c1_samples[t]
        b = c2_samples[t]
        (ismissing(a) || ismissing(b)) && continue
        ratio = (a + epsilon) / (b + epsilon)
        isfinite(ratio) || continue
        k += 1
        delta = ratio - mean
        mean += delta / k
        m2 += delta * (ratio - mean)
    end
    k < 2 && return (Inf, k)
    sd = sqrt(m2 / (k - 1))
    return (abs(mean) < epsilon ? Inf : sd / abs(mean), k)
end

"""
Fill the CV block cache starting at the pair that follows `(i0, j0)`, in parallel.

Only pairs that survive the read-only rejections (balanced complex, trivial pair) get a
CV; the rest are marked with `Inf` and skipped by the sequential pass, which still does
the order-dependent work. The candidate set is identical to the serial filter's.
"""
function fill_cv_block!(filter::StreamingCandidateFilter, i0::Int, j0::Int)
    n = filter.n_complexes
    start_lin = _pair_linear_index(i0, j0, n)
    total = n * (n - 1) ÷ 2
    len = min(filter.cv_block_size, total - start_lin + 1)
    len <= 0 && (filter.cv_block_len = 0; return)

    # Materialise the (i, j) of each slot once, so the threaded loop is a flat range.
    idxs = Vector{Tuple{Int,Int}}(undef, len)
    i, j = i0, j0
    @inbounds for t in 1:len
        idxs[t] = (i, j)
        j += 1
        if j > n
            i += 1
            j = i + 1
        end
    end

    balanced = filter.balanced
    trivial = filter.trivial_pairs
    idx_to_id = filter.idx_to_id
    samples = filter.samples_tree
    eps = filter.cv_epsilon
    cvs = filter.cv_block_cv
    ns = filter.cv_block_n

    nchunks = max(1, min(Threads.nthreads(), len))
    tasks = map(1:nchunks) do c
        Threads.@spawn begin
            @inbounds for t in c:nchunks:len
                a, b = idxs[t]
                if balanced[a] || balanced[b] || ((a, b) in trivial)
                    cvs[t] = Float32(Inf); ns[t] = UInt16(0)
                    continue
                end
                s1 = samples[idx_to_id[a]]
                s2 = samples[idx_to_id[b]]
                if ismissing(s1) || ismissing(s2) || s1 === nothing || s2 === nothing
                    cvs[t] = Float32(Inf); ns[t] = UInt16(0)
                    continue
                end
                cv, k = _pair_cv(s1, s2, eps)
                cvs[t] = Float32(cv)
                ns[t] = UInt16(min(k, typemax(UInt16)))
            end
        end
    end
    foreach(wait, tasks)

    filter.cv_block_start = start_lin
    filter.cv_block_len = len
    return
end

"""
Process a single pair (i,j) and return PairCandidate if it passes all filters.
Returns nothing if pair should be skipped.
Optimized version with cached data access and reduced allocations.
"""
function process_pair(filter::StreamingCandidateFilter, i::Int, j::Int)::Union{PairCandidate,Nothing}
    filter.pairs_tested += 1

    # Read-only rejections, mirrored by `fill_cv_block!` so the cached entry agrees.
    if filter.balanced[i] || filter.balanced[j]
        filter.pairs_balanced_filtered += 1
        return nothing
    end
    if (i, j) in filter.trivial_pairs
        filter.pairs_trivial_filtered += 1
        return nothing
    end

    # Look the CV up in the block cache, refilling it in parallel when we run past it.
    lin = _pair_linear_index(i, j, filter.n_complexes)
    if filter.cv_block_len == 0 || lin < filter.cv_block_start ||
       lin >= filter.cv_block_start + filter.cv_block_len
        fill_cv_block!(filter, i, j)
    end
    slot = lin - filter.cv_block_start + 1
    cv = Float64(filter.cv_block_cv[slot])
    n_valid = Int(filter.cv_block_n[slot])

    if n_valid < 2
        # No usable samples: keep the pair as a candidate rather than discard it on no
        # evidence, and record it. Matches the serial filter's behaviour.
        filter.pairs_missing_samples += 1
        return PairCandidate(i, j, determine_directions_bits(j, filter.positive, filter.negative),
                             Float32(Inf), UInt16(n_valid))
    end

    if n_valid < filter.min_valid_samples
        filter.insufficient_samples += 1
        return PairCandidate(i, j, determine_directions_bits(j, filter.positive, filter.negative),
                             Float32(cv), UInt16(n_valid))
    end

    # The CV gate. NOTE this discards a pair WITHOUT ever testing it by LP, so a
    # genuinely concordant pair whose sampled ratio is poorly resolved is lost for good.
    # Measured on Saccharomyces cerevisiae: raising sample_size from
    # 1000 to 3000 cut the candidate set from 23,636 to 10,844 — more than half of the
    # candidates at 1000 were admitted on an underestimated CV — while the number of
    # concordant pairs FOUND went up.
    if cv > filter.cv_threshold
        filter.pairs_cv_filtered += 1
        return nothing
    end

    # Order-dependent checks stay sequential: they read (and path-compress) the
    # union-find tracker, which the parallel pass must not touch.
    if filter.use_transitivity
        if are_concordant(filter.concordance_tracker, i, j)
            filter.pairs_transitivity_concordant_filtered += 1
            filter.pairs_skipped_by_transitivity += 1
            return nothing
        elseif is_non_concordant(filter.concordance_tracker, i, j;
                                 inherit_modules=filter.inherit_non_concordance)
            filter.pairs_transitivity_non_concordant_filtered += 1
            filter.pairs_skipped_by_transitivity += 1
            return nothing
        end
    end

    return PairCandidate(i, j, determine_directions_bits(j, filter.positive, filter.negative),
                         Float32(cv), UInt16(n_valid))
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