"""
Correlation analysis functionality for COCOA.

This module contains:
- Streaming correlation statistics (StreamingStats, StreamingCorrelation)
- Correlation tracking with memory management (CorrelationTracker)
- Streaming correlation filtering functionality
"""

using Statistics
using Random
using StableRNGs
using Distributed
using COBREXA
using JuMP
using DocStringExtensions
using ProgressMeter
import ConstraintTrees as C
"""
Streaming statistics accumulator for memory-efficient correlation calculation.
"""
mutable struct StreamingStats
    n::Int64
    mean::Float64
    M2::Float64

    StreamingStats() = new(0, 0.0, 0.0)
end

function update!(s::StreamingStats, x::Float64)
    s.n += 1
    delta = x - s.mean
    s.mean += delta / s.n
    s.M2 += delta * (x - s.mean)
end

variance(s::StreamingStats) = s.n > 1 ? s.M2 / (s.n - 1) : 0.0

"""
Streaming correlation accumulator.
"""
mutable struct StreamingCorrelation
    n::Int64
    mean_x::Float64
    mean_y::Float64
    cov_sum::Float64
    var_x_sum::Float64
    var_y_sum::Float64

    StreamingCorrelation() = new(0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function update!(c::StreamingCorrelation, x::Float64, y::Float64)
    c.n += 1

    delta_x = x - c.mean_x
    delta_y = y - c.mean_y

    c.mean_x += delta_x / c.n
    c.mean_y += delta_y / c.n

    c.cov_sum += delta_x * (y - c.mean_y)
    c.var_x_sum += delta_x * (x - c.mean_x)
    c.var_y_sum += delta_y * (y - c.mean_y)
end

function correlation(c::StreamingCorrelation)
    if c.n < 2 || c.var_x_sum ≈ 0 || c.var_y_sum ≈ 0
        return 0.0
    end
    return c.cov_sum / sqrt(c.var_x_sum * c.var_y_sum)
end

"""
Correlation tracker entry for memory-efficient correlation storage.
"""
struct CorrelationEntry
    pair_key::Tuple{Symbol,Symbol}
    correlation_acc::StreamingCorrelation
    current_correlation::Float64
    last_updated::Int
    confidence_upper_bound::Float64  # Statistical upper bound for correlation
end

"""
Hierarchical correlation tracker that manages memory using scientifically principled criteria.
Only rejects pairs based on statistical evidence or promotes them to higher confidence tiers.
"""
mutable struct CorrelationTracker
    # Tier 1: High confidence pairs (>= promotion_threshold)
    high_confidence::Dict{Tuple{Symbol,Symbol},StreamingCorrelation}

    # Tier 2: Under evaluation with LRU eviction
    under_evaluation::Dict{Tuple{Symbol,Symbol},CorrelationEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}  # O(1) LRU position lookup
    lru_counter::Int  # Monotonic counter for LRU ordering

    # Configuration
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64  # Confidence level for statistical rejection

    # Statistics
    pairs_promoted::Int
    pairs_rejected_statistically::Int
    pairs_evicted_lru::Int

    function CorrelationTracker(max_pairs::Int, promotion_threshold::Float64=0.8, rejection_confidence::Float64=0.95)
        new(
            Dict{Tuple{Symbol,Symbol},StreamingCorrelation}(),
            Dict{Tuple{Symbol,Symbol},CorrelationEntry}(),
            Dict{Tuple{Symbol,Symbol},Int}(),  # lru_positions
            0,  # lru_counter
            max_pairs,
            30,  # min_samples_for_decision
            promotion_threshold,
            rejection_confidence,
            0, 0, 0
        )
    end
end

"""
Calculate confidence interval upper bound for correlation using Fisher's z-transformation.
"""
function correlation_confidence_upper_bound(corr_acc::StreamingCorrelation, confidence_level::Float64)
    if corr_acc.n < 4  # Need minimum samples for meaningful CI
        return 1.0  # Conservative: assume could reach any correlation
    end

    r = abs(correlation(corr_acc))
    if r >= 0.999  # Handle numerical issues near 1
        return 1.0
    end

    # Fisher's z-transformation
    z = 0.5 * log((1 + r) / (1 - r))
    se_z = 1 / sqrt(corr_acc.n - 3)

    # Critical value for confidence interval
    z_critical = 1.96  # Approximation for 95% confidence
    if confidence_level ≈ 0.99
        z_critical = 2.576
    elseif confidence_level ≈ 0.90
        z_critical = 1.645
    end

    # Upper bound of confidence interval
    z_upper = z + z_critical * se_z
    r_upper = (exp(2 * z_upper) - 1) / (exp(2 * z_upper) + 1)

    return min(r_upper, 1.0)
end

"""
Update LRU order for a pair key using O(1) operations.
"""
function update_lru!(tracker::CorrelationTracker, pair_key::Tuple{Symbol,Symbol})
    # Update LRU position with O(1) counter-based approach
    tracker.lru_counter += 1
    tracker.lru_positions[pair_key] = tracker.lru_counter
end

"""
Efficient copy constructor for StreamingCorrelation.
Avoids deepcopy overhead by directly copying the simple fields.
"""
function Base.copy(c::StreamingCorrelation)
    new_corr = StreamingCorrelation()  # Use the default constructor
    new_corr.n = c.n
    new_corr.mean_x = c.mean_x
    new_corr.mean_y = c.mean_y
    new_corr.cov_sum = c.cov_sum
    new_corr.var_x_sum = c.var_x_sum
    new_corr.var_y_sum = c.var_y_sum
    return new_corr
end

"""
In-place update of destination StreamingCorrelation from source.
Eliminates allocation entirely for existing entries.
"""
function update_in_place!(dest::StreamingCorrelation, src::StreamingCorrelation)
    dest.n = src.n
    dest.mean_x = src.mean_x
    dest.mean_y = src.mean_y
    dest.cov_sum = src.cov_sum
    dest.var_x_sum = src.var_x_sum
    dest.var_y_sum = src.var_y_sum
    return dest
end

"""
Update the correlation tracker with a new correlation value using scientific criteria.
"""
function update_correlation_tracker!(
    tracker::CorrelationTracker,
    pair_key::Tuple{Symbol,Symbol},
    corr_acc::StreamingCorrelation,
    sample_number::Int,
    final_threshold::Float64=0.95
)
    current_corr = abs(correlation(corr_acc))

    # Check if already in high confidence
    if haskey(tracker.high_confidence, pair_key)
        # In-place update for existing high confidence entries
        update_in_place!(tracker.high_confidence[pair_key], corr_acc)
        return true
    end

    # Calculate confidence interval upper bound
    upper_bound = correlation_confidence_upper_bound(corr_acc, tracker.rejection_confidence)

    # Statistical rejection: upper bound significantly below final threshold
    if corr_acc.n >= tracker.min_samples_for_decision &&
       upper_bound < (final_threshold - 0.05)  # Conservative margin

        # Remove from tracking if present
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end

        tracker.pairs_rejected_statistically += 1
        return false
    end

    # Promotion to high confidence
    if current_corr >= tracker.promotion_threshold && corr_acc.n >= 20
        tracker.high_confidence[pair_key] = copy(corr_acc)  # Only copy when creating new entry

        # Remove from under_evaluation if present
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end

        tracker.pairs_promoted += 1
        return true
    end

    # Keep in under_evaluation tier
    if haskey(tracker.under_evaluation, pair_key)
        # In-place update for existing under_evaluation entries
        existing_entry = tracker.under_evaluation[pair_key]
        update_in_place!(existing_entry.correlation_acc, corr_acc)

        # Update the entry fields that changed
        new_entry = CorrelationEntry(
            pair_key,
            existing_entry.correlation_acc,  # Reuse the updated accumulator
            current_corr,
            sample_number,
            upper_bound
        )
        tracker.under_evaluation[pair_key] = new_entry
        update_lru!(tracker, pair_key)
    else
        # Create new entry only when needed
        entry = CorrelationEntry(
            pair_key,
            copy(corr_acc),  # Only copy when creating new entry
            current_corr,
            sample_number,
            upper_bound
        )

        # Handle capacity limit with LRU eviction
        if length(tracker.under_evaluation) >= tracker.max_under_evaluation
            # Find least recently used key with O(1) lookup
            lru_key = nothing
            min_counter = typemax(Int)
            for (key, position) in tracker.lru_positions
                if position < min_counter
                    min_counter = position
                    lru_key = key
                end
            end

            if lru_key !== nothing
                delete!(tracker.under_evaluation, lru_key)
                delete!(tracker.lru_positions, lru_key)
                tracker.pairs_evicted_lru += 1
            end
        end

        tracker.under_evaluation[pair_key] = entry
        update_lru!(tracker, pair_key)
    end

    return true
end

"""
Get all pairs that meet the final correlation threshold from both tiers.
"""
function get_candidate_pairs(tracker::CorrelationTracker, final_threshold::Float64, min_valid_samples::Int)
    candidates = CorrelationEntry[]

    # High confidence pairs (already meet promotion threshold)
    for (pair_key, corr_acc) in tracker.high_confidence
        if corr_acc.n >= min_valid_samples && abs(correlation(corr_acc)) >= final_threshold
            push!(candidates, CorrelationEntry(
                pair_key,
                corr_acc,
                abs(correlation(corr_acc)),
                0,  # last_updated not needed
                1.0  # confidence_upper_bound not needed
            ))
        end
    end

    # Under evaluation pairs
    for entry in values(tracker.under_evaluation)
        if entry.correlation_acc.n >= min_valid_samples &&
           abs(entry.current_correlation) >= final_threshold
            push!(candidates, entry)
        end
    end

    # Sort by correlation strength
    sort!(candidates, by=e -> abs(e.current_correlation), rev=true)

    return candidates
end

"""
Get statistics about tracker performance.
"""
function get_tracker_stats(tracker::CorrelationTracker)
    return (
        high_confidence_pairs=length(tracker.high_confidence),
        under_evaluation_pairs=length(tracker.under_evaluation),
        max_capacity=tracker.max_under_evaluation,
        pairs_promoted=tracker.pairs_promoted,
        pairs_rejected_statistically=tracker.pairs_rejected_statistically,
        pairs_evicted_lru=tracker.pairs_evicted_lru,
        total_tracked=length(tracker.high_confidence) + length(tracker.under_evaluation)
    )
end

"""
Pair priority information for sorting and filtering.
"""
struct PairPriority
    c1_idx::Int
    c2_idx::Int
    directions::Set{Symbol}
    correlation::Float64
    n_samples::Int
    is_high_confidence::Bool
end

"""
$(TYPEDSIGNATURES)

Perform streaming correlation analysis for complex concordance with direct matrix sampling.
- Uses memory-efficient direct matrix sampling
- Filters out balanced complexes and trivial pairs
- Returns PairPriority objects for high-correlation candidates
"""
function streaming_correlation_filter(
    complexes::Vector{Complex},
    balanced_complexes::Set{Symbol},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    warmup::Matrix{Float64},
    constraints::ConstraintTree;
    tolerance::Float64=1e-12,
    correlation_threshold::Float64=0.95,
    sample_size::Int=100,
    min_valid_samples::Int=30,
    max_correlation_pairs::Int=500_000,
    early_correlation_threshold::Float64=0.8,
    workers=workers,
    seed::Union{Int,Nothing}=42,
)
    # Setup simple RNG for reproducible sampling
    rng = seed === nothing ? StableRNG() : StableRNG(seed)

    # Filter to get only complexes that need to be sampled
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    n_active = length(active_complexes)

    # Create a mapping from complex ID to original index for efficient lookup of trivial pairs
    original_indices = Dict(c.id => i for (i, c) in enumerate(complexes))

    # Initialize the hierarchical, memory-aware correlation tracker
    correlation_tracker = CorrelationTracker(
        max_correlation_pairs,
        early_correlation_threshold,
        0.95, # Confidence level for statistical rejection
    )

    # === OPTIMIZED SAMPLING CONFIGURATION ===
    @info "Configuring optimized sampler"

    # Reduce sampling overhead by using fewer chains with more samples each
    n_chains = min(4, length(workers))  # Use fewer chains
    n_warmup_points_per_chain = min(100, size(warmup, 1))  # Limit warmup points

    # More aggressive burn-in and thinning to reduce total iterations
    burn_in_period = 32
    thinning_interval = 8

    # Calculate required collections more efficiently
    target_samples_per_chain = ceil(Int, sample_size / n_chains)
    n_collections_per_chain = min(target_samples_per_chain, n_warmup_points_per_chain)

    # Generate iteration list more efficiently
    iters_to_collect = collect(burn_in_period:thinning_interval:(burn_in_period+(n_collections_per_chain-1)*thinning_interval))

    @info "Optimized sampling parameters" n_chains n_warmup_points_per_chain n_collections_per_chain iters_to_collect

    # === MEMORY-EFFICIENT SAMPLING ===
    @info "Generating flux samples with memory optimization"

    # Use a subset of warmup points to reduce memory pressure
    if size(warmup, 1) > n_warmup_points_per_chain
        selected_indices = sort(randperm(rng, size(warmup, 1))[1:n_warmup_points_per_chain])
        limited_warmup = warmup[selected_indices, :]
    else
        limited_warmup = warmup
    end

    # Sample with optimized settings
    all_samples = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints;
        output=constraints.concordance_analysis.complexes,
        start_variables=limited_warmup,
        seed=rand(rng, UInt64),
        n_chains=n_chains,
        collect_iterations=iters_to_collect,
        workers=workers,
    )

    # === OPTIMIZED SAMPLE PROCESSING ===
    @info "Processing samples with zero-copy optimization"

    # Pre-filter active complexes more efficiently
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    n_active = length(active_complexes)

    # Build index mappings once
    original_indices = Dict{Symbol,Int}()
    for (i, c) in enumerate(complexes)
        original_indices[c.id] = i
    end

    # Use sparse set for memory-efficient skip tracking (massive memory savings)
    skip_pairs_set = Set{Tuple{Int,Int}}()
    for i = 1:n_active
        ci = active_complexes[i]
        ci_original_idx = get(original_indices, ci.id, 0)
        if ci_original_idx == 0
            continue
        end

        for j = (i+1):n_active
            cj = active_complexes[j]
            cj_original_idx = get(original_indices, cj.id, 0)
            if cj_original_idx == 0
                continue
            end

            canonical_pair = ci_original_idx < cj_original_idx ?
                             (ci_original_idx, cj_original_idx) : (cj_original_idx, ci_original_idx)
            if canonical_pair in trivial_pairs
                push!(skip_pairs_set, (i, j))
            end
        end
    end

    # === ZERO-COPY ACTIVITY EXTRACTION ===
    @info "Extracting activities with zero-copy access"

    # Instead of copying all activities, create views/references
    activity_refs = Dict{Symbol,Vector{Float64}}()
    for c in active_complexes
        if haskey(all_samples, c.id)
            activity_refs[c.id] = all_samples[c.id]  # Direct reference, no copy
        end
    end

    # Filter to only complexes with actual data and apply zero-copy slicing
    valid_complexes = [c for c in active_complexes if haskey(activity_refs, c.id)]
    n_valid = length(valid_complexes)
    @info "Valid complexes with data: $n_valid"

    # === STREAMING CORRELATION COMPUTATION ===
    @info "Computing streaming correlations with memory optimization"

    n_samples = length(first(values(activity_refs)))
    @info "Total samples available: $n_samples"

    # Calculate total number of pairs to process
    total_pairs = 0
    for i = 1:n_valid
        ci = valid_complexes[i]
        ci_original_idx = get(original_indices, ci.id, 0)
        if ci_original_idx == 0
            continue
        end
        for j = (i+1):n_valid
            cj = valid_complexes[j]
            cj_original_idx = get(original_indices, cj.id, 0)
            if cj_original_idx == 0
                continue
            end
            if (i, j) ∉ skip_pairs_set
                total_pairs += 1
            end
        end
    end
    @info "Total pairs to process: $total_pairs"
    # Initialize progress meter
    progress = Progress(total_pairs, desc="Computing correlations: ", showspeed=true)

    # Process correlations in batches to avoid memory explosion
    processed_pairs = 0

    for i = 1:n_valid
        ci = valid_complexes[i]
        ci_original_idx = get(original_indices, ci.id, 0)
        if ci_original_idx == 0
            continue
        end

        for j = (i+1):n_valid
            cj = valid_complexes[j]
            cj_original_idx = get(original_indices, cj.id, 0)
            if cj_original_idx == 0
                continue
            end

            # Check if we should skip this pair
            if (i, j) in skip_pairs_set
                continue
            end

            processed_pairs += 1

            # Create streaming correlation accumulator
            corr_acc = StreamingCorrelation()

            # Compute correlation using direct array access (zero-copy)
            x_values = activity_refs[ci.id]
            y_values = activity_refs[cj.id]

            # Stream the correlation calculation
            for k = 1:n_samples
                update!(corr_acc, x_values[k], y_values[k])
            end

            # Update tracker with this correlation
            pair_key = (ci.id, cj.id)
            update_correlation_tracker!(correlation_tracker, pair_key, corr_acc, n_samples, correlation_threshold)

            # Update progress meter
            ProgressMeter.next!(progress)
        end
    end

    ProgressMeter.finish!(progress)
    @info "Correlation computation complete" processed_pairs

    # === EXTRACT FINAL CANDIDATES ===
    @info "Extracting final candidates"

    candidates = get_candidate_pairs(correlation_tracker, correlation_threshold, min_valid_samples)
    @info "Found $(length(candidates)) candidate pairs"

    # Convert to PairPriority objects
    priorities = PairPriority[]
    for entry in candidates
        c1_id, c2_id = entry.pair_key

        # Find indices in the original complexes array
        c1_idx = get(original_indices, c1_id, 0)
        c2_idx = get(original_indices, c2_id, 0)

        if c1_idx == 0 || c2_idx == 0
            continue
        end

        # Determine directions based on complex types
        directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)

        # Check if this is a high confidence pair
        is_high_conf = haskey(correlation_tracker.high_confidence, entry.pair_key)

        push!(priorities, PairPriority(
            c1_idx, c2_idx, directions,
            abs(entry.current_correlation),
            entry.correlation_acc.n,
            is_high_conf
        ))
    end

    # Sort by correlation strength
    sort!(priorities, by=p -> p.correlation, rev=true)

    # Report tracker statistics
    stats = get_tracker_stats(correlation_tracker)
    @info "Correlation tracker stats" stats

    return priorities
end

"""
$(TYPEDSIGNATURES)

Determine which directions need to be tested based on complex activity patterns.
"""
function determine_directions(
    c1_idx::Int, c2_idx::Int,
    positive_complexes::Set{Int}, negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int}
)::Set{Symbol}
    directions = Set{Symbol}()

    if c2_idx in positive_complexes
        push!(directions, :positive)
    elseif c2_idx in negative_complexes
        push!(directions, :negative)
    else  # unrestricted
        push!(directions, :positive)
        push!(directions, :negative)
    end

    return directions
end