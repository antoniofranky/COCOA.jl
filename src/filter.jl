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
using Base.Threads

"""
Check if threading is available and beneficial for correlation analysis.
"""
function should_use_threading()::Bool
    n_threads = Threads.nthreads()
    if n_threads == 1
        @warn "Julia started with single thread. For better performance, start Julia with multiple threads: julia --threads=auto"
        return false
    else
        @info "Threading enabled with $n_threads threads"
        return true
    end
end

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
Reset a StreamingCorrelation for reuse in object pooling.
"""
function reset!(c::StreamingCorrelation)
    c.n = 0
    c.mean_x = 0.0
    c.mean_y = 0.0
    c.cov_sum = 0.0
    c.var_x_sum = 0.0
    c.var_y_sum = 0.0
    return c
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
Thread-safe implementation with sharded locks for reduced contention in multi-threaded analysis.
"""
mutable struct CorrelationTracker
    # Tier 1: High confidence pairs (>= promotion_threshold)
    high_confidence::Dict{Tuple{Symbol,Symbol},StreamingCorrelation}

    # Tier 2: Under evaluation with LRU eviction
    under_evaluation::Dict{Tuple{Symbol,Symbol},CorrelationEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}  # O(1) LRU position lookup
    lru_counter::Threads.Atomic{Int}

    # Configuration
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64

    # Statistics (thread-safe atomic counters)
    pairs_promoted::Threads.Atomic{Int}
    pairs_rejected_statistically::Threads.Atomic{Int}
    pairs_evicted_lru::Threads.Atomic{Int}

    # Thread synchronization (sharded locks)
    num_locks::Int
    tracker_locks::Vector{ReentrantLock}

    function CorrelationTracker(max_pairs::Int, promotion_threshold::Float64=0.8, rejection_confidence::Float64=0.95)
        num_locks = 256 # A power of 2, adjustable based on expected contention
        new(
            Dict{Tuple{Symbol,Symbol},StreamingCorrelation}(),
            Dict{Tuple{Symbol,Symbol},CorrelationEntry}(),
            Dict{Tuple{Symbol,Symbol},Int}(),
            Threads.Atomic{Int}(0),
            max_pairs,
            30,
            promotion_threshold,
            rejection_confidence,
            Threads.Atomic{Int}(0),
            Threads.Atomic{Int}(0),
            Threads.Atomic{Int}(0),
            num_locks,
            [ReentrantLock() for _ in 1:num_locks] # Sharded locks
        )
    end
end

"""
Calculate confidence interval upper bound for correlation using Fisher's z-transformation.
"""
function correlation_confidence_upper_bound(corr_acc::StreamingCorrelation, confidence_level::Float64)
    if corr_acc.n < 4
        return 1.0
    end

    r = abs(correlation(corr_acc))
    r = min(r, 0.999999) # Clamp to avoid Inf from atanh

    z = atanh(r)
    se_z = 1 / sqrt(corr_acc.n - 3)

    z_critical = 1.96 # Approx for 95%
    if confidence_level ≈ 0.99
        z_critical = 2.576
    elseif confidence_level ≈ 0.90
        z_critical = 1.645
    end

    z_upper = z + z_critical * se_z
    r_upper = tanh(z_upper)

    return min(r_upper, 1.0)
end

"""
Update LRU order for a pair key.
"""
function update_lru!(tracker::CorrelationTracker, pair_key::Tuple{Symbol,Symbol})
    new_counter = Threads.atomic_add!(tracker.lru_counter, 1) + 1
    tracker.lru_positions[pair_key] = new_counter
end

"""
Efficient copy constructor for StreamingCorrelation.
"""
function Base.copy(c::StreamingCorrelation)
    new_corr = StreamingCorrelation()
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
Thread-safe implementation with sharded locking for reduced contention.
"""
function update_correlation_tracker!(
    tracker::CorrelationTracker,
    pair_key::Tuple{Symbol,Symbol},
    corr_acc::StreamingCorrelation,
    sample_number::Int,
    final_threshold::Float64=0.95
)
    current_corr = abs(correlation(corr_acc))

    # Use a sharded lock based on the hash of the pair key
    lock_idx = (hash(pair_key) % tracker.num_locks) + 1

    lock(tracker.tracker_locks[lock_idx]) do
        # Check if already in high confidence
        if haskey(tracker.high_confidence, pair_key)
            update_in_place!(tracker.high_confidence[pair_key], corr_acc)
            return true
        end

        upper_bound = correlation_confidence_upper_bound(corr_acc, tracker.rejection_confidence)

        # Statistical rejection
        if corr_acc.n >= tracker.min_samples_for_decision && upper_bound < (final_threshold - 0.05)
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_rejected_statistically, 1)
            return false
        end

        # Promotion to high confidence
        if current_corr >= tracker.promotion_threshold && corr_acc.n >= 20
            tracker.high_confidence[pair_key] = copy(corr_acc)
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_promoted, 1)
            return true
        end

        # Keep in under_evaluation tier
        if haskey(tracker.under_evaluation, pair_key)
            existing_entry = tracker.under_evaluation[pair_key]
            update_in_place!(existing_entry.correlation_acc, corr_acc)
            new_entry = CorrelationEntry(pair_key, existing_entry.correlation_acc, current_corr, sample_number, upper_bound)
            tracker.under_evaluation[pair_key] = new_entry
            update_lru!(tracker, pair_key)
        else
            entry = CorrelationEntry(pair_key, copy(corr_acc), current_corr, sample_number, upper_bound)
            if length(tracker.under_evaluation) >= tracker.max_under_evaluation
                lru_key = nothing
                min_counter = typemax(Int)
                # This part is not O(1), but eviction is less frequent than updates.
                # A more complex structure (e.g., doubly linked list) would be needed for true O(1).
                for (key, position) in tracker.lru_positions
                    if position < min_counter
                        min_counter = position
                        lru_key = key
                    end
                end
                if lru_key !== nothing
                    delete!(tracker.under_evaluation, lru_key)
                    delete!(tracker.lru_positions, lru_key)
                    Threads.atomic_add!(tracker.pairs_evicted_lru, 1)
                end
            end
            tracker.under_evaluation[pair_key] = entry
            update_lru!(tracker, pair_key)
        end
        return true
    end
end


"""
Get all pairs that meet the final correlation threshold from both tiers.
"""
function get_candidate_pairs(tracker::CorrelationTracker, final_threshold::Float64, min_valid_samples::Int)
    candidates = CorrelationEntry[]

    # Lock all shards to ensure a consistent snapshot
    for l in tracker.tracker_locks
        lock(l)
    end

    try
        # High confidence pairs
        for (pair_key, corr_acc) in tracker.high_confidence
            if corr_acc.n >= min_valid_samples && abs(correlation(corr_acc)) >= final_threshold
                push!(candidates, CorrelationEntry(pair_key, corr_acc, abs(correlation(corr_acc)), 0, 1.0))
            end
        end

        # Under evaluation pairs
        for entry in values(tracker.under_evaluation)
            if entry.correlation_acc.n >= min_valid_samples && abs(entry.current_correlation) >= final_threshold
                push!(candidates, entry)
            end
        end
    finally
        # Ensure all locks are unlocked
        for l in reverse(tracker.tracker_locks)
            unlock(l)
        end
    end

    sort!(candidates, by=e -> abs(e.current_correlation), rev=true)
    return candidates
end

"""
Get statistics about tracker performance.
"""
function get_tracker_stats(tracker::CorrelationTracker)
    # No need to lock for atomic reads
    return (
        high_confidence_pairs=length(tracker.high_confidence),
        under_evaluation_pairs=length(tracker.under_evaluation),
        max_capacity=tracker.max_under_evaluation,
        pairs_promoted=tracker.pairs_promoted[],
        pairs_rejected_statistically=tracker.pairs_rejected_statistically[],
        pairs_evicted_lru=tracker.pairs_evicted_lru[],
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

Compute coefficient of variation for activity ratios between complex pairs.
Pairs with CV <= cv_threshold are considered for concordance testing.
This function requires all samples to be present in memory.
"""
function coefficient_of_variation_filter(
    activity_refs::Dict{Symbol,Vector{Float64}},
    valid_complexes::Vector{MetabolicComplex},
    cv_threshold::Float64,
    cv_epsilon::Float64
)
    n_valid = length(valid_complexes)
    cv_candidates = Vector{Tuple{Int,Int,Float64}}()
    @info "Computing CV filtering for $(n_valid) complexes" cv_threshold

    for i in 1:n_valid
        ci = valid_complexes[i]
        ci_activities = activity_refs[ci.id]

        for j in (i+1):n_valid
            cj = valid_complexes[j]
            cj_activities = activity_refs[cj.id]

            ratios = (ci_activities .+ cv_epsilon) ./ (cj_activities .+ cv_epsilon)
            valid_ratios = ratios[isfinite.(ratios)]

            if length(valid_ratios) >= 2
                ratio_mean = mean(valid_ratios)
                if abs(ratio_mean) > cv_epsilon
                    ratio_std = std(valid_ratios)
                    cv_value = ratio_std / abs(ratio_mean)
                    if cv_value <= cv_threshold
                        push!(cv_candidates, (i, j, cv_value))
                    end
                end
            end
        end
    end

    @info "CV filtering complete" n_pairs_evaluated = div(n_valid * (n_valid - 1), 2) n_cv_candidates = length(cv_candidates)
    return cv_candidates
end

"""
Worker function to process samples from a single chain and update a shared tracker.
This function is designed to be called in parallel for the memory-efficient streaming path.
"""
function process_chain_samples(
    chain_idx::Int,
    constraints::C.ConstraintTree,
    warmup_subset::Matrix{Float64},
    sampler_config::NamedTuple,
    filter_config::NamedTuple,
    shared_data::NamedTuple,
)
    rng = StableRNG(sampler_config.seed + chain_idx) # Ensure each chain has a unique seed

    # 1. Generate samples for this chain ONLY
    chain_samples = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=warmup_subset,
        seed=rand(rng, UInt64),
        n_chains=1, # CRITICAL: only one chain per worker
        collect_iterations=sampler_config.iters_to_collect,
    )

    if isempty(chain_samples)
        @warn "Chain $chain_idx produced no samples."
        return 0
    end

    # 2. Extract activity references (zero-copy)
    activity_refs = Dict{Symbol,Vector{Float64}}()
    for c in shared_data.valid_complexes
        if haskey(chain_samples, c.id)
            activity_refs[c.id] = chain_samples[c.id]
        end
    end

    if isempty(activity_refs)
        @warn "Chain $chain_idx had no valid complexes with data."
        return 0
    end

    n_samples_in_chain = length(first(values(activity_refs)))

    # 3. Process pairs and update shared tracker
    pool = [StreamingCorrelation() for _ in 1:10]
    pool_size = length(pool)

    processed_count = 0
    for (pair_idx, (i, j)) in enumerate(shared_data.valid_pairs)
        ci = shared_data.valid_complexes[i]
        cj = shared_data.valid_complexes[j]

        if !haskey(activity_refs, ci.id) || !haskey(activity_refs, cj.id)
            continue
        end

        corr_acc = pool[(pair_idx-1)%pool_size+1]
        reset!(corr_acc)

        x_values = activity_refs[ci.id]
        y_values = activity_refs[cj.id]

        for k = 1:n_samples_in_chain
            update!(corr_acc, x_values[k], y_values[k])
        end

        pair_key = (ci.id, cj.id)
        update_correlation_tracker!(
            shared_data.correlation_tracker,
            pair_key,
            corr_acc,
            n_samples_in_chain,
            filter_config.correlation_threshold
        )
        processed_count += 1
    end

    return processed_count
end


"""
$(TYPEDSIGNATURES)

Perform streaming correlation analysis for complex concordance with direct matrix sampling.
This version uses a true streaming approach, processing samples from each parallel
chain as they are generated to keep memory usage low.
"""
function streaming_filter(
    complexes::Vector{MetabolicComplex},
    balanced_complexes::Set{Symbol},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    warmup::Matrix{Float64},
    constraints::C.ConstraintTree,
    concordance_tracker::ConcordanceTracker;
    correlation_threshold::Float64=0.95,
    sample_size::Int=100,
    min_valid_samples::Int=30,
    max_correlation_pairs::Int=500_000,
    early_correlation_threshold::Float64=0.8,
    workers=workers,
    seed::Union{Int,Nothing}=42,
    filter::Vector{Symbol}=[:cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-12,
)
    # === 1. Global Setup ===
    rng = seed === nothing ? StableRNG() : StableRNG(seed)
    n_threads = Threads.nthreads()
    n_chains = min(n_threads, 12, length(workers))

    correlation_tracker = CorrelationTracker(
        max_correlation_pairs,
        early_correlation_threshold,
        0.95,
    )
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    original_indices = concordance_tracker.id_to_idx

    # === 2. Select Execution Path (Memory-Efficient vs. CV-Compatible) ===
    if :cv in filter
        # --- CV-based path (higher memory usage) ---
        @warn "CV filtering selected. This requires collecting all samples into memory."

        # A. Bulk sample generation
        iters_to_collect = [32]
        all_samples = COBREXA.sample_constraints(
            COBREXA.sample_chain_achr,
            constraints.balance;
            output=constraints.activities,
            start_variables=warmup,
            seed=rand(rng, UInt64),
            n_chains=n_chains,
            collect_iterations=iters_to_collect,
            workers=workers,
        )

        # B. Create activity refs and identify valid complexes with data
        activity_refs = Dict{Symbol,Vector{Float64}}()
        for c in active_complexes
            if haskey(all_samples, c.id) && !isempty(all_samples[c.id])
                activity_refs[c.id] = all_samples[c.id]
            end
        end
        valid_complexes_with_data = [c for c in active_complexes if haskey(activity_refs, c.id)]
        n_samples_total = isempty(activity_refs) ? 0 : length(first(values(activity_refs)))

        # C. Run CV filter to get pairs to test
        cv_candidates = coefficient_of_variation_filter(activity_refs, valid_complexes_with_data, cv_threshold, cv_epsilon)
        pairs_to_test = [(valid_complexes_with_data[i], valid_complexes_with_data[j]) for (i, j, _) in cv_candidates]

        # D. Run correlation on the filtered pairs
        progress = Progress(length(pairs_to_test), desc="Correlating CV candidates: ", showspeed=true)

        Threads.@threads for (ci, cj) in pairs_to_test
            corr_acc = StreamingCorrelation()
            x_values = activity_refs[ci.id]
            y_values = activity_refs[cj.id]
            for k = 1:n_samples_total
                update!(corr_acc, x_values[k], y_values[k])
            end
            update_correlation_tracker!(correlation_tracker, (ci.id, cj.id), corr_acc, n_samples_total, correlation_threshold)
            ProgressMeter.next!(progress)
        end
        ProgressMeter.finish!(progress)

    else
        # --- Correlation-only path (low memory streaming) ---
        @info "Using memory-efficient streaming for correlation analysis."

        # A. Determine all pairs to test
        n_active = length(active_complexes)
        skip_pairs_set = Set{Tuple{Int,Int}}()
        for i in 1:n_active, j in (i+1):n_active
            ci_original_idx = get(original_indices, active_complexes[i].id, 0)
            cj_original_idx = get(original_indices, active_complexes[j].id, 0)
            if ci_original_idx > 0 && cj_original_idx > 0
                canonical_pair = ci_original_idx < cj_original_idx ? (ci_original_idx, cj_original_idx) : (cj_original_idx, ci_original_idx)
                if canonical_pair in trivial_pairs
                    push!(skip_pairs_set, (i, j))
                end
            end
        end
        valid_pairs = [(i, j) for i in 1:n_active for j in (i+1):n_active if (i, j) ∉ skip_pairs_set]

        # B. Configure and distribute work
        target_samples_per_chain = ceil(Int, sample_size / n_chains)
        sampler_config = (seed=rand(rng, UInt), iters_to_collect=[32])
        filter_config = (correlation_threshold=correlation_threshold,)
        shared_data = (correlation_tracker=correlation_tracker, valid_complexes=active_complexes, valid_pairs=valid_pairs)

        warmup_chunks = []
        n_warmup_available = size(warmup, 1)
        indices = 1:n_warmup_available
        for i in 1:n_chains
            chain_indices = [indices[mod1(k, n_warmup_available)] for k in (i-1)*target_samples_per_chain+1:i*target_samples_per_chain]
            push!(warmup_chunks, warmup[chain_indices, :])
        end

        # C. Parallel chain execution
        progress = Progress(n_chains, desc="Processing chains: ", showspeed=true)
        tasks = [Threads.@spawn begin
            process_chain_samples(i, constraints, warmup_chunks[i], sampler_config, filter_config, shared_data)
            ProgressMeter.next!(progress)
        end for i in 1:n_chains]

        fetch.(tasks)
        ProgressMeter.finish!(progress)
    end

    # === 3. Extract Final Results (Common to both paths) ===
    @info "Extracting final candidates from the aggregated tracker."
    candidates = get_candidate_pairs(correlation_tracker, correlation_threshold, min_valid_samples)
    @info "Found $(length(candidates)) candidate pairs meeting the criteria."

    priorities = PairPriority[]
    for entry in candidates
        c1_id, c2_id = entry.pair_key
        c1_idx = get(original_indices, c1_id, 0)
        c2_idx = get(original_indices, c2_id, 0)
        if c1_idx == 0 || c2_idx == 0
            continue
        end

        directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)
        is_high_conf = haskey(correlation_tracker.high_confidence, entry.pair_key)

        push!(priorities, PairPriority(c1_idx, c2_idx, directions, abs(entry.current_correlation), entry.correlation_acc.n, is_high_conf))
    end

    sort!(priorities, by=p -> p.correlation, rev=true)

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
