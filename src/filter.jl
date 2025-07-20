"""
Correlation analysis functionality for COCOA.

This module contains:
- Streaming correlation statistics (StreamingStats, StreamingCorrelation)
- Correlation tracking with memory management (CorrelationTracker)
- CV tracking with memory management (CVTracker)
- Deterministic, streaming correlation and CV filtering functionality
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

# --- Base Streaming Statistics Structs and Functions ---

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
    if c.n < 2 || c.var_x_sum <= 0 || c.var_y_sum <= 0
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

# --- New Merge Functions for Deterministic Combination ---

"""
Merge two StreamingStats objects deterministically.
"""
function merge!(s1::StreamingStats, s2::StreamingStats)
    if s2.n == 0
        return s1
    end
    if s1.n == 0
        s1.n, s1.mean, s1.M2 = s2.n, s2.mean, s2.M2
        return s1
    end
    n_new = s1.n + s2.n
    delta = s2.mean - s1.mean
    s1.mean = (s1.n * s1.mean + s2.n * s2.mean) / n_new
    s1.M2 += s2.M2 + delta^2 * (s1.n * s2.n) / n_new
    s1.n = n_new
    return s1
end

"""
Merge two StreamingCorrelation objects deterministically.
"""
function merge!(c1::StreamingCorrelation, c2::StreamingCorrelation)
    if c2.n == 0
        return c1
    end
    if c1.n == 0
        c1.n, c1.mean_x, c1.mean_y, c1.cov_sum, c1.var_x_sum, c1.var_y_sum =
            c2.n, c2.mean_x, c2.mean_y, c2.cov_sum, c2.var_x_sum, c2.var_y_sum
        return c1
    end

    n_new = c1.n + c2.n
    delta_x = c2.mean_x - c1.mean_x
    delta_y = c2.mean_y - c1.mean_y

    c1.var_x_sum += c2.var_x_sum + delta_x^2 * (c1.n * c2.n) / n_new
    c1.var_y_sum += c2.var_y_sum + delta_y^2 * (c1.n * c2.n) / n_new
    c1.cov_sum += c2.cov_sum + delta_x * delta_y * (c1.n * c2.n) / n_new

    c1.mean_x = (c1.n * c1.mean_x + c2.n * c2.mean_x) / n_new
    c1.mean_y = (c1.n * c1.mean_y + c2.n * c2.mean_y) / n_new
    c1.n = n_new
    return c1
end

# --- Tracker Data Structures ---

"""
Correlation tracker entry for memory-efficient correlation storage.
"""
struct CorrelationEntry
    pair_key::Tuple{Symbol,Symbol}
    correlation_acc::StreamingCorrelation
    current_correlation::Float64
    last_updated::Int
    confidence_upper_bound::Float64
end

"""
Hierarchical correlation tracker that manages memory using scientifically principled criteria.
"""
mutable struct CorrelationTracker
    high_confidence::Dict{Tuple{Symbol,Symbol},StreamingCorrelation}
    under_evaluation::Dict{Tuple{Symbol,Symbol},CorrelationEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}
    lru_counter::Threads.Atomic{Int}
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64
    pairs_promoted::Threads.Atomic{Int}
    pairs_rejected_statistically::Threads.Atomic{Int}
    pairs_evicted_lru::Threads.Atomic{Int}
    num_locks::Int
    tracker_locks::Vector{ReentrantLock}

    function CorrelationTracker(max_pairs::Int, promotion_threshold::Float64=0.8, rejection_confidence::Float64=0.95)
        num_locks = 256
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
            [ReentrantLock() for _ = 1:num_locks],
        )
    end
end

"""
CV tracker entry for memory-efficient CV storage.
"""
struct CVEntry
    pair_key::Tuple{Symbol,Symbol}
    ratio_stats_acc::StreamingStats
    current_cv::Float64
    last_updated::Int
    confidence_lower_bound::Float64
end

"""
Hierarchical CV tracker that manages memory using scientifically principled criteria.
"""
mutable struct CVTracker
    low_cv_confirmed::Dict{Tuple{Symbol,Symbol},StreamingStats}
    under_evaluation::Dict{Tuple{Symbol,Symbol},CVEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}
    lru_counter::Threads.Atomic{Int}
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64
    pairs_promoted::Threads.Atomic{Int}
    pairs_rejected_statistically::Threads.Atomic{Int}
    pairs_evicted_lru::Threads.Atomic{Int}
    num_locks::Int
    tracker_locks::Vector{ReentrantLock}

    function CVTracker(max_pairs::Int, cv_threshold::Float64=0.01, rejection_confidence::Float64=0.95)
        num_locks = 256
        new(
            Dict{Tuple{Symbol,Symbol},StreamingStats}(),
            Dict{Tuple{Symbol,Symbol},CVEntry}(),
            Dict{Tuple{Symbol,Symbol},Int}(),
            Threads.Atomic{Int}(0),
            max_pairs,
            50,
            cv_threshold,
            rejection_confidence,
            Threads.Atomic{Int}(0),
            Threads.Atomic{Int}(0),
            Threads.Atomic{Int}(0),
            num_locks,
            [ReentrantLock() for _ = 1:num_locks],
        )
    end
end

# --- Tracker Logic Functions ---

function Base.copy(s::StreamingStats)
    new_s = StreamingStats()
    new_s.n = s.n
    new_s.mean = s.mean
    new_s.M2 = s.M2
    return new_s
end

function update_in_place!(dest::StreamingStats, src::StreamingStats)
    dest.n = src.n
    dest.mean = src.mean
    dest.M2 = src.M2
    return dest
end

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
Merge two tracker objects by merging their internal statistics.
"""
function merge_trackers!(t1, t2)
    # This generic function works for both CVTracker and CorrelationTracker
    # by checking for the existence of the relevant fields.
    field_to_merge = isdefined(t2, :high_confidence) ? :high_confidence : :low_cv_confirmed
    dict1 = getfield(t1, field_to_merge)
    dict2 = getfield(t2, field_to_merge)

    for (pair_key, acc2) in dict2
        if haskey(dict1, pair_key)
            merge!(dict1[pair_key], acc2)
        else
            dict1[pair_key] = acc2
        end
    end

    # Also merge the `under_evaluation` dictionaries for a more complete picture,
    # though this is less critical as they are transient.
    if isdefined(t1, :under_evaluation) && isdefined(t2, :under_evaluation)
        for (pair_key, entry2) in t2.under_evaluation
            if haskey(t1.under_evaluation, pair_key)
                entry1 = t1.under_evaluation[pair_key]
                # Merge the stats accumulators within the entries
                merge!(entry1.ratio_stats_acc, entry2.ratio_stats_acc)
            else
                # A simple copy is sufficient here, no need for complex LRU merge
                t1.under_evaluation[pair_key] = entry2
            end
        end
    end
end


function update_lru!(tracker, pair_key::Tuple{Symbol,Symbol})
    new_counter = Threads.atomic_add!(tracker.lru_counter, 1) + 1
    tracker.lru_positions[pair_key] = new_counter
end

function correlation_confidence_upper_bound(corr_acc::StreamingCorrelation, confidence_level::Float64)
    if corr_acc.n < 4
        return 1.0
    end
    r = abs(correlation(corr_acc))
    r = min(r, 0.999999)
    z = atanh(r)
    se_z = 1 / sqrt(corr_acc.n - 3)
    z_critical = 1.96 # Approx for 95%
    if confidence_level >= 0.99
        z_critical = 2.576
    elseif confidence_level >= 0.90
        z_critical = 1.645
    end
    z_upper = z + z_critical * se_z
    r_upper = tanh(z_upper)
    return min(r_upper, 1.0)
end

function update_correlation_tracker!(
    tracker::CorrelationTracker,
    pair_key::Tuple{Symbol,Symbol},
    corr_acc::StreamingCorrelation,
    sample_number::Int,
    final_threshold::Float64=0.95,
)
    current_corr = abs(correlation(corr_acc))
    lock_idx = (hash(pair_key) % tracker.num_locks) + 1
    lock(tracker.tracker_locks[lock_idx]) do
        if haskey(tracker.high_confidence, pair_key)
            update_in_place!(tracker.high_confidence[pair_key], corr_acc)
            return true
        end

        upper_bound =
            correlation_confidence_upper_bound(corr_acc, tracker.rejection_confidence)
        if corr_acc.n >= tracker.min_samples_for_decision &&
           upper_bound < (final_threshold - 0.05)
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_rejected_statistically, 1)
            return false
        end

        if current_corr >= tracker.promotion_threshold && corr_acc.n >= 20
            tracker.high_confidence[pair_key] = copy(corr_acc)
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_promoted, 1)
            return true
        end

        if haskey(tracker.under_evaluation, pair_key)
            existing_entry = tracker.under_evaluation[pair_key]
            update_in_place!(existing_entry.correlation_acc, corr_acc)
            new_entry = CorrelationEntry(
                pair_key,
                existing_entry.correlation_acc,
                current_corr,
                sample_number,
                upper_bound,
            )
            tracker.under_evaluation[pair_key] = new_entry
            update_lru!(tracker, pair_key)
        else
            entry = CorrelationEntry(
                pair_key,
                copy(corr_acc),
                current_corr,
                sample_number,
                upper_bound,
            )
            if length(tracker.under_evaluation) >= tracker.max_under_evaluation
                lru_key = nothing
                min_counter = typemax(Int)
                # This part is not O(1), but eviction is less frequent than updates.
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

function coefficient_of_variation(s::StreamingStats, epsilon::Float64=1e-12)
    m = s.mean
    v = variance(s)
    if abs(m) < epsilon || v < 0
        return Inf
    end
    return sqrt(v) / abs(m)
end

function cv_confidence_lower_bound(
    cv_acc::StreamingStats,
    confidence_level::Float64,
    epsilon::Float64=1e-12,
)
    if cv_acc.n < 10
        return 0.0
    end
    cv = coefficient_of_variation(cv_acc, epsilon)
    if isinf(cv)
        return Inf
    end
    se_cv = cv * sqrt((1 + 2 * (cv^2)) / (2 * cv_acc.n))
    z_critical = 1.645 # For one-sided 95% confidence
    return max(0.0, cv - z_critical * se_cv)
end

function update_cv_tracker!(
    tracker::CVTracker,
    pair_key::Tuple{Symbol,Symbol},
    ratio_stats_acc::StreamingStats,
    sample_number::Int;
    cv_epsilon::Float64=1e-12,
)
    current_cv = coefficient_of_variation(ratio_stats_acc, cv_epsilon)
    if isinf(current_cv)
        return true
    end

    lock_idx = (hash(pair_key) % tracker.num_locks) + 1
    lock(tracker.tracker_locks[lock_idx]) do
        if haskey(tracker.low_cv_confirmed, pair_key)
            update_in_place!(tracker.low_cv_confirmed[pair_key], ratio_stats_acc)
            return true
        end

        lower_bound =
            cv_confidence_lower_bound(ratio_stats_acc, tracker.rejection_confidence, cv_epsilon)

        if ratio_stats_acc.n >= tracker.min_samples_for_decision &&
           lower_bound > tracker.promotion_threshold
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_rejected_statistically, 1)
            return false
        end

        if current_cv <= tracker.promotion_threshold &&
           ratio_stats_acc.n >= tracker.min_samples_for_decision
            tracker.low_cv_confirmed[pair_key] = copy(ratio_stats_acc)
            if haskey(tracker.under_evaluation, pair_key)
                delete!(tracker.under_evaluation, pair_key)
                delete!(tracker.lru_positions, pair_key)
            end
            Threads.atomic_add!(tracker.pairs_promoted, 1)
            return true
        end

        if haskey(tracker.under_evaluation, pair_key)
            existing_entry = tracker.under_evaluation[pair_key]
            update_in_place!(existing_entry.ratio_stats_acc, ratio_stats_acc)
            new_entry = CVEntry(
                pair_key,
                existing_entry.ratio_stats_acc,
                current_cv,
                sample_number,
                lower_bound,
            )
            tracker.under_evaluation[pair_key] = new_entry
            update_lru!(tracker, pair_key)
        else
            if length(tracker.under_evaluation) >= tracker.max_under_evaluation
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
                    Threads.atomic_add!(tracker.pairs_evicted_lru, 1)
                end
            end
            entry = CVEntry(
                pair_key,
                copy(ratio_stats_acc),
                current_cv,
                sample_number,
                lower_bound,
            )
            tracker.under_evaluation[pair_key] = entry
            update_lru!(tracker, pair_key)
        end
        return true
    end
end

function get_cv_candidate_pairs(tracker::CVTracker, final_cv_threshold::Float64, min_valid_samples::Int)
    candidates = Set{Tuple{Symbol,Symbol}}()
    for l in tracker.tracker_locks
        lock(l)
    end
    try
        for (pair_key, stats_acc) in tracker.low_cv_confirmed
            if stats_acc.n >= min_valid_samples &&
               coefficient_of_variation(stats_acc) <= final_cv_threshold
                push!(candidates, pair_key)
            end
        end
        for entry in values(tracker.under_evaluation)
            if entry.ratio_stats_acc.n >= min_valid_samples &&
               entry.current_cv <= final_cv_threshold
                if !haskey(tracker.low_cv_confirmed, entry.pair_key)
                    push!(candidates, entry.pair_key)
                end
            end
        end
    finally
        for l in reverse(tracker.tracker_locks)
            unlock(l)
        end
    end
    return collect(candidates)
end

function get_candidate_pairs(
    tracker::CorrelationTracker,
    final_threshold::Float64,
    min_valid_samples::Int,
)
    candidates = CorrelationEntry[]
    for l in tracker.tracker_locks
        lock(l)
    end
    try
        for (pair_key, corr_acc) in tracker.high_confidence
            if corr_acc.n >= min_valid_samples && abs(correlation(corr_acc)) >= final_threshold
                push!(
                    candidates,
                    CorrelationEntry(pair_key, corr_acc, abs(correlation(corr_acc)), 0, 1.0),
                )
            end
        end
        for entry in values(tracker.under_evaluation)
            if entry.correlation_acc.n >= min_valid_samples &&
               abs(entry.current_correlation) >= final_threshold
                push!(candidates, entry)
            end
        end
    finally
        for l in reverse(tracker.tracker_locks)
            unlock(l)
        end
    end
    sort!(candidates, by=e -> abs(e.current_correlation), rev=true)
    return candidates
end

function get_tracker_stats(tracker)
    return (
        high_confidence_pairs=if isdefined(tracker, :high_confidence)
            length(tracker.high_confidence)
        else
            length(tracker.low_cv_confirmed)
        end,
        under_evaluation_pairs=length(tracker.under_evaluation),
        max_capacity=tracker.max_under_evaluation,
        pairs_promoted=tracker.pairs_promoted[],
        pairs_rejected_statistically=tracker.pairs_rejected_statistically[],
        pairs_evicted_lru=tracker.pairs_evicted_lru[],
        total_tracked=(
            if isdefined(tracker, :high_confidence)
                length(tracker.high_confidence)
            else
                length(tracker.low_cv_confirmed)
            end
        ) + length(tracker.under_evaluation),
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

# --- Parallel Worker Functions ---

function process_chain_samples_for_cor(
    constraints::C.ConstraintTree,
    warmup_subset::Matrix{Float64},
    sampler_config::NamedTuple,
    filter_config::NamedTuple,
    correlation_tracker::CorrelationTracker,
    active_complexes,
    valid_pairs,
)
    rng = StableRNG(sampler_config.seed)
    chain_samples = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=warmup_subset,
        seed=rand(rng, UInt64),
        n_chains=1,
        collect_iterations=sampler_config.iters_to_collect,
    )
    if isempty(chain_samples)
        return 0
    end

    activity_refs = Dict{Symbol,Vector{Float64}}()
    for c in active_complexes
        if haskey(chain_samples, c.id)
            activity_refs[c.id] = chain_samples[c.id]
        end
    end

    if isempty(activity_refs)
        return 0
    end

    n_samples_in_chain = length(first(values(activity_refs)))
    pool = [StreamingCorrelation() for _ = 1:10]

    for (pair_idx, (i, j)) in enumerate(valid_pairs)
        ci = active_complexes[i]
        cj = active_complexes[j]

        if !haskey(activity_refs, ci.id) || !haskey(activity_refs, cj.id)
            continue
        end

        corr_acc = pool[(pair_idx-1)%length(pool)+1]
        reset!(corr_acc)

        x_values = activity_refs[ci.id]
        y_values = activity_refs[cj.id]

        for k = 1:n_samples_in_chain
            update!(corr_acc, x_values[k], y_values[k])
        end

        pair_key = (ci.id, cj.id)
        update_correlation_tracker!(
            correlation_tracker,
            pair_key,
            corr_acc,
            n_samples_in_chain,
            filter_config.correlation_threshold,
        )
    end
    return length(valid_pairs)
end

function process_chain_samples_for_cv(
    constraints::C.ConstraintTree,
    warmup_subset::Matrix{Float64},
    sampler_config::NamedTuple,
    filter_config::NamedTuple,
    cv_tracker::CVTracker,
    active_complexes,
    valid_pairs,
)
    rng = StableRNG(sampler_config.seed)
    chain_samples = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=warmup_subset,
        seed=rand(rng, UInt64),
        n_chains=1,
        collect_iterations=sampler_config.iters_to_collect,
    )
    if isempty(chain_samples)
        return 0
    end

    activity_refs = Dict{Symbol,Vector{Float64}}()
    for c in active_complexes
        if haskey(chain_samples, c.id)
            activity_refs[c.id] = chain_samples[c.id]
        end
    end
    if isempty(activity_refs)
        return 0
    end

    n_samples_in_chain = length(first(values(activity_refs)))
    ratio_stats_acc = StreamingStats()

    for (i, j) in valid_pairs
        ci = active_complexes[i]
        cj = active_complexes[j]

        if !haskey(activity_refs, ci.id) || !haskey(activity_refs, cj.id)
            continue
        end

        ratio_stats_acc.n = 0
        ratio_stats_acc.mean = 0.0
        ratio_stats_acc.M2 = 0.0

        x_values = activity_refs[ci.id]
        y_values = activity_refs[cj.id]

        for k = 1:n_samples_in_chain
            ratio =
                (x_values[k] + filter_config.cv_epsilon) /
                (y_values[k] + filter_config.cv_epsilon)
            if isfinite(ratio)
                update!(ratio_stats_acc, ratio)
            end
        end

        if ratio_stats_acc.n > 1
            pair_key = (ci.id, cj.id)
            update_cv_tracker!(
                cv_tracker,
                pair_key,
                ratio_stats_acc,
                n_samples_in_chain,
                cv_epsilon=filter_config.cv_epsilon,
            )
        end
    end
    return length(valid_pairs)
end

# --- Main `streaming_filter` Function ---

"""
$(TYPEDSIGNATURES)

Perform deterministic, streaming correlation and/or CV analysis for complex concordance.
This version uses thread-local trackers and a final merge step to ensure
reproducibility without sacrificing parallel efficiency.
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
    max_cv_pairs::Int=1_000_000,
    early_correlation_threshold::Float64=0.8,
    workers=workers,
    seed::Union{Int,Nothing}=42,
    filter::Vector{Symbol}=[:cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-12,
)
    # === 1. Global and Deterministic Setup ===
    rng = seed === nothing ? StableRNG() : StableRNG(seed)
    n_threads = Threads.nthreads()
    n_chains = min(n_threads, 12, length(workers))
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    original_indices = concordance_tracker.id_to_idx

    n_active = length(active_complexes)
    all_pairs_indices =
        [(i, j) for i = 1:n_active for j = (i+1):n_active if !((i, j) in trivial_pairs)]

    sampler_config = (seed=rand(rng, UInt), iters_to_collect=[32])
    warmup_chunks = []
    n_warmup_available = size(warmup, 1)
    indices = 1:n_warmup_available
    target_samples_per_chain = ceil(Int, sample_size / n_chains)
    for i = 1:n_chains
        chain_indices = [
            indices[mod1(k, n_warmup_available)] for
            k = (i-1)*target_samples_per_chain+1:i*target_samples_per_chain
        ]
        push!(warmup_chunks, warmup[chain_indices, :])
    end

    # === 2. Execute Filters using Thread-Local Trackers ===
    pairs_to_test_indices = all_pairs_indices
    cv_candidate_keys = []
    final_cv_tracker = nothing

    if :cv in filter
        @info "Using deterministic streaming for CV filtering with $n_chains chains."
        thread_local_cv_trackers =
            [CVTracker(max_cv_pairs, cv_threshold) for _ = 1:n_chains]

        filter_config_cv = (cv_threshold=cv_threshold, cv_epsilon=cv_epsilon)

        progress_cv = Progress(n_chains, desc="Processing chains for CV: ", showspeed=true)
        tasks_cv = [
            Threads.@spawn begin
                process_chain_samples_for_cv(
                    constraints,
                    warmup_chunks[i],
                    (
                        seed=sampler_config.seed + i,
                        iters_to_collect=sampler_config.iters_to_collect,
                    ),
                    filter_config_cv,
                    thread_local_cv_trackers[i],
                    active_complexes,
                    pairs_to_test_indices,
                )
                ProgressMeter.next!(progress_cv)
            end for i = 1:n_chains
        ]
        fetch.(tasks_cv)
        ProgressMeter.finish!(progress_cv)

        @info "Merging CV tracker results..."
        final_cv_tracker = thread_local_cv_trackers[1]
        for i = 2:n_chains
            merge_trackers!(final_cv_tracker, thread_local_cv_trackers[i])
        end

        cv_candidate_keys =
            get_cv_candidate_pairs(final_cv_tracker, cv_threshold, min_valid_samples)
        @info "CV filtering complete. Found $(length(cv_candidate_keys)) candidate pairs."

        id_to_active_idx = Dict(c.id => i for (i, c) in enumerate(active_complexes))
        next_pairs_to_test = Tuple{Int,Int}[]
        for (c1_id, c2_id) in cv_candidate_keys
            i = get(id_to_active_idx, c1_id, 0)
            j = get(id_to_active_idx, c2_id, 0)
            if i > 0 && j > 0
                push!(next_pairs_to_test, i < j ? (i, j) : (j, i))
            end
        end
        pairs_to_test_indices = next_pairs_to_test
    end

    if !(:cor in filter)
        # Handle CV-only case: format CV results into PairPriority
        priorities = PairPriority[]
        if !isnothing(final_cv_tracker)
            for pair_key in cv_candidate_keys
                c1_id, c2_id = pair_key
                c1_idx = get(original_indices, c1_id, 0)
                c2_idx = get(original_indices, c2_id, 0)
                if c1_idx == 0 || c2_idx == 0
                    continue
                end

                stats = final_cv_tracker.low_cv_confirmed[pair_key]
                directions = determine_directions(
                    c1_idx,
                    c2_idx,
                    positive_complexes,
                    negative_complexes,
                    unrestricted_complexes,
                )

                # Create a placeholder priority object. Correlation is -1 to indicate not computed.
                push!(
                    priorities,
                    PairPriority(c1_idx, c2_idx, directions, -1.0, stats.n, true),
                )
            end
        end
        return priorities
    end

    # === 3. Correlation Filtering ===
    @info "Using deterministic streaming for correlation analysis on $(length(pairs_to_test_indices)) pairs."
    thread_local_cor_trackers = [
        CorrelationTracker(max_correlation_pairs, early_correlation_threshold) for
        _ = 1:n_chains
    ]
    filter_config_cor = (correlation_threshold=correlation_threshold,)

    progress_cor = Progress(n_chains, desc="Processing chains for Cor: ", showspeed=true)
    tasks_cor = [
        Threads.@spawn begin
            process_chain_samples_for_cor(
                constraints,
                warmup_chunks[i],
                (
                    seed=sampler_config.seed + i,
                    iters_to_collect=sampler_config.iters_to_collect,
                ),
                filter_config_cor,
                thread_local_cor_trackers[i],
                active_complexes,
                pairs_to_test_indices,
            )
            ProgressMeter.next!(progress_cor)
        end for i = 1:n_chains
    ]
    fetch.(tasks_cor)
    ProgressMeter.finish!(progress_cor)

    @info "Merging correlation tracker results..."
    final_cor_tracker = thread_local_cor_trackers[1]
    for i = 2:n_chains
        merge_trackers!(final_cor_tracker, thread_local_cor_trackers[i])
    end

    # === 4. Extract Final Correlation Results ===
    @info "Extracting final candidates from the merged correlation tracker."
    candidates =
        get_candidate_pairs(final_cor_tracker, correlation_threshold, min_valid_samples)
    @info "Found $(length(candidates)) candidate pairs meeting the criteria."
    priorities = PairPriority[]
    for entry in candidates
        c1_id, c2_id = entry.pair_key
        c1_idx = get(original_indices, c1_id, 0)
        c2_idx = get(original_indices, c2_id, 0)
        if c1_idx == 0 || c2_idx == 0
            continue
        end

        directions = determine_directions(
            c1_idx,
            c2_idx,
            positive_complexes,
            negative_complexes,
            unrestricted_complexes,
        )
        is_high_conf = haskey(final_cor_tracker.high_confidence, entry.pair_key)
        push!(
            priorities,
            PairPriority(
                c1_idx,
                c2_idx,
                directions,
                abs(entry.current_correlation),
                entry.correlation_acc.n,
                is_high_conf,
            ),
        )
    end

    sort!(priorities, by=p -> p.correlation, rev=true)
    return priorities
end

"""
$(TYPEDSIGNATURES)

Determine which directions need to be tested based on complex activity patterns.
"""
function determine_directions(
    c1_idx::Int,
    c2_idx::Int,
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
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