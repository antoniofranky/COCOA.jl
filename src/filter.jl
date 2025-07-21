"""
Correlation analysis functionality for COCOA.
--- MEMORY-EFFICIENT STREAMING REDESIGN ---
This version has been refactored for maximum memory efficiency. It decouples
sample generation from filtering using a producer-consumer model with a Channel.
This avoids both re-running the expensive sampling algorithm and holding all
samples in memory at once.
"""

using Statistics
using Random
using StableRNGs
using COBREXA
using JuMP
using DocStringExtensions
using ProgressMeter
import ConstraintTrees as C
using OnlineStats

# --- Tracker Data Structures and Logic ---
# This section contains the original, unchanged data structures and logic for
# managing the statistical trackers. It is included here for completeness.

struct CorrelationEntry
    pair_key::Tuple{Symbol,Symbol}
    correlation_acc::CovMatrix
    current_correlation::Float64
    last_updated::Int
    confidence_upper_bound::Float64
end

mutable struct CorrelationTracker
    high_confidence::Dict{Tuple{Symbol,Symbol},CovMatrix}
    under_evaluation::Dict{Tuple{Symbol,Symbol},CorrelationEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}
    lru_counter::Int
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64
    pairs_promoted::Int
    pairs_rejected_statistically::Int
    pairs_evicted_lru::Int

    function CorrelationTracker(max_pairs::Int, promotion_threshold::Float64=0.8, rejection_confidence::Float64=0.95)
        new(Dict{Tuple{Symbol,Symbol},CovMatrix}(), Dict{Tuple{Symbol,Symbol},CorrelationEntry}(), Dict{Tuple{Symbol,Symbol},Int}(), 0, max_pairs, 30, promotion_threshold, rejection_confidence, 0, 0, 0)
    end
end

struct CVEntry
    pair_key::Tuple{Symbol,Symbol}
    ratio_stats_acc::Variance
    current_cv::Float64
    last_updated::Int
    confidence_lower_bound::Float64
end

mutable struct CVTracker
    low_cv_confirmed::Dict{Tuple{Symbol,Symbol},Variance}
    under_evaluation::Dict{Tuple{Symbol,Symbol},CVEntry}
    lru_positions::Dict{Tuple{Symbol,Symbol},Int}
    lru_counter::Int
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64
    pairs_promoted::Int
    pairs_rejected_statistically::Int
    pairs_evicted_lru::Int

    function CVTracker(max_pairs::Int, cv_threshold::Float64=0.01, rejection_confidence::Float64=0.95)
        new(Dict{Tuple{Symbol,Symbol},Variance}(), Dict{Tuple{Symbol,Symbol},CVEntry}(), Dict{Tuple{Symbol,Symbol},Int}(), 0, max_pairs, 50, cv_threshold, rejection_confidence, 0, 0, 0)
    end
end

function update_lru!(tracker, pair_key::Tuple{Symbol,Symbol})
    tracker.lru_counter += 1
    tracker.lru_positions[pair_key] = tracker.lru_counter
end

function correlation_confidence_upper_bound(corr_acc::CovMatrix, confidence_level::Float64)
    n = nobs(corr_acc)
    if n < 4
        return 1.0
    end
    r = min(abs(cor(corr_acc)[1, 2]), 0.999999)
    z = atanh(r)
    se_z = 1 / sqrt(n - 3)
    z_critical = confidence_level >= 0.99 ? 2.576 : (confidence_level >= 0.90 ? 1.645 : 1.96)
    return min(tanh(z + z_critical * se_z), 1.0)
end

function update_correlation_tracker!(tracker::CorrelationTracker, pair_key::Tuple{Symbol,Symbol}, corr_acc::CovMatrix, final_threshold::Float64=0.95)
    current_corr = abs(cor(corr_acc)[1, 2])
    if haskey(tracker.high_confidence, pair_key)
        merge!(tracker.high_confidence[pair_key], corr_acc)
        return true
    end
    if haskey(tracker.under_evaluation, pair_key)
        existing_entry = tracker.under_evaluation[pair_key]
        merge!(existing_entry.correlation_acc, corr_acc)
        corr_acc = existing_entry.correlation_acc
        current_corr = abs(cor(corr_acc)[1, 2])
    end
    upper_bound = correlation_confidence_upper_bound(corr_acc, tracker.rejection_confidence)
    if nobs(corr_acc) >= tracker.min_samples_for_decision && upper_bound < (final_threshold - 0.05)
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end
        tracker.pairs_rejected_statistically += 1
        return false
    end
    if current_corr >= tracker.promotion_threshold && nobs(corr_acc) >= 20
        tracker.high_confidence[pair_key] = corr_acc
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end
        tracker.pairs_promoted += 1
        return true
    end
    entry = CorrelationEntry(pair_key, corr_acc, current_corr, nobs(corr_acc), upper_bound)
    if !haskey(tracker.under_evaluation, pair_key) && length(tracker.under_evaluation) >= tracker.max_under_evaluation
        lru_key = first(sort([k for (k, v) in tracker.lru_positions if v == minimum(values(tracker.lru_positions))]))
        delete!(tracker.under_evaluation, lru_key)
        delete!(tracker.lru_positions, lru_key)
        tracker.pairs_evicted_lru += 1
    end
    tracker.under_evaluation[pair_key] = entry
    update_lru!(tracker, pair_key)
    return true
end

function coefficient_of_variation(s::Variance, epsilon::Float64=1e-12)
    m = mean(s)
    v = var(s)
    return abs(m) < epsilon || v < 0 ? Inf : sqrt(v) / abs(m)
end

function cv_confidence_lower_bound(cv_acc::Variance, confidence_level::Float64, epsilon::Float64=1e-12)
    n = nobs(cv_acc)
    if n < 10
        return 0.0
    end
    cv = coefficient_of_variation(cv_acc, epsilon)
    if isinf(cv)
        return Inf
    end
    se_cv = cv * sqrt((1 + 2 * (cv^2)) / (2 * n))
    return max(0.0, cv - 1.645 * se_cv)
end

function update_cv_tracker!(tracker::CVTracker, pair_key::Tuple{Symbol,Symbol}, ratio_stats_acc::Variance; cv_epsilon::Float64=1e-12)
    current_cv = coefficient_of_variation(ratio_stats_acc, cv_epsilon)
    if isinf(current_cv)
        return true
    end
    if haskey(tracker.low_cv_confirmed, pair_key)
        merge!(tracker.low_cv_confirmed[pair_key], ratio_stats_acc)
        return true
    end
    if haskey(tracker.under_evaluation, pair_key)
        existing_entry = tracker.under_evaluation[pair_key]
        merge!(existing_entry.ratio_stats_acc, ratio_stats_acc)
        ratio_stats_acc = existing_entry.ratio_stats_acc
        current_cv = coefficient_of_variation(ratio_stats_acc, cv_epsilon)
    end
    lower_bound = cv_confidence_lower_bound(ratio_stats_acc, tracker.rejection_confidence, cv_epsilon)
    if nobs(ratio_stats_acc) >= tracker.min_samples_for_decision && lower_bound > tracker.promotion_threshold
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end
        tracker.pairs_rejected_statistically += 1
        return false
    end
    if current_cv <= tracker.promotion_threshold && nobs(ratio_stats_acc) >= tracker.min_samples_for_decision
        tracker.low_cv_confirmed[pair_key] = ratio_stats_acc
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            delete!(tracker.lru_positions, pair_key)
        end
        tracker.pairs_promoted += 1
        return true
    end
    entry = CVEntry(pair_key, ratio_stats_acc, current_cv, nobs(ratio_stats_acc), lower_bound)
    if !haskey(tracker.under_evaluation, pair_key) && length(tracker.under_evaluation) >= tracker.max_under_evaluation
        lru_key = first(sort([k for (k, v) in tracker.lru_positions if v == minimum(values(tracker.lru_positions))]))
        delete!(tracker.under_evaluation, lru_key)
        delete!(tracker.lru_positions, lru_key)
        tracker.pairs_evicted_lru += 1
    end
    tracker.under_evaluation[pair_key] = entry
    update_lru!(tracker, pair_key)
    return true
end

function get_cv_candidate_pairs(tracker::CVTracker, final_cv_threshold::Float64, min_valid_samples::Int)
    candidates = Set{Tuple{Symbol,Symbol}}()
    for (pair_key, stats_acc) in tracker.low_cv_confirmed
        if nobs(stats_acc) >= min_valid_samples && coefficient_of_variation(stats_acc) <= final_cv_threshold
            push!(candidates, pair_key)
        end
    end
    for entry in values(tracker.under_evaluation)
        if nobs(entry.ratio_stats_acc) >= min_valid_samples && entry.current_cv <= final_cv_threshold && !haskey(tracker.low_cv_confirmed, entry.pair_key)
            push!(candidates, entry.pair_key)
        end
    end
    return sort(collect(candidates))
end

function get_candidate_pairs(tracker::CorrelationTracker, final_threshold::Float64, min_valid_samples::Int)
    candidates = CorrelationEntry[]
    for (pair_key, corr_acc) in tracker.high_confidence
        if nobs(corr_acc) >= min_valid_samples && abs(cor(corr_acc)[1, 2]) >= final_threshold
            push!(candidates, CorrelationEntry(pair_key, corr_acc, abs(cor(corr_acc)[1, 2]), nobs(corr_acc), 1.0))
        end
    end
    for entry in values(tracker.under_evaluation)
        if nobs(entry.correlation_acc) >= min_valid_samples && abs(entry.current_correlation) >= final_threshold
            push!(candidates, entry)
        end
    end
    sort!(candidates, by=e -> abs(e.current_correlation), rev=true)
    return candidates
end


struct PairPriority
    c1_idx::Int
    c2_idx::Int
    directions::Set{Symbol}
    correlation::Float64
    n_samples::Int
    is_high_confidence::Bool
end

# --- Sample Producer ---
function sample_producer(
    constraints::C.ConstraintTree,
    warmup::Matrix{Float64},
    original_indices::Dict{Symbol,Int};
    sample_size::Int,
    workers,
    seed::Union{Int,Nothing}
)
    sample_channel = Channel{Dict{Symbol,Float64}}(100)
    rng = seed === nothing ? StableRNG() : StableRNG(seed)

    @async begin
        try
            if isempty(warmup)
                @warn "Sample producer received no warmup points. No samples will be generated."
                return
            end

            @info "Starting background sample generation..."
            start_vars_indices = rand(rng, 1:size(warmup, 1), min(sample_size, size(warmup, 1)))
            start_variables = warmup[start_vars_indices, :]

            all_samples_tree = COBREXA.sample_constraints(
                COBREXA.sample_chain_achr,
                constraints.balance;
                output=constraints.activities,
                start_variables=start_variables,
                workers=workers,
                seed=rand(rng, UInt64),
                n_chains=1,
                collect_iterations=[32]
            )

            if isempty(all_samples_tree)
                @warn "Sampling process (sample_constraints) returned an empty result tree."
                return
            end

            samples_dict = Dict{Symbol,Vector{Float64}}(id => vec(s) for (id, s) in all_samples_tree if haskey(original_indices, id))

            if isempty(samples_dict)
                @warn "Sample generation yielded no valid samples after filtering by complex."
                return
            end

            num_samples = length(first(values(samples_dict)))
            @info "Sample generation complete. Streaming $num_samples samples..."
            for i in 1:num_samples
                put!(sample_channel, Dict{Symbol,Float64}(id => val[i] for (id, val) in samples_dict))
            end
        finally
            close(sample_channel)
        end
    end
    return sample_channel
end

# --- Streaming Filter (Consumer) ---
function streaming_filter(
    complexes::Vector,
    balanced_complexes::Set{Int},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    sample_channel::Channel{Dict{Symbol,Float64}},
    concordance_tracker::ConcordanceTracker;
    correlation_threshold::Float64=0.95,
    min_valid_samples::Int=0,
    max_correlation_pairs::Int=100_000_000,
    max_cv_pairs::Int=1_000_000,
    early_correlation_threshold::Float64=0.8,
    filter::Vector{Symbol}=[:cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-8,
)
    original_indices = concordance_tracker.id_to_idx
    active_complexes = [c for c in complexes if !(get(original_indices, c.id, 0) in balanced_complexes)]
    all_pairs = [(c1.id, c2.id) for (i, c1) in enumerate(active_complexes) for (j, c2) in enumerate(active_complexes) if i < j && !((get(original_indices, c1.id, 0), get(original_indices, c2.id, 0)) in trivial_pairs)]

    final_cv_tracker = CVTracker(max_cv_pairs, cv_threshold)
    final_cor_tracker = CorrelationTracker(max_correlation_pairs, early_correlation_threshold)


    @info "Consuming samples and updating trackers directly..."
    for sample in sample_channel
        for (ci_id, cj_id) in all_pairs
            # Ensure both values exist in the current sample
            if !haskey(sample, ci_id) || !haskey(sample, cj_id)
                continue
            end

            # --- Direct CV Tracker Update ---
            if :cv in filter
                ratio = (sample[ci_id] + cv_epsilon) / (sample[cj_id] + cv_epsilon)
                if isfinite(ratio)
                    # Create a temporary Variance object for this single observation.
                    # The tracker will merge it or create a new entry.
                    single_ratio_stat = Variance()
                    fit!(single_ratio_stat, ratio)
                    update_cv_tracker!(final_cv_tracker, (ci_id, cj_id), single_ratio_stat; cv_epsilon=cv_epsilon)
                end
            end

            # --- Direct Correlation Tracker Update ---
            if :cor in filter
                # Create a temporary CovMatrix for this single observation.
                # The tracker will merge it or create a new entry.
                single_corr_stat = CovMatrix(2)
                fit!(single_corr_stat, [sample[ci_id], sample[cj_id]])
                update_correlation_tracker!(final_cor_tracker, (ci_id, cj_id), single_corr_stat, correlation_threshold)
            end
        end
    end
    @info "Finished consuming all samples."

    # --- CORRECTED EXTRACTION LOGIC ---

    # Start with all pairs, then filter down
    pairs_to_process = all_pairs

    # Step 1: Apply CV filter if specified
    if :cv in filter
        cv_candidate_keys = get_cv_candidate_pairs(final_cv_tracker, cv_threshold, min_valid_samples)
        @info "CV filtering complete. Found $(length(cv_candidate_keys)) candidate pairs."

        # The next stage will only process pairs that passed the CV filter
        pairs_to_process = cv_candidate_keys
    end

    # Step 2: Apply Correlation filter if specified
    if :cor in filter
        # If we also did CV, we filter the already-filtered list. Otherwise, we filter all pairs.
        @info "Performing correlation analysis on $(length(pairs_to_process)) pairs."

        # Get correlation candidates from the subset of pairs that passed the CV filter
        # We need to check their stats from the master `final_cor_tracker`
        candidates = get_candidate_pairs(final_cor_tracker, correlation_threshold, min_valid_samples)

        # Filter these candidates further to only include those that also passed the CV step
        cor_candidate_keys = Set(entry.pair_key for entry in candidates)
        final_candidate_keys = intersect(Set(pairs_to_process), cor_candidate_keys)

        final_candidates = [entry for entry in candidates if entry.pair_key in final_candidate_keys]

        @info "Found $(length(final_candidates)) candidate pairs meeting all criteria."

        priorities = PairPriority[]
        for entry in final_candidates
            c1_id, c2_id = entry.pair_key
            c1_idx, c2_idx = get(original_indices, c1_id, 0), get(original_indices, c2_id, 0)
            if c1_idx == 0 || c2_idx == 0
                continue
            end
            directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)
            is_high_conf = haskey(final_cor_tracker.high_confidence, entry.pair_key)
            push!(priorities, PairPriority(c1_idx, c2_idx, directions, abs(entry.current_correlation), nobs(entry.correlation_acc), is_high_conf))
        end
        sort!(priorities, by=p -> (-p.is_high_confidence, p.correlation), rev=true)
        return priorities
    else
        # This block executes if only CV filtering was done
        priorities = PairPriority[]
        for pair_key in pairs_to_process # This is the list from get_cv_candidate_pairs
            c1_id, c2_id = pair_key
            c1_idx, c2_idx = get(original_indices, c1_id, 0), get(original_indices, c2_id, 0)
            if c1_idx == 0 || c2_idx == 0
                continue
            end

            stats_acc = if haskey(final_cv_tracker.low_cv_confirmed, pair_key)
                final_cv_tracker.low_cv_confirmed[pair_key]
            elseif haskey(final_cv_tracker.under_evaluation, pair_key)
                final_cv_tracker.under_evaluation[pair_key].ratio_stats_acc
            else
                nothing
            end

            if isnothing(stats_acc)
                continue
            end

            directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)
            push!(priorities, PairPriority(c1_idx, c2_idx, directions, -1.0, nobs(stats_acc), true))
        end
        sort!(priorities, by=x -> (x.c1_idx, x.c2_idx))
        return priorities
    end
end

function determine_directions(c1_idx::Int, c2_idx::Int, positive_complexes::Set{Int}, negative_complexes::Set{Int}, unrestricted_complexes::Set{Int})::Set{Symbol}
    directions = Set{Symbol}()
    if c2_idx in positive_complexes
        push!(directions, :positive)
    elseif c2_idx in negative_complexes
        push!(directions, :negative)
    else
        push!(directions, :positive)
        push!(directions, :negative)
    end
    return directions
end