"""
Correlation analysis functionality for COCOA.

This module uses OnlineStats.jl for high-performance, memory-efficient
streaming statistics to identify candidate pairs for concordance analysis.
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
using OnlineStats

# --- Tracker Data Structures (Now using OnlineStats) ---

"""
Correlation tracker entry for memory-efficient correlation storage.
"""
struct CorrelationEntry
    pair_key::Tuple{Symbol,Symbol}
    correlation_acc::CovMatrix
    current_correlation::Float64
    last_updated::Int
    confidence_upper_bound::Float64
end

"""
Hierarchical correlation tracker that manages memory using scientifically principled criteria.
"""
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
        new(
            Dict{Tuple{Symbol,Symbol},CovMatrix}(),
            Dict{Tuple{Symbol,Symbol},CorrelationEntry}(),
            Dict{Tuple{Symbol,Symbol},Int}(),
            0, max_pairs, 30, promotion_threshold, rejection_confidence, 0, 0, 0
        )
    end
end

"""
CV tracker entry for memory-efficient CV storage.
"""
struct CVEntry
    pair_key::Tuple{Symbol,Symbol}
    ratio_stats_acc::Variance
    current_cv::Float64
    last_updated::Int
    confidence_lower_bound::Float64
end

"""
Hierarchical CV tracker that manages memory using scientifically principled criteria.
"""
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
        new(
            Dict{Tuple{Symbol,Symbol},Variance}(),
            Dict{Tuple{Symbol,Symbol},CVEntry}(),
            Dict{Tuple{Symbol,Symbol},Int}(),
            0, max_pairs, 50, cv_threshold, rejection_confidence, 0, 0, 0
        )
    end
end

# --- Tracker Logic Functions (Adapted for OnlineStats) ---

"""
Merge two tracker objects by merging their internal OnlineStats accumulators.
"""
function merge_trackers!(t1, t2)
    is_cor_tracker = isdefined(t2, :high_confidence)

    dict1_confirmed = getfield(t1, is_cor_tracker ? :high_confidence : :low_cv_confirmed)
    dict2_confirmed = getfield(t2, is_cor_tracker ? :high_confidence : :low_cv_confirmed)

    for pair_key in sort(collect(keys(dict2_confirmed)))
        acc2 = dict2_confirmed[pair_key]
        if haskey(dict1_confirmed, pair_key)
            merge!(dict1_confirmed[pair_key], acc2)
        else
            dict1_confirmed[pair_key] = acc2
        end
    end

    dict1_eval = t1.under_evaluation
    dict2_eval = t2.under_evaluation
    for pair_key in sort(collect(keys(dict2_eval)))
        entry2 = dict2_eval[pair_key]
        if haskey(dict1_eval, pair_key)
            entry1 = dict1_eval[pair_key]
            acc_field = is_cor_tracker ? :correlation_acc : :ratio_stats_acc
            merge!(getfield(entry1, acc_field), getfield(entry2, acc_field))
        else
            dict1_eval[pair_key] = entry2
        end
    end

    t1.pairs_promoted += t2.pairs_promoted
    t1.pairs_rejected_statistically += t2.pairs_rejected_statistically
    t1.pairs_evicted_lru += t2.pairs_evicted_lru
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
    r = abs(cor(corr_acc)[1, 2])
    r = min(r, 0.999999)
    z = atanh(r)
    se_z = 1 / sqrt(n - 3)
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
    corr_acc::CovMatrix,
    final_threshold::Float64=0.95,
)
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
        min_counter = typemax(Int)
        for position in values(tracker.lru_positions)
            min_counter = min(min_counter, position)
        end
        lru_candidates = [key for (key, pos) in tracker.lru_positions if pos == min_counter]
        if !isempty(lru_candidates)
            sort!(lru_candidates)
            lru_key = lru_candidates[1]
            delete!(tracker.under_evaluation, lru_key)
            delete!(tracker.lru_positions, lru_key)
            tracker.pairs_evicted_lru += 1
        end
    end
    tracker.under_evaluation[pair_key] = entry
    update_lru!(tracker, pair_key)
    return true
end

function coefficient_of_variation(s::Variance, epsilon::Float64=1e-12)
    m = mean(s)
    v = var(s)
    if abs(m) < epsilon || v < 0
        return Inf
    end
    return sqrt(v) / abs(m)
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
    z_critical = 1.645 # For one-sided 95% confidence
    return max(0.0, cv - z_critical * se_cv)
end

function update_cv_tracker!(
    tracker::CVTracker,
    pair_key::Tuple{Symbol,Symbol},
    ratio_stats_acc::Variance;
    cv_epsilon::Float64=1e-12,
)
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
        min_counter = typemax(Int)
        for position in values(tracker.lru_positions)
            min_counter = min(min_counter, position)
        end
        lru_candidates = [key for (key, pos) in tracker.lru_positions if pos == min_counter]
        if !isempty(lru_candidates)
            sort!(lru_candidates)
            lru_key = lru_candidates[1]
            delete!(tracker.under_evaluation, lru_key)
            delete!(tracker.lru_positions, lru_key)
            tracker.pairs_evicted_lru += 1
        end
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
        if nobs(entry.ratio_stats_acc) >= min_valid_samples && entry.current_cv <= final_cv_threshold
            if !haskey(tracker.low_cv_confirmed, entry.pair_key)
                push!(candidates, entry.pair_key)
            end
        end
    end
    return sort(collect(candidates))
end

function get_candidate_pairs(tracker::CorrelationTracker, final_threshold::Float64, min_valid_samples::Int)
    candidates = CorrelationEntry[]
    for (pair_key, corr_acc) in tracker.high_confidence
        if nobs(corr_acc) >= min_valid_samples && abs(cor(corr_acc)[1, 2]) >= final_threshold
            push!(candidates, CorrelationEntry(pair_key, corr_acc, abs(cor(corr_acc)[1, 2]), 0, 1.0))
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

# --- Main `streaming_filter` Function ---

function streaming_filter(
    complexes::Vector{MetabolicComplex},
    balanced_complexes::Set{Int},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    warmup::Matrix{Float64},
    constraints::C.ConstraintTree,
    concordance_tracker::ConcordanceTracker;
    sample_size::Int=1000,
    n_iterations::Int=32,
    n_warmup_points::Int=500,
    correlation_threshold::Float64=0.95,
    min_valid_samples::Int=0,
    max_correlation_pairs::Int=100_000_000,
    max_cv_pairs::Int=1_000_000,
    early_correlation_threshold::Float64=0.8,
    workers=workers,
    seed::Union{Int,Nothing}=42,
    filter::Vector{Symbol}=[:cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-9,
)
    rng = seed === nothing ? StableRNG() : StableRNG(seed)
    n_threads = Threads.nthreads()

    active_complexes = [c for c in complexes if !(Int(c.id) in balanced_complexes)]
    original_indices = concordance_tracker.id_to_idx
    all_pairs = [(c1.id, c2.id) for (i, c1) in enumerate(active_complexes) for (j, c2) in enumerate(active_complexes) if i < j && !((get(original_indices, c1.id, 0), get(original_indices, c2.id, 0)) in trivial_pairs)]

    @info "Generating $sample_size samples..."
    start_vars_indices = rand(rng, 1:size(warmup, 1), min(sample_size, size(warmup, 1)))
    start_variables = warmup[start_vars_indices, :]
    @debug "Size of start variables: $(size(start_variables, 1)) x $(size(start_variables, 2))"
    @debug "Type of start variables: $(typeof(start_variables))"
    @debug "First 5 start variables: $(start_variables[1:5, :])"

    all_samples_tree = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=start_variables,
        seed=rand(rng, UInt64),
        n_chains=1,
        collect_iterations=[n_iterations]
    )

    @debug "First 5 in the samples Tree: $(first(all_samples_tree, 5))"
    samples_dict = Dict{Symbol,Vector{Float64}}(
        id => vec(samples) for (id, samples) in all_samples_tree if haskey(original_indices, id)
    )

    if !isempty(samples_dict)
        @info "Length of one sample: $(length(first(values(samples_dict))))"
    end
    if isempty(all_samples_tree)
        @warn "Sample generation failed. Aborting."
        return PairPriority[]
    end

    samples_dict = Dict{Symbol,Vector{Float64}}(id => vec(samples) for (id, samples) in all_samples_tree if haskey(original_indices, id))

    if isempty(samples_dict)
        @warn "No valid activity samples found after filtering. Aborting."
        return PairPriority[]
    end

    @info "Sample generation complete. Total samples per activity: $(length(first(values(samples_dict))))."

    pairs_to_test = all_pairs
    final_cv_tracker = nothing

    if :cv in filter
        @info "Performing CV filtering with $n_threads threads..."
        thread_local_cv_trackers = [CVTracker(max_cv_pairs, cv_threshold) for _ = 1:n_threads]
        filter_config_cv = (cv_threshold=cv_threshold, cv_epsilon=cv_epsilon)

        progress_cv = Progress(length(pairs_to_test), desc="Processing pairs for CV: ")
        @threads for (ci_id, cj_id) in pairs_to_test
            local_tracker = thread_local_cv_trackers[Threads.threadid()]

            if !haskey(samples_dict, ci_id) || !haskey(samples_dict, cj_id)
                continue
            end

            x_values = samples_dict[ci_id]
            y_values = samples_dict[cj_id]

            # Create a fresh stat for each pair - they are very cheap
            ratio_stat = Variance()
            for k in eachindex(x_values, y_values)
                ratio = (x_values[k] + filter_config_cv.cv_epsilon) / (y_values[k] + filter_config_cv.cv_epsilon)
                if isfinite(ratio)
                    fit!(ratio_stat, ratio)
                end
            end

            if nobs(ratio_stat) > 1
                update_cv_tracker!(local_tracker, (ci_id, cj_id), ratio_stat; cv_epsilon=filter_config_cv.cv_epsilon)
            end
            ProgressMeter.next!(progress_cv)
        end
        ProgressMeter.finish!(progress_cv)

        @info "Merging thread-local CV trackers..."
        final_cv_tracker = thread_local_cv_trackers[1]
        for i = 2:n_threads
            merge_trackers!(final_cv_tracker, thread_local_cv_trackers[i])
        end

        cv_candidate_keys = get_cv_candidate_pairs(final_cv_tracker, cv_threshold, min_valid_samples)
        @info "CV filtering complete. Found $(length(cv_candidate_keys)) candidate pairs."
        pairs_to_test = cv_candidate_keys
    end

    if !(:cor in filter)
        priorities = PairPriority[]
        if !isnothing(final_cv_tracker)
            for pair_key in pairs_to_test
                c1_id, c2_id = pair_key
                c1_idx, c2_idx = get(original_indices, c1_id, 0), get(original_indices, c2_id, 0)
                if c1_idx == 0 || c2_idx == 0
                    continue
                end
                stats = get(final_cv_tracker.low_cv_confirmed, pair_key, nothing)
                if isnothing(stats)
                    continue
                end
                directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)
                push!(priorities, PairPriority(c1_idx, c2_idx, directions, -1.0, nobs(stats), true))
            end
        end
        return priorities
    end

    @info "Performing correlation analysis on $(length(pairs_to_test)) pairs with $n_threads threads."
    thread_local_cor_trackers = [CorrelationTracker(max_correlation_pairs, early_correlation_threshold) for _ = 1:n_threads]
    filter_config_cor = (correlation_threshold=correlation_threshold,)

    progress_cor = Progress(length(pairs_to_test), desc="Processing pairs for Correlation: ")
    @threads for (ci_id, cj_id) in pairs_to_test
        local_tracker = thread_local_cor_trackers[Threads.threadid()]

        if !haskey(samples_dict, ci_id) || !haskey(samples_dict, cj_id)
            continue
        end

        x_values = samples_dict[ci_id]
        y_values = samples_dict[cj_id]

        corr_stat = CovMatrix()
        fit!(corr_stat, zip(x_values, y_values))

        if nobs(corr_stat) > 1
            update_correlation_tracker!(local_tracker, (ci_id, cj_id), corr_stat, filter_config_cor.correlation_threshold)
        end
        ProgressMeter.next!(progress_cor)
    end
    ProgressMeter.finish!(progress_cor)

    @info "Merging thread-local correlation trackers..."
    final_cor_tracker = thread_local_cor_trackers[1]
    for i = 2:n_threads
        merge_trackers!(final_cor_tracker, thread_local_cor_trackers[i])
    end

    @info "Extracting final candidates from the merged correlation tracker."
    candidates = get_candidate_pairs(final_cor_tracker, correlation_threshold, min_valid_samples)
    @info "Found $(length(candidates)) candidate pairs meeting the criteria."

    priorities = PairPriority[]
    for entry in candidates
        c1_id, c2_id = entry.pair_key
        c1_idx, c2_idx = get(original_indices, c1_id, 0), get(original_indices, c2_id, 0)
        if c1_idx == 0 || c2_idx == 0
            continue
        end
        directions = determine_directions(c1_idx, c2_idx, positive_complexes, negative_complexes, unrestricted_complexes)
        is_high_conf = haskey(final_cor_tracker.high_confidence, entry.pair_key)
        push!(priorities, PairPriority(c1_idx, c2_idx, directions, abs(entry.current_correlation), nobs(entry.correlation_acc), is_high_conf))
    end

    sort!(priorities, by=p -> p.correlation, rev=true)
    return priorities
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