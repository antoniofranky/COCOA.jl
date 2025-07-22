"""
filter.jl - Memory-efficient correlation/CV filtering for COCOA

Three-stage filtering pipeline using indices for maximum performance:
1. Structural prefiltering (balanced, trivial, transitivity)
2. Coarse filtering with subset of samples
3. Full statistical analysis with global trackers
"""

using Statistics
using Random
using StableRNGs
using COBREXA
using JuMP
using DocStringExtensions
using ProgressMeter
import ConstraintTrees as C
using Base.Threads
using OnlineStats

# --- Configuration ---
struct FilterConfig
    # Stage 2 parameters
    coarse_sample_count::Int
    coarse_correlation_threshold::Float64

    # Stage 3 parameters  
    correlation_threshold::Float64
    cv_threshold::Float64
    cv_epsilon::Float64

    # Memory limits
    max_correlation_pairs::Int
    max_cv_pairs::Int

    # Performance
    use_batching::Bool
    batch_size::Int
    use_threads::Bool

    # Filter selection
    filter::Vector{Symbol}
end

# Default configuration
function FilterConfig(;
    coarse_sample_count::Int=20,
    coarse_correlation_threshold::Float64=0.5,
    correlation_threshold::Float64=0.95,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-8,
    max_correlation_pairs::Int=10_000_000,
    max_cv_pairs::Int=1_000_000,
    use_batching::Bool=false,
    batch_size::Int=1_000_000,
    use_threads::Bool=true,
    filter::Vector{Symbol}=[:cor]
)
    FilterConfig(
        coarse_sample_count,
        coarse_correlation_threshold,
        correlation_threshold,
        cv_threshold,
        cv_epsilon,
        max_correlation_pairs,
        max_cv_pairs,
        use_batching,
        batch_size,
        use_threads,
        filter
    )
end

# --- Tracker Structures (using indices) ---
struct CorrelationEntry
    pair_key::Tuple{Int,Int}
    correlation_acc::CovMatrix
    current_correlation::Float64
    last_updated::Int
    confidence_upper_bound::Float64
end

mutable struct CorrelationTracker
    high_confidence::Dict{Tuple{Int,Int},CovMatrix}
    under_evaluation::Dict{Tuple{Int,Int},CorrelationEntry}
    lru_positions::Dict{Tuple{Int,Int},Int}
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
            Dict{Tuple{Int,Int},CovMatrix}(),
            Dict{Tuple{Int,Int},CorrelationEntry}(),
            Dict{Tuple{Int,Int},Int}(),
            0, max_pairs, 30, promotion_threshold, rejection_confidence, 0, 0, 0
        )
    end
end

struct CVEntry
    pair_key::Tuple{Int,Int}
    ratio_stats_acc::Variance
    current_cv::Float64
    last_updated::Int
    confidence_lower_bound::Float64
end

mutable struct CVTracker
    low_cv_confirmed::Dict{Tuple{Int,Int},Variance}
    under_evaluation::Dict{Tuple{Int,Int},CVEntry}
    lru_positions::Dict{Tuple{Int,Int},Int}
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
            Dict{Tuple{Int,Int},Variance}(),
            Dict{Tuple{Int,Int},CVEntry}(),
            Dict{Tuple{Int,Int},Int}(),
            0, max_pairs, 50, cv_threshold, rejection_confidence, 0, 0, 0
        )
    end
end

# --- Thread-safe wrappers ---
mutable struct ThreadSafeCorrelationTracker
    tracker::CorrelationTracker
    lock::ReentrantLock
end

mutable struct ThreadSafeCVTracker
    tracker::CVTracker
    lock::ReentrantLock
end

# --- Tracker update functions ---
function update_lru!(tracker, pair_key::Tuple{Int,Int})
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
    z_critical = confidence_level >= 0.99 ? 2.576 : confidence_level >= 0.90 ? 1.645 : 1.96
    z_upper = z + z_critical * se_z
    r_upper = tanh(z_upper)
    return min(r_upper, 1.0)
end

function update_correlation_tracker!(
    tracker::CorrelationTracker,
    pair_key::Tuple{Int,Int},
    corr_acc::CovMatrix,
    final_threshold::Float64=0.95
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
        # Find LRU entry
        min_counter = minimum(values(tracker.lru_positions))
        lru_key = first(k for (k, v) in tracker.lru_positions if v == min_counter)
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
    z_critical = 1.645  # One-sided 95% confidence
    return max(0.0, cv - z_critical * se_cv)
end

function update_cv_tracker!(
    tracker::CVTracker,
    pair_key::Tuple{Int,Int},
    ratio_stats_acc::Variance;
    cv_epsilon::Float64=1e-12
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
        # Find LRU entry
        min_counter = minimum(values(tracker.lru_positions))
        lru_key = first(k for (k, v) in tracker.lru_positions if v == min_counter)
        delete!(tracker.under_evaluation, lru_key)
        delete!(tracker.lru_positions, lru_key)
        tracker.pairs_evicted_lru += 1
    end

    tracker.under_evaluation[pair_key] = entry
    update_lru!(tracker, pair_key)
    return true
end

# --- Priority structure ---
struct PairPriority
    c1_idx::Int
    c2_idx::Int
    directions::Set{Symbol}
    correlation::Float64
    n_samples::Int
    is_high_confidence::Bool
end

# --- Main filtering function ---
"""
$(TYPEDSIGNATURES)

Three-stage filtering pipeline for identifying candidate concordant pairs.
Works entirely with indices for maximum performance.
"""
function streaming_filter(
    complexes::Vector,
    balanced_complexes::Set{Int},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::ConstraintTrees.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    config::FilterConfig=FilterConfig(),
    seed::Union{Int,Nothing}=42
)
    rng = seed === nothing ? StableRNG() : StableRNG(seed)

    @info "Starting 3-stage filtering pipeline"

    # Stage 1: Structural prefiltering (returns indices)
    stage1_start = time()
    candidate_pairs = structural_prefilter(
        complexes,
        balanced_complexes,
        trivial_pairs,
        concordance_tracker
    )

    # Stage 2: Coarse filtering
    stage2_start = time()
    coarse_candidates = coarse_filter(
        candidate_pairs,
        samples_tree,
        concordance_tracker,
        config
    )
    stage2_time = time() - stage2_start
    @info "Stage 2 complete" candidates = length(coarse_candidates) time_sec = round(stage2_time, digits=2)

    # Stage 3: Full statistical analysis
    stage3_start = time()
    priorities = full_statistical_filter(
        coarse_candidates,
        samples_tree,
        positive_complexes,
        negative_complexes,
        unrestricted_complexes,
        concordance_tracker,
        config
    )
    stage3_time = time() - stage3_start
    @info "Stage 3 complete" final_pairs = length(priorities) time_sec = round(stage3_time, digits=2)

    return priorities
end

# --- Stage 1: Structural prefiltering ---
function structural_prefilter(complexes, balanced, trivial, concordance_tracker)
    # Returns an iterator of index pairs
    return (
        (i, j)
        for i in 1:length(complexes)
        for j in (i+1):length(complexes)
        if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker)
    )
end

function should_test_pair_indices(i::Int, j::Int, balanced, trivial, concordance_tracker)
    # Skip if both balanced
    i in balanced && j in balanced && return false

    # Skip trivially concordant
    (i, j) in trivial && return false

    # Skip if transitively concordant
    are_concordant(concordance_tracker, i, j) && return false

    # Skip if already known non-concordant
    is_non_concordant(concordance_tracker, i, j) && return false

    return true
end

# --- Stage 2: Coarse filtering ---
function coarse_filter(candidate_pairs, samples_tree, concordance_tracker, config)
    if config.use_threads
        coarse_filter_threaded(candidate_pairs, samples_tree, concordance_tracker, config)
    else
        @info "Using serial coarse filtering"
        coarse_filter_serial(candidate_pairs, samples_tree, concordance_tracker, config)
    end
end

function coarse_filter_serial(pairs_iter, samples_tree, concordance_tracker, config)
    survivors = Vector{Tuple{Int,Int,Float64}}()
    idx_to_id = concordance_tracker.idx_to_id

    count = 0
    for (i, j) in pairs_iter
        count += 1

        # Convert indices to IDs for tree access
        c1_id = idx_to_id[i]
        c2_id = idx_to_id[j]

        # Check if samples exist
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        isnothing(c1_samples) || isnothing(c2_samples) && continue

        # Use only first N samples
        n_samples = min(config.coarse_sample_count, length(c1_samples), length(c2_samples))
        c1_subset = @view c1_samples[1:n_samples]
        c2_subset = @view c2_samples[1:n_samples]

        # Quick correlation check
        quick_cor = cor(c1_subset, c2_subset)

        if abs(quick_cor) >= config.coarse_correlation_threshold
            push!(survivors, (i, j, quick_cor))
        end

        if count % 100_000 == 0
            @info "Coarse filter progress" pairs_processed = count survivors = length(survivors)
        end
    end

    @info "Coarse filtering complete" total_pairs = count survivors = length(survivors) survival_rate = round(100 * length(survivors) / count, digits=2)
    return survivors
end

function coarse_filter_threaded(pairs_iter, samples_tree, concordance_tracker, config)
    # Collect into chunks for parallel processing
    chunk_size = 10_000
    chunks = Vector{Vector{Tuple{Int,Int}}}()
    current_chunk = Vector{Tuple{Int,Int}}()

    for pair in pairs_iter
        push!(current_chunk, pair)
        if length(current_chunk) >= chunk_size
            push!(chunks, current_chunk)
            current_chunk = Vector{Tuple{Int,Int}}()
        end
    end
    !isempty(current_chunk) && push!(chunks, current_chunk)

    @info "Coarse filtering with threads" chunks = length(chunks) threads = Threads.nthreads()

    # Process chunks in parallel
    results = Vector{Vector{Tuple{Int,Int,Float64}}}(undef, length(chunks))
    idx_to_id = concordance_tracker.idx_to_id

    @threads for chunk_idx in eachindex(chunks)
        chunk_results = Vector{Tuple{Int,Int,Float64}}()
        for (i, j) in chunks[chunk_idx]
            c1_id = idx_to_id[i]
            c2_id = idx_to_id[j]

            c1_samples = samples_tree[c1_id]
            c2_samples = samples_tree[c2_id]

            isnothing(c1_samples) || isnothing(c2_samples) && continue

            n_samples = min(config.coarse_sample_count, length(c1_samples), length(c2_samples))
            c1_subset = @view c1_samples[1:n_samples]
            c2_subset = @view c2_samples[1:n_samples]

            quick_cor = cor(c1_subset, c2_subset)

            if abs(quick_cor) >= config.coarse_correlation_threshold
                push!(chunk_results, (i, j, quick_cor))
            end
        end
        results[chunk_idx] = chunk_results
    end

    # Combine results
    survivors = vcat(results...)
    total_pairs = sum(length(chunk) for chunk in chunks)
    @info "Coarse filtering complete" total_pairs survivors = length(survivors) survival_rate = round(100 * length(survivors) / total_pairs, digits=2)

    return survivors
end

# --- Stage 3: Full statistical analysis ---
function full_statistical_filter(
    coarse_candidates,
    samples_tree,
    positive_complexes,
    negative_complexes,
    unrestricted_complexes,
    concordance_tracker,
    config
)
    # Create thread-safe global trackers
    cor_tracker = :cor in config.filter ?
                  ThreadSafeCorrelationTracker(
        CorrelationTracker(config.max_correlation_pairs, config.correlation_threshold),
        ReentrantLock()
    ) : nothing

    cv_tracker = :cv in config.filter ?
                 ThreadSafeCVTracker(
        CVTracker(config.max_cv_pairs, config.cv_threshold),
        ReentrantLock()
    ) : nothing

    if config.use_batching
        process_in_batches(
            coarse_candidates, samples_tree, cor_tracker, cv_tracker,
            concordance_tracker, positive_complexes, negative_complexes,
            unrestricted_complexes, config
        )
    else
        process_all_at_once(
            coarse_candidates, samples_tree, cor_tracker, cv_tracker,
            concordance_tracker, positive_complexes, negative_complexes,
            unrestricted_complexes, config
        )
    end

    # Extract final results
    return extract_priorities(cor_tracker, cv_tracker, concordance_tracker,
        positive_complexes, negative_complexes,
        unrestricted_complexes, config)
end

# --- Batch processing ---
function process_all_at_once(
    candidates, samples_tree, cor_tracker, cv_tracker,
    concordance_tracker, positive_complexes, negative_complexes,
    unrestricted_complexes, config
)
    idx_to_id = concordance_tracker.idx_to_id
    progress = Progress(length(candidates), desc="Full analysis: ")

    if config.use_threads
        @threads for (i, j, coarse_cor) in candidates
            process_pair_indices(
                i, j, idx_to_id, samples_tree, cor_tracker, cv_tracker, config
            )
            ProgressMeter.next!(progress)
        end
    else
        for (i, j, coarse_cor) in candidates
            process_pair_indices(
                i, j, idx_to_id, samples_tree, cor_tracker, cv_tracker, config
            )
            ProgressMeter.next!(progress)
        end
    end

    ProgressMeter.finish!(progress)
end

function process_in_batches(
    candidates, samples_tree, cor_tracker, cv_tracker,
    concordance_tracker, positive_complexes, negative_complexes,
    unrestricted_complexes, config
)
    n_batches = ceil(Int, length(candidates) / config.batch_size)
    idx_to_id = concordance_tracker.idx_to_id

    for batch_idx in 1:n_batches
        start_idx = (batch_idx - 1) * config.batch_size + 1
        end_idx = min(batch_idx * config.batch_size, length(candidates))
        batch = @view candidates[start_idx:end_idx]

        progress = Progress(length(batch), desc="Batch $batch_idx/$n_batches: ")

        if config.use_threads
            @threads for (i, j, coarse_cor) in batch
                process_pair_indices(
                    i, j, idx_to_id, samples_tree, cor_tracker, cv_tracker, config
                )
                ProgressMeter.next!(progress)
            end
        else
            for (i, j, coarse_cor) in batch
                process_pair_indices(
                    i, j, idx_to_id, samples_tree, cor_tracker, cv_tracker, config
                )
                ProgressMeter.next!(progress)
            end
        end

        ProgressMeter.finish!(progress)

        # Light GC between batches
        GC.safepoint()
    end
end

# --- Process individual pair ---
function process_pair_indices(i::Int, j::Int, idx_to_id, samples_tree, cor_tracker, cv_tracker, config)
    # Convert to IDs for tree access
    c1_id = idx_to_id[i]
    c2_id = idx_to_id[j]

    c1_samples = samples_tree[c1_id]
    c2_samples = samples_tree[c2_id]

    isnothing(c1_samples) || isnothing(c2_samples) && return

    # Correlation analysis
    if !isnothing(cor_tracker)
        corr_stat = CovMatrix()
        fit!(corr_stat, zip(c1_samples, c2_samples))

        lock(cor_tracker.lock) do
            update_correlation_tracker!(
                cor_tracker.tracker, (i, j), corr_stat,
                config.correlation_threshold
            )
        end
    end

    # CV analysis
    if !isnothing(cv_tracker)
        ratio_stat = Variance()
        for (x, y) in zip(c1_samples, c2_samples)
            ratio = (x + config.cv_epsilon) / (y + config.cv_epsilon)
            isfinite(ratio) && fit!(ratio_stat, ratio)
        end

        if nobs(ratio_stat) > 1
            lock(cv_tracker.lock) do
                update_cv_tracker!(
                    cv_tracker.tracker, (i, j), ratio_stat;
                    cv_epsilon=config.cv_epsilon
                )
            end
        end
    end
end

# --- Extract final priorities ---
function extract_priorities(
    cor_tracker, cv_tracker, concordance_tracker,
    positive_complexes, negative_complexes, unrestricted_complexes, config
)
    priorities = PairPriority[]

    # Determine which pairs passed filters
    passing_pairs = Set{Tuple{Int,Int}}()

    # Collect correlation-passing pairs
    if !isnothing(cor_tracker)
        lock(cor_tracker.lock) do
            for (pair_key, corr_acc) in cor_tracker.tracker.high_confidence
                if abs(cor(corr_acc)[1, 2]) >= config.correlation_threshold
                    push!(passing_pairs, pair_key)
                end
            end
            for entry in values(cor_tracker.tracker.under_evaluation)
                if abs(entry.current_correlation) >= config.correlation_threshold
                    push!(passing_pairs, entry.pair_key)
                end
            end
        end
    end

    # Collect CV-passing pairs
    if !isnothing(cv_tracker)
        cv_pairs = Set{Tuple{Int,Int}}()
        lock(cv_tracker.lock) do
            for (pair_key, stats_acc) in cv_tracker.tracker.low_cv_confirmed
                cv = coefficient_of_variation(stats_acc)
                if !isinf(cv) && cv <= config.cv_threshold
                    push!(cv_pairs, pair_key)
                end
            end
            for entry in values(cv_tracker.tracker.under_evaluation)
                if entry.current_cv <= config.cv_threshold
                    push!(cv_pairs, entry.pair_key)
                end
            end
        end

        # If both filters active, take intersection
        if !isnothing(cor_tracker)
            passing_pairs = intersect(passing_pairs, cv_pairs)
        else
            passing_pairs = cv_pairs
        end
    end

    # Build priority list
    for (i, j) in passing_pairs
        # Get correlation value if available
        correlation = -1.0
        n_samples = 0
        is_high_conf = false

        if !isnothing(cor_tracker)
            lock(cor_tracker.lock) do
                if haskey(cor_tracker.tracker.high_confidence, (i, j))
                    acc = cor_tracker.tracker.high_confidence[(i, j)]
                    correlation = abs(cor(acc)[1, 2])
                    n_samples = nobs(acc)
                    is_high_conf = true
                else
                    for entry in values(cor_tracker.tracker.under_evaluation)
                        if entry.pair_key == (i, j)
                            correlation = abs(entry.current_correlation)
                            n_samples = nobs(entry.correlation_acc)
                            break
                        end
                    end
                end
            end
        end

        # Determine directions
        directions = determine_directions(
            i, j, positive_complexes, negative_complexes, unrestricted_complexes
        )

        push!(priorities, PairPriority(
            i, j, directions, correlation, n_samples, is_high_conf
        ))
    end

    # Sort by correlation (highest first)
    sort!(priorities, by=p -> p.correlation, rev=true)

    return priorities
end

function determine_directions(
    c1_idx::Int, c2_idx::Int,
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int}
)::Set{Symbol}
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