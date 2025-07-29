"""
filter.jl - Ultra memory-efficient CV filtering using OnlineStats.jl

Streaming three-stage pipeline focused purely on CV computation.
"""

using Base.Threads
using OnlineStats

# --- Configuration ---
struct FilterConfig
    # Stage 2 parameters
    coarse_sample_size::Int
    coarse_cv_threshold::Float64

    # Stage 3 parameters
    cv_threshold::Float64
    cv_epsilon::Float64
    min_valid_samples::Int

    # Performance
    use_threads::Bool
    chunk_size::Int

    # Memory management
    max_pairs_in_memory::Int
end

function FilterConfig(;
    coarse_sample_size::Int=20,
    coarse_cv_threshold::Float64=0.1,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-15,
    min_valid_samples::Int=50,
    use_threads::Bool=true,
    chunk_size::Int=10_000,
    max_pairs_in_memory::Int=1_000_000,
)
    FilterConfig(
        coarse_sample_size,
        coarse_cv_threshold,
        cv_threshold,
        cv_epsilon,
        min_valid_samples,
        use_threads,
        chunk_size,
        max_pairs_in_memory,
    )
end

# --- Helper functions for CV ---
@inline function compute_cv(variance_stat::Variance, epsilon::Float64=1e-8)
    n = nobs(variance_stat)
    m = mean(variance_stat)
    s = std(variance_stat)

    n < 2 && return Inf
    abs(m) < epsilon && return Inf

    return s / abs(m)
end

# Optimized function to compute ratios and fit to variance stat
@inline function compute_ratios_batch!(
    variance_stat::Variance, 
    c1_samples::Vector{Float64}, 
    c2_samples::Vector{Float64},
    start_idx::Int, 
    end_idx::Int, 
    epsilon::Float64
)
    @inbounds for k in start_idx:end_idx
        ratio = (c1_samples[k] + epsilon) / (c2_samples[k] + epsilon)
        isfinite(ratio) && fit!(variance_stat, ratio)
    end
    return variance_stat
end

# --- Priority structure ---
struct PairPriority
    c1_idx::Int
    c2_idx::Int
    directions::Set{Symbol}
    cv::Float64
    n_samples::Int
end

# --- Main filtering function ---
"""
$(TYPEDSIGNATURES)

Ultra-efficient three-stage streaming pipeline.
"""
function streaming_filter(
    complexes::Vector,
    balanced_complexes::Set{Int},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    samples_tree::C.Tree{Vector{Float64}},
    concordance_tracker::ConcordanceTracker;
    config::FilterConfig=FilterConfig()
)
    @info "Starting memory-efficient CV filtering pipeline"

    # Stage 1: Count pairs without materializing
    stage1_start = time()
    n_pairs = count_valid_pairs(complexes, balanced_complexes, trivial_pairs, concordance_tracker)
    @info "Stage 1: Counted pairs" n_pairs time_sec = round(time() - stage1_start, digits=2)

    # Stage 2 & 3: Stream processing
    stage23_start = time()
    priorities = if n_pairs <= config.max_pairs_in_memory
        # Small enough to process directly
        @info "Processing all pairs directly (small dataset)"
        process_all_pairs(
            complexes, balanced_complexes, trivial_pairs, samples_tree,
            concordance_tracker, positive_complexes, negative_complexes,
            unrestricted_complexes, config
        )
    else
        # Stream in chunks
        @info "Processing in streaming chunks (large dataset)" chunk_size = config.chunk_size
        process_streaming_chunks(
            complexes, balanced_complexes, trivial_pairs, samples_tree,
            concordance_tracker, positive_complexes, negative_complexes,
            unrestricted_complexes, config
        )
    end
    stage23_time = time() - stage23_start

    @info "Filtering complete" total_candidates = length(priorities) time_sec = round(stage23_time, digits=2)

    return priorities
end

# --- Stage 1: Counting without allocation ---
function count_valid_pairs(complexes, balanced, trivial, concordance_tracker)
    count = 0
    n = length(complexes)
    @inbounds for i in 1:n
        # Cache balanced check for i to avoid repeated lookups
        i_balanced = i in balanced
        if i_balanced
            continue  # Skip entire inner loop if i is balanced
        end
        
        for j in (i+1):n
            if should_test_pair_indices_fast(i, j, balanced, trivial, concordance_tracker, i_balanced)
                count += 1
            end
        end
    end
    return count
end

# Optimized version that takes pre-computed i_balanced
@inline function should_test_pair_indices_fast(i::Int, j::Int, balanced, trivial, concordance_tracker, i_balanced::Bool)
    # i_balanced already checked outside
    if j in balanced
        return false
    end
    (i, j) in trivial && return false
    are_concordant(concordance_tracker, i, j) && return false
    is_non_concordant(concordance_tracker, i, j) && return false
    return true
end

@inline function should_test_pair_indices(i::Int, j::Int, balanced, trivial, concordance_tracker)
    if i in balanced || j in balanced
        return false
    end
    (i, j) in trivial && return false
    are_concordant(concordance_tracker, i, j) && return false
    is_non_concordant(concordance_tracker, i, j) && return false
    return true
end

# --- Direct processing for smaller datasets ---
function process_all_pairs(
    complexes, balanced, trivial, samples_tree,
    concordance_tracker, positive, negative, unrestricted, config
)
    # Generator for valid pairs
    valid_pairs = (
        (i, j) for i in 1:length(complexes)
        for j in (i+1):length(complexes)
        if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker)
    )

    # Process with streaming statistics
    priorities = process_pair_stream(
        valid_pairs, samples_tree, concordance_tracker,
        positive, negative, unrestricted, config
    )

    return priorities
end

# --- Chunked streaming for huge datasets ---
function process_streaming_chunks(
    complexes, balanced, trivial, samples_tree,
    concordance_tracker, positive, negative, unrestricted, config
)
    # Pre-allocate with type annotation for better performance
    all_priorities = Vector{PairPriority}()
    sizehint!(all_priorities, config.max_pairs_in_memory)
    chunk_pairs = Vector{Tuple{Int,Int}}()
    sizehint!(chunk_pairs, config.chunk_size)

    n = length(complexes)
    @inbounds for i in 1:n
        # Cache balanced check for i to avoid repeated lookups
        i_balanced = i in balanced
        if i_balanced
            continue  # Skip entire inner loop if i is balanced
        end
        
        for j in (i+1):n
            if should_test_pair_indices_fast(i, j, balanced, trivial, concordance_tracker, i_balanced)
                push!(chunk_pairs, (i, j))

                if length(chunk_pairs) >= config.chunk_size
                    # Process chunk
                    chunk_priorities = process_pair_stream(
                        chunk_pairs, samples_tree, concordance_tracker,
                        positive, negative, unrestricted, config
                    )
                    append!(all_priorities, chunk_priorities)

                    # Clear chunk
                    empty!(chunk_pairs)

                    # Keep only top candidates to save memory
                    if length(all_priorities) > config.max_pairs_in_memory
                        sort!(all_priorities, by=p -> p.cv)
                        resize!(all_priorities, config.max_pairs_in_memory ÷ 2)
                        @info "Memory pruning" kept_best = length(all_priorities)
                    end
                end
            end
        end
    end

    # Process final chunk
    if !isempty(chunk_pairs)
        chunk_priorities = process_pair_stream(
            chunk_pairs, samples_tree, concordance_tracker,
            positive, negative, unrestricted, config
        )
        append!(all_priorities, chunk_priorities)
    end

    # Final sort
    sort!(all_priorities, by=p -> p.cv)
    return all_priorities
end

# --- Core streaming processor ---
function process_pair_stream(
    pairs, samples_tree, concordance_tracker,
    positive, negative, unrestricted, config
)
    idx_to_id = concordance_tracker.idx_to_id

    if config.use_threads
        @info "Using parallel processing with threads"
        process_pairs_parallel(pairs, samples_tree, idx_to_id, positive, negative, unrestricted, config)
    else
        @info "Using serial processing"
        process_pairs_serial(pairs, samples_tree, idx_to_id, positive, negative, unrestricted, config)
    end
end

# --- Serial processing ---
function process_pairs_serial(
    pairs, samples_tree, idx_to_id,
    positive, negative, unrestricted, config
)
    # Pre-allocate with type annotation
    priorities = Vector{PairPriority}()
    sizehint!(priorities, 1000)  # Conservative estimate
    pair_count = 0
    stage2_passed = 0
    stage3_passed = 0

    for (i, j) in pairs
        pair_count += 1

        # Cache lookups to avoid repeated dictionary access
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        # Early exit if samples are missing
        if isnothing(c1_samples) || isnothing(c2_samples)
            continue
        end
        
        # Cache sample lengths to avoid repeated calls
        c1_len, c2_len = length(c1_samples), length(c2_samples)

        # Stage 2: Coarse filter - use cached lengths
        n_coarse = min(config.coarse_sample_size, c1_len, c2_len)
        n_coarse < 2 && continue

        ratio_stat = Variance()
        compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, 1, n_coarse, config.cv_epsilon)

        cv_coarse = compute_cv(ratio_stat, config.cv_epsilon)
        cv_coarse > config.coarse_cv_threshold && continue
        stage2_passed += 1

        # Stage 3: Full analysis - reuse and continue with same stat, use cached lengths
        max_samples = min(c1_len, c2_len)
        if max_samples > n_coarse
            compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, n_coarse + 1, max_samples, config.cv_epsilon)
        end

        n_samples = nobs(ratio_stat)
        n_samples < config.min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, config.cv_epsilon)
        cv_full > config.cv_threshold && continue
        stage3_passed += 1

        # Passed all filters
        directions = determine_directions(j, positive, negative)
        push!(priorities, PairPriority(i, j, directions, cv_full, n_samples))

        if pair_count % 50_000 == 0
            @info "Processing progress" pairs_tested = pair_count coarse_passed = stage2_passed cv_passed = stage3_passed candidates = length(priorities)
        end
    end

    @info "Serial processing complete" pairs_tested = pair_count coarse_passed = stage2_passed cv_passed = stage3_passed final_candidates = length(priorities)
    return priorities
end

# --- Parallel processing ---
function process_pairs_parallel(
    pairs, samples_tree, idx_to_id,
    positive, negative, unrestricted, config
)
    # Convert to vector only if not already a vector, avoiding copy when possible
    pairs_vec = if pairs isa AbstractVector
        pairs
    else
        collect(pairs)
    end
    n_threads = Threads.nthreads()

    # Pre-allocate thread-local results with type annotation and size hints
    thread_results = Vector{Vector{PairPriority}}(undef, n_threads)
    for i in 1:n_threads
        thread_results[i] = Vector{PairPriority}()
        sizehint!(thread_results[i], length(pairs_vec) ÷ n_threads + 100)
    end

    # Process in parallel
    @threads for idx in eachindex(pairs_vec)
        tid = Threads.threadid()
        i, j = pairs_vec[idx]

        # Cache lookups and get samples
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        # Early exit if samples are missing
        if isnothing(c1_samples) || isnothing(c2_samples)
            continue
        end
        
        # Cache sample lengths to avoid repeated calls
        c1_len, c2_len = length(c1_samples), length(c2_samples)

        # Coarse filter - use cached lengths
        n_coarse = min(config.coarse_sample_size, c1_len, c2_len)
        n_coarse < 2 && continue

        ratio_stat = Variance()
        compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, 1, n_coarse, config.cv_epsilon)

        cv_coarse = compute_cv(ratio_stat, config.cv_epsilon)
        cv_coarse > config.coarse_cv_threshold && continue

        # Full analysis - use cached lengths
        max_samples = min(c1_len, c2_len)
        if max_samples > n_coarse
            compute_ratios_batch!(ratio_stat, c1_samples, c2_samples, n_coarse + 1, max_samples, config.cv_epsilon)
        end

        n_samples = nobs(ratio_stat)
        n_samples < config.min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, config.cv_epsilon)
        cv_full > config.cv_threshold && continue

        directions = determine_directions(j, positive, negative)
        push!(thread_results[tid], PairPriority(i, j, directions, cv_full, n_samples))
    end

    # Merge thread results
    return vcat(thread_results...)
end

@inline function determine_directions(c2_idx::Int,
    positive::Set{Int}, negative::Set{Int}
)::Set{Symbol}
    # Pre-allocate direction set with capacity hint
    directions = Set{Symbol}()
    sizehint!(directions, 2)

    # Check constraints on both complexes to determine valid directions
    c2_positive = c2_idx in positive
    c2_negative = c2_idx in negative

    # Positive direction test: set c2 = +1
    # Only feasible if c2 can achieve positive values (not constrained to negative only)
    if !c2_negative
        push!(directions, :positive)
    end

    # Negative direction test: set c2 = -1  
    # Only feasible if c2 can achieve negative values (not constrained to positive only)
    if !c2_positive
        push!(directions, :negative)
    end

    return directions
end