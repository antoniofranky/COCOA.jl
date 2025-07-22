"""
filter.jl - Ultra memory-efficient CV filtering using OnlineStats.jl

Streaming three-stage pipeline focused purely on CV computation.
"""

using Statistics
using Random
using StableRNGs
using COBREXA
using DocStringExtensions
using ProgressMeter
import ConstraintTrees as C
using Base.Threads
using OnlineStats

# --- Configuration ---
struct FilterConfig
    # Stage 2 parameters
    coarse_sample_count::Int
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
    coarse_sample_count::Int=20,
    coarse_cv_threshold::Float64=0.1,
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-8,
    min_valid_samples::Int=50,
    use_threads::Bool=true,
    chunk_size::Int=10_000,
    max_pairs_in_memory::Int=1_000_000,
)
    FilterConfig(
        coarse_sample_count,
        coarse_cv_threshold,
        cv_threshold,
        cv_epsilon,
        min_valid_samples,
        use_threads,
        chunk_size,
        max_pairs_in_memory,
    )
end

# --- Helper function for CV ---
function compute_cv(variance_stat::Variance, epsilon::Float64=1e-8)
    n = nobs(variance_stat)
    m = mean(variance_stat)
    s = std(variance_stat)

    n < 2 && return Inf
    abs(m) < epsilon && return Inf

    return s / abs(m)
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
    config::FilterConfig=FilterConfig(),
    seed::Union{Int,Nothing}=42,
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

    @info "Stages 2 & 3 complete" final_pairs = length(priorities) time_sec = round(stage23_time, digits=2)

    return priorities
end

# --- Stage 1: Counting without allocation ---
function count_valid_pairs(complexes, balanced, trivial, concordance_tracker)
    count = 0
    @inbounds for i in 1:length(complexes)
        for j in (i+1):length(complexes)
            if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker)
                count += 1
            end
        end
    end
    return count
end

@inline function should_test_pair_indices(i::Int, j::Int, balanced, trivial, concordance_tracker)
    i in balanced || j in balanced && return false
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
    all_priorities = PairPriority[]
    chunk_pairs = sizehint!(Vector{Tuple{Int,Int}}(), config.chunk_size)

    for i in 1:length(complexes)
        for j in (i+1):length(complexes)
            if should_test_pair_indices(i, j, balanced, trivial, concordance_tracker)
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
                        @info "Pruned candidates" kept = length(all_priorities)
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
    priorities = PairPriority[]
    pair_count = 0
    stage2_passed = 0
    stage3_passed = 0

    for (i, j) in pairs
        pair_count += 1

        # Get samples once
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        isnothing(c1_samples) || isnothing(c2_samples) && continue

        # Stage 2: Coarse filter
        n_coarse = min(config.coarse_sample_count, length(c1_samples), length(c2_samples))
        n_coarse < 2 && continue

        ratio_stat = Variance()
        @inbounds for k in 1:n_coarse
            ratio = (c1_samples[k] + config.cv_epsilon) / (c2_samples[k] + config.cv_epsilon)
            isfinite(ratio) && fit!(ratio_stat, ratio)
        end

        cv_coarse = compute_cv(ratio_stat, config.cv_epsilon)
        cv_coarse > config.coarse_cv_threshold && continue
        stage2_passed += 1

        # Stage 3: Full analysis - reuse and continue with same stat
        @inbounds for k in (n_coarse+1):min(length(c1_samples), length(c2_samples))
            ratio = (c1_samples[k] + config.cv_epsilon) / (c2_samples[k] + config.cv_epsilon)
            isfinite(ratio) && fit!(ratio_stat, ratio)
        end

        n_samples = nobs(ratio_stat)
        n_samples < config.min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, config.cv_epsilon)
        cv_full > config.cv_threshold && continue
        stage3_passed += 1

        # Passed all filters
        directions = determine_directions(i, j, positive, negative, unrestricted)
        push!(priorities, PairPriority(i, j, directions, cv_full, n_samples))

        if pair_count % 50_000 == 0
            @info "Processing progress" pairs_processed = pair_count stage2_passed = stage2_passed stage3_passed = stage3_passed final_candidates = length(priorities)
        end
    end

    @info "Serial processing complete" total_pairs = pair_count stage2_passed = stage2_passed stage3_passed = stage3_passed final_pairs = length(priorities)
    @info "Stage 2 (coarse filter) complete" candidates_after_coarse = stage2_passed
    @info "Stage 3 (full analysis) complete" final_pairs = stage3_passed
    return priorities
end

# --- Parallel processing ---
function process_pairs_parallel(
    pairs, samples_tree, idx_to_id,
    positive, negative, unrestricted, config
)
    # Collect to vector for parallel indexing
    pairs_vec = collect(pairs)
    n_threads = Threads.nthreads()

    # Thread-local results
    thread_results = [PairPriority[] for _ in 1:n_threads]

    # Process in parallel
    @threads for idx in eachindex(pairs_vec)
        tid = Threads.threadid()
        i, j = pairs_vec[idx]

        # Get samples
        c1_id, c2_id = idx_to_id[i], idx_to_id[j]
        c1_samples = samples_tree[c1_id]
        c2_samples = samples_tree[c2_id]

        isnothing(c1_samples) || isnothing(c2_samples) && continue

        # Coarse filter
        n_coarse = min(config.coarse_sample_count, length(c1_samples), length(c2_samples))
        n_coarse < 2 && continue

        ratio_stat = Variance()
        @inbounds for k in 1:n_coarse
            ratio = (c1_samples[k] + config.cv_epsilon) / (c2_samples[k] + config.cv_epsilon)
            isfinite(ratio) && fit!(ratio_stat, ratio)
        end

        cv_coarse = compute_cv(ratio_stat, config.cv_epsilon)
        cv_coarse > config.coarse_cv_threshold && continue

        # Full analysis
        @inbounds for k in (n_coarse+1):min(length(c1_samples), length(c2_samples))
            ratio = (c1_samples[k] + config.cv_epsilon) / (c2_samples[k] + config.cv_epsilon)
            isfinite(ratio) && fit!(ratio_stat, ratio)
        end

        n_samples = nobs(ratio_stat)
        n_samples < config.min_valid_samples && continue

        cv_full = compute_cv(ratio_stat, config.cv_epsilon)
        cv_full > config.cv_threshold && continue

        directions = determine_directions(i, j, positive, negative, unrestricted)
        push!(thread_results[tid], PairPriority(i, j, directions, cv_full, n_samples))
    end

    # Merge thread results
    return vcat(thread_results...)
end

function determine_directions(
    c1_idx::Int, c2_idx::Int,
    positive::Set{Int}, negative::Set{Int}, unrestricted::Set{Int}
)::Set{Symbol}
    directions = Set{Symbol}()

    if c2_idx in positive
        push!(directions, :positive)
    elseif c2_idx in negative
        push!(directions, :negative)
    else
        push!(directions, :positive)
        push!(directions, :negative)
    end

    return directions
end