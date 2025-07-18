"""
Data structures for COCOA - Core types, buffers, storage, and tracking functionality.

This module contains:
- Memory management buffers (AnalysisBuffers)
- Core data structures (Complex, SharedSparseMatrix, SparseIncidenceMatrix)
- Sample storage abstractions (SampleStorage, MemoryStreamStorage, DiskStreamStorage)
- Concordance tracking (ConcordanceTracker)
"""

using SparseArrays
using Distributed
using SharedArrays
using JLD2
using Random
using StableRNGs

"""
Pre-allocated buffers for memory-efficient concordance analysis.
Reusing these buffers eliminates allocation overhead during analysis.
"""
mutable struct AnalysisBuffers
    # Activity computation buffers
    activities::Matrix{Float64}  # [n_active_complexes, batch_size]
    sample_buffer::Matrix{Float64}  # [batch_size, n_reactions]
    flux_buffer::Vector{Float64}  # [n_reactions]

    # Optimization result buffers
    optimization_results::Vector{Float64}  # For collecting results
    constraint_coeffs::Vector{Float64}  # For building constraints

    # Correlation computation buffers
    x_values::Vector{Float64}  # For correlation calculation
    y_values::Vector{Float64}  # For correlation calculation

    # Sparse matrix operation buffers
    sparse_indices::Vector{Int}  # For sparse operations
    sparse_values::Vector{Float64}  # For sparse operations

    function AnalysisBuffers(n_active_complexes::Int, n_reactions::Int, max_batch_size::Int)
        new(
            zeros(Float64, n_active_complexes, max_batch_size),
            zeros(Float64, max_batch_size, n_reactions),
            zeros(Float64, n_reactions),
            zeros(Float64, max_batch_size),
            zeros(Float64, n_reactions),
            zeros(Float64, max_batch_size),
            zeros(Float64, max_batch_size),
            Vector{Int}(undef, n_reactions),
            Vector{Float64}(undef, n_reactions)
        )
    end
end

"""
Resize buffers if needed to accommodate larger batch sizes.
"""
function ensure_buffer_capacity!(buffers::AnalysisBuffers, required_batch_size::Int, n_reactions::Int)
    current_batch_capacity = size(buffers.activities, 2)

    if required_batch_size > current_batch_capacity
        # Resize activity and sample buffers
        n_active = size(buffers.activities, 1)
        buffers.activities = zeros(Float64, n_active, required_batch_size)
        buffers.sample_buffer = zeros(Float64, required_batch_size, n_reactions)

        # Resize result buffers
        resize!(buffers.optimization_results, required_batch_size)
        resize!(buffers.x_values, required_batch_size)
        resize!(buffers.y_values, required_batch_size)
    end

    # Ensure reaction-sized buffers are adequate
    if length(buffers.flux_buffer) < n_reactions
        resize!(buffers.flux_buffer, n_reactions)
        resize!(buffers.constraint_coeffs, n_reactions)
        resize!(buffers.sparse_indices, n_reactions)
        resize!(buffers.sparse_values, n_reactions)
    end
end

"""
Tracks both concordant and non-concordant relationships between complexes with transitivity.

This structure efficiently tracks:
1. Concordant relationships using Union-Find
2. Non-concordant relationships with transitivity inference
3. Module membership caching for fast lookups
"""
mutable struct ConcordanceTracker
    # Union-Find for concordant relationships
    parent::Vector{Int}
    rank::Vector{Int}

    # Maps complex ID to index
    id_to_idx::Dict{Symbol,Int}
    idx_to_id::Vector{Symbol}

    # Non-concordance tracking
    non_concordant_pairs::Set{Tuple{Int,Int}}  # Direct non-concordant pairs
    non_concordant_modules::Dict{Tuple{Int,Int},Bool}  # Module relationship cache
    module_members_cache::Dict{Int,Vector{Int}}  # Module membership cache

    function ConcordanceTracker(complex_ids::Vector{Symbol})
        n = length(complex_ids)
        parent = collect(1:n)
        rank = zeros(Int, n)
        id_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))
        idx_to_id = copy(complex_ids)

        new(parent, rank, id_to_idx, idx_to_id,
            Set{Tuple{Int,Int}}(),
            Dict{Tuple{Int,Int},Bool}(),
            Dict{Int,Vector{Int}}())
    end
end


# Union-Find operations for regular tracker
function find_set!(tracker::ConcordanceTracker, x::Int)
    if tracker.parent[x] != x
        tracker.parent[x] = find_set!(tracker, tracker.parent[x])
    end
    return tracker.parent[x]
end

function union_sets!(tracker::ConcordanceTracker, x::Int, y::Int)
    root_x = find_set!(tracker, x)
    root_y = find_set!(tracker, y)

    if root_x == root_y
        return root_x
    end

    # Union by rank
    if tracker.rank[root_x] < tracker.rank[root_y]
        tracker.parent[root_x] = root_y
        new_root = root_y
    elseif tracker.rank[root_x] > tracker.rank[root_y]
        tracker.parent[root_y] = root_x
        new_root = root_x
    else
        tracker.parent[root_y] = root_x
        tracker.rank[root_x] += 1
        new_root = root_x
    end

    # Update caches when merging
    on_module_merged!(tracker, root_x, root_y, new_root)

    return new_root
end


# Generic operations
function are_concordant(tracker::ConcordanceTracker, x::Int, y::Int)
    return find_set!(tracker, x) == find_set!(tracker, y)
end

# Non-concordance operations for regular tracker
function add_non_concordant!(tracker::ConcordanceTracker, x::Int, y::Int)
    # Store in canonical order
    pair = x < y ? (x, y) : (y, x)
    if pair ∉ tracker.non_concordant_pairs
        push!(tracker.non_concordant_pairs, pair)

        # Update module relationship cache
        rep_x = find_set!(tracker, x)
        rep_y = find_set!(tracker, y)
        if rep_x != rep_y
            tracker.non_concordant_modules[(rep_x, rep_y)] = true
            tracker.non_concordant_modules[(rep_y, rep_x)] = true
        end
    end
end

function is_non_concordant(tracker::ConcordanceTracker, x::Int, y::Int)
    # Direct check
    pair = x < y ? (x, y) : (y, x)
    if pair in tracker.non_concordant_pairs
        return true
    end

    # Get representatives
    rep_x = find_set!(tracker, x)
    rep_y = find_set!(tracker, y)

    # Same module check
    if rep_x == rep_y
        return false
    end

    # Cached module relationship
    if get(tracker.non_concordant_modules, (rep_x, rep_y), false)
        return true
    end

    # Transitivity check
    ensure_module_cached!(tracker, rep_x)
    ensure_module_cached!(tracker, rep_y)

    for m1 in tracker.module_members_cache[rep_x]
        for m2 in tracker.module_members_cache[rep_y]
            pair_check = m1 < m2 ? (m1, m2) : (m2, m1)
            if pair_check in tracker.non_concordant_pairs
                tracker.non_concordant_modules[(rep_x, rep_y)] = true
                tracker.non_concordant_modules[(rep_y, rep_x)] = true
                return true
            end
        end
    end

    return false
end


function ensure_module_cached!(tracker::ConcordanceTracker, rep::Int)
    if !haskey(tracker.module_members_cache, rep)
        members = Int[]
        for i in 1:length(tracker.parent)
            if find_set!(tracker, i) == rep
                push!(members, i)
            end
        end
        tracker.module_members_cache[rep] = members
    end
end

function on_module_merged!(tracker::ConcordanceTracker, old_rep1::Int, old_rep2::Int, new_rep::Int)
    # Clear old caches
    delete!(tracker.module_members_cache, old_rep1)
    delete!(tracker.module_members_cache, old_rep2)

    # Update module relationships
    keys_to_remove = Tuple{Int,Int}[]
    new_relationships = Set{Tuple{Int,Int}}()

    for (key, _) in tracker.non_concordant_modules
        rep1, rep2 = key

        if rep1 == old_rep1 || rep1 == old_rep2
            push!(keys_to_remove, key)
            if rep2 != old_rep1 && rep2 != old_rep2
                push!(new_relationships, (new_rep, rep2))
            end
        elseif rep2 == old_rep1 || rep2 == old_rep2
            push!(keys_to_remove, key)
            push!(new_relationships, (rep1, new_rep))
        end
    end

    for key in keys_to_remove
        delete!(tracker.non_concordant_modules, key)
    end

    for (rep1, rep2) in new_relationships
        tracker.non_concordant_modules[(rep1, rep2)] = true
        tracker.non_concordant_modules[(rep2, rep1)] = true
    end
end

function clear_module_cache!(tracker::ConcordanceTracker)
    empty!(tracker.module_members_cache)
end