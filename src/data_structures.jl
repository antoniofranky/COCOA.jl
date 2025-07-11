"""
Data structures for COCOA - Core types, buffers, storage, and tracking functionality.

This module contains:
- Memory management buffers (AnalysisBuffers)
- Core data structures (Complex, SharedSparseMatrix, SparseIncidenceMatrix)
- Sample storage abstractions (SampleStorage, MemoryStreamStorage, DiskStreamStorage)
- Concordance tracking (ConcordanceTracker, SharedConcordanceTracker)
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
Internal representation of a complex with memory-efficient storage.
"""
struct Complex
    id::Symbol
    metabolite_indices::Vector{Int32}
    stoichiometry::Vector{Float32}
    hash::UInt64

    function Complex(id::Symbol, met_idxs::Vector{<:Integer}, stoich::Vector{<:Real})
        # Ensure canonical ordering
        perm = sortperm(met_idxs)
        sorted_idxs = Int32.(met_idxs[perm])
        sorted_stoich = Float32.(stoich[perm])
        h = hash((sorted_idxs, sorted_stoich))
        new(id, sorted_idxs, sorted_stoich, h)
    end
end

"""
Shared sparse matrix structure optimized for single-node parallelism.
All processes share the same memory - no duplication!
"""
struct SharedSparseMatrix
    m::Int
    n::Int
    colptr::SharedArray{Int32,1}
    rowval::SharedArray{Int32,1}
    nzval::SharedArray{Float32,1}

    function SharedSparseMatrix(A::SparseMatrixCSC)
        m, n = size(A)

        # Create shared arrays
        colptr = SharedArray{Int32}(length(A.colptr))
        rowval = SharedArray{Int32}(length(A.rowval))
        nzval = SharedArray{Float32}(length(A.nzval))

        # Fill shared arrays (only on process 1)
        if myid() == 1
            colptr[:] = Int32.(A.colptr)
            rowval[:] = Int32.(A.rowval)
            nzval[:] = Float32.(A.nzval)
        end

        # Wait for all processes to see the data
        @sync @distributed for p in workers()
            nothing
        end

        new(m, n, colptr, rowval, nzval)
    end
end

# Convert back to SparseMatrixCSC when needed
function SparseArrays.sparse(S::SharedSparseMatrix)
    SparseMatrixCSC(S.m, S.n,
        convert(Vector{Int}, S.colptr),
        convert(Vector{Int}, S.rowval),
        convert(Vector{Float64}, S.nzval))
end

"""
Memory-efficient storage for sparse complex-reaction incidence matrix.
"""
struct SparseIncidenceMatrix
    n_complexes::Int
    n_reactions::Int
    colptr::Vector{Int32}
    rowval::Vector{Int32}
    nzval::Vector{Float32}

    function SparseIncidenceMatrix(I::Vector{Int}, J::Vector{Int}, V::Vector{<:Real}, m::Int, n::Int)
        A = sparse(I, J, Float32.(V), m, n)
        new(m, n, Int32.(A.colptr), Int32.(A.rowval), Float32.(A.nzval))
    end
end

# Convert back to SparseMatrixCSC when needed
function SparseArrays.sparse(A::SparseIncidenceMatrix)
    SparseMatrixCSC(A.n_complexes, A.n_reactions,
        Int.(A.colptr), Int.(A.rowval), Float64.(A.nzval))
end

# Abstract storage interface
abstract type SampleStorage end

# Memory-based storage for medium models
mutable struct MemoryStreamStorage <: SampleStorage
    activity_matrix::Matrix{Float64}
    n_samples::Int

    function MemoryStreamStorage(n_complexes::Int, max_samples::Int)
        new(zeros(Float64, max_samples, n_complexes), 0)
    end
end

# Disk-based storage for ultra-large models (50K+)
mutable struct DiskStreamStorage <: SampleStorage
    temp_file::String
    n_complexes::Int
    n_samples::Int

    function DiskStreamStorage(n_complexes::Int, max_samples::Int)
        temp_file = tempname() * ".samples.h5"
        h5open(temp_file, "w") do file
            file["n_complexes"] = n_complexes
            file["max_samples"] = max_samples
            # Pre-create dataset with chunking for efficient streaming access
            chunk_size = min(1000, max_samples)
            create_dataset(file, "samples", Float64,
                ((max_samples, n_complexes), (chunk_size, n_complexes)))
        end
        new(temp_file, n_complexes, 0)
    end
end

# Storage interface implementations
function store_sample!(storage::MemoryStreamStorage, activities::Dict, active_complexes::Vector{Complex}, iter::Int)
    if storage.n_samples >= size(storage.activity_matrix, 1)
        return false # Storage full
    end

    storage.n_samples += 1

    for (i, c) in enumerate(active_complexes)
        if haskey(activities, c.id)
            storage.activity_matrix[storage.n_samples, i] = activities[c.id][iter]
        else
            storage.activity_matrix[storage.n_samples, i] = 0.0
        end
    end

    return true
end

function store_sample!(storage::DiskStreamStorage, activities::Dict, active_complexes::Vector{Complex}, iter::Int)
    h5open(storage.temp_file, "r+") do file
        storage.n_samples += 1
        sample_row = zeros(Float64, storage.n_complexes)

        for (i, c) in enumerate(active_complexes)
            if haskey(activities, c.id)
                sample_row[i] = activities[c.id][iter]
            end
        end

        # Write just this sample (efficient HDF5 slicing)
        file["samples"][storage.n_samples, :] = sample_row
    end

    return true
end

function get_sample(storage::MemoryStreamStorage, idx::Int)
    if idx > storage.n_samples
        return zeros(Float64, size(storage.activity_matrix, 2))
    end
    return @view storage.activity_matrix[idx, :]
end

function get_sample(storage::DiskStreamStorage, idx::Int)
    if idx > storage.n_samples
        return zeros(Float64, storage.n_complexes)
    end

    sample_row = zeros(Float64, storage.n_complexes)
    h5open(storage.temp_file, "r") do file
        sample_row = file["samples"][idx, :]
    end
    return sample_row
end

function close(storage::DiskStreamStorage)
    if isfile(storage.temp_file)
        rm(storage.temp_file)
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

"""
Shared concordance tracker that maintains consistency across processes.
"""
mutable struct SharedConcordanceTracker
    # Shared arrays for Union-Find - keep Int32 for memory efficiency in shared memory
    parent::SharedArray{Int32,1}
    rank::SharedArray{Int32,1}

    # Complex mappings (read-only after initialization) - use Int for interface
    id_to_idx::Dict{Symbol,Int}
    idx_to_id::Vector{Symbol}

    # Non-concordance tracking (requires synchronization)
    non_concordant_matrix::SharedArray{Bool,2}

    # Lock for thread-safe updates (on process 1)
    update_lock::ReentrantLock

    function SharedConcordanceTracker(complex_ids::Vector{Symbol})
        n = length(complex_ids)

        # Initialize shared arrays
        parent = SharedArray{Int32}(n)
        rank = SharedArray{Int32}(n)
        non_concordant_matrix = SharedArray{Bool}(n, n)

        # Initialize on process 1
        if myid() == 1
            parent[:] = 1:n
            rank[:] .= 0
            non_concordant_matrix[:] .= false
        end

        # Synchronize
        @sync @distributed for p in workers()
            nothing
        end

        # Create mappings (same on all processes) - use Int for interface
        id_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))
        idx_to_id = copy(complex_ids)

        new(parent, rank, id_to_idx, idx_to_id,
            non_concordant_matrix, ReentrantLock())
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

# Shared tracker operations
function find_set!(tracker::SharedConcordanceTracker, x::Int)
    # Convert to Int32 for internal storage access
    x32 = Int32(x)

    # Use atomic operations for thread safety
    parent_x = tracker.parent[x32]
    if parent_x != x32
        # Path compression with atomic update
        root = find_set!(tracker, Int(parent_x))
        tracker.parent[x32] = Int32(root)
        return root
    end
    return x
end

function union_sets!(tracker::SharedConcordanceTracker, x::Int, y::Int)
    # Only process 1 should modify
    if myid() != 1
        return
    end

    root_x = find_set!(tracker, x)
    root_y = find_set!(tracker, y)

    if root_x == root_y
        return root_x
    end

    # Convert to Int32 for internal access
    root_x32 = Int32(root_x)
    root_y32 = Int32(root_y)

    # Union by rank with atomic operations
    if tracker.rank[root_x32] < tracker.rank[root_y32]
        tracker.parent[root_x32] = root_y32
        return root_y
    elseif tracker.rank[root_x32] > tracker.rank[root_y32]
        tracker.parent[root_y32] = root_x32
        return root_x
    else
        tracker.parent[root_y32] = root_x32
        tracker.rank[root_x32] += 1
        return root_x
    end
end

# Generic operations that work for both trackers
function are_concordant(tracker::Union{ConcordanceTracker,SharedConcordanceTracker}, x::Int, y::Int)
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

# Non-concordance operations for shared tracker
function add_non_concordant!(tracker::SharedConcordanceTracker, x::Int, y::Int)
    if myid() == 1
        tracker.non_concordant_matrix[x, y] = true
        tracker.non_concordant_matrix[y, x] = true
    end
end

function is_non_concordant(tracker::SharedConcordanceTracker, x::Int, y::Int)
    # Direct check
    if tracker.non_concordant_matrix[x, y]
        return true
    end

    # Check module transitivity
    rep_x = find_set!(tracker, x)
    rep_y = find_set!(tracker, y)

    if rep_x == rep_y
        return false
    end

    # Check if any pair between modules is non-concordant
    n = length(tracker.parent)
    for i in 1:n
        if find_set!(tracker, i) == rep_x
            for j in 1:n
                if find_set!(tracker, j) == rep_y && tracker.non_concordant_matrix[i, j]
                    return true
                end
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

"""
Find trivially balanced complexes (containing metabolites that appear in only one complex).
"""
function find_trivially_balanced_complexes(
    complexes::Vector{Complex}
)::Set{Symbol}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Int,Vector{Int}}()

    for (cidx, complex) in enumerate(complexes)
        for met_idx in complex.metabolite_indices
            if !haskey(metabolite_participation, met_idx)
                metabolite_participation[met_idx] = Int[]
            end
            push!(metabolite_participation[met_idx], cidx)
        end
    end

    balanced_complexes = Set{Symbol}()

    # Find metabolites that appear in only one complex
    for (met_idx, complex_indices) in metabolite_participation
        if length(complex_indices) == 1
            complex_idx = complex_indices[1]
            push!(balanced_complexes, complexes[complex_idx].id)
        end
    end

    return balanced_complexes
end

"""
Find trivially concordant complexes based on shared metabolites.
Two complexes are trivially concordant if they share a metabolite that 
appears in exactly those two complexes.
"""
function find_trivially_concordant_pairs(complexes::Vector{Complex})::Set{Tuple{Int,Int}}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Int,Vector{Int}}()

    for (cidx, complex) in enumerate(complexes)
        for met_idx in complex.metabolite_indices
            if !haskey(metabolite_participation, met_idx)
                metabolite_participation[met_idx] = Int[]
            end
            push!(metabolite_participation[met_idx], cidx)
        end
    end

    concordant_pairs = Set{Tuple{Int,Int}}()

    # Find metabolites that appear in exactly two complexes
    for (met_idx, complex_indices) in metabolite_participation
        if length(complex_indices) == 2
            c1, c2 = complex_indices
            pair = c1 < c2 ? (c1, c2) : (c2, c1)
            push!(concordant_pairs, pair)
        end
    end

    return concordant_pairs
end