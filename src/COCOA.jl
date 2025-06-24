"""
    module COCOA

COnstraint-based COncordance Analysis for metabolic networks - optimized for large-scale models.

This module implements memory-efficient methods for identifying concordant complexes in metabolic
networks. Optimized for models with >30,000 complexes and reactions on HPC clusters with
SharedArrays support for single-node parallelism.
"""
module COCOA

using COBREXA
using AbstractFBCModels
using ConstraintTrees
using SparseArrays
using LinearAlgebra
using Distributed
using SharedArrays
using DataFrames
using Statistics
using Random
using JuMP
using DocStringExtensions
using ProgressMeter

include("preprocessing/ElementarySteps.jl")
include("preprocessing/ModelPreparation.jl")
using .ElementarySteps
using .ModelPreparation

# Configuration constants
const DEFAULT_BATCH_SIZE = 10
const DEFAULT_STAGE_SIZE = 100
const DEFAULT_SAMPLE_BATCH_SIZE = 100
const DEFAULT_CORRELATION_THRESHOLD = 0.95
const DEFAULT_TOLERANCE = 1e-9
const MAX_MEMORY_PER_WORKER = 2e9  # 2GB per worker default

# Re-export main functions
export concordance_constraints, concordance_analysis
export split_into_elementary_steps
export prepare_model_for_concordance
export ConcordanceConfig, ConcordanceTracker

"""
Configuration for concordance analysis with memory and performance optimizations.
"""
Base.@kwdef struct ConcordanceConfig
    tolerance::Float64 = DEFAULT_TOLERANCE
    correlation_threshold::Float64 = DEFAULT_CORRELATION_THRESHOLD
    sample_size::Int = 10
    sample_batch_size::Int = DEFAULT_SAMPLE_BATCH_SIZE
    concordance_batch_size::Int = DEFAULT_BATCH_SIZE
    stage_size::Int = DEFAULT_STAGE_SIZE
    max_memory_per_worker::Float64 = MAX_MEMORY_PER_WORKER
    reduce_model::Bool = false
    use_shared_arrays::Bool = true  # Enable SharedArrays by default
    min_size_for_sharing::Int = 1_000_000  # Share arrays > 1MB
    min_valid_samples::Int = 10  # Minimum samples for correlation
    seed::Union{Int,Nothing} = 42
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
    # Shared arrays for Union-Find
    parent::SharedArray{Int32,1}
    rank::SharedArray{Int32,1}

    # Complex mappings (read-only after initialization)
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

        # Create mappings (same on all processes)
        id_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))
        idx_to_id = copy(complex_ids)

        new(parent, rank, id_to_idx, idx_to_id,
            non_concordant_matrix, ReentrantLock())
    end
end

# Union-Find operations for regular tracker
function find_set!(tracker::ConcordanceTracker, x::Union{Int,Int32})
    if tracker.parent[x] != x
        tracker.parent[x] = find_set!(tracker, tracker.parent[x])
    end
    return tracker.parent[x]
end

function union_sets!(tracker::ConcordanceTracker, x::Union{Int,Int32}, y::Union{Int,Int32})
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
function find_set!(tracker::SharedConcordanceTracker, x::Union{Int,Int32})
    # Use atomic operations for thread safety
    parent_x = tracker.parent[x]
    if parent_x != x
        # Path compression with atomic update
        root = find_set!(tracker, parent_x)
        tracker.parent[x] = root
        return root
    end
    return x
end

function union_sets!(tracker::SharedConcordanceTracker, x::Union{Int,Int32}, y::Union{Int,Int32})
    # Only process 1 should modify
    if myid() != 1
        return
    end

    root_x = find_set!(tracker, x)
    root_y = find_set!(tracker, y)

    if root_x == root_y
        return root_x
    end

    # Union by rank with atomic operations
    if tracker.rank[root_x] < tracker.rank[root_y]
        tracker.parent[root_x] = root_y
        return root_y
    elseif tracker.rank[root_x] > tracker.rank[root_y]
        tracker.parent[root_y] = root_x
        return root_x
    else
        tracker.parent[root_y] = root_x
        tracker.rank[root_x] += 1
        return root_x
    end
end

# Generic operations that work for both trackers
function are_concordant(tracker::Union{ConcordanceTracker,SharedConcordanceTracker}, x::Int, y::Int)
    return find_set!(tracker, x) == find_set!(tracker, y)
end

# Non-concordance operations for regular tracker
function add_non_concordant!(tracker::ConcordanceTracker, x::Union{Int,Int32}, y::Union{Int,Int32})
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

function is_non_concordant(tracker::ConcordanceTracker, x::Union{Int,Int32}, y::Union{Int,Int32})
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
$(TYPEDSIGNATURES)

Find trivially balanced complexes (containing metabolites that appear in only one complex).
"""
function find_trivially_balanced_complexes(
    complexes::Vector{Complex}
)::Set{Symbol}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Int32,Vector{Int}}()

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
$(TYPEDSIGNATURES)

Find trivially concordant complexes based on shared metabolites.
Two complexes are trivially concordant if they share a metabolite that 
appears in exactly those two complexes.
"""
function find_trivially_concordant_pairs(complexes::Vector{Complex})::Set{Tuple{Int,Int}}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Int32,Vector{Int}}()

    for (cidx, complex) in enumerate(complexes)
        for met_idx in complex.metabolite_indices
            if !haskey(metabolite_participation, met_idx)
                metabolite_participation[met_idx] = Int[]
            end
            push!(metabolite_participation[met_idx], cidx)
        end
    end

    trivial_pairs = Set{Tuple{Int,Int}}()

    # Find metabolites that appear in exactly two complexes
    for (met_idx, complex_indices) in metabolite_participation
        if length(complex_indices) == 2
            c1_idx, c2_idx = complex_indices[1], complex_indices[2]
            # Store in canonical order
            pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            push!(trivial_pairs, pair)
        end
    end

    return trivial_pairs
end
"""
$(TYPEDSIGNATURES)

Extract complexes and build incidence matrix with optional shared memory.
"""
function extract_complexes_and_incidence(model::AbstractFBCModels.AbstractFBCModel;
    config::ConcordanceConfig=ConcordanceConfig())
    rxns = AbstractFBCModels.reactions(model)
    mets = AbstractFBCModels.metabolites(model)
    n_rxns = length(rxns)
    n_mets = length(mets)

    # Use Int32 for indices to save memory
    met_idx_map = Dict{String,Int32}(m => Int32(i) for (i, m) in enumerate(mets))

    complexes = Complex[]
    complex_dict = Dict{UInt64,Int32}()

    # Pre-allocate for incidence matrix construction
    I_rows = Int32[]
    J_cols = Int32[]
    V_vals = Float32[]

    sizehint!(I_rows, 2 * n_rxns)
    sizehint!(J_cols, 2 * n_rxns)
    sizehint!(V_vals, 2 * n_rxns)

    # Process reactions in batches
    batch_size = min(1000, n_rxns)
    for batch_start in 1:batch_size:n_rxns
        batch_end = min(batch_start + batch_size - 1, n_rxns)

        for ridx in batch_start:batch_end
            rxn = rxns[ridx]
            rxn_stoich = AbstractFBCModels.reaction_stoichiometry(model, rxn)

            # Separate substrates and products
            substrate_mets = Int32[]
            substrate_stoich = Float32[]
            product_mets = Int32[]
            product_stoich = Float32[]

            for (met, coeff) in rxn_stoich
                met_idx = met_idx_map[met]
                if coeff < 0
                    push!(substrate_mets, met_idx)
                    push!(substrate_stoich, -Float32(coeff))
                elseif coeff > 0
                    push!(product_mets, met_idx)
                    push!(product_stoich, Float32(coeff))
                end
            end

            # Process substrate complex
            if !isempty(substrate_mets)
                sub_complex = create_complex(substrate_mets, substrate_stoich, mets)
                complex_idx = get!(complex_dict, sub_complex.hash) do
                    push!(complexes, sub_complex)
                    Int32(length(complexes))
                end
                push!(I_rows, complex_idx)
                push!(J_cols, Int32(ridx))
                push!(V_vals, -1.0f0)
            end

            # Process product complex
            if !isempty(product_mets)
                prod_complex = create_complex(product_mets, product_stoich, mets)
                complex_idx = get!(complex_dict, prod_complex.hash) do
                    push!(complexes, prod_complex)
                    Int32(length(complexes))
                end
                push!(I_rows, complex_idx)
                push!(J_cols, Int32(ridx))
                push!(V_vals, 1.0f0)
            end
        end

        if batch_end < n_rxns
            GC.safepoint()
        end
    end

    # Build sparse incidence matrix
    A_sparse = sparse(Int.(I_rows), Int.(J_cols), V_vals, length(complexes), n_rxns)

    # Use shared memory if enabled and beneficial
    if config.use_shared_arrays && nworkers() > 0 &&
       (nnz(A_sparse) * (sizeof(Int32) + sizeof(Float32)) > config.min_size_for_sharing)
        A_matrix = SharedSparseMatrix(A_sparse)
    else
        A_matrix = SparseIncidenceMatrix(Int.(I_rows), Int.(J_cols), V_vals,
            length(complexes), n_rxns)
    end

    return complexes, A_matrix, complex_dict
end

"""
Create a complex from metabolite indices with memory-efficient ID generation.
"""
function create_complex(met_idxs::Vector{Int32}, stoich::Vector{Float32}, met_names::Vector{String})
    id_parts = IOBuffer()
    sorted_pairs = sort(collect(zip(met_idxs, stoich)))

    for (i, (idx, coeff)) in enumerate(sorted_pairs)
        i > 1 && write(id_parts, '+')
        if isinteger(coeff)
            write(id_parts, string(Int(coeff)), '_', met_names[idx])
        else
            write(id_parts, string(coeff), '_', met_names[idx])
        end
    end

    id = Symbol(String(take!(id_parts)))
    return Complex(id, Int32[p[1] for p in sorted_pairs], Float32[p[2] for p in sorted_pairs])
end

"""
$(TYPEDSIGNATURES)

Memory-efficient concordance constraints that work with large models.
"""
function concordance_constraints(
    model::AbstractFBCModels.AbstractFBCModel;
    modifications=Function[],
    interface=nothing,
    config::ConcordanceConfig=ConcordanceConfig()
)
    # Start with standard flux balance constraints
    constraints = COBREXA.flux_balance_constraints(model; interface)

    # Apply modifications
    for mod in modifications
        constraints = mod(constraints)
    end

    # Extract complexes with memory optimization
    complexes, A_matrix, complex_dict = extract_complexes_and_incidence(model; config)

    # Create complex activity variables
    rxn_ids = Symbol.(AbstractFBCModels.reactions(model))
    complex_activities = ConstraintTree()

    # Build activities in batches
    n_complexes = length(complexes)
    batch_size = min(1000, n_complexes)

    # Convert to appropriate sparse format
    A_sparse = if isa(A_matrix, SharedSparseMatrix)
        sparse(A_matrix)
    else
        sparse(A_matrix)
    end

    for batch_start in 1:batch_size:n_complexes
        batch_end = min(batch_start + batch_size - 1, n_complexes)

        for cidx in batch_start:batch_end
            complex = complexes[cidx]

            # Build activity as linear combination of fluxes
            activity_value = ConstraintTrees.LinearValue(idxs=Int[], weights=Float64[])

            # Get the row efficiently from the sparse matrix
            row = A_sparse[cidx, :]
            nz_indices = findnz(row)

            for (idx, val) in zip(nz_indices[1], nz_indices[2])
                if val != 0
                    activity_value += val * constraints.fluxes[rxn_ids[idx]].value
                end
            end

            complex_activities[complex.id] = ConstraintTrees.Constraint(
                value=activity_value,
                bound=ConstraintTrees.Between(-Inf, Inf)
            )
        end

        GC.safepoint()
    end

    # Add to constraint tree
    constraints[:complexes] = complex_activities
    constraints[:_cocoa_metadata] = ConstraintTree(
        :n_complexes => ConstraintTrees.Constraint(
            value=ConstraintTrees.LinearValue(idxs=Int[], weights=Float64[])
        )
    )

    return constraints
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

"""
$(TYPEDSIGNATURES)

Test concordance using CORRECTED Charnes-Cooper transformation.
"""
function test_concordance_optimized(
    model::JuMP.Model,
    c1_idx::Int,
    c2_idx::Int,
    A_row1::SparseVector{Float64,Int},
    A_row2::SparseVector{Float64,Int},
    direction::Symbol;
    tolerance::Float64=1e-9
)
    n_reactions = length(A_row1)

    # Pre-compute non-zero indices from both rows
    nz1 = findnz(A_row1)
    nz2 = findnz(A_row2)

    # Combine all non-zero indices from both rows
    all_nz_indices = union(Set(nz1[1]), Set(nz2[1]))

    # Clear any existing concordance constraints
    for cname in [:w, :t, :c2_act, :bounds_l, :bounds_u]
        if haskey(model, cname)
            delete(model, model[cname])
            unregister(model, cname)
        end
    end

    # Create transformed variables (only for non-zero indices)
    @variable(model, w[j=collect(all_nz_indices)])
    @variable(model, t)

    # Direction constraint on t
    if direction == :positive
        @constraint(model, t >= tolerance)
    else
        @constraint(model, t <= -tolerance)
    end

    # Extract bounds
    x = model[:x]

    # Complex c2 activity constraint
    c2_expr = AffExpr(0.0)
    for (idx, val) in zip(nz2[1], nz2[2])
        add_to_expression!(c2_expr, val, w[idx])
    end

    target = direction == :positive ? 1.0 : -1.0
    @constraint(model, c2_act, c2_expr == target)

    # Charnes-Cooper bounds constraints (ONLY for non-zero indices)
    if direction == :positive
        for j in all_nz_indices
            lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
            ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
            @constraint(model, w[j] - lb * t >= 0)
            @constraint(model, ub * t - w[j] >= 0)
        end
    else
        for j in all_nz_indices
            lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
            ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
            @constraint(model, w[j] - ub * t >= 0)
            @constraint(model, lb * t - w[j] >= 0)
        end
    end

    # Objective: optimize complex c1 activity
    c1_expr = AffExpr(0.0)
    for (idx, val) in zip(nz1[1], nz1[2])
        add_to_expression!(c1_expr, val, w[idx])
    end

    # Test min and max
    results = Dict{Symbol,Float64}()

    for (sense, key) in [(JuMP.MIN_SENSE, :min), (JuMP.MAX_SENSE, :max)]
        @objective(model, sense, c1_expr)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            results[key] = objective_value(model)
        else
            # Clean up properly based on variable/constraint type
            if haskey(model, :t)
                delete(model, model[:t])
                unregister(model, :t)
            end

            if haskey(model, :c2_act)
                delete(model, model[:c2_act])
                unregister(model, :c2_act)
            end

            # For array variables, just unregister the name
            for cname in [:w, :bounds_l, :bounds_u]
                if haskey(model, cname)
                    unregister(model, cname)
                end
            end
            return (false, nothing)
        end
    end

    # Check concordance
    is_concordant = isapprox(results[:min], results[:max]; atol=tolerance)
    lambda_value = is_concordant ? results[:min] : nothing

    # Clean up - handle arrays and scalars differently
    if haskey(model, :t)
        delete(model, model[:t])
        unregister(model, :t)
    end

    if haskey(model, :c2_act)
        delete(model, model[:c2_act])
        unregister(model, :c2_act)
    end

    # For array variables, just unregister the name
    for cname in [:w, :bounds_l, :bounds_u]
        if haskey(model, cname)
            unregister(model, cname)
        end
    end

    return (is_concordant, lambda_value)
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

Perform streaming correlation analysis with proper filtering:
- Include pairs with high correlation (above threshold)
- Include pairs with insufficient samples or failed correlation
- Filter out pairs with sufficient samples but low correlation
- Filter out trivially concordant pairs already identified
"""
function streaming_correlation_filter(
    model::AbstractFBCModels.AbstractFBCModel,
    complexes::Vector{Complex},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    balanced_complexes::Set{Symbol},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}};
    optimizer,
    settings=[],
    config::ConcordanceConfig=ConcordanceConfig()
)
    n_complexes = length(complexes)
    n_reactions = isa(A_matrix, SharedSparseMatrix) ? A_matrix.n : A_matrix.n_reactions

    # Setup random number generator with seed for reproducibility
    master_rng = Random.MersenneTwister(config.seed === nothing ? rand(UInt) : config.seed)

    # Filter out balanced complexes
    active_indices = Int32[]
    for (i, c) in enumerate(complexes)
        if !(c.id in balanced_complexes)
            push!(active_indices, Int32(i))
        end
    end

    n_active = length(active_indices)
    if n_active == 0
        return PairPriority[]
    end

    # Get sparse matrix for faster operations
    A_sparse = isa(A_matrix, SharedSparseMatrix) ? sparse(A_matrix) : sparse(A_matrix)

    # Prepare streaming statistics for complex activities
    stats = Dict{Int32,StreamingStats}()
    for i in active_indices
        stats[i] = StreamingStats()
    end

    # Calculate optimal batch size based on available memory
    # We want to avoid materializing the entire activity matrix in memory
    memory_per_active_complex = 8  # bytes per sample (Float64)
    available_memory_for_samples = config.max_memory_per_worker * 0.5  # Use half of available memory
    max_samples_in_memory = floor(Int, available_memory_for_samples / (n_active * memory_per_active_complex))

    effective_batch_size = min(
        max_samples_in_memory,
        config.sample_batch_size === nothing ? 100 : config.sample_batch_size,
        config.sample_size  # Never use a batch larger than total requested samples
    )

    # Track high-variance complexes for filtering
    high_variance_complexes = Set{Int32}()
    variance_threshold = config.tolerance * 100  # Threshold for considering a complex as high-variance

    @info "Starting streaming correlation analysis" n_active effective_batch_size total_samples_needed = config.sample_size

    # Use raw JuMP model for sample generation for better memory efficiency
    constraints = flux_balance_constraints(model)    # Create optimization model once for both feasible point and sampling
    opt_model = optimization_model(constraints, optimizer=optimizer)

    # Apply optimizer settings if provided
    for (name, value) in settings
        set_optimizer_attribute(opt_model, name, value)
    end

    # Generate a feasible starting point for sampling
    @info "Finding initial feasible point for sampling"
    optimize!(opt_model)
    if termination_status(opt_model) != MOI.OPTIMAL
        @warn "Failed to find initial feasible point for sampling"
        return PairPriority[]
    end

    # Extract initial point for sampling
    n_vars = num_variables(opt_model)
    start_point = zeros(Float64, 1, n_vars)
    if JuMP.has_values(opt_model)  # Check if model has solution values
        for i in 1:n_vars
            # Don't use has_value for individual variables - just check if model has values
            start_point[1, i] = JuMP.value(opt_model[:x][i])
        end
    end

    # Sample in batches and update statistics
    n_batches = ceil(Int, config.sample_size / effective_batch_size)
    correlation_pairs = Dict{Tuple{Int32,Int32},StreamingCorrelation}()

    # Pre-allocate matrix for batch activities
    activities = zeros(Float64, n_active, effective_batch_size)

    # Track progress
    prog = Progress(config.sample_size, desc="Generating samples: ", barlen=40)

    # Determine optimal number of chains based on available threads and workers
    optimal_chains = min(
        Threads.nthreads(),  # Limited by available threads
        4,                   # Reasonable upper limit for most systems
        n_active ÷ 1000 + 1  # Scale with problem size
    )

    @info "Sampling configuration" batch_size = effective_batch_size chains = optimal_chains
    # Generate samples in batches
    for batch in 1:n_batches
        # Adjust batch size for the last batch
        current_batch_size = min(effective_batch_size, config.sample_size - (batch - 1) * effective_batch_size)

        # Generate random samples using efficient ACHR or uniform sampling
        # Use the master_rng to generate reproducible batch seeds
        batch_seed = rand(master_rng, UInt)

        # For first batch or large problems: use multiple starting points
        # For subsequent batches: use previous samples as starting points for faster convergence
        current_start_variables = if batch == 1 || n_active > 10000
            # For first batch or large problems, create multiple starting points
            if batch == 1 && optimal_chains > 1
                # Generate multiple diverse starting points for first batch
                multi_start = zeros(Float64, optimal_chains, n_vars)
                multi_start[1, :] = start_point[1, :]  # Use our original feasible point

                # Generate additional starting points with small perturbations
                for i in 2:optimal_chains
                    # Copy with small random perturbations (within feasible space)
                    multi_start[i, :] = start_point[1, :]
                    # Add small random noise to non-zero elements
                    for j in 1:n_vars
                        if abs(start_point[1, j]) > config.tolerance
                            # Add up to ±10% noise
                            multi_start[i, j] *= (1.0 + 0.1 * randn(master_rng))
                        end
                    end
                end
                multi_start
            else
                # Single chain for subsequent batches or small problems
                start_point
            end
        else
            # For subsequent batches, use last samples as starting points
            # This significantly improves convergence speed
            samples[end-min(optimal_chains, size(samples, 1))+1:end, :]
        end

        # For complex problems, use multiple chains for better coverage
        use_chains = batch == 1 ? optimal_chains : min(optimal_chains, size(current_start_variables, 1))

        # Create a JuMP model for the optimization with settings applied
        jump_model = nothing
        if !isempty(settings)
            jump_model = Model(optimizer)
            for (name, value) in settings
                set_optimizer_attribute(jump_model, name, value)
            end
        end

        # Run the sampler with optimized parameters
        sample_tree = sample_constraints(
            COBREXA.sample_chain_achr,  # First parameter should be the sampler function
            constraints;        # Second parameter is the constraints
            start_variables=current_start_variables,
            seed=batch_seed,
            n_chains=use_chains,
            collect_iterations=[current_batch_size ÷ use_chains + (batch == n_batches ? current_batch_size % use_chains : 0)],
            workers=workers()
        )

        # Extract the sample matrix from the constraint tree result
        sample_matrix = Matrix{Float64}(undef, n_vars, current_batch_size)
        sample_idx = 1

        # Extract reaction values from the constraint tree
        for rxn_id in Symbol.(AbstractFBCModels.reactions(model))
            if haskey(sample_tree.fluxes, rxn_id) && sample_idx <= n_vars
                # Extract flux values for this reaction - directly access the value without .value
                flux_values = sample_tree.fluxes[rxn_id]  # Remove .value here
                # Store in sample matrix (up to current_batch_size)
                for j in 1:min(current_batch_size, length(flux_values))
                    sample_matrix[sample_idx, j] = flux_values[j]
                end
            end
            sample_idx += 1
        end

        # Compute complex activities in parallel using threads
        Threads.@threads for j in 1:n_active
            complex_idx = active_indices[j]
            row = A_sparse[complex_idx, :]

            for s in 1:current_batch_size
                # Compute activity using sparse dot product
                activities[j, s] = dot(row, sample_matrix[:, s])
            end
        end

        # Update streaming statistics and correlations
        GC.@preserve activities begin
            # First pass: Update statistics and identify high-variance complexes
            for j in 1:n_active
                complex_idx = active_indices[j]

                for s in 1:current_batch_size
                    update!(stats[complex_idx], activities[j, s])
                end

                # Check for high variance
                if variance(stats[complex_idx]) > variance_threshold
                    push!(high_variance_complexes, complex_idx)
                end
            end

            # Second pass: Update correlations only for promising pairs
            if !isempty(high_variance_complexes)
                Threads.@threads for i in 1:n_active
                    ci_idx = active_indices[i]

                    # Skip if low variance (major optimization)
                    if ci_idx ∉ high_variance_complexes
                        continue
                    end

                    for j in (i+1):n_active
                        cj_idx = active_indices[j]

                        # Skip if low variance
                        if cj_idx ∉ high_variance_complexes
                            continue
                        end

                        # Skip trivially concordant pairs
                        pair = (Int(ci_idx) < Int(cj_idx)) ? (Int(ci_idx), Int(cj_idx)) : (Int(cj_idx), Int(ci_idx))
                        if pair in trivial_pairs
                            continue
                        end

                        # Initialize correlation if needed
                        pair_key = (ci_idx, cj_idx)
                        if !haskey(correlation_pairs, pair_key)
                            correlation_pairs[pair_key] = StreamingCorrelation()
                        end

                        # Update correlation with current batch
                        for s in 1:current_batch_size
                            # Only consider samples where at least one complex has activity
                            if abs(activities[i, s]) > config.tolerance || abs(activities[j, s]) > config.tolerance
                                update!(correlation_pairs[pair_key], activities[i, s], activities[j, s])
                            end
                        end
                    end
                end
            end
        end

        # Explicit memory cleanup to ensure we don't keep unnecessary data
        samples = nothing
        GC.gc()

        # Update progress bar
        ProgressMeter.update!(prog, min((batch * effective_batch_size), config.sample_size))
    end

    ProgressMeter.finish!(prog)

    # Generate candidate pairs with priority information
    @info "Generating candidate pairs" high_variance_complexes = length(high_variance_complexes) correlation_pairs = length(correlation_pairs)

    # Process correlations in parallel with threads
    candidate_pairs_lock = ReentrantLock()
    candidate_pairs = PairPriority[]

    # Use parallel processing if there are many pairs
    if length(correlation_pairs) > 10000 && Threads.nthreads() > 1
        @info "Using parallel processing for correlation pairs" n_threads = Threads.nthreads()

        # Thread-local storage for results
        thread_candidates = [PairPriority[] for _ in 1:Threads.nthreads()]

        # Process in parallel chunks
        pair_keys = collect(keys(correlation_pairs))
        chunk_size = max(1, length(pair_keys) ÷ Threads.nthreads())

        # Add a counter for processed pairs to track progress
        progress_counter = Threads.Atomic{Int}(0)

        Threads.@threads for chunk_start in 1:chunk_size:length(pair_keys)
            tid = Threads.threadid()
            chunk_end = min(chunk_start + chunk_size - 1, length(pair_keys))
            chunk_pairs_processed = 0

            for pair_idx in chunk_start:chunk_end
                (ci_idx, cj_idx) = pair_keys[pair_idx]
                corr_acc = correlation_pairs[pair_keys[pair_idx]]

                if corr_acc.n >= config.min_valid_samples
                    # Calculate correlation
                    corr_value = correlation(corr_acc)

                    # Include pairs with high correlation or borderline cases
                    if abs(corr_value) >= (config.correlation_threshold)
                        directions = determine_directions(
                            Int(ci_idx), Int(cj_idx),
                            positive_complexes, negative_complexes, unrestricted_complexes
                        )

                        priority = PairPriority(
                            Int(ci_idx), Int(cj_idx), directions,
                            corr_value, corr_acc.n,
                            abs(corr_value) >= config.correlation_threshold
                        )

                        push!(thread_candidates[tid], priority)
                    end
                else
                    # Insufficient samples - include as low confidence
                    directions = determine_directions(
                        Int(ci_idx), Int(cj_idx),
                        positive_complexes, negative_complexes, unrestricted_complexes
                    )

                    priority = PairPriority(
                        Int(ci_idx), Int(cj_idx), directions,
                        0.0, corr_acc.n, false
                    )

                    push!(thread_candidates[tid], priority)
                end

                chunk_pairs_processed += 1
            end

            # Update the progress counter and progress bar after processing each chunk
            old_count = Threads.atomic_add!(progress_counter, chunk_pairs_processed)
            ProgressMeter.update!(prog, min(old_count + chunk_pairs_processed, length(pair_keys)))
        end

        # Combine results from all threads
        for tid in 1:Threads.nthreads()
            append!(candidate_pairs, thread_candidates[tid])
        end
    else
        @info "Using serial processing for correlation pairs" n_pairs = length(correlation_pairs)
        # Serial processing for smaller problems
        pair_count = 0
        total_pairs = length(correlation_pairs)

        # Create a new progress tracker for serial processing
        prog = Progress(total_pairs, desc="Processing correlations: ", barlen=40)

        for ((ci_idx, cj_idx), corr_acc) in correlation_pairs
            if corr_acc.n >= config.min_valid_samples
                # Calculate correlation
                corr_value = correlation(corr_acc)

                # Include pairs with high correlation or borderline cases
                if abs(corr_value) >= (config.correlation_threshold * 0.9)
                    directions = determine_directions(
                        Int(ci_idx), Int(cj_idx),
                        positive_complexes, negative_complexes, unrestricted_complexes
                    )

                    priority = PairPriority(
                        Int(ci_idx), Int(cj_idx), directions,
                        corr_value, corr_acc.n,
                        abs(corr_value) >= config.correlation_threshold
                    )

                    push!(candidate_pairs, priority)
                end
            else
                # Insufficient samples - include as low confidence
                directions = determine_directions(
                    Int(ci_idx), Int(cj_idx),
                    positive_complexes, negative_complexes, unrestricted_complexes
                )

                priority = PairPriority(
                    Int(ci_idx), Int(cj_idx), directions,
                    0.0, corr_acc.n, false
                )

                push!(candidate_pairs, priority)
            end

            # Update progress after processing each pair
            pair_count += 1
            ProgressMeter.update!(prog, pair_count)
        end

        # Finish the progress bar
        ProgressMeter.finish!(prog)
    end
    # Log statistics
    high_corr_count = count(p -> p.is_high_confidence, candidate_pairs)
    low_sample_count = count(p -> p.n_samples < config.min_valid_samples, candidate_pairs)

    # Calculate total potential pairs and filtered count
    total_possible_pairs = Int(n_active * (n_active - 1) / 2)
    filtered_count = total_possible_pairs - length(candidate_pairs) - length(trivial_pairs)

    @info "Correlation filtering complete" total_pairs = length(candidate_pairs) high_correlation = high_corr_count low_samples = low_sample_count filtered_pairs = filtered_count skipped_trivial = length(trivial_pairs)

    return candidate_pairs
end
"""
$(TYPEDSIGNATURES)

Process concordance analysis in stages with transitivity filtering.
Processes high-confidence pairs first, then uncertain pairs.
"""
function process_in_stages(
    model::AbstractFBCModels.AbstractFBCModel,
    constraints::ConstraintTree,
    complexes::Vector{Complex},
    candidate_priorities::Vector{PairPriority},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    concordance_tracker::Union{ConcordanceTracker,SharedConcordanceTracker},
    A_rows::Union{Vector{SparseVector{Float64,Int}},Nothing}=nothing;
    optimizer,
    settings=[],
    workers=Distributed.workers(),
    config::ConcordanceConfig=ConcordanceConfig()
)
    stage_results = Dict{String,Any}(
        "stages_completed" => 0,
        "pairs_processed" => 0,
        "concordant_pairs" => Set{Tuple{Int,Int}}(),
        "non_concordant_pairs" => 0,
        "skipped_by_transitivity" => 0,
        "optimization_results" => Dict{Tuple{Int,Int,Symbol},Float64}()
    )

    # Sort pairs by priority: high-confidence first, then by correlation
    sorted_pairs = sort(candidate_priorities,
        by=p -> (-p.is_high_confidence, -abs(p.correlation), p.c1_idx, p.c2_idx))

    # Convert to simple format for processing
    remaining_pairs = [(p.c1_idx, p.c2_idx, p.directions) for p in sorted_pairs]

    stage_count = 0
    total_pairs = length(remaining_pairs)
    processed_pairs = 0

    prog = Progress(
        total_pairs,
        desc="Concordance analysis: ",
        dt=1.0,
        barlen=50,                     # Longer bar
        output=stdout,                 # Ensure it goes to standard output
        showspeed=true                 # Show processing speed
    )

    while !isempty(remaining_pairs)
        stage_count += 1

        # Clear cache if using regular tracker
        if isa(concordance_tracker, ConcordanceTracker)
            clear_module_cache!(concordance_tracker)
        end

        @info "Starting stage $stage_count" remaining = length(remaining_pairs)

        # Filter out pairs that can be inferred
        filtered_pairs = Tuple{Int,Int,Set{Symbol}}[]

        for (c1_idx, c2_idx, directions) in remaining_pairs
            # Get tracker indices
            if isa(concordance_tracker, SharedConcordanceTracker)
                tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
                tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
            else
                tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
                tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
            end

            # Skip if already concordant
            if are_concordant(concordance_tracker, tracker_idx1, tracker_idx2)
                stage_results["skipped_by_transitivity"] += 1
                continue
            end

            # Skip if known non-concordant
            if is_non_concordant(concordance_tracker, tracker_idx1, tracker_idx2)
                stage_results["skipped_by_transitivity"] += 1
                continue
            end

            push!(filtered_pairs, (c1_idx, c2_idx, directions))
        end

        if isempty(filtered_pairs)
            break
        end

        # Take stage batch
        stage_size = min(config.stage_size, length(filtered_pairs))
        stage_pairs = filtered_pairs[1:stage_size]

        @info "Processing stage $stage_count" pairs = length(stage_pairs)

        # Process pairs
        batch_results = process_concordance_batch(
            constraints, complexes, stage_pairs, A_matrix, A_rows;
            optimizer=optimizer, settings=settings,
            workers=workers, config=config
        )

        processed_pairs += length(stage_pairs)
        # Update progress meter after processing a batch
        ProgressMeter.update!(prog, processed_pairs)

        # Update tracker with results
        new_concordant = 0
        for result in batch_results
            c1_idx, c2_idx, direction, is_concordant, lambda = result

            # Get tracker indices
            if isa(concordance_tracker, SharedConcordanceTracker)
                tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
                tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
            else
                tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
                tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
            end

            if is_concordant
                union_sets!(concordance_tracker, tracker_idx1, tracker_idx2)
                push!(stage_results["concordant_pairs"], (c1_idx, c2_idx))
                new_concordant += 1

                if !isnothing(lambda)
                    stage_results["optimization_results"][(c1_idx, c2_idx, direction)] = lambda
                end
            else
                add_non_concordant!(concordance_tracker, tracker_idx1, tracker_idx2)
                stage_results["non_concordant_pairs"] += 1
            end
        end

        stage_results["pairs_processed"] += length(stage_pairs)
        stage_results["stages_completed"] = stage_count

        @info "Stage $stage_count complete" new_concordant = new_concordant

        # Update remaining pairs
        remaining_pairs = filtered_pairs[(stage_size+1):end]
    end
    ProgressMeter.finish!(prog)
    return stage_results
end

"""
$(TYPEDSIGNATURES)

Process a batch of concordance tests.
"""
function process_concordance_batch(
    constraints::ConstraintTree,
    complexes::Vector{Complex},
    batch_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    A_rows::Union{Vector{SparseVector{Float64,Int}},Nothing}=nothing;
    optimizer,
    settings=[],
    workers=Distributed.workers(),
    config::ConcordanceConfig=ConcordanceConfig()
)
    # Get sparse matrix representation
    A_sparse = isa(A_matrix, SharedSparseMatrix) ? sparse(A_matrix) : sparse(A_matrix)
    n_complexes = size(A_sparse, 1)

    # Create A_rows if not provided
    if isnothing(A_rows)
        A_rows = Vector{SparseVector{Float64,Int}}(undef, n_complexes)
        for i in 1:n_complexes
            A_rows[i] = A_sparse[i, :]
        end
    end

    # Expand pairs by direction
    expanded_pairs = []
    for (c1_idx, c2_idx, directions) in batch_pairs
        for direction in directions
            push!(expanded_pairs, (c1_idx, c2_idx, direction))
        end
    end

    # Test concordance using COBREXA's parallel infrastructure
    results = COBREXA.screen_optimization_model(
        constraints,
        expanded_pairs;
        optimizer,
        settings,
        workers
    ) do om, (c1_idx, c2_idx, direction)
        is_conc, lambda = test_concordance_optimized(
            om, c1_idx, c2_idx,
            A_rows[c1_idx], A_rows[c2_idx],
            direction;
            tolerance=config.tolerance
        )
        return (c1_idx, c2_idx, direction, is_conc, lambda)
    end

    # Aggregate results by pair
    pair_results = Dict{Tuple{Int,Int},Dict{Symbol,Tuple{Bool,Any}}}()

    for (c1_idx, c2_idx, direction, is_conc, lambda) in results
        pair_key = (c1_idx, c2_idx)
        if !haskey(pair_results, pair_key)
            pair_results[pair_key] = Dict{Symbol,Tuple{Bool,Any}}()
        end
        pair_results[pair_key][direction] = (is_conc, lambda)
    end

    # Check if all required directions are concordant
    final_results = []
    for ((c1_idx, c2_idx), dir_results) in pair_results
        all_concordant = all(r[1] for r in values(dir_results))
        # Get lambda from any concordant direction
        lambda = nothing
        for (is_conc, lam) in values(dir_results)
            if is_conc && !isnothing(lam)
                lambda = lam
                break
            end
        end

        push!(final_results, (c1_idx, c2_idx, :both, all_concordant, lambda))
    end

    return final_results
end

"""
$(TYPEDSIGNATURES)

Main concordance analysis function optimized for large models and HPC execution.
"""
function concordance_analysis(
    model;
    modifications=Function[],
    optimizer,
    settings=[],
    workers=Distributed.workers(),
    config::ConcordanceConfig=ConcordanceConfig()
)
    start_time = time()

    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    @info "Starting concordance analysis" n_workers = length(workers) config

    # Build constraints
    constraints = concordance_constraints(model; modifications, config)

    # Extract complexes with potential shared memory 
    complexes, A_matrix, _ = extract_complexes_and_incidence(model; config)
    n_complexes = length(complexes)
    complex_ids = [c.id for c in complexes]

    @info "Model statistics" n_complexes n_reactions = (isa(A_matrix, SharedSparseMatrix) ? A_matrix.n : A_matrix.n_reactions)

    # Step 1: Find trivially balanced complexes
    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(complexes)
    @info "Found trivially balanced complexes" n_balanced = length(trivially_balanced)

    # Step 2: Find trivially concordant pairs
    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(complexes)
    @info "Found trivially concordant pairs" n_pairs = length(trivial_pairs)

    # Step 3: Identify balanced complexes via FVA (in addition to trivially balanced)
    @info "Identifying balanced complexes via activity varibability analysis (AVA)"

    # Create complex activity expressions for FVA
    complex_expr = Dict{Symbol,ConstraintTrees.Constraint}()

    batch_size = min(1000, n_complexes)
    for batch_start in 1:batch_size:n_complexes
        batch_end = min(batch_start + batch_size - 1, n_complexes)

        for cidx in batch_start:batch_end
            c = complexes[cidx]
            complex_expr[c.id] = constraints.complexes[c.id]
        end
    end

    # Run FVA on complexes
    complex_ranges = COBREXA.constraints_variability(
        constraints,
        ConstraintTree(complex_expr);
        optimizer,
        settings,
        workers
    )

    # Classify complexes by activity patterns
    balanced_complexes = Set{Symbol}()
    positive_complexes = Set{Int}()
    negative_complexes = Set{Int}()
    unrestricted_complexes = Set{Int}()

    # Start with trivially balanced complexes
    union!(balanced_complexes, trivially_balanced)

    for (i, c) in enumerate(complexes)
        cid = c.id

        # Skip if already identified as trivially balanced
        if cid in trivially_balanced
            continue
        end

        if haskey(complex_ranges, cid)
            min_val, max_val = complex_ranges[cid]
            if abs(min_val) < config.tolerance && abs(max_val) < config.tolerance
                push!(balanced_complexes, cid)
            elseif min_val >= -config.tolerance  # Can only be positive
                push!(positive_complexes, i)
            elseif max_val <= config.tolerance  # Can only be negative
                push!(negative_complexes, i)
            else
                push!(unrestricted_complexes, i)
            end
        else
            push!(unrestricted_complexes, i)
        end
    end

    @info "Complex classification" balanced = length(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = length(positive_complexes) negative = length(negative_complexes) unrestricted = length(unrestricted_complexes)

    # Step 3: Initialize concordance tracker (with shared memory if enabled)
    concordance_tracker = if config.use_shared_arrays && nworkers() > 0
        SharedConcordanceTracker(complex_ids)
    else
        ConcordanceTracker(complex_ids)
    end

    # Add trivially concordant pairs
    for (c1_idx, c2_idx) in trivial_pairs
        tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
        tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
        union_sets!(concordance_tracker, tracker_idx1, tracker_idx2)
    end

    # Step 4: Generate candidate pairs using streaming correlation
    @info "Generating candidate pairs via streaming correlation"

    candidate_priorities = streaming_correlation_filter(
        model, complexes, A_matrix, balanced_complexes,
        positive_complexes, negative_complexes, unrestricted_complexes, trivial_pairs;
        optimizer=optimizer,
        settings=settings,
        config=config
    )

    @info "Candidate pairs identified" n_pairs = length(candidate_priorities)

    # Step 5: Process in stages with transitivity
    @info "Processing concordance tests in stages"

    stage_results = process_in_stages(
        model, constraints, complexes, candidate_priorities, A_matrix,
        concordance_tracker;
        optimizer=optimizer, settings=settings,
        workers=workers, config=config
    )

    # Step 6: Build concordance modules
    @info "Building concordance modules"

    modules = extract_modules(concordance_tracker, balanced_complexes)

    # Step 7: Prepare results
    complexes_df = DataFrame(
        :complex_id => [c.id for c in complexes],
        :n_metabolites => [length(c.metabolite_indices) for c in complexes],
        :is_balanced => [c.id in balanced_complexes for c in complexes],
        :is_trivially_balanced => [c.id in trivially_balanced for c in complexes],
        :module => [get_module_id(c.id, modules) for c in complexes]
    )

    # Add activity ranges
    if !isempty(complex_ranges)
        complexes_df.min_activity = [get(complex_ranges, c.id, (NaN, NaN))[1] for c in complexes]
        complexes_df.max_activity = [get(complex_ranges, c.id, (NaN, NaN))[2] for c in complexes]
    end

    modules_df = DataFrame(
        module_id=collect(String.(keys(modules))),
        size=[length(m) for m in values(modules)],
        complexes=[join(String.(m), ", ") for m in values(modules)]
    )

    lambda_df = DataFrame(
        c1_idx=Int[],
        c2_idx=Int[],
        direction=Symbol[],
        lambda=Float64[]
    )

    for ((c1_idx, c2_idx, direction), lambda) in stage_results["optimization_results"]
        push!(lambda_df, (c1_idx, c2_idx, direction, lambda))
    end

    elapsed = time() - start_time

    stats = Dict(
        "n_complexes" => n_complexes,
        "n_balanced" => length(balanced_complexes),
        "n_trivially_balanced" => length(trivially_balanced),
        "n_trivial_pairs" => length(trivial_pairs),
        "n_candidate_pairs" => length(candidate_priorities),
        "n_concordant_pairs" => length(stage_results["concordant_pairs"]),
        "n_non_concordant_pairs" => stage_results["non_concordant_pairs"],
        "n_skipped_transitivity" => stage_results["skipped_by_transitivity"],
        "n_modules" => length(modules),
        "stages_completed" => stage_results["stages_completed"],
        "elapsed_time" => elapsed
    )

    @info "Concordance analysis complete" stats

    return (
        complexes=complexes_df,
        modules=modules_df,
        lambdas=lambda_df,
        stats=stats
    )
end

"""
Extract modules from concordance tracker.
"""
function extract_modules(tracker::Union{ConcordanceTracker,SharedConcordanceTracker}, balanced_complexes::Set{Symbol})
    # Get all disjoint sets
    groups = Dict{Int,Vector{Int}}()
    n = length(tracker.parent)

    for i in 1:n
        root = find_set!(tracker, i)
        if !haskey(groups, root)
            groups[root] = Int[]
        end
        push!(groups[root], i)
    end

    # Create modules
    modules = Dict{Symbol,Set{Symbol}}()

    # Add balanced module if exists
    if !isempty(balanced_complexes)
        modules[:balanced] = balanced_complexes
    end

    # Add other modules
    module_idx = 1
    for (root, members) in groups
        if length(members) > 1
            complex_ids = Set(tracker.idx_to_id[i] for i in members)

            # Skip if subset of balanced
            if !isempty(balanced_complexes) && issubset(complex_ids, balanced_complexes)
                continue
            end

            module_id = Symbol("module_$module_idx")
            modules[module_id] = complex_ids
            module_idx += 1
        end
    end

    return modules
end

"""
Get module ID for a complex.
"""
function get_module_id(complex_id::Symbol, modules::Dict{Symbol,Set{Symbol}})
    for (mid, members) in modules
        if complex_id in members
            return String(mid)
        end
    end
    return "none"
end

"""
Estimate memory usage for concordance analysis.
"""
function estimate_memory_usage(n_complexes::Int, n_reactions::Int,
    sparsity::Float64=0.01; use_shared::Bool=true)
    sparse_matrix_size = n_complexes * n_reactions * sparsity *
                         (sizeof(Int32) + sizeof(Float32)) +
                         n_reactions * sizeof(Int32)

    correlation_overhead = n_complexes^2 * sizeof(Float32) / 2  # Upper triangular
    concordance_tracker = n_complexes * 2 * sizeof(Int32) +
                          n_complexes^2 * sizeof(Bool)

    total = sparse_matrix_size + correlation_overhead + concordance_tracker

    savings_factor = use_shared && nworkers() > 0 ? nworkers() : 1

    return Dict(
        "sparse_matrix_GB" => sparse_matrix_size / 1e9,
        "correlation_overhead_GB" => correlation_overhead / 1e9,
        "tracker_GB" => concordance_tracker / 1e9,
        "total_GB" => total / 1e9,
        "total_with_sharing_GB" => total / 1e9 / savings_factor,
        "savings_factor" => savings_factor
    )
end

end # module COCOA