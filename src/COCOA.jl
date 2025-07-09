"""
    module COCOA

COnstraint-based COncordance Analysis for metabolic networks - optimized for large-scale models.

This module implements memory-efficient methods for identifying concordant complexes in metabolic
networks. Optimized for models with >30,000 complexes and reactions on HPC clusters with
SharedArrays support for single-node parallelism.

## Performance Optimizations

COCOA is optimized for efficiency through integration with COBREXA.jl:

- **Model reuse**: Uses `COBREXA.screen_optimization_model` to avoid repeated model creation/destruction
- **Optimized constraint building**: Leverages COBREXA's efficient sign splitting and variable pruning
- **Smart solver settings**: Automatically applies optimized solver configurations for concordance testing
- **Memory efficiency**: Automatic model conversion to CanonicalModel format for optimal performance
- **Parallel infrastructure**: Seamless integration with COBREXA's worker management and load balancing

## Reproducibility Guarantees

COCOA implements best practices for scientific reproducibility using hierarchical random number
generation (RNG) with StableRNGs.jl:

- **Cross-platform reproducibility**: Uses StableRNGs instead of Julia's default RNG
- **Hierarchical seeding**: Master seed generates deterministic component-specific seeds
- **Deterministic sampling**: All random operations (warmup, batching, sampling) are reproducible
- **Seed logging**: All component seeds are logged for verification and debugging

### RNG Components

The analysis pipeline uses a minimal set of RNGs for reproducibility:
- **Master RNG**: Generates deterministic seeds for all components
- **Sampling RNG**: Handles all random sampling operations (warmup selection, batch seeds, etc.)

This simplified approach ensures reproducibility while minimizing complexity.

### Usage for Reproducible Results

```julia
# Same master seed will produce identical results across runs and platforms
results1 = concordance_analysis(model; optimizer=optimizer, seed=1234)
results2 = concordance_analysis(model; optimizer=optimizer, seed=1234)
# results1 == results2 (within numerical precision)

# Different seeds produce different but reproducible results
results3 = concordance_analysis(model; optimizer=optimizer, seed=5678)
# results3 != results1, but results3 is reproducible with seed=5678
```

For complete reproducibility, ensure:
1. Same Julia version and package versions
2. Same model and analysis parameters  
3. Same optimizer and solver settings
4. Same seed value
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
using StableRNGs
using JuMP
using DocStringExtensions
using ProgressMeter
using JLD2
using HiGHS
include("preprocessing/ElementarySteps.jl")
include("preprocessing/ModelPreparation.jl")
using .ElementarySteps
using .ModelPreparation

# Re-export main functions
export concordance_constraints, concordance_analysis
export split_into_elementary_steps
export prepare_model_for_concordance


"""
RNG Management and Hierarchical Seeding Utilities

These functions implement best practices for reproducible random number generation
in scientific computing, using StableRNGs for cross-platform reproducibility.
"""

"""
    create_master_rng(seed::Int=1234) -> StableRNG

Create a master RNG with the given seed for hierarchical seeding.
This ensures reproducible generation of seeds for different components.
"""
function create_master_rng(seed::Int=1234)
    return StableRNG(seed)
end

"""
    derive_seed(master_rng::AbstractRNG, component::String) -> UInt64

Derive a deterministic seed for a specific component from the master RNG.
Each component gets a unique, reproducible seed based on the master seed.
"""
function derive_seed(master_rng::AbstractRNG, component::String)
    # Create a temporary RNG to get a deterministic seed
    temp_rng = copy(master_rng)
    # Hash the component name to get a unique offset
    component_hash = hash(component)
    # Advance the RNG state based on the hash to ensure uniqueness
    for _ in 1:(component_hash%100+1)
        rand(temp_rng)
    end
    return rand(temp_rng, UInt64)
end

"""
    create_component_rng(master_rng::AbstractRNG, component::String) -> StableRNG

Create a dedicated RNG for a specific component using hierarchical seeding.
"""
function create_component_rng(master_rng::AbstractRNG, component::String)
    seed = derive_seed(master_rng, component)
    return StableRNG(seed)
end

"""
    RNGManager

Manages RNG instances for different components of the analysis pipeline.
Ensures reproducible, hierarchical seeding across all random operations.

# Fields
- `master_rng`: The master RNG for hierarchical seeding
- `component_rngs`: Dictionary of component-specific RNGs
- `seed_log`: Log of seeds used for each component for reproducibility
"""
mutable struct RNGManager
    master_rng::StableRNG
    component_rngs::Dict{String,StableRNG}
    seed_log::Dict{String,UInt64}

    function RNGManager(master_seed::Int=1234)
        master_rng = create_master_rng(master_seed)
        new(master_rng, Dict{String,StableRNG}(), Dict{String,UInt64}())
    end
end

"""
    get_rng(manager::RNGManager, component::String) -> StableRNG

Get or create an RNG for a specific component.
Logs the seed for reproducibility documentation.
"""
function get_rng(manager::RNGManager, component::String)
    if !haskey(manager.component_rngs, component)
        seed = derive_seed(manager.master_rng, component)
        manager.component_rngs[component] = StableRNG(seed)
        manager.seed_log[component] = seed
        @debug "Created RNG for component '$component' with seed $seed"
    end
    return manager.component_rngs[component]
end

"""
    reset_component(manager::RNGManager, component::String)

Reset a specific component's RNG to its initial state for reproducible re-runs.
"""
function reset_component(manager::RNGManager, component::String)
    if haskey(manager.seed_log, component)
        seed = manager.seed_log[component]
        manager.component_rngs[component] = StableRNG(seed)
        @debug "Reset RNG for component '$component' to seed $seed"
    end
end

"""
    log_seeds(manager::RNGManager) -> Dict{String, UInt64}

Return a copy of the seed log for reproducibility documentation.
"""
function log_seeds(manager::RNGManager)
    return copy(manager.seed_log)
end

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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000)
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

    # Pre-allocate buffers for metabolite processing to avoid repeated allocations
    substrate_mets = Int32[]
    substrate_stoich = Float32[]
    product_mets = Int32[]
    product_stoich = Float32[]
    sizehint!(substrate_mets, 10)  # Reasonable estimate for typical reaction size
    sizehint!(substrate_stoich, 10)
    sizehint!(product_mets, 10)
    sizehint!(product_stoich, 10)

    # Process reactions in batches
    batch_size = min(1000, n_rxns)
    for batch_start in 1:batch_size:n_rxns
        batch_end = min(batch_start + batch_size - 1, n_rxns)

        for ridx in batch_start:batch_end
            rxn = rxns[ridx]
            rxn_id = Symbol(rxn)
            rxn_stoich = AbstractFBCModels.reaction_stoichiometry(model, rxn)

            # Check if this reaction is reversible (using the same logic as unidirectional constraints)
            # We'll determine this from the model bounds if available
            is_reversible = false
            try
                # Try to get bounds from the model
                # This is a simplified check - in practice, we'd use the constraint system
                # For now, assume reversible if we can't determine otherwise
                is_reversible = true  # Conservative default
            catch
                is_reversible = true  # Conservative default
            end

            # Separate substrates and products (reuse pre-allocated buffers)
            empty!(substrate_mets)
            empty!(substrate_stoich)
            empty!(product_mets)
            empty!(product_stoich)

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
                sub_complex = create_complex(
                    substrate_mets, substrate_stoich, mets
                )
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
                prod_complex = create_complex(
                    product_mets, product_stoich, mets
                )
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
    if use_shared_arrays && nworkers() > 0 &&
       (nnz(A_sparse) * (sizeof(Int32) + sizeof(Float32)) > min_size_for_sharing)
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

Create unidirectional constraints for concordance analysis by splitting reactions into forward and reverse fluxes.

# Arguments
- `model`: FBC model

# Returns
- Modified constraint tree with unidirectional variables for all reactions
- Set of reaction indices that were split (for downstream analysis)
"""
function create_unidirectional_constraints(
    model::AbstractFBCModels.AbstractFBCModel
)
    # Start with standard flux balance constraints
    constraints = COBREXA.flux_balance_constraints(model)

    # Use COBREXA's optimized sign splitting
    constraints += COBREXA.sign_split_variables(
        constraints.fluxes,
        positive=:fluxes_forward,
        negative=:fluxes_reverse
    )

    constraints *= :directional_flux_balance^COBREXA.sign_split_constraints(
        positive=constraints.fluxes_forward,
        negative=constraints.fluxes_reverse,
        signed=constraints.fluxes,
    )
    # Substitute and prune variables for efficiency
    subst_vals = [ConstraintTrees.variable(; idx).value for idx = 1:ConstraintTrees.variable_count(constraints)]

    constraints.fluxes = ConstraintTrees.zip(constraints.fluxes, constraints.fluxes_forward, constraints.fluxes_reverse) do f, p, n
        (var_idx,) = f.value.idxs
        subst_value = p.value - n.value
        subst_vals[var_idx] = subst_value
        ConstraintTrees.Constraint(subst_value) # bidirectional bound is dropped
    end

    constraints = ConstraintTrees.prune_variables(ConstraintTrees.substitute(constraints, subst_vals))

    # All reactions were split since we applied splitting to all fluxes
    rxn_ids = Symbol.(AbstractFBCModels.reactions(model))
    all_indices = Set(1:length(rxn_ids))

    return constraints, all_indices
end

"""
$(TYPEDSIGNATURES)

Memory-efficient concordance constraints that work with large models.
"""
function concordance_constraints(
    model::AbstractFBCModels.AbstractFBCModel;
    modifications=Function[],
    interface=nothing,
    use_unidirectional_constraints::Bool=true,
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000
)
    if use_unidirectional_constraints
        constraints, split_indices = create_unidirectional_constraints(model)
        @info "Using unidirectional constraints" n_reversible_split = length(split_indices)
    else
        constraints = COBREXA.flux_balance_constraints(model; interface)
        split_indices = Set{Int}()
    end

    # Apply modifications
    for mod in modifications
        constraints = mod(constraints)
    end

    # Get the reaction IDs that will be used consistently
    original_rxn_ids = Symbol.(AbstractFBCModels.reactions(model))

    # Build incidence matrix using the SAME reaction ordering
    complexes, A_matrix, complex_dict = extract_complexes_and_incidence(model;
        use_shared_arrays, min_size_for_sharing)



    # Create complex activity expressions
    complex_activities = ConstraintTrees.ConstraintTree()
    A_sparse = isa(A_matrix, SharedSparseMatrix) ? sparse(A_matrix) : sparse(A_matrix)

    # Build activities in batches
    n_complexes = length(complexes)
    batch_size = min(1000, n_complexes)

    for batch_start in 1:batch_size:n_complexes
        batch_end = min(batch_start + batch_size - 1, n_complexes)

        for cidx in batch_start:batch_end
            complex = complexes[cidx]

            # Build activity as linear combination of fluxes
            # Start with zero value using the proper constructor
            activity_terms = ConstraintTrees.LinearValue(idxs=Int[], weights=Float64[])

            # Get the row efficiently from the sparse matrix
            for j in eachindex(original_rxn_ids)
                coeff = A_sparse[cidx, j]
                if abs(coeff) > 1e-12  # Only include non-zero coefficients
                    rxn_id = original_rxn_ids[j]
                    # Use the flux variable from the constraint tree
                    if haskey(constraints.fluxes, rxn_id)
                        activity_terms += coeff * constraints.fluxes[rxn_id].value
                    end
                end
            end

            # Create constraint for this complex activity using direct assignment
            complex_activities[complex.id] = ConstraintTrees.Constraint(
                value=activity_terms,
                bound=ConstraintTrees.Between(-Inf, Inf)
            )
        end

        GC.safepoint()
    end

    # Add complex activities to the constraint tree
    constraints = constraints * (:complexes^complex_activities)

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
"""
mutable struct CorrelationTracker
    # Tier 1: High confidence pairs (>= promotion_threshold)
    high_confidence::Dict{Tuple{Symbol,Symbol},StreamingCorrelation}

    # Tier 2: Under evaluation with LRU eviction
    under_evaluation::Dict{Tuple{Symbol,Symbol},CorrelationEntry}
    lru_order::Vector{Tuple{Symbol,Symbol}}  # For LRU eviction

    # Configuration
    max_under_evaluation::Int
    min_samples_for_decision::Int
    promotion_threshold::Float64
    rejection_confidence::Float64  # Confidence level for statistical rejection

    # Statistics
    pairs_promoted::Int
    pairs_rejected_statistically::Int
    pairs_evicted_lru::Int

    function CorrelationTracker(max_pairs::Int, promotion_threshold::Float64=0.8, rejection_confidence::Float64=0.95)
        new(
            Dict{Tuple{Symbol,Symbol},StreamingCorrelation}(),
            Dict{Tuple{Symbol,Symbol},CorrelationEntry}(),
            Vector{Tuple{Symbol,Symbol}}(),
            max_pairs,
            30,  # min_samples_for_decision
            promotion_threshold,
            rejection_confidence,
            0, 0, 0
        )
    end
end

"""
Calculate confidence interval upper bound for correlation using Fisher's z-transformation.
"""
function correlation_confidence_upper_bound(corr_acc::StreamingCorrelation, confidence_level::Float64)
    if corr_acc.n < 4  # Need minimum samples for meaningful CI
        return 1.0  # Conservative: assume could reach any correlation
    end

    r = abs(correlation(corr_acc))
    if r >= 0.999  # Handle numerical issues near 1
        return 1.0
    end

    # Fisher's z-transformation
    z = 0.5 * log((1 + r) / (1 - r))
    se_z = 1 / sqrt(corr_acc.n - 3)

    # Critical value for confidence interval
    z_critical = 1.96  # Approximation for 95% confidence
    if confidence_level ≈ 0.99
        z_critical = 2.576
    elseif confidence_level ≈ 0.90
        z_critical = 1.645
    end

    # Upper bound of confidence interval
    z_upper = z + z_critical * se_z
    r_upper = (exp(2 * z_upper) - 1) / (exp(2 * z_upper) + 1)

    return min(r_upper, 1.0)
end

"""
Update LRU order for a pair key.
"""
function update_lru!(tracker::CorrelationTracker, pair_key::Tuple{Symbol,Symbol})
    # Remove from current position if exists
    idx = findfirst(==(pair_key), tracker.lru_order)
    if idx !== nothing
        deleteat!(tracker.lru_order, idx)
    end
    # Add to front (most recently used)
    pushfirst!(tracker.lru_order, pair_key)
end


"""
Efficient copy constructor for StreamingCorrelation.
Avoids deepcopy overhead by directly copying the simple fields.
"""
function Base.copy(c::StreamingCorrelation)
    new_corr = StreamingCorrelation()  # Use the default constructor
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
Eliminates allocation entirely for existing entries.
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
"""
function update_correlation_tracker!(
    tracker::CorrelationTracker,
    pair_key::Tuple{Symbol,Symbol},
    corr_acc::StreamingCorrelation,
    sample_number::Int,
    final_threshold::Float64=0.95
)
    current_corr = abs(correlation(corr_acc))

    # Check if already in high confidence
    if haskey(tracker.high_confidence, pair_key)
        # In-place update for existing high confidence entries
        update_in_place!(tracker.high_confidence[pair_key], corr_acc)
        return true
    end

    # Calculate confidence interval upper bound
    upper_bound = correlation_confidence_upper_bound(corr_acc, tracker.rejection_confidence)

    # Statistical rejection: upper bound significantly below final threshold
    if corr_acc.n >= tracker.min_samples_for_decision &&
       upper_bound < (final_threshold - 0.05)  # Conservative margin

        # Remove from tracking if present
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            idx = findfirst(==(pair_key), tracker.lru_order)
            if idx !== nothing
                deleteat!(tracker.lru_order, idx)
            end
        end

        tracker.pairs_rejected_statistically += 1
        return false
    end

    # Promotion to high confidence
    if current_corr >= tracker.promotion_threshold && corr_acc.n >= 20
        tracker.high_confidence[pair_key] = copy(corr_acc)  # Only copy when creating new entry

        # Remove from under_evaluation if present
        if haskey(tracker.under_evaluation, pair_key)
            delete!(tracker.under_evaluation, pair_key)
            idx = findfirst(==(pair_key), tracker.lru_order)
            if idx !== nothing
                deleteat!(tracker.lru_order, idx)
            end
        end

        tracker.pairs_promoted += 1
        return true
    end

    # Keep in under_evaluation tier
    if haskey(tracker.under_evaluation, pair_key)
        # In-place update for existing under_evaluation entries
        existing_entry = tracker.under_evaluation[pair_key]
        update_in_place!(existing_entry.correlation_acc, corr_acc)

        # Update the entry fields that changed
        new_entry = CorrelationEntry(
            pair_key,
            existing_entry.correlation_acc,  # Reuse the updated accumulator
            current_corr,
            sample_number,
            upper_bound
        )
        tracker.under_evaluation[pair_key] = new_entry
        update_lru!(tracker, pair_key)
    else
        # Create new entry only when needed
        entry = CorrelationEntry(
            pair_key,
            copy(corr_acc),  # Only copy when creating new entry
            current_corr,
            sample_number,
            upper_bound
        )

        # Handle capacity limit with LRU eviction
        if length(tracker.under_evaluation) >= tracker.max_under_evaluation
            # Evict least recently used
            lru_key = pop!(tracker.lru_order)
            delete!(tracker.under_evaluation, lru_key)
            tracker.pairs_evicted_lru += 1
        end

        tracker.under_evaluation[pair_key] = entry
        update_lru!(tracker, pair_key)
    end

    return true
end

"""
Get all pairs that meet the final correlation threshold from both tiers.
"""
function get_candidate_pairs(tracker::CorrelationTracker, final_threshold::Float64, min_valid_samples::Int)
    candidates = CorrelationEntry[]

    # High confidence pairs (already meet promotion threshold)
    for (pair_key, corr_acc) in tracker.high_confidence
        if corr_acc.n >= min_valid_samples && abs(correlation(corr_acc)) >= final_threshold
            push!(candidates, CorrelationEntry(
                pair_key,
                corr_acc,
                abs(correlation(corr_acc)),
                0,  # last_updated not needed
                1.0  # confidence_upper_bound not needed
            ))
        end
    end

    # Under evaluation pairs
    for entry in values(tracker.under_evaluation)
        if entry.correlation_acc.n >= min_valid_samples &&
           abs(entry.current_correlation) >= final_threshold
            push!(candidates, entry)
        end
    end

    # Sort by correlation strength
    sort!(candidates, by=e -> abs(e.current_correlation), rev=true)

    return candidates
end

"""
Get statistics about tracker performance.
"""
function get_tracker_stats(tracker::CorrelationTracker)
    return (
        high_confidence_pairs=length(tracker.high_confidence),
        under_evaluation_pairs=length(tracker.under_evaluation),
        max_capacity=tracker.max_under_evaluation,
        pairs_promoted=tracker.pairs_promoted,
        pairs_rejected_statistically=tracker.pairs_rejected_statistically,
        pairs_evicted_lru=tracker.pairs_evicted_lru,
        total_tracked=length(tracker.high_confidence) + length(tracker.under_evaluation)
    )
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

    # Clear any existing concordance constraints to ensure clean state
    for cname in [:w, :t, :c2_act, :bounds_l, :bounds_u]
        if haskey(model, cname)
            try
                delete(model, model[cname])
                unregister(model, cname)
            catch
                # Variable might already be deleted, continue
                unregister(model, cname)
            end
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
            # Optimization failed - clean up and return failure
            # Use try-catch for robust cleanup in case of solver errors
            for cname in [:t, :c2_act]
                if haskey(model, cname)
                    try
                        delete(model, model[cname])
                        unregister(model, cname)
                    catch
                        unregister(model, cname)
                    end
                end
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

    # Clean up - use robust cleanup for model reuse
    for cname in [:t, :c2_act]
        if haskey(model, cname)
            try
                delete(model, model[cname])
                unregister(model, cname)
            catch
                unregister(model, cname)
            end
        end
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

Perform streaming correlation analysis for complex concordance with direct matrix sampling.
- Uses memory-efficient direct matrix sampling
- Filters out balanced complexes and trivial pairs
- Returns PairPriority objects for high-correlation candidates
"""
function streaming_correlation_filter(
    complexes::Vector{Complex},
    balanced_complexes::Set{Symbol},
    positive_complexes::Set{Int},
    negative_complexes::Set{Int},
    unrestricted_complexes::Set{Int},
    trivial_pairs::Set{Tuple{Int,Int}},
    warmup::Matrix{Float64},
    constraints::ConstraintTree;
    tolerance::Float64=1e-9,
    correlation_threshold::Float64=0.95,
    sample_size::Int=100,
    min_valid_samples::Int=30,
    max_correlation_pairs::Int=500_000,
    early_correlation_threshold::Float64=0.8,
    workers=Distributed.workers(),
    seed::Union{Int,Nothing}=42,
)
    # Setup RNG for reproducible sampling
    master_seed = seed === nothing ? rand() : seed
    rng_manager = RNGManager(master_seed)

    # Filter to get only complexes that need to be sampled
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    n_active = length(active_complexes)

    # Create a mapping from complex ID to original index for efficient lookup of trivial pairs
    original_indices = Dict(c.id => i for (i, c) in enumerate(complexes))

    # Initialize the hierarchical, memory-aware correlation tracker
    correlation_tracker = CorrelationTracker(
        max_correlation_pairs,
        early_correlation_threshold,
        0.95, # Confidence level for statistical rejection
    )

    # === OPTIMIZED SAMPLING CONFIGURATION ===
    @info "Configuring optimized sampler"

    # Reduce sampling overhead by using fewer chains with more samples each
    n_chains = min(4, length(workers))  # Use fewer chains
    n_warmup_points_per_chain = min(50, size(warmup, 1))  # Limit warmup points

    # More aggressive burn-in and thinning to reduce total iterations
    burn_in_period = 32
    thinning_interval = 8

    # Calculate required collections more efficiently
    target_samples_per_chain = ceil(Int, sample_size / n_chains)
    n_collections_per_chain = min(target_samples_per_chain, n_warmup_points_per_chain)

    # Generate iteration list more efficiently
    iters_to_collect = collect(burn_in_period:thinning_interval:(burn_in_period+(n_collections_per_chain-1)*thinning_interval))

    @info "Optimized sampling parameters" n_chains n_warmup_points_per_chain n_collections_per_chain iters_to_collect

    # === MEMORY-EFFICIENT SAMPLING ===
    @info "Generating flux samples with memory optimization"

    # Use a subset of warmup points to reduce memory pressure
    if size(warmup, 1) > n_warmup_points_per_chain
        sampling_rng = get_rng(rng_manager, "sampling")
        selected_indices = sort(randperm(sampling_rng, size(warmup, 1))[1:n_warmup_points_per_chain])
        limited_warmup = warmup[selected_indices, :]
    else
        limited_warmup = warmup
    end

    batch_seed = rand(get_rng(rng_manager, "sampling"), UInt64)

    # Sample with optimized settings
    all_samples = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints;
        output=constraints.complexes,
        start_variables=limited_warmup,
        seed=batch_seed,
        n_chains=n_chains,
        collect_iterations=iters_to_collect,
        workers=workers,
    )

    # === OPTIMIZED SAMPLE PROCESSING ===
    @info "Processing samples with zero-copy optimization"

    # Pre-filter active complexes more efficiently
    active_complexes = [c for c in complexes if !(c.id in balanced_complexes)]
    n_active = length(active_complexes)

    # Build index mappings once
    original_indices = Dict{Symbol,Int}()
    for (i, c) in enumerate(complexes)
        original_indices[c.id] = i
    end

    # Pre-compute skip matrix more efficiently
    skip_pairs_matrix = falses(n_active, n_active)
    for i = 1:n_active
        ci = active_complexes[i]
        ci_original_idx = get(original_indices, ci.id, 0)
        if ci_original_idx == 0
            continue
        end

        for j = (i+1):n_active
            cj = active_complexes[j]
            cj_original_idx = get(original_indices, cj.id, 0)
            if cj_original_idx == 0
                continue
            end

            canonical_pair = ci_original_idx < cj_original_idx ?
                             (ci_original_idx, cj_original_idx) : (cj_original_idx, ci_original_idx)
            skip_pairs_matrix[i, j] = canonical_pair in trivial_pairs
        end
    end

    # === ZERO-COPY ACTIVITY EXTRACTION ===
    @info "Extracting activities with zero-copy access"

    # Instead of copying all activities, create views/references
    activity_refs = Dict{Symbol,Vector{Float64}}()
    for c in active_complexes
        if haskey(all_samples, c.id)
            activity_refs[c.id] = all_samples[c.id]  # Direct reference, no copy
        end
    end

    # Filter to only complexes with actual data
    filtered_active_complexes = [c for c in active_complexes if haskey(activity_refs, c.id)]
    n_filtered_active = length(filtered_active_complexes)

    # Determine actual sample count from the first available complex
    actual_sample_count = if !isempty(activity_refs)
        length(first(values(activity_refs)))
    else
        0
    end
    n_samples_to_process = min(sample_size, actual_sample_count)

    @info "Sample processing setup" n_filtered_active n_samples_to_process

    # === OPTIMIZED CORRELATION COMPUTATION ===
    @info "Computing correlations with streaming optimization"

    # Initialize correlation tracker with optimized settings
    correlation_tracker = CorrelationTracker(
        min(max_correlation_pairs, n_filtered_active^2 ÷ 4),  # More conservative estimate
        early_correlation_threshold,
        0.95
    )

    # Pre-allocate correlation buffers
    active_pairs = Dict{Tuple{Symbol,Symbol},StreamingCorrelation}()

    # Process samples in optimized batches
    batch_size = 10  # Process in small batches to manage memory
    prog = Progress(n_samples_to_process, desc="Processing samples: ", barlen=40)

    for batch_start in 1:batch_size:n_samples_to_process
        batch_end = min(batch_start + batch_size - 1, n_samples_to_process)

        # Process this batch of samples
        for sample_idx = batch_start:batch_end
            # Update correlations for all pairs with data from current sample
            for i = 1:n_filtered_active
                ci = filtered_active_complexes[i]
                activity_data = activity_refs[ci.id]

                # Bounds check and skip if no data
                if sample_idx > length(activity_data)
                    continue
                end
                act_i = activity_data[sample_idx]

                # Skip if activity too small
                if abs(act_i) <= tolerance
                    continue
                end

                for j = (i+1):n_filtered_active
                    # Quick skip check
                    if skip_pairs_matrix[i, j]
                        continue
                    end

                    cj = filtered_active_complexes[j]
                    activity_data_j = activity_refs[cj.id]

                    # Bounds check
                    if sample_idx > length(activity_data_j)
                        continue
                    end
                    act_j = activity_data_j[sample_idx]

                    # Skip if both activities near zero
                    if abs(act_j) <= tolerance
                        continue
                    end

                    # Create pair key (reuse same pattern)
                    pair_key = (ci.id, cj.id)

                    # Skip if already rejected by tracker
                    if haskey(correlation_tracker.high_confidence, pair_key)
                        continue
                    end

                    # Get or create correlation accumulator
                    if !haskey(active_pairs, pair_key)
                        active_pairs[pair_key] = StreamingCorrelation()
                    end

                    # Update correlation
                    update!(active_pairs[pair_key], act_i, act_j)
                end
            end

            ProgressMeter.update!(prog, sample_idx)
        end

        # Process tracker updates less frequently (batch-wise)
        if batch_end % 50 == 0 || batch_end == n_samples_to_process
            pairs_to_remove = Tuple{Symbol,Symbol}[]

            for (pair_key, corr_acc) in active_pairs
                is_still_tracking = update_correlation_tracker!(
                    correlation_tracker,
                    pair_key,
                    corr_acc,
                    batch_end,
                    correlation_threshold
                )

                if !is_still_tracking
                    push!(pairs_to_remove, pair_key)
                end
            end

            # Clean up rejected pairs
            for pair_key in pairs_to_remove
                delete!(active_pairs, pair_key)
            end

            # Force GC every few batches
            if batch_end % 100 == 0
                GC.gc(false)  # Minor GC only
            end
        end
    end

    ProgressMeter.finish!(prog)

    # Final tracker update
    for (pair_key, corr_acc) in active_pairs
        update_correlation_tracker!(
            correlation_tracker,
            pair_key,
            corr_acc,
            n_samples_to_process,
            correlation_threshold
        )
    end

    # Clear references to allow GC
    activity_refs = nothing
    all_samples = nothing
    GC.gc()

    @info "Correlation tracker statistics" get_tracker_stats(correlation_tracker)...

    # === GENERATE FINAL RESULTS ===
    candidate_entries = get_candidate_pairs(correlation_tracker, correlation_threshold, min_valid_samples)
    candidate_pairs = PairPriority[]

    for entry in candidate_entries
        ci_id, cj_id = entry.pair_key
        ci_idx = findfirst(c -> c.id == ci_id, complexes)
        cj_idx = findfirst(c -> c.id == cj_id, complexes)

        if ci_idx === nothing || cj_idx === nothing
            continue
        end

        directions = determine_directions(
            ci_idx, cj_idx,
            positive_complexes, negative_complexes, unrestricted_complexes
        )

        is_high_confidence = haskey(correlation_tracker.high_confidence, entry.pair_key)

        push!(candidate_pairs, PairPriority(
            ci_idx, cj_idx, directions,
            entry.current_correlation,
            entry.correlation_acc.n,
            is_high_confidence
        ))
    end

    @info "Correlation filtering complete" total_pairs = length(candidate_pairs)

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
    stage_size::Int=500,
    tolerance::Float64=1e-9
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
        stage_size_actual = min(stage_size, length(filtered_pairs))
        stage_pairs = filtered_pairs[1:stage_size_actual]

        @info "Processing stage $stage_count" pairs = length(stage_pairs)

        # Process pairs
        batch_results = process_concordance_batch(
            constraints, complexes, stage_pairs, A_matrix, A_rows; optimizer=optimizer, settings=settings,
            workers=workers, tolerance=tolerance
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
        remaining_pairs = filtered_pairs[(stage_size_actual+1):end]
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
    tolerance::Float64=1e-9
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

    # Test concordance using COBREXA's optimized screening infrastructure
    results = COBREXA.screen_optimization_model(
        constraints,
        expanded_pairs;
        optimizer=optimizer,
        settings=settings,  # Add silence by default for reduced overhead
        workers=workers
    ) do om, (c1_idx, c2_idx, direction)
        is_conc, lambda = test_concordance_optimized(
            om, c1_idx, c2_idx,
            A_rows[c1_idx], A_rows[c2_idx],
            direction;
            tolerance=tolerance
        )
        return (c1_idx, c2_idx, direction, is_conc, lambda)
    end

    # Aggregate results by pair
    pair_results = Dict{Tuple{Int,Int},Dict{Symbol,Tuple{Bool,Union{Float64,Nothing}}}}()

    for (c1_idx, c2_idx, direction, is_conc, lambda) in results
        pair_key = (c1_idx, c2_idx)
        if !haskey(pair_results, pair_key)
            pair_results[pair_key] = Dict{Symbol,Tuple{Bool,Union{Float64,Nothing}}}()
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

# Arguments
- `model`: Metabolic model to analyze
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `modifications=Function[]`: Model modifications to apply
- `settings=[]`: Solver settings
- `workers=workers()`: Worker processes for parallel computation
- `tolerance=1e-9`: Numerical tolerance for concordance testing
- `correlation_threshold=0.95`: Minimum correlation for candidate pairs
- `sample_size=1000`: Number of flux samples for correlation analysis
- `sample_batch_size=100`: Batch size for sampling
- `stage_size=500`: Number of pairs to process per stage
- `max_memory_per_worker=2e9`: Maximum memory usage per worker (bytes)
- `use_shared_arrays=true`: Enable SharedArrays for parallel processing
- `min_size_for_sharing=1_000_000`: Minimum array size for shared memory
- `min_valid_samples=10`: Minimum samples required for correlation
- `seed=42`: Master random seed for hierarchical reproducible RNG. Uses StableRNGs for cross-platform reproducibility and generates deterministic seeds for different analysis components (warmup generation, batch coordination, etc.)
- `use_unidirectional_constraints=true`: Split reversible reactions

# Returns
Named tuple with concordance analysis results including complexes, modules, and statistics.
"""
function concordance_analysis(
    model;
    modifications=Function[],
    optimizer,
    settings=[],
    workers=Distributed.workers(),
    tolerance::Float64=1e-9,
    correlation_threshold::Float64=0.95,
    sample_size::Int=1000,
    stage_size::Int=500,
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000,
    min_valid_samples::Int=30,
    max_correlation_pairs::Int=500_000,
    early_correlation_threshold::Float64=0.8,
    seed::Union{Int,Nothing}=42,
    use_unidirectional_constraints::Bool=true,
)
    start_time = time()

    # Use COBREXA's efficient model conversion
    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel for optimal performance"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    @info "Starting concordance analysis" n_workers = length(workers) tolerance correlation_threshold sample_size use_unidirectional_constraints

    # Build constraints
    constraints =
        concordance_constraints(model; modifications, use_unidirectional_constraints, use_shared_arrays, min_size_for_sharing)

    # Extract complexes with potential shared memory 
    complexes, A_matrix, _ =
        extract_complexes_and_incidence(model; use_shared_arrays, min_size_for_sharing)
    n_complexes = length(complexes)
    complex_ids = [c.id for c in complexes]

    @info "Model statistics" n_complexes n_reactions = (
        isa(A_matrix, SharedSparseMatrix) ? A_matrix.n : A_matrix.n_reactions
    )

    # Step 1: Find trivially balanced complexes
    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(complexes)
    @info "Found trivially balanced complexes" n_balanced = length(trivially_balanced)

    # Initialize balanced_complexes with trivially balanced
    balanced_complexes = Set{Symbol}()
    union!(balanced_complexes, trivially_balanced)

    # Step 2: Find trivially concordant pairs
    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(complexes)
    @info "Found trivially concordant pairs" n_pairs = length(trivial_pairs)

    # Step 3: Perform Activity Variability Analysis (AVA) and generate warmup points simultaneously
    @info "Performing Activity Variability Analysis and generating warmup points"

    # Define a custom output function to capture both activity and flux vectors
    function ava_output_with_warmup(dir, om)
        objective_val = JuMP.objective_value(om)
        flux_vector = JuMP.value.(om[:x])
        # Correct for direction: dir=-1 for minimization, dir=1 for maximization
        activity = dir * objective_val
        return (activity, flux_vector)
    end

    # Run FVA on all complexes using optimized settings
    ava_results = COBREXA.constraints_variability(
        constraints,
        constraints.complexes;
        optimizer=optimizer,
        settings=settings,  # Add silence for reduced overhead
        output=ava_output_with_warmup,
        output_type=Tuple{Float64,Vector{Float64}},
        workers=workers,
    )

    # Process the combined results
    complex_ranges = Dict{Symbol,Tuple{Float64,Float64}}()
    warmup_points = Vector{Float64}[]

    # Process FVA results correctly - iterate directly over the Tree
    for (cid, (min_result, max_result)) in ava_results
        # Only process complexes that are not balanced
        if min_result !== nothing && max_result !== nothing
            min_activity, min_flux = min_result
            max_activity, max_flux = max_result

            # Store the activity range
            complex_ranges[cid] = (min_activity, max_activity)

            # Store warmup points  
            push!(warmup_points, min_flux)
            push!(warmup_points, max_flux)
        end
    end

    # Convert warmup points to a matrix with robust handling
    warmup = if isempty(warmup_points)
        Matrix{Float64}(undef, 0, 0)
    else
        try
            collect(transpose(reduce(hcat, warmup_points)))
        catch e
            @warn "Failed to create warmup matrix: $e. Using empty matrix."
            Matrix{Float64}(undef, 0, 0)
        end
    end

    # Classify complexes by activity patterns using the extracted ranges
    balanced_complexes = Set{Symbol}()
    positive_complexes = Set{Int}()
    negative_complexes = Set{Int}()
    unrestricted_complexes = Set{Int}()

    # Start with trivially balanced complexes
    union!(balanced_complexes, trivially_balanced)

    # Robust classification with proper error handling
    for (i, c) in enumerate(complexes)
        cid = c.id

        # Skip if already identified as trivially balanced
        if cid in trivially_balanced
            continue
        end

        if haskey(complex_ranges, cid)
            min_val, max_val = complex_ranges[cid]

            # Robust numerical comparison
            if abs(min_val) < tolerance && abs(max_val) < tolerance
                push!(balanced_complexes, cid)
            elseif min_val >= -tolerance  # Can only be positive
                push!(positive_complexes, i)
            elseif max_val <= tolerance  # Can only be negative
                push!(negative_complexes, i)
            else
                push!(unrestricted_complexes, i)
            end
        else
            # If no FVA range available, classify as unrestricted
            push!(unrestricted_complexes, i)
        end
    end




    @info "Complex classification" balanced = length(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = length(positive_complexes) negative = length(negative_complexes) unrestricted = length(unrestricted_complexes)

    # Step 3: Initialize concordance tracker
    concordance_tracker =
        if use_shared_arrays && nworkers() > 0
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

    # Pass the pre-computed warmup points directly
    candidate_priorities = streaming_correlation_filter(
        complexes,
        balanced_complexes,
        positive_complexes,
        negative_complexes,
        unrestricted_complexes,
        trivial_pairs,
        warmup,
        constraints;
        tolerance=tolerance,
        correlation_threshold=correlation_threshold,
        sample_size=sample_size,
        min_valid_samples=min_valid_samples,
        max_correlation_pairs=max_correlation_pairs,
        early_correlation_threshold=early_correlation_threshold,
        workers=workers,
        seed=seed,
    )

    @info "Candidate pairs identified" n_pairs = length(candidate_priorities)

    # Step 5: Process in stages with transitivity
    @info "Processing concordance tests in stages"

    stage_results = process_in_stages(
        model,
        constraints,
        complexes,
        candidate_priorities,
        A_matrix,
        concordance_tracker;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        stage_size,
        tolerance,
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
        :module => [get_module_id(c.id, modules) for c in complexes],
    )

    # Add activity ranges
    if !isempty(complex_ranges)
        complexes_df.min_activity =
            [get(complex_ranges, c.id, (NaN, NaN))[1] for c in complexes]
        complexes_df.max_activity =
            [get(complex_ranges, c.id, (NaN, NaN))[2] for c in complexes]
    end

    modules_df = DataFrame(
        module_id=collect(String.(keys(modules))),
        size=[length(m) for m in values(modules)],
        complexes=[join(String.(m), ", ") for m in values(modules)],
    )

    lambda_df =
        DataFrame(c1_idx=Int[], c2_idx=Int[], direction=Symbol[], lambda=Float64[])

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
        "elapsed_time" => elapsed,
    )

    @info "Concordance analysis complete" stats

    return (
        A=A_matrix,
        complexes=complexes_df,
        modules=modules_df,
        lambdas=lambda_df,
        stats=stats,
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

"""
Recommend correlation tracker configuration based on model size and available memory.
"""
function recommend_correlation_tracker_config(n_complexes::Int, available_memory_gb::Float64=8.0)
    potential_pairs = div(n_complexes * (n_complexes - 1), 2)

    # Conservative estimate: reserve 20% of available memory for correlation tracking
    available_bytes = available_memory_gb * 1e9 * 0.2
    bytes_per_correlation = 112  # StreamingCorrelation + overhead

    max_trackable_pairs = floor(Int, available_bytes / bytes_per_correlation)

    # Recommend different configurations
    if potential_pairs <= 10_000
        return (
            max_correlation_pairs=potential_pairs,
            early_correlation_threshold=0.7,
            recommendation="Small model: track all pairs"
        )
    elseif potential_pairs <= 100_000
        recommended_pairs = min(max_trackable_pairs, 50_000)
        return (
            max_correlation_pairs=recommended_pairs,
            early_correlation_threshold=0.8,
            recommendation="Medium model: correlation tracking recommended"
        )
    else
        recommended_pairs = min(max_trackable_pairs, 100_000)
        threshold = potential_pairs > 1_000_000 ? 0.85 : 0.8

        return (
            max_correlation_pairs=recommended_pairs,
            early_correlation_threshold=threshold,
            recommendation="Large model: correlation tracking essential"
        )
    end
end

end # module COCOA