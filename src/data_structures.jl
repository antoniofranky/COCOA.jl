"""
Data structures for COCOA - Core types, buffers, storage, and tracking functionality.

This module contains:
- Memory management buffers (AnalysisBuffers)
- Core data structures (Complex, SharedSparseMatrix, SparseIncidenceMatrix)
- Sample storage abstractions (SampleStorage, MemoryStreamStorage, DiskStreamStorage)
- Concordance tracking (ConcordanceTracker)
- Memory-efficient data structures (SparseConcordantPairs, ObjectPool)
- Complete concordance model results
"""


"""
Tracks both concordant and non-concordant relationships between complexes with transitivity.

This structure efficiently tracks:
1. Concordant relationships using Union-Find
2. Non-concordant relationships with transitivity inference
3. Module membership caching for fast lookups
4. BitVector-based boolean flags for memory-efficient complex state tracking
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
    non_concordant_modules::Set{Tuple{Int,Int}}  # Module relationship cache
    module_members_cache::Dict{Int,Vector{Int}}  # Module membership cache

    # BitVector-based boolean flags for memory-efficient tracking (8x memory reduction)
    # These are optional and only allocated when needed for large-scale analyses
    balanced_mask::Union{BitVector,Nothing}     # Tracks balanced complexes
    positive_mask::Union{BitVector,Nothing}     # Tracks positive-only complexes  
    negative_mask::Union{BitVector,Nothing}     # Tracks negative-only complexes
    unrestricted_mask::Union{BitVector,Nothing} # Tracks unrestricted complexes
    processed_mask::Union{BitVector,Nothing}    # Tracks which complexes have been processed

    function ConcordanceTracker(complex_ids::Vector{Symbol})
        n = length(complex_ids)
        parent = collect(1:n)
        rank = zeros(Int, n)

        # Pre-allocate dictionary with known size for better performance
        id_to_idx = Dict{Symbol,Int}()
        sizehint!(id_to_idx, n)
        @inbounds for (i, id) in enumerate(complex_ids)
            id_to_idx[id] = i
        end

        idx_to_id = copy(complex_ids)

        # Pre-allocate sets with estimated sizes
        non_concordant_pairs = Set{Tuple{Int,Int}}()
        sizehint!(non_concordant_pairs, n ÷ 10)  # Conservative estimate

        non_concordant_modules = Set{Tuple{Int,Int}}()
        sizehint!(non_concordant_modules, n ÷ 20)  # Conservative estimate

        module_cache = Dict{Int,Vector{Int}}()
        sizehint!(module_cache, n ÷ 5)  # Conservative estimate

        # Initialize BitVector masks as nothing - they will be allocated on demand
        # This saves memory for small models while enabling efficient tracking for large ones
        balanced_mask = nothing
        positive_mask = nothing
        negative_mask = nothing
        unrestricted_mask = nothing
        processed_mask = nothing

        new(parent, rank, id_to_idx, idx_to_id,
            non_concordant_pairs, non_concordant_modules, module_cache,
            balanced_mask, positive_mask, negative_mask, unrestricted_mask, processed_mask)
    end
end

# Union-Find operations for regular tracker - optimized with iterative path compression
@inline function find_set!(tracker::ConcordanceTracker, x::Int)
    root = x
    # Find root
    while tracker.parent[root] != root
        root = tracker.parent[root]
    end

    # Path compression - flatten the path
    while tracker.parent[x] != root
        next = tracker.parent[x]
        tracker.parent[x] = root
        x = next
    end

    return root
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

# Generic operations - using Int (Julia's native integer type) for consistency
@inline function are_concordant(tracker::ConcordanceTracker, x::Int, y::Int)
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
            # --- MODIFIED: Push to Set instead of setting Dict value ---
            push!(tracker.non_concordant_modules, (rep_x, rep_y))
            push!(tracker.non_concordant_modules, (rep_y, rep_x))
        end
    end
end

function is_non_concordant(tracker::ConcordanceTracker, x::Int, y::Int)
    # Direct check with canonical ordering
    pair = x < y ? (x, y) : (y, x)
    if pair in tracker.non_concordant_pairs
        return true
    end

    # Get representatives once and cache them
    rep_x = find_set!(tracker, x)
    rep_y = find_set!(tracker, y)

    # Same module check
    if rep_x == rep_y
        return false
    end

    # Cached module relationship - check both orderings
    if (rep_x, rep_y) in tracker.non_concordant_modules || (rep_y, rep_x) in tracker.non_concordant_modules
        return true
    end

    # Transitivity check - ensure both modules are cached
    ensure_module_cached!(tracker, rep_x)
    ensure_module_cached!(tracker, rep_y)

    # Cache module members for inner loop efficiency
    members_x = tracker.module_members_cache[rep_x]
    members_y = tracker.module_members_cache[rep_y]
    non_concordant_pairs = tracker.non_concordant_pairs

    @inbounds for m1 in members_x
        for m2 in members_y
            pair_check = m1 < m2 ? (m1, m2) : (m2, m1)
            if pair_check in non_concordant_pairs
                # Cache the result in both directions
                push!(tracker.non_concordant_modules, (rep_x, rep_y))
                push!(tracker.non_concordant_modules, (rep_y, rep_x))
                return true
            end
        end
    end

    return false
end

function ensure_module_cached!(tracker::ConcordanceTracker, rep::Int)
    if !haskey(tracker.module_members_cache, rep)
        # Pre-allocate with estimated size for better performance
        members = Vector{Int}()
        sizehint!(members, 10)  # Conservative estimate

        # Batch the find operations for better cache locality
        n = length(tracker.parent)
        @inbounds for i in 1:n
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

    # Update module relationships efficiently
    # Pre-allocate vectors with estimated sizes
    keys_to_remove = Vector{Tuple{Int,Int}}()
    new_relationships = Set{Tuple{Int,Int}}()
    sizehint!(keys_to_remove, 20)  # Conservative estimate
    sizehint!(new_relationships, 10)

    # Single pass through non_concordant_modules
    for key in tracker.non_concordant_modules
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

    # Batch removal for better performance
    @inbounds for key in keys_to_remove
        delete!(tracker.non_concordant_modules, key)
    end

    # Add new relationships in both directions
    for (rep1, rep2) in new_relationships
        push!(tracker.non_concordant_modules, (rep1, rep2))
        push!(tracker.non_concordant_modules, (rep2, rep1))
    end
end

function clear_module_cache!(tracker::ConcordanceTracker)
    empty!(tracker.module_members_cache)
end

"""
Count total concordant pairs from modules.
For a module with n complexes, there are n*(n-1)/2 concordant pairs.
"""
function count_concordant_pairs_from_modules(tracker::ConcordanceTracker)::Int
    # Group complexes by their module representative
    modules = Dict{Int,Int}()  # representative -> count

    for i in 1:length(tracker.parent)
        rep = find_set!(tracker, i)
        modules[rep] = get(modules, rep, 0) + 1
    end

    # Count pairs in each module
    total_pairs = 0
    for (rep, size) in modules
        if size > 1
            total_pairs += size * (size - 1) ÷ 2
        end
    end

    return total_pairs
end

# --- BitVector mask management functions ---

"""
    ensure_mask_allocated!(tracker::ConcordanceTracker, mask_type::Symbol)

Ensure that the specified BitVector mask is allocated. 
Mask types: :balanced, :positive, :negative, :unrestricted, :processed
"""
function ensure_mask_allocated!(tracker::ConcordanceTracker, mask_type::Symbol)
    n = length(tracker.idx_to_id)

    if mask_type == :balanced
        if tracker.balanced_mask === nothing
            tracker.balanced_mask = falses(n)
        end
    elseif mask_type == :positive
        if tracker.positive_mask === nothing
            tracker.positive_mask = falses(n)
        end
    elseif mask_type == :negative
        if tracker.negative_mask === nothing
            tracker.negative_mask = falses(n)
        end
    elseif mask_type == :unrestricted
        if tracker.unrestricted_mask === nothing
            tracker.unrestricted_mask = falses(n)
        end
    elseif mask_type == :processed
        if tracker.processed_mask === nothing
            tracker.processed_mask = falses(n)
        end
    else
        throw(ArgumentError("Unknown mask type: $mask_type. Valid types: :balanced, :positive, :negative, :unrestricted, :processed"))
    end
end

"""
    get_mask(tracker::ConcordanceTracker, mask_type::Symbol) -> BitVector

Get the specified BitVector mask, allocating it if necessary.
"""
function get_mask(tracker::ConcordanceTracker, mask_type::Symbol)::BitVector
    ensure_mask_allocated!(tracker, mask_type)

    if mask_type == :balanced
        return tracker.balanced_mask
    elseif mask_type == :positive
        return tracker.positive_mask
    elseif mask_type == :negative
        return tracker.negative_mask
    elseif mask_type == :unrestricted
        return tracker.unrestricted_mask
    elseif mask_type == :processed
        return tracker.processed_mask
    else
        throw(ArgumentError("Unknown mask type: $mask_type"))
    end
end

"""
    set_complex_flag!(tracker::ConcordanceTracker, complex_idx::Int, mask_type::Symbol, value::Bool)

Set a boolean flag for a specific complex using BitVector for memory efficiency.
"""
@inline function set_complex_flag!(tracker::ConcordanceTracker, complex_idx::Int, mask_type::Symbol, value::Bool)
    mask = get_mask(tracker, mask_type)
    mask[complex_idx] = value
end

"""
    get_complex_flag(tracker::ConcordanceTracker, complex_idx::Int, mask_type::Symbol) -> Bool

Get a boolean flag for a specific complex. Returns false if mask is not allocated.
"""
@inline function get_complex_flag(tracker::ConcordanceTracker, complex_idx::Int, mask_type::Symbol)::Bool
    if mask_type == :balanced && tracker.balanced_mask !== nothing
        return tracker.balanced_mask[complex_idx]
    elseif mask_type == :positive && tracker.positive_mask !== nothing
        return tracker.positive_mask[complex_idx]
    elseif mask_type == :negative && tracker.negative_mask !== nothing
        return tracker.negative_mask[complex_idx]
    elseif mask_type == :unrestricted && tracker.unrestricted_mask !== nothing
        return tracker.unrestricted_mask[complex_idx]
    elseif mask_type == :processed && tracker.processed_mask !== nothing
        return tracker.processed_mask[complex_idx]
    else
        return false  # Default to false if mask not allocated
    end
end

# ========================================================================================
# Memory-Efficient Sparse Data Structures  
# ========================================================================================

"""
Memory-efficient representation of concordant pairs using sparse bit matrix.
Uses ~1 bit per pair instead of 16 bytes for Set{Tuple{Int,Int}}.
"""
mutable struct SparseConcordantPairs
    n::Int  # Number of complexes
    # Use sparse matrix for memory efficiency - only stores non-zero entries
    matrix::SparseArrays.SparseMatrixCSC{Bool,Int}
    # Track number of pairs for fast counting
    n_pairs::Int

    function SparseConcordantPairs(n_complexes::Int)
        # Initialize empty sparse matrix
        matrix = SparseArrays.spzeros(Bool, n_complexes, n_complexes)
        new(n_complexes, matrix, 0)
    end
end

"""
Add a concordant pair to the sparse matrix.
"""
@inline function add_pair!(pairs::SparseConcordantPairs, i::Int, j::Int)
    if i > j
        i, j = j, i  # Canonical order
    end
    if i <= pairs.n && j <= pairs.n && !pairs.matrix[i, j]
        pairs.matrix[i, j] = true
        pairs.n_pairs += 1
    end
end

"""
Check if a pair is concordant.
"""
@inline function has_pair(pairs::SparseConcordantPairs, i::Int, j::Int)::Bool
    if i > j
        i, j = j, i  # Canonical order
    end
    return i <= pairs.n && j <= pairs.n && pairs.matrix[i, j]
end

"""
Merge another SparseConcordantPairs into this one.
"""
function merge_pairs!(dest::SparseConcordantPairs, src::SparseConcordantPairs)
    # Use sparse matrix operations for efficiency
    dest.matrix = dest.matrix .| src.matrix
    dest.n_pairs = SparseArrays.nnz(dest.matrix)
end

"""
Clear all pairs (for reuse).
"""
function clear!(pairs::SparseConcordantPairs)
    pairs.matrix = SparseArrays.spzeros(Bool, pairs.n, pairs.n)
    pairs.n_pairs = 0
end

@enum ConcordanceType::Int begin
    None = 0
    Concordant = 1
    Trivially_concordant = 2
    Balanced = 3
    Trivially_balanced = 4
end

mutable struct CompleteConcordanceModel
    # === IDENTIFIER MAPPINGS (following AbstractFBCModels pattern) ===
    complex_ids::Vector{Symbol}                # Position = index
    complex_idx::Dict{Symbol,Int}             # Symbol -> index lookup

    reaction_ids::Vector{Symbol}
    reaction_idx::Dict{Symbol,Int}

    metabolite_ids::Vector{Symbol}
    metabolite_idx::Dict{Symbol,Int}

    # === SYMMETRIC CONCORDANCE DATA ===
    concordance_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
    lambda_dict::Dict{Tuple{Int,Int},Float64}

    # === ASYMMETRIC STRUCTURAL DATA (regular sparse) ===
    complex_reaction_matrix::SparseArrays.SparseMatrixCSC{Int,Int}      # Complexes × Reactions
    metabolite_complex_matrix::SparseArrays.SparseMatrixCSC{Float64,Int}  # Metabolites × Complexes (Y matrix)
    metabolite_reaction_matrix::SparseArrays.SparseMatrixCSC{Float64,Int} # Metabolites × Reactions (S matrix)

    # === SPARSE ACRR DATA (simple vector for few pairs) ===
    acrr_pairs::Vector{Tuple{Int,Int}}          # Canonical form: i < j

    # === DENSE NODE PROPERTIES ===
    activity_ranges::Vector{Tuple{Float64,Float64}}
    concordance_modules::Vector{Int}
    kinetic_modules::Vector{Int}                # 0 = no kinetic module

    # === DENSE REACTION PROPERTIES ===
    interface_reactions::BitVector              # true = Interface, false = Internal

    # === METABOLITE PROPERTIES ===
    acr_metabolites::BitVector                   # ACR per metabolite

    # === METADATA ===
    n_complexes::Int
    n_reactions::Int
    n_metabolites::Int
    n_concordant::Int
    n_trivially_concordant::Int
    n_balanced::Int
    n_trivially_balanced::Int
    n_concordance_modules::Int
    n_kinetic_modules::Int
    stats::Dict{String,Any}
end

#  Constructor

function CompleteConcordanceModel(
    complex_ids::Vector{Symbol},
    reaction_ids::Vector{Symbol},
    metabolite_ids::Vector{Symbol};

    # Efficient sparse concordance data (coordinate vectors)
    I_concordance::Vector{Int}=Int[],
    J_concordance::Vector{Int}=Int[],
    V_concordance::Vector{Int}=Int[],
    lambda_dict::Dict{Tuple{Int,Int},Float64}=Dict{Tuple{Int,Int},Float64}(),
    acrr_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],

    # Properties with defaults
    activity_ranges=fill((NaN, NaN), length(complex_ids)),
    concordance_modules=fill(-1, length(complex_ids)),  # -1 = singleton, 0 = balanced, >0 = concordance module
    kinetic_modules=fill(-1, length(complex_ids)),     # -1 = not in kinetic module, >0 = kinetic module
    interface_reactions=falses(length(reaction_ids)),
    acr_metabolites=falses(length(metabolite_ids)),

    # Structural matrices
    complex_reaction_matrix=SparseArrays.spzeros(Int, length(complex_ids), length(reaction_ids)),
    complex_metabolite_matrix=SparseArrays.spzeros(Float64, length(complex_ids), length(metabolite_ids)),
    reaction_metabolite_matrix=SparseArrays.spzeros(Float64, length(reaction_ids), length(metabolite_ids))
)

    # Build index dictionaries (following AbstractFBCModels pattern)
    complex_idx = Dict(id => idx for (idx, id) in enumerate(complex_ids))
    reaction_idx = Dict(id => idx for (idx, id) in enumerate(reaction_ids))
    metabolite_idx = Dict(id => idx for (idx, id) in enumerate(metabolite_ids))

    n_complexes = length(complex_ids)
    n_reactions = length(reaction_ids)
    n_metabolites = length(metabolite_ids)

    # Build concordance matrix directly from coordinate vectors (maximum efficiency)
    if !isempty(I_concordance)
        # Make symmetric by adding both (i,j) and (j,i) entries
        I_symmetric = vcat(I_concordance, J_concordance)
        J_symmetric = vcat(J_concordance, I_concordance)
        V_symmetric = vcat(V_concordance, V_concordance)

        concordance_matrix = SparseArrays.sparse(I_symmetric, J_symmetric, V_symmetric, n_complexes, n_complexes)
    else
        concordance_matrix = SparseArrays.spzeros(Int, n_complexes, n_complexes)
    end

    # Lambda dict is already in the correct format - no conversion needed

    # Store ACRR pairs in canonical form (i < j)
    canonical_acrr_pairs = [i < j ? (i, j) : (j, i) for (i, j) in acrr_pairs]
    acrr_pairs = unique(canonical_acrr_pairs)

    # Count modules
    n_concordance_modules = length(unique(filter(x -> x > 0, concordance_modules)))
    n_kinetic_modules = length(unique(filter(x -> x > 0, kinetic_modules)))

    return CompleteConcordanceModel(
        complex_ids, complex_idx,
        reaction_ids, reaction_idx,
        metabolite_ids, metabolite_idx,
        concordance_matrix, lambda_dict,
        complex_reaction_matrix, complex_metabolite_matrix, reaction_metabolite_matrix,
        acrr_pairs,
        activity_ranges, concordance_modules, kinetic_modules,
        interface_reactions, acr_metabolites,
        n_complexes, n_reactions, n_metabolites,
        length(V_concordance),  # n_concordant
        0,  # n_trivially_concordant (will be set elsewhere)
        0,  # n_balanced (will be set elsewhere) 
        0,  # n_trivially_balanced (will be set elsewhere)
        n_concordance_modules, n_kinetic_modules,
        Dict{String,Any}()
    )
end

#  Query Interface

# === FAST LOOKUPS (following AbstractFBCModels pattern) ===
# Index to ID: O(1) vector access
get_complex_id(model, idx::Int) = model.complex_ids[idx]
get_reaction_id(model, idx::Int) = model.reaction_ids[idx]
get_metabolite_id(model, idx::Int) = model.metabolite_ids[idx]

# ID to index: O(1) dict lookup
get_complex_idx(model, id::Symbol) = model.complex_idx[id]
get_reaction_idx(model, id::Symbol) = model.reaction_idx[id]
get_metabolite_idx(model, id::Symbol) = model.metabolite_idx[id]

# === CONCORDANCE QUERIES ===
function get_concordance_type(model::CompleteConcordanceModel, c1::Symbol, c2::Symbol)
    i, j = model.complex_idx[c1], model.complex_idx[c2]
    canonical_pair = i < j ? (i, j) : (j, i)
    return ConcordanceType(model.concordance_matrix[i, j])  # UpperTriangular handles symmetry
end

function get_lambda_value(model::CompleteConcordanceModel, c1::Symbol, c2::Symbol)
    i, j = model.complex_idx[c1], model.complex_idx[c2]
    key = i < j ? (i, j) : (j, i)
    return haskey(model.lambda_dict, key) ? model.lambda_dict[key] : NaN
end

function is_concordant(model::CompleteConcordanceModel, c1::Symbol, c2::Symbol)
    return get_concordance_type(model, c1, c2) != None
end

# === ACRR QUERIES (efficient for sparse data) ===
function has_acrr(model::CompleteConcordanceModel, c1::Symbol, c2::Symbol)
    i, j = model.complex_idx[c1], model.complex_idx[c2]
    canonical_pair = i < j ? (i, j) : (j, i)
    return canonical_pair ∈ model.acrr_pairs
end

function get_acrr_pairs(model::CompleteConcordanceModel)
    return [(model.complex_ids[i], model.complex_ids[j]) for (i, j) in model.acrr_pairs]
end

function add_acrr_pair!(model::CompleteConcordanceModel, c1::Symbol, c2::Symbol)
    i, j = model.complex_idx[c1], model.complex_idx[c2]
    canonical_pair = i < j ? (i, j) : (j, i)
    if canonical_pair ∉ model.acrr_pairs
        push!(model.acrr_pairs, canonical_pair)
    end
end

# === STRUCTURAL QUERIES ===
function get_complex_reactions(model::CompleteConcordanceModel, complex::Symbol)
    idx = model.complex_idx[complex]
    reaction_idxs, _ = SparseArrays.findnz(model.complex_reaction_matrix[idx, :])
    return model.reaction_ids[reaction_idxs]
end

function get_complex_metabolites(model::CompleteConcordanceModel, complex::Symbol)
    idx = model.complex_idx[complex]
    met_idxs, coeffs = SparseArrays.findnz(model.complex_metabolite_matrix[idx, :])
    return [(model.metabolite_ids[i], coeff) for (i, coeff) in zip(met_idxs, coeffs)]
end

# === ACR QUERIES ===
function has_acr(model::CompleteConcordanceModel, metabolite::Symbol)
    met_idx = model.metabolite_idx[metabolite]
    return model.acr_metabolites[met_idx]
end

function get_acr_metabolites(model::CompleteConcordanceModel)
    acr_idxs = findall(model.acr_metabolites)
    return model.metabolite_ids[acr_idxs]
end

