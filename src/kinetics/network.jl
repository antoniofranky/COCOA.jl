# ================================================================================================
# Reaction Network Data Structure
# ================================================================================================
# Immutable representation of a chemical reaction network
# Built once from a metabolic model, reused throughout analysis

export ReactionNetwork, complex_stoichiometry

using SparseArrays

"""
    ReactionNetwork(model) -> NamedTuple

Construct a reaction network representation from a metabolic model.
Returns a NamedTuple with network information.

The model must have fields: S (stoichiometry), complexes, A (incidence matrix), Y.

# Returns
NamedTuple with fields:
- `Y::Matrix{Float64}`: Complex stoichiometry matrix (m metabolites × n complexes)
- `A::SparseMatrixCSC{Int,Int}`: Incidence matrix (n complexes × r reactions)
- `metabolite_ids::Vector{Symbol}`: Metabolite identifiers
- `complex_ids::Vector{Symbol}`: Complex identifiers
- `complex_to_idx::Dict{Symbol,Int}`: Mapping from complex ID to column index in Y
- `n_metabolites::Int`: Number of metabolites
- `n_complexes::Int`: Number of complexes
- `n_reactions::Int`: Number of reactions
"""
function ReactionNetwork(model)
    # Extract stoichiometry matrix Y
    Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)

    # Get incidence matrix A
    A_matrix = model.A

    # Build complex lookup
    complex_to_idx = Dict{Symbol,Int}(id => i for (i, id) in enumerate(complex_ids))

    return (
        Y = Matrix{Float64}(Y_matrix),
        A = A_matrix,
        metabolite_ids = metabolite_ids,
        complex_ids = complex_ids,
        complex_to_idx = complex_to_idx,
        n_metabolites = length(metabolite_ids),
        n_complexes = length(complex_ids),
        n_reactions = size(A_matrix, 2)
    )
end

"""
    complex_stoichiometry(model; return_ids=false)

Extract complex stoichiometry matrix Y from model.
Y[i,j] = coefficient of metabolite i in complex j.

If return_ids=true, also returns (Y, metabolite_ids, complex_ids).
"""
function complex_stoichiometry(model; return_ids::Bool=false)
    Y = model.Y
    metabolite_ids = Symbol.(model.mets)
    complex_ids = Symbol.(model.complexes)

    if return_ids
        return (Y, metabolite_ids, complex_ids)
    else
        return Y
    end
end

"""
    get_complex_vector(network::ReactionNetwork, complex_id::Symbol) -> Vector{Float64}

Get the stoichiometric vector for a complex.
"""
function get_complex_vector(network::ReactionNetwork, complex_id::Symbol)
    idx = get(network.complex_to_idx, complex_id, 0)
    if idx == 0
        error("Complex $complex_id not found in network")
    end
    return Vector{Float64}(network.Y[:, idx])
end

"""
    get_complex_vectors(network::ReactionNetwork, complex_ids) -> Dict{Symbol,Vector{Float64}}

Get stoichiometric vectors for multiple complexes.
"""
function get_complex_vectors(network::ReactionNetwork, complex_ids)
    result = Dict{Symbol,Vector{Float64}}()
    for c in complex_ids
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            result[c] = Vector{Float64}(network.Y[:, idx])
        end
    end
    return result
end

"""
    stoichiometric_difference(network::ReactionNetwork, c1::Symbol, c2::Symbol) -> Vector{Float64}

Compute Y(c1) - Y(c2), the stoichiometric difference between two complexes.
"""
function stoichiometric_difference(network::ReactionNetwork, c1::Symbol, c2::Symbol)
    idx1 = network.complex_to_idx[c1]
    idx2 = network.complex_to_idx[c2]
    return Vector{Float64}(network.Y[:, idx1] - network.Y[:, idx2])
end

"""
    stoichiometric_difference!(
        result::Vector{Float64},
        network::ReactionNetwork,
        c1::Symbol,
        c2::Symbol
    )

In-place computation of Y(c1) - Y(c2).
"""
function stoichiometric_difference!(
    result::Vector{Float64},
    network::ReactionNetwork,
    c1::Symbol,
    c2::Symbol
)
    idx1 = network.complex_to_idx[c1]
    idx2 = network.complex_to_idx[c2]
    @inbounds for i in 1:network.n_metabolites
        result[i] = network.Y[i, idx1] - network.Y[i, idx2]
    end
    return result
end

# ================================================================================================
# Network Analysis Utilities
# ================================================================================================

"""
    find_reactions_involving(network::ReactionNetwork, complex_idx::Int) -> Vector{Int}

Find all reaction indices where the given complex participates.
"""
function find_reactions_involving(network::ReactionNetwork, complex_idx::Int)
    row = network.A[complex_idx, :]
    return SparseArrays.findnz(row)[1]
end

"""
    find_substrates(network::ReactionNetwork, rxn_idx::Int) -> Vector{Int}

Find complex indices that are substrates in the given reaction.
"""
function find_substrates(network::ReactionNetwork, rxn_idx::Int)
    col = network.A[:, rxn_idx]
    indices, values = SparseArrays.findnz(col)
    return [i for (i, v) in zip(indices, values) if v < 0]
end

"""
    find_products(network::ReactionNetwork, rxn_idx::Int) -> Vector{Int}

Find complex indices that are products in the given reaction.
"""
function find_products(network::ReactionNetwork, rxn_idx::Int)
    col = network.A[:, rxn_idx]
    indices, values = SparseArrays.findnz(col)
    return [i for (i, v) in zip(indices, values) if v > 0]
end

"""
    has_incoming_from_outside(network::ReactionNetwork, complex_idx::Int, inside_set::Set{Int}) -> Bool

Check if a complex has incoming reactions from complexes outside the given set.
Used for identifying entry complexes in the upstream algorithm.
"""
function has_incoming_from_outside(network::ReactionNetwork, complex_idx::Int, inside_set::Set{Int})
    for rxn_idx in find_reactions_involving(network, complex_idx)
        # Check if complex is a product in this reaction
        if network.A[complex_idx, rxn_idx] > 0
            # Find substrates
            for substrate_idx in find_substrates(network, rxn_idx)
                if substrate_idx ∉ inside_set
                    return true
                end
            end
        end
    end
    return false
end
