# ================================================================================================
# Deficiency Calculations and Theorem S4-6
# ================================================================================================
# Structural and mass action deficiency calculations
# Implementation of Theorem S4-6 for deficiency-one networks

export structural_deficiency, mass_action_deficiency, apply_theorem_s4_6

include("network.jl")
include("upstream.jl")

using LinearAlgebra
using SparseArrays

"""
    structural_deficiency(
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Int

Compute the structural deficiency δ of a network.

δ = n - ℓ - s

where:
- n = number of complexes
- ℓ = number of linkage classes
- s = rank of stoichiometry matrix N = YA

# Reference
Section S.1.2 of the paper, equation defining deficiency.
"""
function structural_deficiency(
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    # Count complexes in concordance structure
    all_complexes = reduce(∪, concordance_modules; init=Set{Symbol}())
    n = length(all_complexes)

    if n == 0
        return 0
    end

    # Count linkage classes
    ℓ = count_linkage_classes(all_complexes, network)

    # Compute rank of stoichiometry matrix N = Y * A
    s = stoichiometry_rank(all_complexes, network)

    δ = n - ℓ - s

    return max(0, δ)  # Deficiency is non-negative
end

"""
    count_linkage_classes(complexes::Set{Symbol}, network::ReactionNetwork) -> Int

Count the number of linkage classes (connected components) in the reaction graph.
"""
function count_linkage_classes(complexes::Set{Symbol}, network::ReactionNetwork)
    if isempty(complexes)
        return 0
    end

    # Convert to indices
    indices = Set{Int}()
    for c in complexes
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            push!(indices, idx)
        end
    end

    if isempty(indices)
        return 0
    end

    # Build undirected adjacency (reactions connect complexes)
    adjacency = Dict{Int,Set{Int}}()
    for idx in indices
        adjacency[idx] = Set{Int}()
    end

    A = network.A
    n_rxns = size(A, 2)

    for rxn_idx in 1:n_rxns
        col = A[:, rxn_idx]
        involved = Int[]
        for (cidx, val) in zip(SparseArrays.findnz(col)...)
            if cidx in indices
                push!(involved, cidx)
            end
        end

        # Connect all involved complexes (undirected)
        for i in 1:length(involved)
            for j in (i+1):length(involved)
                push!(adjacency[involved[i]], involved[j])
                push!(adjacency[involved[j]], involved[i])
            end
        end
    end

    # Count connected components using BFS
    visited = Set{Int}()
    n_components = 0

    for start in indices
        if start in visited
            continue
        end

        n_components += 1
        queue = [start]

        while !isempty(queue)
            node = popfirst!(queue)
            if node in visited
                continue
            end
            push!(visited, node)

            for neighbor in adjacency[node]
                if neighbor ∉ visited
                    push!(queue, neighbor)
                end
            end
        end
    end

    return n_components
end

"""
    stoichiometry_rank(complexes::Set{Symbol}, network::ReactionNetwork) -> Int

Compute the rank of the stoichiometry matrix N = Y * A restricted to given complexes.
"""
function stoichiometry_rank(complexes::Set{Symbol}, network::ReactionNetwork)
    if isempty(complexes)
        return 0
    end

    # Find reactions involving only complexes in the set
    complex_indices = Set{Int}()
    for c in complexes
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            push!(complex_indices, idx)
        end
    end

    if isempty(complex_indices)
        return 0
    end

    # Find relevant reactions
    relevant_rxns = Int[]
    A = network.A
    n_rxns = size(A, 2)

    for rxn_idx in 1:n_rxns
        col = A[:, rxn_idx]
        involved, _ = SparseArrays.findnz(col)

        # Check if all involved complexes are in our set
        if all(idx -> idx in complex_indices, involved)
            push!(relevant_rxns, rxn_idx)
        end
    end

    if isempty(relevant_rxns)
        return 0
    end

    # Build restricted matrices
    idx_list = sort(collect(complex_indices))
    Y_restricted = network.Y[:, idx_list]
    A_restricted = A[idx_list, relevant_rxns]

    # Compute N = Y * A
    N = Y_restricted * Matrix{Float64}(A_restricted)

    # Compute rank using SVD
    if isempty(N) || size(N, 1) == 0 || size(N, 2) == 0
        return 0
    end

    svd_result = svd(N)
    tol = 1e-10 * max(size(N)...) * maximum(svd_result.S)

    return count(s -> s > tol, svd_result.S)
end

"""
    mass_action_deficiency(
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Int

Compute the mass action deficiency δₖ.

For networks admitting positive equilibria under mass action kinetics,
δₖ ≤ δ (structural deficiency).

# Reference
Section S.4.2 of the paper.
"""
function mass_action_deficiency(
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    # For this implementation, we use the structural deficiency as an upper bound
    # A more precise calculation would require checking concordance patterns
    return structural_deficiency(concordance_modules, network)
end

"""
    check_deficiency_one(
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Bool

Check if the network has mass action deficiency δₖ = 1.

Conditions (Lemma S4-5):
1. All unbalanced complexes are mutually concordant (one unbalanced module)
2. δ = 1

# Reference
Lemma S4-5 of the paper.
"""
function check_deficiency_one(
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    # Check: exactly one unbalanced module
    n_unbalanced = length(concordance_modules) - 1
    if n_unbalanced != 1
        return false
    end

    # Check structural deficiency
    δ = structural_deficiency(concordance_modules, network)

    return δ == 1
end

"""
    apply_theorem_s4_6(
        kinetic_modules::Vector{Set{Symbol}},
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Vector{Set{Symbol}}

Apply Theorem S4-6: If δₖ = 1, all non-terminal complexes are mutually coupled.

Theorem S4-6: "Let G be a network of mass action deficiency one. Then all
nonterminal complexes in G are mutually coupled."

# Reference
Theorem S4-6 of the paper.
"""
function apply_theorem_s4_6(
    kinetic_modules::Vector{Set{Symbol}},
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    # Check if δₖ = 1
    if !check_deficiency_one(concordance_modules, network)
        return kinetic_modules
    end

    # Get all complexes
    all_complexes = reduce(∪, concordance_modules; init=Set{Symbol}())

    if isempty(all_complexes)
        return kinetic_modules
    end

    # Find terminal and non-terminal complexes
    terminal = find_terminal_complexes(all_complexes, network)
    non_terminal = setdiff(all_complexes, terminal)

    if isempty(non_terminal)
        return kinetic_modules
    end

    # Theorem S4-6: All non-terminal complexes are coupled
    # Return single module with all non-terminal, plus terminal singletons
    result = Set{Symbol}[non_terminal]

    for tc in terminal
        push!(result, Set([tc]))
    end

    return result
end

# ================================================================================================
# Weak Reversibility Check
# ================================================================================================

"""
    is_weakly_reversible(complexes::Set{Symbol}, network::ReactionNetwork) -> Bool

Check if the network restricted to given complexes is weakly reversible.

A network is weakly reversible if every linkage class is strongly connected,
i.e., every linkage class is a single SCC.

# Reference
Section S.1.2 of the paper.
"""
function is_weakly_reversible(complexes::Set{Symbol}, network::ReactionNetwork)
    if isempty(complexes)
        return true
    end

    # Convert to indices
    indices = Set{Int}()
    for c in complexes
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            push!(indices, idx)
        end
    end

    if isempty(indices)
        return true
    end

    # Find SCCs
    sccs = tarjan_scc(indices, network.A)

    # Count linkage classes
    n_linkage = count_linkage_classes(complexes, network)

    # Weakly reversible iff #SCCs == #linkage classes
    return length(sccs) == n_linkage
end
