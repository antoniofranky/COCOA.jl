# ================================================================================================
# Upstream Algorithm Implementation
# ================================================================================================
# Implements the Upstream Algorithm from Section S.2.3 of the paper
# Finds the largest upstream set within an extended concordance module

export upstream_set, compute_upstream_sets

include("tarjan.jl")
include("network.jl")

using SparseArrays

"""
    upstream_set(extended_module::Set{Symbol}, network::ReactionNetwork) -> Set{Symbol}

Compute the largest upstream set within an extended module using the Upstream Algorithm.

The upstream algorithm (Remark S2-1) has two phases:
1. Phase I: Remove entry complexes (ensures autonomy)
2. Phase II: Remove terminal strong linkage classes (ensures feeding property)

# Arguments
- `extended_module`: Set of complex symbols (balanced ∪ unbalanced module)
- `network`: ReactionNetwork containing the stoichiometry and incidence matrices

# Returns
Set of complex symbols comprising the largest upstream subset.

# Reference
Section S.2.3, Remark S2-1 of the paper.
"""
function upstream_set(extended_module::Set{Symbol}, network::ReactionNetwork)
    if isempty(extended_module)
        return Set{Symbol}()
    end

    # Convert to indices for efficient computation
    complex_indices = Set{Int}()
    for c in extended_module
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            push!(complex_indices, idx)
        end
    end

    if isempty(complex_indices)
        return Set{Symbol}()
    end

    # Phase I: Remove entry complexes iteratively
    current = copy(complex_indices)
    current = phase1_remove_entry_complexes!(current, network)

    if isempty(current)
        return Set{Symbol}()
    end

    # Phase II: Remove terminal strong linkage classes
    current = phase2_remove_terminal_sccs!(current, network)

    # Convert back to symbols
    return Set{Symbol}(network.complex_ids[i] for i in current)
end

"""
    phase1_remove_entry_complexes!(indices::Set{Int}, network::ReactionNetwork) -> Set{Int}

Phase I of the Upstream Algorithm: Remove all entry complexes.
An entry complex has incoming reactions from complexes outside the current set.

Modifies `indices` in place and returns it.
"""
function phase1_remove_entry_complexes!(indices::Set{Int}, network::ReactionNetwork)
    max_iterations = length(indices) + 10  # Safety limit

    for _ in 1:max_iterations
        entry_complexes = find_entry_complexes(indices, network)

        if isempty(entry_complexes)
            break
        end

        setdiff!(indices, entry_complexes)

        if isempty(indices)
            break
        end
    end

    return indices
end

"""
    find_entry_complexes(indices::Set{Int}, network::ReactionNetwork) -> Set{Int}

Find all entry complexes in the current set.
An entry complex has at least one incoming reaction from outside the set.
"""
function find_entry_complexes(indices::Set{Int}, network::ReactionNetwork)
    entry = Set{Int}()

    for idx in indices
        if is_entry_complex(idx, indices, network)
            push!(entry, idx)
        end
    end

    return entry
end

"""
    is_entry_complex(complex_idx::Int, current_set::Set{Int}, network::ReactionNetwork) -> Bool

Check if a complex is an entry complex (has incoming reactions from outside current_set).
"""
function is_entry_complex(complex_idx::Int, current_set::Set{Int}, network::ReactionNetwork)
    A = network.A

    # Find reactions where this complex is a product
    row = A[complex_idx, :]
    for rxn_idx in SparseArrays.findnz(row)[1]
        if A[complex_idx, rxn_idx] > 0  # Complex is a product
            # Check if any substrate is outside current_set
            col = A[:, rxn_idx]
            for (substrate_idx, val) in zip(SparseArrays.findnz(col)...)
                if val < 0 && substrate_idx ∉ current_set
                    return true
                end
            end
        end
    end

    return false
end

"""
    phase2_remove_terminal_sccs!(indices::Set{Int}, network::ReactionNetwork) -> Set{Int}

Phase II of the Upstream Algorithm: Remove all terminal strong linkage classes.

According to Remark S2-1, after Phase I we simply remove all terminal SCCs
to get the upstream set.
"""
function phase2_remove_terminal_sccs!(indices::Set{Int}, network::ReactionNetwork)
    if isempty(indices)
        return indices
    end

    # Find all SCCs in the reduced network
    sccs = tarjan_scc(indices, network.A)

    # Identify and remove terminal SCCs
    for scc in sccs
        if is_terminal_scc(scc, indices, network.A)
            setdiff!(indices, scc)
        end
    end

    return indices
end

# ================================================================================================
# Batch Operations for Multiple Modules
# ================================================================================================

"""
    compute_upstream_sets(
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Vector{Set{Symbol}}

Compute upstream sets for all unbalanced concordance modules in parallel.

# Arguments
- `concordance_modules`: Vector where [1] is balanced, [2:end] are unbalanced modules
- `network`: ReactionNetwork

# Returns
Vector of upstream sets, one for each unbalanced module.
"""
function compute_upstream_sets(
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    if length(concordance_modules) < 2
        return Set{Symbol}[]
    end

    balanced = concordance_modules[1]
    unbalanced_modules = concordance_modules[2:end]

    # Compute in parallel
    results = Vector{Set{Symbol}}(undef, length(unbalanced_modules))

    Threads.@threads for i in eachindex(unbalanced_modules)
        extended = balanced ∪ unbalanced_modules[i]
        results[i] = upstream_set(extended, network)
    end

    return results
end

"""
    compute_upstream_sets_serial(
        concordance_modules::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Vector{Set{Symbol}}

Serial version of compute_upstream_sets for debugging or small inputs.
"""
function compute_upstream_sets_serial(
    concordance_modules::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    if length(concordance_modules) < 2
        return Set{Symbol}[]
    end

    balanced = concordance_modules[1]
    return [upstream_set(balanced ∪ m, network) for m in concordance_modules[2:end]]
end

# ================================================================================================
# Analysis Utilities
# ================================================================================================

"""
    find_terminal_complexes(complexes::Set{Symbol}, network::ReactionNetwork) -> Set{Symbol}

Find all terminal complexes in the given set.
A complex is terminal if it belongs to a terminal SCC.
"""
function find_terminal_complexes(complexes::Set{Symbol}, network::ReactionNetwork)
    # Convert to indices
    indices = Set{Int}()
    for c in complexes
        idx = get(network.complex_to_idx, c, 0)
        if idx > 0
            push!(indices, idx)
        end
    end

    if isempty(indices)
        return Set{Symbol}()
    end

    # Find SCCs and identify terminal ones
    sccs = tarjan_scc(indices, network.A)
    terminal_indices = Set{Int}()

    for scc in sccs
        if is_terminal_scc(scc, indices, network.A)
            union!(terminal_indices, scc)
        end
    end

    # Convert back to symbols
    return Set{Symbol}(network.complex_ids[i] for i in terminal_indices)
end

"""
    find_nonterminal_complexes(complexes::Set{Symbol}, network::ReactionNetwork) -> Set{Symbol}

Find all non-terminal complexes in the given set.
"""
function find_nonterminal_complexes(complexes::Set{Symbol}, network::ReactionNetwork)
    terminal = find_terminal_complexes(complexes, network)
    return setdiff(complexes, terminal)
end
