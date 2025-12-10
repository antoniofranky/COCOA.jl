# ================================================================================================
# Tarjan's Strongly Connected Components Algorithm
# ================================================================================================
# Efficient O(V + E) algorithm for finding SCCs in directed graphs
# Used in upstream algorithm for identifying terminal strong linkage classes

export tarjan_scc, is_terminal_scc

using SparseArrays

"""
    tarjan_scc(vertex_indices, adjacency_matrix) -> Vector{Set{Int}}

Find all strongly connected components using Tarjan's algorithm.

# Arguments
- `vertex_indices`: Set of vertex indices to consider (subset of graph)
- `adjacency_matrix`: Sparse incidence matrix where A[i,r] < 0 means i is substrate in reaction r,
                      and A[j,r] > 0 means j is product in reaction r

# Returns
Vector of sets, each set containing vertex indices belonging to one SCC.
"""
function tarjan_scc(
    vertex_indices::Set{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
)
    # Build adjacency list for vertices in the subset
    # Edge i -> j exists if there's a reaction where i is substrate and j is product
    adjacency = Dict{Int,Vector{Int}}()
    for v in vertex_indices
        adjacency[v] = Int[]
    end

    # Find edges: for each vertex, find reactions where it's a substrate
    for src in vertex_indices
        row = A_matrix[src, :]
        for rxn_idx in SparseArrays.findnz(row)[1]
            if A_matrix[src, rxn_idx] < 0  # src is substrate
                # Find products of this reaction that are in our vertex set
                col = A_matrix[:, rxn_idx]
                for (dst, val) in zip(SparseArrays.findnz(col)...)
                    if val > 0 && dst in vertex_indices && dst != src
                        push!(adjacency[src], dst)
                    end
                end
            end
        end
    end

    # Tarjan's algorithm state
    index_counter = Ref(0)
    indices = Dict{Int,Int}()
    lowlinks = Dict{Int,Int}()
    on_stack = Dict{Int,Bool}()
    stack = Int[]
    sccs = Vector{Set{Int}}()

    function strongconnect(v)
        indices[v] = index_counter[]
        lowlinks[v] = index_counter[]
        index_counter[] += 1
        push!(stack, v)
        on_stack[v] = true

        # Consider successors
        for w in adjacency[v]
            if !haskey(indices, w)
                # w has not been visited
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elseif get(on_stack, w, false)
                # w is on stack, part of current SCC
                lowlinks[v] = min(lowlinks[v], indices[w])
            end
        end

        # If v is a root node, pop the stack to get an SCC
        if lowlinks[v] == indices[v]
            scc = Set{Int}()
            while true
                w = pop!(stack)
                on_stack[w] = false
                push!(scc, w)
                w == v && break
            end
            push!(sccs, scc)
        end
    end

    # Run Tarjan's on all vertices
    for v in vertex_indices
        if !haskey(indices, v)
            strongconnect(v)
        end
    end

    return sccs
end

"""
    is_terminal_scc(scc, all_vertices, A_matrix) -> Bool

Check if an SCC is terminal (no outgoing edges to other vertices in all_vertices).

According to the paper: "A terminal strong linkage class is a strong linkage class Λ,
no complex of which converts to any complex in 𝒞∖Λ"

In the context of the reduced network G_𝒞₀, we only consider edges within all_vertices.
"""
function is_terminal_scc(
    scc::Set{Int},
    all_vertices::Set{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
)
    for src in scc
        # Find reactions where src is a substrate
        row = A_matrix[src, :]
        for rxn_idx in SparseArrays.findnz(row)[1]
            if A_matrix[src, rxn_idx] < 0  # src is substrate
                # Check products
                col = A_matrix[:, rxn_idx]
                for (dst, val) in zip(SparseArrays.findnz(col)...)
                    # Non-terminal if there's a product in all_vertices but outside SCC
                    if val > 0 && dst in all_vertices && dst ∉ scc
                        return false
                    end
                end
            end
        end
    end
    return true
end

"""
    find_terminal_sccs(vertices, A_matrix) -> (terminal::Vector{Set{Int}}, non_terminal::Vector{Set{Int}})

Find all SCCs and partition them into terminal and non-terminal.
"""
function find_terminal_sccs(
    vertices::Set{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
)
    sccs = tarjan_scc(vertices, A_matrix)

    terminal = Set{Int}[]
    non_terminal = Set{Int}[]

    for scc in sccs
        if is_terminal_scc(scc, vertices, A_matrix)
            push!(terminal, scc)
        else
            push!(non_terminal, scc)
        end
    end

    return (terminal=terminal, non_terminal=non_terminal)
end
