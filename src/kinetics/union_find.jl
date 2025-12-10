# ================================================================================================
# Union-Find (Disjoint Set Union) Data Structure
# ================================================================================================
# Standalone implementation with path compression and union by rank
# Used for efficient merging of kinetic modules

export UnionFind, find_root!, union!, get_groups

"""
    UnionFind{T}

Efficient disjoint set data structure with path compression and union by rank.
Maps elements of type T to group representatives.
"""
mutable struct UnionFind{T}
    parent::Dict{T,T}
    rank::Dict{T,Int}
end

"""
    UnionFind{T}()

Create an empty Union-Find structure.
"""
UnionFind{T}() where {T} = UnionFind{T}(Dict{T,T}(), Dict{T,Int}())

"""
    UnionFind(elements)

Create a Union-Find structure with each element in its own set.
"""
function UnionFind(elements)
    T = eltype(elements)
    uf = UnionFind{T}()
    for e in elements
        uf.parent[e] = e
        uf.rank[e] = 0
    end
    return uf
end

"""
    find_root!(uf::UnionFind, x)

Find the root representative of element x with path compression.
"""
function find_root!(uf::UnionFind{T}, x::T) where {T}
    if !haskey(uf.parent, x)
        uf.parent[x] = x
        uf.rank[x] = 0
        return x
    end

    if uf.parent[x] != x
        uf.parent[x] = find_root!(uf, uf.parent[x])  # Path compression
    end
    return uf.parent[x]
end

"""
    union!(uf::UnionFind, x, y) -> Bool

Merge the sets containing x and y. Returns true if they were in different sets.
Uses union by rank for balanced trees.
"""
function union!(uf::UnionFind{T}, x::T, y::T) where {T}
    root_x = find_root!(uf, x)
    root_y = find_root!(uf, y)

    if root_x == root_y
        return false  # Already in same set
    end

    # Union by rank
    if uf.rank[root_x] < uf.rank[root_y]
        uf.parent[root_x] = root_y
    elseif uf.rank[root_x] > uf.rank[root_y]
        uf.parent[root_y] = root_x
    else
        uf.parent[root_y] = root_x
        uf.rank[root_x] += 1
    end

    return true
end

"""
    get_groups(uf::UnionFind) -> Dict{T, Vector{T}}

Return all groups as a dictionary mapping root -> members.
"""
function get_groups(uf::UnionFind{T}) where {T}
    groups = Dict{T,Vector{T}}()
    for x in keys(uf.parent)
        root = find_root!(uf, x)
        if !haskey(groups, root)
            groups[root] = T[]
        end
        push!(groups[root], x)
    end
    return groups
end

"""
    get_sets(uf::UnionFind) -> Vector{Set{T}}

Return all groups as a vector of sets.
"""
function get_sets(uf::UnionFind{T}) where {T}
    groups = get_groups(uf)
    return [Set{T}(members) for members in values(groups)]
end
