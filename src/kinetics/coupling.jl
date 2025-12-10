# ================================================================================================
# Coupling Detection and Module Merging
# ================================================================================================
# Implements coupling companion matrix (Y∆) construction and module merging
# Based on Proposition S3-4 and Remark S3-3 from the paper

export build_coupling_matrix, merge_coupled_modules, can_merge

include("union_find.jl")
include("linear_algebra.jl")
include("network.jl")

"""
    build_coupling_matrix(
        coupling_sets::Vector{Set{Symbol}},
        network::ReactionNetwork
    ) -> Matrix{Float64}

Build the coupling companion matrix Y∆ from coupling sets.

Y∆ contains columns Y(eᵢ - eⱼ) for all pairs of coupled complexes.
For each coupling set {C₁, C₂, ..., Cₚ}, we add columns:
  Y(C₁) - Y(C₂), Y(C₁) - Y(C₃), ..., Y(C₁) - Y(Cₚ)

# Reference
Section S.3.1 of the paper.
"""
function build_coupling_matrix(
    coupling_sets::Vector{Set{Symbol}},
    network::ReactionNetwork
)
    columns = Vector{Float64}[]

    for coupling_set in coupling_sets
        if length(coupling_set) < 2
            continue  # Need at least 2 complexes for coupling
        end

        complexes = collect(coupling_set)
        reference = complexes[1]

        ref_idx = get(network.complex_to_idx, reference, 0)
        if ref_idx == 0
            continue
        end

        # Add columns: Y(reference) - Y(other) for each other complex
        for j in 2:length(complexes)
            other_idx = get(network.complex_to_idx, complexes[j], 0)
            if other_idx == 0
                continue
            end

            col = Vector{Float64}(network.Y[:, ref_idx] - network.Y[:, other_idx])
            push!(columns, col)
        end
    end

    if isempty(columns)
        return zeros(Float64, network.n_metabolites, 0)
    end

    return hcat(columns...)
end

"""
    can_merge(
        y_diff::Vector{Float64},
        projector::ColumnSpanProjector
    ) -> Bool

Check if two modules can be merged via Proposition S3-4.

Two modules can be merged if their stoichiometric difference y_diff = Y(Cα) - Y(Cβ)
is in the column span of Y∆, i.e., y_diff ∈ im(Y∆).

# Reference
Proposition S3-4 of the paper.
"""
function can_merge(y_diff::Vector{Float64}, projector::ColumnSpanProjector)
    return is_in_span(y_diff, projector)
end

"""
    merge_coupled_modules(
        upstream_sets::Vector{Set{Symbol}},
        network::ReactionNetwork;
        projector::Union{ColumnSpanProjector,Nothing}=nothing
    ) -> Vector{Set{Symbol}}

Merge coupled upstream sets into kinetic modules.

Uses a three-step process:
1. Trivial merging: Merge sets sharing complexes (Lemma S3-3)
2. Proposition S3-4 merging: Merge if Y(Cα) - Y(Cβ) ∈ im(Y∆)

According to Remark S3-3, merging doesn't change im(Y∆), so we can
use a single cached projector for all merge checks.

# Arguments
- `upstream_sets`: Vector of upstream sets from the upstream algorithm
- `network`: ReactionNetwork
- `projector`: Optional pre-computed projector; if not provided, will be computed

# Returns
Vector of merged kinetic modules.
"""
function merge_coupled_modules(
    upstream_sets::Vector{Set{Symbol}},
    network::ReactionNetwork;
    projector::Union{ColumnSpanProjector,Nothing}=nothing
)
    n = length(upstream_sets)
    if n == 0
        return Set{Symbol}[]
    end

    # Initialize Union-Find with module indices
    uf = UnionFind(1:n)

    # Step 1: Trivial merging - merge sets that share complexes
    trivial_merge!(uf, upstream_sets)

    # Build Y∆ and projector if not provided
    if projector === nothing
        Y_Delta = build_coupling_matrix(upstream_sets, network)
        projector = ColumnSpanProjector(Y_Delta)
    end

    # Step 2: Proposition S3-4 merging
    proposition_s34_merge!(uf, upstream_sets, network, projector)

    # Collect merged modules
    return collect_merged_modules(uf, upstream_sets)
end

"""
    trivial_merge!(uf::UnionFind, sets::Vector{Set{Symbol}})

Merge sets that share at least one complex (Lemma S3-3).
"""
function trivial_merge!(uf::UnionFind{Int}, sets::Vector{Set{Symbol}})
    n = length(sets)

    # Build complex -> set indices mapping
    complex_to_sets = Dict{Symbol,Vector{Int}}()
    for (i, s) in enumerate(sets)
        for c in s
            if !haskey(complex_to_sets, c)
                complex_to_sets[c] = Int[]
            end
            push!(complex_to_sets[c], i)
        end
    end

    # Merge sets sharing complexes
    for (_, set_indices) in complex_to_sets
        if length(set_indices) > 1
            first_idx = set_indices[1]
            for idx in set_indices[2:end]
                union!(uf, first_idx, idx)
            end
        end
    end
end

"""
    proposition_s34_merge!(
        uf::UnionFind{Int},
        upstream_sets::Vector{Set{Symbol}},
        network::ReactionNetwork,
        projector::ColumnSpanProjector
    )

Apply Proposition S3-4 merging using cached projector.

Two modules can be merged if Y(Cα) - Y(Cβ) ∈ im(Y∆) for representative
complexes Cα, Cβ from each module.
"""
function proposition_s34_merge!(
    uf::UnionFind{Int},
    upstream_sets::Vector{Set{Symbol}},
    network::ReactionNetwork,
    projector::ColumnSpanProjector
)
    n = length(upstream_sets)

    # Precompute representative complex vectors
    representatives = Dict{Int,Tuple{Symbol,Vector{Float64}}}()
    for i in 1:n
        if !isempty(upstream_sets[i])
            c = first(upstream_sets[i])
            idx = get(network.complex_to_idx, c, 0)
            if idx > 0
                representatives[i] = (c, Vector{Float64}(network.Y[:, idx]))
            end
        end
    end

    # Build pairs to check (skip already merged pairs)
    pairs_to_check = Tuple{Int,Int}[]
    for i in 1:n
        haskey(representatives, i) || continue
        for j in (i+1):n
            haskey(representatives, j) || continue
            # Skip if already in same group
            find_root!(uf, i) == find_root!(uf, j) && continue
            push!(pairs_to_check, (i, j))
        end
    end

    if isempty(pairs_to_check)
        return
    end

    # Parallel merge check
    merge_results = parallel_merge_check(pairs_to_check, representatives, projector, network)

    # Apply merges (sequential - Union-Find not thread-safe)
    for (i, j) in merge_results
        union!(uf, i, j)
    end
end

"""
    parallel_merge_check(
        pairs::Vector{Tuple{Int,Int}},
        representatives::Dict{Int,Tuple{Symbol,Vector{Float64}}},
        projector::ColumnSpanProjector,
        network::ReactionNetwork
    ) -> Vector{Tuple{Int,Int}}

Check pairs in parallel and return those that should be merged.
"""
function parallel_merge_check(
    pairs::Vector{Tuple{Int,Int}},
    representatives::Dict{Int,Tuple{Symbol,Vector{Float64}}},
    projector::ColumnSpanProjector,
    network::ReactionNetwork
)
    if isempty(pairs)
        return Tuple{Int,Int}[]
    end

    m = network.n_metabolites

    # Thread-local results and workspaces
    results = [Tuple{Int,Int}[] for _ in 1:Threads.maxthreadid()]
    workspaces = [Vector{Float64}(undef, m) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for idx in eachindex(pairs)
        i, j = pairs[idx]
        tid = Threads.threadid()
        y_diff = workspaces[tid]

        # Compute y_diff = Y(Cα) - Y(Cβ) in-place
        _, vec_i = representatives[i]
        _, vec_j = representatives[j]
        @inbounds for k in 1:m
            y_diff[k] = vec_i[k] - vec_j[k]
        end

        # Check if y_diff ∈ im(Y∆)
        if is_in_span(y_diff, projector)
            push!(results[tid], (i, j))
        end
    end

    # Merge thread-local results
    return reduce(vcat, results; init=Tuple{Int,Int}[])
end

"""
    collect_merged_modules(uf::UnionFind{Int}, upstream_sets::Vector{Set{Symbol}}) -> Vector{Set{Symbol}}

Collect merged modules from Union-Find structure.
"""
function collect_merged_modules(uf::UnionFind{Int}, upstream_sets::Vector{Set{Symbol}})
    groups = get_groups(uf)

    modules = Set{Symbol}[]
    for (_, member_indices) in groups
        merged = Set{Symbol}()
        for i in member_indices
            union!(merged, upstream_sets[i])
        end
        if !isempty(merged)
            push!(modules, merged)
        end
    end

    # Sort by size (largest first)
    sort!(modules, by=length, rev=true)

    return modules
end

# ================================================================================================
# Incremental Updates
# ================================================================================================

"""
    augment_coupling_matrix(
        Y_Delta::Matrix{Float64},
        acr_columns::Matrix{Float64}
    ) -> Matrix{Float64}

Augment Y∆ with columns for known ACR metabolites (Remark S3-6).
"""
function augment_coupling_matrix(
    Y_Delta::Matrix{Float64},
    acr_columns::Matrix{Float64}
)
    if size(acr_columns, 2) == 0
        return Y_Delta
    end
    return hcat(Y_Delta, acr_columns)
end

"""
    make_acr_augmentation_column(metabolite_idx::Int, n_metabolites::Int) -> Vector{Float64}

Create a unit vector eᵢ for ACR augmentation.
"""
function make_acr_augmentation_column(metabolite_idx::Int, n_metabolites::Int)
    col = zeros(Float64, n_metabolites)
    col[metabolite_idx] = 1.0
    return col
end
