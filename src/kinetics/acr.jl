# ================================================================================================
# ACR and ACRR Identification
# ================================================================================================
# Batch identification of Absolute Concentration Robustness (ACR) and
# Absolute Concentration Ratio Robustness (ACRR)

export identify_acr, identify_acrr, identify_acr_acrr

include("linear_algebra.jl")
include("coupling.jl")
include("network.jl")

"""
    identify_acr(
        projector::ColumnSpanProjector,
        metabolite_ids::Vector{Symbol};
        tol::Float64=1e-8
    ) -> Vector{Symbol}

Identify metabolites with Absolute Concentration Robustness (ACR).

A metabolite i has ACR if the unit vector eᵢ is in the column span of Y∆,
i.e., eᵢ ∈ im(Y∆).

Uses the batch formula: eᵢ ∈ im(Y∆) ⟺ ||Q[i,:]||² ≈ 1

# Arguments
- `projector`: Cached column span projector for Y∆
- `metabolite_ids`: Metabolite identifiers
- `tol`: Numerical tolerance

# Returns
Vector of metabolite symbols with ACR.
"""
function identify_acr(
    projector::ColumnSpanProjector,
    metabolite_ids::Vector{Symbol};
    tol::Float64=1e-8
)
    n = length(metabolite_ids)
    acr_mask = batch_acr_check(projector, n; tol=tol)

    return metabolite_ids[acr_mask]
end

"""
    identify_acrr(
        projector::ColumnSpanProjector,
        metabolite_ids::Vector{Symbol};
        tol::Float64=1e-8,
        parallel::Bool=true
    ) -> Vector{Tuple{Symbol,Symbol}}

Identify metabolite pairs with Absolute Concentration Ratio Robustness (ACRR).

A pair (i, j) has ACRR if (eᵢ - eⱼ) ∈ im(Y∆).

# Arguments
- `projector`: Cached column span projector for Y∆
- `metabolite_ids`: Metabolite identifiers
- `tol`: Numerical tolerance
- `parallel`: Use parallel computation for large networks

# Returns
Vector of (metabolite_i, metabolite_j) tuples with ACRR, where i < j alphabetically.
"""
function identify_acrr(
    projector::ColumnSpanProjector,
    metabolite_ids::Vector{Symbol};
    tol::Float64=1e-8,
    parallel::Bool=true
)
    n = length(metabolite_ids)

    # Choose serial or parallel based on size and flag
    if parallel && n > 100
        idx_pairs = parallel_acrr_check(projector, n; tol=tol)
    else
        idx_pairs = batch_acrr_check(projector, n; tol=tol)
    end

    # Convert to symbol pairs (sorted alphabetically)
    result = Tuple{Symbol,Symbol}[]
    for (i, j) in idx_pairs
        m1, m2 = metabolite_ids[i], metabolite_ids[j]
        if m1 < m2
            push!(result, (m1, m2))
        else
            push!(result, (m2, m1))
        end
    end

    return result
end

"""
    identify_acr_acrr(
        kinetic_modules::Vector{Set{Symbol}},
        network::ReactionNetwork;
        tol::Float64=1e-8,
        efficient::Bool=true
    ) -> NamedTuple

Identify both ACR and ACRR from kinetic modules.

# Arguments
- `kinetic_modules`: Vector of kinetic module sets
- `network`: ReactionNetwork
- `tol`: Numerical tolerance
- `efficient`: If true, use fast pairwise method; if false, use full matrix method

# Returns
NamedTuple with fields:
- `acr_metabolites`: Vector of ACR metabolite symbols
- `acrr_pairs`: Vector of ACRR metabolite pair tuples
- `acr_mask`: BitVector for efficient lookup
"""
function identify_acr_acrr(
    kinetic_modules::Vector{Set{Symbol}},
    network::ReactionNetwork;
    tol::Float64=1e-8,
    efficient::Bool=true
)
    if efficient
        return identify_acr_acrr_efficient(kinetic_modules, network; tol=tol)
    else
        return identify_acr_acrr_full(kinetic_modules, network; tol=tol)
    end
end

"""
    identify_acr_acrr_efficient(
        kinetic_modules::Vector{Set{Symbol}},
        network::ReactionNetwork;
        tol::Float64=1e-8
    ) -> NamedTuple

Fast ACR/ACRR identification using direct pairwise comparison.
O(N_modules × N_complexes²) but usually fast for small modules.
"""
function identify_acr_acrr_efficient(
    kinetic_modules::Vector{Set{Symbol}},
    network::ReactionNetwork;
    tol::Float64=1e-8
)
    n_metabolites = network.n_metabolites
    metabolite_ids = network.metabolite_ids

    # Thread-safe accumulation
    acr_set = Set{Symbol}()
    acrr_set = Set{Tuple{Symbol,Symbol}}()
    acr_lock = ReentrantLock()
    acrr_lock = ReentrantLock()

    # Parallel over modules
    Threads.@threads for module_set in kinetic_modules
        local_acr = Set{Symbol}()
        local_acrr = Set{Tuple{Symbol,Symbol}}()

        complexes = collect(module_set)
        k = length(complexes)
        k < 2 && continue

        # Check all pairs in the module
        for i in 1:k
            idx_a = get(network.complex_to_idx, complexes[i], 0)
            idx_a == 0 && continue

            for j in (i+1):k
                idx_b = get(network.complex_to_idx, complexes[j], 0)
                idx_b == 0 && continue

                # Count non-zero differences
                nnz_count = 0
                nz_idx1, nz_idx2 = 0, 0
                nz_val1, nz_val2 = 0.0, 0.0

                @inbounds for met_idx in 1:n_metabolites
                    diff = network.Y[met_idx, idx_a] - network.Y[met_idx, idx_b]
                    if abs(diff) > tol
                        nnz_count += 1
                        if nnz_count == 1
                            nz_idx1, nz_val1 = met_idx, diff
                        elseif nnz_count == 2
                            nz_idx2, nz_val2 = met_idx, diff
                        else
                            break  # More than 2 differences
                        end
                    end
                end

                if nnz_count == 1
                    # ACR: exactly one metabolite differs
                    push!(local_acr, metabolite_ids[nz_idx1])
                elseif nnz_count == 2
                    # Potential ACRR: check if opposite changes
                    if abs(nz_val1 + nz_val2) < tol
                        m1, m2 = metabolite_ids[nz_idx1], metabolite_ids[nz_idx2]
                        if m1 < m2
                            push!(local_acrr, (m1, m2))
                        else
                            push!(local_acrr, (m2, m1))
                        end
                    end
                end
            end
        end

        # Merge local results
        if !isempty(local_acr)
            lock(acr_lock) do
                union!(acr_set, local_acr)
            end
        end
        if !isempty(local_acrr)
            lock(acrr_lock) do
                union!(acrr_set, local_acrr)
            end
        end
    end

    # Build result
    acr_list = collect(acr_set)
    acrr_list = collect(acrr_set)
    acr_mask = falses(n_metabolites)
    for m in acr_list
        idx = findfirst(==(m), metabolite_ids)
        if idx !== nothing
            acr_mask[idx] = true
        end
    end

    return (
        acr_metabolites = acr_list,
        acrr_pairs = acrr_list,
        acr_mask = acr_mask
    )
end

"""
    identify_acr_acrr_full(
        kinetic_modules::Vector{Set{Symbol}},
        network::ReactionNetwork;
        tol::Float64=1e-8
    ) -> NamedTuple

Full matrix-based ACR/ACRR identification using column span projection.
More comprehensive but slower than efficient method.
"""
function identify_acr_acrr_full(
    kinetic_modules::Vector{Set{Symbol}},
    network::ReactionNetwork;
    tol::Float64=1e-8
)
    n_metabolites = network.n_metabolites
    metabolite_ids = network.metabolite_ids

    # Build Y∆ from kinetic modules
    Y_Delta = build_coupling_matrix(kinetic_modules, network)

    if size(Y_Delta, 2) == 0
        return (
            acr_metabolites = Symbol[],
            acrr_pairs = Tuple{Symbol,Symbol}[],
            acr_mask = falses(n_metabolites)
        )
    end

    # Build projector
    projector = ColumnSpanProjector(Y_Delta; tol=tol)

    if projector.rank == 0
        return (
            acr_metabolites = Symbol[],
            acrr_pairs = Tuple{Symbol,Symbol}[],
            acr_mask = falses(n_metabolites)
        )
    end

    # Batch ACR check
    acr_mask = batch_acr_check(projector, n_metabolites; tol=tol)
    acr_list = metabolite_ids[acr_mask]

    # Batch ACRR check (parallel for large networks)
    if n_metabolites > 100
        idx_pairs = parallel_acrr_check(projector, n_metabolites; tol=tol)
    else
        idx_pairs = batch_acrr_check(projector, n_metabolites; tol=tol)
    end

    # Convert to symbol pairs
    acrr_list = Tuple{Symbol,Symbol}[]
    for (i, j) in idx_pairs
        m1, m2 = metabolite_ids[i], metabolite_ids[j]
        if m1 < m2
            push!(acrr_list, (m1, m2))
        else
            push!(acrr_list, (m2, m1))
        end
    end

    return (
        acr_metabolites = acr_list,
        acrr_pairs = acrr_list,
        acr_mask = acr_mask
    )
end
