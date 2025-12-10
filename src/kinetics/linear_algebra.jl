# ================================================================================================
# Linear Algebra Utilities for Kinetic Analysis
# ================================================================================================
# Cached QR decomposition and column span operations
# Core numerical routines for coupling detection and ACR identification

export ColumnSpanProjector, is_in_span, ThreadWorkspaces

using LinearAlgebra

"""
    ColumnSpanProjector

Cached QR decomposition for efficient repeated column span membership checks.
Stores orthonormal basis Q for im(M), enabling O(m×k) projection checks
instead of O(m³) QR decomposition per check.

# Fields
- `Q::Matrix{Float64}`: Orthonormal basis for column span (m × k)
- `rank::Int`: Effective rank of the original matrix
- `tol::Float64`: Numerical tolerance used for rank determination
"""
struct ColumnSpanProjector
    Q::Matrix{Float64}
    rank::Int
    tol::Float64
end

"""
    ColumnSpanProjector(M::Matrix{Float64}; tol::Float64=1e-8)

Construct a projector from matrix M using QR decomposition.
"""
function ColumnSpanProjector(M::Matrix{Float64}; tol::Float64=1e-8)
    m = size(M, 1)

    if size(M, 2) == 0
        return ColumnSpanProjector(zeros(Float64, m, 0), 0, tol)
    end

    # QR decomposition
    F = qr(M)
    R = F.R

    # Determine effective rank from diagonal of R
    r_diag = abs.(diag(R))
    k = sum(r_diag .> tol)

    if k == 0
        return ColumnSpanProjector(zeros(Float64, m, 0), 0, tol)
    end

    # Extract reduced Q (orthonormal basis for column span)
    Q_full = Matrix(F.Q)
    Q_reduced = Q_full[:, 1:k]

    return ColumnSpanProjector(Q_reduced, k, tol)
end

"""
    is_in_span(v::AbstractVector, proj::ColumnSpanProjector) -> Bool

Check if vector v is in the column span using cached projection.
Returns true if ||P v - v|| < tol where P = Q Q^T.

Computational cost: O(m × k) where m = length(v), k = proj.rank
"""
function is_in_span(v::AbstractVector{Float64}, proj::ColumnSpanProjector)
    if proj.rank == 0
        return norm(v) < proj.tol
    end

    # Projection: P v = Q (Q^T v)
    coeffs = proj.Q' * v          # k × 1
    proj_v = proj.Q * coeffs      # m × 1
    residual = norm(proj_v - v)

    return residual < proj.tol
end

"""
    is_in_span!(v::AbstractVector, proj::ColumnSpanProjector, workspace::Vector{Float64}) -> Bool

In-place version using pre-allocated workspace to avoid allocations.
workspace must have length >= proj.rank
"""
function is_in_span!(
    v::AbstractVector{Float64},
    proj::ColumnSpanProjector,
    coeffs_workspace::Vector{Float64},
    proj_workspace::Vector{Float64}
)
    if proj.rank == 0
        return norm(v) < proj.tol
    end

    k = proj.rank
    m = length(v)

    # coeffs = Q^T v (in-place)
    @inbounds for j in 1:k
        s = 0.0
        for i in 1:m
            s += proj.Q[i, j] * v[i]
        end
        coeffs_workspace[j] = s
    end

    # Compute ||Q * coeffs - v||² directly without forming proj_v
    residual_sq = 0.0
    @inbounds for i in 1:m
        proj_vi = 0.0
        for j in 1:k
            proj_vi += proj.Q[i, j] * coeffs_workspace[j]
        end
        diff = proj_vi - v[i]
        residual_sq += diff * diff
    end

    return residual_sq < proj.tol^2
end

"""
    projection_residual_sq(v::AbstractVector, proj::ColumnSpanProjector) -> Float64

Compute ||P v - v||² where P is the projection onto the column span.
"""
function projection_residual_sq(v::AbstractVector{Float64}, proj::ColumnSpanProjector)
    if proj.rank == 0
        return sum(abs2, v)
    end

    coeffs = proj.Q' * v
    proj_v = proj.Q * coeffs
    return sum(abs2, proj_v - v)
end

# ================================================================================================
# Thread-safe Workspaces
# ================================================================================================

"""
    ThreadWorkspaces{T}

Pre-allocated workspaces for each thread to avoid allocations in parallel loops.
Uses Threads.maxthreadid() to handle Julia 1.9+ interactive threads correctly.
"""
struct ThreadWorkspaces{T}
    buffers::Vector{T}
end

"""
    ThreadWorkspaces(constructor, n_threads=Threads.maxthreadid())

Create workspaces using the provided constructor function for each thread.
"""
function ThreadWorkspaces(constructor::Function)
    n = Threads.maxthreadid()
    buffers = [constructor() for _ in 1:n]
    return ThreadWorkspaces(buffers)
end

"""
    get_workspace(ws::ThreadWorkspaces) -> T

Get the workspace for the current thread.
"""
function get_workspace(ws::ThreadWorkspaces)
    tid = Threads.threadid()
    return ws.buffers[tid]
end

# ================================================================================================
# Batch Operations for ACR Detection
# ================================================================================================

"""
    batch_acr_check(proj::ColumnSpanProjector, n_metabolites::Int) -> BitVector

Batch check which unit vectors eᵢ are in the column span.
Returns a BitVector where result[i] = true if eᵢ ∈ im(Q).

For ACR: eᵢ ∈ im(Y∆) means metabolite i has ACR.

Mathematical basis:
- ||P eᵢ - eᵢ||² = 1 - ||Q[i,:]||²
- So eᵢ ∈ im(Y∆) ⟺ ||Q[i,:]||² ≈ 1
"""
function batch_acr_check(proj::ColumnSpanProjector, n_metabolites::Int; tol::Float64=1e-8)
    result = falses(n_metabolites)

    if proj.rank == 0
        return result
    end

    Q = proj.Q
    k = proj.rank
    tol_sq = tol^2

    @inbounds for i in 1:n_metabolites
        # ||Q[i,:]||² = sum of squares of i-th row of Q
        row_norm_sq = 0.0
        for j in 1:k
            row_norm_sq += Q[i, j]^2
        end
        # eᵢ ∈ span ⟺ ||P eᵢ - eᵢ||² = 1 - row_norm_sq < tol²
        result[i] = (1.0 - row_norm_sq) < tol_sq
    end

    return result
end

"""
    batch_acrr_check(proj::ColumnSpanProjector, n_metabolites::Int) -> Vector{Tuple{Int,Int}}

Batch check which pairs (i,j) have eᵢ - eⱼ in the column span.
Returns vector of (i,j) pairs where i < j.

For ACRR: (eᵢ - eⱼ) ∈ im(Y∆) means metabolites i,j have ACRR.

Mathematical basis:
- ||P(eᵢ - eⱼ) - (eᵢ - eⱼ)||² = ||c||² - 2(Qc)ᵢ + 2(Qc)ⱼ + 2
  where c = Q^T(eᵢ - eⱼ) = Q[i,:] - Q[j,:]
"""
function batch_acrr_check(proj::ColumnSpanProjector, n_metabolites::Int; tol::Float64=1e-8)
    pairs = Tuple{Int,Int}[]

    if proj.rank == 0
        return pairs
    end

    Q = proj.Q
    k = proj.rank
    tol_sq = tol^2

    # Precompute Q row norms for filtering candidates
    row_norms_sq = zeros(Float64, n_metabolites)
    @inbounds for i in 1:n_metabolites
        for j in 1:k
            row_norms_sq[i] += Q[i, j]^2
        end
    end

    # Only check pairs where at least one has non-trivial projection
    # (if both have row_norm ≈ 0, their difference can't be in span unless it's zero)
    candidates = findall(row_norms_sq .> tol_sq)

    # Check all pairs involving candidates
    coeff_diff = Vector{Float64}(undef, k)

    for i in candidates
        for j in (i+1):n_metabolites
            # c = Q[i,:] - Q[j,:]
            @inbounds for l in 1:k
                coeff_diff[l] = Q[i, l] - Q[j, l]
            end

            # ||c||²
            coeff_norm_sq = sum(abs2, coeff_diff)

            # (Qc)ᵢ = Q[i,:] · c, (Qc)ⱼ = Q[j,:] · c
            Qc_i = 0.0
            Qc_j = 0.0
            @inbounds for l in 1:k
                Qc_i += Q[i, l] * coeff_diff[l]
                Qc_j += Q[j, l] * coeff_diff[l]
            end

            # ||P(eᵢ - eⱼ) - (eᵢ - eⱼ)||² = ||c||² - 2(Qc_i - Qc_j) + 2
            residual_sq = coeff_norm_sq - 2.0 * (Qc_i - Qc_j) + 2.0

            if residual_sq < tol_sq
                push!(pairs, (i, j))
            end
        end
    end

    return pairs
end

"""
    parallel_acrr_check(proj::ColumnSpanProjector, n_metabolites::Int; tol::Float64=1e-8) -> Vector{Tuple{Int,Int}}

Parallel version of ACRR check for large networks.
"""
function parallel_acrr_check(proj::ColumnSpanProjector, n_metabolites::Int; tol::Float64=1e-8)
    if proj.rank == 0
        return Tuple{Int,Int}[]
    end

    Q = proj.Q
    k = proj.rank
    tol_sq = tol^2

    # Precompute Q row norms for filtering
    row_norms_sq = zeros(Float64, n_metabolites)
    @inbounds for i in 1:n_metabolites
        for j in 1:k
            row_norms_sq[i] += Q[i, j]^2
        end
    end

    candidates = findall(row_norms_sq .> tol_sq)

    # Build pairs to check
    pair_list = Tuple{Int,Int}[]
    for i in candidates
        for j in (i+1):n_metabolites
            push!(pair_list, (i, j))
        end
    end

    if isempty(pair_list)
        return Tuple{Int,Int}[]
    end

    # Parallel check with thread-local results
    results = [Tuple{Int,Int}[] for _ in 1:Threads.maxthreadid()]
    workspaces = ThreadWorkspaces(() -> Vector{Float64}(undef, k))

    Threads.@threads for idx in 1:length(pair_list)
        i, j = pair_list[idx]
        tid = Threads.threadid()
        coeff_diff = workspaces.buffers[tid]

        @inbounds for l in 1:k
            coeff_diff[l] = Q[i, l] - Q[j, l]
        end

        coeff_norm_sq = sum(abs2, coeff_diff)

        Qc_i = 0.0
        Qc_j = 0.0
        @inbounds for l in 1:k
            Qc_i += Q[i, l] * coeff_diff[l]
            Qc_j += Q[j, l] * coeff_diff[l]
        end

        residual_sq = coeff_norm_sq - 2.0 * (Qc_i - Qc_j) + 2.0

        if residual_sq < tol_sq
            push!(results[tid], (i, j))
        end
    end

    # Merge thread-local results
    return reduce(vcat, results; init=Tuple{Int,Int}[])
end
