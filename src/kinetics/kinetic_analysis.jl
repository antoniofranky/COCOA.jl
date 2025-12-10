# ================================================================================================
# Kinetic Module Analysis - Main Orchestrator
# ================================================================================================
# Main entry point for kinetic module analysis
# Coordinates upstream algorithm, coupling detection, and ACR identification
#
# Based on: "Kinetic modules are sources of concentration robustness in
#           biochemical networks" (Langary et al., Sci. Adv. 2025)

export kinetic_analysis

# Include all submodules
include("union_find.jl")
include("tarjan.jl")
include("linear_algebra.jl")
include("network.jl")
include("upstream.jl")
include("coupling.jl")
include("acr.jl")
include("deficiency.jl")

using LinearAlgebra
using SparseArrays

"""
    kinetic_analysis(
        concordance_modules::Vector{Set{Symbol}},
        model;
        min_module_size::Int=1,
        efficient::Bool=true,
        known_acr::Vector{Symbol}=Symbol[],
        max_iterations::Int=100
    ) -> NamedTuple

Perform kinetic module analysis on a metabolic model.

Implements the iterative refinement algorithm from Section S.4.1 with feedback loops:
1. Proposition S4-1: Coupling merges → Concordance merges
2. Remark S3-6: ACR identification → Enhanced coupling (only if `efficient=false`)
3. Theorem S4-6: If δₖ = 1, all non-terminal complexes are coupled (only if `efficient=false`)

# Arguments
- `concordance_modules`: Vector{Set{Symbol}} where:
  - `concordance_modules[1]` = balanced complexes (module 0)
  - `concordance_modules[2+]` = unbalanced concordance modules 1, 2, ...
- `model`: Metabolic model with fields: S, mets, complexes, A, Y
- `min_module_size`: Minimum size for returned kinetic modules (default: 1)
- `efficient`: If true, use fast algorithm; if false, use full matrix method (default: true)
- `known_acr`: Pre-known ACR metabolites for augmentation
- `max_iterations`: Maximum iterations for convergence (default: 100)

# Returns
NamedTuple with fields:
- `kinetic_modules`: Vector of kinetic module sets (sorted by size, largest first)
- `acr_metabolites`: Metabolites with Absolute Concentration Robustness
- `acrr_pairs`: Metabolite pairs with Absolute Concentration Ratio Robustness
- `stats`: Dictionary with analysis statistics

# Example
```julia
result = kinetic_analysis(concordance_modules, model; efficient=false)
println("Found \$(length(result.kinetic_modules)) kinetic modules")
println("ACR metabolites: \$(result.acr_metabolites)")

# Destructuring works too:
(; kinetic_modules, acr_metabolites) = result
```
"""
function kinetic_analysis(
    concordance_modules::Vector{Set{Symbol}},
    model;
    min_module_size::Int=1,
    efficient::Bool=true,
    known_acr::Vector{Symbol}=Symbol[],
    max_iterations::Int=100
)
    # Initialize statistics
    stats = Dict{Symbol,Any}(
        :iterations => 0,
        :n_merges => 0,
        :efficient => efficient,
        :initial_modules => length(concordance_modules) - 1
    )

    # Build immutable network representation
    network = ReactionNetwork(model)
    stats[:n_metabolites] = network.n_metabolites
    stats[:n_complexes] = network.n_complexes

    # Validate input
    if length(concordance_modules) < 2
        return (
            kinetic_modules = Set{Symbol}[],
            acr_metabolites = Symbol[],
            acrr_pairs = Tuple{Symbol,Symbol}[],
            stats = stats
        )
    end

    # Extract balanced and unbalanced modules (keep ALL including singletons)
    balanced = concordance_modules[1]
    unbalanced_modules = concordance_modules[2:end]
    current_concordance = copy(concordance_modules)

    # Check for δₖ = 1 early exit (only for !efficient)
    if !efficient && check_deficiency_one(current_concordance, network)
        kinetic_modules = apply_theorem_s4_6(Set{Symbol}[], current_concordance, network)
        acr_result = identify_acr_acrr(kinetic_modules, network; efficient=efficient)

        # Filter by min_module_size
        filter!(m -> length(m) >= min_module_size, kinetic_modules)
        sort!(kinetic_modules, by=length, rev=true)

        stats[:early_exit] = :deficiency_one
        return (
            kinetic_modules = kinetic_modules,
            acr_metabolites = acr_result.acr_metabolites,
            acrr_pairs = acr_result.acrr_pairs,
            stats = stats
        )
    end

    # Initialize known ACR for augmentation
    current_known_acr = Set{Symbol}(known_acr)

    # Main iterative loop
    kinetic_modules = Set{Symbol}[]

    for iteration in 1:max_iterations
        stats[:iterations] = iteration

        # Step 1: Compute upstream sets for each extended module
        upstream_sets = compute_upstream_sets(current_concordance, network)

        if isempty(upstream_sets)
            break
        end

        # Step 2: Build Y∆ and projector (with ACR augmentation if !efficient)
        Y_Delta = build_coupling_matrix(upstream_sets, network)

        if !efficient && !isempty(current_known_acr)
            # Augment with known ACR columns (Remark S3-6)
            acr_cols = build_acr_augmentation(current_known_acr, network)
            if size(acr_cols, 2) > 0
                Y_Delta = hcat(Y_Delta, acr_cols)
            end
        end

        projector = ColumnSpanProjector(Y_Delta)

        # Step 3: Merge coupled modules
        kinetic_modules = merge_coupled_modules(upstream_sets, network; projector=projector)

        # Step 4: Identify ACR/ACRR
        acr_result = identify_acr_acrr(kinetic_modules, network; efficient=efficient)
        newly_found_acr = setdiff(Set(acr_result.acr_metabolites), current_known_acr)

        # Step 5: Check for convergence or update
        if efficient || isempty(newly_found_acr)
            # Converged or efficient mode (single pass)
            break
        end

        # Update known ACR and continue
        union!(current_known_acr, newly_found_acr)

        # Update concordance modules based on merges (Proposition S4-1)
        new_concordance = apply_coupling_to_concordance(kinetic_modules, current_concordance)

        if new_concordance == current_concordance
            break  # No changes, converged
        end

        current_concordance = new_concordance
    end

    # Apply Theorem S4-6 if applicable (only for !efficient)
    if !efficient
        kinetic_modules = apply_theorem_s4_6(kinetic_modules, current_concordance, network)
    end

    # Final ACR/ACRR identification
    acr_result = identify_acr_acrr(kinetic_modules, network; efficient=efficient)

    # Filter by min_module_size and sort
    filter!(m -> length(m) >= min_module_size, kinetic_modules)
    sort!(kinetic_modules, by=length, rev=true)

    stats[:final_modules] = length(kinetic_modules)
    stats[:n_acr] = length(acr_result.acr_metabolites)
    stats[:n_acrr] = length(acr_result.acrr_pairs)

    return (
        kinetic_modules = kinetic_modules,
        acr_metabolites = acr_result.acr_metabolites,
        acrr_pairs = acr_result.acrr_pairs,
        stats = stats
    )
end

# ================================================================================================
# Helper Functions
# ================================================================================================

"""
    build_acr_augmentation(acr_metabolites::Set{Symbol}, network::ReactionNetwork) -> Matrix{Float64}

Build augmentation columns for known ACR metabolites (unit vectors eᵢ).
"""
function build_acr_augmentation(acr_metabolites::Set{Symbol}, network::ReactionNetwork)
    if isempty(acr_metabolites)
        return zeros(Float64, network.n_metabolites, 0)
    end

    columns = Vector{Float64}[]
    for m in acr_metabolites
        idx = findfirst(==(m), network.metabolite_ids)
        if idx !== nothing
            col = zeros(Float64, network.n_metabolites)
            col[idx] = 1.0
            push!(columns, col)
        end
    end

    if isempty(columns)
        return zeros(Float64, network.n_metabolites, 0)
    end

    return hcat(columns...)
end

"""
    apply_coupling_to_concordance(
        kinetic_modules::Vector{Set{Symbol}},
        concordance_modules::Vector{Set{Symbol}}
    ) -> Vector{Set{Symbol}}

Update concordance modules based on coupling merges (Proposition S4-1).
If complexes from different concordance modules are coupled, they become concordant.
"""
function apply_coupling_to_concordance(
    kinetic_modules::Vector{Set{Symbol}},
    concordance_modules::Vector{Set{Symbol}}
)
    if length(concordance_modules) < 2
        return concordance_modules
    end

    balanced = concordance_modules[1]
    unbalanced = concordance_modules[2:end]

    # Build mapping: complex -> concordance module index
    complex_to_module = Dict{Symbol,Int}()
    for (i, m) in enumerate(unbalanced)
        for c in m
            complex_to_module[c] = i
        end
    end

    # Check which concordance modules need merging based on kinetic coupling
    uf = UnionFind(1:length(unbalanced))

    for km in kinetic_modules
        module_indices = Set{Int}()
        for c in km
            idx = get(complex_to_module, c, 0)
            if idx > 0
                push!(module_indices, idx)
            end
        end

        # Merge all concordance modules that have complexes in the same kinetic module
        indices_list = collect(module_indices)
        for i in 2:length(indices_list)
            union!(uf, indices_list[1], indices_list[i])
        end
    end

    # Collect merged concordance modules
    groups = get_groups(uf)
    new_unbalanced = Set{Symbol}[]

    for (_, member_indices) in groups
        merged = Set{Symbol}()
        for idx in member_indices
            union!(merged, unbalanced[idx])
        end
        push!(new_unbalanced, merged)
    end

    return [balanced; new_unbalanced]
end
