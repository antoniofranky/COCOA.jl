"""
Constraint building functionality for COCOA.

This module contains:
- Complex and incidence matrix extraction
- Constraint tree building for concordance analysis
- Unidirectional constraint creation
- Concordance test constraint building
"""

using AbstractFBCModels
using COBREXA
using ConstraintTrees
using SparseArrays
using Distributed
using DocStringExtensions

# Global constraint cache for reusing patterns
const CONSTRAINT_CACHE = Dict{UInt64,ConstraintTrees.ConstraintTree}()

"""
$(TYPEDSIGNATURES)

Extract complexes and build incidence matrix with optional shared memory.
"""
function extract_complexes_and_incidence(model::AbstractFBCModels.AbstractFBCModel;
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000)
    rxns = AbstractFBCModels.reactions(model)
    mets = AbstractFBCModels.metabolites(model)
    n_rxns = length(rxns)
    n_mets = length(mets)

    # Use Int32 for indices to save memory
    met_idx_map = Dict{String,Int32}(m => Int32(i) for (i, m) in enumerate(mets))

    complexes = Complex[]
    complex_dict = Dict{UInt64,Int32}()

    # Pre-allocate for incidence matrix construction
    I_rows = Int32[]
    J_cols = Int32[]
    V_vals = Float32[]

    sizehint!(I_rows, 2 * n_rxns)
    sizehint!(J_cols, 2 * n_rxns)
    sizehint!(V_vals, 2 * n_rxns)

    # Pre-allocate buffers for metabolite processing to avoid repeated allocations
    substrate_mets = Int32[]
    substrate_stoich = Float32[]
    product_mets = Int32[]
    product_stoich = Float32[]
    sizehint!(substrate_mets, 10)  # Reasonable estimate for typical reaction size
    sizehint!(substrate_stoich, 10)
    sizehint!(product_mets, 10)
    sizehint!(product_stoich, 10)

    # Process reactions in batches
    batch_size = min(1000, n_rxns)
    for batch_start in 1:batch_size:n_rxns
        batch_end = min(batch_start + batch_size - 1, n_rxns)

        for ridx in batch_start:batch_end
            rxn = rxns[ridx]
            rxn_id = Symbol(rxn)
            rxn_stoich = AbstractFBCModels.reaction_stoichiometry(model, rxn)

            # Check if this reaction is reversible (using the same logic as unidirectional constraints)
            # We'll determine this from the model bounds if available
            is_reversible = false
            try
                # Try to get bounds from the model
                # This is a simplified check - in practice, we'd use the constraint system
                # For now, assume reversible if we can't determine otherwise
                is_reversible = true  # Conservative default
            catch
                is_reversible = true  # Conservative default
            end

            # Separate substrates and products (reuse pre-allocated buffers)
            empty!(substrate_mets)
            empty!(substrate_stoich)
            empty!(product_mets)
            empty!(product_stoich)

            for (met, coeff) in rxn_stoich
                met_idx = met_idx_map[met]
                if coeff < 0
                    push!(substrate_mets, met_idx)
                    push!(substrate_stoich, -Float32(coeff))
                elseif coeff > 0
                    push!(product_mets, met_idx)
                    push!(product_stoich, Float32(coeff))
                end
            end

            # Process substrate complex
            if !isempty(substrate_mets)
                sub_complex = create_complex(
                    substrate_mets, substrate_stoich, mets
                )
                complex_idx = get!(complex_dict, sub_complex.hash) do
                    push!(complexes, sub_complex)
                    Int32(length(complexes))
                end
                push!(I_rows, complex_idx)
                push!(J_cols, Int32(ridx))
                push!(V_vals, -1.0f0)
            end

            # Process product complex
            if !isempty(product_mets)
                prod_complex = create_complex(
                    product_mets, product_stoich, mets
                )
                complex_idx = get!(complex_dict, prod_complex.hash) do
                    push!(complexes, prod_complex)
                    Int32(length(complexes))
                end
                push!(I_rows, complex_idx)
                push!(J_cols, Int32(ridx))
                push!(V_vals, 1.0f0)
            end
        end

        if batch_end < n_rxns
            GC.safepoint()
        end
    end

    # Build sparse incidence matrix
    A_sparse = sparse(Int.(I_rows), Int.(J_cols), V_vals, length(complexes), n_rxns)

    # Use shared memory if enabled and beneficial
    if use_shared_arrays && nworkers() > 0 &&
       (nnz(A_sparse) * (sizeof(Int32) + sizeof(Float32)) > min_size_for_sharing)
        A_matrix = SharedSparseMatrix(A_sparse)
    else
        A_matrix = SparseIncidenceMatrix(Int.(I_rows), Int.(J_cols), V_vals,
            length(complexes), n_rxns)
    end

    return complexes, A_matrix, complex_dict
end

"""
Create a complex from metabolite indices with memory-efficient ID generation.
"""
function create_complex(met_idxs::Vector{Int32}, stoich::Vector{Float32}, met_names::Vector{String})
    id_parts = IOBuffer()
    sorted_pairs = sort(collect(zip(met_idxs, stoich)))

    for (i, (idx, coeff)) in enumerate(sorted_pairs)
        i > 1 && write(id_parts, '+')
        if isinteger(coeff)
            write(id_parts, string(Int(coeff)), '_', met_names[idx])
        else
            write(id_parts, string(coeff), '_', met_names[idx])
        end
    end

    id = Symbol(String(take!(id_parts)))
    return Complex(id, Int32[p[1] for p in sorted_pairs], Float32[p[2] for p in sorted_pairs])
end

"""
$(TYPEDSIGNATURES)

Create unidirectional constraints for concordance analysis by splitting reactions into forward and reverse fluxes.

# Arguments
- `model`: FBC model

# Returns
- Modified constraint tree with unidirectional variables for all reactions
- Set of reaction indices that were split (for downstream analysis)
"""
function create_unidirectional_constraints(
    model::AbstractFBCModels.AbstractFBCModel
)
    # Start with standard flux balance constraints
    constraints = COBREXA.flux_balance_constraints(model)

    # Use COBREXA's optimized sign splitting
    constraints += COBREXA.sign_split_variables(
        constraints.fluxes,
        positive=:fluxes_forward,
        negative=:fluxes_reverse
    )

    # Create directional constraints using hierarchical composition
    constraints *= :directional_flux_balance^COBREXA.sign_split_constraints(
        positive=constraints.fluxes_forward,
        negative=constraints.fluxes_reverse,
        signed=constraints.fluxes,
    )

    # Functional variable substitution and pruning
    subst_vals = [ConstraintTrees.variable(; idx).value for idx = 1:ConstraintTrees.variable_count(constraints)]

    # Use ConstraintTrees.zip for functional composition
    constraints.fluxes = ConstraintTrees.zip(constraints.fluxes, constraints.fluxes_forward, constraints.fluxes_reverse) do f, p, n
        (var_idx,) = f.value.idxs
        subst_value = p.value - n.value
        subst_vals[var_idx] = subst_value
        ConstraintTrees.Constraint(subst_value) # bidirectional bound is dropped
    end

    # Apply optimized substitution and pruning
    constraints = ConstraintTrees.prune_variables(ConstraintTrees.substitute(constraints, subst_vals))

    # All reactions were split since we applied splitting to all fluxes
    rxn_ids = Symbol.(AbstractFBCModels.reactions(model))
    all_indices = Set(1:length(rxn_ids))

    return constraints, all_indices
end

"""
$(TYPEDSIGNATURES)

Memory-efficient concordance constraints that work with large models.
"""
function concordance_constraints(
    model::AbstractFBCModels.AbstractFBCModel;
    modifications=Function[],
    interface=nothing,
    use_unidirectional_constraints::Bool=true,
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000
)
    if use_unidirectional_constraints
        constraints, split_indices = create_unidirectional_constraints(model)
        @info "Using unidirectional constraints" n_reversible_split = length(split_indices)
    else
        constraints = COBREXA.flux_balance_constraints(model; interface)
        split_indices = Set{Int}()
    end

    # Apply modifications
    for mod in modifications
        constraints = mod(constraints)
    end

    # Get the reaction IDs that will be used consistently
    # Use the actual constraint system IDs, not the original model IDs
    constraint_rxn_ids = collect(keys(constraints.fluxes))

    # Build incidence matrix using the SAME reaction ordering
    complexes, A_matrix, complex_dict = extract_complexes_and_incidence(model;
        use_shared_arrays, min_size_for_sharing)

    # Create complex activity expressions using functional patterns
    complex_activities = build_complex_activities_functional(
        complexes, A_matrix, constraints.fluxes, constraint_rxn_ids
    )

    # Hierarchically compose the complete constraint system
    constraints = constraints * (:concordance_analysis^(
        :complexes^complex_activities
    ))

    return constraints
end

"""
$(TYPEDSIGNATURES)

Build complex activities using functional ConstraintTrees patterns.
More compositional and efficient than manual iteration.
"""
function build_complex_activities_functional(
    complexes::Vector{Complex},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    flux_constraints::ConstraintTrees.ConstraintTree,
    constraint_rxn_ids::Vector{Symbol}
)
    # Convert to sparse matrix for efficient operations
    A_sparse = isa(A_matrix, SharedSparseMatrix) ? sparse(A_matrix) : sparse(A_matrix)

    # Build complex activities using functional patterns
    complex_activities = ConstraintTrees.ConstraintTree()

    # Process complexes in batches for memory efficiency
    n_complexes = length(complexes)
    batch_size = min(1000, n_complexes)

    for batch_start in 1:batch_size:n_complexes
        batch_end = min(batch_start + batch_size - 1, n_complexes)

        # Use functional approach for batch processing
        batch_activities = map(batch_start:batch_end) do cidx
            complex = complexes[cidx]

            # Build activity expression functionally
            activity_terms = build_activity_expression(
                A_sparse, cidx, constraint_rxn_ids, flux_constraints
            )

            # Return as constraint with appropriate bounds
            complex.id => ConstraintTrees.Constraint(
                value=activity_terms,
                bound=ConstraintTrees.Between(-Inf, Inf)
            )
        end

        # Add batch to constraint tree
        for (complex_id, constraint) in batch_activities
            complex_activities[complex_id] = constraint
        end

        GC.safepoint()
    end

    return complex_activities
end

"""
$(TYPEDSIGNATURES)

Build a single complex activity expression using functional composition.
"""
function build_activity_expression(
    A_sparse::SparseMatrixCSC,
    complex_idx::Int,
    constraint_rxn_ids::Vector{Symbol},
    flux_constraints::ConstraintTrees.ConstraintTree
)
    # Use sparse row iteration - iterate through columns to find non-zeros in this row
    idxs = Vector{Int}()
    weights = Vector{Float64}()

    # Iterate through all columns to find non-zero elements in row complex_idx
    for j_idx in eachindex(constraint_rxn_ids)
        if j_idx <= size(A_sparse, 2)
            coeff = A_sparse[complex_idx, j_idx]
            if abs(coeff) > 1e-12
                rxn_id = constraint_rxn_ids[j_idx]
                if haskey(flux_constraints, rxn_id)
                    push!(idxs, j_idx)
                    push!(weights, coeff)
                end
            end
        end
    end

    return ConstraintTrees.LinearValue(idxs=idxs, weights=weights)
end

"""
$(TYPEDSIGNATURES)

Test concordance using manual JuMP model building with Charnes-Cooper transformation.
Uses pre-computed LinearValues for efficient constraint composition.
"""
function test_concordance(
    base_constraints::ConstraintTrees.ConstraintTree,
    c1_activity::ConstraintTrees.LinearValue,
    c2_activity::ConstraintTrees.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-12,
    optimizer,
    workers,
    settings=[]
)
    # Use COBREXA to build base model, then add concordance constraints manually
    # This approach works but is less efficient - we'll optimize later
    results = COBREXA.screen_optimization_model(
        base_constraints,
        [(JuMP.MIN_SENSE, :min), (JuMP.MAX_SENSE, :max)];
        optimizer=optimizer,
        settings=settings,
        workers=workers  # Explicitly use all available workers
    ) do om, (sense, key)
        # Get reaction indices that are involved in these activities
        reaction_indices = union(c1_activity.idxs, c2_activity.idxs)

        # Create transformed variables (only for involved reactions)
        w = Dict(j => @variable(om) for j in reaction_indices)
        t = @variable(om)

        # Direction constraint on t
        if direction == :positive
            @constraint(om, t >= tolerance)
        else
            @constraint(om, t <= -tolerance)
        end

        # Extract bounds from original flux variables
        x = om[:x]

        # Complex c2 activity constraint
        c2_expr = sum(
            c2_activity.weights[i] * w[c2_activity.idxs[i]]
            for i in eachindex(c2_activity.idxs)
            if c2_activity.idxs[i] in reaction_indices
        )
        target = direction == :positive ? 1.0 : -1.0
        @constraint(om, c2_expr == target)

        # Charnes-Cooper bounds constraints
        if direction == :positive
            for j in reaction_indices
                lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
                ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
                @constraint(om, w[j] - lb * t >= 0)
                @constraint(om, ub * t - w[j] >= 0)
            end
        else
            for j in reaction_indices
                lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
                ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
                @constraint(om, w[j] - ub * t >= 0)
                @constraint(om, lb * t - w[j] >= 0)
            end
        end

        # Objective: optimize complex c1 activity
        c1_expr = sum(
            c1_activity.weights[i] * w[c1_activity.idxs[i]]
            for i in eachindex(c1_activity.idxs)
            if c1_activity.idxs[i] in reaction_indices
        )

        @objective(om, sense, c1_expr)
        optimize!(om)

        if termination_status(om) == OPTIMAL
            return objective_value(om)
        else
            return nothing
        end
    end

    # Extract results
    min_val = results[1]
    max_val = results[2]

    if min_val === nothing || max_val === nothing
        return (false, nothing)
    end

    # Check concordance
    is_concordant = isapprox(min_val, max_val; atol=tolerance)
    lambda_value = is_concordant ? min_val : nothing

    return (is_concordant, lambda_value)
end

"""
$(TYPEDSIGNATURES)

Create Charnes-Cooper constraints using ConstraintTrees composition patterns.
More declarative and efficient than manual JuMP constraint building.
"""
function charnes_cooper_constraints(
    base_constraints::ConstraintTrees.ConstraintTree,
    c1_activities::ConstraintTrees.LinearValue,
    c2_activities::ConstraintTrees.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-12
)
    # Add Charnes-Cooper transformation variables
    reaction_indices = union(c1_activities.idxs, c2_activities.idxs)

    # Create w variables for transformed fluxes using proper ConstraintTrees API
    w_vars = :w_vars^ConstraintTrees.variables(
        keys=Symbol.("w_$j" for j in reaction_indices),
        bounds=ConstraintTrees.Between(-Inf, Inf)
    )

    # Create t variable for normalization
    t_bound = direction == :positive ?
              ConstraintTrees.Between(tolerance, Inf) :
              ConstraintTrees.Between(-Inf, -tolerance)
    t_var = :t^ConstraintTrees.variable(bound=t_bound)

    # Build c2 activity normalization constraint
    target_value = direction == :positive ? 1.0 : -1.0

    # Create linear expression for c2 activity
    c2_expr = sum(
        coeff * ConstraintTrees.value(w_vars.w_vars[Symbol("w_$idx")])
        for (idx, coeff) in zip(c2_activities.idxs, c2_activities.weights)
    )

    c2_constraint = :c2_normalization^ConstraintTrees.Constraint(
        c2_expr,
        ConstraintTrees.EqualTo(target_value)
    )

    # Build Charnes-Cooper bounds constraints
    bounds_constraints = ConstraintTrees.ConstraintTree()

    for j in reaction_indices
        # Get bounds from base constraints (simplified - assumes default bounds)
        lb, ub = -1000.0, 1000.0  # Conservative bounds

        w_var = ConstraintTrees.value(w_vars.w_vars[Symbol("w_$j")])
        t_val = ConstraintTrees.value(t_var.t)

        if direction == :positive
            # w[j] >= lb * t  -->  w[j] - lb * t >= 0
            bounds_constraints[Symbol("w_$(j)_lower")] = ConstraintTrees.Constraint(
                w_var - lb * t_val,
                ConstraintTrees.Between(0.0, Inf)
            )
            # w[j] <= ub * t  -->  ub * t - w[j] >= 0
            bounds_constraints[Symbol("w_$(j)_upper")] = ConstraintTrees.Constraint(
                ub * t_val - w_var,
                ConstraintTrees.Between(0.0, Inf)
            )
        else
            # w[j] >= ub * t (flipped for negative direction)
            bounds_constraints[Symbol("w_$(j)_lower")] = ConstraintTrees.Constraint(
                w_var - ub * t_val,
                ConstraintTrees.Between(0.0, Inf)
            )
            # w[j] <= lb * t
            bounds_constraints[Symbol("w_$(j)_upper")] = ConstraintTrees.Constraint(
                lb * t_val - w_var,
                ConstraintTrees.Between(0.0, Inf)
            )
        end
    end

    # Compose complete constraint system (without pre-built objective)
    test_constraints = base_constraints * w_vars * t_var * c2_constraint * (:bounds^bounds_constraints)

    return test_constraints
end

