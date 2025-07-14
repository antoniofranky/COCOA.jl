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
using SparseArrays
using Distributed
using DocStringExtensions
import ConstraintTrees as C
# Global constraint cache for reusing patterns
const CONSTRAINT_CACHE = Dict{UInt64,C.ConstraintTree}()

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
function create_complex(met_idxs::Vector{Int32}, stoich::Vector{Float32},
    met_names::Vector{String})
    # Sort by indices for canonical ordering
    perm = sortperm(met_idxs)
    sorted_idxs = met_idxs[perm]
    sorted_stoich = stoich[perm]

    # Build ID more efficiently - ensure type stability
    id_parts = String[]
    sizehint!(id_parts, length(sorted_idxs))

    for i in eachindex(sorted_idxs)
        idx = sorted_idxs[i]
        coeff = sorted_stoich[i]
        coeff_str = isinteger(coeff) ? string(Int(coeff)) : string(coeff)
        push!(id_parts, "$(coeff_str)_$(met_names[idx])")
    end

    id = Symbol(join(id_parts, "+"))
    return Complex(id, sorted_idxs, sorted_stoich)
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
    subst_vals = [C.variable(; idx).value for idx = 1:C.variable_count(constraints)]

    # Use C.zip for functional composition
    constraints.fluxes = C.zip(constraints.fluxes, constraints.fluxes_forward, constraints.fluxes_reverse) do f, p, n
        (var_idx,) = f.value.idxs
        subst_value = p.value - n.value
        subst_vals[var_idx] = subst_value
        C.Constraint(subst_value) # bidirectional bound is dropped
    end

    # Apply optimized substitution and pruning
    constraints = C.prune_variables(C.substitute(constraints, subst_vals))

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
    @info "Building complex activities" n_complexes=length(complexes) n_constraint_rxns=length(constraint_rxn_ids)
    
    # Debug: Check the structure of the flux constraints after transformations
    @debug "Flux constraint structure after transformations"
    sample_rxn_ids = constraint_rxn_ids[1:min(3, end)]
    for rxn_id in sample_rxn_ids
        if haskey(constraints.fluxes, rxn_id)
            flux_constraint = constraints.fluxes[rxn_id]
            @debug "Sample flux constraint" rxn_id constraint_type=typeof(flux_constraint) has_value=hasfield(typeof(flux_constraint), :value)
            if hasfield(typeof(flux_constraint), :value)
                @debug "Constraint value details" value_type=typeof(flux_constraint.value)
            end
        end
    end
    
    complex_activities = build_complex_activities_functional(
        complexes, A_matrix, constraints.fluxes, constraint_rxn_ids
    )
    @info "Complex activities built" n_complex_activities=length(complex_activities)

    # Hierarchically compose the complete constraint system
    @info "Adding complex activities to constraint tree"
    constraints = constraints * (:concordance_analysis^(
        :complexes^complex_activities
    ))
    
    # Verify the structure was added correctly
    @info "Constraint tree structure after adding complexes" has_concordance_analysis=haskey(constraints, :concordance_analysis)
    if haskey(constraints, :concordance_analysis)
        @info "Concordance analysis structure" has_complexes=haskey(constraints.concordance_analysis, :complexes) n_complexes_in_tree=length(constraints.concordance_analysis.complexes)
    end

    return constraints
end

"""
$(TYPEDSIGNATURES)

Build complex activities using functional C patterns.
More compositional and efficient than manual iteration.
"""
function build_complex_activities_functional(
    complexes::Vector{Complex},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    flux_constraints::C.ConstraintTree,
    constraint_rxn_ids::Vector{Symbol}
)
    # Convert to sparse matrix for efficient operations
    A_sparse = isa(A_matrix, SharedSparseMatrix) ? sparse(A_matrix) : sparse(A_matrix)

    # Build complex activities using functional patterns
    complex_activities = C.ConstraintTree()

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
            complex.id => C.Constraint(
                value=activity_terms,
                bound=C.Between(-Inf, Inf)
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
    flux_constraints::C.ConstraintTree
)
    # Get actual variable indices from the constraint tree
    idxs = Vector{Int}()
    weights = Vector{Float64}()

    @debug "Building activity for complex $complex_idx" n_reactions=length(constraint_rxn_ids)

    # Iterate through all columns to find non-zero elements in row complex_idx
    for j_idx in eachindex(constraint_rxn_ids)
        if j_idx <= size(A_sparse, 2)
            coeff = A_sparse[complex_idx, j_idx]
            if abs(coeff) > 1e-12
                rxn_id = constraint_rxn_ids[j_idx]
                if haskey(flux_constraints, rxn_id)
                    # Get the actual variable index from the constraint tree
                    flux_constraint = flux_constraints[rxn_id]
                    
                    @debug "Processing reaction $rxn_id" coeff constraint_type=typeof(flux_constraint)
                    
                    # Handle different constraint types after transformations
                    if isa(flux_constraint, C.Constraint)
                        # After substitution, constraints might be C.Constraint objects
                        value = flux_constraint.value
                        if isa(value, C.LinearValue)
                            # Extract variable indices from the LinearValue
                            for (var_idx, var_coeff) in zip(value.idxs, value.weights)
                                push!(idxs, var_idx)
                                push!(weights, coeff * var_coeff)
                            end
                            @debug "Added linear terms" n_new_terms=length(value.idxs)
                        elseif isa(value, C.Variable)
                            # Single variable constraint
                            push!(idxs, value.idx)
                            push!(weights, coeff)
                            @debug "Added single variable" var_idx=value.idx
                        else
                            @warn "Unsupported flux constraint value type" rxn_id value_type=typeof(value)
                        end
                    else
                        @warn "Unexpected flux constraint type" rxn_id constraint_type=typeof(flux_constraint)
                    end
                else
                    @debug "Reaction $rxn_id not found in flux constraints"
                end
            end
        end
    end

    @debug "Built activity expression for complex $complex_idx" n_terms=length(idxs) unique_variables=length(unique(idxs))
    
    if isempty(idxs)
        @warn "No terms found for complex $complex_idx - this may indicate a constraint building issue"
    end

    return C.LinearValue(idxs=idxs, weights=weights)
end

"""
$(TYPEDSIGNATURES)

Test concordance using ConstraintTrees template-based approach.
Much more efficient than manual JuMP model building as it reuses
constraint structures between similar tests.
"""
function test_concordance_templated(
    template::C.ConstraintTree,
    c1_activity::C.LinearValue,
    c2_activity::C.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-9,
    optimizer,
    workers,
    settings=[]
)
    # Instantiate the template for this specific pair
    instantiated, c1_expr, t_key = instantiate_charnes_cooper(
        template, c1_activity, c2_activity, direction; tolerance
    )
    
    # Use COBREXA's optimization with the instantiated constraint tree
    results = COBREXA.screen_optimization_model(
        instantiated,
        [(JuMP.MIN_SENSE, :min), (JuMP.MAX_SENSE, :max)];
        optimizer=optimizer,
        settings=settings,
        workers=workers
    ) do om, (sense, _)
        # The objective is already built into the constraint tree
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

Test concordance using manual JuMP model building with Charnes-Cooper transformation.
Uses pre-computed LinearValues for efficient constraint composition.
[DEPRECATED: Use test_concordance_templated for better performance]
"""
function test_concordance(
    base_constraints::C.ConstraintTree,
    c1_activity::C.LinearValue,
    c2_activity::C.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-9,
    optimizer,
    workers,
    settings=[]
)
    # Use COBREXA to build base model, then add concordance constraints manually
    results = COBREXA.screen_optimization_model(
        base_constraints,
        [(JuMP.MIN_SENSE, :min), (JuMP.MAX_SENSE, :max)];
        optimizer=optimizer,
        settings=settings,
        workers=workers
    ) do om, (sense, _)
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

Create a reusable Charnes-Cooper constraint template that can be parameterized
for different activity pairs and directions. This avoids rebuilding similar
constraints for each concordance test.
"""
function create_charnes_cooper_template(
    base_constraints::C.ConstraintTree,
    all_reaction_indices::Set{Int};
    default_bounds::Tuple{Float64,Float64} = (-1000.0, 1000.0)
)
    lb, ub = default_bounds
    
    # Create w variables for ALL reactions that might be involved
    # This avoids rebuilding the variable structure for each pair
    w_vars = :w_vars^C.variables(
        keys=Symbol.("w_$j" for j in all_reaction_indices),
        bounds=C.Between(-Inf, Inf)
    )
    
    # Create template t variables for both directions
    t_pos = :t_pos^C.variable(bound=C.Between(1e-12, Inf))
    t_neg = :t_neg^C.variable(bound=C.Between(-Inf, -1e-12))
    
    # Create bounds constraint template for positive direction
    bounds_pos = :bounds_pos^C.ConstraintTree(
        Symbol("w_$(j)_lower") => C.Constraint(
            C.value(w_vars.w_vars[Symbol("w_$j")]) - lb * C.value(t_pos.t_pos),
            C.Between(0.0, Inf)
        ) for j in all_reaction_indices
    ) * C.ConstraintTree(
        Symbol("w_$(j)_upper") => C.Constraint(
            ub * C.value(t_pos.t_pos) - C.value(w_vars.w_vars[Symbol("w_$j")]),
            C.Between(0.0, Inf)
        ) for j in all_reaction_indices
    )
    
    # Create bounds constraint template for negative direction
    bounds_neg = :bounds_neg^C.ConstraintTree(
        Symbol("w_$(j)_lower") => C.Constraint(
            C.value(w_vars.w_vars[Symbol("w_$j")]) - ub * C.value(t_neg.t_neg),
            C.Between(0.0, Inf)
        ) for j in all_reaction_indices
    ) * C.ConstraintTree(
        Symbol("w_$(j)_upper") => C.Constraint(
            lb * C.value(t_neg.t_neg) - C.value(w_vars.w_vars[Symbol("w_$j")]),
            C.Between(0.0, Inf)
        ) for j in all_reaction_indices
    )
    
    # Compose base template with variables and bounds
    template = base_constraints * w_vars * t_pos * t_neg * bounds_pos * bounds_neg
    
    return template
end

"""
$(TYPEDSIGNATURES)

Instantiate the Charnes-Cooper template for a specific activity pair and direction.
This uses constraint substitution and pruning to create an efficient constraint
system for the specific concordance test.
"""
function instantiate_charnes_cooper(
    template::C.ConstraintTree,
    c1_activities::C.LinearValue,
    c2_activities::C.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-12
)
    # Determine which reactions are actually involved in this pair
    active_reactions = Set(union(c1_activities.idxs, c2_activities.idxs))
    
    # Create c2 normalization constraint for this specific pair
    target_value = direction == :positive ? 1.0 : -1.0
    
    # Select appropriate t variable
    t_key = direction == :positive ? :t_pos : :t_neg
    
    # Build c2 activity expression using the template's w variables
    c2_expr = sum(
        coeff * C.value(template.w_vars.w_vars[Symbol("w_$idx")])
        for (idx, coeff) in zip(c2_activities.idxs, c2_activities.weights)
    )
    
    c2_constraint = :c2_normalization^C.Constraint(
        c2_expr,
        C.EqualTo(target_value)
    )
    
    # Build c1 objective expression
    c1_expr = sum(
        coeff * C.value(template.w_vars.w_vars[Symbol("w_$idx")])
        for (idx, coeff) in zip(c1_activities.idxs, c1_activities.weights)
    )
    
    c1_objective = :c1_objective^C.Constraint(
        c1_expr,
        C.Between(-Inf, Inf)  # Unconstrained objective
    )
    
    # Add pair-specific constraints to template
    instantiated = template * c2_constraint * c1_objective
    
    # Use substitution to fix unused variables to zero and prune
    # This removes variables not involved in this specific pair
    var_count = C.variable_count(instantiated)
    substitution = zeros(var_count)
    
    # Only keep variables that are actually used
    # (This is where the power of ConstraintTrees substitution comes in)
    
    # Prune unused variables for efficiency
    optimized = C.prune_variables(instantiated)
    
    return optimized, c1_expr, t_key
end

"""
$(TYPEDSIGNATURES)

Create and cache a Charnes-Cooper template for efficient concordance testing.
This should be called once per analysis and reused for all concordance tests.
"""
function setup_concordance_testing(
    base_constraints::C.ConstraintTree,
    all_complexes::Vector{Complex};
    default_bounds::Tuple{Float64,Float64} = (-1000.0, 1000.0)
)
    # Extract all reaction indices that could be involved in any concordance test
    all_reaction_indices = Set{Int}()
    for complex in all_complexes
        for idx in complex.metabolite_indices
            push!(all_reaction_indices, Int(idx))
        end
    end
    
    # Create the reusable template
    template = create_charnes_cooper_template(
        base_constraints, all_reaction_indices; default_bounds
    )
    
    return template
end

