"""
Constraint building functionality for COCOA.

This module contains:
- Complex and incidence matrix extraction
- Constraint tree building for concordance analysis
- Unidirectional constraint creation
- Concordance test constraint building
"""

"""
$(TYPEDSIGNATURES)

Create unidirectional constraints for concordance analysis by splitting reactions into forward and reverse fluxes.
Uses COBREXA's symmetric approach with variable substitution and pruning.

# Arguments
- `model`: FBC model

# Returns
- Modified constraint tree with unidirectional variables for all reactions
- Set of reaction indices that were split (for downstream analysis)
"""
function create_unidirectional_constraints(
    model::A.AbstractFBCModel
)
    # Use symmetric approach following COBREXA documentation exactly
    constraints = COBREXA.flux_balance_constraints(model)

    # Add forward and reverse flux variables
    constraints += COBREXA.sign_split_variables(
        constraints.fluxes,
        positive=:fluxes_forward,
        negative=:fluxes_reverse
    )

    # Simplify the system by removing original variables (following COBREXA documentation)
    # Contrary to the documentation we only use variables for reactions that have flux bounds
    # This avoids unnecessary variables for reactions that are already irreversible
    subst_vals = [C.variable(; idx).value for idx = 1:C.var_count(constraints)]

    constraints.fluxes = C.zip(constraints.fluxes, constraints.fluxes_forward, constraints.fluxes_reverse) do f, p, n
        (var_idx,) = f.value.idxs
        if f.bound isa C.Between && f.bound.lower >= 0.0 && f.bound.upper >= 0.0
            subst_value = p.value  # Irreversible forward reaction
            @debug("Irreversible forward reaction detected for variable index $var_idx")
        elseif f.bound isa C.Between && f.bound.lower <= 0.0 && f.bound.upper <= 0.0
            subst_value = -n.value  # Irreversible reverse reaction
            @debug("Irreversible reverse reaction detected for variable index $var_idx")
        else
            subst_value = p.value - n.value
        end
        subst_vals[var_idx] = subst_value
        C.Constraint(subst_value) # the bidirectional bound is dropped here
    end

    # Filter out EqualTo(0.0) constraints from split fluxes before linking
    # This removes artificial reverse/forward variables for irreversible reactions
    # when pruning
    constraints.fluxes_forward = C.filter_leaves(constraints.fluxes_forward) do constraint
        !(constraint.bound isa C.EqualTo && constraint.bound.equal_to == 0.0)
    end
    constraints.fluxes_reverse = C.filter_leaves(constraints.fluxes_reverse) do constraint
        !(constraint.bound isa C.EqualTo && constraint.bound.equal_to == 0.0)
    end

    constraints = C.prune_variables(C.substitute(constraints, subst_vals))



    @info "Using symmetric unidirectional approach with variable pruning" var_count = C.var_count(constraints)

    # Count reactions that were split (all of them in this symmetric approach)
    reactions = A.reactions(model)
    n_reactions = length(reactions)
    all_indices = Set(1:n_reactions)

    return constraints, all_indices
end

"""
$(TYPEDSIGNATURES)

Memory-efficient concordance constraints that work with large models.

Builds constraint trees for concordance analysis following COBREXA patterns.
For custom constraints (like objective bounds), build constraints with this function
then merge additional constraints using the `*` operator before passing to analysis.

# Arguments
- `model::A.AbstractFBCModel`: Metabolic model to analyze
- `return_complexes::Bool=false`: Return complex information along with constraints
- `interface=nothing`: Interface specification (forwarded to COBREXA)
- `use_unidirectional_constraints::Bool=true`: Use unidirectional flux splitting

# Returns
- Constraint tree with balance, activities, and Charnes-Cooper templates
- Optionally: complex information (if `return_complexes=true`)

# Examples
```julia
# Basic usage
constraints = concordance_constraints(model)

# With objective bound (COBREXA pattern with * merge)
constraints = concordance_constraints(model)
obj_flux = COBREXA.optimized_values(
    constraints.balance;
    objective=constraints.balance.objective.value,
    output=constraints.balance.objective,
    optimizer=HiGHS.Optimizer
)
constraints.balance *= :objective_bound^COBREXA.C.Constraint(
    constraints.balance.objective.value,
    COBREXA.relative_tolerance_bound(0.999)(obj_flux)
)
# Now pass modified constraints to analysis...
```
"""
function concordance_constraints(
    model::A.AbstractFBCModel;
    modifications=Function[],
    return_complexes::Bool=false,
    interface=nothing,
    use_unidirectional_constraints::Bool=false,
)
    # TODO: Apply modifications correctly COBREXA style (not implemented yet)
    if use_unidirectional_constraints
        constraints, split_indices = create_unidirectional_constraints(model)
    else
        constraints = COBREXA.flux_balance_constraints(model; interface)
        split_indices = Set{Int}()
    end

    balance_constraints = constraints

    # Build complex activities using C.sum pattern and extract complexes
    activities, complexes_info = extract_activities_from_constraints(constraints)

    # Create Charnes-Cooper templates for both directions
    pos_template = create_charnes_cooper_template(balance_constraints, :positive)
    neg_template = create_charnes_cooper_template(balance_constraints, :negative)

    # Structure the final constraints tree by composing its parts
    final_constraints = C.ConstraintTree(
        :balance => balance_constraints,
        :activities => activities,
        :charnes_cooper => C.ConstraintTree(
            :positive => pos_template,
            :negative => neg_template
        )
    )

    if return_complexes
        return final_constraints, complexes_info
    else
        return final_constraints
    end

end

"""
$(TYPEDSIGNATURES)

Create a Charnes-Cooper template for a specific direction with balance constraints and scaled flux bounds.
"""
function create_charnes_cooper_template(
    base_constraints::C.ConstraintTree,
    direction::Symbol;
)
    # Start with a copy of base constraints (includes stoichiometry, etc.)
    template_constraints = deepcopy(base_constraints)

    # Add t variable
    if direction == :positive
        template_constraints += :t^C.variable(bound=C.Between(0, 999))
    else # direction == :negative
        template_constraints += :t^C.variable(bound=C.Between(-999, 0))
    end

    # Replace flux bounds with scaled bounds
    if haskey(template_constraints, :fluxes_forward)
        template_constraints.fluxes_forward = apply_charnes_cooper_scaling(
            template_constraints.fluxes_forward, template_constraints.t.value, direction
        )
    end
    if haskey(template_constraints, :fluxes_reverse)
        template_constraints.fluxes_reverse = apply_charnes_cooper_scaling(
            template_constraints.fluxes_reverse, template_constraints.t.value, direction
        )
    elseif haskey(template_constraints, :fluxes)
        template_constraints.fluxes = apply_charnes_cooper_scaling(
            template_constraints.fluxes, template_constraints.t.value, direction
        )
    end

    return template_constraints
end

"""
$(TYPEDSIGNATURES)

Apply Charnes-Cooper scaling to flux constraints with direction-aware bound handling.
This replaces the complex method overwriting with a single focused function.
"""
function apply_charnes_cooper_scaling(
    flux_constraints::C.ConstraintTree,
    t_value::C.Value,
    direction::Symbol
)
    return C.ConstraintTree(
        k => apply_charnes_cooper_scaling_to_constraint(v, t_value, direction)
        for (k, v) in flux_constraints
    )
end

"""
$(TYPEDSIGNATURES)

Apply Charnes-Cooper scaling to a single constraint with directional bound handling.
"""
function apply_charnes_cooper_scaling_to_constraint(
    constraint::C.Constraint,
    t_value::C.Value,
    direction::Symbol
)
    bound = constraint.bound
    x = constraint.value

    if bound isa C.Between
        if direction == :negative
            # NEGATIVE CASE: For t <= 0, inequalities flip: v_max*t <= w <= v_min*t
            bounds = [
                bound.upper < Inf ? (:lower => C.Constraint(x - bound.upper * t_value, (0, Inf))) : nothing,
                bound.lower > -Inf ? (:upper => C.Constraint(x - bound.lower * t_value, (-Inf, 0))) : nothing,
            ]
            return C.ConstraintTree(b for b in bounds if !isnothing(b))
        else
            # POSITIVE CASE (default)
            bounds = [
                bound.lower > -Inf ? (:lower => C.Constraint(x - bound.lower * t_value, (0, Inf))) : nothing,
                bound.upper < Inf ? (:upper => C.Constraint(x - bound.upper * t_value, (-Inf, 0))) : nothing,
            ]
            return C.ConstraintTree(b for b in bounds if !isnothing(b))
        end
    elseif bound isa C.EqualTo
        # For equality constraints, direction doesn't matter
        return COBREXA.value_scaled_bound_constraint(x, bound, t_value)
    else
        # Fallback to COBREXA's original implementation for other bound types
        return COBREXA.value_scaled_bound_constraint(x, bound, t_value)
    end
end

"""
$(TYPEDSIGNATURES)

Extract unique metabolite complexes from constraints with proper stoichiometry handling.
Identifies unique metabolite complexes (substrate or product combinations) from flux stoichiometry.

# Arguments
- `constraints`: ConstraintTree with flux_stoichiometry constraints

# Returns
- `complex_info`: Dictionary mapping complex names to their metabolite compositions
- `reaction_metabolite_map`: Dictionary mapping reaction indices to (substrates, products) tuples
"""
function extract_complexes(constraints::C.ConstraintTree)
    # Step 1: Build reaction metabolite profiles with stoichiometry
    reaction_metabolite_map = Dict{Int,Tuple{Vector{Tuple{Symbol,Float64}},Vector{Tuple{Symbol,Float64}}}}()

    # Access flux_stoichiometry from the correct location (either direct or nested in balance)
    flux_stoich = haskey(constraints, :flux_stoichiometry) ? constraints.flux_stoichiometry : constraints.balance.flux_stoichiometry

    for (metabolite, constraint) in flux_stoich
        var_indices = constraint.value.idxs
        coefficients = constraint.value.weights

        for (var_idx, coeff) in zip(var_indices, coefficients)
            if !haskey(reaction_metabolite_map, var_idx)
                reaction_metabolite_map[var_idx] = (Tuple{Symbol,Float64}[], Tuple{Symbol,Float64}[])
            end

            substrates, products = reaction_metabolite_map[var_idx]
            if coeff < 0
                push!(substrates, (metabolite, -coeff))  # Store positive stoichiometry
            elseif coeff > 0
                push!(products, (metabolite, coeff))
            end
        end
    end

    # Step 2: Canonicalize metabolite compositions (sort for consistency)
    for (var_idx, (substrates, products)) in reaction_metabolite_map
        sort!(substrates, by=x -> x[1])
        sort!(products, by=x -> x[1])
        reaction_metabolite_map[var_idx] = (substrates, products)
    end

    # Step 3: Identify unique metabolite complexes - preserve insertion order
    # Use a Dict to track seen complexes while maintaining insertion order (Julia 1.5+)
    seen_complexes = Dict{Vector{Tuple{Symbol,Float64}},Bool}()

    # Iterate through variable indices in sorted order for deterministic complex ordering
    for var_idx in sort(collect(keys(reaction_metabolite_map)))
        substrates, products = reaction_metabolite_map[var_idx]
        if !isempty(substrates) && !haskey(seen_complexes, substrates)
            seen_complexes[substrates] = true
        end
        if !isempty(products) && !haskey(seen_complexes, products)
            seen_complexes[products] = true
        end
    end

    # Step 4: Generate complex IDs - preserving insertion order
    complex_info = Dict{Symbol,Vector{Tuple{Symbol,Float64}}}()
    for metabolite_composition in keys(seen_complexes)
        # Generate readable complex ID
        complex_id = generate_complex_id(metabolite_composition)
        complex_info[complex_id] = metabolite_composition
    end

    return complex_info, reaction_metabolite_map
end

"""
$(TYPEDSIGNATURES)

Extract complex activities directly from constraint system with proper stoichiometry handling.
Creates activity variables for each unique metabolite complex (substrate or product combination).

# Arguments
- `constraints`: ConstraintTree with flux_stoichiometry constraints

# Returns
- `activities_tree`: ConstraintTree with complex activity variables
- `complex_info`: Dictionary mapping complex names to their metabolite compositions
"""
function extract_activities_from_constraints(constraints::C.ConstraintTree)
    # Use shared function to extract complexes
    complex_info, reaction_metabolite_map = extract_complexes(constraints)

    # Build activities constraint tree directly using generator
    activity_pairs = Pair{Symbol,C.Constraint}[]

    # ConstraintTree will sort keys alphabetically regardless of insertion order
    # This is fine since activities are accessed by ID, not by index
    for complex_id in keys(complex_info)
        metabolite_composition = complex_info[complex_id]
        # Build activity as sum of reaction contributions
        reaction_contributions = Dict{Int,Float64}()

        for (var_idx, (substrates, products)) in reaction_metabolite_map
            if metabolite_composition == substrates
                reaction_contributions[var_idx] = -1.0  # Consumed
            elseif metabolite_composition == products
                reaction_contributions[var_idx] = 1.0   # Produced
            end
        end

        # Create constraint if there are contributions
        if !isempty(reaction_contributions)
            # Sort by variable index for consistent ordering
            sorted_vars = sort!(collect(reaction_contributions); by=first)
            var_indices = [var_idx for (var_idx, _) in sorted_vars]
            coefficients = [coeff for (_, coeff) in sorted_vars]
            activity_value = C.LinearValue(var_indices, coefficients)
            constraint = C.Constraint(
                value=activity_value,
                bound=C.Between(-1e9, 1e9)
            )
            push!(activity_pairs, complex_id => constraint)
        end
    end

    activities_tree = C.ConstraintTree(activity_pairs)


    return activities_tree, complex_info
end

# Note: generate_complex_id is defined in matrix_builders.jl and imported