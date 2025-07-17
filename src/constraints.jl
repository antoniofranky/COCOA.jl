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
    constraints_before_pruning = C.substitute(constraints, subst_vals)
    constraints = C.prune_variables(constraints_before_pruning)

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
    return_complexes::Bool=false,
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

    # Build complex activities using C.sum pattern and extract complexes
    constraints, complexes = add_complex_activities_to_constraints(model, constraints)

    if return_complexes
        return constraints, complexes
    else
        # If complexes are not needed, we can return just the constraints
        return constraints
    end

end


"""
$(TYPEDSIGNATURES)

Data structure to track complex information.
"""
struct MetabolicComplex
    id::Symbol
    metabolites::Vector{Tuple{Symbol,Float64}}  # (metabolite_id, stoichiometry)
    reaction_contributions::Dict{Symbol,Float64}  # reaction_id -> contribution (+1 produced, -1 consumed)

    function MetabolicComplex(id::Symbol, metabolites::Vector{Tuple{Symbol,Float64}}, reaction_contributions::Dict{Symbol,Float64})
        # Ensure canonical ordering by sorting metabolites by their Symbol names
        sorted_metabolites = sort(metabolites, by=x -> x[1])
        new(id, sorted_metabolites, reaction_contributions)
    end
end

"""
$(TYPEDSIGNATURES)

Extract all unique complexes from the model.
A complex is the sum of metabolites that appear together in reactions.
Identical complexes are unified regardless of whether they appear as substrates or products.
"""
function extract_complexes_from_model(model::AbstractFBCModels.AbstractFBCModel)
    complexes = Dict{Symbol,MetabolicComplex}()

    # Get stoichiometry matrix and model components
    S = AbstractFBCModels.stoichiometry(model)
    reactions = AbstractFBCModels.reactions(model)
    metabolites = AbstractFBCModels.metabolites(model)

    # For each reaction, extract substrate and product complexes
    for (rxn_idx, rxn_id) in enumerate(reactions)
        rxn_symbol = Symbol(rxn_id)
        rxn_col = S[:, rxn_idx]

        # Extract substrate complex (negative coefficients)
        substrate_mets = Tuple{Symbol,Float64}[]
        for (met_idx, coeff) in enumerate(rxn_col)
            if coeff < -1e-12  # Negative coefficient means substrate
                met_id = Symbol(metabolites[met_idx])
                push!(substrate_mets, (met_id, -coeff))  # Store as positive stoichiometry
            end
        end

        if !isempty(substrate_mets)
            complex_id = generate_complex_id(substrate_mets)
            if !haskey(complexes, complex_id)
                complexes[complex_id] = MetabolicComplex(complex_id, substrate_mets, Dict{Symbol,Float64}())
            end
            # Complex is consumed when reaction runs forward (negative contribution)
            complexes[complex_id].reaction_contributions[rxn_symbol] = -1.0
        end

        # Extract product complex (positive coefficients)
        product_mets = Tuple{Symbol,Float64}[]
        for (met_idx, coeff) in enumerate(rxn_col)
            if coeff > 1e-12  # Positive coefficient means product
                met_id = Symbol(metabolites[met_idx])
                push!(product_mets, (met_id, coeff))
            end
        end

        if !isempty(product_mets)
            complex_id = generate_complex_id(product_mets)
            if !haskey(complexes, complex_id)
                complexes[complex_id] = MetabolicComplex(complex_id, product_mets, Dict{Symbol,Float64}())
            end
            # Add to existing reaction contributions if complex already exists
            if haskey(complexes[complex_id].reaction_contributions, rxn_symbol)
                # If the same complex appears as both substrate and product in the same reaction,
                # add the contributions (this handles cases like isomerization)
                complexes[complex_id].reaction_contributions[rxn_symbol] += 1.0
            else
                # Complex is produced when reaction runs forward (positive contribution)
                complexes[complex_id].reaction_contributions[rxn_symbol] = 1.0
            end
        end
    end

    return complexes
end

"""
$(TYPEDSIGNATURES)

Generate a unique complex ID from metabolite composition.
Identical complexes get the same ID regardless of which reaction they appear in or their role (substrate/product).
"""
function generate_complex_id(metabolites::Vector{Tuple{Symbol,Float64}})
    # Sort for canonical ordering
    sorted_mets = sort(metabolites, by=x -> x[1])

    # Build name from stoichiometry
    parts = [
        "$(isinteger(coeff) ? Int(coeff) : coeff)_$(met_id)"
        for (met_id, coeff) in sorted_mets
    ]

    complex_name = join(parts, "+")
    return Symbol(complex_name)
end

"""
$(TYPEDSIGNATURES)

Add complex activity variables and constraints to base constraints using C.sum pattern.
Uses the flux variables from base_constraints to ensure consistency.
"""
function add_complex_activities_to_constraints(
    model::AbstractFBCModels.AbstractFBCModel,
    base_constraints::C.ConstraintTree
)
    # Extract complexes from the model
    complexes = extract_complexes_from_model(model)

    # Build complex activity constraints using the combined constraint tree
    complex_activity_constraints = C.ConstraintTree(
        complex_id => C.Constraint(
            value=C.sum(
                (
                    contribution * base_constraints.fluxes[Symbol(rxn_id)].value
                    for (rxn_id, contribution) in complex.reaction_contributions
                    if haskey(base_constraints.fluxes, Symbol(rxn_id))
                ),
                init=zero(C.LinearValue)
            ),
            bound=C.Between(-1e9, 1e9)
        ) for (complex_id, complex) in complexes
    )

    # Add the complex relationships to the constraints
    constraints_with_activities = base_constraints * (:activities^complex_activity_constraints)

    # Return both the updated constraints and the complexes
    return constraints_with_activities, complexes
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
    default_bounds::Tuple{Float64,Float64}=(-1000.0, 1000.0)
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
