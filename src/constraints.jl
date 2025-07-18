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

    # --- MODIFIED: Avoided deepcopy for efficiency ---
    # Store the balance constraints for other uses (like variability analysis)
    # By constructing a new tree from existing parts, we avoid a deep copy.
    balance_constraints = constraints

    # Build complex activities using C.sum pattern and extract complexes
    constraints_with_activities, complexes = add_complex_activities_to_constraints(model, constraints)

    # Create Charnes-Cooper templates for both directions
    pos_template = create_charnes_cooper_template(balance_constraints, :positive)
    neg_template = create_charnes_cooper_template(balance_constraints, :negative)

    # Structure the final constraints tree by composing its parts
    final_constraints = C.ConstraintTree(
        :balance => balance_constraints,
        :activities => constraints_with_activities.activities,
        :charnes_cooper => C.ConstraintTree(
            :positive => pos_template,
            :negative => neg_template
        )
    )
    # --- END MODIFICATION ---

    if return_complexes
        return final_constraints, complexes
    else
        # If complexes are not needed, we can return just the constraints
        return final_constraints
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

Create bounds constraints for Charnes-Cooper transformation.
Implements the correct bounds: vmin*t ≤ w ≤ vmax*t for t ≥ 0, and vmax*t ≤ w ≤ vmin*t for t ≤ 0.
"""
function create_bounds_constraints(
    base_constraints::C.ConstraintTree,
    direction::Symbol,
)
    bounds = C.ConstraintTree()

    if direction == :positive
        # Create t_pos variable once
        t_pos = :t_pos^C.variable(bound=C.Between(1e-12, Inf))
        base_constraints += (:charnes_cooper^t_pos)
        t_var = base_constraints.charnes_cooper[:t_pos].value

        # Forward fluxes
        if haskey(base_constraints, :fluxes_forward)
            for (flux_name, flux_constraint) in base_constraints.fluxes_forward
                flux_var = flux_constraint.value
                bound = flux_constraint.bound

                if typeof(bound) <: C.Between
                    vmin = bound.lower
                    vmax = bound.upper
                elseif typeof(bound) <: C.EqualTo
                    vmin = bound.equal_to
                    vmax = bound.equal_to
                else
                    continue
                end

                bounds *= Symbol("upper_fwd_$(flux_name)")^C.Constraint(
                    flux_var - vmax * t_var,
                    C.Between(-Inf, 0.0)
                )
                bounds *= Symbol("lower_fwd_$(flux_name)")^C.Constraint(
                    flux_var - vmin * t_var,
                    C.Between(0.0, Inf)
                )
            end
        end

        # Reverse fluxes
        if haskey(base_constraints, :fluxes_reverse)
            for (flux_name, flux_constraint) in base_constraints.fluxes_reverse
                flux_var = flux_constraint.value
                bound = flux_constraint.bound

                if typeof(bound) <: C.Between
                    vmin = bound.lower
                    vmax = bound.upper
                elseif typeof(bound) <: C.EqualTo
                    vmin = bound.equal_to
                    vmax = bound.equal_to
                else
                    continue
                end


                bounds *= Symbol("upper_rev_$(flux_name)")^C.Constraint(
                    flux_var - vmax * t_var,
                    C.Between(-Inf, 0.0)
                )
                bounds *= Symbol("lower_rev_$(flux_name)")^C.Constraint(
                    flux_var - vmin * t_var,
                    C.Between(0.0, Inf)
                )
            end
        end

    else # direction == :negative
        # Create t_neg variable once
        t_neg = :t_neg^C.variable(bound=C.Between(1e-12, Inf))
        base_constraints += (:charnes_cooper^t_neg)
        t_var = base_constraints.charnes_cooper[:t_neg].value

        # Forward fluxes
        if haskey(base_constraints, :fluxes_forward)
            for (flux_name, flux_constraint) in base_constraints.fluxes_forward
                flux_var = flux_constraint.value
                bound = flux_constraint.bound

                if typeof(bound) <: C.Between
                    vmin = bound.lower
                    vmax = bound.upper
                elseif typeof(bound) <: C.EqualTo
                    vmin = bound.equal_to
                    vmax = bound.equal_to
                else
                    continue
                end

                # Note: bounds swap for negative t
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("upper_fwd_$(flux_name)")^C.Constraint(
                        flux_var - vmin * t_var,
                        C.Between(-Inf, 0.0)
                    )
                end
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("lower_fwd_$(flux_name)")^C.Constraint(
                        flux_var - vmax * t_var,
                        C.Between(0.0, Inf)
                    )
                end
            end
        end

        # Reverse fluxes
        if haskey(base_constraints, :fluxes_reverse)
            for (flux_name, flux_constraint) in base_constraints.fluxes_reverse
                flux_var = flux_constraint.value
                bound = flux_constraint.bound

                if typeof(bound) <: C.Between
                    vmin = bound.lower
                    vmax = bound.upper
                elseif typeof(bound) <: C.EqualTo
                    vmin = bound.equal_to
                    vmax = bound.equal_to
                else
                    continue
                end

                # Note: bounds swap for negative t
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("upper_rev_$(flux_name)")^C.Constraint(
                        flux_var - vmin * t_var,
                        C.Between(-Inf, 0.0)
                    )
                end
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("lower_rev_$(flux_name)")^C.Constraint(
                        flux_var - vmax * t_var,
                        C.Between(0.0, Inf)
                    )
                end
            end
        end
    end

    return bounds
end

"""
$(TYPEDSIGNATURES)

Instantiate the Charnes-Cooper template for a specific activity pair and direction.
This uses constraint substitution and pruning to create an efficient constraint
system for the specific concordance test.
"""
function instantiate_charnes_cooper(
    base_constraints::C.ConstraintTree,
    c1_activities::C.LinearValue,
    c2_activities::C.LinearValue,
    direction::Symbol;
    tolerance::Float64=1e-12
)
    # Create a mutable copy for this instance
    instantiated = copy(base_constraints)
    instantiated += :charnes_cooper^C.ConstraintTree()
    target_value = direction == :positive ? 1.0 : -1.0
    # Create normalization constraint: [Aw]j = ±1
    c2_constraint = :c2_normalization^C.Constraint(
        c2_activities,
        C.EqualTo(target_value)
    )

    # Create bounds constraints using the correct transformation
    bounds_constraints = create_bounds_constraints(instantiated, direction)


    # Add pair-specific constraints to template
    instantiated *= :charnes_cooper^c2_constraint * :charnes_cooper^:fluxes_transformed^bounds_constraints

    instantiated.fluxes_forward = C.ConstraintTree()
    instantiated.fluxes_reverse = C.ConstraintTree()

    # --- MODIFIED: Enabled variable pruning for efficiency ---
    # Pruning removes unused variables and constraints, leading to a much
    # smaller and faster optimization problem.
    optimized = C.prune_variables(instantiated)
    return optimized, c1_activities
end

"""
$(TYPEDSIGNATURES)

Create a Charnes-Cooper template for a specific direction without activity-specific constraints.
This creates the base constraint structure that can be reused for multiple concordance tests
by only modifying the objective and c_j normalization constraint.
The returned constraint tree contains:
- The base flux balance constraints
- Charnes-Cooper bounds constraints for the specified direction
- Empty charnes_cooper section for c_j constraint to be added by workers

This is more efficient than instantiate_charnes_cooper when the same direction
is used repeatedly with different activity pairs.
"""
function create_charnes_cooper_template(
    base_constraints::C.ConstraintTree,
    direction::Symbol;
)
    # Create a copy to avoid modifying the original
    template_constraints = deepcopy(base_constraints)
    template_constraints += :charnes_cooper^C.ConstraintTree()

    # Create bounds constraints using the correct transformation
    bounds_constraints = create_bounds_constraints(template_constraints, direction)

    # Add bounds constraints to template (but not the c_j constraint)
    template_constraints *= :charnes_cooper^:fluxes_transformed^bounds_constraints

    # Remove the original flux variables since they're replaced by transformed ones
    template_constraints.fluxes_forward = C.ConstraintTree()
    template_constraints.fluxes_reverse = C.ConstraintTree()

    return template_constraints
end
