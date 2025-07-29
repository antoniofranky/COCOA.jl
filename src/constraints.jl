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
    objective_bound=nothing,
    optimizer=nothing,
    settings=[],
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

    # Add objective constraint if specified
    if objective_bound !== nothing
        if optimizer === nothing
            throw(ArgumentError("optimizer must be provided when using objective_bound"))
        end

        # Validate objective_bound is callable
        if !isa(objective_bound, Function)
            throw(ArgumentError("objective_bound must be a function that takes optimal objective value and returns a constraint bound"))
        end

        # Get optimal objective value first
        objective_flux = COBREXA.optimized_values(
            constraints;
            objective=constraints.objective.value,
            output=constraints.objective,
            optimizer,
            settings,
        )

        if objective_flux !== nothing
            # Add objective bound constraint to limit feasible space
            constraints = constraints * :objective_bound^C.Constraint(
                constraints.objective.value,
                objective_bound(objective_flux)
            )
        else
            @warn "Could not determine optimal objective value, skipping objective constraint"
        end
    end

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

    if return_complexes
        return final_constraints, complexes
    else
        return final_constraints
    end

end


"""
$(TYPEDSIGNATURES)

Data structure to track complex information in a canonical, deterministic format.
"""
struct MetabolicComplex
    id::Symbol
    metabolites::Vector{Tuple{Symbol,Float64}}
    reaction_contributions::Vector{Tuple{Symbol,Float64}}

    function MetabolicComplex(id::Symbol, metabolites::Vector{Tuple{Symbol,Float64}}, reaction_contributions::Dict{Symbol,Float64})
        # Ensure canonical ordering by sorting all components
        sorted_metabolites = sort(metabolites, by=x -> x[1])
        sorted_reactions = sort([(k, v) for (k, v) in reaction_contributions], by=x -> x[1])
        new(id, sorted_metabolites, sorted_reactions)
    end
end

# Define a custom equality method that is robust to floating-point noise.
import Base.:(==)
function ==(a::MetabolicComplex, b::MetabolicComplex)
    a.id != b.id && return false
    a.metabolites != b.metabolites && return false

    contribs_a = a.reaction_contributions
    contribs_b = b.reaction_contributions

    length(contribs_a) != length(contribs_b) && return false

    for i in 1:length(contribs_a)
        rxn_a, val_a = contribs_a[i]
        rxn_b, val_b = contribs_b[i]
        (rxn_a != rxn_b || !isapprox(val_a, val_b)) && return false
    end

    return true
end


"""
$(TYPEDSIGNATURES)

Efficiently and deterministically extract all unique complexes from the model.
"""
function extract_complexes_from_model(model::AbstractFBCModels.AbstractFBCModel)
    # Step 1: Accumulate reaction contributions for each unique complex.
    # The key is the canonical representation of the complex (a sorted vector of its metabolites).
    complex_data = Dict{Vector{Tuple{Symbol,Float64}},Dict{Symbol,Float64}}()

    S = AbstractFBCModels.stoichiometry(model)
    reactions = AbstractFBCModels.reactions(model)
    metabolites = AbstractFBCModels.metabolites(model)
    reaction_map = Dict(id => i for (i, id) in enumerate(reactions))

    # Iterate over sorted reactions to ensure deterministic processing
    for rxn_id in sort(reactions)
        rxn_idx = reaction_map[rxn_id]
        rxn_symbol = Symbol(rxn_id)
        rxn_col = S[:, rxn_idx]

        # Substrate side - trust all negative coefficients from the model
        substrate_mets = [(Symbol(metabolites[i]), -s) for (i, s) in enumerate(rxn_col) if s < 0]
        if !isempty(substrate_mets)
            canonical_mets = sort(substrate_mets, by=x -> x[1])
            reaction_contribs = get!(complex_data, canonical_mets, Dict{Symbol,Float64}())
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) - 1.0
        end

        # Product side - trust all positive coefficients from the model
        product_mets = [(Symbol(metabolites[i]), s) for (i, s) in enumerate(rxn_col) if s > 0]
        if !isempty(product_mets)
            canonical_mets = sort(product_mets, by=x -> x[1])
            reaction_contribs = get!(complex_data, canonical_mets, Dict{Symbol,Float64}())
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) + 1.0
        end
    end

    # Step 2: Build the final, immutable MetabolicComplex structs.
    complexes = Dict{Symbol,MetabolicComplex}()
    # The sort order of keys doesn't matter for correctness, but sorting makes debugging easier.
    for canonical_mets in sort(collect(keys(complex_data)); by=x -> string(generate_complex_id(x)))
        reaction_contribs = complex_data[canonical_mets]
        complex_id = generate_complex_id(canonical_mets)
        # The constructor handles the final conversion to the canonical struct form
        complexes[complex_id] = MetabolicComplex(complex_id, canonical_mets, reaction_contribs)
    end

    return complexes
end


"""
$(TYPEDSIGNATURES)

Generate a unique, canonical complex ID from metabolite composition.
"""
function generate_complex_id(metabolites::Vector{Tuple{Symbol,Float64}})
    # Sorting is required here to ensure that the ID is canonical.
    sorted_mets = sort(metabolites, by=x -> x[1])
    parts = ["$(isinteger(coeff) ? Int(coeff) : coeff)_$(met_id)" for (met_id, coeff) in sorted_mets]
    return Symbol(join(parts, "+"))
end

"""
$(TYPEDSIGNATURES)

Add complex activity variables and constraints to base constraints.
"""
function add_complex_activities_to_constraints(
    model::AbstractFBCModels.AbstractFBCModel,
    base_constraints::C.ConstraintTree
)
    complexes = extract_complexes_from_model(model)
    # Iterate over sorted complex IDs to ensure deterministic ConstraintTree construction
    sorted_complex_ids = sort(collect(keys(complexes)))

    complex_activity_constraints = C.ConstraintTree(
        complex_id => C.Constraint(
            value=C.sum(
                (
                    contribution * base_constraints.fluxes[Symbol(rxn_id)].value
                    for (rxn_id, contribution) in complexes[complex_id].reaction_contributions
                    if haskey(base_constraints.fluxes, Symbol(rxn_id))
                ),
                init=zero(C.LinearValue)
            ),
            bound=C.Between(-1e9, 1e9)
        ) for complex_id in sorted_complex_ids
    )

    constraints_with_activities = base_constraints * (:activities^complex_activity_constraints)
    return constraints_with_activities, complexes
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
        template_constraints.fluxes_forward = COBREXA.value_scaled_bound_constraints(
            template_constraints.fluxes_forward, template_constraints.t.value, direction
        )
    end
    if haskey(template_constraints, :fluxes_reverse)
        template_constraints.fluxes_reverse = COBREXA.value_scaled_bound_constraints(
            template_constraints.fluxes_reverse, template_constraints.t.value, direction
        )
    end

    return template_constraints
end

# Import both the singular and plural versions of the function
import COBREXA: value_scaled_bound_constraint, value_scaled_bound_constraints

# ------------------------------------------------------------------
# CORRECTED OVERLOADS START HERE
# ------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Overload of `value_scaled_bound_constraint` for `C.Between` bounds that
correctly handles the directionality of the Charnes-Cooper transformation.
"""
function COBREXA.value_scaled_bound_constraint(
    x::C.Value,
    b::C.Between,
    scale::C.Value,
    direction::Symbol,
)
    if direction == :negative
        # NEGATIVE CASE: For t <= 0, inequalities flip: v_max*t <= w <= v_min*t
        bounds = [
            b.upper < Inf ? (:lower => C.Constraint(x - b.upper * scale, (0, Inf))) : nothing,
            b.lower > -Inf ? (:upper => C.Constraint(x - b.lower * scale, (-Inf, 0))) : nothing,
        ]
        return C.ConstraintTree(b for b in bounds if !isnothing(b))
    else
        # POSITIVE CASE (default)
        bounds = [
            b.lower > -Inf ? (:lower => C.Constraint(x - b.lower * scale, (0, Inf))) : nothing,
            b.upper < Inf ? (:upper => C.Constraint(x - b.upper * scale, (-Inf, 0))) : nothing,
        ]
        return C.ConstraintTree(b for b in bounds if !isnothing(b))
    end
end

"""
$(TYPEDSIGNATURES)

Overload of `value_scaled_bound_constraint` for `C.EqualTo` bounds. The
directionality does not change the resulting equality constraint.
"""
function COBREXA.value_scaled_bound_constraint(
    x::C.Value,
    b::C.EqualTo,
    scale::C.Value,
    direction::Symbol, # The direction argument is present for dispatch consistency
)
    # The original 3-argument function is correct for equality, regardless of direction.
    return COBREXA.value_scaled_bound_constraint(x, b, scale)
end


# --- Recursive Wrapper Functions ---

# Wrapper for ConstraintTree: Recursively calls the function, passing the direction.
function COBREXA.value_scaled_bound_constraints(
    x::C.ConstraintTree,
    scale::C.Value,
    direction::Symbol,
)
    return C.ConstraintTree(
        k => v for
        (k, v) in (k => value_scaled_bound_constraints(v, scale, direction) for (k, v) in x) if
        !(v isa C.ConstraintTree) || !isempty(v)
    )
end

# Wrapper for Constraint: Calls the singular version with the direction.
function COBREXA.value_scaled_bound_constraints(
    x::C.Constraint,
    scale::C.Value,
    direction::Symbol,
)
    return value_scaled_bound_constraint(x.value, x.bound, scale, direction)
end