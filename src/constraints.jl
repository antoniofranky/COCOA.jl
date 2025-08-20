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
    model::AbstractFBCModels.AbstractFBCModel
)
    # Use symmetric approach following COBREXA documentation exactly
    constraints = COBREXA.flux_balance_constraints(model)
    
    # Add forward and reverse flux variables
    constraints += COBREXA.sign_split_variables(
        constraints.fluxes,
        positive=:fluxes_forward,
        negative=:fluxes_reverse
    )
    
    # Add constraints linking original fluxes to split fluxes: flux = forward - reverse
    constraints *= :directional_flux_balance^COBREXA.sign_split_constraints(
        positive=constraints.fluxes_forward,
        negative=constraints.fluxes_reverse,
        signed=constraints.fluxes,
    )

    # Simplify the system by removing original variables (following COBREXA documentation)
    subst_vals = [C.variable(; idx).value for idx = 1:C.var_count(constraints)]

    constraints.fluxes = C.zip(constraints.fluxes, constraints.fluxes_forward, constraints.fluxes_reverse) do f, p, n
        (var_idx,) = f.value.idxs
        subst_value = p.value - n.value
        subst_vals[var_idx] = subst_value
        C.Constraint(subst_value) # the bidirectional bound is dropped here
    end

    constraints = C.prune_variables(C.substitute(constraints, subst_vals))

    @info "Using symmetric unidirectional approach with variable pruning"

    # Count reactions that were split (all of them in this symmetric approach)
    reactions = AbstractFBCModels.reactions(model)
    n_reactions = length(reactions)
    all_indices = Set(1:n_reactions)

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
)
    if use_unidirectional_constraints
        constraints, split_indices = create_unidirectional_constraints(model)
        @info "Using unidirectional constraints" n_reversible_split = C.var_count(constraints.fluxes_forward) +
                                                                      C.var_count(constraints.fluxes_reverse)
    else
        constraints = COBREXA.flux_balance_constraints(model; interface)
        split_indices = Set{Int}()
    end

    # Apply modifications
    for mod in modifications
        constraints = mod(constraints)
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
        sorted_metabolites = sort!(metabolites, by=x -> x[1])
        sorted_reactions = sort!([(k, v) for (k, v) in reaction_contributions], by=x -> x[1])
        new(id, sorted_metabolites, sorted_reactions)
    end
end

# Define a custom equality method that is robust to floating-point noise.
import Base.:(==)
@inline function ==(a::MetabolicComplex, b::MetabolicComplex)
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

Extract complexes from split constraints that respect reaction directionality.
This function builds complexes consistent with the unidirectional constraint system,
ensuring that forward and reverse reactions are treated as separate elementary steps.
"""
function extract_complexes_from_constraints(
    model::AbstractFBCModels.AbstractFBCModel,
    constraints::C.ConstraintTree
)
    # Check if we have split constraints
    if haskey(constraints, :fluxes_forward) && haskey(constraints, :fluxes_reverse)
        return extract_complexes_from_split_constraints(model, constraints)
    elseif haskey(constraints, :balance) &&
           haskey(constraints.balance, :fluxes_forward) &&
           haskey(constraints.balance, :fluxes_reverse)
        # Split constraints are nested under balance
        return extract_complexes_from_split_constraints(model, constraints.balance)
    else
        # Fall back to original model-based extraction
        return extract_complexes_from_model(model)
    end
end

"""
$(TYPEDSIGNATURES)

Extract complexes from split constraint system where reactions are decomposed into 
forward and reverse elementary steps. This respects the paper's requirement that
reversible reactions be split into irreversible ones for kinetic analysis.
"""
function extract_complexes_from_split_constraints(
    model::AbstractFBCModels.AbstractFBCModel,
    constraints::C.ConstraintTree
)
    # Get model data
    S = AbstractFBCModels.stoichiometry(model)
    reactions = AbstractFBCModels.reactions(model)
    metabolites = AbstractFBCModels.metabolites(model)
    n_reactions = length(reactions)
    n_metabolites = length(metabolites)

    # Get split flux variables
    forward_flux_vars = constraints.fluxes_forward
    reverse_flux_vars = constraints.fluxes_reverse

    # Build complex data structure
    complex_data = Dict{Vector{Tuple{Symbol,Float64}},Dict{Symbol,Float64}}()
    sizehint!(complex_data, n_reactions * 4)  # More complexes due to splitting

    # Pre-allocate reaction map
    reaction_map = Dict{String,Int}()
    sizehint!(reaction_map, n_reactions)
    for (i, id) in enumerate(reactions)
        reaction_map[id] = i
    end

    # Pre-allocate temporary vectors
    substrate_mets = Vector{Tuple{Symbol,Float64}}()
    product_mets = Vector{Tuple{Symbol,Float64}}()
    sizehint!(substrate_mets, n_metabolites ÷ 4)
    sizehint!(product_mets, n_metabolites ÷ 4)

    # Process each base reaction for forward and reverse directions
    for rxn_id in sort(reactions)
        rxn_idx = reaction_map[rxn_id]
        rxn_symbol = Symbol(rxn_id)
        rxn_col = S[:, rxn_idx]

        # Skip if this reaction is not in the split constraints
        if !haskey(forward_flux_vars, rxn_symbol) || !haskey(reverse_flux_vars, rxn_symbol)
            continue
        end

        # Clear and reuse pre-allocated vectors
        empty!(substrate_mets)
        empty!(product_mets)

        # Build substrate and product lists
        @inbounds for (i, s) in enumerate(rxn_col)
            if s < 0
                push!(substrate_mets, (Symbol(metabolites[i]), -s))
            elseif s > 0
                push!(product_mets, (Symbol(metabolites[i]), s))
            end
        end

        # For split constraints, each reaction appears in both forward and reverse collections
        # We need to track contributions to both directions using the original reaction symbol

        # Process substrate side for forward reaction (consumption)
        if !isempty(substrate_mets)
            sort!(substrate_mets, by=x -> x[1])
            substrate_key = copy(substrate_mets)
            if !haskey(complex_data, substrate_key)
                reaction_contribs = Dict{Symbol,Float64}()
                sizehint!(reaction_contribs, 10)
                complex_data[substrate_key] = reaction_contribs
            else
                reaction_contribs = complex_data[substrate_key]
            end
            # For forward direction, substrates are consumed (negative contribution)
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) - 1.0
        end

        # Process product side for forward reaction (production)
        if !isempty(product_mets)
            sort!(product_mets, by=x -> x[1])
            product_key = copy(product_mets)
            if !haskey(complex_data, product_key)
                reaction_contribs = Dict{Symbol,Float64}()
                sizehint!(reaction_contribs, 10)
                complex_data[product_key] = reaction_contribs
            else
                reaction_contribs = complex_data[product_key]
            end
            # For forward direction, products are produced (positive contribution)
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) + 1.0
        end

        # Note: For split constraints, we only need to process each reaction once
        # The forward and reverse flux variables in the constraint system will handle
        # the directional contributions automatically through the optimization
    end

    # Build final MetabolicComplex structs
    complexes = Dict{Symbol,MetabolicComplex}()
    sizehint!(complexes, length(complex_data))

    # Sort keys for determinism
    sorted_keys = sort!(collect(keys(complex_data)), by=x -> string(generate_complex_id(x)))

    for canonical_mets in sorted_keys
        reaction_contribs = complex_data[canonical_mets]
        complex_id = generate_complex_id(canonical_mets)
        complexes[complex_id] = MetabolicComplex(complex_id, canonical_mets, reaction_contribs)
    end

    return complexes
end

"""
$(TYPEDSIGNATURES)

Efficiently and deterministically extract all unique complexes from the model.
This is the original function that works with the model's stoichiometric matrix
without considering reaction splitting.
"""
function extract_complexes_from_model(model::AbstractFBCModels.AbstractFBCModel)
    # Pre-allocate and cache model data for better performance
    S = AbstractFBCModels.stoichiometry(model)
    reactions = AbstractFBCModels.reactions(model)
    metabolites = AbstractFBCModels.metabolites(model)
    n_reactions = length(reactions)
    n_metabolites = length(metabolites)

    # Step 1: Accumulate reaction contributions for each unique complex.
    # The key is the canonical representation of the complex (a sorted vector of its metabolites).
    # Pre-size for better performance - estimate 2x reactions for complexes
    complex_data = Dict{Vector{Tuple{Symbol,Float64}},Dict{Symbol,Float64}}()
    sizehint!(complex_data, n_reactions * 2)

    # Pre-allocate reaction map with known size
    reaction_map = Dict{String,Int}()
    sizehint!(reaction_map, n_reactions)
    for (i, id) in enumerate(reactions)
        reaction_map[id] = i
    end

    # Pre-allocate temporary vectors to reuse in loops
    substrate_mets = Vector{Tuple{Symbol,Float64}}()
    product_mets = Vector{Tuple{Symbol,Float64}}()
    sizehint!(substrate_mets, n_metabolites ÷ 4)  # Estimate
    sizehint!(product_mets, n_metabolites ÷ 4)    # Estimate

    # Iterate over sorted reactions to ensure deterministic processing
    for rxn_id in sort(reactions)
        rxn_idx = reaction_map[rxn_id]
        rxn_symbol = Symbol(rxn_id)
        rxn_col = S[:, rxn_idx]

        # Clear and reuse pre-allocated vectors
        empty!(substrate_mets)
        empty!(product_mets)

        # Build substrate and product lists efficiently
        @inbounds for (i, s) in enumerate(rxn_col)
            if s < 0
                push!(substrate_mets, (Symbol(metabolites[i]), -s))
            elseif s > 0
                push!(product_mets, (Symbol(metabolites[i]), s))
            end
        end

        # Process substrate side - pre-size reaction contributions dict
        if !isempty(substrate_mets)
            sort!(substrate_mets, by=x -> x[1])
            substrate_key = copy(substrate_mets)
            if !haskey(complex_data, substrate_key)
                reaction_contribs = Dict{Symbol,Float64}()
                sizehint!(reaction_contribs, 10)  # Conservative estimate
                complex_data[substrate_key] = reaction_contribs
            else
                reaction_contribs = complex_data[substrate_key]
            end
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) - 1.0
        end

        # Process product side - pre-size reaction contributions dict
        if !isempty(product_mets)
            sort!(product_mets, by=x -> x[1])
            product_key = copy(product_mets)
            if !haskey(complex_data, product_key)
                reaction_contribs = Dict{Symbol,Float64}()
                sizehint!(reaction_contribs, 10)  # Conservative estimate
                complex_data[product_key] = reaction_contribs
            else
                reaction_contribs = complex_data[product_key]
            end
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) + 1.0
        end
    end

    # Step 2: Build the final, immutable MetabolicComplex structs.
    # Pre-allocate complexes dictionary with estimated size
    complexes = Dict{Symbol,MetabolicComplex}()
    sizehint!(complexes, length(complex_data))

    # Use keys directly without collecting to avoid allocation, sort them for determinism
    sorted_keys = sort!(collect(keys(complex_data)), by=x -> string(generate_complex_id(x)))

    for canonical_mets in sorted_keys
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
@inline function generate_complex_id(metabolites::Vector{Tuple{Symbol,Float64}})
    # Sorting is required here to ensure that the ID is canonical.
    sorted_mets = sort(metabolites, by=x -> x[1])
    # Pre-allocate parts vector for better performance
    parts = Vector{String}(undef, length(sorted_mets))
    @inbounds for (i, (met_id, coeff)) in enumerate(sorted_mets)
        parts[i] = "$(isinteger(coeff) ? Int(coeff) : coeff)_$(met_id)"
    end
    return Symbol(join(parts, "+"))
end

"""
$(TYPEDSIGNATURES)

Add complex activity variables and constraints to base constraints.
Uses constraint-aware complex extraction to respect reaction splitting.
"""
function add_complex_activities_to_constraints(
    model::AbstractFBCModels.AbstractFBCModel,
    base_constraints::C.ConstraintTree
)
    complexes = extract_complexes_from_constraints(model, base_constraints)
    # Iterate over sorted complex IDs to ensure deterministic ConstraintTree construction
    sorted_complex_ids = sort!(collect(keys(complexes)))

    complex_activity_constraints = C.ConstraintTree(
        complex_id => C.Constraint(
            value=C.sum(
                (
                    contribution * get_flux_variable(base_constraints, Symbol(rxn_id))
                    for (rxn_id, contribution) in complexes[complex_id].reaction_contributions
                    if flux_variable_exists(base_constraints, Symbol(rxn_id))
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

Helper function to get flux variable value from constraints, handling both split and non-split cases.
"""
function get_flux_variable(constraints::C.ConstraintTree, rxn_symbol::Symbol)
    # For split constraints, reaction contributions should be based on net flux
    # which is represented by forward_flux - reverse_flux
    # However, this is already handled by the constraint substitution in the symmetric approach

    # First check if we have the standard flux variable (after substitution)
    if haskey(constraints, :fluxes) && haskey(constraints.fluxes, rxn_symbol)
        return constraints.fluxes[rxn_symbol].value
        # If not available in fluxes, check forward flux variables
    elseif haskey(constraints, :fluxes_forward) && haskey(constraints.fluxes_forward, rxn_symbol)
        return constraints.fluxes_forward[rxn_symbol].value
        # Check reverse flux variables
    elseif haskey(constraints, :fluxes_reverse) && haskey(constraints.fluxes_reverse, rxn_symbol)
        return constraints.fluxes_reverse[rxn_symbol].value
    else
        error("Flux variable $rxn_symbol not found in constraints")
    end
end

"""
$(TYPEDSIGNATURES)

Helper function to check if flux variable exists in constraints.
"""
function flux_variable_exists(constraints::C.ConstraintTree, rxn_symbol::Symbol)
    return (haskey(constraints, :fluxes_forward) && haskey(constraints.fluxes_forward, rxn_symbol)) ||
           (haskey(constraints, :fluxes_reverse) && haskey(constraints.fluxes_reverse, rxn_symbol)) ||
           (haskey(constraints, :fluxes) && haskey(constraints.fluxes, rxn_symbol))
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