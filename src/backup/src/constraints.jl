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
    #constraints = C.prune_variables(constraints_before_pruning)

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

        # Substrate side
        substrate_mets = [(Symbol(metabolites[i]), -s) for (i, s) in enumerate(rxn_col) if s < -1e-12]
        if !isempty(substrate_mets)
            canonical_mets = sort(substrate_mets, by=x -> x[1])
            reaction_contribs = get!(complex_data, canonical_mets, Dict{Symbol,Float64}())
            reaction_contribs[rxn_symbol] = get(reaction_contribs, rxn_symbol, 0.0) - 1.0
        end

        # Product side
        product_mets = [(Symbol(metabolites[i]), s) for (i, s) in enumerate(rxn_col) if s > 1e-12]
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

Create bounds constraints for Charnes-Cooper transformation.
"""
function create_bounds_constraints(
    base_constraints::C.ConstraintTree,
    direction::Symbol,
)
    bounds = C.ConstraintTree()

    if direction == :positive
        t_pos = :t_pos^C.variable(bound=C.Between(1e-9, 1e9))
        base_constraints += (:charnes_cooper^t_pos)
        t_var = base_constraints.charnes_cooper[:t_pos].value

        if haskey(base_constraints, :fluxes_forward)
            for flux_name in sort(collect(keys(base_constraints.fluxes_forward)))
                flux_constraint = base_constraints.fluxes_forward[flux_name]
                flux_var = flux_constraint.value
                bound = flux_constraint.bound
                vmin = hasproperty(bound, :lower) ? bound.lower : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                vmax = hasproperty(bound, :upper) ? bound.upper : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("upper_fwd_$(flux_name)")^C.Constraint(flux_var - vmax * t_var, C.Between(-1e9, 0.0))
                end
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("lower_fwd_$(flux_name)")^C.Constraint(flux_var - vmin * t_var, C.Between(0.0, 1e9))
                end
            end
        end

        if haskey(base_constraints, :fluxes_reverse)
            for flux_name in sort(collect(keys(base_constraints.fluxes_reverse)))
                flux_constraint = base_constraints.fluxes_reverse[flux_name]
                flux_var = flux_constraint.value
                bound = flux_constraint.bound
                vmin = hasproperty(bound, :lower) ? bound.lower : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                vmax = hasproperty(bound, :upper) ? bound.upper : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("upper_rev_$(flux_name)")^C.Constraint(flux_var - vmax * t_var, C.Between(-1e9, 0.0))
                end
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("lower_rev_$(flux_name)")^C.Constraint(flux_var - vmin * t_var, C.Between(0.0, 1e9))
                end
            end
        end

    else # direction == :negative
        t_neg = :t_neg^C.variable(bound=C.Between(1e-12, 1e9))
        base_constraints += (:charnes_cooper^t_neg)
        t_var = base_constraints.charnes_cooper[:t_neg].value

        if haskey(base_constraints, :fluxes_forward)
            for flux_name in sort(collect(keys(base_constraints.fluxes_forward)))
                flux_constraint = base_constraints.fluxes_forward[flux_name]
                flux_var = flux_constraint.value
                bound = flux_constraint.bound
                vmin = hasproperty(bound, :lower) ? bound.lower : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                vmax = hasproperty(bound, :upper) ? bound.upper : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("upper_fwd_$(flux_name)")^C.Constraint(flux_var - vmin * t_var, C.Between(-1e9, 0.0))
                end
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("lower_fwd_$(flux_name)")^C.Constraint(flux_var - vmax * t_var, C.Between(0.0, 1e9))
                end
            end
        end

        if haskey(base_constraints, :fluxes_reverse)
            for flux_name in sort(collect(keys(base_constraints.fluxes_reverse)))
                flux_constraint = base_constraints.fluxes_reverse[flux_name]
                flux_var = flux_constraint.value
                bound = flux_constraint.bound
                vmin = hasproperty(bound, :lower) ? bound.lower : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                vmax = hasproperty(bound, :upper) ? bound.upper : (hasproperty(bound, :equal_to) ? bound.equal_to : continue)
                if isfinite(vmin) && !iszero(vmin)
                    bounds *= Symbol("upper_rev_$(flux_name)")^C.Constraint(flux_var - vmin * t_var, C.Between(-1e9, 0.0))
                end
                if isfinite(vmax) && !iszero(vmax)
                    bounds *= Symbol("lower_rev_$(flux_name)")^C.Constraint(flux_var - vmax * t_var, C.Between(0.0, 1e9))
                end
            end
        end
    end

    return bounds
end

"""
$(TYPEDSIGNATURES)

Create a Charnes-Cooper template for a specific direction without activity-specific constraints.
"""
function create_charnes_cooper_template(
    base_constraints::C.ConstraintTree,
    direction::Symbol;
)
    template_constraints = deepcopy(base_constraints)
    template_constraints += :charnes_cooper^C.ConstraintTree()
    bounds_constraints = create_bounds_constraints(template_constraints, direction)
    template_constraints *= :charnes_cooper^:fluxes_transformed^bounds_constraints
    template_constraints.fluxes_forward = C.ConstraintTree()
    template_constraints.fluxes_reverse = C.ConstraintTree()
    return template_constraints
end
