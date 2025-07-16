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

    # Build complex activities using C.sum pattern
    constraints = add_complex_activities_to_constraints(model, constraints)
    #verify_complex_relationships(model, constraints)
    return constraints
end

"""
$(TYPEDSIGNATURES)

Convert a ConstraintTrees.LinearValue to a JuMP expression.
"""
function linear_value_to_jump_expr(linear_value::C.LinearValue, x::Vector{JuMP.VariableRef})
    return sum(
        linear_value.weights[i] * x[linear_value.idxs[i]]
        for i in eachindex(linear_value.idxs)
    )
end

"""
$(TYPEDSIGNATURES)

Data structure to track complex information.
"""
struct ComplexInfo
    metabolites::Vector{Tuple{Symbol,Float64}}  # (metabolite_id, stoichiometry)
    reaction_contributions::Dict{Symbol,Float64}  # reaction_id -> contribution (+1 produced, -1 consumed)
end

"""
$(TYPEDSIGNATURES)

Extract all unique complexes from the model.
A complex is the sum of metabolites that appear together in reactions.
Identical complexes are unified regardless of whether they appear as substrates or products.
"""
function extract_complexes_from_model(model::AbstractFBCModels.AbstractFBCModel)
    complexes = Dict{Symbol,ComplexInfo}()

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
                complexes[complex_id] = ComplexInfo(substrate_mets, Dict{Symbol,Float64}())
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
                complexes[complex_id] = ComplexInfo(product_mets, Dict{Symbol,Float64}())
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

    # Create complex activity variables and add them to base constraints
    complex_vars = C.variables(
        keys=collect(keys(complexes)),
        bounds=C.Between(-Inf, Inf)
    )

    # Add complex variables to base constraints first
    constraints_with_complexes = base_constraints + (:complexes^complex_vars)

    # Build complex activity constraints using the combined constraint tree
    complex_activity_constraints = C.ConstraintTree(
        complex_id => C.Constraint(
            value=constraints_with_complexes.complexes[complex_id].value - C.sum(
                (
                    contribution * constraints_with_complexes.fluxes[Symbol(rxn_id)].value
                    for (rxn_id, contribution) in complex_info.reaction_contributions
                    if haskey(constraints_with_complexes.fluxes, Symbol(rxn_id))
                ),
                init=zero(C.LinearValue)
            ),
            bound=C.EqualTo(0.0)
        ) for (complex_id, complex_info) in complexes
    )

    # Add the complex relationships to the constraints
    return constraints_with_complexes * (:complex_relationships^complex_activity_constraints)
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
function setup_concordance_testing(
    base_constraints::C.ConstraintTree,
    all_complexes::Vector{Complex};
    default_bounds::Tuple{Float64,Float64}=(-1000.0, 1000.0)
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



#  julia> complex_vars = C.variables(keys = [:ATP_F6P_complex], bounds = C.Between(-Inf, Inf))
#   ConstraintTrees.ConstraintTree with 1 element:
#     :ATP_F6P_complex => ConstraintTrees.Constraint(ConstraintTrees.LinearValue(#= ... =#), 
#   ConstraintTrees.Between(-Inf, Inf))

#   julia> constraints = base_constraints + (:complexes^complex_vars)

#          # 3. Create the relationship constraint
#          # ATP_F6P_complex_activity = -R_PFK_flux (consumed in PFK reaction)
#          # (Add other reactions where this complex appears if any)
#   ConstraintTrees.ConstraintTree with 5 elements:
#     :complexes          => ConstraintTrees.ConstraintTree(#= 1 element =#)
#     :coupling           => ConstraintTrees.ConstraintTree(#= 0 elements =#)
#     :flux_stoichiometry => ConstraintTrees.ConstraintTree(#= 72 elements =#)
#     :fluxes             => ConstraintTrees.ConstraintTree(#= 95 elements =#)
#     :objective          => ConstraintTrees.Constraint(ConstraintTrees.LinearValue(#= ... =#))

#   julia> relationship = :ATP_F6P_complex^C.Constraint(
#              value = constraints.complexes.ATP_F6P_complex.value +
#                      constraints.fluxes.R_PFK.value,  # activity = -flux (consumption)
#              bound = C.EqualTo(0.0)
#          )

#          # 4. Add the relationship to the constraint system
#   ConstraintTrees.ConstraintTree with 1 element:
#     :ATP_F6P_complex => ConstraintTrees.Constraint(ConstraintTrees.LinearValue(#= ... =#), 
#   ConstraintTrees.EqualTo(0.0))

#   julia> constraints *= (:complex_relationships^relationship)
#   ConstraintTrees.ConstraintTree with 6 elements:
#     :complex_relationships => ConstraintTrees.ConstraintTree(#= 1 element =#)
#     :complexes             => ConstraintTrees.ConstraintTree(#= 1 element =#)
#     :coupling              => ConstraintTrees.ConstraintTree(#= 0 elements =#)
#     :flux_stoichiometry    => ConstraintTrees.ConstraintTree(#= 72 elements =#)
#     :fluxes                => ConstraintTrees.ConstraintTree(#= 95 elements =#)
#     :objective             => ConstraintTrees.Constraint(ConstraintTrees.LinearValue(#= ... =#))

#   julia> C.pretty(constraints)
#   ┬─complex_relationships
#   │ ╰───ATP_F6P_complex: 1.0*x[72] + 1.0*x[96] = 0.0
#   ├─complexes
#   │ ╰───ATP_F6P_complex: 1.0*x[96] ∈ [-Inf, Inf]
#   ├─coupling
#   ├─flux_stoichiometry
#   │ ╰─┬─M_13dpg_c: 1.0*x[49] + 1.0*x[75] = 0.0
#   │   ├─M_2pg_c: -1.0*x[18] + -1.0*x[77] = 0.0
#   │   ├─M_3pg_c: -1.496*x[13] + -1.0*x[75] + 1.0*x[77] = 0.0
#   │   ├─M_6pgc_c: -1.0*x[57] + 1.0*x[76] = 0.0
#   │   ├─M_6pgl_c: 1.0*x[48] + -1.0*x[76] = 0.0
#   │   ├─M_ac_c: -1.0*x[3] + 1.0*x[6] = 0.0
#   │   ├─M_ac_e: -1.0*x[6] + -1.0*x[20] = 0.0
#   │   ├─M_acald_c: -1.0*x[1] + 1.0*x[2] + 1.0*x[10] = 0.0
#   │   ├─M_acald_e: -1.0*x[2] + -1.0*x[21] = 0.0
#   │   ├─M_accoa_c: 1.0*x[1] + -3.7478*x[13] + -1.0*x[15] + -1.0*x[62] + 1.0*x[71] + 1.0*x[73] + 
#   -1.0*x[82] = 0.0
#   │   ├─M_acon_C_c: 1.0*x[4] + -1.0*x[5] = 0.0
#   │   ├─M_actp_c: 1.0*x[3] + 1.0*x[82] = 0.0
#   │   ├─M_adp_c: 1.0*x[3] + 2.0*x[7] + 1.0*x[11] + -1.0*x[12] + 59.81*x[13] + 1.0*x[51] + 1.0*x[52] + 
#   1.0*x[72] + 1.0*x[75] + 1.0*x[80] + -1.0*x[83] + 1.0*x[90] = 0.0
#   │   ├─M_akg_c: -1.0*x[8] + 1.0*x[9] + 4.1182*x[13] + 1.0*x[53] + -1.0*x[55] + 1.0*x[59] = 0.0
#   │   ├─M_akg_e: -1.0*x[9] + -1.0*x[22] = 0.0
#   │   ├─M_amp_c: -1.0*x[7] + 1.0*x[81] = 0.0
#   │   ├─M_atp_c: -1.0*x[3] + -1.0*x[7] + -1.0*x[11] + 1.0*x[12] + -59.81*x[13] + -1.0*x[51] + 
#   -1.0*x[52] + -1.0*x[72] + -1.0*x[75] + -1.0*x[80] + -1.0*x[81] + 1.0*x[83] + -1.0*x[90] = 0.0
#   │   ├─M_cit_c: -1.0*x[4] + 1.0*x[15] = 0.0
#   │   ├─M_co2_c: 1.0*x[8] + 1.0*x[14] + 1.0*x[57] + 1.0*x[59] + 1.0*x[65] + 1.0*x[66] + 1.0*x[71] + 
#   -1.0*x[79] + 1.0*x[80] = 0.0
#   │   ├─M_co2_e: -1.0*x[14] + -1.0*x[23] = 0.0
#   │   ├─M_coa_c: -1.0*x[1] + -1.0*x[8] + 3.7478*x[13] + 1.0*x[15] + 1.0*x[62] + -1.0*x[71] + -1.0*x[73]
#    + 1.0*x[82] + -1.0*x[90] = 0.0
#   │   ├─M_dhap_c: 1.0*x[40] + -1.0*x[95] = 0.0
#   │   ├─M_e4p_c: -0.361*x[13] + 1.0*x[91] + -1.0*x[94] = 0.0
#   │   ├─M_etoh_c: -1.0*x[10] + 1.0*x[19] = 0.0
#   │   ├─M_etoh_e: -1.0*x[19] + -1.0*x[24] = 0.0
#   │   ├─M_f6p_c: -0.0709*x[13] + 1.0*x[41] + 1.0*x[45] + -1.0*x[72] + 1.0*x[74] + 1.0*x[91] + 1.0*x[94]
#    = 0.0
#   │   ├─M_fdp_c: -1.0*x[40] + -1.0*x[41] + 1.0*x[72] = 0.0
#   │   ├─M_for_c: 1.0*x[42] + 1.0*x[43] + 1.0*x[73] = 0.0
#   │   ├─M_for_e: -1.0*x[25] + -1.0*x[42] + -1.0*x[43] = 0.0
#   │   ├─M_fru_e: -1.0*x[26] + -1.0*x[45] = 0.0
#   │   ├─M_fum_c: -1.0*x[44] + -1.0*x[46] + 1.0*x[47] + 1.0*x[89] = 0.0
#   │   ├─M_fum_e: -1.0*x[27] + -1.0*x[47] = 0.0
#   │   ├─M_g3p_c: -0.129*x[13] + 1.0*x[40] + -1.0*x[49] + -1.0*x[91] + 1.0*x[93] + 1.0*x[94] + 1.0*x[95]
#    = 0.0
#   │   ├─M_g6p_c: -0.205*x[13] + -1.0*x[48] + 1.0*x[50] + -1.0*x[74] = 0.0
#   │   ├─M_glc__D_e: -1.0*x[28] + -1.0*x[50] = 0.0
#   │   ├─M_gln__L_c: -0.2557*x[13] + 1.0*x[51] + 1.0*x[52] + -1.0*x[54] + -1.0*x[55] = 0.0
#   │   ├─M_gln__L_e: -1.0*x[29] + -1.0*x[52] = 0.0
#   │   ├─M_glu__L_c: -4.9414*x[13] + -1.0*x[51] + -1.0*x[53] + 1.0*x[54] + 2.0*x[55] + 1.0*x[56] = 0.0
#   │   ├─M_glu__L_e: -1.0*x[30] + -1.0*x[56] = 0.0
#   │   ├─M_glx_c: 1.0*x[60] + -1.0*x[62] = 0.0
#   │   ├─M_h2o_c: 1.0*x[4] + -1.0*x[5] + -1.0*x[11] + 1.0*x[12] + -59.81*x[13] + -1.0*x[15] + 1.0*x[16] 
#   + 1.0*x[18] + -1.0*x[41] + -1.0*x[46] + -1.0*x[52] + -1.0*x[53] + -1.0*x[54] + 1.0*x[58] + -1.0*x[62]
#    + -1.0*x[76] + -1.0*x[79] + -1.0*x[81] = 0.0
#   │   ├─M_h2o_e: -1.0*x[31] + -1.0*x[58] = 0.0
#   │   ├─M_h_c: 1.0*x[1] + 1.0*x[6] + 1.0*x[9] + 1.0*x[10] + 1.0*x[11] + 3.0*x[12] + 59.81*x[13] + 
#   1.0*x[15] + -2.0*x[16] + 1.0*x[17] + 1.0*x[19] + 1.0*x[43] + 2.0*x[47] + 1.0*x[48] + 1.0*x[49] + 
#   1.0*x[51] + 1.0*x[52] + 1.0*x[53] + -1.0*x[55] + 1.0*x[56] + 1.0*x[61] + 1.0*x[62] + 2.0*x[63] + 
#   1.0*x[64] + -4.0*x[67] + 1.0*x[72] + 1.0*x[76] + 1.0*x[78] + 1.0*x[79] + 2.0*x[81] + -1.0*x[83] + 
#   1.0*x[84] + 2.0*x[87] + 1.0*x[88] + 2.0*x[92] = 0.0
#   │   ├─M_h_e: -1.0*x[6] + -1.0*x[9] + -4.0*x[12] + 2.0*x[16] + -1.0*x[17] + -1.0*x[19] + -1.0*x[32] + 
#   -1.0*x[43] + -2.0*x[47] + -1.0*x[56] + -2.0*x[63] + 3.0*x[67] + -1.0*x[78] + -1.0*x[84] + -2.0*x[87] 
#   + -1.0*x[88] + -2.0*x[92] = 0.0
#   │   ├─M_icit_c: 1.0*x[5] + -1.0*x[59] + -1.0*x[60] = 0.0
#   │   ├─M_lac__D_c: 1.0*x[17] + -1.0*x[61] = 0.0
#   │   ├─M_lac__D_e: -1.0*x[17] + -1.0*x[33] = 0.0
#   │   ├─M_mal__L_c: 1.0*x[46] + 1.0*x[62] + 1.0*x[63] + -1.0*x[64] + -1.0*x[65] + -1.0*x[66] = 0.0
#   │   ├─M_mal__L_e: -1.0*x[34] + -1.0*x[63] = 0.0
#   │   ├─M_nad_c: -1.0*x[1] + -1.0*x[8] + -1.0*x[10] + -3.547*x[13] + -1.0*x[49] + -1.0*x[61] + 
#   -1.0*x[64] + -1.0*x[65] + 1.0*x[67] + -1.0*x[68] + -1.0*x[71] + 1.0*x[92] = 0.0
#   │   ├─M_nadh_c: 1.0*x[1] + 1.0*x[8] + 1.0*x[10] + 3.547*x[13] + 1.0*x[49] + 1.0*x[61] + 1.0*x[64] + 
#   1.0*x[65] + -1.0*x[67] + 1.0*x[68] + 1.0*x[71] + -1.0*x[92] = 0.0
#   │   ├─M_nadp_c: 13.0279*x[13] + -1.0*x[48] + -1.0*x[53] + 1.0*x[55] + -1.0*x[57] + -1.0*x[59] + 
#   -1.0*x[66] + 1.0*x[68] + -1.0*x[92] = 0.0
#   │   ├─M_nadph_c: -13.0279*x[13] + 1.0*x[48] + 1.0*x[53] + -1.0*x[55] + 1.0*x[57] + 1.0*x[59] + 
#   1.0*x[66] + -1.0*x[68] + 1.0*x[92] = 0.0
#   │   ├─M_nh4_c: -1.0*x[51] + 1.0*x[53] + 1.0*x[54] + 1.0*x[69] = 0.0
#   │   ├─M_nh4_e: -1.0*x[35] + -1.0*x[69] = 0.0
#   │   ├─M_o2_c: -0.5*x[16] + 1.0*x[70] = 0.0
#   │   ├─M_o2_e: -1.0*x[36] + -1.0*x[70] = 0.0
#   │   ├─M_oaa_c: -1.7867*x[13] + -1.0*x[15] + 1.0*x[64] + 1.0*x[79] + -1.0*x[80] = 0.0
#   │   ├─M_pep_c: -0.5191*x[13] + 1.0*x[18] + -1.0*x[45] + -1.0*x[50] + -1.0*x[79] + 1.0*x[80] + 
#   1.0*x[81] + -1.0*x[83] = 0.0
#   │   ├─M_pi_c: 1.0*x[11] + -1.0*x[12] + 59.81*x[13] + 1.0*x[41] + -1.0*x[49] + 1.0*x[51] + 1.0*x[52] +
#    1.0*x[78] + 1.0*x[79] + 1.0*x[81] + -1.0*x[82] + 1.0*x[90] = 0.0
#   │   ├─M_pi_e: -1.0*x[37] + -1.0*x[78] = 0.0
#   │   ├─M_pyr_c: -2.8328*x[13] + 1.0*x[45] + 1.0*x[50] + 1.0*x[61] + 1.0*x[65] + 1.0*x[66] + -1.0*x[71]
#    + -1.0*x[73] + -1.0*x[81] + 1.0*x[83] + 1.0*x[84] = 0.0
#   │   ├─M_pyr_e: -1.0*x[38] + -1.0*x[84] = 0.0
#   │   ├─M_q8_c: 1.0*x[16] + 1.0*x[44] + -1.0*x[67] + -1.0*x[89] = 0.0
#   │   ├─M_q8h2_c: -1.0*x[16] + -1.0*x[44] + 1.0*x[67] + 1.0*x[89] = 0.0
#   │   ├─M_r5p_c: -0.8977*x[13] + -1.0*x[86] + -1.0*x[93] = 0.0
#   │   ├─M_ru5p__D_c: 1.0*x[57] + -1.0*x[85] + 1.0*x[86] = 0.0
#   │   ├─M_s7p_c: -1.0*x[91] + 1.0*x[93] = 0.0
#   │   ├─M_succ_c: 1.0*x[44] + 1.0*x[60] + 1.0*x[87] + -1.0*x[88] + -1.0*x[89] + -1.0*x[90] = 0.0
#   │   ├─M_succ_e: -1.0*x[39] + -1.0*x[87] + 1.0*x[88] = 0.0
#   │   ├─M_succoa_c: 1.0*x[8] + 1.0*x[90] = 0.0
#   │   ╰─M_xu5p__D_c: 1.0*x[85] + -1.0*x[93] + -1.0*x[94] = 0.0
#   ├─fluxes
#   │ ╰─┬─R_ACALD: 1.0*x[1] ∈ [-1000.0, 1000.0]
#   │   ├─R_ACALDt: 1.0*x[2] ∈ [-1000.0, 1000.0]
#   │   ├─R_ACKr: 1.0*x[3] ∈ [-1000.0, 1000.0]
#   │   ├─R_ACONTa: 1.0*x[4] ∈ [-1000.0, 1000.0]
#   │   ├─R_ACONTb: 1.0*x[5] ∈ [-1000.0, 1000.0]
#   │   ├─R_ACt2r: 1.0*x[6] ∈ [-1000.0, 1000.0]
#   │   ├─R_ADK1: 1.0*x[7] ∈ [-1000.0, 1000.0]
#   │   ├─R_AKGDH: 1.0*x[8] ∈ [0.0, 1000.0]
#   │   ├─R_AKGt2r: 1.0*x[9] ∈ [-1000.0, 1000.0]
#   │   ├─R_ALCD2x: 1.0*x[10] ∈ [-1000.0, 1000.0]
#   │   ├─R_ATPM: 1.0*x[11] ∈ [8.39, 1000.0]
#   │   ├─R_ATPS4r: 1.0*x[12] ∈ [-1000.0, 1000.0]
#   │   ├─R_BIOMASS_Ecoli_core_w_GAM: 1.0*x[13] ∈ [0.0, 1000.0]
#   │   ├─R_CO2t: 1.0*x[14] ∈ [-1000.0, 1000.0]
#   │   ├─R_CS: 1.0*x[15] ∈ [0.0, 1000.0]
#   │   ├─R_CYTBD: 1.0*x[16] ∈ [0.0, 1000.0]
#   │   ├─R_D_LACt2: 1.0*x[17] ∈ [-1000.0, 1000.0]
#   │   ├─R_ENO: 1.0*x[18] ∈ [-1000.0, 1000.0]
#   │   ├─R_ETOHt2r: 1.0*x[19] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_ac_e: 1.0*x[20] ∈ [0.0, 1000.0]
#   │   ├─R_EX_acald_e: 1.0*x[21] ∈ [0.0, 1000.0]
#   │   ├─R_EX_akg_e: 1.0*x[22] ∈ [0.0, 1000.0]
#   │   ├─R_EX_co2_e: 1.0*x[23] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_etoh_e: 1.0*x[24] ∈ [0.0, 1000.0]
#   │   ├─R_EX_for_e: 1.0*x[25] ∈ [0.0, 1000.0]
#   │   ├─R_EX_fru_e: 1.0*x[26] ∈ [0.0, 1000.0]
#   │   ├─R_EX_fum_e: 1.0*x[27] ∈ [0.0, 1000.0]
#   │   ├─R_EX_glc__D_e: 1.0*x[28] ∈ [-10.0, 1000.0]
#   │   ├─R_EX_gln__L_e: 1.0*x[29] ∈ [0.0, 1000.0]
#   │   ├─R_EX_glu__L_e: 1.0*x[30] ∈ [0.0, 1000.0]
#   │   ├─R_EX_h2o_e: 1.0*x[31] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_h_e: 1.0*x[32] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_lac__D_e: 1.0*x[33] ∈ [0.0, 1000.0]
#   │   ├─R_EX_mal__L_e: 1.0*x[34] ∈ [0.0, 1000.0]
#   │   ├─R_EX_nh4_e: 1.0*x[35] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_o2_e: 1.0*x[36] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_pi_e: 1.0*x[37] ∈ [-1000.0, 1000.0]
#   │   ├─R_EX_pyr_e: 1.0*x[38] ∈ [0.0, 1000.0]
#   │   ├─R_EX_succ_e: 1.0*x[39] ∈ [0.0, 1000.0]
#   │   ├─R_FBA: 1.0*x[40] ∈ [-1000.0, 1000.0]
#   │   ├─R_FBP: 1.0*x[41] ∈ [0.0, 1000.0]
#   │   ├─R_FORt: 1.0*x[42] ∈ [-1000.0, 0.0]
#   │   ├─R_FORt2: 1.0*x[43] ∈ [0.0, 1000.0]
#   │   ├─R_FRD7: 1.0*x[44] ∈ [0.0, 1000.0]
#   │   ├─R_FRUpts2: 1.0*x[45] ∈ [0.0, 1000.0]
#   │   ├─R_FUM: 1.0*x[46] ∈ [-1000.0, 1000.0]
#   │   ├─R_FUMt2_2: 1.0*x[47] ∈ [0.0, 1000.0]
#   │   ├─R_G6PDH2r: 1.0*x[48] ∈ [-1000.0, 1000.0]
#   │   ├─R_GAPD: 1.0*x[49] ∈ [-1000.0, 1000.0]
#   │   ├─R_GLCpts: 1.0*x[50] ∈ [0.0, 1000.0]
#   │   ├─R_GLNS: 1.0*x[51] ∈ [0.0, 1000.0]
#   │   ├─R_GLNabc: 1.0*x[52] ∈ [0.0, 1000.0]
#   │   ├─R_GLUDy: 1.0*x[53] ∈ [-1000.0, 1000.0]
#   │   ├─R_GLUN: 1.0*x[54] ∈ [0.0, 1000.0]
#   │   ├─R_GLUSy: 1.0*x[55] ∈ [0.0, 1000.0]
#   │   ├─R_GLUt2r: 1.0*x[56] ∈ [-1000.0, 1000.0]
#   │   ├─R_GND: 1.0*x[57] ∈ [0.0, 1000.0]
#   │   ├─R_H2Ot: 1.0*x[58] ∈ [-1000.0, 1000.0]
#   │   ├─R_ICDHyr: 1.0*x[59] ∈ [-1000.0, 1000.0]
#   │   ├─R_ICL: 1.0*x[60] ∈ [0.0, 1000.0]
#   │   ├─R_LDH_D: 1.0*x[61] ∈ [-1000.0, 1000.0]
#   │   ├─R_MALS: 1.0*x[62] ∈ [0.0, 1000.0]
#   │   ├─R_MALt2_2: 1.0*x[63] ∈ [0.0, 1000.0]
#   │   ├─R_MDH: 1.0*x[64] ∈ [-1000.0, 1000.0]
#   │   ├─R_ME1: 1.0*x[65] ∈ [0.0, 1000.0]
#   │   ├─R_ME2: 1.0*x[66] ∈ [0.0, 1000.0]
#   │   ├─R_NADH16: 1.0*x[67] ∈ [0.0, 1000.0]
#   │   ├─R_NADTRHD: 1.0*x[68] ∈ [0.0, 1000.0]
#   │   ├─R_NH4t: 1.0*x[69] ∈ [-1000.0, 1000.0]
#   │   ├─R_O2t: 1.0*x[70] ∈ [-1000.0, 1000.0]
#   │   ├─R_PDH: 1.0*x[71] ∈ [0.0, 1000.0]
#   │   ├─R_PFK: 1.0*x[72] ∈ [0.0, 1000.0]
#   │   ├─R_PFL: 1.0*x[73] ∈ [0.0, 1000.0]
#   │   ├─R_PGI: 1.0*x[74] ∈ [-1000.0, 1000.0]
#   │   ├─R_PGK: 1.0*x[75] ∈ [-1000.0, 1000.0]
#   │   ├─R_PGL: 1.0*x[76] ∈ [0.0, 1000.0]
#   │   ├─R_PGM: 1.0*x[77] ∈ [-1000.0, 1000.0]
#   │   ├─R_PIt2r: 1.0*x[78] ∈ [-1000.0, 1000.0]
#   │   ├─R_PPC: 1.0*x[79] ∈ [0.0, 1000.0]
#   │   ├─R_PPCK: 1.0*x[80] ∈ [0.0, 1000.0]
#   │   ├─R_PPS: 1.0*x[81] ∈ [0.0, 1000.0]
#   │   ├─R_PTAr: 1.0*x[82] ∈ [-1000.0, 1000.0]
#   │   ├─R_PYK: 1.0*x[83] ∈ [0.0, 1000.0]
#   │   ├─R_PYRt2: 1.0*x[84] ∈ [-1000.0, 1000.0]
#   │   ├─R_RPE: 1.0*x[85] ∈ [-1000.0, 1000.0]
#   │   ├─R_RPI: 1.0*x[86] ∈ [-1000.0, 1000.0]
#   │   ├─R_SUCCt2_2: 1.0*x[87] ∈ [0.0, 1000.0]
#   │   ├─R_SUCCt3: 1.0*x[88] ∈ [0.0, 1000.0]
#   │   ├─R_SUCDi: 1.0*x[89] ∈ [0.0, 1000.0]
#   │   ├─R_SUCOAS: 1.0*x[90] ∈ [-1000.0, 1000.0]
#   │   ├─R_TALA: 1.0*x[91] ∈ [-1000.0, 1000.0]
#   │   ├─R_THD2: 1.0*x[92] ∈ [0.0, 1000.0]
#   │   ├─R_TKT1: 1.0*x[93] ∈ [-1000.0, 1000.0]
#   │   ├─R_TKT2: 1.0*x[94] ∈ [-1000.0, 1000.0]
#   │   ╰─R_TPI: 1.0*x[95] ∈ [-1000.0, 1000.0]
#   ╰─objective: 1.0*x[13]