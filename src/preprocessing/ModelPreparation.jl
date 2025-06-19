module ModelPreparation

using COBREXA
using AbstractFBCModels
using HiGHS
import AbstractFBCModels.AbstractFBCModel
import AbstractFBCModels.CanonicalModel as CM
using Logging
using ProgressMeter
using Distributed
import JuMP as J
export prepare_model_for_concordance

"""
    prepare_model_for_concordance(model::AbstractFBCModel; 
                                 optimizer=HiGHS.Optimizer,
                                 flux_tolerance::Float64 = 1e-9,
                                 remove_blocked::Bool = true,
                                 split_reversible::Bool = true,
                                 remove_zero_rows::Bool = true,
                                 remove_zero_cols::Bool = true,
                                 workers = Distributed.workers())

Prepare a model for concordance analysis by:
1. Finding and removing blocked reactions
2. Splitting reversible reactions into irreversible pairs
3. Removing zero rows/columns from stoichiometry

Now uses parallel FVA for better performance on large models.
"""
function prepare_model_for_concordance(model::AbstractFBCModel;
    optimizer=HiGHS.Optimizer,
    flux_tolerance::Float64=1e-9,
    remove_blocked::Bool=true,
    split_reversible::Bool=true,
    remove_zero_rows::Bool=true,
    remove_zero_cols::Bool=true,
    workers=Distributed.workers())

    work_model = convert(CM.Model, model)
    n_original_rxns = length(work_model.reactions)

    # 1. Find blocked reactions using parallel FVA
    if remove_blocked
        @info "Finding blocked reactions using parallel FVA..."
        blocked = find_blocked_reactions_parallel(work_model, optimizer, flux_tolerance, workers)
        if !isempty(blocked)
            remove_reactions!(work_model, blocked)
            @info "Removed $(length(blocked)) blocked reactions"
        end
    end

    # 2. Split reversible reactions
    if split_reversible
        @info "Splitting reversible reactions..."
        n_rev = count_reversible_reactions(work_model)
        if n_rev > 0
            split_reversible_reactions_optimized!(work_model)
            @info "Split $n_rev reversible reactions"
        end
    end

    # 3. Remove zero rows/columns
    if remove_zero_rows || remove_zero_cols
        remove_zero_stoichiometry!(work_model, remove_zero_rows, remove_zero_cols)
    end

    @info "Model preparation complete: $(n_original_rxns) → $(length(work_model.reactions)) reactions"
    return work_model

    # 4. Look for blocked reactions again
    if remove_blocked
        @info "Looking for blocked reactions using parallel FVA preprocessing again..."
        blocked_fast = find_blocked_reactions_fast(work_model, optimizer, flux_tolerance)
        if !isempty(blocked_fast)
            remove_reactions!(work_model, blocked_fast)
            @info "Removed $(length(blocked_fast)) blocked reactions (fast method)"
        end
    end
end

"""
Find blocked reactions using COBREXA's parallel FVA infrastructure
"""
function find_blocked_reactions_parallel(model::CM.Model, optimizer, flux_tolerance::Float64, workers)
    # Convert to COBREXA constraints for FVA
    constraints = flux_balance_constraints(model)

    # Run parallel FVA
    @info "Running parallel FVA on $(length(model.reactions)) reactions..."
    fva_results = constraints_variability(
        constraints,
        constraints.fluxes;
        optimizer=optimizer,
        workers=workers,
        settings=[]
    )

    # Identify blocked reactions
    blocked = Symbol[]
    for (rid, (min_flux, max_flux)) in fva_results
        if !isnothing(min_flux) && !isnothing(max_flux) &&
           abs(min_flux) < flux_tolerance && abs(max_flux) < flux_tolerance
            push!(blocked, rid)
        end
    end

    return String.(blocked)
end

"""
Count reversible reactions without allocating unnecessary arrays
"""
function count_reversible_reactions(model::CM.Model)
    count = 0
    for (_, rxn) in model.reactions
        if rxn.lower_bound < 0 && rxn.upper_bound > 0
            count += 1
        end
    end
    return count
end

"""
Optimized version that avoids deepcopy and excessive allocations
"""
function split_reversible_reactions_optimized!(model::CM.Model)
    # Collect reversible reactions first (can't modify while iterating)
    reversible_rxns = Tuple{String,CM.Reaction}[]
    sizehint!(reversible_rxns, length(model.reactions) ÷ 2)  # Pre-allocate

    for (rid, rxn) in model.reactions
        if rxn.lower_bound < 0 && rxn.upper_bound > 0
            push!(reversible_rxns, (rid, rxn))
        end
    end

    # Progress bar for large models
    p = Progress(length(reversible_rxns), desc="Splitting reactions: ")

    # Process all reversible reactions
    for (rid, rxn) in reversible_rxns
        # Create forward reaction (reuse existing reaction object)
        fwd_rxn = CM.Reaction(
            name=isnothing(rxn.name) ? nothing : "$(rxn.name) (forward)",
            lower_bound=0.0,
            upper_bound=rxn.upper_bound,
            stoichiometry=rxn.stoichiometry,  # Reuse dict
            objective_coefficient=rxn.objective_coefficient,
            gene_association_dnf=rxn.gene_association_dnf,  # Reuse
            annotations=rxn.annotations,  # Reuse
            notes=rxn.notes  # Reuse
        )

        # Create backward reaction with inverted stoichiometry
        bwd_stoich = Dict{String,Float64}()
        sizehint!(bwd_stoich, length(rxn.stoichiometry))
        for (k, v) in rxn.stoichiometry
            bwd_stoich[k] = -v
        end

        bwd_rxn = CM.Reaction(
            name=isnothing(rxn.name) ? nothing : "$(rxn.name) (backward)",
            lower_bound=0.0,
            upper_bound=-rxn.lower_bound,
            stoichiometry=bwd_stoich,
            objective_coefficient=-rxn.objective_coefficient,
            gene_association_dnf=rxn.gene_association_dnf,  # Reuse
            annotations=rxn.annotations,  # Reuse  
            notes=rxn.notes  # Reuse
        )

        # Replace original with forward/backward pair
        delete!(model.reactions, rid)
        model.reactions["$(rid)_f"] = fwd_rxn
        model.reactions["$(rid)_b"] = bwd_rxn

        next!(p)
    end
    finish!(p)
end

"""
Optimized version using Sets for faster lookups
"""
function remove_zero_stoichiometry!(model::CM.Model, remove_rows::Bool, remove_cols::Bool)
    # Remove metabolites not participating in any reaction
    if remove_rows
        # Use Set for O(1) lookups
        active_metabolites = Set{String}()
        for rxn in values(model.reactions)
            for met in keys(rxn.stoichiometry)
                push!(active_metabolites, met)
            end
        end

        # Collect inactive metabolites
        inactive = String[]
        for mid in keys(model.metabolites)
            if !(mid in active_metabolites)
                push!(inactive, mid)
            end
        end

        # Batch removal
        for mid in inactive
            delete!(model.metabolites, mid)
        end

        if !isempty(inactive)
            @info "Removed $(length(inactive)) metabolites with no reactions"
        end
    end

    # Remove reactions with no metabolites
    if remove_cols
        empty_reactions = String[]
        for (rid, rxn) in model.reactions
            if isempty(rxn.stoichiometry)
                push!(empty_reactions, rid)
            end
        end

        for rid in empty_reactions
            delete!(model.reactions, rid)
        end

        if !isempty(empty_reactions)
            @info "Removed $(length(empty_reactions)) reactions with no metabolites"
        end
    end
end

"""
Efficient batch removal of reactions
"""
function remove_reactions!(model::CM.Model, rids::Vector{String})
    for rid in rids
        delete!(model.reactions, rid)
    end
    return model
end

# Alternative fast blocked reaction detection using LP preprocessing
# This is typically much faster than full FVA for large models
function find_blocked_reactions_fast(model::CM.Model, optimizer, flux_tolerance::Float64)
    # This uses a single LP to find definitely blocked reactions
    # based on flux consistency checking

    n_rxns = length(model.reactions)
    rxn_ids = collect(keys(model.reactions))

    # Build stoichiometric matrix efficiently
    S = stoichiometry(model)

    # Create LP model
    lp = Model(optimizer)
    set_silent(lp)

    # Variables: fluxes + slack variables for each reaction
    J.@variable(lp, v[1:n_rxns])
    J.@variable(lp, z[1:n_rxns] >= 0)

    # Constraints
    J.@constraint(lp, S * v .== 0)  # Steady state

    # Get bounds
    lbs, ubs = bounds(model)

    # Flux bounds with slack
    J.@constraint(lp, [i = 1:n_rxns], v[i] >= lbs[i] - z[i])
    J.@constraint(lp, [i = 1:n_rxns], v[i] <= ubs[i] + z[i])

    # Minimize total slack
    J.@objective(lp, Min, sum(z))

    optimize!(lp)

    # Reactions with non-zero slack are potentially blocked
    blocked = String[]
    if termination_status(lp) == MOI.OPTIMAL
        z_vals = value.(z)
        for (i, z_val) in enumerate(z_vals)
            if z_val > flux_tolerance
                # Double-check with individual FVA
                # (only for reactions with slack)
                if is_reaction_blocked(model, rxn_ids[i], optimizer, flux_tolerance)
                    push!(blocked, rxn_ids[i])
                end
            end
        end
    end

    return blocked
end

# Helper for checking individual reactions
function is_reaction_blocked(model::CM.Model, rxn_id::String, optimizer, tolerance::Float64)
    constraints = flux_balance_constraints(model)
    rxn_sym = Symbol(rxn_id)

    # Check max flux
    max_result = optimized_values(
        constraints;
        objective=constraints.fluxes[rxn_sym].value,
        optimizer=optimizer,
        settings=[],
        sense=Maximal
    )

    isnothing(max_result) && return true
    abs(max_result.fluxes[rxn_sym]) > tolerance && return false

    # Check min flux
    min_result = optimized_values(
        constraints;
        objective=constraints.fluxes[rxn_sym].value,
        optimizer=optimizer,
        settings=[],
        sense=Minimal
    )

    isnothing(min_result) && return true
    abs(min_result.fluxes[rxn_sym]) > tolerance && return false

    return true
end

end # module