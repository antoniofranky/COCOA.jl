"""
Blocked reaction detection algorithms for model preparation.

This module handles:
- Parallel FVA-based blocked reaction detection
- Fast LP-based blocked reaction detection
- Individual reaction blocking verification
"""

using COBREXA
using HiGHS
using JuMP
using Distributed
import AbstractFBCModels.CanonicalModel as CM
import JuMP as J

"""
Find blocked reactions using COBREXA's parallel FVA infrastructure.
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

    # Identify blocked reactions - pre-allocate for better performance
    blocked = Vector{Symbol}()
    sizehint!(blocked, length(model.reactions) ÷ 10)  # Conservative estimate
    
    @inbounds for (rid, (min_flux, max_flux)) in fva_results
        if !isnothing(min_flux) && !isnothing(max_flux) &&
           abs(min_flux) < flux_tolerance && abs(max_flux) < flux_tolerance
            push!(blocked, rid)
        end
    end

    # Convert to strings efficiently
    blocked_strings = Vector{String}(undef, length(blocked))
    @inbounds for (i, rid) in enumerate(blocked)
        blocked_strings[i] = String(rid)
    end
    return blocked_strings
end

"""
Alternative fast blocked reaction detection using LP preprocessing.
This is typically much faster than full FVA for large models.
"""
function find_blocked_reactions_fast(model::CM.Model, optimizer, flux_tolerance::Float64)
    n_rxns = length(model.reactions)

    # Get reaction info using stable API
    rxn_ids = collect(keys(model.reactions))
    lbs = [model.reactions[rid].lower_bound for rid in rxn_ids]
    ubs = [model.reactions[rid].upper_bound for rid in rxn_ids]

    # Build stoichiometric matrix
    S = AbstractFBCModels.stoichiometry(model)  # This is part of AbstractFBCModels stable API

    # Create LP model
    lp = Model(optimizer)
    set_silent(lp)

    # Variables: fluxes + slack variables
    J.@variable(lp, v[1:n_rxns])
    J.@variable(lp, z[1:n_rxns] >= 0)

    # Constraints
    J.@constraint(lp, S * v .== 0)  # Steady state

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

"""
Helper for checking individual reactions.
"""
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