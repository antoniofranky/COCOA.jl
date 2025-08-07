"""
Model preparation functionality for COCOA.

This module handles model preprocessing including:
- Blocked reaction detection and removal
- Reversible reaction splitting
- Model structure optimization
"""
module ModelPreparation

using COBREXA
using AbstractFBCModels
using HiGHS
using JuMP
import AbstractFBCModels.AbstractFBCModel
import AbstractFBCModels.CanonicalModel as CM
using Logging
using Distributed

# Include sub-modules
include("blocked_reactions.jl")
include("model_optimization.jl")

export prepare_model_for_concordance

"""
Extract numerical tolerance from JuMP optimizer for consistent thresholding.
Creates a temporary model to query the optimizer's tolerance settings.
"""
function extract_solver_tolerance(optimizer, settings=[])::Float64
    default_tolerance = 1e-6  # Conservative fallback

    try
        # Create a minimal model to query optimizer attributes
        temp_model = JuMP.Model(optimizer)

        # Apply settings to get the actual configured tolerances
        for setting in [COBREXA.configuration.default_solver_settings; settings]
            setting(temp_model)
        end

        # Try common tolerance attribute names across different solvers
        tolerance_attrs = [
            "primal_feasibility_tolerance",  # HiGHS, Gurobi
            "dual_feasibility_tolerance",    # HiGHS, Gurobi
            "feasibility_tolerance",         # Some solvers
            "FeasibilityTol",               # Gurobi
            "OptimalityTol",                # Gurobi
            "primal_tolerance",             # CPLEX-style
            "dual_tolerance"                # CPLEX-style
        ]

        detected_tolerances = Float64[]

        for attr in tolerance_attrs
            try
                tol = JuMP.get_optimizer_attribute(temp_model, attr)
                if isa(tol, Real) && tol > 0 && tol < 1e-3
                    push!(detected_tolerances, Float64(tol))
                end
            catch
                # Attribute not supported by this solver, continue
            end
        end

        # Return the most restrictive (smallest) tolerance found
        if !isempty(detected_tolerances)
            return minimum(detected_tolerances)
        end

    catch e
        @debug "Could not extract solver tolerance" exception = e
    end

    return default_tolerance
end

"""
    prepare_model_for_concordance(model::AbstractFBCModel; 
                                 optimizer=HiGHS.Optimizer,
                                 settings=[],
                                 remove_blocked::Bool = true,
                                 split_reversible::Bool = true,
                                 remove_zero_rows::Bool = true,
                                 remove_zero_cols::Bool = true,
                                 workers = Distributed.workers())

Prepare a model for concordance analysis by:
1. Finding and removing blocked reactions using solver-specific tolerance
2. Splitting reversible reactions into irreversible pairs
3. Removing zero rows/columns from stoichiometry

Now uses parallel FVA for better performance on large models.
"""
function prepare_model_for_concordance(model::AbstractFBCModel;
    optimizer=HiGHS.Optimizer,
    settings=[],
    remove_blocked::Bool=true,
    split_reversible::Bool=false, # Already handled through COBREXA constraints
    remove_zero_rows::Bool=true,
    remove_zero_cols::Bool=true,
    workers=Distributed.workers(),
    fast=false)

    work_model = convert(CM.Model, model)
    n_original_rxns = length(work_model.reactions)
    flux_tolerance = extract_solver_tolerance(optimizer, settings)

    # 1. Find blocked reactions using parallel FVA
    if remove_blocked
        if fast
            @info "Finding blocked reactions using fast method with flux tolerance $(flux_tolerance)..."
            blocked = find_blocked_reactions_fast(work_model, optimizer, flux_tolerance)
        else
            @info "Finding blocked reactions using FVA with $(length(workers)) workers and flux tolerance $(flux_tolerance)..."
            blocked = find_blocked_reactions_parallel(work_model, optimizer, flux_tolerance, workers)
        end
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
end

end # module