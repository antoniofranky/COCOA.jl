"""
Model preparation functionality for COCOA.

This module handles model preprocessing including:
- Blocked reaction detection and removal
- Reversible reaction splitting
- Model structure optimization
"""
module ModelPreparation

import COBREXA
import AbstractFBCModels
import HiGHS
import JuMP as J
import AbstractFBCModels.AbstractFBCModel
import SBMLFBCModels.SBMLFBCModels
import AbstractFBCModels.CanonicalModel as CM
import Logging
import Distributed

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
        temp_model = J.Model(optimizer)

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
                tol = J.get_optimizer_attribute(temp_model, attr)
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
    output_type=CM.Model,
    fast=false)

    work_model = convert(CM.Model, model)
    n_original_rxns = length(work_model.reactions)
    flux_tolerance = extract_solver_tolerance(optimizer, settings)

    # 1. Remove zero rows/columns FIRST (matching upstream algorithm Step 1)
    if remove_zero_rows || remove_zero_cols
        @info "Removing zero stoichiometry rows/columns..."
        remove_zero_stoichiometry!(work_model, remove_zero_rows, remove_zero_cols)
    end

    # 2. Split reversible reactions (upstream Step 2 - convert to irreversible)
    if split_reversible
        @info "Splitting reversible reactions..."
        n_rev = count_reversible_reactions(work_model)
        if n_rev > 0
            split_reversible_reactions!(work_model)
            @info "Split $n_rev reversible reactions"
        end
    end

    # 3. Find and remove blocked reactions LAST (matching upstream algorithm Step 3)
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
    if remove_zero_rows || remove_zero_cols
        @info "Removing zero stoichiometry rows/columns..."
        remove_zero_stoichiometry!(work_model, remove_zero_rows, remove_zero_cols)
    end

    @info "Model preparation complete: $(n_original_rxns) → $(length(work_model.reactions)) reactions"

    exported_model = convert(output_type, work_model)
    # Fix objective after conversion if we converted to SBML
    if output_type == SBMLFBCModels.SBMLFBCModel
        exported_model = fix_objective_conversion(exported_model)
    end
    return exported_model

end
"""
    fix_objective_after_conversion(model::SBMLFBCModels.SBMLFBCModel)

Fix objective function after model conversion by updating reaction IDs to match the actual 
reactions in the model. This handles cases where the objective references reaction IDs that 
don't exist due to format conversion (e.g., missing R_ prefix).
"""
function fix_objective_conversion(model::SBMLFBCModels.SBMLFBCModel)
    # Get all reaction IDs in the model
    reaction_ids = Set(AbstractFBCModels.reactions(model))

    # Check if we need to fix the objective
    try
        # Try to get the objective - if this fails, we need to fix it
        obj_dict = AbstractFBCModels.objective(model)
        return model  # If this succeeds, no fix needed
    catch e
        if e isa KeyError
            @info "Fixing objective after model conversion: $(e.key) not found"

            # Find the missing reaction ID and try to map it
            missing_id = string(e.key)

            # Try adding R_ prefix if missing
            if !startswith(missing_id, "R_")
                candidate_id = "R_" * missing_id
                if candidate_id in reaction_ids
                    @info "Mapping objective reaction $missing_id → $candidate_id"

                    # Update the SBML objective directly
                    for (obj_id, obj) in model.sbml.objectives
                        # flux_objectives is a Dict{String, Float64}
                        if haskey(obj.flux_objectives, missing_id)
                            coeff = obj.flux_objectives[missing_id]
                            delete!(obj.flux_objectives, missing_id)
                            obj.flux_objectives[candidate_id] = coeff
                            @info "Updated objective flux reference: $missing_id → $candidate_id (coeff=$coeff)"
                        end
                    end

                    return model
                end
            end

            # If we can't fix it, warn and clear the objective
            @warn "Could not fix objective reference to $missing_id, clearing objective"
            # Clear the objective by removing all flux objectives
            for (obj_id, obj) in model.sbml.objectives
                empty!(obj.flux_objectives)
            end

            return model
        else
            rethrow(e)
        end
    end
end
end # module