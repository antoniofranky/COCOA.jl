"""
Model preparation utilities for COCOA analysis.

This module contains functions to prepare COBRA models for concordance and kinetic
module analysis. It includes functions for removing blocked reactions, converting
to irreversible reactions, and splitting into elementary steps.
"""
module ModelPreparation

using COBREXA
using SparseArrays
using LinearAlgebra
using Logging

import COBREXA: StandardModel

include("elementary_steps.jl")
using .elementary_steps

export prepare_model_for_concordance,
    split_into_elementary_steps,
    standardize_bounds!,
    remove_blocked_reactions!

"""
    prepare_model_for_concordance(model::AbstractFBCModel; 
                                  split_elementary::Bool = false,
                                  mechanism::Symbol = :fixed,
                                  remove_blocked::Bool = true,
                                  threshold::Float64 = 1e-9,
                                  default_lb::Float64 = -1000.0,
                                  default_ub::Float64 = 1000.0)

Prepare a metabolic model for concordance module analysis.

# Arguments
- `model`: The metabolic model to prepare
- `split_elementary`: Whether to split reactions into elementary steps
- `mechanism`: Enzyme mechanism for splitting (`:fixed` or `:random`)
- `remove_blocked`: Whether to remove blocked reactions
- `threshold`: Threshold for considering reactions blocked
- `default_lb`: Default lower bound for reactions
- `default_ub`: Default upper bound for reactions

# Returns
- Preprocessed model ready for concordance analysis
"""
function prepare_model_for_concordance(
    model::AbstractFBCModel;
    split_elementary::Bool=false,
    mechanism::Symbol=:fixed,
    remove_blocked::Bool=true,
    threshold::Float64=1e-9,
    default_lb::Float64=-1000.0,
    default_ub::Float64=1000.0
)
    # Create a working copy
    working_model = convert(StandardModel, model)

    @info "Starting model preprocessing for concordance analysis"

    # Step 1: Initial cleanup
    @info "Step 1: Initial model cleanup"
    remove_zero_rows_columns!(working_model)

    # Step 2: Standardize bounds
    @info "Step 2: Standardizing reaction bounds"
    standardize_bounds!(working_model, default_lb, default_ub)

    # Step 3: Check initial feasibility
    @info "Step 3: Checking initial model feasibility"
    if !check_model_feasibility(working_model)
        @warn "Model is infeasible before preprocessing"
    end

    # Step 4: Remove blocked reactions
    if remove_blocked
        @info "Step 4: Removing blocked reactions"
        remove_blocked_reactions!(working_model, threshold)
    end

    # Step 5: Split into elementary steps (if requested)
    if split_elementary
        @info "Step 5: Splitting reactions into elementary steps"

        working_model = ElementarySteps.split_into_elementary_steps(
            working_model,
            mechanism=mechanism,
            seed=42  # For reproducibility
        )

        # Post-splitting cleanup
        @info "Step 5a: Post-splitting cleanup"
        remove_zero_rows_columns!(working_model)

        if remove_blocked
            @info "Step 5b: Removing blocked reactions after splitting"
            remove_blocked_reactions!(working_model, threshold)
        end
    end

    # Step 6: Convert to irreversible
    @info "Step 6: Converting to irreversible reactions"
    working_model = make_irreversible(working_model)

    # Step 7: Final validation
    @info "Step 7: Final model validation"
    validate_preprocessed_model(working_model)

    @info "Model preprocessing complete"
    @info "Final model: $(n_reactions(working_model)) reactions, $(n_metabolites(working_model)) metabolites"

    return working_model
end

"""
    remove_zero_rows_columns!(model::StandardModel)

Remove metabolites and reactions with all-zero coefficients.
"""
function remove_zero_rows_columns!(model::StandardModel)
    # Get stoichiometric matrix
    S = stoichiometry(model)

    # Find zero columns (reactions)
    zero_reactions = findall(x -> all(iszero, x), eachcol(S))

    # Find zero rows (metabolites)  
    zero_metabolites = findall(x -> all(iszero, x), eachrow(S))

    if !isempty(zero_reactions)
        @info "Removing $(length(zero_reactions)) zero reactions"
        # Remove reactions
        rxn_ids = reactions(model)[zero_reactions]
        for rxn_id in rxn_ids
            delete!(model.reactions, rxn_id)
        end
    end

    if !isempty(zero_metabolites)
        @info "Removing $(length(zero_metabolites)) zero metabolites"
        # Remove metabolites
        met_ids = metabolites(model)[zero_metabolites]
        for met_id in met_ids
            delete!(model.metabolites, met_id)
        end
    end
end

"""
    standardize_bounds!(model::StandardModel, default_lb::Float64, default_ub::Float64)

Standardize reaction bounds to ensure consistent values.
"""
function standardize_bounds!(model::StandardModel, default_lb::Float64, default_ub::Float64)
    for (rid, rxn) in model.reactions
        # Get current bounds
        lb = coalesce(rxn.lower_bound, 0.0)
        ub = coalesce(rxn.upper_bound, default_ub)

        # Standardize bounds
        if lb < 0
            rxn.lower_bound = max(default_lb, lb)
        else
            rxn.lower_bound = max(0.0, lb)
        end

        if ub > 0
            rxn.upper_bound = min(default_ub, ub)
        else
            rxn.upper_bound = min(0.0, ub)
        end

        # Ensure objective reactions are not blocked
        if haskey(model.objective, rid) && model.objective[rid] != 0
            rxn.lower_bound = 0.0
            rxn.upper_bound = default_ub
        end
    end
end

"""
    remove_blocked_reactions!(model::StandardModel, threshold::Float64)

Remove reactions that cannot carry flux.
"""
function remove_blocked_reactions!(model::StandardModel, threshold::Float64)
    # Find blocked reactions using FVA
    constraints = flux_balance_constraints(model)

    blocked_reactions = String[]

    for rid in reactions(model)
        # Check maximum flux
        opt_max = optimized_objective(
            constraints,
            objective=constraints.fluxes[rid].value,
            optimizer=HiGHS.Optimizer,
            sense=Maximal
        )

        # Check minimum flux
        opt_min = optimized_objective(
            constraints,
            objective=constraints.fluxes[rid].value,
            optimizer=HiGHS.Optimizer,
            sense=Minimal
        )

        # If both bounds are below threshold, reaction is blocked
        if abs(opt_max) < threshold && abs(opt_min) < threshold
            push!(blocked_reactions, rid)
        end
    end

    if !isempty(blocked_reactions)
        @info "Removing $(length(blocked_reactions)) blocked reactions"
        for rid in blocked_reactions
            delete!(model.reactions, rid)
        end
    end
end

"""
    check_model_feasibility(model::StandardModel)

Check if the model can achieve a non-zero objective flux.
"""
function check_model_feasibility(model::StandardModel)
    try
        solution = flux_balance_analysis(
            model,
            optimizer=HiGHS.Optimizer
        )

        obj_value = objective_value(solution)
        return !isnothing(obj_value) && abs(obj_value) > 1e-9
    catch e
        @warn "Error checking model feasibility: $e"
        return false
    end
end

"""
    validate_preprocessed_model(model::StandardModel)

Perform final validation checks on the preprocessed model.
"""
function validate_preprocessed_model(model::StandardModel)
    # Check for empty model
    if n_reactions(model) == 0
        error("Model has no reactions after preprocessing")
    end

    if n_metabolites(model) == 0
        error("Model has no metabolites after preprocessing")
    end

    # Check connectivity
    S = stoichiometry(model)
    if rank(Matrix(S)) < min(size(S)...)
        @warn "Stoichiometric matrix may be rank deficient"
    end

    # Check for orphan metabolites
    orphan_metabolites = String[]
    for (mid, met) in model.metabolites
        if !any(r -> haskey(r.stoichiometry, mid), values(model.reactions))
            push!(orphan_metabolites, mid)
        end
    end

    if !isempty(orphan_metabolites)
        @warn "Found $(length(orphan_metabolites)) orphan metabolites"
    end

    @info "Model validation complete"
end

"""
    make_irreversible(model::StandardModel)

Convert all reversible reactions to irreversible reaction pairs.
"""
function make_irreversible(model::StandardModel)
    irreversible_model = StandardModel(model.id)

    # Copy metabolites
    for (mid, met) in model.metabolites
        irreversible_model.metabolites[mid] = deepcopy(met)
    end

    # Process reactions
    for (rid, rxn) in model.reactions
        lb = coalesce(rxn.lower_bound, 0.0)
        ub = coalesce(rxn.upper_bound, 1000.0)

        # Forward reaction (if ub > 0)
        if ub > 0
            fwd_rxn = Reaction(
                name=rxn.name * "_forward",
                lower_bound=max(0.0, lb),
                upper_bound=ub,
                stoichiometry=copy(rxn.stoichiometry),
                gene_association_dnf=rxn.gene_association_dnf,
                annotations=copy(rxn.annotations)
            )
            irreversible_model.reactions[rid*"_fwd"] = fwd_rxn
        end

        # Reverse reaction (if lb < 0)
        if lb < 0
            # Reverse stoichiometry
            rev_stoich = Dict(k => -v for (k, v) in rxn.stoichiometry)

            rev_rxn = Reaction(
                name=rxn.name * "_reverse",
                lower_bound=0.0,
                upper_bound=-lb,
                stoichiometry=rev_stoich,
                gene_association_dnf=rxn.gene_association_dnf,
                annotations=copy(rxn.annotations)
            )
            irreversible_model.reactions[rid*"_rev"] = rev_rxn
        end
    end

    # Update objective
    new_objective = Dict{String,Float64}()
    for (rid, coef) in model.objective
        if haskey(irreversible_model.reactions, rid * "_fwd")
            new_objective[rid*"_fwd"] = coef
        end
        if haskey(irreversible_model.reactions, rid * "_rev")
            new_objective[rid*"_rev"] = -coef
        end
    end
    irreversible_model.objective = new_objective

    return irreversible_model
end

# Placeholder for elementary step splitting
"""
    split_into_elementary_steps(model::StandardModel; mechanism::Symbol = :fixed)

Split reactions into elementary steps based on enzyme mechanisms.
This is a placeholder - implement based on your specific requirements.
"""
function split_into_elementary_steps(model::StandardModel; mechanism::Symbol=:fixed)
    @warn "Elementary step splitting not yet implemented - returning original model"
    return model
end

end # module