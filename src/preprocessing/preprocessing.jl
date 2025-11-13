"""
Model preprocessing utilities for COCOA.jl.

This module provides modular, immutable preprocessing functions following COBREXA patterns.
All functions return modified copies of the model, preserving the original using `deepcopy`.

Based on MATLAB preprocessing pipeline from run_preprocessing.m (Langary et al. 2025).

# Available Functions

- `normalize_bounds` - MATLAB-style bounds normalization
- `remove_orphans` - Remove unused metabolites and reactions  
- `find_blocked_reactions` - Identify blocked reactions via FVA
- `remove_blocked_reactions` - Find and remove blocked reactions
- `split_into_elementary_steps` - Decompose reactions into elementary steps
- `split_into_irreversible` - Convert reversible reactions to irreversible pairs

# Pipeline-Friendly Design

Functions can be composed using Julia's pipe operator:

```julia
using HiGHS
import AbstractFBCModels as A

# Load model
model = A.load("ecoli_core.xml")

# Preprocessing pipeline (immutable - original model preserved)
model_processed = model |>
    normalize_bounds |>
    remove_orphans |>
    m -> remove_blocked_reactions(m, optimizer=HiGHS.Optimizer)[1] |>
    split_into_elementary_steps |>
    split_into_irreversible

# Original model is unchanged
@assert model !== model_processed
```

# Correct Preprocessing Order (per Langary et al. 2025)

1. Normalize bounds
2. Remove blocked reactions (via FVA at 99.9% optimal)
3. **Split into elementary steps** (with reversible reactions)
4. **Then split into irreversible** reactions

# Performance Note

These functions use `deepcopy` to ensure immutability. For very large models (>50k reactions),
if preprocessing time becomes a bottleneck (profile first!), you can manually work with copies.
"""


"""
    normalize_bounds(
        model::A.AbstractFBCModel;
        lower_bound::Float64=-1000.0,
        upper_bound::Float64=1000.0,
        normalize_objective_bounds::Bool=true
    ) -> typeof(model)

Normalize model bounds following MATLAB preprocessing logic (immutable).

Creates a deep copy of the model before applying transformations, preserving the original.

# Transformations Applied

- All negative lower bounds → `lower_bound` (default: -1000, for reversible reactions)
- All positive lower bounds → 0 (irreversible forward)
- All negative upper bounds → 0 (irreversible reverse)
- All positive upper bounds → `upper_bound` (default: 1000, forward unlimited)
- Objective reactions: lb = 0, ub = `upper_bound` (force forward only)

This matches the MATLAB preprocessing in run_preprocessing.m lines 40-46.

# Arguments
- `model::A.AbstractFBCModel`: Model to normalize (will NOT be modified)
- `lower_bound::Float64`: Generic lower bound for reversible reactions (default: -1000)
- `upper_bound::Float64`: Generic upper bound for unlimited reactions (default: 1000)
- `normalize_objective_bounds::Bool`: Force objective reactions to be forward-only

# Returns
- Modified copy of the model with normalized bounds

# Examples
```julia
# Immutable pipeline (original model preserved)
model_normalized = normalize_bounds(model)

# With custom bounds
model_normalized = normalize_bounds(model, lower_bound=-500.0, upper_bound=500.0)
```

# MATLAB Reference
```matlab
model.lb(model.lb<0) = -1000;
model.lb(model.lb>0) = 0;
model.ub(model.ub<0) = 0;
model.ub(model.ub>0) = 1000;
model.lb(model.c~=0) = 0;
model.ub(model.c~=0) = 1000;
```
"""
function normalize_bounds(
    model::A.AbstractFBCModel;
    lower_bound::Float64=-1000.0,
    upper_bound::Float64=1000.0,
    normalize_objective_bounds::Bool=true
)
    # Deep copy to preserve original
    model_copy = deepcopy(model)

    for (_, rxn) in model_copy.reactions
        # MATLAB: model.lb(model.lb<0) = -1000;
        if rxn.lower_bound < 0
            rxn.lower_bound = lower_bound
        end

        # MATLAB: model.lb(model.lb>0) = 0;
        if rxn.lower_bound > 0
            rxn.lower_bound = 0.0
        end

        # MATLAB: model.ub(model.ub<0) = 0;
        if rxn.upper_bound < 0
            rxn.upper_bound = 0.0
        end

        # MATLAB: model.ub(model.ub>0) = 1000;
        if rxn.upper_bound > 0
            rxn.upper_bound = upper_bound
        end

        # MATLAB: model.lb(model.c~=0) = 0; model.ub(model.c~=0) = 1000;
        if normalize_objective_bounds && abs(rxn.objective_coefficient) > 1e-12
            rxn.lower_bound = 0.0
            rxn.upper_bound = upper_bound
        end
    end
    @info "Normalized reaction bounds to lb=$(lower_bound), ub=$(upper_bound)."
    return model_copy
end


"""
    remove_orphans(
        model::A.AbstractFBCModel;
        remove_mets::Bool=true,
        remove_rxns::Bool=true
    ) -> typeof(model)

Remove metabolites and/or reactions with zero stoichiometry (immutable).

Creates a deep copy of the model before removing orphans, preserving the original.
This matches MATLAB's removeRxns and removeMetabolites for empty elements.

# Arguments
- `model::A.AbstractFBCModel`: Model to clean (will NOT be modified)
- `remove_mets::Bool`: Remove metabolites not in any reactions (default: true)
- `remove_rxns::Bool`: Remove reactions with no metabolites (default: true)

# Returns
Modified copy of the model with orphans removed

# Examples
```julia
# Immutable pipeline (original model preserved)
model_cleaned = remove_orphans(model)

# Only remove unused reactions
model_cleaned = remove_orphans(model, remove_mets=false)
```

# MATLAB Reference
```matlab
model = removeRxns(model, model.rxns(find(all(model.S==0))));
model = removeMetabolites(model, model.mets(find(all(model.S'==0))));
```
"""
function remove_orphans(
    model::A.AbstractFBCModel;
    remove_mets::Bool=true,
    remove_rxns::Bool=true
)
    # Deep copy to preserve original
    model_copy = deepcopy(model)

    # Remove reactions with no metabolites
    empty_reactions = String[]
    if remove_rxns
        empty_reactions = [rid for (rid, rxn) in model_copy.reactions if isempty(rxn.stoichiometry)]
        for rid in empty_reactions
            delete!(model_copy.reactions, rid)
        end
    end

    # Remove metabolites not participating in any reaction
    inactive = String[]
    if remove_mets
        active_metabolites = Set{String}()
        for rxn in values(model_copy.reactions)
            union!(active_metabolites, keys(rxn.stoichiometry))
        end

        inactive = [mid for mid in keys(model_copy.metabolites) if !(mid in active_metabolites)]
        for mid in inactive
            delete!(model_copy.metabolites, mid)
        end
    end
    @info "Removed orphans: $(length(empty_reactions)) reactions, $(length(inactive)) metabolites."
    return model_copy
end


"""
    find_blocked_reactions(constraints::COBREXA.C.ConstraintTree;
                           optimizer,
                           flux_tolerance::Float64=1e-9,
                           settings=[],
                           workers=D.workers())

Find blocked reactions using Flux Variability Analysis (FVA) on pre-built constraints.

A reaction is considered blocked if both its minimum and maximum fluxes
are below the flux tolerance threshold. Follows COBREXA.jl patterns by accepting
pre-built constraint trees, allowing users to add objective bounds via constraint
tree merging before calling this function.

This matches MATLAB's FVA-based blocking detection in run_preprocessing.m lines 58-62.

# Arguments
- `constraints::COBREXA.C.ConstraintTree`: Pre-built constraint system (from `flux_balance_constraints`)
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `flux_tolerance::Float64`: Threshold below which reactions are blocked (default: 1e-9)
- `settings`: Additional settings passed to optimizer
- `workers`: Parallel workers for FVA (default: all available via `D.workers()`)

# Returns
- `Vector{String}`: IDs of blocked reactions, or empty vector if model is infeasible

# Examples
```julia
using HiGHS
import COBREXA
import ConstraintTrees as C

# Basic usage - no objective bound
constraints = COBREXA.flux_balance_constraints(model)
blocked = find_blocked_reactions(constraints; optimizer=HiGHS.Optimizer)

# With objective bound (99.9% of optimal) - the COBREXA way
constraints = COBREXA.flux_balance_constraints(model)
obj_flux = COBREXA.optimized_values(
    constraints;
    objective=constraints.objective.value,
    output=constraints.objective,
    optimizer=HiGHS.Optimizer
)
constraints *= :objective_bound^C.Constraint(
    constraints.objective.value,
    COBREXA.relative_tolerance_bound(0.999)(obj_flux)
)
blocked = find_blocked_reactions(constraints; optimizer=HiGHS.Optimizer)

# Or use the model-based convenience wrapper with objective_bound parameter
blocked = find_blocked_reactions(
    model; 
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999)
)
```

# MATLAB Reference
```matlab
[mini, maxi] = linprog_FVA(model, 0.001);  % 0.001 = 0.1% below optimal = 99.9%
thr = 1e-9;
BLK = model.rxns(find(abs(mini)<thr & abs(maxi)<thr));
```
"""
function find_blocked_reactions(
    constraints::COBREXA.C.ConstraintTree;
    optimizer,
    flux_tolerance::Float64=1e-9,
    settings=[],
    workers=D.workers()
)
    # Run variability analysis on all fluxes
    variability = COBREXA.constraints_variability(
        constraints,
        constraints.fluxes;
        optimizer,
        settings,
        workers
    )

    # Find reactions where both min and max are below tolerance
    blocked = String[]
    for (rid, (min_flux, max_flux)) in variability
        # Skip reactions with missing flux values
        if isnothing(min_flux) || isnothing(max_flux)
            @warn "Reaction $rid has missing flux values, skipping blocked check"
            continue
        end
        if abs(min_flux) < flux_tolerance && abs(max_flux) < flux_tolerance
            push!(blocked, String(rid))
        end
    end

    return blocked
end

"""
    find_blocked_reactions(model::A.AbstractFBCModel; optimizer, objective_bound=nothing, kwargs...)

Convenience wrapper that builds constraints from model before finding blocked reactions.

Follows COBREXA pattern: builds constraints, optionally adds objective bound via `*` merge,
then finds blocked reactions. The `objective_bound` parameter should be a bound function
like `relative_tolerance_bound(0.999)` or `absolute_tolerance_bound(1e-5)`, or `nothing`
to skip objective bounding.

# Arguments
- `model::A.AbstractFBCModel`: Model to analyze
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `objective_bound`: Bound function for objective constraint, or `nothing` (default: `nothing`)
- `flux_tolerance::Float64`: Threshold for blocked reactions (default: 1e-9)
- `settings`: Solver settings
- `workers`: Parallel workers

# Examples
```julia
using HiGHS
import COBREXA

# Without objective bound
blocked = find_blocked_reactions(model; optimizer=HiGHS.Optimizer)

# With 99.9% objective bound (MATLAB-style preprocessing)
blocked = find_blocked_reactions(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999)
)
```

See the constraint-tree version for more details.
"""
function find_blocked_reactions(
    model::A.AbstractFBCModel;
    optimizer,
    objective_bound=nothing,
    flux_tolerance::Float64=1e-9,
    settings=[],
    workers=D.workers()
)
    constraints = COBREXA.flux_balance_constraints(model)
    
    # Add objective bound if specified (COBREXA pattern)
    if !isnothing(objective_bound)
        objective = constraints.objective.value
        
        objective_flux = COBREXA.optimized_values(
            constraints;
            objective=constraints.objective.value,
            output=constraints.objective,
            optimizer,
            settings,
        )
        
        isnothing(objective_flux) && return String[]
        
        # Merge objective bound constraint (COBREXA way with *)
        constraints *= :objective_bound^COBREXA.C.Constraint(
            objective,
            objective_bound(objective_flux)
        )
    end
    
    return find_blocked_reactions(
        constraints;
        optimizer,
        flux_tolerance,
        settings,
        workers
    )
end

"""
    remove_blocked_reactions(
        model::A.AbstractFBCModel;
        optimizer,
        objective_bound=nothing,
        flux_tolerance::Float64=1e-9,
        settings=[],
        workers=D.workers()
    ) -> (typeof(model), Vector{String})

Find and remove blocked reactions (immutable).

Creates a deep copy of the model before removing blocked reactions, preserving the original.
Follows COBREXA pattern for objective bounding.

# Arguments
- `model::A.AbstractFBCModel`: Model to process (will NOT be modified)
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `objective_bound`: Bound function for objective (e.g., `relative_tolerance_bound(0.999)`), or `nothing` (default: `nothing`)
- `flux_tolerance`: Threshold for considering a flux zero (default: 1e-9)
- `settings`: Solver settings (default: [])
- `workers`: Distributed workers for parallel computation (default: Distributed.workers())

# Returns
Tuple of:
- Modified model (copy with blocked reactions removed)
- Vector of removed reaction IDs

# Examples
```julia
using HiGHS
import COBREXA

# Without objective bound
model_unblocked, blocked_ids = remove_blocked_reactions(
    model;
    optimizer=HiGHS.Optimizer
)

# With 99.9% objective bound (MATLAB-style preprocessing)
model_unblocked, blocked_ids = remove_blocked_reactions(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999)
)
```
"""
function remove_blocked_reactions(
    model::A.AbstractFBCModel;
    optimizer,
    objective_bound=nothing,
    flux_tolerance::Float64=1e-9,
    settings=[],
    workers=D.workers()
)
    # Deep copy to preserve original
    model_copy = deepcopy(model)

    # Find blocked reactions (uses convenience wrapper with objective_bound)
    blocked = find_blocked_reactions(
        model_copy;
        optimizer,
        objective_bound,
        flux_tolerance,
        settings,
        workers
    )

    # Remove from the copy
    for rid in blocked
        haskey(model_copy.reactions, rid) && delete!(model_copy.reactions, rid)
    end

    @info "Removed $(length(blocked)) blocked reactions."
    return model_copy, blocked
end
