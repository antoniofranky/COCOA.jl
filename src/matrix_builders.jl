"""
    Matrix Building Functions for COCOA.jl

This module provides functions to construct structural matrices from metabolic models
for chemical reaction network analysis. The functions extract stoichiometric and 
incidence matrices that are fundamental to concordance analysis and kinetic module
identification.

# Mathematical Notation (following Küken et al. 2022, Science Advances)
- N: Stoichiometric matrix (metabolites × reactions), N = YA
- Y: Species-complex composition matrix (metabolites × complexes)
- A: Complex-reaction incidence matrix (complexes × reactions)

# Functions
- [`stoichiometry`](@ref): Build metabolite-reaction stoichiometric matrix N
- [`complex_stoichiometry`](@ref): Build metabolite-complex composition matrix Y  
- [`incidence`](@ref): Build complex-reaction incidence matrix A

# Legacy Aliases (for backward compatibility)
- `S_from_constraints` → `stoichiometry`
- `Y_matrix_from_constraints` → `complex_stoichiometry`
- `A_matrix_from_constraints` → `incidence`
- `A_matrix_from_model` → `incidence`

# Dependencies
Requires ConstraintTrees.jl for constraint tree processing, SparseArrays.jl for
efficient sparse matrix construction, and AbstractFBCModels.jl for model interfaces.
"""

# ========================================================================================
# Helper Functions
# ========================================================================================

"""
    get_reaction_names_from_constraints(balance_constraints::C.ConstraintTree)

Extract reaction names from balance constraints, handling both standard and split flux variables.

This function processes constraint trees to identify all reaction names, supporting both
unidirectional (split into forward/reverse) and bidirectional flux representations.

# Arguments
- `balance_constraints::C.ConstraintTree`: Constraint tree containing flux variables

# Returns
- `Vector{String}`: Collection of reaction names with appropriate suffixes for split fluxes

# Notes
- For split fluxes: adds "_forward" and "_reverse" suffixes
- For standard fluxes: uses original reaction names
- Prioritizes split flux representation when available
"""
function get_reaction_names_from_constraints(balance_constraints::C.ConstraintTree)
    reaction_names = Set{String}()

    # Get reaction names from forward fluxes
    if haskey(balance_constraints, :fluxes_forward)
        for rxn_name in keys(balance_constraints.fluxes_forward)
            push!(reaction_names, string(rxn_name) * "_forward")
        end
    end

    # Get reaction names from reverse fluxes
    if haskey(balance_constraints, :fluxes_reverse)
        for rxn_name in keys(balance_constraints.fluxes_reverse)
            push!(reaction_names, string(rxn_name) * "_reverse")
        end
    end

    # Only use standard fluxes if there are no split fluxes (fallback for non-unidirectional)
    if !haskey(balance_constraints, :fluxes_forward) && !haskey(balance_constraints, :fluxes_reverse) && haskey(balance_constraints, :fluxes)
        for rxn_name in keys(balance_constraints.fluxes)
            push!(reaction_names, string(rxn_name))
        end
    end

    return collect(reaction_names)
end

# ========================================================================================
# Stoichiometric Matrix N (metabolites × reactions)
# ========================================================================================

"""
    stoichiometry(constraints::C.ConstraintTree; return_ids::Bool=false)

Build stoichiometric matrix N from constraint tree flux stoichiometry.

Constructs the metabolite-reaction stoichiometric matrix directly from flux_stoichiometry 
constraints in the constraint tree. This matrix represents the stoichiometric coefficients
of metabolites in reactions, where N[i,j] gives the coefficient of metabolite i in reaction j.
Follows notation from Küken et al. (2022), Science Advances: N = YA.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing flux_stoichiometry data
- `return_ids::Bool=false`: If true, also return metabolite and reaction ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Float64,Int}` - The stoichiometric matrix N
- If `return_ids=true`: `Tuple` containing:
  - `N::SparseMatrixCSC{Float64,Int}`: Stoichiometric matrix (metabolites × reactions)
  - `metabolite_ids::Vector{Symbol}`: Ordered vector of metabolite identifiers
  - `reaction_ids::Vector{Symbol}`: Ordered vector of reaction identifiers

# Matrix Structure
- Rows: metabolites (ordered alphabetically by string representation)
- Columns: reactions (ordered alphabetically by string representation)
- Values: stoichiometric coefficients (negative for substrates, positive for products)

# Examples
```julia
# Get just the matrix
S = stoichiometry(constraints)

# Get matrix with ID mappings
S, metabolites, reactions = stoichiometry(constraints; return_ids=true)
```
"""
function stoichiometry(constraints::C.ConstraintTree; return_ids::Bool=false, model::Union{A.AbstractFBCModel,Nothing}=nothing)
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Get metabolite IDs - preserve original model order if model provided
    if model !== nothing
        metabolite_ids = Symbol.(A.metabolites(model))
    else
        # Fallback to insertion order from ConstraintTree keys
        metabolite_ids = collect(keys(balance_constraints.flux_stoichiometry))
    end

    # Get reaction names from the balance constraints
    reaction_names = get_reaction_names_from_constraints(balance_constraints)
    if model !== nothing
        # Preserve original model reaction order, filtering to only those present in constraints
        original_rxn_ids = Symbol.(A.reactions(model))
        reaction_name_set = Set(Symbol.(reaction_names))
        reaction_ids = filter(id -> id in reaction_name_set, original_rxn_ids)
    else
        # Fallback to insertion order
        reaction_ids = Symbol.(reaction_names)
    end

    n_metabolites = length(metabolite_ids)
    n_reactions = length(reaction_ids)

    # Build metabolite index mapping
    metabolite_to_idx = Dict(met => i for (i, met) in enumerate(metabolite_ids))

    # Create variable index to reaction index mapping
    var_to_reaction_idx = Dict{Int,Int}()

    # Build reaction index mapping
    reaction_to_idx = Dict(rxn => j for (j, rxn) in enumerate(reaction_ids))

    # Map forward reaction variables
    if haskey(balance_constraints, :fluxes_forward)
        for (rxn_name, constraint) in balance_constraints.fluxes_forward
            rxn_name_forward = Symbol(string(rxn_name) * "_forward")
            if haskey(reaction_to_idx, rxn_name_forward)
                reaction_idx = reaction_to_idx[rxn_name_forward]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Map reverse reaction variables
    if haskey(balance_constraints, :fluxes_reverse)
        for (rxn_name, constraint) in balance_constraints.fluxes_reverse
            rxn_name_reverse = Symbol(string(rxn_name) * "_reverse")
            if haskey(reaction_to_idx, rxn_name_reverse)
                reaction_idx = reaction_to_idx[rxn_name_reverse]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Map standard reaction variables (if no split)
    if haskey(balance_constraints, :fluxes)
        for (rxn_name, constraint) in balance_constraints.fluxes
            if haskey(reaction_to_idx, rxn_name)
                reaction_idx = reaction_to_idx[rxn_name]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Pre-allocate triplets for sparse matrix construction
    I = Int[]
    J = Int[]
    V = Float64[]

    # Build S matrix directly from flux_stoichiometry constraints
    # Each constraint is: sum(coeff * x[var_idx]) = 0 for one metabolite
    for (met_id, constraint) in balance_constraints.flux_stoichiometry
        i = metabolite_to_idx[met_id]

        var_indices = constraint.value.idxs
        coefficients = constraint.value.weights

        for (var_idx, coeff) in zip(var_indices, coefficients)
            if haskey(var_to_reaction_idx, var_idx)
                j = var_to_reaction_idx[var_idx]
                push!(I, i)
                push!(J, j)
                push!(V, Float64(coeff))
            end
        end
    end

    S_matrix = SparseArrays.sparse(I, J, V, n_metabolites, n_reactions)

    if return_ids
        return S_matrix, metabolite_ids, reaction_ids
    else
        return S_matrix
    end
end

"""
    stoichiometry(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)

Build stoichiometric matrix S directly from an AbstractFBCModel.

This function constructs the stoichiometric matrix by first building a constraint tree
(optionally with unidirectional reactions), then extracting the stoichiometry.

# Arguments
- `model::A.AbstractFBCModel`: FBC model containing reactions and metabolites
- `return_ids::Bool=false`: If true, also return metabolite and reaction ID vectors
- `use_unidirectional::Bool=true`: If true, split reactions into forward/reverse

# Returns
Same as `stoichiometry(::C.ConstraintTree)`

# Examples
```julia
model = load_model("model.xml")
S = stoichiometry(model)
S, metabolites, reactions = stoichiometry(model; return_ids=true)
```
"""
function stoichiometry(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)
    # Build constraints to extract stoichiometry consistently
    if use_unidirectional
        constraints, _ = create_unidirectional_constraints(model)
    else
        constraints = COBREXA.flux_balance_constraints(model)
    end

    return stoichiometry(constraints; return_ids=return_ids, model=model)
end

# ========================================================================================
# Complex Composition Matrix Y (species-complex matrix, metabolites × complexes)
# ========================================================================================

"""
    complex_stoichiometry(constraints::C.ConstraintTree; return_ids::Bool=false)

Build species-complex composition matrix Y from constraint tree.

Constructs the metabolite-complex stoichiometric matrix from the constraint tree by
extracting complex compositions. This matrix represents the stoichiometric coefficients
of metabolites in complexes, where Y[i,j] gives the coefficient of metabolite i in complex j.
Follows notation from Küken et al. (2022), Science Advances.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing complex and flux constraint data
- `return_ids::Bool=false`: If true, also return metabolite and complex ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Float64,Int}` - The complex composition matrix Y
- If `return_ids=true`: `Tuple` containing:
  - `Y::SparseMatrixCSC{Float64,Int}`: Complex composition matrix (metabolites × complexes)
  - `metabolite_ids::Vector{Symbol}`: Ordered vector of metabolite identifiers
  - `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers

# Matrix Structure
- Rows: metabolites (ordered alphabetically by string representation)
- Columns: complexes (ordered alphabetically by string representation)  
- Values: stoichiometric coefficients (typically positive, representing composition)

# Examples
```julia
# Get just the matrix
Y = complex_stoichiometry(constraints)

# Get matrix with ID mappings
Y, metabolites, complexes = complex_stoichiometry(constraints; return_ids=true)
```
"""
function complex_stoichiometry(constraints::C.ConstraintTree; return_ids::Bool=false, model::Union{A.AbstractFBCModel,Nothing}=nothing)
    # Extract complexes using the shared function
    complex_info, _ = extract_complexes(constraints)

    # Get complex IDs - preserve insertion order from ConstraintTree
    # Complexes are generated in a deterministic order by extract_complexes
    complex_ids = collect(keys(complex_info))

    # Get all unique metabolites
    all_metabolites = Set{Symbol}()
    for composition in values(complex_info)
        for (met_id, _) in composition
            push!(all_metabolites, met_id)
        end
    end

    # Preserve original model order for metabolites if model provided
    if model !== nothing
        original_met_ids = Symbol.(A.metabolites(model))
        metabolite_ids = filter(id -> id in all_metabolites, original_met_ids)
    else
        # Fallback to collection order
        metabolite_ids = collect(all_metabolites)
    end

    n_metabolites = length(metabolite_ids)
    n_complexes = length(complex_ids)

    # Build index mappings
    metabolite_to_idx = Dict(met => i for (i, met) in enumerate(metabolite_ids))
    complex_to_idx = Dict(comp => j for (j, comp) in enumerate(complex_ids))

    # Pre-allocate triplets for sparse matrix construction
    I = Int[]
    J = Int[]
    V = Float64[]

    for (complex_id, composition) in complex_info
        j = complex_to_idx[complex_id]
        for (met_id, coeff) in composition
            i = metabolite_to_idx[met_id]
            push!(I, i)
            push!(J, j)
            push!(V, Float64(coeff))
        end
    end

    Y_matrix = SparseArrays.sparse(I, J, V, n_metabolites, n_complexes)

    if return_ids
        return Y_matrix, metabolite_ids, complex_ids
    else
        return Y_matrix
    end
end

"""
    complex_stoichiometry(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)

Build species-complex composition matrix Y directly from an AbstractFBCModel.

This function constructs the Y matrix by first building a constraint tree
(optionally with unidirectional reactions), then extracting complexes using the same
logic as `extract_complexes` to ensure consistency.

# Arguments
- `model::A.AbstractFBCModel`: FBC model containing reactions and metabolites
- `return_ids::Bool=false`: If true, also return metabolite and complex ID vectors
- `use_unidirectional::Bool=true`: If true, split reactions into forward/reverse (matches concordance analysis)

# Returns
Same as `complex_stoichiometry(::C.ConstraintTree)`

# Examples
```julia
model = load_model("model.xml")
Y = complex_stoichiometry(model)
Y, metabolites, complexes = complex_stoichiometry(model; return_ids=true)

# Without unidirectional splitting (may give different complexes)
Y = complex_stoichiometry(model; use_unidirectional=false)
```

# Notes
To ensure complex IDs match those from concordance analysis, use `use_unidirectional=true` (default).
This will build constraints and extract complexes using the same logic as `extract_complexes`.
"""
function complex_stoichiometry(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)
    # Build constraints to extract complexes consistently
    if use_unidirectional
        constraints, _ = create_unidirectional_constraints(model)
    else
        constraints = COBREXA.flux_balance_constraints(model)
    end

    return complex_stoichiometry(constraints; return_ids=return_ids, model=model)
end

# ========================================================================================
# Complex-Reaction Incidence Matrix A (complexes × reactions)
# ========================================================================================

"""
    incidence(constraints::C.ConstraintTree; return_ids::Bool=false)

Build complex-reaction incidence matrix A from constraint tree.

Constructs the incidence matrix representing the participation of complexes in reactions.
This matrix encodes the reaction network structure in chemical reaction network theory,
where A[i,j] indicates how complex i participates in reaction j.
Follows notation from Küken et al. (2022), Science Advances.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing balance constraints and activities
- `return_ids::Bool=false`: If true, also return complex and reaction ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Int,Int}` - The incidence matrix A
- If `return_ids=true`: `Tuple` containing:
  - `A::SparseMatrixCSC{Int,Int}`: Incidence matrix (complexes × reactions)
  - `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers
  - `reaction_ids::Vector{Symbol}`: Ordered vector of reaction identifiers

# Matrix Structure
- Rows: complexes (ordered alphabetically by string representation)
- Columns: reactions (ordered alphabetically by string representation)
- Values: 
  - `-1`: complex i is consumed (substrate) in reaction j
  - `+1`: complex i is produced (product) in reaction j
  - `0`: complex i does not participate in reaction j

# Requirements
The constraint tree must contain an `activities` section that defines complex activities
as linear combinations of reaction fluxes.

# Examples
```julia
# Get just the matrix
A = incidence(constraints)

# Get matrix with ID mappings
A, complexes, reactions = incidence(constraints; return_ids=true)
```

# Throws
- `ArgumentError`: If constraints do not contain required `activities` section
"""
function incidence(constraints::C.ConstraintTree; return_ids::Bool=false, model::Union{A.AbstractFBCModel,Nothing}=nothing)
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Get reaction names from the balance constraints
    reaction_names = get_reaction_names_from_constraints(balance_constraints)
    if model !== nothing
        # Preserve original model reaction order, filtering to only those present in constraints
        original_rxn_ids = Symbol.(A.reactions(model))
        reaction_name_set = Set(Symbol.(reaction_names))
        reaction_ids = filter(id -> id in reaction_name_set, original_rxn_ids)
    else
        # Fallback to insertion order
        reaction_ids = Symbol.(reaction_names)
    end

    # Get complex IDs from activities constraints - preserve insertion order
    if !haskey(constraints, :activities)
        throw(ArgumentError("Constraints must have activities to build incidence matrix"))
    end
    complex_ids = collect(keys(constraints.activities))

    n_complexes = length(complex_ids)
    n_reactions = length(reaction_ids)

    # Build index mappings
    complex_to_idx = Dict(comp => i for (i, comp) in enumerate(complex_ids))
    reaction_to_idx = Dict(rxn => j for (j, rxn) in enumerate(reaction_ids))

    # Create variable index to reaction index mapping
    var_to_reaction_idx = Dict{Int,Int}()

    # Map forward reaction variables
    if haskey(balance_constraints, :fluxes_forward)
        for (rxn_name, constraint) in balance_constraints.fluxes_forward
            rxn_name_forward = Symbol(string(rxn_name) * "_forward")
            if haskey(reaction_to_idx, rxn_name_forward)
                reaction_idx = reaction_to_idx[rxn_name_forward]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Map reverse reaction variables
    if haskey(balance_constraints, :fluxes_reverse)
        for (rxn_name, constraint) in balance_constraints.fluxes_reverse
            rxn_name_reverse = Symbol(string(rxn_name) * "_reverse")
            if haskey(reaction_to_idx, rxn_name_reverse)
                reaction_idx = reaction_to_idx[rxn_name_reverse]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Map standard reaction variables (if no split)
    if haskey(balance_constraints, :fluxes)
        for (rxn_name, constraint) in balance_constraints.fluxes
            if haskey(reaction_to_idx, rxn_name)
                reaction_idx = reaction_to_idx[rxn_name]
                for var_idx in constraint.value.idxs
                    var_to_reaction_idx[var_idx] = reaction_idx
                end
            end
        end
    end

    # Pre-allocate triplets for sparse matrix construction
    I = Int[]
    J = Int[]
    V = Int[]

    # Build A matrix from activities constraints
    # Activities are defined as: activity = sum of reaction fluxes where complex is produced - consumed
    # So coefficient in activity constraint tells us the incidence
    for (complex_id, constraint) in constraints.activities
        if haskey(complex_to_idx, complex_id)
            i = complex_to_idx[complex_id]

            var_indices = constraint.value.idxs
            coefficients = constraint.value.weights

            for (var_idx, coeff) in zip(var_indices, coefficients)
                if haskey(var_to_reaction_idx, var_idx)
                    j = var_to_reaction_idx[var_idx]
                    push!(I, i)
                    push!(J, j)
                    push!(V, Int(coeff))  # -1 for consumed, +1 for produced
                end
            end
        end
    end

    A_matrix = SparseArrays.sparse(I, J, V, n_complexes, n_reactions)

    if return_ids
        return A_matrix, complex_ids, reaction_ids
    else
        return A_matrix
    end
end

"""
    incidence(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)

Build complex-reaction incidence matrix A directly from an AbstractFBCModel.

This function constructs the incidence matrix by first building a constraint tree
(optionally with unidirectional reactions), then extracting complexes using the same
logic as `extract_complexes` to ensure consistency.

# Arguments
- `model::A.AbstractFBCModel`: FBC model containing reactions and metabolites
- `return_ids::Bool=false`: If true, also return complex and reaction ID vectors
- `use_unidirectional::Bool=true`: If true, split reactions into forward/reverse (matches concordance analysis)

# Returns
Same as `incidence(::C.ConstraintTree)`

# Examples
```julia
model = load_model("model.xml")
A = incidence(model)
A, complexes, reactions = incidence(model; return_ids=true)

# Without unidirectional splitting (may give different complexes)
A = incidence(model; use_unidirectional=false)
```

# Notes
To ensure complex IDs match those from concordance analysis, use `use_unidirectional=true` (default).
This will build constraints and extract complexes using the same logic as `extract_complexes`.
"""
function incidence(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=false)
    # Build constraints to extract complexes consistently
    if use_unidirectional
        constraints = concordance_constraints(model, use_unidirectional_constraints=true)
    else
        constraints = concordance_constraints(model, use_unidirectional_constraints=false)
    end

    return incidence(constraints; return_ids=return_ids, model=model)
end

# ========================================================================================
# Legacy Aliases (for backward compatibility)
# ========================================================================================

"""Alias for `stoichiometry(::C.ConstraintTree)`"""
const S_from_constraints = stoichiometry

"""Alias for `complex_stoichiometry(::C.ConstraintTree)`"""
const Y_matrix_from_constraints = complex_stoichiometry

"""Alias for `incidence(::C.ConstraintTree)`"""
const A_matrix_from_constraints = incidence

"""Alias for `incidence(::A.AbstractFBCModel)`"""
const A_matrix_from_model = incidence
