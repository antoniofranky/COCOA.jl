"""
    Matrix Building Functions for COCOA.jl

This module provides functions to construct structural matrices from constraint trees
for chemical reaction network analysis. The functions extract stoichiometric and 
incidence matrices that are fundamental to concordance analysis and kinetic module
identification.

# Functions
- [`S_from_constraints`](@ref): Build metabolite-reaction stoichiometric matrix
- [`Y_matrix_from_constraints`](@ref): Build metabolite-complex stoichiometric matrix  
- [`A_matrix_from_constraints`](@ref): Build complex-reaction incidence matrix

# Dependencies
Requires ConstraintTrees.jl for constraint tree processing and SparseArrays.jl for
efficient sparse matrix construction.
"""

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

"""
    S_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)

Build stoichiometric matrix S from flux stoichiometry constraints.

Constructs the metabolite-reaction stoichiometric matrix directly from flux_stoichiometry 
constraints in the constraint tree. This matrix represents the stoichiometric coefficients
of metabolites in reactions, where S[i,j] gives the coefficient of metabolite i in reaction j.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing flux_stoichiometry data
- `return_ids::Bool=false`: If true, also return metabolite and reaction ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Float64,Int}` - The stoichiometric matrix S
- If `return_ids=true`: `Tuple` containing:
  - `S_matrix::SparseMatrixCSC{Float64,Int}`: Stoichiometric matrix (metabolites × reactions)
  - `metabolite_ids::Vector{Symbol}`: Ordered vector of metabolite identifiers
  - `reaction_ids::Vector{Symbol}`: Ordered vector of reaction identifiers

# Matrix Structure
- Rows: metabolites (ordered alphabetically by string representation)
- Columns: reactions (ordered alphabetically by string representation)
- Values: stoichiometric coefficients (negative for substrates, positive for products)

# Examples
```julia
# Get just the matrix
S = S_from_constraints(constraints)

# Get matrix with ID mappings
S, metabolites, reactions = S_from_constraints(constraints; return_ids=true)
```
"""
function S_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Get metabolite IDs from flux_stoichiometry keys (already sorted)
    metabolite_ids = sort!(collect(keys(balance_constraints.flux_stoichiometry)); by=string)

    # Get reaction names from the balance constraints
    reaction_names = get_reaction_names_from_constraints(balance_constraints)
    reaction_ids = sort!(Symbol.(reaction_names); by=string)

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
    Y_matrix_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)

Build complex stoichiometric matrix Y from constraint tree.

Constructs the metabolite-complex stoichiometric matrix from the constraint tree by
extracting complex compositions. This matrix represents the stoichiometric coefficients
of metabolites in complexes, where Y[i,j] gives the coefficient of metabolite i in complex j.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing complex and flux constraint data
- `return_ids::Bool=false`: If true, also return metabolite and complex ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Float64,Int}` - The complex stoichiometric matrix Y
- If `return_ids=true`: `Tuple` containing:
  - `Y_matrix::SparseMatrixCSC{Float64,Int}`: Complex stoichiometric matrix (metabolites × complexes)
  - `metabolite_ids::Vector{Symbol}`: Ordered vector of metabolite identifiers
  - `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers

# Matrix Structure
- Rows: metabolites (ordered alphabetically by string representation)
- Columns: complexes (ordered alphabetically by string representation)  
- Values: stoichiometric coefficients (typically positive, representing composition)

# Examples
```julia
# Get just the matrix
Y = Y_matrix_from_constraints(constraints)

# Get matrix with ID mappings
Y, metabolites, complexes = Y_matrix_from_constraints(constraints; return_ids=true)
```
"""
function Y_matrix_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)
    # Extract complexes using the shared function
    complex_info, _ = extract_complexes(constraints)

    # Get complex IDs in canonical order
    complex_ids = sort!(collect(keys(complex_info)); by=string)

    # Get all unique metabolites
    all_metabolites = Set{Symbol}()
    for composition in values(complex_info)
        for (met_id, _) in composition
            push!(all_metabolites, met_id)
        end
    end
    metabolite_ids = sort!(collect(all_metabolites); by=string)

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
    A_matrix_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)

Build complex-reaction incidence matrix A from constraint tree.

Constructs the incidence matrix representing the participation of complexes in reactions.
This matrix encodes the reaction network structure in chemical reaction network theory,
where A[i,j] indicates how complex i participates in reaction j.

# Arguments
- `constraints::C.ConstraintTree`: Constraint tree containing balance constraints and activities
- `return_ids::Bool=false`: If true, also return complex and reaction ID vectors

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Int,Int}` - The incidence matrix A
- If `return_ids=true`: `Tuple` containing:
  - `A_matrix::SparseMatrixCSC{Int,Int}`: Incidence matrix (complexes × reactions)
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
A = A_matrix_from_constraints(constraints)

# Get matrix with ID mappings
A, complexes, reactions = A_matrix_from_constraints(constraints; return_ids=true)
```

# Throws
- `ArgumentError`: If constraints do not contain required `activities` section
"""
function A_matrix_from_constraints(constraints::C.ConstraintTree; return_ids::Bool=false)
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Get reaction names from the balance constraints
    reaction_names = get_reaction_names_from_constraints(balance_constraints)
    reaction_ids = sort!(Symbol.(reaction_names); by=string)

    # Get complex IDs from activities constraints
    if !haskey(constraints, :activities)
        throw(ArgumentError("Constraints must have activities to build A matrix"))
    end
    complex_ids = sort!(collect(keys(constraints.activities)); by=string)

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
    if haskey(constraints, :activities)
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
    end

    A_matrix = SparseArrays.sparse(I, J, V, n_complexes, n_reactions)

    if return_ids
        return A_matrix, complex_ids, reaction_ids
    else
        return A_matrix
    end
end

# ========================================================================================
# Direct Matrix Extraction from AbstractFBCModels (similar to A.stoichiometry pattern)
# ========================================================================================

"""
    A_matrix_from_model(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)

Build complex-reaction incidence matrix A directly from an AbstractFBCModel.

This function constructs the incidence matrix by first building a constraint tree
(optionally with unidirectional reactions), then extracting complexes using the same
logic as `extract_complexes` to ensure consistency.

# Arguments
- `model::A.AbstractFBCModel`: FBC model containing reactions and metabolites
- `return_ids::Bool=false`: If true, also return complex and reaction ID vectors
- `use_unidirectional::Bool=true`: If true, split reactions into forward/reverse (matches concordance analysis)

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Int,Int}` - The incidence matrix A
- If `return_ids=true`: `Tuple` containing:
  - `A_matrix::SparseMatrixCSC{Int,Int}`: Incidence matrix (complexes × reactions)
  - `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers  
  - `reaction_ids::Vector{Symbol}`: Ordered vector of reaction identifiers

# Matrix Structure
- Rows: complexes (ordered alphabetically)
- Columns: reactions (ordered alphabetically)
- Values: -1 (substrate), +1 (product), 0 (no participation)

# Examples
```julia
model = load_model("model.xml")
A = A_matrix_from_model(model)
A, complexes, reactions = A_matrix_from_model(model; return_ids=true)

# Without unidirectional splitting (may give different complexes)
A = A_matrix_from_model(model; use_unidirectional=false)
```

# Notes
To ensure complex IDs match those from concordance analysis, use `use_unidirectional=true` (default).
This will build constraints and extract complexes using the same logic as `extract_complexes`.
"""
function A_matrix_from_model(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)
    # Build constraints to extract complexes consistently
    if use_unidirectional
        constraints, _ = create_unidirectional_constraints(model)
    else
        constraints = COBREXA.flux_balance_constraints(model)
    end

    # Use the same extraction logic as constraints.jl
    return A_matrix_from_constraints(constraints; return_ids=return_ids)
end

"""
    Y_matrix_from_model(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)

Build metabolite-complex stoichiometric matrix Y directly from an AbstractFBCModel.

This function constructs the Y matrix by first building a constraint tree
(optionally with unidirectional reactions), then extracting complexes using the same
logic as `extract_complexes` to ensure consistency.

# Arguments
- `model::A.AbstractFBCModel`: FBC model containing reactions and metabolites
- `return_ids::Bool=false`: If true, also return metabolite and complex ID vectors
- `use_unidirectional::Bool=true`: If true, split reactions into forward/reverse (matches concordance analysis)

# Returns
- If `return_ids=false`: `SparseMatrixCSC{Float64,Int}` - The Y matrix
- If `return_ids=true`: `Tuple` containing:
  - `Y_matrix::SparseMatrixCSC{Float64,Int}`: Stoichiometric matrix (metabolites × complexes)
  - `metabolite_ids::Vector{Symbol}`: Ordered vector of metabolite identifiers
  - `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers

# Matrix Structure
- Rows: metabolites (ordered alphabetically)
- Columns: complexes (ordered alphabetically)
- Values: stoichiometric coefficients in complex composition

# Examples
```julia
model = load_model("model.xml")
Y = Y_matrix_from_model(model)
Y, metabolites, complexes = Y_matrix_from_model(model; return_ids=true)

# Without unidirectional splitting (may give different complexes)
Y = Y_matrix_from_model(model; use_unidirectional=false)
```

# Notes
To ensure complex IDs match those from concordance analysis, use `use_unidirectional=true` (default).
This will build constraints and extract complexes using the same logic as `extract_complexes`.
"""
function Y_matrix_from_model(model::A.AbstractFBCModel; return_ids::Bool=false, use_unidirectional::Bool=true)
    # Build constraints to extract complexes consistently
    if use_unidirectional
        constraints, _ = create_unidirectional_constraints(model)
    else
        constraints = COBREXA.flux_balance_constraints(model)
    end

    # Use the same extraction logic as constraints.jl
    return Y_matrix_from_constraints(constraints; return_ids=return_ids)
end