export build_S_matrix_from_flux_stoichiometry, build_Y_matrix_from_constraints, build_A_matrix_from_constraints

"""
Matrix building functions for COCOA - Extract structural matrices from constraint trees.
"""

"""
$(TYPEDSIGNATURES)

Extract reaction names from balance constraints, handling both standard and split flux variables.
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
$(TYPEDSIGNATURES)

Build S matrix (metabolite-reaction stoichiometry matrix) directly from flux_stoichiometry constraints.
This is much simpler than parsing the constraint structure - the flux_stoichiometry directly 
gives us the stoichiometric coefficients.

# Arguments
- `constraints`: Constraint tree with flux_stoichiometry 

# Returns
- `S_matrix`: Sparse matrix where S[i,j] = stoichiometry of metabolite i in reaction j
- `metabolite_ids`: Ordered vector of metabolite IDs
- `reaction_ids`: Ordered vector of reaction IDs  
"""
function build_S_matrix_from_flux_stoichiometry(constraints::C.ConstraintTree)
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

    return S_matrix, metabolite_ids, reaction_ids
end

"""
$(TYPEDSIGNATURES)

Build Y matrix (metabolite-complex stoichiometry matrix) from constraints.

# Arguments
- `constraints`: Constraint tree with flux_stoichiometry constraints

# Returns
- `Y_matrix`: Sparse matrix where Y[i,j] = stoichiometry of metabolite i in complex j  
- `metabolite_ids`: Ordered vector of metabolite IDs
- `complex_ids`: Ordered vector of complex IDs
"""
function build_Y_matrix_from_constraints(constraints::C.ConstraintTree)
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

    return Y_matrix, metabolite_ids, complex_ids
end

"""
$(TYPEDSIGNATURES)

Build A matrix (complex-reaction incidence matrix) from constraint tree.
Each row represents a complex, each column a reaction.
A[i,j] = -1 if complex i is consumed in reaction j
A[i,j] = +1 if complex i is produced in reaction j
A[i,j] = 0 if complex i doesn't participate in reaction j

# Arguments
- `constraints`: Constraint tree with balance constraints and activities

# Returns
- `A_matrix`: Sparse matrix where A[i,j] represents incidence of complex i in reaction j
- `complex_ids`: Ordered vector of complex IDs  
- `reaction_ids`: Ordered vector of reaction IDs
"""
function build_A_matrix_from_constraints(constraints::C.ConstraintTree)
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Get reaction names from the balance constraints
    reaction_names = get_reaction_names_from_constraints(balance_constraints)
    reaction_ids = sort!(Symbol.(reaction_names); by=string)

    # Get complex IDs from activities constraints
    if !haskey(constraints, :activities)
        error("Constraints must have activities to build A matrix")
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

    return A_matrix, complex_ids, reaction_ids
end