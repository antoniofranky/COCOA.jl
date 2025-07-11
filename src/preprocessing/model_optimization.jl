"""
Model structure optimization for model preparation.

This module handles:
- Reversible reaction splitting
- Zero stoichiometry removal
- Reaction removal utilities
- Model structure cleanup
"""

import AbstractFBCModels.CanonicalModel as CM

"""
Count reversible reactions without allocating unnecessary arrays.
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
Optimized version that avoids deepcopy and excessive allocations.
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
    end
end

"""
Optimized version using Sets for faster lookups.
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
Efficient batch removal of reactions.
"""
function remove_reactions!(model::CM.Model, rids::Vector{String})
    for rid in rids
        delete!(model.reactions, rid)
    end
    return model
end