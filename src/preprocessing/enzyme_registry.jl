"""
Enzyme registry functionality for elementary step splitting.

This module handles:
- Enzyme discovery from gene associations
- Enzyme registry management
- Reaction-enzyme mapping
"""

using Random
import AbstractFBCModels.CanonicalModel as CM

export extract_reaction_enzymes, build_enzyme_registry

"""
Build a registry of all enzymes from gene associations.
"""
function build_enzyme_registry(model::CM.Model)
    enzyme_registry = Dict{String,String}()
    enzyme_counter = 0

    for (rid, rxn) in model.reactions
        if !isnothing(rxn.gene_association_dnf)
            for gene_group in rxn.gene_association_dnf
                enzyme_counter += 1
                if length(gene_group) == 1
                    # Single gene = single enzyme
                    enzyme_id = "ENZ_$(gene_group[1])"
                    enzyme_registry[enzyme_id] = gene_group[1]
                else
                    # Multiple genes = enzyme intermediate
                    intermediate_name = join(sort(gene_group), "_")
                    enzyme_id = "ENZ_$intermediate_name"
                    enzyme_registry[enzyme_id] = join(sort(gene_group), " & ")
                end
            end
        end
    end

    return enzyme_registry
end

"""
Extract enzyme IDs for a reaction based on gene associations.
"""
function extract_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String})
    enzyme_ids = String[]

    if !isnothing(rxn.gene_association_dnf)
        for gene_group in rxn.gene_association_dnf
            if length(gene_group) == 1
                enzyme_id = "ENZ_$(gene_group[1])"
            else
                intermediate_name = join(sort(gene_group), "_")
                enzyme_id = "ENZ_$intermediate_name"
            end

            if haskey(enzyme_registry, enzyme_id)
                push!(enzyme_ids, enzyme_id)
            end
        end
    end

    return enzyme_ids
end

"""
Count the number of enzyme variants (isoenzymes) for a reaction.
Used to properly distribute objective coefficients.
"""
function count_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String})::Int
    # Extract enzyme IDs from gene association
    if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
        return 0
    end

    enzyme_count = 0
    for term in rxn.gene_association_dnf
        # Each term represents an enzyme or enzyme complex
        enzyme_count += 1
    end
    return max(1, enzyme_count)  # At least 1 to avoid division by zero
end

"""
Assign ordered or random mechanism to each eligible reaction.
"""
function assign_reaction_mechanisms(
    model::CM.Model, ordered_fraction::Float64,
    max_substrates::Int, max_products::Int, rng::AbstractRNG
)
    mechanisms = Dict{String,Symbol}()

    eligible_reactions = String[]
    for (rid, rxn) in model.reactions
        # Check if reaction has gene association
        if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
            continue
        end

        # Check substrate/product limits
        n_substrates = count(coeff < 0 for (_, coeff) in rxn.stoichiometry)
        n_products = count(coeff > 0 for (_, coeff) in rxn.stoichiometry)

        if n_substrates <= max_substrates && n_products <= max_products
            push!(eligible_reactions, rid)
        end
    end

    # Randomly assign mechanisms
    n_ordered = round(Int, length(eligible_reactions) * ordered_fraction)
    shuffle!(rng, eligible_reactions)

    for (i, rid) in enumerate(eligible_reactions)
        mechanisms[rid] = i <= n_ordered ? :ordered : :random
    end

    return mechanisms
end