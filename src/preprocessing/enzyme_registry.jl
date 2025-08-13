"""
Enzyme registry functionality for elementary step splitting.

This module handles:
- Enzyme discovery from gene associations
- Enzyme registry management
- Reaction-enzyme mapping
"""

using Random
import AbstractFBCModels.CanonicalModel as CM
import AbstractFBCModels

export extract_reaction_enzymes, build_enzyme_registry

"""
Build a registry of all enzymes from gene associations.
Works with any AbstractFBCModel by extracting gene associations via the interface.
"""
function build_enzyme_registry(model::AbstractFBCModels.AbstractFBCModel)
    enzyme_registry = Dict{String,String}()
    enzyme_id_map = Dict{String,String}()  # Maps full enzyme ID to short ID
    enzyme_counter = 0

    # Get reactions via AbstractFBCModel interface
    reaction_ids = AbstractFBCModels.reactions(model)

    for rid in reaction_ids
        # Get gene associations via the interface
        gene_associations = AbstractFBCModels.reaction_gene_association_dnf(model, rid)

        if !isnothing(gene_associations) && !isempty(gene_associations)
            for gene_group in gene_associations
                enzyme_counter += 1
                if length(gene_group) == 1
                    # Single gene = single enzyme
                    enzyme_id = "ENZ_$(gene_group[1])"
                    short_id = "E$enzyme_counter"
                    enzyme_registry[enzyme_id] = gene_group[1]
                    enzyme_id_map[enzyme_id] = short_id
                else
                    # Multiple genes = enzyme intermediate
                    intermediate_name = join(sort(gene_group), "_")
                    enzyme_id = "ENZ_$intermediate_name"
                    short_id = "E$enzyme_counter"
                    enzyme_registry[enzyme_id] = join(sort(gene_group), " & ")
                    enzyme_id_map[enzyme_id] = short_id
                end
            end
        end
    end

    return enzyme_registry, enzyme_id_map
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
Extract enzyme IDs for a reaction from AbstractFBCModel based on gene associations.
"""
function extract_reaction_enzymes(model::AbstractFBCModels.AbstractFBCModel, reaction_id::String, enzyme_registry::Dict{String,String})
    enzyme_ids = String[]

    # Get gene associations via the interface
    gene_associations = AbstractFBCModels.reaction_gene_association_dnf(model, reaction_id)

    if !isnothing(gene_associations) && !isempty(gene_associations)
        for gene_group in gene_associations
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
Count the number of enzyme variants (isoenzymes) for a reaction from AbstractFBCModel.
Used to properly distribute objective coefficients.
"""
function count_reaction_enzymes(model::AbstractFBCModels.AbstractFBCModel, reaction_id::String, enzyme_registry::Dict{String,String})::Int
    # Get gene associations via the interface
    gene_associations = AbstractFBCModels.reaction_gene_association_dnf(model, reaction_id)

    if isnothing(gene_associations) || isempty(gene_associations)
        return 0
    end

    enzyme_count = 0
    for term in gene_associations
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