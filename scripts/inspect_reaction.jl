"""
Inspect specific reactions to understand their GPR structure.
"""

using COBREXA
import AbstractFBCModels.CanonicalModel as CM
import SBMLFBCModels

function inspect_reaction(model_path::String, reaction_ids::Vector{String})
    println("Loading model from: $model_path")
    println("File exists: $(isfile(model_path))")
    sbml_model = load_model(model_path)
    model = convert(CM.Model, sbml_model)

    for rid in reaction_ids
        if !haskey(model.reactions, rid)
            println("\n⚠️  Reaction $rid not found in model")
            continue
        end

        rxn = model.reactions[rid]

        println("\n" * "="^80)
        println("Reaction: $rid")
        println("="^80)
        println("Name: $(rxn.name)")
        println()

        # Stoichiometry
        println("Stoichiometry:")
        substrates = [(mid, coeff) for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        products = [(mid, coeff) for (mid, coeff) in rxn.stoichiometry if coeff > 0]

        for (mid, coeff) in substrates
            met_name = haskey(model.metabolites, mid) ? model.metabolites[mid].name : "Unknown"
            println("  $(abs(coeff)) $mid ($met_name)")
        end
        println("  →")
        for (mid, coeff) in products
            met_name = haskey(model.metabolites, mid) ? model.metabolites[mid].name : "Unknown"
            println("  $coeff $mid ($met_name)")
        end
        println()

        # GPR
        println("Gene Association (DNF):")
        if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
            println("  (none)")
        else
            n_variants = length(rxn.gene_association_dnf)
            println("  $n_variants enzyme variants (OR groups):")
            for (i, gene_group) in enumerate(rxn.gene_association_dnf[1:min(10, n_variants)])
                if length(gene_group) == 1
                    println("    $i. $(gene_group[1])")
                else
                    println("    $i. ($(join(gene_group, " AND ")))")
                end
            end
            if n_variants > 10
                println("    ... and $(n_variants - 10) more")
            end
        end
        println()

        # Annotations
        if !isempty(rxn.annotations)
            println("Annotations:")
            for (key, values) in rxn.annotations
                if !isempty(values)
                    println("  $key: $(join(values, ", "))")
                end
            end
            println()
        end

        # Calculate elementary reactions
        n_substrates = length(substrates)
        n_products = length(products)
        n_enzyme_variants = isnothing(rxn.gene_association_dnf) ? 0 : length(rxn.gene_association_dnf)

        if n_enzyme_variants > 0 && n_substrates <= 4 && n_products <= 4
            ordered_per_enzyme = n_substrates + 1 + n_products
            ordered_total = ordered_per_enzyme * n_enzyme_variants

            println("Elementary Reaction Count (Ordered):")
            println("  Steps per enzyme: $ordered_per_enzyme ($n_substrates binding + 1 catalysis + $n_products release)")
            println("  Enzyme variants:  $n_enzyme_variants")
            println("  Total:            $ordered_total elementary reactions")
        end
    end
    println("\n" * "="^80)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia inspect_reaction.jl <model_path> <reaction_id1> [reaction_id2] ...")
        exit(1)
    end

    model_path = ARGS[1]
    reaction_ids = ARGS[2:end]
    inspect_reaction(model_path, reaction_ids)
end
