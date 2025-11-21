"""
Diagnostic script to analyze elementary splitting behavior with ordered mechanism.

Identifies why models become large after ordered splitting.
"""

using COBREXA
using COCOA
using Statistics
import AbstractFBCModels.CanonicalModel as CM
import SBMLFBCModels

function analyze_splitting_impact(model_path::String)
    println("="^80)
    println("Elementary Splitting Diagnostic Report (Ordered Mechanism)")
    println("="^80)
    println("Model: $(basename(model_path))")
    println()

    # Load model
    println("Loading model...")
    sbml_model = load_model(model_path)
    model = convert(CM.Model, sbml_model)

    # Basic statistics
    n_reactions = length(model.reactions)
    n_metabolites = length(model.metabolites)
    n_genes = length(model.genes)

    println("Original Model Statistics:")
    println("  Reactions:   $n_reactions")
    println("  Metabolites: $n_metabolites")
    println("  Genes:       $n_genes")
    println()

    # Analyze reactions for ordered splitting
    eligible_reactions = String[]
    enzyme_usage = Dict{String,Int}()

    for (rid, rxn) in model.reactions
        # Check GPR
        has_gpr = !isnothing(rxn.gene_association_dnf) &&
                  !isempty(rxn.gene_association_dnf) &&
                  !all(isempty(g) for g in rxn.gene_association_dnf)

        # Check EC code
        ec_code = COCOA.extract_ec_number(rxn)
        has_ec = !isnothing(ec_code)

        # Count substrates and products
        n_substrates = count(coeff -> coeff < 0, values(rxn.stoichiometry))
        n_products = count(coeff -> coeff > 0, values(rxn.stoichiometry))

        # Check if eligible (≤4 substrates, ≤4 products, has GPR or EC)
        is_eligible = (has_gpr || has_ec) && n_substrates <= 4 && n_products <= 4

        if is_eligible
            push!(eligible_reactions, rid)

            # Count enzyme variants
            if has_gpr
                for gene_group in rxn.gene_association_dnf
                    enzyme_name = length(gene_group) == 1 ? gene_group[1] : join(sort(gene_group), " & ")
                    enzyme_usage[enzyme_name] = get(enzyme_usage, enzyme_name, 0) + 1
                end
            elseif has_ec
                enzyme_usage[ec_code] = get(enzyme_usage, ec_code, 0) + 1
            end
        end
    end

    n_enzymes = length(enzyme_usage)
    println("Splitting Statistics:")
    println("  Eligible reactions: $(length(eligible_reactions)) / $n_reactions")
    println("  Unique enzymes:     $n_enzymes")
    println()

    # Calculate expected elementary reactions (ordered mechanism)
    total_elementary = 0
    reaction_breakdown = Tuple{String, Int, Int, Int, Int}[]

    for (rid, rxn) in model.reactions
        has_gpr = !isnothing(rxn.gene_association_dnf) &&
                  !isempty(rxn.gene_association_dnf) &&
                  !all(isempty(g) for g in rxn.gene_association_dnf)
        ec_code = COCOA.extract_ec_number(rxn)
        has_ec = !isnothing(ec_code)

        n_substrates = count(coeff -> coeff < 0, values(rxn.stoichiometry))
        n_products = count(coeff -> coeff > 0, values(rxn.stoichiometry))

        is_eligible = (has_gpr || has_ec) && n_substrates <= 4 && n_products <= 4

        if is_eligible
            # Count enzyme variants
            n_enzyme_variants = has_gpr ? length(rxn.gene_association_dnf) : 1

            # Ordered mechanism: n_substrates + 1 (catalysis) + n_products per enzyme
            rxns_per_enzyme = n_substrates + 1 + n_products
            rxns_total = rxns_per_enzyme * n_enzyme_variants

            total_elementary += rxns_total
            push!(reaction_breakdown, (rid, n_substrates, n_products, n_enzyme_variants, rxns_total))
        else
            # Not split
            total_elementary += 1
        end
    end

    println("Expected Model Size (Ordered Mechanism):")
    println("  Total elementary reactions: $total_elementary")
    println("  Expansion factor:           $(round(total_elementary/n_reactions, digits=2))x")
    println("  Plus enzyme metabolites:    $n_enzymes")
    println("  Plus intermediates:         ~$(length(eligible_reactions) * 2) (estimate)")
    println()

    # Show top contributors
    sort!(reaction_breakdown, by=x->x[5], rev=true)
    println("Top 20 Reactions Creating Most Elementary Steps:")
    println("  #  Reaction ID                      S  P  Enzymes  → Elementary")
    println("  " * "-"^76)
    for (i, (rid, n_s, n_p, n_enz, n_rxns)) in enumerate(reaction_breakdown[1:min(20, length(reaction_breakdown))])
        println("  $(lpad(i, 2)). $(rpad(rid, 30)) $(lpad(n_s, 2)) $(lpad(n_p, 2)) $(lpad(n_enz, 7))  → $(lpad(n_rxns, 10))")
    end
    println()

    # Enzyme usage analysis
    enzyme_rxn_counts = sort(collect(enzyme_usage), by=x->x[2], rev=true)
    println("Top 10 Most Used Enzymes:")
    println("  #  Enzyme/Gene                      Reactions")
    println("  " * "-"^50)
    for (i, (enz, count)) in enumerate(enzyme_rxn_counts[1:min(10, length(enzyme_rxn_counts))])
        enz_display = length(enz) > 30 ? enz[1:27] * "..." : enz
        println("  $(lpad(i, 2)). $(rpad(enz_display, 30)) $(lpad(count, 9))")
    end
    println()

    # Check for reactions with many enzyme variants (OR relationships)
    multi_enzyme_reactions = filter(x -> x[4] > 1, reaction_breakdown)
    if !isempty(multi_enzyme_reactions)
        sort!(multi_enzyme_reactions, by=x->x[4], rev=true)
        println("Reactions with Multiple Enzyme Variants (OR in GPR):")
        println("  #  Reaction ID                      Enzymes  Elementary")
        println("  " * "-"^58)
        for (i, (rid, _, _, n_enz, n_rxns)) in enumerate(multi_enzyme_reactions[1:min(10, length(multi_enzyme_reactions))])
            println("  $(lpad(i, 2)). $(rpad(rid, 30)) $(lpad(n_enz, 7))  $(lpad(n_rxns, 10))")
        end
        println()

        total_multi = sum(x[5] for x in multi_enzyme_reactions)
        println("  Impact: $(length(multi_enzyme_reactions)) reactions with multiple enzymes")
        println("          contribute $total_multi / $total_elementary elementary reactions")
        println("          ($(round(100*total_multi/total_elementary, digits=1))% of total)")
        println()
    end

    println("="^80)

    return (
        original_reactions=n_reactions,
        eligible_reactions=length(eligible_reactions),
        expected_elementary=total_elementary,
        n_enzymes=n_enzymes,
        expansion_factor=total_elementary/n_reactions
    )
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia diagnose_splitting.jl <model_path>")
        exit(1)
    end

    model_path = ARGS[1]
    analyze_splitting_impact(model_path)
end
