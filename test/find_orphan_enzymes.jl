"""
Find orphan enzyme metabolites that are created but never used.
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS

# Load and process model
model = A.load("C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\iIS312_Epimastigote.xml")
model = convert(CM.Model, model)
model = COCOA.normalize_bounds(model)
model = COCOA.remove_orphans(model)
model, _ = COCOA.remove_blocked_reactions(model, optimizer=HiGHS.Optimizer)
model = COCOA.remove_orphans(model)

# Split into elementary
model_elem = COCOA.split_into_elementary(model, random=0.0)
model_irr = COCOA.split_into_irreversible(model_elem)

println("="^80)
println("ORPHAN ENZYME ANALYSIS")
println("="^80)

# Find which metabolites are actually used in reactions
active_metabolites = Set{String}()
for rxn in values(model_irr.reactions)
    union!(active_metabolites, keys(rxn.stoichiometry))
end

# Find enzyme metabolites
enzyme_mets = [mid for mid in keys(model_irr.metabolites) if match(r"^E\d+$", mid) !== nothing]
sort!(enzyme_mets, by=x -> parse(Int, replace(x, "E" => "")))

println("\nTotal enzymes created: $(length(enzyme_mets))")
println("Enzyme range: $(enzyme_mets[1]) to $(enzyme_mets[end])")

# Find orphan enzymes
orphan_enzymes = [mid for mid in enzyme_mets if !(mid in active_metabolites)]
active_enzymes = [mid for mid in enzyme_mets if mid in active_metabolites]

println("\nActive enzymes: $(length(active_enzymes))")
println("Orphan enzymes: $(length(orphan_enzymes))")

if length(orphan_enzymes) > 0
    println("\nOrphan enzyme IDs:")
    for enz_id in sort(orphan_enzymes, by=x -> parse(Int, replace(x, "E" => "")))
        enz_num = parse(Int, replace(enz_id, "E" => ""))
        met = model_irr.metabolites[enz_id]
        println("  $enz_id ($(met.name))")
    end

    println("\n" * "="^80)
    println("INVESTIGATION: Why were these enzymes created?")
    println("="^80)

    # These correspond to the enzyme list generated during splitting
    # Need to trace back to the original GPR rules

    # Re-run the enzyme list generation logic to see what created these
    reactions_with_fallback = Dict{String,Vector{Vector{String}}}()
    for (rid, rxn) in model.reactions
        gpr = rxn.gene_association_dnf
        has_empty_gpr = isnothing(gpr) || isempty(gpr) || all(isempty(g) for g in gpr)

        if has_empty_gpr
            ec_code = COCOA.extract_ec_number(rxn)
            if !isnothing(ec_code)
                reactions_with_fallback[rid] = [[ec_code]]
            end
        else
            reactions_with_fallback[rid] = gpr
        end
    end

    # Build enzyme list
    enzyme_list = String[]
    rxn_for_enzyme = Dict{Int,Vector{String}}()

    for (rid, gpr) in reactions_with_fallback
        for gene_group in gpr
            enzyme_name = length(gene_group) == 1 ? gene_group[1] : join(sort(gene_group), " & ")
            if !(enzyme_name in enzyme_list)
                push!(enzyme_list, enzyme_name)
            end

            enz_idx = findfirst(==(enzyme_name), enzyme_list)
            if !haskey(rxn_for_enzyme, enz_idx)
                rxn_for_enzyme[enz_idx] = String[]
            end
            push!(rxn_for_enzyme[enz_idx], rid)
        end
    end

    println("\nEnzyme list has $(length(enzyme_list)) entries")

    for enz_id in sort(orphan_enzymes, by=x -> parse(Int, replace(x, "E" => "")))
        enz_num = parse(Int, replace(enz_id, "E" => ""))

        println("\n" * "-"^80)
        println("Enzyme $enz_id (index $enz_num):")

        if enz_num <= length(enzyme_list)
            println("  Enzyme name: $(enzyme_list[enz_num])")

            if haskey(rxn_for_enzyme, enz_num)
                rxns = rxn_for_enzyme[enz_num]
                println("  Should be used by reactions: $(join(rxns, ", "))")

                # Check if these reactions were expanded
                for rid in rxns
                    rxn = model.reactions[rid]
                    substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
                    product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

                    println("\n  Reaction $rid:")
                    println("    Name: $(rxn.name)")
                    println("    Substrates: $(length(substrate_ids))")
                    println("    Products: $(length(product_ids))")

                    if length(substrate_ids) > 4 || length(product_ids) > 4
                        println("    ⚠️  NOT EXPANDED (too many substrates/products)")
                        println("    This is why enzyme $enz_id was created but not used!")
                    else
                        # Check if reaction exists in elementary model
                        rid_clean = replace(rid, r"^R_" => "")
                        elem_rxns = [r for r in keys(model_irr.reactions) if startswith(r, rid_clean)]
                        if isempty(elem_rxns)
                            println("    ⚠️  MISSING in elementary model!")
                        else
                            println("    ✓ Expanded into $(length(elem_rxns)) elementary reactions")
                        end
                    end
                end
            else
                println("  ERROR: No reactions found for this enzyme!")
            end
        else
            println("  ERROR: Enzyme index out of bounds!")
        end
    end
end

println("\n" * "="^80)
println("CONCLUSION")
println("="^80)
println("""
Orphan enzymes are created because:
1. The enzyme list is built from ALL GPR rules (including EC code fallbacks)
2. An enzyme metabolite (E1, E2, etc.) is created for EACH unique enzyme/gene combo
3. But reactions are only split if they have ≤4 substrates AND ≤4 products
4. If a reaction has >4 substrates or >4 products, it's kept as-is (not split)
5. The enzyme for that reaction is created but never used in any elementary step

FIX: Only create enzyme metabolites for reactions that will actually be split!
""")
