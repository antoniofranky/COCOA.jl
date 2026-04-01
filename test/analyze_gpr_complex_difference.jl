"""
Analyze whether the 73 extra complexes in Julia come from GPR parsing differences.

This script identifies which enzymes create the extra complexes and traces them
back to specific reactions to verify if they're from the GPR parsing bug.
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS

# Load and process model
println("="^80)
println("GPR PARSING vs COMPLEX METABOLITE ANALYSIS")
println("="^80)

model = A.load("C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\iIS312_Epimastigote.xml")
model = convert(CM.Model, model)
model = COCOA.normalize_bounds(model)
model = COCOA.remove_orphans(model)
model, _ = COCOA.remove_blocked_reactions(model, optimizer=HiGHS.Optimizer)
model = COCOA.remove_orphans(model)

# Split into elementary
model_elem = COCOA.split_into_elementary(model, random=0.0)
model_irr = COCOA.split_into_irreversible(model_elem)

println("\n📊 Metabolite Counts:")
println("  Total metabolites: $(length(model_irr.metabolites))")

# Categorize metabolites
enzyme_mets = String[]
complex_mets = String[]
regular_mets = String[]

for (mid, met) in model_irr.metabolites
    if match(r"^E\d+$", mid) !== nothing
        push!(enzyme_mets, mid)
    elseif startswith(mid, "M_CPLX_E")
        push!(complex_mets, mid)
    else
        push!(regular_mets, mid)
    end
end

println("  Enzyme metabolites:  $(length(enzyme_mets))")
println("  Complex metabolites: $(length(complex_mets))")
println("  Regular metabolites: $(length(regular_mets))")

# Extract enzyme numbers from complexes
function extract_enzyme_from_complex(complex_id::String)
    m = match(r"M_CPLX_E(\d+)__", complex_id)
    return m !== nothing ? parse(Int, m.captures[1]) : nothing
end

# Group complexes by enzyme
complexes_by_enzyme = Dict{Int,Vector{String}}()
for cplx_id in complex_mets
    enz_num = extract_enzyme_from_complex(cplx_id)
    if !isnothing(enz_num)
        if !haskey(complexes_by_enzyme, enz_num)
            complexes_by_enzyme[enz_num] = String[]
        end
        push!(complexes_by_enzyme[enz_num], cplx_id)
    end
end

println("\n📈 Complexes per Enzyme:")
for enz_num in sort(collect(keys(complexes_by_enzyme)))
    n_complexes = length(complexes_by_enzyme[enz_num])
    enz_id = "E$enz_num"
    met = model_irr.metabolites[enz_id]
    println("  E$(enz_num) ($(met.name)): $n_complexes complexes")
end

# From MATLAB comparison, we know:
# - MATLAB: 782 complexes, 115 enzymes create complexes
# - Julia: 855 complexes, 118 enzymes create complexes
# - Julia-only enzymes: E_8, E_47, E_49
matlab_complex_count = 782
julia_complex_count = length(complex_mets)
difference = julia_complex_count - matlab_complex_count

println("\n" * "="^80)
println("COMPLEX COUNT COMPARISON")
println("="^80)
println("  MATLAB complexes: $matlab_complex_count")
println("  Julia complexes:  $julia_complex_count")
println("  Difference:       $difference extra in Julia")

# Identify the Julia-only enzymes (from previous comparison)
# These are suspected to come from GPR parsing differences
suspected_enzymes = [8, 47, 49]

println("\n" * "="^80)
println("SUSPECTED GPR PARSING DIFFERENCE ENZYMES")
println("="^80)
println("Julia creates complexes for E_8, E_47, E_49 that MATLAB doesn't")
println()

global total_complexes_from_suspected = 0
for enz_num in suspected_enzymes
    if haskey(complexes_by_enzyme, enz_num)
        n = length(complexes_by_enzyme[enz_num])
        global total_complexes_from_suspected += n
        enz_id = "E$enz_num"
        met = model_irr.metabolites[enz_id]
        println("  E$(enz_num) ($(met.name)): $n complexes")
    end
end

println("\n  Total complexes from suspected enzymes: $total_complexes_from_suspected")
println("  Difference to explain: $difference")
println("  Accounted for: $(round(100 * total_complexes_from_suspected / difference, digits=1))%")

# Now trace these enzymes back to their reactions
println("\n" * "="^80)
println("REACTION TRACING: What reactions use these enzymes?")
println("="^80)

# Rebuild enzyme list to find which reactions map to which enzymes
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

# Build enzyme list (matching splitting logic)
enzyme_list = String[]
enzyme_to_reactions = Dict{String,Vector{String}}()

for (rid, gpr) in reactions_with_fallback
    rxn = model.reactions[rid]
    substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
    product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

    # Only process enzymes from reactions that will be split
    if length(substrate_ids) <= 4 && length(product_ids) <= 4
        for gene_group in gpr
            enzyme_name = length(gene_group) == 1 ? gene_group[1] : join(sort(gene_group), " & ")
            if !(enzyme_name in enzyme_list)
                push!(enzyme_list, enzyme_name)
                enzyme_to_reactions[enzyme_name] = String[]
            end
            push!(enzyme_to_reactions[enzyme_name], rid)
        end
    end
end

println("\nBuilt enzyme list with $(length(enzyme_list)) entries")

for enz_num in suspected_enzymes
    println("\n" * "-"^80)
    println("Enzyme E$enz_num:")

    if enz_num <= length(enzyme_list)
        enzyme_name = enzyme_list[enz_num]
        println("  Enzyme name: $enzyme_name")

        if haskey(enzyme_to_reactions, enzyme_name)
            rxns = enzyme_to_reactions[enzyme_name]
            println("  Used by reactions: $(join(rxns, ", "))")

            # Show GPR details for these reactions
            for rid in rxns
                rxn = model.reactions[rid]
                println("\n  Reaction: $rid")
                println("    Name: $(rxn.name)")

                # Get original GPR
                original_gpr = rxn.gene_association_dnf
                if !isnothing(original_gpr) && !isempty(original_gpr)
                    println("    GPR (DNF form):")
                    for (i, gene_group) in enumerate(original_gpr)
                        if isempty(gene_group)
                            println("      Group $i: [empty]")
                        else
                            isozyme = length(gene_group) > 1 ? " (isozyme: " * join(gene_group, " & ") * ")" : ""
                            println("      Group $i: $(gene_group[1])$isozyme")
                        end
                    end

                    println("    Total enzyme combinations: $(length(original_gpr))")

                    # Count how many complexes this reaction creates
                    substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
                    product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

                    # Ordered mechanism creates complexes at each binding step
                    # For n substrates and m products:
                    # - n substrate binding steps create n intermediates
                    # - 1 catalytic step (substrate→product complex transformation)
                    # - m product release steps create m-1 intermediates
                    # Total: n + m intermediate types (some may be reused)

                    n_subs = length(substrate_ids)
                    n_prods = length(product_ids)

                    # Estimate complexes per enzyme for ordered mechanism
                    # Conservative estimate: ~(n_subs + n_prods) complexes per enzyme
                    estimated_complexes = n_subs + n_prods

                    println("    Substrates: $n_subs, Products: $n_prods")
                    println("    Est. complexes per enzyme: ~$estimated_complexes")
                    println("    Total enzymes from GPR: $(length(original_gpr))")
                    println("    Est. total complexes: ~$(length(original_gpr) * estimated_complexes)")
                end
            end
        end
    end
end

# Calculate if the numbers add up
println("\n" * "="^80)
println("VERIFICATION: Do the numbers match?")
println("="^80)

println("""
Expected scenario if GPR parsing is the cause:
- MATLAB's naive GPR parser creates fewer enzyme combinations
- Julia's correct GPR parser creates more enzyme combinations
- More enzymes → more complexes (each enzyme creates its own set)

From the analysis:
1. Julia has 3 extra enzymes (E8, E47, E49) compared to MATLAB
2. These 3 enzymes create $total_complexes_from_suspected complexes
3. The total difference is $difference complexes
4. Accounted for: $(round(100 * total_complexes_from_suspected / difference, digits=1))%

""")

if total_complexes_from_suspected >= difference - 5
    println("✅ CONFIRMED: The extra complexes are almost entirely explained by")
    println("   the GPR parsing differences. Julia correctly parses complex GPR")
    println("   rules into more enzyme combinations, and each enzyme creates its")
    println("   own set of intermediate complexes.")
else
    remaining = difference - total_complexes_from_suspected
    println("⚠️  PARTIAL: Suspected enzymes account for $total_complexes_from_suspected complexes,")
    println("   but $remaining complexes remain unexplained. There may be additional")
    println("   differences in mechanism or other enzyme-specific variations.")

    # Find which other enzymes might contribute
    println("\n  Other enzymes with many complexes:")
    sorted_enzymes = sort(collect(keys(complexes_by_enzyme)), by=x -> length(complexes_by_enzyme[x]), rev=true)
    for enz_num in sorted_enzymes[1:min(10, length(sorted_enzymes))]
        if !(enz_num in suspected_enzymes)
            n = length(complexes_by_enzyme[enz_num])
            if n > 10
                enz_id = "E$enz_num"
                met = model_irr.metabolites[enz_id]
                println("    E$(enz_num) ($(met.name)): $n complexes")
            end
        end
    end
end

println("\n" * "="^80)
