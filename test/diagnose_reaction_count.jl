"""
Diagnostic script to understand the MATLAB vs Julia reaction count discrepancy.

MATLAB result: 2335 reactions
Julia result: 1331 reactions

This script analyzes what might be different.
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS

println("="^80)
println("Reaction Count Diagnosis")
println("="^80)

# Load model
MODEL_PATH = "test/iIS312_Epimastigote.xml"
model = A.load(MODEL_PATH) |> CM.Model

println("\nInitial model:")
println("  Reactions: $(length(model.reactions))")
println("  Metabolites: $(length(model.metabolites))")

# Step by step
model = COCOA.normalize_bounds(model)
println("\nAfter normalize_bounds:")
println("  Reactions: $(length(model.reactions))")

model = COCOA.remove_orphans(model)
println("\nAfter remove_orphans:")
println("  Reactions: $(length(model.reactions))")

model, _ = COCOA.remove_blocked_reactions(model, optimizer=HiGHS.Optimizer)
println("\nAfter remove_blocked_reactions:")
println("  Reactions: $(length(model.reactions))")

model_elem = COCOA.split_into_elementary_steps(model, random=0.0)
println("\nAfter split_into_elementary_steps:")
println("  Reactions: $(length(model_elem.reactions))")
println("  Metabolites: $(length(model_elem.metabolites))")

# Analyze elementary reactions
println("\nElementary reaction breakdown:")
enzyme_rxns = filter(r -> occursin("_enz", first(r)), model_elem.reactions)
println("  Enzyme reactions (contain '_enz'): $(length(enzyme_rxns))")

subst_rxns = filter(r -> occursin("_subst", first(r)), model_elem.reactions)
println("  Substrate binding (_subst): $(length(subst_rxns))")

product_complex = filter(r -> occursin("_product_complex", first(r)), model_elem.reactions)
println("  Product complex formation: $(length(product_complex))")

product_rxns = filter(r -> occursin("_product_", first(r)) && !occursin("_product_complex", first(r)), model_elem.reactions)
println("  Product release: $(length(product_rxns))")

no_enz = filter(r -> !occursin("_enz", first(r)), model_elem.reactions)
println("  Non-enzyme reactions: $(length(no_enz))")

model_irrev = COCOA.split_into_irreversible(model_elem)
println("\nAfter split_into_irreversible:")
println("  Reactions: $(length(model_irrev.reactions))")
println("  Metabolites: $(length(model_irrev.metabolites))")

# Analyze irreversible splits
fwd = filter(r -> endswith(first(r), "_f"), model_irrev.reactions)
bwd = filter(r -> endswith(first(r), "_b"), model_irrev.reactions)
rev = filter(r -> endswith(first(r), "_r"), model_irrev.reactions)
neither = filter(r -> !endswith(first(r), "_f") && !endswith(first(r), "_b") && !endswith(first(r), "_r"), model_irrev.reactions)

println("\nIrreversible reaction breakdown:")
println("  Forward (_f): $(length(fwd))")
println("  Backward (_b): $(length(bwd))")
println("  Reversed (_r): $(length(rev))")
println("  Unsplit: $(length(neither))")
println("  Total: $(length(fwd) + length(bwd) + length(rev) + length(neither))")

println("\n" * "="^80)
println("COMPARISON WITH MATLAB")
println("="^80)
println("MATLAB final count: 2335 reactions")
println("Julia final count:  $(length(model_irrev.reactions)) reactions")
println("Difference:         $(2335 - length(model_irrev.reactions)) reactions")
println("\nPossible explanations:")
println("1. MATLAB might not remove as many blocked reactions")
println("2. MATLAB might create more elementary steps per enzyme")
println("3. MATLAB might use different irreversible splitting rules")
println("4. MATLAB might include duplicate reactions in the count")
println("5. MATLAB might include exchange/transport reactions differently")

# Sample some reaction names to compare with MATLAB format
println("\n" * "="^80)
println("Sample Julia reaction names (first 20 elementary):")
println("="^80)
count = 0
for (rid, _) in model_irrev.reactions
    if occursin("_enz", rid)
        println(rid)
        count += 1
        count >= 20 && break
    end
end

println("\n" * "="^80)
println("Sample Julia reaction names (first 20 transport/exchange):")
println("="^80)
count = 0
for (rid, _) in model_irrev.reactions
    if !occursin("_enz", rid)
        println(rid)
        count += 1
        count >= 20 && break
    end
end
