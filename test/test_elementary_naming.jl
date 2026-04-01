"""
Quick test to verify the elementary splitting changes work correctly.
Tests the new M_CPLX_ naming and elementary_mechanism annotations.
"""

using Pkg
Pkg.activate(".")

include("../src/COCOA.jl")
using .COCOA
import AbstractFBCModels.CanonicalModel as CM

println("Testing elementary splitting with new naming conventions...")

# Create a simple test model
model = CM.Model()

# Add metabolites
model.metabolites["M_atp_c"] = CM.Metabolite(name="ATP", compartment="c")
model.metabolites["M_adp_c"] = CM.Metabolite(name="ADP", compartment="c")
model.metabolites["M_pi_c"] = CM.Metabolite(name="Phosphate", compartment="c")
model.metabolites["M_h2o_c"] = CM.Metabolite(name="Water", compartment="c")

# Add a simple reaction with GPR
model.reactions["R_ATPASE"] = CM.Reaction(
    name="ATP hydrolysis",
    stoichiometry=Dict(
        "M_atp_c" => -1.0,
        "M_h2o_c" => -1.0,
        "M_adp_c" => 1.0,
        "M_pi_c" => 1.0
    ),
    lower_bound=-1000.0,
    upper_bound=1000.0,
    gene_association_dnf=[["GENE1"]]
)

model.genes["GENE1"] = CM.Gene(name="Gene 1")

println("Original model: $(length(model.reactions)) reactions, $(length(model.metabolites)) metabolites")

# Split to elementary
model_elem = COCOA.split_into_elementary_steps(model)

println("Elementary model: $(length(model_elem.reactions)) reactions, $(length(model_elem.metabolites)) metabolites")

# Check metabolite categories
enzymes = [mid for mid in keys(model_elem.metabolites) if occursin(r"^E\d+$", mid)]
complexes = [mid for mid in keys(model_elem.metabolites) if startswith(mid, "M_CPLX_")]
original = [mid for mid in keys(model_elem.metabolites) if !occursin(r"^E\d+$", mid) && !startswith(mid, "M_CPLX_")]

println("\nMetabolite breakdown:")
println("  Enzymes (E<n>): $(length(enzymes))")
for enz in enzymes
    println("    - $enz")
end

println("  Complexes (M_CPLX_): $(length(complexes))")
for cplx in sort(complexes)
    println("    - $cplx")
end

println("  Original: $(length(original))")

# Check one complex has correct structure
if !isempty(complexes)
    sample_cplx = model_elem.metabolites[complexes[1]]
    println("\nSample complex metadata:")
    println("  ID: $(complexes[1])")
    println("  Components: $(sample_cplx.notes["components"])")
    println("  Compartment: $(sample_cplx.compartment)")
    if haskey(sample_cplx.annotations, "reaction_compartment")
        println("  Reaction compartment: $(sample_cplx.annotations["reaction_compartment"])")
    end
end

# Check reactions have elementary_mechanism annotation
println("\nChecking reaction annotations:")
for (rid, rxn) in model_elem.reactions
    if haskey(rxn.annotations, "elementary_mechanism")
        mech = rxn.annotations["elementary_mechanism"][1]
        step_type = get(rxn.annotations, "elementary_step_type", ["unknown"])[1]
        println("  $rid: $step_type ($mech)")
    end
end

# Verify all original metabolites present
original_ids = Set(keys(model.metabolites))
elem_original_ids = Set(original)

if original_ids == elem_original_ids
    println("\n✓ All original metabolites preserved!")
else
    missing = setdiff(original_ids, elem_original_ids)
    extra = setdiff(elem_original_ids, original_ids)
    println("\n✗ Metabolite mismatch:")
    if !isempty(missing)
        println("  Missing: $missing")
    end
    if !isempty(extra)
        println("  Extra: $extra")
    end
end

println("\n✓ Test complete!")
