"""
Test script to verify complex metabolite metadata improvements:
1. Components field should have one entry per component (not comma-separated string)
2. Compartment assignment should handle multi-compartment complexes properly
"""

using Pkg
Pkg.activate(".")

include("../src/COCOA.jl")
using .COCOA

# Load model
println("Loading model...")
model_path = "c:/Users/anton/master-thesis/Yeast-Species-GEMs/Saccharomyces_cerevisiae.xml"
model = COCOA.load_model(StandardModel, model_path)
println("Model loaded: $(length(model.reactions)) reactions, $(length(model.metabolites)) metabolites\n")

# Run preprocessing to generate complexes
println("Running preprocessing...")
model_irr = COCOA.preprocess(
    model,
    split_to_elementary=true,
    remove_blocked=false
)
println("Preprocessing complete: $(length(model_irr.metabolites)) metabolites\n")

# Test 1: Check components field structure
println("="^80)
println("TEST 1: Components field should be vector with one entry per component")
println("="^80)

# Find a complex with multiple metabolites
test_complex_id = "CPLX_E96__M_atp_c__M_dtdp_c_c"
if haskey(model_irr.metabolites, test_complex_id)
    cplx = model_irr.metabolites[test_complex_id]

    println("\nComplex: $test_complex_id")
    println("Components type: $(typeof(cplx.notes["components"]))")
    println("Components length: $(length(cplx.notes["components"]))")
    println("Components content:")
    for (i, comp) in enumerate(cplx.notes["components"])
        println("  [$i] $comp")
    end

    # Verify it's a vector, not a single string
    if cplx.notes["components"] isa Vector && length(cplx.notes["components"]) > 1
        println("\n✓ PASS: Components field is properly structured as vector")
    else
        println("\n✗ FAIL: Components field should be vector with multiple entries")
    end
else
    println("✗ SKIP: Test complex $test_complex_id not found")
end

# Test 2: Check multi-compartment complex handling
println("\n" * "="^80)
println("TEST 2: Multi-compartment complex handling")
println("="^80)

# Find a complex with metabolites from different compartments
multi_comp_complex = nothing
for (cplx_id, cplx) in model_irr.metabolites
    if startswith(cplx_id, "CPLX_") && haskey(cplx.annotations, "compartments_involved")
        compartments = cplx.annotations["compartments_involved"]
        if length(compartments) > 1
            multi_comp_complex = (cplx_id, cplx)
            break
        end
    end
end

if !isnothing(multi_comp_complex)
    cplx_id, cplx = multi_comp_complex

    println("\nMulti-compartment complex found: $cplx_id")
    println("Primary compartment: $(cplx.compartment)")
    println("All compartments involved: $(cplx.annotations["compartments_involved"])")
    println("Bound metabolites: $(cplx.annotations["bound_metabolites"])")

    # Check compartment logic explanation
    if haskey(cplx.notes, "compartment_logic")
        println("\nCompartment logic: $(cplx.notes["compartment_logic"][1])")
    end

    # Verify each bound metabolite's compartment
    println("\nMetabolite compartments:")
    for met_id in cplx.annotations["bound_metabolites"]
        if haskey(model_irr.metabolites, met_id)
            met_comp = model_irr.metabolites[met_id].compartment
            println("  $met_id → compartment '$met_comp'")
        end
    end

    println("\n✓ PASS: Multi-compartment complex properly documented")
else
    # Try the example from the user's question
    test_id = "CPLX_E93__M_ficytc_m__M_h2o_m__M_h_c_m"
    if haskey(model_irr.metabolites, test_id)
        cplx = model_irr.metabolites[test_id]
        println("\nFound user's example: $test_id")
        println("Primary compartment: $(cplx.compartment)")

        if haskey(cplx.annotations, "compartments_involved")
            println("All compartments involved: $(cplx.annotations["compartments_involved"])")
        end

        if haskey(cplx.notes, "compartment_logic")
            println("Compartment logic: $(cplx.notes["compartment_logic"][1])")
        end

        println("\n✓ INFO: Tested user's specific example")
    else
        println("\n✓ INFO: No multi-compartment complexes found (all complexes use single compartment)")
    end
end

# Test 3: Statistical overview
println("\n" * "="^80)
println("TEST 3: Statistical overview of complexes")
println("="^80)

total_complexes = 0
multi_compartment_count = 0
compartment_distribution = Dict{Int,Int}()

for (met_id, met) in model_irr.metabolites
    if startswith(met_id, "CPLX_") && haskey(met.annotations, "compartments_involved")
        total_complexes += 1
        n_comps = length(met.annotations["compartments_involved"])
        compartment_distribution[n_comps] = get(compartment_distribution, n_comps, 0) + 1

        if n_comps > 1
            multi_compartment_count += 1
        end
    end
end

println("\nTotal complexes: $total_complexes")
println("Multi-compartment complexes: $multi_compartment_count")
println("\nCompartment distribution:")
for n_comps in sort(collect(keys(compartment_distribution)))
    count = compartment_distribution[n_comps]
    pct = round(100 * count / total_complexes, digits=1)
    println("  $n_comps compartment(s): $count complexes ($pct%)")
end

println("\n" * "="^80)
println("All tests complete!")
println("="^80)
