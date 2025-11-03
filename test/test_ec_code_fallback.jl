"""
Test EC code fallback for reactions without GPR rules.

This tests that reactions with EC codes but no GPR rules can still be split
into elementary steps, matching MATLAB's behavior.
"""

using Test
using AbstractFBCModels
using AbstractFBCModels.CanonicalModel: Model, Metabolite, Reaction

# Add COCOA.jl to the load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using COCOA

@testset "EC Code Fallback" begin
    # Create a simple model with a reaction that has EC code but no GPR
    model = Model()

    # Add metabolites
    model.metabolites["M_A"] = Metabolite(name="A", compartment="c")
    model.metabolites["M_B"] = Metabolite(name="B", compartment="c")
    model.metabolites["M_C"] = Metabolite(name="C", compartment="c")

    # Reaction 1: Has GPR rule (should be split)
    model.reactions["R1"] = Reaction(
        name="Reaction with GPR",
        stoichiometry=Dict("M_A" => -1.0, "M_B" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_association_dnf=[["G_gene1"]],
        annotations=Dict{String,Vector{String}}()
    )

    # Reaction 2: Has EC code in annotations but NO GPR (should be split using EC code)
    model.reactions["R2"] = Reaction(
        name="Reaction with EC code in annotations",
        stoichiometry=Dict("M_B" => -1.0, "M_C" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_association_dnf=nothing,  # No GPR!
        annotations=Dict("ec-code" => ["1.7.1.3"]),  # Has EC code in annotations
        notes=Dict{String,Vector{String}}()
    )

    # Reaction 3: Has EC code in notes field but NO GPR (should be split using EC code from notes)
    model.reactions["R3"] = Reaction(
        name="Reaction with EC code in notes",
        stoichiometry=Dict("M_C" => -1.0, "M_A" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_association_dnf=nothing,  # No GPR!
        annotations=Dict{String,Vector{String}}(),
        notes=Dict("" => ["<notes>\n  <body xmlns=\"http://www.w3.org/1999/xhtml\">\n    <p>EC Number: 2.6.1.79</p>\n  </body>\n</notes>"])
    )

    # Reaction 4: Has invalid EC code placeholder (should NOT be split)
    model.reactions["R4"] = Reaction(
        name="Reaction with invalid EC code",
        stoichiometry=Dict("M_A" => -1.0, "M_B" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_association_dnf=nothing,
        annotations=Dict{String,Vector{String}}(),
        notes=Dict("" => ["<notes>\n  <body xmlns=\"http://www.w3.org/1999/xhtml\">\n    <p>EC Number: --</p>\n  </body>\n</notes>"])
    )

    # Reaction 5: No GPR and no EC code (should NOT be split)
    model.reactions["R5"] = Reaction(
        name="Reaction without GPR or EC",
        stoichiometry=Dict("M_B" => -1.0, "M_C" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0,
        gene_association_dnf=nothing,
        annotations=Dict{String,Vector{String}}(),
        notes=Dict{String,Vector{String}}()
    )

    # Split into elementary steps
    elem_model = COCOA.split_into_elementary_steps(model)

    # Test 1: R1 should be split (has GPR)
    @test !haskey(elem_model.reactions, "R1")  # Original should be removed
    @test haskey(elem_model.reactions, "R1_S1")  # Elementary steps should exist
    @test haskey(elem_model.reactions, "R1_CAT")
    @test haskey(elem_model.reactions, "R1_P1")

    # Test 2: R2 should be split (has EC code in annotations)
    @test !haskey(elem_model.reactions, "R2")  # Original should be removed
    @test haskey(elem_model.reactions, "R2_S1")  # Elementary steps should exist
    @test haskey(elem_model.reactions, "R2_CAT")
    @test haskey(elem_model.reactions, "R2_P1")

    # Test 3: R3 should be split (has EC code in notes field)
    @test !haskey(elem_model.reactions, "R3")  # Original should be removed
    @test haskey(elem_model.reactions, "R3_S1")  # Elementary steps should exist
    @test haskey(elem_model.reactions, "R3_CAT")
    @test haskey(elem_model.reactions, "R3_P1")

    # Test 4: R4 should NOT be split (invalid EC code placeholder)
    @test haskey(elem_model.reactions, "R4")  # Original should remain
    @test !haskey(elem_model.reactions, "R4_S1")  # No elementary steps

    # Test 5: R5 should NOT be split (no GPR or EC code)
    @test haskey(elem_model.reactions, "R5")  # Original should remain
    @test !haskey(elem_model.reactions, "R5_S1")  # No elementary steps

    # Test 6: Check that enzyme metabolites were created
    # Should have 3 enzymes: one for gene1, one for EC 1.7.1.3, one for EC 2.6.1.79
    enzyme_count = count(k -> startswith(k, "E_"), keys(elem_model.metabolites))
    @test enzyme_count == 3

    # Test 7: Verify enzyme names include both gene and EC codes
    enzyme_names = [m.name for (k, m) in elem_model.metabolites if startswith(k, "E_")]
    @test any(contains(name, "G_gene1") for name in enzyme_names)
    @test any(contains(name, "1.7.1.3") for name in enzyme_names)
    @test any(contains(name, "2.6.1.79") for name in enzyme_names)

    println("✓ All EC code fallback tests passed!")
    println("  - Reactions with GPR: split correctly")
    println("  - Reactions with EC code in annotations: split using EC code fallback")
    println("  - Reactions with EC code in notes field: split by parsing notes")
    println("  - Reactions with invalid EC placeholders (--): kept as-is")
    println("  - Reactions with neither: kept as-is")
    println("  - Enzyme metabolites created for genes and EC codes from both sources")
end
