"""
EnvZ-OmpR System Model Creation

This script recreates the EnvZ-OmpR system from the kinetic modules paper
(https://www.science.org/doi/10.1126/sciadv.ads7269) Figure 1A using COBREXA.jl.

The system contains:
- 9 species (A, B, C, D, E, F, G, H, I)
- 14 reactions (R1-R14) 
- 13 complexes as described in the paper

This represents the EnvZ-OmpR two-component regulatory system for kinetic analysis.
"""

import AbstractFBCModels.CanonicalModel as CM
using AbstractFBCModels
using COBREXA
using SparseArrays

"""
Create the EnvZ-OmpR system model as described in the kinetic modules paper.

Based on Figure 1A from "Kinetic modules are sources of concentration robustness 
in biochemical networks" (Science Advances, 2025).

Returns a CanonicalModel representing the EnvZ-OmpR system.
"""
function create_envz_ompr_model()
    # Create empty model
    model = CM.Model()

    # Add metabolites (species A through I - J is a complex, not a species)
    model.metabolites = Dict(
        "A" => CM.Metabolite(name="Species A"),
        "B" => CM.Metabolite(name="Species B"),
        "C" => CM.Metabolite(name="Species C"),
        "D" => CM.Metabolite(name="Species D"),
        "E" => CM.Metabolite(name="Species E"),
        "F" => CM.Metabolite(name="Species F"),
        "G" => CM.Metabolite(name="Species G"),
        "H" => CM.Metabolite(name="Species H"),
        "I" => CM.Metabolite(name="Species I")
    )

    # Add reactions based on Figure 1A
    # The figure shows 14 reactions connecting 13 complexes
    # Complexes: A, B, C, D, D+E, F, B+G, C+G, H, C+I, A+G, J, A+I
    # J is a complex formed from A+G, not a separate species

    # R1: A → B (forward direction of A ⇄ B)
    model.reactions["R1"] = CM.Reaction(
        name="Reaction 1: A to B",
        stoichiometry=Dict("A" => -1.0, "B" => 1.0),
        lower_bound=0.0,  # Forward reaction
        upper_bound=1000.0
    )

    # R2: B → A (reverse direction of A ⇄ B)
    model.reactions["R2"] = CM.Reaction(
        name="Reaction 2: B to A",
        stoichiometry=Dict("B" => -1.0, "A" => 1.0),
        lower_bound=0.0,  # Reverse reaction
        upper_bound=1000.0
    )

    # R3: B → C (forward direction of B ⇄ C)
    model.reactions["R3"] = CM.Reaction(
        name="Reaction 3: B to C",
        stoichiometry=Dict("B" => -1.0, "C" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R4: C → B (reverse direction of B ⇄ C)
    model.reactions["R4"] = CM.Reaction(
        name="Reaction 4: C to B",
        stoichiometry=Dict("C" => -1.0, "B" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R5: C → D (irreversible)
    model.reactions["R5"] = CM.Reaction(
        name="Reaction 5: C to D",
        stoichiometry=Dict("C" => -1.0, "D" => 1.0),
        lower_bound=0.0,  # Irreversible
        upper_bound=1000.0
    )

    # R6: D + E → F (forward direction of D+E ⇄ F)
    model.reactions["R6"] = CM.Reaction(
        name="Reaction 6: D+E to F",
        stoichiometry=Dict("D" => -1.0, "E" => -1.0, "F" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R7: F → D + E (reverse direction of D+E ⇄ F)
    model.reactions["R7"] = CM.Reaction(
        name="Reaction 7: F to D+E",
        stoichiometry=Dict("F" => -1.0, "D" => 1.0, "E" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R8: F → B + G (irreversible)
    model.reactions["R8"] = CM.Reaction(
        name="Reaction 8: F to B+G",
        stoichiometry=Dict("F" => -1.0, "B" => 1.0, "G" => 1.0),
        lower_bound=0.0,  # Irreversible
        upper_bound=1000.0
    )

    # R9: C + G → H (forward direction of C+G ⇄ H)
    model.reactions["R9"] = CM.Reaction(
        name="Reaction 9: C+G to H",
        stoichiometry=Dict("C" => -1.0, "G" => -1.0, "H" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R10: H → C + G (reverse direction of C+G ⇄ H)
    model.reactions["R10"] = CM.Reaction(
        name="Reaction 10: H to C+G",
        stoichiometry=Dict("H" => -1.0, "C" => 1.0, "G" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R11: H → C + I (irreversible - produces C and I separately, not as complex)
    model.reactions["R11"] = CM.Reaction(
        name="Reaction 11: H to C and I",
        stoichiometry=Dict("H" => -1.0, "C" => 1.0, "I" => 1.0),
        lower_bound=0.0,  # Irreversible
        upper_bound=1000.0
    )

    # R12: A + G → J (forward direction of A+G ⇄ J)
    # Note: J is represented as a complex in the stoichiometry, we model it implicitly
    # Since J is not a metabolite but a complex state, we can represent this as 
    # consumption of A and G with appropriate products in R14
    model.reactions["R12"] = CM.Reaction(
        name="Reaction 12: A+G to complex J",
        stoichiometry=Dict("A" => -1.0, "G" => -1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R13: J → A + G (reverse direction of A+G ⇄ J)
    # This represents the dissociation back from complex J
    model.reactions["R13"] = CM.Reaction(
        name="Reaction 13: complex J to A+G",
        stoichiometry=Dict("A" => 1.0, "G" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    # R14: J → A + I (irreversible)
    # Since J is formed from A+G, this effectively transforms G to I while releasing A
    model.reactions["R14"] = CM.Reaction(
        name="Reaction 14: complex J to A+I",
        stoichiometry=Dict("A" => 1.0, "I" => 1.0),
        lower_bound=0.0,  # Irreversible
        upper_bound=1000.0
    )

    # Note: To properly model the J complex, we need to ensure R12, R13, and R14
    # are properly coupled. In the actual system, J is an intermediate complex
    # formed from A+G that can either dissociate back (R13) or proceed to form A+I (R14)

    # For a more accurate representation, we could add J as a pseudo-metabolite:
    # But based on the paper's approach, complexes are handled differently in the kinetic analysis

    # Alternative approach: Add J as a metabolite to properly track the complex
    model.metabolites["J"] = CM.Metabolite(name="Complex J (A+G bound)")

    # Update reactions to properly use J
    model.reactions["R12"] = CM.Reaction(
        name="Reaction 12: A+G to J",
        stoichiometry=Dict("A" => -1.0, "G" => -1.0, "J" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    model.reactions["R13"] = CM.Reaction(
        name="Reaction 13: J to A+G",
        stoichiometry=Dict("J" => -1.0, "A" => 1.0, "G" => 1.0),
        lower_bound=0.0,
        upper_bound=1000.0
    )

    model.reactions["R14"] = CM.Reaction(
        name="Reaction 14: J to A and I",
        stoichiometry=Dict("J" => -1.0, "A" => 1.0, "I" => 1.0),
        lower_bound=0.0,  # Irreversible
        upper_bound=1000.0
    )

    # Add exchange reactions for boundary conditions (to make system feasible)
    # These allow for input/output of metabolites at the system boundary

    # Source reactions for initial substrates
    model.reactions["EX_A"] = CM.Reaction(
        name="Exchange A",
        stoichiometry=Dict("A" => 1.0),
        lower_bound=-10.0,  # Can import or export
        upper_bound=10.0
    )

    model.reactions["EX_E"] = CM.Reaction(
        name="Exchange E",
        stoichiometry=Dict("E" => 1.0),
        lower_bound=-10.0,
        upper_bound=10.0
    )

    model.reactions["EX_G"] = CM.Reaction(
        name="Exchange G",
        stoichiometry=Dict("G" => 1.0),
        lower_bound=-10.0,
        upper_bound=10.0
    )

    # Sink reactions for terminal products
    model.reactions["EX_D"] = CM.Reaction(
        name="Exchange D",
        stoichiometry=Dict("D" => -1.0),
        lower_bound=0.0,
        upper_bound=10.0
    )

    model.reactions["EX_I"] = CM.Reaction(
        name="Exchange I",
        stoichiometry=Dict("I" => -1.0),
        lower_bound=0.0,
        upper_bound=10.0
    )

    return model
end
