import AbstractFBCModels.CanonicalModel as CM

"""
Create the deficiency two network model from Fig. S-6 of the paper.
This network demonstrates advanced merging capabilities.

Reactions and rates (from Fig. S-6):
1.  F ⇄ A      (k1, k2)
2.  A → B      (k3)
3.  C + F → E  (k4)
4.  E → D + F  (k5)
5.  B + D → A + C (k6)
6.  X ⇄ S      (k7, k8)
7.  S → T      (k9)
8.  U + X → W  (k10)
9.  W → V + X  (k11)
10. T + Z → S + U (k12)
11. V → D + Y  (k13)
12. D + Y → Z  (k14)
"""
function create_deficiency_two_model()
    model = CM.Model()

    # Metabolites
    metabolites = [:F, :A, :B, :C, :E, :D, :X, :S, :T, :U, :W, :V, :Y, :Z]
    for met in metabolites
        model.metabolites[String(met)] = CM.Metabolite(name=String(met))
    end

    # Reactions
    reactions = [
        ("R1", Dict("F" => -1.0, "A" => 1.0)),
        ("R2", Dict("A" => -1.0, "F" => 1.0)),
        ("R3", Dict("A" => -1.0, "B" => 1.0)),
        ("R4", Dict("C" => -1.0, "F" => -1.0, "E" => 1.0)),
        ("R5", Dict("E" => -1.0, "D" => 1.0, "F" => 1.0)),
        ("R6", Dict("B" => -1.0, "D" => -1.0, "A" => 1.0, "C" => 1.0)),
        ("R7", Dict("X" => -1.0, "S" => 1.0)),
        ("R8", Dict("S" => -1.0, "X" => 1.0)),
        ("R9", Dict("S" => -1.0, "T" => 1.0)),
        ("R10", Dict("U" => -1.0, "X" => -1.0, "W" => 1.0)),
        ("R11", Dict("W" => -1.0, "V" => 1.0, "X" => 1.0)),
        ("R12", Dict("T" => -1.0, "Z" => -1.0, "S" => 1.0, "U" => 1.0)),
        ("R13", Dict("V" => -1.0, "D" => 1.0, "Y" => 1.0)),
        ("R14", Dict("D" => -1.0, "Y" => -1.0, "Z" => 1.0))
    ]

    for (rxn_id, stoich) in reactions
        model.reactions[rxn_id] = CM.Reaction(
            name=rxn_id,
            stoichiometry=stoich,
            lower_bound=0.0,
            upper_bound=1000.0
        )
    end

    return model
end
