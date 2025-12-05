import AbstractFBCModels.CanonicalModel as CM

"""
Create the EnvZ-OmpR network for kinetic module testing (without exchange reactions).
This has the correct topology from the paper for testing kinetic module identification.

Species mapping to paper notation:
- XD (EnvZ, deactivating form)
- X (EnvZ, inactive form)
- XT (EnvZ, activating form)
- Xp (EnvZ-phosphorylated)
- Y (OmpR)
- XpY (Xp+Y complex)
- Yp (OmpR-phosphorylated)
- XTYp (XT+Yp complex)
- XDYp (XD+Yp complex)
"""
function create_envz_ompr_model()
    model = CM.Model()

    # Add metabolites (using paper notation)
    metabolites = [:XD, :X, :XT, :Xp, :Y, :XpY, :Yp, :XTYp, :XDYp]
    for met in metabolites
        model.metabolites[String(met)] = CM.Metabolite(name=String(met))
    end

    # Add reactions matching the network structure (using paper notation)
    # R1-R2: XD ⇄ X (EnvZ deactivating ⇄ inactive)
    # R3-R4: X ⇄ XT (EnvZ inactive ⇄ activating)
    # R5: XT → Xp (phosphorylation)
    # R6-R7: Xp + Y ⇄ XpY (complex formation)
    # R8: XpY → X + Yp (phosphotransfer)
    # R9-R10: XT + Yp ⇄ XTYp (complex formation)
    # R11: XTYp → XT + Y (dephosphorylation)
    # R12-R13: XD + Yp ⇄ XDYp (complex formation)
    # R14: XDYp → XD + Y (dephosphorylation)
    reactions = [
        ("R1", Dict("XD" => -1.0, "X" => 1.0)),
        ("R2", Dict("X" => -1.0, "XD" => 1.0)),
        ("R3", Dict("X" => -1.0, "XT" => 1.0)),
        ("R4", Dict("XT" => -1.0, "X" => 1.0)),
        ("R5", Dict("XT" => -1.0, "Xp" => 1.0)),
        ("R6", Dict("Xp" => -1.0, "Y" => -1.0, "XpY" => 1.0)),
        ("R7", Dict("XpY" => -1.0, "Xp" => 1.0, "Y" => 1.0)),
        ("R8", Dict("XpY" => -1.0, "X" => 1.0, "Yp" => 1.0)),
        ("R9", Dict("XT" => -1.0, "Yp" => -1.0, "XTYp" => 1.0)),
        ("R10", Dict("XTYp" => -1.0, "XT" => 1.0, "Yp" => 1.0)),
        ("R11", Dict("XTYp" => -1.0, "XT" => 1.0, "Y" => 1.0)),
        ("R12", Dict("XD" => -1.0, "Yp" => -1.0, "XDYp" => 1.0)),
        ("R13", Dict("XDYp" => -1.0, "XD" => 1.0, "Yp" => 1.0)),
        ("R14", Dict("XDYp" => -1.0, "XD" => 1.0, "Y" => 1.0))
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