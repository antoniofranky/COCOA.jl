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
    model = CM.Model()

    # Paper-style species naming used throughout the strict validation tests.
    model.metabolites = Dict(
        "X" => CM.Metabolite(name="X"),
        "Xp" => CM.Metabolite(name="Xp"),
        "XD" => CM.Metabolite(name="XD"),
        "XT" => CM.Metabolite(name="XT"),
        "XpY" => CM.Metabolite(name="XpY"),
        "XTYp" => CM.Metabolite(name="XTYp"),
        "XDYp" => CM.Metabolite(name="XDYp"),
        "Y" => CM.Metabolite(name="Y"),
        "Yp" => CM.Metabolite(name="Yp")
    )

    # 14 irreversible reactions that realize the 13-complex EnvZ-OmpR paper network:
    # {XD, X, XT, Xp, Xp+Y, XpY, X+Yp, XT+Yp, XTYp, XT+Y, XD+Yp, XDYp, XD+Y}
    model.reactions["R1"] = CM.Reaction(name="XD_to_X", stoichiometry=Dict("XD" => -1.0, "X" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R2"] = CM.Reaction(name="X_to_XD", stoichiometry=Dict("X" => -1.0, "XD" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R3"] = CM.Reaction(name="X_to_XT", stoichiometry=Dict("X" => -1.0, "XT" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R4"] = CM.Reaction(name="XT_to_X", stoichiometry=Dict("XT" => -1.0, "X" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R5"] = CM.Reaction(name="XT_to_Xp", stoichiometry=Dict("XT" => -1.0, "Xp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R6"] = CM.Reaction(name="XpY_binding", stoichiometry=Dict("Xp" => -1.0, "Y" => -1.0, "XpY" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R7"] = CM.Reaction(name="XpY_dissociation", stoichiometry=Dict("XpY" => -1.0, "Xp" => 1.0, "Y" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R8"] = CM.Reaction(name="XpY_to_X_plus_Yp", stoichiometry=Dict("XpY" => -1.0, "X" => 1.0, "Yp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R9"] = CM.Reaction(name="XT_plus_Yp_to_XTYp", stoichiometry=Dict("XT" => -1.0, "Yp" => -1.0, "XTYp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R10"] = CM.Reaction(name="XTYp_to_XT_plus_Yp", stoichiometry=Dict("XTYp" => -1.0, "XT" => 1.0, "Yp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R11"] = CM.Reaction(name="XTYp_to_XT_plus_Y", stoichiometry=Dict("XTYp" => -1.0, "XT" => 1.0, "Y" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R12"] = CM.Reaction(name="XD_plus_Yp_to_XDYp", stoichiometry=Dict("XD" => -1.0, "Yp" => -1.0, "XDYp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R13"] = CM.Reaction(name="XDYp_to_XD_plus_Yp", stoichiometry=Dict("XDYp" => -1.0, "XD" => 1.0, "Yp" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    model.reactions["R14"] = CM.Reaction(name="XDYp_to_XD_plus_Y", stoichiometry=Dict("XDYp" => -1.0, "XD" => 1.0, "Y" => 1.0), lower_bound=0.0, upper_bound=1000.0)

    return model
end
