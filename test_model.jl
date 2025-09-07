using AbstractFBCModels
using AbstractFBCModels.CanonicalModel
# Create the toy model from Langary et al. 2025 paper (Figure 1)
model = CanonicalModel.Model()

# Define metabolites A through J
metabolite_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
for met_id in metabolite_ids
    model.metabolites[met_id] = CanonicalModel.Metabolite(
        name="Metabolite $met_id",
        compartment=nothing,
        formula=nothing,
        charge=nothing,
        balance=0.0,
        annotations=Dict{String,Vector{String}}(),
        notes=Dict{String,Vector{String}}()
    )
end

# Define reactions based on the diagram
# R1: A ⇌ B (reversible)
model.reactions["R1"] = CanonicalModel.Reaction(
    name="A to B",
    lower_bound=-1000.0,
    upper_bound=1000.0,
    stoichiometry=Dict("A" => -1.0, "B" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R3: B ⇌ C (reversible)
model.reactions["R3"] = CanonicalModel.Reaction(
    name="B to C",
    lower_bound=-1000.0,
    upper_bound=1000.0,
    stoichiometry=Dict("B" => -1.0, "C" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R5: C → D (irreversible)
model.reactions["R5"] = CanonicalModel.Reaction(
    name="C to D",
    lower_bound=0.0,
    upper_bound=1000.0,
    stoichiometry=Dict("C" => -1.0, "D" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R6: D + E ⇌ F (reversible)
model.reactions["R6"] = CanonicalModel.Reaction(
    name="D + E to F",
    lower_bound=-1000.0,
    upper_bound=1000.0,
    stoichiometry=Dict("D" => -1.0, "E" => -1.0, "F" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R8: F → B + G (irreversible)
model.reactions["R8"] = CanonicalModel.Reaction(
    name="F to B + G",
    lower_bound=0.0,
    upper_bound=1000.0,
    stoichiometry=Dict("F" => -1.0, "B" => 1.0, "G" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R9: C + G ⇌ H (reversible)
model.reactions["R9"] = CanonicalModel.Reaction(
    name="C + G to H",
    lower_bound=-1000.0,
    upper_bound=1000.0,
    stoichiometry=Dict("C" => -1.0, "G" => -1.0, "H" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R11: H → C + I (irreversible)
model.reactions["R11"] = CanonicalModel.Reaction(
    name="H to C + I",
    lower_bound=0.0,
    upper_bound=1000.0,
    stoichiometry=Dict("H" => -1.0, "C" => 1.0, "I" => 1.0),
    objective_coefficient=1.0,  # Maximize I production
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R12: A + G ⇌ J (reversible)
model.reactions["R12"] = CanonicalModel.Reaction(
    name="A + G to J",
    lower_bound=-1000.0,
    upper_bound=1000.0,
    stoichiometry=Dict("A" => -1.0, "G" => -1.0, "J" => 1.0),
    objective_coefficient=0.0,
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

# R14: J → A + I (irreversible)
model.reactions["R14"] = CanonicalModel.Reaction(
    name="J to A + I",
    lower_bound=0.0,
    upper_bound=1000.0,
    stoichiometry=Dict("J" => -1.0, "A" => 1.0, "I" => 1.0),
    objective_coefficient=1.0,  # Also maximize I production
    gene_association_dnf=nothing,
    annotations=Dict{String,Vector{String}}(),
    notes=Dict{String,Vector{String}}()
)

println("Model created with $(length(AbstractFBCModels.metabolites(model))) metabolites and
$(length(AbstractFBCModels.reactions(model))) reactions")
println("Objective: Maximize I production (R11 + R14)")

# Display reactions
println("\nReactions:")
for rxn_id in AbstractFBCModels.reactions(model)
    rxn = model.reactions[rxn_id]
    stoich_parts = String[]
    substrates = String[]
    products = String[]

    for (met, coeff) in rxn.stoichiometry
        if coeff < 0
            if abs(coeff) == 1
                push!(substrates, met)
            else
                push!(substrates, "$(abs(coeff))$met")
            end
        else
            if coeff == 1
                push!(products, met)
            else
                push!(products, "$(coeff)$met")
            end
        end
    end

    direction = rxn.lower_bound < 0 ? " ⇌ " : " → "
    reaction_str = join(substrates, " + ") * direction * join(products, " + ")
end
