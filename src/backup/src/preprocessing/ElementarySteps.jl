"""
Elementary step splitting functionality for COCOA.

This module implements the decomposition of metabolic reactions into elementary
steps based on enzyme mechanisms (ordered or random binding).
"""
module ElementarySteps

using COBREXA
using AbstractFBCModels
using SBMLFBCModels
using JSONFBCModels
using Random
using SparseArrays
using Logging

import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM

# Include sub-modules
include("enzyme_registry.jl")
include("intermediates.jl")
include("mechanisms.jl")

export split_into_elementary_steps

"""
    split_into_elementary_steps(model::A.AbstractFBCModel; 
                               ordered_fraction::Float64 = 1.0,
                               mechanism_assignment::Union{Nothing, Dict{String,Symbol}} = nothing,
                               output_type::Type{<:A.AbstractFBCModel} = SBMLFBCModels.SBMLFBCModel,
                               max_substrates::Int = 4,
                               max_products::Int = 4,
                               max_random_orders::Int = 10,
                               seed::Union{Int, Nothing} = nothing)

Split reactions into elementary steps based on enzyme mechanisms.

# Arguments
- `model`: The metabolic model to split
- `ordered_fraction`: Fraction of reactions to use ordered mechanism (default: 1.0)
- `mechanism_assignment`: Optional dict specifying mechanism per reaction
- `output_type`: Model type to return (default: CM.Model (for AbstractFBCModels.CanonicalModel.Model), also supports JSONModel, SBMLFBCModel)
- `max_substrates`: Maximum substrates per reaction to split (default: 4)
- `max_products`: Maximum products per reaction to split (default: 4)
- `max_random_orders`: Maximum binding orders to generate for random mechanism (default: 10)
- `seed`: Random seed for reproducibility

# Returns
- Model with reactions split into elementary steps in the requested format
"""
function split_into_elementary_steps(
    model::A.AbstractFBCModel;
    ordered_fraction::Float64=1.0,
    mechanism_assignment::Union{Nothing,Dict{String,Symbol}}=nothing,
    output_type::Type{<:A.AbstractFBCModel}=CM.Model,
    max_substrates::Int=4,
    max_products::Int=4,
    max_random_orders::Int=10,
    seed::Union{Int,Nothing}=nothing
)
    # Initialize RNG
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Convert to canonical model for easier manipulation
    work_model = convert(CM.Model, model)

    # Extract and prepare enzyme information
    enzyme_registry = build_enzyme_registry(work_model)

    # Assign mechanisms to reactions
    if isnothing(mechanism_assignment)
        mechanism_assignment = assign_reaction_mechanisms(
            work_model, ordered_fraction, max_substrates, max_products, rng
        )
    end

    # Initialize elementary model
    elementary_model = CM.Model()

    # Copy metabolites and add enzymes
    for (mid, met) in work_model.metabolites
        elementary_model.metabolites[mid] = deepcopy(met)
    end

    # Add enzyme metabolites
    for (enzyme_id, enzyme_name) in enzyme_registry
        elementary_model.metabolites[enzyme_id] = CM.Metabolite(
            name="Enzyme: $enzyme_name",
            compartment="c",  # Default to cytosol
            annotations=Dict("sbo" => ["SBO:0000252"])  # polypeptide chain
        )
    end

    # Track intermediate metabolites and mappings
    intermediate_registry = Dict{Vector{String},String}()
    reaction_count = 0

    # Process each reaction
    for (rid, rxn) in work_model.reactions
        if haskey(mechanism_assignment, rid)
            mechanism = mechanism_assignment[rid]

            # Get substrates and products
            substrates = [(mid, -coeff) for (mid, coeff) in rxn.stoichiometry if coeff < 0]
            products = [(mid, coeff) for (mid, coeff) in rxn.stoichiometry if coeff > 0]

            # Extract enzymes for this reaction
            reaction_enzymes = extract_reaction_enzymes(rxn, enzyme_registry)

            if isempty(reaction_enzymes)
                # No enzymes - keep original reaction
                elementary_model.reactions[rid] = deepcopy(rxn)
            else
                # Split reaction for each enzyme
                for enzyme_id in reaction_enzymes
                    if mechanism == :ordered
                        add_ordered_reactions!(
                            elementary_model, rid, rxn, enzyme_id,
                            substrates, products, intermediate_registry,
                            reaction_count, rng
                        )
                    else  # :random
                        add_random_reactions!(
                            elementary_model, rid, rxn, enzyme_id,
                            substrates, products, intermediate_registry,
                            reaction_count, max_random_orders, rng
                        )
                    end
                end
            end
        else
            # Keep original reaction
            elementary_model.reactions[rid] = deepcopy(rxn)
        end
    end

    # Copy genes
    elementary_model.genes = deepcopy(work_model.genes)

    # Copy couplings if any
    elementary_model.couplings = deepcopy(work_model.couplings)

    @info "Split $(length(work_model.reactions)) reactions into $(length(elementary_model.reactions)) elementary steps"
    @info "Created $(length(enzyme_registry)) enzymes and $(length(intermediate_registry)) intermediate complexes"

    #TODO: Handle objective setting for converstion (e.g. SBML adds R_ and then can not find objective reaction)

    # Convert to requested output type
    exported_model = convert(output_type, elementary_model)

    return exported_model
end

# For Julia < 1.7 compatibility
if !isdefined(Base, :allequal)
    allequal(itr) = isempty(itr) || all(==(first(itr)), itr)
end

end # module