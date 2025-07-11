"""
Intermediate metabolite management for elementary step splitting.

This module handles:
- Creation and management of enzyme-substrate intermediates
- Intermediate metabolite registry
- Intermediate ID generation and tracking
"""

import AbstractFBCModels.CanonicalModel as CM

"""
Get existing intermediate or create new one.
"""
function get_or_create_intermediate!(
    elementary_model::CM.Model,
    intermediate_registry::Dict{Vector{String},String},
    metabolites::Vector{String},
    enzyme_id::String
)
    # Create canonical representation
    intermediate_key = sort([metabolites; enzyme_id])

    if haskey(intermediate_registry, intermediate_key)
        return intermediate_registry[intermediate_key]
    end

    # Create new intermediate
    intermediate_id = "INTRM_$(length(intermediate_registry) + 1)"
    intermediate_registry[intermediate_key] = intermediate_id

    # Add to model
    metabolite_names = join(sort(metabolites), ", ")
    elementary_model.metabolites[intermediate_id] = CM.Metabolite(
        name="Intermediate: $enzyme_id with $metabolite_names",
        compartment="c",  # Inherit from metabolites in future version
        annotations=Dict("sbo" => ["SBO:0000297"])  # protein-small molecule intermediate
    )

    return intermediate_id
end