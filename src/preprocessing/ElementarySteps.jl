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
using HiGHS
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM

# Include sub-modules
include("enzyme_registry.jl")
include("intermediates.jl")
include("mechanisms.jl")

export split_into_elementary_steps, validate_split_model, normalize_bounds!

"""
    normalize_bounds!(model::CM.Model; 
                     lower_bound::Float64 = -1000.0,
                     upper_bound::Float64 = 1000.0,
                     normalize_objective_bounds::Bool = true)

Normalize model bounds following MATLAB preprocessing logic.

# MATLAB Logic:
# - All negative lower bounds → -1000 (reversible)
# - All positive lower bounds → 0 (irreversible forward)  
# - All negative upper bounds → 0 (irreversible reverse)
# - All positive upper bounds → 1000 (forward unlimited)
# - Objective reactions: lb = 0, ub = 1000 (force forward)

# Arguments
- `model`: Model to normalize (modified in-place)
- `lower_bound`: Generic lower bound for reversible reactions (default: -1000)
- `upper_bound`: Generic upper bound for unlimited reactions (default: 1000)  
- `normalize_objective_bounds`: Whether to force objective reactions to be forward-only
"""
function normalize_bounds!(
    model::CM.Model;
    lower_bound::Float64=-1000.0,
    upper_bound::Float64=1000.0,
    normalize_objective_bounds::Bool=true
)
    @info "Normalizing model bounds"

    original_bounds = []
    normalized_count = 0

    for (rid, rxn) in model.reactions
        original_lb = rxn.lower_bound
        original_ub = rxn.upper_bound
        push!(original_bounds, (rid, original_lb, original_ub))

        # MATLAB logic: model.lb(model.lb<0) = -1000;
        if rxn.lower_bound < 0
            rxn.lower_bound = lower_bound
            normalized_count += 1
        end

        # MATLAB logic: model.lb(model.lb>0) = 0;
        if rxn.lower_bound > 0
            rxn.lower_bound = 0.0
            normalized_count += 1
        end

        # MATLAB logic: model.ub(model.ub<0) = 0;
        if rxn.upper_bound < 0
            rxn.upper_bound = 0.0
            normalized_count += 1
        end

        # MATLAB logic: model.ub(model.ub>0) = 1000;
        if rxn.upper_bound > 0
            rxn.upper_bound = upper_bound
            normalized_count += 1
        end

        # MATLAB logic: model.lb(model.c~=0) = 0; model.ub(model.c~=0) = 1000;
        if normalize_objective_bounds && abs(rxn.objective_coefficient) > 1e-12
            rxn.lower_bound = 0.0
            rxn.upper_bound = upper_bound
            @info "Forced objective reaction $rid to forward-only: lb=0, ub=$upper_bound"
        end
    end

    @info "Normalized bounds for $normalized_count reaction bounds"

    return original_bounds  # Return for potential restoration
end

"""
    split_into_elementary_steps(model::A.AbstractFBCModel; 
                               ordered_fraction::Float64 = 1.0,
                               mechanism_assignment::Union{Nothing, Dict{String,Symbol}} = nothing,
                               output_type::Type{<:A.AbstractFBCModel} = SBMLFBCModels.SBMLFBCModel,
                               max_substrates::Int = 4,
                               max_products::Int = 4,
                               max_random_orders::Int = 10,
                               seed::Union{Int, Nothing} = nothing,
                               normalize_bounds::Bool = false)

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
- `normalize_bounds`: Whether to apply MATLAB-style bounds normalization (default: false)

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
    seed::Union{Int,Nothing}=nothing,
    normalize_bounds::Bool=false
)
    # Initialize RNG
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Convert to canonical model for easier manipulation
    work_model = convert(CM.Model, model)

    # Apply bounds normalization if requested
    if normalize_bounds
        normalize_bounds!(work_model)
    end

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

    #TODO: Handle objective setting for conversion (e.g. SBML adds R_ and then can not find objective reaction)

    # Convert to requested output type
    exported_model = convert(output_type, elementary_model)

    # Validate model integrity
    validate_split_model(model, exported_model)

    return exported_model
end

# For Julia < 1.7 compatibility
if !isdefined(Base, :allequal)
    allequal(itr) = isempty(itr) || all(==(first(itr)), itr)
end

"""
Validate that the split model preserves key properties.
"""
function validate_split_model(original_model::A.AbstractFBCModel,
    split_model::A.AbstractFBCModel)
    # Check objective preservation
    original_obj_sum = sum(abs(v) for v in A.objective(original_model))
    split_obj_sum = sum(abs(v) for v in A.objective(split_model))

    if abs(original_obj_sum - split_obj_sum) > 1e-6
        @warn "Objective sum changed: $original_obj_sum → $split_obj_sum"
    end

    # Check metabolite balance
    original_mets = Set(A.metabolites(original_model))
    split_mets = Set(A.metabolites(split_model))
    co = COBREXA.flux_balance_constraints(original_model)
    oo = COBREXA.optimized_values(co, optimizer=HiGHS.Optimizer, objective=co.objective.value, output=co.objective)
    csplt = COBREXA.flux_balance_constraints(split_model)
    so = COBREXA.optimized_values(csplt, optimizer=HiGHS.Optimizer, objective=csplt.objective.value, output=csplt.objective)

    if abs(oo - so) > 1e-6
        @warn "Flux balance objective value changed: $oo → $so"
    else
        @info "Flux balance objective value preserved: $oo → $so"
    end
    # Original metabolites should all be present
    missing_mets = setdiff(original_mets, split_mets)
    if !isempty(missing_mets)
        @warn "Missing metabolites: $missing_mets"
    end

    return true
end

end # module