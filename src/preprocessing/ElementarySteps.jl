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
                        add_ordered_mechanism_reactions!(
                            elementary_model, rid, rxn, enzyme_id,
                            substrates, products, intermediate_registry,
                            reaction_count, rng
                        )
                    else  # :random
                        add_random_mechanism_reactions!(
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

# Helper functions

function build_enzyme_registry(model::CM.Model)
    """Build a registry of all enzymes from gene associations."""
    enzyme_registry = Dict{String,String}()
    enzyme_counter = 0

    for (rid, rxn) in model.reactions
        if !isnothing(rxn.gene_association_dnf)
            for gene_group in rxn.gene_association_dnf
                enzyme_counter += 1
                if length(gene_group) == 1
                    # Single gene = single enzyme
                    enzyme_id = "ENZ_$(gene_group[1])"
                    enzyme_registry[enzyme_id] = gene_group[1]
                else
                    # Multiple genes = enzyme intermediate
                    intermediate_name = join(sort(gene_group), "_")
                    enzyme_id = "ENZ_$intermediate_name"
                    enzyme_registry[enzyme_id] = join(sort(gene_group), " & ")
                end
            end
        end
    end

    return enzyme_registry
end

function assign_reaction_mechanisms(
    model::CM.Model, ordered_fraction::Float64,
    max_substrates::Int, max_products::Int, rng::AbstractRNG
)
    """Assign ordered or random mechanism to each eligible reaction."""
    mechanisms = Dict{String,Symbol}()

    eligible_reactions = String[]
    for (rid, rxn) in model.reactions
        # Check if reaction has gene association
        if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
            continue
        end

        # Check substrate/product limits
        n_substrates = count(coeff < 0 for (_, coeff) in rxn.stoichiometry)
        n_products = count(coeff > 0 for (_, coeff) in rxn.stoichiometry)

        if n_substrates <= max_substrates && n_products <= max_products
            push!(eligible_reactions, rid)
        end
    end

    # Randomly assign mechanisms
    n_ordered = round(Int, length(eligible_reactions) * ordered_fraction)
    shuffle!(rng, eligible_reactions)

    for (i, rid) in enumerate(eligible_reactions)
        mechanisms[rid] = i <= n_ordered ? :ordered : :random
    end

    return mechanisms
end

function extract_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String})
    """Extract enzyme IDs for a reaction based on gene associations."""
    enzyme_ids = String[]

    if !isnothing(rxn.gene_association_dnf)
        for gene_group in rxn.gene_association_dnf
            if length(gene_group) == 1
                enzyme_id = "ENZ_$(gene_group[1])"
            else
                intermediate_name = join(sort(gene_group), "_")
                enzyme_id = "ENZ_$intermediate_name"
            end

            if haskey(enzyme_registry, enzyme_id)
                push!(enzyme_ids, enzyme_id)
            end
        end
    end

    return enzyme_ids
end

function get_or_create_intermediate!(
    elementary_model::CM.Model,
    intermediate_registry::Dict{Vector{String},String},
    metabolites::Vector{String},
    enzyme_id::String
)
    """Get existing intermediate or create new one."""
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

function add_ordered_mechanism_reactions!(
    elementary_model::CM.Model,
    original_rid::String,
    original_rxn::CM.Reaction,
    enzyme_id::String,
    substrates::Vector{Tuple{String,Float64}},
    products::Vector{Tuple{String,Float64}},
    intermediate_registry::Dict{Vector{String},String},
    reaction_count::Int,
    rng::AbstractRNG
)
    """Add elementary reactions for ordered binding mechanism."""

    # Randomize but fix the order for this enzyme
    substrates_ordered = shuffle(rng, substrates)
    products_ordered = shuffle(rng, products)

    current_intermediate_metabolites = String[]

    # 1. Substrate binding steps
    for (idx, (met_id, coeff)) in enumerate(substrates_ordered)
        reaction_count += 1
        rxn_id = "$(original_rid)_$(enzyme_id)_S$(idx)"

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): substrate binding step $idx",
            lower_bound=-1000.0,  # Reversible binding
            upper_bound=1000.0,
            gene_association_dnf=original_rxn.gene_association_dnf
        )

        # Add substrate
        elem_rxn.stoichiometry[met_id] = -abs(coeff)

        if idx == 1
            # First substrate binds to free enzyme
            elem_rxn.stoichiometry[enzyme_id] = -1.0
        else
            # Subsequent substrates bind to intermediate
            prev_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_intermediate_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[prev_intermediate] = -1.0
        end

        # Create product intermediate
        push!(current_intermediate_metabolites, met_id)
        new_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            current_intermediate_metabolites, enzyme_id
        )
        elem_rxn.stoichiometry[new_intermediate] = 1.0

        elementary_model.reactions[rxn_id] = elem_rxn
    end

    # 2. Catalytic step
    reaction_count += 1
    cat_rxn_id = "$(original_rid)_$(enzyme_id)_CAT"
    cat_rxn = CM.Reaction(
        name="$(original_rxn.name): catalysis",
        lower_bound=original_rxn.lower_bound >= 0 ? 0.0 : -1000.0,
        upper_bound=1000.0,
        gene_association_dnf=original_rxn.gene_association_dnf,
        objective_coefficient=original_rxn.objective_coefficient  # Preserve objective
    )

    # Substrate intermediate to product intermediate
    substrate_intermediate = get_or_create_intermediate!(
        elementary_model, intermediate_registry,
        current_intermediate_metabolites, enzyme_id
    )
    cat_rxn.stoichiometry[substrate_intermediate] = -1.0

    if !isempty(products_ordered)
        product_metabolites = [p[1] for p in products_ordered]
        product_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            product_metabolites, enzyme_id
        )
        cat_rxn.stoichiometry[product_intermediate] = 1.0
        current_intermediate_metabolites = product_metabolites
    else
        # No products - release enzyme
        cat_rxn.stoichiometry[enzyme_id] = 1.0
        current_intermediate_metabolites = String[]
    end

    elementary_model.reactions[cat_rxn_id] = cat_rxn

    # 3. Product release steps
    for (idx, (met_id, coeff)) in enumerate(products_ordered)
        reaction_count += 1
        rxn_id = "$(original_rid)_$(enzyme_id)_P$(idx)"

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): product release step $idx",
            lower_bound=-1000.0,  # Reversible release
            upper_bound=1000.0,
            gene_association_dnf=original_rxn.gene_association_dnf
        )

        # Current intermediate as substrate
        current_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            current_intermediate_metabolites, enzyme_id
        )
        elem_rxn.stoichiometry[current_intermediate] = -1.0

        # Release product
        elem_rxn.stoichiometry[met_id] = abs(coeff)

        # Remove this metabolite from intermediate
        filter!(m -> m != met_id, current_intermediate_metabolites)

        if !isempty(current_intermediate_metabolites)
            # Create smaller intermediate
            new_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_intermediate_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[new_intermediate] = 1.0
        else
            # Last product - release enzyme
            elem_rxn.stoichiometry[enzyme_id] = 1.0
        end

        elementary_model.reactions[rxn_id] = elem_rxn
    end
end

function add_random_mechanism_reactions!(
    elementary_model::CM.Model,
    original_rid::String,
    original_rxn::CM.Reaction,
    enzyme_id::String,
    substrates::Vector{Tuple{String,Float64}},
    products::Vector{Tuple{String,Float64}},
    intermediate_registry::Dict{Vector{String},String},
    reaction_count::Int,
    max_orders::Int,
    rng::AbstractRNG
)
    """Add elementary reactions for random binding mechanism.
    
    Instead of generating all permutations, we sample a limited number
    of binding orders to keep the model tractable.
    """

    n_substrates = length(substrates)
    n_products = length(products)

    # Determine number of orders to generate
    n_possible_orders = factorial(n_substrates)
    n_orders = min(max_orders, n_possible_orders)

    # Generate unique binding orders
    if n_orders == n_possible_orders
        # Generate all permutations
        substrate_orders = collect(permutations(1:n_substrates))
    else
        # Sample random orders
        substrate_orders = Set{Vector{Int}}()
        while length(substrate_orders) < n_orders
            push!(substrate_orders, shuffle(rng, collect(1:n_substrates)))
        end
        substrate_orders = collect(substrate_orders)
    end

    # Track all possible final substrate intermediates
    final_substrate_intermediates = String[]

    # Generate reactions for each binding order
    for (order_idx, order) in enumerate(substrate_orders)
        ordered_substrates = substrates[order]

        # Create hierarchical binding for this order
        for level in 1:n_substrates
            # Get all combinations of metabolites at this level
            current_metabolites = [ordered_substrates[i][1] for i in 1:level]
            prev_metabolites = level > 1 ? [ordered_substrates[i][1] for i in 1:(level-1)] : String[]

            reaction_count += 1
            rxn_id = "$(original_rid)_$(enzyme_id)_ORD$(order_idx)_L$(level)"

            elem_rxn = CM.Reaction(
                name="$(original_rxn.name): random binding order $order_idx level $level",
                lower_bound=-1000.0,
                upper_bound=1000.0,
                gene_association_dnf=original_rxn.gene_association_dnf
            )

            # Add new substrate
            met_id, coeff = ordered_substrates[level]
            elem_rxn.stoichiometry[met_id] = -abs(coeff)

            if level == 1
                # Bind to enzyme
                elem_rxn.stoichiometry[enzyme_id] = -1.0
            else
                # Bind to previous intermediate
                prev_intermediate = get_or_create_intermediate!(
                    elementary_model, intermediate_registry,
                    prev_metabolites, enzyme_id
                )
                elem_rxn.stoichiometry[prev_intermediate] = -1.0
            end

            # Create new intermediate
            new_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[new_intermediate] = 1.0

            elementary_model.reactions[rxn_id] = elem_rxn

            # Track final intermediate for this order
            if level == n_substrates
                push!(final_substrate_intermediates, new_intermediate)
            end
        end
    end

    # Single catalytic step (all orders lead to same product intermediate)
    reaction_count += 1
    cat_rxn_id = "$(original_rid)_$(enzyme_id)_CAT"
    cat_rxn = CM.Reaction(
        name="$(original_rxn.name): catalysis",
        lower_bound=original_rxn.lower_bound >= 0 ? 0.0 : -1000.0,
        upper_bound=1000.0,
        gene_association_dnf=original_rxn.gene_association_dnf,
        objective_coefficient=original_rxn.objective_coefficient
    )

    # All substrate intermediates can undergo catalysis
    for substrate_intermediate in final_substrate_intermediates
        # Create separate catalytic reaction for each
        cat_rxn_specific_id = "$(cat_rxn_id)_$(substrate_intermediate)"
        cat_rxn_specific = deepcopy(cat_rxn)
        cat_rxn_specific.stoichiometry[substrate_intermediate] = -1.0

        if !isempty(products)
            product_metabolites = [p[1] for p in products]
            product_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                product_metabolites, enzyme_id
            )
            cat_rxn_specific.stoichiometry[product_intermediate] = 1.0
        else
            cat_rxn_specific.stoichiometry[enzyme_id] = 1.0
        end

        elementary_model.reactions[cat_rxn_specific_id] = cat_rxn_specific
    end

    # Product release (same for all binding orders)
    if !isempty(products)
        # Only generate one set of release reactions
        current_metabolites = [p[1] for p in products]

        for (idx, (met_id, coeff)) in enumerate(products)
            reaction_count += 1
            rxn_id = "$(original_rid)_$(enzyme_id)_P$(idx)"

            elem_rxn = CM.Reaction(
                name="$(original_rxn.name): product release step $idx",
                lower_bound=-1000.0,
                upper_bound=1000.0,
                gene_association_dnf=original_rxn.gene_association_dnf
            )

            current_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[current_intermediate] = -1.0
            elem_rxn.stoichiometry[met_id] = abs(coeff)

            filter!(m -> m != met_id, current_metabolites)

            if !isempty(current_metabolites)
                new_intermediate = get_or_create_intermediate!(
                    elementary_model, intermediate_registry,
                    current_metabolites, enzyme_id
                )
                elem_rxn.stoichiometry[new_intermediate] = 1.0
            else
                elem_rxn.stoichiometry[enzyme_id] = 1.0
            end

            elementary_model.reactions[rxn_id] = elem_rxn
        end
    end
end

# For Julia < 1.7 compatibility
if !isdefined(Base, :allequal)
    allequal(itr) = isempty(itr) || all(==(first(itr)), itr)
end

function permutations(a::AbstractVector)
    # Simple implementation for small arrays
    n = length(a)
    n == 0 && return [eltype(a)[]]
    n == 1 && return [copy(a)]

    perms = Vector{eltype(a)}[]
    for i in 1:n
        rest = [a[j] for j in 1:n if j != i]
        for p in permutations(rest)
            push!(perms, vcat(a[i], p))
        end
    end
    return perms
end

end # module