"""
Elementary step splitting functionality for COCOA.

This module implements the decomposition of metabolic reactions into elementary
steps based on enzyme mechanisms (fixed/ordered or random binding).
"""
module ElementarySteps

using COBREXA
using AbstractFBCModels
using Random
using Logging

import COBREXA: StandardModel

export split_into_elementary_steps

# Custom types for enzyme and intermediate metabolites
struct EnzymeMetabolite
    id::String
    name::String
    enzyme_name::String
    compartment::String
end

struct IntermediateMetabolite
    id::String
    name::String
    enzyme_id::String
    metabolites::Vector{String}
    coefficients::Vector{Float64}
    intermediate_type::Symbol  # :substrate or :product
    compartment::String
end

"""
    split_into_elementary_steps(model::StandardModel; 
                               mechanism::Symbol = :fixed,
                               max_substrates::Int = 4,
                               max_products::Int = 4,
                               seed::Union{Int, Nothing} = nothing)

Split reactions into elementary steps based on enzyme mechanisms.

# Arguments
- `model`: The metabolic model to split
- `mechanism`: Enzyme mechanism (`:fixed` for ordered binding, `:random` for random binding)
- `max_substrates`: Maximum substrates per reaction to split (default: 4)
- `max_products`: Maximum products per reaction to split (default: 4)
- `seed`: Random seed for reproducible substrate/product ordering

# Returns
- Model with reactions split into elementary steps
"""
function split_into_elementary_steps(
    model::StandardModel;
    mechanism::Symbol=:fixed,
    max_substrates::Int=4,
    max_products::Int=4,
    seed::Union{Int,Nothing}=nothing
)
    # Create RNG for deterministic behavior
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Create new model for elementary steps
    elementary_model = StandardModel("$(model.id)_elementary")

    # Copy metabolites
    for (mid, met) in model.metabolites
        elementary_model.metabolites[mid] = deepcopy(met)
    end

    # Track enzymes and intermediates
    enzyme_registry = Dict{String,EnzymeMetabolite}()
    intermediate_registry = Dict{String,IntermediateMetabolite}()

    # Process each reaction
    reactions_to_process = collect(model.reactions)

    for (rid, rxn) in reactions_to_process
        # Get substrates and products
        substrates = [(mid, -coef) for (mid, coef) in rxn.stoichiometry if coef < 0]
        products = [(mid, coef) for (mid, coef) in rxn.stoichiometry if coef > 0]

        # Skip if no gene rule or too many metabolites
        if isempty(rxn.gene_association_dnf) ||
           length(substrates) > max_substrates ||
           length(products) > max_products
            # Keep original reaction
            elementary_model.reactions[rid] = deepcopy(rxn)
            continue
        end

        @debug "Processing reaction $rid with $mechanism mechanism"

        # Extract enzymes from gene rules
        enzymes = extract_enzymes_from_gene_rule(rxn.gene_association_dnf)

        if isempty(enzymes)
            # Keep original reaction if no enzymes found
            elementary_model.reactions[rid] = deepcopy(rxn)
            continue
        end

        # Determine reaction compartment
        compartment = get_reaction_compartment(rxn, model)

        # Process each enzyme (isoenzyme)
        for enzyme_name in enzymes
            # Get or create enzyme metabolite
            enzyme_met = get_or_create_enzyme!(
                enzyme_registry,
                elementary_model,
                enzyme_name,
                compartment
            )

            # Split reaction based on mechanism
            if mechanism == :fixed
                elementary_rxns = fixed_binding_mechanism(
                    rid, rxn, enzyme_met,
                    substrates, products,
                    elementary_model, intermediate_registry,
                    compartment, rng
                )
            elseif mechanism == :random
                elementary_rxns = random_binding_mechanism(
                    rid, rxn, enzyme_met,
                    substrates, products,
                    elementary_model, intermediate_registry,
                    compartment, rng
                )
            else
                error("Unknown mechanism: $mechanism. Use :fixed or :random")
            end

            # Add elementary reactions to model
            for (elem_id, elem_rxn) in elementary_rxns
                elementary_model.reactions[elem_id] = elem_rxn
            end
        end
    end

    # Update objective function
    update_objective_for_elementary!(elementary_model, model)

    @info "Split $(length(model.reactions)) reactions into $(length(elementary_model.reactions)) elementary steps"
    @info "Created $(length(enzyme_registry)) enzymes and $(length(intermediate_registry)) intermediates"

    return elementary_model
end

"""
    extract_enzymes_from_gene_rule(gene_rule::Vector{Vector{String}})

Extract enzyme names from gene association DNF format.
Each inner vector represents an AND relationship, outer vector represents OR.
"""
function extract_enzymes_from_gene_rule(gene_rule::Vector{Vector{String}})
    enzymes = String[]

    for gene_group in gene_rule
        if length(gene_group) == 1
            # Single gene = single enzyme
            push!(enzymes, gene_group[1])
        else
            # Multiple genes = enzyme complex
            enzyme_complex = join(sort(gene_group), "&")
            push!(enzymes, enzyme_complex)
        end
    end

    return unique(enzymes)
end

"""
    get_reaction_compartment(rxn::Reaction, model::StandardModel)

Determine the compartment of a reaction based on its metabolites.
"""
function get_reaction_compartment(rxn::Reaction, model::StandardModel)
    compartments = String[]

    for (mid, _) in rxn.stoichiometry
        if haskey(model.metabolites, mid)
            met = model.metabolites[mid]
            if !isnothing(met.compartment)
                push!(compartments, met.compartment)
            end
        end
    end

    # Return most common compartment, default to cytosol
    isempty(compartments) ? "c" : mode(compartments)
end

# Helper function to find mode (most common element)
function mode(arr::Vector{String})
    counts = Dict{String,Int}()
    for x in arr
        counts[x] = get(counts, x, 0) + 1
    end
    return argmax(counts)
end

"""
    get_or_create_enzyme!(registry, model, enzyme_name, compartment)

Get existing enzyme or create new one.
"""
function get_or_create_enzyme!(
    registry::Dict{String,EnzymeMetabolite},
    model::StandardModel,
    enzyme_name::String,
    compartment::String
)
    if haskey(registry, enzyme_name)
        return registry[enzyme_name]
    end

    # Create enzyme ID
    enzyme_id = "ENZ_" * replace(enzyme_name, "&" => "_")

    # Create enzyme metabolite
    enzyme_met = EnzymeMetabolite(
        enzyme_id,
        "Enzyme: $enzyme_name",
        enzyme_name,
        compartment
    )

    # Add to model as regular metabolite
    model.metabolites[enzyme_id] = Metabolite(
        name=enzyme_met.name,
        compartment=compartment,
        annotations=Dict("enzyme" => enzyme_name)
    )

    # Register
    registry[enzyme_name] = enzyme_met

    return enzyme_met
end

"""
    fixed_binding_mechanism(rid, rxn, enzyme_met, substrates, products, 
                           model, intermediate_registry, compartment, rng)

Implement ordered (fixed) binding mechanism.
"""
function fixed_binding_mechanism(
    rid::String,
    rxn::Reaction,
    enzyme_met::EnzymeMetabolite,
    substrates::Vector{Tuple{String,Float64}},
    products::Vector{Tuple{String,Float64}},
    model::StandardModel,
    intermediate_registry::Dict{String,IntermediateMetabolite},
    compartment::String,
    rng::AbstractRNG
)
    elementary_rxns = Dict{String,Reaction}()

    # Shuffle substrates and products for random but fixed order
    substrates_ordered = shuffle(rng, substrates)
    products_ordered = shuffle(rng, products)

    # Track current bound metabolites
    current_bound = Tuple{String,Float64}[]
    prev_intermediate_id = nothing

    # 1. Substrate binding steps
    for (idx, (sub_id, coef)) in enumerate(substrates_ordered)
        rxn_id = "$(rid)_$(enzyme_met.id)_bind_$(idx)"
        elem_rxn = Reaction(
            name="$(rxn.name): $(enzyme_met.enzyme_name) binding $sub_id",
            lower_bound=-1000.0,  # Reversible for binding
            upper_bound=1000.0,
            gene_association_dnf=rxn.gene_association_dnf
        )

        # Add substrate as reactant
        elem_rxn.stoichiometry[sub_id] = -abs(coef)

        if idx == 1
            # First substrate binds to free enzyme
            elem_rxn.stoichiometry[enzyme_met.id] = -1.0
        else
            # Subsequent substrates bind to previous intermediate
            elem_rxn.stoichiometry[prev_intermediate_id] = -1.0
        end

        # Create product intermediate
        push!(current_bound, (sub_id, abs(coef)))
        intermediate_id = create_intermediate_id(
            enzyme_met, current_bound, :substrate
        )

        # Get or create intermediate
        intermediate = get_or_create_intermediate!(
            intermediate_registry, model,
            intermediate_id, enzyme_met, current_bound,
            :substrate, compartment
        )

        # Add intermediate as product
        elem_rxn.stoichiometry[intermediate_id] = 1.0

        elementary_rxns[rxn_id] = elem_rxn
        prev_intermediate_id = intermediate_id
    end

    # 2. Catalytic step
    cat_rxn_id = "$(rid)_$(enzyme_met.id)_catalysis"
    cat_rxn = Reaction(
        name="$(rxn.name): $(enzyme_met.enzyme_name) catalysis",
        lower_bound=rxn.lower_bound ≥ 0 ? 0.0 : -1000.0,
        upper_bound=1000.0,
        gene_association_dnf=rxn.gene_association_dnf
    )

    # Substrate complex converts to product complex
    if !isnothing(prev_intermediate_id)
        cat_rxn.stoichiometry[prev_intermediate_id] = -1.0
    end

    # Create product intermediate
    if !isempty(products_ordered)
        product_intermediate_id = create_intermediate_id(
            enzyme_met, products_ordered, :product
        )

        get_or_create_intermediate!(
            intermediate_registry, model,
            product_intermediate_id, enzyme_met, products_ordered,
            :product, compartment
        )

        cat_rxn.stoichiometry[product_intermediate_id] = 1.0
        prev_intermediate_id = product_intermediate_id
    else
        # No products - release enzyme directly
        cat_rxn.stoichiometry[enzyme_met.id] = 1.0
    end

    elementary_rxns[cat_rxn_id] = cat_rxn

    # 3. Product release steps
    remaining_products = copy(products_ordered)

    for (idx, (prod_id, coef)) in enumerate(products_ordered)
        rxn_id = "$(rid)_$(enzyme_met.id)_release_$(idx)"
        elem_rxn = Reaction(
            name="$(rxn.name): $(enzyme_met.enzyme_name) releasing $prod_id",
            lower_bound=-1000.0,  # Reversible for release
            upper_bound=1000.0,
            gene_association_dnf=rxn.gene_association_dnf
        )

        # Current intermediate as reactant
        elem_rxn.stoichiometry[prev_intermediate_id] = -1.0

        # Release product
        elem_rxn.stoichiometry[prod_id] = abs(coef)

        # Remove this product from remaining
        filter!(p -> p[1] != prod_id, remaining_products)

        if !isempty(remaining_products)
            # Create intermediate with remaining products
            next_intermediate_id = create_intermediate_id(
                enzyme_met, remaining_products, :product
            )

            get_or_create_intermediate!(
                intermediate_registry, model,
                next_intermediate_id, enzyme_met, remaining_products,
                :product, compartment
            )

            elem_rxn.stoichiometry[next_intermediate_id] = 1.0
            prev_intermediate_id = next_intermediate_id
        else
            # Last product - release enzyme
            elem_rxn.stoichiometry[enzyme_met.id] = 1.0
        end

        elementary_rxns[rxn_id] = elem_rxn
    end

    return elementary_rxns
end

"""
    random_binding_mechanism(rid, rxn, enzyme_met, substrates, products,
                            model, intermediate_registry, compartment, rng)

Implement random binding mechanism with all possible binding orders.
"""
function random_binding_mechanism(
    rid::String,
    rxn::Reaction,
    enzyme_met::EnzymeMetabolite,
    substrates::Vector{Tuple{String,Float64}},
    products::Vector{Tuple{String,Float64}},
    model::StandardModel,
    intermediate_registry::Dict{String,IntermediateMetabolite},
    compartment::String,
    rng::AbstractRNG
)
    elementary_rxns = Dict{String,Reaction}()

    # For random binding, we need to generate all possible binding orders
    # This can lead to combinatorial explosion, so we limit it

    if length(substrates) > 3
        @warn "Random binding with >3 substrates can create many reactions. Consider using fixed binding."
    end

    # Generate all permutations of substrate binding orders
    substrate_orders = collect(permutations(1:length(substrates)))

    # Track all final substrate complexes
    final_substrate_complexes = String[]

    # Process each binding order
    for (order_idx, order) in enumerate(substrate_orders)
        ordered_substrates = substrates[order]

        # Create binding steps for this order
        current_bound = Tuple{String,Float64}[]
        prev_intermediate_id = nothing

        for (step_idx, (sub_id, coef)) in enumerate(ordered_substrates)
            rxn_id = "$(rid)_$(enzyme_met.id)_bind_ord$(order_idx)_s$(step_idx)"

            # Skip if this reaction already exists
            if haskey(elementary_rxns, rxn_id)
                continue
            end

            elem_rxn = Reaction(
                name="$(rxn.name): $(enzyme_met.enzyme_name) binding $sub_id (order $order_idx)",
                lower_bound=-1000.0,
                upper_bound=1000.0,
                gene_association_dnf=rxn.gene_association_dnf
            )

            # Add substrate
            elem_rxn.stoichiometry[sub_id] = -abs(coef)

            if step_idx == 1
                # First substrate binds to enzyme
                elem_rxn.stoichiometry[enzyme_met.id] = -1.0
            else
                # Bind to previous intermediate
                elem_rxn.stoichiometry[prev_intermediate_id] = -1.0
            end

            # Create new intermediate
            push!(current_bound, (sub_id, abs(coef)))
            intermediate_id = create_intermediate_id(
                enzyme_met, current_bound, :substrate
            )

            get_or_create_intermediate!(
                intermediate_registry, model,
                intermediate_id, enzyme_met, current_bound,
                :substrate, compartment
            )

            elem_rxn.stoichiometry[intermediate_id] = 1.0
            elementary_rxns[rxn_id] = elem_rxn
            prev_intermediate_id = intermediate_id
        end

        # Track the final complex for this order
        push!(final_substrate_complexes, prev_intermediate_id)

        # Create catalytic step for this binding order
        cat_rxn_id = "$(rid)_$(enzyme_met.id)_catalysis_ord$(order_idx)"
        cat_rxn = Reaction(
            name="$(rxn.name): $(enzyme_met.enzyme_name) catalysis (order $order_idx)",
            lower_bound=rxn.lower_bound ≥ 0 ? 0.0 : -1000.0,
            upper_bound=1000.0,
            gene_association_dnf=rxn.gene_association_dnf
        )

        cat_rxn.stoichiometry[prev_intermediate_id] = -1.0

        # All catalytic steps produce the same product complex
        product_intermediate_id = create_intermediate_id(
            enzyme_met, products, :product
        )

        get_or_create_intermediate!(
            intermediate_registry, model,
            product_intermediate_id, enzyme_met, products,
            :product, compartment
        )

        cat_rxn.stoichiometry[product_intermediate_id] = 1.0
        elementary_rxns[cat_rxn_id] = cat_rxn
    end

    # Product release steps (same for all binding orders)
    # Only create one set of release steps to avoid explosion
    prev_intermediate_id = create_intermediate_id(enzyme_met, products, :product)
    remaining_products = copy(products)

    for (idx, (prod_id, coef)) in enumerate(products)
        rxn_id = "$(rid)_$(enzyme_met.id)_release_$(idx)"

        if !haskey(elementary_rxns, rxn_id)
            elem_rxn = Reaction(
                name="$(rxn.name): $(enzyme_met.enzyme_name) releasing $prod_id",
                lower_bound=-1000.0,
                upper_bound=1000.0,
                gene_association_dnf=rxn.gene_association_dnf
            )

            elem_rxn.stoichiometry[prev_intermediate_id] = -1.0
            elem_rxn.stoichiometry[prod_id] = abs(coef)

            filter!(p -> p[1] != prod_id, remaining_products)

            if !isempty(remaining_products)
                next_intermediate_id = create_intermediate_id(
                    enzyme_met, remaining_products, :product
                )

                get_or_create_intermediate!(
                    intermediate_registry, model,
                    next_intermediate_id, enzyme_met, remaining_products,
                    :product, compartment
                )

                elem_rxn.stoichiometry[next_intermediate_id] = 1.0
                prev_intermediate_id = next_intermediate_id
            else
                elem_rxn.stoichiometry[enzyme_met.id] = 1.0
            end

            elementary_rxns[rxn_id] = elem_rxn
        end
    end

    return elementary_rxns
end

"""
    create_intermediate_id(enzyme_met, metabolites, intermediate_type)

Create a deterministic ID for an enzyme-metabolite intermediate.
"""
function create_intermediate_id(
    enzyme_met::EnzymeMetabolite,
    metabolites::Vector{Tuple{String,Float64}},
    intermediate_type::Symbol
)
    # Sort metabolites for deterministic ID
    sorted_mets = sort(metabolites, by=x -> x[1])

    # Create ID components
    met_str = join(["$(m)_$(c)" for (m, c) in sorted_mets], "_")
    type_str = intermediate_type == :substrate ? "SUB" : "PROD"

    # Use hash for shorter ID
    met_hash = string(hash(met_str), base=16)[1:8]

    return "$(enzyme_met.id)_$(type_str)_$(met_hash)"
end

"""
    get_or_create_intermediate!(registry, model, intermediate_id, 
                               enzyme_met, metabolites, intermediate_type, compartment)

Get existing intermediate or create new one.
"""
function get_or_create_intermediate!(
    registry::Dict{String,IntermediateMetabolite},
    model::StandardModel,
    intermediate_id::String,
    enzyme_met::EnzymeMetabolite,
    metabolites::Vector{Tuple{String,Float64}},
    intermediate_type::Symbol,
    compartment::String
)
    if haskey(registry, intermediate_id)
        return registry[intermediate_id]
    end

    # Create intermediate
    met_names = [m for (m, _) in metabolites]
    coeffs = [c for (_, c) in metabolites]

    intermediate = IntermediateMetabolite(
        intermediate_id,
        "Intermediate: $(enzyme_met.enzyme_name) with $(join(met_names, ", "))",
        enzyme_met.id,
        met_names,
        coeffs,
        intermediate_type,
        compartment
    )

    # Add to model as regular metabolite
    model.metabolites[intermediate_id] = Metabolite(
        name=intermediate.name,
        compartment=compartment,
        annotations=Dict(
            "intermediate_type" => String(intermediate_type),
            "enzyme" => enzyme_met.id,
            "metabolites" => join(met_names, ";")
        )
    )

    # Register
    registry[intermediate_id] = intermediate

    return intermediate
end

"""
    update_objective_for_elementary!(elementary_model, original_model)

Update the objective function to work with elementary reactions.
"""
function update_objective_for_elementary!(
    elementary_model::StandardModel,
    original_model::StandardModel
)
    new_objective = Dict{String,Float64}()

    for (rid, coef) in original_model.objective
        # Find catalytic reactions for this original reaction
        for (elem_rid, elem_rxn) in elementary_model.reactions
            if occursin("$(rid)_", elem_rid) && occursin("_catalysis", elem_rid)
                new_objective[elem_rid] = coef
            end
        end
    end

    elementary_model.objective = new_objective
end

end # module