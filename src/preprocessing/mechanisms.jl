"""
Binding mechanism implementations for elementary step splitting.

This module handles:
- Ordered binding mechanism implementation
- Random binding mechanism implementation
- Enzyme-substrate interaction modeling
"""

using Random
import AbstractFBCModels.CanonicalModel as CM

"""
Add elementary reactions for ordered binding mechanism.
"""
function add_ordered_reactions!(
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
    # MATLAB behavior: Copy objective to ALL elementary steps (not distribute)
    objective_coefficient = original_rxn.objective_coefficient

    # Get global bounds for reversible steps (matching MATLAB logic)
    # Since bounds are already normalized, this will typically be -1000 and 1000
    reversible_lb = -1000.0  # Could use min(-1000, minimum_model_lb) if needed
    reversible_ub = 1000.0   # Could use max(1000, maximum_model_ub) if needed

    # Randomize order
    substrates_ordered = shuffle(rng, substrates)
    products_ordered = shuffle(rng, products)

    current_intermediate_metabolites = String[]

    # 1. Substrate binding steps - ALWAYS REVERSIBLE
    for (idx, (met_id, coeff)) in enumerate(substrates_ordered)
        reaction_count += 1
        # Create shorter reaction ID: original_rid_enzyme_number_step_type_step_number
        enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
        rxn_id = "$(original_rid)_$(enzyme_num)_S$(idx)"

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): substrate binding step $idx",
            lower_bound=reversible_lb,  # Always -1000
            upper_bound=reversible_ub,  # Always 1000
            gene_association_dnf=original_rxn.gene_association_dnf,
            objective_coefficient=objective_coefficient  # Copy original objective (MATLAB behavior)
        )

        # Add stoichiometry (same as before)
        elem_rxn.stoichiometry[met_id] = -abs(coeff)

        if idx == 1
            elem_rxn.stoichiometry[enzyme_id] = -1.0
        else
            prev_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_intermediate_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[prev_intermediate] = -1.0
        end

        push!(current_intermediate_metabolites, met_id)
        new_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            current_intermediate_metabolites, enzyme_id
        )
        elem_rxn.stoichiometry[new_intermediate] = 1.0

        elementary_model.reactions[rxn_id] = elem_rxn
    end

    # 2. Catalytic step - SPECIAL HANDLING
    reaction_count += 1
    # Create shorter reaction ID for catalytic step
    enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
    cat_rxn_id = "$(original_rid)_$(enzyme_num)_CAT"

    # Match MATLAB logic for catalytic step bounds
    cat_lower = if original_rxn.lower_bound >= 0
        0.0  # Keep irreversible if original was forward-only
    else
        reversible_lb  # Otherwise make reversible
    end

    cat_rxn = CM.Reaction(
        name="$(original_rxn.name): catalysis",
        lower_bound=cat_lower,
        upper_bound=reversible_ub,  # Always use max bound
        gene_association_dnf=original_rxn.gene_association_dnf,
        objective_coefficient=objective_coefficient  # Copy original objective (MATLAB behavior)
    )

    # Add stoichiometry (same as before)
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
        cat_rxn.stoichiometry[enzyme_id] = 1.0
        current_intermediate_metabolites = String[]
    end

    elementary_model.reactions[cat_rxn_id] = cat_rxn

    # 3. Product release steps - ALWAYS REVERSIBLE
    for (idx, (met_id, coeff)) in enumerate(products_ordered)
        reaction_count += 1
        # Create shorter reaction ID for product steps
        enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
        rxn_id = "$(original_rid)_$(enzyme_num)_P$(idx)"

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): product release step $idx",
            lower_bound=reversible_lb,  # Always -1000
            upper_bound=reversible_ub,  # Always 1000
            gene_association_dnf=original_rxn.gene_association_dnf,
            objective_coefficient=objective_coefficient  # Copy original objective (MATLAB behavior)
        )

        # Add stoichiometry (same as before)
        current_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            current_intermediate_metabolites, enzyme_id
        )
        elem_rxn.stoichiometry[current_intermediate] = -1.0
        elem_rxn.stoichiometry[met_id] = abs(coeff)

        filter!(m -> m != met_id, current_intermediate_metabolites)

        if !isempty(current_intermediate_metabolites)
            new_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_intermediate_metabolites, enzyme_id
            )
            elem_rxn.stoichiometry[new_intermediate] = 1.0
        else
            elem_rxn.stoichiometry[enzyme_id] = 1.0
        end

        elementary_model.reactions[rxn_id] = elem_rxn
    end
end

"""
Add elementary reactions for random binding mechanism.

Instead of generating all permutations, we sample a limited number
of binding orders to keep the model tractable.
"""
function add_random_reactions!(
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
    # MATLAB behavior: Copy objective to ALL elementary steps (not distribute)
    objective_coefficient = original_rxn.objective_coefficient

    # ALL steps in random mechanism are fully reversible (matching MATLAB)
    reversible_lb = -1000.0
    reversible_ub = 1000.0

    n_substrates = length(substrates)
    n_products = length(products)    # Determine number of orders to generate
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
            # Create shorter reaction ID for random binding
            enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
            rxn_id = "$(original_rid)_$(enzyme_num)_ORD$(order_idx)_L$(level)"

            elem_rxn = CM.Reaction(
                name="$(original_rxn.name): random binding order $order_idx level $level",
                lower_bound=reversible_lb,  # Always -1000
                upper_bound=reversible_ub,  # Always 1000
                gene_association_dnf=original_rxn.gene_association_dnf,
                objective_coefficient=objective_coefficient  # Copy original objective (MATLAB behavior)
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
    # Create shorter reaction ID for catalytic step
    enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
    cat_rxn_id = "$(original_rid)_$(enzyme_num)_CAT"

    # Match MATLAB logic for catalytic step bounds (same as ordered mechanism)
    cat_lower = if original_rxn.lower_bound >= 0
        0.0  # Keep irreversible if original was forward-only
    else
        reversible_lb  # Otherwise make reversible
    end

    # Distribute objective across all catalytic pathways (MATLAB behavior)
    distributed_catalytic_objective = objective_coefficient / length(final_substrate_intermediates)

    # All substrate intermediates can undergo catalysis
    for substrate_intermediate in final_substrate_intermediates
        # Create separate catalytic reaction for each
        cat_rxn_specific_id = "$(cat_rxn_id)_$(substrate_intermediate)"
        cat_rxn_specific = CM.Reaction(
            name="$(original_rxn.name): catalysis",
            lower_bound=cat_lower,
            upper_bound=reversible_ub,
            gene_association_dnf=original_rxn.gene_association_dnf,
            objective_coefficient=distributed_catalytic_objective  # Distribute objective across catalytic steps
        )
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
            # Create shorter reaction ID for product steps
            enzyme_num = replace(enzyme_id, "E" => "")  # Extract number from E123 -> 123
            rxn_id = "$(original_rid)_$(enzyme_num)_P$(idx)"

            elem_rxn = CM.Reaction(
                name="$(original_rxn.name): product release step $idx",
                lower_bound=reversible_lb,  # Always -1000
                upper_bound=reversible_ub,  # Always 1000
                gene_association_dnf=original_rxn.gene_association_dnf,
                objective_coefficient=objective_coefficient  # Copy original objective (MATLAB behavior)
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

"""
Simple implementation of permutations for small arrays.
"""
function permutations(a::AbstractVector)
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