"""
Binding mechanism implementations for elementary step splitting.

This module handles:
- Ordered binding mechanism implementation
- Random binding mechanism implementation
- Enzyme-substrate intermediate management
- Enzyme-substrate interaction modeling
"""

import Combinatorics

"""
    get_reaction_compartment(model, reaction)

Determine the compartment where a reaction occurs based on its metabolites.
Returns the most common compartment among the reaction's metabolites.

# Arguments
- `model::CM.Model`: The metabolic model
- `reaction::CM.Reaction`: The reaction to analyze

# Returns
- `String`: The reaction's primary compartment (default: "c")
"""
function get_reaction_compartment(model::CM.Model, reaction::CM.Reaction)
    compartment_counts = Dict{String,Int}()

    for (met_id, _) in reaction.stoichiometry
        if haskey(model.metabolites, met_id)
            met = model.metabolites[met_id]
            if !isnothing(met.compartment)
                compartment_counts[met.compartment] = get(compartment_counts, met.compartment, 0) + 1
            end
        end
    end

    if isempty(compartment_counts)
        return "c"  # Default to cytosol
    end

    # Return the most common compartment
    return sort(collect(compartment_counts), by=x -> (-x[2], x[1]))[1][1]
end

"""
    get_or_create_intermediate!(elementary_model, intermediate_registry, metabolites, enzyme_id, reaction_compartment)

Get existing intermediate or create new one.

Uses the reaction's compartment for the complex (where catalysis occurs).

# Arguments
- `elementary_model::CM.Model`: Model to add intermediate to
- `intermediate_registry::Dict{Vector{String},String}`: Registry mapping metabolite sets to intermediate IDs
- `metabolites::Vector{String}`: Metabolite IDs in this complex
- `enzyme_id::String`: Enzyme metabolite ID
- `reaction_compartment::String`: Compartment where the reaction occurs

# Returns
- `String`: Intermediate metabolite ID
"""
function get_or_create_intermediate!(
    elementary_model::CM.Model,
    intermediate_registry::Dict{Vector{String},String},
    metabolites::Vector{String},
    enzyme_id::String,
    reaction_compartment::String="c"
)
    # Create canonical representation (sorted for consistent lookup)
    intermediate_key = sort([metabolites; enzyme_id])

    if haskey(intermediate_registry, intermediate_key)
        return intermediate_registry[intermediate_key]
    end

    # Use the reaction's compartment (where catalysis occurs)
    # This matches the biological reality: enzyme-substrate complexes form
    # in the compartment where the enzyme catalyzes the reaction
    compartment = reaction_compartment

    # Track which compartments the bound metabolites come from
    metabolite_compartments = String[]
    for met_id in metabolites
        met = get(elementary_model.metabolites, met_id, nothing)
        if !isnothing(met) && !isnothing(met.compartment)
            push!(metabolite_compartments, met.compartment)
        end
    end
    unique_met_compartments = sort(unique(metabolite_compartments))

    # Build systematic identifier following BiGG/MetaNetX conventions
    # Format: CPLX_<enzyme>_<sorted_metabolites>_<compartment>
    # CPLX = Complex (enzyme-substrate intermediate)
    # Extract enzyme number (e.g., "E3" -> "3")
    enz_num = replace(enzyme_id, "E" => "")

    # Get clean metabolite names (sorted for consistency)
    met_names = sort(metabolites)

    # Create intermediate ID with systematic naming
    if isempty(met_names)
        # Enzyme-only intermediate (rare edge case)
        intermediate_id = "CPLX_E$(enz_num)_$(compartment)"
    else
        # Standard enzyme-substrate complex
        # Join with double underscore to separate logical groups
        intermediate_id = "CPLX_E$(enz_num)__" * join(met_names, "__") * "_$(compartment)"
    end

    intermediate_registry[intermediate_key] = intermediate_id

    # Build human-readable name for display
    clean_mets = [replace(replace(m, r"^M_" => ""), r"_[a-z]$" => "") for m in sort(metabolites)]
    clean_enzyme = "E$(enz_num)"

    if isempty(metabolites)
        complex_name = clean_enzyme
    else
        complex_name = join([clean_enzyme; clean_mets], ":")
    end

    # Aggregate formula and charge from bound metabolites (if available)
    aggregate_formula = Dict{String,Int}()
    total_charge = 0
    has_charge_info = true
    all_metabolite_names = String[]

    for met_id in sort(metabolites)
        if haskey(elementary_model.metabolites, met_id)
            met = elementary_model.metabolites[met_id]

            # Aggregate chemical formula
            if !isnothing(met.formula) && met.formula isa Dict
                for (element, count) in met.formula
                    aggregate_formula[element] = get(aggregate_formula, element, 0) + count
                end
            end

            # Sum charges (only if all metabolites have charge info)
            if !isnothing(met.charge)
                total_charge += met.charge
            else
                has_charge_info = false
            end

            # Collect metabolite names for human-readable description
            if !isnothing(met.name) && !isempty(met.name)
                push!(all_metabolite_names, met.name)
            end
        else
            has_charge_info = false
        end
    end

    # Set formula and charge based on what we could aggregate
    final_formula = isempty(aggregate_formula) ? nothing : aggregate_formula
    final_charge = has_charge_info ? total_charge : nothing

    # Add to model with proper metadata and systematic annotations
    # Note: ALL metadata goes in annotations - SBML export rejects ANY notes (even single strings)
    elementary_model.metabolites[intermediate_id] = CM.Metabolite(
        name=complex_name,
        compartment=compartment,
        formula=final_formula,
        charge=final_charge,
        balance=0.0,  # Intermediates are balanced (not accumulated)
        annotations=Dict(
            "sbo" => ["SBO:0000297"],  # SBO:0000297 = protein-small molecule complex
            "complex_type" => ["enzyme_substrate"],
            "enzyme" => [enzyme_id],
            "bound_metabolites" => sort(metabolites),
            "systematic_name" => [intermediate_id],
            "complex_stoichiometry" => ["1 enzyme + " * string(length(metabolites)) * " metabolite(s)"],
            "metabolite_compartments" => unique_met_compartments,  # Track metabolite origins
            "reaction_compartment" => [compartment],  # Where catalysis occurs
            "description" => ["Enzyme-substrate intermediate complex in elementary reaction mechanism"],
            "components" => [enzyme_id; sort(metabolites)],  # One entry per component
            "component_names" => isempty(all_metabolite_names) ? ["Unknown metabolites"] : all_metabolite_names,
            "naming_convention" => ["BiGG-compatible systematic identifier with compartment suffix"],
            "formula_aggregation" => [isnothing(final_formula) ? "Formula information unavailable" : "Aggregated from bound metabolites"],
            "charge_aggregation" => [isnothing(final_charge) ? "Charge information unavailable" : "Sum of bound metabolite charges"],
            "compartment_logic" => [length(unique_met_compartments) <= 1 ?
                                    "All metabolites in same compartment ($(compartment))" :
                                    "Multi-compartment binding: reaction in $(compartment), metabolites from $(join(unique_met_compartments, ", "))"]
        )
    )

    return intermediate_id
end

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
    original_model::CM.Model  # Added to get reaction compartment
)
    # Determine reaction compartment
    reaction_compartment = get_reaction_compartment(original_model, original_rxn)
    # FIX: Only catalytic step should carry objective coefficient
    # Assigning objective to all steps artificially inflates the objective value
    objective_coefficient = original_rxn.objective_coefficient

    # Get global bounds for reversible steps (matching MATLAB logic)
    # Since bounds are already normalized, this will typically be -1000 and 1000
    reversible_lb = -1000.0  # Could use min(-1000, minimum_model_lb) if needed
    reversible_ub = 1000.0   # Could use max(1000, maximum_model_ub) if needed

    # Use deterministic order (matching MATLAB fixed mechanism)
    # MATLAB uses order from stoichiometric matrix (find(model.S(:,i)<0))
    substrates_ordered = substrates
    products_ordered = products

    current_intermediate_metabolites = String[]

    # 1. Substrate binding steps - ALWAYS REVERSIBLE
    for (idx, (met_id, coeff)) in enumerate(substrates_ordered)
        # Systematic reaction ID: <original>_E<enzyme>_SB<step>
        # SB = Substrate Binding
        enz_num = replace(enzyme_id, "E" => "")
        rxn_id = "$(original_rid)_E$(enz_num)_SB$(idx)"

        # Build annotations (all new metadata goes here for SBML compatibility)
        elem_annotations = deepcopy(original_rxn.annotations)
        elem_annotations["elementary_step_type"] = ["substrate_binding"]
        elem_annotations["elementary_mechanism"] = ["ordered"]  # Track mechanism type
        elem_annotations["original_reaction"] = [original_rid]
        elem_annotations["binding_order"] = [string(idx)]
        elem_annotations["substrate"] = [met_id]
        elem_annotations["elementary_step"] = ["Substrate binding reaction in ordered mechanism"]
        elem_annotations["bound_metabolite"] = [met_id]

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): substrate binding step $idx",
            lower_bound=reversible_lb,  # Always -1000
            upper_bound=reversible_ub,  # Always 1000
            gene_association_dnf=original_rxn.gene_association_dnf,
            objective_coefficient=0.0,  # Only catalytic step has objective
            annotations=elem_annotations,
            notes=original_rxn.notes  # Keep original notes unchanged
        )

        # Add stoichiometry (same as before)
        elem_rxn.stoichiometry[met_id] = -abs(coeff)

        if idx == 1
            elem_rxn.stoichiometry[enzyme_id] = -1.0
        else
            prev_intermediate = get_or_create_intermediate!(
                elementary_model, intermediate_registry,
                current_intermediate_metabolites, enzyme_id, reaction_compartment
            )
            elem_rxn.stoichiometry[prev_intermediate] = -1.0
        end

        push!(current_intermediate_metabolites, met_id)
        new_intermediate = get_or_create_intermediate!(
            elementary_model, intermediate_registry,
            current_intermediate_metabolites, enzyme_id, reaction_compartment
        )
        elem_rxn.stoichiometry[new_intermediate] = 1.0

        elementary_model.reactions[rxn_id] = elem_rxn
    end

    # 2. Catalytic step - SPECIAL HANDLING
    # Systematic reaction ID: <original>_E<enzyme>_CAT (catalysis)
    enz_num = replace(enzyme_id, "E" => "")
    cat_rxn_id = "$(original_rid)_E$(enz_num)_CAT"

    # Match MATLAB logic for catalytic step bounds
    cat_lower = if original_rxn.lower_bound >= 0
        0.0  # Keep irreversible if original was forward-only
    else
        reversible_lb  # Otherwise make reversible
    end

    # Build annotations for catalytic step (all new metadata goes here)
    cat_annotations = deepcopy(original_rxn.annotations)
    cat_annotations["elementary_step_type"] = ["catalysis"]
    cat_annotations["elementary_mechanism"] = ["ordered"]  # Track mechanism type
    cat_annotations["original_reaction"] = [original_rid]
    cat_annotations["enzyme"] = [enzyme_id]
    cat_annotations["elementary_step"] = ["Catalytic transformation step"]

    cat_rxn = CM.Reaction(
        name="$(original_rxn.name): catalysis",
        lower_bound=cat_lower,
        upper_bound=reversible_ub,  # Always use max bound
        gene_association_dnf=original_rxn.gene_association_dnf,
        objective_coefficient=objective_coefficient,  # Copy original objective (MATLAB behavior)
        annotations=cat_annotations,
        notes=original_rxn.notes  # Keep original notes unchanged
    )

    # Add stoichiometry - matching MATLAB behavior for exchange reactions
    # MATLAB line 129: Use enzyme directly when no substrates
    if !isempty(current_intermediate_metabolites)
        substrate_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, current_intermediate_metabolites, enzyme_id, reaction_compartment)
        cat_rxn.stoichiometry[substrate_intermediate] = -1.0
    else
        # No substrates - consume enzyme directly (MATLAB line 129)
        cat_rxn.stoichiometry[enzyme_id] = -1.0
    end

    if !isempty(products_ordered)
        product_metabolites = [p[1] for p in products_ordered]
        product_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, product_metabolites, enzyme_id, reaction_compartment)
        cat_rxn.stoichiometry[product_intermediate] = 1.0
        current_intermediate_metabolites = product_metabolites
    else
        cat_rxn.stoichiometry[enzyme_id] = 1.0
        current_intermediate_metabolites = String[]
    end

    elementary_model.reactions[cat_rxn_id] = cat_rxn

    # 3. Product release steps - ALWAYS REVERSIBLE
    for (idx, (met_id, coeff)) in enumerate(products_ordered)
        # Systematic reaction ID: <original>_E<enzyme>_PR<step>
        # PR = Product Release
        enz_num = replace(enzyme_id, "E" => "")
        rxn_id = "$(original_rid)_E$(enz_num)_PR$(idx)"

        # Build annotations for product release (all new metadata goes here)
        prod_annotations = deepcopy(original_rxn.annotations)
        prod_annotations["elementary_step_type"] = ["product_release"]
        prod_annotations["elementary_mechanism"] = ["ordered"]  # Track mechanism type
        prod_annotations["original_reaction"] = [original_rid]
        prod_annotations["release_order"] = [string(idx)]
        prod_annotations["product"] = [met_id]
        prod_annotations["elementary_step"] = ["Product release reaction in ordered mechanism"]
        prod_annotations["released_metabolite"] = [met_id]

        elem_rxn = CM.Reaction(
            name="$(original_rxn.name): product release step $idx",
            lower_bound=reversible_lb,  # Always -1000
            upper_bound=reversible_ub,  # Always 1000
            gene_association_dnf=original_rxn.gene_association_dnf,
            objective_coefficient=0.0,  # Only catalytic step has objective
            annotations=prod_annotations,
            notes=original_rxn.notes  # Keep original notes unchanged
        )

        # Add stoichiometry (same as before)
        current_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, current_intermediate_metabolites, enzyme_id, reaction_compartment)
        elem_rxn.stoichiometry[current_intermediate] = -1.0
        elem_rxn.stoichiometry[met_id] = abs(coeff)

        filter!(m -> m != met_id, current_intermediate_metabolites)

        if !isempty(current_intermediate_metabolites)
            new_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, current_intermediate_metabolites, enzyme_id, reaction_compartment)
            elem_rxn.stoichiometry[new_intermediate] = 1.0
        else
            elem_rxn.stoichiometry[enzyme_id] = 1.0
        end

        elementary_model.reactions[rxn_id] = elem_rxn
    end
end

"""
Add elementary reactions for random binding mechanism.

Follows MATLAB nchoosek logic: generates ALL possible binding combinations
at each level (not sampling). For n substrates with k bound at each level,
creates C(n,k) reactions per level.

This is deterministic - generates all C(n,k) combinations at each binding level.
"""
function add_random_reactions!(
    elementary_model::CM.Model,
    original_rid::String,
    original_rxn::CM.Reaction,
    enzyme_id::String,
    substrates::Vector{Tuple{String,Float64}},
    products::Vector{Tuple{String,Float64}},
    intermediate_registry::Dict{Vector{String},String},
    original_model::CM.Model  # Added to get reaction compartment
)
    # Determine reaction compartment
    reaction_compartment = get_reaction_compartment(original_model, original_rxn)
    # FIX: Only catalytic step should carry objective coefficient
    # Assigning objective to all steps artificially inflates the objective value
    objective_coefficient = original_rxn.objective_coefficient

    # ALL steps in random mechanism are fully reversible (matching MATLAB)
    reversible_lb = -1000.0
    reversible_ub = 1000.0

    n_substrates = length(substrates)
    n_products = length(products)

    # Track complexes at each level (MATLAB: complex_formed_old)
    # Each element is a set of substrate indices bound at that level
    current_level_complexes = Vector{Vector{Int}}[]

    # Substrate binding - level by level using nchoosek
    for level in 1:n_substrates
        if level == 1
            # Level 1: All single substrates (MATLAB line 228)
            # nchoosek(1:n_substrates, 1) gives [[1], [2], ..., [n]]
            substrate_combos = collect(Combinatorics.combinations(1:n_substrates, 1))

            for combo in substrate_combos
                sub_idx = combo[1]
                met_id, coeff = substrates[sub_idx]

                # Systematic reaction ID: <original>_E<enzyme>_SBR<level>_<substrate_index>
                # SBR = Substrate Binding Random mechanism
                enz_num = replace(enzyme_id, "E" => "")
                rxn_id = "$(original_rid)_E$(enz_num)_SBR1_S$(sub_idx)"

                # Build annotations for random binding (level 1, all new metadata goes here)
                rand_annotations = deepcopy(original_rxn.annotations)
                rand_annotations["elementary_step_type"] = ["substrate_binding"]
                rand_annotations["elementary_mechanism"] = ["random"]  # Track mechanism type
                rand_annotations["original_reaction"] = [original_rid]
                rand_annotations["binding_level"] = ["1"]
                rand_annotations["substrate"] = [met_id]
                rand_annotations["elementary_step"] = ["Substrate binding reaction in random mechanism"]
                rand_annotations["bound_metabolite"] = [met_id]

                elem_rxn = CM.Reaction(
                    name="$(original_rxn.name): substrate binding step $sub_idx",
                    lower_bound=reversible_lb,
                    upper_bound=reversible_ub,
                    gene_association_dnf=original_rxn.gene_association_dnf,
                    objective_coefficient=0.0,  # Only catalytic step has objective
                    annotations=rand_annotations,
                    notes=original_rxn.notes  # Keep original notes unchanged
                )

                # Stoichiometry
                elem_rxn.stoichiometry[met_id] = -abs(coeff)
                elem_rxn.stoichiometry[enzyme_id] = -1.0

                # Create intermediate
                metabolites_in_complex = [met_id]
                new_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, metabolites_in_complex, enzyme_id, reaction_compartment)
                elem_rxn.stoichiometry[new_intermediate] = 1.0

                elementary_model.reactions[rxn_id] = elem_rxn
            end

            current_level_complexes = [[i] for i in 1:n_substrates]
        else
            # Level > 1: Expand each previous complex by adding one more substrate
            # (MATLAB lines 277-330)
            next_level_complexes = Vector{Int}[]

            for prev_complex in current_level_complexes
                # Find substrates not yet in this complex (MATLAB: setdiff)
                available_substrates = setdiff(1:n_substrates, prev_complex)

                # Add each available substrate to form new complexes
                for new_sub_idx in available_substrates
                    new_complex = sort([prev_complex; new_sub_idx])

                    # Check if we"ve already created this complex at this level
                    if !(new_complex in next_level_complexes)
                        push!(next_level_complexes, new_complex)

                        met_id, coeff = substrates[new_sub_idx]

                        # Systematic reaction ID for level > 1
                        enz_num = replace(enzyme_id, "E" => "")
                        rxn_id = "$(original_rid)_E$(enz_num)_SBR$(level)_S$(new_sub_idx)_from_" * join(string.(prev_complex), "-")

                        # Build annotations for random binding (level > 1, all new metadata goes here)
                        rand_annotations = deepcopy(original_rxn.annotations)
                        rand_annotations["elementary_step_type"] = ["substrate_binding"]
                        rand_annotations["elementary_mechanism"] = ["random"]  # Track mechanism type
                        rand_annotations["original_reaction"] = [original_rid]
                        rand_annotations["binding_level"] = [string(level)]
                        rand_annotations["substrate"] = [met_id]
                        rand_annotations["previous_complex"] = [join(prev_complex, ",")]
                        rand_annotations["elementary_step"] = ["Substrate binding reaction in random mechanism"]
                        rand_annotations["bound_metabolite"] = [met_id]

                        elem_rxn = CM.Reaction(
                            name="$(original_rxn.name): substrate binding step $new_sub_idx",
                            lower_bound=reversible_lb,
                            upper_bound=reversible_ub,
                            gene_association_dnf=original_rxn.gene_association_dnf,
                            objective_coefficient=0.0,  # Only catalytic step has objective
                            annotations=rand_annotations,
                            notes=original_rxn.notes  # Keep original notes unchanged
                        )

                        # Stoichiometry: consume new substrate and previous complex
                        elem_rxn.stoichiometry[met_id] = -abs(coeff)

                        # Get previous complex intermediate
                        prev_metabolites = [substrates[i][1] for i in prev_complex]
                        prev_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, prev_metabolites, enzyme_id, reaction_compartment)
                        elem_rxn.stoichiometry[prev_intermediate] = -1.0

                        # Create new complex intermediate
                        new_metabolites = [substrates[i][1] for i in new_complex]
                        new_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, new_metabolites, enzyme_id, reaction_compartment)
                        elem_rxn.stoichiometry[new_intermediate] = 1.0

                        elementary_model.reactions[rxn_id] = elem_rxn
                    end
                end
            end

            current_level_complexes = next_level_complexes
        end
    end

    # Catalytic step (MATLAB lines 334-382)
    # MATLAB insight: ALL binding paths converge to ONE unique final complex (E + all substrates)
    # After unique(), current_level_complexes should contain only [1,2,...,n_substrates]
    # So there"s only ONE catalytic reaction
    # Systematic reaction ID: <original>_E<enzyme>_CAT_RND (random mechanism)
    enz_num = replace(enzyme_id, "E" => "")
    cat_rxn_id = "$(original_rid)_E$(enz_num)_CAT_RND"

    cat_lower = if original_rxn.lower_bound >= 0
        0.0
    else
        reversible_lb
    end

    # Build annotations for catalytic step (random mechanism, all new metadata goes here)
    cat_annotations = deepcopy(original_rxn.annotations)
    cat_annotations["elementary_step_type"] = ["catalysis"]
    cat_annotations["elementary_mechanism"] = ["random"]  # Track mechanism type
    cat_annotations["original_reaction"] = [original_rid]
    cat_annotations["enzyme"] = [enzyme_id]
    cat_annotations["elementary_step"] = ["Catalytic transformation step in random mechanism"]

    cat_rxn = CM.Reaction(
        name="$(original_rxn.name): catalysis",
        lower_bound=cat_lower,
        upper_bound=reversible_ub,
        gene_association_dnf=original_rxn.gene_association_dnf,
        objective_coefficient=objective_coefficient,
        annotations=cat_annotations,
        notes=original_rxn.notes  # Keep original notes unchanged
    )

    # Consume the full substrate complex (E + all substrates)
    if !isempty(substrates)
        all_substrate_metabolites = [s[1] for s in substrates]
        substrate_complex = get_or_create_intermediate!(elementary_model, intermediate_registry, all_substrate_metabolites, enzyme_id, reaction_compartment)
        cat_rxn.stoichiometry[substrate_complex] = -1.0
    else
        # No substrates - consume enzyme directly
        cat_rxn.stoichiometry[enzyme_id] = -1.0
    end

    # Produce the full product complex (E + all products)
    if !isempty(products)
        all_product_metabolites = [p[1] for p in products]
        product_complex = get_or_create_intermediate!(elementary_model, intermediate_registry, all_product_metabolites, enzyme_id, reaction_compartment)
        cat_rxn.stoichiometry[product_complex] = 1.0
    else
        # No products - release enzyme directly
        cat_rxn.stoichiometry[enzyme_id] = 1.0
    end

    elementary_model.reactions[cat_rxn_id] = cat_rxn

    # Product release - level by level (matching MATLAB lines 385-522)
    # MATLAB logic: Start with full complex, release products one at a time
    # Track which complexes were created, then expand from those
    if !isempty(products)
        # Start with full product complex from catalytic step
        complex_formed_old = Vector{Vector{Int}}()

        for level in 1:n_products
            if level == 1
                # Level 1: Release one product from FULL complex (MATLAB line 388)
                # ls = products(nchoosek(1:length(products),1)) - all products

                new_complexes = Vector{Int}[]

                for prod_idx in 1:n_products
                    met_id, coeff = products[prod_idx]

                    # Systematic reaction ID: <original>_E<enzyme>_PRR<level>_P<product_index>
                    # PRR = Product Release Random mechanism
                    enz_num = replace(enzyme_id, "E" => "")
                    rxn_id = "$(original_rid)_E$(enz_num)_PRR1_P$(prod_idx)"

                    # Build annotations for product release (level 1, all new metadata goes here)
                    prod_annotations = deepcopy(original_rxn.annotations)
                    prod_annotations["elementary_step_type"] = ["product_release"]
                    prod_annotations["elementary_mechanism"] = ["random"]  # Track mechanism type
                    prod_annotations["original_reaction"] = [original_rid]
                    prod_annotations["release_level"] = ["1"]
                    prod_annotations["product"] = [met_id]
                    prod_annotations["elementary_step"] = ["Product release reaction in random mechanism"]
                    prod_annotations["released_metabolite"] = [met_id]

                    elem_rxn = CM.Reaction(
                        name="$(original_rxn.name): product release step $prod_idx",
                        lower_bound=reversible_lb,
                        upper_bound=reversible_ub,
                        gene_association_dnf=original_rxn.gene_association_dnf,
                        objective_coefficient=0.0,  # Only catalytic step has objective
                        annotations=prod_annotations,
                        notes=original_rxn.notes  # Keep original notes unchanged
                    )

                    # Consume full product complex
                    all_product_metabolites = [p[1] for p in products]
                    full_complex = get_or_create_intermediate!(elementary_model, intermediate_registry, all_product_metabolites, enzyme_id, reaction_compartment)
                    elem_rxn.stoichiometry[full_complex] = -1.0

                    # Release this product
                    elem_rxn.stoichiometry[met_id] = abs(coeff)

                    # Create intermediate with remaining products
                    remaining_indices = [i for i in 1:n_products if i != prod_idx]
                    if !isempty(remaining_indices)
                        remaining_metabolites = [products[i][1] for i in remaining_indices]
                        new_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, remaining_metabolites, enzyme_id, reaction_compartment)
                        elem_rxn.stoichiometry[new_intermediate] = 1.0

                        # Track this complex for next level
                        if !(sort(remaining_indices) in new_complexes)
                            push!(new_complexes, sort(remaining_indices))
                        end
                    else
                        elem_rxn.stoichiometry[enzyme_id] = 1.0
                    end

                    elementary_model.reactions[rxn_id] = elem_rxn
                end

                complex_formed_old = new_complexes
            else
                # Level > 1: For each complex from previous level, release one more product
                # (MATLAB lines 449-521)
                new_complexes = Vector{Int}[]

                for prev_complex in complex_formed_old
                    # Products still in this complex (MATLAB: isec = intersect(...))
                    products_in_complex = prev_complex

                    # Release each product that"s still in the complex
                    for prod_idx_to_release in products_in_complex
                        met_id, coeff = products[prod_idx_to_release]

                        # Systematic reaction ID for level > 1
                        enz_num = replace(enzyme_id, "E" => "")
                        rxn_id = "$(original_rid)_E$(enz_num)_PRR$(level)_P$(prod_idx_to_release)_from_" * join(string.(prev_complex), "-")

                        # Build annotations for product release (level > 1, all new metadata goes here)
                        prod_annotations = deepcopy(original_rxn.annotations)
                        prod_annotations["elementary_step_type"] = ["product_release"]
                        prod_annotations["elementary_mechanism"] = ["random"]  # Track mechanism type
                        prod_annotations["original_reaction"] = [original_rid]
                        prod_annotations["release_level"] = [string(level)]
                        prod_annotations["product"] = [met_id]
                        prod_annotations["previous_complex"] = [join(prev_complex, ",")]
                        prod_annotations["elementary_step"] = ["Product release reaction in random mechanism"]
                        prod_annotations["released_metabolite"] = [met_id]

                        elem_rxn = CM.Reaction(
                            name="$(original_rxn.name): product release step $prod_idx_to_release",
                            lower_bound=reversible_lb,
                            upper_bound=reversible_ub,
                            gene_association_dnf=original_rxn.gene_association_dnf,
                            objective_coefficient=0.0,  # Only catalytic step has objective
                            annotations=prod_annotations,
                            notes=original_rxn.notes  # Keep original notes unchanged
                        )

                        # Consume current complex
                        current_metabolites = [products[i][1] for i in prev_complex]
                        current_complex = get_or_create_intermediate!(elementary_model, intermediate_registry, current_metabolites, enzyme_id, reaction_compartment)
                        elem_rxn.stoichiometry[current_complex] = -1.0

                        # Release product
                        elem_rxn.stoichiometry[met_id] = abs(coeff)

                        # Create smaller complex or return enzyme
                        remaining_indices = [i for i in prev_complex if i != prod_idx_to_release]
                        if !isempty(remaining_indices)
                            remaining_metabolites = [products[i][1] for i in remaining_indices]
                            new_intermediate = get_or_create_intermediate!(elementary_model, intermediate_registry, remaining_metabolites, enzyme_id, reaction_compartment)
                            elem_rxn.stoichiometry[new_intermediate] = 1.0

                            # Track this complex for next level (avoid duplicates)
                            sorted_remaining = sort(remaining_indices)
                            if !(sorted_remaining in new_complexes)
                                push!(new_complexes, sorted_remaining)
                            end
                        else
                            elem_rxn.stoichiometry[enzyme_id] = 1.0
                        end

                        elementary_model.reactions[rxn_id] = elem_rxn
                    end
                end

                complex_formed_old = new_complexes
            end
        end
    end
end