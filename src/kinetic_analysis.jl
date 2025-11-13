"""
Kinetic Module Analysis via Upstream Algorithm

Implements the upstream algorithm for identifying kinetic modules in metabolic networks.
The algorithm identifies sets of complexes that satisfy both autonomous and feeding properties,
then merges coupled modules through shared balanced complexes.

# Algorithm Overview
1. **Extended Modules**: For each concordance module, form extended module = balanced ∪ concordance module
2. **Upstream Algorithm**: Apply 4-phase algorithm to identify kinetic modules
3. **Coupling Detection**: Merge modules that share complexes (via balanced complexes)
4. **ACR/ACRR Analysis**: Detect concentration robustness properties

# References
- Langary et al. (2025) "Kinetic modules in metabolic reaction networks"
"""

using LinearAlgebra
using Graphs

# ========================================================================================
# Phase I-IV: Core Upstream Algorithm Utilities
# ========================================================================================

"""
    find_entry_complexes(complexes, A_matrix) -> Vector{Int}

Find complexes with incoming reactions from outside the given set.
Entry complexes violate the autonomous property (Phase I).
"""
function find_entry_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    entry_complexes = Int[]

    for complex_idx in complexes
        # Check reactions where this complex is produced
        for rxn_idx in SparseArrays.findnz(A_matrix[complex_idx, :])[1]
            if A_matrix[complex_idx, rxn_idx] > 0  # Complex is product
                # Check if any substrate is outside our set
                substrates = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                has_external_substrate = any(s -> A_matrix[s, rxn_idx] < 0 && s ∉ complexes, substrates)

                if has_external_substrate
                    push!(entry_complexes, complex_idx)
                    break
                end
            end
        end
    end

    return unique!(entry_complexes)
end

"""
    find_nonreactant_complexes(complexes, A_matrix) -> Vector{Int}

Find complexes with no outgoing reactions (non-reactant complexes).
These violate the feeding property (Phase II).
"""
function find_nonreactant_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    nonreactant_complexes = Int[]

    for complex_idx in complexes
        # Check if complex is consumed in any reaction
        reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]
        is_reactant = any(rxn -> A_matrix[complex_idx, rxn] < 0, reactions)

        if !is_reactant
            push!(nonreactant_complexes, complex_idx)
        end
    end

    return nonreactant_complexes
end

"""
    find_exit_complexes(complexes, A_matrix) -> Vector{Int}

Find complexes with outgoing reactions to outside the given set.
Exit complexes violate the feeding property (Phase III).
"""
function find_exit_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    exit_complexes = Int[]

    for complex_idx in complexes
        # Check reactions where this complex is consumed
        for rxn_idx in SparseArrays.findnz(A_matrix[complex_idx, :])[1]
            if A_matrix[complex_idx, rxn_idx] < 0  # Complex is substrate
                # Check if any product is outside our set
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                has_external_product = any(p -> A_matrix[p, rxn_idx] > 0 && p ∉ complexes, products)

                if has_external_product
                    push!(exit_complexes, complex_idx)
                    break
                end
            end
        end
    end

    return unique!(exit_complexes)
end

"""
    find_strongly_connected_components(A_matrix, complexes_subset) -> Vector{Vector{Int}}

Find strongly connected components using Tarjan's algorithm on the reaction graph.
Works directly with sparse incidence matrix A for efficiency (Phase IV).

# Arguments
- `A_matrix`: Complex-reaction incidence matrix (complexes × reactions)
- `complexes_subset`: Subset of complex indices to analyze (nothing = all complexes)

# Returns
Vector of SCCs, each containing complex indices
"""
function find_strongly_connected_components(
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    complexes_subset::Union{Vector{Int},Nothing}=nothing
)
    # Determine working set
    working_complexes = complexes_subset !== nothing ? complexes_subset : collect(1:size(A_matrix, 1))
    isempty(working_complexes) && return Vector{Int}[]

    n_working = length(working_complexes)
    orig_to_working = Dict(working_complexes[i] => i for i in 1:n_working)

    # Tarjan's algorithm state
    index_ref = Ref{Int}(0)
    indices = fill(-1, n_working)
    lowlinks = zeros(Int, n_working)
    on_stack = falses(n_working)
    stack = Int[]
    components = Vector{Int}[]

    # Get successors in reaction graph
    function get_neighbors(working_idx::Int)
        orig_idx = working_complexes[working_idx]
        neighbors = Int[]

        # Find reactions where this complex is a substrate
        for (rxn_idx, val) in zip(SparseArrays.findnz(A_matrix[orig_idx, :])...)
            val < 0 || continue  # Skip if not substrate

            # Find products in this reaction
            for (prod_idx, prod_val) in zip(SparseArrays.findnz(A_matrix[:, rxn_idx])...)
                if prod_val > 0 && haskey(orig_to_working, prod_idx)
                    push!(neighbors, orig_to_working[prod_idx])
                end
            end
        end

        return unique!(neighbors)
    end

    # Tarjan's DFS
    function strongconnect(v::Int)
        indices[v] = lowlinks[v] = index_ref[]
        index_ref[] += 1
        push!(stack, v)
        on_stack[v] = true

        # Process successors
        for w in get_neighbors(v)
            if indices[w] == -1
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elseif on_stack[w]
                lowlinks[v] = min(lowlinks[v], indices[w])
            end
        end

        # Root of SCC - pop stack
        if lowlinks[v] == indices[v]
            component = Int[]
            while true
                w = pop!(stack)
                on_stack[w] = false
                push!(component, working_complexes[w])
                w == v && break
            end
            push!(components, component)
        end
    end

    # Run on all unvisited nodes
    for v in 1:n_working
        indices[v] == -1 && strongconnect(v)
    end

    return components
end

"""
    is_terminal_component(component, remaining_complexes, A_matrix) -> Bool

Check if an SCC is terminal (has no outgoing edges to other components).
Terminal components violate the feeding property (Phase IV).
"""
function is_terminal_component(
    component::Vector{Int},
    remaining_complexes::Vector{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}
)
    for complex_idx in component
        # Check all reactions where this complex is a substrate
        for rxn_idx in SparseArrays.findnz(A_matrix[complex_idx, :])[1]
            A_matrix[complex_idx, rxn_idx] < 0 || continue  # Skip if not substrate

            # Check if any product is in remaining set but outside component
            for product_idx in SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                if A_matrix[product_idx, rxn_idx] > 0 &&
                   product_idx ∈ remaining_complexes &&
                   product_idx ∉ component
                    return false  # Found outgoing edge
                end
            end
        end
    end

    return true  # No outgoing edges
end

# ========================================================================================
# Upstream Algorithm
# ========================================================================================

"""
    upstream_algorithm(extended_module, A_matrix; verbose) -> Vector{Int}

Four-phase upstream algorithm to identify kinetic modules.

# Algorithm
1. **Phase I**: Iteratively remove entry complexes (autonomous property)
2. **Phase II**: Remove non-reactant complexes (feeding property)
3. **Phase III**: Iteratively remove exit complexes (feeding property)
4. **Phase IV**: Remove terminal SCCs (feeding property)

# Returns
Union of Phase III and Phase IV complexes (the upstream set)
"""
function upstream_algorithm(
    extended_module::Vector{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int};
    verbose::Bool=true
)
    isempty(extended_module) && return Int[]

    verbose && @debug "Starting upstream algorithm" initial_size = length(extended_module)

    # Phase I: Autonomous property (remove entry complexes)
    current_set = copy(extended_module)
    phase_i_excluded = Int[]

    verbose && @debug "Phase I: Removing entry complexes (autonomous property)"
    while true
        entries = find_entry_complexes(current_set, A_matrix)
        isempty(entries) && break

        append!(phase_i_excluded, entries)
        current_set = setdiff(current_set, entries)
        isempty(current_set) && return Int[]
    end
    verbose && @debug "Phase I complete" excluded = length(phase_i_excluded) remaining = length(current_set)

    # Phase II: Feeding property (remove non-reactants)
    verbose && @debug "Phase II: Removing non-reactant complexes (feeding property)"
    nonreactants = find_nonreactant_complexes(current_set, A_matrix)
    current_set = setdiff(current_set, nonreactants)
    verbose && @debug "Phase II complete" excluded = length(nonreactants) remaining = length(current_set)
    isempty(current_set) && return Int[]

    # Phase III: Feeding property (remove exit complexes)
    phase_iii_complexes = Int[]

    verbose && @debug "Phase III: Removing exit complexes (feeding property)"
    while true
        exits = find_exit_complexes(current_set, A_matrix)
        isempty(exits) && break

        append!(phase_iii_complexes, exits)
        current_set = setdiff(current_set, exits)
    end
    verbose && @debug "Phase III complete" excluded = length(phase_iii_complexes) remaining = length(current_set)

    # Phase IV: Feeding property (remove terminal SCCs)
    phase_iv_complexes = Int[]

    if !isempty(current_set)
        verbose && @debug "Phase IV: Processing strongly connected components"
        components = find_strongly_connected_components(A_matrix, current_set)

        for component in components
            if !is_terminal_component(component, current_set, A_matrix)
                append!(phase_iv_complexes, component)
            end
        end
        verbose && @debug "Phase IV complete" n_sccs = length(components) non_terminal = length(phase_iv_complexes)
    end

    # Upstream set = Phase III ∪ Phase IV
    upstream_set = union(phase_iii_complexes, phase_iv_complexes)

    verbose && @debug "Upstream algorithm complete" (
        initial=length(extended_module),
        phase_i_excluded=length(phase_i_excluded),
        phase_ii_excluded=length(nonreactants),
        phase_iii=length(phase_iii_complexes),
        phase_iv=length(phase_iv_complexes),
        upstream_set=length(upstream_set)
    )

    return upstream_set
end
# ========================================================================================
# Module Merging via Coupling Detection
# ========================================================================================

"""
    merge_coupled_upstream_sets(upstream_sets) -> Vector{Vector{Int}}

Merge upstream sets that share complexes (coupled via balanced complexes).
Uses graph-based connected component detection.
"""
function merge_coupled_upstream_sets(upstream_sets::Vector{Vector{Int}})
    n_sets = length(upstream_sets)
    isempty(upstream_sets) && return Vector{Int}[]

    # Build intersection graph
    intersection_graph = [Set{Int}() for _ in 1:n_sets]
    for i in 1:n_sets, j in (i+1):n_sets
        if !isempty(intersect(upstream_sets[i], upstream_sets[j]))
            push!(intersection_graph[i], j)
            push!(intersection_graph[j], i)
        end
    end

    # Find connected components via DFS
    visited = falses(n_sets)
    components = Vector{Vector{Int}}()

    function dfs!(node::Int, component::Vector{Int})
        visited[node] && return
        visited[node] = true
        push!(component, node)
        for neighbor in intersection_graph[node]
            dfs!(neighbor, component)
        end
    end

    for i in 1:n_sets
        if !visited[i]
            component = Int[]
            dfs!(i, component)
            push!(components, component)
        end
    end

    # Merge sets in each component
    merged_modules = Vector{Int}[]
    for component in components
        merged = union(upstream_sets[component]...)
        push!(merged_modules, merged)
    end

    return merged_modules
end

# ========================================================================================
# Main Kinetic Analysis Function
# ========================================================================================


"""
    apply_kinetic_analysis!(results, constraints; min_module_size, verbose)

Apply upstream algorithm with coupling detection to identify kinetic modules.

# Algorithm Steps
1. For each concordance module: form extended module (balanced ∪ concordance)
2. Apply upstream algorithm to each extended module
3. Restore balanced complexes to upstream sets (coupling enablers)
4. Merge upstream sets that share complexes
5. Detect ACR/ACRR properties and interface reactions

# Updates
Modifies `results` in-place, setting:
- `kinetic_modules`: module assignment for each complex
- `giant_id`: ID of largest module
- `acr_metabolites`, `acrr_pairs`: robustness properties
- `interface_reactions`: inter-module vs intra-module reactions
"""
function apply_kinetic_analysis!(
    results::ConcordanceResults,
    constraints::C.ConstraintTree;
    min_module_size::Int=1,
    verbose::Bool=true
)
    verbose && @info "Starting kinetic module analysis"

    # Initialize results
    results.kinetic_modules = fill(0, length(results.complex_ids))
    results.interface_reactions = falses(results.stats["n_reactions"])
    results.acr_metabolites = Symbol[]
    results.acrr_pairs = Tuple{Symbol,Symbol}[]
    results.giant_id = 0

    # Build A matrix from constraints
    A_matrix = A_matrix_from_constraints(constraints)
    balanced_complexes = findall(==(0), results.concordance_modules)

    verbose && @debug "Setup" n_complexes = length(results.complex_ids) n_balanced = length(balanced_complexes)

    # Step 1: Compute upstream sets for each concordance module
    upstream_sets = compute_upstream_sets(
        results.concordance_modules,
        balanced_complexes,
        A_matrix,
        results.complex_ids,
        results.stats["n_concordance_modules"],
        verbose
    )

    isempty(upstream_sets) && (results.stats["n_kinetic_modules"] = 0; return 0)

    # Step 2: Merge coupled upstream sets
    verbose && @info "Merging coupled modules" n_upstream_sets = length(upstream_sets)
    kinetic_modules = merge_coupled_upstream_sets(upstream_sets)

    # Step 3: Add singleton complexes from extended modules
    add_singleton_complexes!(kinetic_modules, results.concordance_modules, balanced_complexes, upstream_sets)

    # Step 4: Assign kinetic module IDs
    kinetic_module_count = assign_module_ids!(results, kinetic_modules, min_module_size)
    results.stats["n_kinetic_modules"] = kinetic_module_count

    # Step 5: Identify giant module
    identify_giant_module!(results, kinetic_modules, min_module_size)

    # Step 6: Analyze concentration robustness (ACR/ACRR) and interface reactions
    if kinetic_module_count > 0
        analyze_robustness_properties!(results, kinetic_modules, constraints, min_module_size, verbose)
    end

    verbose && @info "Kinetic analysis complete" (
        n_modules=kinetic_module_count,
        n_acr=length(results.acr_metabolites),
        n_acrr=length(results.acrr_pairs),
        giant_size=kinetic_module_count > 0 ? length(kinetic_modules[results.giant_id]) : 0
    )

    return kinetic_module_count
end
# ========================================================================================
# Helper Functions for Kinetic Analysis
# ========================================================================================

"""
    compute_upstream_sets(concordance_modules, balanced_complexes, A_matrix, complex_ids, n_conc_modules, verbose)

Compute upstream sets for each concordance module using the upstream algorithm.
Restores balanced complexes to enable coupling detection.
"""
function compute_upstream_sets(
    concordance_modules::Vector{Int},
    balanced_complexes::Vector{Int},
    A_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    complex_ids::Vector{Symbol},
    n_concordance_modules::Int,
    verbose::Bool
)
    upstream_sets = Vector{Int}[]

    for conc_mod_id in 1:n_concordance_modules
        # Get complexes in this concordance module
        conc_complexes = findall(==(conc_mod_id), concordance_modules)
        isempty(conc_complexes) && continue

        # Form extended module: balanced ∪ concordance
        extended_module = union(balanced_complexes, conc_complexes)

        # Apply upstream algorithm
        upstream_set = upstream_algorithm(extended_module, A_matrix, verbose=verbose)

        # Restore balanced complexes (critical for coupling detection)
        for bal_idx in balanced_complexes
            if bal_idx ∈ extended_module && bal_idx ∉ upstream_set
                push!(upstream_set, bal_idx)
                verbose && @debug "Restored balanced complex" complex_id = complex_ids[bal_idx] conc_module = conc_mod_id
            end
        end

        !isempty(upstream_set) && push!(upstream_sets, upstream_set)
    end

    verbose && @info "Computed upstream sets" n_sets = length(upstream_sets) n_concordance = n_concordance_modules
    return upstream_sets
end

"""
    add_singleton_complexes!(kinetic_modules, concordance_modules, balanced_complexes, upstream_sets)

Add complexes from extended modules that aren't in any upstream set as singletons.
"""
function add_singleton_complexes!(
    kinetic_modules::Vector{Vector{Int}},
    concordance_modules::Vector{Int},
    balanced_complexes::Vector{Int},
    upstream_sets::Vector{Vector{Int}}
)
    # Find all complexes in upstream sets
    all_upstream = Set{Int}()
    for upstream_set in kinetic_modules
        union!(all_upstream, upstream_set)
    end

    # Find all complexes in extended modules
    all_extended = Set{Int}(balanced_complexes)
    for conc_id in unique(filter(>=(0), concordance_modules))
        union!(all_extended, findall(==(conc_id), concordance_modules))
    end

    # Add missing complexes as singletons
    for complex_idx in all_extended
        if complex_idx ∉ all_upstream
            push!(kinetic_modules, [complex_idx])
        end
    end
end

"""
    assign_module_ids!(results, kinetic_modules, min_module_size) -> Int

Assign kinetic module IDs to complexes in results. Returns count of modules.
"""
function assign_module_ids!(
    results::ConcordanceResults,
    kinetic_modules::Vector{Vector{Int}},
    min_module_size::Int
)
    kinetic_module_count = 0

    for module_complexes in kinetic_modules
        if length(module_complexes) >= min_module_size
            kinetic_module_count += 1
            for complex_idx in module_complexes
                results.kinetic_modules[complex_idx] = kinetic_module_count
            end
        end
    end

    return kinetic_module_count
end

"""
    identify_giant_module!(results, kinetic_modules, min_module_size)

Identify and store the ID of the largest kinetic module (giant component).
"""
function identify_giant_module!(
    results::ConcordanceResults,
    kinetic_modules::Vector{Vector{Int}},
    min_module_size::Int
)
    valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)

    if !isempty(valid_modules)
        module_sizes = [length(km) for km in valid_modules]
        giant_idx = argmax(module_sizes)
        results.giant_id = giant_idx
    else
        results.giant_id = 0
    end
end

"""
    analyze_robustness_properties!(results, kinetic_modules, constraints, min_module_size, verbose)

Detect ACR/ACRR properties and interface reactions.
"""
function analyze_robustness_properties!(
    results::ConcordanceResults,
    kinetic_modules::Vector{Vector{Int}},
    constraints::C.ConstraintTree,
    min_module_size::Int,
    verbose::Bool
)
    verbose && @info "Analyzing concentration robustness properties"

    # Prepare kinetic groups
    kinetic_groups = Dict{Int,Vector{Int}}()
    for (i, km) in enumerate(kinetic_modules)
        length(km) >= min_module_size && (kinetic_groups[i] = km)
    end

    # Build Y matrix
    Y_matrix, _, _ = Y_matrix_from_constraints(constraints; return_ids=true)

    # Detect ACR metabolites
    acr_indices = detect_acr_metabolites(kinetic_groups, Y_matrix)
    for met_idx in acr_indices
        met_idx <= length(results.metabolite_ids) && push!(results.acr_metabolites, results.metabolite_ids[met_idx])
    end

    # Detect ACRR pairs
    acrr_index_pairs = detect_acrr_metabolite_pairs(kinetic_groups, Y_matrix)
    for (met1_idx, met2_idx) in acrr_index_pairs
        if met1_idx <= length(results.metabolite_ids) && met2_idx <= length(results.metabolite_ids)
            push!(results.acrr_pairs, (results.metabolite_ids[met1_idx], results.metabolite_ids[met2_idx]))
        end
    end

    # Detect interface reactions
    detect_interface_reactions!(results, kinetic_modules, constraints)

    # Update stats
    results.stats["n_acr_metabolites"] = length(results.acr_metabolites)
    results.stats["n_acrr_metabolite_pairs"] = length(results.acrr_pairs)
    results.stats["n_interface_reactions"] = count(results.interface_reactions)
    results.stats["n_intra_module_reactions"] = results.stats["n_reactions"] - count(results.interface_reactions)

    verbose && @info "Robustness analysis complete" n_acr = length(acr_indices) n_acrr = length(acrr_index_pairs)
end

# ========================================================================================
# ACR/ACRR Detection
# ========================================================================================

"""
    detect_acr_metabolites(kinetic_groups, Y_matrix) -> Vector{Int}

Detect metabolites with Absolute Concentration Robustness (ACR).
ACR occurs when two complexes in the same kinetic module differ by exactly one metabolite.
"""
function detect_acr_metabolites(kinetic_groups::Dict{Int,Vector{Int}}, Y_matrix::AbstractMatrix)
    acr_metabolites = Set{Int}()

    for (group_id, complexes) in kinetic_groups
        length(complexes) < 2 && continue

        # Check all pairs of complexes
        for i in 1:(length(complexes)-1), j in (i+1):length(complexes)
            c1_idx, c2_idx = complexes[i], complexes[j]
            diff_vector = Y_matrix[:, c1_idx] - Y_matrix[:, c2_idx]
            diff_indices = findall(x -> abs(x) > 1e-10, diff_vector)

            # ACR condition: exactly one metabolite differs
            length(diff_indices) == 1 && push!(acr_metabolites, diff_indices[1])
        end
    end

    return sort(collect(acr_metabolites))
end

"""
    detect_acrr_metabolite_pairs(kinetic_groups, Y_matrix) -> Vector{Tuple{Int,Int}}

Detect metabolite pairs with Absolute Concentration Ratio Robustness (ACRR).
ACRR occurs when two complexes differ in exactly 2 metabolites with specific structural conditions.
"""
function detect_acrr_metabolite_pairs(kinetic_groups::Dict{Int,Vector{Int}}, Y_matrix::AbstractMatrix)
    acrr_pairs = Set{Tuple{Int,Int}}()

    for (group_id, complexes) in kinetic_groups
        length(complexes) < 2 && continue

        # Check all pairs of complexes
        for i in 1:(length(complexes)-1), j in (i+1):length(complexes)
            c1_idx, c2_idx = complexes[i], complexes[j]
            diff_vector = Y_matrix[:, c1_idx] - Y_matrix[:, c2_idx]
            diff_indices = findall(x -> abs(x) > 1e-10, diff_vector)

            # ACRR condition: exactly 2 metabolites differ + structural constraints
            if length(diff_indices) == 2
                c1_mets = findall(x -> abs(x) > 1e-10, Y_matrix[:, c1_idx])
                c2_mets = findall(x -> abs(x) > 1e-10, Y_matrix[:, c2_idx])

                # Each complex must have exactly 1 metabolite not in the difference
                if length(setdiff(c1_mets, diff_indices)) == 1 && length(setdiff(c2_mets, diff_indices)) == 1
                    canonical_pair = diff_indices[1] < diff_indices[2] ?
                                     (diff_indices[1], diff_indices[2]) :
                                     (diff_indices[2], diff_indices[1])
                    push!(acrr_pairs, canonical_pair)
                end
            end
        end
    end

    return sort(collect(acrr_pairs))
end

# ========================================================================================
# Interface Reaction Detection
# ========================================================================================

"""
    detect_interface_reactions!(results, kinetic_modules, constraints)

Detect interface reactions (connect different kinetic modules) vs intra-module reactions.
Updates `results.interface_reactions` BitVector in-place.
"""
function detect_interface_reactions!(
    results::ConcordanceResults,
    kinetic_modules::Vector{Vector{Int}},
    constraints::C.ConstraintTree
)
    # Initialize all as interface reactions
    fill!(results.interface_reactions, true)

    # Build A matrix and complex-to-module mapping
    A_matrix = A_matrix_from_constraints(constraints)
    complex_to_module = Dict{Int,Int}()
    for (mod_id, complexes) in enumerate(kinetic_modules)
        for complex_idx in complexes
            complex_to_module[complex_idx] = mod_id
        end
    end

    # Identify intra-module reactions
    intra_module_reactions = Set{Int}()

    for (mod_id, module_complexes) in enumerate(kinetic_modules)
        length(module_complexes) < 2 && continue
        module_set = Set(module_complexes)

        # Check each reaction
        for rxn_idx in 1:results.stats["n_reactions"]
            # Get substrates and products
            substrates = findall(i -> A_matrix[i, rxn_idx] < 0, 1:size(A_matrix, 1))
            products = findall(i -> A_matrix[i, rxn_idx] > 0, 1:size(A_matrix, 1))

            # Intra-module if all substrates and products in this module
            if !isempty(substrates) && !isempty(products) &&
               all(s -> s ∈ module_set, substrates) && all(p -> p ∈ module_set, products)
                push!(intra_module_reactions, rxn_idx)
            end
        end
    end

    # Mark intra-module reactions as false (not interface)
    for rxn_idx in intra_module_reactions
        results.interface_reactions[rxn_idx] = false
    end
end

