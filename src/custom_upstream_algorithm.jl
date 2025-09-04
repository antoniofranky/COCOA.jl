"""
Upstream Algorithm

Optimized 4-phase upstream algorithm designed specifically for COCOA.jl's 
CompleteConcordanceModel data structure and performance requirements.

Based on theoretical foundations but adapted for practical use with:
- Large metabolic networks (50k+ reactions)
- Sparse matrix operations
- Memory efficiency
- Integration with CompleteConcordanceModel
"""

using LinearAlgebra
using Graphs

"""
Find complexes with incoming reactions from outside the given set.
Entry complexes violate the autonomous property.
"""
function find_entry_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    entry_complexes = Int[]

    for complex_idx in complexes
        # Find reactions where this complex is a product (incoming)
        incoming_reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]

        for rxn_idx in incoming_reactions
            if A_matrix[complex_idx, rxn_idx] > 0  # This complex is produced
                # Check if any substrate is outside our set
                substrates = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                for substrate_idx in substrates
                    if A_matrix[substrate_idx, rxn_idx] < 0 && substrate_idx ∉ complexes
                        push!(entry_complexes, complex_idx)
                        @goto next_complex
                    end
                end
            end
        end
        @label next_complex
    end

    return unique!(entry_complexes)
end

"""
Find complexes with no outgoing reactions (non-reactant complexes).
These violate the feeding property.
"""
function find_nonreactant_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    nonreactant_complexes = Int[]

    for complex_idx in complexes
        # Check if complex has any outgoing reactions
        outgoing_reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]
        has_outgoing = false

        for rxn_idx in outgoing_reactions
            if A_matrix[complex_idx, rxn_idx] < 0  # This complex is consumed
                has_outgoing = true
                break
            end
        end

        if !has_outgoing
            push!(nonreactant_complexes, complex_idx)
        end
    end

    return nonreactant_complexes
end

"""
Find complexes with outgoing reactions to outside the given set.
Exit complexes violate internal closure but are part of the feeding property.
"""
function find_exit_complexes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    exit_complexes = Int[]

    for complex_idx in complexes
        # Find reactions where this complex is a substrate (outgoing)
        outgoing_reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]

        for rxn_idx in outgoing_reactions
            if A_matrix[complex_idx, rxn_idx] < 0  # This complex is consumed
                # Check if any product is outside our set
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                for product_idx in products
                    if A_matrix[product_idx, rxn_idx] > 0 && product_idx ∉ complexes
                        push!(exit_complexes, complex_idx)
                        @goto next_complex
                    end
                end
            end
        end
        @label next_complex
    end

    return unique!(exit_complexes)
end

"""
Find strongly connected components using Graphs.jl (modern, tested implementation).
Much cleaner than custom Tarjan's algorithm.
"""
function find_strongly_connected_components(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    if isempty(complexes)
        return Vector{Int}[]
    end

    n_complexes = length(complexes)
    complex_to_local = Dict(complexes[i] => i for i in 1:n_complexes)

    # Build directed graph from A matrix using Graphs.jl
    g = SimpleDiGraph(n_complexes)

    for (local_i, global_i) in enumerate(complexes)
        # Find reactions where this complex is a substrate
        reactions = SparseArrays.findnz(A_matrix[global_i, :])[1]

        for rxn_idx in reactions
            if A_matrix[global_i, rxn_idx] < 0  # Substrate
                # Find products of this reaction
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                for prod_global in products
                    if A_matrix[prod_global, rxn_idx] > 0 && haskey(complex_to_local, prod_global)
                        local_j = complex_to_local[prod_global]
                        add_edge!(g, local_i, local_j)
                    end
                end
            end
        end
    end

    # Find strongly connected components using Graphs.jl
    scc_indices = strongly_connected_components(g)

    # Convert back to global complex indices
    components = Vector{Int}[]
    for component in scc_indices
        push!(components, [complexes[i] for i in component])
    end

    return components
end

"""
Check if a strongly connected component is terminal.
A component is terminal if it has no outgoing connections to other components within the given set.
"""
function is_terminal_component(component::Vector{Int}, remaining_complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    for complex_idx in component
        # Find outgoing reactions
        reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]

        for rxn_idx in reactions
            if A_matrix[complex_idx, rxn_idx] < 0  # This complex is a substrate
                # Check products
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                for product_idx in products
                    if A_matrix[product_idx, rxn_idx] > 0 &&
                       product_idx ∈ remaining_complexes &&
                       product_idx ∉ component
                        return false  # Found connection to other component
                    end
                end
            end
        end
    end

    return true  # No external connections found
end

"""
4-Phase Upstream Algorithm

Optimized implementation for large metabolic networks with sparse operations.

# Arguments
- `extended_module::Vector{Int}`: Complex indices (balanced + concordance module)
- `A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}`: Complex-reaction incidence matrix

# Returns  
- `Vector{Int}`: Complexes in the identified kinetic module (empty if none)

# Algorithm Phases
1. **Phase I**: Iteratively remove entry complexes (autonomous property)
2. **Phase II**: Remove non-reactant complexes (feeding property)  
3. **Phase III**: Iteratively remove exit complexes (feeding property)
4. **Phase IV**: Remove terminal strongly connected components (feeding property)

# Performance Notes
- Uses sparse matrix operations for memory efficiency
- Optimized for networks with 50k+ reactions
- Returns union of Phase III and IV complexes as per theoretical specification
"""
function upstream_algorithm(extended_module::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}; verbose::Bool=true)
    if isempty(extended_module)
        verbose && @info "Empty extended module provided"
        return Int[]
    end

    verbose && @info "Starting upstream algorithm" initial_size = length(extended_module)

    # Phase I: Iteratively exclude entry complexes (autonomous property)
    current_set = copy(extended_module)
    phase_i_excluded = Int[]
    phase_i_iterations = 0

    verbose && @info " Phase I: Removing entry complexes (autonomous property)"

    while true
        phase_i_iterations += 1
        entry_complexes = find_entry_complexes(current_set, A_matrix)

        if isempty(entry_complexes)
            verbose && @info "Phase I complete" iterations = phase_i_iterations excluded_total = length(phase_i_excluded) remaining = length(current_set)
            break  # Achieved autonomous property
        end

        verbose && @info "Iteration $phase_i_iterations" entry_found = length(entry_complexes) entry_complexes = entry_complexes

        append!(phase_i_excluded, entry_complexes)
        current_set = setdiff(current_set, entry_complexes)

        if isempty(current_set)
            verbose && @info "  All complexes excluded in Phase I - no autonomous subset exists"
            return Int[]  # All complexes excluded
        end

        verbose && @info "  After removing entries" remaining = length(current_set)
    end

    # Phase II: Exclude non-reactant complexes (feeding property)
    verbose && @info "Phase II: Removing non-reactant complexes (feeding property)"

    nonreactant = find_nonreactant_complexes(current_set, A_matrix)
    phase_ii_excluded = nonreactant
    current_set = setdiff(current_set, nonreactant)

    verbose && @info "Phase II result" excluded = length(nonreactant) excluded_complexes = nonreactant remaining = length(current_set)

    if isempty(current_set)
        verbose && @info "All remaining complexes excluded in Phase II - no feeding subset exists"
        return Int[]
    end

    # Phase III: Iteratively exclude exit complexes (feeding property) 
    verbose && @info "Phase III: Removing exit complexes (feeding property)"

    phase_iii_complexes = Int[]
    phase_iii_iterations = 0

    while true
        phase_iii_iterations += 1
        exit_complexes = find_exit_complexes(current_set, A_matrix)

        if isempty(exit_complexes)
            verbose && @info "   Phase III complete" iterations = phase_iii_iterations excluded_total = length(phase_iii_complexes) remaining = length(current_set)
            break  # No more exit complexes
        end

        verbose && @info "Iteration $phase_iii_iterations" exit_found = length(exit_complexes) exit_complexes = exit_complexes

        append!(phase_iii_complexes, exit_complexes)
        current_set = setdiff(current_set, exit_complexes)

        if isempty(current_set)
            verbose && @info "  All remaining complexes are exit complexes"
            break
        end

        verbose && @info "  After removing exits" remaining = length(current_set)
    end

    # Phase IV: Exclude terminal strongly connected components (feeding property)
    verbose && @info "Phase IV: Processing strongly connected components"

    phase_iv_complexes = Int[]

    if !isempty(current_set)
        components = find_strongly_connected_components(current_set, A_matrix)
        verbose && @info "  Found $(length(components)) strongly connected components"

        for (comp_idx, component) in enumerate(components)
            is_terminal = is_terminal_component(component, current_set, A_matrix)

            if !is_terminal
                # Non-terminal component - these are part of kinetic module
                verbose && @info "  Component $comp_idx is NON-TERMINAL" size = length(component) complexes = component
                append!(phase_iv_complexes, component)
            else
                # Terminal components are excluded (not added to phase_iv_complexes)
                verbose && @info "  Component $comp_idx is TERMINAL (excluded)" size = length(component) complexes = component
            end
        end
    else
        verbose && @info "  No complexes remaining for SCC analysis"
    end

    # Return union of Phase III and Phase IV complexes (the upstream set)
    # This matches the theoretical specification and R implementation
    upstream_set = union(phase_iii_complexes, phase_iv_complexes)

    verbose && @info " Upstream algorithm complete" (
        initial_complexes=length(extended_module),
        phase_i_excluded=length(phase_i_excluded),
        phase_ii_excluded=length(phase_ii_excluded),
        phase_iii_complexes=length(phase_iii_complexes),
        phase_iv_complexes=length(phase_iv_complexes),
        final_upstream_set=length(upstream_set),
        upstream_complexes=upstream_set
    )

    return upstream_set
end


"""
Get linkage classes from the complete concordance model.
Linkage classes are strongly connected components in the reaction network graph.
"""
function extract_linkage_classes(complex_reaction_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    n_complexes = size(complex_reaction_matrix, 1)

    if n_complexes == 0
        return Vector{Int}[]
    end

    # Find all complexes and get their strongly connected components
    all_complexes = collect(1:n_complexes)
    linkage_classes = find_strongly_connected_components(all_complexes, complex_reaction_matrix)

    # Filter out singleton classes (individual complexes)
    return [lc for lc in linkage_classes if length(lc) > 1]
end


"""
Apply upstream algorithm with coupling detection to identify kinetic modules.

This implements the complete theoretical algorithm:
1. Apply upstream algorithm to each extended module independently  
2. Check for coupling between upstream sets from different modules
3. Merge coupled upstream sets to form kinetic modules

Updates the model's kinetic_modules field efficiently.
"""
function apply_kinetic_analysis!(complete_model::CompleteConcordanceModel; min_module_size::Int=1)
    @info "Applying upstream algorithm with coupling detection for kinetic module identification"

    # Reset kinetic modules
    fill!(complete_model.kinetic_modules, 0)

    # Get balanced complexes (concordance module 0) 
    balanced_complexes = findall(==(0), complete_model.concordance_modules)

    # Use the A matrix from the model (now built correctly without blocked reactions)
    A_matrix = complete_model.complex_reaction_matrix

    # Extract linkage classes for coupling detection
    linkage_classes = extract_linkage_classes(A_matrix)

    @info "Setup for coupling detection" n_balanced = length(balanced_complexes) n_linkage_classes = length(linkage_classes)

    # Step 1: Apply upstream algorithm to each extended module to get upstream sets
    upstream_sets = Vector{Int}[]  # Store upstream sets from each concordance module
    module_to_upstream = Dict{Int,Vector{Int}}()  # Map concordance module ID to its upstream set

    # The balanced complexes are those with concordance_modules[i] == 0
    # We should only process actual concordance modules (1 to n), not the balanced "module 0"
    # This matches R logic where balanced complexes are separate from concordance modules

    # Process each concordance module (1 to n) - balanced complexes are already identified above
    for concordance_module_id in 1:complete_model.n_concordance_modules
        # Get complexes in this concordance module
        concordance_complexes = findall(==(concordance_module_id), complete_model.concordance_modules)

        if isempty(concordance_complexes)
            continue
        end

        # CRITICAL FIX 1: Construct extended module = balanced ∪ concordance module
        # This follows the theoretical definition 𝒞ₘ̅ = 𝒞ᵦ ∪ 𝒞ₘ from the paper
        extended_module = union(balanced_complexes, concordance_complexes)

        # Apply upstream algorithm to get upstream set
        upstream_set = upstream_algorithm(extended_module, A_matrix, verbose=true)

        if !isempty(upstream_set)
            push!(upstream_sets, upstream_set)
            module_to_upstream[concordance_module_id] = upstream_set
            @info "Upstream set found" concordance_module = concordance_module_id upstream_size = length(upstream_set)
        end
    end

    @info "Found $(length(upstream_sets)) upstream sets from $(complete_model.n_concordance_modules) concordance modules"

    # CRITICAL FIX 2: R-style module merging based on intersection matrix
    # This follows the R code: mdiff[i,j] = length(intersect(lres[[i]], lres[[j]]))
    kinetic_modules = Vector{Int}[]  # Final kinetic modules after merging

    if isempty(upstream_sets)
        @info "No upstream sets found - no kinetic modules identified"
        complete_model.n_kinetic_modules = 0
        return 0
    end

    @info "Applying R-style module merging" n_upstream_sets = length(upstream_sets)

    # Step 1: Create intersection matrix (like R code mdiff matrix)
    n_sets = length(upstream_sets)
    intersection_matrix = zeros(Int, n_sets, n_sets)

    for i in 1:n_sets
        for j in 1:n_sets
            if i != j
                intersection_count = length(intersect(upstream_sets[i], upstream_sets[j]))
                intersection_matrix[i, j] = intersection_count
                if intersection_count > 0
                    @info "Intersection found between upstream sets $i and $j" shared_complexes = intersection_count
                end
            end
        end
    end

    # Step 2: Create graph where modules with shared complexes are connected
    # (equivalent to R code: mg <- graph_from_adjacency_matrix(mdiff, mode = "undirected"))
    module_graph = [Set{Int}() for _ in 1:n_sets]
    for i in 1:n_sets
        for j in 1:n_sets
            if intersection_matrix[i, j] > 0  # Modules share complexes
                push!(module_graph[i], j)
            end
        end
    end

    # Step 3: Find connected components (like R code: clustmg <- components(mg))
    visited = falses(n_sets)
    components = Vector{Vector{Int}}()

    function dfs_component(node::Int, current_component::Vector{Int})
        if visited[node]
            return
        end
        visited[node] = true
        push!(current_component, node)

        for neighbor in module_graph[node]
            dfs_component(neighbor, current_component)
        end
    end

    for i in 1:n_sets
        if !visited[i]
            component = Int[]
            dfs_component(i, component)
            push!(components, component)
        end
    end

    @info "Found $(length(components)) connected components for merging" component_sizes = [length(c) for c in components]

    # Step 4: Merge modules in same connected component 
    # (like R code: final_lres[[length(final_lres) + 1]] <- union(lres[[p[1]]], lres[[p[j]]]))
    for (comp_id, component) in enumerate(components)
        if length(component) == 1
            # Single module - no merging needed
            merged_module = upstream_sets[component[1]]
        else
            # Merge all modules in this component
            merged_module = upstream_sets[component[1]]  # Start with first module
            for j in 2:length(component)
                merged_module = union(merged_module, upstream_sets[component[j]])
            end
            @info "Merged component $comp_id" original_modules = component merged_size = length(merged_module)
        end

        # Add merged module to final kinetic modules
        if length(merged_module) >= min_module_size
            push!(kinetic_modules, merged_module)
        end
    end

    # Step 5: Add remaining singleton complexes (like R code)
    # Find all complexes that participated in extended modules but aren't in any upstream set
    all_upstream_complexes = Set{Int}()
    for kinetic_module in kinetic_modules
        union!(all_upstream_complexes, kinetic_module)
    end

    # Get all complexes that were in extended modules (balanced + all concordance complexes)
    all_extended_complexes = Set{Int}(balanced_complexes)
    for concordance_id in 1:complete_model.n_concordance_modules
        concordance_complexes = findall(==(concordance_id), complete_model.concordance_modules)
        union!(all_extended_complexes, concordance_complexes)
    end

    # Add complexes that were in extended modules but not captured in upstream sets as singletons
    for complex_idx in all_extended_complexes
        if complex_idx ∉ all_upstream_complexes
            push!(kinetic_modules, [complex_idx])
        end
    end

    @info "Module merging completed" n_final_modules = length(kinetic_modules) n_merged_from = length(upstream_sets)

    # Step 6: Update model with final kinetic modules
    kinetic_module_count = 0
    total_kinetic_complexes = 0

    for (module_id, kinetic_complexes) in enumerate(kinetic_modules)
        if length(kinetic_complexes) >= min_module_size
            kinetic_module_count += 1
            total_kinetic_complexes += length(kinetic_complexes)

            # Update kinetic_modules field in model
            for complex_idx in kinetic_complexes
                complete_model.kinetic_modules[complex_idx] = kinetic_module_count
            end

            @info "Kinetic module $kinetic_module_count created" size = length(kinetic_complexes)
        end
    end

    # Update model statistics
    complete_model.n_kinetic_modules = kinetic_module_count

    # Step 7: Detect ACR and ACRR properties from kinetic modules
    if kinetic_module_count > 0
        @info "Analyzing concentration robustness properties"

        # Convert kinetic_modules to Dict format expected by ACR/ACRR detection
        kinetic_groups_dict = Dict{Int,Vector{Int}}()
        for (i, km) in enumerate(kinetic_modules)
            if length(km) >= min_module_size
                kinetic_groups_dict[i] = km
            end
        end
        Y_matrix = complete_model.metabolite_complex_matrix
        # Detect ACR metabolites
        acr_metabolites = detect_acr_metabolites(kinetic_groups_dict, Y_matrix)

        # Detect ACRR metabolite pairs  
        acrr_pairs = detect_acrr_metabolite_pairs(kinetic_groups_dict, Y_matrix)

        # Step 7.1: Detect interface reactions (following R reference logic)
        @info "Detecting interface reactions"
        detect_interface_reactions!(complete_model, kinetic_modules)

        # Convert indices to metabolite IDs for storage
        acr_metabolite_ids = Vector{Symbol}()
        if !isempty(acr_metabolites)
            for met_idx in acr_metabolites
                if met_idx <= length(complete_model.metabolite_ids)
                    push!(acr_metabolite_ids, complete_model.metabolite_ids[met_idx])
                end
            end
        end

        acrr_metabolite_id_pairs = Vector{Tuple{Symbol,Symbol}}()
        if !isempty(acrr_pairs)
            for (met1_idx, met2_idx) in acrr_pairs
                if met1_idx <= length(complete_model.metabolite_ids) && met2_idx <= length(complete_model.metabolite_ids)
                    met1_id = complete_model.metabolite_ids[met1_idx]
                    met2_id = complete_model.metabolite_ids[met2_idx]
                    push!(acrr_metabolite_id_pairs, (met1_id, met2_id))
                end
            end
        end

        # Update model's ACR metabolites BitVector
        fill!(complete_model.acr_metabolites, false)  # Reset
        for met_idx in acr_metabolites
            if met_idx <= length(complete_model.acr_metabolites)
                complete_model.acr_metabolites[met_idx] = true
            end
        end

        # Store results in stats  
        n_interface_reactions = count(complete_model.interface_reactions)
        n_intra_module_reactions = complete_model.n_reactions - n_interface_reactions
        
        complete_model.stats["n_acr_metabolites"] = length(acr_metabolites)
        complete_model.stats["n_acrr_metabolite_pairs"] = length(acrr_pairs)
        complete_model.stats["acr_metabolite_ids"] = acr_metabolite_ids
        complete_model.stats["acrr_metabolite_id_pairs"] = acrr_metabolite_id_pairs
        complete_model.stats["n_interface_reactions"] = n_interface_reactions
        complete_model.stats["n_intra_module_reactions"] = n_intra_module_reactions

        @info " Concentration robustness analysis complete" (
            acr_metabolites=length(acr_metabolites),
            acrr_pairs=length(acrr_pairs)
        )
    else
        @info "  No kinetic modules found - skipping ACR/ACRR analysis"
        complete_model.stats["n_acr_metabolites"] = 0
        complete_model.stats["n_acrr_metabolite_pairs"] = 0
        complete_model.stats["acr_metabolite_ids"] = Symbol[]
        complete_model.stats["acrr_metabolite_id_pairs"] = Tuple{Symbol,Symbol}[]
    end

    @info "Kinetic analysis with coupling detection and robustness analysis completed" (
        kinetic_modules=kinetic_module_count,
        total_complexes=total_kinetic_complexes,
        acr_metabolites=get(complete_model.stats, "n_acr_metabolites", 0),
        acrr_pairs=get(complete_model.stats, "n_acrr_metabolite_pairs", 0)
    )

    return kinetic_module_count
end


"""
Get stoichiometric difference between two complexes.
Returns the difference vector and indices of non-zero entries.
"""
function get_stoichiometric_difference(Y_matrix::AbstractMatrix, complex1_idx::Int, complex2_idx::Int)
    if complex1_idx > size(Y_matrix, 2) || complex2_idx > size(Y_matrix, 2)
        return Float64[], Int[]
    end

    diff_vector = Y_matrix[:, complex1_idx] - Y_matrix[:, complex2_idx]
    nonzero_indices = findall(x -> abs(x) > 1e-10, diff_vector)

    return diff_vector, nonzero_indices
end

"""
Check if two complexes satisfy ACRR conditions.
Implements the specific criteria from the Upstream_Algorithm R code.
"""
function check_acrr_conditions(Y_matrix::AbstractMatrix, complex1_idx::Int, complex2_idx::Int, diff_indices::Vector{Int})
    if length(diff_indices) != 2
        return false
    end

    # Get non-zero metabolite indices for each complex
    complex1_metabolites = findall(x -> abs(x) > 1e-10, Y_matrix[:, complex1_idx])
    complex2_metabolites = findall(x -> abs(x) > 1e-10, Y_matrix[:, complex2_idx])

    # Check ACRR conditions from R code:
    # qq2: complex1 has exactly 1 metabolite not in the difference
    metabolites_only_in_complex1 = setdiff(complex1_metabolites, diff_indices)
    condition2 = length(metabolites_only_in_complex1) == 1

    # qq3: complex2 has exactly 1 metabolite not in the difference  
    metabolites_only_in_complex2 = setdiff(complex2_metabolites, diff_indices)
    condition3 = length(metabolites_only_in_complex2) == 1

    return condition2 && condition3
end

"""
Detect metabolites with Absolute Concentration Robustness (ACR).

ACR metabolites are found by identifying cases where two complexes in the same 
kinetic module differ by exactly one metabolite. Based on the Upstream_Algorithm R code.

# Arguments
- `kinetic_groups::Dict{Int,Vector{Int}}`: Kinetic modules (group_id => complex_indices)
- `Y_matrix::AbstractMatrix`: Stoichiometric matrix (metabolites × complexes)

# Returns
- `Vector{Int}`: Indices of metabolites with ACR properties
"""
function detect_acr_metabolites(kinetic_groups::Dict{Int,Vector{Int}}, Y_matrix::AbstractMatrix)
    acr_metabolites = Set{Int}()

    @info " Detecting ACR metabolites from $(length(kinetic_groups)) kinetic modules"

    for (group_id, kinetic_complexes) in kinetic_groups
        if length(kinetic_complexes) < 2
            continue  # Need at least 2 complexes for comparison
        end

        @debug "Checking kinetic module $group_id with $(length(kinetic_complexes)) complexes"

        # Check all pairs of complexes within this kinetic module
        for i in 1:(length(kinetic_complexes)-1)
            for j in (i+1):length(kinetic_complexes)
                complex1_idx = kinetic_complexes[i]
                complex2_idx = kinetic_complexes[j]

                # Get stoichiometric difference
                diff_vector, diff_indices = get_stoichiometric_difference(Y_matrix, complex1_idx, complex2_idx)

                # ACR condition: exactly one metabolite differs
                if length(diff_indices) == 1
                    metabolite_idx = diff_indices[1]
                    push!(acr_metabolites, metabolite_idx)
                    @debug "ACR found: metabolite $metabolite_idx (complexes $complex1_idx, $complex2_idx in module $group_id)"
                end
            end
        end
    end

    acr_result = sort(collect(acr_metabolites))
    @info "ACR detection complete: found $(length(acr_result)) metabolites with ACR"

    return acr_result
end

"""
Detect metabolite pairs with Absolute Concentration Ratio Robustness (ACRR).

ACRR pairs are found by identifying cases where two complexes in the same kinetic module
differ in exactly 2 metabolites with specific structural conditions. Based on the 
Upstream_Algorithm R code.

# Arguments
- `kinetic_groups::Dict{Int,Vector{Int}}`: Kinetic modules (group_id => complex_indices) 
- `Y_matrix::AbstractMatrix`: Stoichiometric matrix (metabolites × complexes)

# Returns
- `Vector{Tuple{Int,Int}}`: Pairs of metabolite indices with ACRR properties
"""
function detect_acrr_metabolite_pairs(kinetic_groups::Dict{Int,Vector{Int}}, Y_matrix::AbstractMatrix)
    acrr_pairs = Set{Tuple{Int,Int}}()

    @info " Detecting ACRR metabolite pairs from $(length(kinetic_groups)) kinetic modules"

    for (group_id, kinetic_complexes) in kinetic_groups
        if length(kinetic_complexes) < 2
            continue  # Need at least 2 complexes for comparison
        end

        @debug "Checking kinetic module $group_id with $(length(kinetic_complexes)) complexes for ACRR"

        # Check all pairs of complexes within this kinetic module
        for i in 1:(length(kinetic_complexes)-1)
            for j in (i+1):length(kinetic_complexes)
                complex1_idx = kinetic_complexes[i]
                complex2_idx = kinetic_complexes[j]

                # Get stoichiometric difference
                diff_vector, diff_indices = get_stoichiometric_difference(Y_matrix, complex1_idx, complex2_idx)

                # ACRR condition: exactly 2 metabolites differ with specific constraints
                if length(diff_indices) == 2 && check_acrr_conditions(Y_matrix, complex1_idx, complex2_idx, diff_indices)
                    # Store in canonical order (smaller index first)
                    met1, met2 = diff_indices[1], diff_indices[2]
                    canonical_pair = met1 < met2 ? (met1, met2) : (met2, met1)
                    push!(acrr_pairs, canonical_pair)
                    @debug "ACRR found: metabolites $canonical_pair (complexes $complex1_idx, $complex2_idx in module $group_id)"
                end
            end
        end
    end

    acrr_result = sort(collect(acrr_pairs))
    @info "ACRR detection complete: found $(length(acrr_result)) metabolite pairs with ACRR"

    return acrr_result
end

"""
Detect interface reactions based on kinetic modules.

Interface reactions connect complexes from different kinetic modules.
Intra-module reactions connect complexes within the same kinetic module.

This follows the R reference logic:
- Get all edges (reactions) in the network
- Identify which reactions have both source and target in the same kinetic module (intra-module)
- All other reactions are interface reactions

Updates the interface_reactions BitVector in the model.
"""
function detect_interface_reactions!(complete_model::CompleteConcordanceModel, kinetic_modules::Vector{Vector{Int}})
    @info "Detecting interface reactions following R reference logic"
    
    # Reset interface reactions (assume all are interface initially)
    fill!(complete_model.interface_reactions, true)
    
    A_matrix = complete_model.complex_reaction_matrix
    
    # Create mapping from complex to its kinetic module
    complex_to_module = Dict{Int,Int}()
    for (module_id, complexes) in enumerate(kinetic_modules)
        for complex_idx in complexes
            complex_to_module[complex_idx] = module_id
        end
    end
    
    @info "Built complex-to-module mapping" n_complexes_in_modules = length(complex_to_module)
    
    intra_module_reactions = Set{Int}()
    
    # Check each kinetic module for intra-module reactions  
    for (module_id, module_complexes) in enumerate(kinetic_modules)
        if length(module_complexes) < 2
            continue  # Need at least 2 complexes to have intra-module reactions
        end
        
        # Find reactions where both source and target are in this module
        # Following R logic: c1 <- which(eg[,1] %in% final_lres[[j]]) and c2 <- which(eg[,2] %in% final_lres[[j]])
        module_set = Set(module_complexes)
        
        # Check each reaction to see if both substrate and product are in this module
        for reaction_idx in 1:complete_model.n_reactions
            reaction_column = A_matrix[:, reaction_idx]
            
            # Find substrate complexes (negative entries)
            substrate_complexes = findall(x -> x < 0, reaction_column)
            # Find product complexes (positive entries)  
            product_complexes = findall(x -> x > 0, reaction_column)
            
            # Check if all substrates and products are in this kinetic module
            substrates_in_module = all(c -> c ∈ module_set, substrate_complexes)
            products_in_module = all(c -> c ∈ module_set, product_complexes)
            
            if substrates_in_module && products_in_module && !isempty(substrate_complexes) && !isempty(product_complexes)
                push!(intra_module_reactions, reaction_idx)
            end
        end
    end
    
    # Mark intra-module reactions as false (not interface)
    for reaction_idx in intra_module_reactions
        complete_model.interface_reactions[reaction_idx] = false
    end
    
    n_interface = count(complete_model.interface_reactions)  
    n_intra_module = length(intra_module_reactions)
    
    @info "Interface reaction detection complete" (
        total_reactions=complete_model.n_reactions,
        intra_module_reactions=n_intra_module,
        interface_reactions=n_interface
    )
    
    return nothing
end

"""
Comprehensive validation function for iterative testing of upstream algorithm phases.
Returns detailed validation report for debugging and verification.
"""
function validate_upstream_algorithm(complete_model::CompleteConcordanceModel; verbose::Bool=true)
    @info "Starting comprehensive validation of upstream algorithm"

    validation_results = Dict{String,Any}()

    # Get test data
    balanced_complexes = findall(==(0), complete_model.concordance_modules)
    A_matrix = complete_model.complex_reaction_matrix

    validation_results["n_balanced"] = length(balanced_complexes)
    validation_results["n_concordance_modules"] = complete_model.n_concordance_modules
    validation_results["matrix_size"] = size(A_matrix)

    verbose && @info "Validation setup" n_balanced = length(balanced_complexes) n_concordance_modules = complete_model.n_concordance_modules matrix_size = size(A_matrix)

    # Test each concordance module
    module_results = Dict{Int,Any}()

    for concordance_module_id in 1:complete_model.n_concordance_modules
        if concordance_module_id == 0
            continue
        end

        concordance_complexes = findall(==(concordance_module_id), complete_model.concordance_modules)

        if isempty(concordance_complexes)
            continue
        end

        extended_module = union(balanced_complexes, concordance_complexes)

        verbose && @info "Testing concordance module $concordance_module_id" n_complexes = length(concordance_complexes) extended_size = length(extended_module)

        # Phase-by-phase validation
        phase_results = validate_algorithm_phases(extended_module, A_matrix, verbose)
        module_results[concordance_module_id] = phase_results

        # Final algorithm result
        kinetic_complexes = upstream_algorithm(extended_module, A_matrix)
        phase_results["final_kinetic_size"] = length(kinetic_complexes)
        phase_results["kinetic_complexes"] = kinetic_complexes

        verbose && @info "Module $concordance_module_id result" kinetic_size = length(kinetic_complexes)
    end

    validation_results["module_results"] = module_results

    # Overall summary
    total_kinetic_complexes = sum(result["final_kinetic_size"] for result in values(module_results))
    modules_with_kinetic = count(result["final_kinetic_size"] > 0 for result in values(module_results))

    validation_results["summary"] = (
        total_modules_tested=length(module_results),
        modules_with_kinetic=modules_with_kinetic,
        total_kinetic_complexes=total_kinetic_complexes
    )

    @info "Validation complete" summary = validation_results["summary"]

    return validation_results
end

"""
Debug Phase III exit complex detection logic specifically.
"""
function debug_exit_complex_detection(current_set::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}; verbose::Bool=true)
    verbose && @info "Debugging exit complex detection for $(length(current_set)) complexes"

    exit_complexes_found = Int[]

    for complex_idx in current_set
        verbose && @info "Checking complex $complex_idx for exit connections"

        # Find outgoing reactions where this complex is consumed
        outgoing_reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]

        exits_to_external = false
        external_products = Int[]

        for rxn_idx in outgoing_reactions
            if A_matrix[complex_idx, rxn_idx] < 0  # This complex is consumed
                verbose && @info "  Complex $complex_idx is consumed in reaction $rxn_idx"

                # Check products of this reaction
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                for product_idx in products
                    if A_matrix[product_idx, rxn_idx] > 0  # Is a product
                        if product_idx ∉ current_set  # Product is OUTSIDE our set
                            verbose && @info "    → Product $product_idx is OUTSIDE current set - EXIT FOUND!"
                            exits_to_external = true
                            push!(external_products, product_idx)
                        else
                            verbose && @info "    → Product $product_idx is inside current set"
                        end
                    end
                end
            end
        end

        if exits_to_external
            push!(exit_complexes_found, complex_idx)
            verbose && @info "   Complex $complex_idx is an EXIT complex (connects to external: $external_products)"
        else
            verbose && @info "  ✗ Complex $complex_idx has no external connections"
        end
    end

    verbose && @info "Phase III debug complete: found $(length(exit_complexes_found)) exit complexes: $exit_complexes_found"

    return exit_complexes_found
end

"""
Debug the terminal component classification logic specifically.
"""
function debug_terminal_components(remaining_complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}; verbose::Bool=true)
    if isempty(remaining_complexes)
        @info "No complexes to analyze"
        return
    end

    # Get SCCs
    components = find_strongly_connected_components(remaining_complexes, A_matrix)

    verbose && @info "Found $(length(components)) strongly connected components"

    for (i, component) in enumerate(components)
        verbose && @info "Analyzing component $i with $(length(component)) complexes: $component"

        # Check each complex in component for outgoing connections
        outgoing_connections = Dict{Int,Vector{Int}}()

        for complex_idx in component
            connections = Int[]

            # Find outgoing reactions where this complex is consumed
            reactions = SparseArrays.findnz(A_matrix[complex_idx, :])[1]

            for rxn_idx in reactions
                if A_matrix[complex_idx, rxn_idx] < 0  # This complex is a substrate
                    # Find products of this reaction
                    products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]

                    for product_idx in products
                        if A_matrix[product_idx, rxn_idx] > 0 &&  # Is a product
                           product_idx ∈ remaining_complexes &&   # Is in our remaining set
                           product_idx ∉ component               # Is NOT in same component
                            push!(connections, product_idx)
                        end
                    end
                end
            end

            outgoing_connections[complex_idx] = connections
        end

        # Determine if terminal
        has_external_connections = any(!isempty(connections) for connections in values(outgoing_connections))
        is_terminal = !has_external_connections

        verbose && @info "Component $i analysis" is_terminal = is_terminal outgoing_connections = outgoing_connections

        if !is_terminal
            verbose && @info "Component $i is NON-TERMINAL - should be included in kinetic module!"
        else
            verbose && @info "Component $i is terminal - correctly excluded"
        end
    end

    return components
end

"""
Validate each phase of the upstream algorithm individually.
"""
function validate_algorithm_phases(extended_module::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int}, verbose::Bool=false)
    phase_results = Dict{String,Any}()

    # Initial state
    current_set = copy(extended_module)
    phase_results["initial_size"] = length(current_set)

    # Phase I: Entry complexes validation
    phase_i_iterations = 0
    phase_i_total_excluded = Int[]

    while true
        phase_i_iterations += 1
        entry_complexes = find_entry_complexes(current_set, A_matrix)

        if isempty(entry_complexes)
            break
        end

        verbose && @info "Phase I iteration $phase_i_iterations" entry_found = length(entry_complexes) remaining = length(current_set)

        append!(phase_i_total_excluded, entry_complexes)
        current_set = setdiff(current_set, entry_complexes)

        if isempty(current_set)
            break
        end
    end

    phase_results["phase_i"] = (
        iterations=phase_i_iterations,
        excluded=phase_i_total_excluded,
        excluded_count=length(phase_i_total_excluded),
        remaining_after=length(current_set)
    )

    verbose && @info "Phase I complete" iterations = phase_i_iterations excluded = length(phase_i_total_excluded) remaining = length(current_set)

    if isempty(current_set)
        phase_results["early_termination"] = "phase_i"
        return phase_results
    end

    # Phase II: Non-reactant complexes validation
    nonreactant = find_nonreactant_complexes(current_set, A_matrix)
    current_set = setdiff(current_set, nonreactant)

    phase_results["phase_ii"] = (
        excluded=nonreactant,
        excluded_count=length(nonreactant),
        remaining_after=length(current_set)
    )

    verbose && @info "Phase II complete" excluded = length(nonreactant) remaining = length(current_set)

    if isempty(current_set)
        phase_results["early_termination"] = "phase_ii"
        return phase_results
    end

    # Phase III: Exit complexes validation
    phase_iii_iterations = 0
    phase_iii_total_excluded = Int[]

    while true
        phase_iii_iterations += 1
        exit_complexes = find_exit_complexes(current_set, A_matrix)

        if isempty(exit_complexes)
            break
        end

        verbose && @info "Phase III iteration $phase_iii_iterations" exit_found = length(exit_complexes) remaining = length(current_set)

        append!(phase_iii_total_excluded, exit_complexes)
        current_set = setdiff(current_set, exit_complexes)

        if isempty(current_set)
            break
        end
    end

    phase_results["phase_iii"] = (
        iterations=phase_iii_iterations,
        excluded=phase_iii_total_excluded,
        excluded_count=length(phase_iii_total_excluded),
        remaining_after=length(current_set)
    )

    verbose && @info "Phase III complete" iterations = phase_iii_iterations excluded = length(phase_iii_total_excluded) remaining = length(current_set)

    # Phase IV: Strongly connected components validation
    phase_iv_complexes = Int[]
    phase_iv_components = Vector{Int}[]
    phase_iv_terminal_components = Vector{Int}[]

    if !isempty(current_set)
        components = find_strongly_connected_components(current_set, A_matrix)
        phase_iv_components = components

        for component in components
            if !is_terminal_component(component, current_set, A_matrix)
                append!(phase_iv_complexes, component)
            else
                push!(phase_iv_terminal_components, component)
            end
        end
    end

    phase_results["phase_iv"] = (
        total_components=length(phase_iv_components),
        terminal_components=phase_iv_terminal_components,
        non_terminal_complexes=phase_iv_complexes,
        non_terminal_count=length(phase_iv_complexes)
    )

    verbose && @info "Phase IV complete" total_components = length(phase_iv_components) terminal = length(phase_iv_terminal_components) non_terminal = length(phase_iv_complexes)

    # Final upstream set (union of Phase III and IV)
    upstream_set = union(phase_iii_total_excluded, phase_iv_complexes)
    phase_results["upstream_set_size"] = length(upstream_set)

    return phase_results
end