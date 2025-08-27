"""
Kinetic module analysis for COCOA.

This module implements:
- Kinetic module identification based on concordance modules
- Four-phase autonomy filtering algorithm
- Concentration robustness analysis for metabolites
- Memory-efficient processing for large-scale HPC models
"""


"""
Results from kinetic module analysis.

Contains the original concordance results plus kinetic-specific analysis:
- Kinetic modules identified through autonomy filtering
- Network topology matrices for position queries
- Interface reactions between modules
"""
struct KineticModuleResults
    # Reuse concordance results (no duplication)
    concordance_results::NamedTuple

    # Kinetic-specific additions
    kinetic_modules::Dict{Symbol,Vector{Symbol}}
    giant_module_id::Symbol
    interface_reactions::Vector{Symbol}

    # Essential matrices (sparse, memory-efficient)
    Y_matrix::SparseArrays.SparseMatrixCSC{Float64}  # Species-complex matrix (m × n) - stoichiometric coefficients
    A_matrix::SparseArrays.SparseMatrixCSC{Int8}  # Complex-reaction matrix (n × r) - incidence matrix (1/-1)

    # Index mappings for network position queries
    metabolite_names::Vector{String}
    reaction_names::Vector{String}
    complex_to_idx::Dict{String,Int}
    metabolite_to_idx::Dict{String,Int}
    reaction_to_idx::Dict{String,Int}

    # Comprehensive summary for large-scale analysis
    summary::Union{Dict{String,Int},Nothing}
end

"""
Results from concentration robustness analysis.

Identifies metabolites showing absolute concentration robustness
and metabolite pairs showing concentration ratio robustness.
"""
struct ConcentrationRobustnessResults
    # Reuse kinetic results
    kinetic_results::KineticModuleResults

    # Robustness findings
    robust_metabolites::Vector{String}
    robust_metabolite_pairs::Vector{Tuple{String,String}}

    # Statistics
    n_robust_metabolites::Int
    n_robust_pairs::Int
    giant_kinetic_module_size::Int

    # Comprehensive summary for large-scale analysis
    summary::Union{Dict{String,Int},Nothing}
end

"""
$(TYPEDSIGNATURES)

Extract Y (species-complex) and A (complex-reaction) matrices from COCOA constraints.
Uses the actual complexes extracted by COCOA's constraint system for consistency.
This ensures compatibility with concordance analysis and follows the paper's definitions.

Y matrix: metabolites × complexes, entries are stoichiometric coefficients
A matrix: complexes × reactions, entries are +1/-1 for product/substrate relationships
"""
function extract_network_matrices_from_constraints(constraints, model::AbstractFBCModels.AbstractFBCModel)
    # Extract complexes using COCOA's system
    complexes = COCOA.extract_complexes_from_model(model)

    # Get reaction information - check if we have split reactions
    if haskey(constraints.balance, :fluxes_forward) && haskey(constraints.balance, :fluxes_reverse)
        # We have split reactions
        forward_flux_vars = constraints.balance.fluxes_forward
        reverse_flux_vars = constraints.balance.fluxes_reverse

        reaction_names = collect(string.(keys(forward_flux_vars)))
        n_base_reactions = length(reaction_names)

        # Create full reaction list with forward and reverse directions
        full_reaction_names = Vector{String}()
        sizehint!(full_reaction_names, 2 * n_base_reactions)

        for rxn_name in reaction_names
            push!(full_reaction_names, rxn_name * "_forward")
            push!(full_reaction_names, rxn_name * "_reverse")
        end
    else
        # Standard reactions without splitting
        reaction_names = collect(string.(keys(constraints.balance.fluxes)))
        full_reaction_names = reaction_names
    end

    # Get metabolite information from the model
    metabolites = AbstractFBCModels.metabolites(model)
    metabolite_names = collect(metabolites)
    n_metabolites = length(metabolite_names)

    # Get complex information  
    complex_names = collect(string.(keys(complexes)))
    n_complexes = length(complex_names)

    @info "Building matrices from COCOA complexes" n_metabolites = n_metabolites n_complexes = n_complexes n_reactions = length(full_reaction_names)

    # Create index mappings
    metabolite_to_idx = Dict{String,Int}()
    for (i, met_name) in enumerate(metabolite_names)
        metabolite_to_idx[met_name] = i
    end

    complex_to_idx = Dict{String,Int}()
    for (i, complex_name) in enumerate(complex_names)
        complex_to_idx[complex_name] = i
    end

    reaction_to_idx = Dict{String,Int}()
    for (i, rxn_name) in enumerate(full_reaction_names)
        reaction_to_idx[rxn_name] = i
    end

    # Build Y matrix: metabolites × complexes
    Y_rows = Vector{Int}()
    Y_cols = Vector{Int}()
    Y_vals = Vector{Float64}()

    for (complex_idx, (complex_id, complex_obj)) in enumerate(complexes)
        for (met_symbol, stoich_coeff) in complex_obj.metabolites
            met_name = string(met_symbol)
            if haskey(metabolite_to_idx, met_name)
                met_idx = metabolite_to_idx[met_name]
                push!(Y_rows, met_idx)
                push!(Y_cols, complex_idx)
                push!(Y_vals, stoich_coeff)
            end
        end
    end

    Y_matrix = SparseArrays.sparse(Y_rows, Y_cols, Y_vals, n_metabolites, n_complexes)

    # Build A matrix: complexes × reactions
    A_rows = Vector{Int}()
    A_cols = Vector{Int}()
    A_vals = Vector{Int8}()

    for (complex_idx, (complex_id, complex_obj)) in enumerate(complexes)
        for (rxn_symbol, contribution) in complex_obj.reaction_contributions
            rxn_name = string(rxn_symbol)

            # Handle split reactions by checking both forward and reverse
            if haskey(reaction_to_idx, rxn_name * "_forward")
                # Split reaction case
                forward_rxn_idx = reaction_to_idx[rxn_name*"_forward"]
                reverse_rxn_idx = reaction_to_idx[rxn_name*"_reverse"]

                if contribution > 0
                    # This complex is a product of the forward reaction
                    push!(A_rows, complex_idx)
                    push!(A_cols, forward_rxn_idx)
                    push!(A_vals, Int8(1))

                    # And a substrate of the reverse reaction
                    push!(A_rows, complex_idx)
                    push!(A_cols, reverse_rxn_idx)
                    push!(A_vals, Int8(-1))
                elseif contribution < 0
                    # This complex is a substrate of the forward reaction
                    push!(A_rows, complex_idx)
                    push!(A_cols, forward_rxn_idx)
                    push!(A_vals, Int8(-1))

                    # And a product of the reverse reaction
                    push!(A_rows, complex_idx)
                    push!(A_cols, reverse_rxn_idx)
                    push!(A_vals, Int8(1))
                end
            elseif haskey(reaction_to_idx, rxn_name)
                # Standard reaction case
                rxn_idx = reaction_to_idx[rxn_name]
                push!(A_rows, complex_idx)
                push!(A_cols, rxn_idx)
                push!(A_vals, Int8(sign(contribution)))
            end
        end
    end

    A_matrix = SparseArrays.sparse(A_rows, A_cols, A_vals, n_complexes, length(full_reaction_names))

    # Return complex activities from constraints if available
    complex_activities = haskey(constraints, :activities) ? constraints.activities : nothing

    return (
        Y_matrix=Y_matrix,
        A_matrix=A_matrix,
        metabolite_names=metabolite_names,
        complex_names=complex_names,
        reaction_names=full_reaction_names,
        complex_to_idx=complex_to_idx,
        metabolite_to_idx=metabolite_to_idx,
        reaction_to_idx=reaction_to_idx,
        complexes=complexes,  # Return the actual complex objects for validation
        complex_activities=complex_activities
    )
end

"""
Helper function to extract coefficient of a variable from a ConstraintTrees LinearValue.
"""
function extract_coefficient(linear_value, target_var_idx)
    # ConstraintTrees.LinearValue has idxs and weights fields
    if hasfield(typeof(linear_value), :idxs) && hasfield(typeof(linear_value), :weights)
        for (i, idx) in enumerate(linear_value.idxs)
            if idx == target_var_idx
                return linear_value.weights[i]
            end
        end
    end
    return 0.0
end

"""
$(TYPEDSIGNATURES)

Extract Y (species-complex) and A (complex-reaction) matrices from model.
Y matrix uses Float64 for stoichiometric coefficients, A matrix uses Int8 for incidence (1/-1).
"""
function extract_network_matrices(model)
    # Get model information and cache lengths
    metabolites = AbstractFBCModels.metabolites(model)
    reactions = AbstractFBCModels.reactions(model)
    n_metabolites = length(metabolites)
    n_reactions = length(reactions)

    # Build species-reaction stoichiometric matrix first
    S = AbstractFBCModels.stoichiometry(model)

    # Pre-allocate vectors with estimated sizes for better performance
    estimated_complexes = 2 * n_reactions  # substrate + product per reaction
    estimated_entries = n_metabolites * 2  # rough estimate for matrix entries

    complexes = Vector{String}()
    sizehint!(complexes, estimated_complexes)

    Y_rows = Vector{Int}()
    Y_cols = Vector{Int}()
    Y_vals = Vector{Float64}()
    sizehint!(Y_rows, estimated_entries)
    sizehint!(Y_cols, estimated_entries)
    sizehint!(Y_vals, estimated_entries)

    A_rows = Vector{Int}()
    A_cols = Vector{Int}()
    A_vals = Vector{Int8}()
    sizehint!(A_rows, estimated_complexes)
    sizehint!(A_cols, estimated_complexes)
    sizehint!(A_vals, estimated_complexes)

    complex_idx = 1

    # Simplified: create substrate and product complexes for each reaction
    for (rxn_idx, rxn_id) in enumerate(reactions)
        # Get stoichiometry for this reaction
        rxn_stoich = S[:, rxn_idx]

        # Create substrate complex
        substrate_indices = findall(x -> x < 0, rxn_stoich)
        if !isempty(substrate_indices)
            push!(complexes, "$(rxn_id)_substrate")
            for met_idx in substrate_indices
                push!(Y_rows, met_idx)
                push!(Y_cols, complex_idx)
                push!(Y_vals, -rxn_stoich[met_idx])
            end
            # Add to A matrix (substrate -> reaction)
            push!(A_rows, complex_idx)
            push!(A_cols, rxn_idx)
            push!(A_vals, Int8(-1))
            complex_idx += 1
        end

        # Create product complex
        product_indices = findall(x -> x > 0, rxn_stoich)
        if !isempty(product_indices)
            push!(complexes, "$(rxn_id)_product")
            for met_idx in product_indices
                push!(Y_rows, met_idx)
                push!(Y_cols, complex_idx)
                push!(Y_vals, rxn_stoich[met_idx])
            end
            # Add to A matrix (reaction -> product)
            push!(A_rows, complex_idx)
            push!(A_cols, rxn_idx)
            push!(A_vals, Int8(1))
            complex_idx += 1
        end
    end

    n_complexes = length(complexes)

    Y_matrix = SparseArrays.sparse(Y_rows, Y_cols, Y_vals, n_metabolites, n_complexes)
    A_matrix = SparseArrays.sparse(A_rows, A_cols, A_vals, n_complexes, n_reactions)

    # Create index mappings with pre-allocated sizes
    complex_to_idx = Dict{String,Int}()
    sizehint!(complex_to_idx, n_complexes)
    @inbounds for i in eachindex(complexes)
        complex_to_idx[complexes[i]] = i
    end

    metabolite_to_idx = Dict{String,Int}()
    sizehint!(metabolite_to_idx, n_metabolites)
    @inbounds for i in eachindex(metabolites)
        metabolite_to_idx[metabolites[i]] = i
    end

    reaction_to_idx = Dict{String,Int}()
    sizehint!(reaction_to_idx, n_reactions)
    @inbounds for i in eachindex(reactions)
        reaction_to_idx[reactions[i]] = i
    end

    return (
        Y_matrix=Y_matrix,
        A_matrix=A_matrix,
        metabolite_names=collect(metabolites),
        complex_names=complexes,
        reaction_names=collect(reactions),
        complex_to_idx=complex_to_idx,
        metabolite_to_idx=metabolite_to_idx,
        reaction_to_idx=reaction_to_idx
    )
end

"""
$(TYPEDSIGNATURES)

Create class_with_balanced groups by grouping mutually concordant complexes using transitive closure.
This replicates the MATLAB preprocessing step that creates `class_with_balanced`.
"""
function create_class_with_balanced_groups(concordance_results, network_data)
    complexes_df = concordance_results.complexes
    
    # Create concordance pairs (CP) - equivalent to [CP(:,1),CP(:,2)] = find(CC~=0) in MATLAB
    concordance_pairs = Vector{Tuple{Int,Int}}()
    
    # Add pairs within each concordance module (including balanced)
    for module_id in unique(complexes_df.module)
        module_complexes = DF.filter(row -> row.module == module_id, complexes_df)
        if DF.nrow(module_complexes) > 1
            complex_indices = Int[]
            for row in DF.eachrow(module_complexes)
                complex_name = String(row.id)
                if haskey(network_data.complex_to_idx, complex_name)
                    push!(complex_indices, network_data.complex_to_idx[complex_name])
                end
            end
            
            # Create all pairs within this module
            for i in 1:length(complex_indices)
                for j in (i+1):length(complex_indices)
                    push!(concordance_pairs, (complex_indices[i], complex_indices[j]))
                end
            end
        end
    end
    
    # Get all unique complex indices
    all_indices = Set{Int}()
    for (i, j) in concordance_pairs
        push!(all_indices, i)
        push!(all_indices, j)
    end
    unclassified = collect(all_indices)
    
    # Group using transitive closure (equivalent to MATLAB while loop)
    class_with_balanced = Vector{Vector{Int}}()
    
    while !isempty(unclassified)
        # Start new group with first unclassified complex
        i = unclassified[1]
        current_group = [i]
        
        # Find all pairs involving complexes in current group
        changed = true
        while changed
            old_size = length(current_group)
            
            # Find all concordance pairs that involve any complex in current group
            for (c1, c2) in concordance_pairs
                if c1 in current_group && !(c2 in current_group)
                    push!(current_group, c2)
                elseif c2 in current_group && !(c1 in current_group)
                    push!(current_group, c1)
                end
            end
            
            # Remove duplicates
            current_group = unique(current_group)
            changed = length(current_group) > old_size
        end
        
        # Remove classified complexes from unclassified
        unclassified = setdiff(unclassified, current_group)
        
        # Add completed group
        push!(class_with_balanced, current_group)
    end
    
    return class_with_balanced
end

"""
$(TYPEDSIGNATURES)

Add pure balanced linkage classes as separate kinetic modules.
This replicates the R code that finds weak connected components containing only balanced complexes.
"""
function add_pure_balanced_linkage_classes(candidate_modules, largest_group_indices, network_data)
    # Find weak connected components in the network graph
    A_matrix = network_data.A_matrix
    n_complexes = size(A_matrix, 1)
    
    # Create adjacency matrix for undirected graph (weak connectivity)
    adj_matrix = falses(n_complexes, n_complexes)
    for reaction_idx in 1:size(A_matrix, 2)
        substrates = findall(x -> x < 0, A_matrix[:, reaction_idx])
        products = findall(x -> x > 0, A_matrix[:, reaction_idx])
        
        # Connect all substrates to all products (undirected)
        for s in substrates
            for p in products
                adj_matrix[s, p] = adj_matrix[p, s] = true
            end
        end
    end
    
    # Find connected components using simple BFS
    visited = falses(n_complexes)
    components = Vector{Vector{Int}}()
    
    for i in 1:n_complexes
        if !visited[i]
            component = [i]
            queue = [i]
            visited[i] = true
            
            while !isempty(queue)
                current = popfirst!(queue)
                for neighbor in 1:n_complexes
                    if !visited[neighbor] && adj_matrix[current, neighbor]
                        push!(component, neighbor)
                        push!(queue, neighbor)
                        visited[neighbor] = true
                    end
                end
            end
            
            push!(components, component)
        end
    end
    
    # Check which components contain only balanced complexes
    balanced_set = Set(largest_group_indices)  # largest group contains balanced complexes
    result_modules = copy(candidate_modules)
    
    for component in components
        if length(component) > 1  # Skip singleton components
            # Check if component contains only balanced complexes
            if all(idx -> idx in balanced_set, component)
                push!(result_modules, component)
            end
        end
    end
    
    return result_modules
end

"""
$(TYPEDSIGNATURES)

Merge overlapping kinetic modules using connected components on the overlap graph.
This replicates the R merging logic with mdiff matrix and graph connected components.
"""
function merge_overlapping_modules(candidate_modules)
    if isempty(candidate_modules)
        return Vector{Vector{Int}}()
    end
    
    n_modules = length(candidate_modules)
    
    # Create overlap matrix (mdiff in R)
    overlap_matrix = zeros(Int, n_modules, n_modules)
    for i in 1:(n_modules-1)
        for j in (i+1):n_modules
            overlap = length(intersect(candidate_modules[i], candidate_modules[j]))
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap
        end
    end
    
    # Create adjacency matrix (any overlap > 0 means modules should be merged)
    adj_matrix = overlap_matrix .> 0
    
    # Find connected components
    visited = falses(n_modules)
    final_modules = Vector{Vector{Int}}()
    
    for i in 1:n_modules
        if !visited[i]
            # Find connected component starting from i
            component_indices = [i]
            queue = [i]
            visited[i] = true
            
            while !isempty(queue)
                current = popfirst!(queue)
                for j in 1:n_modules
                    if !visited[j] && adj_matrix[current, j]
                        push!(component_indices, j)
                        push!(queue, j)
                        visited[j] = true
                    end
                end
            end
            
            # Merge all modules in this component
            merged_module = Vector{Int}()
            for module_idx in component_indices
                append!(merged_module, candidate_modules[module_idx])
            end
            
            # Remove duplicates
            merged_module = unique(merged_module)
            push!(final_modules, merged_module)
        end
    end
    
    return final_modules
end

"""
$(TYPEDSIGNATURES)

Apply the four-phase autonomy algorithm from the kinetic modules paper.
Filters complexes to identify autonomous sets that form kinetic modules.
"""
function apply_r_autonomy_filter(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    """
    Implements the Upstream Algorithm autonomy filter from Section S.2.3.
    
    CRITICAL: Returns the UNION of removed complexes from Phase III and Phase IV.
    The remaining complexes after removal form the autonomous and feeding upstream set.
    
    Upstream Algorithm:
    - Phase I: Remove entry complexes iteratively until autonomous  
    - Phase II: Remove sink complexes (degree 0)
    - Phase III: Remove exit complexes iteratively 
    - Phase IV: Remove non-terminal strongly connected components
    
    Returns: Vector of complex indices that were REMOVED (to be excluded from kinetic module)
    """
    if isempty(complex_indices)
        return Int[]
    end

    x = copy(complex_indices)  # Current set being processed
    original_count = length(x)
    @info "Autonomy filter starting" input_size = original_count

    # Phase I: Remove entry complexes iteratively
    f = true
    while f
        res = ones(Int, length(x))
        for i in 1:length(x)
            # Get incoming neighbors (ego with mode "in")
            complex_idx = x[i]
            incoming_complexes = Set{Int}()

            # Find reactions where this complex is a product
            product_reactions = findall(v -> v > 0, A_matrix[complex_idx, :])
            for rxn_idx in product_reactions
                # Find substrate complexes for this reaction (incoming neighbors)
                substrate_complexes = findall(v -> v < 0, A_matrix[:, rxn_idx])
                union!(incoming_complexes, substrate_complexes)
            end

            # Check if there are incoming complexes outside our current set
            check = setdiff(incoming_complexes, Set(x))
            if length(check) > 0
                res[i] = 0
            end
        end

        removed_this_round = sum(res .== 0)
        x = x[res.==1]  # Keep only complexes with res == 1
        if removed_this_round == 0 || length(x) == 0
            f = false
        end
        if removed_this_round > 0
            @info "Phase I removed complexes" removed_count = removed_this_round remaining_count = length(x)
        end
    end
    
    @info "Phase I completed" remaining_after_phase_I = length(x)

    # Phase II: Remove sink complexes (degree 0)
    if length(x) > 0
        res = ones(Int, length(x))
        for i in 1:length(x)
            complex_idx = x[i]
            # Calculate out-degree: number of reactions where this complex is a substrate
            out_degree = sum(A_matrix[complex_idx, :] .< 0)
            if out_degree == 0
                res[i] = 0
            end
        end
        removed_phase_II = sum(res .== 0)
        x = x[res.==1]
        if removed_phase_II > 0
            @info "Phase II removed sinks" removed_count = removed_phase_II remaining_count = length(x)
        end
    end
    
    @info "Phase II completed" remaining_after_phase_II = length(x)

    # Phase III: Remove exit complexes iteratively
    x1 = copy(x)  # Save state before Phase III
    if length(x) > 0
        f = true
        while f
            res = ones(Int, length(x))
            for i in 1:length(x)
                # Get outgoing neighbors (ego with mode "out")
                complex_idx = x[i]
                outgoing_complexes = Set{Int}()

                # Find reactions where this complex is a substrate
                substrate_reactions = findall(v -> v < 0, A_matrix[complex_idx, :])
                for rxn_idx in substrate_reactions
                    # Find product complexes for this reaction (outgoing neighbors)
                    product_complexes = findall(v -> v > 0, A_matrix[:, rxn_idx])
                    union!(outgoing_complexes, product_complexes)
                end

                # Check if there are outgoing complexes outside our current set
                check = setdiff(outgoing_complexes, Set(x))
                if length(check) > 0
                    res[i] = 0
                end
            end

            removed_this_round = sum(res .== 0)
            x = x[res.==1]  # Keep only complexes with res == 1
            if removed_this_round == 0 || length(x) == 0
                f = false
            end
            if removed_this_round > 0
                @info "Phase III removed complexes" removed_count = removed_this_round remaining_count = length(x)
            end
        end
    end

    phase_III_removed = setdiff(x1, x)  # Complexes removed in Phase III
    @info "Phase III completed" removed_count = length(phase_III_removed) remaining_after_phase_III = length(x)

    # Phase IV: Remove non-terminal strongly connected components
    phase_IV_removed = Int[]
    if length(x) > 0
        # Find strongly connected components in the full graph (R uses full graph g)
        n_complexes = size(A_matrix, 1)

        # Build adjacency matrix for full graph
        adj_matrix = zeros(Bool, n_complexes, n_complexes)

        for rxn_idx in 1:size(A_matrix, 2)
            substrate_complexes = findall(v -> v < 0, A_matrix[:, rxn_idx])
            product_complexes = findall(v -> v > 0, A_matrix[:, rxn_idx])

            # Add edges from substrates to products
            for sub_idx in substrate_complexes
                for prod_idx in product_complexes
                    adj_matrix[sub_idx, prod_idx] = true
                end
            end
        end

        # Find strongly connected components using simple connected components
        # (R uses igraph components function with mode="strong")
        graph = Graphs.SimpleDiGraph(adj_matrix)
        sccs = Graphs.strongly_connected_components(graph)

        # Create membership array
        scc_membership = zeros(Int, n_complexes)
        for (scc_id, scc) in enumerate(sccs)
            for node in scc
                scc_membership[node] = scc_id
            end
        end

        n_sccs = length(sccs)

        # Check which SCCs are terminal
        terminal_c = ones(Int, n_sccs)  # 1 means terminal
        for i in 1:n_sccs
            vc = findall(scc_membership .== i)  # complexes in this SCC

            # Get outgoing neighbors for all complexes in this SCC
            outgoing_neighbors = Set{Int}()
            for complex_idx in vc
                substrate_reactions = findall(v -> v < 0, A_matrix[complex_idx, :])
                for rxn_idx in substrate_reactions
                    product_complexes = findall(v -> v > 0, A_matrix[:, rxn_idx])
                    union!(outgoing_neighbors, product_complexes)
                end
            end

            # Check if outgoing neighbors include complexes outside this SCC
            sdsn = setdiff(outgoing_neighbors, Set(vc))
            if length(sdsn) > 0
                terminal_c[i] = 0  # Not terminal
            end
        end

        # Mark complexes based on their SCC's terminal status
        terminal_complexes = ones(Int, n_complexes)
        for i in 1:n_complexes
            terminal_complexes[i] = terminal_c[scc_membership[i]]
        end

        # Phase IV removes complexes in x that are non-terminal (terminal == 0)
        phase_IV_removed = x[terminal_complexes[x].==0]
        @info "Phase IV completed" removed_count = length(phase_IV_removed)
    end

    total_removed = union(phase_III_removed, phase_IV_removed)
    @info "Autonomy filter completed" total_removed_count = length(total_removed) original_size = original_count final_size = original_count - length(total_removed)

    # Return union of removed complexes (this is what R returns!)
    return total_removed
end

"""
$(TYPEDSIGNATURES)

Find interface reactions that connect different kinetic modules.
"""
function find_interface_reactions(kinetic_modules::Dict{Symbol,Vector{Symbol}},
    A_matrix::SparseArrays.SparseMatrixCSC,
    complex_to_idx::Dict{String,Int},
    reaction_names::Vector{String})
    interface_reactions = Set{Symbol}()

    # Create module membership mapping
    complex_to_module = Dict{String,Symbol}()
    for (module_id, complexes) in kinetic_modules
        for complex_id in complexes
            complex_to_module[string(complex_id)] = module_id
        end
    end

    # Check each reaction
    for (rxn_idx, rxn_name) in enumerate(reaction_names)
        # Find substrate and product complexes for this reaction
        substrate_complexes = findall(x -> x < 0, A_matrix[:, rxn_idx])
        product_complexes = findall(x -> x > 0, A_matrix[:, rxn_idx])

        # Get modules of substrates and products
        substrate_modules = Set{Symbol}()
        product_modules = Set{Symbol}()

        for complex_idx in substrate_complexes
            # Find complex name from index - correct lookup
            complex_name = nothing
            for (name, idx) in complex_to_idx
                if idx == complex_idx
                    complex_name = name
                    break
                end
            end
            if complex_name !== nothing && haskey(complex_to_module, complex_name)
                push!(substrate_modules, complex_to_module[complex_name])
            end
        end

        for complex_idx in product_complexes
            # Find complex name from index - correct lookup
            complex_name = nothing
            for (name, idx) in complex_to_idx
                if idx == complex_idx
                    complex_name = name
                    break
                end
            end
            if complex_name !== nothing && haskey(complex_to_module, complex_name)
                push!(product_modules, complex_to_module[complex_name])
            end
        end

        # If substrates and products are in different modules, it's an interface reaction
        all_modules = union(substrate_modules, product_modules)
        if length(all_modules) > 1
            push!(interface_reactions, Symbol(rxn_name))
        end
    end

    return collect(interface_reactions)
end

"""
$(TYPEDSIGNATURES)

Save kinetic analysis results to JLD2 file with compression.
"""
function save_kinetic_results(results::KineticModuleResults, filepath::String)
    JLD2.jldsave(filepath;
        # Concordance results
        concordance_complexes=results.concordance_results.complexes,
        concordance_modules=results.concordance_results.modules,
        concordance_lambdas=results.concordance_results.lambdas,
        concordance_stats=results.concordance_results.stats,

        # Kinetic results
        kinetic_modules=results.kinetic_modules,
        giant_module_id=results.giant_module_id,
        interface_reactions=results.interface_reactions,

        # Network matrices
        Y_matrix=results.Y_matrix,
        A_matrix=results.A_matrix,

        # Names and mappings
        metabolite_names=results.metabolite_names,
        reaction_names=results.reaction_names,
        complex_to_idx=results.complex_to_idx,
        metabolite_to_idx=results.metabolite_to_idx,
        reaction_to_idx=results.reaction_to_idx, compress=true
    )
end

"""
$(TYPEDSIGNATURES)

Load kinetic analysis results from JLD2 file.
"""
function load_kinetic_results(filepath::String)
    data = JLD2.load(filepath)

    concordance_results = (
        complexes=data["concordance_complexes"],
        modules=data["concordance_modules"],
        lambdas=data["concordance_lambdas"],
        stats=data["concordance_stats"]
    )

    return KineticModuleResults(
        concordance_results,
        data["kinetic_modules"],
        data["giant_module_id"],
        data["interface_reactions"],
        data["Y_matrix"],
        data["A_matrix"],
        data["metabolite_names"],
        data["reaction_names"],
        data["complex_to_idx"],
        data["metabolite_to_idx"],
        data["reaction_to_idx"],
        nothing  # summary will be populated by analysis functions
    )
end

"""
$(TYPEDSIGNATURES)

Save concentration robustness results to JLD2 file.
"""
function save_robustness_results(results::ConcentrationRobustnessResults, filepath::String)
    # Save kinetic results first
    kinetic_file = replace(filepath, ".jld2" => "_kinetic.jld2")
    save_kinetic_results(results.kinetic_results, kinetic_file)

    # Save robustness-specific data
    JLD2.jldsave(filepath;
        kinetic_results_file=kinetic_file,
        robust_metabolites=results.robust_metabolites,
        robust_metabolite_pairs=results.robust_metabolite_pairs,
        n_robust_metabolites=results.n_robust_metabolites,
        n_robust_pairs=results.n_robust_pairs,
        giant_kinetic_module_size=results.giant_kinetic_module_size,
        compress=true
    )
end

"""
$(TYPEDSIGNATURES)

Load concentration robustness results from JLD2 file.
"""
function load_robustness_results(filepath::String)
    data = JLD2.load(filepath)
    kinetic_results = load_kinetic_results(data["kinetic_results_file"])

    return ConcentrationRobustnessResults(
        kinetic_results,
        data["robust_metabolites"],
        data["robust_metabolite_pairs"],
        data["n_robust_metabolites"],
        data["n_robust_pairs"],
        data["giant_kinetic_module_size"],
        nothing  # summary will be populated by analysis functions
    )
end

"""
$(TYPEDSIGNATURES)

Identify kinetic modules from concordance analysis results using constraints.

Takes concordance analysis results and constraints to identify kinetic modules
according to the kinetic modules paper, using the same split reactions as concordance.

# Arguments
- `constraints`: Constraint tree from `concordance_constraints`
- `concordance_results`: Results from `activity_concordance_analysis`
- `min_module_size::Int=2`: Minimum size for a kinetic module
- `workers=D.workers()`: Worker processes for parallel computation

# Returns
`KineticModuleResults` containing kinetic modules and network topology.
"""
function identify_kinetic_modules(
    constraints,
    model::AbstractFBCModels.AbstractFBCModel,
    concordance_results;
    min_module_size::Int=2,
    workers=D.workers()
)
    @info "Extracting network matrices from constraints"
    network_data = extract_network_matrices_from_constraints(constraints, model)

    @info "Identifying kinetic modules from concordance results"

    # Extract concordance modules from results
    concordance_modules_df = concordance_results.modules
    complexes_df = concordance_results.complexes

    # STEP 1: Create class_with_balanced groups (equivalent to MATLAB preprocessing)
    # This groups mutually concordant complexes including balanced complexes
    class_with_balanced = create_class_with_balanced_groups(concordance_results, network_data)
    
    @info "Created class_with_balanced groups" n_groups = length(class_with_balanced) group_sizes = [length(g) for g in class_with_balanced]

    # STEP 2: Find largest class_with_balanced group (equivalent to maxj in R)
    largest_group_idx = 1
    largest_group_size = length(class_with_balanced[1])
    for i in 2:length(class_with_balanced)
        if length(class_with_balanced[i]) > largest_group_size
            largest_group_size = length(class_with_balanced[i])
            largest_group_idx = i
        end
    end
    
    @info "Found largest group" largest_group_idx = largest_group_idx largest_group_size = largest_group_size

    # STEP 3: Process smaller groups with largest group using Upstream Algorithm
    candidate_modules = Vector{Vector{Int}}()
    
    @info "Processing smaller groups" n_smaller_groups = length(class_with_balanced) - 1
    
    for i in 1:length(class_with_balanced)
        # Skip the largest group (equivalent to R: if (i != maxj))
        if i != largest_group_idx
            @info "Processing group $i" group_size = length(class_with_balanced[i])
            
            # Combine smaller group with largest group
            combined_indices = union(class_with_balanced[i], class_with_balanced[largest_group_idx])
            @info "Combined with largest group" combined_size = length(combined_indices)
            
            # Apply Upstream Algorithm to combined set
            removed_indices = apply_r_autonomy_filter(combined_indices, network_data.A_matrix)
            remaining_indices = setdiff(combined_indices, removed_indices)
            @info "After Upstream Algorithm" removed_count = length(removed_indices) remaining_count = length(remaining_indices)
            
            if length(remaining_indices) >= min_module_size
                push!(candidate_modules, remaining_indices)
                @info "Added candidate module" module_size = length(remaining_indices)
            else
                @info "Rejected small module" module_size = length(remaining_indices) min_size = min_module_size
            end
        end
    end
    
    @info "Created candidate modules" n_candidates = length(candidate_modules)

    # STEP 4: Add pure balanced linkage classes
    candidate_modules = add_pure_balanced_linkage_classes(candidate_modules, class_with_balanced[largest_group_idx], network_data)

    # STEP 5: Merge overlapping modules via overlap graph
    final_modules = merge_overlapping_modules(candidate_modules)

    # STEP 6: Convert to kinetic modules dictionary
    kinetic_modules = Dict{Symbol,Vector{Symbol}}()
    kinetic_module_counter = 1
    
    # Convert final modules to kinetic modules dictionary
    for (i, module_indices) in enumerate(final_modules)
        if length(module_indices) >= min_module_size
            # Convert indices back to complex names
            upstream_names = Symbol[]
            for idx in module_indices
                # Find complex name from index
                complex_name = nothing
                for (name, idx_val) in network_data.complex_to_idx
                    if idx_val == idx
                        complex_name = name
                        break
                    end
                end
                if complex_name !== nothing
                    push!(upstream_names, Symbol(complex_name))
                end
            end

            if !isempty(upstream_names)
                # Sort for deterministic output
                sort!(upstream_names)

                # Create unique kinetic module name
                kinetic_module_name = Symbol("kinetic_module_$kinetic_module_counter")
                kinetic_modules[kinetic_module_name] = upstream_names
                kinetic_module_counter += 1
            end
        end
    end

    # Find giant kinetic module (largest by complex count)
    giant_module_id = :none
    max_size = 0
    for (module_id, complexes) in kinetic_modules
        if length(complexes) > max_size
            max_size = length(complexes)
            giant_module_id = module_id
        end
    end

    @info "Kinetic module identification completed" n_modules = length(kinetic_modules) giant_size = max_size

    # Find interface reactions
    @info "Finding interface reactions"
    interface_reactions = find_interface_reactions(
        kinetic_modules,
        network_data.A_matrix,
        network_data.complex_to_idx,
        network_data.reaction_names
    )
    # Sort for deterministic output
    sort!(interface_reactions)

    @info "Kinetic module identification complete" n_interface_reactions = length(interface_reactions)

    return KineticModuleResults(
        concordance_results,
        kinetic_modules,
        giant_module_id,
        interface_reactions,
        network_data.Y_matrix,
        network_data.A_matrix,
        network_data.metabolite_names,
        network_data.reaction_names,
        network_data.complex_to_idx,
        network_data.metabolite_to_idx,
        network_data.reaction_to_idx,
        nothing  # summary will be populated by analysis functions
    )
end

"""
$(TYPEDSIGNATURES)

Identify metabolites with concentration robustness from kinetic modules.

Analyzes kinetic modules to find:
1. Metabolites with absolute concentration robustness (single metabolite differences)
2. Metabolite pairs with concentration ratio robustness (two metabolite differences)

# Arguments
- `kinetic_results::KineticModuleResults`: Results from `identify_kinetic_modules`
- `include_pairs::Bool=true`: Whether to analyze metabolite pairs

# Returns
`ConcentrationRobustnessResults` containing robust metabolites and pairs.
"""
function identify_concentration_robustness(
    kinetic_results::KineticModuleResults;
    include_pairs::Bool=true
)
    @info "Analyzing concentration robustness"

    robust_metabolites = Set{String}()
    robust_metabolite_pairs = Set{Tuple{String,String}}()
    giant_kinetic_module_size = 0

    Y_matrix = kinetic_results.Y_matrix
    metabolite_names = kinetic_results.metabolite_names
    complex_to_idx = kinetic_results.complex_to_idx

    # Analyze each kinetic module
    for (module_id, complex_ids) in kinetic_results.kinetic_modules
        if length(complex_ids) < 2
            continue
        end

        giant_kinetic_module_size = max(giant_kinetic_module_size, length(complex_ids))

        @debug "Analyzing module $module_id" n_complexes = length(complex_ids)

        # Get indices for this module's complexes
        complex_indices = Int[]
        for complex_id in complex_ids
            complex_name = string(complex_id)
            if haskey(complex_to_idx, complex_name)
                push!(complex_indices, complex_to_idx[complex_name])
            end
        end

        if length(complex_indices) < 2
            continue
        end

        # Compare all pairs within this module
        for i in 1:(length(complex_indices)-1)
            for j in (i+1):length(complex_indices)
                idx1, idx2 = complex_indices[i], complex_indices[j]

                # Get stoichiometry vectors for both complexes
                vec1 = Y_matrix[:, idx1]
                vec2 = Y_matrix[:, idx2]

                # Find differences
                diff = vec1 - vec2
                differing_indices = findall(!iszero, diff)

                n_differences = length(differing_indices)

                if n_differences == 1
                    # Single metabolite difference = absolute concentration robustness
                    met_idx = differing_indices[1]
                    if met_idx <= length(metabolite_names)
                        met_name = metabolite_names[met_idx]
                        push!(robust_metabolites, met_name)
                    end

                elseif n_differences == 2 && include_pairs
                    # Two metabolite differences = concentration ratio robustness
                    # Apply upstream algorithm's strict structural validation (qq1, qq2, qq3)
                    # These conditions ensure the complexes have minimal structure required for robustness:
                    # - qq1: Exactly 2 metabolite differences between complexes
                    # - qq2: Complex 1 participates in exactly 1 additional metabolite (beyond the 2 differences)  
                    # - qq3: Complex 2 participates in exactly 1 additional metabolite (beyond the 2 differences)
                    # This validates the specific stoichiometric relationship needed for ratio robustness
                    met_idx1, met_idx2 = differing_indices

                    if met_idx1 <= length(metabolite_names) && met_idx2 <= length(metabolite_names)
                        # qq1: Exactly 2 differences (already confirmed above)
                        qq1 = true

                        # qq2: Complex 1 has exactly 1 additional non-zero metabolite beyond the differences
                        non_zero_vec1 = findall(!iszero, vec1)
                        qq2 = length(setdiff(non_zero_vec1, differing_indices)) == 1

                        # qq3: Complex 2 has exactly 1 additional non-zero metabolite beyond the differences  
                        non_zero_vec2 = findall(!iszero, vec2)
                        qq3 = length(setdiff(non_zero_vec2, differing_indices)) == 1

                        # Only accept pairs that satisfy ALL three conditions (upstream algorithm logic)
                        if qq1 && qq2 && qq3
                            met_name1 = metabolite_names[met_idx1]
                            met_name2 = metabolite_names[met_idx2]

                            # Store pair in canonical order
                            pair = met_name1 < met_name2 ? (met_name1, met_name2) : (met_name2, met_name1)
                            push!(robust_metabolite_pairs, pair)
                        end
                    end
                end
            end
        end
    end

    # Sort for deterministic output
    robust_metabolites_vec = sort(collect(robust_metabolites))
    robust_pairs_vec = sort(collect(robust_metabolite_pairs))

    @info "Concentration robustness analysis complete" (
        n_robust_metabolites=length(robust_metabolites_vec),
        n_robust_pairs=length(robust_pairs_vec),
        largest_module_size=giant_kinetic_module_size
    )

    return ConcentrationRobustnessResults(
        kinetic_results,
        robust_metabolites_vec,
        robust_pairs_vec,
        length(robust_metabolites_vec),
        length(robust_pairs_vec),
        giant_kinetic_module_size,
        nothing  # summary will be populated by analysis functions
    )
end

"""
$(TYPEDSIGNATURES)

Complete kinetic concordance analysis pipeline with constraint-based workflow.

Enhanced version that returns results with a comprehensive summary field containing
all key metrics for analyzing 343 models efficiently.

# Arguments
- `model`: The metabolic model
- `optimizer`: Optimization solver
- `include_kinetic_modules::Bool=true`: Whether to identify kinetic modules
- `include_robustness::Bool=true`: Whether to analyze concentration robustness
- `min_module_size::Int=2`: Minimum size for kinetic modules
- `kwargs...`: Additional arguments passed to `activity_concordance_analysis`

# Returns
Enhanced results with `.summary` field containing:
- `n_reactions`: Number of reactions
- `n_metabolites`: Number of metabolites  
- `n_complexes`: Number of complexes
- `n_concordant_pairs`: Number of concordant pairs
- `n_balanced_complexes`: Number of balanced complexes
- `n_trivially_balanced`: Number of trivially balanced complexes
- `n_trivially_concordant`: Number of trivially concordant pairs
- `n_concordance_modules`: Number of concordance modules
- `n_kinetic_modules`: Number of kinetic modules
- `giant_kinetic_module_size`: Size of largest kinetic module
- `n_metabolites_absolute_robust`: Metabolites with absolute concentration robustness
- `n_metabolite_pairs_ratio_robust`: Metabolite pairs with concentration ratio robustness
"""
function kinetic_concordance_analysis(
    model;
    optimizer,
    include_kinetic_modules::Bool=true,
    include_robustness::Bool=true,
    min_module_size::Int=2,
    workers=D.workers(),
    kwargs...
)
    @info "Starting kinetic concordance analysis pipeline"

    # Step 1: Run concordance analysis (this builds constraints internally)
    @info "Running concordance analysis"
    concordance_results = activity_concordance_analysis(model; optimizer=optimizer, workers=workers, kwargs...)

    # Extract basic model statistics
    n_reactions = length(AbstractFBCModels.reactions(model))
    n_metabolites = length(AbstractFBCModels.metabolites(model))
    n_complexes = concordance_results.stats["n_complexes"]

    # Extract concordance statistics
    n_concordant_pairs = concordance_results.stats["n_concordant_total"]
    n_balanced_complexes = concordance_results.stats["n_balanced"]
    n_trivially_balanced = concordance_results.stats["n_trivially_balanced"]
    n_trivially_concordant = concordance_results.stats["n_trivial_pairs"]
    n_concordance_modules = concordance_results.stats["n_modules"]

    # Initialize kinetic and robustness metrics
    n_kinetic_modules = 0
    giant_kinetic_module_size = 0
    n_metabolites_absolute_robust = 0
    n_metabolite_pairs_ratio_robust = 0

    # Initialize results to return
    final_results = concordance_results

    if !include_kinetic_modules
        @info "Kinetic concordance analysis complete (concordance only)"
    else
        # Step 2: Build constraints separately for kinetic analysis
        @info "Building concordance constraints for kinetic analysis"

        # Filter kwargs to only include those accepted by concordance_constraints
        constraints_kwargs = Dict(
            :modifications => get(kwargs, :modifications, Function[]),
            :interface => get(kwargs, :interface, nothing),
            :use_unidirectional_constraints => get(kwargs, :use_unidirectional_constraints, true)
        )

        constraints = concordance_constraints(model; constraints_kwargs...)

        # Step 3: Run kinetic module analysis using constraints
        @info "Running kinetic module analysis"
        kinetic_results = identify_kinetic_modules(constraints, model, concordance_results; min_module_size, workers)

        # Extract kinetic module statistics
        n_kinetic_modules = length(kinetic_results.kinetic_modules)
        giant_kinetic_module_size = if kinetic_results.giant_module_id != :none
            length(kinetic_results.kinetic_modules[kinetic_results.giant_module_id])
        else
            0
        end

        if !include_robustness
            @info "Kinetic concordance analysis complete (no robustness analysis)"
        else
            # Step 4: Run concentration robustness analysis
            @info "Running concentration robustness analysis"
            robustness_results = identify_concentration_robustness(kinetic_results)

            # Extract robustness statistics
            n_metabolites_absolute_robust = robustness_results.n_robust_metabolites
            n_metabolite_pairs_ratio_robust = robustness_results.n_robust_pairs
        end
    end

    # Create comprehensive summary with all requested metrics
    summary = Dict{String,Int}(
        "n_reactions" => n_reactions,
        "n_metabolites" => n_metabolites,
        "n_complexes" => n_complexes,
        "n_concordant_pairs" => n_concordant_pairs,
        "n_balanced_complexes" => n_balanced_complexes,
        "n_trivially_balanced" => n_trivially_balanced,
        "n_trivially_concordant" => n_trivially_concordant,
        "n_concordance_modules" => n_concordance_modules,
        "n_kinetic_modules" => n_kinetic_modules,
        "giant_kinetic_module_size" => giant_kinetic_module_size,
        "n_metabolites_absolute_robust" => n_metabolites_absolute_robust,
        "n_metabolite_pairs_ratio_robust" => n_metabolite_pairs_ratio_robust
    )

    @info "Kinetic concordance analysis pipeline complete"
    @info "Analysis summary" summary

    # Return results with improved structure - kinetic_results and concordance_results at top level
    if include_robustness && include_kinetic_modules
        return (
            concordance_results=concordance_results,
            kinetic_results=(
                kinetic_modules=robustness_results.kinetic_results.kinetic_modules,
                giant_module_id=robustness_results.kinetic_results.giant_module_id,
                interface_reactions=robustness_results.kinetic_results.interface_reactions,
                Y_matrix=robustness_results.kinetic_results.Y_matrix,
                A_matrix=robustness_results.kinetic_results.A_matrix,
                metabolite_names=robustness_results.kinetic_results.metabolite_names,
                reaction_names=robustness_results.kinetic_results.reaction_names,
                complex_to_idx=robustness_results.kinetic_results.complex_to_idx,
                metabolite_to_idx=robustness_results.kinetic_results.metabolite_to_idx,
                reaction_to_idx=robustness_results.kinetic_results.reaction_to_idx
            ),
            robust_metabolites=robustness_results.robust_metabolites,
            robust_metabolite_pairs=robustness_results.robust_metabolite_pairs,
            n_robust_metabolites=robustness_results.n_robust_metabolites,
            n_robust_pairs=robustness_results.n_robust_pairs,
            giant_kinetic_module_size=robustness_results.giant_kinetic_module_size,
            summary=summary
        )
    elseif include_kinetic_modules
        return (
            concordance_results=concordance_results,
            kinetic_results=(
                kinetic_modules=kinetic_results.kinetic_modules,
                giant_module_id=kinetic_results.giant_module_id,
                interface_reactions=kinetic_results.interface_reactions,
                Y_matrix=kinetic_results.Y_matrix,
                A_matrix=kinetic_results.A_matrix,
                metabolite_names=kinetic_results.metabolite_names,
                reaction_names=kinetic_results.reaction_names,
                complex_to_idx=kinetic_results.complex_to_idx,
                metabolite_to_idx=kinetic_results.metabolite_to_idx,
                reaction_to_idx=kinetic_results.reaction_to_idx
            ),
            summary=summary
        )
    else
        # Concordance results only - convert to named tuple with summary
        return (
            complexes=final_results.complexes,
            modules=final_results.modules,
            lambdas=final_results.lambdas,
            stats=final_results.stats,
            summary=summary
        )
    end
end