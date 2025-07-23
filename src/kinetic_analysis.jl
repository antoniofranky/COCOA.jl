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
    Y_matrix::SparseArrays.SparseMatrixCSC{Int8}  # Species-complex matrix (m × n)
    A_matrix::SparseArrays.SparseMatrixCSC{Int8}  # Complex-reaction matrix (n × r)
    
    # Index mappings for network position queries
    metabolite_names::Vector{String}
    reaction_names::Vector{String}
    complex_to_idx::Dict{String,Int}
    metabolite_to_idx::Dict{String,Int}
    reaction_to_idx::Dict{String,Int}
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
    largest_robust_module_size::Int
end

"""
$(TYPEDSIGNATURES)

Extract Y (species-complex) and A (complex-reaction) matrices from model.
Uses memory-efficient Int8 storage for stoichiometric coefficients.
"""
function extract_network_matrices(model)
    # Get model information
    metabolites = AbstractFBCModels.metabolites(model)
    reactions = AbstractFBCModels.reactions(model)
    
    # Build species-reaction stoichiometric matrix first
    S = AbstractFBCModels.stoichiometry(model)
    
    # For now, we need to extract complexes from the model
    # This is a simplified approach - in a full implementation,
    # we'd extract the actual complex structure from the model
    n_metabolites = length(metabolites)
    n_reactions = length(reactions)
    
    # Create simplified Y matrix where each reaction corresponds to substrate/product complexes
    # This is a placeholder - real implementation would parse reaction complexes
    complexes = Vector{String}()
    Y_rows = Vector{Int}()
    Y_cols = Vector{Int}()
    Y_vals = Vector{Int8}()
    
    complex_idx = 1
    A_rows = Vector{Int}()
    A_cols = Vector{Int}()
    A_vals = Vector{Int8}()
    
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
                push!(Y_vals, Int8(-rxn_stoich[met_idx]))
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
                push!(Y_vals, Int8(rxn_stoich[met_idx]))
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
    
    # Create index mappings
    complex_to_idx = Dict(complexes[i] => i for i in eachindex(complexes))
    metabolite_to_idx = Dict(metabolites[i] => i for i in eachindex(metabolites))
    reaction_to_idx = Dict(reactions[i] => i for i in eachindex(reactions))
    
    return (
        Y_matrix = Y_matrix,
        A_matrix = A_matrix,
        metabolite_names = collect(metabolites),
        complex_names = complexes,
        reaction_names = collect(reactions),
        complex_to_idx = complex_to_idx,
        metabolite_to_idx = metabolite_to_idx,
        reaction_to_idx = reaction_to_idx
    )
end

"""
$(TYPEDSIGNATURES)

Apply the four-phase autonomy algorithm from the kinetic modules paper.
Filters complexes to identify autonomous sets that form kinetic modules.
"""
function apply_four_phase_autonomy_filter(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(complex_indices)
        return Int[]
    end
    
    current_set = Set(complex_indices)
    
    # Build adjacency information from A matrix
    # A[i,j] = -1 means complex i is substrate of reaction j
    # A[i,j] = 1 means complex i is product of reaction j
    
    # Phase I: Remove complexes with external inputs
    # A complex has external input if it's a product of reactions whose substrates are outside the set
    phase_1_removed = Set{Int}()
    for complex_idx in current_set
        # Find reactions where this complex is a product
        product_reactions = findall(x -> x > 0, A_matrix[complex_idx, :])
        
        for rxn_idx in product_reactions
            # Find substrate complexes for this reaction
            substrate_complexes = findall(x -> x < 0, A_matrix[:, rxn_idx])
            
            # If any substrate is outside our set, this complex has external input
            if !all(sub_idx in current_set for sub_idx in substrate_complexes)
                push!(phase_1_removed, complex_idx)
                break
            end
        end
    end
    setdiff!(current_set, phase_1_removed)
    
    if isempty(current_set)
        return Int[]
    end
    
    # Phase II: Remove terminal complexes (no outgoing reactions)
    phase_2_removed = Set{Int}()
    for complex_idx in current_set
        # Check if complex has any outgoing reactions (is substrate of any reaction)
        substrate_reactions = findall(x -> x < 0, A_matrix[complex_idx, :])
        if isempty(substrate_reactions)
            push!(phase_2_removed, complex_idx)
        end
    end
    setdiff!(current_set, phase_2_removed)
    
    if isempty(current_set)
        return Int[]
    end
    
    # Phase III: Remove complexes with external outputs
    phase_3_removed = Set{Int}()
    for complex_idx in current_set
        # Find reactions where this complex is a substrate
        substrate_reactions = findall(x -> x < 0, A_matrix[complex_idx, :])
        
        for rxn_idx in substrate_reactions
            # Find product complexes for this reaction
            product_complexes = findall(x -> x > 0, A_matrix[:, rxn_idx])
            
            # If any product is outside our set, this complex has external output
            if !all(prod_idx in current_set for prod_idx in product_complexes)
                push!(phase_3_removed, complex_idx)
                break
            end
        end
    end
    setdiff!(current_set, phase_3_removed)
    
    if isempty(current_set)
        return Int[]
    end
    
    # Phase IV: Keep only non-terminal complexes in same strongly connected component
    # Build directed graph from remaining complexes
    remaining_indices = collect(current_set)
    n_remaining = length(remaining_indices)
    idx_map = Dict(remaining_indices[i] => i for i in 1:n_remaining)
    
    # Create adjacency matrix for remaining complexes
    adj_matrix = zeros(Bool, n_remaining, n_remaining)
    for (i, complex_i) in enumerate(remaining_indices)
        substrate_reactions = findall(x -> x < 0, A_matrix[complex_i, :])
        for rxn_idx in substrate_reactions
            product_complexes = findall(x -> x > 0, A_matrix[:, rxn_idx])
            for complex_j in product_complexes
                if complex_j in current_set && complex_j != complex_i
                    j = idx_map[complex_j]
                    adj_matrix[i, j] = true
                end
            end
        end
    end
    
    # Find strongly connected components
    graph = Graphs.SimpleDiGraph(adj_matrix)
    sccs = Graphs.strongly_connected_components(graph)
    
    # Keep only non-terminal complexes (those in SCCs with outgoing edges to other SCCs)
    terminal_complexes = Set{Int}()
    for scc in sccs
        is_terminal = true
        for node in scc
            complex_idx = remaining_indices[node]
            substrate_reactions = findall(x -> x < 0, A_matrix[complex_idx, :])
            for rxn_idx in substrate_reactions
                product_complexes = findall(x -> x > 0, A_matrix[:, rxn_idx])
                # Check if any products are in different SCCs
                for prod_idx in product_complexes
                    if prod_idx in current_set
                        prod_node = idx_map[prod_idx]
                        # Check if product is in different SCC
                        if !any(prod_node in other_scc for other_scc in sccs if other_scc != scc)
                            is_terminal = false
                            break
                        end
                    end
                end
                if !is_terminal break end
            end
            if !is_terminal break end
        end
        
        if is_terminal
            for node in scc
                push!(terminal_complexes, remaining_indices[node])
            end
        end
    end
    
    setdiff!(current_set, terminal_complexes)
    
    return collect(current_set)
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
            # Find complex name from index
            complex_name = findfirst(x -> x == complex_idx, complex_to_idx)
            if complex_name !== nothing && haskey(complex_to_module, complex_name)
                push!(substrate_modules, complex_to_module[complex_name])
            end
        end
        
        for complex_idx in product_complexes
            complex_name = findfirst(x -> x == complex_idx, complex_to_idx)
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
        concordance_complexes = results.concordance_results.complexes,
        concordance_modules = results.concordance_results.modules,
        concordance_lambdas = results.concordance_results.lambdas,
        concordance_stats = results.concordance_results.stats,
        
        # Kinetic results
        kinetic_modules = results.kinetic_modules,
        giant_module_id = results.giant_module_id,
        interface_reactions = results.interface_reactions,
        
        # Network matrices
        Y_matrix = results.Y_matrix,
        A_matrix = results.A_matrix,
        
        # Names and mappings
        metabolite_names = results.metabolite_names,
        reaction_names = results.reaction_names,
        complex_to_idx = results.complex_to_idx,
        metabolite_to_idx = results.metabolite_to_idx,
        reaction_to_idx = results.reaction_to_idx,
        
        compress = true
    )
end

"""
$(TYPEDSIGNATURES)

Load kinetic analysis results from JLD2 file.
"""
function load_kinetic_results(filepath::String)
    data = JLD2.load(filepath)
    
    concordance_results = (
        complexes = data["concordance_complexes"],
        modules = data["concordance_modules"],
        lambdas = data["concordance_lambdas"],
        stats = data["concordance_stats"]
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
        data["reaction_to_idx"]
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
        kinetic_results_file = kinetic_file,
        robust_metabolites = results.robust_metabolites,
        robust_metabolite_pairs = results.robust_metabolite_pairs,
        n_robust_metabolites = results.n_robust_metabolites,
        n_robust_pairs = results.n_robust_pairs,
        largest_robust_module_size = results.largest_robust_module_size,
        compress = true
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
        data["largest_robust_module_size"]
    )
end

"""
$(TYPEDSIGNATURES)

Identify kinetic modules from concordance analysis results.

Takes concordance analysis results and applies the four-phase autonomy algorithm
to identify kinetic modules according to the kinetic modules paper.

# Arguments
- `concordance_results`: Results from `concordance_analysis`
- `model`: The metabolic model
- `min_module_size::Int=2`: Minimum size for a kinetic module
- `workers=D.workers()`: Worker processes for parallel computation

# Returns
`KineticModuleResults` containing kinetic modules and network topology.
"""
function identify_kinetic_modules(
    concordance_results, 
    model;
    min_module_size::Int=2,
    workers=D.workers()
)
    @info "Extracting network matrices from model"
    network_data = extract_network_matrices(model)
    
    @info "Identifying kinetic modules from concordance results"
    
    # Extract concordance modules from results
    concordance_modules_df = concordance_results.modules
    complexes_df = concordance_results.complexes
    
    # Convert concordance modules to kinetic modules using autonomy filtering
    kinetic_modules = Dict{Symbol,Vector{Symbol}}()
    
    # Process each concordance module
    for row in eachrow(concordance_modules_df)
        module_id = Symbol(row.module_id)
        complex_names = split(row.complexes, ", ")
        
        if length(complex_names) < min_module_size
            continue
        end
        
        # Get complex indices
        complex_indices = Int[]
        for complex_name in complex_names
            if haskey(network_data.complex_to_idx, complex_name)
                push!(complex_indices, network_data.complex_to_idx[complex_name])
            end
        end
        
        if length(complex_indices) < min_module_size
            continue
        end
        
        # Apply four-phase autonomy filtering
        autonomous_indices = apply_four_phase_autonomy_filter(complex_indices, network_data.A_matrix)
        
        if length(autonomous_indices) >= min_module_size
            # Convert back to complex names
            autonomous_names = Symbol[]
            for idx in autonomous_indices
                # Find complex name from index
                complex_name = findfirst(x -> x == idx, network_data.complex_to_idx)
                if complex_name !== nothing
                    push!(autonomous_names, Symbol(complex_name))
                end
            end
            
            if !isempty(autonomous_names)
                # Sort for deterministic output
                sort!(autonomous_names)
                kinetic_modules[module_id] = autonomous_names
            end
        end
    end
    
    # Add balanced complexes as separate module if they exist
    balanced_complexes = filter(row -> row.is_balanced, complexes_df)
    if nrow(balanced_complexes) >= min_module_size
        balanced_names = Symbol[]
        for row in eachrow(balanced_complexes)
            push!(balanced_names, Symbol(row.complex_id))
        end
        # Sort for deterministic output
        sort!(balanced_names)
        kinetic_modules[:balanced] = balanced_names
    end
    
    # Find giant module (largest kinetic module)
    giant_module_id = :none
    max_size = 0
    for (module_id, complexes) in kinetic_modules
        if length(complexes) > max_size
            max_size = length(complexes)
            giant_module_id = module_id
        end
    end
    
    @info "Identified kinetic modules" n_modules = length(kinetic_modules) giant_size = max_size
    
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
        network_data.reaction_to_idx
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
    largest_robust_module_size = 0
    
    Y_matrix = kinetic_results.Y_matrix
    metabolite_names = kinetic_results.metabolite_names
    complex_to_idx = kinetic_results.complex_to_idx
    
    # Analyze each kinetic module
    for (module_id, complex_ids) in kinetic_results.kinetic_modules
        if length(complex_ids) < 2 
            continue
        end
        
        largest_robust_module_size = max(largest_robust_module_size, length(complex_ids))
        
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
                    # Check that complexes share all but two metabolites
                    met_idx1, met_idx2 = differing_indices
                    
                    if met_idx1 <= length(metabolite_names) && met_idx2 <= length(metabolite_names)
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
    
    # Sort for deterministic output
    robust_metabolites_vec = sort(collect(robust_metabolites))
    robust_pairs_vec = sort(collect(robust_metabolite_pairs))
    
    @info "Concentration robustness analysis complete" (
        n_robust_metabolites = length(robust_metabolites_vec),
        n_robust_pairs = length(robust_pairs_vec),
        largest_module_size = largest_robust_module_size
    )
    
    return ConcentrationRobustnessResults(
        kinetic_results,
        robust_metabolites_vec,
        robust_pairs_vec,
        length(robust_metabolites_vec),
        length(robust_pairs_vec),
        largest_robust_module_size
    )
end

"""
$(TYPEDSIGNATURES)

Complete kinetic concordance analysis pipeline.

Convenience wrapper that runs concordance analysis followed by kinetic module
identification and concentration robustness analysis.

# Arguments
- `model`: The metabolic model
- `optimizer`: Optimization solver
- `include_kinetic_modules::Bool=true`: Whether to identify kinetic modules
- `include_robustness::Bool=true`: Whether to analyze concentration robustness
- `min_module_size::Int=2`: Minimum size for kinetic modules
- `kwargs...`: Additional arguments passed to `concordance_analysis`

# Returns
Depending on options:
- Concordance results only (if `include_kinetic_modules=false`)
- `KineticModuleResults` (if `include_robustness=false`)  
- `ConcentrationRobustnessResults` (if both options true)
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
    
    # Run existing concordance analysis
    @info "Running concordance analysis"
    concordance_results = concordance_analysis(model; optimizer=optimizer, workers=workers, kwargs...)
    
    if !include_kinetic_modules
        @info "Kinetic concordance analysis complete (concordance only)"
        return concordance_results
    end
    
    # Add kinetic module analysis
    @info "Running kinetic module analysis"
    kinetic_results = identify_kinetic_modules(concordance_results, model; min_module_size, workers)
    
    if !include_robustness
        @info "Kinetic concordance analysis complete (no robustness analysis)"
        return kinetic_results
    end
    
    # Add concentration robustness analysis
    @info "Running concentration robustness analysis"
    robustness_results = identify_concentration_robustness(kinetic_results)
    
    @info "Kinetic concordance analysis pipeline complete"
    return robustness_results
end