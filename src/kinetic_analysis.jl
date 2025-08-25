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
    largest_robust_module_size::Int

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
                        # Check if product is in a different SCC - if yes, this SCC is not terminal
                        if !(prod_node in scc)
                            is_terminal = false
                            break
                        end
                    end
                end
                if !is_terminal
                    break
                end
            end
            if !is_terminal
                break
            end
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
        largest_robust_module_size=results.largest_robust_module_size,
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
        data["largest_robust_module_size"],
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

    # Convert concordance modules to kinetic modules using autonomy filtering
    kinetic_modules = Dict{Symbol,Vector{Symbol}}()

    # Process each concordance module
    for row in DF.eachrow(concordance_modules_df)
        module_id = Symbol(row.module_id)
        # Handle both string and vector format for complexes
        complex_names = if isa(row.complexes, String)
            split(row.complexes, ", ")
        else
            row.complexes  # Assume it's already a vector
        end

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
                # Find complex name from index - correct lookup
                complex_name = nothing
                for (name, idx_val) in network_data.complex_to_idx
                    if idx_val == idx
                        complex_name = name
                        break
                    end
                end
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
    # Look for balanced complexes in the complexes DataFrame
    balanced_complexes = DF.filter(row -> row.module == "balanced", complexes_df)
    if DF.nrow(balanced_complexes) >= min_module_size
        balanced_names = Symbol[]
        for row in DF.eachrow(balanced_complexes)
            push!(balanced_names, Symbol(row.id))
        end
        # Sort for deterministic output
        sort!(balanced_names)
        kinetic_modules[:balanced] = balanced_names
    end

    # Apply module merging logic
    @info "Applying module merging logic"
    if !isempty(kinetic_modules)
        module_list = collect(values(kinetic_modules))
        module_ids = collect(keys(kinetic_modules))
        n_modules = length(module_list)

        # Build overlap matrix
        overlap_matrix = zeros(Int, n_modules, n_modules)
        for i in 1:(n_modules-1)
            for j in (i+1):n_modules
                overlap = length(intersect(Set(module_list[i]), Set(module_list[j])))
                overlap_matrix[i, j] = overlap_matrix[j, i] = overlap
            end
        end

        # Find connected components of overlapping modules
        # Any overlap > 0 means modules should be merged
        adj_matrix = overlap_matrix .> 0

        # Simple connected components algorithm
        visited = falses(n_modules)
        merged_modules = Dict{Symbol,Vector{Symbol}}()
        singleton_complexes = Set{Symbol}()

        # Collect all complexes that are in modules
        all_module_complexes = Set{Symbol}()
        for complex_list in module_list
            union!(all_module_complexes, Set(complex_list))
        end

        for i in 1:n_modules
            if !visited[i]
                # Find connected component starting from i
                component = [i]
                queue = [i]
                visited[i] = true

                while !isempty(queue)
                    current = popfirst!(queue)
                    for j in 1:n_modules
                        if !visited[j] && adj_matrix[current, j]
                            push!(component, j)
                            push!(queue, j)
                            visited[j] = true
                        end
                    end
                end

                # Merge all modules in this component
                merged_complexes = Vector{Symbol}()
                for module_idx in component
                    append!(merged_complexes, module_list[module_idx])
                end

                # Remove duplicates and sort
                merged_complexes = sort(collect(Set(merged_complexes)))

                if length(merged_complexes) >= min_module_size
                    # Use first module ID as the merged module ID
                    merged_module_id = module_ids[component[1]]
                    merged_modules[merged_module_id] = merged_complexes
                else
                    # Add to singletons if too small
                    union!(singleton_complexes, Set(merged_complexes))
                end
            end
        end

        # Handle singleton complexes (those not in any kinetic module)
        # Add complexes that passed autonomy filtering but ended up as singletons
        for complex_symbol in singleton_complexes
            if length([complex_symbol]) >= 1  # Each singleton is its own "module"
                singleton_id = Symbol("singleton_", complex_symbol)
                merged_modules[singleton_id] = [complex_symbol]
            end
        end

        kinetic_modules = merged_modules
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

    @info "Identified kinetic modules after merging" n_modules = length(kinetic_modules) giant_size = max_size

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
        largest_module_size=largest_robust_module_size
    )

    return ConcentrationRobustnessResults(
        kinetic_results,
        robust_metabolites_vec,
        robust_pairs_vec,
        length(robust_metabolites_vec),
        length(robust_pairs_vec),
        largest_robust_module_size,
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

    # Return results with summary - handle different result types
    if include_robustness && include_kinetic_modules
        return ConcentrationRobustnessResults(
            robustness_results.kinetic_results,
            robustness_results.robust_metabolites,
            robustness_results.robust_metabolite_pairs,
            robustness_results.n_robust_metabolites,
            robustness_results.n_robust_pairs,
            robustness_results.largest_robust_module_size,
            summary
        )
    elseif include_kinetic_modules
        return KineticModuleResults(
            kinetic_results.concordance_results,
            kinetic_results.kinetic_modules,
            kinetic_results.giant_module_id,
            kinetic_results.interface_reactions,
            kinetic_results.Y_matrix,
            kinetic_results.A_matrix,
            kinetic_results.metabolite_names,
            kinetic_results.reaction_names,
            kinetic_results.complex_to_idx,
            kinetic_results.metabolite_to_idx,
            kinetic_results.reaction_to_idx,
            summary
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