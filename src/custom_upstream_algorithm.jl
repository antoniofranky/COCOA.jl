"""
COCOA.jl Custom Upstream Algorithm

Optimized 4-phase upstream algorithm designed specifically for COCOA.jl's 
CompleteConcordanceModel data structure and performance requirements.

Based on theoretical foundations but adapted for practical use with:
- Large metabolic networks (50k+ reactions)
- Sparse matrix operations
- Memory efficiency
- Integration with CompleteConcordanceModel
"""

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
Find strongly connected components using Tarjan's algorithm.
Optimized for sparse matrices and large networks.
"""
function find_strongly_connected_components(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    if isempty(complexes)
        return Vector{Int}[]
    end

    n_complexes = length(complexes)
    complex_to_local = Dict(complexes[i] => i for i in 1:n_complexes)

    # Build adjacency list for efficiency
    adjacency = [Int[] for _ in 1:n_complexes]

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
                        push!(adjacency[local_i], local_j)
                    end
                end
            end
        end
    end

    # Tarjan's algorithm
    index_counter = [0]
    stack = Int[]
    indices = zeros(Int, n_complexes)
    lowlinks = zeros(Int, n_complexes)
    on_stack = falses(n_complexes)
    components = Vector{Int}[]

    function strongconnect(v)
        indices[v] = index_counter[1]
        lowlinks[v] = index_counter[1]
        index_counter[1] += 1
        push!(stack, v)
        on_stack[v] = true

        for w in adjacency[v]
            if indices[w] == 0
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elseif on_stack[w]
                lowlinks[v] = min(lowlinks[v], indices[w])
            end
        end

        if lowlinks[v] == indices[v]
            component = Int[]
            while true
                w = pop!(stack)
                on_stack[w] = false
                push!(component, complexes[w])  # Convert back to global indices
                if w == v
                    break
                end
            end
            push!(components, component)
        end
    end

    for v in 1:n_complexes
        if indices[v] == 0
            strongconnect(v)
        end
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
COCOA.jl Custom 4-Phase Upstream Algorithm

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
function custom_upstream_algorithm(extended_module::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    if isempty(extended_module)
        return Int[]
    end

    # Phase I: Iteratively exclude entry complexes (autonomous property)
    current_set = copy(extended_module)
    phase_i_excluded = Int[]

    while true
        entry_complexes = find_entry_complexes(current_set, A_matrix)
        if isempty(entry_complexes)
            break  # Achieved autonomous property
        end

        append!(phase_i_excluded, entry_complexes)
        current_set = setdiff(current_set, entry_complexes)

        if isempty(current_set)
            return Int[]  # All complexes excluded
        end
    end

    # Phase II: Exclude non-reactant complexes (feeding property)
    nonreactant = find_nonreactant_complexes(current_set, A_matrix)
    phase_ii_excluded = nonreactant
    current_set = setdiff(current_set, nonreactant)

    if isempty(current_set)
        return Int[]
    end

    # Phase III: Iteratively exclude exit complexes (feeding property) 
    phase_iii_complexes = Int[]

    while true
        exit_complexes = find_exit_complexes(current_set, A_matrix)
        if isempty(exit_complexes)
            break  # No more exit complexes
        end

        append!(phase_iii_complexes, exit_complexes)
        current_set = setdiff(current_set, exit_complexes)

        if isempty(current_set)
            break
        end
    end

    # Phase IV: Exclude terminal strongly connected components (feeding property)
    phase_iv_complexes = Int[]

    if !isempty(current_set)
        components = find_strongly_connected_components(current_set, A_matrix)

        for component in components
            if !is_terminal_component(component, current_set, A_matrix)
                # Non-terminal component - these are part of kinetic module
                append!(phase_iv_complexes, component)
            end
            # Terminal components are excluded (not added to phase_iv_complexes)
        end
    end

    # Return union of Phase III and Phase IV complexes (the upstream set)
    # This matches the theoretical specification and R implementation
    upstream_set = union(phase_iii_complexes, phase_iv_complexes)

    return upstream_set
end

"""
Apply custom upstream algorithm to identify kinetic modules in CompleteConcordanceModel.

Updates the model's kinetic_modules field efficiently.
"""
function apply_custom_kinetic_analysis!(complete_model::CompleteConcordanceModel; min_module_size::Int=2)
    @info "Applying custom upstream algorithm for kinetic module identification"

    # Reset kinetic modules
    fill!(complete_model.kinetic_modules, 0)

    # Get balanced complexes (concordance module 0)
    balanced_complexes = findall(==(0), complete_model.concordance_modules)

    # Process each non-balanced concordance module
    kinetic_module_count = 0
    total_kinetic_complexes = 0

    for concordance_module_id in 1:complete_model.n_concordance_modules
        if concordance_module_id == 0
            continue  # Skip balanced complexes module
        end

        # Get complexes in this concordance module
        concordance_complexes = findall(==(concordance_module_id), complete_model.concordance_modules)

        if isempty(concordance_complexes)
            continue
        end

        # Form extended module (balanced + this concordance module)
        extended_module = union(balanced_complexes, concordance_complexes)

        # Apply custom upstream algorithm
        kinetic_complexes = custom_upstream_algorithm(extended_module, complete_model.complex_reaction_matrix)

        # Only create kinetic module if it meets minimum size requirement
        if length(kinetic_complexes) >= min_module_size
            kinetic_module_count += 1
            total_kinetic_complexes += length(kinetic_complexes)

            # Update kinetic_modules field
            for complex_idx in kinetic_complexes
                complete_model.kinetic_modules[complex_idx] = kinetic_module_count
            end
        end
    end

    # Update model statistics
    complete_model.n_kinetic_modules = kinetic_module_count

    @info "Custom kinetic analysis completed: $kinetic_module_count modules, $total_kinetic_complexes total complexes"

    return kinetic_module_count
end