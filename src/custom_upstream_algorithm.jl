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

"""
Comprehensive validation function for iterative testing of custom upstream algorithm phases.
Returns detailed validation report for debugging and verification.
"""
function validate_custom_upstream_algorithm(complete_model::CompleteConcordanceModel; verbose::Bool=true)
    @info "Starting comprehensive validation of custom upstream algorithm"
    
    validation_results = Dict{String, Any}()
    
    # Get test data
    balanced_complexes = findall(==(0), complete_model.concordance_modules)
    A_matrix = complete_model.complex_reaction_matrix
    
    validation_results["n_balanced"] = length(balanced_complexes)
    validation_results["n_concordance_modules"] = complete_model.n_concordance_modules
    validation_results["matrix_size"] = size(A_matrix)
    
    verbose && @info "Validation setup" n_balanced=length(balanced_complexes) n_concordance_modules=complete_model.n_concordance_modules matrix_size=size(A_matrix)
    
    # Test each concordance module
    module_results = Dict{Int, Any}()
    
    for concordance_module_id in 1:complete_model.n_concordance_modules
        if concordance_module_id == 0
            continue
        end
        
        concordance_complexes = findall(==(concordance_module_id), complete_model.concordance_modules)
        
        if isempty(concordance_complexes)
            continue
        end
        
        extended_module = union(balanced_complexes, concordance_complexes)
        
        verbose && @info "Testing concordance module $concordance_module_id" n_complexes=length(concordance_complexes) extended_size=length(extended_module)
        
        # Phase-by-phase validation
        phase_results = validate_algorithm_phases(extended_module, A_matrix, verbose)
        module_results[concordance_module_id] = phase_results
        
        # Final algorithm result
        kinetic_complexes = custom_upstream_algorithm(extended_module, A_matrix)
        phase_results["final_kinetic_size"] = length(kinetic_complexes)
        phase_results["kinetic_complexes"] = kinetic_complexes
        
        verbose && @info "Module $concordance_module_id result" kinetic_size=length(kinetic_complexes)
    end
    
    validation_results["module_results"] = module_results
    
    # Overall summary
    total_kinetic_complexes = sum(result["final_kinetic_size"] for result in values(module_results))
    modules_with_kinetic = count(result["final_kinetic_size"] > 0 for result in values(module_results))
    
    validation_results["summary"] = (
        total_modules_tested = length(module_results),
        modules_with_kinetic = modules_with_kinetic,
        total_kinetic_complexes = total_kinetic_complexes
    )
    
    @info "Validation complete" summary=validation_results["summary"]
    
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
            verbose && @info "  ✓ Complex $complex_idx is an EXIT complex (connects to external: $external_products)"
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
        outgoing_connections = Dict{Int, Vector{Int}}()
        
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
        
        verbose && @info "Component $i analysis" is_terminal=is_terminal outgoing_connections=outgoing_connections
        
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
    phase_results = Dict{String, Any}()
    
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
        
        verbose && @info "Phase I iteration $phase_i_iterations" entry_found=length(entry_complexes) remaining=length(current_set)
        
        append!(phase_i_total_excluded, entry_complexes)
        current_set = setdiff(current_set, entry_complexes)
        
        if isempty(current_set)
            break
        end
    end
    
    phase_results["phase_i"] = (
        iterations = phase_i_iterations,
        excluded = phase_i_total_excluded,
        excluded_count = length(phase_i_total_excluded),
        remaining_after = length(current_set)
    )
    
    verbose && @info "Phase I complete" iterations=phase_i_iterations excluded=length(phase_i_total_excluded) remaining=length(current_set)
    
    if isempty(current_set)
        phase_results["early_termination"] = "phase_i"
        return phase_results
    end
    
    # Phase II: Non-reactant complexes validation
    nonreactant = find_nonreactant_complexes(current_set, A_matrix)
    current_set = setdiff(current_set, nonreactant)
    
    phase_results["phase_ii"] = (
        excluded = nonreactant,
        excluded_count = length(nonreactant),
        remaining_after = length(current_set)
    )
    
    verbose && @info "Phase II complete" excluded=length(nonreactant) remaining=length(current_set)
    
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
        
        verbose && @info "Phase III iteration $phase_iii_iterations" exit_found=length(exit_complexes) remaining=length(current_set)
        
        append!(phase_iii_total_excluded, exit_complexes)
        current_set = setdiff(current_set, exit_complexes)
        
        if isempty(current_set)
            break
        end
    end
    
    phase_results["phase_iii"] = (
        iterations = phase_iii_iterations,
        excluded = phase_iii_total_excluded,
        excluded_count = length(phase_iii_total_excluded),
        remaining_after = length(current_set)
    )
    
    verbose && @info "Phase III complete" iterations=phase_iii_iterations excluded=length(phase_iii_total_excluded) remaining=length(current_set)
    
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
        total_components = length(phase_iv_components),
        terminal_components = phase_iv_terminal_components,
        non_terminal_complexes = phase_iv_complexes,
        non_terminal_count = length(phase_iv_complexes)
    )
    
    verbose && @info "Phase IV complete" total_components=length(phase_iv_components) terminal=length(phase_iv_terminal_components) non_terminal=length(phase_iv_complexes)
    
    # Final upstream set (union of Phase III and IV)
    upstream_set = union(phase_iii_total_excluded, phase_iv_complexes)
    phase_results["upstream_set_size"] = length(upstream_set)
    
    return phase_results
end