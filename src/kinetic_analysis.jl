"""
Kinetic module analysis for COCOA.

This module implements:
- Kinetic module identification based on concordance modules
- Upstream Algorithm for autonomous and nonterminal complex identification
- Concentration robustness analysis for metabolites
- Memory-efficient processing for large-scale HPC models
"""


"""
Results from kinetic module analysis.

Contains the original concordance results plus kinetic-specific analysis:
- Kinetic modules identified through the Upstream Algorithm
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
    # Extract complexes using COCOA's system - returns Dict{Symbol, MetabolicComplex}
    complexes_dict = extract_complexes_from_constraints(model, constraints)

    # Use universal mapping to get reaction names consistently
    reaction_names = get_reaction_names_from_constraints(constraints)
    reaction_to_idx = Dict(rxn => i for (i, rxn) in enumerate(reaction_names))

    # Build S matrix from flux_stoichiometry constraints using universal mapping
    S_matrix, metabolite_names = build_S_matrix_from_constraints(constraints)

    # Build A matrix from complexes using universal mapping approach  
    A_matrix, complexes_vec, complex_to_idx = build_A_matrix_from_complexes(complexes_dict, constraints)

    # Build Y matrix (metabolites × complexes) from complex metabolite data
    n_metabolites = length(metabolite_names)
    n_complexes = length(complexes_vec)
    metabolite_to_idx = Dict(met => i for (i, met) in enumerate(metabolite_names))

    Y_matrix = SparseArrays.spzeros(Float64, n_metabolites, n_complexes)

    for (complex_idx, complex) in enumerate(complexes_vec)
        if hasfield(typeof(complex), :metabolites)
            for (metabolite_symbol, stoich) in complex.metabolites
                met_name = string(metabolite_symbol)
                if haskey(metabolite_to_idx, met_name)
                    met_idx = metabolite_to_idx[met_name]
                    Y_matrix[met_idx, complex_idx] = Float64(stoich)
                end
            end
        end
    end

    return (
        Y_matrix=Y_matrix,
        A_matrix=A_matrix,
        metabolite_names=metabolite_names,
        reaction_names=reaction_names,
        complex_to_idx=complex_to_idx,
        metabolite_to_idx=metabolite_to_idx,
        reaction_to_idx=reaction_to_idx
    )
end

"""
Build S matrix (metabolites × reactions) from flux_stoichiometry constraints.
Uses universal mapping for consistency with A matrix construction.
"""
function build_S_matrix_from_constraints(constraints)
    # Get the balance constraints
    balance_constraints = haskey(constraints, :balance) ? constraints.balance : constraints

    # Extract metabolite names from flux_stoichiometry keys
    metabolite_names = collect(string.(keys(balance_constraints.flux_stoichiometry)))

    # Use universal mapping to get reaction names
    reaction_names = get_reaction_names_from_constraints(constraints)
    reaction_to_idx = Dict(rxn => i for (i, rxn) in enumerate(reaction_names))

    # Create temporary storage for reactions and coefficients
    reaction_indices = Int[]
    metabolite_indices = Int[]
    coefficients = Float64[]

    # Build metabolite index mapping
    metabolite_to_idx = Dict(met => i for (i, met) in enumerate(metabolite_names))

    # Process each metabolite's balance constraint
    for (met_symbol, balance_constraint) in balance_constraints.flux_stoichiometry
        met_name = string(met_symbol)
        met_idx = metabolite_to_idx[met_name]

        # Extract coefficients from LinearValue structure
        if hasfield(typeof(balance_constraint.value), :idxs) && hasfield(typeof(balance_constraint.value), :weights)
            for (var_idx, coeff) in zip(balance_constraint.value.idxs, balance_constraint.value.weights)
                # Find which reaction this variable index corresponds to
                rxn_idx = findfirst_reaction_for_variable(var_idx, balance_constraints, reaction_names, reaction_to_idx)
                if rxn_idx !== nothing
                    push!(metabolite_indices, met_idx)
                    push!(reaction_indices, rxn_idx)
                    push!(coefficients, Float64(coeff))
                end
            end
        end
    end

    # Build sparse S matrix
    n_metabolites = length(metabolite_names)
    n_reactions = length(reaction_names)
    S_matrix = SparseArrays.sparse(metabolite_indices, reaction_indices, coefficients, n_metabolites, n_reactions)

    return S_matrix, metabolite_names
end

"""
Find which reaction a variable index corresponds to.
"""
function findfirst_reaction_for_variable(var_idx, constraints, reaction_names, reaction_to_idx)
    # Check split reactions first
    if haskey(constraints, :fluxes_forward) && haskey(constraints, :fluxes_reverse)
        # Check forward reactions
        for (rxn_symbol, var) in constraints.fluxes_forward
            if hasfield(typeof(var.value), :idxs) && var_idx in var.value.idxs
                rxn_name = string(rxn_symbol) * "_forward"
                return get(reaction_to_idx, rxn_name, nothing)
            end
        end
        # Check reverse reactions
        for (rxn_symbol, var) in constraints.fluxes_reverse
            if hasfield(typeof(var.value), :idxs) && var_idx in var.value.idxs
                rxn_name = string(rxn_symbol) * "_reverse"
                return get(reaction_to_idx, rxn_name, nothing)
            end
        end
    else
        # Check regular reactions
        for (rxn_symbol, var) in constraints.fluxes
            if hasfield(typeof(var.value), :idxs) && var_idx in var.value.idxs
                rxn_name = string(rxn_symbol)
                return get(reaction_to_idx, rxn_name, nothing)
            end
        end
    end
    return nothing
end

"""
Build A matrix (complexes × reactions) using universal reaction mapping.
Uses the constraint-based mapping system for consistency.
"""
function build_A_matrix_from_complexes(complexes_dict, constraints)
    complexes_vec = collect(Base.values(complexes_dict))
    n_complexes = length(complexes_vec)

    # Get reaction names from constraints using universal mapping
    reaction_names = get_reaction_names_from_constraints(constraints)
    n_reactions = length(reaction_names)
    reaction_to_idx = Dict(rxn => i for (i, rxn) in enumerate(reaction_names))

    # Get the reaction name mapping for base → constraint name conversion
    mapping = build_reaction_name_mapping(constraints)

    # Create complex name to index mapping
    complex_to_idx_map = Dict{String,Int}()
    for (i, complex) in enumerate(complexes_vec)
        if hasfield(typeof(complex), :id)
            complex_to_idx_map[string(complex.id)] = i
        end
    end

    # Storage for sparse matrix construction
    complex_indices = Int[]
    reaction_indices = Int[]
    coefficients = Int8[]

    # Process each complex's reaction contributions
    for (complex_idx, complex) in enumerate(complexes_vec)
        if hasfield(typeof(complex), :reaction_contributions)
            for (rxn_symbol, contribution) in complex.reaction_contributions
                base_rxn_name = string(rxn_symbol)

                # Use universal mapping to get constraint reaction names
                if haskey(mapping.base_to_constraint, base_rxn_name)
                    constraint_rxn_names = mapping.base_to_constraint[base_rxn_name]

                    for constraint_rxn_name in constraint_rxn_names
                        rxn_idx = get(reaction_to_idx, constraint_rxn_name, nothing)

                        if rxn_idx !== nothing
                            push!(complex_indices, complex_idx)
                            push!(reaction_indices, rxn_idx)

                            # Handle split reaction sign logic
                            if endswith(constraint_rxn_name, "_forward")
                                # Forward reaction: preserve contribution sign
                                coeff = contribution > 0 ? Int8(1) : Int8(-1)
                            elseif endswith(constraint_rxn_name, "_reverse")
                                # Reverse reaction: flip contribution sign
                                coeff = contribution > 0 ? Int8(-1) : Int8(1)
                            else
                                # Standard reaction: preserve contribution sign
                                coeff = contribution > 0 ? Int8(1) : Int8(-1)
                            end

                            push!(coefficients, coeff)
                        end
                    end
                end
            end
        end
    end

    # Build sparse A matrix
    A_matrix = SparseArrays.sparse(complex_indices, reaction_indices, coefficients, n_complexes, n_reactions)

    return A_matrix, complexes_vec, complex_to_idx_map
end

"""
Convert a complex to a metabolite dictionary for matrix construction.
"""
function complex_to_metabolite_dict(complex, model)
    # Handle MetabolicComplex structure from COCOA constraints
    if hasfield(typeof(complex), :metabolites)
        # Convert Vector{Tuple{Symbol,Float64}} to Dict{Symbol,Float64}
        return Dict(complex.metabolites)
    elseif isa(complex, Dict)
        # Already a metabolite dictionary
        return complex
    elseif isa(complex, Vector) && length(complex) == 2
        # Tuple-like representation [metabolites_dict, direction]
        metabolites, direction = complex
        return metabolites
    else
        # Unknown format - return empty dict to avoid errors
        @debug "Unknown complex format: $(typeof(complex))"
        return Dict()
    end
end

"""
Find a complex that matches the given metabolite composition.
"""
function find_complex_for_metabolites(target_metabolites, complexes, model)
    for complex in complexes
        complex_dict = complex_to_metabolite_dict(complex, model)

        # Check if metabolite sets match
        if Set(keys(complex_dict)) == Set(keys(target_metabolites))
            # Check if stoichiometries match (within tolerance)
            match = true
            for (met, stoich) in target_metabolites
                if !haskey(complex_dict, met) || abs(complex_dict[met] - stoich) > 1e-6
                    match = false
                    break
                end
            end
            if match
                return complex
            end
        end
    end
    return nothing
end

"""
Phase I of Upstream Algorithm: Exclude entry complexes.
An entry complex has incoming reactions from complexes outside the set C0.
Iteratively removes entry complexes until none remain.
"""
function phase_i_exclude_entry_complexes(C0::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(C0)
        return Int[]
    end

    C1 = copy(C0)

    while true
        C1_set = Set(C1)
        entry_complexes = Int[]

        # Identify entry complexes in current C1
        for complex_idx in C1
            # Find reactions where this complex is a product (A[complex, reaction] > 0)
            product_reactions = findall(A_matrix[complex_idx, :] .> 0)

            for rxn_idx in product_reactions
                # Find substrate complexes for this reaction (A[complex, reaction] < 0)
                substrate_complexes = findall(A_matrix[:, rxn_idx] .< 0)

                # If any substrate is outside C1, this complex is an entry complex
                if !all(sub_idx in C1_set for sub_idx in substrate_complexes)
                    push!(entry_complexes, complex_idx)
                    break  # No need to check other reactions for this complex
                end
            end
        end

        # If no entry complexes found, we're done
        if isempty(entry_complexes)
            break
        end

        # Remove entry complexes from C1
        C1 = setdiff(C1, entry_complexes)

        @debug "Phase I: Removed $(length(entry_complexes)) entry complexes, $(length(C1)) remaining"

        # Safety check to prevent infinite loops
        if isempty(C1)
            break
        end
    end

    return C1
end

"""
Phase II of Upstream Algorithm: Remove complexes with no outgoing connections.
Removes complexes that have no outgoing reactions (degree = 0).
"""
function phase_ii_exclude_no_outgoing(C1::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(C1)
        return Int[]
    end

    C2 = Int[]

    for complex_idx in C1
        # Check if complex has any outgoing reactions (appears as substrate)
        outgoing_reactions = findall(A_matrix[complex_idx, :] .< 0)

        # Keep complexes that have outgoing connections
        if !isempty(outgoing_reactions)
            push!(C2, complex_idx)
        end
    end

    @debug "Phase II: Removed $(length(C1) - length(C2)) complexes with no outgoing connections"

    return C2
end

"""
Phase III of Upstream Algorithm: Exclude exit complexes iteratively.
Removes complexes that have outgoing connections to complexes outside the current set.
Returns both the remaining core and the exit complexes that were removed.
"""
function phase_iii_exclude_exit_complexes(C2::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(C2)
        return Int[], Int[]
    end

    C3 = copy(C2)
    exit_complexes = Int[]

    while true
        C3_set = Set(C3)
        current_exits = Int[]

        # Find exit complexes in current C3
        for complex_idx in C3
            # Find outgoing reactions from this complex
            substrate_reactions = findall(A_matrix[complex_idx, :] .< 0)

            is_exit = false
            for rxn_idx in substrate_reactions
                # Find products of this reaction
                product_complexes = findall(A_matrix[:, rxn_idx] .> 0)

                # If any product is outside C3, this is an exit complex
                if !all(prod_idx in C3_set for prod_idx in product_complexes)
                    is_exit = true
                    break
                end
            end

            if is_exit
                push!(current_exits, complex_idx)
            end
        end

        # Stop if no exit complexes found
        if isempty(current_exits)
            break
        end

        # Move exit complexes out of C3
        append!(exit_complexes, current_exits)
        C3 = setdiff(C3, current_exits)

        @debug "Phase III: Removed $(length(current_exits)) exit complexes, $(length(C3)) remaining"

        if isempty(C3)
            break
        end
    end

    return C3, exit_complexes
end

"""
Phase IV of Upstream Algorithm: Exclude terminal complexes.
Identifies and removes all terminal strong linkage classes from C_minus4.
A terminal strong linkage class has no outgoing reactions to other complexes in the same set.
Returns non-terminal complexes (those that remain after excluding terminal ones).
"""
function phase_iv_exclude_terminal_complexes(C_minus4::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(C_minus4)
        return Int[]
    end

    C_minus4_set = Set(C_minus4)

    # Find strongly connected components (strong linkage classes) in the subgraph induced by C_minus4
    strong_linkage_classes = find_strong_linkage_classes(C_minus4, A_matrix)

    # Identify terminal strong linkage classes
    nonterminal_complexes = Int[]

    for class_complexes in strong_linkage_classes
        is_terminal = true

        # Check if this strong linkage class has outgoing reactions to OTHER classes within C_minus4
        for complex_idx in class_complexes
            # Find reactions where this complex is a substrate (A[complex, reaction] < 0)
            substrate_reactions = findall(A_matrix[complex_idx, :] .< 0)

            for rxn_idx in substrate_reactions
                # Find product complexes for this reaction (A[complex, reaction] > 0)
                product_complexes = findall(A_matrix[:, rxn_idx] .> 0)

                # Check if any product is in C_minus4 but NOT in this same class
                for prod_idx in product_complexes
                    if prod_idx in C_minus4_set && prod_idx ∉ class_complexes
                        is_terminal = false
                        break
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

        # If NOT terminal, include in nonterminal complexes
        if !is_terminal
            append!(nonterminal_complexes, class_complexes)
        end
    end

    @debug "Phase IV: Found $(length(strong_linkage_classes)) strong linkage classes, $(length(nonterminal_complexes)) nonterminal complexes remain"

    return nonterminal_complexes
end

"""
Find strongly connected components (strong linkage classes) using Tarjan's algorithm.
Returns vector of vectors, where each inner vector contains complexes in one strong linkage class.
"""
function find_strong_linkage_classes(complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(complexes)
        return Vector{Int}[]
    end

    n_complexes = length(complexes)
    complex_to_local = Dict(complexes[i] => i for i in 1:n_complexes)

    # Build adjacency matrix for the subgraph
    adj_matrix = zeros(Bool, n_complexes, n_complexes)

    for (local_i, global_i) in enumerate(complexes)
        # Find reactions where this complex is a substrate
        substrate_reactions = findall(A_matrix[global_i, :] .< 0)

        for rxn_idx in substrate_reactions
            # Find product complexes for this reaction
            product_complexes = findall(A_matrix[:, rxn_idx] .> 0)

            for prod_global in product_complexes
                if haskey(complex_to_local, prod_global)
                    local_j = complex_to_local[prod_global]
                    adj_matrix[local_i, local_j] = true
                end
            end
        end
    end

    # Apply Tarjan's algorithm
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

        # Consider successors of v
        for w in 1:n_complexes
            if adj_matrix[v, w]  # Edge v -> w exists
                if indices[w] == 0
                    # Successor w has not been visited; recurse
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elseif on_stack[w]
                    # Successor w is in stack and hence in current SCC
                    lowlinks[v] = min(lowlinks[v], indices[w])
                end
            end
        end

        # If v is a root node, pop the stack and create SCC
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

    # Run DFS for each unvisited node
    for v in 1:n_complexes
        if indices[v] == 0
            strongconnect(v)
        end
    end

    return components
end

"""
The Upstream Algorithm W(C0): Find the largest upstream subset.
Implements the theoretical upstream function that returns the unique maximal
upstream subset of C0 according to Theorem S2-7.

An upstream set is both autonomous (no external dependencies) and feeding
(every complex eventually converts to complexes outside the set).

# Algorithm Steps:
1. Phase I: Exclude entry complexes iteratively
2. Phase IV: Exclude terminal strong linkage classes

# Arguments:
- `C0::Vector{Int}`: Input complex indices (typically an extended module C_b ∪ C_m)
- `A_matrix`: Complex-reaction incidence matrix

# Returns:
- `Vector{Int}`: Indices of complexes in the largest upstream subset
"""
function upstream_algorithm_W(C0::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(C0)
        return Int[]
    end

    @debug "Upstream Algorithm W: Starting with $(length(C0)) complexes"

    # Phase I: Exclusion of entry complexes (autonomous property)
    C1 = phase_i_exclude_entry_complexes(C0, A_matrix)
    @debug "Phase I: $(length(C0)) → $(length(C1)) complexes"

    if isempty(C1)
        return Int[]
    end

    # Phase II: Remove complexes with no outgoing connections from the complex itself
    C2 = phase_ii_exclude_no_outgoing(C1, A_matrix)
    @debug "Phase II: $(length(C1)) → $(length(C2)) complexes"

    if isempty(C2)
        return Int[]
    end

    # Phase III: Exclusion of exit complexes (complexes with outgoing connections from the module)
    C3, phase_III_complexes = phase_iii_exclude_exit_complexes(C2, A_matrix)
    @debug "Phase III: $(length(C2)) → $(length(C3)) core + $(length(phase_III_complexes)) exit complexes"

    # Phase IV: Exclude terminal strong linkage classes from remaining complexes
    phase_IV_complexes = Int[]
    if !isempty(C3)
        phase_IV_complexes = phase_iv_exclude_terminal_complexes(C3, A_matrix)
        @debug "Phase IV: $(length(C3)) → $(length(phase_IV_complexes)) nonterminal complexes"
    end

    # Final kinetic module: nonterminal complexes that survive all phases
    # Theory: "autonomous set of nonterminal complexes"
    kinetic_module_complexes = phase_IV_complexes

    @debug "Upstream Algorithm W: Final result $(length(kinetic_module_complexes)) kinetic module complexes"

    return kinetic_module_complexes
end

"""
Apply Upstream Algorithm filtering to identify kinetic module candidates.
Wrapper function that calls the complete upstream algorithm W(C0).
"""
function apply_upstream_algorithm(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    return upstream_algorithm_W(complex_indices, A_matrix)
end

"""
Debug version of upstream algorithm that shows what happens at each phase.
"""
function debug_upstream_algorithm(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    println("Starting upstream algorithm with $(length(complex_indices)) complexes")
    
    # Phase I: Exclusion of entry complexes (autonomous property)
    C1 = phase_i_exclude_entry_complexes(complex_indices, A_matrix)
    println("Phase I: $(length(complex_indices)) → $(length(C1)) complexes")
    
    if isempty(C1)
        println("Phase I eliminated all complexes - stopping")
        return Int[]
    end
    
    # Phase II: Remove complexes with no outgoing connections from the complex itself
    C2 = phase_ii_exclude_no_outgoing(C1, A_matrix)
    println("Phase II: $(length(C1)) → $(length(C2)) complexes")
    
    if isempty(C2)
        println("Phase II eliminated all complexes - stopping")
        return Int[]
    end
    
    # Phase III: Exclusion of exit complexes (complexes with outgoing connections from the module)
    C3, phase_III_complexes = phase_iii_exclude_exit_complexes(C2, A_matrix)
    println("Phase III: $(length(C2)) → $(length(C3)) core + $(length(phase_III_complexes)) exit complexes")
    
    # Phase IV: Exclude terminal strong linkage classes from remaining complexes
    phase_IV_complexes = Int[]
    if !isempty(C3)
        println("Phase IV debug: Analyzing $(length(C3)) complexes")
        
        # Debug strong linkage classes
        strong_linkage_classes = find_strong_linkage_classes(C3, A_matrix)
        println("  Found $(length(strong_linkage_classes)) strong linkage classes")
        
        for (i, class_complexes) in enumerate(strong_linkage_classes)
            println("  Class $i: $(length(class_complexes)) complexes")
            
            # Check connections to other classes
            connections_found = false
            for complex_idx in class_complexes
                substrate_reactions = findall(A_matrix[complex_idx, :] .< 0)
                for rxn_idx in substrate_reactions
                    product_complexes = findall(A_matrix[:, rxn_idx] .> 0)
                    for prod_idx in product_complexes
                        if prod_idx in C3 && prod_idx ∉ class_complexes
                            connections_found = true
                            break
                        end
                    end
                    if connections_found break end
                end
                if connections_found break end
            end
            println("    Connections to other classes: $connections_found")
        end
        
        phase_IV_complexes = phase_iv_exclude_terminal_complexes(C3, A_matrix)
        println("Phase IV: $(length(C3)) → $(length(phase_IV_complexes)) nonterminal complexes")
    else
        println("Phase III eliminated all complexes - Phase IV has nothing to process")
    end
    
    return phase_IV_complexes
end

"""
Kinetic module identification using the proven redesigned approach.
"""
function identify_kinetic_modules(
    complete_model::CompleteConcordanceModel;
    min_module_size::Int=2,
    workers=D.workers()
)
    @info "Identifying kinetic modules using CompleteConcordanceModel"

    # Extract concordance modules from CompleteConcordanceModel
    # Group complexes by their concordance module ID (>0 means in a module)
    concordance_module_groups = Dict{Int,Vector{Int}}()
    for (complex_idx, module_id) in enumerate(complete_model.concordance_modules)
        if module_id > 0  # Skip complexes not in any concordance module
            if !haskey(concordance_module_groups, module_id)
                concordance_module_groups[module_id] = Int[]
            end
            push!(concordance_module_groups[module_id], complex_idx)
        end
    end

    @info "Processing concordance modules: $(length(concordance_module_groups))"

    module_candidates = Vector{Vector{Int}}()

    # Debug: Track upstream algorithm results
    upstream_results = []

    # Get balanced complexes (module_id = 0)
    balanced_indices = Int[]
    for (complex_idx, module_id) in enumerate(complete_model.concordance_modules)
        if module_id == 0
            push!(balanced_indices, complex_idx)
        end
    end

    @info "Processing extended modules (balanced + concordance)"
    @debug "Balanced complexes: $(length(balanced_indices))"

    # Apply upstream algorithm to extended modules (balanced + each concordance module)
    for (module_id, concordance_indices) in concordance_module_groups
        @debug "Processing extended module: balanced + concordance module $module_id"

        # Create extended module combining balanced and concordance complexes
        extended_module = union(balanced_indices, concordance_indices)

        # Apply upstream algorithm to find kinetic module candidates
        kinetic_candidates = apply_upstream_algorithm(extended_module, complete_model.complex_reaction_matrix)

        # Track results
        push!(upstream_results, ("extended_$module_id", length(extended_module), length(kinetic_candidates)))

        # Keep all successful results
        if length(kinetic_candidates) >= 1
            push!(module_candidates, kinetic_candidates)
            @info "Extended module $module_id: $(length(extended_module)) → $(length(kinetic_candidates)) kinetic candidates"
        end
    end

    # Process pure balanced linkage classes as potential kinetic modules
    if !isempty(balanced_indices)
        @debug "Processing pure balanced linkage classes"

        # Apply upstream algorithm to balanced complexes alone
        balanced_kinetic_candidates = apply_upstream_algorithm(balanced_indices, complete_model.complex_reaction_matrix)

        push!(upstream_results, ("balanced_linkage", length(balanced_indices), length(balanced_kinetic_candidates)))

        if length(balanced_kinetic_candidates) >= 1
            push!(module_candidates, balanced_kinetic_candidates)
            @debug "Added balanced linkage class: $(length(balanced_kinetic_candidates)) complexes"
        end
    end

    # Debug: Show what the upstream algorithm found
    @info "Upstream algorithm results:"
    for (module_id, input_size, output_size) in upstream_results
        @info "  Module $module_id: $input_size → $output_size"
    end

    # Merge overlapping candidates using graph-based approach
    @info "Merging overlapping kinetic modules: $(length(module_candidates))"

    if isempty(module_candidates)
        final_modules = Vector{Vector{Int}}()
    else
        # Create overlap matrix between kinetic module candidates
        n_candidates = length(module_candidates)
        overlap_matrix = zeros(Int, n_candidates, n_candidates)

        for i in 1:(n_candidates-1)
            for j in (i+1):n_candidates
                overlap_size = length(intersect(Set(module_candidates[i]), Set(module_candidates[j])))
                overlap_matrix[i, j] = overlap_size
                overlap_matrix[j, i] = overlap_size
            end
        end

        # Find connected components of overlapping modules
        # Two modules are connected if they share any complexes
        adjacency_matrix = (overlap_matrix .> 0)

        # Use simple DFS to find connected components
        visited = falses(n_candidates)
        final_modules = Vector{Vector{Int}}()

        for start_idx in 1:n_candidates
            if !visited[start_idx]
                # Find all modules connected to this one
                component = Int[]
                stack = [start_idx]

                while !isempty(stack)
                    current = pop!(stack)
                    if !visited[current]
                        visited[current] = true
                        push!(component, current)

                        # Add unvisited neighbors to stack
                        for neighbor in 1:n_candidates
                            if adjacency_matrix[current, neighbor] && !visited[neighbor]
                                push!(stack, neighbor)
                            end
                        end
                    end
                end

                # Merge all modules in this connected component
                merged_module = Int[]
                for module_idx in component
                    merged_module = union(merged_module, module_candidates[module_idx])
                end

                push!(final_modules, merged_module)
                @debug "Merged $(length(component)) overlapping modules into one with $(length(merged_module)) complexes"
            end
        end
    end

    # Convert to final format with names and create kinetic modules dict
    kinetic_modules = Dict{Symbol,Vector{Symbol}}()

    for (i, module_complexes) in enumerate(final_modules)
        module_symbol = Symbol("kinetic_module_$i")
        complex_symbols = Symbol[]

        for complex_idx in module_complexes
            # Get complex name directly from CompleteConcordanceModel
            if complex_idx <= length(complete_model.complex_ids)
                complex_symbol = complete_model.complex_ids[complex_idx]
                push!(complex_symbols, complex_symbol)
            end
        end

        if length(complex_symbols) >= min_module_size
            kinetic_modules[module_symbol] = complex_symbols
        end
    end

    # Identify giant kinetic module
    giant_module_id = :none
    giant_module_size = 0

    if !isempty(kinetic_modules)
        for (module_id, complexes) in kinetic_modules
            if length(complexes) > giant_module_size
                giant_module_size = length(complexes)
                giant_module_id = module_id
            end
        end
    end

    # Identify interface reactions using our proven approach  
    interface_reactions = identify_interface_reactions(kinetic_modules, complete_model, min_module_size)

    # Update CompleteConcordanceModel.kinetic_modules directly
    for (module_idx, (module_symbol, complex_symbols)) in enumerate(kinetic_modules)
        for complex_symbol in complex_symbols
            if haskey(complete_model.complex_idx, complex_symbol)
                complex_idx = complete_model.complex_idx[complex_symbol]
                complete_model.kinetic_modules[complex_idx] = module_idx
            end
        end
    end

    # Update kinetic module count
    complete_model.n_kinetic_modules = length(kinetic_modules)

    # Update interface reactions
    for reaction_name in interface_reactions
        reaction_idx = findfirst(==(Symbol(reaction_name)), complete_model.reaction_ids)
        if reaction_idx !== nothing
            complete_model.interface_reactions[reaction_idx] = true
        end
    end

    @info "Kinetic module identification completed: $(length(kinetic_modules)) modules, giant size: $giant_module_size"

    # Return minimal result info for logging/debugging including giant module size
    return (
        interface_reactions=interface_reactions,
        giant_module_size=giant_module_size
    )
end

"""
Identify interface reactions based on kinetic modules using our proven approach.
"""
function identify_interface_reactions(kinetic_modules::Dict{Symbol,Vector{Symbol}},
    complete_model::CompleteConcordanceModel, min_module_size::Int)

    # Build mapping from complex to kinetic module
    complex_to_module = Dict{String,Symbol}()
    for (module_id, complexes) in kinetic_modules
        for complex_symbol in complexes
            complex_to_module[string(complex_symbol)] = module_id
        end
    end

    interface_reactions = Symbol[]
    internal_reaction_count = 0

    for (rxn_name, rxn_idx) in complete_model.reaction_idx
        # Find all complexes involved in this reaction
        involved_complexes = findall(abs.(complete_model.complex_reaction_matrix[:, rxn_idx]) .> 0)

        # Get modules for involved complexes
        modules_involved = Set{Symbol}()
        has_unassigned = false

        for complex_idx in involved_complexes
            # Find complex name using CompleteConcordanceModel
            complex_name = string(complete_model.complex_ids[complex_idx])

            if complex_name !== nothing && haskey(complex_to_module, complex_name)
                push!(modules_involved, complex_to_module[complex_name])
            else
                has_unassigned = true
            end
        end

        # Reaction is internal if all complexes belong to same module (and no unassigned)
        is_internal = !has_unassigned && length(modules_involved) == 1

        if is_internal
            internal_reaction_count += 1
        else
            push!(interface_reactions, Symbol(rxn_name))
        end
    end

    @info "Interface reaction analysis: $(length(complete_model.reaction_ids)) total, $internal_reaction_count internal, $(length(interface_reactions)) interface"

    return interface_reactions
end

"""
$(TYPEDSIGNATURES)

Comprehensive kinetic and concordance analysis combining activity concordance with kinetic module identification.

This is the main entry point for kinetic analysis in COCOA, providing both concordance analysis
and kinetic module identification in a single workflow. The analysis includes:

1. **Activity concordance analysis**: Identifies concordant complexes using optimized constraint-based methods
2. **Kinetic module identification**: Uses the Upstream Algorithm to find kinetic modules from concordance results  
3. **Interface reaction analysis**: Identifies reactions connecting different kinetic modules
4. **Giant module detection**: Finds the largest kinetic module for network characterization

# Arguments
- `model::AbstractFBCModels.AbstractFBCModel`: Metabolic network model
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `min_module_size::Int=2`: Minimum size for kinetic modules
- `workers=D.workers()`: Worker pool for parallel processing
- `kwargs...`: Additional arguments passed to concordance analysis

# Returns
`KineticModuleResults` containing:
- `concordance_results`: Full concordance analysis results (complexes, modules, linkage classes)
- `kinetic_modules`: Dictionary mapping kinetic module IDs to complex lists
- `giant_module_id`: ID of the largest kinetic module
- `interface_reactions`: Vector of reaction IDs connecting different modules
- `Y_matrix`, `A_matrix`: Network topology matrices for advanced analysis
- `*_names`, `*_to_idx`: Name mappings and index dictionaries for position queries
- `summary`: Comprehensive statistics for large-scale reporting

# Performance Notes

The function is optimized for large-scale metabolic networks:

- **Memory efficiency**: Constraint trees avoid dense matrix storage
- **Parallel processing**: Utilizes distributed workers where beneficial  
- **Sparse matrices**: All network matrices stored in compressed sparse format
- **Incremental analysis**: Reuses concordance results for kinetic analysis

# Usage Example

```julia
using COCOA, HiGHS, AbstractFBCModels

model = load_model("path/to/model.json")
optimizer = HiGHS.Optimizer

results = kinetic_concordance_analysis(
    model, optimizer;
    min_module_size=3,
    seed=1234,
    interface=:split_forward
)

# Access results
println("Found \$(length(results.kinetic_modules)) kinetic modules")
println("Giant module: \$(results.giant_module_id) with \$(length(results.kinetic_modules[results.giant_module_id])) complexes")
println("Interface reactions: \$(length(results.interface_reactions))")
```

# Theoretical Background

The analysis implements established theory connecting activity concordance to kinetic modularity:

- **Concordance modules**: Complexes with proportional steady-state activities
- **Kinetic modules**: Maximal sets of mutually kinetically coupled complexes  
- **Upstream Algorithm**: Identifies autonomous and nonterminal complex sets
- **Interface reactions**: Connect kinetic modules and determine system-level behavior

# References

"""
function kinetic_concordance_analysis(
    model::AbstractFBCModels.AbstractFBCModel;
    optimizer,
    min_module_size::Int=2,
    workers=D.workers(),
    kwargs...
)
    @info "Starting comprehensive kinetic concordance analysis"

    # Step 1: Run activity concordance analysis - now returns CompleteConcordanceModel directly
    @info "Running activity concordance analysis"
    complete_model = activity_concordance_analysis(model; optimizer=optimizer, kwargs...)
    n_complexes = complete_model.n_complexes
    n_modules = complete_model.n_concordance_modules

    @info "Concordance analysis completed: $n_complexes complexes in $n_modules modules"

    # Step 2: Build constraints for kinetic analysis
    @info "Building constraints for kinetic module analysis"
    constraints_kwargs = (
        :interface => get(kwargs, :interface, nothing),
        :use_unidirectional_constraints => get(kwargs, :use_unidirectional_constraints, true)
    )

    constraints = concordance_constraints(model; constraints_kwargs...)

    # Step 3: Run kinetic module analysis using CompleteConcordanceModel directly
    @info "Running kinetic module analysis"
    kinetic_results = identify_kinetic_modules(complete_model; min_module_size, workers)

    # Extract kinetic module statistics - now efficiently returned by the function
    n_kinetic_modules = complete_model.n_kinetic_modules
    giant_kinetic_module_size = kinetic_results.giant_module_size

    # Create comprehensive summary
    summary = Dict{String,Int}(
        "n_reactions" => complete_model.n_reactions,
        "n_complexes" => complete_model.n_complexes,
        "n_concordance_modules" => complete_model.n_concordance_modules,
        "n_kinetic_modules" => n_kinetic_modules,
        "giant_kinetic_module_size" => giant_kinetic_module_size,
        "n_interface_reactions" => length(kinetic_results.interface_reactions)
    )

    @info "Kinetic concordance analysis completed: $n_kinetic_modules kinetic modules, $(length(kinetic_results.interface_reactions)) interface reactions"

    # Return the updated CompleteConcordanceModel with kinetic analysis results
    return complete_model
end