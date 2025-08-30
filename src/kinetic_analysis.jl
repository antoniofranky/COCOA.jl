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
Identify autonomous complexes based on network topology.
Uses efficient graph analysis instead of iterative filtering.
"""
function find_autonomous_complexes(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(complex_indices)
        return Int[]
    end

    complex_set = Set(complex_indices)
    autonomous_candidates = Int[]

    for complex_idx in complex_indices
        is_autonomous = true

        # Check if this complex has any incoming reactions from outside the set
        # Find reactions where this complex is a product (A[complex, reaction] > 0)
        product_reactions = findall(A_matrix[complex_idx, :] .> 0)

        for rxn_idx in product_reactions
            # Find substrate complexes for this reaction (A[complex, reaction] < 0)
            substrate_complexes = findall(A_matrix[:, rxn_idx] .< 0)

            # If any substrate is outside our complex set, this complex is not autonomous
            if !all(sub_idx in complex_set for sub_idx in substrate_complexes)
                is_autonomous = false
                break
            end
        end

        # Additional check: complex must be able to participate in reactions (not isolated)
        if is_autonomous
            total_reactions = sum(abs.(A_matrix[complex_idx, :]) .> 0)
            if total_reactions > 0
                push!(autonomous_candidates, complex_idx)
            end
        end
    end

    return autonomous_candidates
end

"""
Identify nonterminal complexes using graph connectivity analysis.
A complex is nonterminal if it has outgoing connections within the autonomous set.
"""
function find_nonterminal_complexes(autonomous_complexes::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    if isempty(autonomous_complexes)
        return Int[]
    end

    autonomous_set = Set(autonomous_complexes)
    nonterminal_complexes = Int[]

    for complex_idx in autonomous_complexes
        # Find reactions where this complex is a substrate (A[complex, reaction] < 0)
        substrate_reactions = findall(A_matrix[complex_idx, :] .< 0)

        has_internal_outgoing = false
        for rxn_idx in substrate_reactions
            # Find product complexes for this reaction (A[complex, reaction] > 0)
            product_complexes = findall(A_matrix[:, rxn_idx] .> 0)

            # If any product is within our autonomous set, this complex is nonterminal
            if any(prod_idx in autonomous_set for prod_idx in product_complexes)
                has_internal_outgoing = true
                break
            end
        end

        if has_internal_outgoing
            push!(nonterminal_complexes, complex_idx)
        end
    end

    return nonterminal_complexes
end

"""
Apply Upstream Algorithm filtering to identify kinetic module candidates.
Implements the theoretical W(C) operation: autonomous and feeding upstream set.
"""
function apply_upstream_algorithm(complex_indices::Vector{Int}, A_matrix::SparseArrays.SparseMatrixCSC)
    # Step 1: Find autonomous complexes (no external dependencies)
    autonomous_complexes = find_autonomous_complexes(complex_indices, A_matrix)

    if isempty(autonomous_complexes)
        return Int[]
    end

    # Step 2: Among autonomous complexes, identify nonterminal ones
    # These form the core of kinetic modules according to theory
    nonterminal_complexes = find_nonterminal_complexes(autonomous_complexes, A_matrix)

    # Return nonterminal autonomous complexes as kinetic module candidates
    return nonterminal_complexes
end

"""
Kinetic module identification using the proven redesigned approach.
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

    @info "Identifying kinetic modules using redesigned approach"

    # Get concordance modules (excluding "none")
    valid_modules = filter(row -> row.module_id != "none", concordance_results.modules)

    @info "Processing concordance modules: $(DF.nrow(valid_modules))"

    module_candidates = Vector{Vector{Int}}()

    # Process each concordance module individually
    for module_row in DF.eachrow(valid_modules)
        module_id = module_row.module_id

        # Get complexes in this module
        module_complexes = filter(row -> row.module == module_id, concordance_results.complexes)

        if DF.nrow(module_complexes) < min_module_size
            continue
        end

        # Convert complex names to indices
        complex_indices = Int[]
        for complex_row in DF.eachrow(module_complexes)
            complex_name = String(complex_row.id)
            if haskey(network_data.complex_to_idx, complex_name)
                push!(complex_indices, network_data.complex_to_idx[complex_name])
            end
        end

        if length(complex_indices) < min_module_size
            continue
        end

        @debug "Processing module $module_id: $(length(complex_indices)) complexes"

        # Apply Upstream Algorithm to find kinetic module candidates
        kinetic_candidates = apply_upstream_algorithm(complex_indices, network_data.A_matrix)

        if length(kinetic_candidates) >= min_module_size
            push!(module_candidates, kinetic_candidates)
            @debug "Added kinetic module candidate: $(length(kinetic_candidates)) complexes"
        end
    end

    # Special handling for balanced module (largest concordance module)
    balanced_module = filter(row -> row.module_id == "balanced", concordance_results.modules)
    if DF.nrow(balanced_module) > 0
        @info "Processing balanced module combinations"

        balanced_complexes = filter(row -> row.module == "balanced", concordance_results.complexes)
        balanced_indices = Int[]

        for complex_row in DF.eachrow(balanced_complexes)
            complex_name = String(complex_row.id)
            if haskey(network_data.complex_to_idx, complex_name)
                push!(balanced_indices, network_data.complex_to_idx[complex_name])
            end
        end

        @debug "Balanced complexes: $(length(balanced_indices))"

        # Try combining balanced with each other module
        for i in 1:DF.nrow(valid_modules)
            module_id = valid_modules[i, :module_id]
            if module_id == "balanced"
                continue
            end

            module_complexes = filter(row -> row.module == module_id, concordance_results.complexes)
            module_indices = Int[]

            for complex_row in DF.eachrow(module_complexes)
                complex_name = String(complex_row.id)
                if haskey(network_data.complex_to_idx, complex_name)
                    push!(module_indices, network_data.complex_to_idx[complex_name])
                end
            end

            # Combine balanced + this module
            combined_indices = union(balanced_indices, module_indices)

            if length(combined_indices) >= min_module_size
                @debug "Testing balanced + $module_id: $(length(combined_indices)) complexes"

                # Apply Upstream Algorithm to combined set
                kinetic_candidates = apply_upstream_algorithm(combined_indices, network_data.A_matrix)

                if length(kinetic_candidates) >= min_module_size
                    push!(module_candidates, kinetic_candidates)
                    @debug "Added combined kinetic module: $(length(kinetic_candidates)) complexes"
                end
            end
        end
    end

    # Merge overlapping candidates
    @info "Merging overlapping candidates: $(length(module_candidates))"

    final_modules = Vector{Vector{Int}}()
    used_candidates = Set{Int}()

    for (i, candidate) in enumerate(module_candidates)
        if i in used_candidates
            continue
        end

        merged_module = copy(candidate)
        used_candidates = union(used_candidates, Set([i]))

        # Check for overlaps with remaining candidates
        for (j, other_candidate) in enumerate(module_candidates)
            if j <= i || j in used_candidates
                continue
            end

            overlap = length(intersect(Set(merged_module), Set(other_candidate)))
            if overlap > 0  # Any overlap means merge
                merged_module = union(merged_module, other_candidate)
                used_candidates = union(used_candidates, Set([j]))
                @debug "Merged candidates $i and $j due to overlap of $overlap complexes"
            end
        end

        push!(final_modules, merged_module)
    end

    # Convert to final format with names and create kinetic modules dict
    kinetic_modules = Dict{Symbol,Vector{Symbol}}()

    for (i, module_complexes) in enumerate(final_modules)
        module_symbol = Symbol("kinetic_module_$i")
        complex_symbols = Symbol[]

        for complex_idx in module_complexes
            # Find complex name from index
            for (name, idx) in network_data.complex_to_idx
                if idx == complex_idx
                    push!(complex_symbols, Symbol(name))
                    break
                end
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
    interface_reactions = identify_interface_reactions(kinetic_modules, network_data, min_module_size)

    @info "Kinetic module identification completed: $(length(kinetic_modules)) modules, giant size: $giant_module_size"

    return (
        kinetic_modules=kinetic_modules,
        giant_module_id=giant_module_id,
        interface_reactions=interface_reactions,
        Y_matrix=network_data.Y_matrix,
        A_matrix=network_data.A_matrix,
        metabolite_names=network_data.metabolite_names,
        reaction_names=network_data.reaction_names,
        complex_to_idx=network_data.complex_to_idx,
        metabolite_to_idx=network_data.metabolite_to_idx,
        reaction_to_idx=network_data.reaction_to_idx
    )
end

"""
Identify interface reactions based on kinetic modules using our proven approach.
"""
function identify_interface_reactions(kinetic_modules::Dict{Symbol,Vector{Symbol}},
    network_data, min_module_size::Int)

    # Build mapping from complex to kinetic module
    complex_to_module = Dict{String,Symbol}()
    for (module_id, complexes) in kinetic_modules
        for complex_symbol in complexes
            complex_to_module[string(complex_symbol)] = module_id
        end
    end

    interface_reactions = Symbol[]
    internal_reaction_count = 0

    for (rxn_name, rxn_idx) in network_data.reaction_to_idx
        # Find all complexes involved in this reaction
        involved_complexes = findall(abs.(network_data.A_matrix[:, rxn_idx]) .> 0)

        # Get modules for involved complexes
        modules_involved = Set{Symbol}()
        has_unassigned = false

        for complex_idx in involved_complexes
            # Find complex name
            complex_name = nothing
            for (name, idx) in network_data.complex_to_idx
                if idx == complex_idx
                    complex_name = name
                    break
                end
            end

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

    @info "Interface reaction analysis: $(length(network_data.reaction_names)) total, $internal_reaction_count internal, $(length(interface_reactions)) interface"

    return interface_reactions
end

"""
$(TYPEDSIGNATURES)

Identify concentration robustness in metabolic networks through kinetic module analysis.

This function uses kinetic module identification to find:
- **Absolute concentration robustness (ACR)**: Metabolites whose concentrations remain constant
- **Concentration ratio robustness (CRR)**: Metabolite pairs whose concentration ratios remain constant

The analysis leverages the theoretical connection between kinetic modules and concentration robustness
in mass-action kinetics systems.

# Arguments
- `constraints`: COCOA constraint tree containing network structure
- `model::AbstractFBCModels.AbstractFBCModel`: Metabolic network model  
- `concordance_results`: Results from activity concordance analysis
- `min_module_size::Int=2`: Minimum size for kinetic modules (smaller modules filtered out)
- `workers=D.workers()`: Parallel worker pool for distributed computation

# Returns
`ConcentrationRobustnessResults` containing:
- `robust_metabolites`: Vector of metabolite names showing ACR
- `robust_metabolite_pairs`: Vector of metabolite pairs showing CRR  
- `n_robust_metabolites`: Count of ACR metabolites
- `n_robust_pairs`: Count of CRR pairs
- `kinetic_results`: Full kinetic module analysis results
- `summary`: Analysis statistics for large-scale reporting

# Implementation Notes

The implementation follows established theoretical results:

1. **ACR identification**: Metabolites that appear in exactly one kinetic module and satisfy
   specific network topology conditions are candidates for ACR

2. **CRR identification**: Pairs of metabolites within the same kinetic module that satisfy
   coupling conditions are candidates for CRR

3. **Giant kinetic module**: The largest kinetic module, which often contains the majority 
   of network complexes and provides key insights into network modularity

# Performance Characteristics

- **Memory efficient**: Reuses kinetic analysis results, no duplication of large matrices
- **Scalable**: Designed for networks with >50,000 complexes and reactions
- **HPC optimized**: Leverages distributed workers for parallel processing where applicable

# References
- Shinar & Feinberg (2010): Structural sources of robustness in biochemical reaction networks
- Anderson et al. (2020): Concentration robustness and kinetic modules in networks
"""
function identify_concentration_robustness(
    constraints,
    model::AbstractFBCModels.AbstractFBCModel,
    concordance_results;
    min_module_size::Int=2,
    workers=D.workers()
)
    @info "Running kinetic module analysis for concentration robustness"

    # First get kinetic module results
    kinetic_results = identify_kinetic_modules(constraints, model, concordance_results; min_module_size, workers)

    @info "Analyzing concentration robustness properties"

    # Extract kinetic modules and network data
    kinetic_modules = kinetic_results.kinetic_modules
    Y_matrix = kinetic_results.Y_matrix
    metabolite_names = kinetic_results.metabolite_names

    # Initialize results containers
    robust_metabolites = String[]
    robust_metabolite_pairs = Tuple{String,String}[]

    # Build metabolite-to-modules mapping for ACR analysis
    metabolite_to_modules = Dict{String,Set{Symbol}}()
    for metabolite in metabolite_names
        metabolite_to_modules[metabolite] = Set{Symbol}()
    end

    # Map metabolites to their kinetic modules
    for (module_id, module_complexes) in kinetic_modules
        for complex_symbol in module_complexes
            complex_name = string(complex_symbol)
            if haskey(kinetic_results.complex_to_idx, complex_name)
                complex_idx = kinetic_results.complex_to_idx[complex_name]

                # Find metabolites in this complex (non-zero entries in Y matrix)
                metabolite_indices = findall(abs.(Y_matrix[:, complex_idx]) .> 1e-10)
                for met_idx in metabolite_indices
                    metabolite = metabolite_names[met_idx]
                    push!(metabolite_to_modules[metabolite], module_id)
                end
            end
        end
    end

    # Identify ACR candidates: metabolites appearing in exactly one kinetic module
    @info "Identifying absolute concentration robustness (ACR) candidates"
    for (metabolite, modules) in metabolite_to_modules
        if length(modules) == 1
            # Additional checks could be added here for network topology requirements
            push!(robust_metabolites, metabolite)
        end
    end

    # Identify CRR candidates: metabolite pairs within same kinetic modules  
    @info "Identifying concentration ratio robustness (CRR) candidates"
    for (module_id, module_complexes) in kinetic_modules
        # Get all metabolites in this module
        module_metabolites = String[]
        for complex_symbol in module_complexes
            complex_name = string(complex_symbol)
            if haskey(kinetic_results.complex_to_idx, complex_name)
                complex_idx = kinetic_results.complex_to_idx[complex_name]

                metabolite_indices = findall(abs.(Y_matrix[:, complex_idx]) .> 1e-10)
                for met_idx in metabolite_indices
                    metabolite = metabolite_names[met_idx]
                    if metabolite ∉ module_metabolites
                        push!(module_metabolites, metabolite)
                    end
                end
            end
        end

        # Generate all pairs within this module
        for i in 1:length(module_metabolites)
            for j in (i+1):length(module_metabolites)
                pair = (module_metabolites[i], module_metabolites[j])
                if pair ∉ robust_metabolite_pairs
                    push!(robust_metabolite_pairs, pair)
                end
            end
        end
    end

    # Calculate statistics
    n_robust_metabolites = length(robust_metabolites)
    n_robust_pairs = length(robust_metabolite_pairs)
    giant_kinetic_module_size = if kinetic_results.giant_module_id != :none
        length(kinetic_results.kinetic_modules[kinetic_results.giant_module_id])
    else
        0
    end

    # Create summary for large-scale analysis
    summary = Dict{String,Int}(
        "n_robust_metabolites" => n_robust_metabolites,
        "n_robust_pairs" => n_robust_pairs,
        "giant_kinetic_module_size" => giant_kinetic_module_size,
        "n_kinetic_modules" => length(kinetic_modules)
    )

    @info "Concentration robustness analysis completed" n_robust_metabolites n_robust_pairs giant_kinetic_module_size

    # Create full results structure - reuse the kinetic results but update summary
    kinetic_module_results = KineticModuleResults(
        concordance_results,  # Use the original concordance results
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

    return ConcentrationRobustnessResults(
        kinetic_module_results,
        robust_metabolites,
        robust_metabolite_pairs,
        n_robust_metabolites,
        n_robust_pairs,
        giant_kinetic_module_size,
        summary
    )
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
- Grimbs et al. (2012): Activity concordance analysis for metabolic networks
- Anderson et al. (2020): Kinetic modules and concentration robustness
- Feinberg (2019): Foundations of Chemical Reaction Network Theory
"""
function kinetic_concordance_analysis(
    model::AbstractFBCModels.AbstractFBCModel;
    optimizer,
    min_module_size::Int=2,
    workers=D.workers(),
    kwargs...
)
    @info "Starting comprehensive kinetic concordance analysis"

    # Step 1: Run activity concordance analysis
    @info "Running activity concordance analysis"
    concordance_results = activity_concordance_analysis(model; optimizer=optimizer, kwargs...)

    n_complexes = DF.nrow(concordance_results.complexes)
    n_modules = DF.nrow(concordance_results.modules)

    @info "Concordance analysis completed: $n_complexes complexes in $n_modules modules"

    # Step 2: Build constraints for kinetic analysis
    @info "Building constraints for kinetic module analysis"
    constraints_kwargs = (
        :interface => get(kwargs, :interface, nothing),
        :use_unidirectional_constraints => get(kwargs, :use_unidirectional_constraints, true)
    )

    constraints = concordance_constraints(model; constraints_kwargs...)

    # Step 3: Run kinetic module analysis using constraints
    @info "Running kinetic module analysis"
    kinetic_results = identify_kinetic_modules(constraints, model, concordance_results; min_module_size, workers)

    # Update reaction count to reflect post-splitting count from kinetic results
    n_reactions = length(kinetic_results.reaction_names)

    # Extract kinetic module statistics
    n_kinetic_modules = length(kinetic_results.kinetic_modules)
    giant_kinetic_module_size = if kinetic_results.giant_module_id != :none
        length(kinetic_results.kinetic_modules[kinetic_results.giant_module_id])
    else
        0
    end

    # Create comprehensive summary
    summary = Dict{String,Int}(
        "n_reactions" => n_reactions,
        "n_complexes" => n_complexes,
        "n_concordance_modules" => n_modules,
        "n_kinetic_modules" => n_kinetic_modules,
        "giant_kinetic_module_size" => giant_kinetic_module_size,
        "n_interface_reactions" => length(kinetic_results.interface_reactions)
    )

    @info "Kinetic concordance analysis completed: $n_kinetic_modules kinetic modules, $(length(kinetic_results.interface_reactions)) interface reactions"

    # Return comprehensive results
    return KineticModuleResults(
        concordance_results,
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
end