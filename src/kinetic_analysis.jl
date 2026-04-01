"""
Kinetic Module Analysis with Set-based Interface

Clean implementation that works natively with Vector{Set{Symbol}} structure
for concordance modules, avoiding index mapping issues.

Uses the simplified 2-phase upstream algorithm from the paper (Remark S2-1).
"""

import Graphs
import LinearAlgebra: norm, dot
import Base.Threads

# Forward declarations of types used throughout the module
"""
    CachedAdjacency

Pre-computed adjacency structure for efficient graph traversal.
Avoids repeated sparse matrix indexing in hot loops.
"""
struct CachedAdjacency
    out_neighbors::Vector{Vector{Int}}   # out_neighbors[v] = complexes v can convert to
    in_neighbors::Vector{Vector{Int}}    # in_neighbors[v] = complexes that can convert to v
    has_external_in::Vector{Bool}        # has_external_in[v] = true if v has incoming from any source
end

"""
    build_cached_adjacency(A_matrix, n_complexes)

Build cached adjacency lists from incidence matrix for O(1) neighbor lookup.
Replaces repeated `findnz` calls with pre-computed adjacency.
"""
function build_cached_adjacency(A_matrix::SparseArrays.SparseMatrixCSC, n_complexes::Int)
    out_neighbors = [Int[] for _ in 1:n_complexes]
    in_neighbors = [Int[] for _ in 1:n_complexes]
    has_external_in = fill(false, n_complexes)

    rows = SparseArrays.rowvals(A_matrix)
    vals = SparseArrays.nonzeros(A_matrix)
    n_reactions = size(A_matrix, 2)

    for rxn_idx in 1:n_reactions
        substrates = Int[]
        products = Int[]

        for idx in SparseArrays.nzrange(A_matrix, rxn_idx)
            row = rows[idx]
            val = vals[idx]
            if val < 0
                push!(substrates, row)
            elseif val > 0
                push!(products, row)
            end
        end

        # Build adjacency: substrates -> products
        for sub in substrates
            for prod in products
                push!(out_neighbors[sub], prod)
                push!(in_neighbors[prod], sub)
                has_external_in[prod] = true
            end
        end
    end

    return CachedAdjacency(out_neighbors, in_neighbors, has_external_in)
end

"""
    kinetic_analysis(concordance_modules, model; min_module_size=1, known_acr=Symbol[], efficient=true)

Apply kinetic module analysis using set-based concordance module structure.

Implements the iterative refinement algorithm from Section S.4.1 with three feedback loops:
1. Proposition S4-1: Coupling merges → Concordance merges
2. Remark S3-6: ACR identification → Enhanced coupling via augmented Y𝚫 (only if `efficient=false`)
3. Theorem S4-6: If δₖ = 1, all non-terminal complexes are coupled (only if `efficient=false`)

# Arguments
- `concordance_modules`: Vector{Set{Symbol}} where:
  - `concordance_modules[1]` = balanced complexes (module 0)
  - `concordance_modules[2+]` = concordance modules 1, 2, ...
- `model`: AbstractFBCModel for network topology
- `min_module_size`: Minimum size for reported modules (default: 1)
- `known_acr`: Vector of metabolite IDs with known ACR (from external sources)
- `efficient`: Boolean flag for performance optimization (default: `true`)
    - `true`: Fast pairwise ACR/ACRR detection, trivial merging only (no matrix inversions/rank checks).
    - `false`: Full analysis including matrix-based ACR/ACRR, Proposition S3-4 advanced merging, and deficiency checks.

# Returns
- `Vector{Set{Symbol}}`: Kinetic modules (sets of complex IDs), sorted by size (largest first)

# Examples
```julia
concordance = [
    Set([:A, :C, :F]),              # balanced
    Set([:B, :D, Symbol("D+E")]),   # concordance module 1
    Set([Symbol("C+G")])             # concordance module 2
]

# Include all modules (including singletons)
results = kinetic_analysis(concordance, model)

# Exclude singleton modules (focus on coupled complexes)
results = kinetic_analysis(concordance, model; min_module_size=2)

# With known ACR metabolites
results = kinetic_analysis(concordance, model; known_acr=[:metabolite_X])
```
"""
function kinetic_analysis(
    concordance_modules::Vector{Set{Symbol}},
    model::A.AbstractFBCModel;
    min_module_size::Int=1,
    known_acr::Vector{Symbol}=Symbol[],
    efficient::Bool=true
)
    # Map efficient flag to advanced merging control
    # efficient=true  -> enable_advanced_merging=false (trivial merging only)
    # efficient=false -> enable_advanced_merging=true (full matrix merging)
    enable_advanced_merging = !efficient

    @debug "Starting kinetic module analysis" n_concordance_modules = length(concordance_modules) - 1 efficient

    # Extract network topology and build ID mappings ONCE
    A_matrix, complex_ids = incidence(model; return_ids=true)
    Y_matrix, metabolite_ids, _ = complex_stoichiometry(model; return_ids=true)

    # Build mappings once and reuse throughout
    complex_to_idx = Dict{Symbol,Int}(id => i for (i, id) in enumerate(complex_ids))

    # Build augmentation for known ACR metabolites (Remark S3-6)
    # Works in both modes - helps with merging even in efficient mode
    acr_augmentation = build_acr_augmentation(known_acr, metabolite_ids, size(Y_matrix, 1))

    if !isempty(known_acr)
        @debug "Using known ACR metabolites" n_known_acr = length(known_acr)
    end

    # OPTIMIZATION: Pre-compute adjacency structure for efficient graph traversal
    # Avoids repeated sparse matrix indexing in find_entry_complexes and tarjan_scc
    cached_adj = build_cached_adjacency(A_matrix, length(complex_ids))

    # Store in a named tuple for easy passing
    network = (
        A=A_matrix,
        Y=Y_matrix,
        complex_ids=complex_ids,
        metabolite_ids=metabolite_ids,
        complex_to_idx=complex_to_idx,
        acr_augmentation=acr_augmentation,
        enable_advanced_merging=enable_advanced_merging,
        cached_adj=cached_adj
    )

    # Extract balanced and unbalanced modules (keep ALL modules including singletons)
    # Singletons can still contribute to cross-module couplings via Proposition S3-4
    # Filtering by min_module_size happens only at the end when returning results
    balanced = concordance_modules[1]
    unbalanced_modules = concordance_modules[2:end]

    # Keep all concordance modules for analysis
    filtered_concordance = [balanced; unbalanced_modules]

    # Compute initial structural deficiency only if doing full analysis
    initial_delta = -1
    if !efficient
        initial_delta = compute_structural_deficiency(filtered_concordance, network)
    end

    # Initial check for δₖ = 1 (only if efficient=false)
    if !efficient
        current_concordance = copy(filtered_concordance)
        if length(current_concordance) == 2  # balanced + 1 unbalanced
            @debug "Initial check: δₖ = 1 detected"
            kinetic_modules = apply_theorem_s4_6(Set{Symbol}[], current_concordance, network)
            valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)
            sort!(valid_modules, by=length, rev=true)

            # Detect ACR/ACRR using already-available network data
            # Always use efficient=true for ACR detection (faster and more complete pairwise detection)
            acr_results = _detect_acr_acrr(valid_modules, Y_matrix, metabolite_ids, complex_to_idx;
                efficient=efficient)
            return (
                kinetic_modules=valid_modules,
                acr_metabolites=acr_results.acr_metabolites,
                acrr_pairs=acr_results.acrr_pairs
            )
        end
    else
        current_concordance = filtered_concordance
    end

    # Iterative refinement loop with convergence detection
    # Continue until we reach a fixed point (no modules change)
    current_known_acr = Set{Symbol}(known_acr)

    outer_iteration = 0
    max_iterations = 100  # Safety limit to prevent infinite loops

    # Track state for convergence detection
    previous_kinetic_modules = Set{Symbol}[]
    previous_concordance_state = Set{Symbol}[]

    # Track merges for deficiency calculation
    n_concordance_merges = 0

    while outer_iteration < max_iterations
        outer_iteration += 1
        @debug "=== Iteration $outer_iteration ===" efficient

        balanced = current_concordance[1]

        # Update network with current known ACR for merging
        # In efficient mode: still use ACR but skip expensive matrix operations
        if !isempty(current_known_acr) && outer_iteration > 1
            acr_augmentation = build_acr_augmentation(collect(current_known_acr), metabolite_ids, size(Y_matrix, 1))
            network = (
                A=network.A, Y=network.Y, complex_ids=network.complex_ids,
                metabolite_ids=network.metabolite_ids, complex_to_idx=network.complex_to_idx,
                acr_augmentation=acr_augmentation,
                enable_advanced_merging=enable_advanced_merging,
                cached_adj=network.cached_adj
            )
            @debug "Updated network with ACR augmentation" n_acr = length(current_known_acr)
        end

        # Step 1: Compute upstream sets (parallelized)
        n_modules = length(current_concordance) - 1
        upstream_results = Vector{Union{Nothing,Set{Symbol}}}(undef, n_modules)

        Threads.@threads for idx in 1:n_modules
            conc_module = current_concordance[idx+1]
            extended_module = balanced ∪ conc_module
            upstream = upstream_algorithm(extended_module, network)
            upstream_results[idx] = isempty(upstream) ? nothing : upstream
        end

        # Collect non-empty results into properly typed vector
        upstream_sets = Set{Symbol}[r for r in upstream_results if !isnothing(r)]

        if isempty(upstream_sets)
            return (
                kinetic_modules=Set{Symbol}[],
                acr_metabolites=Symbol[],
                acrr_pairs=Tuple{Symbol,Symbol}[]
            )
        end

        # Step 2: Merge coupled modules
        # Use simple merging if efficient=true (handled by enable_advanced_merging=false in network)
        kinetic_modules, merge_map = merge_coupled_sets_tracked(upstream_sets, network)

        # Step 3: Check concordance merges
        previous_n_unbalanced = length(current_concordance) - 1
        concordance_changed = apply_concordance_merging!(current_concordance, merge_map)

        if concordance_changed && !efficient
            n_concordance_merges += (previous_n_unbalanced - (length(current_concordance) - 1))
        end

        # Step 4: Identify ACR and trigger feedback loop
        # Works in both efficient and full modes (just uses different detection methods)

        # Identify ACR from current kinetic modules
        acr_results = identify_acr_acrr(kinetic_modules, model; efficient=efficient)
        newly_identified_acr = setdiff(Set(acr_results.acr_metabolites), current_known_acr)

        if !isempty(newly_identified_acr)
            union!(current_known_acr, newly_identified_acr)
            @debug "Identified new ACR metabolites" count = length(newly_identified_acr) acr_list = collect(newly_identified_acr)
        end

        # Check convergence: Has the kinetic module partition changed?
        # Convert to canonical form for comparison (sorted sets of sorted symbols)
        current_state = Set(Set(sort(collect(km), by=string)) for km in kinetic_modules)
        concordance_state = Set(Set(sort(collect(cm), by=string)) for cm in current_concordance)

        converged = (current_state == previous_kinetic_modules &&
                     concordance_state == previous_concordance_state)

        if converged
            @debug "Convergence detected - no changes in modules" iteration = outer_iteration
            break
        end

        # Store current state for next iteration
        previous_kinetic_modules = current_state
        previous_concordance_state = concordance_state
    end

    # Check if we hit max iterations (should rarely happen with proper convergence)
    if outer_iteration >= max_iterations
        @warn "Maximum iterations reached without convergence" max_iterations
    else
        @debug "Algorithm converged" iterations = outer_iteration
    end

    # Final deficiency check for full mode
    if !efficient && initial_delta >= 0
        (is_delta_k_one, should_merge) = check_mass_action_deficiency(
            current_concordance, n_concordance_merges, initial_delta, network
        )
        if is_delta_k_one && should_merge
            # Merge all unbalanced
            all_unbalanced = reduce(∪, current_concordance[2:end]; init=Set{Symbol}())
            current_concordance = [current_concordance[1], all_unbalanced]

            # Recompute modules with merged concordance
            upstream_sets = Set{Symbol}[]
            balanced = current_concordance[1]
            for conc_module in current_concordance[2:end]
                extended_module = balanced ∪ conc_module
                upstream = upstream_algorithm(extended_module, network)
                !isempty(upstream) && push!(upstream_sets, upstream)
            end

            kinetic_modules, _ = merge_coupled_sets_tracked(upstream_sets, network)
        end
    end

    # Finalize with singletons and balanced modules
    kinetic_modules = add_singleton_balanced(kinetic_modules, current_concordance[1], current_concordance[2:end], network)

    # Final ACR-based merging pass for singletons added by add_singleton_balanced
    # This handles cases like {A+C} merging with {F, A, C+F, E, B+D}
    if !isempty(current_known_acr)
        @debug "Final ACR-based merging pass" n_modules_before = length(kinetic_modules)

        # Try merging with known ACR (converting to vector form expected by merge_coupled_sets)
        final_merged = merge_coupled_sets(kinetic_modules, network)

        if length(final_merged) < length(kinetic_modules)
            @debug "Final ACR merging reduced module count" before = length(kinetic_modules) after = length(final_merged)
            kinetic_modules = final_merged
        end
    end

    if !efficient
        kinetic_modules = apply_theorem_s4_6(kinetic_modules, current_concordance, network)
    end

    valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)
    sort!(valid_modules, by=length, rev=true)

    # Detect ACR/ACRR using already-available network data (no redundant extraction)
    acr_results = _detect_acr_acrr(valid_modules, Y_matrix, metabolite_ids, complex_to_idx;
        efficient=efficient)

    return (
        kinetic_modules=valid_modules,
        acr_metabolites=acr_results.acr_metabolites,
        acrr_pairs=acr_results.acrr_pairs
    )
end

"""
    kinetic_analysis(concordance_result::NamedTuple, model; kwargs...)

Run kinetic module analysis on the output of `activity_concordance_analysis`.

Returns an updated NamedTuple with `kinetic_module`, `acr`, and `acrr` fields populated.

# Example
```julia
result = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer)
result = kinetic_analysis(result, model)

# Or in one call:
result = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, kinetic_analysis=true)
```
"""
function kinetic_analysis(
    concordance_result::NamedTuple,
    model::A.AbstractFBCModel;
    min_module_size::Int=1,
    known_acr::Vector{Symbol}=Symbol[],
    efficient::Bool=true
)
    complexes = concordance_result.complexes
    complex_ids = Symbol.(complexes.complex_id)

    # Use extract_concordance_modules to get Vector{Set{Symbol}} for the core algorithm
    concordance_modules = extract_concordance_modules(concordance_result)

    # Run the core kinetic analysis
    kin_result = kinetic_analysis(concordance_modules, model;
        min_module_size=min_module_size, known_acr=known_acr, efficient=efficient)

    # Map kinetic modules (Vector{Set{Symbol}}) back to per-complex Int assignments
    n = length(complex_ids)
    km_mapping = zeros(Int, n)
    complex_to_idx = Dict{Symbol,Int}(id => i for (i, id) in enumerate(complex_ids))
    for (mod_id, mod_set) in enumerate(kin_result.kinetic_modules)
        for cid in mod_set
            if haskey(complex_to_idx, cid)
                km_mapping[complex_to_idx[cid]] = mod_id
            end
        end
    end

    # Build updated complexes table with kinetic_module filled in
    updated_complexes = if haskey(complexes, :min_activity)
        (
            complex_id         = complexes.complex_id,
            concordance_module = complexes.concordance_module,
            kinetic_module     = km_mapping,
            classification     = complexes.classification,
            min_activity       = complexes.min_activity,
            max_activity       = complexes.max_activity,
            lambda             = complexes.lambda,
            trivially_balanced = complexes.trivially_balanced,
        )
    else
        (
            complex_id         = complexes.complex_id,
            concordance_module = complexes.concordance_module,
            kinetic_module     = km_mapping,
            classification     = complexes.classification,
        )
    end

    # Build updated ACR/ACRR tables
    acr = (metabolite_id = String.(kin_result.acr_metabolites),)
    acrr = (
        metabolite_1 = String[String(p[1]) for p in kin_result.acrr_pairs],
        metabolite_2 = String[String(p[2]) for p in kin_result.acrr_pairs],
    )

    if haskey(concordance_result, :lambda_pairs)
        return (complexes=updated_complexes, acr=acr, acrr=acrr,
                lambda_pairs=concordance_result.lambda_pairs)
    end
    return (complexes=updated_complexes, acr=acr, acrr=acrr)
end

"""
    upstream_algorithm(extended_module, network)

Apply simplified 2-phase upstream algorithm to identify kinetic module (Remark S2-1 from paper).

# Phases
1. Phase I: Remove entry complexes (autonomous property)
2. Phase IV: Remove terminal strong linkage classes (feeding property)

Note: Phases II and III are skipped as per Remark S2-1 (Simplified Upstream Algorithm).
Returns the upstream set (complexes satisfying both properties).
"""
function upstream_algorithm(
    extended_module::Set{Symbol},
    network::NamedTuple
)
    # Unpack network structures
    A_matrix = network.A
    complex_ids = network.complex_ids
    complex_to_idx = network.complex_to_idx

    # Convert to indices for matrix operations
    current_indices = Set(complex_to_idx[c] for c in extended_module if haskey(complex_to_idx, c))

    @debug "Starting upstream algorithm (simplified 2-phase)" extended_size = length(extended_module) indices_found = length(current_indices)
    remaining = [complex_ids[i] for i in current_indices]
    @debug "  Extended module complexes: $remaining"

    # Phase I: Remove entry complexes (autonomous property)
    @debug "Phase I: Removing entry complexes" thread_id = Threads.threadid() initial_size = length(current_indices)
    iteration = 0
    max_phase1_iterations = 1000  # Safety limit
    last_size = length(current_indices)

    # Use cached adjacency if available for O(1) neighbor lookup
    has_cached_adj = haskey(network, :cached_adj)

    while iteration < max_phase1_iterations
        iteration += 1
        entries = if has_cached_adj
            find_entry_complexes_cached(current_indices, network.cached_adj)
        else
            find_entry_complexes_idx(current_indices, A_matrix)
        end

        # Progress update every 10 iterations or when finding entries
        if !isempty(entries) || iteration % 10 == 0
            @debug "  Phase I iteration $iteration" thread_id = Threads.threadid() n_remaining = length(current_indices) n_entries = length(entries)
        end

        isempty(entries) && break
        setdiff!(current_indices, entries)

        if isempty(current_indices)
            @debug "  All complexes removed in Phase I" thread_id = Threads.threadid()
            return Set{Symbol}()
        end

        # Detect if stuck (no progress)
        current_size = length(current_indices)
        if current_size == last_size
            @warn "Phase I appears stuck - no progress in iteration $iteration" thread_id = Threads.threadid() n_complexes = current_size
            break
        end
        last_size = current_size
    end

    if iteration >= max_phase1_iterations
        @warn "Phase I exceeded maximum iterations" thread_id = Threads.threadid() max_iterations = max_phase1_iterations n_complexes = length(current_indices)
    end

    remaining = [complex_ids[i] for i in current_indices]
    remaining = [complex_ids[i] for i in current_indices]
    @debug "Phase IV: Removing terminal strong linkage classes" remaining

    # Phase IV: Find and remove all terminal strong linkage classes using Tarjan's algorithm
    # As per Remark S2-1, we identify terminal SCCs in the current set (C_-4).
    # Exit complexes correspond to NON-TERMINAL SCCs (because they have edges to outside)
    # and are thus preserved.
    # Use cached adjacency if available for O(1) neighbor lookup
    sccs = if has_cached_adj
        tarjan_scc_cached(current_indices, network.cached_adj)
    else
        tarjan_scc(current_indices, A_matrix)
    end
    @debug "  Found $(length(sccs)) strongly connected components"

    # Debug: Show all SCCs with their terminal status
    @debug "  SCC analysis:" begin
        for (i, scc) in enumerate(sccs)
            scc_symbols = [complex_ids[j] for j in scc]
            is_term = if has_cached_adj
                is_terminal_scc_cached(scc, current_indices, network.cached_adj)
            else
                is_terminal_scc_idx(scc, current_indices, A_matrix)
            end
            @debug "    SCC $i (size $(length(scc))): $scc_symbols - Terminal: $is_term"
        end
    end

    # Identify terminal/non-terminal SCCs WITHOUT modifying current_indices
    # CRITICAL: Must check all SCCs against the SAME set of complexes
    original_indices = copy(current_indices)
    terminal_sccs = Set{Int}[]
    non_terminal_sccs = Set{Int}[]

    for scc in sccs
        scc_symbols = [complex_ids[i] for i in scc]
        is_term = if has_cached_adj
            is_terminal_scc_cached(scc, original_indices, network.cached_adj)
        else
            is_terminal_scc_idx(scc, original_indices, A_matrix)
        end
        @debug "  SCC (size $(length(scc))): $scc_symbols - Terminal: $is_term"

        if is_term
            push!(terminal_sccs, scc)
        else
            push!(non_terminal_sccs, scc)
        end
    end

    # Remove all terminal SCCs
    for scc in terminal_sccs
        setdiff!(current_indices, scc)
    end

    @debug "  Removed $(length(terminal_sccs)) terminal SCCs, kept $(length(non_terminal_sccs)) non-terminal SCCs"

    # Convert back to symbols using complex_ids vector
    upstream_set = Set(complex_ids[i] for i in current_indices if i <= length(complex_ids))

    @debug "Upstream algorithm complete" upstream_size = length(upstream_set) complexes = sort(collect(upstream_set), by=string)

    return upstream_set
end

"""
Find entry complexes using index-based operations.

An entry complex has an incoming reaction from a complex outside the current set.
"""
function find_entry_complexes_idx(complexes::Set{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    entries = Set{Int}()
    sizehint!(entries, length(complexes) >> 2)  # Preallocate (expect ~25% entries)

    @inbounds for cidx in complexes
        # Check reactions where this complex is produced
        for rxn_idx in SparseArrays.findnz(A_matrix[cidx, :])[1]
            if A_matrix[cidx, rxn_idx] > 0  # Complex is product
                # Check if any substrate is outside our set
                substrates = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                has_external = any(s -> A_matrix[s, rxn_idx] < 0 && s ∉ complexes, substrates)

                if has_external
                    push!(entries, cidx)
                    break
                end
            end
        end
    end

    return entries
end

"""
Find entry complexes using cached adjacency (optimized version).

An entry complex has an incoming edge from a complex outside the current set.
Uses pre-computed adjacency lists instead of repeated sparse matrix indexing.
"""
function find_entry_complexes_cached(complexes::Set{Int}, cached_adj::CachedAdjacency)
    entries = Set{Int}()
    sizehint!(entries, length(complexes) >> 2)  # Preallocate (expect ~25% entries)

    @inbounds for cidx in complexes
        # Check if any in-neighbor is outside our set
        for in_neighbor in cached_adj.in_neighbors[cidx]
            if in_neighbor ∉ complexes
                push!(entries, cidx)
                break
            end
        end
    end

    return entries
end

"""
    tarjan_scc(complexes, A_matrix)

Find all strongly connected components using Tarjan's algorithm.

Returns a vector of SCCs, where each SCC is a Set{Int} of complex indices.
"""
function tarjan_scc(complexes::Set{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    # Convert set to vector for indexing
    nodes = collect(complexes)
    n = length(nodes)

    # Initialize Tarjan's algorithm state with preallocation
    index = 0
    stack = Vector{Int}()
    sizehint!(stack, n)
    indices = Dict{Int,Int}()
    sizehint!(indices, n)
    lowlinks = Dict{Int,Int}()
    sizehint!(lowlinks, n)
    on_stack = Set{Int}()
    sizehint!(on_stack, n)
    sccs = Vector{Set{Int}}()
    sizehint!(sccs, max(1, n >> 3))  # Estimate ~n/8 SCCs

    function strongconnect(v::Int)
        # Set the depth index for v
        indices[v] = index
        lowlinks[v] = index
        index += 1
        push!(stack, v)
        push!(on_stack, v)

        # Consider successors of v (complexes that v converts to)
        for rxn_idx in SparseArrays.findnz(A_matrix[v, :])[1]
            if A_matrix[v, rxn_idx] < 0  # v is substrate in this reaction
                # Find products of this reaction
                for w in SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                    if A_matrix[w, rxn_idx] > 0 && w ∈ complexes  # w is product and in our set
                        if !haskey(indices, w)
                            # Successor w has not yet been visited; recurse on it
                            strongconnect(w)
                            lowlinks[v] = min(lowlinks[v], lowlinks[w])
                        elseif w ∈ on_stack
                            # Successor w is in stack and hence in the current SCC
                            lowlinks[v] = min(lowlinks[v], indices[w])
                        end
                    end
                end
            end
        end

        # If v is a root node, pop the stack and create an SCC
        if lowlinks[v] == indices[v]
            scc = Set{Int}()
            while true
                w = pop!(stack)
                delete!(on_stack, w)
                push!(scc, w)
                if w == v
                    break
                end
            end
            push!(sccs, scc)
        end
    end

    # Run Tarjan's algorithm on all nodes
    for v in nodes
        if !haskey(indices, v)
            strongconnect(v)
        end
    end

    return sccs
end

"""
Check if SCC is terminal (no outgoing edges to other complexes in the current set).

A terminal SCC has no complex that converts to any complex outside the SCC
(but still within the current set of complexes being considered).
"""
function is_terminal_scc_idx(scc::Set{Int}, all_complexes::Set{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    # Check if any complex in SCC has outgoing edge to complex outside SCC (but in all_complexes)
    for cidx in scc
        for rxn_idx in SparseArrays.findnz(A_matrix[cidx, :])[1]
            if A_matrix[cidx, rxn_idx] < 0  # Complex is substrate
                # Check products
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                for pidx in products
                    if A_matrix[pidx, rxn_idx] > 0 && pidx ∉ scc
                        @debug "SCC contains complex $cidx which converts (via rxn $rxn_idx) to $pidx outside SCC → NON-TERMINAL"
                        return false  # Has edge outside SCC
                    end
                end
            end
        end
    end

    return true
end

"""
    tarjan_scc_cached(complexes, cached_adj)

Find all strongly connected components using Tarjan's algorithm with cached adjacency.
Optimized version that avoids repeated sparse matrix indexing.
"""
function tarjan_scc_cached(complexes::Set{Int}, cached_adj::CachedAdjacency)
    nodes = collect(complexes)
    n = length(nodes)

    # Initialize Tarjan's algorithm state with preallocation
    index = 0
    stack = Vector{Int}()
    sizehint!(stack, n)
    indices = Dict{Int,Int}()
    sizehint!(indices, n)
    lowlinks = Dict{Int,Int}()
    sizehint!(lowlinks, n)
    on_stack = Set{Int}()
    sizehint!(on_stack, n)
    sccs = Vector{Set{Int}}()
    sizehint!(sccs, max(1, n >> 3))

    function strongconnect(v::Int)
        indices[v] = index
        lowlinks[v] = index
        index += 1
        push!(stack, v)
        push!(on_stack, v)

        # Use cached out_neighbors instead of sparse matrix traversal
        @inbounds for w in cached_adj.out_neighbors[v]
            if w ∈ complexes  # w is in our current set
                if !haskey(indices, w)
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elseif w ∈ on_stack
                    lowlinks[v] = min(lowlinks[v], indices[w])
                end
            end
        end

        if lowlinks[v] == indices[v]
            scc = Set{Int}()
            while true
                w = pop!(stack)
                delete!(on_stack, w)
                push!(scc, w)
                if w == v
                    break
                end
            end
            push!(sccs, scc)
        end
    end

    for v in nodes
        if !haskey(indices, v)
            strongconnect(v)
        end
    end

    return sccs
end

"""
Check if SCC is terminal using cached adjacency (optimized version).
"""
function is_terminal_scc_cached(scc::Set{Int}, all_complexes::Set{Int}, cached_adj::CachedAdjacency)
    @inbounds for cidx in scc
        for neighbor in cached_adj.out_neighbors[cidx]
            if neighbor ∉ scc
                return false  # Has edge outside SCC
            end
        end
    end
    return true
end

"""
    merge_coupled_sets_tracked(upstream_sets, network)

Merge coupled upstream sets and track which modules merged together.
Returns (merged_modules, merge_map) where merge_map[i] gives the final group ID for module i.
"""
function merge_coupled_sets_tracked(upstream_sets::Vector{Set{Symbol}}, network::NamedTuple)
    kinetic_modules = merge_coupled_sets(upstream_sets, network)

    # Build merge map by checking which original modules ended up in which final groups
    n = length(upstream_sets)
    merge_map = zeros(Int, n)

    for i in 1:n
        for (group_id, final_module) in enumerate(kinetic_modules)
            if !isdisjoint(upstream_sets[i], final_module)
                merge_map[i] = group_id
                break
            end
        end
    end

    return kinetic_modules, merge_map
end

"""
    apply_concordance_merging!(concordance_modules, merge_map)

Apply Proposition S4-1: When coupling sets merge, their concordance modules also merge.
Returns true if any merges occurred, false otherwise.
"""
function apply_concordance_merging!(concordance_modules::Vector{Set{Symbol}}, merge_map::Vector{Int})
    if length(merge_map) + 1 != length(concordance_modules)
        # Mismatch - can't apply merging
        return false
    end

    # Find which concordance modules should be merged
    groups = Dict{Int,Set{Int}}()
    for (i, group_id) in enumerate(merge_map)
        if !haskey(groups, group_id)
            groups[group_id] = Set{Int}()
        end
        push!(groups[group_id], i)
    end

    # Check if any group has more than one module (indicating a merge)
    any_merged = any(length(group) > 1 for group in values(groups))

    if !any_merged
        return false
    end

    # Merge concordance modules
    @debug "Applying Proposition S4-1: Merging concordance modules"
    balanced = concordance_modules[1]
    new_concordance = [balanced]

    for (group_id, module_indices) in sort(collect(groups), by=first)
        merged_module = Set{Symbol}()
        for idx in module_indices
            Base.union!(merged_module, concordance_modules[idx+1])  # +1 because concordance_modules[1] is balanced
        end

        if length(module_indices) > 1
            @debug "  Merging concordance modules $(collect(module_indices)) (coupling sets merged)"
        end

        push!(new_concordance, merged_module)
    end

    # Replace concordance_modules in-place
    empty!(concordance_modules)
    append!(concordance_modules, new_concordance)

    return true
end

"""
Merge coupled upstream sets using Proposition S3-4.

Two coupling sets can be merged if:
1. They share complexes directly (trivial merging - Lemma S3-1), OR
2. For complexes C_α ∈ 𝒞_i and C_β ∈ 𝒞_j: Y[:,α] - Y[:,β] ∈ im(Y𝚫)
   (Proposition S3-4 - their stoichiometric difference lies in the span of coupled complex differences)
"""
function merge_coupled_sets(upstream_sets::Vector{Set{Symbol}}, network::NamedTuple)
    n = length(upstream_sets)
    isempty(upstream_sets) && return Set{Symbol}[]

    # Unpack network structures (no recreating mappings!)
    Y_matrix = network.Y
    complex_to_idx = network.complex_to_idx

    # Build union-find structure
    parent = collect(1:n)

    function find_root(idx::Int)
        while parent[idx] != idx
            parent[idx] = parent[parent[idx]]  # Path compression
            idx = parent[idx]
        end
        return idx
    end

    function union_indices!(idx1::Int, idx2::Int)
        root_i = find_root(idx1)
        root_j = find_root(idx2)
        if root_i != root_j
            parent[root_j] = root_i
            return true
        end
        return false
    end

    # Step 1: Trivial merging - merge sets that share complexes directly (Lemma S3-1)
    @debug "Step 1: Trivial merging (Lemma S3-1)"
    for i in 1:n
        for j in (i+1):n
            if !isdisjoint(upstream_sets[i], upstream_sets[j])
                union_indices!(i, j)
                @debug "  Merging modules $i and $j (shared complexes)" shared = upstream_sets[i] ∩ upstream_sets[j]
            end
        end
    end

    # Step 1.5: ACR-based merging - merge sets where complexes differ only by ACR species
    # This handles cases like {A} and {A+C} where C is ACR
    if haskey(network, :acr_augmentation) && size(network.acr_augmentation, 2) > 0
        @debug "Step 1.5: ACR-based merging (Remark S3-6)"

        # Get ACR metabolite indices from augmentation matrix
        acr_indices = Set{Int}()
        for col in 1:size(network.acr_augmentation, 2)
            # Each column is a unit vector for an ACR metabolite
            nz_idx = findfirst(x -> abs(x) > 1e-10, network.acr_augmentation[:, col])
            if !isnothing(nz_idx)
                push!(acr_indices, nz_idx)
            end
        end

        if !isempty(acr_indices)
            for i in 1:n
                for j in (i+1):n
                    # Skip if already merged
                    if find_root(i) == find_root(j)
                        continue
                    end

                    # Check if any pair of complexes between the two sets differ only by ACR species
                    for c_i in upstream_sets[i]
                        idx_i = get(complex_to_idx, c_i, 0)
                        idx_i == 0 && continue

                        for c_j in upstream_sets[j]
                            idx_j = get(complex_to_idx, c_j, 0)
                            idx_j == 0 && continue

                            # Compute stoichiometric difference
                            diff = Y_matrix[:, idx_i] - Y_matrix[:, idx_j]
                            nz_indices = SparseArrays.findnz(diff)[1]

                            # If all differences are in ACR metabolites, merge!
                            if !isempty(nz_indices) && all(idx -> idx in acr_indices, nz_indices)
                                if union_indices!(i, j)
                                    @debug "  Merging modules $i and $j (ACR difference)" complexes = (c_i, c_j) acr_diff = [network.metabolite_ids[idx] for idx in nz_indices if idx <= length(network.metabolite_ids)]
                                end
                                @goto next_pair_acr
                            end
                        end
                    end
                    @label next_pair_acr
                end
            end
        end
    end

    # Step 2: Advanced merging via Proposition S3-4
    # Check if Y[:,α] - Y[:,β] ∈ im(Y𝚫) for complexes from different coupling sets
    if !get(network, :enable_advanced_merging, true)
        @debug "Skipping Step 2: Advanced merging (disabled)"

        # Return results from Step 1
        groups = Dict{Int,Set{Symbol}}()
        for i in 1:n
            root = find_root(i)
            if !haskey(groups, root)
                groups[root] = Set{Symbol}()
            end
            Base.union!(groups[root], upstream_sets[i])
        end

        return collect(values(groups))
    end

    @debug "Step 2: Advanced merging via Proposition S3-4"

    # Build coupling companion map 𝚫 from current upstream sets
    # 𝚫 = [𝚫1 ... 𝚫q] where each 𝚫i encodes coupling relations in upstream_sets[i]
    Y_Delta = build_coupling_companion_matrix(upstream_sets, Y_matrix, complex_to_idx)

    # Augment with known ACR columns (Remark S3-6)
    if haskey(network, :acr_augmentation) && size(network.acr_augmentation, 2) > 0
        Y_Delta = hcat(Y_Delta, network.acr_augmentation)
        @debug "  Augmented Y𝚫 with known ACR metabolites" n_augmentation_cols = size(network.acr_augmentation, 2)
    end

    # OPTIMIZATION: Build cached QR decomposition ONCE for all merge checks
    # According to Remark S3-3, merging coupling sets does NOT alter the column span of Y𝚫.
    # Therefore, we do not need to rebuild or update Y𝚫 after merges.
    # A single pass over all pairs is sufficient to find all merging opportunities.
    # The Union-Find structure handles the transitive closure of merges.
    cache = build_cached_column_span(Y_Delta)
    @debug "  Built cached column span" rank = cache.rank n_cols = size(Y_Delta, 2)

    # OPTIMIZATION: Precompute stoichiometric vectors for all relevant complexes
    # Avoids repeated sparse-to-dense conversions in the loop
    n_metabolites = size(Y_matrix, 1)
    complex_vectors = Dict{Symbol,Vector{Float64}}()
    for upstream_set in upstream_sets
        c = first(upstream_set)  # We only need one complex per set (Lemma S3-3)
        if haskey(complex_to_idx, c) && !haskey(complex_vectors, c)
            complex_vectors[c] = Vector(Y_matrix[:, complex_to_idx[c]])
        end
    end

    # OPTIMIZATION: Collect pairs to check, then parallelize
    # Build list of (i, j, c_alpha, c_beta) tuples for pairs that need checking
    pairs_to_check = Tuple{Int,Int,Symbol,Symbol}[]
    for i in 1:n
        c_alpha = first(upstream_sets[i])
        haskey(complex_vectors, c_alpha) || continue

        for j in (i+1):n
            # Skip if already merged from Step 1/1.5
            find_root(i) == find_root(j) && continue

            c_beta = first(upstream_sets[j])
            haskey(complex_vectors, c_beta) || continue

            push!(pairs_to_check, (i, j, c_alpha, c_beta))
        end
    end

    @debug "  Checking $(length(pairs_to_check)) pairs for Proposition S3-4 merging"

    # OPTIMIZATION: Parallel merge check using thread-local result arrays
    # Avoids Channel synchronization overhead by accumulating results per-thread
    merge_count = 0

    if length(pairs_to_check) > 0
        # Preallocate workspace per thread for y_diff computation
        # Use maxthreadid() instead of nthreads() to handle interactive threads in Julia 1.9+
        max_tid = Threads.maxthreadid()
        workspaces = [Vector{Float64}(undef, n_metabolites) for _ in 1:max_tid]

        # OPTIMIZATION: Additional workspaces for in-place BLAS operations in is_in_span!
        # coeffs_workspace: size = cache.rank (for Q' * v result)
        # proj_workspace: size = n_metabolites (for Q * coeffs and residual)
        coeffs_workspaces = [Vector{Float64}(undef, max(1, cache.rank)) for _ in 1:max_tid]
        proj_workspaces = [Vector{Float64}(undef, n_metabolites) for _ in 1:max_tid]

        # Thread-local result buffers (no synchronization during parallel section)
        thread_results = [Vector{Tuple{Int,Int}}() for _ in 1:max_tid]
        for buf in thread_results
            sizehint!(buf, max(1, length(pairs_to_check) ÷ max_tid + 10))
        end

        Threads.@threads for pair_idx in 1:length(pairs_to_check)
            i, j, c_alpha, c_beta = pairs_to_check[pair_idx]

            # Use thread-local workspaces
            tid = Threads.threadid()
            y_diff = workspaces[tid]
            coeffs_ws = coeffs_workspaces[tid]
            proj_ws = proj_workspaces[tid]

            # Compute difference in-place: y_alpha - y_beta
            y_alpha = complex_vectors[c_alpha]
            y_beta = complex_vectors[c_beta]
            @inbounds for k in 1:n_metabolites
                y_diff[k] = y_alpha[k] - y_beta[k]
            end

            # Check if y_diff ∈ im(Y𝚫) using in-place BLAS operations
            if is_in_span!(y_diff, cache, coeffs_ws, proj_ws)
                push!(thread_results[tid], (i, j))
            end
        end

        # Sequential merge of thread-local results (union-find not thread-safe)
        for thread_result in thread_results
            for (i, j) in thread_result
                if union_indices!(i, j)
                    merge_count += 1
                    @debug "  Merging modules $i and $j via Proposition S3-4"
                end
            end
        end
    end
    @debug "  Proposition S3-4 merged $merge_count pairs"



    # Collect merged sets
    groups = Dict{Int,Set{Symbol}}()
    for i in 1:n
        root = find_root(i)
        if !haskey(groups, root)
            groups[root] = Set{Symbol}()
        end
        Base.union!(groups[root], upstream_sets[i])
    end

    return collect(values(groups))
end

"""
    build_acr_augmentation(known_acr, metabolite_ids, n_metabolites)

Build augmentation matrix for known ACR metabolites (Remark S3-6).

For each known ACR metabolite S_a, build unit vector e_{S_a}.
Returns matrix with columns [e_{S_a1}, e_{S_a2}, ...].
"""
function build_acr_augmentation(
    known_acr::Vector{Symbol},
    metabolite_ids::Vector{Symbol},
    n_metabolites::Int
)
    if isempty(known_acr)
        return zeros(Float64, n_metabolites, 0)
    end

    metabolite_to_idx = Dict(id => i for (i, id) in enumerate(metabolite_ids))
    columns = Vector{Float64}[]

    for met_id in known_acr
        if !haskey(metabolite_to_idx, met_id)
            @warn "Known ACR metabolite not found in model" metabolite = met_id
            continue
        end

        met_idx = metabolite_to_idx[met_id]
        e_S = zeros(Float64, n_metabolites)
        e_S[met_idx] = 1.0
        push!(columns, e_S)
    end

    if isempty(columns)
        return zeros(Float64, n_metabolites, 0)
    end

    return hcat(columns...)
end

"""
Build the coupling companion map 𝚫 from coupling sets (Eq. S3-5).

For each coupling set 𝒞i = {C_i1, ..., C_ip}, build companion map:
𝚫i = [e_i1 - e_i2, e_i1 - e_i3, ..., e_i1 - e_ip]

Returns Y𝚫, the "species information matrix".
"""
function build_coupling_companion_matrix(
    coupling_sets::Vector{Set{Symbol}},
    Y_matrix::AbstractMatrix,
    complex_to_idx::Dict{Symbol,Int}
)
    n_metabolites = size(Y_matrix, 1)

    # Pre-count columns needed to avoid reallocation
    n_columns = 0
    for coupling_set in coupling_sets
        if length(coupling_set) >= 2
            # Count valid complexes for this set
            valid_count = count(c -> haskey(complex_to_idx, c), coupling_set)
            if valid_count >= 2
                n_columns += valid_count - 1  # (p-1) columns per set
            end
        end
    end

    if n_columns == 0
        return zeros(Float64, n_metabolites, 0)
    end

    # Pre-allocate result matrix
    result = zeros(Float64, n_metabolites, n_columns)
    col_idx = 1

    # OPTIMIZATION: Use sparse-aware difference computation
    # Avoids intermediate sparse allocations from Y[:,ref] - Y[:,other]
    Y_sparse = Y_matrix isa SparseArrays.SparseMatrixCSC ? Y_matrix : nothing

    for coupling_set in coupling_sets
        if length(coupling_set) < 2
            continue  # Need at least 2 complexes to form coupling relations
        end

        complexes_list = collect(coupling_set)
        reference_complex = complexes_list[1]

        if !haskey(complex_to_idx, reference_complex)
            continue
        end

        ref_idx = complex_to_idx[reference_complex]

        # Add columns: Y(e_i1 - e_ij) for j = 2, ..., p
        for j in 2:length(complexes_list)
            if !haskey(complex_to_idx, complexes_list[j])
                continue
            end

            other_idx = complex_to_idx[complexes_list[j]]

            # OPTIMIZED: Direct sparse-aware column difference
            # Write directly to result matrix column, no intermediate allocation
            if Y_sparse !== nothing
                # Zero the column (result is already zeroed, but be explicit for safety)
                # Actually skip this since we know result starts as zeros

                # Add ref_idx column values
                rows = SparseArrays.rowvals(Y_sparse)
                vals = SparseArrays.nonzeros(Y_sparse)
                for k in SparseArrays.nzrange(Y_sparse, ref_idx)
                    result[rows[k], col_idx] += vals[k]
                end

                # Subtract other_idx column values
                for k in SparseArrays.nzrange(Y_sparse, other_idx)
                    result[rows[k], col_idx] -= vals[k]
                end
            else
                # Fallback for dense matrices
                @inbounds for row in 1:n_metabolites
                    result[row, col_idx] = Y_matrix[row, ref_idx] - Y_matrix[row, other_idx]
                end
            end

            col_idx += 1
        end
    end

    return result
end

"""
Cached QR decomposition for efficient column span checks.

Stores the orthonormal basis Q_reduced for the column span of Y𝚫,
enabling O(m·k) projection checks instead of O(m³) QR per check.
"""
struct CachedColumnSpan
    Q_reduced::Matrix{Float64}  # Orthonormal basis for im(Y𝚫)
    rank::Int                   # Effective rank of Y𝚫
    tolerance::Float64          # Numerical tolerance
end

"""
    build_cached_column_span(Y_Delta; tolerance=1e-8)

Build a cached QR decomposition for efficient column span membership checks.
"""
function build_cached_column_span(Y_Delta::Matrix{Float64}; tolerance::Float64=1e-8)
    n_rows = size(Y_Delta, 1)

    if size(Y_Delta, 2) == 0
        return CachedColumnSpan(zeros(Float64, n_rows, 0), 0, tolerance)
    end

    try
        Q, R = LinearAlgebra.qr(Y_Delta)
        r_diag = abs.(LinearAlgebra.diag(R))
        rank_ydelta = sum(r_diag .> tolerance)

        if rank_ydelta == 0
            return CachedColumnSpan(zeros(Float64, n_rows, 0), 0, tolerance)
        end

        Q_reduced = Matrix(Q[:, 1:rank_ydelta])
        return CachedColumnSpan(Q_reduced, rank_ydelta, tolerance)
    catch e
        @debug "Error building cached column span" exception = e
        return CachedColumnSpan(zeros(Float64, n_rows, 0), 0, tolerance)
    end
end

"""
    is_in_span(v, cache::CachedColumnSpan)

Check if vector v is in the column span using cached QR decomposition.
O(m·k) instead of O(m³) per check.
"""
function is_in_span(v::AbstractVector{Float64}, cache::CachedColumnSpan)
    if cache.rank == 0
        return norm(v) < cache.tolerance
    end

    # Project v onto column space: P v = Q Q^T v
    proj = cache.Q_reduced * (cache.Q_reduced' * v)
    return norm(proj - v) < cache.tolerance
end

"""
    is_in_span!(v, cache, coeffs_workspace, proj_workspace)

In-place version of is_in_span using pre-allocated workspaces.
Avoids allocations in hot loops. Uses BLAS operations for efficiency.

Arguments:
- v: Vector to check (not modified)
- cache: CachedColumnSpan with orthonormal basis Q
- coeffs_workspace: Pre-allocated vector of length cache.rank for Q'*v
- proj_workspace: Pre-allocated vector of length(v) for Q*coeffs and residual
"""
function is_in_span!(
    v::AbstractVector{Float64},
    cache::CachedColumnSpan,
    coeffs_workspace::Vector{Float64},
    proj_workspace::Vector{Float64}
)
    if cache.rank == 0
        return norm(v) < cache.tolerance
    end

    Q = cache.Q_reduced

    # coeffs = Q' * v  (BLAS gemv: y = α*A'*x + β*y)
    LinearAlgebra.BLAS.gemv!('T', 1.0, Q, v, 0.0, coeffs_workspace)

    # proj = Q * coeffs  (BLAS gemv)
    LinearAlgebra.BLAS.gemv!('N', 1.0, Q, coeffs_workspace, 0.0, proj_workspace)

    # Compute ||proj - v|| without allocating (proj_workspace -= v, then norm)
    # Use axpy!: y = α*x + y, so proj_workspace = -1.0*v + proj_workspace
    LinearAlgebra.BLAS.axpy!(-1.0, v, proj_workspace)

    return norm(proj_workspace) < cache.tolerance
end

"""
Check if y_diff ∈ im(Y𝚫) using QR-based projection.

More robust than direct least squares solve - handles rank-deficient matrices gracefully.
Uses orthogonal projection: y_diff ∈ im(Y𝚫) ⟺ ||P y_diff - y_diff|| < tol
where P = Q Q^T is the projection onto im(Y𝚫).
"""
function can_merge_via_proposition_s34(y_diff::Vector{Float64}, Y_Delta::Matrix{Float64})
    if size(Y_Delta, 2) == 0
        # No coupling information available yet
        return false
    end

    tolerance = 1e-8

    try
        # Use QR decomposition for robust column span check
        Q, R = LinearAlgebra.qr(Y_Delta)

        # Determine effective rank
        r_diag = abs.(LinearAlgebra.diag(R))
        rank_ydelta = sum(r_diag .> tolerance)

        if rank_ydelta == 0
            # Y_Delta is effectively zero - only zero vector is in span
            return norm(y_diff) < tolerance
        end

        # Project onto column space using reduced Q
        Q_reduced = Matrix(Q[:, 1:rank_ydelta])
        proj = Q_reduced * (Q_reduced' * y_diff)

        # Check if projection recovers original vector
        residual_norm = norm(proj - y_diff)
        return residual_norm < tolerance
    catch e
        # Should rarely happen with QR, but catch just in case
        @debug "Error in QR-based column span check" exception = e
        return false
    end
end

"""
Check if y_diff ∈ im(Y𝚫) using pre-cached QR decomposition.
"""
function can_merge_via_proposition_s34(y_diff::Vector{Float64}, cache::CachedColumnSpan)
    return is_in_span(y_diff, cache)
end

"""
    is_weakly_reversible(network)

Check if the network is weakly reversible.

A network is weakly reversible if every linkage class is strongly connected.
Used in Lemma S4-7: If not weakly reversible → δₖ ≥ 1.
"""
function is_weakly_reversible(network::NamedTuple)
    A_matrix = network.A
    complex_ids = network.complex_ids
    n_complexes = length(complex_ids)

    # OPTIMIZATION: Use helper function for graph building
    g = build_reaction_graph(A_matrix, n_complexes)

    # Find weakly connected components (linkage classes)
    weak_components = Graphs.weakly_connected_components(g)

    # Check if each linkage class is strongly connected
    for component in weak_components
        subgraph_vertices = Set(component)

        # Check if this component is strongly connected
        # by verifying all vertices can reach each other
        sccs_in_component = tarjan_scc(subgraph_vertices, A_matrix)

        # If linkage class has multiple SCCs, it's not strongly connected
        if length(sccs_in_component) > 1
            return false
        end
    end

    return true
end

"""
    build_reaction_graph(A_matrix, n_complexes)

Build a directed graph from the incidence matrix.
Returns the graph and weakly connected components (linkage classes).
"""
function build_reaction_graph(A_matrix::SparseArrays.SparseMatrixCSC, n_complexes::Int)
    g = Graphs.SimpleDiGraph(n_complexes)
    n_reactions = size(A_matrix, 2)

    # OPTIMIZATION: Use sparse matrix structure directly
    rows = SparseArrays.rowvals(A_matrix)
    vals = SparseArrays.nonzeros(A_matrix)

    for rxn_idx in 1:n_reactions
        substrates = Int[]
        products = Int[]

        for idx in SparseArrays.nzrange(A_matrix, rxn_idx)
            row = rows[idx]
            val = vals[idx]
            if val < 0
                push!(substrates, row)
            elseif val > 0
                push!(products, row)
            end
        end

        for sub in substrates, prod in products
            Graphs.add_edge!(g, sub, prod)
        end
    end

    return g
end

"""
    robust_rank(M; tolerance=1e-10)

Compute numerical rank using SVD with configurable tolerance.
More robust than LinearAlgebra.rank for ill-conditioned matrices.
"""
function robust_rank(M::AbstractMatrix; tolerance::Float64=1e-10)
    if isempty(M) || all(iszero, M)
        return 0
    end

    # Use SVD for robust rank computation
    svd_result = LinearAlgebra.svd(M)
    max_sv = maximum(svd_result.S)

    if max_sv < tolerance
        return 0
    end

    # Count singular values above relative tolerance
    threshold = tolerance * max_sv
    return count(s -> s > threshold, svd_result.S)
end

"""
    compute_structural_deficiency(concordance_modules, network)

Compute structural deficiency δ using Equation S4-6:
    δ = ℯ − rank([Y^T; U^T] M)

Where:
- ℯ = number of unbalanced concordance modules
- Y = stoichiometric matrix
- U = complex-linkage class incidence matrix
- M = concordance ratio matrix

Uses sparse matrices for efficiency on large-scale models.
"""
function compute_structural_deficiency(
    concordance_modules::Vector{Set{Symbol}},
    network::NamedTuple
)
    Y_matrix = network.Y  # Already sparse
    A_matrix = network.A
    complex_ids = network.complex_ids
    complex_to_idx = network.complex_to_idx

    balanced = concordance_modules[1]
    unbalanced_modules = concordance_modules[2:end]
    e = length(unbalanced_modules)

    if e == 0
        return 0
    end

    n_complexes = length(complex_ids)

    # OPTIMIZATION: Use helper function for graph building
    g = build_reaction_graph(A_matrix, n_complexes)
    linkage_classes = Graphs.weakly_connected_components(g)
    n_linkages = length(linkage_classes)

    # Build U matrix as sparse (complex-linkage class incidence)
    # OPTIMIZATION: Preallocate with known size
    total_entries = sum(length, linkage_classes)
    I_u = Vector{Int}(undef, total_entries)
    J_u = Vector{Int}(undef, total_entries)

    idx = 1
    for (j, lc) in enumerate(linkage_classes)
        for complex_idx in lc
            I_u[idx] = complex_idx
            J_u[idx] = j
            idx += 1
        end
    end
    U_matrix = SparseArrays.sparse(I_u, J_u, ones(Int, total_entries), n_complexes, n_linkages)

    # Build concordance ratio matrix M (sparse)
    # OPTIMIZATION: Estimate size for preallocation
    estimated_entries = sum(length, unbalanced_modules)
    I_m = Vector{Int}()
    J_m = Vector{Int}()
    sizehint!(I_m, estimated_entries)
    sizehint!(J_m, estimated_entries)

    for (module_idx, conc_module) in enumerate(unbalanced_modules)
        for complex_symbol in conc_module
            if haskey(complex_to_idx, complex_symbol)
                complex_idx = complex_to_idx[complex_symbol]
                push!(I_m, complex_idx)
                push!(J_m, module_idx)
            end
        end
    end
    M_matrix = SparseArrays.sparse(I_m, J_m, ones(Float64, length(I_m)), n_complexes, e)

    # Compute [Y; U^T] M using sparse operations (Equation S4-6)
    YM = Y_matrix * M_matrix
    UM = U_matrix' * M_matrix

    # Stack and convert to dense for rank computation (small matrix: rows = m + ℓ, cols = ℯ)
    stacked = Matrix(vcat(YM, UM))

    # OPTIMIZATION: Use robust SVD-based rank computation
    matrix_rank = robust_rank(stacked)

    delta = e - matrix_rank

    # Compute classical deficiency for comparison: δ = n − ℓ − s
    # where s = dim(im(S)) and S = Y * A (stoichiometry matrix)
    S_matrix = Y_matrix * A_matrix
    s_dim = robust_rank(Matrix(S_matrix))
    classical_delta = n_complexes - n_linkages - s_dim

    @debug "Structural deficiency computation" ℯ = e n_linkages n_complexes rank_YUM = matrix_rank s_stoich = s_dim δ_equation_S46 = delta δ_classical = classical_delta

    # Return the classical deficiency (should match Equation S4-6, but let's use classical for now)
    return classical_delta
end

"""
    check_mass_action_deficiency(
        concordance_modules::Vector{Set{Symbol}},
        n_concordance_merges::Int,
        initial_delta::Int,
        network::NamedTuple
    )

Determine if mass action deficiency δₖ = 1 using Proposition S4-8 and Lemma S4-7.

# Algorithm
1. Apply Lemma S4-7: If not weakly reversible → δₖ ≥ 1
2. Apply Proposition S4-8: Each concordance merge reduces δₖ by at least 1
   - δₖ ≤ δ₀ - n_concordance_merges (where δ₀ is initial structural deficiency)
3. If δₖ ≥ 1 AND δₖ ≤ 1 → δₖ = 1

Returns (is_delta_k_one, should_merge_concordance)
"""
function check_mass_action_deficiency(
    concordance_modules::Vector{Set{Symbol}},
    n_concordance_merges::Int,
    initial_delta::Int,
    network::NamedTuple
)
    # Current number of unbalanced modules
    current_n_unbalanced = length(concordance_modules) - 1

    # Lower bound from Lemma S4-7
    weakly_rev = is_weakly_reversible(network)
    delta_k_lower_bound = weakly_rev ? 0 : 1

    # Upper bound from Proposition S4-8
    # Each concordance merge (via Proposition S4-1 from coupling merges) reduces δₖ by at least 1
    # δₖ ≤ δ₀ - n_concordance_merges
    delta_k_upper_bound = initial_delta - n_concordance_merges

    @debug "Mass action deficiency bounds" δ₀ = initial_delta weakly_reversible = weakly_rev δₖ_lower = delta_k_lower_bound δₖ_upper = delta_k_upper_bound n_merges = n_concordance_merges

    # Check if δₖ = 1
    is_delta_k_one = (delta_k_lower_bound == 1 && delta_k_upper_bound == 1)

    # If δₖ = 1, we should merge all unbalanced concordance modules (Lemma S4-5)
    should_merge_concordance = is_delta_k_one && current_n_unbalanced > 1

    return (is_delta_k_one, should_merge_concordance)
end

"""
    apply_theorem_s4_6(kinetic_modules, concordance_modules, network)

Apply Theorem S4-6: If mass action deficiency δₖ = 1, then all non-terminal
complexes are mutually coupled and should be merged into a single kinetic module.

Theorem S4-6 states: "Let G be a network of mass action deficiency one. Then all
nonterminal complexes in G are mutually coupled."

This is checked by:
1. Verifying that all unbalanced complexes are mutually concordant (Lemma S4-5)
   - Indicated by having exactly one unbalanced concordance module
2. If δₖ = 1, then all non-terminal complexes in the ENTIRE network are coupled

Note: concordance_modules = [balanced, unbalanced_1, unbalanced_2, ...]
"""
function apply_theorem_s4_6(
    kinetic_modules::Vector{Set{Symbol}},
    concordance_modules::Vector{Set{Symbol}},
    network::NamedTuple
)
    # Extract balanced and unbalanced modules
    balanced = concordance_modules[1]
    unbalanced_modules = concordance_modules[2:end]
    n_unbalanced_modules = length(unbalanced_modules)

    # Lemma S4-5: δₖ = 1 ⟺ exactly one unbalanced concordance module
    if n_unbalanced_modules == 1
        @debug "Applying Theorem S4-6: δₖ = 1 (one unbalanced concordance module)"

        # ALL complexes in the network (balanced + unbalanced)
        all_complexes = reduce(∪, [balanced; unbalanced_modules]; init=Set{Symbol}())
        @debug "  Total complexes in network" n_total = length(all_complexes)

        # Find terminal complexes
        terminal_complexes = find_terminal_complexes(all_complexes, network)
        @debug "  Terminal complexes" n_terminal = length(terminal_complexes) complexes = terminal_complexes

        # Theorem S4-6: All non-terminal complexes are mutually coupled
        non_terminal = setdiff(all_complexes, terminal_complexes)

        if !isempty(non_terminal)
            @debug "  Merging all non-terminal complexes into single kinetic module" n_non_terminal = length(non_terminal)
            # Return single module with all non-terminal complexes, plus terminal singletons
            result = [non_terminal]
            for terminal_complex in terminal_complexes
                push!(result, Set([terminal_complex]))
            end
            return result
        end
    end

    return kinetic_modules
end

"""
Find terminal complexes using strong linkage class (SCC) analysis.

A complex is terminal if it belongs to a terminal SCC. A terminal SCC is one
that has no outgoing edges to other complexes within the current set.

This matches the paper's definition: "A complex C ∈ 𝒞 is called terminal, if it is
a member of some terminal strong linkage class Λ_l."
"""
function find_terminal_complexes(
    complexes::Set{Symbol},
    network::NamedTuple
)
    A_matrix = network.A
    complex_ids = network.complex_ids
    complex_to_idx = network.complex_to_idx

    # Convert to indices
    complex_indices = Set(complex_to_idx[c] for c in complexes if haskey(complex_to_idx, c))

    # Find all SCCs
    sccs = tarjan_scc(complex_indices, A_matrix)

    # Identify terminal SCCs
    terminal_complex_set = Set{Symbol}()
    for scc in sccs
        if is_terminal_scc_idx(scc, complex_indices, A_matrix)
            # All complexes in this SCC are terminal
            for idx in scc
                if idx <= length(complex_ids)
                    push!(terminal_complex_set, complex_ids[idx])
                end
            end
        end
    end

    return terminal_complex_set
end

"""
Add singleton complexes and weak linkage classes as kinetic modules.

Following the reference R implementation (code_kineticModule_analysis.R):
1. Weak linkage classes composed entirely of balanced complexes (lines 141-155)
2. All unassigned complexes as singletons (lines 185-189)
"""
function add_singleton_balanced(
    kinetic_modules::Vector{Set{Symbol}},
    balanced::Set{Symbol},
    concordance_modules::Vector{Set{Symbol}},
    network::NamedTuple
)
    result = copy(kinetic_modules)

    # All complexes that participated (balanced + concordance)
    all_complexes = reduce(∪, [balanced; concordance_modules]; init=Set{Symbol}())

    # Find complexes already assigned to kinetic modules
    assigned = reduce(∪, kinetic_modules; init=Set{Symbol}())

    # Step 1: Add weak linkage classes composed entirely of balanced complexes
    # (R implementation lines 141-155)
    weak_modules = find_balanced_weak_linkage_classes(balanced, all_complexes, network)
    for wm in weak_modules
        if isdisjoint(wm, assigned)
            push!(result, wm)
            union!(assigned, wm)
            @debug "Added weak linkage class as kinetic module" size = length(wm)
        end
    end

    # Step 2: Add all remaining unassigned complexes as singletons
    # (R implementation lines 185-189)
    singletons = setdiff(all_complexes, assigned)
    @debug "Adding singleton complexes" n_singletons = length(singletons)

    for singleton in singletons
        push!(result, Set([singleton]))
    end

    return result
end

"""
Find weak linkage classes that are composed entirely of balanced complexes.

A weak linkage class is a weakly connected component in the reaction graph.
Following R implementation at lines 141-155.
"""
function find_balanced_weak_linkage_classes(
    balanced::Set{Symbol},
    all_complexes::Set{Symbol},
    network::NamedTuple
)
    A_matrix = network.A
    complex_ids = network.complex_ids
    complex_to_idx = network.complex_to_idx

    # Build undirected graph from incidence matrix for weak connectivity
    n_complexes = length(complex_ids)

    # Use Graphs.jl to find weakly connected components
    g = Graphs.SimpleDiGraph(n_complexes)

    # Add edges from incidence matrix
    n_reactions = size(A_matrix, 2)
    for rxn_idx in 1:n_reactions
        # Find substrates and products for this reaction
        substrates = findall(A_matrix[:, rxn_idx] .< 0)
        products = findall(A_matrix[:, rxn_idx] .> 0)

        # Add directed edges from each substrate to each product
        for sub in substrates
            for prod in products
                Graphs.add_edge!(g, sub, prod)
            end
        end
    end

    # Find weakly connected components
    weak_components = Graphs.weakly_connected_components(g)

    # Filter to keep only components composed entirely of balanced complexes
    balanced_weak_modules = Set{Symbol}[]

    for component in weak_components
        # Convert indices to symbols
        component_symbols = Set(complex_ids[idx] for idx in component if idx <= length(complex_ids))

        # Check if all complexes in this component are balanced
        if !isempty(component_symbols) && issubset(component_symbols, balanced)
            push!(balanced_weak_modules, component_symbols)
            @debug "Found weak linkage class of balanced complexes" size = length(component_symbols) complexes = component_symbols
        end
    end

    return balanced_weak_modules
end

"""
    _detect_acr_acrr(kinetic_modules, Y_matrix, metabolite_ids, complex_to_idx; tolerance=1e-8, efficient=true)

Internal helper for ACR/ACRR detection using pre-computed network data.
Called by both `kinetic_analysis` (integrated) and `identify_acr_acrr` (standalone).
"""
function _detect_acr_acrr(
    kinetic_modules::Vector{Set{Symbol}},
    Y_matrix::AbstractMatrix,
    metabolite_ids::Vector{Symbol},
    complex_to_idx::Dict{Symbol,Int};
    tolerance::Float64=1e-8,
    efficient::Bool=true
)
    n_metabolites = length(metabolite_ids)

    if efficient
        # Fast efficient path: Direct pairwise comparison
        max_tid = Threads.maxthreadid()
        thread_acr = [Set{Symbol}() for _ in 1:max_tid]
        thread_acrr = [Set{Tuple{Symbol,Symbol}}() for _ in 1:max_tid]
        thread_nz_indices = [Vector{Int}(undef, 2) for _ in 1:max_tid]
        thread_nz_vals = [Vector{Float64}(undef, 2) for _ in 1:max_tid]

        Threads.@threads for module_set in kinetic_modules
            tid = Threads.threadid()
            local_acr = thread_acr[tid]
            local_acrr = thread_acrr[tid]
            nz_indices_buf = thread_nz_indices[tid]
            nz_vals_buf = thread_nz_vals[tid]

            complexes = collect(module_set)
            k = length(complexes)
            k < 2 && continue

            @inbounds for i in 1:k
                idx_a = get(complex_to_idx, complexes[i], 0)
                idx_a == 0 && continue

                for j in (i+1):k
                    idx_b = get(complex_to_idx, complexes[j], 0)
                    idx_b == 0 && continue

                    nnz_count = 0
                    for met_idx in 1:n_metabolites
                        val_diff = Y_matrix[met_idx, idx_a] - Y_matrix[met_idx, idx_b]
                        if abs(val_diff) > tolerance
                            nnz_count += 1
                            if nnz_count <= 2
                                nz_indices_buf[nnz_count] = met_idx
                                nz_vals_buf[nnz_count] = val_diff
                            else
                                break
                            end
                        end
                    end

                    if nnz_count == 1
                        push!(local_acr, metabolite_ids[nz_indices_buf[1]])
                    elseif nnz_count == 2
                        if abs(nz_vals_buf[1] + nz_vals_buf[2]) < tolerance
                            m1, m2 = metabolite_ids[nz_indices_buf[1]], metabolite_ids[nz_indices_buf[2]]
                            push!(local_acrr, m1 < m2 ? (m1, m2) : (m2, m1))
                        end
                    end
                end
            end
        end

        acr_set = Set{Symbol}()
        acrr_set = Set{Tuple{Symbol,Symbol}}()
        for tid in 1:max_tid
            union!(acr_set, thread_acr[tid])
            union!(acrr_set, thread_acrr[tid])
        end

        return (acr_metabolites=collect(acr_set), acrr_pairs=collect(acrr_set))
    else
        # Full/thorough path: Do BOTH pairwise comparison AND column span check
        # This finds all ACR/ACRR relationships

        acr_set = Set{Symbol}()
        acrr_set = Set{Tuple{Symbol,Symbol}}()

        # --- Part 1: Pairwise comparison (same as efficient=true) ---
        # This finds direct ACR/ACRR from stoichiometric differences
        for module_set in kinetic_modules
            complexes = collect(module_set)
            k = length(complexes)
            k < 2 && continue

            for i in 1:k
                idx_a = get(complex_to_idx, complexes[i], 0)
                idx_a == 0 && continue

                for j in (i+1):k
                    idx_b = get(complex_to_idx, complexes[j], 0)
                    idx_b == 0 && continue

                    # Count non-zero differences
                    nz_indices = Int[]
                    nz_vals = Float64[]
                    for met_idx in 1:n_metabolites
                        val_diff = Y_matrix[met_idx, idx_a] - Y_matrix[met_idx, idx_b]
                        if abs(val_diff) > tolerance
                            push!(nz_indices, met_idx)
                            push!(nz_vals, val_diff)
                            length(nz_indices) > 2 && break
                        end
                    end

                    if length(nz_indices) == 1
                        push!(acr_set, metabolite_ids[nz_indices[1]])
                    elseif length(nz_indices) == 2
                        if abs(nz_vals[1] + nz_vals[2]) < tolerance
                            m1, m2 = metabolite_ids[nz_indices[1]], metabolite_ids[nz_indices[2]]
                            push!(acrr_set, m1 < m2 ? (m1, m2) : (m2, m1))
                        end
                    end
                end
            end
        end

        # --- Part 2: Column span check (Propositions S3-5, S3-6) ---
        # This finds ACR/ACRR from linear combinations of coupling relations
        Y_Delta = build_coupling_companion_matrix(kinetic_modules, Y_matrix, complex_to_idx)

        if size(Y_Delta, 2) > 0
            cache = build_cached_column_span(Y_Delta; tolerance=tolerance)

            if cache.rank > 0
                Q_reduced = cache.Q_reduced
                coeffs = Q_reduced'

                # ACR: Check if e_S ∈ im(Y∆) for each metabolite
                for i in 1:n_metabolites
                    coeff_col = @view coeffs[:, i]
                    coeff_norm_sq = sum(abs2, coeff_col)
                    proj_diag = sum(Q_reduced[i, k] * coeff_col[k] for k in 1:cache.rank)
                    residual_norm_sq = coeff_norm_sq - 2 * proj_diag + 1.0

                    if residual_norm_sq < tolerance^2
                        push!(acr_set, metabolite_ids[i])
                    end
                end

                # ACRR: Check if e_i - e_j ∈ im(Y∆) for each metabolite pair
                diff_vec = zeros(Float64, n_metabolites)
                for i in 1:n_metabolites
                    for j in (i+1):n_metabolites
                        fill!(diff_vec, 0.0)
                        diff_vec[i] = 1.0
                        diff_vec[j] = -1.0

                        if is_in_span(diff_vec, cache)
                            m1, m2 = metabolite_ids[i], metabolite_ids[j]
                            push!(acrr_set, m1 < m2 ? (m1, m2) : (m2, m1))
                        end
                    end
                end
            end
        end

        return (acr_metabolites=collect(acr_set), acrr_pairs=collect(acrr_set))
    end
end

"""
    identify_acr_acrr(kinetic_modules, model; tolerance=1e-8, efficient=true)

Identify metabolites with Absolute Concentration Robustness (ACR) and
Absolute Concentration Ratio Robustness (ACRR) from kinetic modules.

Uses Propositions S3-5 and S3-6 from the paper:
- ACR: Metabolite S has ACR if e_S ∈ im(YΔ)
- ACRR: Metabolites S1, S2 have ACRR if e_{S1} - e_{S2} ∈ im(YΔ)

# Arguments
- `kinetic_modules`: Vector of kinetic modules (from kinetic_analysis)
- `model`: AbstractFBCModel for network topology
- `tolerance`: Tolerance for linear system solving (default: 1e-8)
- `efficient`: Boolean flag for performance optimization (default: `true`)
    - `true`: Use fast pairwise comparison of coupled complexes. Identifies ACR/ACRR
      only from direct stoichiometric differences within modules.
    - `false`: Use full matrix analysis (Proposition S3-5/S3-6) checking column span of Y𝚫.
      Can identify implicit relationships from linear combinations.

# Returns
Named tuple with:
- `acr_metabolites`: Vector of metabolite IDs with ACR
- `acrr_pairs`: Vector of tuples (S1, S2) with ACRR

# Examples
```julia
kinetic_modules = kinetic_analysis(concordance_modules, model)
acr_results = identify_acr_acrr(kinetic_modules, model)
println("ACR metabolites: ", acr_results.acr_metabolites)
println("ACRR pairs: ", acr_results.acrr_pairs)
```
"""
function identify_acr_acrr(
    kinetic_modules::Vector{Set{Symbol}},
    model::A.AbstractFBCModel;
    tolerance::Float64=1e-8,
    efficient::Bool=true
)
    # Extract network topology and delegate to helper
    Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)
    complex_to_idx = Dict{Symbol,Int}(id => i for (i, id) in enumerate(complex_ids))

    return _detect_acr_acrr(kinetic_modules, Y_matrix, metabolite_ids, complex_to_idx;
        tolerance=tolerance, efficient=efficient)
end

"""
    is_in_column_span(v, M, tolerance)

Check if vector v is in the column span of matrix M using least squares.

Returns true if ||M ξ - v|| < tolerance for some ξ.
"""
function is_in_column_span(v::Vector{Float64}, M::Matrix{Float64}, tolerance::Float64)
    if size(M, 2) == 0
        # Empty matrix - only zero vector is in span
        return norm(v) < tolerance
    end

    try
        # Solve least squares: minimize ||M ξ - v||
        xi = M \ v
        residual = M * xi - v
        residual_norm = norm(residual)

        return residual_norm < tolerance
    catch e
        @warn "Error solving linear system in column span check" exception = e
        return false
    end
end

# ================================================================================================
# Public API: Deficiency Calculation Functions
# ================================================================================================

"""
    structural_deficiency(concordance_modules, model::AbstractFBCModel)

Compute the structural deficiency δ using the classical formula:
    δ = n - ℓ - s

Where:
- n = number of complexes
- ℓ = number of linkage classes (weakly connected components)
- s = dimension of stoichiometric subspace (rank of stoichiometry matrix S = Y·A)

This is a wrapper for `compute_structural_deficiency` with cleaner naming.

# Example
```julia
model = create_envz_ompr_model()
concordance_modules = extract_concordance_modules(results)
δ = structural_deficiency(concordance_modules, model)  # Returns 2 for EnvZ-OmpR
```
"""
function structural_deficiency(
    concordance_modules::Vector{Set{Symbol}},
    model::A.AbstractFBCModel
)
    Y_matrix, _, complex_ids = complex_stoichiometry(model; return_ids=true)
    A_matrix, _, _ = incidence(model; return_ids=true)
    complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

    network = (
        Y=Y_matrix,
        A=A_matrix,
        complex_ids=complex_ids,
        complex_to_idx=complex_to_idx
    )

    return compute_structural_deficiency(concordance_modules, network)
end

"""
    mass_action_deficiency_bounds(
        concordance_modules,
        model::AbstractFBCModel;
        n_concordance_merges::Int=0
    )

Compute bounds on mass action deficiency δₖ using:
- **Lower bound** (Lemma S4-7): δₖ ≥ 1 if not weakly reversible, else δₖ ≥ 0
- **Upper bound** (Proposition S4-8): δₖ ≤ δ₀ - n_concordance_merges

Returns a named tuple `(lower=..., upper=..., is_exact=...)` where:
- `lower`: Lower bound on δₖ
- `upper`: Upper bound on δₖ
- `is_exact`: true if lower == upper (δₖ is uniquely determined)

# Arguments
- `concordance_modules`: Vector of concordance modules
- `model`: AbstractFBCModel
- `n_concordance_merges`: Number of concordance merges applied (default: 0)

# Example
```julia
model = create_envz_ompr_model()
bounds = mass_action_deficiency_bounds(concordance_modules, model)
# For EnvZ-OmpR after proper merging: (lower=1, upper=1, is_exact=true)
```
"""
function mass_action_deficiency_bounds(
    concordance_modules::Vector{Set{Symbol}},
    model::A.AbstractFBCModel;
    n_concordance_merges::Int=0
)
    Y_matrix, _, complex_ids = complex_stoichiometry(model; return_ids=true)
    A_matrix, _, _ = incidence(model; return_ids=true)
    complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

    network = (
        Y=Y_matrix,
        A=A_matrix,
        complex_ids=complex_ids,
        complex_to_idx=complex_to_idx
    )

    # Compute initial structural deficiency
    initial_delta = compute_structural_deficiency(concordance_modules, network)

    # Get bounds
    weakly_rev = is_weakly_reversible(network)
    lower_bound = weakly_rev ? 0 : 1
    upper_bound = initial_delta - n_concordance_merges

    return (
        lower=lower_bound,
        upper=upper_bound,
        is_exact=(lower_bound == upper_bound),
        weakly_reversible=weakly_rev
    )
end
