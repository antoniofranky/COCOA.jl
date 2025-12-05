"""
Kinetic Module Analysis with Set-based Interface

Clean implementation that works natively with Vector{Set{Symbol}} structure
for concordance modules, avoiding index mapping issues.

Uses the simplified 2-phase upstream algorithm from the paper (Remark S2-1).
"""

import Graphs
import LinearAlgebra: norm

"""
    kinetic_analysis(concordance_modules, model; min_module_size=1, known_acr=Symbol[])

Apply kinetic module analysis using set-based concordance module structure.

Implements the iterative refinement algorithm from Section S.4.1 with three feedback loops:
1. Proposition S4-1: Coupling merges → Concordance merges
2. Remark S3-6: ACR identification → Enhanced coupling via augmented Y𝚫
3. Theorem S4-6: If δₖ = 1, all non-terminal complexes are coupled

# Arguments
- `concordance_modules`: Vector{Set{Symbol}} where:
  - `concordance_modules[1]` = balanced complexes (module 0)
  - `concordance_modules[2+]` = concordance modules 1, 2, ...
- `model`: AbstractFBCModel for network topology
- `min_module_size`: Minimum size for reported modules (default: 1)
  - Set to 1 to include singleton modules (terminal and unassigned complexes)
  - Set to 2 to exclude singletons and focus on coupled modules
- `known_acr`: Vector of metabolite IDs with known ACR (from external sources)
  Used to enhance merging via Remark S3-6 (default: Symbol[])

# Returns
- `Vector{Set{Symbol}}`: Kinetic modules (sets of complex IDs), sorted by size (largest first)

# Notes
- Singleton modules include: terminal complexes (dead ends) and balanced/unbalanced
  complexes not assigned to larger modules
- Terminal complexes don't participate in steady-state dynamics (their monomials
  don't appear in ODEs), but are included for completeness
- Set `min_module_size=2` to exclude singletons and focus on "interesting" kinetic modules

# Example
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
    known_acr::Vector{Symbol}=Symbol[]
)
    @info "Starting kinetic module analysis" n_concordance_modules = length(concordance_modules) - 1 n_balanced = length(concordance_modules[1])

    # Extract network topology and build ID mappings ONCE
    A_matrix, complex_ids = incidence(model; return_ids=true)
    Y_matrix, metabolite_ids, _ = complex_stoichiometry(model; return_ids=true)

    # Build mappings once and reuse throughout
    complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))

    # Build augmentation for known ACR metabolites (Remark S3-6)
    # For each known ACR metabolite, add its unit vector e_S as an augmentation column
    acr_augmentation = build_acr_augmentation(known_acr, metabolite_ids, size(Y_matrix, 1))

    if !isempty(known_acr)
        @info "Using known ACR metabolites for enhanced merging" n_known_acr = length(known_acr) metabolites = known_acr
    end

    # Store in a named tuple for easy passing
    network = (
        A=A_matrix,
        Y=Y_matrix,
        complex_ids=complex_ids,
        metabolite_ids=metabolite_ids,
        complex_to_idx=complex_to_idx,
        acr_augmentation=acr_augmentation
    )

    # Compute initial structural deficiency δ₀ before any concordance merging (for Proposition S4-8)
    initial_delta = compute_structural_deficiency(concordance_modules, network)

    # Check for mass action deficiency δₖ = 1 before entering the loop (Lemma S4-5)
    # If δₖ = 1 (one unbalanced concordance module), we can directly apply Theorem S4-6
    current_concordance = copy(concordance_modules)

    if length(current_concordance) == 2  # balanced + 1 unbalanced
        @info "Initial check: δₖ = 1 detected (one unbalanced concordance module)"
        @info "Directly applying Theorem S4-6 without iterative refinement"
        kinetic_modules = apply_theorem_s4_6(Set{Symbol}[], current_concordance, network)
        valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)
        sort!(valid_modules, by=length, rev=true)
        @info "Kinetic analysis complete via Theorem S4-6" n_modules = length(valid_modules)
        return valid_modules
    end

    # Iterative refinement loop (Section S.4.1 + ACR feedback)
    # When coupling sets merge, concordance modules also merge (S4-1)
    # When ACR is identified, it can enable more merging (Remark S3-6)
    # This may lead to larger upstream sets and more merging opportunities
    max_outer_iterations = 10
    outer_iteration = 0
    concordance_changed = true
    acr_changed = false
    current_known_acr = Set{Symbol}(known_acr)  # Track discovered ACR metabolites

    # Track number of concordance merges for Proposition S4-8
    n_concordance_merges = 0

    while (concordance_changed || acr_changed) && outer_iteration < max_outer_iterations
        outer_iteration += 1
        @info "=== Iteration $outer_iteration: Concordance-Coupling-ACR Refinement ===" n_modules = length(current_concordance) - 1 n_known_acr = length(current_known_acr)

        balanced = current_concordance[1]

        # Update network with current known ACR (Remark S3-6)
        if !isempty(current_known_acr) && acr_changed
            @info "Updating Y𝚫 augmentation with newly identified ACR metabolites" new_acr = setdiff(current_known_acr, Set(known_acr))
            acr_augmentation = build_acr_augmentation(collect(current_known_acr), metabolite_ids, size(Y_matrix, 1))
            network = (
                A=network.A,
                Y=network.Y,
                complex_ids=network.complex_ids,
                metabolite_ids=network.metabolite_ids,
                complex_to_idx=network.complex_to_idx,
                acr_augmentation=acr_augmentation
            )
        end

        # Step 1: Compute upstream sets for each concordance module
        @info "Computing upstream sets"
        upstream_sets = Set{Symbol}[]

        for (i, conc_module) in enumerate(current_concordance[2:end])
            @debug "Processing concordance module $i" size = length(conc_module) complexes = conc_module

            # Form extended module: balanced ∪ concordance
            extended_module = balanced ∪ conc_module

            # Apply upstream algorithm
            upstream = upstream_algorithm(extended_module, network)

            if !isempty(upstream)
                push!(upstream_sets, upstream)
                @debug "Upstream set computed" size = length(upstream) complexes = upstream
            end
        end

        @info "Computed upstream sets" n_sets = length(upstream_sets)

        if isempty(upstream_sets)
            @warn "No upstream sets found"
            return Set{Symbol}[]
        end

        # Step 2: Merge coupled modules and track which concordance modules should merge
        @info "Merging coupled modules"
        kinetic_modules, merge_map = merge_coupled_sets_tracked(upstream_sets, network)

        # Step 3: Check if any merges occurred (Proposition S4-1)
        # If coupling sets i and j merged, their concordance modules should also merge
        previous_n_unbalanced = length(current_concordance) - 1
        concordance_changed = apply_concordance_merging!(current_concordance, merge_map)
        current_n_unbalanced = length(current_concordance) - 1

        # Track number of concordance merges for Proposition S4-8
        if concordance_changed
            n_concordance_merges += (previous_n_unbalanced - current_n_unbalanced)
        end

        # Step 4: Identify ACR metabolites from current kinetic modules (Proposition S3-5)
        # This enables further merging via Remark S3-6
        @info "Identifying ACR metabolites from current kinetic modules"
        acr_results = identify_acr_acrr(kinetic_modules, model)
        newly_identified_acr = setdiff(Set(acr_results.acr_metabolites), current_known_acr)

        if !isempty(newly_identified_acr)
            @info "Identified new ACR metabolites" metabolites = newly_identified_acr
            union!(current_known_acr, newly_identified_acr)
            acr_changed = true
        else
            acr_changed = false
        end

        if concordance_changed
            @info "Concordance modules merged via Proposition S4-1, recomputing..."

            # Check if concordance merging resulted in δₖ = 1
            if length(current_concordance) == 2  # balanced + 1 unbalanced
                @info "After merging: δₖ = 1 detected (one unbalanced concordance module)"
                @info "Directly applying Theorem S4-6"
                kinetic_modules = apply_theorem_s4_6(Set{Symbol}[], current_concordance, network)
                valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)
                sort!(valid_modules, by=length, rev=true)
                @info "Kinetic analysis complete via Theorem S4-6" n_modules = length(valid_modules)
                return valid_modules
            end
        elseif acr_changed
            @info "New ACR metabolites identified, will continue iteration with augmented Y𝚫"
        else
            @info "No concordance changes and no new ACR, checking for δₖ = 1..."

            # Check if δₖ = 1 using Proposition S4-8 and Lemma S4-7
            (is_delta_k_one, should_merge_concordance) = check_mass_action_deficiency(
                current_concordance,
                n_concordance_merges,
                initial_delta,
                network
            )

            if is_delta_k_one && should_merge_concordance
                @info "δₖ = 1 detected! Forcing merge of all unbalanced concordance modules (Lemma S4-5)"

                # Merge all unbalanced concordance modules into one
                all_unbalanced = reduce(∪, current_concordance[2:end]; init=Set{Symbol}())
                current_concordance = [current_concordance[1], all_unbalanced]

                @info "Recomputing with merged concordance modules..."
                # Continue loop to recompute upstream sets
                concordance_changed = true
                continue
            end

            @info "Refinement converged"

            # Add singleton balanced complexes from extended modules
            kinetic_modules = add_singleton_balanced(kinetic_modules, balanced, current_concordance[2:end], network)

            # Apply Theorem S4-6: if mass action deficiency δₖ = 1, merge all non-terminal modules
            kinetic_modules = apply_theorem_s4_6(kinetic_modules, current_concordance, network)

            # Filter by size
            valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)

            # Sort by size (largest first) so giant module is at index 1
            sort!(valid_modules, by=length, rev=true)

            @info "Kinetic analysis complete" n_modules = length(valid_modules) giant_size = (isempty(valid_modules) ? 0 : maximum(length.(valid_modules)))

            return valid_modules
        end
    end

    if outer_iteration >= max_outer_iterations
        @warn "Reached maximum iterations in concordance-coupling refinement loop"
    end

    # Fallback: return current kinetic modules
    balanced = current_concordance[1]
    upstream_sets = Set{Symbol}[]
    for conc_module in current_concordance[2:end]
        extended_module = balanced ∪ conc_module
        upstream = upstream_algorithm(extended_module, network)
        !isempty(upstream) && push!(upstream_sets, upstream)
    end

    kinetic_modules = merge_coupled_sets(upstream_sets, network)
    kinetic_modules = add_singleton_balanced(kinetic_modules, balanced, current_concordance[2:end], network)
    kinetic_modules = apply_theorem_s4_6(kinetic_modules, current_concordance, network)
    valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)

    # Sort by size (largest first) so giant module is at index 1
    sort!(valid_modules, by=length, rev=true)

    return valid_modules
end

"""
    upstream_algorithm(extended_module, network)

Apply simplified 2-phase upstream algorithm to identify kinetic module (Remark S2-1 from paper).

# Phases
1. Phase I: Remove entry complexes (autonomous property)
2. Phase II: Remove terminal strong linkage classes (feeding property)

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

    @info "Starting upstream algorithm (simplified 2-phase)" extended_size = length(extended_module) indices_found = length(current_indices)
    remaining = [complex_ids[i] for i in current_indices]
    @info "  Extended module complexes: $remaining"

    # Phase I: Remove entry complexes (autonomous property)
    @info "Phase I: Removing entry complexes" initial_size = length(current_indices)
    iteration = 0
    while true
        iteration += 1
        entries = find_entry_complexes_idx(current_indices, A_matrix)
        if !isempty(entries)
            entry_symbols = [complex_ids[i] for i in entries]
            @info "  Phase I iteration $iteration: removing $(length(entries)) entries: $entry_symbols"
        end

        isempty(entries) && break
        setdiff!(current_indices, entries)

        if isempty(current_indices)
            @info "  All complexes removed in Phase I"
            return Set{Symbol}()
        end
    end

    remaining = [complex_ids[i] for i in current_indices]
    @info "Phase II: Removing terminal strong linkage classes" remaining

    # Phase II: Find and remove all terminal strong linkage classes using Tarjan's algorithm
    sccs = tarjan_scc(current_indices, A_matrix)
    @info "  Found $(length(sccs)) strongly connected components"

    # Debug: Show all SCCs with their terminal status
    @debug "  SCC analysis:" begin
        for (i, scc) in enumerate(sccs)
            scc_symbols = [complex_ids[j] for j in scc]
            is_term = is_terminal_scc_idx(scc, current_indices, A_matrix)
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
        is_term = is_terminal_scc_idx(scc, original_indices, A_matrix)
        @info "  SCC (size $(length(scc))): $scc_symbols - Terminal: $is_term"

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

    @info "  Removed $(length(terminal_sccs)) terminal SCCs, kept $(length(non_terminal_sccs)) non-terminal SCCs"

    # Convert back to symbols using complex_ids vector
    upstream_set = Set(complex_ids[i] for i in current_indices if i <= length(complex_ids))

    @info "Upstream algorithm complete" upstream_size = length(upstream_set) complexes = sort(collect(upstream_set), by=string)

    return upstream_set
end

"""
Find entry complexes using index-based operations.

An entry complex has an incoming reaction from a complex outside the current set.
"""
function find_entry_complexes_idx(complexes::Set{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    entries = Set{Int}()

    for cidx in complexes
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
    tarjan_scc(complexes, A_matrix)

Find all strongly connected components using Tarjan's algorithm.

Returns a vector of SCCs, where each SCC is a Set{Int} of complex indices.
"""
function tarjan_scc(complexes::Set{Int}, A_matrix::SparseArrays.SparseMatrixCSC{Int,Int})
    # Convert set to vector for indexing
    nodes = collect(complexes)
    n = length(nodes)

    # Initialize Tarjan's algorithm state
    index = 0
    stack = Int[]
    indices = Dict{Int,Int}()
    lowlinks = Dict{Int,Int}()
    on_stack = Set{Int}()
    sccs = Vector{Set{Int}}()

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
                    if A_matrix[pidx, rxn_idx] > 0 && pidx ∈ all_complexes && pidx ∉ scc
                        @debug "SCC contains complex $cidx which converts (via rxn $rxn_idx) to $pidx outside SCC → NON-TERMINAL"
                        return false  # Has edge outside SCC but within all_complexes
                    end
                end
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
    @info "Applying Proposition S4-1: Merging concordance modules"
    balanced = concordance_modules[1]
    new_concordance = [balanced]

    for (group_id, module_indices) in sort(collect(groups), by=first)
        merged_module = Set{Symbol}()
        for idx in module_indices
            Base.union!(merged_module, concordance_modules[idx+1])  # +1 because concordance_modules[1] is balanced
        end

        if length(module_indices) > 1
            @info "  Merging concordance modules $(collect(module_indices)) (coupling sets merged)"
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
    @info "Step 1: Trivial merging (Lemma S3-1)"
    for i in 1:n
        for j in (i+1):n
            if !isdisjoint(upstream_sets[i], upstream_sets[j])
                union_indices!(i, j)
                @info "  Merging modules $i and $j (shared complexes)" shared = upstream_sets[i] ∩ upstream_sets[j]
            end
        end
    end

    # Step 2: Advanced merging via Proposition S3-4
    # Check if Y[:,α] - Y[:,β] ∈ im(Y𝚫) for complexes from different coupling sets
    @info "Step 2: Advanced merging via Proposition S3-4"

    # Build coupling companion map 𝚫 from current upstream sets
    # 𝚫 = [𝚫1 ... 𝚫q] where each 𝚫i encodes coupling relations in upstream_sets[i]
    Y_Delta = build_coupling_companion_matrix(upstream_sets, Y_matrix, complex_to_idx)

    # Augment with known ACR columns (Remark S3-6)
    if haskey(network, :acr_augmentation) && size(network.acr_augmentation, 2) > 0
        Y_Delta = hcat(Y_Delta, network.acr_augmentation)
        @info "  Augmented Y𝚫 with known ACR metabolites" n_augmentation_cols = size(network.acr_augmentation, 2)
    end

    # Check all pairs of distinct coupling sets
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    merged_any = true

    while merged_any && iteration < max_iterations
        iteration += 1
        merged_any = false

        for i in 1:n
            for j in (i+1):n
                # Skip if already merged
                if find_root(i) == find_root(j)
                    continue
                end

                # Check Proposition S3-4: can we merge sets i and j?
                # Pick arbitrary complexes from each set
                c_alpha = first(upstream_sets[i])
                c_beta = first(upstream_sets[j])

                # Get their stoichiometric vectors
                if !haskey(complex_to_idx, c_alpha) || !haskey(complex_to_idx, c_beta)
                    continue
                end

                idx_alpha = complex_to_idx[c_alpha]
                idx_beta = complex_to_idx[c_beta]

                y_diff = collect(Y_matrix[:, idx_alpha] - Y_matrix[:, idx_beta])

                # Check if y_diff ∈ im(Y𝚫) by solving: Y𝚫 ξ = y_diff
                if can_merge_via_proposition_s34(y_diff, Y_Delta)
                    if union_indices!(i, j)
                        @info "  Merging modules $i and $j via Proposition S3-4" complexes = (c_alpha, c_beta)
                        merged_any = true

                        # Rebuild Y_Delta with updated coupling information
                        groups_temp = Dict{Int,Set{Symbol}}()
                        for k in 1:n
                            root = find_root(k)
                            if !haskey(groups_temp, root)
                                groups_temp[root] = Set{Symbol}()
                            end
                            Base.union!(groups_temp[root], upstream_sets[k])
                        end
                        current_sets = collect(values(groups_temp))
                        Y_Delta = build_coupling_companion_matrix(current_sets, Y_matrix, complex_to_idx)

                        # Re-augment with known ACR columns after rebuilding
                        if haskey(network, :acr_augmentation) && size(network.acr_augmentation, 2) > 0
                            Y_Delta = hcat(Y_Delta, network.acr_augmentation)
                        end
                    end
                end
            end
        end
    end

    if iteration >= max_iterations
        @warn "Reached maximum iterations in Proposition S3-4 merging"
    end

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
    columns = Vector{Float64}[]

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
            # Column: Y[:,ref] - Y[:,other]
            # collect() converts sparse to dense vector
            col = collect(Y_matrix[:, ref_idx] - Y_matrix[:, other_idx])
            push!(columns, col)
        end
    end

    if isempty(columns)
        return zeros(Float64, n_metabolites, 0)
    end

    return hcat(columns...)
end

"""
Check if y_diff ∈ im(Y𝚫) by solving the linear system: Y𝚫 ξ = y_diff

Uses least squares with tolerance to handle numerical errors.
"""
function can_merge_via_proposition_s34(y_diff::Vector{Float64}, Y_Delta::Matrix{Float64})
    if size(Y_Delta, 2) == 0
        # No coupling information available yet
        return false
    end

    # Solve least squares: minimize ||Y𝚫 ξ - y_diff||
    # If residual is small, then y_diff ∈ im(Y𝚫)
    try
        xi = Y_Delta \ y_diff
        residual = Y_Delta * xi - y_diff
        residual_norm = norm(residual)

        tolerance = 1e-8
        return residual_norm < tolerance
    catch e
        @warn "Error solving linear system in Proposition S3-4" exception = e
        return false
    end
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

    # Build directed graph from incidence matrix
    g = Graphs.SimpleDiGraph(n_complexes)
    n_reactions = size(A_matrix, 2)

    for rxn_idx in 1:n_reactions
        substrates = findall(A_matrix[:, rxn_idx] .< 0)
        products = findall(A_matrix[:, rxn_idx] .> 0)

        for sub in substrates
            for prod in products
                Graphs.add_edge!(g, sub, prod)
            end
        end
    end

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

    # Build complex-linkage class incidence matrix U (sparse)
    n_complexes = length(complex_ids)
    g = Graphs.SimpleDiGraph(n_complexes)
    n_reactions = size(A_matrix, 2)

    for rxn_idx in 1:n_reactions
        substrates = findall(A_matrix[:, rxn_idx] .< 0)
        products = findall(A_matrix[:, rxn_idx] .> 0)
        for sub in substrates, prod in products
            Graphs.add_edge!(g, sub, prod)
        end
    end

    linkage_classes = Graphs.weakly_connected_components(g)
    n_linkages = length(linkage_classes)

    # Build U matrix as sparse
    I_u = Int[]
    J_u = Int[]
    for (j, lc) in enumerate(linkage_classes)
        for complex_idx in lc
            push!(I_u, complex_idx)
            push!(J_u, j)
        end
    end
    U_matrix = SparseArrays.sparse(I_u, J_u, ones(Int, length(I_u)), n_complexes, n_linkages)

    # Build concordance ratio matrix M (sparse)
    I_m = Int[]
    J_m = Int[]
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

    # Stack and convert to dense for rank computation (small matrix)
    stacked = Matrix(vcat(YM, UM))

    # Compute rank
    matrix_rank = LinearAlgebra.rank(stacked)

    delta = e - matrix_rank

    # Compute classical deficiency for comparison: δ = n − ℓ − s
    # where s = dim(im(S)) and S = Y * A (stoichiometry matrix)
    S_matrix = Y_matrix * A_matrix
    s_dim = LinearAlgebra.rank(Matrix(S_matrix))
    classical_delta = n_complexes - n_linkages - s_dim

    @info "Structural deficiency computation" ℯ = e n_linkages n_complexes rank_YUM = matrix_rank s_stoich = s_dim δ_equation_S46 = delta δ_classical = classical_delta

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

    @info "Mass action deficiency bounds" δ₀ = initial_delta weakly_reversible = weakly_rev δₖ_lower = delta_k_lower_bound δₖ_upper = delta_k_upper_bound n_merges = n_concordance_merges

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
        @info "Applying Theorem S4-6: δₖ = 1 (one unbalanced concordance module)"

        # ALL complexes in the network (balanced + unbalanced)
        all_complexes = reduce(∪, [balanced; unbalanced_modules]; init=Set{Symbol}())
        @info "  Total complexes in network" n_total = length(all_complexes)

        # Find terminal complexes
        terminal_complexes = find_terminal_complexes(all_complexes, network)
        @info "  Terminal complexes" n_terminal = length(terminal_complexes) complexes = terminal_complexes

        # Theorem S4-6: All non-terminal complexes are mutually coupled
        non_terminal = setdiff(all_complexes, terminal_complexes)

        if !isempty(non_terminal)
            @info "  Merging all non-terminal complexes into single kinetic module" n_non_terminal = length(non_terminal)
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
    identify_acr_acrr(kinetic_modules, model; tolerance=1e-8)

Identify metabolites with Absolute Concentration Robustness (ACR) and
Absolute Concentration Ratio Robustness (ACRR) from kinetic modules.

Uses Propositions S3-5 and S3-6 from the paper:
- ACR: Metabolite S has ACR if e_S ∈ im(YΔ)
- ACRR: Metabolites S1, S2 have ACRR if e_{S1} - e_{S2} ∈ im(YΔ)

# Arguments
- `kinetic_modules`: Vector of kinetic modules (from kinetic_analysis)
- `model`: AbstractFBCModel for network topology
- `tolerance`: Tolerance for linear system solving (default: 1e-8)

# Returns
Named tuple with:
- `acr_metabolites`: Vector of metabolite IDs with ACR
- `acrr_pairs`: Vector of tuples (S1, S2) with ACRR

# Example
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
    tolerance::Float64=1e-8
)
    # Extract network topology
    Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)
    complex_to_idx = Dict(id => i for (i, id) in enumerate(complex_ids))
    metabolite_to_idx = Dict(id => i for (i, id) in enumerate(metabolite_ids))

    # Build Y𝚫 from final kinetic modules
    Y_Delta = build_coupling_companion_matrix(kinetic_modules, Y_matrix, complex_to_idx)

    n_metabolites = length(metabolite_ids)

    # Check ACR for each metabolite (Proposition S3-5)
    acr_metabolites = Symbol[]
    for (met_idx, met_id) in enumerate(metabolite_ids)
        # Build unit vector e_S
        e_S = zeros(Float64, n_metabolites)
        e_S[met_idx] = 1.0

        # Check if e_S ∈ im(YΔ)
        if is_in_column_span(e_S, Y_Delta, tolerance)
            push!(acr_metabolites, met_id)
        end
    end

    # Check ACRR for all pairs of metabolites (Proposition S3-6)
    acrr_pairs = Tuple{Symbol,Symbol}[]
    for i in 1:n_metabolites
        for j in (i+1):n_metabolites
            # Build difference e_{S1} - e_{S2}
            e_diff = zeros(Float64, n_metabolites)
            e_diff[i] = 1.0
            e_diff[j] = -1.0

            # Check if e_diff ∈ im(YΔ)
            if is_in_column_span(e_diff, Y_Delta, tolerance)
                push!(acrr_pairs, (metabolite_ids[i], metabolite_ids[j]))
            end
        end
    end

    @info "ACR/ACRR identification complete" n_acr = length(acr_metabolites) n_acrr = length(acrr_pairs)

    return (acr_metabolites=acr_metabolites, acrr_pairs=acrr_pairs)
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
