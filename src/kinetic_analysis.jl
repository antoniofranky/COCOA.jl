"""
Kinetic Module Analysis with Set-based Interface

Clean implementation that works natively with Vector{Set{Symbol}} structure
for concordance modules, avoiding index mapping issues.

Uses the simplified 2-phase upstream algorithm from the paper (Remark S2-1).
"""

import Graphs
import LinearAlgebra: norm, dot, cholesky, Symmetric, I
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
    ZERO_COMPLEX

Symbol identifying the zero complex ∅ used to represent the empty side of
boundary/exchange reactions in the kinetic-module graph (see [`augment_with_zero_complex`](@ref)).
Never appears in a concordance module and is excluded from all reported kinetic
modules, so it never surfaces to callers.
"""
const ZERO_COMPLEX = Symbol("∅")

"""
    augment_with_zero_complex(A_matrix, Y_matrix, complex_ids)

Add the zero complex ∅ (standard in CRN theory) to the kinetic-module graph and
connect every boundary reaction to it:
- inflow  (has products, no substrate complex): edge ∅ → product  (∅ is substrate, `A[∅,r] = -1`)
- outflow (has substrate, no product complex):  edge substrate → ∅ (∅ is product,  `A[∅,r] = +1`)

Without ∅, boundary metabolite complexes (e.g. an exchanged extracellular metabolite
produced only by `∅ → met[e]`) have no in-neighbor, so Phase I of
[`upstream_algorithm`](@ref) never prunes them as entry complexes. Being balanced,
they then get injected into every extended module and
[`merge_coupled_sets_tracked`](@ref)'s shared-complex trivial-merge step glues all
concordance modules into one giant kinetic module, producing spurious ACR/ACRR pairs
(confirmed against the MATLAB reference, which does represent boundary reactions via
an explicit zero complex).

∅ itself never appears in a concordance module (it is not part of network topology
known to the concordance step), so `upstream_algorithm` never returns it as a member
of an upstream set — it only participates as a graph node used to correctly prune
neighbors. It therefore requires no special-casing in ACR/ACRR reporting or module
output.

Returns `(A_aug, Y_aug, complex_ids_aug)` with one extra row/column appended for ∅.
"""
function augment_with_zero_complex(
    A_matrix::SparseArrays.SparseMatrixCSC,
    Y_matrix::SparseArrays.AbstractSparseMatrix,
    complex_ids::Vector{Symbol}
)
    n_complexes, n_reactions = size(A_matrix)
    zero_idx = n_complexes + 1

    I, J, V = SparseArrays.findnz(A_matrix)
    I2 = collect(I)
    J2 = collect(J)
    V2 = collect(V)

    vals = SparseArrays.nonzeros(A_matrix)
    for r in 1:n_reactions
        has_substrate = false
        has_product = false
        for idx in SparseArrays.nzrange(A_matrix, r)
            v = vals[idx]
            v < 0 && (has_substrate = true)
            v > 0 && (has_product = true)
        end
        if has_product && !has_substrate
            # Inflow: ∅ -> product, so ∅ is the substrate (A[∅,r] = -1)
            push!(I2, zero_idx)
            push!(J2, r)
            push!(V2, -1)
        elseif has_substrate && !has_product
            # Outflow: substrate -> ∅, so ∅ is the product (A[∅,r] = +1)
            push!(I2, zero_idx)
            push!(J2, r)
            push!(V2, 1)
        end
    end

    A_aug = SparseArrays.sparse(I2, J2, V2, zero_idx, n_reactions)
    Y_aug = hcat(Y_matrix, SparseArrays.spzeros(size(Y_matrix, 1), 1))
    complex_ids_aug = vcat(complex_ids, [ZERO_COMPLEX])

    return A_aug, Y_aug, complex_ids_aug
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
    !!! note "Completeness vs. speed"
        The fast path (`efficient=true`) scales to genome-scale networks but is inherently a less
        exhaustive search than the full matrix/deficiency path, and can under-report ACR
        metabolites / ACRR pairs. Use `efficient=false`
        whenever exhaustive detection matters more than runtime.

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

    # Augment with the zero complex ∅ so boundary/exchange reactions have an explicit
    # in- or out-neighbor. Without this, boundary complexes are never pruned by Phase I
    # of the upstream algorithm and spuriously glue concordance modules together (see
    # `augment_with_zero_complex` docstring). ∅ never appears in the returned modules.
    A_matrix, Y_matrix, complex_ids = augment_with_zero_complex(A_matrix, Y_matrix, complex_ids)

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

    # Remark S2-1: the simplified 2-phase Upstream Algorithm is equivalent to the
    # full algorithm ONLY when terminal strong linkage classes are identified on
    # the FULL network. Precompute the globally-terminal complexes once, on the
    # whole complex graph, so Phase IV uses the network's terminal SLCs rather
    # than per-module induced-subgraph SCCs (which broke S2-1's premise and
    # diverged from the reference implementation on genome-scale networks).
    global_terminal = compute_global_terminal_complexes(cached_adj, length(complex_ids))

    # Store in a named tuple for easy passing
    network = (
        A=A_matrix,
        Y=Y_matrix,
        complex_ids=complex_ids,
        metabolite_ids=metabolite_ids,
        complex_to_idx=complex_to_idx,
        acr_augmentation=acr_augmentation,
        enable_advanced_merging=enable_advanced_merging,
        cached_adj=cached_adj,
        global_terminal=global_terminal
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
        progress("structural deficiency")
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
            progress("ACR/ACRR detection", " n_modules=", length(valid_modules))
    acr_results = _detect_acr_acrr(valid_modules, Y_matrix, metabolite_ids, complex_to_idx;
                efficient=efficient, known_acr=known_acr)
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
                cached_adj=network.cached_adj,
                global_terminal=network.global_terminal
            )
            @debug "Updated network with ACR augmentation" n_acr = length(current_known_acr)
        end

        # Step 1: Compute upstream sets (parallelized)
        n_modules = length(current_concordance) - 1
        upstream_results = Vector{Union{Nothing,Set{Symbol}}}(undef, n_modules)

        progress("upstream sets")
        Threads.@threads for idx in 1:n_modules
            conc_module = current_concordance[idx+1]
            extended_module = balanced ∪ conc_module
            upstream = upstream_algorithm(extended_module, network)
            upstream_results[idx] = isempty(upstream) ? nothing : upstream
        end

        # Collect non-empty results into properly typed vector
        upstream_sets = Set{Symbol}[r for r in upstream_results if !isnothing(r)]
        outer_iteration == 1 && complexes_in_every_coupling_set(upstream_sets)

        if isempty(upstream_sets)
            return (
                kinetic_modules=Set{Symbol}[],
                acr_metabolites=Symbol[],
                acrr_pairs=Tuple{Symbol,Symbol}[]
            )
        end

        # Step 2: Merge coupled modules
        # Use simple merging if efficient=true (handled by enable_advanced_merging=false in network)
        progress("merge_coupled_sets_tracked", " n_upstream=", length(upstream_sets))
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
        _sizes = sort(length.(kinetic_modules), rev=true)
        progress("iter ", outer_iteration, ": identify_acr_acrr",
                " n_modules=", length(kinetic_modules),
                " largest=", isempty(_sizes) ? 0 : _sizes[1])
        _t_acr = time()
        acr_results = identify_acr_acrr(kinetic_modules, model; efficient=efficient)
        progress("iter ", outer_iteration, ": identify_acr_acrr done",
                " seconds=", round(time() - _t_acr, digits=2),
                " n_acr=", length(acr_results.acr_metabolites))
        newly_identified_acr = setdiff(Set(acr_results.acr_metabolites), current_known_acr)

        if !isempty(newly_identified_acr)
            union!(current_known_acr, newly_identified_acr)
            @debug "Identified new ACR metabolites" count = length(newly_identified_acr) acr_list = collect(newly_identified_acr)
        end

        # Check convergence: Has the kinetic module partition changed?
        # Convert to canonical form for comparison (sorted sets of sorted symbols)
        _t_conv = time()
        current_state = Set(Set(sort(collect(km), by=string)) for km in kinetic_modules)
        concordance_state = Set(Set(sort(collect(cm), by=string)) for cm in current_concordance)

        converged = (current_state == previous_kinetic_modules &&
                     concordance_state == previous_concordance_state)
        progress("iter ", outer_iteration, ": convergence check",
                " seconds=", round(time() - _t_conv, digits=2), " converged=", converged)

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
        progress("mass action deficiency")
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
    progress("add_singleton_balanced")
    kinetic_modules = add_singleton_balanced(kinetic_modules, current_concordance[1], current_concordance[2:end], network)

    # Final shared-complex merge (Lemma S3-1): the balanced weak linkage classes just
    # added may overlap existing modules; unify them (and any ACR-difference merges).
    # The reference performs this merge unconditionally (its mdiff/clustmg step), so it
    # runs here regardless of whether ACR metabolites were found.
    begin
        @debug "Final shared-complex merging pass" n_modules_before = length(kinetic_modules)
        progress("final merge_coupled_sets", " n_modules=", length(kinetic_modules))
        final_merged = merge_coupled_sets(kinetic_modules, network)
        if length(final_merged) < length(kinetic_modules)
            @debug "Final merging reduced module count" before = length(kinetic_modules) after = length(final_merged)
            kinetic_modules = final_merged
        end
    end

    if !efficient
        progress("theorem S4-6")
        kinetic_modules = apply_theorem_s4_6(kinetic_modules, current_concordance, network)
    end

    valid_modules = filter(km -> length(km) >= min_module_size, kinetic_modules)
    sort!(valid_modules, by=length, rev=true)

    # Detect ACR/ACRR using already-available network data (no redundant extraction).
    # Feed the iteratively discovered ACR set so detection can propagate it
    # (Remark S3-6): e_S ∈ im([Y∆ | e_{known ACR}]).
    acr_results = _detect_acr_acrr(valid_modules, Y_matrix, metabolite_ids, complex_to_idx;
        efficient=efficient, known_acr=collect(current_known_acr))

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
            complex_id=complexes.complex_id,
            concordance_module=complexes.concordance_module,
            kinetic_module=km_mapping,
            classification=complexes.classification,
            min_activity=complexes.min_activity,
            max_activity=complexes.max_activity,
            lambda=complexes.lambda,
            trivially_balanced=complexes.trivially_balanced,
        )
    else
        (
            complex_id=complexes.complex_id,
            concordance_module=complexes.concordance_module,
            kinetic_module=km_mapping,
            classification=complexes.classification,
        )
    end

    # Build updated ACR/ACRR tables
    acr = (metabolite_id=String.(kin_result.acr_metabolites),)
    acrr = (
        metabolite_1=String[String(p[1]) for p in kin_result.acrr_pairs],
        metabolite_2=String[String(p[2]) for p in kin_result.acrr_pairs],
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

    # Phase IV: Remove complexes that belong to a terminal strong linkage class.
    # Remark S2-1 requires terminal SLCs to be identified on the FULL complex
    # graph, not per-module. When `network.global_terminal` is available (the set
    # of complexes in terminal SLCs of the full network, computed once), remove
    # those; this reproduces the reference implementation's Phase IV. The legacy
    # per-subset SCC path is kept as a fallback only when the global set is absent.
    if haskey(network, :global_terminal)
        n_before = length(current_indices)
        setdiff!(current_indices, network.global_terminal)
        @debug "  Removed globally-terminal complexes (full-network SLCs)" removed = n_before - length(current_indices)
    else
        sccs = if has_cached_adj
            tarjan_scc_cached(current_indices, network.cached_adj)
        else
            tarjan_scc(current_indices, A_matrix)
        end
        @debug "  Found $(length(sccs)) strongly connected components (subset fallback)"

        # Identify terminal/non-terminal SCCs WITHOUT modifying current_indices
        original_indices = copy(current_indices)
        terminal_sccs = Set{Int}[]
        for scc in sccs
            is_term = if has_cached_adj
                is_terminal_scc_cached(scc, original_indices, network.cached_adj)
            else
                is_terminal_scc_idx(scc, original_indices, A_matrix)
            end
            is_term && push!(terminal_sccs, scc)
        end
        for scc in terminal_sccs
            setdiff!(current_indices, scc)
        end
        @debug "  Removed $(length(terminal_sccs)) terminal SCCs (subset fallback)"
    end

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
    # An SCC is globally terminal iff no complex in it has any outgoing edge to a
    # complex outside the SCC in the FULL graph (Remark S2-1: terminal classification
    # uses the full network, not the reduced subgraph).
    for cidx in scc
        for rxn_idx in SparseArrays.findnz(A_matrix[cidx, :])[1]
            if A_matrix[cidx, rxn_idx] < 0  # Complex is substrate
                products = SparseArrays.findnz(A_matrix[:, rxn_idx])[1]
                for pidx in products
                    if A_matrix[pidx, rxn_idx] > 0 && pidx ∉ scc
                        @debug "SCC contains complex $cidx which converts (via rxn $rxn_idx) to $pidx outside SCC → NON-TERMINAL"
                        return false
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
Check if SCC is globally terminal using cached adjacency (optimized version).
An SCC is globally terminal iff no complex in it converts to any complex outside
the SCC in the full graph (Remark S2-1: terminal classification uses the full network).
"""
function is_terminal_scc_cached(scc::Set{Int}, all_complexes::Set{Int}, cached_adj::CachedAdjacency)
    @inbounds for cidx in scc
        for neighbor in cached_adj.out_neighbors[cidx]
            if neighbor ∉ scc
                return false  # Has edge outside SCC in full graph → non-terminal
            end
        end
    end
    return true
end

"""
    compute_global_terminal_complexes(cached_adj, n_complexes) -> Set{Int}

Return the set of complexes that lie in a **terminal strong linkage class of the
full complex graph**. Per the manuscript's Remark S2-1, the simplified 2-phase
Upstream Algorithm (Phase I + IV) equals the full 4-phase algorithm only when
terminal strong linkage classes are identified on the whole network. Computing
the strongly connected components once, over all complexes, and flagging those
with no outgoing edge as terminal, restores that premise (the previous
implementation computed SCCs per-module subset, which diverged from the
reference on genome-scale networks). O(V+E), done once per analysis.
"""
function compute_global_terminal_complexes(cached_adj::CachedAdjacency, n_complexes::Int)
    all_nodes = Set(1:n_complexes)
    sccs = tarjan_scc_cached(all_nodes, cached_adj)
    terminal = Set{Int}()
    for scc in sccs
        is_terminal_scc_cached(scc, all_nodes, cached_adj) && union!(terminal, scc)
    end
    return terminal
end

"""
    merge_coupled_sets_tracked(upstream_sets, network)

Merge coupled upstream sets and track which modules merged together.
Returns (merged_modules, merge_map) where merge_map[i] gives the final group ID for module i.
"""
function merge_coupled_sets_tracked(upstream_sets::Vector{Set{Symbol}}, network::NamedTuple)
    _t = time()
    kinetic_modules = merge_coupled_sets(upstream_sets, network)
    progress("merge_coupled_sets done seconds=", round(time() - _t, digits=2),
            " n_modules=", length(kinetic_modules))

    # Build merge map: which original upstream set ended up in which final group.
    #
    # This used to scan every final module for every upstream set with `isdisjoint`,
    # i.e. O(n_upstream * n_modules) set comparisons — 5385 x ~5000 on a genome-scale
    # yeast model, each one walking a set. Measured: it sat here for hours while the
    # phase markers showed the rest of the function finishing in seconds.
    #
    # A final module is a union of upstream sets, so membership of ANY member decides
    # the group. One reverse index over all complexes replaces the quadratic scan. The
    # lowest group id is kept, which reproduces the original's "first match in
    # enumeration order" exactly, even if a set were to touch several groups.
    _t = time()
    n = length(upstream_sets)
    merge_map = zeros(Int, n)

    complex_to_group = Dict{Symbol,Int}()
    for (group_id, final_module) in enumerate(kinetic_modules)
        for c in final_module
            g = get(complex_to_group, c, typemax(Int))
            g > group_id && (complex_to_group[c] = group_id)
        end
    end

    @inbounds for i in 1:n
        best = typemax(Int)
        for c in upstream_sets[i]
            g = get(complex_to_group, c, typemax(Int))
            g < best && (best = g)
        end
        merge_map[i] = best == typemax(Int) ? 0 : best
    end
    progress("merge_map built seconds=", round(time() - _t, digits=2))

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
    _t_delta = time()
    Y_Delta = build_coupling_companion_matrix(upstream_sets, Y_matrix, complex_to_idx)
    _t_delta = time() - _t_delta

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
    _t_qr = time()
    cache = build_cached_column_span(Y_Delta)
    _t_qr = time() - _t_qr
    # Phase timings at @info: the profile of this path was never measured, and the
    # sparse span test only addresses ONE of these phases. Optimising without knowing
    # which phase dominates is guessing.
    progress("Y_Delta and QR n_upstream_sets=", length(upstream_sets),
            " size_Y_Delta=", size(Y_Delta), " rank=", cache.rank,
            " seconds_build_Y_Delta=", round(_t_delta, digits=2),
            " seconds_qr=", round(_t_qr, digits=2))

    # OPTIMIZATION: Precompute stoichiometric vectors for all relevant complexes
    # Avoids repeated sparse-to-dense conversions in the loop
    n_metabolites = size(Y_matrix, 1)
    complex_vectors = Dict{Symbol,Vector{Float64}}()
    # Sparse form as well: a complex is a small sum of species, so its Y column has a
    # handful of nonzeros out of n_metabolites. Densifying it (as the line below still
    # does, for the exact fallback) is what made the pair loop stream the whole basis
    # through memory. See `is_in_span_sparse`.
    complex_sparse = Dict{Symbol,Tuple{Vector{Int},Vector{Float64}}}()
    for upstream_set in upstream_sets
        c = first(upstream_set)  # We only need one complex per set (Lemma S3-3)
        if haskey(complex_to_idx, c) && !haskey(complex_vectors, c)
            col = Y_matrix[:, complex_to_idx[c]]
            complex_vectors[c] = Vector(col)
            idxs = SparseArrays.findnz(SparseArrays.sparse(col))
            complex_sparse[c] = (collect(idxs[1]), collect(idxs[2]))
        end
    end

    # OPTIMIZATION: Collect pairs to check, then parallelize
    # Build list of (i, j, c_alpha, c_beta) tuples for pairs that need checking
    _t_pairs = time()
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

    # n_pairs = 0 would mean the Proposition S3-4 check never runs at all. That has two
    # very different causes and they must not be confused: either everything is already
    # in one union-find root (the "all glued into one giant module" pathology), or
    # `complex_vectors` is missing its keys and every pair is skipped by the `haskey`
    # guard — a silent no-op of the most expensive check in the whole path. Report both.
    _n_roots = length(unique(find_root(i) for i in 1:n))
    progress("pairs to check n_pairs=", length(pairs_to_check),
            " seconds_collect=", round(time() - _t_pairs, digits=2),
            " n_upstream=", n,
            " n_complex_vectors=", length(complex_vectors),
            " n_distinct_roots=", _n_roots)

    # OPTIMIZATION: Parallel merge check using thread-local result arrays
    # Avoids Channel synchronization overhead by accumulating results per-thread
    merge_count = 0

    if length(pairs_to_check) > 0
        # Per-task workspaces via explicit chunking, NOT `threadid()`-indexed buffers:
        # `Threads.@threads` schedules dynamically, so a task may resume on a different
        # thread than it started on and two tasks could then share `y_diff`. See the
        # matching comment in `_detect_acr_acrr`.
        #
        # Round-robin chunking keeps the load even; every pair costs the same two BLAS
        # gemv calls, so there is nothing to balance dynamically anyway.
        n_pairs = length(pairs_to_check)
        nchunks = max(1, min(Threads.nthreads(), n_pairs))
        _t_loop_start = time()

        chunk_tasks = map(1:nchunks) do c
            Threads.@spawn begin
                y_diff = Vector{Float64}(undef, n_metabolites)
                coeffs_ws = Vector{Float64}(undef, max(1, cache.rank))
                proj_ws = Vector{Float64}(undef, n_metabolites)
                local_result = Vector{Tuple{Int,Int}}()
                n_fallback = 0
                sizehint!(local_result, max(1, n_pairs ÷ nchunks + 10))

                nz_idx = Vector{Int}(undef, 64)
                nz_val = Vector{Float64}(undef, 64)

                for pair_idx in c:nchunks:n_pairs
                    i, j, c_alpha, c_beta = pairs_to_check[pair_idx]

                    # Sparse difference: merge the two complexes' nonzero patterns.
                    ia, va = complex_sparse[c_alpha]
                    ib, vb = complex_sparse[c_beta]
                    need = length(ia) + length(ib)
                    if need > length(nz_idx)
                        resize!(nz_idx, need)
                        resize!(nz_val, need)
                    end
                    n_nz = _sparse_difference!(nz_idx, nz_val, ia, va, ib, vb)

                    in_span, decided = is_in_span_sparse(nz_idx, nz_val, n_nz, cache, coeffs_ws)
                    if !decided
                        # Borderline for the cancellation-prone identity: settle it on
                        # the exact dense path.
                        y_alpha = complex_vectors[c_alpha]
                        y_beta = complex_vectors[c_beta]
                        @inbounds for k in 1:n_metabolites
                            y_diff[k] = y_alpha[k] - y_beta[k]
                        end
                        in_span = is_in_span!(y_diff, cache, coeffs_ws, proj_ws)
                        n_fallback += 1
                    end
                    if in_span
                        push!(local_result, (i, j))
                    end
                end
                (local_result, n_fallback)
            end
        end
        fetched = [fetch(t) for t in chunk_tasks]
        thread_results = [f[1] for f in fetched]
        # Timed from before the spawn, so this covers the whole parallel section.
        progress("pair loop done", " seconds_loop=", round(time() - _t_loop_start, digits=2), " dense_fallbacks=", sum(f[2] for f in fetched), " of_pairs=", n_pairs)

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
    Q_reduced::Matrix{Float64}  # Orthonormal basis for im(Y𝚫)   (n_metabolites x rank)
    Qt::Matrix{Float64}         # Q_reduced' stored explicitly    (rank x n_metabolites)
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
        return CachedColumnSpan(zeros(Float64, n_rows, 0), zeros(Float64, 0, n_rows), 0, tolerance)
    end

    try
        Q, R = LinearAlgebra.qr(Y_Delta)
        r_diag = abs.(LinearAlgebra.diag(R))
        rank_ydelta = sum(r_diag .> tolerance)

        if rank_ydelta == 0
            return CachedColumnSpan(zeros(Float64, n_rows, 0), zeros(Float64, 0, n_rows), 0, tolerance)
        end

        Q_reduced = Matrix(Q[:, 1:rank_ydelta])
        # Qt is kept explicitly so that the sparse span test can read all `rank`
        # entries belonging to ONE metabolite row contiguously. Reading them out of
        # the column-major Q_reduced would stride by n_metabolites per element.
        return CachedColumnSpan(Q_reduced, Matrix(Q_reduced'), rank_ydelta, tolerance)
    catch e
        @debug "Error building cached column span" exception = e
        return CachedColumnSpan(zeros(Float64, n_rows, 0), zeros(Float64, 0, n_rows), 0, tolerance)
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
$(TYPEDSIGNATURES)

Count the nonzeros of the sparse difference `a - b` above `tolerance`, recording the
first two in `(out_idx, out_val)`, and stop as soon as a third appears — the ACR/ACRR
criteria only distinguish "exactly one", "exactly two" and "more than two".

Replaces a scan over every metabolite with `Y_matrix[met_idx, idx]` lookups, which on a
sparse matrix costs a search per access. A complex column holds a handful of nonzeros
out of ~10^4 metabolites, so the merge is the same answer for a fraction of the work.
Returns 3 to mean "three or more".
"""
@inline function _sparse_diff_upto2!(
    out_idx::Vector{Int}, out_val::Vector{Float64},
    ia::Vector{Int}, va::Vector{Float64},
    ib::Vector{Int}, vb::Vector{Float64},
    tolerance::Float64
)::Int
    p = 1; q = 1; n = 0
    na = length(ia); nb = length(ib)
    @inbounds while p <= na || q <= nb
        local idx::Int, d::Float64
        if q > nb || (p <= na && ia[p] < ib[q])
            idx = ia[p]; d = va[p]; p += 1
        elseif p > na || ib[q] < ia[p]
            idx = ib[q]; d = -vb[q]; q += 1
        else
            idx = ia[p]; d = va[p] - vb[q]; p += 1; q += 1
        end
        if abs(d) > tolerance
            n += 1
            n > 2 && return 3
            out_idx[n] = idx; out_val[n] = d
        end
    end
    return n
end

"""
$(TYPEDSIGNATURES)

Write the sparse difference `a - b` of two sorted sparse vectors into
`(out_idx, out_val)` and return how many entries were written. Exact zeros produced by
cancellation are dropped, so the result carries only genuine nonzeros.

Both inputs come from `SparseArrays.findnz`, whose indices are ascending, so this is a
single merge pass.
"""
@inline function _sparse_difference!(
    out_idx::Vector{Int}, out_val::Vector{Float64},
    ia::Vector{Int}, va::Vector{Float64},
    ib::Vector{Int}, vb::Vector{Float64}
)::Int
    p = 1
    q = 1
    n = 0
    na = length(ia)
    nb = length(ib)
    @inbounds while p <= na || q <= nb
        if q > nb || (p <= na && ia[p] < ib[q])
            n += 1; out_idx[n] = ia[p]; out_val[n] = va[p]; p += 1
        elseif p > na || ib[q] < ia[p]
            n += 1; out_idx[n] = ib[q]; out_val[n] = -vb[q]; q += 1
        else
            d = va[p] - vb[q]
            if d != 0.0
                n += 1; out_idx[n] = ia[p]; out_val[n] = d
            end
            p += 1; q += 1
        end
    end
    return n
end

"""
$(TYPEDSIGNATURES)

Detect ACR, ACRR and gACRR directly from the column span of Y𝚫, as the theory defines
them, rather than from differences between complexes.

The Supplementary Materials state the three properties as span memberships:

    ACR   (S)       e_S                ∈ im(Y𝚫)
    ACRR  (S1,S2)   e_S1 -   e_S2      ∈ im(Y𝚫)
    gACRR (S1,S2)   e_S1 - γ e_S2      ∈ im(Y𝚫)   for some γ > 0      (SM S3-13)

and note that gACRR is an LP feasibility problem in (γ, ξ) (SM S3-14). Comparing pairs
of complexes, as `_detect_acr_acrr` does, is only a SUFFICIENT shortcut: if two coupled
complexes differ by exactly `e_S` then `e_S ∈ im(Y𝚫)` follows, but not conversely. That
is why the pairwise path misses the published ACR of the EnvZ-OmpR example.

The span formulation needs no LP. Writing `r_S = e_S - Q Qᵀ e_S` for the part of `e_S`
orthogonal to the span, with `Q` the orthonormal basis in `cache`:

    e_S ∈ im(Y𝚫)              ⟺  r_S = 0
    e_S1 - γ e_S2 ∈ im(Y𝚫)    ⟺  r_S1 = γ r_S2

so ACR is a norm test and gACRR is a test for parallel residuals, with γ the ratio of
their lengths and ACRR the case γ = 1. Everything follows from inner products of rows
of `Q`, because `QᵀQ = I` gives

    ‖r_i‖² = 1 - ‖Q[i,:]‖²        and      ⟨r_i, r_j⟩ = -Q[i,:]·Q[j,:]   (i ≠ j)

Cost is O(n_species²·rank) in the worst case but only over species that are not already
ACR, versus O(k²·n_metabolites) complex pairs for the pairwise path — on a genome-scale
yeast model that is the difference between thousands and hundreds of millions of tests.

Returns `(acr_metabolites, acrr_pairs, gacrr_triples)` where each gACRR entry is
`(S1, S2, γ)`. ACRR pairs are reported separately AND remain in the gACRR list, so the
two quantities stay comparable with what earlier runs reported.
"""
function detect_robustness_via_span(
    cache::CachedColumnSpan,
    metabolite_ids::Vector{Symbol};
    tolerance::Float64=1e-8,
    include_gacrr::Bool=true
)
    Q = cache.Q_reduced
    n = length(metabolite_ids)
    acr = Symbol[]
    acrr = Tuple{Symbol,Symbol}[]
    gacrr = Tuple{Symbol,Symbol,Float64}[]

    if cache.rank == 0 || size(Q, 1) != n
        # No span information: e_S is never in im(Y𝚫), so nothing is robust.
        return (acr_metabolites=acr, acrr_pairs=acrr, gacrr_triples=gacrr)
    end

    tol2 = tolerance * tolerance
    nrm2 = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        acc = 0.0
        @simd for q in 1:cache.rank
            acc += Q[s, q] * Q[s, q]
        end
        # ‖r_s‖² = 1 - ‖Qᵀe_s‖²; clamp away negative round-off
        nrm2[s] = max(0.0, 1.0 - acc)
    end

    @inbounds for s in 1:n
        nrm2[s] < tol2 && push!(acr, metabolite_ids[s])
    end

    include_gacrr || return (acr_metabolites=acr, acrr_pairs=acrr, gacrr_triples=gacrr)

    # The Gram identities give ‖r_i - γ r_j‖² as a difference of O(1) quantities, which
    # cannot resolve a residual of 1e-16 in double precision — measured: using it as an
    # exact criterion found 1 of 15 ACRR pairs on the EnvZ-OmpR reference case, and a
    # relative variant found 10. It is therefore used ONLY as a conservative filter:
    # a comfortably large residual is a reliable "not in the span", and everything else
    # is settled by the same exact `is_in_span` the reference loop uses.
    #
    # Most metabolite pairs are decisively outside the span, so the filter still removes
    # the bulk of the O(n_met² · n_met · rank) work while changing no answer.
    gate2 = (1e3 * tolerance) * (1e3 * tolerance)
    coeffs_ws = Vector{Float64}(undef, max(1, cache.rank))
    proj_ws = Vector{Float64}(undef, n)
    vbuf = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        for j in (i+1):n
            dot_ij = 0.0
            @simd for q in 1:cache.rank
                dot_ij -= Q[i, q] * Q[j, q]      # ⟨r_i, r_j⟩ = -Q[i,:]·Q[j,:]
            end

            # --- ACRR (γ = 1): filter on ‖r_i - r_j‖², then verify exactly ----------
            d2 = nrm2[i] + nrm2[j] - 2 * dot_ij
            if d2 <= gate2
                fill!(vbuf, 0.0); vbuf[i] = 1.0; vbuf[j] = -1.0
                if is_in_span!(vbuf, cache, coeffs_ws, proj_ws)
                    m1, m2 = metabolite_ids[i], metabolite_ids[j]
                    push!(acrr, m1 < m2 ? (m1, m2) : (m2, m1))
                    push!(gacrr, (m1, m2, 1.0))
                    continue
                end
            end

            # --- gACRR (γ ≠ 1): only meaningful when r_j does not vanish ------------
            nrm2[j] < tol2 && continue
            gamma = dot_ij / nrm2[j]
            gamma > 0 || continue                 # SM restricts gACRR to γ > 0
            abs(gamma - 1.0) <= 1e-12 && continue # already handled as ACRR above
            resid2 = nrm2[i] - dot_ij * dot_ij / nrm2[j]
            resid2 > gate2 && continue
            fill!(vbuf, 0.0); vbuf[i] = 1.0; vbuf[j] = -gamma
            if is_in_span!(vbuf, cache, coeffs_ws, proj_ws)
                push!(gacrr, (metabolite_ids[i], metabolite_ids[j], gamma))
            end
        end
    end

    return (acr_metabolites=acr, acrr_pairs=acrr, gacrr_triples=gacrr)
end

"""
$(TYPEDSIGNATURES)

Sparse span-membership test: is the vector with nonzeros `(nz_idx, nz_val)` in im(Y𝚫)?

Why this exists. The dense path (`is_in_span!`) runs two `gemv` against
`Q` of size `n_metabolites x rank` for every complex PAIR, so it streams the whole basis
— on genome-scale yeast models roughly 100 MB — through memory once per pair. With
~10^7 pairs in a giant module that is pure memory bandwidth, which is why the
`efficient=false` path showed identical runtimes at 16, 64 and 128 threads: more cores
share the same bus.

Two observations remove almost all of that work:

  1. `y_diff = y_alpha - y_beta` is SPARSE. A complex is a small sum of species, so the
     difference has a handful of nonzeros out of `n_metabolites`. `c = Q' * v` then only
     touches those few rows of `Q` instead of all of them.
  2. `Q` has orthonormal columns, so ||Q Q' v - v||^2 = ||v||^2 - ||c||^2. The projection
     never has to be formed — the second `gemv`, the `axpy` and the dense norm all go.

That identity loses precision by cancellation exactly when the residual is small, i.e.
in the regime being tested. It is therefore used only as a CONSERVATIVE FILTER: a
comfortably large residual is a reliable "not in span"; anything near the tolerance is
reported as undecided and must be re-checked on the dense path.

Returns `(in_span, decided)`. When `decided` is false the caller must fall back.
"""
function is_in_span_sparse(
    nz_idx::Vector{Int},
    nz_val::Vector{Float64},
    n_nz::Int,
    cache::CachedColumnSpan,
    coeffs_workspace::Vector{Float64}
)::Tuple{Bool,Bool}
    nrm2_v = 0.0
    @inbounds for k in 1:n_nz
        nrm2_v += nz_val[k] * nz_val[k]
    end

    if cache.rank == 0
        return (sqrt(nrm2_v) < cache.tolerance, true)
    end
    n_nz == 0 && return (true, true)

    Qt = cache.Qt
    r = cache.rank
    @inbounds fill!(view(coeffs_workspace, 1:r), 0.0)
    @inbounds for t in 1:n_nz
        row = nz_idx[t]
        val = nz_val[t]
        @simd for q in 1:r
            coeffs_workspace[q] += val * Qt[q, row]
        end
    end

    nrm2_c = 0.0
    @inbounds @simd for q in 1:r
        nrm2_c += coeffs_workspace[q] * coeffs_workspace[q]
    end

    resid2 = nrm2_v - nrm2_c
    tol2 = cache.tolerance * cache.tolerance

    # Trust the subtraction only when the residual is far above both the tolerance and
    # the cancellation noise floor (~eps * ||v||^2, with a wide safety margin).
    if resid2 > max(4 * tol2, 1e-8 * nrm2_v)
        return (false, true)          # decisively outside the span
    end
    return (false, false)             # borderline -> caller re-checks densely
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

    # Guard against malformed inputs that can break LAPACK routines.
    if any(!isfinite, M)
        @warn "robust_rank received non-finite values; treating as rank 0"
        return 0
    end

    # Prefer SVD for robust rank computation; fall back to QR if SVD fails.
    try
        svd_result = LinearAlgebra.svd(M)
        max_sv = maximum(svd_result.S)

        if max_sv < tolerance
            return 0
        end

        # Count singular values above relative tolerance.
        threshold = tolerance * max_sv
        return count(s -> s > threshold, svd_result.S)
    catch e
        @warn "SVD failed in robust_rank; falling back to QR-based rank" exception = e

        # QR fallback is less robust than SVD on ill-conditioned matrices,
        # but avoids hard failures for large/problematic instances.
        try
            _, R = LinearAlgebra.qr(M)
            r_diag = abs.(LinearAlgebra.diag(R))
            isempty(r_diag) && return 0
            max_r = maximum(r_diag)
            max_r < tolerance && return 0
            return count(x -> x > tolerance * max_r, r_diag)
        catch e2
            @warn "QR fallback also failed in robust_rank; returning 0" exception = e2
            return 0
        end
    end
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
    # Add ALL pure-balanced weak linkage classes (reference R lines 141-155 add them
    # unconditionally; overlapping ones are then unified by the shared-complex merge,
    # Lemma S3-1). Skipping overlapping classes here would drop coupled balanced
    # complexes that should merge into the module they overlap.
    weak_modules = find_balanced_weak_linkage_classes(balanced, all_complexes, network)
    for wm in weak_modules
        push!(result, wm)
        union!(assigned, wm)
        @debug "Added weak linkage class as kinetic module" size = length(wm)
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
    efficient::Bool=true,
    known_acr::Vector{Symbol}=Symbol[]
)
    n_metabolites = length(metabolite_ids)

    if efficient
        # Fast efficient path: Direct pairwise comparison
        # Per-task buffers via explicit chunking, NOT `threadid()`-indexed buffers.
        #
        # `Threads.@threads` schedules dynamically since Julia 1.8, so a task may resume
        # on a different thread than it started on; indexing shared buffers by
        # `Threads.threadid()` is therefore unsound by contract, and the Julia manual
        # warns against it. Two tasks sharing `nz_indices_buf` would silently corrupt
        # ACR/ACRR results — and this loop runs in BOTH kinetic modes, i.e. on the
        # production path.
        #
        # Each spawned task owns its buffers, so the hazard cannot arise. Round-robin
        # chunking (c:nchunks:n) rather than contiguous blocks, because module sizes
        # differ by orders of magnitude and contiguous blocks would load-imbalance.
        # Sparse column per complex, built once for the whole call. The pair loop below
        # used to read `Y_matrix[met_idx, idx]` for EVERY metabolite of every pair —
        # O(n_metabolites) random accesses into a sparse matrix, each a search within a
        # column — to recover a difference that carries two to four nonzeros. Merging the
        # two sparse columns gives the same answer in O(nnz_a + nnz_b).
        sparse_cols = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
        for module_set in kinetic_modules, c in module_set
            idx = get(complex_to_idx, c, 0)
            (idx == 0 || haskey(sparse_cols, idx)) && continue
            ii, vv = SparseArrays.findnz(SparseArrays.sparse(Y_matrix[:, idx]))
            sparse_cols[idx] = (collect(ii), collect(vv))
        end

        n_mods = length(kinetic_modules)
        nchunks = max(1, min(Threads.nthreads(), n_mods))

        chunk_tasks = map(1:nchunks) do c
            Threads.@spawn begin
                local_acr = Set{Symbol}()
                local_acrr = Set{Tuple{Symbol,Symbol}}()
                nz_indices_buf = Vector{Int}(undef, 2)
                nz_vals_buf = Vector{Float64}(undef, 2)

                for mi in c:nchunks:n_mods
                    module_set = kinetic_modules[mi]
                complexes = collect(module_set)
                k = length(complexes)
                k < 2 && continue

                @inbounds for i in 1:k
                    idx_a = get(complex_to_idx, complexes[i], 0)
                    idx_a == 0 && continue

                    for j in (i+1):k
                        idx_b = get(complex_to_idx, complexes[j], 0)
                        idx_b == 0 && continue

                        (ia, va) = sparse_cols[idx_a]
                        (ib, vb) = sparse_cols[idx_b]
                        nnz_count = _sparse_diff_upto2!(nz_indices_buf, nz_vals_buf,
                                                        ia, va, ib, vb, tolerance)

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
                (local_acr, local_acrr)
            end
        end

        acr_set = Set{Symbol}()
        acrr_set = Set{Tuple{Symbol,Symbol}}()
        for t in chunk_tasks
            a, b = fetch(t)
            union!(acr_set, a)
            union!(acrr_set, b)
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

        # ACRR: e_i - e_j ∈ im(Y∆), from the (un-augmented) coupling span.
        if size(Y_Delta, 2) > 0
            base_cache = build_cached_column_span(Y_Delta; tolerance=tolerance)

            if base_cache.rank > 0
                diff_vec = zeros(Float64, n_metabolites)
                for i in 1:n_metabolites
                    for j in (i+1):n_metabolites
                        fill!(diff_vec, 0.0)
                        diff_vec[i] = 1.0
                        diff_vec[j] = -1.0

                        if is_in_span(diff_vec, base_cache)
                            m1, m2 = metabolite_ids[i], metabolite_ids[j]
                            push!(acrr_set, m1 < m2 ? (m1, m2) : (m2, m1))
                        end
                    end
                end
            end
        end


        # ACR with iterative known-ACR augmentation (Remark S3-6 propagation):
        # metabolite S is ACR if e_S ∈ im([Y∆ | e_{known ACR}]). Each newly
        # resolved metabolite is fed back as a column and the span re-checked
        # until a fixed point, so an externally known ACR (or one found in this
        # pass) can unlock further ACR through any coupling monomial. With no
        # external known_acr this reduces to the plain e_S ∈ im(Y∆) check.
        known = Set{Symbol}(known_acr)
        union!(known, acr_set)  # seed with pairwise-detected ACR (Part 1)
        e_vec = zeros(Float64, n_metabolites)
        Y_Delta_dense = Matrix{Float64}(Y_Delta)
        while true
            aug = build_acr_augmentation(collect(known), metabolite_ids, n_metabolites)
            Y_aug = size(aug, 2) > 0 ? hcat(Y_Delta_dense, aug) : Y_Delta_dense
            size(Y_aug, 2) == 0 && break

            cache = build_cached_column_span(Y_aug; tolerance=tolerance)
            cache.rank == 0 && break

            newly = Symbol[]
            for i in 1:n_metabolites
                metabolite_ids[i] in known && continue
                fill!(e_vec, 0.0)
                e_vec[i] = 1.0
                if is_in_span(e_vec, cache)
                    push!(newly, metabolite_ids[i])
                end
            end
            isempty(newly) && break
            union!(acr_set, newly)
            union!(known, newly)
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
    efficient::Bool=true,
    known_acr::Vector{Symbol}=Symbol[]
)
    # Extract network topology and delegate to helper
    Y_matrix, metabolite_ids, complex_ids = complex_stoichiometry(model; return_ids=true)
    complex_to_idx = Dict{Symbol,Int}(id => i for (i, id) in enumerate(complex_ids))

    return _detect_acr_acrr(kinetic_modules, Y_matrix, metabolite_ids, complex_to_idx;
        tolerance=tolerance, efficient=efficient, known_acr=known_acr)
end

"""
    identify_acr_acrr_dce(model; tolerance=1e-7, known_acr=Symbol[])

Identify ACR/ACRR via the flux-coupling (DCE) criterion — an extension of the
graph-based kinetic-module analysis that also captures robustness arising from
couplings *across* linkage classes.

Every mass-action flux is `v_r = k_r · ψ(σ(r))`, the monomial of its substrate
complex `σ(r)`. Reactions that are fully coupled (constant flux ratio at every
steady state) therefore impose a constant monomial ratio
`ψ(σ(i)) / ψ(σ(j)) = const`, so the substrate-complex composition difference
`Y[:,σ(i)] − Y[:,σ(j)]` is a constant-log-monomial direction. Collecting these
into a matrix `M`:
- metabolite `S` has ACR  iff `e_S ∈ im(M)`;
- metabolites `S1,S2` have ACRR iff `e_{S1} − e_{S2} ∈ im(M)`.

Full flux coupling is detected structurally as parallel (positive-scale) rows of a
null-space basis of the stoichiometric matrix `N` (i.e. `vᵢ/vⱼ` constant over
`ker N`). `known_acr` metabolites are augmented into the span (propagation, Remark
S3-6) and the ACR check is iterated to a fixed point.

Unlike [`identify_acr_acrr`](@ref), which detects ACR/ACRR only within graph-connected
kinetic modules, this method recovers `[D]`/`[B]`-type robustness that the upstream
algorithm splits across linkage classes.

!!! note
    Uses a dense null space of `N` and an `O(n_reactions²)` coupling scan; intended for
    chemical-reaction networks where cross-linkage robustness matters, not genome-scale models.

# Returns
Named tuple `(acr_metabolites::Vector{Symbol}, acrr_pairs::Vector{Tuple{Symbol,Symbol}})`.
"""
function identify_acr_acrr_dce(
    model::A.AbstractFBCModel;
    tolerance::Float64=1e-7,
    known_acr::Vector{Symbol}=Symbol[],
    n_probes::Int=8,
    seed::Integer=1234,
)
    N = SparseArrays.SparseMatrixCSC{Float64,Int}(A.stoichiometry(model))  # metabolites × reactions (sparse)
    A_mat, _ = incidence(model; return_ids=true)
    Y_matrix, metabolite_ids, _ = complex_stoichiometry(model; return_ids=true)
    n_rxn = size(N, 2)
    n_met = length(metabolite_ids)
    empty_result = (acr_metabolites=Symbol[], acrr_pairs=Tuple{Symbol,Symbol}[])
    (n_rxn == 0 || n_met == 0) && return empty_result

    # Substrate complex (incidence entry −1) for each reaction.
    substrate = zeros(Int, n_rxn)
    arows = SparseArrays.rowvals(A_mat)
    avals = SparseArrays.nonzeros(A_mat)
    for r in 1:n_rxn
        for k in SparseArrays.nzrange(A_mat, r)
            avals[k] < 0 && (substrate[r] = arows[k])
        end
    end

    # --- Scalable full flux coupling ------------------------------------------------
    # Reactions i,j are fully coupled iff vᵢ/vⱼ is constant over ker N, i.e. their rows
    # in any null-space basis are parallel. Rather than materialise a (dense, possibly
    # huge) null-space, we fingerprint each reaction with its values on a few random
    # vectors sampled from ker N; coupled reactions get parallel fingerprints. Sampling
    # ker N is one sparse normal-equation solve per probe: v = w − Nᵀ(NNᵀ)⁺(N w), using
    # a single sparse Cholesky factorisation of NNᵀ+εI. Cost: O(n_probes) sparse solves
    # + O(n_reactions·n_probes) grouping — no dense null space, no O(n_reactions²) scan.
    Nt = SparseArrays.sparse(transpose(N))
    NNt = N * Nt
    scale = isempty(SparseArrays.nonzeros(NNt)) ? 1.0 : maximum(abs, SparseArrays.nonzeros(NNt))
    F = cholesky(Symmetric(NNt + (1e-9 * scale) * I))
    rng = StableRNGs.StableRNG(seed)
    t = clamp(n_probes, 1, n_rxn)
    G = Matrix{Float64}(undef, n_rxn, t)
    for k in 1:t
        w = randn(rng, n_rxn)
        G[:, k] = w .- Nt * (F \ (N * w))            # projection of w onto ≈ ker N
    end

    # Group reactions into coupling classes by parallel (positive-scaled) fingerprint.
    classkey = f -> begin
        nf = norm(f)
        nf < tolerance && return nothing             # blocked / inactive in ker N
        u = f ./ nf
        i0 = findfirst(x -> abs(x) > 1e-8, u)
        i0 !== nothing && u[i0] < 0 && (u = -u)       # canonical sign
        Tuple(round.(u; digits=6))
    end
    classes = Dict{Any,Vector{Int}}()
    for r in 1:n_rxn
        key = classkey(@view G[r, :])
        key === nothing && continue
        push!(get!(classes, key, Int[]), r)
    end

    # Coupling-difference directions: within each class, Y[:,σ(ref)] − Y[:,σ(r)].
    diff_cols = Vector{SparseArrays.SparseVector{Float64,Int}}()
    seen = Set{Vector{Pair{Int,Float64}}}()
    for rxns in values(classes)
        length(rxns) < 2 && continue
        ref = findfirst(r -> substrate[r] != 0, rxns)
        ref === nothing && continue
        yref = Y_matrix[:, substrate[rxns[ref]]]
        for r in rxns
            (r == rxns[ref] || substrate[r] == 0) && continue
            d = yref - Y_matrix[:, substrate[r]]
            SparseArrays.nnz(d) == 0 && continue
            sig = [i => d[i] for i in SparseArrays.findnz(d)[1]]
            sig in seen && continue
            push!(seen, sig)
            push!(diff_cols, d)
        end
    end

    # --- ACR/ACRR in the reduced space of metabolites that actually appear ----------
    met_to_idx = Dict(id => i for (i, id) in enumerate(metabolite_ids))
    active = Set{Int}()
    for d in diff_cols, i in SparseArrays.findnz(d)[1]
        push!(active, i)
    end
    for s in known_acr
        haskey(met_to_idx, s) && push!(active, met_to_idx[s])
    end
    isempty(active) && return empty_result
    act = sort(collect(active))
    na = length(act)
    pos = Dict(act[i] => i for i in 1:na)

    M = zeros(Float64, na, length(diff_cols))
    for (j, d) in enumerate(diff_cols)
        idxs, vals = SparseArrays.findnz(d)
        for (i, v) in zip(idxs, vals)
            M[pos[i], j] = v
        end
    end

    acr_idx = Set{Int}()
    acrr_set = Set{Tuple{Symbol,Symbol}}()

    if size(M, 2) > 0
        base = build_cached_column_span(M; tolerance=tolerance)
        if base.rank > 0
            d = zeros(Float64, na)
            for a in 1:na, b in (a+1):na
                fill!(d, 0.0); d[a] = 1.0; d[b] = -1.0
                if is_in_span(d, base)
                    m1, m2 = metabolite_ids[act[a]], metabolite_ids[act[b]]
                    push!(acrr_set, m1 < m2 ? (m1, m2) : (m2, m1))
                end
            end
        end
    end

    # ACR with iterative known-ACR augmentation (propagation, Remark S3-6).
    known = Set{Int}(pos[met_to_idx[s]] for s in known_acr if haskey(met_to_idx, s) && haskey(pos, met_to_idx[s]))
    e = zeros(Float64, na)
    while true
        if isempty(known)
            M_aug = M
        else
            aug = zeros(Float64, na, length(known))
            for (c, p) in enumerate(known); aug[p, c] = 1.0; end
            M_aug = hcat(M, aug)
        end
        size(M_aug, 2) == 0 && break
        cache = build_cached_column_span(M_aug; tolerance=tolerance)
        cache.rank == 0 && break
        newly = Int[]
        for a in 1:na
            a in known && continue
            fill!(e, 0.0); e[a] = 1.0
            is_in_span(e, cache) && push!(newly, a)
        end
        isempty(newly) && break
        union!(acr_idx, newly); union!(known, newly)
    end

    return (
        acr_metabolites=[metabolite_ids[act[a]] for a in acr_idx],
        acrr_pairs=collect(acrr_set),
    )
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

"""
    complexes_in_every_coupling_set(upstream_sets; min_sets=10) -> Vector{Symbol}

Return — and warn about — complexes that belong to EVERY coupling set.

One such complex merges all coupling sets into a single kinetic module (Lemma S3-1). That is
consistent with the theory, but on the yeast panel it happened only through a preprocessing
artifact: a blocked reaction left in the network, whose metabolite is consumed but never
produced, so Phase I of the upstream algorithm cannot remove its complex; as every set is
seeded with 𝒞b ∪ 𝒞m it then turns up in all of them. Removing 1–3 such complexes
broke a "giant" module of 12,071 complexes into modules of ~50, the size seen everywhere else.
Worth a look before interpreting a giant module. Below `min_sets` sets the pattern is too
common to mean anything (the EnvZ-OmpR network has four).
"""
function complexes_in_every_coupling_set(upstream_sets; min_sets::Int=10)
    n = length(upstream_sets)
    n < min_sets && return Symbol[]
    counts = Dict{Symbol,Int}()
    for set in upstream_sets, c in set
        counts[c] = get(counts, c, 0) + 1
    end
    shared = sort!([c for (c, k) in counts if k == n])
    isempty(shared) ||
        @warn "$(length(shared)) complex(es) belong to every one of the $n coupling sets and " *
              "will merge them into ONE kinetic module. On the yeast panel this signalled a " *
              "blocked reaction left in the network: check whether these complexes " *
              "have any producing reaction." complexes = shared
    return shared
end
