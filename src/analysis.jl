"""
Main analysis algorithms for COCOA.

This module contains:
- Staged processing with transitivity optimization
- Concordance batch processing
- Main concordance analysis function
- Module extraction and result formatting
"""

using AbstractFBCModels
using COBREXA
using DataFrames
using Distributed
using Statistics
using Random
using StableRNGs
using JuMP
using DocStringExtensions
using ProgressMeter
using LinearAlgebra

import ConstraintTrees as C
import Base.Iterators

"""
$(TYPEDSIGNATURES)

Process concordance analysis in stages.
When concordant pairs are found, remaining pairs involving those complexes are prioritized.
This exploits the clustering tendency of concordant complexes to dramatically improve efficiency.
"""
function process_in_stages(
    constraints::C.ConstraintTree,
    complexes::Vector{Complex},
    candidate_priorities::Vector{PairPriority},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix},
    concordance_tracker::Union{ConcordanceTracker,SharedConcordanceTracker};
    optimizer,
    settings=[],
    workers=workers,
    stage_size::Int=500,
    batch_size::Int=100,
    tolerance::Float64=1e-12
)
    stage_results = Dict{String,Any}(
        "stages_completed" => 0,
        "pairs_processed" => 0,
        "concordant_pairs" => Set{Tuple{Int,Int}}(),
        "non_concordant_pairs" => 0,
        "skipped_by_transitivity" => 0,
        "optimization_results" => Dict{Tuple{Int,Int,Symbol},Float64}()
    )

    # Sort pairs by priority: high-confidence first, then by correlation
    sorted_pairs = sort(candidate_priorities,
        by=p -> (-p.is_high_confidence, -abs(p.correlation), p.c1_idx, p.c2_idx))

    # Convert to simple format for processing
    remaining_pairs = [(p.c1_idx, p.c2_idx, p.directions) for p in sorted_pairs]

    stage_count = 0
    total_pairs = length(remaining_pairs)
    processed_pairs = 0
    all_concordant_complexes = Set{Symbol}()  # Accumulate across stages

    prog = Progress(
        total_pairs,
        desc="Concordance analysis: ",
        dt=1.0,
        barlen=50,
        output=stdout,
        showspeed=true
    )

    concordant_count = 0
    eliminated_count = 0
    prioritized_count = 0

    while !isempty(remaining_pairs)
        stage_count += 1

        # Clear cache if using regular tracker
        if isa(concordance_tracker, ConcordanceTracker)
            clear_module_cache!(concordance_tracker)
        end

        @debug "Starting stage $stage_count" remaining = length(remaining_pairs)

        # Filter out pairs that can be inferred by transitivity
        filtered_pairs, skipped_count = filter_transitive_pairs(
            remaining_pairs,
            complexes,
            concordance_tracker
        )
        stage_results["skipped_by_transitivity"] += skipped_count

        if isempty(filtered_pairs)
            @debug "No more pairs to process after transitivity filtering"
            break
        end

        # Take stage batch
        stage_size_actual = min(stage_size, length(filtered_pairs))
        stage_pairs = filtered_pairs[1:stage_size_actual]
        remaining_pairs = filtered_pairs[(stage_size_actual+1):end]

        @debug "Processing stage $stage_count" pairs = length(stage_pairs) batch_size

        # Split stage into batches for memory management
        all_batch_results = []
        n_batches = ceil(Int, length(stage_pairs) / batch_size)

        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(stage_pairs))
            batch_pairs = stage_pairs[start_idx:end_idx]

            @debug "Processing batch $batch_idx/$n_batches" batch_pairs = length(batch_pairs)

            # Process batch
            batch_results = process_concordance_batch(
                constraints, complexes, batch_pairs, A_matrix;
                optimizer=optimizer,
                settings=settings,
                workers=workers,
                tolerance=tolerance
            )

            append!(all_batch_results, batch_results)

            # Optional: force GC between batches for large models
            if batch_idx < n_batches
                GC.gc()
            end
        end

        batch_results = all_batch_results

        processed_pairs += length(stage_pairs)
        ProgressMeter.update!(prog, processed_pairs)

        # Extract newly concordant pairs from this stage
        newly_concordant = Vector{Tuple{Symbol,Symbol}}()
        stage_concordant_count = 0
        stage_prioritized_count = 0
        pairs_eliminated = 0

        # Update tracker with results
        for result in batch_results
            c1_idx, c2_idx, direction, is_concordant, lambda = result

            # Get tracker indices
            tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
            tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]

            if is_concordant
                union_sets!(concordance_tracker, tracker_idx1, tracker_idx2)
                push!(stage_results["concordant_pairs"], (c1_idx, c2_idx))
                push!(newly_concordant, (complexes[c1_idx].id, complexes[c2_idx].id))
                stage_concordant_count += 1

                if !isnothing(lambda)
                    stage_results["optimization_results"][(c1_idx, c2_idx, direction)] = lambda
                end
            else
                add_non_concordant!(concordance_tracker, tracker_idx1, tracker_idx2)
                stage_results["non_concordant_pairs"] += 1
            end
        end

        concordant_count = length(stage_results["concordant_pairs"])

        # Apply transitivity elimination after finding concordant pairs
        if stage_concordant_count > 0
            prev_num_pairs = length(remaining_pairs)
            remaining_pairs = apply_transitivity_elimination(
                remaining_pairs,
                newly_concordant,
                complexes,
                concordance_tracker
            )
            pairs_eliminated = prev_num_pairs - length(remaining_pairs)
            eliminated_count += pairs_eliminated
        end

        # Reprioritize if we found concordant pairs
        if stage_concordant_count >= 1
            # Add newly concordant complexes to accumulated set
            for (c1_id, c2_id) in newly_concordant
                push!(all_concordant_complexes, c1_id, c2_id)
            end

            # Reprioritize remaining pairs
            remaining_pairs = reprioritize_by_concordant_complexes(
                remaining_pairs,
                all_concordant_complexes,
                complexes
            )

            # Count prioritized pairs (those involving concordant complexes)
            for (c1_idx, c2_idx, _) in remaining_pairs
                if (complexes[c1_idx].id in all_concordant_complexes) || (complexes[c2_idx].id in all_concordant_complexes)
                    stage_prioritized_count += 1
                end
            end
            prioritized_count = stage_prioritized_count
        end

        stage_results["pairs_processed"] += length(stage_pairs)
        stage_results["stages_completed"] = stage_count

        @debug "Stage $stage_count complete" new_concordant = stage_concordant_count

        # Update progress bar with live values
        ProgressMeter.update!(prog, processed_pairs;
            showvalues=[
                (:concordant, concordant_count),
                (:eliminated, eliminated_count),
                (:prioritized, prioritized_count)
            ]
        )
    end

    ProgressMeter.finish!(prog)

    @info "Concordance testing complete" (
        total_stages=stage_count,
        total_concordant_pairs=length(stage_results["concordant_pairs"]),
        total_concordant_complexes=length(all_concordant_complexes)
    )

    return stage_results
end

"""
$(TYPEDSIGNATURES)

Filter out pairs that can be inferred by transitivity to avoid redundant testing.
"""
function filter_transitive_pairs(
    remaining_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    complexes::Vector{Complex},
    concordance_tracker::Union{ConcordanceTracker,SharedConcordanceTracker}
)
    filtered_pairs = Tuple{Int,Int,Set{Symbol}}[]
    skipped_count = 0

    for (c1_idx, c2_idx, directions) in remaining_pairs
        # Get tracker indices
        tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
        tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]

        # Skip if already concordant
        if are_concordant(concordance_tracker, tracker_idx1, tracker_idx2)
            skipped_count += 1
            continue
        end

        # Skip if known non-concordant
        if is_non_concordant(concordance_tracker, tracker_idx1, tracker_idx2)
            skipped_count += 1
            continue
        end

        push!(filtered_pairs, (c1_idx, c2_idx, directions))
    end

    if skipped_count > 0
        @debug "Filtered pairs by transitivity" skipped = skipped_count remaining = length(filtered_pairs)
    end

    return filtered_pairs, skipped_count
end

"""
$(TYPEDSIGNATURES)

Reprioritize remaining pairs by moving pairs involving concordant complexes to the front.
Pairs involving concordant complexes are sorted by correlation strength.
"""
function reprioritize_by_concordant_complexes(
    remaining_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    concordant_complexes::Set{Symbol},
    complexes::Vector{Complex}
)
    if isempty(concordant_complexes)
        return remaining_pairs
    end

    # Partition pairs: those involving concordant complexes vs others
    priority_pairs = Vector{Tuple{Int,Int,Set{Symbol}}}()
    regular_pairs = Vector{Tuple{Int,Int,Set{Symbol}}}()

    for (c1_idx, c2_idx, directions) in remaining_pairs
        c1_id = complexes[c1_idx].id
        c2_id = complexes[c2_idx].id

        if c1_id in concordant_complexes || c2_id in concordant_complexes
            push!(priority_pairs, (c1_idx, c2_idx, directions))
        else
            push!(regular_pairs, (c1_idx, c2_idx, directions))
        end
    end

    # Sort priority pairs by complex indices for deterministic ordering
    # (We don't have correlation info at this stage, so use indices as proxy)
    sort!(priority_pairs, by=p -> (p[1], p[2]))

    @debug "Reprioritization effect" (
        priority_pairs=length(priority_pairs),
        regular_pairs=length(regular_pairs),
        priority_percentage=round(100 * length(priority_pairs) / length(remaining_pairs), digits=1)
    )

    # Return reprioritized list: priority pairs first, then regular pairs
    return vcat(priority_pairs, regular_pairs)
end

"""
$(TYPEDSIGNATURES)

Apply transitivity elimination to remove pairs that are guaranteed to be concordant
based on already discovered concordant pairs.
"""
function apply_transitivity_elimination(
    remaining_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    newly_concordant::Vector{Tuple{Symbol,Symbol}},
    complexes::Vector{Complex},
    concordance_tracker::Union{ConcordanceTracker,SharedConcordanceTracker}
)
    if isempty(newly_concordant)
        return remaining_pairs
    end

    # Filter out pairs that are now transitively concordant or non-concordant
    filtered_pairs = filter(remaining_pairs) do (c1_idx, c2_idx, directions)
        tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
        tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]

        # Keep pair if it's not transitively determined
        !are_concordant(concordance_tracker, tracker_idx1, tracker_idx2) &&
            !is_non_concordant(concordance_tracker, tracker_idx1, tracker_idx2)
    end

    eliminated_count = length(remaining_pairs) - length(filtered_pairs)

    if eliminated_count > 0
        @debug "Transitivity elimination" (
            pairs_eliminated=eliminated_count,
            remaining_pairs=length(filtered_pairs)
        )
    end

    return filtered_pairs
end

"""
$(TYPEDSIGNATURES)

Pre-compute reaction indices for all expanded pairs to avoid repeated union operations.
"""
function precompute_reaction_indices(expanded_pairs, activity_lookup, complexes)
    reaction_indices_cache = Dict{Tuple{Int,Int},Set{Int}}()

    for (c1_idx, c2_idx, _) in expanded_pairs
        key = (c1_idx, c2_idx)
        if !haskey(reaction_indices_cache, key)
            c1_activity = activity_lookup[complexes[c1_idx].id]
            c2_activity = activity_lookup[complexes[c2_idx].id]
            reaction_indices_cache[key] = Set(union(c1_activity.idxs, c2_activity.idxs))
        end
    end

    return reaction_indices_cache
end

"""
$(TYPEDSIGNATURES)

Build efficient activity expression avoiding repeated allocations.
"""
function build_activity_expression(activity::C.LinearValue, w::Dict{Int,JuMP.VariableRef}, reaction_indices::Set{Int})
    # Pre-filter relevant indices to avoid repeated checks
    relevant_terms = JuMP.AffExpr()

    for i in eachindex(activity.idxs)
        if activity.idxs[i] in reaction_indices
            JuMP.add_to_expression!(relevant_terms, activity.weights[i], w[activity.idxs[i]])
        end
    end

    return relevant_terms
end

"""
$(TYPEDSIGNATURES)

Process a batch of concordance tests using ConstraintTrees approach with parallel batch processing.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    complexes::Vector{Complex},
    batch_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    A_matrix::Union{SparseIncidenceMatrix,SharedSparseMatrix};
    optimizer,
    settings=[],
    workers=workers,
    tolerance::Float64=1e-12
)
    # Optimize constraint trees by pruning unused variables
    pruned_constraints = C.prune_variables(constraints)

    # Extract activity lookup from pre-computed ConstraintTrees
    activity_lookup = Dict{Symbol,C.LinearValue}()

    for (complex_id, constraint) in pruned_constraints.concordance_analysis.complexes
        activity_lookup[complex_id] = constraint.value
    end

    # Expand pairs by direction
    expanded_pairs = []
    for (c1_idx, c2_idx, directions) in batch_pairs
        for direction in directions
            push!(expanded_pairs, (c1_idx, c2_idx, direction))
        end
    end

    # Pre-compute reaction indices to avoid repeated union operations
    reaction_indices_cache = precompute_reaction_indices(expanded_pairs, activity_lookup, complexes)

    # Pre-build ALL test data on the main thread
    # This moves all preparation work out of the worker functions
    all_test_data = []
    pair_mappings = []

    for (c1_idx, c2_idx, direction) in expanded_pairs
        c1_activity = activity_lookup[complexes[c1_idx].id]
        c2_activity = activity_lookup[complexes[c2_idx].id]
        reaction_indices = reaction_indices_cache[(c1_idx, c2_idx)]

        # Pre-build all test data for this specific concordance test
        test_data = (
            c1_activity=c1_activity,
            c2_activity=c2_activity,
            direction=direction,
            tolerance=tolerance,
            reaction_indices=reaction_indices
        )

        push!(all_test_data, test_data)
        push!(pair_mappings, (c1_idx, c2_idx, direction))
    end

    # Process ALL concordance tests in parallel using screen_optimization_model
    # Each worker gets pre-built test data, minimal constraint building needed
    batch_results = COBREXA.screen_optimization_model(
        pruned_constraints,
        all_test_data;
        optimizer=optimizer,
        settings=settings,
        workers=workers
    ) do om, test_data
        # Extract test parameters
        c1_activity = test_data.c1_activity
        c2_activity = test_data.c2_activity
        direction = test_data.direction
        tolerance = test_data.tolerance
        reaction_indices = test_data.reaction_indices

        # Process both MIN and MAX senses for this test
        results = []
        for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
            # Clear any existing objective
            @objective(om, sense, 0)

            # Pre-allocate variable Dict with expected size
            w = Dict{Int,JuMP.VariableRef}()
            sizehint!(w, length(reaction_indices))

            # Create transformed variables (only for involved reactions)
            for j in reaction_indices
                w[j] = @variable(om)
            end
            t = @variable(om)

            # Direction constraint on t
            if direction == :positive
                @constraint(om, t >= tolerance)
            else
                @constraint(om, t <= -tolerance)
            end

            # Extract bounds from original flux variables
            x = om[:x]

            # Complex c2 activity constraint using efficient expression building
            c2_expr = build_activity_expression(c2_activity, w, reaction_indices)
            target = direction == :positive ? 1.0 : -1.0
            @constraint(om, c2_expr == target)

            # Charnes-Cooper bounds constraints
            if direction == :positive
                for j in reaction_indices
                    lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
                    ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
                    @constraint(om, w[j] - lb * t >= 0)
                    @constraint(om, ub * t - w[j] >= 0)
                end
            else
                for j in reaction_indices
                    lb = has_lower_bound(x[j]) ? lower_bound(x[j]) : -1e6
                    ub = has_upper_bound(x[j]) ? upper_bound(x[j]) : 1e6
                    @constraint(om, w[j] - ub * t >= 0)
                    @constraint(om, lb * t - w[j] >= 0)
                end
            end

            # Objective: optimize complex c1 activity using efficient expression building
            c1_expr = build_activity_expression(c1_activity, w, reaction_indices)
            @objective(om, sense, c1_expr)
            optimize!(om)

            if termination_status(om) == OPTIMAL
                push!(results, objective_value(om))
            else
                push!(results, nothing)
            end
        end

        return results  # [min_val, max_val]
    end

    # Process batch results back to individual concordance test results
    results = []

    # batch_results contains results for each test, with each test processed for both MIN and MAX
    for (i, test_results) in enumerate(batch_results)
        c1_idx, c2_idx, direction = pair_mappings[i]

        # Extract min and max values from the test results
        min_val = test_results[1]  # MIN_SENSE result
        max_val = test_results[2]  # MAX_SENSE result

        if min_val === nothing || max_val === nothing
            is_conc = false
            lambda = nothing
        else
            # Check concordance
            is_conc = isapprox(min_val, max_val; atol=tolerance)
            lambda = is_conc ? min_val : nothing
        end

        push!(results, (c1_idx, c2_idx, direction, is_conc, lambda))
    end

    # Aggregate results by pair
    pair_results = Dict{Tuple{Int,Int},Dict{Symbol,Tuple{Bool,Union{Float64,Nothing}}}}()

    for (c1_idx, c2_idx, direction, is_conc, lambda) in results
        pair_key = (c1_idx, c2_idx)
        if !haskey(pair_results, pair_key)
            pair_results[pair_key] = Dict{Symbol,Tuple{Bool,Union{Float64,Nothing}}}()
        end
        pair_results[pair_key][direction] = (is_conc, lambda)
    end

    # Check if all required directions are concordant
    final_results = []
    for ((c1_idx, c2_idx), dir_results) in pair_results
        all_concordant = all(r[1] for r in values(dir_results))
        # Get lambda from any concordant direction
        lambda = nothing
        for (is_conc, lam) in values(dir_results)
            if is_conc && !isnothing(lam)
                lambda = lam
                break
            end
        end

        push!(final_results, (c1_idx, c2_idx, :both, all_concordant, lambda))
    end

    return final_results
end

"""
$(TYPEDSIGNATURES)

Main concordance analysis function optimized for large models and HPC execution.

# Arguments
- `model`: Metabolic model to analyze
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `modifications=Function[]`: Model modifications to apply
- `settings=[]`: Solver settings
- `workers=workers()`: Worker processes for parallel computation
- `tolerance=1e-12`: Numerical tolerance for concordance testing
- `correlation_threshold=0.99`: Minimum correlation for candidate pairs
- `early_correlation_threshold=0.95`: Early correlation threshold for filtering
- `sample_size=100`: Number of flux samples for correlation analysis
- `stage_size=500`: Number of pairs to process per stage
- `batch_size=100`: Number of pairs to process per batch (within each stage)
- `use_shared_arrays=true`: Enable SharedArrays for parallel processing
- `min_size_for_sharing=1_000_000`: Minimum array size for shared memory
- `min_valid_samples=30`: Minimum samples required for correlation
- `max_correlation_pairs=500_000`: Maximum correlation pairs to evaluate
- `seed=42`: Master random seed for hierarchical reproducible RNG. Uses StableRNGs for cross-platform reproducibility and generates deterministic seeds for different analysis components (warmup generation, batch coordination, etc.)
- `use_unidirectional_constraints=true`: Split reversible reactions
- `filter=[:cv, :cor]`: Filtering methods to use. Can contain `:cv` for coefficient of variation filtering, `:cor` for correlation filtering, or both
- `cv_threshold=0.01`: Maximum coefficient of variation for activity ratios (lower = more stringent)
- `cv_epsilon=1e-12`: Numerical stability epsilon for ratio calculations

# Returns
Named tuple with concordance analysis results including complexes, modules, and statistics.
"""
function concordance_analysis(
    model;
    modifications=Function[],
    optimizer,
    settings=[],
    workers=Distributed.workers(),
    tolerance::Float64=1e-12,
    correlation_threshold::Float64=0.99,
    early_correlation_threshold::Float64=0.95,
    sample_size::Int=100,
    stage_size::Int=500,
    batch_size::Int=100,
    use_shared_arrays::Bool=true,
    min_size_for_sharing::Int=1_000_000,
    min_valid_samples::Int=30,
    max_correlation_pairs::Int=500_000,
    seed::Union{Int,Nothing}=42,
    use_unidirectional_constraints::Bool=true,
    # Filtering parameters
    filter::Vector{Symbol}=[:cv, :cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-12,
)
    start_time = time()

    # Use COBREXA's efficient model conversion
    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel for optimal performance"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    # Scale batch sizes with number of workers for better utilization
    n_workers = length(workers)
    effective_batch_size = max(batch_size, 50 * n_workers)  # At least 50 tasks per worker
    effective_stage_size = max(stage_size, 200 * n_workers)  # Scale stage size accordingly

    @info "Starting concordance analysis" n_workers tolerance early_correlation_threshold correlation_threshold sample_size use_unidirectional_constraints effective_batch_size effective_stage_size

    # Build constraints
    constraints =
        concordance_constraints(model; modifications, use_unidirectional_constraints, use_shared_arrays, min_size_for_sharing)

    # Extract complexes with potential shared memory 
    complexes, A_matrix, _ =
        extract_complexes_and_incidence(model; use_shared_arrays, min_size_for_sharing)
    n_complexes = length(complexes)
    complex_ids = [c.id for c in complexes]

    @info "Model statistics" n_complexes n_reactions = (
        isa(A_matrix, SharedSparseMatrix) ? A_matrix.n : A_matrix.n_reactions
    )

    # Step 1: Find trivially balanced complexes
    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(complexes)
    @info "Found trivially balanced complexes" n_trivivally_balanced = length(trivially_balanced)

    # Initialize balanced_complexes with trivially balanced
    balanced_complexes = Set{Symbol}()
    union!(balanced_complexes, trivially_balanced)

    # Step 2: Find trivially concordant pairs
    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(complexes)
    @info "Found trivially concordant pairs" n_trivially_concordant = length(trivial_pairs)

    # Step 3: Perform Activity Variability Analysis (AVA) and generate warmup points simultaneously
    @info "Performing Activity Variability Analysis and generating warmup points"

    # Add validation before AVA
    @debug "Validating constraint structure before AVA" n_complex_constraints = length(constraints.concordance_analysis.complexes)

    # Sample a few complexes to check their structure
    if haskey(constraints, :concordance_analysis) && haskey(constraints.concordance_analysis, :complexes)
        sample_count = min(3, length(constraints.concordance_analysis.complexes))
        sample_complexes = collect(Iterators.take(constraints.concordance_analysis.complexes, sample_count))
        for (cid, constraint) in sample_complexes
            @debug "Sample complex constraint" complex_id = cid constraint_type = typeof(constraint.value) n_terms = length(constraint.value.idxs)
        end
    else
        @warn "Complex activities not found in constraint tree" has_concordance_analysis = haskey(constraints, :concordance_analysis)
    end

    # Run FVA on all complexes using optimized settings
    ava_time = @elapsed ava_results = COBREXA.constraints_variability(
        constraints,
        constraints.concordance_analysis.complexes;
        optimizer=optimizer,
        settings=settings,  # Add silence for reduced overhead
        output=ava_output_with_warmup,
        output_type=Tuple{Float64,Vector{Float64}},
        workers=workers,
    )

    # Process the combined results
    complex_ranges = Dict{Symbol,Tuple{Float64,Float64}}()
    warmup_points = Vector{Float64}[]

    # Process FVA results correctly - iterate directly over the Tree
    @debug "Processing AVA results" n_ava_results = length(ava_results)

    for (cid, (min_result, max_result)) in ava_results
        @debug "Processing AVA result for complex $cid" min_result max_result

        # Only process complexes that are not balanced
        if min_result !== nothing && max_result !== nothing
            min_activity, min_flux = min_result
            max_activity, max_flux = max_result

            @debug "AVA result for $cid" min_activity max_activity

            # Store the activity range
            complex_ranges[cid] = (min_activity, max_activity)

            # Store warmup points  
            push!(warmup_points, min_flux)
            push!(warmup_points, max_flux)
        else
            @warn "AVA result is nothing for complex $cid" min_result max_result
        end
    end

    @info "AVA processing complete" n_complex_ranges = length(complex_ranges) ava_time_sec = round(ava_time, digits=2)

    # Convert warmup points to a matrix with robust handling
    warmup = if isempty(warmup_points)
        Matrix{Float64}(undef, 0, 0)
    else
        try
            collect(transpose(reduce(hcat, warmup_points)))
        catch e
            @warn "Failed to create warmup matrix: $e. Using empty matrix."
            Matrix{Float64}(undef, 0, 0)
        end
    end

    # Classify complexes by activity patterns using the extracted ranges
    balanced_complexes = Set{Symbol}()
    positive_complexes = Set{Int}()
    negative_complexes = Set{Int}()
    unrestricted_complexes = Set{Int}()

    # Note: Don't automatically include trivially balanced - verify through AVA first

    # Robust classification with proper error handling
    n_ava_balanced = 0
    n_trivial_ava_confirmed = 0
    n_trivial_ava_rejected = 0

    for (i, c) in enumerate(complexes)
        cid = c.id

        if haskey(complex_ranges, cid)
            min_val, max_val = complex_ranges[cid]

            @debug "Classifying complex $cid" min_val max_val tolerance is_trivially_balanced = (cid in trivially_balanced)

            # Robust numerical comparison - treat all complexes the same way
            if abs(min_val) < tolerance && abs(max_val) < tolerance
                push!(balanced_complexes, cid)
                n_ava_balanced += 1

                if cid in trivially_balanced
                    n_trivial_ava_confirmed += 1
                    @debug "Complex $cid: trivially balanced confirmed by AVA"
                else
                    @debug "Complex $cid: classified as balanced by AVA (not trivially balanced)"
                end
            elseif min_val >= -tolerance  # Can only be positive
                push!(positive_complexes, i)
                if cid in trivially_balanced
                    n_trivial_ava_rejected += 1
                    @warn "Trivially balanced complex $cid has positive-only activity" min_val max_val
                end
                @debug "Complex $cid classified as positive-only"
            elseif max_val <= tolerance  # Can only be negative
                push!(negative_complexes, i)
                if cid in trivially_balanced
                    n_trivial_ava_rejected += 1
                    @warn "Trivially balanced complex $cid has negative-only activity" min_val max_val
                end
                @debug "Complex $cid classified as negative-only"
            else
                push!(unrestricted_complexes, i)
                if cid in trivially_balanced
                    n_trivial_ava_rejected += 1
                    @warn "Trivially balanced complex $cid has unrestricted activity" min_val max_val
                end
                @debug "Complex $cid classified as unrestricted"
            end
        else
            # If no FVA range available, classify as unrestricted
            push!(unrestricted_complexes, i)
            if cid in trivially_balanced
                n_trivial_ava_rejected += 1
                @warn "No AVA result for trivially balanced complex $cid"
            end
            @debug "Complex $cid has no AVA result, classified as unrestricted"
        end
    end

    @debug "Complex classification complete" n_ava_balanced tolerance n_trivial_ava_confirmed n_trivial_ava_rejected

    @info "Complex classification" balanced = length(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = length(positive_complexes) negative = length(negative_complexes) unrestricted = length(unrestricted_complexes)

    # Step 3: Initialize concordance tracker
    concordance_tracker =
        if use_shared_arrays && nworkers() > 0
            SharedConcordanceTracker(complex_ids)
        else
            ConcordanceTracker(complex_ids)
        end

    # Add trivially concordant pairs
    for (c1_idx, c2_idx) in trivial_pairs
        tracker_idx1 = concordance_tracker.id_to_idx[complexes[c1_idx].id]
        tracker_idx2 = concordance_tracker.id_to_idx[complexes[c2_idx].id]
        union_sets!(concordance_tracker, tracker_idx1, tracker_idx2)
    end

    # Step 4: Generate candidate pairs using streaming correlation
    @info "Generating candidate pairs via streaming correlation"

    # Pass the pre-computed warmup points directly
    correlation_time = @elapsed candidate_priorities = streaming_filter(
        complexes,
        balanced_complexes,
        positive_complexes,
        negative_complexes,
        unrestricted_complexes,
        trivial_pairs,
        warmup,
        constraints;
        tolerance=tolerance,
        correlation_threshold=correlation_threshold,
        sample_size=sample_size,
        min_valid_samples=min_valid_samples,
        max_correlation_pairs=max_correlation_pairs,
        early_correlation_threshold=early_correlation_threshold,
        workers=workers,
        seed=seed,
        # Filtering parameters
        filter=filter,
        cv_threshold=cv_threshold,
        cv_epsilon=cv_epsilon,
    )

    @info "Candidate pairs identified" n_pairs = length(candidate_priorities) correlation_time_sec = round(correlation_time, digits=2)

    # Step 5: Process in stages with transitivity
    @info "Processing concordance tests in stages"

    concordance_time = @elapsed stage_results = process_in_stages(
        constraints,
        complexes,
        candidate_priorities,
        A_matrix,
        concordance_tracker;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        stage_size=effective_stage_size,
        batch_size=effective_batch_size,
        tolerance,
    )

    # Step 6: Build concordance modules
    @info "Building concordance modules" concordance_time_sec = round(concordance_time, digits=2)

    modules = extract_modules(concordance_tracker, balanced_complexes)

    # Step 7: Prepare results
    complexes_df = DataFrame(
        :complex_id => [c.id for c in complexes],
        :n_metabolites => [length(c.metabolite_indices) for c in complexes],
        :is_balanced => [c.id in balanced_complexes for c in complexes],
        :is_trivially_balanced => [c.id in trivially_balanced for c in complexes],
        :module => [get_module_id(c.id, modules) for c in complexes],
    )

    # Add activity ranges - verify trivially balanced complexes through AVA
    if !isempty(complex_ranges) || !isempty(trivially_balanced)
        # Pre-allocate vectors with the correct length
        min_activities = Vector{Float64}(undef, length(complexes))
        max_activities = Vector{Float64}(undef, length(complexes))
        ava_confirms = Vector{Bool}(undef, length(complexes))

        n_trivial_confirmed = 0
        n_trivial_contradicted = 0

        for (i, c) in enumerate(complexes)
            if haskey(complex_ranges, c.id)
                min_act, max_act = complex_ranges[c.id]
                min_activities[i] = min_act
                max_activities[i] = max_act

                # Check if AVA confirms trivially balanced classification
                if c.id in trivially_balanced
                    is_ava_balanced = abs(min_act) < tolerance && abs(max_act) < tolerance
                    ava_confirms[i] = is_ava_balanced
                    if is_ava_balanced
                        n_trivial_confirmed += 1
                    else
                        n_trivial_contradicted += 1
                        @warn "Trivially balanced complex has non-zero AVA range" complex_id = c.id min_activity = min_act max_activity = max_act
                    end
                else
                    ava_confirms[i] = true  # Not claimed to be trivially balanced
                end
            elseif c.id in trivially_balanced
                # Trivially balanced but no AVA result - this might indicate an issue
                @warn "No AVA result for trivially balanced complex" complex_id = c.id
                min_activities[i] = NaN
                max_activities[i] = NaN
                ava_confirms[i] = false
                n_trivial_contradicted += 1
            else
                # No AVA result available and not trivially balanced
                min_activities[i] = NaN
                max_activities[i] = NaN
                ava_confirms[i] = true
            end
        end

        # Add all columns at once
        complexes_df.min_activity = min_activities
        complexes_df.max_activity = max_activities
        complexes_df.ava_confirms_balanced = ava_confirms

        @debug "Trivially balanced verification" n_trivial_confirmed n_trivial_contradicted n_total_trivial = length(trivially_balanced)
    end

    modules_df = DataFrame(
        module_id=collect(String.(keys(modules))),
        size=[length(m) for m in values(modules)],
        complexes=[join(String.(m), ", ") for m in values(modules)],
    )

    lambda_df =
        DataFrame(c1_idx=Int[], c2_idx=Int[], direction=Symbol[], lambda=Float64[])

    for ((c1_idx, c2_idx, direction), lambda) in stage_results["optimization_results"]
        push!(lambda_df, (c1_idx, c2_idx, direction, lambda))
    end

    elapsed = time() - start_time

    stats = Dict(
        "n_complexes" => n_complexes,
        "n_balanced" => length(balanced_complexes),
        "n_trivially_balanced" => length(trivially_balanced),
        "n_trivial_pairs" => length(trivial_pairs),
        "n_candidate_pairs" => length(candidate_priorities),
        "n_concordant_pairs" => length(stage_results["concordant_pairs"]) + length(trivial_pairs),
        "n_non_concordant_pairs" => stage_results["non_concordant_pairs"],
        "n_skipped_transitivity" => stage_results["skipped_by_transitivity"],
        "n_modules" => length(modules),
        "stages_completed" => stage_results["stages_completed"],
        "elapsed_time" => elapsed,
    )

    @info "Concordance analysis complete" stats

    return (
        A=A_matrix,
        complexes=complexes_df,
        modules=modules_df,
        lambdas=lambda_df,
        stats=stats,
    )
end
# Define a custom output function to capture both activity and flux vectors
function ava_output_with_warmup(dir, om)
    if JuMP.termination_status(om) != JuMP.OPTIMAL
        @debug "AVA optimization not optimal" status = JuMP.termination_status(om)
        return nothing
    end

    objective_val = JuMP.objective_value(om)
    flux_vector = JuMP.value.(om[:x])
    # Correct for direction: dir=-1 for minimization, dir=1 for maximization
    activity = dir * objective_val

    @debug "AVA result" direction = dir objective_val activity flux_norm = norm(flux_vector)

    return (activity, flux_vector)
end
"""
Extract modules from concordance tracker.
"""
function extract_modules(tracker::Union{ConcordanceTracker,SharedConcordanceTracker}, balanced_complexes::Set{Symbol})
    # Get all disjoint sets
    groups = Dict{Int,Vector{Int}}()
    n = length(tracker.parent)

    for i in 1:n
        root = find_set!(tracker, i)
        if !haskey(groups, root)
            groups[root] = Int[]
        end
        push!(groups[root], i)
    end

    # Create modules
    modules = Dict{Symbol,Set{Symbol}}()

    # Add balanced module if exists
    if !isempty(balanced_complexes)
        modules[:balanced] = balanced_complexes
    end

    # Add other modules
    module_idx = 1
    for (root, members) in groups
        if length(members) > 1
            complex_ids = Set(tracker.idx_to_id[i] for i in members)

            # Skip if subset of balanced
            if !isempty(balanced_complexes) && issubset(complex_ids, balanced_complexes)
                continue
            end

            module_id = Symbol("module_$module_idx")
            modules[module_id] = complex_ids
            module_idx += 1
        end
    end

    return modules
end

"""
Get module ID for a complex.
"""
function get_module_id(complex_id::Symbol, modules::Dict{Symbol,Set{Symbol}})
    for (mid, members) in modules
        if complex_id in members
            return String(mid)
        end
    end
    return "none"
end