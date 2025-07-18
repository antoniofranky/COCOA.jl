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
import COBREXA: worker_local_data, get_worker_local_data
using DataFrames
import Distributed as D
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
    complexes::Dict{Symbol,MetabolicComplex},
    candidate_priorities::Vector{PairPriority},
    concordance_tracker::ConcordanceTracker;
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
                constraints, batch_pairs, concordance_tracker;
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
            if is_concordant
                union_sets!(concordance_tracker, c1_idx, c2_idx)
                push!(stage_results["concordant_pairs"], (c1_idx, c2_idx))
                push!(newly_concordant, (concordance_tracker.idx_to_id[c1_idx], concordance_tracker.idx_to_id[c2_idx]))
                stage_concordant_count += 1

                if !isnothing(lambda)
                    stage_results["optimization_results"][(c1_idx, c2_idx, direction)] = lambda
                end
            else
                add_non_concordant!(concordance_tracker, c1_idx, c2_idx)
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
                concordance_tracker
            )

            # Count prioritized pairs (those involving concordant complexes)
            for (c1_idx, c2_idx, _) in remaining_pairs
                if (concordance_tracker.idx_to_id[c1_idx] in all_concordant_complexes) || (concordance_tracker.idx_to_id[c2_idx] in all_concordant_complexes)
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
    concordance_tracker::ConcordanceTracker
)
    filtered_pairs = Tuple{Int,Int,Set{Symbol}}[]
    skipped_count = 0

    for (c1_idx, c2_idx, directions) in remaining_pairs

        # Skip if already concordant
        if are_concordant(concordance_tracker, c1_idx, c2_idx)
            skipped_count += 1
            continue
        end

        # Skip if known non-concordant
        if is_non_concordant(concordance_tracker, c1_idx, c2_idx)
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
    concordance_tracker::ConcordanceTracker)
    if isempty(concordant_complexes)
        return remaining_pairs
    end

    # Partition pairs: those involving concordant complexes vs others
    priority_pairs = Vector{Tuple{Int,Int,Set{Symbol}}}()
    regular_pairs = Vector{Tuple{Int,Int,Set{Symbol}}}()

    for (c1_idx, c2_idx, directions) in remaining_pairs
        c1_id = concordance_tracker.idx_to_id[c1_idx]
        c2_id = concordance_tracker.idx_to_id[c2_idx]

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
    concordance_tracker::ConcordanceTracker
)
    if isempty(newly_concordant)
        return remaining_pairs
    end

    # Filter out pairs that are now transitively concordant or non-concordant
    filtered_pairs = filter(remaining_pairs) do (c1_idx, c2_idx, directions)
        # Keep pair if it's not transitively determined
        !are_concordant(concordance_tracker, c1_idx, c2_idx) &&
            !is_non_concordant(concordance_tracker, c1_idx, c2_idx)
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
function precompute_reaction_indices(expanded_pairs, activity_lookup, idx_to_complex)
    reaction_indices_cache = Dict{Tuple{Int,Int},Set{Int}}()

    for (c1_idx, c2_idx, _) in expanded_pairs
        key = (c1_idx, c2_idx)
        if !haskey(reaction_indices_cache, key)
            c1_activity = activity_lookup[idx_to_complex[c1_idx].id]
            c2_activity = activity_lookup[idx_to_complex[c2_idx].id]
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

Process a batch of concordance tests using ConstraintTrees templated approach with parallel batch processing.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    batch_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    tolerance::Float64=1e-12
)
    # Convert indices to IDs before passing to workers (to avoid passing tracker to workers)
    batch_pairs_with_ids = [(concordance_tracker.idx_to_id[c1_idx], concordance_tracker.idx_to_id[c2_idx], directions) for (c1_idx, c2_idx, directions) in batch_pairs]

    # Process ALL concordance tests in parallel using the dual model approach
    batch_results = screen_dual_optimization_model(
        constraints,
        batch_pairs_with_ids;
        optimizer=optimizer,
        settings=settings,
        workers=workers
    ) do models_cache, (c1_id, c2_id, directions)
        # Process each direction for this pair
        pair_results = []

        for direction in directions
            # Select the appropriate model based on direction
            om = direction == :positive ? models_cache.positive : models_cache.negative

            # Set the c_j normalization constraint
            target_value = direction == :positive ? 1.0 : -1.0

            # Add the c_j constraint to the model
            c2_expr = C.substitute(constraints.activities[c2_id].value, om[:x])
            c2_constraint = @constraint(om, c2_expr == target_value)

            # Run optimization for both min and max
            results = []
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                @objective(om, sense, C.substitute(constraints.activities[c1_id].value, om[:x]))
                optimize!(om)

                if termination_status(om) == OPTIMAL
                    push!(results, objective_value(om))
                else
                    push!(results, nothing)
                end
            end

            # Clean up the c_j constraint for next use
            delete(om, c2_constraint)

            # Store result for this direction
            push!(pair_results, (direction, results))
        end

        return pair_results  # [(direction, [min_val, max_val]), ...]
    end

    # Process batch results back to individual concordance test results
    results = []

    # Process results for each pair
    for (i, pair_results) in enumerate(batch_results)
        c1_idx, c2_idx, directions = batch_pairs[i]  # Original indices

        # Process each direction result
        for (direction, test_results) in pair_results
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

    # Sort pairs by indices to ensure deterministic iteration order
    sorted_pairs = sort(collect(pair_results), by=x -> x[1])

    for ((c1_idx, c2_idx), dir_results) in sorted_pairs
        all_concordant = all(r[1] for r in values(dir_results))
        # Get lambda from any concordant direction (deterministic ordering)
        lambda = nothing
        for direction in sort(collect(keys(dir_results)))  # Sort directions for determinism
            is_conc, lam = dir_results[direction]
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
- `min_size_for_sharing=1_000_000`: Minimum array size for shared memory
- `min_valid_samples=30`: Minimum samples required for correlation
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
    workers=D.workers(),
    tolerance::Float64=1e-12,
    correlation_threshold::Float64=0.99,
    early_correlation_threshold::Float64=0.95,
    sample_size::Int=100,
    stage_size::Int=500,
    batch_size::Int=100,
    min_size_for_sharing::Int=1_000_000,
    min_valid_samples::Int=30,
    seed::Union{Int,Nothing}=42,
    use_unidirectional_constraints::Bool=true,
    # Filtering parameters
    filter::Union{Symbol,Vector{Symbol}}=[:cv, :cor],
    cv_threshold::Float64=0.01,
    cv_epsilon::Float64=1e-12,
)
    start_time = time()

    # Normalize filter parameter to vector
    filter_vec = isa(filter, Symbol) ? [filter] : filter

    # Use COBREXA's efficient model conversion
    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel for optimal performance"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    @info "Starting concordance analysis" n_workers = length(workers) tolerance early_correlation_threshold correlation_threshold sample_size use_unidirectional_constraints batch_size stage_size

    # Build constraints and extract complexes (includes Charnes-Cooper templates)
    constraints, complexes =
        concordance_constraints(model; modifications, use_unidirectional_constraints, min_size_for_sharing, return_complexes=true)

    # Extract complex information - complexes is now a Dict{Symbol,MetabolicComplex}
    n_complexes = length(complexes)
    complex_ids = [c.id for c in values(complexes)]

    n_reactions = length(AbstractFBCModels.reactions(model))

    @info "Model statistics" n_complexes n_reactions

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
    @debug "Validating constraint structure before AVA" n_complex_constraints = length(constraints.activities)

    # Sample a few complexes to check their structure
    if haskey(constraints, :activities)
        sample_count = min(3, length(constraints.activities))
        sample_complexes = collect(Iterators.take(constraints.activities, sample_count))
        for (cid, constraint) in sample_complexes
            @debug "Sample complex constraint" complex_id = cid constraint_type = typeof(constraint.value) n_terms = length(constraint.value.idxs)
        end
    else
        @warn "Complex activities not found in constraint tree" has_activities = haskey(constraints, :activities)
    end

    # Run FVA on all complexes using optimized settings
    ava_time = @elapsed ava_results = COBREXA.constraints_variability(
        constraints.balance,
        constraints.activities;
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

    for (i, (cid, c)) in enumerate(complexes)
        # cid is the Symbol key, c is the MetabolicComplex value

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
    concordance_tracker = ConcordanceTracker(complex_ids)

    # Add trivially concordant pairs
    for (c1_id, c2_id) in trivial_pairs
        union_sets!(concordance_tracker, concordance_tracker.id_to_idx[c1_id], concordance_tracker.id_to_idx[c2_id])
    end

    # Step 4: Generate candidate pairs using streaming correlation
    @info "Generating candidate pairs via streaming correlation"

    # Convert complexes dict to vector for streaming_filter
    # Create complexes vector in the same order as concordance tracker
    complexes_vector = [complexes[id] for id in concordance_tracker.idx_to_id]

    # Convert trivial_pairs from Symbol pairs to index pairs using concordance tracker
    trivial_pairs_indices = Set{Tuple{Int,Int}}()
    for (c1_id, c2_id) in trivial_pairs
        if haskey(concordance_tracker.id_to_idx, c1_id) && haskey(concordance_tracker.id_to_idx, c2_id)
            c1_idx = concordance_tracker.id_to_idx[c1_id]
            c2_idx = concordance_tracker.id_to_idx[c2_id]
            canonical_pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            push!(trivial_pairs_indices, canonical_pair)
        end
    end

    # Pass the pre-computed warmup points directly
    correlation_time = @elapsed candidate_priorities = streaming_filter(
        complexes_vector,
        balanced_complexes,
        positive_complexes,
        negative_complexes,
        unrestricted_complexes,
        trivial_pairs_indices,
        warmup,
        constraints,
        concordance_tracker;
        tolerance=tolerance,
        correlation_threshold=correlation_threshold,
        sample_size=sample_size,
        min_valid_samples=min_valid_samples,
        early_correlation_threshold=early_correlation_threshold,
        workers=workers,
        seed=seed,
        # Filtering parameters
        filter=filter_vec,
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
        concordance_tracker;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        stage_size=stage_size,
        batch_size=batch_size,
        tolerance,
    )

    # Step 6: Build concordance modules
    @info "Building concordance modules" concordance_time_sec = round(concordance_time, digits=2)

    modules = extract_modules(concordance_tracker, balanced_complexes)

    # Step 7: Prepare results
    complexes_df = DataFrame(
        :complex_id => [c.id for c in values(complexes)],
        :n_metabolites => [length(c.metabolites) for c in values(complexes)],
        :is_balanced => [c.id in balanced_complexes for c in values(complexes)],
        :is_trivially_balanced => [c.id in trivially_balanced for c in values(complexes)],
        :module => [get_module_id(c.id, modules) for c in values(complexes)],
    )

    # Add activity ranges - verify trivially balanced complexes through AVA
    if !isempty(complex_ranges) || !isempty(trivially_balanced)
        # Pre-allocate vectors with the correct length
        min_activities = Vector{Float64}(undef, length(complexes))
        max_activities = Vector{Float64}(undef, length(complexes))
        ava_confirms = Vector{Bool}(undef, length(complexes))

        n_trivial_confirmed = 0
        n_trivial_contradicted = 0

        for (i, c) in enumerate(values(complexes))
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
function extract_modules(tracker::ConcordanceTracker, balanced_complexes::Set{Symbol})
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

"""
Find trivially balanced complexes (containing metabolites that appear in only one complex).
"""
function find_trivially_balanced_complexes(
    complexes::Dict{Symbol,MetabolicComplex}
)::Set{Symbol}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    for (complex_id, complex) in complexes
        for (met_id, _) in complex.metabolites
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    balanced_complexes = Set{Symbol}()

    # Find metabolites that appear in only one complex
    for (met_id, complex_ids) in metabolite_participation
        if length(complex_ids) == 1
            complex_id = complex_ids[1]
            push!(balanced_complexes, complex_id)
        end
    end

    return balanced_complexes
end

"""
Find trivially concordant complexes based on shared metabolites.
Two complexes are trivially concordant if they share a metabolite that 
appears in exactly those two complexes.
"""
function find_trivially_concordant_pairs(complexes::Dict{Symbol,MetabolicComplex})::Set{Tuple{Symbol,Symbol}}
    # Build metabolite participation mapping
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    for (complex_id, complex) in complexes
        for (met_id, _) in complex.metabolites
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    concordant_pairs = Set{Tuple{Symbol,Symbol}}()

    # Find metabolites that appear in exactly two complexes
    for (met_id, complex_ids) in metabolite_participation
        if length(complex_ids) == 2
            c1, c2 = complex_ids
            pair = c1 < c2 ? (c1, c2) : (c2, c1)
            push!(concordant_pairs, pair)
        end
    end

    return concordant_pairs
end

"""
$(TYPEDSIGNATURES)

Execute a function with arguments from arrays `args` on `workers`, with dual pre-cached
JuMP optimization models created for positive and negative directions using Charnes-Cooper
transformation.

This function is optimized for concordance analysis where the same constraint structure
is used repeatedly with different objectives and c_j normalization constraints. It 
maintains two cached models per worker to avoid repeated constraint tree instantiation.

The function `f` takes the models cache and arguments. It should select the appropriate
model based on the test requirements and modify only the objective and c_j constraint.

# Arguments
- `f`: Function to execute, signature: `f(models_cache, args...)`
- `base_constraints`: Base constraint tree before Charnes-Cooper transformation
- `args...`: Arguments to pass to `f`

# Keyword Arguments
- `optimizer`: Optimization solver
- `settings`: Solver settings
- `workers`: Worker processes
"""
function screen_dual_optimization_model(
    f,
    base_constraints::C.ConstraintTree,
    args...;
    optimizer,
    settings=[],
    workers=D.workers(),
)
    # Extract pre-constructed templates from the constraint tree
    pos_constraints = base_constraints.charnes_cooper.positive
    neg_constraints = base_constraints.charnes_cooper.negative

    # Create dual model cache
    worker_cache = COBREXA.worker_local_data(
        constraints_tuple -> begin
            pos_const, neg_const = constraints_tuple
            # Create optimization models for both directions
            pos_model = COBREXA.optimization_model(pos_const; optimizer=optimizer)
            neg_model = COBREXA.optimization_model(neg_const; optimizer=optimizer)

            # Apply settings to both models
            for s in [COBREXA.configuration.default_solver_settings; settings]
                s(pos_model)
                s(neg_model)
            end

            return (positive=pos_model, negative=neg_model)
        end,
        (pos_constraints, neg_constraints)
    )

    D.pmap(
        (as...) -> f(COBREXA.get_worker_local_data(worker_cache), as...),
        D.CachingPool(workers),
        args...,
    )
end
