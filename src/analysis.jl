"""
Main analysis algorithms for COCOA.

This module contains:
- Batch processing with transitivity optimization
- Concordance analysis using unified batch_size parameter
- Main concordance analysis function
- Module extraction and result formatting
"""

import COBREXA: worker_local_data, get_worker_local_data
import Base.Iterators

"""
Extract numerical tolerance from JuMP optimizer for consistent thresholding.
Creates a temporary model to query the optimizer's tolerance settings.
"""
function extract_solver_tolerance(optimizer, settings=[])::Float64
    default_tolerance = 1e-6  # Conservative fallback

    try
        # Create a minimal model to query optimizer attributes
        temp_model = JuMP.Model(optimizer)

        # Apply settings to get the actual configured tolerances
        for setting in [COBREXA.configuration.default_solver_settings; settings]
            setting(temp_model)
        end

        # Try common tolerance attribute names across different solvers
        tolerance_attrs = [
            "primal_feasibility_tolerance",  # HiGHS, Gurobi
            "dual_feasibility_tolerance",    # HiGHS, Gurobi
            "feasibility_tolerance",         # Some solvers
            "FeasibilityTol",               # Gurobi
            "OptimalityTol",                # Gurobi
            "primal_tolerance",             # CPLEX-style
            "dual_tolerance"                # CPLEX-style
        ]

        detected_tolerances = Float64[]

        for attr in tolerance_attrs
            try
                tol = JuMP.get_optimizer_attribute(temp_model, attr)
                if isa(tol, Real) && tol > 0 && tol < 1e-3
                    push!(detected_tolerances, Float64(tol))
                end
            catch
                # Attribute not supported by this solver, continue
            end
        end

        # Return the most restrictive (smallest) tolerance found
        if !isempty(detected_tolerances)
            return minimum(detected_tolerances)
        end

    catch e
        @debug "Could not extract solver tolerance" exception = e
    end

    return default_tolerance
end

# Helper function for iterating directions from bit flags
@inline function iterate_directions(bits::UInt8)
    directions = Symbol[]
    (bits & 0x01) != 0 && push!(directions, :positive)
    (bits & 0x02) != 0 && push!(directions, :negative)
    return directions
end

"""
$(TYPEDSIGNATURES)

Process concordance analysis in batches.
When concordant pairs are found, remaining pairs involving those complexes are prioritized.
This exploits the clustering tendency of concordant complexes to dramatically improve efficiency.

When `use_transitivity=false`, transitivity filtering and elimination are disabled,
which forces testing of all candidate pairs regardless of already known concordant relationships.
Trivially concordant pairs are still automatically recognized and don't need explicit testing.
"""
function process_in_batches(
    constraints::C.ConstraintTree,
    candidate_priorities::Vector{PairPriority},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    batch_size::Int=50,
    optimization_tolerance::Float64=1e-6,
    concordance_tolerance::Float64=1e-4,
    use_transitivity::Bool=true
)
    # Use separate variables - following Julia performance best practices
    batches_completed = 0
    pairs_processed = 0
    concordant_pairs = Set{Tuple{Int,Int}}()
    non_concordant_pairs = 0
    skipped_by_transitivity = 0
    transitive_pairs = 0
    timeout_pairs = 0
    optimization_results = Dict{Tuple{Int,Int,Symbol},Float64}()

    sorted_pairs = sort(candidate_priorities,
        by=p -> (p.cv, p.c1_idx, p.c2_idx))

    remaining_pairs = [(p.c1_idx, p.c2_idx, p.directions_bits) for p in sorted_pairs]

    batch_count = 0
    total_pairs = length(remaining_pairs)
    processed_pairs = 0
    all_concordant_complexes = Set{Symbol}()

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
        batch_count += 1

        if isa(concordance_tracker, ConcordanceTracker)
            clear_module_cache!(concordance_tracker)
        end

        @debug "Starting batch $(batch_count)" remaining = length(remaining_pairs)

        if use_transitivity
            filtered_pairs, skipped_count = filter_transitive_pairs(
                remaining_pairs,
                concordance_tracker
            )
            skipped_by_transitivity += skipped_count
        else
            filtered_pairs = remaining_pairs
            skipped_count = 0
        end

        if isempty(filtered_pairs)
            break
        end

        batch_size_actual = min(batch_size, length(filtered_pairs))
        batch_pairs = filtered_pairs[1:batch_size_actual]
        remaining_pairs = filtered_pairs[(batch_size_actual+1):end]

        current_batch_results = try
            process_concordance_batch(
                constraints, batch_pairs, concordance_tracker;
                optimizer=optimizer,
                settings=settings,
                workers=workers,
                optimization_tolerance=optimization_tolerance,
                concordance_tolerance=concordance_tolerance
            )
        catch e
            @warn "Error processing batch $(batch_count)" error = string(e)
            continue
        end

        processed_pairs += length(batch_pairs)

        newly_concordant = Vector{Tuple{Symbol,Symbol}}()
        batch_concordant_count = 0
        batch_prioritized_count = 0
        pairs_eliminated = 0

        # Cache idx_to_id for better performance
        idx_to_id = concordance_tracker.idx_to_id

        # Track processed complexes for progress monitoring
        ensure_mask_allocated!(concordance_tracker, :processed)
        processed_mask = get_mask(concordance_tracker, :processed)

        for result in current_batch_results
            c1_idx, c2_idx, direction, is_concordant, lambda, has_timeout = result

            # Mark both complexes as processed
            processed_mask[c1_idx] = true
            processed_mask[c2_idx] = true

            if has_timeout
                timeout_pairs += 1
            end

            if is_concordant
                union_sets!(concordance_tracker, c1_idx, c2_idx)
                push!(concordant_pairs, (c1_idx, c2_idx))
                push!(newly_concordant, (idx_to_id[c1_idx], idx_to_id[c2_idx]))
                batch_concordant_count += 1

                if !isnothing(lambda)
                    optimization_results[(c1_idx, c2_idx, direction)] = lambda
                end
            else
                add_non_concordant!(concordance_tracker, c1_idx, c2_idx)
                non_concordant_pairs += 1
            end
        end

        concordant_count = length(concordant_pairs)

        if batch_concordant_count > 0 && use_transitivity
            prev_num_pairs = length(remaining_pairs)
            remaining_pairs, transitive_count = apply_transitivity_elimination(
                remaining_pairs,
                newly_concordant,
                concordance_tracker
            )
            pairs_eliminated = prev_num_pairs - length(remaining_pairs)
            eliminated_count += pairs_eliminated
            transitive_pairs += transitive_count
        end

        if batch_concordant_count >= 1
            for (c1_id, c2_id) in newly_concordant
                push!(all_concordant_complexes, c1_id, c2_id)
            end

            remaining_pairs = reprioritize_by_concordant_complexes(
                remaining_pairs,
                all_concordant_complexes,
                concordance_tracker
            )

            for (c1_idx, c2_idx, _) in remaining_pairs
                if (idx_to_id[c1_idx] in all_concordant_complexes) || (idx_to_id[c2_idx] in all_concordant_complexes)
                    batch_prioritized_count += 1
                end
            end
            prioritized_count = batch_prioritized_count
        end

        pairs_processed += length(batch_pairs)
        batches_completed = batch_count

        @debug "Batch $(batch_count) complete" new_concordant = batch_concordant_count

        ProgressMeter.update!(prog, processed_pairs;
            showvalues=[
                (:batch, batch_count),
                (:computed, concordant_count),
                (:transitive, transitive_pairs),
                (:eliminated, eliminated_count)
            ]
        )
    end

    ProgressMeter.finish!(prog)

    @info "Concordance testing complete" (
        total_batches=batch_count,
        total_concordant_pairs=length(concordant_pairs),
        total_concordant_complexes=length(all_concordant_complexes)
    )

    # Return results as a named tuple for type stability
    return (
        batches_completed=batches_completed,
        pairs_processed=pairs_processed,
        concordant_pairs=concordant_pairs,
        non_concordant_pairs=non_concordant_pairs,
        skipped_by_transitivity=skipped_by_transitivity,
        transitive_pairs=transitive_pairs,
        timeout_pairs=timeout_pairs,
        optimization_results=optimization_results
    )
end

"""
$(TYPEDSIGNATURES)

Filter out pairs that can be inferred by transitivity to avoid redundant testing.
"""
function filter_transitive_pairs(
    remaining_pairs::Vector{Tuple{Int,Int,UInt8}},
    concordance_tracker::ConcordanceTracker
)
    filtered_pairs = Tuple{Int,Int,UInt8}[]
    skipped_count = 0

    for (c1_idx, c2_idx, directions_bits) in remaining_pairs

        if are_concordant(concordance_tracker, c1_idx, c2_idx)
            skipped_count += 1
            continue
        end

        if is_non_concordant(concordance_tracker, c1_idx, c2_idx)
            skipped_count += 1
            continue
        end

        push!(filtered_pairs, (c1_idx, c2_idx, directions_bits))
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
    remaining_pairs::Vector{Tuple{Int,Int,UInt8}},
    concordant_complexes::Set{Symbol},
    concordance_tracker::ConcordanceTracker)
    if isempty(concordant_complexes)
        return remaining_pairs
    end

    # Pre-allocate with size hints for better performance
    n_pairs = length(remaining_pairs)
    priority_pairs = Vector{Tuple{Int,Int,UInt8}}()
    regular_pairs = Vector{Tuple{Int,Int,UInt8}}()
    sizehint!(priority_pairs, n_pairs ÷ 2)
    sizehint!(regular_pairs, n_pairs ÷ 2)

    # Cache the idx_to_id lookups to avoid repeated dictionary access
    idx_to_id = concordance_tracker.idx_to_id

    for (c1_idx, c2_idx, directions_bits) in remaining_pairs
        c1_id = idx_to_id[c1_idx]
        c2_id = idx_to_id[c2_idx]

        if c1_id in concordant_complexes || c2_id in concordant_complexes
            push!(priority_pairs, (c1_idx, c2_idx, directions_bits))
        else
            push!(regular_pairs, (c1_idx, c2_idx, directions_bits))
        end
    end

    sort!(priority_pairs, by=p -> (p[1], p[2]))

    @debug "Reprioritization effect" (
        priority_pairs=length(priority_pairs),
        regular_pairs=length(regular_pairs),
        priority_percentage=round(100 * length(priority_pairs) / length(remaining_pairs), digits=1)
    )

    return vcat(priority_pairs, regular_pairs)
end

"""
$(TYPEDSIGNATURES)

Apply transitivity elimination to remove pairs that are guaranteed to be concordant
based on already discovered concordant pairs.
"""
function apply_transitivity_elimination(
    remaining_pairs::Vector{Tuple{Int,Int,UInt8}},
    newly_concordant::Vector{Tuple{Symbol,Symbol}},
    concordance_tracker::ConcordanceTracker
)
    if isempty(newly_concordant)
        return remaining_pairs, 0
    end

    # Count pairs that are now concordant through transitivity
    transitive_concordant_count = 0
    filtered_pairs = filter(remaining_pairs) do (c1_idx, c2_idx, directions_bits)
        if are_concordant(concordance_tracker, c1_idx, c2_idx)
            transitive_concordant_count += 1
            return false
        elseif is_non_concordant(concordance_tracker, c1_idx, c2_idx)
            return false
        else
            return true
        end
    end

    eliminated_count = length(remaining_pairs) - length(filtered_pairs)

    if eliminated_count > 0
        @debug "Transitivity elimination" (
            pairs_eliminated=eliminated_count,
            transitive_concordant=transitive_concordant_count,
            remaining_pairs=length(filtered_pairs)
        )
    end

    return filtered_pairs, transitive_concordant_count
end

"""
$(TYPEDSIGNATURES)

Helper function to process optimization results into the expected format.
"""
# Define concrete types for better type stability
const OptValue = Union{Float64,Nothing}
const TestResult = Tuple{OptValue,OptValue,Bool}  # (min_val, max_val, timeout)
const DirectionResult = Tuple{Symbol,TestResult}  # (direction, test_result)

function process_optimization_results(optimization_results, batch_pairs, concordance_tracker)
    # Group results by (c1_id, c2_id, direction) - use concrete types
    result_dict = Dict{Tuple{Symbol,Symbol,Symbol},Dict{Int,Tuple{OptValue,Bool}}}()

    for result in optimization_results
        c1_id, c2_id, direction, dir_multiplier, actual_value, timeout = result
        key = (c1_id, c2_id, direction)
        if !haskey(result_dict, key)
            result_dict[key] = Dict{Int,Tuple{OptValue,Bool}}()
        end
        result_dict[key][dir_multiplier] = (actual_value, timeout)
    end

    # Convert to pair format - use concrete types
    pair_results_dict = Dict{Tuple{Symbol,Symbol},Vector{DirectionResult}}()

    for ((c1_id, c2_id, direction), minmax_results) in result_dict
        key = (c1_id, c2_id)
        if !haskey(pair_results_dict, key)
            pair_results_dict[key] = DirectionResult[]
        end

        min_result = get(minmax_results, -1, (nothing, false))
        max_result = get(minmax_results, +1, (nothing, false))
        min_val, min_timeout = min_result
        max_val, max_timeout = max_result

        if min_timeout || max_timeout
            @warn "Solver timeout" c1_id c2_id direction
        end

        push!(pair_results_dict[key], (direction, (min_val, max_val, min_timeout || max_timeout)))
    end

    # Pre-allocate results array
    batch_results = Vector{Vector{DirectionResult}}(undef, length(batch_pairs))

    # Cache idx_to_id for better performance
    idx_to_id = concordance_tracker.idx_to_id

    for (i, (c1_idx, c2_idx, _)) in enumerate(batch_pairs)
        c1_id, c2_id = idx_to_id[c1_idx], idx_to_id[c2_idx]
        batch_results[i] = get(pair_results_dict, (c1_id, c2_id), DirectionResult[])
    end

    return batch_results
end

"""
$(TYPEDSIGNATURES)

Process a batch of concordance tests using ConstraintTrees templated approach with parallel batch processing.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    batch_pairs::Vector{Tuple{Int,Int,UInt8}},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    optimization_tolerance::Float64=1e-6,
    concordance_tolerance::Float64=1e-4
)
    # Create COBREXA-style test array with direction multipliers
    test_array = []
    for (c1_idx, c2_idx, directions_bits) in batch_pairs
        c1_id, c2_id = concordance_tracker.idx_to_id[c1_idx], concordance_tracker.idx_to_id[c2_idx]
        # Iterate over directions using bit flags
        for direction in iterate_directions(directions_bits)
            # Add both MIN (-1) and MAX (+1) tests
            push!(test_array, (c1_id, c2_id, direction, -1))  # MIN test
            push!(test_array, (c1_id, c2_id, direction, +1))  # MAX test
        end
    end

    # Process using pure COBREXA pattern
    optimization_results = screen_directions_optimization_model(
        (models_cache, (c1_id, c2_id, direction, dir_multiplier)) -> begin
            om = direction == :positive ? models_cache.positive : models_cache.negative

            try
                # Set c2 constraint to 1.0 and optimize c1
                c2_constraint = @constraint(om, c2_constraint,
                    C.substitute(constraints.activities[c2_id].value, om[:x]) == 1.0)

                @objective(om, JuMP.MAX_SENSE,
                    C.substitute(dir_multiplier * constraints.activities[c1_id].value, om[:x]))

                optimize!(om)

                # Get result
                raw_value = if termination_status(om) in (OPTIMAL, LOCALLY_SOLVED) && is_solved_and_feasible(om)
                    objective_value(om)
                else
                    nothing
                end

                # Convert back: if we maximized -f(x), negate result to get min f(x)
                actual_value = dir_multiplier == -1 ? (raw_value !== nothing ? -raw_value : nothing) : raw_value

                # Cleanup
                delete(om, c2_constraint)
                JuMP.unregister(om, :c2_constraint)

                return (c1_id, c2_id, direction, dir_multiplier, actual_value, termination_status(om) == TIME_LIMIT)
            catch e
                @warn "Optimization error" c1_id c2_id direction error = string(e)
                return (c1_id, c2_id, direction, dir_multiplier, nothing, false)
            end
        end,
        constraints,
        test_array;
        optimizer=optimizer,
        settings=settings,
        workers=workers
    )

    # Process results into final format
    batch_results = process_optimization_results(optimization_results, batch_pairs, concordance_tracker)

    final_results = []
    for (i, pair_results_by_dir) in enumerate(batch_results)
        c1_idx, c2_idx, original_directions_bits = batch_pairs[i]

        all_concordant = true
        final_lambda = nothing
        concordant_directions = Set{Symbol}()
        has_timeout = false

        if isempty(pair_results_by_dir)
            all_concordant = false
        else
            for (direction, test_results) in pair_results_by_dir
                min_val = test_results[1]
                max_val = test_results[2]
                timeout_occurred = test_results[3]

                if timeout_occurred === true
                    has_timeout = true
                end

                if min_val === nothing || max_val === nothing || abs(min_val - max_val) > concordance_tolerance
                    all_concordant = false
                    break
                else
                    push!(concordant_directions, direction)
                    if isnothing(final_lambda)
                        final_lambda = min_val
                    end
                end
            end
        end

        final_direction = if all_concordant
            if length(concordant_directions) == 2
                :both
            elseif :positive in concordant_directions
                :positive
            elseif :negative in concordant_directions
                :negative
            else
                :both
            end
        else
            :both
        end

        push!(final_results, (c1_idx, c2_idx, final_direction, all_concordant, all_concordant ? final_lambda : nothing, has_timeout))
    end

    return final_results
end


"""
$(TYPEDSIGNATURES)

Main concordance analysis function optimized for large models and HPC execution.
"""
function concordance_analysis(
    model;
    modifications=Function[],
    optimizer,
    settings=[],
    workers=D.workers(),
    optimization_tolerance::Union{Float64,Nothing}=nothing,
    concordance_tolerance::Union{Float64,Nothing}=nothing,
    balanced_tolerance::Union{Float64,Nothing}=nothing,
    cv_threshold::Union{Float64,Nothing}=nothing,
    coarse_cv_threshold::Union{Float64,Nothing}=nothing,
    cv_epsilon::Union{Float64,Nothing}=1e-16,
    sample_size::Int=100,
    coarse_sample_size::Int=Int(sample_size ÷ 5),
    batch_size::Union{Int,Nothing}=nothing,
    min_valid_samples::Int=10,
    seed::Union{Int,Nothing}=nothing,
    use_unidirectional_constraints::Bool=true,
    chunk_size_filter::Int=100_000,
    max_pairs_in_memory::Int=500_000,
    objective_bound=nothing,
    use_transitivity::Bool=true,
    n_burnin::Int=50,
    n_chains::Int=1,
)
    start_time = time()

    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel for optimal performance"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    # Detect solver tolerance for consistent numerical thresholds
    solver_tolerance = extract_solver_tolerance(optimizer, settings)

    # Set default values based on solver tolerance if not provided
    actual_optimization_tolerance = optimization_tolerance !== nothing ? optimization_tolerance : max(solver_tolerance * 10, 1e-6)
    actual_concordance_tolerance = concordance_tolerance !== nothing ? concordance_tolerance : max(solver_tolerance * 100, 1e-4)
    actual_balanced_tolerance = balanced_tolerance !== nothing ? balanced_tolerance : solver_tolerance
    actual_cv_threshold = cv_threshold !== nothing ? cv_threshold : max(solver_tolerance * 100, 1e-2)
    actual_coarse_cv_threshold = coarse_cv_threshold !== nothing ? coarse_cv_threshold : cv_threshold * 10

    @info "Starting concordance analysis" n_workers = length(workers) optimization_tolerance = actual_optimization_tolerance concordance_tolerance = actual_concordance_tolerance balanced_tolerance = actual_balanced_tolerance coarse_cv_threshold = actual_coarse_cv_threshold cv_threshold = actual_cv_threshold sample_size use_unidirectional_constraints batch_size solver_tolerance

    constraints, complexes =
        concordance_constraints(model; modifications, use_unidirectional_constraints, return_complexes=true)

    # Add objective constraint if specified
    if objective_bound !== nothing
        # Validate objective_bound is callable
        if !isa(objective_bound, Function)
            throw(ArgumentError("objective_bound must be a function that takes optimal objective value and returns a constraint bound"))
        end

        # Get optimal objective value first
        objective_flux = COBREXA.optimized_values(
            constraints.balance;
            objective=constraints.balance.objective.value,
            output=constraints.balance.objective,
            optimizer,
            settings,
        )

        if objective_flux !== nothing
            @info "Objective flux determined" objective_flux
            # Add objective bound constraint to limit feasible space
            constraints.balance *= :objective_bound^C.Constraint(
                constraints.balance.objective.value,
                objective_bound(objective_flux)
            )
            constraints.charnes_cooper.positive *= :objective_bound^C.Constraint(
                constraints.charnes_cooper.positive.objective.value,
                objective_bound(objective_flux)
            )
            constraints.charnes_cooper.negative *= :objective_bound^C.Constraint(
                constraints.charnes_cooper.negative.objective.value,
                objective_bound(objective_flux)
            )
            @info "Added objective bound constraint" optimal_value = objective_flux bound_value = objective_bound(objective_flux)
        else
            @warn "Could not determine optimal objective value, skipping objective constraint"
        end
    end

    n_complexes = length(complexes)
    # Get sorted complex IDs for fully deterministic ConcordanceTracker initialization
    # Sort by string representation to ensure consistent ordering across platforms
    complex_ids = sort!(collect(keys(complexes)); by=string)
    n_reactions = length(AbstractFBCModels.reactions(model))
    @info "Model statistics" n_complexes n_reactions

    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(complexes)
    @info "Found trivially balanced complexes" n_trivivally_balanced = length(trivially_balanced)

    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(complexes)
    @info "Found trivially concordant pairs" n_trivially_concordant = length(trivial_pairs)

    # Set up RNG early for deterministic sampling
    rng = if seed === nothing
        Random.GLOBAL_RNG
    else
        StableRNG(seed::Int)
    end

    @info "Performing Activity Variability Analysis and generating warmup points"
    ava_time = @elapsed ava_results = COBREXA.constraints_variability(
        constraints.balance,
        constraints.activities;
        optimizer=optimizer,
        settings=settings,
        output=ava_output_with_warmup,
        output_type=Tuple{Float64,Vector{Float64}},
        workers=workers,
    )
    @info "AVA processing complete" ava_time_sec = round(ava_time, digits=2)

    concordance_tracker = ConcordanceTracker(complex_ids)

    # Memory-efficient: extract warmup matrix and activity ranges directly without intermediate storage
    n_complexes = length(concordance_tracker.idx_to_id)
    warmup_points = Vector{Vector{Float64}}()

    # Memory-efficient activity ranges: use NaN for missing values instead of Union types
    # NaN values indicate complexes without valid AVA results (more efficient than Union{Float64,Nothing})
    activity_ranges = Vector{Tuple{Float64,Float64}}(undef, n_complexes)
    sizehint!(warmup_points, n_complexes * 2)  # Estimate: 2 points per active complex

    for (i, cid) in enumerate(concordance_tracker.idx_to_id)
        if haskey(ava_results, cid)
            result = ava_results[cid]
            if result !== nothing && length(result) == 2
                min_res, max_res = result
                if min_res !== nothing && max_res !== nothing
                    min_activity, min_flux = min_res
                    max_activity, max_flux = max_res
                    activity_ranges[i] = (min_activity, max_activity)
                    push!(warmup_points, min_flux, max_flux)
                else
                    # Mark as inactive using NaN values
                    activity_ranges[i] = (NaN, NaN)
                end
            else
                # Mark as inactive using NaN values
                activity_ranges[i] = (NaN, NaN)
            end
        else
            # Mark as inactive using NaN values
            activity_ranges[i] = (NaN, NaN)
        end
    end

    warmup = if isempty(warmup_points)
        Matrix{Float64}(undef, 0, 0)
    else
        # Build matrix directly without transpose - more efficient
        n_points = length(warmup_points)
        n_vars = length(warmup_points[1])
        warmup_matrix = Matrix{Float64}(undef, n_points, n_vars)
        for (i, point) in enumerate(warmup_points)
            warmup_matrix[i, :] = point
        end
        warmup_matrix
    end

    # # Check feasibility of warmup points if objective bound is applied
    # if objective_bound !== nothing && !isempty(warmup_points) && haskey(constraints.balance, :objective_bound)
    #     @info "Checking feasibility of warmup points with objective bound"
    #     n_feasible = 0
    #     n_infeasible = 0

    #     obj_constraint = constraints.balance.objective_bound

    #     for (i, point) in enumerate(warmup_points)
    #         # Check if point satisfies the objective bound constraint
    #         obj_value = sum(obj_constraint.value.weights[j] * point[obj_constraint.value.idxs[j]] for j in 1:length(obj_constraint.value.idxs) if obj_constraint.value.idxs[j] <= length(point))

    #         bound = obj_constraint.bound
    #         is_feasible = if bound isa C.Between
    #             bound.lower <= obj_value <= bound.upper
    #         elseif bound isa C.EqualTo
    #             abs(obj_value - bound.equal_to) < 1e-9
    #         else
    #             true  # Unknown bound type, assume feasible
    #         end

    #         if is_feasible
    #             n_feasible += 1
    #         else
    #             n_infeasible += 1
    #             @debug "Infeasible warmup point" point_idx = i obj_value = obj_value bound = bound
    #         end
    #     end

    #     @info "Warmup point feasibility check" n_total = length(warmup_points) n_feasible = n_feasible n_infeasible = n_infeasible

    #     if n_infeasible > 0
    #         @warn "$(n_infeasible) warmup points are infeasible with objective bound - this may cause sampling issues"
    #     end
    # end

    # Use BitVectors consistently for optimal performance with large models (50K+ reactions)
    n_complexes = length(concordance_tracker.idx_to_id)

    # Initialize BitVector masks in the ConcordanceTracker for memory efficiency
    ensure_mask_allocated!(concordance_tracker, :balanced)
    ensure_mask_allocated!(concordance_tracker, :positive)
    ensure_mask_allocated!(concordance_tracker, :negative)
    ensure_mask_allocated!(concordance_tracker, :unrestricted)

    # Get references to the BitVector masks for efficient access
    balanced_complexes = get_mask(concordance_tracker, :balanced)
    positive_complexes = get_mask(concordance_tracker, :positive)
    negative_complexes = get_mask(concordance_tracker, :negative)
    unrestricted_complexes = get_mask(concordance_tracker, :unrestricted)

    # Use balanced tolerance for balanced complex detection - ensures consistency with optimization
    balanced_threshold = actual_balanced_tolerance

    # Build BitVector mask for trivially balanced complexes for O(1) lookups
    trivially_balanced_mask = falses(n_complexes)
    for (i, cid) in enumerate(concordance_tracker.idx_to_id)
        if cid in trivially_balanced
            trivially_balanced_mask[i] = true
        end
    end

    # Process complexes using direct indexing with optimized comparisons
    @inbounds for i in eachindex(activity_ranges)
        # Fast O(1) BitVector lookup instead of O(log n) Set lookup
        if trivially_balanced_mask[i]
            balanced_complexes[i] = true
            continue  # Skip AVA classification for trivially balanced complexes
        end

        min_val, max_val = activity_ranges[i]

        # Check if complex has valid AVA results (NaN indicates inactive)
        if isnan(min_val) || isnan(max_val)
            # Complex is inactive (no valid AVA results)
            unrestricted_complexes[i] = true
        else
            # Complex has valid activity range - classify based on activity bounds
            # Use direct comparison instead of rounding for better performance
            if abs(min_val) < balanced_threshold && abs(max_val) < balanced_threshold
                balanced_complexes[i] = true
            elseif min_val >= -balanced_threshold  # Effectively >= 0 with tolerance
                positive_complexes[i] = true
            elseif max_val <= balanced_threshold   # Effectively <= 0 with tolerance
                negative_complexes[i] = true
            else
                unrestricted_complexes[i] = true
            end
        end
    end

    @info "Complex classification" balanced = count(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = count(positive_complexes) negative = count(negative_complexes) unrestricted = count(unrestricted_complexes)


    for (c1_id, c2_id) in trivial_pairs
        union_sets!(concordance_tracker, concordance_tracker.id_to_idx[c1_id], concordance_tracker.id_to_idx[c2_id])
    end

    # CRITICAL: Union all balanced complexes into a single module
    # This includes both trivially balanced complexes AND AVA-detected balanced complexes
    # All balanced complexes have zero net flux and are therefore concordant with each other
    balanced_indices = findall(balanced_complexes)
    if length(balanced_indices) > 1
        @info "Unioning all balanced complexes into single module" n_balanced = length(balanced_indices) n_trivially_balanced = length(trivially_balanced)
        # Union all balanced complexes with the first balanced complex
        first_balanced_idx = balanced_indices[1]
        for i in eachindex(balanced_indices)[2:end]
            union_sets!(concordance_tracker, first_balanced_idx, balanced_indices[i])
        end
    end

    # Use tracker's idx_to_id directly as complexes_vector (it's already a Vector{Symbol})
    complexes_vector = concordance_tracker.idx_to_id

    # Process trivial pairs into indices set
    trivial_pairs_indices = Set{Tuple{Int,Int}}()
    sizehint!(trivial_pairs_indices, length(trivial_pairs))
    for (c1_id, c2_id) in trivial_pairs
        if haskey(concordance_tracker.id_to_idx, c1_id) && haskey(concordance_tracker.id_to_idx, c2_id)
            c1_idx = concordance_tracker.id_to_idx[c1_id]
            c2_idx = concordance_tracker.id_to_idx[c2_id]
            canonical_pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            push!(trivial_pairs_indices, canonical_pair)
        end
    end
    @info "Generating candidate pairs via coefficient of variance..."

    decimals = max(0, -floor(Int, log10(solver_tolerance)))
    aggregate = rows -> round.(vec(hcat(rows...)), digits=decimals)
    @info "Sampling schedule"

    # Sampling strategy for concordance analysis
    # Formula: n_samples_collected = n_chains × n_starting_points × n_iterations_collected

    # 1. Configure to produce exactly sample_size samples
    n_iterations_to_collect = 1  # Single sample per starting point
    n_starting_points = sample_size  # Exactly sample_size starting points for sample_size samples

    # 2. Minimal burn-in since each starting point is already diverse
    spacing = 1   # No spacing needed since each starting point is independent

    # 3. Calculate expected total
    expected_samples = n_chains * n_starting_points * n_iterations_to_collect

    @info "Sampling configuration" n_chains n_starting_points n_iterations_to_collect n_burnin spacing target = sample_size expected = expected_samples warmup_size = size(warmup, 1)

    # 4. Define single iteration to collect after burn-in
    iterations_to_collect = [n_burnin]  # Single well-converged sample per starting point

    # 5. Generate exactly sample_size starting points using stratified approach
    # Distribute across strategies to ensure comprehensive coverage
    n_extreme_points = min(sample_size ÷ 3, size(warmup, 1))  # 1/3 from extremes
    n_center_points = min(1, sample_size - n_extreme_points)  # 1 center point if space allows
    n_random_points = sample_size - n_extreme_points - n_center_points  # Rest as random combinations

    # Ensure we don't exceed sample_size
    total_planned = n_extreme_points + n_center_points + n_random_points
    if total_planned != sample_size
        @warn "Starting point allocation mismatch" planned = total_planned target = sample_size
        n_random_points = sample_size - n_extreme_points - n_center_points
    end

    # Pre-allocate start_variables_list with known size
    start_variables_list = Vector{Vector{Float64}}()
    sizehint!(start_variables_list, sample_size)

    # Strategy 1: Use extreme boundary points for maximum activity ranges
    if n_extreme_points > 0 && !isempty(warmup)
        # Select most diverse extreme points (max distance from each other)
        extreme_indices = rand(rng, 1:size(warmup, 1), n_extreme_points)
        for idx in extreme_indices
            push!(start_variables_list, warmup[idx, :])
        end
    end

    # Strategy 2: Use center point for balanced exploration
    if n_center_points > 0 && !isempty(warmup)
        # Create center point as average of all warmup points
        center_point = vec(mean(warmup, dims=1))
        push!(start_variables_list, center_point)
    end

    # Strategy 3: Generate diverse random points as convex combinations
    # Key insight: Use true randomness for interior exploration
    if n_random_points > 0 && size(warmup, 1) >= 2
        # Pre-allocate weights vector to reuse
        weights = Vector{Float64}(undef, 4)  # max n_base_points is 4

        for i in 1:n_random_points
            # Use varying numbers of base points for different exploration depths
            n_base_points = 2 + (i % 3)  # Alternate between 2, 3, 4 base points
            n_base_points = min(n_base_points, size(warmup, 1))

            # Select points for maximum diversity
            base_indices = rand(rng, 1:size(warmup, 1), n_base_points)

            # Generate weights for uniform interior exploration
            resize!(weights, n_base_points)
            rand!(rng, weights)
            weights ./= sum(weights)

            # Create convex combination - pre-allocate and reuse
            random_flux = zeros(size(warmup, 2))
            for (j, weight) in enumerate(weights)
                random_flux .+= weight .* warmup[base_indices[j], :]
            end

            push!(start_variables_list, random_flux)
        end
    end

    start_variables = if isempty(start_variables_list)
        warmup  # Fallback to all warmup points
    else
        # Use more efficient matrix construction with vectorized copying
        n_points = length(start_variables_list)
        n_vars = length(start_variables_list[1])

        # Pre-allocate matrix and use column-major filling for better cache performance
        start_matrix = Matrix{Float64}(undef, n_points, n_vars)

        # Vectorized copy for better performance
        @inbounds for (i, point) in enumerate(start_variables_list)
            # Use copyto! for vectorized copy instead of element-wise assignment
            copyto!(view(start_matrix, i, :), point)
        end
        start_matrix
    end

    @info "Starting point composition" extreme_points = n_extreme_points center_points = n_center_points random_combinations = n_random_points total = size(start_variables, 1)    # 6. Run the sampler with deterministic seeding for reproducible results
    # Use a fixed offset from the main seed to ensure deterministic but different sampling
    sampling_seed = if seed === nothing
        UInt64(12345)  # Fixed fallback seed for deterministic behavior even without user seed
    else
        UInt64(seed + 1000)  # Deterministic offset from main seed
    end

    samples_tree = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=start_variables,
        workers=workers,
        seed=sampling_seed,
        n_chains=n_chains,
        collect_iterations=iterations_to_collect,
        aggregate=aggregate,
        aggregate_type=Vector{Float64}
    )

    n_samples_collected = length(first(samples_tree)[2])
    @info "Sampling complete." n_samples_collected
    @info "First 5 samples collected" first_samples = collect(Iterators.take(samples_tree, 5))
    @info "Number of samples per activity variable" n_samples = length(first(samples_tree)[2])

    @debug "type of samples_tree" typeof(samples_tree)
    @info "Generating candidate pairs via streaming filter..."
    filter_time = @elapsed candidate_priorities = streaming_filter(
        complexes_vector,
        trivial_pairs_indices,
        samples_tree, # Pass the collected samples
        concordance_tracker;
        coarse_cv_threshold=actual_coarse_cv_threshold,
        cv_threshold=actual_cv_threshold,
        cv_epsilon=cv_epsilon,
        coarse_sample_size=coarse_sample_size,
        min_valid_samples=min_valid_samples,
        chunk_size=chunk_size_filter,
        max_pairs_in_memory=max_pairs_in_memory
    )

    @info "Candidate pairs identified" n_pairs = length(candidate_priorities) filter_time_sec = round(filter_time, digits=2)

    # Calculate adaptive batch size if not specified - ensure type stability
    n_candidates = length(candidate_priorities)
    adaptive_batch_size::Int = if batch_size === nothing && n_candidates > 0
        # Aim for approximately 10 batches
        max(1, n_candidates ÷ 10)
    else
        batch_size !== nothing ? batch_size::Int : 50  # fallback to 50 if no candidates
    end

    @info "Processing concordance tests in batches" batch_size_used = adaptive_batch_size target_batches = (n_candidates > 0 ? n_candidates ÷ adaptive_batch_size : 0)
    concordance_time = @elapsed batch_results = process_in_batches(
        constraints,
        candidate_priorities,
        concordance_tracker;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        batch_size=adaptive_batch_size,
        optimization_tolerance=actual_optimization_tolerance,
        concordance_tolerance=actual_concordance_tolerance,
        use_transitivity=use_transitivity,
    )

    @info "Building concordance modules" concordance_time_sec = round(concordance_time, digits=2)
    modules = extract_modules(concordance_tracker)

    # Build DataFrame with pre-computed columns for better performance
    complexes_df = build_complexes_dataframe(
        concordance_tracker,
        complexes,
        modules,
        trivially_balanced
    )

    # Add activity columns if data is available
    if !isempty(activity_ranges) || !isempty(trivially_balanced)
        add_activity_columns!(
            complexes_df,
            concordance_tracker,
            activity_ranges,
            trivially_balanced,
            actual_balanced_tolerance
        )
    end

    # Sort modules for deterministic output
    sorted_module_keys = sort(collect(keys(modules)))
    modules_df = DataFrame(
        module_id=String.(sorted_module_keys),
        size=[length(modules[k]) for k in sorted_module_keys],
        complexes=[join(sort(String.(modules[k])), ", ") for k in sorted_module_keys],
    )

    # Pre-allocate lambda DataFrame with known size
    n_lambda_results = length(batch_results.optimization_results)
    lambda_c1_idx = Vector{Int}(undef, n_lambda_results)
    lambda_c2_idx = Vector{Int}(undef, n_lambda_results)
    lambda_direction = Vector{Symbol}(undef, n_lambda_results)
    lambda_values = Vector{Float64}(undef, n_lambda_results)

    # Sort lambda results for deterministic output and fill columns
    sorted_lambda_results = sort!(collect(batch_results.optimization_results), by=x -> x[1])
    for (i, ((c1_idx, c2_idx, direction), lambda)) in enumerate(sorted_lambda_results)
        lambda_c1_idx[i] = c1_idx
        lambda_c2_idx[i] = c2_idx
        lambda_direction[i] = direction
        lambda_values[i] = lambda
    end

    lambda_df = DataFrame(
        c1_idx=lambda_c1_idx,
        c2_idx=lambda_c2_idx,
        direction=lambda_direction,
        lambda=lambda_values
    )

    elapsed = time() - start_time

    stats = Dict(
        # Analysis results
        "n_complexes" => n_complexes,
        "n_balanced" => count(balanced_complexes),
        "n_trivially_balanced" => length(trivially_balanced),
        "n_trivial_pairs" => length(trivial_pairs),
        "n_candidate_pairs" => length(candidate_priorities),
        "n_computed_pairs" => length(batch_results.concordant_pairs),
        "n_transitive_pairs" => batch_results.transitive_pairs,
        "n_concordant_pairs" => length(batch_results.concordant_pairs) + batch_results.transitive_pairs + length(trivial_pairs),
        "n_non_concordant_pairs" => batch_results.non_concordant_pairs,
        "n_skipped_transitivity" => batch_results.skipped_by_transitivity,
        "n_timeout_pairs" => batch_results.timeout_pairs,
        "n_modules" => length(modules),
        "batches_completed" => batch_results.batches_completed,
        "elapsed_time" => elapsed,

        # Algorithm parameters
        "optimization_tolerance" => actual_optimization_tolerance,
        "concordance_tolerance" => actual_concordance_tolerance,
        "balanced_tolerance" => actual_balanced_tolerance,
        "cv_threshold" => actual_cv_threshold,
        "coarse_cv_threshold" => actual_coarse_cv_threshold,
        "cv_epsilon" => cv_epsilon,
        "solver_tolerance" => solver_tolerance,

        # Sampling parameters
        "sample_size" => sample_size,
        "coarse_sample_size" => coarse_sample_size,
        "n_burnin" => n_burnin,
        "n_chains" => n_chains,
        "seed" => seed,

        # Processing parameters
        "batch_size" => adaptive_batch_size,
        "min_valid_samples" => min_valid_samples,
        "chunk_size_filter" => chunk_size_filter,
        "max_pairs_in_memory" => max_pairs_in_memory,
        "use_transitivity" => use_transitivity,
        "use_unidirectional_constraints" => use_unidirectional_constraints,

        # Model parameters
        "n_workers" => length(workers),
        "objective_bound" => objective_bound !== nothing ? "applied" : "none",
    )

    # Add processing statistics if processed_mask is available
    if concordance_tracker.processed_mask !== nothing
        n_processed = count(concordance_tracker.processed_mask)
        stats["n_processed_complexes"] = n_processed
        stats["processing_completion_percent"] = round(n_processed / n_complexes * 100, digits=1)
    end


    @info "Concordance analysis complete" stats

    return (
        complexes=complexes_df,
        modules=modules_df,
        lambdas=lambda_df,
        stats=stats,
    )
end

function ava_output_with_warmup(dir, om; digits=6, collect_flux=true)
    if JuMP.termination_status(om) != JuMP.OPTIMAL
        return (nothing, nothing)
    end

    # Round the results to a reasonable precision to mitigate floating point noise
    objective_val = round(JuMP.objective_value(om), digits=digits)
    activity = dir * objective_val

    # Only collect flux vector if requested (for memory efficiency)
    if collect_flux
        flux_vector = round.(JuMP.value.(om[:x]), digits=digits)
    else
        flux_vector = Float64[]  # Empty vector to save memory
    end

    return (activity, flux_vector)
end

function extract_modules(tracker::ConcordanceTracker)
    groups = Dict{Int,Vector{Int}}()
    n = length(tracker.parent)

    for i in 1:n
        root = find_set!(tracker, i)
        if !haskey(groups, root)
            groups[root] = Int[]
        end
        push!(groups[root], i)
    end

    modules = Dict{Symbol,Set{Symbol}}()

    # Get balanced complexes BitVector from tracker
    balanced_complexes = tracker.balanced_mask
    if balanced_complexes !== nothing && any(balanced_complexes)
        modules[:balanced] = Set(tracker.idx_to_id[i] for i in findall(balanced_complexes))
    end

    module_idx = 1
    for (root, members) in groups
        if length(members) > 1
            complex_ids = Set(tracker.idx_to_id[i] for i in members)
            # Check if all members are balanced using BitVector from tracker
            if balanced_complexes !== nothing && any(balanced_complexes) && all(i <= length(balanced_complexes) && balanced_complexes[i] for i in members)
                continue
            end
            module_id = Symbol("module_$(module_idx)")
            modules[module_id] = complex_ids
            module_idx += 1
        end
    end

    return modules
end

function get_module_id(complex_id::Symbol, modules::Dict{Symbol,Set{Symbol}})
    for (mid, members) in modules
        if complex_id in members
            return String(mid)
        end
    end
    return "none"
end

function find_trivially_balanced_complexes(
    complexes::Dict{Symbol,MetabolicComplex}
)::Set{Symbol}
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
    for (met_id, complex_ids) in metabolite_participation
        if length(complex_ids) == 1
            complex_id = complex_ids[1]
            push!(balanced_complexes, complex_id)
        end
    end
    return balanced_complexes
end

function find_trivially_concordant_pairs(complexes::Dict{Symbol,MetabolicComplex})::Set{Tuple{Symbol,Symbol}}
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
    for (met_id, complex_ids) in metabolite_participation
        if length(complex_ids) == 2
            c1, c2 = complex_ids
            pair = c1 < c2 ? (c1, c2) : (c2, c1)
            push!(concordant_pairs, pair)
        end
    end
    return concordant_pairs
end

function screen_directions_optimization_model(
    f,
    constraints::C.ConstraintTree,
    args...;
    optimizer,
    settings=[],
    workers=D.workers(),
)
    # We pass one model for positive and one for negative constraints
    # to avoid problems with constraint modification of cached models
    # Otherwise would need to adjust the charnes cooper bounds of the cached models repeatedly
    pos_constraints = constraints.charnes_cooper.positive
    neg_constraints = constraints.charnes_cooper.negative

    worker_cache = COBREXA.worker_local_data(
        constraints_tuple -> begin
            pos_const, neg_const = constraints_tuple
            pos_model = COBREXA.optimization_model(pos_const; optimizer=optimizer)
            neg_model = COBREXA.optimization_model(neg_const; optimizer=optimizer)
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

# --- Type-stable helper functions for better performance ---

"""
Build DataFrame for complexes with pre-computed columns for better performance.
"""
function build_complexes_dataframe(
    concordance_tracker::ConcordanceTracker,
    complexes::Dict{Symbol,<:Any},
    modules::Dict{Symbol,Set{Symbol}},
    trivially_balanced::Set{Symbol}
)::DataFrame
    # Pre-compute all columns using vectorized operations
    ids = concordance_tracker.idx_to_id
    metabolite_counts = [length(complexes[cid].metabolites) for cid in ids]

    # Use BitVector mask for trivially balanced instead of repeated Set lookups
    is_trivially_balanced_lookup = [cid in trivially_balanced for cid in ids]

    # Vectorized module assignment
    module_assignments = [get_module_id(cid, modules) for cid in ids]

    # Construct DataFrame with pre-computed columns (more efficient)
    return DataFrame(
        :id => ids,
        :n_metabolites => metabolite_counts,
        :is_trivially_balanced => is_trivially_balanced_lookup,
        :module => module_assignments
    )
end

"""
Add activity columns to DataFrame using type-stable operations.
"""
function add_activity_columns!(
    complexes_df::DataFrame,
    concordance_tracker::ConcordanceTracker,
    activity_ranges::Vector{Tuple{Float64,Float64}},
    trivially_balanced::Set{Symbol},
    actual_balanced_tolerance::Float64
)::Nothing
    n_complexes_df = nrow(complexes_df)

    # Use concrete types instead of Union types for better performance
    min_activities = Vector{Float64}(undef, n_complexes_df)
    max_activities = Vector{Float64}(undef, n_complexes_df)
    ava_confirms = Vector{Bool}(undef, n_complexes_df)

    # Build BitVector mask for trivially balanced complexes for O(1) lookups
    trivially_balanced_mask = falses(n_complexes_df)
    @inbounds for (i, cid) in enumerate(concordance_tracker.idx_to_id)
        if cid in trivially_balanced
            trivially_balanced_mask[i] = true
        end
    end

    # Since DataFrame uses concordance_tracker ordering, row index = tracker index
    @inbounds for (df_row_idx, cid) in enumerate(concordance_tracker.idx_to_id)
        min_act, max_act = activity_ranges[df_row_idx]

        if isnan(min_act) || isnan(max_act)
            # Use NaN instead of missing for Float64 columns - avoids Union types
            min_activities[df_row_idx] = NaN
            max_activities[df_row_idx] = NaN
            ava_confirms[df_row_idx] = !trivially_balanced_mask[df_row_idx]  # Fast BitVector lookup
        else
            # Complex has valid activity range
            min_activities[df_row_idx] = min_act
            max_activities[df_row_idx] = max_act
            if trivially_balanced_mask[df_row_idx]  # Fast BitVector lookup
                ava_confirms[df_row_idx] = abs(min_act) < actual_balanced_tolerance && abs(max_act) < actual_balanced_tolerance
            else
                ava_confirms[df_row_idx] = false  # Use false instead of nothing
            end
        end
    end

    complexes_df.min_activity = min_activities
    complexes_df.max_activity = max_activities
    complexes_df.ava_confirms_balanced = ava_confirms

    return nothing
end
