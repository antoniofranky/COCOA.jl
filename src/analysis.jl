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
    tolerance::Float64=1e-8
)
    stage_results = Dict{String,Any}(
        "stages_completed" => 0,
        "pairs_processed" => 0,
        "concordant_pairs" => Set{Tuple{Int,Int}}(),
        "non_concordant_pairs" => 0,
        "skipped_by_transitivity" => 0,
        "optimization_results" => Dict{Tuple{Int,Int,Symbol},Float64}()
    )

    sorted_pairs = sort(candidate_priorities,
        by=p -> (p.cv, p.c1_idx, p.c2_idx))

    remaining_pairs = [(p.c1_idx, p.c2_idx, p.directions) for p in sorted_pairs]

    stage_count = 0
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
        stage_count += 1

        if isa(concordance_tracker, ConcordanceTracker)
            clear_module_cache!(concordance_tracker)
        end

        @debug "Starting stage $stage_count" remaining = length(remaining_pairs)

        filtered_pairs, skipped_count = filter_transitive_pairs(
            remaining_pairs,
            concordance_tracker
        )
        stage_results["skipped_by_transitivity"] += skipped_count

        if isempty(filtered_pairs)
            @debug "No more pairs to process after transitivity filtering"
            break
        end

        stage_size_actual = min(stage_size, length(filtered_pairs))
        stage_pairs = filtered_pairs[1:stage_size_actual]
        remaining_pairs = filtered_pairs[(stage_size_actual+1):end]

        @debug "Processing stage $stage_count" pairs = length(stage_pairs) batch_size

        all_batch_results = []
        n_batches = ceil(Int, length(stage_pairs) / batch_size)

        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(stage_pairs))
            batch_pairs = stage_pairs[start_idx:end_idx]

            @debug "Processing batch $batch_idx/$n_batches" batch_pairs = length(batch_pairs)

            batch_results = process_concordance_batch(
                constraints, batch_pairs, concordance_tracker;
                optimizer=optimizer,
                settings=settings,
                workers=workers,
                tolerance=tolerance
            )

            append!(all_batch_results, batch_results)
        end

        batch_results = all_batch_results

        processed_pairs += length(stage_pairs)
        ProgressMeter.update!(prog, processed_pairs)

        newly_concordant = Vector{Tuple{Symbol,Symbol}}()
        stage_concordant_count = 0
        stage_prioritized_count = 0
        pairs_eliminated = 0

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

        if stage_concordant_count >= 1
            for (c1_id, c2_id) in newly_concordant
                push!(all_concordant_complexes, c1_id, c2_id)
            end

            remaining_pairs = reprioritize_by_concordant_complexes(
                remaining_pairs,
                all_concordant_complexes,
                concordance_tracker
            )

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

        if are_concordant(concordance_tracker, c1_idx, c2_idx)
            skipped_count += 1
            continue
        end

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
    remaining_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    newly_concordant::Vector{Tuple{Symbol,Symbol}},
    concordance_tracker::ConcordanceTracker
)
    if isempty(newly_concordant)
        return remaining_pairs
    end

    filtered_pairs = filter(remaining_pairs) do (c1_idx, c2_idx, directions)
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

Process a batch of concordance tests using ConstraintTrees templated approach with parallel batch processing.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    batch_pairs::Vector{Tuple{Int,Int,Set{Symbol}}},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    tolerance::Float64=1e-8
)
    # Convert indices to IDs for constraint lookup - this is the critical mapping step
    batch_pairs_with_ids = [(concordance_tracker.idx_to_id[c1_idx], concordance_tracker.idx_to_id[c2_idx], directions) for (c1_idx, c2_idx, directions) in batch_pairs]

    batch_results = screen_dual_optimization_model(
        constraints,
        batch_pairs_with_ids;
        optimizer=optimizer,
        settings=settings,
        workers=workers
    ) do models_cache, (c1_id, c2_id, directions)
        pair_results = []

        for direction in directions
            om = direction == :positive ? models_cache.positive : models_cache.negative
            target_value = direction == :positive ? 1.0 : -1.0
            c2_expr = C.substitute(constraints.activities[c2_id].value, om[:x])
            c2_constraint = @constraint(om, c2_expr == target_value)
            @objective(om, JuMP.MIN_SENSE, C.substitute(constraints.activities[c1_id].value, om[:x]))

            optimize!(om)
            min_val = termination_status(om) == OPTIMAL ? objective_value(om) : nothing

            JuMP.set_objective_sense(om, JuMP.MAX_SENSE)
            optimize!(om)
            max_val = termination_status(om) == OPTIMAL ? objective_value(om) : nothing

            if min_val !== nothing && max_val !== nothing && abs(min_val) < 1e-6 && abs(max_val) < 1e-6
                @info "Found zero lambda candidate" c1_id c2_id direction min_val max_val
            end

            delete(om, c2_constraint)
            push!(pair_results, (direction, [min_val, max_val]))
        end

        return pair_results
    end

    final_results = []
    for (i, pair_results_by_dir) in enumerate(batch_results)
        c1_idx, c2_idx, original_directions = batch_pairs[i]

        all_concordant = true
        final_lambda = nothing
        concordant_directions = Set{Symbol}()

        if isempty(pair_results_by_dir)
            all_concordant = false
        else
            for (direction, test_results) in pair_results_by_dir
                min_val, max_val = test_results

                if min_val === nothing || max_val === nothing || round(min_val, digits=2) != round(max_val, digits=2)
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

        push!(final_results, (c1_idx, c2_idx, final_direction, all_concordant, all_concordant ? final_lambda : nothing))
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
    tolerance::Float64=1e-2,
    coarse_cv_threshold::Float64=0.95,
    coarse_sample_count::Int=20,
    sample_size::Int=100,
    stage_size::Int=1000,
    batch_size::Int=1000,
    min_size_for_sharing::Int=1_000_000,
    min_valid_samples::Int=0,
    seed::Union{Int,Nothing}=42,
    use_unidirectional_constraints::Bool=true,
    filter::Union{Symbol,Vector{Symbol}}=[:cv, :cor],
    cv_threshold::Float64=0.01,
)
    start_time = time()
    filter_vec = isa(filter, Symbol) ? [filter] : filter

    model = if !isa(model, AbstractFBCModels.CanonicalModel.Model)
        @info "Converting model to CanonicalModel for optimal performance"
        convert(AbstractFBCModels.CanonicalModel.Model, model)
    else
        model
    end

    @info "Starting concordance analysis" n_workers = length(workers) tolerance coarse_cv_threshold cv_threshold sample_size use_unidirectional_constraints batch_size stage_size

    constraints, complexes =
        concordance_constraints(model; modifications, use_unidirectional_constraints, min_size_for_sharing, return_complexes=true)

    n_complexes = length(complexes)
    complex_ids = sort(collect(keys(complexes)))
    n_reactions = length(AbstractFBCModels.reactions(model))
    @info "Model statistics" n_complexes n_reactions

    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(complexes)
    @info "Found trivially balanced complexes" n_trivivally_balanced = length(trivially_balanced)

    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(complexes)
    @info "Found trivially concordant pairs" n_trivially_concordant = length(trivial_pairs)

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
    # Collect AVA results into dictionaries first to ensure determinism,
    # then process the dictionaries in a sorted order.
    complex_ranges = Dict{Symbol,Tuple{Float64,Float64}}()
    warmup_fluxes = Dict{Symbol,Vector{Vector{Float64}}}()
    for (cid, result) in ava_results
        if result !== nothing && length(result) == 2
            min_res, max_res = result
            if min_res !== nothing && max_res !== nothing
                min_activity, min_flux = min_res
                max_activity, max_flux = max_res
                complex_ranges[cid] = (min_activity, max_activity)
                warmup_fluxes[cid] = [min_flux, max_flux]
            end
        end
    end

    @info "AVA processing complete" n_complex_ranges = length(complex_ranges) ava_time_sec = round(ava_time, digits=2)

    concordance_tracker = ConcordanceTracker(complex_ids)

    # Build the warmup points vector in a deterministic order matching idx_to_id ordering
    warmup_points = Vector{Float64}[]
    for cid in concordance_tracker.idx_to_id  # Use tracker ordering for consistency
        if haskey(warmup_fluxes, cid)
            fluxes = warmup_fluxes[cid]
            push!(warmup_points, fluxes[1]) # min_flux
            push!(warmup_points, fluxes[2]) # max_flux
        end
    end

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

    balanced_complexes = Set{Int}()
    positive_complexes = Set{Int}()
    negative_complexes = Set{Int}()
    unrestricted_complexes = Set{Int}()

    # MATLAB-style balanced complex detection with tighter threshold
    balanced_threshold = 1e-9

    # Process complexes using proper concordance tracker indices
    for cid in concordance_tracker.idx_to_id
        idx = concordance_tracker.id_to_idx[cid]  # Use actual tracker index

        if haskey(complex_ranges, cid)
            min_val, max_val = complex_ranges[cid]

            # MATLAB-style rounding to threshold precision
            rounded_min = round(min_val / balanced_threshold) * balanced_threshold
            rounded_max = round(max_val / balanced_threshold) * balanced_threshold

            if rounded_min == 0.0 && rounded_max == 0.0
                push!(balanced_complexes, idx)
            elseif rounded_min >= 0.0
                push!(positive_complexes, idx)
            elseif rounded_max <= 0.0
                push!(negative_complexes, idx)
            else
                push!(unrestricted_complexes, idx)
            end
        else
            push!(unrestricted_complexes, idx)
        end
    end

    @info "Complex classification" balanced = length(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = length(positive_complexes) negative = length(negative_complexes) unrestricted = length(unrestricted_complexes)

    for (c1_id, c2_id) in trivial_pairs
        union_sets!(concordance_tracker, concordance_tracker.id_to_idx[c1_id], concordance_tracker.id_to_idx[c2_id])
    end

    complexes_vector = concordance_tracker.idx_to_id
    trivial_pairs_indices = Set{Tuple{Int,Int}}()
    for (c1_id, c2_id) in trivial_pairs
        if haskey(concordance_tracker.id_to_idx, c1_id) && haskey(concordance_tracker.id_to_idx, c2_id)
            c1_idx = concordance_tracker.id_to_idx[c1_id]
            c2_idx = concordance_tracker.id_to_idx[c2_id]
            canonical_pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            push!(trivial_pairs_indices, canonical_pair)
        end
    end
    @info "Generating candidate pairs via coefficient of variance..."

    # Use consistent ordering: build complexes_vector from tracker's idx_to_id  
    complexes_vector = [complexes[id] for id in concordance_tracker.idx_to_id]
    trivial_pairs_indices = Set(
        (concordance_tracker.id_to_idx[c1], concordance_tracker.id_to_idx[c2])
        for (c1, c2) in trivial_pairs if haskey(concordance_tracker.id_to_idx, c1) && haskey(concordance_tracker.id_to_idx, c2)
    )

    rng = StableRNG(seed)
    decimals = max(0, -floor(Int, log10(1e-9)))
    aggregate = rows -> round.(vec(hcat(rows...)), digits=decimals)
    start_vars_indices = rand(rng, 1:size(warmup, 1), min(sample_size, size(warmup, 1)))
    start_variables = warmup[start_vars_indices, :]
    samples_tree = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=constraints.activities,
        start_variables=start_variables,
        workers=workers,
        seed=rand(rng, UInt64),
        n_chains=1,
        collect_iterations=[32],
        aggregate=aggregate,
        aggregate_type=Vector{Float64}
    )
    @debug "First 5 samples collected" first_samples = collect(Iterators.take(samples_tree, 5))
    filter_config = FilterConfig(
        coarse_sample_count=coarse_sample_count,
        coarse_cv_threshold=coarse_cv_threshold,
        cv_threshold=cv_threshold,
        min_valid_samples=min_valid_samples,
        use_threads=true,
        chunk_size=1_000_000,
        max_pairs_in_memory=1_000_000,
    )

    @debug "type of samples_tree" typeof(samples_tree)
    @info "Generating candidate pairs via streaming filter..."
    filter_time = @elapsed candidate_priorities = streaming_filter(
        complexes_vector,
        balanced_complexes,
        positive_complexes,
        negative_complexes,
        unrestricted_complexes,
        trivial_pairs_indices,
        samples_tree, # Pass the collected samples
        concordance_tracker;
        config=filter_config # Pass the config object
    )

    @info "Candidate pairs identified" n_pairs = length(candidate_priorities) filter_time_sec = round(filter_time, digits=2)

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

    @info "Building concordance modules" concordance_time_sec = round(concordance_time, digits=2)
    modules = extract_modules(concordance_tracker, balanced_complexes)

    # Use concordance_tracker ordering as single source of truth for DataFrame
    complexes_df = DataFrame(
        :complex_id => concordance_tracker.idx_to_id,
        :n_metabolites => [length(complexes[cid].metabolites) for cid in concordance_tracker.idx_to_id],
        :is_balanced => [concordance_tracker.id_to_idx[cid] in balanced_complexes for cid in concordance_tracker.idx_to_id],
        :is_trivially_balanced => [cid in trivially_balanced for cid in concordance_tracker.idx_to_id],
        :module => [get_module_id(cid, modules) for cid in concordance_tracker.idx_to_id],
    )

    if !isempty(complex_ranges) || !isempty(trivially_balanced)
        min_activities = Vector{Any}(undef, nrow(complexes_df))
        max_activities = Vector{Any}(undef, nrow(complexes_df))
        ava_confirms = Vector{Any}(undef, nrow(complexes_df))

        # Since DataFrame uses concordance_tracker ordering, row index = tracker index
        for (df_row_idx, cid) in enumerate(concordance_tracker.idx_to_id)

            if haskey(complex_ranges, cid)
                min_act, max_act = complex_ranges[cid]
                min_activities[df_row_idx] = min_act
                max_activities[df_row_idx] = max_act
                if cid in trivially_balanced
                    ava_confirms[df_row_idx] = abs(min_act) < tolerance && abs(max_act) < tolerance
                else
                    ava_confirms[df_row_idx] = true
                end
            else
                min_activities[df_row_idx] = missing
                max_activities[df_row_idx] = missing
                ava_confirms[df_row_idx] = !(cid in trivially_balanced)
            end
        end

        complexes_df.min_activity = min_activities
        complexes_df.max_activity = max_activities
        complexes_df.ava_confirms_balanced = ava_confirms
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

function ava_output_with_warmup(dir, om; digits=6)
    if JuMP.termination_status(om) != JuMP.OPTIMAL
        return (nothing, nothing)
    end

    # Round the results to a reasonable precision to mitigate floating point noise
    objective_val = round(JuMP.objective_value(om), digits=digits)
    flux_vector = round.(JuMP.value.(om[:x]), digits=digits)

    activity = dir * objective_val
    return (activity, flux_vector)
end

function extract_modules(tracker::ConcordanceTracker, balanced_complexes::Set{Int})
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

    if !isempty(balanced_complexes)
        modules[:balanced] = Set(tracker.idx_to_id[i] for i in balanced_complexes)
    end

    module_idx = 1
    for (root, members) in groups
        if length(members) > 1
            complex_ids = Set(tracker.idx_to_id[i] for i in members)
            if !isempty(balanced_complexes) && issubset(Set(members), balanced_complexes)
                continue
            end
            module_id = Symbol("module_$module_idx")
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

function screen_dual_optimization_model(
    f,
    base_constraints::C.ConstraintTree,
    args...;
    optimizer,
    settings=[],
    workers=D.workers(),
)
    pos_constraints = base_constraints.charnes_cooper.positive
    neg_constraints = base_constraints.charnes_cooper.negative

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
