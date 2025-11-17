"""
Concordance Analysis for COCOA.jl

Main algorithms for identifying concordant complex pairs in metabolic networks.
"""

# ========================================================================================
# Memory-Efficient Result Accumulator
# ========================================================================================

"""
Container for batch counts with concrete types.
"""
mutable struct MutableCounts
    concordant::Int
    non_concordant::Int
    timeout::Int
    infeasible::Int           # INFEASIBLE, LOCALLY_INFEASIBLE, ALMOST_INFEASIBLE
    unbounded::Int           # DUAL_INFEASIBLE, NORM_LIMIT, ALMOST_DUAL_INFEASIBLE  
    infeasible_or_unbounded::Int  # INFEASIBLE_OR_UNBOUNDED
    resource_limit::Int      # ITERATION_LIMIT, MEMORY_LIMIT, NODE_LIMIT, etc.
    numerical_error::Int     # NUMERICAL_ERROR, INVALID_MODEL, INVALID_OPTION
    other_error::Int         # OTHER_ERROR, INTERRUPTED, etc.
    transitive::Int
    skipped::Int
    total_optimizations::Int  # Total number of optimization attempts
end

MutableCounts() = MutableCounts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

"""
Configuration for batch processing.
"""
struct BatchProcessingConfig
    batch_size::Int
    concordance_tolerance::Float64
    use_transitivity::Bool
    optimizer
    settings::Vector
    workers::Vector
end

struct BatchOptimizationResult
    concordant_pairs::SparseConcordantPairs
    non_concordant::Int
    timeout::Int
    infeasible::Int
    unbounded::Int
    infeasible_or_unbounded::Int
    resource_limit::Int
    numerical_error::Int
    other_error::Int
    transitive::Int
    skipped::Int
    total_optimizations::Int
    optimization_results::Vector{Tuple{Int,Int,Symbol,Float64}}
    direct_concordant_pairs::Union{Nothing,Set{Tuple{Int,Int}}}
end


"""
Memory-efficient accumulator for batch processing results.
Uses concrete types and pre-allocated buffers for optimal performance.
"""
struct BatchResultAccumulator
    concordant_pairs::SparseConcordantPairs
    counts::MutableCounts
    optimization_results::Vector{Tuple{Int,Int,Symbol,Float64}}
    temp_pairs::Vector{Tuple{Int,Int}}
    temp_values::Vector{Float64}
end

function BatchResultAccumulator(n_complexes::Int)
    BatchResultAccumulator(
        SparseConcordantPairs(n_complexes),
        MutableCounts(),
        Vector{Tuple{Int,Int,Symbol,Float64}}(),
        Vector{Tuple{Int,Int}}(),
        Vector{Float64}()
    )
end

"""
Add batch results to accumulator with type-stable operations.
"""
function accumulate_results!(
    acc::BatchResultAccumulator,
    concordant_pairs::SparseConcordantPairs,
    counts::MutableCounts,
    optimization_results::Vector{Tuple{Int,Int,Symbol,Float64}}
)::Nothing
    # Update counts atomically
    acc.counts.concordant += counts.concordant
    acc.counts.non_concordant += counts.non_concordant
    acc.counts.timeout += counts.timeout
    acc.counts.infeasible += counts.infeasible
    acc.counts.unbounded += counts.unbounded
    acc.counts.infeasible_or_unbounded += counts.infeasible_or_unbounded
    acc.counts.resource_limit += counts.resource_limit
    acc.counts.numerical_error += counts.numerical_error
    acc.counts.other_error += counts.other_error
    acc.counts.transitive += counts.transitive
    acc.counts.skipped += counts.skipped
    acc.counts.total_optimizations += counts.total_optimizations

    # Merge concordant pairs efficiently
    merge_pairs!(acc.concordant_pairs, concordant_pairs)

    # Add optimization results
    append!(acc.optimization_results, optimization_results)
    return nothing
end

"""
Reset accumulator for reuse with efficient buffer clearing.
"""
function reset!(acc::BatchResultAccumulator)::Nothing
    clear!(acc.concordant_pairs)
    acc.counts.concordant = 0
    acc.counts.non_concordant = 0
    acc.counts.timeout = 0
    acc.counts.infeasible = 0
    acc.counts.unbounded = 0
    acc.counts.infeasible_or_unbounded = 0
    acc.counts.resource_limit = 0
    acc.counts.numerical_error = 0
    acc.counts.other_error = 0
    acc.counts.transitive = 0
    acc.counts.skipped = 0
    acc.counts.total_optimizations = 0
    empty!(acc.optimization_results)
    empty!(acc.temp_pairs)
    empty!(acc.temp_values)
    return nothing
end

# ========================================================================================
# Memory Monitoring
# ========================================================================================

"""
Monitor memory usage and suggest batch size adjustments.
"""
mutable struct MemoryMonitor
    initial_memory::Float64
    peak_memory::Float64
    gc_time::Float64
    gc_count::Int
    last_check_time::Float64

    function MemoryMonitor()
        GC.gc()  # Clean slate
        initial = Base.gc_live_bytes() / 1024^3  # GB
        new(initial, initial, 0.0, 0, time())
    end
end

"""
Update memory statistics and return current usage in GB.
"""
function check_memory!(monitor::MemoryMonitor)::Float64
    current = Base.gc_live_bytes() / 1024^3
    monitor.peak_memory = max(monitor.peak_memory, current)

    # Check GC statistics
    gc_stats = Base.gc_num()
    monitor.gc_count = gc_stats.full_sweep
    monitor.gc_time = gc_stats.total_time / 1e9  # Convert to seconds

    return current
end

"""
Categorize JuMP termination status into failure mode categories.
Returns symbol indicating the category of termination status.
"""
@inline function categorize_termination_status(status)::Symbol
    if status in (J.OPTIMAL, J.LOCALLY_SOLVED, J.ALMOST_OPTIMAL)
        return :success
    elseif status == J.TIME_LIMIT
        return :timeout
    elseif status in (J.INFEASIBLE, J.LOCALLY_INFEASIBLE, J.ALMOST_INFEASIBLE)
        return :infeasible
    elseif status in (J.DUAL_INFEASIBLE, J.NORM_LIMIT, J.ALMOST_DUAL_INFEASIBLE)
        return :unbounded
    elseif status == J.INFEASIBLE_OR_UNBOUNDED
        return :infeasible_or_unbounded
    elseif status in (J.ITERATION_LIMIT, J.MEMORY_LIMIT, J.NODE_LIMIT, J.SOLUTION_LIMIT, J.OBJECTIVE_LIMIT, J.OTHER_LIMIT)
        return :resource_limit
    elseif status in (J.NUMERICAL_ERROR, J.INVALID_MODEL, J.INVALID_OPTION)
        return :numerical_error
    else  # J.OTHER_ERROR, J.INTERRUPTED, J.OPTIMIZE_NOT_CALLED, J.SLOW_PROGRESS
        @debug "Other error encountered" status = status
        return :other_error
    end
end

# ========================================================================================
# Batch Processing State Management  
# ========================================================================================

"""
State container for streaming batch processing with concrete types.
"""
mutable struct BatchProcessingState
    # Core processing state
    accumulator::BatchResultAccumulator
    memory_monitor::MemoryMonitor
    current_batch::Vector{PairCandidate}

    # Counters
    total_batches_completed::Int
    total_pairs_processed::Int
    total_candidates_seen::Int
    batches_processed::Int

    # Filter tracking for logging
    last_filtered_cv::Int
    last_filtered_concordant::Int
    last_filtered_non_concordant::Int

    # Timing
    batch_collection_start_time::Float64
    total_possible_pairs::Int
end

function initialize_processing_state(
    streaming_filter::StreamingCandidateFilter,
    concordance_tracker::ConcordanceTracker,
    config::BatchProcessingConfig
)::BatchProcessingState
    n_complexes = length(concordance_tracker.idx_to_id)
    current_batch = Vector{PairCandidate}()
    sizehint!(current_batch, config.batch_size)

    # Calculate total possible pairs for progress tracking
    # Filter examines ALL pairs, so use total pairs as denominator for accurate progress
    total_possible_pairs = n_complexes * (n_complexes - 1) ÷ 2

    # Calculate expected candidates for information
    n_balanced = count(concordance_tracker.balanced_mask)
    n_unbalanced = n_complexes - n_balanced
    n_trivial = length(streaming_filter.trivial_pairs)
    total_possible_candidates = n_unbalanced * (n_unbalanced - 1) ÷ 2 - n_trivial

    @info "Progress tracking initialized" n_complexes = n_complexes n_balanced = n_balanced n_unbalanced = n_unbalanced n_trivial = n_trivial total_possible_pairs = total_possible_pairs expected_candidates = total_possible_candidates

    return BatchProcessingState(
        BatchResultAccumulator(n_complexes),
        MemoryMonitor(),
        current_batch,
        0, 0, 0, 0,  # counters
        0, 0, 0,     # filter tracking
        time(),      # timing
        total_possible_pairs  # total_possible_pairs
    )
end

@inline function collect_candidate!(state::BatchProcessingState, candidate::PairCandidate)::Nothing
    push!(state.current_batch, candidate)
    state.total_candidates_seen += 1
    return nothing
end

@inline function should_process_batch(state::BatchProcessingState, config::BatchProcessingConfig)::Bool
    return length(state.current_batch) >= config.batch_size
end

function process_full_batch!(
    constraints::C.ConstraintTree,
    state::BatchProcessingState,
    concordance_tracker::ConcordanceTracker,
    config::BatchProcessingConfig,
    streaming_filter::StreamingCandidateFilter;
    track_direct_pairs::Bool=false
)::Union{Nothing,Set{Tuple{Int,Int}}}
    state.batches_processed += 1

    # Log batch collection completion
    log_batch_collection!(state, streaming_filter)

    # Clear module cache
    clear_module_cache!(concordance_tracker)

    # Process the batch
    batch_result = execute_batch_optimization!(constraints, state, concordance_tracker, config; track_direct_pairs=track_direct_pairs)

    # Create counts for accumulation - extract concordant count from batch_result
    concordant_count = batch_result.concordant_pairs.n_pairs  # This was set correctly in execute_batch_optimization!
    batch_counts = MutableCounts(
        concordant_count,
        batch_result.non_concordant,
        batch_result.timeout,
        batch_result.infeasible,
        batch_result.unbounded,
        batch_result.infeasible_or_unbounded,
        batch_result.resource_limit,
        batch_result.numerical_error,
        batch_result.other_error,
        batch_result.transitive,
        batch_result.skipped,
        batch_result.total_optimizations
    )

    # Accumulate results
    accumulate_results!(state.accumulator, batch_result.concordant_pairs, batch_counts, batch_result.optimization_results)
    state.total_batches_completed += 1
    state.total_pairs_processed += length(state.current_batch)

    # Reset for next batch
    empty!(state.current_batch)
    state.batch_collection_start_time = time()

    # Return direct concordant pairs if tracking was enabled
    if track_direct_pairs && batch_result.direct_concordant_pairs !== nothing
        return batch_result.direct_concordant_pairs
    else
        return nothing
    end
end

function log_batch_collection!(state::BatchProcessingState, streaming_filter::StreamingCandidateFilter)::Nothing
    collection_time = time() - state.batch_collection_start_time
    collection_time_str = Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, collection_time * 1000)), "HH:MM:SS.s")

    # Calculate progress percentage based on pairs actually examined by filter
    pairs_examined = streaming_filter.pairs_tested
    progress_pct = round(pairs_examined / max(1, state.total_possible_pairs) * 100, digits=1)

    # Get filtering statistics from streaming filter
    filter_report = get_filter_report(streaming_filter)

    @info "Batch $(state.batches_processed): $(length(state.current_batch)) candidates collected [$collection_time_str] ($(pairs_examined)/$(state.total_possible_pairs) = $(progress_pct)% examined) \n [Candidates: $(state.total_candidates_seen), Filtered - CV: $(filter_report.breakdown.cv_threshold), Known: concordant: $(filter_report.breakdown.known_concordant) non-concordant: $(filter_report.breakdown.known_non_concordant), Trivial: $(filter_report.breakdown.trivial), Balanced: $(filter_report.breakdown.balanced)]"
    return nothing
end


function execute_batch_optimization!(
    constraints::C.ConstraintTree,
    state::BatchProcessingState,
    concordance_tracker::ConcordanceTracker,
    config::BatchProcessingConfig;
    track_direct_pairs::Bool=false
)::BatchOptimizationResult
    optimization_start_time = time()

    # OPTIMIZED: Run optimizations and update concordance tracker in single pass
    # Eliminates process_batch_results! entirely
    batch_counts = process_concordance_batch(
        constraints,
        state.current_batch,
        concordance_tracker;
        optimizer=config.optimizer,
        settings=config.settings,
        workers=config.workers,
        concordance_tolerance=config.concordance_tolerance,
        track_direct_pairs=track_direct_pairs
    )

    # Log timing
    optimization_time = time() - optimization_start_time
    opt_time_str = Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, optimization_time * 1000)), "HH:MM:SS.s")
    @info "Batch $(state.batches_processed): $(length(state.current_batch)) optimized → $(batch_counts.concordant_count) concordant, $(batch_counts.non_concordant_count) non-concordant [$opt_time_str]"

    # Create concordant pairs structure with correct count for statistics
    n_complexes = length(concordance_tracker.idx_to_id)
    concordant_pairs = SparseConcordantPairs(n_complexes)
    # Set the correct count for statistics reporting (pairs are already in concordance_tracker)
    concordant_pairs.n_pairs = batch_counts.concordant_count

    return BatchOptimizationResult(
        concordant_pairs,
        batch_counts.non_concordant_count,
        batch_counts.timeout_count,
        batch_counts.infeasible_count,
        batch_counts.unbounded_count,
        batch_counts.infeasible_or_unbounded_count,
        batch_counts.resource_limit_count,
        batch_counts.numerical_error_count,
        batch_counts.other_error_count,
        0, 0,  # transitive counts handled elsewhere  
        batch_counts.total_optimizations,
        batch_counts.optimization_results,
        batch_counts.direct_concordant_pairs
    )
end

function process_batch_results!(opt_results, concordance_tracker, n_complexes)
    concordant_pairs = SparseConcordantPairs(n_complexes)
    non_concordant_count = 0
    timeout_count = 0
    opt_results_vec = Vector{Tuple{Int,Int,Symbol,Float64}}()

    for result in opt_results
        c1_idx, c2_idx, direction, is_concordant, lambda, has_timeout = result

        has_timeout && (timeout_count += 1)

        if is_concordant
            union_sets!(concordance_tracker, c1_idx, c2_idx)
            add_pair!(concordant_pairs, c1_idx, c2_idx)
            if !isnothing(lambda) && !isnan(lambda)
                push!(opt_results_vec, (c1_idx, c2_idx, direction, lambda))
            end
        else
            add_non_concordant!(concordance_tracker, c1_idx, c2_idx)
            non_concordant_count += 1
        end
    end

    counts = (non_concordant=non_concordant_count, timeout=timeout_count, optimization_results=opt_results_vec)
    return concordant_pairs, counts
end

"""
Minimal state for tracking pair concordance during streaming processing.
Optimized for zero-allocation incremental updates.
"""
mutable struct PairConcordanceState
    positive_min::Float64
    positive_max::Float64
    negative_min::Float64
    negative_max::Float64
    has_positive::Bool
    has_negative::Bool
    has_timeout::Bool
    is_concordant::Bool
    reference_lambda::Float64

    PairConcordanceState() = new(NaN, NaN, NaN, NaN, false, false, false, false, NaN)
end


"""
    activity_concordance_analysis(model; optimizer, kwargs...)

Perform comprehensive concordance analysis to identify concordant complex pairs in metabolic networks.

This is the main function for concordance analysis in COCOA.jl. It systematically identifies
pairs of complexes that can maintain the same activity ratio across the feasible flux space,
which is fundamental for understanding concentration robustness in biochemical networks.

Follows COBREXA.jl patterns for constraint building and modification.

# Result Overview
Returns a `ConcordanceResults` with:
- **Concordance matrix**: Binary relationships between all complex pairs
    - 0 = non-concordant, 1 = concordant,   2 = trivially concordant, 3 = balanced, 4 = trivally balanced 
- **Activity ranges**: Min/max activity values for each complex  
- **Concordance modules**: Groups of mutually concordant complexes
- **Analysis statistics**: Comprehensive metrics and processing information
- **Optional kinetic data**: Kinetic modules and interface reactions (if requested)

# Arguments
- `model`: Metabolic model (supports COBREXA.jl compatible formats)

# Required Keyword Arguments
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)

# Optional Keyword Arguments

## Model Processing
- `objective_bound=nothing`: Objective bound function (e.g., `relative_tolerance_bound(0.999)`), or `nothing` to skip
- `use_unidirectional_constraints::Bool=true`: Use unidirectional flux constraints

# Objective Bound Pattern
The `objective_bound` parameter follows COBREXA's pattern from `flux_variability_analysis`.
Pass a bound function like `relative_tolerance_bound(0.999)` or `absolute_tolerance_bound(1e-5)`,
or `nothing` to perform analysis without objective bounding.

# Examples
```julia
using HiGHS
import COBREXA

# Basic concordance analysis (no objective bound)
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer
)

# With 99.9% objective bound (following COBREXA FVA pattern)
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999)
)

# With absolute tolerance bound
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.absolute_tolerance_bound(1e-5)
)

# With custom tolerances and batch size
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.95),
    concordance_tolerance=1e-5,
    batch_size=100_000,
    seed=1234
)
```

## Analysis Parameters
- `concordance_tolerance::Float64=NaN`: Tolerance for concordance detection
- `balanced_threshold::Float64=NaN`: Threshold for balanced complex detection
- `cv_threshold::Float64=NaN`: Coefficient of variation threshold for filtering
- `cv_epsilon::Float64=1e-16`: Small value added to avoid division by zero in CV calculation
- `sample_size::Int=100`: Number of samples for coefficient of variation estimation
- `min_valid_samples::Int=10`: Minimum valid samples required for CV calculation
- `seed::Int=0`: Random seed for reproducible sampling (0 uses global RNG)

## Performance Settings
- `batch_size::Int=50_000`: Number of candidate pairs processed per batch
- `workers=D.workers()`: Worker processes for parallel computation
- `settings=[]`: Solver-specific settings vector
- `use_transitivity::Bool=true`: Use transitivity relationships to reduce computation

## Sampling Options  
- `n_burnin::Int=50`: Burnin samples for warmup point generation
- `n_chains::Int=1`: Number of parallel chains (for future use)
- `kinetic_analysis::Bool=false`: Apply kinetic module analysis to results

# Returns
`ConcordanceResults` containing:
- `concordance_matrix`: Sparse boolean matrix of concordant pairs
- `activity_ranges`: Activity variability ranges for each complex
- `concordance_modules`: Concordance modules (connected components)
- `stats`: Comprehensive analysis statistics
- Additional fields for kinetic analysis (if enabled)

# Statistics Dictionary
The `stats` field contains detailed analysis metrics:

## Core Results
- `n_concordant_pairs`: Total concordant pairs found
- `n_candidate_pairs`: Candidate pairs after filtering  
- `n_computed_pairs`: Pairs found via optimization
- `n_trivial_pairs`: Trivially concordant pairs
- `n_non_concordant_pairs`: Non-concordant pairs

## Processing Statistics
- `batches_completed`: Number of optimization batches
- `elapsed_time`: Total analysis time (seconds)
- `n_candidates_skipped_by_transitivity`: Pairs skipped via transitivity

## Model Information
- `n_complexes`: Total complexes in model
- `n_balanced`: Balanced complexes
- `n_modules`: Concordance modules found
"""
function activity_concordance_analysis(
    model;
    optimizer,
    settings=[],
    objective_bound=nothing,
    workers=D.workers(),
    balanced_threshold::Float64=1e-8,
    concordance_tolerance::Float64=1e-7,
    cv_threshold::Float64=1e-7,
    cv_epsilon::Float64=1e-16,
    sample_size::Int=100,
    batch_size::Int=50_000,
    min_valid_samples::Int=10,
    seed::UInt=rand(UInt),
    use_unidirectional_constraints::Bool=false,
    use_transitivity::Bool=true,
    n_burnin::Int=50,
    n_chains::Int=1,
    kinetic_analysis::Bool=true,
)
    start_time = time()

    @info "Starting concordance analysis" n_workers = length(workers) concordance_tolerance balanced_threshold cv_threshold sample_size use_unidirectional_constraints batch_size

    constraints, complexes =
        concordance_constraints(model; use_unidirectional_constraints, return_complexes=true)

    # Add objective bound constraint if specified (COBREXA pattern)
    if !isnothing(objective_bound)
        # Get optimal objective value first
        objective_flux = COBREXA.optimized_values(
            constraints.balance;
            objective=constraints.balance.objective.value,
            output=constraints.balance.objective,
            optimizer,
            settings,
        )

        if !isnothing(objective_flux)
            @info "Objective flux determined" objective_flux
            # Add objective bound constraint to limit feasible space (COBREXA way with *)
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
            @warn "Could not determine optimal objective value, skipping objective bound"
        end
    end

    n_complexes = length(complexes)
    # Get sorted complex IDs for fully deterministic ConcordanceTracker initialization
    # Sort by string representation to ensure consistent ordering across platforms
    complex_ids = sort!(collect(keys(complexes)); by=string)

    # Get correct reaction count from constraints (accounts for reaction splitting)
    n_reactions = C.var_count(constraints.balance)
    n_metabolites = length(constraints.balance.flux_stoichiometry)

    @info "Model statistics" n_complexes n_reactions n_metabolites

    # Get canonical metabolite ordering
    all_metabolites = Set{Symbol}()
    for complex in values(complexes)
        for (met_id, _) in complex
            push!(all_metabolites, met_id)
        end
    end
    metabolite_ids = sort!(collect(all_metabolites); by=string)

    # Construct Y matrix once for all trivial relationship detection
    @info "Building Y matrix for trivial relationship detection"
    Y_matrix, Y_metabolite_ids, Y_complex_ids = Y_matrix_from_constraints(constraints; return_ids=true)

    @info "Finding trivially balanced complexes"
    initial_balanced = find_trivially_balanced_complexes_sparse(Y_matrix, Y_metabolite_ids, Y_complex_ids)
    trivially_balanced = extend_trivially_balanced_complexes_sparse(Y_matrix, Y_metabolite_ids, Y_complex_ids, initial_balanced)
    @info "Found trivially balanced complexes" n_trivivally_balanced = length(trivially_balanced)

    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs_sparse(Y_matrix, Y_metabolite_ids, Y_complex_ids)
    @info "Found trivially concordant pairs" n_trivially_concordant = length(trivial_pairs)

    # Initialize basic matrix structure - we'll build the complete matrix later
    n_complexes = length(complex_ids)
    complex_idx = Dict(id => idx for (idx, id) in enumerate(complex_ids))


    # Set up RNG early for deterministic sampling
    rng = StableRNGs.StableRNG(seed)

    # Use the separated AVA function with warmup point generation and external timing
    ava_digits = max(1, -floor(Int, log10(concordance_tolerance)))
    ava_output_func = (dir, om) -> ava_output_with_warmup(dir, om; digits=ava_digits)
    @info "Running Activity Variability Analysis (AVA)"
    ava_time = @elapsed ava_results = activity_variability_analysis(
        constraints,
        complex_ids;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        output=ava_output_func,
        output_type=Tuple{Float64,Vector{Float64}},
        return_warmup_points=true
    )

    ava_time_str = Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, ava_time * 1000)), "HH:MM:SS.s")
    @info "AVA processing complete [$ava_time_str]"

    concordance_tracker = ConcordanceTracker(complex_ids)

    # Extract results from AVA (now returns a named tuple when return_warmup_points=true)
    activity_ranges = ava_results.activity_ranges
    warmup = ava_results.warmup_points

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

    # Initialize BitVector masks in the ConcordanceTracker
    ensure_mask_allocated!(concordance_tracker, :balanced)
    ensure_mask_allocated!(concordance_tracker, :positive)
    ensure_mask_allocated!(concordance_tracker, :negative)
    ensure_mask_allocated!(concordance_tracker, :unrestricted)

    # Get references to the BitVector masks for efficient access
    balanced_complexes = get_mask(concordance_tracker, :balanced)
    positive_complexes = get_mask(concordance_tracker, :positive)
    negative_complexes = get_mask(concordance_tracker, :negative)
    unrestricted_complexes = get_mask(concordance_tracker, :unrestricted)

    # Build BitVector mask for trivially balanced complexes for O(1) lookups
    trivially_balanced_mask = falses(n_complexes)
    for (i, cid) in enumerate(concordance_tracker.idx_to_id)
        if cid in trivially_balanced
            trivially_balanced_mask[i] = true
        end
    end

    # Optimized complex classification with minimal branching
    @inbounds for i in eachindex(activity_ranges)
        # Fast BitVector lookup with early exit
        if trivially_balanced_mask[i]
            balanced_complexes[i] = true
            continue
        end

        min_val, max_val = activity_ranges[i]

        # Branchless classification using boolean arithmetic
        is_inactive = isnan(min_val) || isnan(max_val)

        if is_inactive
            unrestricted_complexes[i] = true
        else
            @fastmath begin
                abs_min = abs(min_val)
                abs_max = abs(max_val)

                is_balanced = (abs_min < balanced_threshold) & (abs_max < balanced_threshold)
                is_positive = !is_balanced & (min_val >= -balanced_threshold)
                is_negative = !is_balanced & !is_positive & (max_val <= balanced_threshold)

                balanced_complexes[i] = is_balanced
                positive_complexes[i] = is_positive
                negative_complexes[i] = is_negative
                unrestricted_complexes[i] = !(is_balanced | is_positive | is_negative)
            end
        end
    end

    @info "Complex classification" balanced = count(balanced_complexes) trivially_balanced = length(trivially_balanced) positive = count(positive_complexes) negative = count(negative_complexes) unrestricted = count(unrestricted_complexes)

    # Capture counts for statistics before any potential variable changes
    n_balanced_complexes = count(balanced_complexes)
    n_trivially_balanced_complexes = length(trivially_balanced)

    for (c1_id, c2_id) in trivial_pairs
        union_sets!(concordance_tracker, concordance_tracker.id_to_idx[c1_id], concordance_tracker.id_to_idx[c2_id])
    end

    # Note: Balanced complexes are handled separately in the matrix construction
    # They should not be added to the concordance tracker as they don't participate
    # in flux ratio concordance analysis (zero flux = undefined ratios)

    # Create filtered activities constraint tree excluding balanced complexes for memory efficiency
    active_complex_ids = [
        concordance_tracker.idx_to_id[i]
        for i in eachindex(concordance_tracker.idx_to_id)
        if !balanced_complexes[i]
    ]

    filtered_activities = C.ConstraintTree(
        id => constraints.activities[id]
        for id in active_complex_ids
        if haskey(constraints.activities, id)
    )

    # Use tracker's idx_to_id directly as complexes_vector (it's already a Vector{Symbol})
    complexes_vector = concordance_tracker.idx_to_id

    # Process trivial pairs into indices set with pre-allocated capacity
    trivial_pairs_indices = Set{Tuple{Int,Int}}()
    sizehint!(trivial_pairs_indices, length(trivial_pairs))

    # Cache lookups for better performance
    id_to_idx = concordance_tracker.id_to_idx

    for (c1_id, c2_id) in trivial_pairs
        if haskey(id_to_idx, c1_id) && haskey(id_to_idx, c2_id)
            c1_idx = id_to_idx[c1_id]
            c2_idx = id_to_idx[c2_id]
            canonical_pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            push!(trivial_pairs_indices, canonical_pair)
        end
    end
    @info "Generating candidate pairs via coefficient of variance..."

    decimals = max(0, -floor(Int, log10(concordance_tolerance)))
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

    # Pre-allocate start_variables_list with exact size for better performance
    start_variables_list = Vector{Vector{Float64}}()
    sizehint!(start_variables_list, sample_size)

    # Strategy 1: Use extreme boundary points for maximum activity ranges
    if n_extreme_points > 0 && !isempty(warmup)
        # Select most diverse extreme points (max distance from each other)
        extreme_indices = Random.rand(rng, 1:size(warmup, 1), n_extreme_points)
        for idx in extreme_indices
            push!(start_variables_list, warmup[idx, :])
        end
    end

    # Strategy 2: Use center point for balanced exploration
    if n_center_points > 0 && !isempty(warmup)
        # Create center point as average of all warmup points
        center_point = vec(Statistics.mean(warmup, dims=1))
        push!(start_variables_list, center_point)
    end

    # Strategy 3: Generate diverse random points as convex combinations
    # Optimized with pre-allocated buffers to minimize allocations
    if n_random_points > 0 && size(warmup, 1) >= 2
        # Pre-allocate reusable buffers
        max_base_points = min(4, size(warmup, 1))
        weights = Vector{Float64}(undef, max_base_points)
        base_indices = Vector{Int}(undef, max_base_points)
        random_flux = Vector{Float64}(undef, size(warmup, 2))

        @inbounds for i in 1:n_random_points
            # Use varying numbers of base points for different exploration depths
            n_base_points = 2 + (i % 3)  # Alternate between 2, 3, 4 base points
            n_base_points = min(n_base_points, size(warmup, 1))

            # Reuse pre-allocated arrays
            Random.rand!(rng, view(base_indices, 1:n_base_points), 1:size(warmup, 1))
            Random.rand!(rng, view(weights, 1:n_base_points))

            # Normalize weights in-place
            weight_sum = sum(view(weights, 1:n_base_points))
            @simd for j in 1:n_base_points
                weights[j] /= weight_sum
            end

            # Create convex combination with pre-allocated buffer
            fill!(random_flux, 0.0)
            @simd for j in 1:n_base_points
                @. random_flux += weights[j] * warmup[base_indices[j], :]
            end

            push!(start_variables_list, copy(random_flux))  # Copy to avoid aliasing
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

    @info "Sampling..."
    samples_tree = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=filtered_activities,
        start_variables=start_variables,
        workers=workers,
        seed=seed,
        n_chains=n_chains,
        collect_iterations=iterations_to_collect,
        aggregate=aggregate,
        aggregate_type=Vector{Float64}
    )

    n_samples_collected = length(first(samples_tree)[2])
    @info "Sampling complete." n_samples_collected
    @debug "First 5 samples collected" first_samples = collect(Iterators.take(samples_tree, 5))
    @debug "Number of samples per activity variable" n_samples = length(first(samples_tree)[2])

    # @debug "type of samples_tree" typeof(samples_tree)
    @info "Creating chunked streaming filter for memory-efficient processing..."

    @info "Using fixed batch size" batch_size = batch_size n_complexes = length(complexes_vector)

    # Create direct streaming filter (eliminates redundant chunking layer)
    streaming_filter = try
        StreamingCandidateFilter(
            complexes_vector,
            trivial_pairs_indices,
            samples_tree, # Pass the collected samples
            concordance_tracker;
            cv_threshold=cv_threshold,
            cv_epsilon=cv_epsilon,
            min_valid_samples=min_valid_samples,
            use_transitivity=use_transitivity
        )
    catch e
        @error "Failed to create streaming filter." exception = e
        rethrow(e)
    end

    @info "Direct streaming filter created"

    @info "Processing concordance tests with direct streaming (deterministic batch processing)"

    # Create configuration object for type-stable processing
    config = BatchProcessingConfig(
        batch_size,
        concordance_tolerance,
        use_transitivity,
        optimizer,
        settings,
        workers
    )

    concordance_time = @elapsed batch_results = process_streaming_batches(
        constraints,
        streaming_filter,
        concordance_tracker,
        config;
        track_direct_pairs=!use_transitivity
    )

    concordance_time_str = Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, concordance_time * 1000)), "HH:MM:SS.s")
    @info "Building concordance modules [$concordance_time_str]"
    modules = extract_modules(concordance_tracker)

    elapsed = time() - start_time

    # Store intermediate processing statistics (but not final concordance counts yet)
    intermediate_stats = Dict(
        # Model information
        "n_reactions" => n_reactions,
        "n_metabolites" => n_metabolites,
        "n_complexes" => n_complexes,
        "n_balanced" => n_balanced_complexes,
        "n_trivially_balanced" => n_trivially_balanced_complexes,
        "n_trivial_pairs" => length(trivial_pairs),

        # Processing statistics (keep the useful logging info)
        "n_candidate_pairs" => batch_results.pairs_processed,
        "n_non_concordant_pairs" => batch_results.non_concordant_pairs,
        "n_candidates_skipped_by_transitivity" => streaming_filter.pairs_transitivity_concordant_filtered + streaming_filter.pairs_transitivity_non_concordant_filtered,
        "n_timeout_pairs" => batch_results.timeout_pairs,
        "n_infeasible_pairs" => batch_results.infeasible_pairs,
        "n_unbounded_pairs" => batch_results.unbounded_pairs,
        "n_infeasible_or_unbounded_pairs" => batch_results.infeasible_or_unbounded_pairs,
        "n_resource_limit_pairs" => batch_results.resource_limit_pairs,
        "n_numerical_error_pairs" => batch_results.numerical_error_pairs,
        "n_other_error_pairs" => batch_results.other_error_pairs,
        "n_total_optimizations" => batch_results.total_optimizations,
        "batches_completed" => batch_results.batches_completed,
        "elapsed_time" => elapsed,

        # Algorithm parameters
        "concordance_tolerance" => concordance_tolerance,
        "balanced_threshold" => balanced_threshold,
        "cv_threshold" => cv_threshold,
        "cv_epsilon" => cv_epsilon,

        # Sampling parameters
        "sample_size" => sample_size,
        "n_burnin" => n_burnin,
        "n_chains" => n_chains,
        "seed" => seed,

        # Processing parameters 
        "batch_size" => batch_size,
        "min_valid_samples" => min_valid_samples,
        "use_transitivity" => use_transitivity,
        "use_unidirectional_constraints" => use_unidirectional_constraints,

        # Model parameters
        "n_workers" => length(workers),
        "objective_bound" => objective_bound !== nothing ? "applied" : "none",

        # Store batch processing results for reference
        "batch_results_concordant_count" => batch_results.concordant_count,
    )

    # Add processing statistics if processed_mask is available
    if concordance_tracker.processed_mask !== nothing
        n_processed = count(concordance_tracker.processed_mask)
        intermediate_stats["n_processed_complexes"] = n_processed
        intermediate_stats["processing_completion_percent"] = round(n_processed / n_complexes * 100, digits=1)
    end

    @info "Building complete concordance matrix with all relationships"
    concordance_matrix = build_complete_concordance_matrix(
        n_complexes, complex_idx, trivially_balanced, trivial_pairs,
        concordance_tracker, balanced_complexes, use_transitivity, batch_results
    )

    # Build ConcordanceResults using canonical ordering established earlier
    # complex_ids already canonical from concordance_tracker
    # metabolite_ids already canonical from earlier in function
    # Get reaction names directly from constraints
    feasible_reaction_names = get_reaction_names_from_constraints(constraints.balance)
    reaction_ids = Symbol.(sort!(feasible_reaction_names; by=string))

    # Create module mapping with balanced complexes as 0 and consecutive module IDs
    module_mapping = fill(-1, length(complex_ids))  # Default: -1 for singleton complexes

    # Create mapping from root representative to consecutive module ID
    root_to_consecutive = Dict{Int,Int}()
    next_module_id = 1

    for (module_root, complex_list) in modules
        if length(complex_list) > 1  # Only assign IDs to modules with multiple complexes
            root_to_consecutive[module_root] = next_module_id
            next_module_id += 1
        end
    end

    # Assign module IDs to complexes
    for (module_root, complex_list) in modules
        for complex_id in complex_list
            idx = concordance_tracker.id_to_idx[complex_id]
            if balanced_complexes[idx]
                # Balanced complexes get module_id = 0 (highest priority)
                module_mapping[idx] = 0
            elseif length(complex_list) > 1
                # Regular concordance modules get consecutive positive IDs
                module_mapping[idx] = root_to_consecutive[module_root]
            else
                # Singleton modules remain -1 (no concordance module)
                module_mapping[idx] = -1
            end
        end
    end

    # Process lambda results into dict
    lambda_dict = Dict{Tuple{Int,Int},Float64}()
    for ((c1_idx, c2_idx, direction), lambda_val) in batch_results.optimization_results
        if c1_idx <= length(complex_ids) && c2_idx <= length(complex_ids)
            canonical_pair = c1_idx < c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
            lambda_dict[canonical_pair] = lambda_val
        end
    end

    # Build coordinate vectors for efficient concordance matrix construction
    # All concordance relationships are now populated directly in the concordance_matrix
    # via populate_trivial_relationships! and populate_discovered_relationships!

    # Use the constructor with pre-built concordance matrix (more efficient than coordinate vectors)
    complete_model = ConcordanceResults(
        complex_ids,
        reaction_ids,
        Symbol.(metabolite_ids);
        concordance_matrix=concordance_matrix,  # Pass pre-built matrix
        activity_ranges=activity_ranges,
        concordance_modules=module_mapping,
        interface_reactions=falses(length(reaction_ids)),
        acr_metabolites=Symbol[],
        lambda_dict=lambda_dict,
        stats=intermediate_stats  # Use intermediate stats for now
    )

    # NOW calculate the final concordance statistics from the actual concordance matrix
    final_stats = calculate_final_concordance_statistics!(
        complete_model,
        intermediate_stats,
        use_transitivity
    )

    # Update the complete model with the final statistics
    complete_model.stats = final_stats

    # Apply kinetic analysis if requested
    if kinetic_analysis
        apply_kinetic_analysis!(complete_model, constraints)
    end

    return complete_model
end


"""
Update pair concordance state with new solver result.
Returns true if pair is still potentially concordant, false if non-concordant.
"""
@inline function update_pair_concordance!(
    state::PairConcordanceState,
    direction::Symbol,
    dir_multiplier::Int,
    value::Float64,
    timeout::Bool,
    concordance_tolerance::Float64
)::Bool
    # Early exit if timeout
    if timeout
        state.has_timeout = true
        state.is_concordant = false
        return false
    end

    # Update min/max values for this direction
    if direction == :positive
        if dir_multiplier == -1
            state.positive_min = value
        else
            state.positive_max = value
        end
        state.has_positive = true
    else # direction == :negative
        if dir_multiplier == -1
            state.negative_min = value
        else
            state.negative_max = value
        end
        state.has_negative = true
    end

    # Check concordance when we have both min/max for a direction
    if direction == :positive && !isnan(state.positive_min) && !isnan(state.positive_max)
        # Check if min/max are concordant within tolerance
        if abs(state.positive_min - state.positive_max) > concordance_tolerance
            state.is_concordant = false
            return false
        end
        # Check lambda consistency
        current_lambda = (state.positive_min + state.positive_max) / 2
        if isnan(state.reference_lambda)
            state.reference_lambda = current_lambda
        elseif abs(current_lambda - state.reference_lambda) > concordance_tolerance
            state.is_concordant = false
            return false
        end
    elseif direction == :negative && !isnan(state.negative_min) && !isnan(state.negative_max)
        # Check if min/max are concordant within tolerance
        if abs(state.negative_min - state.negative_max) > concordance_tolerance
            state.is_concordant = false
            return false
        end
        # Check lambda consistency
        current_lambda = (state.negative_min + state.negative_max) / 2
        if isnan(state.reference_lambda)
            state.reference_lambda = current_lambda
        elseif abs(current_lambda - state.reference_lambda) > concordance_tolerance
            state.is_concordant = false
            return false
        end
    end

    # Mark as concordant if all tested directions are complete and valid
    has_complete_positive = state.has_positive && !isnan(state.positive_min) && !isnan(state.positive_max)
    has_complete_negative = state.has_negative && !isnan(state.negative_min) && !isnan(state.negative_max)

    if (state.has_positive || state.has_negative) && (has_complete_positive || has_complete_negative)
        # Only mark concordant if all tested directions are complete
        if (!state.has_positive || has_complete_positive) && (!state.has_negative || has_complete_negative)
            state.is_concordant = true
        end
    end

    return state.is_concordant
end

function process_solver_results_streaming!(
    solver_results,
    batch_pairs::Vector{PairCandidate},
    concordance_tracker,
    concordance_tolerance::Float64;
    track_direct_pairs::Bool=false
)
    # Pre-allocate lookup for O(1) pair finding
    pair_lookup = Dict{Tuple{Symbol,Symbol},Int}()
    idx_to_id = concordance_tracker.idx_to_id

    for (i, candidate) in enumerate(batch_pairs)
        c1_id = idx_to_id[candidate.c1_idx]
        c2_id = idx_to_id[candidate.c2_idx]
        pair_lookup[(c1_id, c2_id)] = i
    end

    # Minimal state tracking - O(n) memory
    pair_states = [PairConcordanceState() for _ in 1:length(batch_pairs)]

    # Process each solver result immediately - zero intermediate storage
    for result in solver_results
        c1_id, c2_id, direction, dir_multiplier, value, timeout = result
        pair_idx = get(pair_lookup, (c1_id, c2_id), 0)
        pair_idx == 0 && continue

        # Update concordance state directly - no allocations
        update_pair_concordance!(pair_states[pair_idx], direction, dir_multiplier, value, timeout, concordance_tolerance)
    end

    # Final concordance decision and tracker update
    concordant_count = 0
    non_concordant_count = 0
    timeout_count = 0
    direct_concordant_pairs = track_direct_pairs ? Set{Tuple{Int,Int}}() : nothing

    for (i, state) in enumerate(pair_states)
        candidate = batch_pairs[i]

        if state.has_timeout
            timeout_count += 1
        end

        if state.is_concordant
            union_sets!(concordance_tracker, candidate.c1_idx, candidate.c2_idx)
            concordant_count += 1

            # Track this pair if requested
            if track_direct_pairs
                pair = candidate.c1_idx <= candidate.c2_idx ? (candidate.c1_idx, candidate.c2_idx) : (candidate.c2_idx, candidate.c1_idx)
                push!(direct_concordant_pairs, pair)
            end
        else
            add_non_concordant!(concordance_tracker, candidate.c1_idx, candidate.c2_idx)
            non_concordant_count += 1
        end
    end

    return concordant_count, non_concordant_count, timeout_count, direct_concordant_pairs
end

"""
Iterate directions from bit flags with minimal allocations.
"""
@inline function iterate_directions(bits::UInt8)::Tuple{Vararg{Symbol}}
    pos = (bits & 0x01) != 0
    neg = (bits & 0x02) != 0

    if pos && neg
        return (:positive, :negative)
    elseif pos
        return (:positive,)
    elseif neg
        return (:negative,)
    else
        return ()
    end
end

"""
Process streaming candidates with deterministic batch processing.
"""
function process_streaming_batches(
    constraints::C.ConstraintTree,
    streaming_filter::StreamingCandidateFilter,
    concordance_tracker::ConcordanceTracker,
    config::BatchProcessingConfig;
    track_direct_pairs::Bool=false
)
    state = initialize_processing_state(streaming_filter, concordance_tracker, config)
    all_direct_pairs = track_direct_pairs ? Set{Tuple{Int,Int}}() : nothing

    @info "Using fixed batch size: $(config.batch_size) candidates per batch"
    @info "Collecting candidates from streaming filter..."

    # Main processing loop
    for candidate in streaming_filter
        collect_candidate!(state, candidate)

        if should_process_batch(state, config)
            batch_direct_pairs = process_full_batch!(constraints, state, concordance_tracker, config, streaming_filter; track_direct_pairs=track_direct_pairs)
            # Collect direct pairs from this batch if tracking
            if track_direct_pairs && batch_direct_pairs !== nothing
                union!(all_direct_pairs, batch_direct_pairs)
            end
        end
    end

    # Process any remaining candidates in final batch
    if !isempty(state.current_batch)
        batch_direct_pairs = process_full_batch!(constraints, state, concordance_tracker, config, streaming_filter; track_direct_pairs=track_direct_pairs)
        # Collect direct pairs from this batch if tracking
        if track_direct_pairs && batch_direct_pairs !== nothing
            union!(all_direct_pairs, batch_direct_pairs)
        end
    end

    results = build_final_results(state)

    # Add direct pairs to results if we were tracking them
    if track_direct_pairs
        return merge(results, (direct_concordant_pairs=all_direct_pairs,))
    else
        return results
    end
end


function build_final_results(state::BatchProcessingState)
    # Convert optimization results to Dict for compatibility
    opt_results_dict = Dict{Tuple{Int,Int,Symbol},Float64}()
    for (i, j, dir, val) in state.accumulator.optimization_results
        opt_results_dict[(i, j, dir)] = val
    end

    return (
        batches_completed=state.total_batches_completed,
        pairs_processed=state.total_pairs_processed,
        concordant_pairs=state.accumulator.concordant_pairs,
        concordant_count=state.accumulator.counts.concordant,  # Add total concordant count
        non_concordant_pairs=state.accumulator.counts.non_concordant,
        skipped_by_transitivity=state.accumulator.counts.skipped,
        transitive_pairs=state.accumulator.counts.transitive,
        timeout_pairs=state.accumulator.counts.timeout,
        infeasible_pairs=state.accumulator.counts.infeasible,
        unbounded_pairs=state.accumulator.counts.unbounded,
        infeasible_or_unbounded_pairs=state.accumulator.counts.infeasible_or_unbounded,
        resource_limit_pairs=state.accumulator.counts.resource_limit,
        numerical_error_pairs=state.accumulator.counts.numerical_error,
        other_error_pairs=state.accumulator.counts.other_error,
        optimization_results=opt_results_dict,
        total_optimizations=state.accumulator.counts.total_optimizations
    )
end



# Define concrete types for better type stability
# Use NaN as sentinel value instead of Nothing for better performance
const OptValue = Float64  # NaN represents missing/invalid values
const TestResult = Tuple{OptValue,OptValue,Bool}  # (min_val, max_val, timeout)
const DirectionResult = Tuple{Symbol,TestResult}  # (direction, test_result)

# --- Memory-efficient result pools ---

"""
Object pool for reusing optimization result containers.
Reduces memory allocations during batch processing.
"""
mutable struct ResultPool{T}
    pool::Vector{T}
    used::Int
    constructor::Function
end

function ResultPool(constructor::Function, initial_size::Int)
    pool = [constructor() for _ in 1:initial_size]
    ResultPool(pool, 0, constructor)
end

function get_result!(pool::ResultPool{T}) where T
    if pool.used < length(pool.pool)
        pool.used += 1
        return pool.pool[pool.used]
    else
        # Expand pool if needed
        new_result = pool.constructor()
        push!(pool.pool, new_result)
        pool.used = length(pool.pool)
        return new_result
    end
end

function reset_pool!(pool::ResultPool)
    pool.used = 0
    # Clear all containers for reuse
    for item in pool.pool
        if item isa Dict
            empty!(item)
        elseif item isa Vector
            empty!(item)
        end
    end
end

# Thread-local result pools to avoid contention
const RESULT_DICT_POOL = ResultPool(() -> Dict{Tuple{Symbol,Symbol,Symbol},Dict{Int,Tuple{OptValue,Bool}}}(), 4)
const PAIR_DICT_POOL = ResultPool(() -> Dict{Tuple{Symbol,Symbol},Vector{DirectionResult}}(), 4)

function process_optimization_results(optimization_results, batch_pairs::Vector{PairCandidate}, concordance_tracker)
    # Get pooled dictionaries to avoid allocations
    result_dict = get_result!(RESULT_DICT_POOL)
    pair_results_dict = get_result!(PAIR_DICT_POOL)

    for result in optimization_results
        c1_id, c2_id, direction, dir_multiplier, actual_value, timeout = result
        key = (c1_id, c2_id, direction)
        if !haskey(result_dict, key)
            result_dict[key] = Dict{Int,Tuple{OptValue,Bool}}()
        end
        result_dict[key][dir_multiplier] = (actual_value, timeout)
    end

    # Convert to pair format - reusing pooled container
    for ((c1_id, c2_id, direction), minmax_results) in result_dict
        key = (c1_id, c2_id)
        if !haskey(pair_results_dict, key)
            pair_results_dict[key] = DirectionResult[]
        end

        min_result = get(minmax_results, -1, (NaN, false))
        max_result = get(minmax_results, +1, (NaN, false))
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

    for (i, candidate) in enumerate(batch_pairs)
        c1_id, c2_id = idx_to_id[candidate.c1_idx], idx_to_id[candidate.c2_idx]
        batch_results[i] = get(pair_results_dict, (c1_id, c2_id), DirectionResult[])
    end

    # Reset pools for reuse (but don't return them yet - they'll be cleared automatically)
    # This prevents memory accumulation across many batches
    if length(result_dict) > 1000  # Reset when getting too large
        empty!(result_dict)
        empty!(pair_results_dict)
    end

    return batch_results
end

"""
Process a batch of concordance tests using parallel optimization.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    batch_pairs::Vector{PairCandidate},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    concordance_tolerance::Float64,
    track_direct_pairs::Bool=false
)

    # Create COBREXA-style test array with direction multipliers directly from PairCandidate
    test_array = []
    debug_candidate_pairs = Set{Tuple{Symbol,Symbol}}()

    for candidate in batch_pairs
        c1_id = concordance_tracker.idx_to_id[candidate.c1_idx]
        c2_id = concordance_tracker.idx_to_id[candidate.c2_idx]

        # Track all candidate pairs being tested
        push!(debug_candidate_pairs, (c1_id, c2_id))

        # Iterate over directions using bit flags
        for direction in iterate_directions(candidate.directions_bits)
            # Add both MIN (-1) and MAX (+1) tests
            push!(test_array, (c1_id, c2_id, direction, -1))  # MIN test
            push!(test_array, (c1_id, c2_id, direction, +1))  # MAX test
        end
    end

    # Process using pure COBREXA pattern
    optimization_results = screen_directions_optimization_model(
        (models_cache, (c1_id, c2_id, direction, dir_multiplier)) -> begin
            @debug "Processing pair in optimization function" c1_id c2_id direction dir_multiplier
            om = direction == :positive ? models_cache.positive : models_cache.negative

            # Check if model creation failed
            if om === nothing
                println("NULL MODEL: Worker failed to create optimization model for $(c1_id), $(c2_id)")
                return (c1_id, c2_id, direction, dir_multiplier, NaN, J.OPTIMIZE_NOT_CALLED)
            end
            try
                # Check if activities exist for both complexes
                if !haskey(constraints.activities, c1_id)
                    @warn("MISSING ACTIVITY: c1_id=$(c1_id)")
                    return (c1_id, c2_id, direction, dir_multiplier, NaN, J.OPTIMIZE_NOT_CALLED)
                end
                if !haskey(constraints.activities, c2_id)
                    @warn("MISSING ACTIVITY: c2_id=$(c2_id)")
                    return (c1_id, c2_id, direction, dir_multiplier, NaN, J.OPTIMIZE_NOT_CALLED)
                end

                # Set c2 constraint to 1.0 and optimize c1
                @debug "About to set constraint for" c2_id
                c2_constraint = J.@constraint(om, c2_constraint,
                    C.substitute(constraints.activities[c2_id].value, om[:x]) == 1.0)
                @debug "Constraint set successfully"

                @debug "About to set objective for" c1_id dir_multiplier
                J.@objective(om, J.MAX_SENSE,
                    C.substitute(dir_multiplier * constraints.activities[c1_id].value, om[:x]))
                @debug "Objective set successfully"

                @debug "About to optimize"
                J.optimize!(om)
                @debug "Optimization completed"

                # Check termination status immediately after optimization
                immediate_status = J.termination_status(om)
                @debug "Termination status after optimize!" status = immediate_status
                @debug "About to use immediate_status" immediate_status

                # Get result (use NaN for invalid/missing values)
                raw_value = if J.termination_status(om) in (J.OPTIMAL, J.LOCALLY_SOLVED) && J.is_solved_and_feasible(om)
                    J.objective_value(om)
                else
                    NaN
                end

                # Convert back: if we maximized -f(x), negate result to get min f(x)
                actual_value = dir_multiplier == -1 ? (!isnan(raw_value) ? -raw_value : NaN) : raw_value

                # Cleanup
                J.delete(om, c2_constraint)
                J.unregister(om, :c2_constraint)
                result = (c1_id, c2_id, direction, dir_multiplier, actual_value, immediate_status)
                @debug "Returning result" result
                return result
            catch e
                @warn "Optimization error" c1_id c2_id direction error = string(e)
                return (c1_id, c2_id, direction, dir_multiplier, NaN, J.OTHER_ERROR)
            end
        end,
        constraints,
        test_array;
        optimizer=optimizer,
        settings=settings,
        workers=workers
    )

    # OPTIMIZED: Process results directly using efficient streaming processor
    # Eliminates all intermediate data structures and multiple processing passes

    # Pre-allocate lookup for O(1) pair finding
    pair_lookup = Dict{Tuple{Symbol,Symbol},Int}()
    idx_to_id = concordance_tracker.idx_to_id

    for (i, candidate) in enumerate(batch_pairs)
        c1_id = idx_to_id[candidate.c1_idx]
        c2_id = idx_to_id[candidate.c2_idx]
        pair_lookup[(c1_id, c2_id)] = i
    end

    # Minimal state tracking - O(n) memory
    pair_states = [PairConcordanceState() for _ in 1:length(batch_pairs)]

    # Track failure statistics by pair for counting
    pair_failure_counts = Dict{Int,Dict{Symbol,Int}}()
    for i in 1:length(batch_pairs)
        pair_failure_counts[i] = Dict(:timeout => 0, :infeasible => 0, :unbounded => 0,
            :infeasible_or_unbounded => 0, :resource_limit => 0,
            :numerical_error => 0, :other_error => 0, :success => 0)
    end

    # Debug: Track actual termination status frequencies and total optimizations
    status_frequency = Dict{Any,Int}()
    total_optimizations_attempted = 0

    # Process each solver result
    for result in optimization_results
        total_optimizations_attempted += 1
        c1_id, c2_id, direction, dir_multiplier, value, termination_status = result
        pair_idx = get(pair_lookup, (c1_id, c2_id), 0)
        pair_idx == 0 && continue

        # Categorize the termination status
        status_category = categorize_termination_status(termination_status)
        timeout = (status_category == :timeout)

        # Debug: Track termination status frequency
        status_frequency[termination_status] = get(status_frequency, termination_status, 0) + 1

        # Track failure statistics per pair
        pair_failure_counts[pair_idx][status_category] += 1

        # Update concordance state directly - no allocations
        update_pair_concordance!(pair_states[pair_idx], direction, dir_multiplier, value, timeout, concordance_tolerance)
    end

    # OPTIMIZED: Update concordance tracker directly and return counts only
    # Eliminates the need for process_batch_results! entirely

    concordant_count = 0
    non_concordant_count = 0
    timeout_count = 0
    infeasible_count = 0
    unbounded_count = 0
    infeasible_or_unbounded_count = 0
    resource_limit_count = 0
    numerical_error_count = 0
    other_error_count = 0
    optimization_results_vec = Vector{Tuple{Int,Int,Symbol,Float64}}()
    direct_concordant_pairs = track_direct_pairs ? Set{Tuple{Int,Int}}() : nothing

    for (i, state) in enumerate(pair_states)
        candidate = batch_pairs[i]
        c1_idx, c2_idx = candidate.c1_idx, candidate.c2_idx

        # Count different failure types for this pair
        failure_stats = pair_failure_counts[i]
        failure_stats[:timeout] > 0 && (timeout_count += 1)
        failure_stats[:infeasible] > 0 && (infeasible_count += 1)
        failure_stats[:unbounded] > 0 && (unbounded_count += 1)
        failure_stats[:infeasible_or_unbounded] > 0 && (infeasible_or_unbounded_count += 1)
        failure_stats[:resource_limit] > 0 && (resource_limit_count += 1)
        failure_stats[:numerical_error] > 0 && (numerical_error_count += 1)
        failure_stats[:other_error] > 0 && (other_error_count += 1)

        # Update concordance tracker directly based on results
        if state.is_concordant
            union_sets!(concordance_tracker, c1_idx, c2_idx)
            concordant_count += 1

            # Track this pair if requested
            if track_direct_pairs
                pair = c1_idx <= c2_idx ? (c1_idx, c2_idx) : (c2_idx, c1_idx)
                push!(direct_concordant_pairs, pair)
            end

            # Store lambda result if valid
            if !isnan(state.reference_lambda)
                final_direction = if state.has_positive && state.has_negative
                    :both
                elseif state.has_positive
                    :positive
                else
                    :negative
                end
                push!(optimization_results_vec, (c1_idx, c2_idx, final_direction, state.reference_lambda))
            end
        else
            add_non_concordant!(concordance_tracker, c1_idx, c2_idx)
            non_concordant_count += 1
        end
    end

    # Debug: Log termination status frequency and total optimizations for troubleshooting
    @debug "Optimization statistics" total_optimizations_attempted status_frequency

    # Return counts in same format as process_batch_results! expected
    return (
        concordant_count=concordant_count,
        non_concordant_count=non_concordant_count,
        timeout_count=timeout_count,
        infeasible_count=infeasible_count,
        unbounded_count=unbounded_count,
        infeasible_or_unbounded_count=infeasible_or_unbounded_count,
        resource_limit_count=resource_limit_count,
        numerical_error_count=numerical_error_count,
        other_error_count=other_error_count,
        total_optimizations=total_optimizations_attempted,
        optimization_results=optimization_results_vec,
        direct_concordant_pairs=direct_concordant_pairs
    )
end


"""
$(TYPEDSIGNATURES)

Custom screening function for concordance analysis that handles directional constraints.

This function is optimized for concordance testing by caching both positive and negative 
Charnes-Cooper constraint models separately, avoiding the need to repeatedly modify 
constraint bounds on cached models during optimization.
"""
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
            try
                pos_const, neg_const = constraints_tuple
                pos_model = COBREXA.optimization_model(pos_const; optimizer=optimizer)
                neg_model = COBREXA.optimization_model(neg_const; optimizer=optimizer)
                for s in [COBREXA.configuration.default_solver_settings; settings]
                    s(pos_model)
                    s(neg_model)
                end
                return (positive=pos_model, negative=neg_model)
            catch e
                @error "WORKER FAILED TO CREATE MODELS" worker_id = myid() exception = e
                return (positive=nothing, negative=nothing)
            end
        end,
        (pos_constraints, neg_constraints)
    )

    D.pmap(
        (as...) -> f(COBREXA.get_worker_local_data(worker_cache), as...),
        D.CachingPool(workers),
        args...,
    )
end

"""
$(TYPEDSIGNATURES)

Extract concordance modules from the ConcordanceTracker.
"""
function extract_modules(tracker::ConcordanceTracker)
    modules = Dict{Int,Vector{Symbol}}()

    for (id, idx) in tracker.id_to_idx
        root = find_set!(tracker, idx)
        if !haskey(modules, root)
            modules[root] = Symbol[]
        end
        push!(modules[root], id)
    end

    return modules
end




"""
$(TYPEDSIGNATURES)

Find trivially balanced complexes directly from complex compositions.
A complex is trivially balanced if all its metabolites appear in only this complex.
"""
function find_trivially_balanced_complexes(
    complexes::Dict{Symbol,Vector{Tuple{Symbol,Float64}}}
)::Set{Symbol}
    # Build metabolite participation map
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    for (complex_id, metabolite_composition) in complexes
        for (met_id, _) in metabolite_composition
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    # Find complexes that have at least one metabolite unique to that complex
    balanced_complexes = Set{Symbol}()
    for (complex_id, metabolite_composition) in complexes
        for (met_id, _) in metabolite_composition
            if length(metabolite_participation[met_id]) == 1
                push!(balanced_complexes, complex_id)
                break  # Found at least one unique metabolite, complex is balanced
            end
        end
    end

    return balanced_complexes
end

"""
$(TYPEDSIGNATURES)

Find trivially balanced complexes using sparse matrix operations for memory efficiency.
A complex is trivially balanced if all its metabolites appear in only this complex.
"""
function find_trivially_balanced_complexes_sparse(
    Y_matrix::SparseArrays.SparseMatrixCSC,
    metabolite_ids::Vector{Symbol},
    complex_ids::Vector{Symbol}
)::Set{Symbol}
    # For each complex, check if it has any metabolites that appear only in this complex
    n_metabolites, n_complexes = size(Y_matrix)

    balanced_complexes = Set{Symbol}()

    # For each complex, check if it has any metabolites that appear only in this complex
    for (j, complex_id) in enumerate(complex_ids)
        # Get metabolites in this complex (non-zero entries in column j)
        metabolite_indices = SparseArrays.findnz(Y_matrix[:, j])[1]

        # Check if any of these metabolites appear in only this complex
        for i in metabolite_indices
            # Count non-zero entries in row i (how many complexes this metabolite participates in)
            row_nnz = SparseArrays.nnz(Y_matrix[i, :])
            if row_nnz == 1
                # This metabolite appears only in this complex
                push!(balanced_complexes, complex_id)
                break  # Found at least one unique metabolite, complex is balanced
            end
        end
    end

    return balanced_complexes
end

"""
$(TYPEDSIGNATURES)

Extend trivially balanced complexes using sparse matrix operations for memory efficiency.
Identifies complexes connected to balanced complexes through metabolites that only appear 
in balanced complexes plus one additional complex.
"""
function extend_trivially_balanced_complexes_sparse(
    Y_matrix::SparseArrays.SparseMatrixCSC,
    metabolite_ids::Vector{Symbol},
    complex_ids::Vector{Symbol},
    initial_balanced::Set{Symbol}
)::Set{Symbol}
    # Build index mappings
    complex_to_idx = Dict(comp => j for (j, comp) in enumerate(complex_ids))

    # Create boolean mask for currently balanced complexes
    n_complexes = length(complex_ids)
    extended_balanced = copy(initial_balanced)
    newly_added = true
    iteration = 0

    while newly_added && iteration < 10  # Safety limit
        newly_added = false
        iteration += 1

        # Create current balanced mask
        balanced_mask = falses(n_complexes)
        for complex_id in extended_balanced
            if haskey(complex_to_idx, complex_id)
                balanced_mask[complex_to_idx[complex_id]] = true
            end
        end

        # For each metabolite, check if it appears only in balanced complexes + one other
        for (i, met_id) in enumerate(metabolite_ids)
            # Get complexes this metabolite participates in (non-zero entries in row i)
            complex_indices = SparseArrays.findnz(Y_matrix[i, :])[1]

            if length(complex_indices) >= 2
                # Count balanced vs unbalanced complexes for this metabolite
                balanced_count = count(j -> balanced_mask[j], complex_indices)
                unbalanced_indices = filter(j -> !balanced_mask[j], complex_indices)

                # If exactly one unbalanced complex and all others are balanced
                if length(unbalanced_indices) == 1 && balanced_count == length(complex_indices) - 1
                    new_balanced_idx = unbalanced_indices[1]
                    new_balanced_id = complex_ids[new_balanced_idx]
                    push!(extended_balanced, new_balanced_id)
                    newly_added = true
                    @debug "Extended trivially balanced: added $new_balanced_id (connected via metabolite $met_id)"
                end
            end
        end
    end

    if length(extended_balanced) > length(initial_balanced)
        @info "Extended trivially balanced complexes" initial = length(initial_balanced) final = length(extended_balanced) added = length(extended_balanced) - length(initial_balanced)
    end

    return extended_balanced
end

"""
$(TYPEDSIGNATURES)

Extend trivially balanced complexes by identifying complexes connected to balanced complexes
through metabolites that only appear in balanced complexes plus one additional complex.

If a metabolite appears only in balanced complexes and one other complex (not yet identified
as trivially balanced), then that additional complex should also be trivially balanced.
"""
function extend_trivially_balanced_complexes(
    complexes::Dict{Symbol,Vector{Tuple{Symbol,Float64}}},
    initial_balanced::Set{Symbol}
)::Set{Symbol}
    # Build metabolite participation map
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    for (complex_id, metabolite_composition) in complexes
        for (met_id, _) in metabolite_composition
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    # Get metabolites that appear in trivially balanced complexes
    balanced_metabolites = Set{Symbol}()
    for balanced_complex in initial_balanced
        if haskey(complexes, balanced_complex)
            for (met_id, _) in complexes[balanced_complex]
                push!(balanced_metabolites, met_id)
            end
        end
    end

    extended_balanced = copy(initial_balanced)
    newly_added = true
    iteration = 0

    # Iterate until no new complexes are added
    while newly_added && iteration < 10  # Safety limit to prevent infinite loops
        newly_added = false
        iteration += 1

        # Only check metabolites that appear in currently balanced complexes
        current_balanced_metabolites = Set{Symbol}()
        for balanced_complex in extended_balanced
            if haskey(complexes, balanced_complex)
                for (met_id, _) in complexes[balanced_complex]
                    push!(current_balanced_metabolites, met_id)
                end
            end
        end

        # Check each metabolite from balanced complexes
        for met_id in current_balanced_metabolites
            if haskey(metabolite_participation, met_id)
                participating_complexes = metabolite_participation[met_id]
                if length(participating_complexes) >= 2
                    # Count how many of the participating complexes are already balanced
                    balanced_count = count(c -> c in extended_balanced, participating_complexes)
                    unbalanced_complexes = filter(c -> c ∉ extended_balanced, participating_complexes)

                    # If exactly one unbalanced complex remains and all others are balanced
                    if length(unbalanced_complexes) == 1 && balanced_count == length(participating_complexes) - 1
                        new_balanced = unbalanced_complexes[1]
                        push!(extended_balanced, new_balanced)
                        newly_added = true
                        @debug "Extended trivially balanced: added $new_balanced (connected via metabolite $met_id)"
                    end
                end
            end
        end
    end

    if length(extended_balanced) > length(initial_balanced)
        @info "Extended trivially balanced complexes" initial = length(initial_balanced) final = length(extended_balanced) added = length(extended_balanced) - length(initial_balanced)
    end

    return extended_balanced
end

"""
$(TYPEDSIGNATURES)

Find trivially concordant pairs directly from complex compositions.
Pairs are trivially concordant if they share exactly the same metabolite composition.
"""
function find_trivially_concordant_pairs(
    complexes::Dict{Symbol,Vector{Tuple{Symbol,Float64}}}
)::Set{Tuple{Symbol,Symbol}}
    # Build metabolite participation map
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    for (complex_id, metabolite_composition) in complexes
        for (met_id, _) in metabolite_composition
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    # Find pairs that share a metabolite that only participates in those two complexes
    concordant_pairs = Set{Tuple{Symbol,Symbol}}()

    for (met_id, participating_complexes) in metabolite_participation
        if length(participating_complexes) == 2
            # This metabolite only participates in exactly two complexes
            complex1, complex2 = participating_complexes[1], participating_complexes[2]
            pair = complex1 < complex2 ? (complex1, complex2) : (complex2, complex1)
            push!(concordant_pairs, pair)
        end
    end

    return concordant_pairs
end

"""
$(TYPEDSIGNATURES)

Find trivially concordant pairs using sparse matrix operations for memory efficiency.
Pairs are trivially concordant if they share a metabolite that appears in exactly two complexes.
"""
function find_trivially_concordant_pairs_sparse(
    Y_matrix::SparseArrays.SparseMatrixCSC,
    metabolite_ids::Vector{Symbol},
    complex_ids::Vector{Symbol}
)::Set{Tuple{Symbol,Symbol}}
    concordant_pairs = Set{Tuple{Symbol,Symbol}}()

    # For each metabolite (row), check if it appears in exactly two complexes
    for (i, met_id) in enumerate(metabolite_ids)
        # Get complexes this metabolite participates in (non-zero entries in row i)
        complex_indices = SparseArrays.findnz(Y_matrix[i, :])[1]

        if length(complex_indices) == 2
            # This metabolite appears in exactly two complexes
            complex1_id = complex_ids[complex_indices[1]]
            complex2_id = complex_ids[complex_indices[2]]
            pair = complex1_id < complex2_id ? (complex1_id, complex2_id) : (complex2_id, complex1_id)
            push!(concordant_pairs, pair)
        end
    end

    return concordant_pairs
end

"""
Build complete concordance matrix in one efficient step with all relationships.
This is the most efficient approach - single sparse matrix construction with all data.
"""
function build_complete_concordance_matrix(
    n_complexes::Int,
    complex_idx::Dict{Symbol,Int},
    trivially_balanced::Set{Symbol},
    trivial_pairs::Set{Tuple{Symbol,Symbol}},
    concordance_tracker::ConcordanceTracker,
    balanced_complexes::BitVector,
    use_transitivity::Bool,
    batch_results::Union{Nothing,NamedTuple}
)
    # Collect all matrix entries for single batch construction
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Int[]

    # Pre-allocate with generous estimate to avoid repeated allocations
    estimated_balanced = length(trivially_balanced) + count(balanced_complexes)
    estimated_pairs = length(trivial_pairs) + n_complexes ÷ 4  # Conservative estimate
    total_estimate = estimated_balanced + estimated_pairs
    sizehint!(I_vals, total_estimate)
    sizehint!(J_vals, total_estimate)
    sizehint!(V_vals, total_estimate)

    # 1. Add trivially balanced complexes to diagonal (Trivially_balanced = 4)
    # These have highest priority and shouldn't be overwritten
    for complex_id in trivially_balanced
        idx = get(complex_idx, complex_id, 0)
        if idx > 0
            push!(I_vals, idx)
            push!(J_vals, idx)
            push!(V_vals, Int(Trivially_balanced))  # 4
        end
    end

    # Track which diagonal positions are already taken
    trivially_balanced_indices = Set{Int}()
    for complex_id in trivially_balanced
        idx = get(complex_idx, complex_id, 0)
        if idx > 0
            push!(trivially_balanced_indices, idx)
        end
    end

    # 2. Add other balanced complexes to diagonal (Balanced = 3)
    # Only if not already trivially balanced
    for i in 1:n_complexes
        if balanced_complexes[i] && i ∉ trivially_balanced_indices
            push!(I_vals, i)
            push!(J_vals, i)
            push!(V_vals, Int(Balanced))  # 3
        end
    end

    # 3. Add discovered concordant pairs (Concordant = 1) FIRST
    if use_transitivity
        @debug "Using transitivity: extracting pairs from concordance modules"
        # Extract modules from concordance tracker
        root_to_complexes = Dict{Int,Vector{Int}}()
        sizehint!(root_to_complexes, n_complexes ÷ 4)

        for i in 1:n_complexes
            root = find_set!(concordance_tracker, i)
            group = get!(root_to_complexes, root) do
                Vector{Int}()
            end
            push!(group, i)
        end

        # Create set of trivial pair positions to avoid overwriting them
        trivial_positions = Set{Tuple{Int,Int}}()
        for (id1, id2) in trivial_pairs
            i = get(complex_idx, id1, 0)
            j = get(complex_idx, id2, 0)
            if i > 0 && j > 0 && i != j
                upper_i, upper_j = i <= j ? (i, j) : (j, i)
                push!(trivial_positions, (upper_i, upper_j))
            end
        end

        # Add concordant pairs from modules, but skip trivial pairs
        concordant_pairs_added = 0
        for complex_group in values(root_to_complexes)
            group_size = length(complex_group)
            if group_size > 1
                for i in 1:group_size-1
                    for j in i+1:group_size
                        idx1, idx2 = complex_group[i], complex_group[j]
                        # Ensure upper triangular ordering
                        upper_i, upper_j = idx1 <= idx2 ? (idx1, idx2) : (idx2, idx1)

                        # Skip if this is a trivial pair (should remain value 2)
                        if (upper_i, upper_j) ∉ trivial_positions
                            push!(I_vals, upper_i)
                            push!(J_vals, upper_j)
                            push!(V_vals, Int(Concordant))  # 1
                            concordant_pairs_added += 1
                        end
                    end
                end
            end
        end
        @info "Concordant pairs added from modules" concordant_pairs_added total_trivial_skipped = length(trivial_positions)
    else
        @info "Not using transitivity: using direct optimization results only"
        # Use the directly tracked concordant pairs from batch processing

        direct_pairs = haskey(batch_results, :direct_concordant_pairs) ? batch_results.direct_concordant_pairs : nothing
        n_direct_pairs = direct_pairs !== nothing ? length(direct_pairs) : 0

        @info "Batch results debug" concordant_count = (batch_results !== nothing ? batch_results.concordant_count : 0) directly_tracked_pairs = n_direct_pairs

        # Add directly tracked concordant pairs to the matrix
        pairs_added = 0
        if direct_pairs !== nothing
            for (i, j) in direct_pairs
                # Check if this pair is balanced from the balanced_complexes BitVector
                is_balanced = balanced_complexes[i] && balanced_complexes[j]
                if is_balanced
                    push!(I_vals, i)
                    push!(J_vals, j)
                    push!(V_vals, Int(Balanced))  # 3
                    pairs_added += 1
                else
                    push!(I_vals, i)
                    push!(J_vals, j)
                    push!(V_vals, Int(Concordant))  # 1
                    pairs_added += 1
                end
            end
        end

        @info "Added $pairs_added optimization pairs to matrix (found $n_direct_pairs directly tracked concordant pairs)"
    end

    # 4. Add trivial pairs (Trivially_concordant = 2)
    trivial_pairs_added = 0
    for (id1, id2) in trivial_pairs
        i = get(complex_idx, id1, 0)
        j = get(complex_idx, id2, 0)
        if i > 0 && j > 0 && i != j
            # Ensure upper triangular ordering
            upper_i, upper_j = i <= j ? (i, j) : (j, i)
            push!(I_vals, upper_i)
            push!(J_vals, upper_j)
            push!(V_vals, Int(Trivially_concordant))  # 2
            trivial_pairs_added += 1
        end
    end

    # Build complete sparse matrix in one efficient operation
    @info "Matrix construction debug" total_entries = length(I_vals) unique_positions = length(Set(zip(I_vals, J_vals)))
    sparse_matrix = SparseArrays.sparse(I_vals, J_vals, V_vals, n_complexes, n_complexes)
    @info "Sparse matrix created" nnz = SparseArrays.nnz(sparse_matrix) count_1s = count(==(1), sparse_matrix) count_2s = count(==(2), sparse_matrix) count_3s = count(==(3), sparse_matrix) count_4s = count(==(4), sparse_matrix)
    concordance_matrix = LinearAlgebra.UpperTriangular(sparse_matrix)

    return concordance_matrix
end

"""
Calculate final concordance statistics from the completed ConcordanceResults object.
This is the single source of truth for all concordance counts.
"""
function calculate_final_concordance_statistics!(
    results::ConcordanceResults,
    intermediate_stats::Dict,
    use_transitivity::Bool
)
    n_complexes = length(results.complex_ids)

    # Count different types of concordance from the final matrix
    n_concordant_total = 0
    n_trivially_concordant = 0
    n_balanced = 0
    n_trivially_balanced = 0

    # Count from upper triangular matrix (avoid double counting)
    for i in 1:n_complexes
        for j in (i+1):n_complexes  # Only upper triangle
            concordance_type = ConcordanceType(results.concordance_matrix[i, j])

            if concordance_type == Concordant
                n_concordant_total += 1
            elseif concordance_type == Trivially_concordant
                n_trivially_concordant += 1
                n_concordant_total += 1  # Trivially concordant are also concordant
            end
        end

        # Check diagonal for balanced complexes
        diagonal_type = ConcordanceType(results.concordance_matrix[i, i])
        if diagonal_type == Balanced
            n_balanced += 1
        elseif diagonal_type == Trivially_balanced
            n_trivially_balanced += 1
            n_balanced += 1  # Trivially balanced are also balanced
        end
    end

    # Count concordance modules (from the final concordance_modules field)
    module_counts = Dict{Int,Int}()
    for module_id in results.concordance_modules
        if module_id >= 0  # Only count actual modules (-1 = singleton)
            module_counts[module_id] = get(module_counts, module_id, 0) + 1
        end
    end
    n_concordance_modules = count(>(1), values(module_counts))

    # Get optimization counts from the batch processing
    n_directly_tested = intermediate_stats["n_total_optimizations"]
    n_found_via_optimization = intermediate_stats["batch_results_concordant_count"]

    # Calculate inferred count ONLY if transitivity was used
    if use_transitivity
        n_concordant_inferred = n_concordant_total - n_found_via_optimization
    else
        n_concordant_inferred = 0
        # Sanity check: without transitivity, all concordant pairs should be directly found
        # (excluding trivial pairs which are never optimized)
        non_trivial_concordant = n_concordant_total - n_trivially_concordant
        if non_trivial_concordant != n_found_via_optimization
            @warn "Inconsistency detected: non-trivial concordant pairs don't match optimization results" (
                non_trivial_concordant=non_trivial_concordant,
                n_found_via_optimization=n_found_via_optimization,
                difference=non_trivial_concordant - n_found_via_optimization
            )
        end
    end

    # Build complete statistics dict
    final_stats = copy(intermediate_stats)

    # Add the final concordance counts
    final_stats["n_concordant_total"] = n_concordant_total
    final_stats["n_concordant_opt"] = n_found_via_optimization
    final_stats["n_concordant_inferred"] = n_concordant_inferred
    final_stats["n_concordant_found"] = n_concordant_total - n_trivially_concordant  # Non-trivial concordant pairs
    final_stats["n_trivially_concordant"] = n_trivially_concordant
    final_stats["n_balanced_total"] = n_balanced
    final_stats["n_balanced_complexes"] = n_balanced
    final_stats["n_trivially_balanced_complexes"] = n_trivially_balanced
    final_stats["n_concordance_modules"] = n_concordance_modules

    # Validation
    expected_total_concordant = final_stats["n_concordant_found"] + final_stats["n_trivially_concordant"]
    actual_total_concordant = final_stats["n_concordant_total"]
    validation_passed = (expected_total_concordant == actual_total_concordant)

    if !validation_passed
        @warn "Concordant pair accounting mismatch detected!" (
            expected_total=expected_total_concordant,
            actual_total=actual_total_concordant,
            found_pairs=final_stats["n_concordant_found"],
            trivial_pairs=final_stats["n_trivial_pairs"],
            via_direct_testing=final_stats["n_concordant_opt"],
            via_inference=final_stats["n_concordant_inferred"],
            difference=actual_total_concordant - expected_total_concordant
        )
    else
        @debug "Concordant pair validation passed" (
            total_concordant=actual_total_concordant,
            expected_concordant=expected_total_concordant
        )
    end

    final_stats["validation_passed"] = validation_passed

    # Clear summary of the analysis results
    @info "Concordance analysis complete"
    @info "  Total concordant pairs: $(final_stats["n_concordant_total"])"
    @info "  Breakdown: $(final_stats["n_concordant_opt"]) from optimization, $(final_stats["n_concordant_inferred"]) inferred, $(final_stats["n_trivially_concordant"]) trivial"
    @info "  Processing: $(final_stats["n_candidate_pairs"]) candidates, $(final_stats["n_candidates_skipped_by_transitivity"]) filtered by transitivity"
    @info "  Elapsed time: $(Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, final_stats["elapsed_time"] * 1000)), "HH:MM:SS.s"))"

    @debug "Full concordance analysis statistics" final_stats

    return final_stats
end