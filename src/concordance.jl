"""
Main analysis algorithms for COCOA.

This module contains:
- Batch processing with transitivity optimization
- Concordance analysis using unified batch_size parameter
- Main concordance analysis function
- Module extraction and result formatting
"""

# Note: All imports are handled via the main COCOA.jl module

# ========================================================================================
# Memory-Efficient Result Accumulator
# ========================================================================================

"""
Mutable container for batch counts with concrete types.
"""
mutable struct MutableCounts
    concordant::Int
    non_concordant::Int
    timeout::Int
    transitive::Int
    skipped::Int
end

MutableCounts() = MutableCounts(0, 0, 0, 0, 0)

"""
Configuration for batch processing with concrete types.
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
    transitive::Int
    skipped::Int
    optimization_results::Vector{Tuple{Int,Int,Symbol,Float64}}
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
    acc.counts.transitive += counts.transitive
    acc.counts.skipped += counts.skipped

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
    acc.counts.non_concordant = 0
    acc.counts.timeout = 0
    acc.counts.transitive = 0
    acc.counts.skipped = 0
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
Suggest batch size based on memory pressure.
"""
function suggest_batch_size(monitor::MemoryMonitor, current_batch_size::Int)::Int
    current_mem = check_memory!(monitor)
    mem_pressure = (current_mem - monitor.initial_memory) / monitor.initial_memory

    if mem_pressure > 0.8  # High pressure
        return max(10, current_batch_size ÷ 2)
    elseif mem_pressure < 0.3  # Low pressure
        return min(1000, current_batch_size * 2)
    else
        return current_batch_size
    end
end

# --- Memory monitoring utilities ---

"""
Get available system memory in GB.
"""
function get_available_memory_gb()::Float64
    try
        if Sys.islinux()
            # Read from /proc/meminfo on Linux
            meminfo = read("/proc/meminfo", String)
            for line in split(meminfo, '\n')
                if startswith(line, "MemAvailable:")
                    kb = parse(Int, split(line)[2])
                    return kb / (1024^2)  # Convert KB to GB
                end
            end
        elseif Sys.iswindows()
            # Use WMI on Windows (fallback to rough estimate)
            return 8.0  # Conservative default for Windows
        elseif Sys.isapple()
            # Use vm_stat on macOS (simplified)
            return 8.0  # Conservative default for macOS  
        end
    catch
        # Fallback if system memory detection fails
    end
    return 4.0  # Conservative fallback
end

const DEFAULT_SOLVER_TOLERANCE = 1e-6
const MAX_VALID_TOLERANCE = 1e-3

const TOLERANCE_ATTRIBUTES = (
    "primal_feasibility_tolerance",  # HiGHS, Gurobi
    "dual_feasibility_tolerance",    # HiGHS, Gurobi
    "feasibility_tolerance",         # Some solvers
    "FeasibilityTol",               # Gurobi
    "OptimalityTol",                # Gurobi
    "primal_tolerance",             # CPLEX-style
    "dual_tolerance"                # CPLEX-style
)

"""
Extract numerical tolerance from JuMP optimizer using const arrays for performance.
"""
function extract_solver_tolerance(optimizer, settings=())::Float64
    temp_model = create_temp_model(optimizer, settings)
    temp_model === nothing && return DEFAULT_SOLVER_TOLERANCE

    tolerances = extract_tolerances(temp_model)
    return isempty(tolerances) ? DEFAULT_SOLVER_TOLERANCE : minimum(tolerances)
end

function create_temp_model(optimizer, settings)
    try
        model = J.Model(optimizer)
        for setting in [COBREXA.configuration.default_solver_settings; settings]
            setting(model)
        end
        return model
    catch e
        @debug "Could not create temporary model" exception = e
        return nothing
    end
end

function extract_tolerances(model)::Vector{Float64}
    tolerances = Float64[]
    sizehint!(tolerances, length(TOLERANCE_ATTRIBUTES))

    for attr in TOLERANCE_ATTRIBUTES
        tol = get_tolerance_attribute(model, attr)
        !isnan(tol) && push!(tolerances, tol)
    end
    return tolerances
end

function get_tolerance_attribute(model, attr::String)::Float64
    try
        tol = J.get_optimizer_attribute(model, attr)
        return (tol isa Real && 0 < tol < MAX_VALID_TOLERANCE) ? Float64(tol) : NaN
    catch
        return NaN
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
    streaming_filter::StreamingCandidateFilter
)::Nothing
    state.batches_processed += 1

    # Log batch collection completion
    log_batch_collection!(state, streaming_filter)

    # Clear module cache
    clear_module_cache!(concordance_tracker)

    # Process the batch
    batch_result = execute_batch_optimization!(constraints, state, concordance_tracker, config)

    # Create counts for accumulation - extract concordant count from batch_result
    concordant_count = batch_result.concordant_pairs.n_pairs  # This was set correctly in execute_batch_optimization!
    batch_counts = MutableCounts(
        concordant_count,
        batch_result.non_concordant,
        batch_result.timeout,
        batch_result.transitive,
        batch_result.skipped
    )

    # Accumulate results
    accumulate_results!(state.accumulator, batch_result.concordant_pairs, batch_counts, batch_result.optimization_results)
    state.total_batches_completed += 1
    state.total_pairs_processed += length(state.current_batch)

    # Reset for next batch
    empty!(state.current_batch)
    state.batch_collection_start_time = time()
    return nothing
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
    config::BatchProcessingConfig
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
        concordance_tolerance=config.concordance_tolerance
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
        0, 0,  # transitive counts handled elsewhere  
        batch_counts.optimization_results
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
    
    PairConcordanceState() = new(NaN, NaN, NaN, NaN, false, false, false, true, NaN)
end

"""
Update pair concordance state with new solver result.
Returns true if pair is still potentially concordant, false if definitely non-concordant.
"""
@inline function update_pair_concordance!(
    state::PairConcordanceState,
    direction::Symbol,
    dir_multiplier::Int,
    value::Float64,
    timeout::Bool,
    concordance_tolerance::Float64
)::Bool
    # Early exit if already failed or timeout
    timeout && (state.has_timeout = true; state.is_concordant = false; return false)
    !state.is_concordant && return false
    
    # Update min/max values for this direction
    if direction == :positive
        if dir_multiplier == -1
            state.positive_min = value
        else  # dir_multiplier == 1
            state.positive_max = value
        end
        state.has_positive = true
    else  # direction == :negative
        if dir_multiplier == -1
            state.negative_min = value
        else  # dir_multiplier == 1
            state.negative_max = value
        end
        state.has_negative = true
    end
    
    # Check concordance for completed directions
    if direction == :positive && !isnan(state.positive_min) && !isnan(state.positive_max)
        if abs(state.positive_min - state.positive_max) > concordance_tolerance
            state.is_concordant = false
            return false
        end
        current_lambda = (state.positive_min + state.positive_max) / 2
        if isnan(state.reference_lambda)
            state.reference_lambda = current_lambda
        elseif abs(current_lambda - state.reference_lambda) > concordance_tolerance
            state.is_concordant = false
            return false
        end
    elseif direction == :negative && !isnan(state.negative_min) && !isnan(state.negative_max)
        if abs(state.negative_min - state.negative_max) > concordance_tolerance
            state.is_concordant = false
            return false
        end
        current_lambda = (state.negative_min + state.negative_max) / 2
        if isnan(state.reference_lambda)
            state.reference_lambda = current_lambda
        elseif abs(current_lambda - state.reference_lambda) > concordance_tolerance
            state.is_concordant = false
            return false
        end
    end
    
    return state.is_concordant
end

"""
Most efficient streaming processor for solver results.
Eliminates all intermediate data structures and processes in single pass.
"""
function process_solver_results_streaming!(
    solver_results,
    batch_pairs::Vector{PairCandidate}, 
    concordance_tracker,
    concordance_tolerance::Float64
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
    
    for (i, state) in enumerate(pair_states)
        candidate = batch_pairs[i]
        
        if state.has_timeout
            timeout_count += 1
        end
        
        if state.is_concordant
            union_sets!(concordance_tracker, candidate.c1_idx, candidate.c2_idx)
            concordant_count += 1
        else
            add_non_concordant!(concordance_tracker, candidate.c1_idx, candidate.c2_idx)
            non_concordant_count += 1
        end
    end
    
    return concordant_count, non_concordant_count, timeout_count
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
    config::BatchProcessingConfig
)
    state = initialize_processing_state(streaming_filter, concordance_tracker, config)

    @info "Using fixed batch size: $(config.batch_size) candidates per batch"
    @info "Collecting candidates from streaming filter..."

    # Main processing loop
    for candidate in streaming_filter
        collect_candidate!(state, candidate)

        if should_process_batch(state, config)
            process_full_batch!(constraints, state, concordance_tracker, config, streaming_filter)
        end
    end

    # Process any remaining candidates in final batch
    if !isempty(state.current_batch)
        process_full_batch!(constraints, state, concordance_tracker, config, streaming_filter)
    end

    return build_final_results(state)
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
        optimization_results=opt_results_dict
    )
end



"""
$(TYPEDSIGNATURES)

Helper function to process optimization results into the expected format.
"""
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
$(TYPEDSIGNATURES)

Process a batch of concordance tests using ConstraintTrees templated approach with parallel batch processing.
"""
function process_concordance_batch(
    constraints::C.ConstraintTree,
    batch_pairs::Vector{PairCandidate},
    concordance_tracker::ConcordanceTracker;
    optimizer,
    settings=[],
    workers=workers,
    concordance_tolerance::Float64=1e-4
)
    # Create COBREXA-style test array with direction multipliers directly from PairCandidate
    test_array = []
    for candidate in batch_pairs
        c1_id = concordance_tracker.idx_to_id[candidate.c1_idx]
        c2_id = concordance_tracker.idx_to_id[candidate.c2_idx]
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
            om = direction == :positive ? models_cache.positive : models_cache.negative

            try
                # Set c2 constraint to 1.0 and optimize c1
                c2_constraint = J.@constraint(om, c2_constraint,
                    C.substitute(constraints.activities[c2_id].value, om[:x]) == 1.0)

                J.@objective(om, J.MAX_SENSE,
                    C.substitute(dir_multiplier * constraints.activities[c1_id].value, om[:x]))

                J.optimize!(om)

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

                return (c1_id, c2_id, direction, dir_multiplier, actual_value, J.termination_status(om) == J.TIME_LIMIT)
            catch e
                @warn "Optimization error" c1_id c2_id direction error = string(e)
                return (c1_id, c2_id, direction, dir_multiplier, NaN, false)
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
    
    # Process each solver result immediately - zero intermediate storage
    for result in optimization_results
        c1_id, c2_id, direction, dir_multiplier, value, timeout = result
        pair_idx = get(pair_lookup, (c1_id, c2_id), 0)
        pair_idx == 0 && continue
        
        # Update concordance state directly - no allocations
        update_pair_concordance!(pair_states[pair_idx], direction, dir_multiplier, value, timeout, concordance_tolerance)
    end
    
    # OPTIMIZED: Update concordance tracker directly and return counts only
    # Eliminates the need for process_batch_results! entirely
    
    concordant_count = 0
    non_concordant_count = 0
    timeout_count = 0
    optimization_results_vec = Vector{Tuple{Int,Int,Symbol,Float64}}()
    
    for (i, state) in enumerate(pair_states)
        candidate = batch_pairs[i]
        c1_idx, c2_idx = candidate.c1_idx, candidate.c2_idx
        
        # Count timeouts
        state.has_timeout && (timeout_count += 1)
        
        # Update concordance tracker directly based on results
        if state.is_concordant
            union_sets!(concordance_tracker, c1_idx, c2_idx)
            concordant_count += 1
            
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

    # Return counts in same format as process_batch_results! expected
    return (
        concordant_count = concordant_count,
        non_concordant_count = non_concordant_count, 
        timeout_count = timeout_count,
        optimization_results = optimization_results_vec
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

Build Y_matrix (species-complex stoichiometric matrix) from constraints and complexes.
Uses canonical ordering provided by caller or generates sorted ordering.
Returns (Y_matrix, complex_ids, metabolite_ids) where:
- Y_matrix[i,j] = stoichiometric coefficient of metabolite i in complex j
- complex_ids[j] = Symbol ID of complex j
- metabolite_ids[i] = Symbol ID of metabolite i
"""
function build_Y_matrix_from_constraints(
    constraints,
    complexes::Dict{Symbol,MetabolicComplex},
    complex_ids::Union{Vector{Symbol},Nothing}=nothing,
    metabolite_ids::Union{Vector{Symbol},Nothing}=nothing
)
    # Use provided ordering or create canonical ordering
    if complex_ids === nothing
        complex_ids = sort!(collect(keys(complexes)); by=string)
    end

    if metabolite_ids === nothing
        # Get all unique metabolites
        all_metabolites = Set{Symbol}()
        for complex in values(complexes)
            for (met_id, _) in complex.metabolites
                push!(all_metabolites, met_id)
            end
        end
        metabolite_ids = sort!(collect(all_metabolites); by=string)
    end

    n_metabolites = length(metabolite_ids)
    n_complexes = length(complex_ids)

    # Build index mappings
    metabolite_to_idx = Dict(met => i for (i, met) in enumerate(metabolite_ids))
    complex_to_idx = Dict(comp => i for (i, comp) in enumerate(complex_ids))

    # Build Y_matrix
    Y_matrix = SparseArrays.spzeros(Float64, n_metabolites, n_complexes)

    for (complex_id, complex) in complexes
        if haskey(complex_to_idx, complex_id)  # Only process complexes in our ordering
            complex_idx = complex_to_idx[complex_id]
            for (metabolite_id, stoich) in complex.metabolites
                if haskey(metabolite_to_idx, metabolite_id)
                    met_idx = metabolite_to_idx[metabolite_id]
                    Y_matrix[met_idx, complex_idx] = Float64(stoich)
                end
            end
        end
    end

    return Y_matrix, complex_ids, metabolite_ids
end

"""
$(TYPEDSIGNATURES)

Find trivially balanced complexes using Y_matrix.
A complex is trivially balanced if all its metabolites appear in only this complex.
"""
function find_trivially_balanced_complexes(
    Y_matrix::SparseArrays.SparseMatrixCSC{Float64,Int},
    complex_ids::Vector{Symbol},
    metabolite_ids::Vector{Symbol}
)::Set{Symbol}
    n_metabolites, n_complexes = size(Y_matrix)
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    # Build metabolite participation map from Y_matrix
    for complex_idx in 1:n_complexes
        complex_id = complex_ids[complex_idx]
        metabolite_idxs, _ = SparseArrays.findnz(Y_matrix[:, complex_idx])

        for met_idx in metabolite_idxs
            met_id = metabolite_ids[met_idx]
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    balanced_complexes = Set{Symbol}()
    for (met_id, complex_ids_list) in metabolite_participation
        if length(complex_ids_list) == 1
            push!(balanced_complexes, complex_ids_list[1])
        end
    end

    return balanced_complexes
end

"""
$(TYPEDSIGNATURES)

Find trivially concordant pairs using Y_matrix.
Pairs are trivially concordant if they share exactly the same metabolites.
"""
function find_trivially_concordant_pairs(
    Y_matrix::SparseArrays.SparseMatrixCSC{Float64,Int},
    complex_ids::Vector{Symbol},
    metabolite_ids::Vector{Symbol}
)::Set{Tuple{Symbol,Symbol}}
    n_metabolites, n_complexes = size(Y_matrix)
    metabolite_participation = Dict{Symbol,Vector{Symbol}}()

    # Build metabolite participation map from Y_matrix
    for complex_idx in 1:n_complexes
        complex_id = complex_ids[complex_idx]
        metabolite_idxs, _ = SparseArrays.SparseArrays.findnz(Y_matrix[:, complex_idx])

        for met_idx in metabolite_idxs
            met_id = metabolite_ids[met_idx]
            if !haskey(metabolite_participation, met_id)
                metabolite_participation[met_id] = Symbol[]
            end
            push!(metabolite_participation[met_id], complex_id)
        end
    end

    concordant_pairs = Set{Tuple{Symbol,Symbol}}()
    for (met_id, complex_ids_list) in metabolite_participation
        if length(complex_ids_list) == 2
            c1, c2 = complex_ids_list
            pair = c1 < c2 ? (c1, c2) : (c2, c1)
            push!(concordant_pairs, pair)
        end
    end

    return concordant_pairs
end

function find_trivially_balanced_complexes(
    complexes::Dict{Symbol,MetabolicComplex}
)::Set{Symbol}
    Y_matrix, complex_ids, metabolite_ids = build_Y_matrix_from_constraints(nothing, complexes)
    return find_trivially_balanced_complexes(Y_matrix, complex_ids, metabolite_ids)
end

"""
$(TYPEDSIGNATURES)

Find trivially concordant pairs (complexes that share the same metabolite composition).
"""
function find_trivially_concordant_pairs(complexes::Dict{Symbol,MetabolicComplex})::Set{Tuple{Symbol,Symbol}}
    Y_matrix, complex_ids, metabolite_ids = build_Y_matrix_from_constraints(nothing, complexes)
    return find_trivially_concordant_pairs(Y_matrix, complex_ids, metabolite_ids)
end

"""
$(TYPEDSIGNATURES)

Main concordance analysis function optimized for large models and HPC execution.

## Statistics Dictionary Output

The analysis returns a comprehensive statistics dictionary with the following keys:

### Core Analysis Results
- `n_candidate_pairs`: Total candidate pairs generated by the filter (after CV filtering)
- `n_computed_pairs`: Concordant pairs found through analysis (direct testing + transitivity)
- `n_concordant_via_testing`: Pairs found concordant through optimization testing
- `n_concordant_via_transitivity`: Pairs found concordant through transitivity relationships
- `n_non_concordant_pairs`: Pairs determined to be non-concordant
- `n_trivial_pairs`: Trivially concordant pairs (balanced complexes)
- `n_concordant_pairs`: Total concordant pairs (computed + trivial)

### Processing Statistics
- `n_candidates_skipped_by_transitivity`: Candidate pairs skipped during batch processing due to known transitivity relationships
- `batches_completed`: Number of optimization batches processed
- `elapsed_time`: Total analysis time in seconds

### Model Information
- `n_complexes`: Total number of complexes in the model
- `n_balanced`: Number of balanced complexes
- `n_trivially_balanced`: Number of trivially balanced complexes
- `n_modules`: Number of concordance modules found

### Validation
- `validation_passed`: Whether internal consistency checks passed

## Key Relationships

The statistics follow these mathematical relationships:
- `n_concordant_pairs` = `n_computed_pairs` + `n_trivial_pairs`
- `n_computed_pairs` = `n_concordant_via_testing` + `n_concordant_via_transitivity`
- Total work done ≈ `n_candidate_pairs` - `n_candidates_skipped_by_transitivity`

The transitivity optimization means that many candidate pairs are identified as concordant
without requiring expensive optimization, significantly reducing computational cost.
"""
function activity_concordance_analysis(
    model;
    modifications=Function[],
    optimizer,
    settings=[],
    workers=D.workers(),
    concordance_tolerance::Float64=NaN,
    balanced_tolerance::Float64=NaN,
    cv_threshold::Float64=NaN,
    cv_epsilon::Float64=1e-16,
    sample_size::Int=100,
    batch_size::Int=50_000,
    min_valid_samples::Int=10,
    seed::Int=0,
    use_unidirectional_constraints::Bool=true,
    objective_bound=nothing,
    use_transitivity::Bool=true,
    n_burnin::Int=50,
    n_chains::Int=1,
)
    start_time = time()

    # Detect solver tolerance for consistent numerical thresholds
    solver_tolerance = extract_solver_tolerance(optimizer, settings)

    # Set default values based on solver tolerance if not provided (NaN indicates default should be used)
    actual_concordance_tolerance = !isnan(concordance_tolerance) ? concordance_tolerance : max(solver_tolerance * 100, 1e-4)
    actual_balanced_tolerance = !isnan(balanced_tolerance) ? balanced_tolerance : solver_tolerance
    actual_cv_threshold = !isnan(cv_threshold) ? cv_threshold : max(solver_tolerance * 100, 1e-2)

    @info "Starting concordance analysis" n_workers = length(workers) concordance_tolerance = actual_concordance_tolerance balanced_tolerance = actual_balanced_tolerance cv_threshold = actual_cv_threshold sample_size use_unidirectional_constraints batch_size solver_tolerance

    # Fix model objective if it has conversion issues (e.g., missing R_ prefix)
    if isa(model, SBMLFBCModels.SBMLFBCModel)
        model = COCOA.ElementarySteps.fix_objective_after_conversion(model)
    end

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

    # Get correct reaction count from constraints (accounts for reaction splitting)
    n_reactions = C.var_count(constraints.balance)

    @info "Model statistics" n_complexes n_reactions

    # Get canonical metabolite ordering
    all_metabolites = Set{Symbol}()
    for complex in values(complexes)
        for (met_id, _) in complex.metabolites
            push!(all_metabolites, met_id)
        end
    end
    metabolite_ids = sort!(collect(all_metabolites); by=string)

    # Build Y_matrix for efficient trivial analysis using canonical ordering
    @info "Building Y_matrix for trivial analysis"
    Y_matrix, _, _ = build_Y_matrix_from_constraints(constraints, complexes, complex_ids, metabolite_ids)

    @info "Finding trivially balanced complexes"
    trivially_balanced = find_trivially_balanced_complexes(Y_matrix, complex_ids, metabolite_ids)
    @info "Found trivially balanced complexes" n_trivivally_balanced = length(trivially_balanced)

    @info "Finding trivially concordant pairs"
    trivial_pairs = find_trivially_concordant_pairs(Y_matrix, complex_ids, metabolite_ids)
    @info "Found trivially concordant pairs" n_trivially_concordant = length(trivial_pairs)

    # Initialize concordance matrix and populate trivial relationships early
    n_complexes = length(complex_ids)
    complex_idx = Dict(id => idx for (idx, id) in enumerate(complex_ids))
    concordance_matrix = SparseArrays.spzeros(Int, n_complexes, n_complexes)

    @info "Populating trivial relationships in concordance matrix"
    populate_trivial_relationships!(concordance_matrix, trivially_balanced, trivial_pairs, complex_idx)

    # Set up RNG early for deterministic sampling
    rng = if seed == 0
        Random.GLOBAL_RNG
    else
        StableRNGs.StableRNG(seed)
    end

    # Use the separated AVA function with warmup point generation and external timing
    ava_digits = max(1, -floor(Int, log10(solver_tolerance)))
    ava_output_func = (dir, om) -> ava_output_with_warmup(dir, om; digits=ava_digits)

    ava_time = @elapsed ava_results = activity_variability_analysis(
        constraints,
        complex_ids;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        solver_tolerance=solver_tolerance,
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

    # Use balanced tolerance for balanced complex detection - ensures consistency with optimization
    balanced_threshold = actual_balanced_tolerance

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
            # Use @fastmath for performance-critical comparisons
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

    @inbounds for (c1_id, c2_id) in trivial_pairs
        if haskey(id_to_idx, c1_id) && haskey(id_to_idx, c2_id)
            c1_idx = id_to_idx[c1_id]
            c2_idx = id_to_idx[c2_id]
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
    # Use a fixed offset from the main seed to ensure deterministic but different sampling
    sampling_seed = if seed == 0
        UInt64(12345)  # Fixed fallback seed for deterministic behavior even without user seed
    else
        UInt64(seed + 1000)  # Deterministic offset from main seed
    end
    @info "Sampling..."
    samples_tree = COBREXA.sample_constraints(
        COBREXA.sample_chain_achr,
        constraints.balance;
        output=filtered_activities,
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
            cv_threshold=actual_cv_threshold,
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
        actual_concordance_tolerance,
        use_transitivity,
        optimizer,
        settings,
        workers
    )

    concordance_time = @elapsed batch_results = process_streaming_batches(
        constraints,
        streaming_filter,
        concordance_tracker,
        config
    )

    concordance_time_str = Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, concordance_time * 1000)), "HH:MM:SS.s")
    @info "Building concordance modules [$concordance_time_str]"
    modules = extract_modules(concordance_tracker)

    # Use module-based counting for accurate totals
    total_concordant_from_modules = count_concordant_pairs_from_modules(concordance_tracker)

    # Debug: compare with tracked count
    tracked_concordant = batch_results.concordant_pairs.n_pairs
    @debug "Concordance counting comparison" (
        from_modules=total_concordant_from_modules - length(trivial_pairs),
        from_tracking=tracked_concordant,
        trivial_pairs=length(trivial_pairs),
        total_modules=total_concordant_from_modules
    )


    elapsed = time() - start_time

    stats = Dict(
        # Analysis results
        "n_complexes" => n_complexes,
        "n_balanced" => count(balanced_complexes),
        "n_trivially_balanced" => length(trivially_balanced),
        "n_trivial_pairs" => length(trivial_pairs),
        "n_candidate_pairs" => batch_results.pairs_processed,  # Total candidates processed across all chunks
        "n_concordant_found" => total_concordant_from_modules - length(trivial_pairs),  # Non-trivial concordant pairs from modules
        "n_concordant_opt" => batch_results.concordant_count,  # Found via direct optimization testing
        "n_concordant_inferred" => total_concordant_from_modules - length(trivial_pairs) - batch_results.concordant_count,  # Found via transitivity
        "n_concordant_total" => total_concordant_from_modules,  # All concordant pairs from modules
        "n_non_concordant_pairs" => batch_results.non_concordant_pairs,
        "n_candidates_skipped_by_transitivity" => streaming_filter.pairs_transitivity_concordant_filtered + streaming_filter.pairs_transitivity_non_concordant_filtered,  # Candidates not tested due to transitivity
        "n_timeout_pairs" => batch_results.timeout_pairs,
        "n_modules" => length(modules),
        "batches_completed" => batch_results.batches_completed,
        "elapsed_time" => elapsed,

        # Algorithm parameters
        "concordance_tolerance" => actual_concordance_tolerance,
        "balanced_tolerance" => actual_balanced_tolerance,
        "cv_threshold" => actual_cv_threshold,
        "cv_epsilon" => cv_epsilon,
        "solver_tolerance" => solver_tolerance,

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
    )

    # Add processing statistics if processed_mask is available
    if concordance_tracker.processed_mask !== nothing
        n_processed = count(concordance_tracker.processed_mask)
        stats["n_processed_complexes"] = n_processed
        stats["processing_completion_percent"] = round(n_processed / n_complexes * 100, digits=1)
    end

    # Validate consistency of concordant pair accounting
    expected_total_concordant = stats["n_concordant_found"] + stats["n_trivial_pairs"]
    actual_total_concordant = stats["n_concordant_total"]

    validation_passed = (expected_total_concordant == actual_total_concordant)

    # Validate core concordant pair accounting (most important check)
    if !validation_passed
        @warn "Concordant pair accounting mismatch detected!" (
            expected_total=expected_total_concordant,
            actual_total=actual_total_concordant,
            found_pairs=stats["n_concordant_found"],
            trivial_pairs=stats["n_trivial_pairs"],
            via_direct_testing=stats["n_concordant_opt"],
            via_inference=stats["n_concordant_inferred"],
            difference=actual_total_concordant - expected_total_concordant
        )
    else
        @debug "Concordant pair validation passed" (
            total_concordant=actual_total_concordant,
            expected_concordant=expected_total_concordant
        )
    end

    stats["validation_passed"] = validation_passed

    # Clear summary of the analysis results
    @info "Concordance analysis complete"
    @info "  Total concordant pairs: $(stats["n_concordant_total"])"
    @info "  Breakdown: $(stats["n_concordant_opt"]) from optimization, $(stats["n_candidates_skipped_by_transitivity"]) filtered by transitivity"
    @info "  Processing: $(stats["n_candidate_pairs"]) candidates, $(stats["n_trivial_pairs"]) trivial pairs"
    @info "  Elapsed time: $(Dates.format(Dates.Time(0) + Dates.Millisecond(round(Int, stats["elapsed_time"] * 1000)), "HH:MM:SS.s"))"

    @debug "Full concordance analysis statistics" stats

    # Populate discovered concordant relationships in concordance matrix after streaming analysis
    @info "Populating discovered concordant relationships in concordance matrix"
    populate_discovered_relationships!(concordance_matrix, concordance_tracker, balanced_complexes)

    # Build CompleteConcordanceModel using canonical ordering established earlier
    # complex_ids already canonical from concordance_tracker
    # metabolite_ids already canonical from earlier in function
    # reaction_ids need to be sorted for canonical order
    reaction_names = get_reaction_names_from_constraints(constraints.balance)
    reaction_ids = Symbol.(sort!(reaction_names; by=string))

    # Build all matrices upfront
    A_matrix, complexes_vec, complex_to_idx_map = build_A_matrix_from_complexes(complexes, constraints)
    S_matrix = build_S_matrix_from_constraints(constraints)[1]

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
    complete_model = CompleteConcordanceModel(
        complex_ids,
        reaction_ids,
        Symbol.(metabolite_ids);
        concordance_matrix=concordance_matrix,  # Pass pre-built matrix
        activity_ranges=activity_ranges,
        concordance_modules=module_mapping,
        interface_reactions=falses(length(reaction_ids)),
        acr_metabolites=falses(length(metabolite_ids)),
        complex_reaction_matrix=A_matrix,
        complex_metabolite_matrix=Y_matrix,
        reaction_metabolite_matrix=S_matrix,
        lambda_dict=lambda_dict
    )

    return complete_model
end

"""
Helper function to populate trivial relationships in concordance matrix.
Maintains symmetry and respects hierarchy (trivial > regular).
"""
function populate_trivial_relationships!(
    concordance_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    trivially_balanced::Set{Symbol},
    trivial_pairs::Set{Tuple{Symbol,Symbol}},
    complex_idx::Dict{Symbol,Int}
)
    # Populate trivially balanced complexes on diagonal (value 4)
    for complex_id in trivially_balanced
        if haskey(complex_idx, complex_id)
            idx = complex_idx[complex_id]
            concordance_matrix[idx, idx] = Int(Trivially_balanced)  # 4
        end
    end

    # Populate trivially concordant pairs (value 2, maintain symmetry)
    for (c1_id, c2_id) in trivial_pairs
        if haskey(complex_idx, c1_id) && haskey(complex_idx, c2_id)
            i = complex_idx[c1_id]
            j = complex_idx[c2_id]
            concordance_matrix[i, j] = Int(Trivially_concordant)  # 2
            concordance_matrix[j, i] = Int(Trivially_concordant)  # 2 (symmetry)
        end
    end
end

"""
Helper function to populate discovered concordant relationships from ConcordanceTracker.
Respects existing trivial relationships and maintains symmetry.
"""
function populate_discovered_relationships!(
    concordance_matrix::SparseArrays.SparseMatrixCSC{Int,Int},
    concordance_tracker::ConcordanceTracker,
    balanced_complexes::BitVector
)
    n_complexes = length(balanced_complexes)

    # Populate balanced complexes on diagonal (value 3, only if not already trivially balanced)
    for i in 1:n_complexes
        if balanced_complexes[i] && concordance_matrix[i, i] == 0
            concordance_matrix[i, i] = Int(Balanced)  # 3
        end
    end

    # Extract concordant pairs from ConcordanceTracker
    # Use union-find structure to identify all concordant relationships
    for i in 1:n_complexes-1
        for j in i+1:n_complexes
            if are_concordant(concordance_tracker, i, j)
                # Only populate if not already set (preserves trivial relationships)
                if concordance_matrix[i, j] == 0
                    concordance_matrix[i, j] = Int(Concordant)  # 1
                    concordance_matrix[j, i] = Int(Concordant)  # 1 (symmetry)
                end
            end
        end
    end
end
