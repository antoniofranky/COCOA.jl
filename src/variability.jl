
"""
Activity Variability Analysis for COCOA.jl

Functions for computing the variability of complex activities in metabolic networks.
"""

"""
Extract warmup points and activity ranges from AVA results.

Internal function that processes activity variability analysis results to extract
flux vectors for use as warmup points in concordance analysis.
"""
function _extract_warmup_points(
    ava_results,
    complex_ids::Vector{Symbol}
)::@NamedTuple{activity_ranges::Vector{Tuple{Float64,Float64}}, warmup_points::Matrix{Float64}}
    n_complexes = length(complex_ids)
    warmup_points = Vector{Vector{Float64}}()
    sizehint!(warmup_points, n_complexes * 2)

    activity_ranges = Vector{Tuple{Float64,Float64}}(undef, n_complexes)

    @inbounds for (i, cid) in enumerate(complex_ids)
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
                    activity_ranges[i] = (NaN, NaN)
                end
            else
                activity_ranges[i] = (NaN, NaN)
            end
        else
            activity_ranges[i] = (NaN, NaN)
        end
    end

    warmup_matrix = if isempty(warmup_points)
        Matrix{Float64}(undef, 0, 0)
    else
        n_points = length(warmup_points)
        n_vars = length(warmup_points[1])
        warmup_matrix = Matrix{Float64}(undef, n_points, n_vars)
        @inbounds for (i, point) in enumerate(warmup_points)
            warmup_matrix[i, :] = point
        end
        warmup_matrix
    end

    (activity_ranges=activity_ranges, warmup_points=warmup_matrix)
end

"""
    activity_variability_analysis(constraints::C.ConstraintTree, complex_ids::Vector{Symbol}; kwargs...)

Perform Activity Variability Analysis using pre-built constraints.

Computes the minimum and maximum achievable values for complex activities within
the feasible space defined by the constraint system. This method provides direct
control over the constraint system and is suitable for advanced usage scenarios.

# Arguments
- `constraints::C.ConstraintTree`: Pre-built constraint tree containing balance and activity constraints
- `complex_ids::Vector{Symbol}`: Ordered vector of complex identifiers to analyze

# Keyword Arguments
- `optimizer`: Optimization solver (e.g., HiGHS.Optimizer)
- `settings=[]`: Solver-specific settings vector
- `workers=D.workers()`: Worker processes for parallel computation
- `output=nothing`: Custom output function for results processing
- `output_type=nothing`: Expected return type of output function
- `return_warmup_points::Bool=false`: If true, extract warmup points when `output_type` is appropriate

# Returns
- If `return_warmup_points=false`: Dictionary mapping complex IDs to `(min_activity, max_activity)` tuples
- If `return_warmup_points=true` and `output_type=Tuple{Float64,Vector{Float64}}`: 
  Named tuple with `activity_ranges` and `warmup_points` fields

# Notes
- This method requires pre-built constraints from [`concordance_constraints`](@ref)
- Parameters `optimizer`, `settings`, and `workers` are forwarded to [`COBREXA.constraints_variability`](@ref)
- For warmup point generation, use `output=ava_output_with_warmup` and appropriate output type

# Examples
```julia
# Basic activity variability analysis
constraints, complexes = concordance_constraints(model; return_complexes=true)
complex_ids = collect(keys(complexes))
results = activity_variability_analysis(constraints, complex_ids; optimizer=HiGHS.Optimizer)

# Generate warmup points for concordance analysis
warmup_data = activity_variability_analysis(
    constraints, complex_ids; 
    optimizer=HiGHS.Optimizer,
    output=ava_output_with_warmup,
    output_type=Tuple{Float64,Vector{Float64}},
    return_warmup_points=true
)
```
"""

"""
$(TYPEDSIGNATURES)

Deterministic replacement for `COBREXA.constraints_variability` in the AVA stage.

Why this exists. `COBREXA.constraints_variability` runs through
`screen_optimization_model`, i.e. `D.pmap` over a `D.CachingPool` — tasks are handed
out as workers fall idle — while each worker mutates ONE cached JuMP model, setting a
new objective per target. On a degenerate flux polytope several optimal vertices share
the same optimal value, and which one the solver returns depends on the warm-start
basis, hence on which targets that worker happened to process before. The scalar
min/max are rounded downstream and come out stable, but the full flux vectors do not —
and those vectors are the ACHR sampler's start points. Everything downstream of the
sample (candidate pairs, concordance, modules) then varies between identical runs.

Measured on *Saccharomyces cerevisiae*: three identical runs produced three different
partitions, with the candidate-pair count swinging by 20-63 %.

The fix is the same one used in [`screen_directions_optimization_model`](@ref): the
worker-local model is rebuilt on every call, so all workers start from an identical
fresh model, and it is enough to fix the *partition* of targets into chunks. Chunk k is
always the index set k, k+n_chunks, k+2*n_chunks, …, always processed in that order, so
which physical worker executes it does not affect the result. Round-robin rather than
contiguous blocks keeps the load balanced when LP cost varies between targets.

Reproducibility holds at a fixed worker count; the count is recorded in the run stats.
"""
function constraints_variability_static(
    constraints::C.ConstraintTree,
    targets::Vector{<:C.Value};
    output,
    output_type::Type{T},
    optimizer,
    settings=[],
    workers=D.workers(),
) where {T}
    target_array = [(dir, tgt) for tgt in targets, dir in (-1, 1)]
    n = length(target_array)
    results = Matrix{Union{Nothing,T}}(undef, size(target_array))
    n == 0 && return results

    worker_cache = COBREXA.worker_local_data(constraints) do c
        om = COBREXA.optimization_model(c; optimizer=optimizer)
        for s in [COBREXA.configuration.default_solver_settings; settings]
            s(om)
        end
        return om
    end

    nw = max(1, length(workers))
    n_chunks = min(nw, n)

    chunk_results = D.pmap(
        k -> begin
            om = COBREXA.get_worker_local_data(worker_cache)
            map(k:n_chunks:n) do i
                dir, tgt = target_array[i]
                J.@objective(om, COBREXA.Maximal, C.substitute(dir * tgt, om[:x]))
                optimize_verified!(om)
                COBREXA.is_solved(om) ? output(dir, om) : nothing
            end
        end,
        D.CachingPool(workers),
        1:n_chunks,
    )

    for (k, res) in enumerate(chunk_results)
        for (j, i) in enumerate(k:n_chunks:n)
            results[i] = res[j]
        end
    end
    return results
end

"""
$(TYPEDSIGNATURES)

`ConstraintTree`-shaped wrapper around [`constraints_variability_static`](@ref), doing
the same deflate/reinflate round trip as COBREXA's own overload so the result tree keeps
the shape and order callers expect.

Exists so that the AVA stage and blocked-reaction detection share ONE deterministic
implementation rather than each growing their own copy.
"""
function constraints_variability_tree_static(
    constraints::C.ConstraintTree,
    targets::C.ConstraintTree;
    output=nothing,
    output_type=nothing,
    kwargs...
)
    out_f = output === nothing ? ((dir, om) -> dir * J.objective_value(om)) : output
    out_T = output_type === nothing ? Float64 : output_type
    result_array = constraints_variability_static(
        constraints,
        COBREXA.tree_deflate(C.value, targets, C.Value);
        output=out_f, output_type=out_T, kwargs...
    )
    return COBREXA.tree_reinflate(
        targets,
        Tuple{eltype(result_array),eltype(result_array)}[
            tuple(a, b) for (a, b) in eachrow(result_array)
        ],
    )
end

function activity_variability_analysis(
    constraints::C.ConstraintTree,
    complex_ids::Vector{Symbol};
    optimizer,
    settings=[],
    workers=D.workers(),
    output=nothing,
    output_type=nothing,
    return_warmup_points::Bool=false,
    scheduling::Symbol=:static
)
    ava_results = if scheduling === :static
        constraints_variability_tree_static(
            constraints.balance,
            constraints.activities;
            output=output, output_type=output_type,
            optimizer=optimizer, settings=settings, workers=workers,
        )
    elseif scheduling === :dynamic
        COBREXA.constraints_variability(
            constraints.balance,
            constraints.activities;
            optimizer,
            settings,
            workers,
            (output === nothing ? () : (output=output,))...,
            (output_type === nothing ? () : (output_type=output_type,))...,
        )
    else
        throw(ArgumentError("scheduling must be :static or :dynamic, got $(repr(scheduling))"))
    end

    if return_warmup_points && output_type == Tuple{Float64,Vector{Float64}}
        return _extract_warmup_points(ava_results, complex_ids)
    else
        return ava_results
    end
end

"""
    activity_variability_analysis(model; optimizer, kwargs...)

Perform Activity Variability Analysis on complex activities in a metabolic model.

This is the high-level interface for activity variability analysis. It automatically
constructs the constraint system using [`concordance_constraints`](@ref) and analyzes
variability for all complex activities in the model.

# Arguments
- `model`: Metabolic model (supports COBREXA.jl compatible formats)

# Keyword Arguments
- `optimizer`: Optimization solver (required, e.g., HiGHS.Optimizer)
- `modifications=Function[]`: Model modifications to apply before analysis
- `settings=[]`: Solver-specific settings vector
- `workers=D.workers()`: Worker processes for parallel computation
- `use_unidirectional_constraints::Bool=true`: Use unidirectional flux constraints
- `kwargs...`: Additional parameters forwarded to constraint-based method

# Returns
- ConstraintTree mapping complex IDs to activity variability results
- Return type depends on `output` and `output_type` parameters (see constraint-based method)

# Notes
- Complex IDs are automatically extracted and sorted alphabetically
- For advanced usage with custom constraints, use the constraint-based method directly

# Examples
```julia
# Basic usage
results = activity_variability_analysis(model; optimizer=HiGHS.Optimizer)

# With model modifications
modifications = [change_bound("EX_glc__D_e", lower=-10.0)]
results = activity_variability_analysis(
    model; 
    optimizer=HiGHS.Optimizer,
    modifications=modifications
)

# Generate warmup points
warmup_data = activity_variability_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    output=ava_output_with_warmup,
    output_type=Tuple{Float64,Vector{Float64}},
    return_warmup_points=true
)
```
"""
function activity_variability_analysis(
    model;
    optimizer,
    modifications=Function[],
    settings=[],
    workers=D.workers(),
    use_unidirectional_constraints::Bool=true,
    kwargs...
)
    constraints, complexes = concordance_constraints(
        model;
        modifications,
        use_unidirectional_constraints,
        return_complexes=true
    )

    complex_ids = sort!(collect(keys(complexes)); by=string)

    activity_variability_analysis(
        constraints,
        complex_ids;
        optimizer,
        settings,
        workers,
        kwargs...
    )
end

"""
Custom output function for AVA that collects activity values and flux vectors.

Returns `(activity, flux_vector)` if optimization succeeds, `(nothing, nothing)` otherwise.
Used to generate warmup points for concordance analysis.
"""
function ava_output_with_warmup(dir, om; digits, collect_flux=true)
    # Accept ALMOST_OPTIMAL too. Requiring strict OPTIMAL discarded solves whose value
    # is perfectly usable at the thresholds this feeds (balanced_threshold 1e-7 against
    # relaxed solver tolerances ~1e-6..1e-8), and a discarded AVA result becomes a NaN
    # activity range, which `activity_concordance_analysis` classifies as `unrestricted`
    # — i.e. a failed solve turned into a statement about the complex.
    # Return `nothing`, NOT `(nothing, nothing)`: the caller stores this in a
    # `Union{Nothing,Tuple{Float64,Vector{Float64}}}` slot, which a tuple of Nothings
    # cannot convert into. The old code returned the tuple too, but only when the
    # termination status was not OPTIMAL — and in that case the caller never invoked
    # this function at all, so the branch was dead. Accepting ALMOST_OPTIMAL made it
    # live for OPTIMAL-but-primal-infeasible solves, and 40 of 343 models died on the
    # resulting conversion error.
    J.is_solved_and_feasible(om; allow_local=true, allow_almost=true) || return nothing

    objective_val = round(J.objective_value(om), digits=digits)
    activity = dir * objective_val

    flux_values = collect_flux ?
                  round.(J.value.(J.all_variables(om)), digits=digits) : nothing

    (activity, flux_values)
end