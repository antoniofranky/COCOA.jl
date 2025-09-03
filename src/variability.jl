
"""
$(TYPEDSIGNATURES)

Extract warmup points from activity variability analysis results.

Takes the results from [`activity_variability_analysis`](@ref) when run with
custom output function and processes them into activity ranges and warmup points
for use in concordance analysis.
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
$(TYPEDSIGNATURES)

Perform Activity Variability Analysis using pre-built constraints.

Computes the variability of complex activities in the feasible space specified
by `constraints`. The analysis examines all activities in `constraints.activities`
corresponding to the given `complex_ids`.

Parameters `optimizer`, `settings`, and `workers` are forwarded to
[`COBREXA.constraints_variability`](@ref).

For warmup point generation (used in concordance analysis), provide custom
`output` and `output_type` parameters and set `return_warmup_points=true`.
"""
function activity_variability_analysis(
    constraints::C.ConstraintTree,
    complex_ids::Vector{Symbol};
    optimizer,
    settings=[],
    workers=D.workers(),
    output=nothing,
    output_type=nothing,
    return_warmup_points::Bool=false
)
    ava_results = COBREXA.constraints_variability(
        constraints.balance,
        constraints.activities;
        optimizer,
        settings,
        workers,
        (output === nothing ? () : (output=output,))...,
        (output_type === nothing ? () : (output_type=output_type,))...,
    )

    if return_warmup_points && output_type == Tuple{Float64,Vector{Float64}}
        return _extract_warmup_points(ava_results, complex_ids)
    else
        return ava_results
    end
end

"""
$(TYPEDSIGNATURES)

Perform Activity Variability Analysis on complex activities in the `model`.

The constraint system is constructed using [`concordance_constraints`](@ref),
and variability is examined on all complex activities. Parameters `optimizer`,
`settings`, and `workers` are forwarded to the constraint-based method.

Use the constraint-based method directly for advanced usage with pre-built
constraints.
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
    if isa(model, SBMLFBCModels.SBMLFBCModel)
        model = COCOA.ElementarySteps.fix_objective_after_conversion(model)
    end

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
$(TYPEDSIGNATURES)

Output function for activity variability analysis that collects both activity
values and flux vectors for warmup point generation.

Returns `(activity, flux_vector)` tuple if optimization is successful,
`(nothing, nothing)` otherwise. Results are rounded to `digits` precision.
"""
function ava_output_with_warmup(dir, om; digits, collect_flux=true)
    J.termination_status(om) != J.OPTIMAL && return (nothing, nothing)

    objective_val = round(J.objective_value(om), digits=digits)
    activity = dir * objective_val

    flux_values = collect_flux ?
                  round.(J.value.(J.all_variables(om)), digits=digits) : nothing

    (activity, flux_values)
end

export activity_variability_analysis, ava_output_with_warmup