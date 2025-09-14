
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
- SBML models are automatically processed with [`ElementarySteps.fix_objective_after_conversion`](@ref)
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
Custom output function for AVA that collects activity values and flux vectors.

Returns `(activity, flux_vector)` if optimization succeeds, `(nothing, nothing)` otherwise.
Used to generate warmup points for concordance analysis.
"""
function ava_output_with_warmup(dir, om; digits, collect_flux=true)
    J.termination_status(om) != J.OPTIMAL && return (nothing, nothing)

    objective_val = round(J.objective_value(om), digits=digits)
    activity = dir * objective_val

    flux_values = collect_flux ?
                  round.(J.value.(J.all_variables(om)), digits=digits) : nothing

    (activity, flux_values)
end