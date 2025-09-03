"""
Activity Variability Analysis for COCOA.

This module contains the activity variability analysis functionality,
separated from concordance analysis for independent use.
"""
# Note: Types and functions are imported via the main COCOA.jl module


"""
$(TYPEDSIGNATURES)

Advanced method: Perform Activity Variability Analysis using pre-built constraints.

This method operates directly on a constraint system and complex IDs, allowing
for custom constraint modifications before analysis. The variability is examined
on all activities in `constraints.activities` corresponding to the given `complex_ids`.

Parameter `workers` may be used to enable parallel or distributed processing;
the execution defaults to all available workers. Parameters `optimizer` and 
`settings` are internally forwarded to [`constraints_variability`](@ref).

Use [`activity_variability_analysis(model; optimizer, ...)`](@ref) for the 
standard interface that handles constraint generation automatically.
"""
function activity_variability_analysis(
    constraints::C.ConstraintTree,
    complex_ids::Vector{Symbol};
    optimizer,
    settings=[],
    workers=D.workers(),
    output=nothing,  # Allow custom output function
    output_type=nothing,  # Allow custom output type
    return_warmup_points::Bool=false  # Control whether to process warmup points
)
    @info "Performing Activity Variability Analysis"

    # Determine output configuration
    if output === nothing
        # Default: just return ConstraintTree (clean & simple)
        ava_results = COBREXA.constraints_variability(
            constraints.balance,
            constraints.activities;
            optimizer=optimizer,
            settings=settings,
            workers=workers,
        )

        @info "AVA processing complete"
        return ava_results
    else
        # Custom output (e.g., for warmup point generation)  
        ava_results = COBREXA.constraints_variability(
            constraints.balance,
            constraints.activities;
            optimizer=optimizer,
            settings=settings,
            output=output,
            output_type=output_type,
            workers=workers,
        )

        @info "AVA processing complete"

        if return_warmup_points && output_type == Tuple{Float64,Vector{Float64}}
            # Process results into activity ranges and warmup points (for concordance analysis)
            n_complexes = length(complex_ids)
            warmup_points = Vector{Vector{Float64}}()
            sizehint!(warmup_points, n_complexes * 2)  # Pre-allocate estimate: 2 points per active complex

            # Pre-allocate activity ranges with concrete types for better performance
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

            # Build warmup matrix efficiently
            warmup = if isempty(warmup_points)
                Matrix{Float64}(undef, 0, 0)
            else
                # Build matrix directly without transpose - more efficient
                n_points = length(warmup_points)
                n_vars = length(warmup_points[1])
                warmup_matrix = Matrix{Float64}(undef, n_points, n_vars)
                @inbounds for (i, point) in enumerate(warmup_points)
                    warmup_matrix[i, :] = point
                end
                warmup_matrix
            end

            return (activity_ranges=activity_ranges, warmup_points=warmup)
        else
            # Just return the raw results from COBREXA
            return ava_results
        end
    end
end

"""
$(TYPEDSIGNATURES)

Perform Activity Variability Analysis on complex activities in the `model`.

This is the main interface following COBREXA conventions. The constraint system 
is constructed using [`concordance_constraints`](@ref), and variability is 
examined on all complex activities.

For advanced usage with pre-built constraints, use the method that takes 
`constraints` and `complex_ids` directly.
"""
function activity_variability_analysis(
    model;
    optimizer,
    modifications=Function[],
    settings=[],
    workers=D.workers(),
    solver_tolerance::Float64=1e-6,
    use_unidirectional_constraints::Bool=true,
    output=nothing,
    output_type=nothing,
    return_warmup_points::Bool=false
)
    # Fix model objective if it has conversion issues (e.g., missing R_ prefix)
    if isa(model, SBMLFBCModels.SBMLFBCModel)
        model = COCOA.ElementarySteps.fix_objective_after_conversion(model)
    end

    # Generate constraints and extract complexes
    constraints, complexes = concordance_constraints(
        model;
        modifications,
        use_unidirectional_constraints,
        return_complexes=true
    )

    # Get sorted complex IDs for deterministic results
    complex_ids = sort!(collect(keys(complexes)); by=string)

    # Call the main AVA function
    return activity_variability_analysis(
        constraints,
        complex_ids;
        optimizer=optimizer,
        settings=settings,
        workers=workers,
        solver_tolerance=solver_tolerance,
        output=output,
        output_type=output_type,
        return_warmup_points=return_warmup_points
    )
end

function ava_output_with_warmup(dir, om; digits, collect_flux=true)
    if J.termination_status(om) != J.OPTIMAL
        return (nothing, nothing)
    end

    # Round the results to a reasonable precision to mitigate floating point noise
    objective_val = round(J.objective_value(om), digits=digits)
    activity = dir * objective_val

    # Only collect flux vector if requested
    if collect_flux
        flux_values = round.(J.value.(J.all_variables(om)), digits=digits)
        return (activity, flux_values)
    else
        return (activity, nothing)
    end
end