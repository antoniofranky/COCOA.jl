
"""
    module COCOA

Concordance analysis for metabolic networks integrated with COBREXA.

This module implements methods for identifying concordant complexes in metabolic
networks as described in Küken et al. (2022). Two complexes are concordant if their
activities maintain a constant ratio across all steady-state flux distributions.
"""
module COCOA

using COBREXA
using AbstractFBCModels
using SBMLFBCModels
using ConstraintTrees
using JuMP
using GLPK, HiGHS, Gurobi
using SparseArrays
using LinearAlgebra
using Distributed
using DataFrames
using Statistics
using Random
using ProgressMeter
using Printf

import COBREXA: C, A, J, D
import Base: @kwdef

# Get function from other modules 
include("preprocessing/ModelPreparation.jl")

export find_concordant_complexes, ConcordanceResults, activity_variability_analysis

# Type aliases for clarity
const Maybe{T} = Union{Nothing,T}

"""
    ComplexInfo

Lightweight structure for complex representation optimized for HPC.
"""
struct ComplexInfo
    id::String
    metabolite_idxs::Vector{Int32}
    stoichiometry::Vector{Float32}
    hash::UInt64

    function ComplexInfo(met_idxs::Vector{<:Integer}, stoich::Vector{<:Real}, met_ids::Vector{String})
        perm = sortperm(met_idxs)
        sorted_idxs = Int32.(met_idxs[perm])
        sorted_stoich = Float32.(stoich[perm])

        # Generate unique ID
        id = join(["$(isinteger(c) ? Int(c) : c)_$(met_ids[idx])"
                   for (idx, c) in zip(sorted_idxs, sorted_stoich)], "+")

        h = hash((sorted_idxs, sorted_stoich))
        new(id, sorted_idxs, sorted_stoich, h)
    end
end

Base.hash(c::ComplexInfo, h::UInt) = hash(c.hash, h)
Base.:(==)(c1::ComplexInfo, c2::ComplexInfo) = c1.hash == c2.hash &&
                                               c1.metabolite_idxs == c2.metabolite_idxs && c1.stoichiometry == c2.stoichiometry

"""
    ConcordanceResults

Container for concordance analysis results.
"""
@kwdef struct ConcordanceResults
    complexes::DataFrame
    pairs::DataFrame
    modules::DataFrame
    metadata::Dict{String,Any}
end

"""
    ConcordanceTracker

Efficient disjoint-set data structure with path compression and union by rank,
including non-concordance tracking for transitivity-based filtering.
"""
mutable struct ConcordanceTracker
    parent::Dict{String,String}
    rank::Dict{String,Int}
    non_concordant::Set{Set{String}}

    ConcordanceTracker() = new(Dict{String,String}(), Dict{String,Int}(), Set{Set{String}}())
end

function make_set!(tracker::ConcordanceTracker, x::String)
    if !haskey(tracker.parent, x)
        tracker.parent[x] = x
        tracker.rank[x] = 0
    end
end

function find_set!(tracker::ConcordanceTracker, x::String)
    if tracker.parent[x] != x
        tracker.parent[x] = find_set!(tracker, tracker.parent[x])  # Path compression
    end
    return tracker.parent[x]
end

function union_sets!(tracker::ConcordanceTracker, x::String, y::String)
    root_x = find_set!(tracker, x)
    root_y = find_set!(tracker, y)

    if root_x == root_y
        return
    end

    # Union by rank
    if tracker.rank[root_x] < tracker.rank[root_y]
        tracker.parent[root_x] = root_y
    elseif tracker.rank[root_x] > tracker.rank[root_y]
        tracker.parent[root_y] = root_x
    else
        tracker.parent[root_y] = root_x
        tracker.rank[root_x] += 1
    end
end

function add_non_concordant!(tracker::ConcordanceTracker, x::String, y::String)
    push!(tracker.non_concordant, Set([x, y]))
end

function is_non_concordant(tracker::ConcordanceTracker, x::String, y::String)
    Set([x, y]) in tracker.non_concordant
end

"""
    extract_complexes(model::A.AbstractFBCModel; kwargs...)

Extract complexes from model reactions using efficient sparse operations.
"""
function extract_complexes(model::A.AbstractFBCModel; tolerance=1e-10)
    rxn_ids = A.reactions(model)
    met_ids = A.metabolites(model)
    n_rxns = length(rxn_ids)

    complexes = ComplexInfo[]
    complex_dict = Dict{UInt64,Int}()

    # Build incidence matrix entries
    I_rows = Int32[]
    I_cols = Int32[]
    I_vals = Int8[]

    # Process reactions in parallel chunks if possible
    for (ridx, rxn_id) in enumerate(rxn_ids)
        rxn_stoich = A.reaction_stoichiometry(model, rxn_id)
        isempty(rxn_stoich) && continue

        # Separate substrates and products
        substrate_mets = Int32[]
        substrate_stoich = Float32[]
        product_mets = Int32[]
        product_stoich = Float32[]

        for (met_id, coef) in rxn_stoich
            met_idx = findfirst(==(met_id), met_ids)
            isnothing(met_idx) && continue

            if coef < -tolerance
                push!(substrate_mets, met_idx)
                push!(substrate_stoich, -coef)
            elseif coef > tolerance
                push!(product_mets, met_idx)
                push!(product_stoich, coef)
            end
        end

        # Process substrate complex
        if !isempty(substrate_mets)
            sub_complex = ComplexInfo(substrate_mets, substrate_stoich, met_ids)
            complex_idx = get!(complex_dict, sub_complex.hash) do
                push!(complexes, sub_complex)
                length(complexes)
            end
            push!(I_rows, complex_idx)
            push!(I_cols, ridx)
            push!(I_vals, -1)
        end

        # Process product complex
        if !isempty(product_mets)
            prod_complex = ComplexInfo(product_mets, product_stoich, met_ids)
            complex_idx = get!(complex_dict, prod_complex.hash) do
                push!(complexes, prod_complex)
                length(complexes)
            end
            push!(I_rows, complex_idx)
            push!(I_cols, ridx)
            push!(I_vals, 1)
        end
    end

    # Build sparse incidence matrix
    A_matrix = sparse(I_rows, I_cols, I_vals, length(complexes), n_rxns)

    return complexes, A_matrix
end

"""
    find_trivially_balanced_complexes(complexes, A_matrix, model)

Find complexes that are trivially balanced due to unique metabolites.
"""
function find_trivially_balanced_complexes(complexes::Vector{ComplexInfo}, A_matrix, model::A.AbstractFBCModel)
    n_metabolites = length(A.metabolites(model))

    # Count metabolite appearances in complexes
    met_complex_counts = zeros(Int, n_metabolites)
    met_to_complex = Dict{Int,Int}()

    for (cidx, complex) in enumerate(complexes)
        for met_idx in complex.metabolite_idxs
            met_complex_counts[met_idx] += 1
            if met_complex_counts[met_idx] == 1
                met_to_complex[met_idx] = cidx
            end
        end
    end

    # Find trivially balanced (metabolites in only one complex)
    trivially_balanced = Set{String}()
    for (met_idx, count) in enumerate(met_complex_counts)
        if count == 1 && haskey(met_to_complex, met_idx)
            push!(trivially_balanced, complexes[met_to_complex[met_idx]].id)
        end
    end

    return trivially_balanced
end

"""
    find_trivially_concordant_complexes(complexes, model)

Find trivially concordant complex pairs.
"""
function find_trivially_concordant_complexes(complexes::Vector{ComplexInfo}, model::A.AbstractFBCModel)
    n_metabolites = length(A.metabolites(model))

    # Map metabolites to complexes
    met_to_complexes = [Int[] for _ in 1:n_metabolites]
    for (cidx, complex) in enumerate(complexes)
        for met_idx in complex.metabolite_idxs
            push!(met_to_complexes[met_idx], cidx)
        end
    end

    # Find trivially concordant pairs
    trivially_concordant = Set{Set{String}}()
    for complex_idxs in met_to_complexes
        if length(complex_idxs) == 2
            c1_id = complexes[complex_idxs[1]].id
            c2_id = complexes[complex_idxs[2]].id
            push!(trivially_concordant, Set([c1_id, c2_id]))
        end
    end

    return trivially_concordant
end

"""
    activity_variability_analysis_cobrexa(model, complexes, A_matrix; kwargs...)

Perform activity variability analysis using COBREXA's optimized implementation.
"""
function activity_variability_analysis(
    model::A.AbstractFBCModel,
    complexes::Vector{ComplexInfo},
    A_matrix;
    optimizer=HiGHS.Optimizer,
    workers=D.workers(),
    batch_size=100
)
    n_complexes = length(complexes)

    # Build constraint system
    constraints = COBREXA.flux_balance_constraints(model)

    # Prepare complex activities
    complex_activities = [
        let row = A_matrix[i, :]
            C.sum(
                coef * constraints.fluxes[Symbol(rxn)].value
                for (rxn, coef) in zip(A.reactions(model), row) if coef != 0;
                init=zero(C.LinearValue)
            )
        end for i in 1:n_complexes
    ]

    # Run variability analysis
    results = COBREXA.constraints_variability(
        constraints,
        complex_activities;
        optimizer=optimizer,
        workers=workers
    )

    # Categorize complexes
    categories = Dict{String,Set{String}}(
        "balanced" => Set{String}(),
        "positive" => Set{String}(),
        "negative" => Set{String}(),
        "unrestricted" => Set{String}()
    )

    activity_ranges = Dict{String,Tuple{Float64,Float64}}()

    for (i, (min_val, max_val)) in enumerate(eachrow(results))
        complex_id = complexes[i].id

        if isnothing(min_val) || isnothing(max_val)
            activity_ranges[complex_id] = (0.0, 0.0)
            push!(categories["balanced"], complex_id)
        else
            activity_ranges[complex_id] = (min_val, max_val)

            if abs(min_val) < 1e-9 && abs(max_val) < 1e-9
                push!(categories["balanced"], complex_id)
            elseif min_val >= -1e-9 && max_val > 1e-9
                push!(categories["positive"], complex_id)
            elseif min_val < -1e-9 && max_val <= 1e-9
                push!(categories["negative"], complex_id)
            else
                push!(categories["unrestricted"], complex_id)
            end
        end
    end

    return categories, activity_ranges
end

"""
    check_concordance_charnes_cooper(model, ci, cj, ci_idx, cj_idx, A_matrix; kwargs...)

Check concordance using Charnes-Cooper transformation.
"""
function check_concordance_charnes_cooper(
    model::A.AbstractFBCModel,
    ci::ComplexInfo,
    cj::ComplexInfo,
    ci_idx::Int,
    cj_idx::Int,
    A_matrix;
    direction=:positive,
    optimizer=HiGHS.Optimizer,
    tolerance=1e-9
)
    # Get model information
    rxn_ids = A.reactions(model)
    met_ids = A.metabolites(model)
    n_reactions = length(rxn_ids)
    n_metabolites = length(met_ids)

    # Create a fresh optimization model
    om = J.Model(optimizer)
    J.set_silent(om)

    # Create w variables (representing v * t where v are the original fluxes)
    J.@variable(om, w[1:n_reactions])

    # Create transformation variable t
    J.@variable(om, t)

    # Constrain t based on direction
    if direction == :positive
        J.set_lower_bound(t, tolerance)  # t > 0
    else
        J.set_upper_bound(t, -tolerance)  # t < 0
    end

    # Get stoichiometry matrix and bounds
    S = A.stoichiometry(model)
    lbs, ubs = A.bounds(model)

    # Add steady-state constraint: S * w = 0
    # Since w = v * t and S * v = 0, we have S * w = S * (v * t) = (S * v) * t = 0 * t = 0
    for i in 1:n_metabolites
        J.@constraint(om, sum(S[i, j] * w[j] for j in 1:n_reactions if S[i, j] != 0) == 0)
    end

    # Complex j activity constraint: sum(A[j,k] * w[k]) = ±1
    # Since w = v * t, this gives us: sum(A[j,k] * v[k] * t) = ±1
    # Which means: activity_j * t = ±1
    target = direction == :positive ? 1.0 : -1.0
    cj_activity = sum(A_matrix[cj_idx, k] * w[k] for k in 1:n_reactions if A_matrix[cj_idx, k] != 0)
    J.@constraint(om, cj_activity == target)

    # Add Charnes-Cooper bounds
    # Original bounds: lb ≤ v ≤ ub
    # With w = v * t:
    #   If t > 0: lb * t ≤ w ≤ ub * t
    #   If t < 0: ub * t ≤ w ≤ lb * t (inequalities flip)
    for i in 1:n_reactions
        lb = lbs[i]
        ub = ubs[i]

        if direction == :positive
            # t > 0 case
            if isfinite(lb)
                J.@constraint(om, w[i] >= lb * t)
            end
            if isfinite(ub)
                J.@constraint(om, w[i] <= ub * t)
            end
        else
            # t < 0 case: bounds flip
            if isfinite(ub)
                J.@constraint(om, w[i] >= ub * t)
            end
            if isfinite(lb)
                J.@constraint(om, w[i] <= lb * t)
            end
        end
    end

    # Objective: optimize complex i activity = sum(A[i,k] * w[k])
    ci_activity = sum(A_matrix[ci_idx, k] * w[k] for k in 1:n_reactions if A_matrix[ci_idx, k] != 0)

    # Find minimum
    J.@objective(om, Min, ci_activity)
    J.optimize!(om)

    if J.termination_status(om) != J.OPTIMAL && J.termination_status(om) != J.LOCALLY_SOLVED
        @debug "Minimization failed" status = J.termination_status(om)
        return false, nothing
    end

    min_val = J.objective_value(om)

    # Find maximum
    J.@objective(om, Max, ci_activity)
    J.optimize!(om)

    if J.termination_status(om) != J.OPTIMAL && J.termination_status(om) != J.LOCALLY_SOLVED
        @debug "Maximization failed" status = J.termination_status(om)
        return false, nothing
    end

    max_val = J.objective_value(om)

    # Check if the activity is constant (within tolerance)
    is_concordant = isapprox(min_val, max_val, atol=tolerance)

    # Lambda is the constant ratio
    # Since complex_j_activity = ±1 and complex_i_activity = lambda * complex_j_activity
    # We have lambda = complex_i_activity / complex_j_activity = complex_i_activity / (±1)
    lambda_value = is_concordant ? min_val : nothing

    return is_concordant, lambda_value
end

"""
    sampling_prefilter(model, complexes, A_matrix, active_complexes; kwargs...)

Use flux sampling for correlation-based prefiltering with efficient memory usage.
"""
function sampling_prefilter(
    model::A.AbstractFBCModel,
    complexes::Vector{ComplexInfo},
    A_matrix,
    active_complexes::Set{String},
    categories::Dict{String,Set{String}};
    n_samples=10,
    correlation_threshold=0.85,
    optimizer=HiGHS.Optimizer,
    workers=D.workers(),
    seed=UInt64(42)
)
    active_mask = [c.id in active_complexes for c in complexes]
    active_indices = findall(active_mask)
    n_active = length(active_indices)

    if n_active < 2
        return Tuple{String,String,Symbol}[]
    end

    # Sample fluxes using COBREXA with improved parameters
    samples = COBREXA.flux_sample(
        model;
        n_chains=n_samples,  # Fewer chains for efficiency
        collect_iterations=[max(1, n_samples ÷ 10)],
        optimizer=optimizer,
        workers=workers,
        seed=seed
    )

    if isnothing(samples)
        @warn "Flux sampling failed, falling back to direct candidate generation"
        return Tuple{String,String,Symbol}[]
    end

    # Get reaction symbols in consistent order
    rxn_symbols = Symbol.(A.reactions(model))
    n_reactions = length(rxn_symbols)

    # Convert samples to matrix efficiently
    sample_matrix = Matrix{Float64}(undef, n_reactions, length(samples[rxn_symbols[1]]))
    for (i, rxn) in enumerate(rxn_symbols)
        sample_matrix[i, :] = samples[rxn]
    end

    # Process in batches to avoid memory issues with large matrices
    batch_size = min(50, n_active)  # Process complexes in batches
    candidates = Tuple{String,String,Symbol}[]
    tolerance = 1e-9

    # Track statistics for correlation calculation
    complex_stats = Dict{String,Dict{String,Any}}()

    for batch_start in 1:batch_size:n_active
        batch_end = min(batch_start + batch_size - 1, n_active)
        batch_indices = active_indices[batch_start:batch_end]

        # Calculate activities for this batch
        A_batch = A_matrix[batch_indices, :]
        activities = A_batch * sample_matrix

        # Process each complex in the batch
        for (local_i, global_i) in enumerate(batch_indices)
            ci = complexes[global_i]
            ci_activities = activities[local_i, :]

            # Skip if no variation
            if all(abs(x - ci_activities[1]) < tolerance for x in ci_activities)
                continue
            end

            # Store statistics for this complex
            valid_mask = abs.(ci_activities) .> tolerance
            if sum(valid_mask) < 5  # Need minimum samples
                continue
            end

            valid_activities = ci_activities[valid_mask]
            complex_stats[ci.id] = Dict(
                "mean" => mean(valid_activities),
                "std" => std(valid_activities),
                "n_valid" => sum(valid_mask),
                "activities" => valid_activities
            )
        end
    end

    # Now calculate correlations between all valid complexes
    valid_complex_ids = collect(keys(complex_stats))
    n_valid = length(valid_complex_ids)

    @info "Calculating correlations for $(n_valid) complexes with sufficient variation"

    for i in 1:n_valid
        ci_id = valid_complex_ids[i]
        ci_stats = complex_stats[ci_id]

        for j in (i+1):n_valid
            cj_id = valid_complex_ids[j]
            cj_stats = complex_stats[cj_id]

            # Calculate correlation between valid samples only
            ci_act = ci_stats["activities"]
            cj_act = cj_stats["activities"]

            # Find overlapping valid samples
            min_len = min(length(ci_act), length(cj_act))
            if min_len < 5
                continue
            end

            # Use same indices for both (assuming sampling is synchronized)
            correlation = cor(ci_act[1:min_len], cj_act[1:min_len])

            if abs(correlation) >= correlation_threshold
                # Determine directions based on activity ranges of cj
                directions = _determine_directions_from_ranges(cj_id, categories)

                for direction in directions
                    push!(candidates, (ci_id, cj_id, direction))
                end
            end
        end
    end

    return candidates
end
"""
    _determine_directions_from_ranges(complex_id, activity_ranges, categories)

Determine which directions to test based on complex activity constraints.
"""
function _determine_directions_from_ranges(complex_id::String, categories::Dict)
    directions = Symbol[]

    # Check which category the complex belongs to
    if complex_id in categories["positive"]
        # Complex can only have positive activity
        push!(directions, :positive)
    elseif complex_id in categories["negative"]
        # Complex can only have negative activity  
        push!(directions, :negative)
    else
        # Complex can have both positive and negative activity
        push!(directions, :positive)
        push!(directions, :negative)
    end

    return directions
end

"""
    process_concordance_stage(candidates, tracker, complexes, model, A_matrix; kwargs...)

Process a stage of concordance candidates with transitivity filtering.
"""
function process_concordance_stage(
    candidates::Vector{Tuple{String,String,Symbol}},
    tracker::ConcordanceTracker,
    complexes::Vector{ComplexInfo},
    model::A.AbstractFBCModel,
    A_matrix;
    batch_size=500,
    optimizer=HiGHS.Optimizer,
    workers=D.workers()
)
    # Filter using transitivity
    filtered = Tuple{String,String,Symbol}[]
    complex_to_idx = Dict(c.id => i for (i, c) in enumerate(complexes))

    for (ci_id, cj_id, direction) in candidates
        # Skip if already connected
        find_set!(tracker, ci_id) == find_set!(tracker, cj_id) && continue

        # Skip if known non-concordant
        is_non_concordant(tracker, ci_id, cj_id) && continue

        push!(filtered, (ci_id, cj_id, direction))
    end

    @info "Filtered $(length(candidates)) to $(length(filtered)) using transitivity"

    # Process in batches using COBREXA's screen
    results = []

    # Create batches
    batches = [filtered[i:min(i + batch_size - 1, end)] for i in 1:batch_size:length(filtered)]

    # Process batches using COBREXA's distributed infrastructure
    batch_results = COBREXA.screen(batches; workers=workers) do batch
        local_results = []
        for (ci_id, cj_id, direction) in batch
            ci_idx = complex_to_idx[ci_id]
            cj_idx = complex_to_idx[cj_id]
            ci = complexes[ci_idx]
            cj = complexes[cj_idx]

            is_concordant, lambda_val = check_concordance_charnes_cooper(
                model, ci, cj, ci_idx, cj_idx, A_matrix;
                direction=direction,
                optimizer=optimizer
            )

            push!(local_results, (ci_id, cj_id, is_concordant, lambda_val))
        end
        local_results
    end

    # Flatten batch results
    for batch_result in batch_results
        append!(results, batch_result)
    end

    # Update tracker
    concordant_count = 0
    for (ci_id, cj_id, is_concordant, lambda_val) in results
        if is_concordant
            union_sets!(tracker, ci_id, cj_id)
            concordant_count += 1
        else
            add_non_concordant!(tracker, ci_id, cj_id)
        end
    end

    @info "Stage complete: found $concordant_count concordant pairs"
    return results
end

"""
    find_concordant_complexes(model::A.AbstractFBCModel; kwargs...)

Main entry point for concordance analysis using COBREXA-style implementation.
"""
function find_concordant_complexes(
    model::A.AbstractFBCModel;
    optimizer=HiGHS.Optimizer,
    workers=D.workers(),
    sample_size::Maybe{Int}=1000,
    correlation_threshold=0.85,
    max_iterations=10,
    stage_size=10000,
    batch_size=500,
    seed=42
)
    start_time = time()
    seed = UInt64(seed)
    @info "Starting concordance analysis" optimizer = optimizer workers = length(workers)

    # Step 1: Extract complexes
    complexes, A_matrix = extract_complexes(model)
    n_complexes = length(complexes)
    n_metabolites = length(A.metabolites(model))

    @info "Extracted $n_complexes complexes from $(length(A.reactions(model))) reactions"

    # Step 2: Find trivial relationships
    trivially_balanced = find_trivially_balanced_complexes(complexes, A_matrix, model)
    trivially_concordant = find_trivially_concordant_complexes(complexes, model)

    @info "Found $(length(trivially_balanced)) trivially balanced complexes"
    @info "Found $(length(trivially_concordant)) trivially concordant pairs"

    # Step 3: Activity variability analysis
    categories, activity_ranges = activity_variability_analysis(
        model, complexes, A_matrix;
        optimizer=optimizer,
        workers=workers
    )
    @info "Activity variability analysis complete"
    @info "Balanced complexes: $(length(categories["balanced"]))"
    @info "Positive complexes: $(length(categories["positive"]))"
    @info "Negative complexes: $(length(categories["negative"]))"
    @info "Unrestricted complexes: $(length(categories["unrestricted"]))"
    # Step 4: Initialize concordance tracker
    tracker = ConcordanceTracker()
    for complex in complexes
        make_set!(tracker, complex.id)
    end

    # Add trivial concordances
    for pair in trivially_concordant
        c1, c2 = collect(pair)
        union_sets!(tracker, c1, c2)
    end

    # Add balanced concordances
    balanced_list = collect(categories["balanced"])
    if length(balanced_list) > 1
        for i in 2:length(balanced_list)
            union_sets!(tracker, balanced_list[1], balanced_list[i])
        end
    end

    # Step 5: Generate candidates
    active_complexes = setdiff(
        Set(c.id for c in complexes),
        categories["balanced"]
    )

    candidates = if !isnothing(sample_size) && sample_size > 0
        sampling_prefilter(
            model, complexes, A_matrix, active_complexes, categories;
            n_samples=sample_size,
            correlation_threshold=correlation_threshold,
            optimizer=optimizer,
            workers=workers,
            seed=seed
        )
    else
        # Generate all non-trivial candidates
        all_candidates = Tuple{String,String,Symbol}[]
        for (i, ci) in enumerate(complexes)
            ci.id in categories["balanced"] && continue
            for (j, cj) in enumerate(complexes)
                i >= j && continue
                cj.id in categories["balanced"] && continue
                Set([ci.id, cj.id]) in trivially_concordant && continue

                # Determine direction based on categories
                if cj.id in categories["positive"]
                    push!(all_candidates, (ci.id, cj.id, :positive))
                elseif cj.id in categories["negative"]
                    push!(all_candidates, (ci.id, cj.id, :negative))
                else
                    push!(all_candidates, (ci.id, cj.id, :positive))
                    push!(all_candidates, (ci.id, cj.id, :negative))
                end
            end
        end
        all_candidates
    end

    @info "Generated $(length(candidates)) candidate pairs"

    # Step 6: Iterative concordance testing
    all_results = []
    iteration = 1

    while !isempty(candidates) && iteration <= max_iterations
        @info "Iteration $iteration with $(length(candidates)) candidates"

        # Take a stage
        stage_candidates = candidates[1:min(stage_size, length(candidates))]
        remaining = candidates[min(stage_size + 1, length(candidates)):end]

        # Process stage
        stage_results = process_concordance_stage(
            stage_candidates, tracker, complexes, model, A_matrix;
            batch_size=batch_size,
            optimizer=optimizer,
            workers=workers
        )

        append!(all_results, stage_results)
        candidates = remaining
        iteration += 1
    end

    # Step 7: Extract modules and create results
    modules = extract_modules_from_tracker(tracker, complexes, categories["balanced"])

    # Create result DataFrames
    results = create_result_dataframes(
        complexes, modules, all_results,
        categories, activity_ranges,
        trivially_concordant, tracker
    )

    elapsed = time() - start_time
    @info "Concordance analysis complete" time = elapsed modules = length(modules)

    return results
end

"""
    extract_modules_from_tracker(tracker, complexes, balanced_set)

Extract concordance modules from the disjoint set tracker.
"""
function extract_modules_from_tracker(
    tracker::ConcordanceTracker,
    complexes::Vector{ComplexInfo},
    balanced_set::Set{String}
)
    modules = Dict{String,Set{String}}()

    # Group by representative
    groups = Dict{String,Vector{String}}()
    for complex in complexes
        root = find_set!(tracker, complex.id)
        if !haskey(groups, root)
            groups[root] = String[]
        end
        push!(groups[root], complex.id)
    end

    # Create modules
    if !isempty(balanced_set)
        modules["balanced"] = balanced_set
    end

    module_idx = 1
    for (root, members) in groups
        if length(members) > 1
            member_set = Set(members)

            # Skip if it's the balanced module
            if !isempty(balanced_set) && member_set ⊆ balanced_set
                continue
            end

            modules["module$module_idx"] = member_set
            module_idx += 1
        end
    end

    return modules
end

"""
    create_result_dataframes(...)

Create result DataFrames from analysis results.
"""
function create_result_dataframes(
    complexes::Vector{ComplexInfo},
    modules::Dict{String,Set{String}},
    concordance_results::Vector,
    categories::Dict{String,Set{String}},
    activity_ranges::Dict{String,Tuple{Float64,Float64}},
    trivial_pairs::Set{Set{String}},
    tracker::ConcordanceTracker
)
    # Complexes DataFrame
    complex_data = []
    for complex in complexes
        module_id = findfirst(m -> complex.id in m[2], collect(modules))
        module_name = isnothing(module_id) ? nothing : collect(keys(modules))[module_id]

        min_act, max_act = get(activity_ranges, complex.id, (NaN, NaN))

        push!(complex_data, (
            complex_id=complex.id,
            module_id=module_name,
            min_activity=min_act,
            max_activity=max_act,
            is_balanced=complex.id in categories["balanced"]
        ))
    end
    complexes_df = DataFrame(complex_data)

    # Pairs DataFrame
    pairs_data = []
    seen_pairs = Set{Set{String}}()

    for (ci_id, cj_id, is_concordant, lambda_val) in concordance_results
        if is_concordant
            pair = Set([ci_id, cj_id])
            pair in seen_pairs && continue
            push!(seen_pairs, pair)

            push!(pairs_data, (
                complex1=ci_id < cj_id ? ci_id : cj_id,
                complex2=ci_id < cj_id ? cj_id : ci_id,
                is_trivial=pair in trivial_pairs,
                lambda_value=something(lambda_val, NaN)
            ))
        end
    end
    pairs_df = DataFrame(pairs_data)

    # Modules DataFrame
    modules_data = []
    for (module_id, members) in modules
        push!(modules_data, (
            module_id=module_id,
            size=length(members),
            complexes=join(sort(collect(members)), ", ")
        ))
    end
    modules_df = DataFrame(modules_data)

    # Metadata
    metadata = Dict{String,Any}(
        "total_complexes" => length(complexes),
        "balanced_complexes" => length(categories["balanced"]),
        "concordant_pairs" => nrow(pairs_df),
        "trivial_pairs" => length(trivial_pairs),
        "modules" => length(modules)
    )

    return ConcordanceResults(
        complexes=complexes_df,
        pairs=pairs_df,
        modules=modules_df,
        metadata=metadata
    )
end

end # module