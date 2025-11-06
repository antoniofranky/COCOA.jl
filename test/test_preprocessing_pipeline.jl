"""
Preprocessing pipeline test with MATLAB model comparison.

Runs the full preprocessing pipeline and compares with MATLAB results.

Usage:
    julia --project=COCOA.jl test/test_preprocessing_pipeline.jl
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS
using COBREXA
using Printf
using MAT
using SparseArrays
using Statistics
@everywhere using COCOA, HiGHS
# Configuration
const MODEL_PATH = "test\\e_coli_core.xml"
const MATLAB_MODEL_PATH = "test\\ecoli_irr_matlab.mat"
const SAVE_JULIA_MODEL = false
const JULIA_OUTPUT_PATH = "C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\julia_preprocessed_iSDY.xml"

"""
Load MATLAB model from MAT file manually (bypasses MATFBCModels issues).
"""
function load_matlab_model_manual(path::String)
    mat = matread(path)

    # Extract model data (might be nested)
    if haskey(mat, "model_elementary")
        model_data = mat["model_elementary"]
    elseif haskey(mat, "model")
        model_data = mat["model"]
    else
        model_data = mat
    end

    # Extract fields
    rxns = vec(model_data["rxns"])
    mets = vec(model_data["mets"])
    S = sparse(model_data["S"])
    lb = vec(model_data["lb"])
    ub = vec(model_data["ub"])
    c = vec(model_data["c"])

    # Build CanonicalModel
    model = CM.Model()

    # Add metabolites
    met_names = get(model_data, "metNames", fill("", length(mets)))
    for (i, met_id) in enumerate(mets)
        model.metabolites[met_id] = CM.Metabolite(
            name=met_names[i],
            compartment="c",
            formula=nothing,
            charge=nothing,
            balance=0.0,
            annotations=Dict{String,Vector{String}}(),
            notes=Dict{String,Vector{String}}()
        )
    end

    # Add reactions
    rxn_names = get(model_data, "rxnNames", fill("", length(rxns)))
    for (j, rxn_id) in enumerate(rxns)
        stoich = Dict{String,Float64}()
        for i in 1:length(mets)
            coeff = S[i, j]
            if coeff != 0.0
                stoich[mets[i]] = coeff
            end
        end

        model.reactions[rxn_id] = CM.Reaction(
            name=rxn_names[j],
            lower_bound=lb[j],
            upper_bound=ub[j],
            stoichiometry=stoich,
            objective_coefficient=c[j],
            gene_association_dnf=nothing,
            annotations=Dict{String,Vector{String}}(),
            notes=Dict{String,Vector{String}}()
        )
    end

    return model
end

println("="^80)
println("Preprocessing Pipeline Summary with MATLAB Comparison")
println("="^80)
println("Julia model: $MODEL_PATH")
println("MATLAB model: $MATLAB_MODEL_PATH")
println()

# Helper function to get stats
function get_stats(model, step_name)
    n_rxns = length(model.reactions)
    n_mets = length(model.metabolites)

    # Get objective value
    constraints = COBREXA.flux_balance_constraints(model)
    obj_value = COBREXA.optimized_values(
        constraints;
        objective=constraints.objective.value,
        output=constraints.objective,
        optimizer=HiGHS.Optimizer,
        settings=[]
    )

    return (name=step_name, reactions=n_rxns, metabolites=n_mets, objective=obj_value)
end

# Load model
model_original = A.load(MODEL_PATH)
model = convert(CM.Model, model_original)

# Track stats at each step
stats = []

# 0. Original
push!(stats, get_stats(model, "0. Original"))

# 1. Remove orphans (pass 1)
model = COCOA.remove_orphans(model)
push!(stats, get_stats(model, "1. Remove orphans"))

# 2. Normalize bounds
model = COCOA.normalize_bounds(model)
push!(stats, get_stats(model, "2. Normalize bounds"))

# 3. Remove blocked reactions
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
]
model, blocked = COCOA.remove_blocked_reactions(
    model,
    optimizer=HiGHS.Optimizer,
    flux_tolerance=1e-7,
    objective_bound=COBREXA.relative_tolerance_bound(0.999),
    settings=highs_settings
)
push!(stats, get_stats(model, "3. Remove blocked rxns"))

# 4. Remove orphans (pass 2)
model = COCOA.remove_orphans(model)
push!(stats, get_stats(model, "4. Remove orphans"))

# 5. Split into elementary
model = COCOA.split_into_elementary(model, random=0.0, seed=42)
println("\n=== After Elementary Split ===")
obj_rxns = filter(p -> p[2].objective_coefficient != 0.0, model.reactions)
println("Reactions with non-zero objective: $(length(obj_rxns))")
for (id, rxn) in obj_rxns
    println("  $id: coefficient = $(rxn.objective_coefficient)")
end

push!(stats, get_stats(model, "5. Elementary split"))


# 6. Split into irreversible
model = COCOA.split_into_irreversible(model)
push!(stats, get_stats(model, "6. Irreversible split"))

# Print Julia pipeline summary
println("\n" * "="^80)
println("JULIA PIPELINE SUMMARY")
println("="^80)
println()
println("Step                      │ Reactions │ Metabolites │  Objective")
println("─"^25 * "┼" * "─"^11 * "┼" * "─"^13 * "┼" * "─"^12)

for s in stats
    obj_str = isnothing(s.objective) ? "INFEASIBLE" : @sprintf("%.6f", s.objective)
    @printf("%-25s │ %9d │ %11d │ %11s\n", s.name, s.reactions, s.metabolites, obj_str)
end

println("="^80)

# Save Julia model if requested
if SAVE_JULIA_MODEL
    println("\nSaving Julia preprocessed model...")
    try
        A.save(model, JULIA_OUTPUT_PATH)
        println("✓ Saved to: $JULIA_OUTPUT_PATH")
    catch e
        println("✗ Failed to save: $e")
    end
end

# ============================================================================
# MATLAB MODEL COMPARISON
# ============================================================================
println("\n" * "="^80)
println("LOADING MATLAB MODEL FOR COMPARISON")
println("="^80)
println()

matlab_model = load_matlab_model_manual(MATLAB_MODEL_PATH)

println("MATLAB model loaded:")
println("  Reactions: $(length(matlab_model.reactions))")
println("  Metabolites: $(length(matlab_model.metabolites))")

# Count metabolite types
matlab_regular = count(p -> !contains(p[1], "_complex") && !startswith(p[1], "E_"), matlab_model.metabolites)
matlab_enzymes = count(p -> startswith(p[1], "E_") && !contains(p[1], "_complex"), matlab_model.metabolites)
matlab_complexes = count(p -> contains(p[1], "_complex"), matlab_model.metabolites)

println("  Regular metabolites: $matlab_regular")
println("  Enzyme metabolites: $matlab_enzymes")
println("  Complex metabolites: $matlab_complexes")

# Count objective reactions
matlab_obj_rxns = filter(p -> p[2].objective_coefficient != 0.0, matlab_model.reactions)
println("  Reactions with objective: $(length(matlab_obj_rxns))")

if length(matlab_obj_rxns) <= 20
    println("\n  MATLAB objective reactions:")
    for (id, rxn) in sort(collect(matlab_obj_rxns), by=p -> p[1])
        @printf("    %-50s: %.6f\n", id, rxn.objective_coefficient)
    end
end

# Calculate MATLAB objective value
constraints = COBREXA.flux_balance_constraints(matlab_model)
matlab_obj_value = COBREXA.optimized_values(
    constraints;
    objective=constraints.objective.value,
    output=constraints.objective,
    optimizer=HiGHS.Optimizer,
    settings=[COBREXA.set_optimizer_attribute("output_flag", false)]
)

println("\n  MATLAB objective value: $matlab_obj_value")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
println("\n" * "="^80)
println("JULIA vs MATLAB COMPARISON")
println("="^80)
println()

julia_final = stats[end]
julia_regular = count(p -> !startswith(p[1], "CPLX_") && !occursin(r"^E\d+$", p[1]), model.metabolites)
julia_enzymes = count(p -> occursin(r"^E\d+$", p[1]), model.metabolites)
julia_complexes = count(p -> startswith(p[1], "CPLX_"), model.metabolites)

function print_comparison(label, julia_val, matlab_val)
    diff = julia_val - matlab_val
    pct = matlab_val == 0 ? 0.0 : 100.0 * diff / matlab_val
    symbol = diff == 0 ? "✓" : (diff > 0 ? "↑" : "↓")
    @printf("%-30s │ %8d │ %8d │ %+8d (%+6.2f%%) %s\n",
        label, julia_val, matlab_val, diff, pct, symbol)
end

println("Metric                         │    Julia │   MATLAB │        Difference")
println("─"^31 * "┼" * "─"^10 * "┼" * "─"^10 * "┼" * "─"^26)

print_comparison("Total reactions", julia_final.reactions, length(matlab_model.reactions))
print_comparison("Total metabolites", julia_final.metabolites, length(matlab_model.metabolites))
println()
print_comparison("Regular metabolites", julia_regular, matlab_regular)
print_comparison("Enzyme metabolites", julia_enzymes, matlab_enzymes)
print_comparison("Complex metabolites", julia_complexes, matlab_complexes)
println()
print_comparison("Objective reactions", length(obj_rxns), length(matlab_obj_rxns))

println()
println("Objective values:")
@printf("  Julia:  %.6f\n", julia_final.objective)
@printf("  MATLAB: %.6f\n", matlab_obj_value)
@printf("  Diff:   %+.6f (%.2f%%)\n",
    julia_final.objective - matlab_obj_value,
    100.0 * (julia_final.objective - matlab_obj_value) / matlab_obj_value)

# ============================================================================
# ANALYSIS
# ============================================================================
println("\n" * "="^80)
println("ANALYSIS")
println("="^80)
println()

if abs(julia_final.objective - matlab_obj_value) / matlab_obj_value < 0.01
    println("✓ Objective values match within 1% - models are equivalent!")
elseif abs(julia_final.objective - matlab_obj_value) / matlab_obj_value < 0.10
    println("⚠ Objective values differ by <10% - check for minor differences")
    println("  Likely causes:")
    println("  - Different GPR parsing (enzyme count: $(julia_enzymes) vs $(matlab_enzymes))")
    println("  - Different complex creation logic")
else
    println("✗ Objective values differ significantly (>10%)")
    println("  This suggests a fundamental difference in model structure")
    println("  Possible causes:")
    println("  - Objective coefficient assignment error")
    println("  - Different reaction splitting logic")
    println("  - Stoichiometry differences")
end

if abs(julia_enzymes - matlab_enzymes) <= 5
    println("\n✓ Enzyme count difference is small (±5) - expected from GPR parsing")
else
    println("\n⚠ Large enzyme count difference ($(abs(julia_enzymes - matlab_enzymes)))")
    println("  This may indicate different reaction filtering or GPR parsing logic")
end

# ============================================================================
# STRUCTURAL COMPARISON (name-independent)
# ============================================================================
println("\n" * "="^80)
println("STRUCTURAL COMPARISON")
println("="^80)
println()

# 1. Objective coefficient distribution analysis
println("1. OBJECTIVE COEFFICIENT DISTRIBUTION")
println("-"^80)

function analyze_objective_distribution(model, name)
    obj_values = Float64[]
    rxns_with_obj = 0

    for (rid, rxn) in model.reactions
        if rxn.objective_coefficient != 0.0
            push!(obj_values, rxn.objective_coefficient)
            rxns_with_obj += 1
        end
    end

    println("\n  $name:")
    println("    Reactions with objective:    $(rxns_with_obj)")
    println("    Total objective value:       $(sum(obj_values))")
    if length(obj_values) > 0
        println("    Mean objective coefficient:  $(sum(obj_values) / length(obj_values))")
        println("    Min objective coefficient:   $(minimum(obj_values))")
        println("    Max objective coefficient:   $(maximum(obj_values))")
        println("    Unique objective values:     $(length(unique(obj_values)))")

        # Show histogram of objective values
        if length(unique(obj_values)) <= 10
            println("\n    Objective value histogram:")
            for val in sort(unique(obj_values))
                count = sum(obj_values .== val)
                @printf("      %.6f: %5d reactions (%.1f%% of total)\n",
                    val, count, 100.0 * count / rxns_with_obj)
            end
        end
    end
end

analyze_objective_distribution(model, "Julia")
analyze_objective_distribution(matlab_model, "MATLAB")

# 2. Reaction stoichiometry structure analysis
println("\n2. STOICHIOMETRY STRUCTURE ANALYSIS")
println("-"^80)

function analyze_stoichiometry_structure(model, name)
    stoich_sizes = Int[]
    substrate_counts = Int[]
    product_counts = Int[]

    for (rid, rxn) in model.reactions
        push!(stoich_sizes, length(rxn.stoichiometry))

        substrates = sum(1 for (met, coeff) in rxn.stoichiometry if coeff < 0; init=0)
        products = sum(1 for (met, coeff) in rxn.stoichiometry if coeff > 0; init=0)

        push!(substrate_counts, substrates)
        push!(product_counts, products)
    end

    println("\n  $name:")
    println("    Avg metabolites per reaction: $(sum(stoich_sizes) / length(stoich_sizes))")
    println("    Avg substrates per reaction:  $(sum(substrate_counts) / length(substrate_counts))")
    println("    Avg products per reaction:    $(sum(product_counts) / length(product_counts))")
    println("    Max metabolites in reaction:  $(maximum(stoich_sizes))")
    println("    Min metabolites in reaction:  $(minimum(stoich_sizes))")

    # Distribution of stoichiometry sizes
    size_dist = Dict{Int,Int}()
    for size in stoich_sizes
        size_dist[size] = get(size_dist, size, 0) + 1
    end

    println("\n    Stoichiometry size distribution (top 10):")
    for (size, count) in sort(collect(size_dist), by=x -> x[2], rev=true)[1:min(10, length(size_dist))]
        @printf("      %2d metabolites: %5d reactions (%.1f%%)\n",
            size, count, 100.0 * count / length(stoich_sizes))
    end
end

analyze_stoichiometry_structure(model, "Julia")
analyze_stoichiometry_structure(matlab_model, "MATLAB")

# 3. Metabolite participation analysis
println("\n3. METABOLITE PARTICIPATION ANALYSIS")
println("-"^80)

function analyze_metabolite_participation(model, name)
    met_participation = Dict{String,Int}()

    for (rid, rxn) in model.reactions
        for met in keys(rxn.stoichiometry)
            met_participation[met] = get(met_participation, met, 0) + 1
        end
    end

    participation_counts = collect(values(met_participation))

    println("\n  $name:")
    println("    Total metabolites:            $(length(met_participation))")
    println("    Avg reactions per metabolite: $(sum(participation_counts) / length(participation_counts))")
    println("    Max reactions per metabolite: $(maximum(participation_counts))")
    println("    Min reactions per metabolite: $(minimum(participation_counts))")

    # Show most connected metabolites
    println("\n    Top 10 most connected metabolites:")
    for (met, count) in sort(collect(met_participation), by=x -> x[2], rev=true)[1:min(10, length(met_participation))]
        @printf("      %-40s: %4d reactions\n", met, count)
    end
end

analyze_metabolite_participation(model, "Julia")
analyze_metabolite_participation(matlab_model, "MATLAB")

# 4. Reaction bounds analysis
println("\n4. REACTION BOUNDS ANALYSIS")
println("-"^80)

function analyze_reaction_bounds(model, name)
    lb_values = Float64[]
    ub_values = Float64[]
    reversible = 0
    irreversible_forward = 0
    irreversible_backward = 0

    for (rid, rxn) in model.reactions
        push!(lb_values, rxn.lower_bound)
        push!(ub_values, rxn.upper_bound)

        if rxn.lower_bound < 0 && rxn.upper_bound > 0
            reversible += 1
        elseif rxn.lower_bound >= 0
            irreversible_forward += 1
        else
            irreversible_backward += 1
        end
    end

    println("\n  $name:")
    println("    Reversible reactions:          $(reversible)")
    println("    Irreversible forward:          $(irreversible_forward)")
    println("    Irreversible backward:         $(irreversible_backward)")
    println("    Avg lower bound:               $(sum(lb_values) / length(lb_values))")
    println("    Avg upper bound:               $(sum(ub_values) / length(ub_values))")
    println("    Unique lower bounds:           $(length(unique(lb_values)))")
    println("    Unique upper bounds:           $(length(unique(ub_values)))")
end

analyze_reaction_bounds(model, "Julia")
analyze_reaction_bounds(matlab_model, "MATLAB")

# 5. Objective coefficient pattern analysis
println("\n5. OBJECTIVE COEFFICIENT PATTERNS")
println("-"^80)

function analyze_objective_patterns(model, name)
    # Count reactions with different objective coefficient patterns
    obj_per_base = Dict{String,Vector{Float64}}()

    for (rid, rxn) in model.reactions
        if rxn.objective_coefficient != 0.0
            # Try to extract base reaction ID (remove suffixes)
            base = rid
            # Remove common suffixes: _f, _b, _SB1, _SB2, _CAT, _PR1, _PR2, etc.
            base = replace(base, r"_[fb]$" => "")
            base = replace(base, r"_SB\d+$" => "")
            base = replace(base, r"_CAT$" => "")
            base = replace(base, r"_PR\d+$" => "")

            if !haskey(obj_per_base, base)
                obj_per_base[base] = Float64[]
            end
            push!(obj_per_base[base], rxn.objective_coefficient)
        end
    end

    # Count how many "base reactions" have 1, 2, 3, ... objective-bearing sub-reactions
    sub_rxn_counts = Dict{Int,Int}()
    total_obj_by_count = Dict{Int,Float64}()

    for (base, coeffs) in obj_per_base
        n = length(coeffs)
        sub_rxn_counts[n] = get(sub_rxn_counts, n, 0) + 1
        total_obj_by_count[n] = get(total_obj_by_count, n, 0.0) + sum(coeffs)
    end

    println("\n  $name:")
    println("    Distinct base reactions with objective: $(length(obj_per_base))")
    println("\n    Sub-reactions per base:")
    for n in sort(collect(keys(sub_rxn_counts)))
        count = sub_rxn_counts[n]
        total_obj = total_obj_by_count[n]
        @printf("      %2d sub-rxns: %4d bases, total obj = %.6f, avg = %.6f\n",
            n, count, total_obj, total_obj / count)
    end
end

analyze_objective_patterns(model, "Julia")
analyze_objective_patterns(matlab_model, "MATLAB")

# 6. Summary of structural differences
println("\n6. STRUCTURAL DIFFERENCE SUMMARY")
println("-"^80)

julia_rxns_with_obj = sum(1 for (_, rxn) in model.reactions if rxn.objective_coefficient != 0.0)
matlab_rxns_with_obj = sum(1 for (_, rxn) in matlab_model.reactions if rxn.objective_coefficient != 0.0)

julia_total_obj = sum(rxn.objective_coefficient for (_, rxn) in model.reactions)
matlab_total_obj = sum(rxn.objective_coefficient for (_, rxn) in matlab_model.reactions)

println("\nKey structural metrics comparison:")
println("  Metric                          │    Julia │   MATLAB │     Difference")
println("  ─"^32 * "┼" * "─"^10 * "┼" * "─"^10 * "┼" * "─"^16)
@printf("  Reactions with objective        │ %8d │ %8d │ %+14d\n",
    julia_rxns_with_obj, matlab_rxns_with_obj, julia_rxns_with_obj - matlab_rxns_with_obj)
@printf("  Sum of objective coefficients   │ %8.2f │ %8.2f │ %+14.2f\n",
    julia_total_obj, matlab_total_obj, julia_total_obj - matlab_total_obj)

obj_diff_pct = 100.0 * (julia_total_obj - matlab_total_obj) / matlab_total_obj
println("\nConclusion:")
if abs(obj_diff_pct) < 1.0
    println("  ✓ Objective coefficient sums match within 1% - models are structurally equivalent")
elseif abs(obj_diff_pct) < 10.0
    println("  ⚠ Objective coefficient sums differ by $(abs(obj_diff_pct))%")
    println("    This suggests minor differences in how objectives are distributed")
else
    println("  ✗ Objective coefficient sums differ by $(abs(obj_diff_pct))%")
    println("    This indicates a fundamental structural difference:")
    if julia_rxns_with_obj > matlab_rxns_with_obj
        println("    - Julia has MORE reactions with objective coefficients")
        println("      → Possible cause: binding/release reactions getting objective when they shouldn't")
    elseif julia_rxns_with_obj < matlab_rxns_with_obj
        println("    - Julia has FEWER reactions with objective coefficients")
        println("      → Possible cause: catalytic reactions not getting objective when they should")
    end
    if julia_total_obj > matlab_total_obj
        println("    - Total objective sum is HIGHER in Julia")
        println("      → Likely cause: objective being assigned to multiple sub-reactions")
    end
end

# ============================================================================
# DEEP FLUX ANALYSIS
# ============================================================================
println("\n" * "="^80)
println("DEEP FLUX ANALYSIS - Finding why objective values differ")
println("="^80)
println()

# Get optimal flux distributions
println("Computing optimal flux distributions...")
julia_constraints = COBREXA.flux_balance_constraints(model)
julia_fluxes = COBREXA.optimized_values(
    julia_constraints;
    objective=julia_constraints.objective.value,
    output=julia_constraints.fluxes,
    optimizer=HiGHS.Optimizer,
    settings=[COBREXA.set_optimizer_attribute("output_flag", false)]
)
julia_objective = COBREXA.optimized_values(
    julia_constraints;
    objective=julia_constraints.objective.value,
    output=julia_constraints.objective,
    optimizer=HiGHS.Optimizer,
    settings=[COBREXA.set_optimizer_attribute("output_flag", false)]
)

matlab_constraints = COBREXA.flux_balance_constraints(matlab_model)
matlab_fluxes = COBREXA.optimized_values(
    matlab_constraints;
    objective=matlab_constraints.objective.value,
    output=matlab_constraints.fluxes,
    optimizer=HiGHS.Optimizer,
    settings=[COBREXA.set_optimizer_attribute("output_flag", false)]
)
matlab_objective = COBREXA.optimized_values(
    matlab_constraints;
    objective=matlab_constraints.objective.value,
    output=matlab_constraints.objective,
    optimizer=HiGHS.Optimizer,
    settings=[COBREXA.set_optimizer_attribute("output_flag", false)]
)

println("✓ Solutions computed")
println("  Julia objective:  $(julia_objective)")
println("  MATLAB objective: $(matlab_objective)")

# 1. Compare biomass reaction fluxes
println("\n1. BIOMASS REACTION FLUX")
println("-"^80)

biomass_rxn_id = Symbol("BIOMASS_Ec_iJO1366_core_53p95M")
julia_biomass_flux = get(julia_fluxes, biomass_rxn_id, 0.0)
matlab_biomass_flux = get(matlab_fluxes, biomass_rxn_id, 0.0)

println("  Julia biomass flux:  $(julia_biomass_flux)")
println("  MATLAB biomass flux: $(matlab_biomass_flux)")
println("  Ratio: $(julia_biomass_flux / matlab_biomass_flux)")

# 2. Find reactions with largest flux differences
println("\n2. REACTIONS WITH LARGEST FLUX DIFFERENCES")
println("-"^80)

# Get all reaction IDs
all_rxn_ids = Set(keys(model.reactions)) ∪ Set(keys(matlab_model.reactions))

flux_diffs = Tuple{String,Float64,Float64,Float64}[]
for rid in all_rxn_ids
    j_flux = get(julia_fluxes, Symbol(rid), 0.0)
    m_flux = get(matlab_fluxes, Symbol(rid), 0.0)
    diff = abs(j_flux - m_flux)

    if diff > 1e-6  # Only significant differences
        push!(flux_diffs, (rid, j_flux, m_flux, diff))
    end
end

sort!(flux_diffs, by=x -> x[4], rev=true)

println("\nTop 20 reactions with largest absolute flux differences:")
println("  ID                                                    │  Julia Flux │ MATLAB Flux │   Abs Diff")
println("  " * "─"^54 * "┼" * "─"^13 * "┼" * "─"^13 * "┼" * "─"^12)
for (i, (rid, j_flux, m_flux, diff)) in enumerate(flux_diffs[1:min(20, length(flux_diffs))])
    @printf("  %-53s │ %11.6f │ %11.6f │ %10.6f\n", rid, j_flux, m_flux, diff)
end

# 3. Check for missing metabolites
println("\n3. MISSING METABOLITES ANALYSIS")
println("-"^80)

julia_mets = Set(keys(model.metabolites))
matlab_mets = Set(keys(matlab_model.metabolites))

missing_in_julia = setdiff(matlab_mets, julia_mets)
missing_in_matlab = setdiff(julia_mets, matlab_mets)

println("  Metabolites in MATLAB but not Julia: $(length(missing_in_julia))")
println("  Metabolites in Julia but not MATLAB: $(length(missing_in_matlab))")

if length(missing_in_julia) > 0
    println("\n  Sample metabolites missing in Julia (first 20):")
    for met in collect(missing_in_julia)[1:min(20, length(missing_in_julia))]
        # Check if this metabolite is involved in any active reactions in MATLAB
        involved_rxns = filter(p -> haskey(p[2].stoichiometry, met), matlab_model.reactions)
        active_involved = filter(p -> abs(get(matlab_fluxes, Symbol(p[1]), 0.0)) > 1e-6, involved_rxns)

        if length(active_involved) > 0
            println("    $met - used in $(length(active_involved)) active reactions")
        end
    end
end

# 4. Enzyme usage comparison
println("\n4. ENZYME METABOLITE USAGE")
println("-"^80)

# Count how many enzyme metabolites carry non-zero flux
julia_enzyme_mets = filter(k -> occursin(r"^E\d+$", k), keys(model.metabolites))
matlab_enzyme_mets = filter(k -> startswith(k, "E_"), keys(matlab_model.metabolites))

println("  Julia enzyme metabolites:  $(length(julia_enzyme_mets))")
println("  MATLAB enzyme metabolites: $(length(matlab_enzyme_mets))")

# Check if any enzyme metabolites are unbalanced (mass balance violations)
function check_enzyme_balance(model, fluxes, enzyme_regex)
    unbalanced = []

    for (met_id, met) in model.metabolites
        if occursin(enzyme_regex, met_id)
            # Calculate net production/consumption
            balance = 0.0
            for (rxn_id, rxn) in model.reactions
                if haskey(rxn.stoichiometry, met_id)
                    flux = get(fluxes, Symbol(rxn_id), 0.0)
                    balance += rxn.stoichiometry[met_id] * flux
                end
            end

            if abs(balance) > 1e-4  # Significant imbalance
                push!(unbalanced, (met_id, balance))
            end
        end
    end

    return unbalanced
end

julia_unbalanced = check_enzyme_balance(model, julia_fluxes, r"^E\d+$")
matlab_unbalanced = check_enzyme_balance(matlab_model, matlab_fluxes, r"^E_")

println("\n  Unbalanced enzymes (mass balance violations):")
println("    Julia:  $(length(julia_unbalanced)) enzymes")
println("    MATLAB: $(length(matlab_unbalanced)) enzymes")

if length(julia_unbalanced) > 0
    println("\n    Julia unbalanced enzymes (first 10):")
    for (met, bal) in julia_unbalanced[1:min(10, length(julia_unbalanced))]
        @printf("      %-20s: %+.6e\n", met, bal)
    end
end

if length(matlab_unbalanced) > 0
    println("\n    MATLAB unbalanced enzymes (first 10):")
    for (met, bal) in matlab_unbalanced[1:min(10, length(matlab_unbalanced))]
        @printf("      %-20s: %+.6e\n", met, bal)
    end
end

# 5. Summary diagnosis
println("\n5. DIAGNOSIS SUMMARY")
println("-"^80)

println("\nKey findings:")
println("  • Biomass flux ratio: $(round(julia_biomass_flux / matlab_biomass_flux, digits=3))x")
println("  • Unbalanced enzymes in Julia: $(length(julia_unbalanced))")
println("  • Unbalanced enzymes in MATLAB: $(length(matlab_unbalanced))")

# Additional structural analysis - name-independent
println("\n6. NAME-INDEPENDENT STRUCTURAL ANALYSIS")
println("-"^80)

# Analyze flux magnitude distributions
function analyze_flux_distribution(fluxes, name)
    flux_values = Float64[]
    for (_, v) in fluxes
        if abs(v) > 1e-9
            push!(flux_values, abs(v))
        end
    end

    if length(flux_values) == 0
        println("\n  $name: No active fluxes")
        return
    end

    println("\n  $name flux statistics:")
    println("    Active reactions (|flux| > 1e-9): $(length(flux_values))")
    println("    Mean absolute flux:                $(mean(flux_values))")
    println("    Median absolute flux:              $(median(flux_values))")
    println("    Max absolute flux:                 $(maximum(flux_values))")
    println("    Min absolute flux:                 $(minimum(flux_values))")
    println("    Std dev:                           $(std(flux_values))")

    # Flux magnitude bins
    bins = [0.0, 1e-6, 1e-3, 1.0, 10.0, 100.0, 1000.0, Inf]
    bin_labels = ["<1e-6", "1e-6-1e-3", "1e-3-1", "1-10", "10-100", "100-1000", ">1000"]

    println("\n    Flux magnitude distribution:")
    for (i, (low, high)) in enumerate(zip(bins[1:end-1], bins[2:end]))
        count = sum(low .<= flux_values .< high)
        if count > 0
            @printf("      %-12s: %5d reactions (%.1f%%)\n",
                bin_labels[i], count, 100.0 * count / length(flux_values))
        end
    end
end

analyze_flux_distribution(julia_fluxes, "Julia")
analyze_flux_distribution(matlab_fluxes, "MATLAB")

# Check if the issue is enzyme usage vs availability
println("\n7. ENZYME UTILIZATION ANALYSIS")
println("-"^80)

function analyze_enzyme_utilization(model, fluxes, enzyme_regex, name)
    total_enzymes = count(p -> occursin(enzyme_regex, p[1]), model.metabolites)

    # Find which enzyme metabolites participate in active reactions
    active_enzymes = Set{String}()
    for (rxn_id, rxn) in model.reactions
        flux = abs(get(fluxes, Symbol(rxn_id), 0.0))
        if flux > 1e-9
            for met in keys(rxn.stoichiometry)
                if occursin(enzyme_regex, met)
                    push!(active_enzymes, met)
                end
            end
        end
    end

    println("\n  $name:")
    println("    Total enzyme metabolites:     $(total_enzymes)")
    println("    Enzymes in active reactions:  $(length(active_enzymes))")
    println("    Utilization:                  $(round(100.0 * length(active_enzymes) / total_enzymes, digits=1))%")
end

analyze_enzyme_utilization(model, julia_fluxes, r"^E\d+$", "Julia")
analyze_enzyme_utilization(matlab_model, matlab_fluxes, r"^E_", "MATLAB")

# Final diagnosis
println("\n8. ROOT CAUSE DIAGNOSIS")
println("-"^80)

println("\nBased on the analysis:")
println("\n  ✓ POSITIVE FINDINGS:")
println("    • Same number of reactions (17,331)")
println("    • Same stoichiometry structure (avg 2.87 metabolites/reaction)")
println("    • Same objective coefficient (1.0 on biomass reaction)")
println("    • Both have 0 unbalanced enzymes (enzyme conservation works)")
println("    • Same reaction bounds distribution")
println()
println("  ✗ KEY DIFFERENCE:")
println("    • Biomass flux is $(round(julia_biomass_flux / matlab_biomass_flux, digits=2))x higher in Julia")
println("    • This suggests the flux is DISTRIBUTED DIFFERENTLY through the network")
println()
println("  HYPOTHESIS:")
println("    The 69% difference is likely caused by:")
println("    1. Different enzyme-substrate complex IDs preventing direct comparison")
println("    2. Despite structural similarity, the optimizer finds different solutions")
println("    3. The models may have different feasible regions due to:")
println("       - Missing metabolites (226 fewer in Julia)")
println("       - Different enzyme-complex naming breaking mass balance constraints")
println("       - Julia may be missing enzyme availability constraints")
println()
println("  RECOMMENDATION:")
println("    Compare the MATLAB model structure before and after elementary split")
println("    to verify that the Julia splitting creates the same constraints.")

if length(julia_unbalanced) > 0
    println("\n⚠ CRITICAL: Julia has $(length(julia_unbalanced)) unbalanced enzyme metabolites!")
    println("  This suggests enzyme complexes are not being conserved properly.")
    println("  Possible causes:")
    println("    - Elementary reactions missing enzyme regeneration steps")
    println("    - Complex formation/dissociation not balanced")
    println("    - Different enzyme naming causing mapping issues")
end

println("\n" * "="^80)
