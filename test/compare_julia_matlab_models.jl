"""
Compare Julia and MATLAB preprocessed models to identify discrepancies.

This script:
1. Exports Julia model to text files (reactions, metabolites)
2. Loads MATLAB model lists from text files
3. Identifies reactions/metabolites present in one but not the other
4. Analyzes patterns in the differences

Usage:
    # First run the preprocessing pipeline to generate model_irreversible
    julia --project=COCOA.jl test/test_preprocessing_pipeline.jl
    
    # Export MATLAB model to text files (in MATLAB):
    #   fid = fopen('matlab_reactions.txt', 'w');
    #   for i = 1:length(model.rxns)
    #       fprintf(fid, '%s\n', model.rxns{i});
    #   end
    #   fclose(fid);
    #   
    #   fid = fopen('matlab_metabolites.txt', 'w');
    #   for i = 1:length(model.mets)
    #       fprintf(fid, '%s\n', model.mets{i});
    #   end
    #   fclose(fid);
    
    # Then run this comparison
    julia --project=COCOA.jl test/compare_julia_matlab_models.jl
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS

println("="^80)
println("JULIA vs MATLAB MODEL COMPARISON")
println("="^80)
println()

#=============================================================================
STEP 1: Run Julia Preprocessing Pipeline
=============================================================================#
println("STEP 1: Running Julia preprocessing pipeline...")
println()

MODEL_PATH = "C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\iIS312_Epimastigote.xml"

# Load model
model_original = A.load(MODEL_PATH)
model = convert(CM.Model, model_original)
println("✓ Loaded model: $(length(model.reactions)) rxns, $(length(model.metabolites)) mets")

# Normalize bounds
model = COCOA.normalize_bounds(model)
println("✓ Normalized bounds")

# Remove orphans
model = COCOA.remove_orphans(model)
println("✓ Removed orphans: $(length(model.reactions)) rxns, $(length(model.metabolites)) mets")

# Remove blocked reactions
using COBREXA
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("random_seed", 42),
    COBREXA.set_optimizer_attribute("time_limit", 1200.0),
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

println("\nFinding and removing blocked reactions...")
model, blocked = COCOA.remove_blocked_reactions(
    model,
    optimizer=HiGHS.Optimizer,
    flux_tolerance=1e-3,
    objective_bound=COBREXA.relative_tolerance_bound(0.999),
    settings=highs_settings
)
println("✓ Removed $(length(blocked)) blocked reactions: $(length(model.reactions)) rxns, $(length(model.metabolites)) mets")

# Split into elementary reactions
println("\nSplitting into elementary reactions...")
model_elementary = COCOA.split_into_elementary_steps(model, random=0.0)
println("✓ Elementary split: $(length(model_elementary.reactions)) rxns, $(length(model_elementary.metabolites)) mets")

# Convert to irreversible
println("\nConverting to irreversible...")
model_irreversible = COCOA.split_into_irreversible(model_elementary)
println("✓ Irreversible: $(length(model_irreversible.reactions)) rxns, $(length(model_irreversible.metabolites)) mets")

#=============================================================================
STEP 2: Export Julia Model Lists
=============================================================================#
println("\n" * "="^80)
println("STEP 2: Exporting Julia model to text files...")
println("="^80)
println()

# Get sorted lists
julia_rxns = sort(collect(keys(model_irreversible.reactions)))
julia_mets = sort(collect(keys(model_irreversible.metabolites)))

println("Julia model summary:")
println("  Total reactions: $(length(julia_rxns))")
println("  Total metabolites: $(length(julia_mets))")

# Count reaction types
n_cat = count(contains(r, "_s_p_transition") for r in julia_rxns)
n_sub = count(r -> contains(r, "_s") && !contains(r, "_s_p_transition"), julia_rxns)
n_prod = count(r -> contains(r, "_p") && !contains(r, "_s_p_transition"), julia_rxns)
n_forward = count(endswith(r, "_f") for r in julia_rxns)
n_backward = count(endswith(r, "_b") for r in julia_rxns)
n_unexpanded = length(julia_rxns) - n_cat - n_sub - n_prod

println("\nReaction types:")
println("  Catalytic (_s_p_transition): $n_cat")
println("  Substrate binding (_s): $n_sub")
println("  Product release (_p): $n_prod")
println("  Forward splits (_f): $n_forward")
println("  Backward splits (_b): $n_backward")
println("  Unexpanded: $n_unexpanded")

# Count metabolite types
n_enzymes = count(startswith(m, "E") && !contains(m, "_complex") for m in julia_mets)
n_intermediates = count(contains(m, "_complex") for m in julia_mets)
n_original = length(julia_mets) - n_enzymes - n_intermediates

println("\nMetabolite types:")
println("  Original metabolites: $n_original")
println("  Enzyme metabolites (E#): $n_enzymes")
println("  Intermediates (*_complex): $n_intermediates")

# Save to files
julia_rxns_file = "julia_reactions.txt"
julia_mets_file = "julia_metabolites.txt"

open(julia_rxns_file, "w") do f
    for rxn in julia_rxns
        println(f, rxn)
    end
end

open(julia_mets_file, "w") do f
    for met in julia_mets
        println(f, met)
    end
end

println("\n✓ Saved Julia lists:")
println("  $julia_rxns_file")
println("  $julia_mets_file")

#=============================================================================
STEP 3: Load MATLAB Model Lists
=============================================================================#
println("\n" * "="^80)
println("STEP 3: Loading MATLAB model lists...")
println("="^80)
println()

matlab_rxns_file = "test/matlab_reactions.txt"
matlab_mets_file = "test/matlab_metabolites.txt"

if !isfile(matlab_rxns_file) || !isfile(matlab_mets_file)
    println("ERROR: MATLAB model files not found!")
    println("\nPlease export MATLAB model to text files using:")
    println()
    println("=== In MATLAB ===")
    println("fid = fopen('matlab_reactions.txt', 'w');")
    println("for i = 1:length(model.rxns)")
    println("    fprintf(fid, '%s\\n', model.rxns{i});")
    println("end")
    println("fclose(fid);")
    println()
    println("fid = fopen('matlab_metabolites.txt', 'w');")
    println("for i = 1:length(model.mets)")
    println("    fprintf(fid, '%s\\n', model.mets{i});")
    println("end")
    println("fclose(fid);")
    println("================")
    println()
    println("Then run this script again.")
    exit(1)
end

# Read MATLAB lists
matlab_rxns = readlines(matlab_rxns_file)
matlab_mets = readlines(matlab_mets_file)

# Remove any empty lines or whitespace
matlab_rxns = filter(!isempty, strip.(matlab_rxns))
matlab_mets = filter(!isempty, strip.(matlab_mets))

println("MATLAB model summary:")
println("  Total reactions: $(length(matlab_rxns))")
println("  Total metabolites: $(length(matlab_mets))")

# Count MATLAB reaction types
n_cat_m = count(contains(r, "_s_p_transition") for r in matlab_rxns)
n_sub_m = count(r -> contains(r, "_s") && !contains(r, "_s_p_transition"), matlab_rxns)
n_prod_m = count(contains(r, "_p") && !contains(r, "_s_p_transition") for r in matlab_rxns)
n_forward_m = count(endswith(r, "_f") for r in matlab_rxns)
n_backward_m = count(endswith(r, "_b") for r in matlab_rxns)
n_unexpanded_m = length(matlab_rxns) - n_cat_m - n_sub_m - n_prod_m

println("\nReaction types:")
println("  Catalytic (_s_p_transition): $n_cat_m")
println("  Substrate binding (_s): $n_sub_m")
println("  Product release (_p): $n_prod_m")
println("  Forward splits (_f): $n_forward_m")
println("  Backward splits (_b): $n_backward_m")
println("  Unexpanded: $n_unexpanded_m")

# Count MATLAB metabolite types
n_enzymes_m = count(startswith(m, "E") && !contains(m, "_complex") for m in matlab_mets)
n_intermediates_m = count(contains(m, "_complex") for m in matlab_mets)
n_original_m = length(matlab_mets) - n_enzymes_m - n_intermediates_m

println("\nMetabolite types:")
println("  Original metabolites: $n_original_m")
println("  Enzyme metabolites (E): $n_enzymes_m")
println("  Intermediates (_complex): $n_intermediates_m")

#=============================================================================
STEP 4: Compare Models
=============================================================================#
println("\n" * "="^80)
println("STEP 4: Comparing Julia and MATLAB models...")
println("="^80)
println()

# Find differences in reactions
julia_only_rxns = setdiff(Set(julia_rxns), Set(matlab_rxns))
matlab_only_rxns = setdiff(Set(matlab_rxns), Set(julia_rxns))
common_rxns = intersect(Set(julia_rxns), Set(matlab_rxns))

println("=== REACTION COMPARISON ===")
println("  Common reactions: $(length(common_rxns))")
println("  Julia-only reactions: $(length(julia_only_rxns))")
println("  MATLAB-only reactions: $(length(matlab_only_rxns))")
println("  Difference: $(length(julia_rxns) - length(matlab_rxns))")

if length(julia_only_rxns) > 0
    println("\nJulia has $(length(julia_only_rxns)) reactions not in MATLAB:")
    for rxn in sort(collect(julia_only_rxns))[1:min(50, length(julia_only_rxns))]
        println("  $rxn")
    end
    if length(julia_only_rxns) > 50
        println("  ... and $(length(julia_only_rxns) - 50) more")
    end
end

if length(matlab_only_rxns) > 0
    println("\nMATLAB has $(length(matlab_only_rxns)) reactions not in Julia:")
    for rxn in sort(collect(matlab_only_rxns))[1:min(50, length(matlab_only_rxns))]
        println("  $rxn")
    end
    if length(matlab_only_rxns) > 50
        println("  ... and $(length(matlab_only_rxns) - 50) more")
    end
end

# Find differences in metabolites
julia_only_mets = setdiff(Set(julia_mets), Set(matlab_mets))
matlab_only_mets = setdiff(Set(matlab_mets), Set(julia_mets))
common_mets = intersect(Set(julia_mets), Set(matlab_mets))

println("\n=== METABOLITE COMPARISON ===")
println("  Common metabolites: $(length(common_mets))")
println("  Julia-only metabolites: $(length(julia_only_mets))")
println("  MATLAB-only metabolites: $(length(matlab_only_mets))")
println("  Difference: $(length(julia_mets) - length(matlab_mets))")

if length(julia_only_mets) > 0
    println("\nJulia has $(length(julia_only_mets)) metabolites not in MATLAB:")
    for met in sort(collect(julia_only_mets))[1:min(50, length(julia_only_mets))]
        println("  $met")
    end
    if length(julia_only_mets) > 50
        println("  ... and $(length(julia_only_mets) - 50) more")
    end
end

if length(matlab_only_mets) > 0
    println("\nMATLAB has $(length(matlab_only_mets)) metabolites not in Julia:")
    for met in sort(collect(matlab_only_mets))[1:min(50, length(matlab_only_mets))]
        println("  $met")
    end
    if length(matlab_only_mets) > 50
        println("  ... and $(length(matlab_only_mets) - 50) more")
    end
end

#=============================================================================
STEP 5: Pattern Analysis
=============================================================================#
println("\n" * "="^80)
println("STEP 5: Analyzing patterns in differences...")
println("="^80)
println()

println("=== JULIA-ONLY REACTIONS BY TYPE ===")
if length(julia_only_rxns) > 0
    j_cat = count(contains(r, "_s_p_transition") for r in julia_only_rxns)
    j_sub = count(r -> contains(r, "_s") && !contains(r, "_s_p_transition"), julia_only_rxns)
    j_prod = count(r -> contains(r, "_p") && !contains(r, "_s_p_transition"), julia_only_rxns)
    j_fwd = count(endswith(r, "_f") for r in julia_only_rxns)
    j_bwd = count(endswith(r, "_b") for r in julia_only_rxns)
    j_other = length(julia_only_rxns) - j_cat - j_sub - j_prod

    println("  Catalytic (_s_p_transition): $j_cat")
    println("  Substrate binding (_s): $j_sub")
    println("  Product release (_p): $j_prod")
    println("  Forward splits (_f): $j_fwd")
    println("  Backward splits (_b): $j_bwd")
    println("  Other: $j_other")

    # Analyze which base reactions these come from
    println("\n  Base reactions (removing _e*, _s*, _p*, _s_p_transition, _f, _b):")
    base_rxns = Set{String}()
    for rxn in julia_only_rxns
        # Remove enzyme-specific suffixes
        base = replace(rxn, r"_e\d+" => "")
        base = replace(base, r"_s\d+" => "")
        base = replace(base, r"_p\d+" => "")
        base = replace(base, r"_s_p_transition" => "")
        base = replace(base, r"_[fb]$" => "")
        push!(base_rxns, base)
    end
    for base in sort(collect(base_rxns))[1:min(30, length(base_rxns))]
        # Count how many Julia-only reactions come from this base
        count_from_base = count(startswith(r, base) for r in julia_only_rxns)
        println("    $base: $count_from_base reactions")
    end
    if length(base_rxns) > 30
        println("    ... and $(length(base_rxns) - 30) more base reactions")
    end
end

println("\n=== MATLAB-ONLY REACTIONS BY TYPE ===")
if length(matlab_only_rxns) > 0
    m_cat = count(contains(r, "_s_p_transition") for r in matlab_only_rxns)
    m_sub = count(r -> contains(r, "_s") && !contains(r, "_s_p_transition"), matlab_only_rxns)
    m_prod = count(r -> contains(r, "_p") && !contains(r, "_s_p_transition"), matlab_only_rxns)
    m_fwd = count(endswith(r, "_f") for r in matlab_only_rxns)
    m_bwd = count(endswith(r, "_b") for r in matlab_only_rxns)
    m_other = length(matlab_only_rxns) - m_cat - m_sub - m_prod

    println("  Catalytic (_s_p_transition): $m_cat")
    println("  Substrate binding (_s): $m_sub")
    println("  Product release (_p): $m_prod")
    println("  Forward splits (_f): $m_fwd")
    println("  Backward splits (_b): $m_bwd")
    println("  Other: $m_other")

    # Analyze which base reactions these come from
    println("\n  Base reactions (removing _e*, _s*, _p*, _s_p_transition, _f, _b):")
    base_rxns_m = Set{String}()
    for rxn in matlab_only_rxns
        # Remove enzyme-specific suffixes
        base = replace(rxn, r"_e\d+" => "")
        base = replace(base, r"_s\d+" => "")
        base = replace(base, r"_p\d+" => "")
        base = replace(base, r"_s_p_transition" => "")
        base = replace(base, r"_[fb]$" => "")
        push!(base_rxns_m, base)
    end
    for base in sort(collect(base_rxns_m))[1:min(30, length(base_rxns_m))]
        # Count how many MATLAB-only reactions come from this base
        count_from_base = count(startswith(r, base) for r in matlab_only_rxns)
        println("    $base: $count_from_base reactions")
    end
    if length(base_rxns_m) > 30
        println("    ... and $(length(base_rxns_m) - 30) more base reactions")
    end
end

println("\n=== JULIA-ONLY METABOLITES BY TYPE ===")
if length(julia_only_mets) > 0
    j_enz = count(startswith(m, "E") && !contains(m, "_complex") for m in julia_only_mets)
    j_intrm = count(contains(m, "_complex") for m in julia_only_mets)
    j_orig = length(julia_only_mets) - j_enz - j_intrm

    println("  Enzyme metabolites (E#): $j_enz")
    println("  Intermediates (*_complex): $j_intrm")
    println("  Original metabolites: $j_orig")
end

println("\n=== MATLAB-ONLY METABOLITES BY TYPE ===")
if length(matlab_only_mets) > 0
    m_enz = count(startswith(m, "E") && !contains(m, "_complex") for m in matlab_only_mets)
    m_intrm = count(contains(m, "_complex") for m in matlab_only_mets)
    m_orig = length(matlab_only_mets) - m_enz - m_intrm

    println("  Enzyme metabolites (E*): $m_enz")
    println("  Intermediates (*_complex): $m_intrm")
    println("  Original metabolites: $m_orig")
end

#=============================================================================
SUMMARY
=============================================================================#
println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)
println()

println("Reactions:")
println("  Julia:  $(length(julia_rxns))")
println("  MATLAB: $(length(matlab_rxns))")
println("  Difference: $(length(julia_rxns) - length(matlab_rxns)) (Julia has $(length(julia_only_rxns)) extra, MATLAB has $(length(matlab_only_rxns)) extra)")
println()

println("Metabolites:")
println("  Julia:  $(length(julia_mets))")
println("  MATLAB: $(length(matlab_mets))")
println("  Difference: $(length(julia_mets) - length(matlab_mets)) (Julia has $(length(julia_only_mets)) extra, MATLAB has $(length(matlab_only_mets)) extra)")
println()

if length(julia_only_rxns) > length(matlab_only_rxns)
    println("⚠ Julia has $(length(julia_only_rxns) - length(matlab_only_rxns)) more unique reactions than MATLAB")
elseif length(matlab_only_rxns) > length(julia_only_rxns)
    println("⚠ MATLAB has $(length(matlab_only_rxns) - length(julia_only_rxns)) more unique reactions than Julia")
else
    println("✓ Same number of unique reactions in both, but sets differ")
end

println("\n" * "="^80)
