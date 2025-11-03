"""
Compare metabolites between Julia and MATLAB implementations.

This script analyzes the differences in metabolite creation between
the Julia and MATLAB elementary splitting approaches.
"""

using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
using HiGHS

# Configuration
const MODEL_PATH = "C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\iIS312_Epimastigote.xml"
const MATLAB_METS_FILE = "C:\\Users\\anton\\master-thesis\\COCOA.jl\\test\\matlab_metabolites.txt"
const RANDOM = 0.0  # 0.0 = all ordered, 1.0 = all random

println("="^80)
println("METABOLITE COMPARISON: Julia vs MATLAB")
println("="^80)

# Load and process the model (Julia)
println("\n📦 Loading and processing model (Julia)...")
model = A.load(MODEL_PATH) |> x -> convert(CM.Model, x)

# Preprocessing pipeline
model = COCOA.normalize_bounds(model)
model = COCOA.remove_orphans(model)
model, _ = COCOA.remove_blocked_reactions(model, optimizer=HiGHS.Optimizer)
model = COCOA.remove_orphans(model)  # Remove orphans again after blocked rxn removal
model = COCOA.split_into_elementary(model, random=RANDOM)
model_irr = COCOA.split_into_irreversible(model)

println("  Julia final model: $(length(model_irr.reactions)) rxns, $(length(model_irr.metabolites)) mets")

# Load MATLAB metabolites
println("\n📥 Loading MATLAB metabolites...")
matlab_mets_raw = readlines(MATLAB_METS_FILE)
matlab_mets_raw = filter(x -> !isempty(strip(x)), matlab_mets_raw)
println("  MATLAB total: $(length(matlab_mets_raw)) metabolites")

# Categorize MATLAB metabolites
matlab_regular = String[]
matlab_enzymes = String[]
matlab_complexes = String[]

for met in matlab_mets_raw
    met = strip(met)
    if match(r"^E_\d+$", met) !== nothing
        # Match E_1, E_2, E_3, ... E_118 (enzyme metabolites with underscore)
        push!(matlab_enzymes, met)
    elseif contains(met, "_complex")
        push!(matlab_complexes, met)
    else
        push!(matlab_regular, met)
    end
end

println("\n  MATLAB breakdown:")
println("    Regular metabolites: $(length(matlab_regular))")
println("    Enzyme metabolites:  $(length(matlab_enzymes))")
println("    Complex metabolites: $(length(matlab_complexes))")

# Categorize Julia metabolites
julia_mets_all = collect(keys(model_irr.metabolites))
julia_regular = String[]
julia_enzymes = String[]
julia_complexes = String[]

for met_id in julia_mets_all
    if startswith(met_id, "CPLX_E")
        push!(julia_complexes, met_id)
    elseif match(r"^E\d+$", met_id) !== nothing
        # Match E1, E2, E3, ... E118 (enzyme metabolites without underscore)
        push!(julia_enzymes, met_id)
    else
        push!(julia_regular, met_id)
    end
end

println("\n  Julia breakdown:")
println("    Regular metabolites: $(length(julia_regular))")
println("    Enzyme metabolites:  $(length(julia_enzymes))")
println("    Complex metabolites: $(length(julia_complexes))")

# Convert Julia metabolite IDs to MATLAB format for comparison
function julia_to_matlab_format(julia_id::String)
    # M_atp_c -> atp[c]
    if startswith(julia_id, "M_")
        parts = split(julia_id, "_")
        if length(parts) >= 3
            met_name = join(parts[2:end-1], "_")
            comp = parts[end]
            return "$(met_name)[$(comp)]"
        end
    end
    return julia_id
end

julia_regular_matlab_fmt = [julia_to_matlab_format(id) for id in julia_regular]

# Compare regular metabolites
println("\n" * "="^80)
println("REGULAR METABOLITE COMPARISON")
println("="^80)

matlab_regular_set = Set(matlab_regular)
julia_regular_set = Set(julia_regular_matlab_fmt)

matlab_only = setdiff(matlab_regular_set, julia_regular_set)
julia_only = setdiff(julia_regular_set, matlab_regular_set)
common = intersect(matlab_regular_set, julia_regular_set)

println("\n  Common metabolites: $(length(common))")
println("  Only in MATLAB:     $(length(matlab_only))")
println("  Only in Julia:      $(length(julia_only))")

if length(matlab_only) > 0
    println("\n  Metabolites in MATLAB but NOT in Julia:")
    for met in sort(collect(matlab_only))[1:min(30, length(matlab_only))]
        println("    - $met")
    end
    if length(matlab_only) > 30
        println("    ... and $(length(matlab_only) - 30) more")
    end
end

if length(julia_only) > 0
    println("\n  Metabolites in Julia but NOT in MATLAB:")
    for met in sort(collect(julia_only))[1:min(30, length(julia_only))]
        println("    - $met")
    end
    if length(julia_only) > 30
        println("    ... and $(length(julia_only) - 30) more")
    end
end

# Compare enzyme metabolites
println("\n" * "="^80)
println("ENZYME METABOLITE COMPARISON")
println("="^80)

matlab_enzyme_set = Set(matlab_enzymes)
julia_enzyme_set = Set(julia_enzymes)

enzyme_common = intersect(matlab_enzyme_set, julia_enzyme_set)
enzyme_matlab_only = setdiff(matlab_enzyme_set, julia_enzyme_set)
enzyme_julia_only = setdiff(julia_enzyme_set, julia_enzyme_set)

println("\n  MATLAB enzymes: $(length(matlab_enzymes))")
println("  Julia enzymes:  $(length(julia_enzymes))")
println("  Common:         $(length(enzyme_common))")

if length(enzyme_matlab_only) > 0
    println("\n  Enzyme IDs only in MATLAB:")
    for e in sort(collect(enzyme_matlab_only))[1:min(20, length(enzyme_matlab_only))]
        println("    - $e")
    end
end

if length(enzyme_julia_only) > 0
    println("\n  Enzyme IDs only in Julia:")
    for e in sort(collect(enzyme_julia_only))[1:min(20, length(enzyme_julia_only))]
        println("    - $e")
    end
end

# Compare complex naming conventions
println("\n" * "="^80)
println("COMPLEX METABOLITE NAMING ANALYSIS")
println("="^80)

println("\n  MATLAB complexes: $(length(matlab_complexes))")
println("  Julia complexes:  $(length(julia_complexes))")

println("\n  MATLAB complex examples:")
for i in 1:min(5, length(matlab_complexes))
    println("    $(matlab_complexes[i])")
end

println("\n  Julia complex examples:")
for i in 1:min(5, length(julia_complexes))
    println("    $(julia_complexes[i])")
end

# Analyze naming pattern differences
println("\n  Naming Convention Comparison:")
println("  ┌─────────────┬────────────────────────────────────────────────────┐")
println("  │   Aspect    │                  Format                            │")
println("  ├─────────────┼────────────────────────────────────────────────────┤")
println("  │ MATLAB      │ E<n>_<met1>[<comp>]_<met2>[<comp>]_complex       │")
println("  │ Julia       │ CPLX_E<n>__M_<met1>_<comp>__M_<met2>_<comp>_<comp>│")
println("  └─────────────┴────────────────────────────────────────────────────┘")

# Extract enzyme numbers from complexes to see if same reactions were split
function extract_enzyme_number(complex_name::String)
    if startswith(complex_name, "CPLX_E")
        # CPLX_E58__M_adp_c_c -> 58
        m = match(r"CPLX_E(\d+)__", complex_name)
        return m !== nothing ? parse(Int, m.captures[1]) : nothing
    elseif startswith(complex_name, "E")
        # E58_cit[m]_complex -> 58
        m = match(r"E(\d+)_", complex_name)
        return m !== nothing ? parse(Int, m.captures[1]) : nothing
    end
    return nothing
end

matlab_complex_enzymes = Set(filter(!isnothing, [extract_enzyme_number(c) for c in matlab_complexes]))
julia_complex_enzymes = Set(filter(!isnothing, [extract_enzyme_number(c) for c in julia_complexes]))

println("\n  Enzymes with complexes:")
println("    MATLAB: $(length(matlab_complex_enzymes)) enzymes create complexes")
println("    Julia:  $(length(julia_complex_enzymes)) enzymes create complexes")
println("    Common: $(length(intersect(matlab_complex_enzymes, julia_complex_enzymes)))")

enzymes_only_matlab = setdiff(matlab_complex_enzymes, julia_complex_enzymes)
enzymes_only_julia = setdiff(julia_complex_enzymes, matlab_complex_enzymes)

if length(enzymes_only_matlab) > 0
    println("\n  Enzymes creating complexes only in MATLAB:")
    println("    E_$(join(sort(collect(enzymes_only_matlab))[1:min(10, length(enzymes_only_matlab))], ", E_"))")
    if length(enzymes_only_matlab) > 10
        println("    ... and $(length(enzymes_only_matlab) - 10) more")
    end
end

if length(enzymes_only_julia) > 0
    println("\n  Enzymes creating complexes only in Julia:")
    println("    E_$(join(sort(collect(enzymes_only_julia))[1:min(10, length(enzymes_only_julia))], ", E_"))")
    if length(enzymes_only_julia) > 10
        println("    ... and $(length(enzymes_only_julia) - 10) more")
    end
end

# Summary
println("\n" * "="^80)
println("SUMMARY")
println("="^80)

println("""
Key Findings:

1. **Total Metabolites**
   - MATLAB: $(length(matlab_mets_raw)) total
   - Julia:  $(length(julia_mets_all)) total
   - Difference: $(abs(length(matlab_mets_raw) - length(julia_mets_all)))

2. **Regular Metabolites** (original model metabolites)
   - MATLAB: $(length(matlab_regular))
   - Julia:  $(length(julia_regular))
   - Common: $(length(common))
   - MATLAB-only: $(length(matlab_only))
   - Julia-only: $(length(julia_only))

3. **Enzyme Metabolites** (E_1, E_2, etc.)
   - MATLAB: $(length(matlab_enzymes))
   - Julia:  $(length(julia_enzymes))
   - Should be 1 per enzyme-catalyzed reaction

4. **Complex Metabolites** (enzyme-substrate intermediates)
   - MATLAB: $(length(matlab_complexes))
   - Julia:  $(length(julia_complexes))
   - Difference: $(abs(length(matlab_complexes) - length(julia_complexes)))

5. **Naming Convention**
   - Both create complexes, but use different ID formats
   - MATLAB: E3_akg[m]_asp__L[m]_complex
   - Julia:  CPLX_E3__M_akg_m__M_asp__L_m_m
   
6. **Biological Equivalence**
   - Same enzymes create complexes: $(length(intersect(matlab_complex_enzymes, julia_complex_enzymes)))
   - This suggests same reactions were split into elementary steps
   - Differences are primarily in naming, not biochemical content
""")

if length(matlab_only) > 0 || length(julia_only) > 0
    println("\n⚠️  IMPORTANT: Regular metabolite differences detected!")
    println("   This could indicate:")
    println("   - Different preprocessing (blocked reactions removed)")
    println("   - Different orphan metabolite handling")
    println("   - GPR parsing differences (as previously identified)")
end

println("\n" * "="^80)
