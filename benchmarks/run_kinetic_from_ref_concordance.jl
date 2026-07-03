"""
run_kinetic_from_ref_concordance.jl — RUN B (diagnostic).

Runs COCOA's kinetic_analysis on the REFERENCE's concordance partition instead of COCOA's
own, to localise the COCOA-vs-reference disagreement:

  • If Run B's kinetic giant module ≈ the reference's (Table S1 col 10), the kinetic/upstream
    step AGREES and the whole divergence lives in the concordance step.
  • If Run B still diverges from the reference, the kinetic step itself differs.

Reference concordance: <REF_DIR>/<model>_concordant_<variant>.mat, with
  - Results_balanced.MODEL_r.{complexes,mets} : complex names + metabolite list. Reference
    complex names are index-encoded, e.g. "2*5+3*7" = 2·mets[5] + 3·mets[7]; "1*0" = ∅.
  - class_with_balanced : cell array of concordance classes (vectors of complex indices);
    the largest class is the balanced module. Singleton complexes are not stored.

We map each reference complex -> COCOA complex id via COCOA.generate_complex_id on its
(met, coef) composition, drop ∅, and assemble Vector{Set{Symbol}} with balanced first.

VALIDATION (cluster agent, do this first on iIS312 before the full array):
  - iIS312_Trypomastigote fixed : reference kinetic giant = 67 (Table S1). Run B should land
    near 67 if the kinetic step agrees (COCOA-own-concordance Run A gives 11).
  - iIS312_Amastigote     random: reference kinetic giant = 1. Run B should land near 1.
  - The script prints `mapped/total` complex-id match rate against COCOA's own complex set;
    if that is not ~1.0 the name↔met mapping is off and must be fixed before trusting results.

Usage / env: same as run_kinetic_step.jl, plus REF_DIR for the reference concordant mats.
"""

using COCOA
using JLD2
import MATFBCModels
import MATFBCModels.MAT
import AbstractFBCModels as A
using Dates

const REF_DIR = get(ENV, "REF_DIR", "/work/schaffran1/kinetic_rerun/ref_concordant")
const MODEL_DIR = get(ENV, "MODEL_DIR", "/work/schaffran1/kinetic_rerun/models")
const OUT_DIR = get(ENV, "OUT_DIR", "/work/schaffran1/kinetic_rerun/out_B")

function resolve_model_variant()
    length(ARGS) >= 2 && return ARGS[1], ARGS[2]
    task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    lines = readlines(joinpath(@__DIR__, "model_manifest.txt"))
    parts = split(lines[task_id])
    return String(parts[1]), String(parts[2])
end

# Drill through MATLAB struct-array nesting until we reach the Dict holding "complexes".
function unwrap_model_r(x)
    x isa AbstractDict && haskey(x, "complexes") && return x
    if x isa AbstractArray
        for el in x
            r = unwrap_model_r(el)
            r !== nothing && return r
        end
    end
    return nothing
end

# Parse a reference complex name ("2*5+3*7", "1*0", "1*12") into a COCOA complex id.
# Returns COCOA.ZERO_COMPLEX-like nothing for ∅ (met index 0).
function ref_name_to_cocoa_id(name::AbstractString, mets)
    comp = Tuple{Symbol,Float64}[]
    for term in split(name, '+')
        cs = split(term, '*')
        length(cs) == 2 || continue
        coef = parse(Float64, cs[1]); midx = parse(Int, cs[2])
        midx == 0 && return nothing            # ∅ zero complex
        push!(comp, (Symbol(String(mets[midx])), coef))
    end
    sort!(comp, by = x -> x[1])
    return COCOA.generate_complex_id(comp)
end

model_name, variant = resolve_model_variant()
mkpath(OUT_DIR)
println("="^60); println("RUN B: COCOA kinetic on REFERENCE concordance"); println("="^60)
println("Model: $model_name  Variant: $variant  Threads: $(Threads.nthreads())  $(now())")

# --- reference concordance ------------------------------------------------------------
ref_path = joinpath(REF_DIR, "$(model_name)_concordant_$(variant).mat")
println("\n1. Reference concordance: $ref_path")
rf = MAT.matopen(ref_path); rd = MAT.read(rf); MAT.close(rf)
mr = unwrap_model_r(rd["Results_balanced"]["MODEL_r"])
mr === nothing && error("Could not locate MODEL_r struct with 'complexes'")
ref_complexes = vec(mr["complexes"]); ref_mets = vec(mr["mets"])
cwb = rd["class_with_balanced"]
class_sizes = [length(cwb[i]) for i in eachindex(cwb)]
bal_idx = argmax(class_sizes)   # balanced module = largest class
println("   reference: $(length(ref_complexes)) complexes, $(length(cwb)) classes, balanced size=$(class_sizes[bal_idx])")

# --- COCOA model + its complex-id universe --------------------------------------------
model_path = joinpath(MODEL_DIR, "$(model_name)_pre_balanced_$(variant).mat")
key = "model_elementary_$(variant)"
mf = MAT.matopen(model_path); md = MAT.read(mf, key); MAT.close(mf)
model = MATFBCModels.MATFBCModel(key, md)
_, cocoa_ids = COCOA.incidence(model; return_ids=true, include_zero_complex=true)
cocoa_set = Set(cocoa_ids)

# --- map reference classes -> COCOA concordance modules -------------------------------
mapped_ok = Ref(0); mapped_tot = Ref(0)
function tocc(class)
    s = Set{Symbol}()
    for i in vec(class)
        id = ref_name_to_cocoa_id(String(ref_complexes[Int(i)]), ref_mets)
        id === nothing && continue          # ∅ excluded from concordance
        mapped_tot[] += 1
        (id in cocoa_set) && (mapped_ok[] += 1)
        push!(s, id)
    end
    s
end
balanced = tocc(cwb[bal_idx])
others = [tocc(cwb[i]) for i in eachindex(cwb) if i != bal_idx]
# singleton complexes not covered by any reference class
covered = union(balanced, others...)
singletons = [Set([c]) for c in cocoa_ids if c != COCOA.ZERO_COMPLEX && !(c in covered)]
conc_modules = vcat([balanced], others, singletons)

match_rate = mapped_tot[] == 0 ? 0.0 : mapped_ok[] / mapped_tot[]
println("\n2. Mapping: $(mapped_ok[])/$(mapped_tot[]) reference complexes matched a COCOA id (rate=$(round(match_rate, digits=4)))")
match_rate < 0.98 && @warn "Low match rate — reference name↔met mapping likely off; results suspect."

# --- kinetic analysis on the reference partition --------------------------------------
println("\n3. kinetic_analysis on reference concordance (efficient=true)...")
t0 = time()
result = COCOA.kinetic_analysis(conc_modules, model; efficient=true, min_module_size=1)
println("   done in $(round(time()-t0, digits=2)) s")
giant = isempty(result.kinetic_modules) ? 0 : maximum(length, result.kinetic_modules)
println("   giant kinetic module: $giant   ACR: $(length(result.acr_metabolites))   ACRR: $(length(result.acrr_pairs))")

out_path = joinpath(OUT_DIR, "kineticB_$(model_name)_$(variant)_seed42.jld2")
JLD2.save(out_path,
    "kinetic_modules", result.kinetic_modules,
    "acr_metabolites", result.acr_metabolites,
    "acrr_pairs", result.acrr_pairs,
    "concordance_modules", conc_modules,
    "match_rate", match_rate,
    "source", "reference_concordance",
    "model_path", model_path)
println("\n4. Saved: $out_path")
println("="^60); println("Done."); println("="^60)
