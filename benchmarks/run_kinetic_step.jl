"""
run_kinetic_step.jl — SLURM-array-compatible kinetic analysis driver.

Reconstructed to match the I/O contract of the original benchmark run (see the
Bioinformatics application note revision, WP2.7-WP2.9): loads a precomputed
concordance JLD2, loads the matching elementary-step .mat model, runs
`kinetic_analysis(...; efficient=true, min_module_size=1)`, and writes a
kinetic_<model>_<variant>_seed42.jld2 with the same keys the original run used
(kinetic_modules, acr_metabolites, acrr_pairs, concordance_modules, model_path),
so it's a drop-in replacement for the pre-fix cluster kinetic step and remains
compatible with compare_results.jl.

Model/variant selection: either pass `MODEL_NAME` and `VARIANT` as the first two
command-line args (single run), or set the environment variable
`SLURM_ARRAY_TASK_ID` (1-based) to index into `model_manifest.txt` (one
"model_name variant" pair per line) for an array job.

Usage (single run):
    julia -t 16 --project=/work/schaffran1/COCOA.jl benchmarks/run_kinetic_step.jl \
        iIS312_Trypomastigote fixed

Usage (SLURM array; see submit_kinetic_array.sbatch):
    SLURM_ARRAY_TASK_ID=1 julia -t 16 --project=/work/schaffran1/COCOA.jl \
        benchmarks/run_kinetic_step.jl

Environment overrides (defaults match the cluster layout used for the original run):
    CONC_DIR   default: /work/schaffran1/application_note/results/matlab_preprocessed/concordance
    MODEL_DIR  default: /work/schaffran1/Upstream_Algorithm/Models/models_with_elementary_steps
    OUT_DIR    default: /work/schaffran1/application_note/results/matlab_preprocessed/kinetic_fixed
               (deliberately a NEW directory, not overwriting the pre-fix results,
               so before/after can still be compared)
"""

using COCOA
using JLD2
import MATFBCModels
import MATFBCModels.MAT
import AbstractFBCModels as A
using Dates

const CONC_DIR = get(ENV, "CONC_DIR",
    "/work/schaffran1/application_note/results/matlab_preprocessed/concordance")
const MODEL_DIR = get(ENV, "MODEL_DIR",
    "/work/schaffran1/Upstream_Algorithm/Models/models_with_elementary_steps")
const OUT_DIR = get(ENV, "OUT_DIR",
    "/work/schaffran1/application_note/results/matlab_preprocessed/kinetic_fixed")

function resolve_model_variant()
    if length(ARGS) >= 2
        return ARGS[1], ARGS[2]
    end
    task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    manifest_path = joinpath(@__DIR__, "model_manifest.txt")
    lines = readlines(manifest_path)
    parts = split(lines[task_id])
    return String(parts[1]), String(parts[2])
end

model_name, variant = resolve_model_variant()
mkpath(OUT_DIR)

println("="^60)
println("COCOA Kinetic Analysis Step (post zero-complex fix)")
println("="^60)
println("Model:      $model_name")
println("Variant:    $variant")
println("Threads:    $(Threads.nthreads())")
println("Timestamp:  $(now())")
println("="^60)

conc_path = joinpath(CONC_DIR, "concordance_$(model_name)_$(variant)_seed42.jld2")
println("\n1. Loading concordance results...")
println("   File: $conc_path")
conc_data = JLD2.load(conc_path)["results"]
conc_modules = COCOA.extract_concordance_modules(conc_data)
println("   Total complexes: $(sum(length, conc_modules))")

model_path = joinpath(MODEL_DIR, "$(model_name)_pre_balanced_$(variant).mat")
println("\n2. Loading model: $model_path")
key = "model_elementary_$(variant)"
f = MAT.matopen(model_path)
d = MAT.read(f, key)
MAT.close(f)
model = MATFBCModels.MATFBCModel(key, d)
println("   Loaded: $(length(A.reactions(model))) reactions, $(length(A.metabolites(model))) metabolites")

println("\n3. Running kinetic_analysis (efficient=true, min_module_size=1, WITH zero-complex fix)...")
t0 = time()
result = COCOA.kinetic_analysis(conc_modules, model; efficient=true, min_module_size=1)
elapsed = time() - t0
println("   Kinetic analysis completed in $(round(elapsed, digits=2)) seconds")

println("\n4. Results Summary:")
println("   Kinetic modules:  $(length(result.kinetic_modules))")
println("   Largest module:   $(isempty(result.kinetic_modules) ? 0 : maximum(length, result.kinetic_modules))")
println("   ACR metabolites:  $(length(result.acr_metabolites))")
println("   ACRR pairs:       $(length(result.acrr_pairs))")

out_path = joinpath(OUT_DIR, "kinetic_$(model_name)_$(variant)_seed42.jld2")
println("\n5. Saving results to: $out_path")
JLD2.save(out_path,
    "kinetic_modules", result.kinetic_modules,
    "acr_metabolites", result.acr_metabolites,
    "acrr_pairs", result.acrr_pairs,
    "concordance_modules", conc_modules,
    "model_path", model_path,
)

println("\n" * "="^60)
println("Done. Exit code: 0")
println("="^60)
