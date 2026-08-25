#!/usr/bin/env julia
# Report post-preprocessing size (complex count = the real cost driver for concordance)
# for a list of models, so small/medium/large tiers can be chosen sensibly.
#
# Usage:
#   MODELS="/path/e_coli_core.xml,/path/iAB_RBC_283.xml,..." \
#   OUTDIR=results \
#   julia --project=examples/reproducibility examples/reproducibility/model_sizing.jl

using Distributed
const NPROCS = parse(Int, get(ENV, "NPROCS", string(min(15, max(1, Sys.CPU_THREADS - 1)))))
if NPROCS > 0 && nprocs() <= NPROCS
    addprocs(NPROCS - (nprocs() - 1); exeflags="--project=$(Base.active_project())")
end
@everywhere begin
    using COCOA
    import COBREXA
    import SBMLFBCModels
    import AbstractFBCModels as A
    import HiGHS
end
import CSV, DataFrames

const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "results"))
mkpath(OUTDIR)
const MODELS = split(get(ENV, "MODELS",
    joinpath(pkgdir(COCOA), "test", "e_coli_core.xml")), ",")

rows = NamedTuple[]
for path in MODELS
    name = splitext(basename(path))[1]
    println("\n=== $name ===")
    try
        canon = convert(A.CanonicalModel.Model, COBREXA.load_model(String(path)))
        n_rxn0 = length(canon.reactions); n_met0 = length(canon.metabolites)
        t0 = time()
        mp = canon |>
            normalize_bounds |>
            m -> remove_blocked_reactions(m; optimizer=HiGHS.Optimizer) |>
            remove_orphans |>
            split_into_elementary |>
            split_into_irreversible
        pt = time() - t0
        n_rxn1 = length(mp.reactions); n_met1 = length(mp.metabolites)
        # complex count via COCOA incidence
        A_mat, cids = COCOA.incidence(mp; return_ids=true)
        n_complexes = length(cids)
        row = (model=name, rxn_raw=n_rxn0, met_raw=n_met0,
               rxn_pp=n_rxn1, met_pp=n_met1, complexes_pp=n_complexes,
               preprocess_s=round(pt, digits=1))
        push!(rows, row)
        println(row)
    catch e
        println("FAILED: ", e)
        push!(rows, (model=name, rxn_raw=-1, met_raw=-1, rxn_pp=-1, met_pp=-1,
                     complexes_pp=-1, preprocess_s=-1.0))
    end
    CSV.write(joinpath(OUTDIR, "model_sizing.csv"), DataFrames.DataFrame(rows))
end
println("\nWrote ", joinpath(OUTDIR, "model_sizing.csv"))
