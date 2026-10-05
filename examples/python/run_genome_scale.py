#!/usr/bin/env python3
"""
Run COCOA.jl (concordance + kinetic modules) on one genome-scale model from Python, in parallel.

The embedded Julia (juliacall) starts ordinary Julia worker processes with `addprocs`; the LPs
are solved in those workers, so the run scales with the number of cores. Results are written as
CSV/JSON so nothing is lost if the Python session ends.

One-time setup: examples/python/README.md, "Path B" (build the Julia project with
a native julia, then point juliacall at it with the PYTHON_JULIAPKG_* variables).

Usage:
    python -u examples/python/run_genome_scale.py MODEL.xml OUTDIR [--workers N] [--binding ordered|random]
                           [--flux-tol 1e-6] [--sample-size 5000] [--exhaustive]

Outputs in OUTDIR: complexes.csv (concordance and kinetic module per complex), acr.csv,
acrr.csv, stats.json, and the preprocessed model as model_preprocessed.xml.
"""
import argparse
import csv
import json
import os
import time

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model")
p.add_argument("outdir")
p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
               help="Julia worker processes (default: all cores minus one; share fairly on a shared machine)")
p.add_argument("--binding", choices=["ordered", "random"], default="ordered",
               help="elementary-step mechanism: ordered (paper setting) or random binding")
p.add_argument("--flux-tol", type=float, default=1e-6,
               help="flux below which a reaction counts as blocked (1e-6 for the yeast GEMs; "
                    "use 1e-9 if biomass is built from nested pools)")
p.add_argument("--sample-size", type=int, default=5000, help="samples for the CV pre-filter")
p.add_argument("--cv-threshold", type=float, default=0.01)
p.add_argument("--concordance-tolerance", type=float, default=0.01)
p.add_argument("--lp-time-limit", type=float, default=3600.0, help="seconds per LP")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exhaustive", action="store_true",
               help="exhaustive kinetic/ACR path (kinetic_efficient=false); much slower")
a = p.parse_args()
os.makedirs(a.outdir, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


from juliacall import Main as jl  # noqa: E402  (import after argument parsing: starting Julia takes time)

log(f"starting {a.workers} Julia workers")
jl.seval("using Distributed")
# Workers must load the same Julia project as the master; one thread each (the parallelism is
# across processes, and BLAS threads would oversubscribe the cores).
jl.seval(f'addprocs({a.workers}; exeflags="--project=$(dirname(Base.active_project()))", '
         'env=["JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1"])')
jl.seval("@everywhere using COCOA, HiGHS")
jl.seval("import COBREXA, SBMLFBCModels; import AbstractFBCModels as A")
log(f"workers ready: {jl.seval('nworkers()')}")

jl.seval(f"""
settings = [COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-8),
            COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-8),
            COBREXA.set_optimizer_attribute("random_seed", {a.seed}),
            COBREXA.set_optimizer_attribute("time_limit", {a.lp_time_limit})]
""")

log(f"loading {a.model}")
jl.seval(f'model = convert(A.CanonicalModel.Model, A.load(SBMLFBCModels.SBMLFBCModel, raw"{os.path.abspath(a.model)}"))')
log(f"model: {jl.seval('length(A.reactions(model))')} reactions, {jl.seval('length(A.metabolites(model))')} metabolites")

log("preprocessing: remove blocked reactions (growth kept within 0.1-100 % of optimum)")
jl.seval(f"""
m = model |> remove_orphans |> normalize_bounds
opt_before = COBREXA.flux_balance_analysis(m; optimizer=HiGHS.Optimizer, settings).objective
m = remove_blocked_reactions(m; optimizer=HiGHS.Optimizer, settings, workers=workers(),
        flux_tolerance={a.flux_tol}, objective_bound=o -> COBREXA.C.Between(0.001 * o, o))
m = remove_orphans(m)
opt_after = COBREXA.flux_balance_analysis(m; optimizer=HiGHS.Optimizer, settings).objective
isapprox(opt_after, opt_before; rtol=1e-6) ||
    error("growth changed by blocked-reaction removal ($opt_before -> $opt_after): lower --flux-tol")
""")
log(f"blocked removal kept growth: {jl.seval('opt_after')}; "
    f"{jl.seval('length(A.reactions(m))')} reactions left")

random = 0.0 if a.binding == "ordered" else 1.0
jl.seval(f"m = split_into_irreversible(split_into_elementary(m; random={random}, seed=UInt({a.seed})))")
jl.seval(f'A.save(convert(SBMLFBCModels.SBMLFBCModel, m), raw"{os.path.join(os.path.abspath(a.outdir), "model_preprocessed.xml")}")')
log(f"split into elementary steps ({a.binding} binding): {jl.seval('length(A.reactions(m))')} reactions")

log("concordance + kinetic analysis (progress messages follow)")
result = jl.seval(f"""
activity_concordance_analysis(m; optimizer=HiGHS.Optimizer, settings, workers=workers(),
    objective_bound=nothing,            # full steady-state cone (paper setting)
    sample_size={a.sample_size}, cv_threshold={a.cv_threshold},
    concordance_tolerance={a.concordance_tolerance}, seed=UInt({a.seed}),
    kinetic_analysis=true, kinetic_efficient={'false' if a.exhaustive else 'true'},
    detailed_results=true)
""")


def write_table(name, table):
    cols = list(jl.seval("t -> String.(collect(keys(t)))")(table))
    data = [list(getattr(table, c)) for c in cols]
    with open(os.path.join(a.outdir, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(zip(*data))


write_table("complexes.csv", result.complexes)
write_table("acr.csv", result.acr)
write_table("acrr.csv", result.acrr)
stats = {str(k): (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in result.stats.items()}
stats.update(dict(model=os.path.abspath(a.model), binding=a.binding, flux_tol=a.flux_tol,
                  workers=a.workers, wall_seconds=round(time.time() - t0, 1)))
json.dump(stats, open(os.path.join(a.outdir, "stats.json"), "w"), indent=1, sort_keys=True)

km = [k for k in result.complexes.kinetic_module if k > 0]
largest = max((km.count(k) for k in set(km)), default=0)
log(f"done: {stats.get('n_concordance_modules')} concordance modules, largest kinetic module "
    f"{largest} complexes, {len(list(result.acr.metabolite_id))} ACR, "
    f"{len(list(result.acrr.metabolite_1))} ACRR; undecided pairs {stats.get('n_unknown_pairs', 'n/a')}")
log(f"results in {a.outdir}")
# Return normally: an abrupt exit while Julia objects are alive can crash the teardown.
