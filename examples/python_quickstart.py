"""
Minimal Python quick-start for COCOA.jl via JuliaCall.

Setup (one-time)
----------------
1. Install Python bindings:      pip install juliacall
2. The first run installs Julia and the required Julia packages into an
   isolated environment managed by juliacall (see the `Pkg.add` block below).
   This can take several minutes the first time only.

Run
---
    python examples/python_quickstart.py

Notes
-----
- Do NOT call `os._exit()` / `sys.exit()` abruptly while Julia objects are
  still alive: forcing an immediate teardown can trigger a "bus error" as the
  Julia GC finalizers run against an already–torn-down runtime. Let the script
  return normally (as below) so JuliaCall shuts the Julia runtime down cleanly.
"""

from juliacall import Main as jl

# --- Install Julia dependencies into the juliacall environment (idempotent) ---
jl.seval("import Pkg")
for pkg in ("COCOA", "COBREXA", "HiGHS", "SBMLFBCModels", "AbstractFBCModels"):
    jl.seval(f'haskey(Pkg.project().dependencies, "{pkg}") || Pkg.add("{pkg}")')

# --- Load packages ---
jl.seval("import COBREXA")
jl.seval("import SBMLFBCModels")
jl.seval("import AbstractFBCModels as A")
jl.seval("using HiGHS, COCOA")

# --- Load the bundled example model and convert to a canonical model ---
jl.seval('model = COBREXA.load_model(joinpath(pkgdir(COCOA), "test", "e_coli_core.xml"))')
jl.seval("model_canon = convert(A.CanonicalModel.Model, model)")

# --- Preprocess and run concordance analysis (serial; add workers for speed) ---
jl.seval("""
model_processed = model_canon |>
    normalize_bounds |>
    m -> remove_blocked_reactions(m; optimizer=HiGHS.Optimizer) |>
    remove_orphans |>
    split_into_elementary |>
    split_into_irreversible
""")

result = jl.seval("""
activity_concordance_analysis(
    model_processed;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999),
    kinetic_analysis=true,
)
""")

# --- Inspect results from Python ---
print("Concordance modules found:", result.stats["n_concordance_modules"])
print("ACR metabolites:", list(result.acr.metabolite_id))
print("Elapsed time (s):", result.stats["elapsed_time"])

# Returning normally lets JuliaCall tear down the Julia runtime cleanly.
