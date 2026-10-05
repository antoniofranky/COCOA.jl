"""
COCOA.jl from Python — quick-start (OFFLINE / pre-built project).

This script performs NO Julia package management: it assumes the Julia project
was already instantiated by a native julia via ``build_env.jl`` and that
juliacall is pointed at it read-only (see ``run_offline.sh``). It therefore only
does ``using`` + the analysis, which is robust even on HPC systems where running
``Pkg`` operations from inside juliacall crashes.

Run it through ``run_offline.sh`` (which sets the required environment
variables), or set them yourself:

    PYTHON_JULIAPKG_PROJECT=/path/to/cocoa_pyenv \
    PYTHON_JULIAPKG_OFFLINE=yes \
    PYTHON_JULIAPKG_EXE=$(which julia) \
    JULIA_CONDAPKG_BACKEND=Null \
    python examples/python/quickstart.py

Requirements
------------
- Python >= 3.10 (PythonCall.jl does not support 3.9 or older).
- ``pip install juliacall``.

Note
----
Let the script return normally; do not call ``os._exit`` / ``sys.exit`` while
Julia objects are alive, or the Julia GC finalizers may run against an
already-torn-down runtime and raise a "bus error" at teardown.
"""
import sys


def step(msg):
    print(f"STEP: {msg}", flush=True)


step("importing juliacall (offline, pre-built project)")
from juliacall import Main as jl

# --- Load packages (no Pkg operations) ---
step("using packages")
jl.seval("import COBREXA")
jl.seval("import SBMLFBCModels")
jl.seval("import AbstractFBCModels as A")
jl.seval("using HiGHS, COCOA")

# --- Load the bundled example model and convert to a canonical model ---
step("load model")
jl.seval(
    'model = A.load(SBMLFBCModels.SBMLFBCModel, '
    'joinpath(pkgdir(COCOA), "test", "e_coli_core.xml"))'
)
jl.seval("model_canon = convert(A.CanonicalModel.Model, model)")

# --- Preprocess and run concordance analysis ---
# Serial, which is fine for this small model. For genome-scale models use
# run_genome_scale.py, which runs in parallel (see README.md).
step("preprocess")
jl.seval("""
model_processed = model_canon |>
    normalize_bounds |>
    m -> remove_blocked_reactions(m; optimizer=HiGHS.Optimizer) |>
    remove_orphans |>
    split_into_elementary |>
    split_into_irreversible
""")

step("activity_concordance_analysis")
result = jl.seval("""
activity_concordance_analysis(
    model_processed;
    optimizer=HiGHS.Optimizer,
    objective_bound=nothing,     # full flux cone -> deterministic
    kinetic_analysis=true,
)
""")

# --- Inspect results from Python ---
step("read results in Python")
print("Concordance modules:", result.stats["n_concordance_modules"], flush=True)
print("ACR metabolites   :", len(list(result.acr.metabolite_id)), flush=True)
print("ACRR pairs        :", len(list(result.acrr.metabolite_1)), flush=True)

step("done (returning normally for clean Julia teardown)")
print("PYCHECK RESULT: SUCCESS", flush=True)
