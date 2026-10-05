# Calling COCOA.jl from Python

COCOA.jl runs from Python through
[`juliacall`](https://juliapy.github.io/PythonCall.jl/). There are two supported
paths.

## Requirements (both paths)

- **Python ≥ 3.10** — PythonCall.jl does not support Python 3.9 or older.
  On HPC, load a recent Python first (e.g. `module load python/3.11`).
- `pip install juliacall`

## Path A — simple (standard workstation)

Use juliacall's auto-managed environment; it installs Julia and the required
packages on first run. See [`quickstart_simple.py`](quickstart_simple.py).

```bash
pip install juliacall
python examples/python/quickstart_simple.py
```

This does `Pkg.add(...)` from inside juliacall on first run. It is the easiest
path and works on a normal machine.

## Path B — offline / pre-built project (HPC-robust, recommended for clusters)

On some HPC systems, running Julia package management (`Pkg.develop` / `Pkg.add`
+ precompile) from *inside* the embedded libjulia runtime segfaults. The fix is
to build the project **once with a native julia**, then point juliacall at it
read-only so it only does `using` — no `Pkg` operations at runtime.

```bash
# 1. Build the project with a NATIVE julia (one time).
#    COCOA_DEV_PATH develops your local checkout; omit it to use the
#    registered COCOA package.
#    PythonCall.jl is pinned to the version of your pip package juliacall
#    (they must match exactly); set COCOA_PYTHON if `python3` is not the
#    Python you will run the scripts with.
COCOA_PYENV=$PWD/cocoa_pyenv COCOA_DEV_PATH=$PWD \
    julia examples/python/build_env.jl

# 2. Run the quick-start against that project, offline.
module load python/3.11          # if needed; must be >= 3.10
COCOA_PYENV=$PWD/cocoa_pyenv JULIA_BIN=$(which julia) \
    bash examples/python/run_offline.sh
```

`run_offline.sh` sets the environment variables that make juliacall use the
pre-built project without touching `Pkg`:

| Variable | Purpose |
|---|---|
| `PYTHON_JULIAPKG_PROJECT` | Path to the pre-built project |
| `PYTHON_JULIAPKG_OFFLINE=yes` | Forbid runtime Pkg resolution/install |
| `PYTHON_JULIAPKG_EXE` | The native julia binary to use |
| `JULIA_CONDAPKG_BACKEND=Null` | Bind to the current Python; don't fetch one |

Expected output for the bundled `e_coli_core` model (full flux cone,
deterministic):

```
Concordance modules: 109
ACR metabolites   : 0
ACRR pairs        : 1
PYCHECK RESULT: SUCCESS
```

These match the native-Julia COCOA results exactly, confirming the Python
bindings are faithful.

## Genome-scale models (parallel)

The two quick-starts run **serially**, which is fine for `e_coli_core` (about a minute) but
not for a genome-scale model: after splitting into elementary steps a yeast GEM has
10,000–35,000 complexes and well over 100,000 candidate pairs, which takes weeks on one core.
Use [`run_genome_scale.py`](run_genome_scale.py) instead. It starts Julia worker processes
from Python (`addprocs` in the embedded Julia; the LPs run in the workers), sets a time limit
per LP, logs progress and writes all results to files:

```bash
# after the one-time Path B build above, with its environment variables set:
python -u examples/python/run_genome_scale.py model.xml outdir --workers 31 > run.log 2>&1
```

- **`python -u`**, or Python holds back all output until the end, and a running job looks
  exactly like a stuck one.
- **Workers**: one per core, minus one for the main process. On a shared interactive machine
  run inside `tmux`/`screen` so the job survives logging out; under SLURM request
  `--cpus-per-task=<workers + 1>`.
- **Resources** (measured, ordered binding, 64 workers): yeast GEMs 4 h – 1.5 days and
  100–250 GB memory (roughly 2–4 GB per worker). Random binding roughly doubles the network
  and takes about 3.5× as long.
- **Outputs** in `outdir`: `complexes.csv` (concordance and kinetic module per complex),
  `acr.csv`, `acrr.csv`, `stats.json` (all counts and the settings used) and
  `model_preprocessed.xml`.
- **Options**: `--binding ordered|random`, `--flux-tol` (blocked-reaction threshold; 1e-6 suits
  the yeast GEMs; models whose biomass is built from nested pools can need 1e-9, and the script
  stops if blocked-reaction removal changes the growth rate), `--sample-size` (default 5000),
  `--cv-threshold`, `--concordance-tolerance`, `--lp-time-limit`, `--exhaustive`
  (`kinetic_efficient=false`). `--help` lists them all.

On `e_coli_core` (`--workers 7`) the script reproduces the serial result: 109 concordance
modules, 0 ACR, 1 ACRR.

## Teardown note

Let the script return normally. Do **not** call `os._exit()` / `sys.exit()`
while Julia objects are still alive — forcing an abrupt teardown can trigger a
"bus error" as the Julia GC finalizers run against an already-torn-down runtime.
