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

## Teardown note

Let the script return normally. Do **not** call `os._exit()` / `sys.exit()`
while Julia objects are still alive — forcing an abrupt teardown can trigger a
"bus error" as the Julia GC finalizers run against an already-torn-down runtime.
