# cocoa-jl — concordance and kinetic modules from Python

`cocoa-jl` is the Python interface to [COCOA.jl](https://github.com/antoniofranky/COCOA.jl).
It takes a genome-scale metabolic model (SBML, COBRApy JSON or COBRA `.mat`) and finds its
**concordance modules** and **kinetic modules**: groups of complexes whose activities are
proportional in every steady state, and the parts of the network whose concentrations and
fluxes are coupled under mass-action kinetics. Metabolites with absolute concentration
robustness (ACR) and metabolite pairs with robust concentration ratios (ACRR) come out of
the same algorithm.

The computation runs in Julia. This package sets up the Julia side once, starts every
analysis as a separate Julia process and returns the results as CSV files and Python
tables. You use it from the command line (`cocoa-jl`) or from Python (`import cocoajl`).
It has no Python dependencies.

- [Installation](#installation)
- [Quick start](#quick-start)
- [What the pipeline does](#what-the-pipeline-does)
- [Output files](#output-files)
- [Parameters](#parameters)
- [Parameter sweeps](#parameter-sweeps)
- [Running on a cluster](#running-on-a-cluster)
- [Models that are already split into elementary steps](#models-that-are-already-split-into-elementary-steps)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

You need **Python ≥ 3.9** and **Julia ≥ 1.12**.

1. Install the package (a virtual environment is recommended):

   ```bash
   python -m venv cocoa-env && source cocoa-env/bin/activate
   pip install "git+https://github.com/antoniofranky/COCOA.jl#subdirectory=python"
   ```

2. Install Julia if you do not have it: `curl -fsSL https://install.julialang.org | sh`
   (installs [juliaup](https://github.com/JuliaLang/juliaup)), or `module load julia` on a
   cluster. Check with `julia --version`.

3. Build the Julia side, once:

   ```bash
   cocoa-jl setup
   ```

   This downloads COCOA.jl and its dependencies into a Julia project
   (`~/.local/share/cocoajl/julia_env`) and precompiles them: 5–20 minutes, once. The
   project is recorded in `~/.config/cocoajl/config.json`, so every later run starts without
   any installation step and **needs no network access** — compute nodes without internet
   are fine. On a cluster, put the project on a file system the compute nodes can read:
   `cocoa-jl setup --project /work/$USER/cocoajl_env`.

   Options: `--julia PATH` (a specific julia binary), `--cocoa-rev main` (COCOA.jl from
   GitHub instead of the registry, e.g. for a fix that is not released yet),
   `--cocoa-path DIR` (a local COCOA.jl checkout), `--cocoa-version V`.

4. Check: `cocoa-jl info` prints the project, the Julia and COCOA.jl versions in use.


## Quick start

Try it on the small *E. coli* core model shipped with COCOA.jl (about a minute):

```bash
cocoa-jl run $(python -c "import cocoajl; print(cocoajl.example_model())") out_ecoli --flux-tol 1e-9
```

A genome-scale model, on 31 worker processes:

```bash
python -u -m cocoajl run iJR904.xml out_iJR904 --workers 31 > run.log 2>&1
```

`python -u` (or `PYTHONUNBUFFERED=1`) makes the progress messages appear as they happen;
without it a running job looks exactly like a stuck one.

From Python:

```python
import cocoajl as cj

res = cj.run("iJR904.xml", "out_iJR904", workers=31)
res.modules.to_pandas()          # one row per kinetic module, largest first
res.complexes.to_pandas()        # concordance and kinetic module of every complex
```

Results of a finished run can be read again at any time with `cj.load_result("out_iJR904")`.
Each run also leaves its job file (`job.json`); if a run fails, the error message prints
the `julia` command that reruns exactly that job, which is the quickest way to debug.

## What the pipeline does

1. **Preprocessing** (also on its own: `cocoa-jl preprocess`, `cj.preprocess`)
   1. remove orphan metabolites; replace the bounds by the flux cone: a negative lower bound
      becomes −1000, a positive upper bound +1000, all others 0 (the analysis is about which
      steady states exist, not about measured uptake rates);
   2. remove **blocked reactions**, i.e. reactions whose flux range stays below `flux_tol`
      while growth is held between 0.1 % and 100 % of its optimum. The run stops if an LP
      cannot be solved, if a blocked reaction is left afterwards, or if the growth optimum
      changed — a silently kept blocked reaction can glue unrelated modules together;
   3. remove orphans again;
   4. split every enzymatic reaction into **elementary steps** (enzyme binding, catalysis,
      release) with an **ordered** or **random** binding mechanism, and every reversible
      reaction into two irreversible ones. Kinetic modules assume mass-action kinetics, which
      holds for elementary steps, not for lumped enzymatic reactions.
2. **Concordance analysis**: samples the steady-state flux cone, keeps
   the complex pairs whose sampled activity ratio is nearly constant, and decides each by a
   pair of LPs. This is the expensive step.
3. **Kinetic modules**: the upstream algorithm of Langary et al.
   (2025) on the concordance modules. Fast (seconds to minutes).

## Output files

`cocoa-jl run` writes to the output directory:

| File | Content |
|---|---|
| `kinetic_modules.csv` | one row per kinetic module: `kinetic_module` (1 = largest), `n_complexes`, `complexes` (`;`-separated) |
| `complexes.csv` | one row per complex: `complex_id`, `concordance_module` (0 = balanced, −1 = in no module), `kinetic_module` (0 = none), `classification`, `min_activity`, `max_activity`, `lambda`, `trivially_balanced` |
| `acr.csv`, `acrr.csv` | ACR metabolites and ACRR metabolite pairs (skip with `--no-acr`; they cost nothing extra) |
| `model_preprocessed.xml` | the model that was analysed (SBML) |
| `concordance.jld2` | the concordance result with the exact model, for `cocoa-jl kinetic` |
| `run.json` | all parameters, the preprocessing report, versions (Julia, COCOA.jl and its git commit), all counts and timings |
| `job.json` | the job as executed (rerun by hand: see [Troubleshooting](#troubleshooting)) |

Complex ids are the metabolites of the complex joined by `+`, with the stoichiometric
coefficient as prefix when it is not 1 (e.g. `2.0_M_atp_c+M_glc__D_e`).

Check `n_unknown_pairs` in `run.json`: the number of complex pairs whose LPs failed and
which are therefore neither concordant nor non-concordant. It should be a small fraction
of `n_candidate_pairs`.

## Parameters

All parameters have defaults that suit genome-scale models; you only set what you want to
change. On the command line they are flags (`--concordance-tolerance 0.001`), in Python
keyword arguments (`concordance_tolerance=0.001`). `cocoa-jl run --help` lists them.

**Preprocessing**

| Parameter | Default | Meaning |
|---|---|---|
| `binding` | `ordered` | elementary-step mechanism: `ordered`, `random`, or `none` (no splitting; kinetic modules then have no theoretical backing) |
| `random_fraction` | 1.0 | with `binding="random"`: fraction of the enzymes split randomly, the rest ordered |
| `flux_tol` | 1e-6 | flux below which a reaction counts as blocked. If the run stops because removal changed growth (biomass built from nested pools, e.g. some BiGG models), use 1e-9 |
| `growth_window` | (0.001, 1.0) | growth range, as fraction of the optimum, while blocked reactions are found (`--growth-window 0.001:1`) |
| `split_seed` | 42 | seed of the random mechanism assignment |

**Concordance analysis**

| Parameter | Default | Meaning |
|---|---|---|
| `concordance_tolerance` | 0.01 | maximum spread of the activity ratio for two complexes to count as concordant |
| `cv_threshold` | 0.01 | pairs whose sampled ratio has a larger coefficient of variation are not tested by LP. Larger = more pairs tested (slower, finds more) |
| `sample_size` | 5000 | flux samples for that pre-filter. Fewer samples underestimate the variation and send many hopeless pairs to the LP stage |
| `balanced_threshold` | 1e-7 | activity range below which a complex counts as balanced |
| `seed` | 42 | sampling seed |
| `use_transitivity` | on | skip pairs whose verdict follows from earlier ones (`--no-transitivity` to test all) |
| `growth_fraction` | none | none: full steady-state cone (the setting of the paper); f: growth held at ≥ f × optimum |
| `solver_tol` | 1e-8 | HiGHS feasibility tolerance |
| `lp_time_limit` | 3600 | seconds per LP; a pair whose LP times out stays undecided |

**Kinetic modules**: `--exhaustive` (`exhaustive=True`) runs the full matrix-based
algorithm instead of the fast one. It can merge further modules and find more ACR, but is
much slower at genome scale.

Expert keywords of `COCOA.activity_concordance_analysis` that are not listed here can be
passed unchanged with `cj.run(..., extra={"keyword": value})`.

## Parameter sweeps

To compare kinetic modules across settings, give each parameter a list of values:

```bash
cocoa-jl sweep iJR904.xml sweep_iJR904 --workers 31 \
    --grid concordance_tolerance=0.01,0.001 --grid binding=ordered,random
```

Every combination is a full run in its own subdirectory, and `summary.csv` has one row per
combination (parameters, number of complexes, concordance and kinetic modules, size of the
largest kinetic module, undecided pairs, runtime). The model is preprocessed once per
preprocessing setting and the Julia workers are reused, so only the concordance analysis is
repeated; the whole sweep runs in one Julia process. A combination that already has its
`run.json` is skipped: an interrupted sweep
continues where it stopped when you run the same command again. Parameters given as normal
flags are held fixed for all combinations.

In Python:

```python
summary = cj.sweep("iJR904.xml", "sweep_iJR904",
                   {"concordance_tolerance": [0.01, 0.001], "binding": ["ordered", "random"]},
                   workers=31)
summary.to_pandas()
```

Only the kinetic-module algorithm changes? Recompute without repeating the concordance
analysis: `cocoa-jl kinetic out_iJR904 --exhaustive --out out_iJR904_exhaustive`.

## Running on a cluster

The LPs run in Julia worker processes; give `--workers` the number of cores minus one.
Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=cocoa
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
source /path/to/cocoa-env/bin/activate
export PYTHONUNBUFFERED=1
cocoa-jl run model.xml out_model --workers $((SLURM_CPUS_PER_TASK - 1))
```

On an interactive node, run inside `tmux` or `screen` so the job survives logging out.

**Resources** (measured with COCOA.jl, ordered binding, 64 workers): yeast genome-scale
models take 4 h – 1.5 days and 100–250 GB of memory (roughly 2–4 GB per worker); random
binding roughly doubles the network and takes about 3.5× as long. Nearly all of the time
is the concordance analysis.

## Models that are already split into elementary steps

A model that is already split into elementary, irreversible steps (for example the
`*_pre_balanced_{fixed,random}.mat` files of the reference implementation) is analysed as
given with `--no-preprocess`. A `.mat` file that holds several models needs `--mat-key`:

```bash
cocoa-jl run iJR904_pre_balanced_fixed.mat out --no-preprocess \
    --mat-key model_elementary_fixed --workers 31
```

Do not preprocess such a model again: it would be split a second time.

## Python API

| Function | Does |
|---|---|
| `run(model, outdir, workers=0, **params)` | the whole pipeline; files as above → `Result` |
| `sweep(model, outdir, grid, workers=0, **fixed)` | all combinations → summary `Table` |
| `preprocess(model, output, **params)` | preprocessed model as SBML → report `dict` |
| `kinetic_from_run(rundir, outdir=None, exhaustive=False)` | kinetic modules again from a finished run → `Result` |
| `load_result(outdir)` | read a finished run → `Result` |
| `example_model()` | path of the bundled `e_coli_core` model |
| `versions()` | Julia and COCOA.jl versions of the configured project |
| `setup(...)` | as `cocoa-jl setup` |
| `configure(project=None, julia=None)` | use another Julia project / binary in this process |

A `Result` has `.modules` (one row per kinetic module), `.complexes`, `.acr`, `.acrr`
(tables), `.info` (the run's `run.json`), `.stats` and `.largest_module`. A `Table` is a
dict of columns with `.to_pandas()`, `.to_csv(path)` and `.rows()`. `PREPROCESS_PARAMS` and
`CONCORDANCE_PARAMS` hold all defaults; every function has a docstring (`help(cj.run)`). A
failed job raises `cocoajl.CocoaError`.

The Julia project can also be chosen with the environment variables `COCOAJL_PROJECT` and
`COCOAJL_JULIA`, and the config file moved with `COCOAJL_CONFIG`.

**How it works.** `cocoa-jl setup` builds a Julia project with COCOA.jl and precompiles it.
Every run writes a job file and executes it as `julia --project=<env> driver.jl job.json`;
the Julia process starts its own workers and writes the result files, which the Python side
reads. Julia never runs inside the Python process, which avoids the problems of embedding it
(clashing OpenSSL libraries, version pins, crashes during package installation on
clusters). The Julia code is in `src/cocoajl/julia/` and can be used without Python.

## Troubleshooting

- **A job failed.** The error message names the job file and the command that reruns it,
  e.g. `julia --project=~/.local/share/cocoajl/julia_env .../driver.jl out/job.json`. Run
  that command directly to see the full Julia output.
- **No output for a long time.** Run with `python -u` or `PYTHONUNBUFFERED=1`. The
  concordance analysis prints a progress line per phase; the LP phase of a genome-scale
  model takes hours.
- **"the Julia side is not set up".** Run `cocoa-jl setup` (or point `COCOAJL_PROJECT` at a
  project built by it). `cocoa-jl info` shows which project and julia are used.
- **"blocked-reaction removal changed the growth optimum".** `flux_tol` is too coarse for
  this model; use `--flux-tol 1e-9`.
- **Out of memory.** Memory grows with the number of workers (2–4 GB each at genome scale);
  use fewer workers or request more memory.
- **COCOA.jl was updated.** Rerun `cocoa-jl setup` (same options) to update the Julia
  project; with `--cocoa-path` edits to the checkout take effect without it.

## Citation

If you use cocoa-jl, please cite COCOA.jl (see the
[main README](https://github.com/antoniofranky/COCOA.jl#citation)) and, for the theory,
Langary et al. (2025).
