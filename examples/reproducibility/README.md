# Reproducibility & robustness experiments

Scripts that reproduce the determinism / numerical-stability checks reported in the
COCOA.jl application note, and that can be re-run by reviewers on their own hardware or
on an HPC cluster.

## What it runs

`robustness_experiments.jl` runs three controlled experiments on one model
(default: the bundled `e_coli_core`), each recording a **label-invariant partition
fingerprint** so that identical counts are distinguished from identical partitions:

| Experiment | Grid | Question |
|---|---|---|
| **A. seed × transitivity** | 10 seeds × `use_transitivity ∈ {true,false}` | Does the transitivity filter amplify borderline, sampling-dependent pairwise decisions into whole-module (and thus seed) changes? Direct LP testing (`false`) should be more seed-stable, at higher cost. |

> **Baseline = paper setting: full steady-state cone, no optimal-biomass constraint**
> (`objective_bound=nothing`). Experiments A/C/D use this; experiment B adds an
> obj-bound variant only for contrast.
| **B. biomass isolation** | {biomass + obj-bound, biomass + no bound, biomass removed + no bound} × 3 seeds | Is the biomass reaction's wide-magnitude stoichiometry the source of numerical instability? Compare *biomass removed* vs *biomass kept, no bound* (both full-cone). |
| **C. tolerance sweep** | `concordance_tolerance,balanced_threshold ∈ {1e-6,1e-8,1e-10}` × 3 seeds | How sensitive are modules / ACR / ACRR to the concordance/balanced tolerances? |
| **D. blocked-tolerance sweep** | `remove_blocked_reactions flux_tolerance ∈ {1e-6,1e-8,1e-10}` × 3 seeds (re-preprocessed per level) | The third ("blocked") threshold from the reviewer/author note. |
| **E. efficient=false** | `kinetic_efficient=false` × 3 seeds, full cone, ordered binding | Exhaustive ACR/ACRR via the full matrix/deficiency path (reference-comparable, slow). Not in the default set — run with `EXPERIMENTS=E`. |

## Run locally

```bash
julia --project=examples/reproducibility -e 'import Pkg; Pkg.instantiate()'
NPROCS=15 julia --project=examples/reproducibility examples/reproducibility/robustness_experiments.jl
```

## Run on SLURM

```bash
sbatch examples/reproducibility/submit_slurm.sh
# or for a genome-scale model, one experiment, more cores:
sbatch --export=ALL,MODEL=/path/iAB_RBC_283.xml,EXPERIMENTS=A \
       --cpus-per-task=32 examples/reproducibility/submit_slurm.sh
```

## Configuration (environment variables)

`MODEL`, `BIOMASS_IDS`, `NPROCS`, `SEEDS`, `EXPERIMENTS` (subset of `A,B,C`),
`OUTDIR`, `OPTIMIZER` (`HiGHS`|`GLPK`). See the header of
`robustness_experiments.jl` for details.

## Output

CSVs in `OUTDIR` (default `examples/reproducibility/results/`): `A_seed_transitivity.csv`,
`B_biomass_isolation.csv`, `C_tolerance_sweep.csv`, and a combined `all_runs.csv`.
Columns include `n_concordance_modules`, `n_concordant_total`, `n_acr`, `n_acrr`,
`partition_fp` (16-hex partition fingerprint), and `elapsed_s`.
