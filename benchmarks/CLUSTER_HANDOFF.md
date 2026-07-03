# COCOA Kinetic Recompute — Cluster Handoff

You are an agent running on the University of Potsdam HPC (`login1.hpc.uni-potsdam.de`,
user `schaffran1`). Your job: recompute the 62-model kinetic-module benchmark with the
**zero-complex fix**, in two variants, and summarise the results. Everything is staged under
`/work/schaffran1/kinetic_rerun/`.

## Goal & context

COCOA.jl (Bioinformatics application-note revision) recomputes *kinetic modules* from
*concordance modules* of genome-scale metabolic networks, and reports ACR / ACRR
(absolute concentration (ratio) robustness). A bug — the incidence matrix omitted the CRN
**zero complex ∅** for boundary/exchange reactions — inflated kinetic modules and ACR/ACRR.
The fix (branch `revision/zero-complex-fix`, already checked out) adds ∅ as a shared complex
inside `incidence`/`complex_stoichiometry` via `include_zero_complex=true`. It is theorem-sound
(verified with im(YΔ) inclusion checks, Prop S3-4/S3-5/S3-6 of Langary et al. 2025).

The goal is **not** to match the MATLAB reference (`Upstream_Algorithm`) exactly — it is to
confirm COCOA's results are theory-backed and to *explain* the differences. Two runs:

- **Run A** — kinetic from COCOA's own concordance (the official recompute, with the fix).
- **Run B** — kinetic from the *reference's* concordance partition, to localise divergence:
  if Run B's giant modules ≈ the reference's (Table S1 below), the kinetic/upstream step
  agrees and the remaining divergence is entirely in the concordance step (attributable to
  COCOA's transitivity filter, solver, tolerance, sampling — not the kinetic algorithm).

## What is already set up

- `/work/schaffran1/COCOA.jl` — checked out to `revision/zero-complex-fix` (commit `27f1fdd`
  or later; run `git pull` to get this handoff + Run B scripts).
- `/work/schaffran1/kinetic_rerun/` with subdirs `concordance/ models/ ref_concordant/
  out_A/ out_B/ logs/ env/`.
- Julia via juliaup: `export JULIA_DEPOT_PATH=/work/schaffran1/.julia` and
  `export PATH=/work/schaffran1/.juliaup/bin:$PATH` (non-interactive shells don't source
  `.bashrc`, so set these yourself).
- Benchmark env building at `/work/schaffran1/kinetic_rerun/env` (dev COCOA + MATFBCModels
  + JLD2 + HiGHS + AbstractFBCModels). Check `logs/env_setup.log` for completion.
- Data transfer from the workstation was in progress at handoff — **verify counts in Step 0**.

## Step 0 — verify environment and data (do this first)

```bash
export JULIA_DEPOT_PATH=/work/schaffran1/.julia
export PATH=/work/schaffran1/.juliaup/bin:$PATH
cd /work/schaffran1/COCOA.jl && git pull        # get Run B scripts + this doc

# data present? expect 62 each (Example_model excluded; 63 files incl. it is fine)
ls /work/schaffran1/kinetic_rerun/concordance/*.jld2 | wc -l   # COCOA concordance (Run A input)
ls /work/schaffran1/kinetic_rerun/models/*.mat       | wc -l   # elementary models (both runs)
ls /work/schaffran1/kinetic_rerun/ref_concordant/*_concordant_*.mat | wc -l  # Run B input (may be 0 until staged)

# env ready? this should print a giant-module number without error:
julia --project=/work/schaffran1/kinetic_rerun/env \
  /work/schaffran1/COCOA.jl/benchmarks/run_kinetic_step.jl iIS312_Trypomastigote fixed
```

If concordance/models are missing, they must be copied from the workstation
(`application_note/concordance/*.jld2` and
`Upstream_Algorithm/Models/models_with_elementary_steps/*_pre_balanced_*.mat`). The reference
concordant mats for Run B come from `Upstream_Algorithm/Results/concordant/*_concordant_*.mat`.

## Step 1 — Run A (COCOA concordance → kinetic, all 62)

The single-model smoke test above writes `out_A/kinetic_iIS312_Trypomastigote_fixed_seed42.jld2`.
Expected: **giant module 11, ACRR 1** ({(gua_e, ura_e)}). If so, submit the array:

```bash
cd /work/schaffran1/COCOA.jl/benchmarks
AJOB=$(sbatch --parsable submit_kinetic_array.sbatch)
echo "Run A array: $AJOB"
# email a single 'done' summary after the array finishes:
sbatch --dependency=afterany:$AJOB collect_and_notify.sbatch
squeue -u schaffran1
```

Each task loads a concordance JLD2 + model .mat and runs `kinetic_analysis(...; efficient=true)`
(the fix is on), writing `out_A/kinetic_<model>_<variant>_seed42.jld2`.

## Step 2 — Run B (reference concordance → kinetic) — diagnostic

Needs `ref_concordant/` populated. **Validate the reference→COCOA complex mapping on iIS312
before the array** (the reference encodes complexes as `coef*metidx`, e.g. `2*5+3*7`, `1*0`=∅):

```bash
export REF_DIR=/work/schaffran1/kinetic_rerun/ref_concordant
export MODEL_DIR=/work/schaffran1/kinetic_rerun/models
export OUT_DIR=/work/schaffran1/kinetic_rerun/out_B
julia --project=/work/schaffran1/kinetic_rerun/env \
  /work/schaffran1/COCOA.jl/benchmarks/run_kinetic_from_ref_concordance.jl iIS312_Trypomastigote fixed
```

The script prints a `mapped/total` match rate against COCOA's complex set — it must be **~1.0**;
if not, the name↔metabolite mapping is off (fix `ref_name_to_cocoa_id` / `unwrap_model_r` in
`run_kinetic_from_ref_concordance.jl`) before trusting results. **Validation targets** (kinetic
step agrees iff Run B lands near the reference):

| model | reference giant (Table S1) | Run A (COCOA conc.) | Run B should be ≈ |
|---|---|---|---|
| iIS312_Trypomastigote fixed | 67 | 11 | ~67 |
| iIS312_Amastigote random | 1 | 57 | ~1 |

If those hold, submit the array:

```bash
BJOB=$(sbatch --parsable submit_kinetic_runB.sbatch)
sbatch --dependency=afterany:$BJOB collect_and_notify.sbatch
```

## Step 3 — collect & compare

`collect_and_notify.sbatch` writes `/work/schaffran1/kinetic_rerun/kinetic_summary.csv`
(columns: `model_variant, giant_A, acr_A, acrr_A, giant_B, acr_B, acrr_B, ref_match_rate`)
and emails on completion. Compare `giant_A`/`giant_B` against the reference Table S1 values
below and report: (a) does Run B ≈ reference (→ divergence is concordance-only), and
(b) which models Run A over/under-sizes vs the reference, with the magnitude.

## Reference Table S1 values (for validation) — iIS312 + iMM904

`ordered` = fixed. Columns: complexes / kinetic giant (max size) / free-ACR / free-ACRR.

| variant | complexes | giant | ACR-free | ACRR-free |
|---|---|---|---|---|
| iMM904 ordered | 8536 | 1193 | 0 | 10 |
| iMM904 random | 19668 | 69 | 0 | 0 |
| iIS312_Amastigote ordered | 1446 | 3 | 0 | 0 |
| iIS312_Amastigote random | 3526 | 1 | 0 | 0 |
| iIS312_Epimastigote ordered | 1738 | 136 | 0 | 8 |
| iIS312_Epimastigote random | 4265 | 23 | 0 | 7 |
| iIS312_Trypomastigote ordered | 1046 | 67 | 0 | 1 |
| iIS312_Trypomastigote random | 2185 | 1 | 0 | 0 |
| iIS312 ordered | 1740 | 5 | 1 | 0 |
| iIS312 random | 4281 | 1 | 0 | 0 |

(Full Table S1: `toolbox/data/kinetic_modules_are_source/ads7269_table_s1.xlsx` on the
workstation — kinetic-coupling block is cols 8–15, free-metabolite ACR/ACRR are cols 20/22.)

## Known facts / gotchas

- MAT variable name per model: `model_elementary_fixed` or `model_elementary_random`.
- `test/Project.toml` does NOT have MATFBCModels — that's why Run A/B use the dedicated
  `kinetic_rerun/env`, not `--project=test`.
- ∅ is excluded from concordance modules and never appears in reported kinetic modules; the
  fix is a no-op on closed networks (EnvZ-OmpR, deficiency-two).
- The concordance step diverges from the reference (COCOA's unbalanced modules are coarser;
  the balanced module matches exactly). This is expected and attributed to COCOA's transitivity
  filter + numerical/solver/tolerance/sampling differences — do NOT try to force a match.
- Email notifications go to the address in the sbatch `--mail-user` line — **verify it**
  (`schaffran1` vs `schaffrank1` vs `schaffranke1` @uni-potsdam.de) before submitting.
