# Changelog

## v1.2.0

Correctness fixes found while re-running a 343-species yeast analysis, and better guidance for
genome-scale runs (in particular from Python). **Recommended for everyone analysing
genome-scale models**: two of the fixes below can change results there.

### Fixes that can change results

- **Blocked-reaction removal no longer keeps reactions whose FVA LP failed.** Inside the
  warm-started variability sweep some LPs came back unsolved, and their reactions were kept as
  "not blocked". A single leftover blocked reaction can sit in every coupling set and merge
  the whole network into one artificial giant kinetic module with spurious ACR/ACRR (on a yeast
  GEM: 4,248 complexes and 339 ACR instead of 50 and 0). Unsolved LPs are now re-solved from a
  cleared solver state (`resolve_cold!`); whatever still fails is reported
  (`on_undecided=:warn`, default) or stops the run (`on_undecided=:error`).
- **A failed pair LP is no longer recorded as "not concordant".** Pair verdicts are now
  three-way: concordant, not concordant, undecided (`stats["n_unknown_pairs"]`). Recording a
  failure as "not concordant" was additionally lifted to whole module pairs under
  `use_transitivity=true`.
- **LPs that HiGHS reports as OPTIMAL without a feasible point are recovered**
  (`optimize_verified!`, counter `n_recovered_lps`) instead of being left without a value. On
  split yeast models this had affected ~7 % of candidate pairs.
- A refuted pair can no longer be marked concordant by its second direction.
- `ALMOST_OPTIMAL` solves are used consistently (they were counted as success but yielded no
  value).

### Reproducibility

- Static scheduling in all LP stages (pair tests, AVA, blocked-reaction detection): identical
  runs now give identical partitions.
- `remove_blocked_reactions` warns when `flux_tolerance` is at or below the solver's accuracy.
- No thread-indexed buffers in threaded loops.

### New

- `kinetic_analysis` warns when one complex belongs to every coupling set — the signature of
  the artificial giant module above.
- `classify_blocked`, `report_undecided`, `resolve_cold!`, `optimize_verified!`;
  `cc_scale_bound` is a parameter of `activity_concordance_analysis` (default 999, as before).
- `detect_robustness_via_span`: ACR/ACRR/gACRR from the column span of Y_Delta (exported, not
  yet used by the pipeline).
- Progress messages via `@info` (`COCOA_PROGRESS=0` silences them).

### Examples and documentation

- `examples/python/run_genome_scale.py`: the full pipeline from Python **in parallel**
  (Julia workers started from the embedded Julia), with a time limit per LP, progress logging,
  results written to CSV/JSON, and a check that blocked-reaction removal leaves the growth rate
  unchanged.
- `examples/python/build_env.jl` pins PythonCall.jl to the installed `juliacall` version (a
  mismatch stops juliacall with "PythonCall.jl did not start properly").
- `quickstart_simple.py` analyses the full steady-state cone (`objective_bound=nothing`), as the
  README recommends, instead of a near-optimal face.
- README: choosing `flux_tolerance`, the growth window for blocked-reaction removal,
  `on_undecided`, a time limit per LP, `sample_size` ≥ 5000 for genome-scale models, and
  running genome-scale models from Python. Corrected: `remove_blocked_reactions` returns the
  model, not a tuple.

### Known issues (open, under investigation)

- **Undecided pairs remain** on genome-scale models (median ~0.9 % of candidate pairs on split
  yeast models, ~5 % on cyanobacterial models). They are pair LPs that stay infeasible even when
  re-solved from scratch; whether these are structural or numerical is open. Undecided pairs are
  never merged, so they can only make concordance modules smaller, never larger. Check
  `stats["n_unknown_pairs"]`.
- **The fast kinetic path (`kinetic_efficient=true`) can miss ACR** (one model of 20 tested:
  1 vs 2 ACR). On one network whose preprocessing left a complex in every coupling set, the
  exhaustive path also produced a *smaller* largest module than the fast path, which should not
  happen; this is being traced.
- **The CV pre-filter is a heuristic**: a pair it discards is never tested by LP. Its miss rate
  has not been measured yet; `sample_size` ≥ 5000 keeps it low.
- **`use_transitivity=true`** is a shortcut whose effect on module boundaries has not been fully
  quantified.
- **`flux_tolerance` is model-dependent** (see README); check that growth is unchanged after
  blocked-reaction removal.

### Performance

- Sparse difference in ACR/ACRR detection; linear `merge_map` construction; parallel CV block
  cache in the candidate filter.

## v1.1.0

Release accompanying the Bioinformatics application note.
