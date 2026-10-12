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
- **The CV pre-filter works on full-precision samples.** Samples were rounded before the ratio
  CV was computed, so pairs with small activities (e.g. uptakes coupled to growth with
  λ ≈ 0.001–0.004) got a large CV and were discarded without an LP: false splits. Samples are
  no longer rounded, and activities below an explicit zero tolerance count as zero.
- **No cap on the Charnes–Cooper variable t** (`cc_scale_bound=Inf`, was 999). The cap cut off
  every steady state whose reference activity is below ~1e-3 of the flux scale; those pair LPs
  were reported infeasible and left undecided (on one yeast model 615 undecided pairs with the
  cap, 0 without, and faster). A pair LP that is proven unbounded now refutes concordance.
- **Concordance is decided with a relative tolerance**:
  `|max − min| ≤ concordance_rtol·max(|max|,|min|) + concordance_tolerance`, defaults
  `concordance_rtol=1e-5`, `concordance_tolerance=1e-9`. The old absolute 0.01 on λ, tested in
  one orientation, accepted pairs with small λ that are not concordant (λ ∈ [0.0080, 0.0086]),
  and through transitivity made results depend on the seed. On split yeast models truly
  concordant pairs have a spread ≤ 1e-10, non-concordant ones ≥ 2e-3. The old rule:
  `concordance_tolerance=0.01, concordance_rtol=0`.
- **The growth bound is scaled in the pair LPs** (`objective_bound` / growth fraction). The
  pair LPs work in w = t·v, where every flux bound is multiplied by t; the objective bound was
  not, so it bounded growth/t instead of growth and the LPs searched the wrong set of steady
  states (on iJR904 at the optimum 92 % of the LPs infeasible, 90 % of pairs undecided; after
  the fix 0.6 % undecided). Runs over the full cone (`objective_bound=nothing`) are unaffected.

### Reproducibility

- Static scheduling in all LP stages (pair tests, AVA, blocked-reaction detection): identical
  runs now give identical partitions.
- `remove_blocked_reactions` warns when `flux_tolerance` is at or below the solver's accuracy.
- No thread-indexed buffers in threaded loops.

### New

- `kinetic_analysis` warns when one complex belongs to every coupling set — the signature of
  the artificial giant module above.
- `classify_blocked`, `report_undecided`, `resolve_cold!`, `optimize_verified!`,
  `add_objective_bound!`; `cc_scale_bound` (default `Inf`) and `concordance_rtol` are
  parameters of `activity_concordance_analysis`.
- **Python package `cocoa-jl`** (`python/`): `pip install`, `cocoa-jl setup` installs the Julia
  side, then `cocoa-jl run model.xml|.json|.mat out --workers N` runs preprocessing,
  concordance and kinetic modules in parallel and writes CSV/JSON plus the preprocessed model
  (`model_preprocessed.xml`); also `preprocess`, `kinetic` (recompute from a run), `sweep` and
  a Python API. `.mat` files holding several models (COBRA workspaces) take `--mat-key`; ids are
  made SBML-safe for the saved model and shown unescaped in the CSV files.
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

- **Undecided pairs** were mostly caused by the Charnes–Cooper cap (removed, see above); with
  no cap the split yeast models tested have none or very few. Undecided pairs are never merged,
  so they can only make concordance modules smaller, never larger. Check
  `stats["n_unknown_pairs"]`.
- **The fast kinetic path (`kinetic_efficient=true`) can miss ACR** (one model of 20 tested:
  1 vs 2 ACR). On one network whose preprocessing left a complex in every coupling set, the
  exhaustive path also produced a *smaller* largest module than the fast path, which should not
  happen; this is being traced.
- **The CV pre-filter is a heuristic**: a pair it discards is never tested by LP. An
  independent check (every complex outside the five largest modules of seven yeast models
  against the module, by LP) found no missed member after the sampling fix; `sample_size` ≥ 5000
  keeps the risk low.
- **`use_transitivity=true`** is a shortcut whose effect on module boundaries has not been fully
  quantified.
- **`flux_tolerance` is model-dependent** (see README); check that growth is unchanged after
  blocked-reaction removal.

### Performance

- Sparse difference in ACR/ACRR detection; linear `merge_map` construction; parallel CV block
  cache in the candidate filter.

## v1.1.0

Release accompanying the Bioinformatics application note.
