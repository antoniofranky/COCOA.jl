# COCOA.jl

[![Build Status](https://github.com/antoniofranky/COCOA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/antoniofranky/COCOA.jl/actions/workflows/CI.yml?query=branch%3Amain)

**COnstraint-based COncordance Analysis** for biochemical networks.

## Overview

COCOA.jl identifies **concordant complexes** in biochemical networks - pairs of complexes that maintain constant activity ratios across all feasible steady states. This property can then be used for the identification of kinetic modules and metabolites exhibiting absolute concentration robustness (ACR) or pairs of metabolites with absolute concentration ratio robustness (ACRR).
## Installation

```julia
using Pkg
Pkg.add("COCOA")
```


### Workflow

For optimal results, preprocess your model following the recommended workflow:

```julia
import COBREXA
import SBMLFBCModels
import AbstractFBCModels as A
# For parallel optimizations use Distributed
using Distributed
# You can add a maximum of n-1 processes, where n is the number of available cores (one process is the main process already running, here n=16)

addprocs(15)

#Load required packages on worker processes 
@everywhere using HiGHS, COCOA

# Load the bundled example model (resolves to the copy shipped with COCOA.jl)
model = COBREXA.load_model(joinpath(pkgdir(COCOA), "test", "e_coli_core.xml"))

model_canon = convert(A.CanonicalModel.Model, model)

# Recommended preprocessing pipeline for kinetic module analysis (immutable - preserves original)
model_processed = model_canon |>
    normalize_bounds |>
    m -> remove_blocked_reactions(m; optimizer=HiGHS.Optimizer) |>
    remove_orphans |>
    split_into_elementary |> # This ensures mass action kinetics. Skip this step, if you only want concordance modules
    split_into_irreversible

# Run concordance analysis on preprocessed model
results = activity_concordance_analysis(
    model_processed;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999),
    kinetic_analysis=true # false, if you only want concordance modules
)
```

#### Preprocessing Steps Explained

1. **`normalize_bounds`** - Standardize reaction bounds (-1000/1000 for reversible/irreversible)
2. **`remove_orphans`** - Remove unused metabolites and reactions
3. **`remove_blocked_reactions`** - Identify and remove reactions with zero flux via Flux Variabiltiy Analysis (FVA)
4. **`split_into_elementary`** - Decompose reactions into elementary steps with either ordered or random mechanisms, to ensure mass action kinetics
5. **`split_into_irreversible`** - Convert reversible reactions into forward/reverse pairs


### Advanced Analysis Options

```julia
results = activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,

    # Objective constraint
    objective_bound=COBREXA.relative_tolerance_bound(0.999),

    # Analysis parameters (values shown are the defaults)
    concordance_tolerance=0.01,      # Tolerance for concordance detection
    balanced_threshold=1e-7,         # Threshold for balanced complexes
    cv_threshold=0.01,               # Coefficient of variation filtering

    # Performance settings
    batch_size=50_000,               # Candidates per optimization batch
    workers=workers(),               # Parallel workers
    use_transitivity=true,           # Exploit transitivity to reduce tests

    # Sampling configuration
    sample_size=1000,                # Samples for CV estimation
    seed=UInt(1234),                 # Random seed (deterministic by default)

    # Additional analysis
    kinetic_analysis=true,           # Identify ACR metabolites and modules
    kinetic_efficient=true,          # Fast ACR/ACRR path (see note below)
    detailed_results=false           # Include activity ranges, lambda, and lambda_pairs table
)
```

> **ACR/ACRR detection completeness.** `kinetic_efficient=true` (the default) uses a fast pairwise
> search that scales to genome-scale networks, but it is inherently *less exhaustive* than the full
> matrix/deficiency path and can under-report ACR metabolites / ACRR pairs. We have confirmed it under-reports on the
> EnvZ-OmpR example (verified against the analytically known result); the exhaustive path. Pass `kinetic_efficient=false` whenever exhaustive
> ACR/ACRR detection matters more than runtime.

## Results Structure

`activity_concordance_analysis` returns a NamedTuple of columnar tables that can be converted to DataFrames:

```julia
using DataFrames
df_complexes = DataFrame(result.complexes)   # one row per complex
df_acr       = DataFrame(result.acr)         # ACR metabolites
df_acrr      = DataFrame(result.acrr)        # ACRR metabolite pairs
```

### `result.complexes`

One row per complex in the model (order matches `COCOA.complex_stoichiometry(model)`):

| Column | Type | Description |
|--------|------|-------------|
| `complex_id` | `String` | Complex identifier |
| `concordance_module` | `Int` | Module ID: `0` = balanced, `-1` = singleton, positive = module index |
| `kinetic_module` | `Int` | Kinetic module ID (`0` = not assigned; requires `kinetic_analysis=true`) |
| `classification` | `String` | `"balanced"`, `"positive"`, `"negative"`, or `"unrestricted"` |

### `result.acr`

| Column | Type | Description |
|--------|------|-------------|
| `metabolite_id` | `String` | ID of an ACR metabolite candidate (requires `kinetic_analysis=true`) |

### `result.acrr`

| Column | Type | Description |
|--------|------|-------------|
| `metabolite_1` | `String` | First metabolite of an ACRR pair (requires `kinetic_analysis=true`) |
| `metabolite_2` | `String` | Second metabolite of an ACRR pair |

### Detailed results (`detailed_results=true`)

Pass `detailed_results=true` to include additional columns in `result.complexes` and an extra `result.lambda_pairs` table:

```julia
result = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, detailed_results=true)
```

Additional columns in `result.complexes`:

| Column | Type | Description |
|--------|------|-------------|
| `min_activity` | `Float64` | Minimum activity across feasible steady states |
| `max_activity` | `Float64` | Maximum activity across feasible steady states |
| `lambda` | `Float64` | Activity ratio relative to the first complex in the concordance module (`1.0` for balanced; `NaN` for singletons) |
| `trivially_balanced` | `Bool` | Whether the complex is trivially balanced |

Additional table `result.lambda_pairs` — directly measured pairwise lambda values:

| Column | Type | Description |
|--------|------|-------------|
| `complex_1` | `String` | First complex in the concordant pair |
| `complex_2` | `String` | Second complex in the concordant pair |
| `lambda` | `Float64` | Measured activity ratio `λ(complex_1, complex_2)` |

`result.stats` — a `Dict{String,Any}` with comprehensive analysis metrics:

```julia
# Model information
result.stats["n_complexes"]                       # Total complexes
result.stats["n_reactions"]                       # Total reactions (after splitting)
result.stats["n_metabolites"]                     # Total metabolites
result.stats["n_balanced"]                        # Balanced complexes
result.stats["n_trivially_balanced"]              # Trivially balanced complexes

# Concordance results
result.stats["n_concordant_total"]                # Total concordant pairs
result.stats["n_concordant_opt"]                  # Pairs found by optimization
result.stats["n_concordant_inferred"]             # Pairs inferred via transitivity
result.stats["n_trivially_concordant"]            # Trivially concordant pairs
result.stats["n_trivial_pairs"]                   # Trivially concordant pairs (pre-optimization)
result.stats["n_non_concordant_pairs"]            # Non-concordant pairs
result.stats["n_concordance_modules"]             # Concordance modules found
result.stats["n_candidate_pairs"]                 # Pairs tested by optimization

# Error accounting
result.stats["n_timeout_pairs"]                   # Pairs that timed out
result.stats["n_infeasible_or_unbounded_pairs"]   # Infeasible or unbounded pairs
result.stats["n_numerical_error_pairs"]           # Numerical error pairs

# Performance
result.stats["batches_completed"]                 # Optimization batches run
result.stats["n_total_optimizations"]             # Total LP solves
result.stats["elapsed_time"]                      # Total analysis time (seconds)
result.stats["n_workers"]                         # Workers used

# Algorithm parameters
result.stats["concordance_tolerance"]
result.stats["balanced_threshold"]
result.stats["cv_threshold"]
result.stats["batch_size"]
result.stats["use_transitivity"]
result.stats["seed"]
```

## Preprocessing Functions

All preprocessing functions follow an **immutable pattern** - they return modified copies and preserve the original model:

### `normalize_bounds`

Standardize reaction bounds:

```julia
model_normalized = normalize_bounds(
    model;
    lower_bound=-1000.0,              # Bound for reversible reactions
    upper_bound=1000.0,               # Bound for unlimited reactions
    normalize_objective_bounds=true   # Force objective reactions forward-only
)
```

### `remove_orphans`

Remove metabolites and reactions with zero stoichiometry:

```julia
model_cleaned = remove_orphans(
    model;
    remove_mets=true,    # Remove unused metabolites
    remove_rxns=true     # Remove empty reactions
)
```

### `find_blocked_reactions` / `remove_blocked_reactions`

Identify and remove reactions with zero flux via FVA:

```julia
# Find blocked reactions
blocked_ids = find_blocked_reactions(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999),
    flux_tolerance=1e-9
)

# Remove blocked reactions (returns tuple: model + removed IDs)
model_unblocked, blocked_ids = remove_blocked_reactions(
    model;
    optimizer=HiGHS.Optimizer,
    objective_bound=COBREXA.relative_tolerance_bound(0.999)
)
```

### `split_into_elementary`

Decompose reactions into elementary steps with reaction mechanisms:

```julia
# Split all eligible reactions (default: ordered mechanism)
model_elementary = split_into_elementary(model)

# Split only specific reactions
model_elementary = split_into_elementary(
    model;
    split_reactions=["R_PGI", "R_PFK", "R_FBA"]
)

# Split specific reactions with different mechanisms
model_elementary = split_into_elementary(
    model;
    split_reactions=["R_PGI", "R_PFK", "R_FBA"],
    random_reactions=["R_FBA"]  # Use random mechanism for R_FBA
)

# Split all eligible reactions, 50% with random mechanism
model_elementary = split_into_elementary(
    model;
    random=0.5,
    seed=1234  # For reproducibility
)
```

**Parameters:**
- `split_reactions`: Vector of reaction IDs to split (empty = all eligible)
- `random_reactions`: Reactions to split using random mechanism
- `random`: Fraction (0.0-1.0) of remaining reactions to split randomly
- `seed`: Random seed for reproducible mechanism assignment

### `split_into_irreversible`

Convert reversible reactions into forward/reverse pairs:

```julia
model_irreversible = split_into_irreversible(model)
```

### Parallelization

Concordance testing scales with the number of available worker processes. Add workers with
`Distributed.jl` **before** loading `COCOA` on the workers, then pass `workers=workers()` to the
analysis:

```julia
using Distributed
addprocs(15)                         # n-1 workers, where n = number of physical cores
@everywhere using HiGHS, COCOA

results = activity_concordance_analysis(
    model_processed;
    optimizer=HiGHS.Optimizer,
    workers=workers(),               # distribute LP solves across all workers
    kinetic_analysis=true,
)
```


## Calling COCOA.jl from Python

COCOA.jl can be driven from Python through [JuliaCall](https://github.com/JuliaPy/PythonCall.jl).

```bash
pip install juliacall
```

```python
from juliacall import Main as jl

jl.seval("import Pkg")
for pkg in ("COCOA", "COBREXA", "HiGHS", "SBMLFBCModels", "AbstractFBCModels"):
    jl.seval(f'haskey(Pkg.project().dependencies, "{pkg}") || Pkg.add("{pkg}")')

jl.seval("import COBREXA; import SBMLFBCModels; import AbstractFBCModels as A")
jl.seval("using HiGHS, COCOA")

jl.seval('model = COBREXA.load_model(joinpath(pkgdir(COCOA), "test", "e_coli_core.xml"))')
jl.seval("model_canon = convert(A.CanonicalModel.Model, model)")
result = jl.seval("activity_concordance_analysis(model_canon; optimizer=HiGHS.Optimizer, kinetic_analysis=true)")

print("ACR metabolites:", list(result.acr.metabolite_id))
```

A complete, runnable script (with preprocessing) is in
[`examples/python_quickstart.py`](examples/python_quickstart.py).

> **Clean shutdown.** Let the Python process return normally rather than calling `os._exit()` while
> Julia objects are still alive — forcing an abrupt teardown can trigger a *bus error* as the Julia
> GC finalizers run against an already–torn-down runtime.

## Requirements

- Julia ≥ 1.12
- A supported LP solver (e.g. [HiGHS.jl](https://github.com/jump-dev/HiGHS.jl))
- Metabolic models loadable via [AbstractFBCModels.jl](https://github.com/COBREXA/AbstractFBCModels.jl) (e.g. SBML via SBMLFBCModels.jl)


## Citation

If you use COCOA.jl in your research, please cite:

```bibtex
@article{Schaffranke2026COCOA,
  author  = {Schaffranke, Anton and K{\"u}ken, Anika and Nikoloski, Zoran},
  title   = {COCOA.jl: A Julia package for high-performance analysis of concordance and kinetic modules in biochemical networks},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {In revision}
}
```

An archival snapshot of the code is available on Zenodo: <!-- DOI-PLACEHOLDER: replace after minting the Zenodo release (WP4) -->
[DOI pending].

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- Küken, A., Langary, D. and Nikoloski, Z. (2022) The hidden simplicity of metabolic networks is revealed by multireaction dependencies. *Sci. Adv.*, **8**, eabl6962. https://doi.org/10.1126/sciadv.abl6962
- Langary, D., Küken, A. and Nikoloski, Z. (2025) Kinetic modules are sources of concentration robustness in biochemical networks. *Sci. Adv.*, **11**, eads7269. https://doi.org/10.1126/sciadv.ads7269
- Kratochvíl, M., Wilken, S.E., Ebenhöh, O., Schneider, R. and Satagopam, V.P. (2025) COBREXA 2: tidy and scalable construction of complex metabolic models. *Bioinformatics*, **41**, btaf056. https://doi.org/10.1093/bioinformatics/btaf056
- Kaufman, D.E. and Smith, R.L. (1998) Direction choice for accelerated convergence in hit-and-run sampling. *Oper. Res.*, **46**, 84–95. https://doi.org/10.1287/opre.46.1.84
