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
# For parallel optimizations us Distributed
using Distributed
# You can add a maximum of n-1 processes, where n is the number of available cores (one process is the main process already running, here n=16)

addprocs(15)

#Load required packages on worker processes 
@everywhere using HiGHS, COCOA

# Load model
model = COBREXA.load_model("ecoli_core.xml")

model_canon = convert(A.CanonicalModel.Model,model)

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

    # Analysis parameters
    concordance_tolerance=1e-7,      # Tolerance for concordance detection
    balanced_threshold=1e-8,         # Threshold for balanced complexes
    cv_threshold=1e-7,               # Coefficient of variation filtering

    # Performance settings
    batch_size=50_000,               # Candidates per optimization batch
    workers=workers(),               # Parallel workers
    use_transitivity=true,           # Exploit transitivity to reduce tests

    # Sampling configuration
    sample_size=100,                 # Samples for CV estimation
    seed=1234,                       # Random seed for reproducibility

    # Additional analysis
    kinetic_analysis=true,           # Identify ACR metabolites and modules
    detailed_results=false           # Include activity ranges, lambda, and lambda_pairs table
)
```

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

For parallel optimizations use the Distributed.jl package.

```julia

```


## Requirements

- Julia ≥ 1.12
- A supported LP solver (e.g. [HiGHS.jl](https://github.com/jump-dev/HiGHS.jl))
- Metabolic models loadable via [AbstractFBCModels.jl](https://github.com/COBREXA/AbstractFBCModels.jl) (e.g. SBML via SBMLFBCModels.jl)


## Citation

If you use COCOA.jl in your research, please cite:

```
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

