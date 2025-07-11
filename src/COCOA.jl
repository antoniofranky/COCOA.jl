"""
    module COCOA

COnstraint-based COncordance Analysis for metabolic networks - optimized for large-scale models.

This module implements memory-efficient methods for identifying concordant complexes in metabolic
networks. Optimized for models with >30,000 complexes and reactions on HPC clusters with
SharedArrays support for single-node parallelism.

## Performance Optimizations

COCOA is optimized for efficiency through integration with COBREXA.jl:

- **Model reuse**: Uses `COBREXA.screen_optimization_model` to avoid repeated model creation/destruction
- **Optimized constraint building**: Leverages COBREXA's efficient sign splitting and variable pruning
- **Smart solver settings**: Automatically applies optimized solver configurations for concordance testing
- **Memory efficiency**: Automatic model conversion to CanonicalModel format for optimal performance
- **Parallel infrastructure**: Seamless integration with COBREXA's worker management and load balancing

## Reproducibility Guarantees

COCOA implements reproducible random number generation using StableRNGs.jl:

- **Cross-platform reproducibility**: Uses StableRNGs instead of Julia's default RNG
- **Deterministic sampling**: All random operations (warmup selection, batch seeds) are reproducible
- **Simple seeding**: Single RNG with configurable seed for all random operations

### Usage for Reproducible Results

```julia
# Same seed will produce identical results across runs and platforms
results1 = concordance_analysis(model; optimizer=optimizer, seed=1234)
results2 = concordance_analysis(model; optimizer=optimizer, seed=1234)
# results1 == results2 (within numerical precision)

# Different seeds produce different but reproducible results
results3 = concordance_analysis(model; optimizer=optimizer, seed=5678)
# results3 != results1, but results3 is reproducible with seed=5678
```

For complete reproducibility, ensure:
1. Same Julia version and package versions
2. Same model and analysis parameters  
3. Same optimizer and solver settings
4. Same seed value
"""
module COCOA

using COBREXA
using AbstractFBCModels
using ConstraintTrees
using SparseArrays
using LinearAlgebra
using Distributed
using SharedArrays
using DataFrames
using Statistics
using Random
using StableRNGs
using JuMP
using DocStringExtensions
using ProgressMeter
using JLD2
using HiGHS

# Include preprocessing modules
include("preprocessing/ElementarySteps.jl")
include("preprocessing/ModelPreparation.jl")
using .ElementarySteps
using .ModelPreparation

# Include main modules
include("data_structures.jl")
include("correlation.jl")
include("constraints.jl")
include("analysis.jl")


# Re-export main functions
export concordance_constraints, concordance_analysis
export split_into_elementary_steps
export prepare_model_for_concordance

end # module COCOA