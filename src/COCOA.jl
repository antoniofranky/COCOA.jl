"""
COCOA

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
results1 = activity_concordance_analysis(model; optimizer=optimizer, seed=1234)
results2 = activity_concordance_analysis(model; optimizer=optimizer, seed=1234)
# results1 == results2 (within numerical precision)

# Different seeds produce different but reproducible results
results3 = activity_concordance_analysis(model; optimizer=optimizer, seed=5678)
# results3 != results1, but results3 is reproducible with seed=5678
```

For complete reproducibility, ensure:
1. Same Julia version and package versions
2. Same model and analysis parameters  
3. Same optimizer and solver settings
4. Same seed value
"""
module COCOA

# Core dependencies - following JuMP style guide recommendations
# Use qualified imports to avoid namespace pollution and improve code clarity
import COBREXA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM
import SBMLFBCModels
import SparseArrays
import Statistics
import Random
import StableRNGs
import JuMP as J
import Dates
import ConstraintTrees as C
import Distributed as D
import JLD2

import Graphs
import LinearAlgebra

using DocStringExtensions


# Include main modules
include("data_structures.jl")
include("matrix_builders.jl")  # Must come before constraints.jl (provides generate_complex_id)
include("constraints.jl")
include("filter.jl")
include("variability.jl")
include("concordance.jl")
include("kinetic_analysis.jl")
include("create_envz_ompr_model.jl")
include("create_deficiency_two_model.jl")

# Include preprocessing functions
include("preprocessing/preprocessing.jl")
include("preprocessing/elementary_splitting.jl")  # This includes mechanisms.jl
include("preprocessing/irreversible_splitting.jl")



# Export preprocessing functions
export split_into_elementary
export split_into_irreversible
export find_blocked_reactions, remove_blocked_reactions!, remove_blocked_reactions
export normalize_bounds!, normalize_bounds
export remove_orphans!, remove_orphans


# Export main analysis functions
export concordance_constraints, activity_variability_analysis, activity_concordance_analysis

# Export matrix building functions
export incidence, stoichiometry, complex_stoichiometry


# Export kinetic analysis functions
export kinetic_analysis, upstream_algorithm, identify_acr_acrr, create_envz_ompr_model, create_deficiency_two_model

# Export deficiency calculation functions
export structural_deficiency, mass_action_deficiency_bounds
end # module COCOA