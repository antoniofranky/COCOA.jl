# ================================================================================================
# Kinetics Module - Consolidated Exports
# ================================================================================================
# This module provides the redesigned kinetic analysis implementation
# with modular architecture, cached computations, and batch optimizations.
#
# Usage:
#   include("kinetics/Kinetics.jl")
#   using .Kinetics
#
# Or to replace the old implementation in COCOA.jl:
#   include("kinetics/Kinetics.jl")
#   # Then use Kinetics.kinetic_analysis instead of the old kinetic_analysis

module Kinetics

using LinearAlgebra
using SparseArrays

# Include all submodules in dependency order
include("union_find.jl")
include("tarjan.jl")
include("linear_algebra.jl")
include("network.jl")
include("upstream.jl")
include("coupling.jl")
include("acr.jl")
include("deficiency.jl")
include("kinetic_analysis.jl")

# ================================================================================================
# Public API Exports
# ================================================================================================

# Main entry point (returns NamedTuple, not struct)
export kinetic_analysis

# Network representation
export ReactionNetwork, complex_stoichiometry

# Upstream algorithm
export upstream_set, compute_upstream_sets
export find_terminal_complexes, find_nonterminal_complexes

# Coupling and merging
export build_coupling_matrix, merge_coupled_modules, can_merge
export ColumnSpanProjector, is_in_span

# ACR/ACRR identification (returns NamedTuple, not struct)
export identify_acr, identify_acrr, identify_acr_acrr

# Deficiency calculations
export structural_deficiency, mass_action_deficiency
export check_deficiency_one, apply_theorem_s4_6
export is_weakly_reversible

# Utility data structures
export UnionFind, find_root!, union!, get_groups, get_sets
export tarjan_scc, is_terminal_scc

# Batch operations
export batch_acr_check, batch_acrr_check, parallel_acrr_check
export ThreadWorkspaces, get_workspace

end # module Kinetics
