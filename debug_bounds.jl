#!/usr/bin/env julia

using Pkg
Pkg.activate("/home/anton/master-thesis/toolbox/julia/COCOA")

import ConstraintTrees as C

# Create different types of bounds to debug
between_bound = C.Between(0.0, 1.0)
equal_bound = C.EqualTo(0.5)

println("Between bound:")
println("  Type: $(typeof(between_bound))")
println("  Fields: $(fieldnames(typeof(between_bound)))")
println("  Lower: $(between_bound.lower)")
println("  Upper: $(between_bound.upper)")

println("\nEqualTo bound:")
println("  Type: $(typeof(equal_bound))")
println("  Fields: $(fieldnames(typeof(equal_bound)))")
for field in fieldnames(typeof(equal_bound))
    println("  $field: $(getfield(equal_bound, field))")
end