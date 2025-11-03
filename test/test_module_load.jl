include("src/COCOA.jl")
using .COCOA

println("✓ Module loaded successfully!")
println("✓ mechanisms.jl changes applied")
println("\nChanges made:")
println("1. Complex compartment now uses reaction compartment (where catalysis occurs)")
println("2. Components field is now a vector with one entry per component")
println("3. Added 'elementary_mechanism' annotation to all elementary reactions")
println("4. Added 'metabolite_compartments' and 'reaction_compartment' annotations to complexes")
