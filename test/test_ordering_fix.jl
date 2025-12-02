using COCOA
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

# Get canonical complex ordering
_, _, complex_ids = COCOA.complex_stoichiometry(constraints; return_ids=true, model=model)

println("=== Complex IDs (canonical order) ===")
for (i, id) in enumerate(complex_ids)
    println("$i. $id")
end
println()

# Check activities ordering
println("=== Activities Constraint Tree Order ===")
activities = constraints.activities
activity_keys = collect(keys(activities))
for (i, id) in enumerate(activity_keys)
    println("$i. $id")
end
println()

# Check if ordering matches
if complex_ids == activity_keys
    println("✓ Orderings match!")
else
    println("✗ MISMATCH! Activities constraint tree has different ordering")
    println()
    println("Differences:")
    for (i, (cid, aid)) in enumerate(zip(complex_ids, activity_keys))
        if cid != aid
            println("  Position $i: complex_ids=$cid, activities=$aid")
        end
    end
end
