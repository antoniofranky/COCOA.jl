using COCOA
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get constraints
constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

println("=== Activity Constraints for Suspected Concordant Complexes ===")
for id in [Symbol("C+I"), Symbol("C+G"), Symbol("A+I"), Symbol("A+G")]
    if haskey(constraints.activities, id)
        activity = constraints.activities[id]
        println("\n$id:")
        println("  Value: ", activity.value)
        println("  Bound: ", activity.bound)
        println("  Variable indices: ", activity.value.idxs)
        println("  Coefficients: ", activity.value.weights)
    else
        println("\n$id: NOT FOUND IN ACTIVITIES!")
    end
end

# Check if these activities can be non-zero
println("\n\n=== Testing if activities can be non-zero ===")
using HiGHS
for id in [Symbol("C+I"), Symbol("C+G"), Symbol("A+I"), Symbol("A+G")]
    if haskey(constraints.activities, id)
        # Test if we can make this activity = 1
        test_constraints = constraints.balance
        test_constraints *= :test_activity^COBREXA.C.Constraint(
            constraints.activities[id].value, COBREXA.C.EqualTo(1.0)
        )

        result = COBREXA.optimized_values(
            test_constraints;
            objective=test_constraints.objective.value,
            optimizer=HiGHS.Optimizer,
            output=test_constraints.test_activity
        )

        println("$id = 1.0: ", isnothing(result) || (result isa Number && isnan(result)) ? "INFEASIBLE" : "FEASIBLE (value=$(result))")
    end
end