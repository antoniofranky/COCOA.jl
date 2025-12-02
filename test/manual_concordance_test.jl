using COCOA, HiGHS, ConstraintTrees
const C = ConstraintTrees
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get constraints
constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

# Test C+I vs C+G manually using Charnes-Cooper
println("=== Manual Concordance Test: C+I vs C+G ===\n")

# Test 1: Set C+I = 1, maximize C+G
println("Test 1: c2 = C+I = 1.0, maximize c1 = C+G")
test_constraints = deepcopy(constraints.charnes_cooper.positive)
test_constraints *= :c2_constraint^C.Constraint(
    constraints.activities[Symbol("C+I")].value, C.EqualTo(1.0)
)

using JuMP
om = JuMP.Model(HiGHS.Optimizer)
JuMP.set_silent(om)
JuMP.@variable(om, x[1:C.var_count(test_constraints)])

# Add constraints
for (_, constraint) in test_constraints
    if constraint isa C.Constraint
        val = C.substitute(constraint.value, x)
        if constraint.bound isa C.Between
            if constraint.bound.lower > -Inf
                JuMP.@constraint(om, val >= constraint.bound.lower)
            end
            if constraint.bound.upper < Inf
                JuMP.@constraint(om, val <= constraint.bound.upper)
            end
        elseif constraint.bound isa C.EqualTo
            JuMP.@constraint(om, val == constraint.bound.equal_to)
        end
    end
end

# Set objective: maximize C+G
obj_expr = C.substitute(constraints.activities[Symbol("C+G")].value, x)
JuMP.@objective(om, Max, obj_expr)
JuMP.optimize!(om)

if JuMP.termination_status(om) in (JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED)
    max_val = JuMP.objective_value(om)
    println("  Result: C+G = $max_val (when C+I = 1.0)")
    println("  Status: ", JuMP.termination_status(om))
else
    println("  Result: INFEASIBLE or ERROR")
    println("  Status: ", JuMP.termination_status(om))
end
