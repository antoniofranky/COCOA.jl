using Pkg
using Distributed

# Debug: Check initial environment state
println("=== MAIN PROCESS DEBUG ===")
println("Julia version: $(VERSION)")
println("Active project: $(Pkg.project().path)")
println("JULIA_LOAD_PATH: $(get(ENV, "JULIA_LOAD_PATH", "not set"))")
println("JULIA_DEPOT_PATH: $(get(ENV, "JULIA_DEPOT_PATH", "not set"))")

# Activate project environment on all workers FIRST
@everywhere using Pkg
@everywhere Pkg.activate("/work/schaffran1/COCOA.jl")

# Debug worker environments too
@everywhere println("Worker $(myid()) - Project: $(Pkg.project().path)")
@everywhere println("Worker $(myid()) - Julia: $(VERSION)")

# Load packages on main process first
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2

# Then load on workers - this ensures consistent loading order
@everywhere using COCOA, HiGHS

# Debug: Check that ConstraintTrees loaded consistently
@everywhere try
    using ConstraintTrees
    println("Worker $(myid()) - ConstraintTrees loaded successfully")
    # Test the constructor that's failing
    test_val = ConstraintTrees.variable(; idx=1).value
    test_constraint = ConstraintTrees.Constraint(value=test_val)
    println("Worker $(myid()) - ConstraintTrees constructor works")
catch e
    println("Worker $(myid()) - ConstraintTrees ERROR: $e")
end
# --- Hardcoded paths ---
test_model_path = "/work/schaffran1/COCOA.jl/test/e_coli_core.xml"         # Change as needed
model_path = "/work/schaffran1/toolbox/models/Schizosaccharomyces_pombe.xml"           # Change as needed
output_path = "/work/schaffran1/concordance_results_schizo_split.jld2"    # Change as needed

# Load the model
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("random_seed", 42),
    COBREXA.set_optimizer_attribute("time_limit", 600.0),  # 10 minutes per optimization
    COBREXA.set_optimizer_attribute("presolve", "on"),
]
test_model = COBREXA.load_model(test_model_path)
model = COBREXA.load_model(model_path)
res = concordance_analysis(
    test_model;
    optimizer=HiGHS.Optimizer,
    sample_size=10,                # Adjust as needed
    batch_size=1000,
    seed=42,
    settings=highs_settings
)
model = COCOA.split_into_elementary_steps(model)
# Run concordance analysis and capture timing
results = COCOA.concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    settings=highs_settings,
    sample_size=100,                # Adjust as needed
    seed=42
)

# Save results and timing
JLD2.save(output_path, "results", results)

println("✓ Concordance analysis complete. Results and timing saved to $output_path")