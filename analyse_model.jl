using Pkg
using Distributed

# Activate and instantiate the environment on all workers
@everywhere begin
    import Pkg
    Pkg.activate("/work/schaffran1/COCOA.jl")
    Pkg.instantiate()
end
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2
@everywhere using COCOA, HiGHS
# --- Hardcoded paths ---
test_model_path = "/work/schaffran1/COCOA.jl/test/e_coli_core.xml"         # Change as needed
model_path = "/work/schaffran1/COCOA.jl/test/e_coli_core_splt_prpd.xml"           # Change as needed
output_path = "/work/schaffran1/concordance_results_e_coli_core_splt_prpd.jld2"    # Change as needed

# Load the model
test_model = COBREXA.load_model(test_model_path)
model = COBREXA.load_model(model_path)
res = concordance_analysis(
    test_model;
    optimizer=HiGHS.Optimizer,
    sample_size=10,                # Adjust as needed
    stage_size=1000,
    batch_size=1000,
    seed=42
)

# Run concordance analysis and capture timing
results = COCOA.concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    sample_size=100,                # Adjust as needed
    stage_size=1000,
    batch_size=1000,
    seed=42
)

# Save results and timing
JLD2.save(output_path, "results", results)

println("✓ Concordance analysis complete. Results and timing saved to $output_path")