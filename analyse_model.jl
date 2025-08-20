using Distributed
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2
@everywhere using COCOA, HiGHS

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
    sample_size=100,
    seed=42,
    settings=highs_settings
)
model = COCOA.split_into_elementary_steps(model)
model = COCOA.prepare_model_for_concordance(model, optimizer=HiGHS.Optimizer)
# Run concordance analysis and capture timing
results = COCOA.concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    settings=highs_settings,
    sample_size=100,                # Adjust as needed
    seed=42,
    coarse_cv_threshold=0.001,
    cv_threshold=0.0001,
)

# Save results and timing
JLD2.save(output_path, "results", results)

