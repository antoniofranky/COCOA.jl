using Distributed
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2
@everywhere using COCOA, HiGHS

# --- Hardcoded paths ---
model_path = "/work/schaffran1/COCOA.jl/test/E_coli_K12_iJO1366.xml"           # Change as needed
# Parameters for activity_concordance_analysis
sample_size = 1000
seed = 42
cv_threshold = 0.01
batch_size = 500_000
use_transitivity = true
# Construct output path based on parameters
output_path =
    "/work/schaffran1/results_testjobs/E_coli_K12_iJO1366_result_reversible" *
    lpad(string(seed), 2, "0") * "_" *
    string(batch_size) * "_cv" *
    replace(string(cv_threshold), "." => "p") * "_samples" *
    string(sample_size) *
    "_transitivity" * string(use_transitivity) * ".jld2"

# Load the model
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("random_seed", seed),
    COBREXA.set_optimizer_attribute("time_limit", 1200.0),  # 20 minutes per optimization
    COBREXA.set_optimizer_attribute("presolve", "on"),
]
model = COBREXA.load_model(model_path)
model = convert(AbstractFBCModels.CanonicalModel.Model, model)
model = COCOA.remove_blocked_reactions(model, optimizer=HiGHS.Optimizer, settings=highs_settings)
model = COCOA.remove_orphans(model)
model = COCOA.split_into_irreversible(model)
model = convert(SBMLFBCModels.SBMLFBCModel, model)
results = COCOA.activity_concordance_analysis(
    model;
    optimizer=HiGHS.Optimizer,
    settings=highs_settings,
    sample_size=sample_size,
    seed=seed,
    cv_threshold=cv_threshold,
    concordance_tolerance=0.01,
    batch_size=batch_size,
    use_transitivity=use_transitivity,
    balanced_threshold=1e-9,
    kinetic_analysis=false,
)

# Save results and timing
JLD2.save(output_path, "results", results)
