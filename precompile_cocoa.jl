# precompile_cocoa.jl - runs during sysimage creation
using COCOA, COBREXA, HiGHS, JLD2, SBMLFBCModels, AbstractFBCModels

# Create a small test model to trigger compilation
test_model = load_model("test/e_coli_core.xml")

# Run analysis to compile all methods
results = COCOA.concordance_analysis(
    test_model;
    optimizer=HiGHS.Optimizer,
    sample_size=20
)