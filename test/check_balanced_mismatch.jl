using COCOA, HiGHS
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get the correct balanced complexes
constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)
Y_matrix, metabolite_ids, complex_ids = COCOA.complex_stoichiometry(constraints; return_ids=true, model=model)

initial_balanced_ids = COCOA.find_trivially_balanced_complexes_sparse(Y_matrix, metabolite_ids, complex_ids)

println("=== Expected Balanced Complexes (from Y matrix analysis) ===")
println(initial_balanced_ids)
println()

# Run full analysis
results = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, kinetic_analysis=false, use_unidirectional_constraints=false)

println("=== Results: Concordance Modules ===")
for (i, id) in enumerate(results.complex_ids)
    module_id = results.concordance_modules[i]
    is_expected_balanced = id in initial_balanced_ids
    marker = if module_id == 0 && is_expected_balanced
        "✓ CORRECT (balanced)"
    elseif module_id == 0 && !is_expected_balanced
        "✗ WRONG (not actually balanced)"
    elseif module_id != 0 && is_expected_balanced
        "✗ WRONG (should be balanced)"
    else
        "  (unbalanced)"
    end
    println("$marker Index $i: $id → module $module_id")
end
