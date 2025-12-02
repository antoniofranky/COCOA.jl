using COCOA
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get constraints
constraints = COCOA.concordance_constraints(model, use_unidirectional_constraints=false)

# Extract from constraints only
Y_matrix, metabolite_ids, complex_ids = COCOA.complex_stoichiometry(constraints; return_ids=true)

println("=== Complex ordering from constraint extraction ===")
for (i, id) in enumerate(complex_ids)
    println("$i. $id")
end
println()

# Show Y matrix to understand structure
println("=== Y Matrix Structure ===")
println("Metabolites: ", metabolite_ids)
println("Complexes: ", complex_ids)
println("Matrix size: ", size(Y_matrix))
println()

# Check which complexes should be concordant based on Y matrix
println("=== Checking for potential ACRR pairs (qq conditions) ===")
for i in 1:(length(complex_ids)-1)
    for j in (i+1):length(complex_ids)
        vec1 = Y_matrix[:, i]
        vec2 = Y_matrix[:, j]

        diff_mask = vec1 .!= vec2
        differing_indices = findall(diff_mask)
        n_differences = length(differing_indices)

        if n_differences == 2
            met_idx1, met_idx2 = differing_indices

            # qq conditions
            non_zero_vec1 = findall(!iszero, vec1)
            non_zero_vec2 = findall(!iszero, vec2)
            qq1 = true  # already confirmed (2 differences)
            qq2 = length(setdiff(non_zero_vec1, differing_indices)) == 1
            qq3 = length(setdiff(non_zero_vec2, differing_indices)) == 1

            if qq1 && qq2 && qq3
                met_name1 = metabolite_ids[met_idx1]
                met_name2 = metabolite_ids[met_idx2]
                println("✓ ACRR pair: $(complex_ids[i]) vs $(complex_ids[j])")
                println("  Metabolites: $met_name1 ↔ $met_name2")
                println("  Catalyst: ", [metabolite_ids[k] for k in setdiff(non_zero_vec1, differing_indices)])
                println()
            end
        end
    end
end
