using COCOA
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get full Y matrix to check which complexes should be concordant
Y, mets, complexes = COCOA.complex_stoichiometry(model; return_ids=true)

println("=== Expected ACRR pairs (manual analysis from Y matrix) ===")
println("Looking for complex pairs that differ in exactly 2 metabolites")
println("and each has exactly 1 additional metabolite (qq conditions)")
println()

for i in 1:(length(complexes)-1)
    for j in (i+1):length(complexes)
        vec1 = Y[:, i]
        vec2 = Y[:, j]

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
                met_name1 = mets[met_idx1]
                met_name2 = mets[met_idx2]
                println("✓ ACRR pair found: ", complexes[i], " vs ", complexes[j])
                println("  Metabolites: ", met_name1, " ↔ ", met_name2)
                println("  Catalyst: ", [mets[k] for k in setdiff(non_zero_vec1, differing_indices)])
                println()
            end
        end
    end
end
