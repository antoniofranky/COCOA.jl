using COCOA
include("envz_ompr_model.jl")
model = create_envz_ompr_model()

# Get the incidence matrix to see which reactions each complex participates in
A_matrix, complex_ids, reaction_ids = COCOA.incidence(model; return_ids=true)

println("=== Complex Participation in Reactions ===")
for (i, complex_id) in enumerate(complex_ids)
    row = A_matrix[i, :]
    consumed_in = [reaction_ids[j] for j in 1:length(reaction_ids) if row[j] == -1]
    produced_in = [reaction_ids[j] for j in 1:length(reaction_ids) if row[j] == 1]

    println("$complex_id:")
    if !isempty(consumed_in)
        println("  Consumed in: ", join(consumed_in, ", "))
    end
    if !isempty(produced_in)
        println("  Produced in: ", join(produced_in, ", "))
    end

    # Check if balanced
    if length(consumed_in) == length(produced_in) && length(consumed_in) > 0
        println("  → Potentially balanced")
    end
    println()
end
