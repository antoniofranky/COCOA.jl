using COBREXA, SBMLFBCModels, AbstractFBCModels
using COCOA, HiGHS
using JLD2, FileIO
using ConstraintTrees
import ConstraintTrees as C

# Get all model files
model_files = readdir("/work/schaffran1/toolbox/Yeast-Species-GEMs", join=true)

# Get the array task ID (1-indexed)
task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# Check if task_id is valid
if task_id > length(model_files)
    println("Task ID $task_id exceeds number of models ($(length(model_files))). Exiting.")
    exit(0)
end

# Process the specific model for this task
f = model_files[task_id]
println("Processing model $task_id/$(length(model_files)): $(basename(f))")

try
    # Load model
    model = load_model(f)

    # Create elementary step constraints with full preprocessing (COBREXA pattern)
    # This replaces: split_into_elementary_steps + prepare_model_for_concordance
    constraints, complexes = concordance_constraints(
        model;
        use_elementary_steps=true,
        remove_blocked=true,
        remove_orphaned=true,
        use_unidirectional_constraints=true,
        optimizer=HiGHS.Optimizer,
        return_complexes=true
    )

    # Save as native Julia objects following COBREXA recommendations
    base_name = splitext(basename(f))[1]
    output_path = "/work/schaffran1/toolbox/prpd_models/$(base_name)_constraints.jld2"

    save(output_path, Dict(
        "constraints" => constraints,
        "complexes" => complexes,
        "original_model_path" => f,
        "processing_timestamp" => now(),
        "preprocessing_config" => Dict(
            "remove_blocked" => true,
            "remove_orphaned" => true,
            "use_unidirectional_constraints" => true,
            "optimizer" => "HiGHS.Optimizer"
        )
    ))

    println("Successfully processed: $(basename(f)) -> $(basename(output_path))")

catch e
    println("Error processing $(basename(f)): $e")
    exit(1)
end