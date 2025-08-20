using COBREXA, SBMLFBCModels, AbstractFBCModels
using COCOA, HiGHS

# Get all model files
model_files = readdir("/work/schaffran1/Yeast-Species-GEMs", join=true)

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
    model = load_model(f)
    model_splt = split_into_elementary_steps(model, seed=42, normalize_bounds=true)
    model_splt_prpd = prepare_model_for_concordance(model_splt, optimizer=HiGHS.Optimizer, output_type=SBMLFBCModels.SBMLFBCModel)
    save_model(model_splt_prpd, "/work/schaffran1/toolbox/prpd_models/$(splitext(basename(f))[1]).xml")
    println("Successfully processed: $(basename(f))")
catch e
    println("Error processing $(basename(f)): $e")
    exit(1)
end
