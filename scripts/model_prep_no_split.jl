
import AbstractFBCModels as M
using COBREXA, SBMLFBCModels
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
println("Processing model array task $task_id/$(length(model_files))...")
println("Preprocessing pipeline: remove_orphans -> normalize_bounds -> remove_blocked_reactions -> remove_orphans -> split_into_elementary -> split_into_irreversible")

try
    # Do common preprocessing once (steps 1-4)
    println("Loading and preprocessing base model...")
    base_model = convert(M.CanonicalModel.Model, M.load(f))
    
    # 1. Remove orphans (unused metabolites/reactions)
    base_model = remove_orphans(base_model)
    
    # 2. Normalize bounds
    base_model = normalize_bounds(base_model)
    
    # 3. Remove blocked reactions (requires optimizer)
    base_model = remove_blocked_reactions(
        base_model,
        optimizer=HiGHS.Optimizer
    )
    
    # 4. Remove orphans again (from blocked reaction removal)
    base_model = remove_orphans(base_model)
    
    
    
    # Copy base model for this random fraction
    model_canonical = base_model
    
    # 5. Split into irreversible reactions
    model_canonical = split_into_irreversible(model_canonical)
    
    # Convert to SBML format for saving
    model_sbml = convert(SBMLFBCModels.SBMLFBCModel, model_canonical)
    
    # Create output directory for this random fraction
    dir_name = "no_split"
    output_dir = "/work/schaffran1/toolbox/prpd_models/$(dir_name)"
    mkpath(output_dir)  # Ensure directory exists
    
    # Save the preprocessed model
    output_path = joinpath(output_dir, "$(splitext(basename(f))[1]).xml")
    M.save(model_sbml, output_path)
    
    println("Successfully processed $(basename(f))")
    
catch e
    println("Error processing $(basename(f)): $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end
