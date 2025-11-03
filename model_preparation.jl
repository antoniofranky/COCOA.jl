
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
println("Processing model $task_id/$(length(model_files)): $(basename(f))")

try
    for rdm in collect(0.0:0.1:0.9)
        println("Processing random fraction: $rdm")

        model_canonical = convert(M.CanonicalModel.Model, M.load("toolbox\\prpd_models\\random_0\\Alloascoidea_hylecoeti.xml"))

        # Preprocessing pipeline:
        # 1. Remove orphans (unused metabolites/reactions)
        model_canonical = remove_orphans(model_canonical)

        # 2. Normalize bounds
        model_canonical = normalize_bounds(model_canonical)

        # 3. Remove blocked reactions (requires optimizer)
        model_canonical, blocked = remove_blocked_reactions(
            model_canonical,
            optimizer=HiGHS.Optimizer
        )

        # 4. Remove orphans again (from blocked reaction removal)
        model_canonical = remove_orphans(model_canonical)

        # 5. Split into elementary steps (with random fraction parameter)
        model_canonical = split_into_elementary(
            model_canonical,
            random=rdm,
            seed=42
        )

        # 6. Split into irreversible reactions
        model_canonical = split_into_irreversible(model_canonical)

        # Convert to SBML format for saving
        model_sbml = convert(SBMLFBCModels.SBMLFBCModel, model_canonical)

        # Create output directory for this random fraction
        dir_name = "random_$(Int(rdm * 100))"
        output_dir = "/work/schaffran1/toolbox/prpd_models/$(dir_name)"
        mkpath(output_dir)  # Ensure directory exists

        # Save the preprocessed model
        output_path = joinpath(output_dir, "$(splitext(basename(f))[1]).xml")
        M.save(output_path, model_sbml)

        println("Successfully processed $(basename(f)) with random=$rdm")
    end
catch e
    println("Error processing $(basename(f)): $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end
