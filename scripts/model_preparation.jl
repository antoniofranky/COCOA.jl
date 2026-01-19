
using Distributed
import AbstractFBCModels as M
using COBREXA, SBMLFBCModels
@everywhere using COCOA, HiGHS

# Define recommended models to process
RECOMMENDED_MODELS = [
    "Lipomyces_starkeyi",
    "Tortispora_caseinolytica",
    "Yarrowia_deformans",
    "Alloascoidea_hylecoeti",
    "Sporopachydermia_quercuum",
    "Pachysolen_tannophilus",
    "Komagataella_pastoris",
    "Debaryomyces_hansenii",
    "Saccharomycopsis_malanga",
    "Wickerhamomyces_ciferrii",
    "Hanseniaspora_vinae",
    "Torulaspora_delbrueckii",
    "Neurospora_crassa"
]

# Define seeds
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Get the array task ID (1-indexed)
task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# Map task_id to (model_idx, seed_idx)
N_MODELS = length(RECOMMENDED_MODELS)
N_SEEDS = length(SEEDS)

if task_id > N_MODELS * N_SEEDS
    println("Task ID $task_id exceeds number of tasks ($(N_MODELS * N_SEEDS)). Exiting.")
    exit(0)
end

task_idx = task_id - 1  # Convert to 0-indexed
model_idx = task_idx ÷ N_SEEDS + 1
seed_idx = task_idx % N_SEEDS + 1

model_name = RECOMMENDED_MODELS[model_idx]
seed = SEEDS[seed_idx]

println("Task $task_id: Processing model $model_name with seed $seed")
println("Using $(nprocs()) processes ($(nworkers()) workers)")

# Find the model file
model_files_list = readdir("/work/schaffran1/Yeast-Species-GEMs", join=true)
f_idx = findfirst(x -> occursin(model_name, x), model_files_list)
if f_idx === nothing
    println("ERROR: Model file not found for $model_name")
    exit(1)
end
f = model_files_list[f_idx]
println("Processing model: $(basename(f))")
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

    println("Base model preprocessed. Starting random splits...")

    # Now process each random fraction using the base model
    for rdm in collect(0.0:0.25:1.0)
        println("Processing random fraction: $rdm")

        # Copy base model for this random fraction
        model_canonical = deepcopy(base_model)

        # 5. Split into elementary steps (with random fraction parameter)
        model_canonical = split_into_elementary(
            model_canonical,
            random=rdm,
            seed=UInt(seed)
        )

        # 6. Split into irreversible reactions
        model_canonical = split_into_irreversible(model_canonical)

        # Convert to SBML format for saving
        model_sbml = convert(SBMLFBCModels.SBMLFBCModel, model_canonical)

        # Create output directory for this random fraction and seed
        dir_name = "random_$(Int(rdm * 100))"
        output_dir = "/work/schaffran1/toolbox/prpd_models/seed_$(seed)/$(dir_name)"
        mkpath(output_dir)  # Ensure directory exists

        # Save the preprocessed model
        output_path = joinpath(output_dir, "$(splitext(basename(f))[1]).xml")
        M.save(model_sbml, output_path)

        println("Successfully processed $(basename(f)) with random=$rdm")
    end
catch e
    println("Error processing $(basename(f)): $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end
