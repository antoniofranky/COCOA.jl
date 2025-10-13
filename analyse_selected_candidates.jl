using Distributed
using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2, Dates, CSV, DataFrames
@everywhere using COCOA, HiGHS

# Parse command line arguments
if length(ARGS) < 2
    error("Usage: julia analyse_selected_candidates.jl <candidates_csv> <results_dir>")
end

candidates_csv = ARGS[1]
results_dir = ARGS[2]

# Optional: Get specific model index from array job
model_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing

# --- Analysis Parameters ---
sample_size = 1000
seed = 42
cv_threshold = 0.01
batch_size = 100_000
use_transitivity = true

# HiGHS solver settings
highs_settings = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-7),
    COBREXA.set_optimizer_attribute("random_seed", seed),
    COBREXA.set_optimizer_attribute("time_limit", 2400.0),  # 40 minutes per optimization
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

# Create results directory if it doesn't exist
mkpath(results_dir)

# Load candidates CSV
println("="^60)
println("COCOA Kinetic Concordance Analysis - Selected Candidates")
println("="^60)
println("Loading candidates from: $candidates_csv")

if !isfile(candidates_csv)
    error("Candidates CSV file not found: $candidates_csv")
end

candidates_df = CSV.read(candidates_csv, DataFrame)
println("Found $(nrow(candidates_df)) candidate species")

# Filter to specific model if index provided
if model_index !== nothing
    if model_index < 1 || model_index > nrow(candidates_df)
        error("Model index $model_index out of range (1-$(nrow(candidates_df)))")
    end
    candidates_df = candidates_df[model_index:model_index, :]
    println("Processing single model (index $model_index)")
end

# Process each candidate
for (idx, row) in enumerate(eachrow(candidates_df))
    model_id = row.old_species_id
    species_name = row.Species_name
    clade = row.Major_clade

    println("\n" * "="^60)
    println("[$idx/$(nrow(candidates_df))] Processing: $species_name")
    println("Clade: $clade")
    println("Model ID: $model_id")
    println("="^60)

    # Construct model file path
    # Adjust this path based on where your models are stored
    model_file = joinpath("/work", "schaffran1", "toolbox", "prpd_models", "random_90", "$(model_id).xml")

    if !isfile(model_file)
        println("WARNING: Model file not found: $model_file")
        println("Skipping $species_name")
        continue
    end

    # Construct output filename
    model_name = model_id
    output_filename = "kinetic_results_" * model_name * "_" *
                      lpad(string(seed), 2, "0") * "_" *
                      string(batch_size) * "_cv" *
                      replace(string(cv_threshold), "." => "p") * "_samples" *
                      string(sample_size) *
                      "_transitivity" * string(use_transitivity) * ".jld2"

    output_path = joinpath(results_dir, output_filename)

    # Check if already processed
    if isfile(output_path)
        println("Results already exist: $output_path")
        println("Skipping $species_name")
        continue
    end

    println("Output file: $output_path")
    println("Parameters:")
    println("  Sample size: $sample_size")
    println("  Seed: $seed")
    println("  CV threshold: $cv_threshold")
    println("  Batch size: $batch_size")
    println("  Use transitivity: $use_transitivity")

    try
        # Load the model
        println("\nLoading model: $model_file")
        model = COBREXA.load_model(model_file)

        # Get basic model info
        n_reactions = length(AbstractFBCModels.reactions(model))
        n_metabolites = length(AbstractFBCModels.metabolites(model))
        println("Model loaded successfully:")
        println("  Reactions: $n_reactions")
        println("  Metabolites: $n_metabolites")

        # Run kinetic concordance analysis
        println("\nStarting kinetic concordance analysis...")

        # Run analysis with timing
        analysis_timing = @timed begin
            results = COCOA.activity_concordance_analysis(
                model;
                optimizer=HiGHS.Optimizer,
                settings=highs_settings,
                sample_size=sample_size,
                seed=seed,
                cv_threshold=cv_threshold,
                batch_size=batch_size,
                use_transitivity=use_transitivity,
                kinetic_analysis=true
            )
        end

        # Extract timing information
        analysis_duration = analysis_timing.time
        gc_time = analysis_timing.gctime
        memory_allocated = analysis_timing.bytes

        println("Analysis completed in $(round(analysis_duration, digits=2)) seconds")
        println("Memory allocated: $(round(memory_allocated / 1e9, digits=2)) GB")
        println("GC time: $(round(gc_time, digits=2)) seconds")

        # Save results
        println("\nSaving results to: $output_path")

        JLD2.save(output_path,
            "results", results,
            "model_name", model_name,
            "model_id", model_id,
            "species_name", species_name,
            "major_clade", clade,
            "model_file", model_file,
            "analysis_parameters", Dict(
                "sample_size" => sample_size,
                "seed" => seed,
                "cv_threshold" => cv_threshold,
                "batch_size" => batch_size,
                "use_transitivity" => use_transitivity
            ),
            "timing_statistics", Dict(
                "analysis_duration_seconds" => analysis_duration,
                "memory_allocated_bytes" => memory_allocated,
                "memory_allocated_gb" => memory_allocated / 1e9,
                "gc_time_seconds" => gc_time,
                "gc_time_fraction" => gc_time / analysis_duration
            ),
            "timestamp", Dates.now())

        # Print summary
        println("\nKinetic Concordance Analysis Summary:")
        println("="^60)
        println("Analysis completed successfully!")
        println("Duration: $(round(analysis_duration/60, digits=2)) minutes")
        println("Memory allocated: $(round(memory_allocated / 1e9, digits=2)) GB")
        println("GC time: $(round(gc_time, digits=2))s ($(round(gc_time/analysis_duration*100, digits=1))%)")

        # Display kinetic and robustness results
        println("\nRobustness Results:")
        println("  Robust metabolites: $(length(results.acr_metabolites))")
        println("  Robust metabolite pairs: $(length(results.acrr_pairs))")
        println("  Largest robust module size: $(count(==(results.giant_id),results.kinetic_modules))")

        # Print general statistics from summary if available
        if results.summary !== nothing
            stats = results.summary
            if haskey(stats, "n_complexes")
                println("\nModel Statistics:")
                println("  Complexes: $(stats["n_complexes"])")
            end
            if haskey(stats, "n_balanced")
                println("  Balanced complexes: $(stats["n_balanced"])")
            end
            if haskey(stats, "n_modules")
                println("  Modules found: $(stats["n_modules"])")
            end
            if haskey(stats, "n_concordant_total")
                println("\nConcordance Results:")
                println("  Total concordant pairs: $(stats["n_concordant_total"])")
            end
            if haskey(stats, "n_kinetic_modules")
                println("\nKinetic Statistics:")
                println("  Kinetic modules: $(stats["n_kinetic_modules"])")
            end
        else
            println("\nNote: Summary statistics not available")
        end
        println("="^60)

    catch e
        println("\nERROR: Analysis failed for model $model_name ($species_name)")
        println("Error details: ", e)
        println("Stacktrace:")
        for line in stacktrace(catch_backtrace())
            println("  ", line)
        end

        # Save error information
        error_filename = "error_" * model_name * "_" * string(Int(round(time()))) * ".txt"
        error_path = joinpath(results_dir, error_filename)

        open(error_path, "w") do f
            println(f, "Error in analysis of model: $model_name")
            println(f, "Species: $species_name")
            println(f, "Clade: $clade")
            println(f, "Model file: $model_file")
            println(f, "Timestamp: $(Dates.now())")
            println(f, "Error: $e")
            println(f, "\nStacktrace:")
            for line in stacktrace(catch_backtrace())
                println(f, "  $line")
            end
        end

        println("Error details saved to: $error_path")
        # Continue with next model instead of failing completely
        continue
    end
end

println("\n" * "="^60)
println("All selected candidates processed!")
println("="^60)
