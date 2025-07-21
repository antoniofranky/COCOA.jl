#!/usr/bin/env julia

# Activate project environment
import Pkg
# Simply go up one directory level from src/ to reach the project root
project_path = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(project_path)

using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

# Load necessary packages
using Distributed
using JLD2
using DataFrames

# Check if we should add workers
if nworkers() == 1
    # No workers added via -p flag, try to add based on available CPUs
    n_cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))

    if n_cpus > 1
        # Decide on worker/thread split based on total CPUs
        if n_cpus <= 8
            n_workers = n_cpus
            n_threads = 1
        elseif n_cpus <= 32
            n_workers = div(n_cpus, 2)
            n_threads = 2
        else
            n_workers = div(n_cpus, 4)
            n_threads = 4
        end

        println("=== Auto-configuring workers ===")
        println("Available CPUs: $n_cpus")
        println("Adding $(n_workers-1) workers with $n_threads threads each")

        addprocs(n_workers - 1; exeflags=`--threads=$n_threads`)
    else
        println("⚠️ Running with single process (no parallelism)")
    end
else
    println("✓ Running with $(nworkers()) pre-configured workers")
end

# Load packages on all workers (if any exist)
@everywhere begin
    # Simply use the parent directory of the script location
    project_path = abspath(joinpath(@__DIR__, ".."))
    import Pkg
    Pkg.activate(project_path)

    using Logging
    global_logger(ConsoleLogger(stderr, Logging.Info))

    using COCOA
    using COBREXA
    using HiGHS
    using GLPK
    using Gurobi
    println("Worker $(myid()) ready on $(gethostname()) with $(Threads.nthreads()) threads")
end

# Parse command line arguments with full support for all COCOA commands
using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description="COCOA - Concordance analysis for metabolic networks",
        version="1.0.0",
        add_version=true
    )

    @add_arg_table! s begin
        "command"
        help = "Command: split, prepare, concordance, or split-and-analyze"
        required = true
        "input"
        help = "Input model file (.xml, .json, .mat, .jld2)"
        required = true
        "output"
        help = "Output file/directory"
        required = true
        "--optimizer", "-o"
        help = "Optimizer to use: HiGHS, GLPK, Gurobi"
        default = "HiGHS"
        "--mechanism", "-m"
        help = "Enzyme mechanism for splitting: fixed or random"
        arg_type = Symbol
        default = :fixed
        "--max-substrates"
        help = "Maximum substrates per reaction"
        arg_type = Int
        default = 4
        "--max-products"
        help = "Maximum products per reaction"
        arg_type = Int
        default = 4
        "--seed"
        help = "Random seed for reproducibility"
        arg_type = Int
        default = 42
        "--split-fraction"
        help = "Fraction of reactions to split (0.0-1.0)"
        arg_type = Float64
        default = 1.0
        "--split-elementary"
        help = "Split into elementary steps if not already done"
        action = :store_true
        "--remove-blocked"
        help = "Remove blocked reactions"
        action = :store_true
        "--threshold"
        help = "Threshold for blocked reactions"
        arg_type = Float64
        default = 1e-9
        "--default-lb"
        help = "Default lower bound for reactions"
        arg_type = Float64
        default = -1000.0
        "--default-ub"
        help = "Default upper bound for reactions"
        arg_type = Float64
        default = 1000.0
        "--sample-size", "-s"
        help = "Sample size for concordance analysis"
        arg_type = Int
        default = 1000
        "--correlation-threshold", "-c"
        help = "Correlation threshold for concordance"
        arg_type = Float64
        default = 0.95
        "--stage-size"
        help = "Stage size for concordance analysis"
        arg_type = Int
        default = 500
        "--batch-size"
        help = "Batch size within each stage (for memory management)"
        arg_type = Int
        default = 100
        "--early-correlation-threshold"
        help = "Early correlation threshold for filtering"
        arg_type = Float64
        default = 0.95
    end

    return parse_args(s)
end

# Helper function to load models with better error handling
function load_model_smart(filepath::String)
    println("Loading model from: $filepath")

    try
        if endswith(filepath, ".jld2")
            jld_data = JLD2.load(filepath)

            # Try to find model in JLD2 file using common keys
            for key in ["model", "prepared_model", "split_model"]
                if haskey(jld_data, key)
                    println("✓ Found model with key '$key'")
                    return jld_data[key]
                end
            end

            # If no standard keys found, use the first object
            first_key = first(keys(jld_data))
            println("⚠ No standard model key found, using '$first_key'")
            return jld_data[first_key]
        else
            # Use COBREXA for standard formats
            return COBREXA.load_model(filepath)
        end
    catch e
        println("❌ Error loading model: $(sprint(showerror, e))")
        rethrow(e)
    end
end

# Save model with better error handling
function save_model_smart(model, filepath::String; key="model")
    try
        if endswith(filepath, ".jld2")
            JLD2.save(filepath, key, model)
        else
            COBREXA.save_model(model, filepath)
        end
        println("✓ Model saved to: $filepath")
    catch e
        println("❌ Error saving model: $(sprint(showerror, e))")
        rethrow(e)
    end
end

# Get optimizer from string name
function get_optimizer(name::String)
    optimizers = Dict(
        "HiGHS" => HiGHS.Optimizer,
        "GLPK" => GLPK.Optimizer,
        "Gurobi" => Gurobi.Optimizer
    )

    return get(optimizers, name, HiGHS.Optimizer)
end

# Save results for concordance analysis
function save_results(results, output_dir)
    mkpath(output_dir)

    # Save complete results as JLD2
    jld2_path = joinpath(output_dir, "concordance_results.jld2")
    JLD2.save(jld2_path, "results", results)
    println("  ✓ Complete results: $(basename(jld2_path))")
end

# Print model summary
function print_summary(model)
    try
        n_rxns = length(AbstractFBCModels.reactions(model))
        n_mets = length(AbstractFBCModels.metabolites(model))
        println("  Model contains $n_rxns reactions and $n_mets metabolites")
    catch
        println("  Model loaded successfully")
    end
end

# Main function
function main()
    args = parse_commandline()
    command = args["command"]

    if command == "split"
        println("\n=== Splitting Reactions into Elementary Steps ===")

        # Load model
        model = load_model_smart(args["input"])
        print_summary(model)

        # Split into elementary steps
        println("\nSplitting reactions...")
        println("  Mechanism: $(args["mechanism"])")
        println("  Max substrates: $(args["max-substrates"])")
        println("  Max products: $(args["max-products"])")
        println("  Random seed: $(args["seed"])")

        split_model = COCOA.split_into_elementary_steps(
            model;
            ordered_fraction=args["mechanism"] == :fixed ? 1.0 : 0.0,
            max_substrates=args["max-substrates"],
            max_products=args["max-products"],
            seed=args["seed"]
        )

        # Save result
        println("\nSaving split model to: $(args["output"])")
        save_model_smart(split_model, args["output"]; key="split_model")
        print_summary(split_model)
        println("✓ Done!")

    elseif command == "prepare"
        println("\n=== Preparing Model for Concordance Analysis ===")

        # Load model
        model = load_model_smart(args["input"])
        print_summary(model)

        # Prepare for concordance
        println("\nPreparing model...")
        options = Dict{Symbol,Any}(
            :split_elementary => args["split-elementary"],
            :mechanism => args["mechanism"],
            :remove_blocked => args["remove-blocked"],
            :threshold => args["threshold"],
            :default_lb => args["default-lb"],
            :default_ub => args["default-ub"],
            :workers => workers()
        )

        # Print selected options
        for (key, value) in options
            if value !== nothing && value !== false
                println("  $key: $value")
            end
        end

        prepared_model = COCOA.prepare_model_for_concordance(
            model;
            optimizer=get_optimizer(args["optimizer"]),
            workers=workers(),
            flux_tolerance=args["threshold"]
        )

        # Save result
        println("\nSaving prepared model to: $(args["output"])")
        save_model_smart(prepared_model, args["output"]; key="prepared_model")
        print_summary(prepared_model)
        println("✓ Done!")

    elseif command == "concordance"
        println("\n=== Running Concordance Analysis ===")

        # Load model
        model = load_model_smart(args["input"])
        print_summary(model)

        # Run concordance analysis
        println("\nAnalysis parameters:")
        println("  Workers: $(nworkers())")
        println("  Optimizer: $(args["optimizer"])")
        println("  Stage size: $(args["stage-size"])")
        println("  Batch size: $(args["batch-size"])")
        println("  Sample size: $(args["sample-size"])")
        println("  Correlation threshold: $(args["correlation-threshold"])")
        println("  Early correlation threshold: $(args["early-correlation-threshold"])")
        println("  Tolerance: $(args["threshold"])")
        println("  Random seed: $(args["seed"])")

        println("\nRunning concordance analysis...")
        results = COCOA.concordance_analysis(
            model;
            optimizer=get_optimizer(args["optimizer"]),
            workers=workers(),
            sample_size=args["sample-size"],
            correlation_threshold=args["correlation-threshold"],
            early_correlation_threshold=args["early-correlation-threshold"],
            stage_size=args["stage-size"],
            batch_size=args["batch-size"],
            seed=args["seed"],
            tolerance=args["threshold"]
        )

        # Save results
        output_dir = args["output"]
        println("\nSaving results to: $output_dir/")
        save_results(results, output_dir)

        # Print summary
        println("\nAnalysis Summary:")
        println("  Total complexes: $(nrow(results.complexes))")
        println("  Kinetic modules: $(nrow(results.modules))")
        println("  Processing time: $(round(results.stats["elapsed_time"] / 60, digits=2)) minutes")
        println("\n✓ Analysis complete!")

    elseif command == "split-and-analyze"
        println("\n=== Split and Analyze Workflow ===")

        # Load original model
        println("Loading original model...")
        original_model = load_model_smart(args["input"])
        print_summary(original_model)

        # Step 1: Split into elementary steps
        println("\nStep 1: Splitting reactions into elementary steps...")
        println("  Ordered fraction: $(args["split-fraction"])")
        println("  Max substrates: $(args["max-substrates"])")
        println("  Max products: $(args["max-products"])")
        println("  Random seed: $(args["seed"])")

        println("\nTiming split step:")
        split_time = @time split_model = COCOA.split_into_elementary_steps(
            original_model;
            ordered_fraction=args["split-fraction"],
            max_substrates=args["max-substrates"],
            max_products=args["max-products"],
            seed=args["seed"]
        )
        print_summary(split_model)

        # Step 2: Prepare model for concordance
        println("\nStep 2: Preparing split model for concordance analysis...")
        println("Timing preparation step:")
        prep_time = @time prepared_model = COCOA.prepare_model_for_concordance(
            split_model;
            optimizer=get_optimizer(args["optimizer"]),
            workers=workers(),
            flux_tolerance=args["threshold"]
        )
        print_summary(prepared_model)

        # Step 3: Save prepared model with parameterized filename
        output_dir = args["output"]
        mkpath(output_dir)
        model_filename = joinpath(output_dir, "prepared_model_split$(args["split-fraction"])_subs$(args["max-substrates"])_prods$(args["max-products"]).jld2")
        save_model_smart(prepared_model, model_filename; key="prepared_model")
        println("✓ Prepared model saved to: $(model_filename)")

        # Step 4: Run concordance analysis
        println("\nStep 4: Running concordance analysis...")
        println("Timing concordance analysis:")
        concordance_time = @time results = COCOA.concordance_analysis(
            prepared_model;
            optimizer=get_optimizer(args["optimizer"]),
            workers=workers(),
            sample_size=args["sample-size"],
            correlation_threshold=args["correlation-threshold"],
            early_correlation_threshold=args["early-correlation-threshold"],
            stage_size=args["stage-size"],
            batch_size=args["batch-size"],
            seed=args["seed"],
            tolerance=args["threshold"]
        )

        # Step 5: Save concordance results as JLD2
        results_filename = joinpath(output_dir, "concordance_results_split$(args["split-fraction"])_subs$(args["max-substrates"])_prods$(args["max-products"]).jld2")
        JLD2.save(results_filename, "results", results)
        println("✓ Concordance results saved to: $(results_filename)")

        # Step 6: Save comprehensive model and performance statistics
        model_stats = Dict(
            "original_reactions" => length(AbstractFBCModels.reactions(original_model)),
            "original_metabolites" => length(AbstractFBCModels.metabolites(original_model)),
            "split_reactions" => length(AbstractFBCModels.reactions(split_model)),
            "split_metabolites" => length(AbstractFBCModels.metabolites(split_model)),
            "prepared_reactions" => length(AbstractFBCModels.reactions(prepared_model)),
            "prepared_metabolites" => length(AbstractFBCModels.metabolites(prepared_model)),
            "split_fraction" => args["split-fraction"],
            "max_substrates" => args["max-substrates"],
            "max_products" => args["max-products"],
            "seed" => args["seed"],
            "split_time_seconds" => split_time,
            "preparation_time_seconds" => prep_time,
            "concordance_time_seconds" => concordance_time,
            "workers" => nworkers(),
            "optimizer" => args["optimizer"]
        )
        
        # Merge with concordance analysis stats
        merged_stats = merge(results.stats, model_stats)
        stats_file = joinpath(output_dir, "model_stats_split$(args["split-fraction"])_subs$(args["max-substrates"])_prods$(args["max-products"]).jld2")
        JLD2.save(stats_file, "stats", merged_stats)
        println("✓ Model statistics saved to: $(stats_file)")

        # Log comprehensive timing info for scaling analysis
        total_time = split_time + prep_time + concordance_time
        @info "Performance Analysis" split_fraction=args["split-fraction"] max_substrates=args["max-substrates"] max_products=args["max-products"] n_reactions=model_stats["prepared_reactions"] n_metabolites=model_stats["prepared_metabolites"] total_time_min=round(total_time/60, digits=2) concordance_time_min=round(concordance_time/60, digits=2) n_complexes=nrow(results.complexes) n_modules=nrow(results.modules)

        println("\nSplit-and-Analyze Summary:")
        println("  Original model: $(model_stats["original_reactions"]) reactions, $(model_stats["original_metabolites"]) metabolites")
        println("  Split model: $(model_stats["split_reactions"]) reactions, $(model_stats["split_metabolites"]) metabolites")
        println("  Prepared model: $(model_stats["prepared_reactions"]) reactions, $(model_stats["prepared_metabolites"]) metabolites")
        println("  Complexes: $(nrow(results.complexes))")
        println("  Kinetic modules: $(nrow(results.modules))")
        println("  Total processing time: $(round(total_time / 60, digits=2)) minutes")
        println("    - Splitting: $(round(split_time / 60, digits=2)) minutes")
        println("    - Preparation: $(round(prep_time / 60, digits=2)) minutes")
        println("    - Concordance: $(round(concordance_time / 60, digits=2)) minutes")
        println("\n✓ Split-and-analyze complete!")

    else
        println("❌ Unknown command: $command")
        println("Available commands: split, prepare, concordance, split-and-analyze")
        exit(1)
    end
end

# Run the main function
main()