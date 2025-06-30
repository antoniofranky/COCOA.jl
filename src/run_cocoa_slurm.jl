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
        help = "Command: split, prepare, or concordance"
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
        "--batch-size", "-b"
        help = "Batch size for concordance analysis"
        arg_type = Int
        default = 10
        "--stage-size"
        help = "Stage size for concordance analysis"
        arg_type = Int
        default = 100
        "--max-iterations"
        help = "Maximum iterations for concordance analysis"
        arg_type = Int
        default = 100
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
        println("  Batch size: $(args["batch-size"])")
        println("  Stage size: $(args["stage-size"])")
        println("  Sample size: $(args["sample-size"])")
        println("  Correlation threshold: $(args["correlation-threshold"])")
        println("  Max iterations: $(args["max-iterations"])")
        println("  Random seed: $(args["seed"])")

        println("\nRunning concordance analysis...")
        config = COCOA.ConcordanceConfig(
            sample_size=args["sample-size"],
            correlation_threshold=args["correlation-threshold"],
            concordance_batch_size=args["batch-size"],
            stage_size=args["stage-size"],
            seed=args["seed"]
        )

        results = COCOA.concordance_analysis(
            model;
            optimizer=get_optimizer(args["optimizer"]),
            workers=workers(),
            config=config
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

    else
        println("❌ Unknown command: $command")
        println("Available commands: split, prepare, concordance")
        exit(1)
    end
end

# Run the main function
main()