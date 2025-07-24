using Pkg
using Distributed

# Activate and instantiate the environment on all workers
@everywhere begin
    import Pkg
    Pkg.activate("/work/schaffran1/COCOA.jl")
    Pkg.instantiate()
end

using SBMLFBCModels, AbstractFBCModels, COBREXA, JLD2, DataFrames, JSON, CSV, Dates, Statistics
@everywhere using COCOA, HiGHS

# Configuration for parameter testing
const MODEL_PATH = "/work/schaffran1/COCOA.jl/test/e_coli_core_splt_prpd.xml"  # ~1000 complex model
const RESULTS_DIR = "/work/schaffran1/parameter_test_results"
const SEED = 42  # Fixed seed for reproducibility

# HiGHS solver settings (matching the updated analyse_model.jl)
const HIGHS_SETTINGS = [
    COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("mip_feasibility_tolerance", 1e-10),
    COBREXA.set_optimizer_attribute("random_seed", 42),
    COBREXA.set_optimizer_attribute("time_limit", 240.0),  # 4 minutes per optimization
    COBREXA.set_optimizer_attribute("presolve", "on"),
]

# Create results directory if it doesn't exist
mkpath(RESULTS_DIR)

"""
Parameter configurations to test, organized by impact on performance for large models.
Each configuration is a named tuple with parameters and expected performance characteristics.
"""
function get_parameter_configurations()
    return [
        # Baseline configuration (from current working setup)
        (
            name = "baseline",
            sample_size = 100,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Current working configuration"
        ),
        
        # Memory-optimized configurations for large models
        (
            name = "memory_conservative",
            sample_size = 50,
            stage_size = 500,
            batch_size = 250,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 50_000,
            max_pairs_in_memory = 50_000,
            description = "Conservative memory usage for large models"
        ),
        
        (
            name = "memory_aggressive",
            sample_size = 200,
            stage_size = 2000,
            batch_size = 2000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 200_000,
            max_pairs_in_memory = 200_000,
            description = "Aggressive memory usage for faster processing"
        ),
        
        # Sample size variations (major impact on accuracy vs speed)
        (
            name = "small_sample",
            sample_size = 25,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Small sample size for speed"
        ),
        
        (
            name = "large_sample",
            sample_size = 300,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Large sample size for accuracy"
        ),
        
        # Tolerance variations (impacts concordance detection sensitivity)
        (
            name = "tight_tolerance",
            sample_size = 100,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.005,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.005,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Tighter tolerance for higher precision"
        ),
        
        (
            name = "loose_tolerance",
            sample_size = 100,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.02,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.02,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Looser tolerance for faster processing"
        ),
        
        # CV threshold variations (impacts candidate pair filtering)
        (
            name = "strict_cv_filter",
            sample_size = 100,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.98,
            cv_threshold = 0.005,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Strict CV filtering to reduce candidate pairs"
        ),
        
        (
            name = "relaxed_cv_filter",
            sample_size = 100,
            stage_size = 1000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.90,
            cv_threshold = 0.02,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Relaxed CV filtering for more candidate pairs"
        ),
        
        # Stage and batch size optimization for large models
        (
            name = "small_stages",
            sample_size = 100,
            stage_size = 200,
            batch_size = 100,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Small stages for memory efficiency"
        ),
        
        (
            name = "large_stages",
            sample_size = 100,
            stage_size = 5000,
            batch_size = 1000,
            tolerance = 0.01,
            coarse_cv_threshold = 0.95,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 100_000,
            max_pairs_in_memory = 100_000,
            description = "Large stages for throughput optimization"
        ),
        
        # Optimized configuration for 50k+ models (extrapolated from smaller tests)
        (
            name = "large_model_optimized",
            sample_size = 75,
            stage_size = 800,
            batch_size = 400,
            tolerance = 0.015,
            coarse_cv_threshold = 0.96,
            cv_threshold = 0.01,
            use_unidirectional_constraints = true,
            chunk_size_filter = 150_000,
            max_pairs_in_memory = 75_000,
            description = "Optimized for 50k+ complex models"
        )
    ]
end

"""
Run concordance analysis with specific parameters and collect detailed metrics.
"""
function run_test_configuration(config, model)
    println("Starting configuration: $(config.name)")
    start_time = time()
    
    # Initialize timing and memory tracking
    gc_start = Base.gc_num()
    memory_start = Base.gc_bytes()
    
    try
        # Run concordance analysis
        results = concordance_analysis(
            model;
            optimizer = HiGHS.Optimizer,
            settings = HIGHS_SETTINGS,
            sample_size = config.sample_size,
            stage_size = config.stage_size,
            batch_size = config.batch_size,
            tolerance = config.tolerance,
            coarse_cv_threshold = config.coarse_cv_threshold,
            cv_threshold = config.cv_threshold,
            use_unidirectional_constraints = config.use_unidirectional_constraints,
            chunk_size_filter = config.chunk_size_filter,
            max_pairs_in_memory = config.max_pairs_in_memory,
            seed = SEED
        )
        
        elapsed_time = time() - start_time
        gc_end = Base.gc_num()
        memory_end = Base.gc_bytes()
        
        # Calculate memory and GC metrics
        memory_allocated = (memory_end - memory_start) / 1024^3  # GB
        total_gc_time = (gc_end.total_time - gc_start.total_time) / 1e9  # seconds
        
        # Extract key metrics from results
        test_results = Dict(
            "config_name" => config.name,
            "description" => config.description,
            "success" => true,
            "elapsed_time" => elapsed_time,
            "memory_allocated_gb" => memory_allocated,
            "gc_time_sec" => total_gc_time,
            "n_complexes" => results.stats["n_complexes"],
            "n_balanced" => results.stats["n_balanced"],
            "n_candidate_pairs" => results.stats["n_candidate_pairs"],
            "n_concordant_pairs" => results.stats["n_concordant_pairs"],
            "n_modules" => results.stats["n_modules"],
            "stages_completed" => results.stats["stages_completed"],
            "n_timeout_pairs" => results.stats["n_timeout_pairs"],
            "concordant_ratio" => results.stats["n_concordant_pairs"] / max(1, results.stats["n_candidate_pairs"]),
            "pairs_per_second" => results.stats["n_candidate_pairs"] / elapsed_time,
            "memory_per_pair_mb" => (memory_allocated * 1024) / max(1, results.stats["n_candidate_pairs"]),
            # Configuration parameters
            "sample_size" => config.sample_size,
            "stage_size" => config.stage_size,
            "batch_size" => config.batch_size,
            "tolerance" => config.tolerance,
            "coarse_cv_threshold" => config.coarse_cv_threshold,
            "cv_threshold" => config.cv_threshold,
            "use_unidirectional_constraints" => config.use_unidirectional_constraints,
            "chunk_size_filter" => config.chunk_size_filter,
            "max_pairs_in_memory" => config.max_pairs_in_memory
        )
        
        println("✓ Configuration $(config.name) completed in $(round(elapsed_time, digits=2))s")
        println("  - Candidate pairs: $(results.stats["n_candidate_pairs"])")
        println("  - Concordant pairs: $(results.stats["n_concordant_pairs"])")
        println("  - Modules found: $(results.stats["n_modules"])")
        println("  - Memory used: $(round(memory_allocated, digits=2)) GB")
        
        return test_results
        
    catch e
        elapsed_time = time() - start_time
        println("✗ Configuration $(config.name) failed after $(round(elapsed_time, digits=2))s: $e")
        
        return Dict(
            "config_name" => config.name,
            "description" => config.description,
            "success" => false,
            "elapsed_time" => elapsed_time,
            "error" => string(e),
            "sample_size" => config.sample_size,
            "stage_size" => config.stage_size,
            "batch_size" => config.batch_size,
            "tolerance" => config.tolerance,
            "coarse_cv_threshold" => config.coarse_cv_threshold,
            "cv_threshold" => config.cv_threshold,
            "use_unidirectional_constraints" => config.use_unidirectional_constraints,
            "chunk_size_filter" => config.chunk_size_filter,
            "max_pairs_in_memory" => config.max_pairs_in_memory
        )
    end
end

"""
Analyze results and provide recommendations for large model parameters.
"""
function analyze_results(all_results)
    println("\n" * "="^80)
    println("PARAMETER TESTING RESULTS ANALYSIS")
    println("="^80)
    
    successful_results = filter(r -> r["success"], all_results)
    failed_results = filter(r -> !r["success"], all_results)
    
    if isempty(successful_results)
        println("❌ All configurations failed!")
        return
    end
    
    println("\n📊 SUCCESS RATE")
    println("Successful configurations: $(length(successful_results))/$(length(all_results))")
    
    if !isempty(failed_results)
        println("\n❌ FAILED CONFIGURATIONS:")
        for result in failed_results
            println("  - $(result["config_name"]): $(result["error"])")
        end
    end
    
    # Performance analysis
    println("\n⚡ PERFORMANCE RANKING (by speed)")
    speed_ranking = sort(successful_results, by = r -> r["elapsed_time"])
    for (i, result) in enumerate(speed_ranking[1:min(5, end)])
        println("  $i. $(result["config_name"]): $(round(result["elapsed_time"], digits=1))s " *
                "($(round(result["pairs_per_second"], digits=1)) pairs/s)")
    end
    
    println("\n💾 MEMORY EFFICIENCY RANKING")
    memory_ranking = sort(successful_results, by = r -> r["memory_allocated_gb"])
    for (i, result) in enumerate(memory_ranking[1:min(5, end)])
        println("  $i. $(result["config_name"]): $(round(result["memory_allocated_gb"], digits=2)) GB " *
                "($(round(result["memory_per_pair_mb"], digits=3)) MB/pair)")
    end
    
    println("\n🎯 ACCURACY RANKING (by concordant pairs found)")
    accuracy_ranking = sort(successful_results, by = r -> r["n_concordant_pairs"], rev=true)
    for (i, result) in enumerate(accuracy_ranking[1:min(5, end)])
        println("  $i. $(result["config_name"]): $(result["n_concordant_pairs"]) concordant pairs " *
                "($(round(result["concordant_ratio"] * 100, digits=1))% of candidates)")
    end
    
    # Recommendations for large models (50k+ complexes)
    println("\n🚀 RECOMMENDATIONS FOR LARGE MODELS (50k+ complexes)")
    
    # Find best balance of speed and memory efficiency
    efficiency_scores = []
    for result in successful_results
        speed_score = 1.0 / result["elapsed_time"]  # Higher is better
        memory_score = 1.0 / result["memory_allocated_gb"]  # Higher is better
        accuracy_score = result["concordant_ratio"]  # Higher is better
        
        # Weighted combination (speed and memory more important for large models)
        combined_score = 0.4 * speed_score + 0.4 * memory_score + 0.2 * accuracy_score
        push!(efficiency_scores, (result, combined_score))
    end
    
    best_configs = sort(efficiency_scores, by = x -> x[2], rev=true)[1:min(3, end)]
    
    println("\nTop 3 recommended configurations for large models:")
    for (i, (result, score)) in enumerate(best_configs)
        println("\n$i. $(result["config_name"]) (efficiency score: $(round(score, digits=4)))")
        println("   Description: $(result["description"])")
        println("   Performance: $(round(result["elapsed_time"], digits=1))s, $(round(result["memory_allocated_gb"], digits=2)) GB")
        println("   Results: $(result["n_concordant_pairs"]) concordant pairs, $(result["n_modules"]) modules")
        println("   Parameters:")
        println("     - sample_size: $(result["sample_size"])")
        println("     - stage_size: $(result["stage_size"])")  
        println("     - batch_size: $(result["batch_size"])")
        println("     - tolerance: $(result["tolerance"])")
        println("     - cv_threshold: $(result["cv_threshold"])")
        println("     - chunk_size_filter: $(result["chunk_size_filter"])")
        println("     - max_pairs_in_memory: $(result["max_pairs_in_memory"])")
    end
    
    # Parameter sensitivity analysis
    println("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
    
    # Sample size impact
    sample_size_results = Dict()
    for result in successful_results
        ss = result["sample_size"]
        if !haskey(sample_size_results, ss)
            sample_size_results[ss] = []
        end
        push!(sample_size_results[ss], (result["elapsed_time"], result["n_concordant_pairs"]))
    end
    
    if length(sample_size_results) > 1
        println("\nSample size impact:")
        for (ss, times_pairs) in sort(collect(sample_size_results))
            avg_time = mean([tp[1] for tp in times_pairs])
            avg_pairs = mean([tp[2] for tp in times_pairs])
            println("  Sample size $ss: $(round(avg_time, digits=1))s avg, $(round(avg_pairs, digits=0)) concordant pairs avg")
        end
    end
    
    println("\n💡 SCALING RECOMMENDATIONS:")
    best_config = best_configs[1][1]
    
    # Extrapolate recommendations for 50k model
    model_size_factor = 50000 / best_config["n_complexes"]  # Scaling factor
    
    println("For a 50,000 complex model ($(round(model_size_factor, digits=1))x larger):")
    println("  - Recommended sample_size: $(round(Int, best_config["sample_size"] * sqrt(model_size_factor)))")
    println("  - Recommended stage_size: $(round(Int, best_config["stage_size"] * 0.8))  # Slightly smaller for memory")
    println("  - Recommended batch_size: $(round(Int, best_config["batch_size"] * 0.7))  # Smaller for memory")
    println("  - Keep tolerance: $(best_config["tolerance"])")
    println("  - Keep cv_threshold: $(best_config["cv_threshold"])")
    println("  - Estimated runtime: $(round(best_config["elapsed_time"] * model_size_factor^1.5 / 3600, digits=1)) hours")
    println("  - Estimated memory: $(round(best_config["memory_allocated_gb"] * model_size_factor^0.8, digits=1)) GB")
end

"""
Main function to run all parameter tests.
"""
function main()
    println("COCOA.jl Parameter Testing Suite")
    println("="^50)
    println("Model: $MODEL_PATH")
    println("Results directory: $RESULTS_DIR")
    println("Number of workers: $(length(workers()))")
    
    # Load model
    println("\nLoading model...")
    model = COBREXA.load_model(MODEL_PATH)
    println("Model loaded: $(length(AbstractFBCModels.reactions(model))) reactions")
    
    # Get test configurations
    configs = get_parameter_configurations()
    println("Testing $(length(configs)) parameter configurations...")
    
    # Run tests
    all_results = []
    start_time = time()
    
    for (i, config) in enumerate(configs)
        println("\n" * "-"^60)
        println("Test $i/$(length(configs)): $(config.name)")
        println("-"^60)
        
        result = run_test_configuration(config, model)
        push!(all_results, result)
        
        # Force garbage collection between tests
        GC.gc()
        
        # Save intermediate results
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        intermediate_file = joinpath(RESULTS_DIR, "intermediate_results_$(timestamp).json")
        open(intermediate_file, "w") do f
            JSON.print(f, all_results, 2)
        end
    end
    
    total_time = time() - start_time
    println("\n" * "="^80)
    println("ALL TESTS COMPLETED in $(round(total_time/60, digits=1)) minutes")
    println("="^80)
    
    # Analyze results
    analyze_results(all_results)
    
    # Save final results
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Save as JSON
    json_file = joinpath(RESULTS_DIR, "parameter_test_results_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, all_results, 2)
    end
    
    # Save as CSV for easy analysis
    csv_file = joinpath(RESULTS_DIR, "parameter_test_results_$(timestamp).csv")
    df = DataFrame(all_results)
    CSV.write(csv_file, df)
    
    println("\n📁 Results saved to:")
    println("  - JSON: $json_file")
    println("  - CSV: $csv_file")
    
    return all_results
end

# Run the parameter testing suite
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end