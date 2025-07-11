#!/usr/bin/env julia

"""
Example: COCOA Parameter Optimization for Large Models

This example demonstrates how to optimize parameters for large metabolic models
on HPC clusters using different optimization strategies.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Load required modules
include("../src/parameter_optimization.jl")
include("../src/bayesian_optimization.jl")
include("../src/COCOA.jl")
using .COCOA
using AbstractFBCModels
using Printf

function main()
    println("🧬 COCOA Parameter Optimization Example")
    println("=" ^ 50)
    
    # Example model path (adjust for your setup)
    model_path = "../test/e_coli_core.xml"
    
    if !isfile(model_path)
        println("⚠️  Model file not found: $model_path")
        println("Please adjust the model_path variable in this script.")
        return
    end
    
    # HPC cluster specifications (adjust for your cluster)
    available_memory_gb = 128.0  # 128 GB RAM
    max_workers = 64            # 64 CPU cores
    target_runtime_hours = 8.0   # 8 hour job limit
    
    println("🖥️  HPC Cluster Specifications:")
    println("   - Available memory: $(available_memory_gb) GB")
    println("   - Max workers: $(max_workers)")
    println("   - Target runtime: $(target_runtime_hours) hours")
    println()
    
    # Method 1: Quick Parameter Estimation
    println("📈 Method 1: Quick Parameter Estimation")
    println("-" ^ 40)
    
    try
        estimation_result = estimate_optimal_parameters(
            AbstractFBCModels.load_model(model_path);
            available_memory_gb = available_memory_gb,
            max_workers = max_workers,
            target_runtime_hours = target_runtime_hours,
            fast_estimation = true  # Use fast estimation for example
        )
        
        println("✅ Estimation complete!")
        println("   - Recommended workers: $(estimation_result.workers)")
        println("   - Sample size: $(estimation_result.sample_size)")
        println("   - Stage size: $(estimation_result.stage_size)")
        println("   - Correlation threshold: $(@sprintf("%.3f", estimation_result.correlation_threshold))")
        println("   - Estimated runtime: $(@sprintf("%.1f", estimation_result.estimated_runtime_hours)) hours")
        println("   - Memory requirement: $(estimation_result.recommended_memory_gb) GB")
        println("   - Recommendation: $(estimation_result.recommendation)")
        
    catch e
        println("❌ Estimation failed: $e")
    end
    
    println()
    
    # Method 2: Bayesian Optimization (simplified example)
    println("🧠 Method 2: Bayesian Optimization")
    println("-" ^ 40)
    
    try
        bayesian_result = recommend_parameters_bayesian(
            model_path;
            available_memory_gb = available_memory_gb,
            max_workers = max_workers,
            n_iterations = 10,  # Reduced for example
            seed = 42
        )
        
        println("✅ Bayesian optimization complete!")
        params = bayesian_result.parameters
        println("   - Optimal workers: $(params.workers)")
        println("   - Sample size: $(params.sample_size)")
        println("   - Stage size: $(params.stage_size)")
        println("   - Correlation threshold: $(@sprintf("%.3f", params.correlation_threshold))")
        println("   - Early correlation threshold: $(@sprintf("%.3f", params.early_correlation_threshold))")
        println("   - Optimization score: $(@sprintf("%.3f", bayesian_result.optimization_result.best_score))")
        
    catch e
        println("❌ Bayesian optimization failed: $e")
    end
    
    println()
    
    # Method 3: Parameter Sweep Configuration
    println("🔄 Method 3: Parameter Sweep")
    println("-" ^ 40)
    
    try
        sweep_configs = parameter_sweep_config(
            model_path,
            available_memory_gb,
            max_workers;
            sweep_type = :quick
        )
        
        println("✅ Parameter sweep configurations generated!")
        println("   - Number of configurations: $(length(sweep_configs))")
        println("   - Configurations:")
        
        for (i, config) in enumerate(sweep_configs)
            println("     Config $i: threshold=$(config.correlation_threshold), " * 
                   "sample_size=$(config.sample_size), stage_size=$(config.stage_size)")
        end
        
    catch e
        println("❌ Parameter sweep failed: $e")
    end
    
    println()
    
    # Method 4: Memory Usage Estimation
    println("💾 Method 4: Memory Usage Analysis")
    println("-" ^ 40)
    
    try
        # Estimate model size
        model = AbstractFBCModels.load_model(model_path)
        n_reactions = length(AbstractFBCModels.reactions(model))
        n_complexes_estimate = n_reactions * 2  # Quick estimate
        
        memory_usage = estimate_memory_usage(n_complexes_estimate, n_reactions)
        
        println("✅ Memory analysis complete!")
        println("   - Estimated complexes: $(n_complexes_estimate)")
        println("   - Reactions: $(n_reactions)")
        println("   - Sparse matrix: $(@sprintf("%.2f", memory_usage["sparse_matrix_GB"])) GB")
        println("   - Correlation overhead: $(@sprintf("%.2f", memory_usage["correlation_overhead_GB"])) GB")
        println("   - Tracker overhead: $(@sprintf("%.2f", memory_usage["tracker_GB"])) GB")
        println("   - Total memory: $(@sprintf("%.2f", memory_usage["total_GB"])) GB")
        println("   - With sharing: $(@sprintf("%.2f", memory_usage["total_with_sharing_GB"])) GB")
        
    catch e
        println("❌ Memory analysis failed: $e")
    end
    
    println()
    
    # Best Practices Summary
    println("💡 Best Practices for Large Models")
    println("-" ^ 40)
    println("1. 🎯 Use correlation_threshold ≥ 0.95 for models with >10K complexes")
    println("2. 🚀 Start with sample_size = 500-1000 for initial runs")
    println("3. ⚡ Use stage_size = 1000-2000 for optimal batching")
    println("4. 💾 Ensure 1.5x memory headroom for safety")
    println("5. 🔄 Use Bayesian optimization for fine-tuning")
    println("6. 🧪 Test with parameter sweeps for robustness")
    println("7. 📊 Monitor memory usage during initial runs")
    println("8. ⏱️  Set realistic time limits (6-12 hours for large models)")
    
    println()
    println("🎉 Parameter optimization example complete!")
    println()
    println("📋 Next Steps:")
    println("1. Run: julia optimize_parameters.jl estimate YOUR_MODEL.xml")
    println("2. Generate SLURM script with --generate-slurm flag")
    println("3. Submit job: sbatch run_cocoa_optimized.sh")
    println("4. Fine-tune with Bayesian optimization if needed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end