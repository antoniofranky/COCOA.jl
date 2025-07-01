"""
Example usage of the COCOABenchmark module for comprehensive model analysis.

This script demonstrates how to use the unified benchmarking module to analyze
a metabolic model and get detailed performance recommendations.
"""

using COBREXA

# Include the benchmark module
include("COCOABenchmark.jl")
using .COCOABenchmark

function demo_analysis()
    println("🧪 COCOABenchmark Module Demo")
    println("="^40)

    # Load a model (replace with your model path)
    model_path = "e_coli_core.xml"

    if !isfile(model_path)
        println("⚠️  Model file not found: $model_path")
        println("Please update the model_path variable to point to your model file")
        return
    end

    println("📂 Loading model: $model_path")
    model = load_model(model_path)

    # Run comprehensive analysis
    println("\n🚀 Running comprehensive analysis...")
    results = analyze_model_performance(model; sample_size=50, n_configs=10)

    # Display summary results
    println("\n📊 ANALYSIS SUMMARY")
    println("-"^30)
    println("Model size: $(results.model_characteristics.size_category)")
    println("System memory: $(round(results.system_info.available_memory_gb, digits=1)) GB")
    println("Optimal batch size: $(results.optimal_config.batch_size)")
    println("Optimal stage size: $(results.optimal_config.stage_size)")
    println("Buffer improvement: $(round(results.buffer_results.efficiency_improvement, digits=1))%")

    println("\n💡 TOP RECOMMENDATIONS:")
    for (i, rec) in enumerate(results.recommendations[1:min(3, length(results.recommendations))])
        println("  $i. $rec")
    end

    # Generate detailed report
    report_file = "cocoa_analysis_report.txt"
    println("\n📄 Generating detailed report...")
    generate_optimization_report(results, report_file)

    println("\n✅ Demo completed successfully!")
    println("Check '$report_file' for detailed analysis results.")

    return results
end

# Run demo if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    demo_analysis()
end
