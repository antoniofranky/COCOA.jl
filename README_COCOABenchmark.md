# COCOABenchmark Module

A comprehensive benchmarking, profiling, and optimization module for COCOA.jl metabolic network concordance analysis.

## Overview

The COCOABenchmark module provides unified tools for analyzing model performance, optimizing memory usage, and generating configuration recommendations for concordance analysis. It's designed to work with any preprocessed metabolic model and provides detailed insights for both local workstations and HPC environments.

## Features

- **🔍 Model Analysis**: Automatically characterizes your model's complexity and computational requirements
- **💾 Memory Optimization**: Benchmarks buffer strategies and estimates memory scaling
- **⚙️ Parameter Optimization**: Tests different configurations to find optimal settings
- **📈 Performance Prediction**: Estimates runtime and memory for different model sizes
- **💡 Smart Recommendations**: Provides system-specific and model-specific optimization advice
- **📄 Comprehensive Reporting**: Generates detailed reports with configuration code

## Quick Start

```julia
using COBREXA

# Include the benchmark module
include("COCOABenchmark.jl")
using .COCOABenchmark

# Load your preprocessed model
model = load_model("path/to/your/model.xml")

# Run comprehensive analysis
results = analyze_model_performance(model)

# Generate detailed report
generate_optimization_report(results, "optimization_report.txt")

# Access optimal configuration
optimal_config = results.optimal_config
println("Recommended batch size: ", optimal_config.batch_size)
```

## Main Functions

### `analyze_model_performance(model; sample_size=100, n_configs=15)`

Main entry point that performs comprehensive analysis:

- **Input**: Preprocessed metabolic model from COBREXA.jl
- **Output**: `ModelAnalysisResults` with complete benchmark data
- **Parameters**:
  - `sample_size`: Size for buffer benchmarking (default: 100)
  - `n_configs`: Number of parameter configurations to test (default: 15)

### `generate_optimization_report(results, output_file)`

Generates a detailed text report with:
- System information and model characteristics
- Buffer efficiency results
- Optimal configuration parameters
- Performance predictions for different model sizes
- Memory scaling estimates
- Specific recommendations
- Ready-to-use Julia configuration code

## Result Structure

The `ModelAnalysisResults` contains:

```julia
struct ModelAnalysisResults
    system_info::SystemInfo              # System resources and capabilities
    model_characteristics::ModelCharacteristics  # Model complexity metrics
    buffer_results::BufferBenchmarkResults       # Buffer efficiency analysis
    parameter_results::ParameterBenchmarkResults # Parameter optimization
    optimal_config::COCOA.ConcordanceConfig     # Best configuration found
    memory_scaling::Dict{String, Float64}        # Memory estimates for different scales
    performance_predictions::Dict{String, Dict{String, Float64}}  # Performance forecasts
    recommendations::Vector{String}              # Actionable recommendations
    analysis_timestamp::DateTime                # When analysis was performed
end
```

## Example Output

```
🚀 Starting comprehensive COCOA model analysis...
============================================================
📊 Analyzing system resources and model characteristics...
   System: 16.0GB RAM, 8 threads
   Model: 2712 reactions, 1805 metabolites (medium)

🧪 Benchmarking buffer efficiency...
  Testing with pre-allocated buffers...
  Testing without pre-allocated buffers...

⚙️  Optimizing parameters...
  Testing 15 configurations...
    Config 1/15... time: 0.0234s, memory: 2.34MB
    Config 2/15... time: 0.0198s, memory: 1.87MB
    ...

📈 Generating scaling estimates and performance predictions...

💡 Generating recommendations...

✅ Analysis complete!
============================================================
```

## Use Cases

### 1. Local Development
```julia
# Quick analysis for development
results = analyze_model_performance(model; sample_size=50, n_configs=8)
config = results.optimal_config
```

### 2. Production Optimization
```julia
# Comprehensive analysis for production deployment
results = analyze_model_performance(model; sample_size=200, n_configs=25)
generate_optimization_report(results, "production_config.txt")
```

### 3. HPC Planning
```julia
# Analysis for HPC resource planning
results = analyze_model_performance(model)
memory_scaling = results.memory_scaling
predictions = results.performance_predictions
```

## Recommendations Interpretation

The module provides several types of recommendations:

- **⚠️ Warnings**: Critical system limitations or model issues
- **✅ Good**: Current system capabilities and settings
- **🎯 Optimization**: Specific parameter recommendations
- **ℹ️ Information**: General guidance and tips

## Memory Scaling

The module estimates memory requirements for different model sizes:

- **1x**: Current model size
- **2x**: Double size (2.5x memory scaling)
- **5x**: Five times larger (7x memory scaling)
- **10x**: Ten times larger (15x memory scaling)
- **20x**: Twenty times larger (35x memory scaling)

## Performance Predictions

Estimates for different scenarios:

- **current_model**: Your current model
- **double_size**: 2x larger model
- **five_times**: 5x larger model
- **ten_times**: 10x larger model

Each prediction includes:
- Estimated runtime
- Memory requirements
- Feasibility on current system
- Recommended batch/stage sizes

## Integration with COCOA.jl

Use the optimized configuration with COCOA:

```julia
# Get optimal configuration
results = analyze_model_performance(model)
config = results.optimal_config

# Run concordance analysis with optimized settings
concordance_results = COCOA.concordance_analysis(model, config)
```

## Files

- `COCOABenchmark.jl`: Main module implementation
- `demo_benchmark.jl`: Example usage script
- `README.md`: This documentation

## Requirements

- Julia 1.6+
- COBREXA.jl
- COCOA.jl (in parent directory)
- Standard Julia libraries: Statistics, LinearAlgebra, SparseArrays, etc.

## Tips

1. **Start Small**: Use smaller `sample_size` and `n_configs` for initial testing
2. **Save Reports**: Always generate reports for important analyses
3. **Monitor Memory**: Watch memory usage during analysis on memory-constrained systems
4. **HPC Planning**: Use scaling estimates to plan cluster resource requests
5. **Iterative Optimization**: Re-run analysis after model changes or system upgrades
