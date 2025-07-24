# COCOA.jl Parameter Testing Guide

## 📋 Overview
This guide explains how to use the parameter testing scripts to find optimal settings for running COCOA.jl concordance analysis on large models (50k+ complexes).

## 📁 Files Created
```
/home/anton/master-thesis/
├── parameter_test.jl           # Main Julia script (tests 12 configurations)
├── parameter_test.sh           # SLURM batch script 
├── precompile_cocoa.jl         # Optional: reduces startup overhead
└── README_PARAMETER_TESTING.md # This guide
```

## 🚀 Quick Start (Recommended)

### Step 1: Submit the Parameter Testing Job
```bash
# Navigate to your working directory
cd /home/anton/master-thesis

# Submit the job to SLURM
sbatch parameter_test.sh
```

### Step 2: Monitor the Job
```bash
# Check job status
squeue -u $USER

# View live output (replace JOBID with actual job ID)
tail -f /work/schaffran1/results_testjobs/cocoa_parameter_test_JOBID.out

# Check for errors
tail -f /work/schaffran1/results_testjobs/cocoa_parameter_test_JOBID.err
```

### Step 3: Collect Results
Results will be saved in: `/work/schaffran1/parameter_test_results/run_TIMESTAMP/`

## 📊 Understanding the Results

### Output Files
- `parameter_test_results_TIMESTAMP.json` - Detailed results with all metrics
- `parameter_test_results_TIMESTAMP.csv` - Tabular data for analysis
- `test_summary_TIMESTAMP.txt` - Quick summary
- `logs/parameter_test_TIMESTAMP.log` - Detailed execution log

### Key Metrics to Look For
1. **Runtime Performance**: `elapsed_time`, `pairs_per_second`
2. **Memory Efficiency**: `memory_allocated_gb`, `memory_per_pair_mb`
3. **Analysis Quality**: `n_concordant_pairs`, `n_modules`
4. **Success Rate**: `success` (true/false)

## 🔧 Advanced Usage

### Option A: Interactive Testing
```bash
# Load modules (if needed)
module load julia

# Navigate to COCOA directory
cd /work/schaffran1/COCOA.jl

# Run interactively with specific resources
salloc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G --time=4:00:00

# Once allocated, run:
julia --project=/work/schaffran1/COCOA.jl -p 31 -t auto \
      --startup-file=no --optimize=2 \
      /home/anton/master-thesis/parameter_test.jl
```

### Option B: Custom Configuration Testing
Edit `/home/anton/master-thesis/parameter_test.jl` to modify configurations:

```julia
# Add your custom configuration to the get_parameter_configurations() function
(
    name = "my_custom_test",
    sample_size = 150,
    stage_size = 1500,
    batch_size = 750,
    tolerance = 0.008,
    coarse_cv_threshold = 0.96,
    cv_threshold = 0.008,
    use_unidirectional_constraints = true,
    chunk_size_filter = 125_000,
    max_pairs_in_memory = 80_000,
    description = "My optimized configuration"
),
```

### Option C: Single Configuration Test
```bash
# Create a minimal test script for one configuration
cat > test_single_config.jl << 'EOF'
using Pkg
Pkg.activate("/work/schaffran1/COCOA.jl")
using COCOA, COBREXA, HiGHS

model = COBREXA.load_model("/work/schaffran1/COCOA.jl/test/e_coli_core_splt_prpd.xml")

# Test your specific parameters
results = concordance_analysis(
    model;
    optimizer = HiGHS.Optimizer,
    sample_size = 100,     # Your choice
    stage_size = 1000,     # Your choice
    batch_size = 500,      # Your choice
    tolerance = 0.01,      # Your choice
    seed = 42
)

println("Results: $(results.stats)")
EOF

# Run it
julia --project=/work/schaffran1/COCOA.jl test_single_config.jl
```

## ⚡ Performance Optimization

### Before Running (Optional but Recommended)
```bash
# 1. Precompile to reduce startup overhead
julia --project=/work/schaffran1/COCOA.jl /home/anton/master-thesis/precompile_cocoa.jl

# 2. Check available resources
sinfo -p your_partition_name
```

### Monitoring During Execution
```bash
# Monitor memory usage
watch -n 30 'free -h && ps aux | grep julia | grep -v grep'

# Monitor job efficiency
seff JOBID  # After job completes
```

## 🎯 Interpreting Results for Large Models

### Look for These Patterns:
1. **Speed vs Memory Trade-off**: 
   - Fast configs may use more memory
   - Memory-efficient configs may be slower

2. **Sample Size Impact**:
   - Smaller samples: faster but may miss concordances
   - Larger samples: slower but more accurate

3. **Scaling Indicators**:
   - `pairs_per_second` - higher is better for large models
   - `memory_per_pair_mb` - lower is better for 50k+ models

### Expected Recommendations Output:
The script will automatically provide:
```
🚀 RECOMMENDATIONS FOR LARGE MODELS (50k+ complexes)

Top 3 recommended configurations for large models:

1. memory_conservative (efficiency score: 0.8234)
   Description: Conservative memory usage for large models
   Performance: 245.1s, 12.3 GB  
   Results: 1247 concordant pairs, 89 modules
   Parameters:
     - sample_size: 50
     - stage_size: 500
     - batch_size: 250
     [etc...]

💡 SCALING RECOMMENDATIONS:
For a 50,000 complex model (56.4x larger):
  - Recommended sample_size: 375
  - Recommended stage_size: 400
  - Recommended batch_size: 175
  - Estimated runtime: 18.3 hours
  - Estimated memory: 45.2 GB
```

## 🔍 Troubleshooting

### Common Issues:

1. **Job Fails Immediately**
   ```bash
   # Check file permissions
   ls -la parameter_test.sh
   chmod +x parameter_test.sh  # If needed
   ```

2. **Out of Memory Errors**
   - Reduce `--mem=60G` to `--mem=40G` in `parameter_test.sh`
   - The script will adjust heap size automatically

3. **Julia Package Issues**
   ```bash
   # Update packages manually
   julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.update()"
   ```

4. **Partial Results Only**
   - Check `intermediate_results_*.json` files
   - The script saves progress periodically

### Getting Help:
```bash
# Check SLURM job details
scontrol show job JOBID

# View full error log
less /work/schaffran1/results_testjobs/cocoa_parameter_test_JOBID.err

# Check disk space
df -h /work/schaffran1/
```

## 📈 Next Steps After Testing

1. **Analyze Results**: Use the CSV file with your preferred analysis tool
2. **Apply to Large Model**: Use recommended parameters for your 50k+ model
3. **Monitor Performance**: Track actual vs predicted performance
4. **Iterate**: Refine parameters based on real large-model performance

## 💡 Tips for Success

- **Start Small**: Run the parameter test first before attempting 50k+ models
- **Monitor Resources**: Watch memory and CPU usage patterns
- **Save Intermediate Results**: The script does this automatically
- **Document Settings**: Keep track of what works for your specific models
- **Plan Time**: Allow 8-12 hours for comprehensive parameter testing

Good luck with your parameter optimization! 🚀