# COCOA Array Job Scripts

This directory contains scripts for running COCOA concordance analysis on multiple models using SLURM array jobs.

## Files

- `analyse_models_array.sh` - SLURM array job script
- `analyse_models_array.jl` - Julia analysis script (called by array job)
- `submit_array_job.sh` - Helper script to submit array jobs
- `collect_array_results.jl` - Script to collect and summarize results

## Usage

### 1. Basic Usage

```bash
# Submit array job with default settings
./submit_array_job.sh

# Specify custom directories
./submit_array_job.sh /path/to/models /path/to/results
```

### 2. Manual Configuration

Edit the parameters in these files before submission:

**In `submit_array_job.sh`:**
```bash
DEFAULT_MODELS_DIR="/work/schaffran1/toolbox/prpd_models/ordered"
DEFAULT_RESULTS_DIR="/work/schaffran1/results_testjobs"
DEFAULT_EMAIL="your.email@uni-potsdam.de"
```

**In `analyse_models_array.jl`:**
```julia
sample_size = 1000
seed = 42
cv_threshold = 0.01
batch_size = 100_000
use_transitivity = true
```

### 3. Submit Array Job

```bash
# Make scripts executable
chmod +x submit_array_job.sh analyse_models_array.sh

# Submit the array job
./submit_array_job.sh
```

The script will:
- Count `.xml` files in the models directory
- Show you a preview of the job configuration
- Ask for confirmation before submission
- Submit the array job to SLURM

### 4. Monitor Jobs

```bash
# Check job status
squeue -j <JOB_ID>

# Check detailed job accounting
sacct -j <JOB_ID>

# Cancel all array tasks
scancel <JOB_ID>
```

### 5. Collect Results

After jobs complete, collect and summarize results:

```bash
# Collect results from default directory
julia collect_array_results.jl /path/to/results

# Specify custom output prefix
julia collect_array_results.jl /path/to/results my_analysis
```

This will generate:
- `cocoa_summary_YYYYMMDD_HHMMSS.csv` - Summary statistics
- `cocoa_detailed_YYYYMMDD_HHMMSS.csv` - Detailed statistics

## Job Configuration

Each array task uses:
- **Time limit:** 24 hours
- **Memory:** 128GB
- **CPUs:** 64 cores
- **Max concurrent jobs:** 10

## Output Files

For each model, the analysis generates:
- `concordance_results_<model_name>_<parameters>.jld2` - Main results
- `cocoa_model_<JOB_ID>_<TASK_ID>.out` - SLURM log file
- `error_<model_name>_<timestamp>.txt` - Error log (if analysis fails)

## Troubleshooting

### Common Issues

1. **No models found:**
   - Check that `.xml` files exist in the models directory
   - Verify directory path is correct

2. **Analysis fails:**
   - Check error files in results directory
   - Verify model files are valid SBML
   - Check memory/time limits

3. **Jobs don't start:**
   - Check SLURM queue: `squeue -u $USER`
   - Verify resource requests are reasonable
   - Check account/partition settings

### Debugging Individual Models

To test a single model before running the array job:

```bash
# Test single model
julia analyse_models_array.jl /path/to/model.xml /path/to/results test_model
```

### Modifying Resource Requirements

Edit `analyse_models_array.sh` to adjust:

```bash
#SBATCH --time=48:00:00      # Increase time limit
#SBATCH --mem=256G           # Increase memory
#SBATCH --cpus-per-task=32   # Reduce CPUs if needed
#SBATCH --array=1-N%5        # Reduce concurrent jobs
```

## Analysis Parameters

The scripts use the same parameters as the original `analyse_model.jl`:

- **sample_size:** Number of flux samples (default: 1000)
- **cv_threshold:** Coefficient of variation threshold (default: 0.00001)
- **batch_size:** Optimization batch size (default: 100,000)
- **use_transitivity:** Enable transitivity optimization (default: true)
- **seed:** Random seed for reproducibility (default: 42)

## Results Structure

The JLD2 results files contain:
```julia
"results" => concordance_analysis_results
"model_name" => model_name  
"model_file" => path_to_model_file
"analysis_parameters" => parameter_dictionary
"analysis_duration_seconds" => total_runtime
```

## Performance Tips

1. **Memory:** Large models may need >128GB memory
2. **Time:** Complex models may take >24 hours
3. **Concurrency:** Reduce array job concurrency if cluster is busy
4. **Batch size:** Increase for large models, decrease if memory limited