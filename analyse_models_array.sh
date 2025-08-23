#!/bin/bash
#SBATCH --job-name=cocoa_array
#SBATCH --chdir=/work/schaffran1/results_testjobs
#SBATCH --output=results_testjobs/cocoa_model_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-ARRAY_SIZE%15  # Process up to 15 jobs concurrently

# Parameters to modify
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/ordered"  # Directory containing models
RESULTS_BASE_DIR="/work/schaffran1/results_testjobs"       # Base directory for results

# Use single results directory
RESULTS_DIR="$RESULTS_BASE_DIR"
mkdir -p "$RESULTS_DIR"

# Calculate heap size hint (80% of allocated memory from SLURM_MEM_PER_NODE)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 8 / 10 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# HPC optimizations for Julia
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_GC_MEASURE_MALLOC=0
export JULIA_GC_PARALLEL_COLLECT=1

# Julia optimization flags
JULIA_OPTS="--project=/work/schaffran1/COCOA.jl"
JULIA_OPTS="$JULIA_OPTS -p $((SLURM_CPUS_PER_TASK - 1))"
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"

cd /work/schaffran1/COCOA.jl

# Get the model file for this array task
MODEL_FILES=($(find "$MODELS_DIR" -name "*.xml" | sort))
MODEL_COUNT=${#MODEL_FILES[@]}

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No .xml model files found in $MODELS_DIR"
    exit 1
fi

if [ $SLURM_ARRAY_TASK_ID -gt $MODEL_COUNT ]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of models ($MODEL_COUNT)"
    exit 0
fi

# Select model file based on array task ID (1-indexed)
MODEL_FILE="${MODEL_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"
MODEL_NAME=$(basename "$MODEL_FILE" .xml)

echo "==================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing model: $MODEL_NAME"
echo "Model file: $MODEL_FILE"
echo "Results directory: $RESULTS_DIR"
echo "==================================="

# Force consistent package precompilation (only for first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting analysis for $MODEL_NAME..."

# Run analysis with LIKWID CPU pinning (memory tracked by SLURM)
julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME"
EXIT_CODE=$?

# Create single master CSV for all results (append mode)
MASTER_CSV="$RESULTS_BASE_DIR/cocoa_performance_results.csv"

# Create header if file doesn't exist
if [ ! -f "$MASTER_CSV" ]; then
  echo "model_name,job_id,task_id,timestamp,runtime_sec,peak_memory_mb,peak_vmem_mb" > "$MASTER_CSV"
fi

# Get memory and runtime info from SLURM accounting (wait briefly for accounting data)
sleep 2
SLURM_METRICS=$(sacct -j $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID --format=MaxRSS,MaxVMSize,Elapsed --noheader --parsable2)
if [ -n "$SLURM_METRICS" ]; then
    PEAK_MEMORY_KB=$(echo "$SLURM_METRICS" | cut -d'|' -f1 | sed 's/K$//')
    PEAK_VMEM_KB=$(echo "$SLURM_METRICS" | cut -d'|' -f2 | sed 's/K$//')
    RUNTIME_STR=$(echo "$SLURM_METRICS" | cut -d'|' -f3)
    
    # Convert memory from KB to MB
    PEAK_MEMORY_MB=$((PEAK_MEMORY_KB / 1024))
    PEAK_VMEM_MB=$((PEAK_VMEM_KB / 1024))
    
    # Convert runtime to seconds (format: HH:MM:SS or MM:SS)
    if [[ $RUNTIME_STR =~ ^[0-9]+:[0-9]+:[0-9]+$ ]]; then
        # HH:MM:SS format
        IFS=':' read -r hours minutes seconds <<< "$RUNTIME_STR"
        RUNTIME_SEC=$((hours * 3600 + minutes * 60 + seconds))
    elif [[ $RUNTIME_STR =~ ^[0-9]+:[0-9]+$ ]]; then
        # MM:SS format
        IFS=':' read -r minutes seconds <<< "$RUNTIME_STR"
        RUNTIME_SEC=$((minutes * 60 + seconds))
    else
        RUNTIME_SEC=0
    fi
else
    PEAK_MEMORY_MB=0; PEAK_VMEM_MB=0; RUNTIME_SEC=0
fi

# # Extract analysis results from JLD2 file (if exists)
# RESULTS_FILE=$(find "$RESULTS_DIR" -name "kinetic_results_${MODEL_NAME}_*.jld2" | head -1)
# if [ -f "$RESULTS_FILE" ]; then
#     # Use Julia to extract key stats and append to CSV
#     julia --project=/work/schaffran1/COCOA.jl -e "
#     using JLD2
#     data = JLD2.load(\"$RESULTS_FILE\")
#     results = data[\"results\"]
#     n_robust_mets = results.n_robust_metabolites
#     n_robust_pairs = results.n_robust_pairs
#     largest_module = results.largest_robust_module_size
#     summary = results.summary
#     n_complexes = summary !== nothing ? get(summary, \"n_complexes\", 0) : 0
#     n_modules = summary !== nothing ? get(summary, \"n_modules\", 0) : 0
#     println(\"${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date -Iseconds),${RUNTIME_SEC},${PEAK_MEMORY_MB},${PEAK_VMEM_MB},\$n_robust_mets,\$n_robust_pairs,\$n_complexes,\$n_modules,\$largest_module\")
#     " >> "$MASTER_CSV"
# else
#     # Fallback if no JLD2 file found
#     echo "${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date -Iseconds),${RUNTIME_SEC},${PEAK_MEMORY_MB},${PEAK_VMEM_MB},0,0,0,0,0" >> "$MASTER_CSV"
# fi

  # Write basic performance data to CSV (no JLD2 analysis needed)        
  echo
  "${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date     
   -Iseconds),${RUNTIME_SEC},${PEAK_MEMORY_MB},${PEAK_VMEM_MB}" >>       
  "$MASTER_CSV"

  echo "Results appended to master CSV: $MASTER_CSV"

echo "Analysis completed for $MODEL_NAME with exit code: $EXIT_CODE"
exit $EXIT_CODE