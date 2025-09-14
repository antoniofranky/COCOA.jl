#!/bin/bash
#SBATCH --job-name=cocoa_array
#SBATCH --chdir=/work/schaffran1/jobresults/random_0
#SBATCH --output=/work/schaffran1/jobresults/random_0/cocoa_model_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-ARRAY_SIZE

# Parameters to modify
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/random_0"  # Directory containing models
RESULTS_BASE_DIR="/work/schaffran1/jobresults/random_0"       # Base directory for results

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

# Run analysis and capture output for parsing
OUTPUT_FILE="/tmp/cocoa_output_${SLURM_ARRAY_TASK_ID}.log"
julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME" 2>&1 | tee "$OUTPUT_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# Extract model statistics from output
N_REACTIONS=$(grep "Reactions:" "$OUTPUT_FILE" | head -1 | awk '{print $2}' || echo "0")
N_METABOLITES=$(grep "Metabolites:" "$OUTPUT_FILE" | head -1 | awk '{print $2}' || echo "0")
N_COMPLEXES=$(grep "Total Complexes:" "$OUTPUT_FILE" | head -1 | awk '{print $3}' || echo "0")

# Clean up temp file
rm -f "$OUTPUT_FILE"

# Create single master CSV for all results (append mode)
MASTER_CSV="$RESULTS_BASE_DIR/cocoa_performance_results.csv"

# Create header if file doesn't exist
if [ ! -f "$MASTER_CSV" ]; then
  echo "model_name,job_id,task_id,timestamp,runtime_sec,peak_memory_mb,peak_vmem_mb,n_reactions,n_metabolites,n_complexes" > "$MASTER_CSV"
fi

# Get memory and runtime info from SLURM accounting (wait for accounting data)
sleep 10  # Increased wait time for accounting data
SLURM_METRICS=$(sacct -j $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID --format=MaxRSS,MaxVMSize,Elapsed --noheader --parsable2)
echo "DEBUG: SLURM_METRICS = '$SLURM_METRICS'"  # Debug output
if [ -n "$SLURM_METRICS" ]; then
    echo "DEBUG: Parsing SLURM metrics: $SLURM_METRICS"
    PEAK_MEMORY_KB=$(echo "$SLURM_METRICS" | cut -d'|' -f1 | sed 's/K$//')
    PEAK_VMEM_KB=$(echo "$SLURM_METRICS" | cut -d'|' -f2 | sed 's/K$//')
    RUNTIME_STR=$(echo "$SLURM_METRICS" | cut -d'|' -f3)
    
    echo "DEBUG: PEAK_MEMORY_KB=$PEAK_MEMORY_KB, PEAK_VMEM_KB=$PEAK_VMEM_KB, RUNTIME_STR=$RUNTIME_STR"
    
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
    
    echo "DEBUG: Final values - RUNTIME_SEC=$RUNTIME_SEC, PEAK_MEMORY_MB=$PEAK_MEMORY_MB, PEAK_VMEM_MB=$PEAK_VMEM_MB"
else
    echo "DEBUG: No SLURM metrics found - using zeros"
    PEAK_MEMORY_MB=0; PEAK_VMEM_MB=0; RUNTIME_SEC=0
fi

# Write performance data to CSV
echo "${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date -Iseconds),${RUNTIME_SEC},${PEAK_MEMORY_MB},${PEAK_VMEM_MB},${N_REACTIONS},${N_METABOLITES},${N_COMPLEXES}" >> "$MASTER_CSV"

  echo "Results appended to master CSV: $MASTER_CSV"

echo "Analysis completed for $MODEL_NAME with exit code: $EXIT_CODE"
exit $EXIT_CODE