#!/bin/bash
#SBATCH --job-name=cocoa_array
#SBATCH --chdir=RESULTS_DIR_PLACEHOLDER
#SBATCH --output=RESULTS_DIR_PLACEHOLDER/cocoa_model_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=92
#SBATCH --mem=500G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-ARRAY_SIZE

# ============================================================================
# CONFIGURATION - Edit these two paths to match your setup
# ============================================================================
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/random_0"  # Directory containing model .xml files
RESULTS_DIR="/work/schaffran1/jobresults/random_0_real"      # Directory where results will be saved
# ============================================================================

# Create results directory
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

# Save job mapping for later performance analysis
JOB_MAPPING_FILE="$RESULTS_DIR/job_mapping_${SLURM_ARRAY_JOB_ID}.txt"
echo "${SLURM_ARRAY_TASK_ID}|${MODEL_NAME}|${MODEL_FILE}|$(date '+%Y-%m-%d %H:%M:%S')" >> "$JOB_MAPPING_FILE"

# Force consistent package precompilation (only for first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting analysis for $MODEL_NAME..."

# Run analysis
julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME"
EXIT_CODE=$?

echo "Analysis completed for $MODEL_NAME with exit code: $EXIT_CODE"
exit $EXIT_CODE