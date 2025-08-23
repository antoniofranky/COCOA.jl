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

# Create timestamp for this array job run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE_DIR}/cocoa_array_${SLURM_ARRAY_JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Calculate heap size hint (80% of allocated memory from SLURM_MEM_PER_NODE)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 8 / 10 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# Load LIKWID for memory monitoring
module load arch/r1/zen4
module load linux-rocky9-zen4/gcc-14.2.0/likwid/5.3.0-gose7xd

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

# Setup LIKWID memory monitoring
MEMORY_OUTPUT="$RESULTS_DIR/memory_${MODEL_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
MEMORY_SUMMARY="$RESULTS_DIR/memory_summary_${MODEL_NAME}_${SLURM_ARRAY_TASK_ID}.csv"

echo "Starting analysis for $MODEL_NAME with LIKWID memory monitoring..."

# Run analysis with LIKWID memory profiling (zero computational overhead)
likwid-perfctr -g MEM -m -o "$MEMORY_OUTPUT" julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME"
EXIT_CODE=$?

# Create single master CSV for all results (append mode)
MASTER_CSV="$RESULTS_BASE_DIR/cocoa_performance_results.csv"

# Create header if file doesn't exist
if [ ! -f "$MASTER_CSV" ]; then
    echo "model_name,job_id,task_id,timestamp,runtime_sec,peak_memory_mb,memory_bandwidth_mbps,memory_volume_gb,n_reactions,n_metabolites,n_complexes,n_modules,validation_passed" > "$MASTER_CSV"
fi

# Extract key metrics from LIKWID output
if [ -f "$MEMORY_OUTPUT" ]; then
    RUNTIME=$(grep -E "Runtime.*sec" "$MEMORY_OUTPUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    PEAK_MEMORY=$(grep -E "Memory.*MB" "$MEMORY_OUTPUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    MEMORY_BW=$(grep -E "Memory bandwidth" "$MEMORY_OUTPUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    MEMORY_VOL=$(grep -E "Memory.*volume.*GB" "$MEMORY_OUTPUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
else
    RUNTIME=""; PEAK_MEMORY=""; MEMORY_BW=""; MEMORY_VOL=""
fi

# Extract analysis results from JLD2 file (if exists)
RESULTS_FILE=$(find "$RESULTS_DIR" -name "concordance_results_${MODEL_NAME}_*.jld2" | head -1)
if [ -f "$RESULTS_FILE" ]; then
    # Use Julia to extract key stats and append to CSV
    julia --project=/work/schaffran1/COCOA.jl -e "
    using JLD2
    data = JLD2.load(\"$RESULTS_FILE\")
    stats = data[\"results\"].stats
    println(\"${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date -Iseconds),${RUNTIME:-0},${PEAK_MEMORY:-0},${MEMORY_BW:-0},${MEMORY_VOL:-0},\$(get(stats, \"n_reactions\", 0)),\$(get(stats, \"n_metabolites\", 0)),\$(get(stats, \"n_complexes\", 0)),\$(get(stats, \"n_modules\", 0)),\$(get(stats, \"validation_passed\", false))\")" >> "$MASTER_CSV"
else
    # Fallback if no JLD2 file found
    echo "${MODEL_NAME},${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},$(date -Iseconds),${RUNTIME:-0},${PEAK_MEMORY:-0},${MEMORY_BW:-0},${MEMORY_VOL:-0},0,0,0,0,false" >> "$MASTER_CSV"
fi

echo "Results appended to master CSV: $MASTER_CSV"

echo "Analysis completed for $MODEL_NAME with exit code: $EXIT_CODE"
exit $EXIT_CODE