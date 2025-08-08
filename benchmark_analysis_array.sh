#!/bin/bash
#SBATCH --job-name=cocoa_benchmark
#SBATCH --chdir=/work/schaffran1/COCOA.jl
#SBATCH --output=/work/schaffran1/results_testjobs/cocoa_benchmark_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-9  # Adjust based on number of models

# Model files array (must match array indices)
MODEL_FILES=(
    "e_coli_core.xml"
    "ecoli567_splt_prpd.xml"
    "iJR904.xml"
    "iAF1260.xml"
    "iML1515.xml"
    "iJF4097_splt_prpd.xml"
    "iAF12599_splt_prpd.xml"
    "iML15211_splt_prpd.xml"
    "iML28686_splt_prpd.xml"
)

# Get current model file
MODEL_FILE="${MODEL_FILES[$((SLURM_ARRAY_TASK_ID-1))]}"

# Create results directory
mkdir -p /work/schaffran1/results_testjobs/benchmark_results

# Calculate heap size hint (80% of allocated memory for safety)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 8 / 10 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# HPC optimizations for Julia
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_GC_MEASURE_MALLOC=0
export JULIA_GC_PARALLEL_COLLECT=1

# Julia optimization flags for benchmarking
JULIA_OPTS="--project=/work/schaffran1/COCOA.jl"
JULIA_OPTS="$JULIA_OPTS -p 31"  # Use 31 worker processes
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"
JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"
JULIA_OPTS="$JULIA_OPTS --optimize=2"

# System information
echo "=== System Information ==="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: $MODEL_FILE"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Heap Size: $HEAP_SIZE"
echo "=========================="

cd /work/schaffran1/COCOA.jl

# Verify model exists
MODEL_PATH="/work/schaffran1/COCOA.jl/benchmark/models/$MODEL_FILE"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "=== Model: $MODEL_FILE ==="
echo "Start time: $(date)"

# Run single model benchmark
time julia $JULIA_OPTS benchmark_single_model.jl "$MODEL_FILE"

EXIT_CODE=$?
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
