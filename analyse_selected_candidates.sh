#!/bin/bash
#SBATCH --job-name=cocoa_candidates
#SBATCH --chdir=/work/schaffran1/jobresults
#SBATCH --output=/work/schaffran1/jobresults/cocoa_candidate_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-13  # 13 candidates

# Parameters to modify
CANDIDATES_CSV="/work/schaffran1/toolbox/results/analysis/recommended_candidates.csv"
RESULTS_DIR="/work/schaffran1/jobresults/selected_candidates"

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

echo "===================================="
echo "COCOA Selected Candidates Analysis"
echo "===================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Candidates CSV: $CANDIDATES_CSV"
echo "Results directory: $RESULTS_DIR"
echo "===================================="

# Check if candidates file exists
if [ ! -f "$CANDIDATES_CSV" ]; then
    echo "ERROR: Candidates CSV file not found: $CANDIDATES_CSV"
    exit 1
fi

# Force consistent package precompilation (only for first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting analysis for candidate $SLURM_ARRAY_TASK_ID..."

# Run analysis with model index from array task ID
julia $JULIA_OPTS analyse_selected_candidates.jl "$CANDIDATES_CSV" "$RESULTS_DIR" "$SLURM_ARRAY_TASK_ID"
EXIT_CODE=$?

echo "Analysis completed for candidate $SLURM_ARRAY_TASK_ID with exit code: $EXIT_CODE"
exit $EXIT_CODE
