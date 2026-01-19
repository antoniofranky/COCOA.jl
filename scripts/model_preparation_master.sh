#!/bin/bash
#SBATCH --job-name=prep_models
#SBATCH --chdir=/work/schaffran1/jobresults
#SBATCH --output=/work/schaffran1/jobresults/master_logs/prep_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-130%20

# ============================================================================
# Model preparation script
# Processes 13 models × 10 seeds = 130 tasks
# Each task processes one model with one seed
# Uses distributed processing for blocked reaction detection
# ============================================================================

echo "==================================="
echo "Prep Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "==================================="

# Create master logs directory
mkdir -p /work/schaffran1/jobresults/master_logs

# Julia optimization flags
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Julia options with distributed processing
JULIA_OPTS="--project=/work/schaffran1/COCOA.jl"
JULIA_OPTS="$JULIA_OPTS -p $((SLURM_CPUS_PER_TASK - 1))"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"

cd /work/schaffran1/COCOA.jl/scripts

# Precompile packages only once per job (first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting model preparation with $((SLURM_CPUS_PER_TASK - 1)) workers..."
julia $JULIA_OPTS model_preparation.jl
EXIT_CODE=$?

echo "Model preparation completed for seed $SEED with exit code: $EXIT_CODE"
exit $EXIT_CODE
