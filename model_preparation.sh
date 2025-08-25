#!/bin/bash
#SBATCH --job-name=model_prep_array
#SBATCH --chdir=/work/schaffran1/results_testjobs
#SBATCH --output=/model_prep_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-4

# Calculate heap size hint (80% of allocated memory)
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
JULIA_OPTS="$JULIA_OPTS -p 63"
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"
JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"
JULIA_OPTS="$JULIA_OPTS --optimize=2"
JULIA_OPTS="$JULIA_OPTS --check-bounds=no"

cd /work/schaffran1/COCOA.jl

echo "Processing model array task ${SLURM_ARRAY_TASK_ID}/$NUM_MODELS..."
julia $JULIA_OPTS custom_model_prep.jl