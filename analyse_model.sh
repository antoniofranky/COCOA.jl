#!/bin/bash
#SBATCH --job-name=bmtest
#SBATCH --chdir=/work/schaffran1/results_testjobs
#SBATCH --output=results_testjobs/testiS_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --qos=normal

# Create a unique output directory based on job ID and timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
# JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"
# JULIA_OPTS="$JULIA_OPTS --optimize=2"
# JULIA_OPTS="$JULIA_OPTS --check-bounds=no"

cd /work/schaffran1/COCOA.jl

# Force consistent package precompilation
# echo "Precompiling packages..."
# julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"

echo "Starting analysis..."
time julia $JULIA_OPTS analyse_model.jl