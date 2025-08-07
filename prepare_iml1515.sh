#!/bin/bash
#SBATCH --job-name=prepare_iml1515
#SBATCH --chdir=/work/schaffran1/COCOA.jl
#SBATCH --output=/work/schaffran1/results_testjobs/prepare_iml1515_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=164G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread

# Create results directory
mkdir -p /work/schaffran1/results_testjobs

# Calculate heap size hint (70% of allocated memory for safety)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 7 / 10 / 1024 ))
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
JULIA_OPTS="$JULIA_OPTS -p 63"  # Use 63 worker processes (64 total cores - 1 master)
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"
JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"
JULIA_OPTS="$JULIA_OPTS --optimize=2"

# System information
echo "=== System Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Heap Size: $HEAP_SIZE"
echo "Julia Options: $JULIA_OPTS"
echo "Working Directory: $(pwd)"
echo "=========================="

cd /work/schaffran1/COCOA.jl

# Force consistent package precompilation
echo "=== Precompiling packages ==="
julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"

# Run the preparation script
echo "=== Preparing iML1515 model ==="
echo "Start time: $(date)"

time julia $JULIA_OPTS prepare_iml1515.jl

EXIT_CODE=$?
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# Check if output file was created
OUTPUT_FILE="/work/schaffran1/COCOA.jl/benchmark/iML1515_splt_prpd.xml"
if [ -f "$OUTPUT_FILE" ]; then
    echo "=== SUCCESS: Model preparation complete ==="
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
else
    echo "=== ERROR: Output file not found ==="
    EXIT_CODE=1
fi

echo "=== Job Complete ==="
exit $EXIT_CODE
