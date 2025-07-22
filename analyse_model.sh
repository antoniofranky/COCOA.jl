#!/bin/bash
#SBATCH --job-name=cocoa_scaling_test
#SBATCH --chdir=/work/schaffran1/results_testjobs
#SBATCH --output=results_testjobs/cocoa_scaling_test_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread

# Create a unique output directory based on job ID and timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Calculate heap size hint (80% of available memory)
HEAP_SIZE_GB=$(( 128 * 8 / 10 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"
cd /work/schaffran1/COCOA.jl

time julia -p 31 -t auto --heap-size-hint $HEAP_SIZE analyse_model.jl