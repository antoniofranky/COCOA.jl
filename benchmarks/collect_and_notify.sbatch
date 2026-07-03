#!/bin/bash
#SBATCH --job-name=kin_collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --time=00:30:00
#SBATCH --output=/work/schaffran1/kinetic_rerun/logs/collect_%j.out
#SBATCH --error=/work/schaffran1/kinetic_rerun/logs/collect_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de   # <-- VERIFY address

# Summarises out_A (and out_B if present) into a CSV and emails on completion.
# Submit with a dependency so it runs after the arrays finish, e.g.:
#   AJOB=$(sbatch --parsable submit_kinetic_array.sbatch)
#   sbatch --dependency=afterany:$AJOB collect_and_notify.sbatch

export JULIA_DEPOT_PATH=/work/schaffran1/.julia
export PATH=/work/schaffran1/.juliaup/bin:$PATH

export OUT_DIR=/work/schaffran1/kinetic_rerun/out_A
export OUT_DIR_B=/work/schaffran1/kinetic_rerun/out_B
export SUMMARY_CSV=/work/schaffran1/kinetic_rerun/kinetic_summary.csv

julia --project=/work/schaffran1/kinetic_rerun/env \
    /work/schaffran1/COCOA.jl/benchmarks/collect_results.jl
