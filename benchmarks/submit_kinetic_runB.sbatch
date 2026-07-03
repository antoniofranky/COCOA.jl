#!/bin/bash
#SBATCH --job-name=kin_B
#SBATCH --array=1-62
#SBATCH --cpus-per-task=16
#SBATCH --mem=65536
#SBATCH --time=04:00:00
#SBATCH --output=/work/schaffran1/kinetic_rerun/logs/kinB_%A_%a.out
#SBATCH --error=/work/schaffran1/kinetic_rerun/logs/kinB_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de   # <-- VERIFY address

# RUN B (diagnostic) — COCOA kinetic on the REFERENCE's concordance partition.
# Needs the reference concordant mats staged in /work/schaffran1/kinetic_rerun/ref_concordant.
# VALIDATE the mapping on iIS312 first (see CLUSTER_HANDOFF.md) before trusting the array.

export JULIA_DEPOT_PATH=/work/schaffran1/.julia
export PATH=/work/schaffran1/.juliaup/bin:$PATH
mkdir -p /work/schaffran1/kinetic_rerun/logs

export REF_DIR=/work/schaffran1/kinetic_rerun/ref_concordant
export MODEL_DIR=/work/schaffran1/kinetic_rerun/models
export OUT_DIR=/work/schaffran1/kinetic_rerun/out_B

julia -t "${SLURM_CPUS_PER_TASK}" --project=/work/schaffran1/kinetic_rerun/env \
    /work/schaffran1/COCOA.jl/benchmarks/run_kinetic_from_ref_concordance.jl
