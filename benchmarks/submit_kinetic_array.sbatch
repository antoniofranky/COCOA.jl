#!/bin/bash
#SBATCH --job-name=kin_A
#SBATCH --array=1-62
#SBATCH --cpus-per-task=16
#SBATCH --mem=65536
#SBATCH --time=04:00:00
#SBATCH --output=/work/schaffran1/kinetic_rerun/logs/kinA_%A_%a.out
#SBATCH --error=/work/schaffran1/kinetic_rerun/logs/kinA_%A_%a.out
#SBATCH --mail-type=FAIL     # array: only email on a failed task (avoids 62 success mails)
#SBATCH --mail-user=schaffran1@uni-potsdam.de   # <-- VERIFY: schaffran1 / schaffrank1 / schaffranke1 ?

# RUN A — recompute kinetic modules from COCOA's OWN concordance, WITH the zero-complex
# fix (branch revision/zero-complex-fix). One array task per line of model_manifest.txt.
#
# Prereqs (see CLUSTER_HANDOFF.md):
#   - /work/schaffran1/COCOA.jl checked out to revision/zero-complex-fix
#   - benchmark env instantiated at /work/schaffran1/kinetic_rerun/env
#     (dev COCOA + MATFBCModels + JLD2 + HiGHS + AbstractFBCModels)
#   - concordance JLD2s in       /work/schaffran1/kinetic_rerun/concordance
#   - elementary .mat models in  /work/schaffran1/kinetic_rerun/models

export JULIA_DEPOT_PATH=/work/schaffran1/.julia
export PATH=/work/schaffran1/.juliaup/bin:$PATH
mkdir -p /work/schaffran1/kinetic_rerun/logs

export CONC_DIR=/work/schaffran1/kinetic_rerun/concordance
export MODEL_DIR=/work/schaffran1/kinetic_rerun/models
export OUT_DIR=/work/schaffran1/kinetic_rerun/out_A

julia -t "${SLURM_CPUS_PER_TASK}" --project=/work/schaffran1/kinetic_rerun/env \
    /work/schaffran1/COCOA.jl/benchmarks/run_kinetic_step.jl
