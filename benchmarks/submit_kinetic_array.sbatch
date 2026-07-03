#!/bin/bash
#SBATCH --job-name=kin_fixed
#SBATCH --array=1-62
#SBATCH --cpus-per-task=16
#SBATCH --mem=65536
#SBATCH --time=04:00:00
#SBATCH --output=/work/schaffran1/COCOA.jl/benchmarks/logs/kin_fixed_%A_%a.out
#SBATCH --error=/work/schaffran1/COCOA.jl/benchmarks/logs/kin_fixed_%A_%a.out

# Reruns the kinetic-analysis step of the 62-model benchmark with the
# zero-complex fix (COCOA.jl branch: revision/zero-complex-fix). Resource
# request matches the original pre-fix run (16 CPUs / 64 GB per task, see
# application_note/kinetic/job_mapping_*.txt from the original run).
#
# Prerequisites on this machine:
#   1. git clone (or pull) git@github.com:antoniofranky/COCOA.jl.git at
#      /work/schaffran1/COCOA.jl, checked out to revision/zero-complex-fix
#   2. Julia environment at /work/schaffran1/COCOA.jl/test instantiated
#      (`julia --project=test -e 'using Pkg; Pkg.instantiate()'`) — this
#      already has MATFBCModels, JLD2, HiGHS, GLPK etc.
#   3. Concordance JLD2s already present at
#      /work/schaffran1/application_note/results/matlab_preprocessed/concordance
#      (unchanged by the fix — only the kinetic step needs rerunning)
#   4. Elementary-step .mat models present at
#      /work/schaffran1/Upstream_Algorithm/Models/models_with_elementary_steps
#
# Usage: sbatch submit_kinetic_array.sbatch

mkdir -p /work/schaffran1/COCOA.jl/benchmarks/logs

module load julia 2>/dev/null || true  # adjust to this cluster's module system if needed

julia -t "${SLURM_CPUS_PER_TASK}" --project=/work/schaffran1/COCOA.jl/test \
    /work/schaffran1/COCOA.jl/benchmarks/run_kinetic_step.jl
