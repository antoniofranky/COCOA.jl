#!/bin/bash
# =====================================================================================
# SLURM submission script for COCOA.jl robustness experiments.
#
#   sbatch examples/reproducibility/submit_slurm.sh                  # bundled e_coli_core
#   MODEL=/path/to/model.xml sbatch examples/reproducibility/submit_slurm.sh
#
# Override any of MODEL / SEEDS / EXPERIMENTS / NPROCS / OPTIMIZER / OUTDIR via
# `--export`, e.g.:
#   sbatch --export=ALL,MODEL=iAB_RBC_283.xml,EXPERIMENTS=A examples/reproducibility/submit_slurm.sh
# =====================================================================================
#SBATCH --job-name=cocoa-robust
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de

set -euo pipefail

# NOTE: SLURM copies the batch script into a spool dir, so ${BASH_SOURCE[0]} does NOT
# point at the repo. Locate the project via the submit directory instead (run `sbatch`
# from the repo root), or override REPRO_DIR explicitly.
#   sbatch examples/reproducibility/submit_slurm.sh              # from repo root
#   REPRO_DIR=/abs/path/examples/reproducibility sbatch .../submit_slurm.sh
REPRO_DIR="${REPRO_DIR:-${SLURM_SUBMIT_DIR}/examples/reproducibility}"
if [[ ! -f "${REPRO_DIR}/robustness_experiments.jl" ]]; then
    echo "ERROR: cannot find robustness_experiments.jl under REPRO_DIR=${REPRO_DIR}." >&2
    echo "Run 'sbatch' from the repo root, or set REPRO_DIR to the examples/reproducibility path." >&2
    exit 1
fi
SCRIPT_DIR="${REPRO_DIR}"
PROJECT="${REPRO_DIR}"

# Leave one core for the master process.
export NPROCS="${NPROCS:-$((SLURM_CPUS_PER_TASK - 1))}"

module load julia 2>/dev/null || true   # no-op if julia is already on PATH

echo "Instantiating environment ..."
julia --project="${PROJECT}" -e 'import Pkg; Pkg.instantiate()'

echo "Running robustness experiments ..."
julia --project="${PROJECT}" "${SCRIPT_DIR}/robustness_experiments.jl"
