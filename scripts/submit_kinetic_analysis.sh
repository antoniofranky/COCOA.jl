#!/bin/bash
#SBATCH --job-name=kinetic_analysis
#SBATCH --output=kinetic_%A_%a.out
#SBATCH --error=kinetic_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-330

# Note on thread count:
# - 8 threads is a good balance for most genome-scale models
# - Kinetic analysis parallelizes: upstream algorithm, Prop S3-4 merging, ACR/ACRR detection
# - Beyond 8-16 threads, diminishing returns due to memory bandwidth limits
# - Adjust based on your cluster's node configuration and queue policies

# =============================================================================
# Kinetic Module Analysis - SLURM Array Job Submission Script
# =============================================================================
#
# This script submits kinetic module analysis jobs as a SLURM array job.
# Each array task processes one concordance result file.
#
# Usage:
#   1. Edit the configuration section below
#   2. Submit with: sbatch --array=1-N submit_kinetic_analysis.sh
#      where N is the number of files to process
#
# To find N (number of files):
#   ls /path/to/results/*.jld2 | wc -l
#
# Example:
#   sbatch --array=1-330 submit_kinetic_analysis.sh
#
# =============================================================================

# ========================= CONFIGURATION =========================
# Edit these paths for your setup

# Directory containing concordance result JLD2 files
RESULTS_DIR="/work/schaffran1/jobresults/1e-10/random_0"

# Directory containing original model XML files
# Adjust to match your cluster's model directory structure
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/random_0"

# Output directory for kinetic analysis results
OUTPUT_DIR="/work/schaffran1/jobresults/kinetic_analysis/random_0"

# Path to COCOA.jl
COCOA_DIR="/work/schaffran1/COCOA.jl"

# Julia executable (adjust if using modules)
JULIA_BIN="julia"

# Number of threads for kinetic analysis
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ========================= END CONFIGURATION =========================

echo "=========================================="
echo "Kinetic Module Analysis - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Julia threads: ${JULIA_NUM_THREADS}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "Configuration:"
echo "  Results directory: ${RESULTS_DIR}"
echo "  Models directory: ${MODELS_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  COCOA directory: ${COCOA_DIR}"
echo ""

# Activate Julia project environment
cd "${COCOA_DIR}"

# Run the kinetic analysis script
echo "Running kinetic analysis..."
echo ""

${JULIA_BIN} --threads=${JULIA_NUM_THREADS} --project="${COCOA_DIR}" \
    "${COCOA_DIR}/scripts/kinetic_module_analysis.jl" \
    "${RESULTS_DIR}" \
    "${MODELS_DIR}" \
    "${OUTPUT_DIR}" \
    "${SLURM_ARRAY_TASK_ID}"

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: ${exit_code}"
echo "End time: $(date)"
echo "=========================================="

exit ${exit_code}
