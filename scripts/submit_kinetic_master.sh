#!/bin/bash
#SBATCH --job-name=kinetic_master
#SBATCH --output=/work/schaffran1/jobresults/kinetic_analysis/kinetic_master_%A_%a.out
#SBATCH --error=/work/schaffran1/jobresults/kinetic_analysis/kinetic_master_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-130

# =============================================================================
# Kinetic Module Analysis - Master SLURM Array Job
# =============================================================================
#
# This script processes concordance results for random_0 across all seeds:
# - 13 models × 1 variant (random_0) × 10 seeds
# - Total: 130 tasks
#
# Usage:
#   sbatch submit_kinetic_master.sh
#
# =============================================================================

# ========================= CONFIGURATION =========================

# Define all parameters - processing random_0 only
VARIANTS=("random_0")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Define recommended models (must match the ones analyzed)
RECOMMENDED_MODELS=(
    "Lipomyces_starkeyi"
    "Tortispora_caseinolytica"
    "Yarrowia_deformans"
    "Alloascoidea_hylecoeti"
    "Sporopachydermia_quercuum"
    "Pachysolen_tannophilus"
    "Komagataella_pastoris"
    "Debaryomyces_hansenii"
    "Saccharomycopsis_malanga"
    "Wickerhamomyces_ciferrii"
    "Hanseniaspora_vinae"
    "Torulaspora_delbrueckii"
    "Neurospora_crassa"
)

# Base paths
COCOA_DIR="/work/schaffran1/COCOA.jl"
JULIA_BIN="julia"
JULIA_THREADS=${SLURM_CPUS_PER_TASK}

# ========================= END CONFIGURATION =========================

N_MODELS=${#RECOMMENDED_MODELS[@]}
N_VARIANTS=${#VARIANTS[@]}
N_SEEDS=${#SEEDS[@]}

# Map array task ID to (seed_idx, variant_idx, model_idx)
# Formula: task_id = seed_idx * (N_VARIANTS * N_MODELS) + variant_idx * N_MODELS + model_idx + 1
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # Convert to 0-indexed

SEED_IDX=$((TASK_ID / (N_VARIANTS * N_MODELS)))
REMAINING=$((TASK_ID % (N_VARIANTS * N_MODELS)))
VARIANT_IDX=$((REMAINING / N_MODELS))
MODEL_IDX=$((REMAINING % N_MODELS))

# Get actual values
SEED=${SEEDS[$SEED_IDX]}
VARIANT=${VARIANTS[$VARIANT_IDX]}
MODEL_NAME=${RECOMMENDED_MODELS[$MODEL_IDX]}

# Set paths based on variant and seed
RESULTS_DIR="/work/schaffran1/jobresults/${SEED}/${VARIANT}"
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/seed_${SEED}/${VARIANT}"
OUTPUT_DIR="/work/schaffran1/jobresults/kinetic_analysis/${SEED}/${VARIANT}"

# Create log and output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

# Move initial log files to variant-specific directory
LOG_FILE="$OUTPUT_DIR/logs/kinetic_master_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="$OUTPUT_DIR/logs/kinetic_master_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Redirect all output to variant-specific log files
exec > >(tee "$LOG_FILE")
exec 2> >(tee "$ERR_FILE" >&2)

echo "=========================================="
echo "Kinetic Module Analysis - Master Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Julia threads: ${JULIA_THREADS}"
echo ""
echo "Configuration:"
echo "  Seed: $SEED"
echo "  Variant: $VARIANT"
echo "  Model: $MODEL_NAME"
echo "  Results directory: $RESULTS_DIR"
echo "  Models directory: $MODELS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Find the concordance result file
RESULT_FILE=$(find "$RESULTS_DIR" -name "kinetic_results_${MODEL_NAME}_*.jld2" -type f | head -1)

if [ -z "$RESULT_FILE" ]; then
    echo "ERROR: Concordance result file not found for $MODEL_NAME in $RESULTS_DIR"
    echo "Searched pattern: kinetic_results_${MODEL_NAME}_*.jld2"
    exit 1
fi

echo "Found concordance result: $(basename $RESULT_FILE)"
echo ""

# Activate Julia project environment
cd "${COCOA_DIR}"

# Run the kinetic analysis script
echo "Running kinetic module analysis..."
echo ""

${JULIA_BIN} --threads=${JULIA_THREADS} --project="${COCOA_DIR}" \
    "${COCOA_DIR}/scripts/kinetic_module_analysis.jl" \
    "$RESULT_FILE" \
    "$MODELS_DIR" \
    "$OUTPUT_DIR"

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: ${exit_code}"
echo "End time: $(date)"
echo "=========================================="

exit ${exit_code}
