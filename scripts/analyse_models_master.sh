#!/bin/bash
#SBATCH --job-name=cocoa_master
#SBATCH --chdir=/work/schaffran1/jobresults
#SBATCH --output=/work/schaffran1/jobresults/master_logs/cocoa_%A_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-520

# ============================================================================
# Master script for running COCOA analysis across multiple variants and seeds
# 
# This script runs 13 models × 4 variants × 10 seeds = 520 total tasks
# - Models: 13 recommended models
# - Variants: random_25, random_50, random_75, random_100
# - Seeds: 42, 44-52
# 
# ============================================================================

# Define all parameters
VARIANTS=("random_25" "random_50" "random_75" "random_100")
MEMORY_PER_VARIANT=(150 300 600 600)  # GB per variant
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Define recommended models
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
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/seed_${SEED}/${VARIANT}"
RESULTS_DIR="/work/schaffran1/jobresults/${SEED}/${VARIANT}"
KINETIC_ANALYSIS="false"

# Create master logs directory
mkdir -p /work/schaffran1/jobresults/master_logs

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "==================================="
echo "Master Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Variant: $VARIANT"
echo "Model: $MODEL_NAME"
echo "Models directory: $MODELS_DIR"
echo "Results directory: $RESULTS_DIR"
echo "==================================="

# Find the model file
MODEL_FILE=$(find "$MODELS_DIR" -name "${MODEL_NAME}.xml" | head -1)

if [ -z "$MODEL_FILE" ]; then
    echo "ERROR: Model file not found: ${MODEL_NAME}.xml in $MODELS_DIR"
    exit 1
fi

echo "Model file: $MODEL_FILE"

# Calculate heap size hint (80% of allocated memory from SLURM_MEM_PER_NODE)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 8 / 10 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# HPC optimizations for Julia
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Julia optimization flags
JULIA_OPTS="--project=/work/schaffran1/COCOA.jl"
JULIA_OPTS="$JULIA_OPTS -p $((SLURM_CPUS_PER_TASK - 1))"
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"

cd /work/schaffran1/COCOA.jl/scripts

# Save job mapping for later analysis
JOB_MAPPING_FILE="$RESULTS_DIR/job_mapping_master_${SLURM_ARRAY_JOB_ID}.txt"
echo "${SLURM_ARRAY_TASK_ID}|${SEED}|${VARIANT}|${MODEL_NAME}|${MODEL_FILE}|$(date '+%Y-%m-%d %H:%M:%S')" >> "$JOB_MAPPING_FILE"

# Precompile packages only once per job (first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting analysis..."
echo "Kinetic analysis: $KINETIC_ANALYSIS"

# Run analysis with the specified seed
julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME" "$KINETIC_ANALYSIS" "$SEED"
EXIT_CODE=$?

echo "Analysis completed for $MODEL_NAME (seed $SEED, variant $VARIANT) with exit code: $EXIT_CODE"
exit $EXIT_CODE
