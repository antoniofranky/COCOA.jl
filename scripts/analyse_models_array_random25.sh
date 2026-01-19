#!/bin/bash
#SBATCH --job-name=r25
#SBATCH --chdir=/work/schaffran1/jobresults/random_25
#SBATCH --output=/work/schaffran1/jobresults/random_25/ka_cocoa_model_%A_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=150G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-13

# ============================================================================
# CONFIGURATION - Edit these paths to match your setup
# ============================================================================
MODELS_DIR="/work/schaffran1/toolbox/prpd_models/random_25"  # Directory containing model .xml files
RESULTS_DIR="/work/schaffran1/jobresults/random_25"      # Directory where results will be saved
KINETIC_ANALYSIS="false"  # Set to "true" or "false"
# ============================================================================

# Create results directory
mkdir -p "$RESULTS_DIR"

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

# Define recommended candidate models from CSV
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

# Get only the recommended model files from the directory
MODEL_FILES=()
for model_name in "${RECOMMENDED_MODELS[@]}"; do
    model_file=$(find "$MODELS_DIR" -name "${model_name}.xml" | head -1)
    if [ -n "$model_file" ]; then
        MODEL_FILES+=("$model_file")
    fi
done

MODEL_COUNT=${#MODEL_FILES[@]}

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No .xml model files found in $MODELS_DIR"
    exit 1
fi

if [ $SLURM_ARRAY_TASK_ID -gt $MODEL_COUNT ]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of models ($MODEL_COUNT)"
    exit 0
fi

# Select model file based on array task ID (1-indexed)
MODEL_FILE="${MODEL_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"
MODEL_NAME=$(basename "$MODEL_FILE" .xml)

echo "==================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing model: $MODEL_NAME"
echo "Model file: $MODEL_FILE"
echo "Results directory: $RESULTS_DIR"
echo "==================================="

# Save job mapping for later performance analysis
JOB_MAPPING_FILE="$RESULTS_DIR/job_mapping_${SLURM_ARRAY_JOB_ID}.txt"
echo "${SLURM_ARRAY_TASK_ID}|${MODEL_NAME}|${MODEL_FILE}|$(date '+%Y-%m-%d %H:%M:%S')" >> "$JOB_MAPPING_FILE"

# Force consistent package precompilation (only for first task)
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Precompiling packages..."
    julia --project=/work/schaffran1/COCOA.jl -e "using Pkg; Pkg.precompile()"
fi

echo "Starting analysis for $MODEL_NAME..."
echo "Kinetic analysis: $KINETIC_ANALYSIS"

# Run analysis
julia $JULIA_OPTS analyse_models_array.jl "$MODEL_FILE" "$RESULTS_DIR" "$MODEL_NAME" "$KINETIC_ANALYSIS"
EXIT_CODE=$?

echo "Analysis completed for $MODEL_NAME with exit code: $EXIT_CODE"
exit $EXIT_CODE
