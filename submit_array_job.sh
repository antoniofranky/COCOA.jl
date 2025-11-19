#!/bin/bash

# Helper script to submit COCOA array job
# Configuration is read from analyse_models_array.sh
# Simply edit the MODELS_DIR and RESULTS_DIR in analyse_models_array.sh and run this script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/analyse_models_array.sh"

# Extract configuration from the main script
MODELS_DIR=$(grep '^MODELS_DIR=' "$MAIN_SCRIPT" | cut -d'"' -f2)
RESULTS_DIR=$(grep '^RESULTS_DIR=' "$MAIN_SCRIPT" | cut -d'"' -f2)

echo "COCOA Array Job Submission Script"
echo "=================================="
echo "Configuration read from: analyse_models_array.sh"
echo "Models directory: $MODELS_DIR"
echo "Results directory: $RESULTS_DIR"

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory does not exist: $MODELS_DIR"
    exit 1
fi

# Count .xml model files
MODEL_FILES=($(find "$MODELS_DIR" -name "*.xml" | sort))
MODEL_COUNT=${#MODEL_FILES[@]}

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No .xml model files found in $MODELS_DIR"
    exit 1
fi

echo "Found $MODEL_COUNT model files:"
for ((i=0; i<$MODEL_COUNT; i++)); do
    echo "  $((i+1)). $(basename "${MODEL_FILES[$i]}")"
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Create a temporary copy of the job script with the correct array size
TEMP_SCRIPT="analyse_models_array_temp_$$.sh"
cp "$MAIN_SCRIPT" "$TEMP_SCRIPT"

# Replace ARRAY_SIZE placeholder with actual count
sed -i "s|ARRAY_SIZE|$MODEL_COUNT|g" "$TEMP_SCRIPT"

# Update SLURM output directives with actual RESULTS_DIR
sed -i "s|RESULTS_DIR_PLACEHOLDER|$RESULTS_DIR|g" "$TEMP_SCRIPT"

echo ""
read -p "Submit array job? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting array job..."
    JOB_ID=$(sbatch "$TEMP_SCRIPT" | grep -o '[0-9]*')
    
    if [ $? -eq 0 ] && [ ! -z "$JOB_ID" ]; then
        echo "Array job submitted successfully!"
        echo "Job ID: $JOB_ID"
        
        # Save model list for this job
        MODEL_LIST_FILE="$RESULTS_DIR/model_list_${JOB_ID}.txt"
        printf "# Job ID: %s\n" "$JOB_ID" > "$MODEL_LIST_FILE"
        printf "# Submitted: %s\n" "$(date)" >> "$MODEL_LIST_FILE"
        printf "# Array Size: %d\n" "$MODEL_COUNT" >> "$MODEL_LIST_FILE"
        printf "# TaskID|ModelName|ModelFile\n" >> "$MODEL_LIST_FILE"
        for ((i=0; i<$MODEL_COUNT; i++)); do
            printf "%d|%s|%s\n" "$((i+1))" "$(basename "${MODEL_FILES[$i]}" .xml)" "${MODEL_FILES[$i]}" >> "$MODEL_LIST_FILE"
        done
        echo "Model list saved to: $MODEL_LIST_FILE"
        
        echo ""
        echo "Monitor job status with:"
        echo "  squeue -j $JOB_ID"
        echo "  sacct -j $JOB_ID"
        echo ""
        echo "View job outputs in: $RESULTS_DIR"
        echo "Log files will be named: cocoa_model_${JOB_ID}_*.out"
        echo ""
        echo "After job completion, collect performance metrics with:"
        echo "  ./collect_slurm_stats.sh $JOB_ID $RESULTS_DIR"
        echo ""
        echo "Cancel all array tasks with:"
        echo "  scancel $JOB_ID"
    else
        echo "ERROR: Failed to submit job"
        rm "$TEMP_SCRIPT"
        exit 1
    fi
else
    echo "Job submission cancelled."
fi

# Clean up temporary script
rm "$TEMP_SCRIPT"

echo ""
echo "Done."