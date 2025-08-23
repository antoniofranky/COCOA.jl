#!/bin/bash

# Helper script to submit COCOA array job for all models in a directory
# Usage: ./submit_array_job.sh [models_directory] [results_directory]

# Default parameters - modify as needed
DEFAULT_MODELS_DIR="/work/schaffran1/toolbox/prpd_models/ordered"
DEFAULT_RESULTS_DIR="/work/schaffran1/jobresults"
DEFAULT_EMAIL="schaffran1@uni-potsdam.de"

# Parse command line arguments
MODELS_DIR="${1:-$DEFAULT_MODELS_DIR}"
RESULTS_DIR="${2:-$DEFAULT_RESULTS_DIR}"
EMAIL="${3:-$DEFAULT_EMAIL}"

echo "COCOA Array Job Submission Script"
echo "=================================="
echo "Models directory: $MODELS_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Email: $EMAIL"

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

echo ""
echo "Job Configuration:"
echo "  Array size: 1-$MODEL_COUNT"
echo "  Max concurrent jobs: 10"
echo "  Time limit: 24 hours per job"
echo "  Memory: 128GB per job"
echo "  CPUs: 64 per job"

# Create a temporary copy of the job script with the correct array size and paths
TEMP_SCRIPT="analyse_models_array_temp_$$.sh"
cp analyse_models_array.sh "$TEMP_SCRIPT"

# Replace placeholders in the temporary script
sed -i "s|ARRAY_SIZE|$MODEL_COUNT|g" "$TEMP_SCRIPT"
sed -i "s|MODELS_DIR=\".*\"|MODELS_DIR=\"$MODELS_DIR\"|g" "$TEMP_SCRIPT"  
sed -i "s|RESULTS_BASE_DIR=\".*\"|RESULTS_BASE_DIR=\"$RESULTS_DIR\"|g" "$TEMP_SCRIPT"
sed -i "s|schaffran1@uni-potsdam.de|$EMAIL|g" "$TEMP_SCRIPT"

echo ""
read -p "Submit array job? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting array job..."
    JOB_ID=$(sbatch "$TEMP_SCRIPT" | grep -o '[0-9]*')
    
    if [ $? -eq 0 ] && [ ! -z "$JOB_ID" ]; then
        echo "Array job submitted successfully!"
        echo "Job ID: $JOB_ID"
        echo ""
        echo "Monitor job status with:"
        echo "  squeue -j $JOB_ID"
        echo "  sacct -j $JOB_ID"
        echo ""
        echo "View job outputs in: $RESULTS_DIR"
        echo "Log files will be named: cocoa_model_${JOB_ID}_*.out"
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