#!/bin/bash

# ============================================================================
# Master submission script for COCOA pipeline
# 
# This script:
# 1. Submits model preparation jobs (10 seeds, array 1-10)
# 2. Automatically submits analysis jobs that depend on prep completion
# 
# Total: 10 prep jobs + 520 analysis jobs (13 models × 4 variants × 10 seeds)
# ============================================================================

set -e  # Exit on error

echo "============================================================"
echo "COCOA Pipeline Submission"
echo "============================================================"
echo ""
echo "This will submit:"
echo "  - 10 model preparation jobs (seeds 42-51)"
echo "  - 520 analysis jobs (13 models × 4 variants × 10 seeds)"
echo ""
echo "Analysis jobs will start automatically after prep completes."
echo "============================================================"
echo ""

# Submit model preparation jobs
echo "Submitting model preparation jobs..."
PREP_JOB_ID=$(sbatch --parsable /work/schaffran1/COCOA.jl/scripts/model_preparation_master.sh)

if [ -z "$PREP_JOB_ID" ]; then
    echo "ERROR: Failed to submit preparation jobs"
    exit 1
fi

echo "Model preparation jobs submitted: $PREP_JOB_ID"
echo ""

# Submit analysis jobs with dependency on preparation
# The analysis will only start after ALL preparation tasks complete successfully
echo "Submitting analysis jobs (will start after prep completes)..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$PREP_JOB_ID /work/schaffran1/COCOA.jl/scripts/analyse_models_master.sh)

if [ -z "$ANALYSIS_JOB_ID" ]; then
    echo "ERROR: Failed to submit analysis jobs"
    exit 1
fi

echo "Analysis jobs submitted: $ANALYSIS_JOB_ID"
echo ""
echo "============================================================"
echo "Pipeline submitted successfully!"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Preparation: $PREP_JOB_ID (10 tasks)"
echo "  Analysis:    $ANALYSIS_JOB_ID (520 tasks)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -j $PREP_JOB_ID"
echo "  squeue -j $ANALYSIS_JOB_ID"
echo ""
echo "Check logs in: /work/schaffran1/jobresults/master_logs/"
echo "============================================================"
