#!/bin/bash

# Collect SLURM performance statistics for array jobs
# Usage: ./collect_slurm_stats.sh <job_id> [results_directory]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job_id> [results_directory]"
    echo "Example: $0 12345 /work/schaffran1/jobresults/random_0"
    exit 1
fi

JOB_ID=$1
RESULTS_DIR=${2:-.}

echo "Collecting SLURM performance statistics for Job ID: $JOB_ID"
echo "Results directory: $RESULTS_DIR"

# Check if job mapping file exists
JOB_MAPPING_FILE="$RESULTS_DIR/job_mapping_${JOB_ID}.txt"
MODEL_LIST_FILE="$RESULTS_DIR/model_list_${JOB_ID}.txt"

# Create output CSV file
OUTPUT_CSV="$RESULTS_DIR/slurm_performance_${JOB_ID}.csv"

echo "Querying SLURM accounting database..."

# Get detailed job statistics from sacct
# Format: JobID, JobName, State, Elapsed, TotalCPU, MaxRSS, MaxVMSize, ExitCode
sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ExitCode,Start,End -P --noheader > /tmp/sacct_${JOB_ID}.txt

if [ ! -s /tmp/sacct_${JOB_ID}.txt ]; then
    echo "ERROR: No SLURM accounting data found for job $JOB_ID"
    echo "The job may be too recent or too old for the accounting database"
    rm -f /tmp/sacct_${JOB_ID}.txt
    exit 1
fi

# Create CSV header
echo "TaskID,ModelName,JobState,WallTimeElapsed,CPUTime,MaxMemoryRSS_MB,MaxMemoryVMSize_MB,ExitCode,StartTime,EndTime" > "$OUTPUT_CSV"

# Process each array task
if [ -f "$JOB_MAPPING_FILE" ]; then
    echo "Using job mapping file: $JOB_MAPPING_FILE"
    
    # Read mapping file and join with SLURM stats
    while IFS='|' read -r task_id model_name model_file timestamp; do
        # Find matching SLURM entry for this array task
        job_entry=$(grep "^${JOB_ID}_${task_id}|" /tmp/sacct_${JOB_ID}.txt | head -n 1)
        
        if [ ! -z "$job_entry" ]; then
            # Parse SLURM data
            IFS='|' read -r job_id job_name state elapsed cpu_time max_rss max_vmsize exit_code start_time end_time <<< "$job_entry"
            
            # Convert memory from KB to MB (remove K suffix and divide by 1024)
            max_rss_mb=$(echo "$max_rss" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
            max_vmsize_mb=$(echo "$max_vmsize" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
            
            # Handle empty values
            [ -z "$max_rss_mb" ] && max_rss_mb="0"
            [ -z "$max_vmsize_mb" ] && max_vmsize_mb="0"
            
            echo "${task_id},${model_name},${state},${elapsed},${cpu_time},${max_rss_mb},${max_vmsize_mb},${exit_code},${start_time},${end_time}" >> "$OUTPUT_CSV"
        else
            echo "${task_id},${model_name},NOTFOUND,,,,,," >> "$OUTPUT_CSV"
        fi
    done < "$JOB_MAPPING_FILE"
    
elif [ -f "$MODEL_LIST_FILE" ]; then
    echo "Using model list file: $MODEL_LIST_FILE"
    
    # Read model list (skip comment lines)
    while IFS='|' read -r task_id model_name model_file; do
        # Skip comment lines
        [[ "$task_id" =~ ^#.*$ ]] && continue
        
        # Find matching SLURM entry for this array task
        job_entry=$(grep "^${JOB_ID}_${task_id}|" /tmp/sacct_${JOB_ID}.txt | head -n 1)
        
        if [ ! -z "$job_entry" ]; then
            # Parse SLURM data
            IFS='|' read -r job_id job_name state elapsed cpu_time max_rss max_vmsize exit_code start_time end_time <<< "$job_entry"
            
            # Convert memory from KB to MB (remove K suffix and divide by 1024)
            max_rss_mb=$(echo "$max_rss" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
            max_vmsize_mb=$(echo "$max_vmsize" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
            
            # Handle empty values
            [ -z "$max_rss_mb" ] && max_rss_mb="0"
            [ -z "$max_vmsize_mb" ] && max_vmsize_mb="0"
            
            echo "${task_id},${model_name},${state},${elapsed},${cpu_time},${max_rss_mb},${max_vmsize_mb},${exit_code},${start_time},${end_time}" >> "$OUTPUT_CSV"
        else
            echo "${task_id},${model_name},NOTFOUND,,,,,," >> "$OUTPUT_CSV"
        fi
    done < "$MODEL_LIST_FILE"
else
    echo "WARNING: No mapping file found. Creating basic statistics without model names."
    
    # Process all array tasks without model names
    grep "^${JOB_ID}_" /tmp/sacct_${JOB_ID}.txt | while IFS='|' read -r job_id job_name state elapsed cpu_time max_rss max_vmsize exit_code start_time end_time; do
        # Extract task ID from job_id (format: JOBID_TASKID)
        task_id=$(echo "$job_id" | cut -d'_' -f2)
        
        # Convert memory from KB to MB
        max_rss_mb=$(echo "$max_rss" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
        max_vmsize_mb=$(echo "$max_vmsize" | sed 's/K$//' | awk '{printf "%.2f", $1/1024}')
        
        # Handle empty values
        [ -z "$max_rss_mb" ] && max_rss_mb="0"
        [ -z "$max_vmsize_mb" ] && max_vmsize_mb="0"
        
        echo "${task_id},Unknown,${state},${elapsed},${cpu_time},${max_rss_mb},${max_vmsize_mb},${exit_code},${start_time},${end_time}" >> "$OUTPUT_CSV"
    done
fi

# Clean up temp file
rm -f /tmp/sacct_${JOB_ID}.txt

echo ""
echo "Performance statistics saved to: $OUTPUT_CSV"
echo ""

# Print summary statistics
if [ -s "$OUTPUT_CSV" ]; then
    total_tasks=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
    completed_tasks=$(tail -n +2 "$OUTPUT_CSV" | grep -c "COMPLETED")
    failed_tasks=$(tail -n +2 "$OUTPUT_CSV" | grep -c "FAILED")
    
    echo "Summary:"
    echo "  Total tasks: $total_tasks"
    echo "  Completed: $completed_tasks"
    echo "  Failed: $failed_tasks"
    echo ""
    
    # Calculate average memory usage for completed tasks
    avg_memory=$(tail -n +2 "$OUTPUT_CSV" | grep "COMPLETED" | cut -d',' -f6 | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}')
    max_memory=$(tail -n +2 "$OUTPUT_CSV" | grep "COMPLETED" | cut -d',' -f6 | sort -n | tail -n 1)
    
    echo "  Average memory (completed): ${avg_memory} MB"
    echo "  Max memory (completed): ${max_memory} MB"
    echo ""
    
    # Show failed tasks if any
    if [ $failed_tasks -gt 0 ]; then
        echo "Failed tasks:"
        tail -n +2 "$OUTPUT_CSV" | grep "FAILED" | cut -d',' -f1,2 | while IFS=',' read -r task model; do
            echo "  Task $task: $model"
        done
        echo ""
    fi
fi

echo "Done!"
