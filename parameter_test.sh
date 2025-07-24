#!/bin/bash
#SBATCH --job-name=cocoa_parameter_test
#SBATCH --chdir=/work/schaffran1/results_testjobs
#SBATCH --output=results_testjobs/cocoa_parameter_test_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread

# COCOA.jl Parameter Testing Script
# This script runs comprehensive parameter testing for concordance analysis
# to find optimal settings for large models (50k+ complexes and reactions)

echo "=================================================="
echo "COCOA.jl Parameter Testing Suite"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "=================================================="

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="/work/schaffran1/parameter_test_results"
RUN_DIR="${RESULTS_BASE}/run_${TIMESTAMP}"

# Create directories
mkdir -p "$RUN_DIR"
mkdir -p "${RESULTS_BASE}/logs"

# Define paths
COCOA_DIR="/work/schaffran1/COCOA.jl"
SCRIPT_DIR="//work/schaffran1/COCOA.jl"
PARAMETER_SCRIPT="${SCRIPT_DIR}/parameter_test.jl"
LOG_FILE="${RESULTS_BASE}/logs/parameter_test_${TIMESTAMP}.log"

# Calculate heap size hint (80% of available memory, matching analyse_model.sh)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 80 / 100 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# Julia performance optimization flags
JULIA_OPTS="--project=$COCOA_DIR"
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"           # Skip .julia/config/startup.jl
JULIA_OPTS="$JULIA_OPTS --history-file=no"           # Disable history file
JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"      # Use precompiled modules
JULIA_OPTS="$JULIA_OPTS --optimize=2"                # Enable optimizations
JULIA_OPTS="$JULIA_OPTS --check-bounds=no"           # Disable bounds checking for speed
JULIA_OPTS="$JULIA_OPTS -p 31 -t auto"

echo "Configuration:"
echo "  - COCOA directory: $COCOA_DIR"
echo "  - Parameter script: $PARAMETER_SCRIPT"
echo "  - Results directory: $RUN_DIR"
echo "  - Log file: $LOG_FILE"
echo "  - Heap size: $HEAP_SIZE"
echo "  - Workers: 31 (leaving 1 CPU for coordination)"
echo "  - Julia options: $JULIA_OPTS"
echo "=================================================="

# Change to COCOA directory
cd "$COCOA_DIR" || {
    echo "ERROR: Could not change to COCOA directory: $COCOA_DIR"
    exit 1
}

# Verify that the parameter test script exists
if [[ ! -f "$PARAMETER_SCRIPT" ]]; then
    echo "ERROR: Parameter test script not found: $PARAMETER_SCRIPT"
    exit 1
fi

# Set up environment variables for the Julia script
export RESULTS_DIR="$RUN_DIR"
export JULIA_NUM_THREADS="auto"

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "=================================================="
    echo "Cleaning up..."
    
    # Kill any remaining Julia processes
    pkill -f "julia.*parameter_test.jl" 2>/dev/null || true
    
    # Final results summary
    if [[ -d "$RUN_DIR" ]]; then
        echo "Results directory: $RUN_DIR"
        echo "Files created:"
        find "$RUN_DIR" -type f -name "*.json" -o -name "*.csv" | sort
        
        # Count results
        JSON_FILES=$(find "$RUN_DIR" -name "*.json" | wc -l)
        CSV_FILES=$(find "$RUN_DIR" -name "*.csv" | wc -l)
        echo "  - JSON files: $JSON_FILES"
        echo "  - CSV files: $CSV_FILES"
    fi
    
    echo "End time: $(date)"
    echo "=================================================="
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Function to monitor system resources
monitor_resources() {
    local log_file="$1"
    while true; do
        {
            echo "=== Resource Monitor $(date) ==="
            echo "Memory usage:"
            free -h
            echo ""
            echo "CPU usage:"
            top -bn1 | grep "Cpu(s)" 
            echo ""
            echo "Julia processes:"
            ps aux | grep julia | grep -v grep | wc -l
            echo ""
        } >> "$log_file"
        sleep 300  # Monitor every 5 minutes
    done
}

# Start resource monitoring in background
monitor_resources "$LOG_FILE" &
MONITOR_PID=$!

# Function to run parameter tests with error handling and retries
run_parameter_tests() {
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        echo ""
        echo "=================================================="
        echo "Starting parameter testing attempt $attempt/$max_attempts"
        echo "=================================================="
        
        # Run the Julia parameter testing script with optimization flags
        echo "Executing: julia $JULIA_OPTS $PARAMETER_SCRIPT"
        
        # Capture both stdout and stderr, and handle timeouts
        timeout 11h julia $JULIA_OPTS "$PARAMETER_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
        local exit_code=${PIPESTATUS[0]}
        
        echo ""
        echo "Julia process exit code: $exit_code"
        
        case $exit_code in
            0)
                echo "✓ Parameter testing completed successfully!"
                return 0
                ;;
            124)
                echo "⚠ Parameter testing timed out (11 hours). This might be expected for comprehensive testing."
                # Check if we got partial results
                if [[ -n "$(find "$RUN_DIR" -name "*.json" -o -name "*.csv" 2>/dev/null)" ]]; then
                    echo "✓ Partial results found. Considering this a successful run."
                    return 0
                else
                    echo "✗ No results found after timeout."
                fi
                ;;
            137)
                echo "✗ Parameter testing was killed (likely out of memory)"
                ;;
            *)
                echo "✗ Parameter testing failed with exit code: $exit_code"
                ;;
        esac
        
        if [[ $attempt -lt $max_attempts ]]; then
            echo "Retrying in 60 seconds..."
            sleep 60
            
            # Clean up any hanging processes
            pkill -f "julia.*parameter_test.jl" 2>/dev/null || true
            sleep 10
            
            # Force garbage collection
            sync
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
            
            attempt=$((attempt + 1))
        else
            echo "✗ All attempts failed. Check logs for details."
            return 1
        fi
    done
}

# Additional HPC optimizations
echo "Setting up HPC optimizations..."

# Set Julia environment variables for better HPC performance
export JULIA_NUM_THREADS="auto"
export JULIA_CPU_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1                    # Prevent OpenMP conflicts
export OPENBLAS_NUM_THREADS=1               # Single-threaded BLAS for distributed computing
export MKL_NUM_THREADS=1                    # Single-threaded MKL if available
export JULIA_EXCLUSIVE=1                    # Exclusive access mode
export JULIA_DEPOT_PATH="/work/schaffran1/.julia:/opt/julia/depot"  # Optimize depot path

# Optimize memory allocation
export JULIA_GC_MEASURE_MALLOC=0           # Disable malloc measurement overhead
export JULIA_GC_PARALLEL_COLLECT=1         # Enable parallel garbage collection

# Pre-flight checks
echo "Pre-flight checks:"

# Check available memory
AVAILABLE_MEM_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
AVAILABLE_MEM_GB=$((AVAILABLE_MEM_KB / 1024 / 1024))
echo "  - Available memory: ${AVAILABLE_MEM_GB}GB"

if [[ $AVAILABLE_MEM_GB -lt 60 ]]; then
    echo "  ⚠ Warning: Low available memory. Consider requesting more memory."
fi

# Check disk space
AVAILABLE_DISK=$(df -BG "$RESULTS_BASE" | awk 'NR==2 {print $4}' | sed 's/G//')
echo "  - Available disk space: ${AVAILABLE_DISK}GB"

if [[ $AVAILABLE_DISK -lt 10 ]]; then
    echo "  ⚠ Warning: Low disk space. Results might be large."
fi

# Check Julia installation
if command -v julia &> /dev/null; then
    JULIA_VERSION=$(julia --version)
    echo "  - Julia: $JULIA_VERSION"
else
    echo "  ✗ ERROR: Julia not found in PATH"
    exit 1
fi

echo "  ✓ All pre-flight checks passed"
echo ""

# Main execution
echo "Starting parameter testing with the following strategy:"
echo "1. Test 12 different parameter configurations"
echo "2. Each configuration targets different aspects:"
echo "   - Memory usage optimization"
echo "   - Processing speed optimization" 
echo "   - Accuracy/sensitivity tuning"
echo "   - Batch and stage size variations"
echo "3. Collect comprehensive performance metrics"
echo "4. Analyze results and provide recommendations for 50k+ models"
echo ""

# Record environment information
{
    echo "=== Environment Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Working directory: $(pwd)"
    echo "Julia version: $(julia --version)"
    echo "Number of workers: 31"
    echo "Heap size: $HEAP_SIZE"
    echo "Memory limit: ${SLURM_MEM_PER_NODE}MB"
    echo "Time limit: 12 hours"
    echo ""
    echo "=== Parameter Testing Log ==="
} > "$LOG_FILE"

# Run the main parameter testing
if run_parameter_tests; then
    echo ""
    echo "=================================================="
    echo "PARAMETER TESTING COMPLETED SUCCESSFULLY!"
    echo "=================================================="
    
    # Generate summary report
    SUMMARY_FILE="${RUN_DIR}/test_summary_${TIMESTAMP}.txt"
    {
        echo "COCOA.jl Parameter Testing Summary"
        echo "=================================="
        echo "Date: $(date)"
        echo "Job ID: $SLURM_JOB_ID"
        echo "Duration: Started at $(head -n1 "$LOG_FILE" | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' || echo 'Unknown')"
        echo "         Completed at $(date)"
        echo ""
        echo "Results:"
        find "$RUN_DIR" -name "*.json" -o -name "*.csv" | while read -r file; do
            echo "  - $(basename "$file"): $(stat -c%s "$file" | numfmt --to=iec)B"
        done
        echo ""
        echo "Key findings and recommendations should be in the JSON/CSV files."
        echo "Check the final analysis output in the log file for detailed recommendations."
    } > "$SUMMARY_FILE"
    
    echo "Summary report: $SUMMARY_FILE"
    
    # If we have CSV results, provide quick statistics
    LATEST_CSV=$(find "$RUN_DIR" -name "*.csv" | sort | tail -n1)
    if [[ -n "$LATEST_CSV" && -f "$LATEST_CSV" ]]; then
        echo ""
        echo "Quick Statistics from $LATEST_CSV:"
        echo "  Total configurations tested: $(tail -n +2 "$LATEST_CSV" | wc -l)"
        echo "  Successful configurations: $(tail -n +2 "$LATEST_CSV" | awk -F, '$3=="true"' | wc -l)"
        echo "  Failed configurations: $(tail -n +2 "$LATEST_CSV" | awk -F, '$3=="false"' | wc -l)"
    fi
    
else
    echo ""
    echo "=================================================="
    echo "PARAMETER TESTING FAILED"
    echo "=================================================="
    echo "Check the log file for details: $LOG_FILE"
    
    # Still try to create a summary of what we have
    if [[ -d "$RUN_DIR" ]]; then
        PARTIAL_RESULTS=$(find "$RUN_DIR" -name "*.json" -o -name "*.csv" | wc -l)
        if [[ $PARTIAL_RESULTS -gt 0 ]]; then
            echo "Found $PARTIAL_RESULTS partial result files that might still be useful."
        fi
    fi
    
    exit 1
fi

# Stop resource monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "For large model recommendations, check the analysis output in:"
echo "  - Log file: $LOG_FILE"  
echo "  - Results directory: $RUN_DIR"
echo ""
echo "Happy parameter optimization! 🚀"