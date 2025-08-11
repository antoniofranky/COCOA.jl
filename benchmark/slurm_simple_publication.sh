#!/bin/bash
#SBATCH --job-name=cocoa_pub_simple
#SBATCH --chdir=/work/schaffran1/COCOA.jl/
#SBATCH --output=/work/schaffran1/results_testjobs/publication_bench_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=schaffran1@uni-potsdam.de
#SBATCH --hint=nomultithread
#SBATCH --array=1-9  # One job per model

# Model files array - one model per job
MODEL_FILES=(
    "e_coli_core.xml"
    "ecoli567_splt_prpd.xml"
    "iJR904.xml"
    "iAF1260.xml"
    "iML1515.xml"
    "iJF4097_splt_prpd.xml"
    "iAF12599_splt_prpd.xml"
    "iML15211_splt_prpd.xml"
    "iML28686_splt_prpd.xml"
)

MODEL_FILE="${MODEL_FILES[$((SLURM_ARRAY_TASK_ID-1))]}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load architecture and tools modules
module load arch/r1/zen4
module load linux-rocky9-zen4/gcc-14.2.0/likwid/5.3.0-gose7xd

# Create directories
mkdir -p /work/schaffran1/results_testjobs/publication_benchmarks
mkdir -p /work/schaffran1/results_testjobs/likwid_results

# Calculate heap size (75% of allocated memory)
HEAP_SIZE_GB=$(( SLURM_MEM_PER_NODE * 75 / 100 / 1024 ))
HEAP_SIZE="${HEAP_SIZE_GB}G"

# LIKWID configuration
LIKWID_OUTPUT="/work/schaffran1/results_testjobs/likwid_results/likwid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

# ============================================================================
# JULIA ENVIRONMENT OPTIMIZATION
# ============================================================================

# Thread configuration for 32 cores
export JULIA_NUM_THREADS=1  # Main thread
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# GC optimization
export JULIA_GC_PARALLEL_COLLECT=1

# Julia optimization flags
JULIA_OPTS="--project=/work/schaffran1/COCOA.jl"
JULIA_OPTS="$JULIA_OPTS -p 63"  # 63 workers + 1 main = 64 cores total
JULIA_OPTS="$JULIA_OPTS --heap-size-hint=$HEAP_SIZE"
JULIA_OPTS="$JULIA_OPTS --startup-file=no"
JULIA_OPTS="$JULIA_OPTS --history-file=no"
JULIA_OPTS="$JULIA_OPTS --compiled-modules=yes"
JULIA_OPTS="$JULIA_OPTS --optimize=2"
JULIA_OPTS="$JULIA_OPTS --inline=yes"
JULIA_OPTS="$JULIA_OPTS --check-bounds=no"

# ============================================================================
# SYSTEM INFORMATION
# ============================================================================

echo "=== SIMPLE PUBLICATION BENCHMARK ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: $MODEL_FILE"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Heap Size: $HEAP_SIZE"
echo ""

echo "=== HARDWARE INFO ==="
echo "CPU info:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket|Core"
echo ""
echo "Memory info:"
free -h
echo ""

# Get hardware topology with likwid
echo "=== LIKWID TOPOLOGY ==="
likwid-topology | head -15
echo ""

echo "=== NUMA CONFIGURATION ==="
if command -v numactl &> /dev/null; then
    numactl --hardware | head -10
fi
echo ""
echo "Memory info:"
free -h
echo ""

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

cd /work/schaffran1/COCOA.jl

MODEL_PATH="/work/schaffran1/COCOA.jl/benchmark/models/$MODEL_FILE"
BENCHMARK_SCRIPT="/work/schaffran1/COCOA.jl/benchmark/simple_publication_benchmark.jl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found: $BENCHMARK_SCRIPT"
    exit 1
fi

echo "=== Starting Benchmark: $MODEL_FILE ==="
echo "Start time: $(date)"
echo ""

# Configure LIKWID performance monitoring
PERFORMANCE_GROUP="FLOPS_DP"  # Double-precision floating point operations

# Test available performance groups
echo "Testing LIKWID configuration..."
echo "Available performance groups:"
likwid-perfctr -a 2>/dev/null | grep -E "(MEM|FLOPS|CACHE)" | head -5 || echo "  No performance groups available"

# Test if LIKWID has permission to access performance counters
echo ""
echo "Testing LIKWID permissions with group: $PERFORMANCE_GROUP"
echo "Testing likwid-pin CPU affinity..."

# Test likwid-pin first (this usually works)
if likwid-pin -c 0 echo "Pin test successful" >/dev/null 2>&1; then
    echo "✓ likwid-pin available for CPU affinity"
    PIN_AVAILABLE=1
else
    echo "⚠ likwid-pin not available"
    PIN_AVAILABLE=0
fi

# Test performance counters (this often fails in HPC)
if likwid-perfctr -g "$PERFORMANCE_GROUP" -c 0 sleep 0.1 >/dev/null 2>&1; then
    echo "✓ LIKWID has permission to access performance counters"
    LIKWID_AVAILABLE=1
else
    echo "⚠ LIKWID does not have permission to access performance counters"
    echo "  This is normal in HPC environments without root access"
    echo "  Will use likwid-pin for CPU affinity only"
    LIKWID_AVAILABLE=0
fi

# Run benchmark with optimal configuration based on available LIKWID features
echo ""
echo "Starting benchmark..."
if [ "$LIKWID_AVAILABLE" -eq 1 ]; then
    echo "Using LIKWID performance monitoring (no explicit pinning to avoid cpuset conflicts)"
    echo "Performance group: $PERFORMANCE_GROUP"
    # Use performance monitoring without explicit pinning to avoid cpuset issues
    likwid-perfctr -g "$PERFORMANCE_GROUP" -o "$LIKWID_OUTPUT" julia $JULIA_OPTS "$BENCHMARK_SCRIPT" "$MODEL_FILE"
    EXIT_CODE=$?
    LIKWID_SUCCESS=1
else
    echo "Running without LIKWID performance monitoring"
    # No LIKWID features available
    julia $JULIA_OPTS "$BENCHMARK_SCRIPT" "$MODEL_FILE" 2>&1 | tee -a "$LIKWID_OUTPUT"
    EXIT_CODE=$?
    LIKWID_SUCCESS=0
fi

echo ""
echo "Benchmark completed with exit code: $EXIT_CODE"

# ============================================================================
# LIKWID PERFORMANCE ANALYSIS
# ============================================================================

if [[ -f "$LIKWID_OUTPUT" && $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "=== LIKWID PERFORMANCE SUMMARY ==="
    
    # Extract key performance metrics
    echo "Memory Performance:"
    grep -E "Memory bandwidth|Memory data volume" "$LIKWID_OUTPUT" || true
    
    echo ""
    echo "Runtime Information:"
    grep -E "Runtime|Time" "$LIKWID_OUTPUT" | head -5 || true
    
    echo ""
    echo "Cache Performance:"
    grep -E "Cache|L2|L3" "$LIKWID_OUTPUT" | head -5 || true
    
    # Create summary file
    SUMMARY_FILE="/work/schaffran1/results_testjobs/likwid_results/summary_${MODEL_FILE%.xml}_${SLURM_ARRAY_TASK_ID}.txt"
    {
        echo "=== LIKWID SUMMARY FOR $MODEL_FILE ==="
        echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        echo "Node: $SLURMD_NODENAME"
        echo "Date: $(date)"
        echo ""
        echo "Key Metrics:"
        grep -E "Runtime|Memory bandwidth|Cache miss|Energy|FLOP" "$LIKWID_OUTPUT" | head -10 || true
    } > "$SUMMARY_FILE"
    
    echo "Detailed likwid results: $LIKWID_OUTPUT"
    echo "Summary saved to: $SUMMARY_FILE"
else
    echo "No likwid output available (exit code: $EXIT_CODE)"
fi

# ============================================================================
# CLEANUP AND REPORTING
# ============================================================================

echo ""
echo "=== BENCHMARK SUMMARY ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL_FILE"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"

if [ "$LIKWID_SUCCESS" -eq 1 ]; then
    echo "LIKWID monitoring: ✓ SUCCESS (data in $LIKWID_OUTPUT)"
    echo "Performance group: $PERFORMANCE_GROUP"
else
    echo "LIKWID monitoring: ✗ DISABLED (permission issues)"
    echo "Note: Benchmark timing and memory data still collected via Julia GC"
fi
echo "Exit Code: $EXIT_CODE"
echo "Completion Time: $(date)"

# Log completion
echo "Job completed for $MODEL_FILE at $(date)" >> /work/schaffran1/results_testjobs/completion_log.txt

exit $EXIT_CODE
