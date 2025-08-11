#!/bin/bash
#SBATCH --job-name=test_likwid
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/work/schaffran1/results_testjobs/test_likwid_%j.out

# Test script to debug LIKWID issues

echo "=== LIKWID DEBUG TEST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# Load modules
echo "Loading modules..."
module load arch/r1/zen4
module load linux-rocky9-zen4/gcc-14.2.0/likwid/5.3.0-gose7xd

echo "Modules loaded successfully"
echo ""

# Check SLURM environment
echo "=== SLURM ENVIRONMENT ==="
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo ""

# Check current process affinity
echo "=== PROCESS AFFINITY ==="
TASKSET_OUTPUT=$(taskset -p $$ 2>/dev/null || echo "taskset failed")
echo "Current process affinity: $TASKSET_OUTPUT"
echo ""

# Check available CPUs
echo "=== CPU AVAILABILITY ==="
if command -v nproc >/dev/null; then
    echo "nproc: $(nproc)"
fi

if [ -f /proc/cpuinfo ]; then
    echo "CPUs in /proc/cpuinfo: $(grep -c ^processor /proc/cpuinfo)"
fi
echo ""

# Test likwid-topology
echo "=== LIKWID TOPOLOGY ==="
likwid-topology | head -20
echo ""

# Test different likwid-perfctr approaches
echo "=== TESTING LIKWID-PERFCTR APPROACHES ==="

# Test 1: No CPU specification
echo "Test 1: No CPU specification with FLOPS_DP"
likwid-perfctr -g FLOPS_DP -m sleep 1
if [ $? -eq 0 ]; then
    echo "✓ Test 1 PASSED"
else
    echo "✗ Test 1 FAILED"
fi
echo ""

# Test 1b: Try MEM1 group
echo "Test 1b: No CPU specification with MEM1"
likwid-perfctr -g MEM1 -m sleep 1
if [ $? -eq 0 ]; then
    echo "✓ Test 1b PASSED"
else
    echo "✗ Test 1b FAILED"
fi
echo ""

# Test 2: Try with explicit CPU list (first 4 CPUs from affinity)
if [[ "$TASKSET_OUTPUT" =~ ([0-9,-]+) ]]; then
    CPU_LIST="${BASH_REMATCH[1]}"
    # Extract first few CPUs for testing
    FIRST_CPUS=$(echo "$CPU_LIST" | tr ',' '\n' | head -4 | tr '\n' ',' | sed 's/,$//')
    
    if [ -n "$FIRST_CPUS" ]; then
        echo "Test 2: Specific CPUs ($FIRST_CPUS) with FLOPS_DP"
        likwid-perfctr -C "$FIRST_CPUS" -g FLOPS_DP -m sleep 1
        if [ $? -eq 0 ]; then
            echo "✓ Test 2 PASSED"
        else
            echo "✗ Test 2 FAILED"
        fi
    else
        echo "Test 2: SKIPPED (couldn't parse CPU list)"
    fi
else
    echo "Test 2: SKIPPED (no CPU affinity found)"
fi
echo ""

# Test 3: Try socket notation if possible
echo "Test 3: Socket notation (S0:0-3) with FLOPS_DP"
likwid-perfctr -C S0:0-3 -g FLOPS_DP -m sleep 1 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Test 3 PASSED"
else
    echo "✗ Test 3 FAILED"
fi
echo ""

# Test with a simple command
echo "=== FINAL TEST WITH SIMPLE COMMAND ==="
echo "Running: echo 'Hello from likwid' with FLOPS_DP performance monitoring"

likwid-perfctr -g FLOPS_DP -m echo "Hello from likwid"
FINAL_EXIT=$?

echo ""
echo "=== SUMMARY ==="
echo "Final test exit code: $FINAL_EXIT"
if [ $FINAL_EXIT -eq 0 ]; then
    echo "✓ LIKWID appears to be working on this node"
    echo "  You can use: likwid-perfctr -g FLOPS_DP -m <your_command>"
    echo "  Or try: likwid-perfctr -g MEM1 -m <your_command> for memory analysis"
else
    echo "✗ LIKWID has issues on this node"
    echo "  Consider running without LIKWID performance monitoring"
fi

echo ""
echo "Test completed at: $(date)"
