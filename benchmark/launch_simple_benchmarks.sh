#!/bin/bash

# launch_simple_benchmarks.sh
# Simple launcher for publication benchmarks - one job per model

# Configuration
BENCHMARK_DIR="/work/schaffran1/COCOA.jl/benchmark"
RESULTS_DIR="/work/schaffran1/results_testjobs/publication_benchmarks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}COCOA.jl Simple Publication Benchmark${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Check if running on login node
if [[ "$HOSTNAME" != *"login"* ]]; then
    echo -e "${YELLOW}Warning: This script should be run from a login node${NC}"
fi

# Create directories
echo "Setting up directories..."
mkdir -p "$RESULTS_DIR"
mkdir -p "$BENCHMARK_DIR/models"

cd /work/schaffran1/COCOA.jl

# Check Julia project
if [[ ! -f "Project.toml" ]]; then
    echo -e "${RED}Error: Julia project not found at $(pwd)${NC}"
    exit 1
fi

# List of all models
MODELS=(
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

# Check all model files exist
echo "Checking model files..."
MISSING_MODELS=0
for MODEL in "${MODELS[@]}"; do
    if [[ ! -f "$BENCHMARK_DIR/models/$MODEL" ]]; then
        echo -e "${RED}  Missing: $MODEL${NC}"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    else
        echo -e "${GREEN}  Found: $MODEL${NC}"
    fi
done

if [[ $MISSING_MODELS -gt 0 ]]; then
    echo -e "${RED}Error: $MISSING_MODELS model files are missing${NC}"
    echo "Please ensure all model files are in $BENCHMARK_DIR/models/"
    exit 1
fi

# Check current queue status
echo ""
echo "Current queue status:"
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
echo "You currently have $CURRENT_JOBS jobs in the queue"

# Display job plan
echo ""
echo -e "${YELLOW}SIMPLE BENCHMARKING PLAN${NC}"
echo "=========================="
echo "Strategy: One separate job per model"
echo "Total jobs: ${#MODELS[@]}"
echo "Each job: 64 cores, 256GB RAM, 48h max time"
echo ""
echo "Models to benchmark:"
for i in "${!MODELS[@]}"; do
    echo "  Job $((i+1)): ${MODELS[$i]}"
done
echo ""
echo "Key optimizations:"
echo "  - Julia precompilation and warmup"
echo "  - Model loading excluded from computation timing"
echo "  - 3 runs per model for statistical significance"
echo "  - Memory sampling during execution"
echo "  - LIKWID performance monitoring (MEM_DP group)"
echo "  - Hardware topology analysis"

# Ask for confirmation
echo ""
echo -e "${YELLOW}Ready to submit ${#MODELS[@]} benchmark jobs${NC}"
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark submission cancelled"
    exit 0
fi

# Submit the job array
echo ""
echo "Submitting benchmark jobs..."

JOB_ID=$(sbatch --array=1-${#MODELS[@]} --parsable benchmark/slurm_simple_publication.sh)

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Successfully submitted job array: $JOB_ID${NC}"
    echo ""
    echo "Job breakdown:"
    for i in "${!MODELS[@]}"; do
        echo "  Job $JOB_ID.$((i+1)): ${MODELS[$i]}"
    done
    echo ""
    echo "Monitor your jobs with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo "  sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS"
    echo ""
    echo "View job output:"
    echo "  tail -f $RESULTS_DIR/../publication_bench_${JOB_ID}_*.out"
    echo ""
    echo "Check results:"
    echo "  ls -lht $RESULTS_DIR/"
    echo "  ls -lht $RESULTS_DIR/../likwid_results/"
    
    # Create simple monitoring script
    cat > "monitor_benchmarks.sh" << EOF
#!/bin/bash
# Simple benchmark monitor

echo "COCOA.jl Benchmark Monitor - \$(date)"
echo "Job Array: $JOB_ID"
echo "======================================"
echo ""

echo "Job Status:"
squeue -j $JOB_ID --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

echo ""
echo "Completed Results:"
ls -lht "$RESULTS_DIR"/*.jld2 2>/dev/null | head -5

echo ""
echo "LIKWID Performance Results:"
ls -lht "$RESULTS_DIR/../likwid_results/"/*.txt 2>/dev/null | head -5

echo ""
echo "Latest Output (last 3 lines from each active job):"
for task_id in {1..${#MODELS[@]}}; do
    output_file="$RESULTS_DIR/../publication_bench_${JOB_ID}_\${task_id}.out"
    if [[ -f "\$output_file" ]]; then
        model="\${MODELS[\$((task_id-1))]}"
        echo "Task \$task_id (\$model):"
        tail -n 3 "\$output_file" 2>/dev/null | sed 's/^/  /'
        echo ""
    fi
done

echo "Refresh: watch -n 30 ./monitor_benchmarks.sh"
EOF

    chmod +x monitor_benchmarks.sh
    
    echo "Real-time monitoring: ./monitor_benchmarks.sh"
    
else
    echo -e "${RED}Failed to submit jobs${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Simple benchmark submission complete!${NC}"
echo ""
echo "This approach:"
echo "  ✓ One job per model (simple and reliable)"
echo "  ✓ Startup overhead eliminated via warmup"
echo "  ✓ Pure computation time measurement"
echo "  ✓ Statistical robustness (3 runs per model)"
echo "  ✓ No complex job scheduling logic"
echo "  ✓ LIKWID performance profiling (MEM_DP counters)"
echo "  ✓ Hardware topology and NUMA analysis"
echo ""
echo "Performance monitoring includes:"
echo "  - Memory bandwidth and data volume"
echo "  - Cache performance (L2/L3 miss rates)"
echo "  - NUMA topology analysis"
echo "  - Floating-point operations (when available)"
echo ""
echo "Total expected runtime: 48 hours max per model"
echo "Results will be in: $RESULTS_DIR"
echo "LIKWID results will be in: $RESULTS_DIR/../likwid_results/"
