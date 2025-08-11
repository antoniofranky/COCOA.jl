#!/bin/bash

# launch_publication_benchmarks.sh
# Script to launch publication benchmark jobs on the HPC cluster

# Configuration
BENCHMARK_DIR="/work/schaffran1/COCOA.jl/benchmark"
RESULTS_DIR="/work/schaffran1/results_testjobs/publication_benchmarks"
SCRIPT_DIR="/work/schaffran1/COCOA.jl/scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}COCOA.jl Publication Benchmark Launcher${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on login node
if [[ "$HOSTNAME" != *"login"* ]]; then
    echo -e "${YELLOW}Warning: This script should be run from a login node${NC}"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/likwid_results"
mkdir -p "$BENCHMARK_DIR/models"
mkdir -p "$SCRIPT_DIR"

# Check if Julia project is properly set up
if [[ ! -f "/work/schaffran1/COCOA.jl/Project.toml" ]]; then
    echo -e "${RED}Error: Julia project not found at /work/schaffran1/COCOA.jl/${NC}"
    exit 1
fi

# Check if all model files exist
echo "Checking model files..."
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

# Copy benchmark scripts to the correct location
echo ""
echo "Setting up benchmark scripts..."
cp publication_benchmark.jl "$BENCHMARK_DIR/" 2>/dev/null || echo "  benchmark script already in place"
cp slurm_publication.sh "$SCRIPT_DIR/" 2>/dev/null || echo "  SLURM script already in place"

# Check current queue status
echo ""
echo "Current queue status:"
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# Count current jobs
CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
echo "You currently have $CURRENT_JOBS jobs in the queue"

# Ask for confirmation
echo ""
echo -e "${YELLOW}Ready to submit publication benchmark jobs${NC}"
echo "This will submit:"
echo "  - 9 array jobs (one per model)"
echo "  - Each using 32 cores and 128GB memory"
echo "  - Maximum runtime: 24 hours per job"
echo "  - Results will be saved to: $RESULTS_DIR"
echo ""
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark submission cancelled"
    exit 0
fi

# Submit the job array
echo ""
echo "Submitting benchmark jobs..."

JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/slurm_publication.sh")

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Successfully submitted job array: $JOB_ID${NC}"
    echo ""
    echo "Monitor your jobs with:"
    echo "  squeue -u $USER"
    echo "  sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS"
    echo ""
    echo "View job output with:"
    echo "  tail -f $RESULTS_DIR/../cocoa_benchmark_${JOB_ID}_*.out"
    echo ""
    echo "Check results with:"
    echo "  ls -lh $RESULTS_DIR/"
    echo ""
    echo "Monitor on dashboard:"
    echo "  https://monitor.hpc.uni-potsdam.de"
else
    echo -e "${RED}Failed to submit jobs${NC}"
    exit 1
fi

# Optional: Set up monitoring
echo ""
read -p "Do you want to set up automatic monitoring? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create monitoring script
    cat > "$SCRIPT_DIR/monitor_benchmarks.sh" << 'EOF'
#!/bin/bash
# Monitor benchmark progress

RESULTS_DIR="/work/schaffran1/results_testjobs/publication_benchmarks"

while true; do
    clear
    echo "COCOA.jl Benchmark Monitor - $(date)"
    echo "========================================"
    echo ""
    
    # Show job status
    echo "Active Jobs:"
    squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | grep cocoa
    
    echo ""
    echo "Completed Results:"
    ls -lht "$RESULTS_DIR"/*.jld2 2>/dev/null | head -10
    
    echo ""
    echo "Latest Log Output:"
    tail -n 5 /work/schaffran1/results_testjobs/cocoa_benchmark_*.out 2>/dev/null | tail -5
    
    echo ""
    echo "Press Ctrl+C to exit"
    
    sleep 30
done
EOF
    
    chmod +x "$SCRIPT_DIR/monitor_benchmarks.sh"
    echo -e "${GREEN}Monitoring script created: $SCRIPT_DIR/monitor_benchmarks.sh${NC}"
    echo "Run it with: $SCRIPT_DIR/monitor_benchmarks.sh"
fi

echo ""
echo -e "${GREEN}Benchmark submission complete!${NC}"
echo "Results will be available in: $RESULTS_DIR"
echo "Generate publication figures with the plot_publication.jl script after completion"