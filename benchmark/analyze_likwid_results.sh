#!/bin/bash

# analyze_likwid_results.sh
# Analyze LIKWID performance monitoring results from publication benchmarks

LIKWID_DIR="/work/schaffran1/results_testjobs/likwid_results"
RESULTS_DIR="/work/schaffran1/results_testjobs/publication_benchmarks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}COCOA.jl LIKWID Performance Analysis${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

if [[ ! -d "$LIKWID_DIR" ]]; then
    echo -e "${RED}Error: LIKWID results directory not found: $LIKWID_DIR${NC}"
    exit 1
fi

# Find all LIKWID output files
LIKWID_FILES=($(find "$LIKWID_DIR" -name "likwid_*.txt" -type f))

if [[ ${#LIKWID_FILES[@]} -eq 0 ]]; then
    echo -e "${YELLOW}No LIKWID output files found in $LIKWID_DIR${NC}"
    echo "Make sure your benchmarks have completed successfully."
    exit 0
fi

echo "Found ${#LIKWID_FILES[@]} LIKWID result files"
echo ""

# Create comprehensive summary
SUMMARY_FILE="$LIKWID_DIR/comprehensive_analysis_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "COCOA.jl LIKWID Performance Analysis"
    echo "Generated: $(date)"
    echo "=========================================="
    echo ""
    
    echo "SUMMARY TABLE"
    echo "============="
    printf "%-25s %15s %15s %15s %15s\n" "Model" "Runtime(s)" "MemBW(MB/s)" "L3Miss%" "Energy(J)"
    echo "-------------------------------------------------------------------------------------"
} > "$SUMMARY_FILE"

# Analyze each file
for file in "${LIKWID_FILES[@]}"; do
    filename=$(basename "$file")
    
    # Extract job info from filename (format: likwid_JOBID_TASKID.txt)
    if [[ $filename =~ likwid_([0-9]+)_([0-9]+)\.txt ]]; then
        job_id="${BASH_REMATCH[1]}"
        task_id="${BASH_REMATCH[2]}"
        
        # Try to match with model (you may need to adjust this based on your array setup)
        models=("e_coli_core" "ecoli567_splt_prpd" "iJR904" "iAF1260" "iML1515" 
                "iJF4097_splt_prpd" "iAF12599_splt_prpd" "iML15211_splt_prpd" "iML28686_splt_prpd")
        
        if [[ $task_id -ge 1 && $task_id -le ${#models[@]} ]]; then
            model_name="${models[$((task_id-1))]}"
        else
            model_name="unknown_task_$task_id"
        fi
    else
        model_name="unknown_$(basename "$file" .txt)"
    fi
    
    echo -e "${BLUE}Analyzing: $model_name${NC}"
    
    if [[ ! -f "$file" ]] || [[ ! -s "$file" ]]; then
        echo -e "${RED}  Warning: File is empty or not found${NC}"
        continue
    fi
    
    # Extract key metrics (adjust patterns based on actual LIKWID output)
    runtime=$(grep -E "Runtime.*[0-9]+\.[0-9]+" "$file" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    mem_bw=$(grep -E "Memory bandwidth.*[0-9]+\.[0-9]+" "$file" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    l3_miss=$(grep -E "L3.*miss.*rate.*[0-9]+\.[0-9]+" "$file" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    energy=$(grep -E "Energy.*[0-9]+\.[0-9]+" "$file" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    # Default values if not found
    runtime=${runtime:-"N/A"}
    mem_bw=${mem_bw:-"N/A"}
    l3_miss=${l3_miss:-"N/A"}
    energy=${energy:-"N/A"}
    
    # Print to console
    echo "  Runtime: $runtime s"
    echo "  Memory Bandwidth: $mem_bw MB/s"
    echo "  L3 Cache Miss Rate: $l3_miss %"
    echo "  Energy: $energy J"
    echo ""
    
    # Add to summary file
    printf "%-25s %15s %15s %15s %15s\n" "$model_name" "$runtime" "$mem_bw" "$l3_miss" "$energy" >> "$SUMMARY_FILE"
done

# Add detailed analysis to summary
{
    echo ""
    echo ""
    echo "DETAILED ANALYSIS"
    echo "================="
    echo ""
    
    for file in "${LIKWID_FILES[@]}"; do
        filename=$(basename "$file")
        echo "File: $filename"
        echo "----------------------------------------"
        
        # Extract most relevant sections
        echo "Performance Summary:"
        grep -A 5 -B 5 -E "Runtime|Memory bandwidth|Cache.*miss|Energy|FLOP" "$file" | head -20
        
        echo ""
        echo "Memory Metrics:"
        grep -E "Memory|Cache|L2|L3" "$file" | head -10
        
        echo ""
        echo "=========================================="
        echo ""
    done
} >> "$SUMMARY_FILE"

echo -e "${GREEN}Analysis complete!${NC}"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "Quick overview:"
cat "$SUMMARY_FILE" | head -20

echo ""
echo -e "${YELLOW}Tips for interpretation:${NC}"
echo "- Runtime: Total execution time (includes Julia startup)"
echo "- Memory Bandwidth: How efficiently memory is used"
echo "- L3 Cache Miss Rate: Higher values indicate memory-bound workloads"
echo "- Energy: Total energy consumed during execution"
echo ""
echo "For detailed analysis, examine: $SUMMARY_FILE"

# Optional: Create CSV for easy plotting
CSV_FILE="$LIKWID_DIR/performance_summary.csv"
{
    echo "Model,Runtime_s,MemoryBandwidth_MBs,L3MissRate_percent,Energy_J"
    grep -v -E "^(Model|----)" "$SUMMARY_FILE" | grep -E "^[a-zA-Z]" | tr -s ' ' ',' | sed 's/N\/A//g'
} > "$CSV_FILE"

echo "CSV data for plotting: $CSV_FILE"
