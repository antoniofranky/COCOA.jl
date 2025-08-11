# COCOA.jl Publication Benchmarking with LIKWID

## Overview

This setup provides a simple, reliable benchmarking approach for your COCOA.jl publication with integrated LIKWID performance monitoring.

## Key Features

✓ **One job per model** - Simple and reliable, no complex scheduling
✓ **Startup overhead eliminated** - Julia precompilation and warmup
✓ **Pure computation timing** - Model loading excluded from measurements  
✓ **Statistical robustness** - 3 runs per model
✓ **LIKWID performance profiling** - Memory bandwidth, cache performance, energy
✓ **Hardware analysis** - NUMA topology and CPU architecture details

## Files Created

### Main Scripts
- `simple_publication_benchmark.jl` - Optimized benchmark script with warmup
- `slurm_simple_publication.sh` - SLURM script with LIKWID integration
- `launch_simple_benchmarks.sh` - Interactive launcher script

### Analysis Tools
- `analyze_likwid_results.sh` - Post-processing script for LIKWID data

## LIKWID Integration

### Modules Loaded
```bash
module load arch/r1/zen4
module load linux-rocky9-zen4/gcc-14.2.0/likwid/5.3.0-gose7xd
```

### Performance Monitoring
- **Group**: `MEM_DP` (Memory + Double Precision operations)
- **Cores**: All 32 cores (`S0:0-31`)
- **Output**: Detailed performance counters in `/work/schaffran1/results_testjobs/likwid_results/`

### Metrics Collected
- Memory bandwidth and data volume
- Cache performance (L2/L3 miss rates)
- Runtime information
- Energy consumption
- NUMA topology analysis

## Usage

### 1. Launch Benchmarks
```bash
cd /work/schaffran1/COCOA.jl
./benchmark/launch_simple_benchmarks.sh
```

This will:
- Check all model files exist
- Show you the execution plan
- Submit 9 separate jobs (one per model)
- Create monitoring scripts

### 2. Monitor Progress
```bash
# Use the auto-generated monitor
./monitor_benchmarks.sh

# Or manual monitoring
squeue -u $USER
ls -lht /work/schaffran1/results_testjobs/publication_benchmarks/
ls -lht /work/schaffran1/results_testjobs/likwid_results/
```

### 3. Analyze Results
```bash
# After jobs complete, analyze LIKWID performance data
./benchmark/analyze_likwid_results.sh
```

## Expected Output

### Computational Results
- **Location**: `/work/schaffran1/results_testjobs/publication_benchmarks/`
- **Format**: `.jld2` files with detailed benchmark data
- **Contains**: 
  - Execution times (3 runs per model)
  - Memory usage statistics
  - Number of complexes and modules found
  - Model statistics

### Performance Results  
- **Location**: `/work/schaffran1/results_testjobs/likwid_results/`
- **Format**: `.txt` files with detailed LIKWID output
- **Contains**:
  - Memory bandwidth utilization
  - Cache hit/miss ratios
  - Energy consumption
  - Hardware topology information

## Job Specifications

- **Resources**: 32 cores, 128GB RAM per job
- **Time limit**: 24 hours per model
- **Total jobs**: 9 (one per model)
- **Architecture**: zen4 nodes with LIKWID support

## Models Benchmarked

1. `e_coli_core.xml`
2. `ecoli567_splt_prpd.xml` 
3. `iJR904.xml`
4. `iAF1260.xml`
5. `iML1515.xml`
6. `iJF4097_splt_prpd.xml`
7. `iAF12599_splt_prpd.xml`
8. `iML15211_splt_prpd.xml`
9. `iML28686_splt_prpd.xml`

## Key Optimizations

### Julia Startup Overhead Elimination
- Pre-compilation of all packages
- Warmup run with small model
- Model loading separated from computation timing

### Performance Monitoring
- LIKWID hardware performance counters
- Memory access pattern analysis
- Cache efficiency measurement
- Energy consumption tracking

### Statistical Robustness
- 3 independent runs per model
- Mean ± standard deviation reported
- Consistent random seeds for reproducibility

## Troubleshooting

### If jobs fail:
1. Check SLURM output files: `/work/schaffran1/results_testjobs/publication_bench_*.out`
2. Verify model files exist in `/work/schaffran1/COCOA.jl/benchmark/models/`
3. Check LIKWID module loading works on compute nodes

### If LIKWID data is missing:
1. Verify the architecture modules loaded correctly
2. Check that the zen4 nodes support the requested performance counters
3. Look for error messages in the SLURM output

### If memory issues occur:
- Large models may need full 128GB
- Check memory usage in LIKWID output
- Monitor with `free -h` during execution

## Performance Analysis Tips

The LIKWID results will help you understand:

1. **Memory-bound vs CPU-bound**: High memory bandwidth utilization indicates memory-bound workloads
2. **Cache efficiency**: Low L3 miss rates suggest good locality
3. **Scaling efficiency**: Compare performance across different model sizes
4. **Energy efficiency**: Energy per complex found

This data will be valuable for your publication's performance characterization section.
