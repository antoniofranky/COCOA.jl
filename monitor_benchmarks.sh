#!/bin/bash
# Simple benchmark monitor

echo "COCOA.jl Benchmark Monitor - $(date)"
echo "Job Array: 35262"
echo "======================================"
echo ""

echo "Job Status:"
squeue -j 35262 --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

echo ""
echo "Completed Results:"
ls -lht "/work/schaffran1/results_testjobs/publication_benchmarks"/*.jld2 2>/dev/null | head -5

echo ""
echo "LIKWID Performance Results:"
ls -lht "/work/schaffran1/results_testjobs/publication_benchmarks/../likwid_results/"/*.txt 2>/dev/null | head -5

echo ""
echo "Latest Output (last 3 lines from each active job):"
for task_id in {1..9}; do
    output_file="/work/schaffran1/results_testjobs/publication_benchmarks/../publication_bench_35262_${task_id}.out"
    if [[ -f "$output_file" ]]; then
        model="${MODELS[$((task_id-1))]}"
        echo "Task $task_id ($model):"
        tail -n 3 "$output_file" 2>/dev/null | sed 's/^/  /'
        echo ""
    fi
done

echo "Refresh: watch -n 30 ./monitor_benchmarks.sh"
