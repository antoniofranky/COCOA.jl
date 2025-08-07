using JLD2, Dates, Statistics

const RESULTS_DIR = "/work/schaffran1/results_testjobs/benchmark_results"
const OUTPUT_FILE = joinpath(RESULTS_DIR, "aggregated_benchmark_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2")

function aggregate_results()
    @info "Aggregating benchmark results from" RESULTS_DIR

    # Find all benchmark result files
    result_files = filter(f -> startswith(f, "benchmark_") && endswith(f, ".jld2") && !startswith(f, "benchmark_ERROR"), readdir(RESULTS_DIR))
    error_files = filter(f -> startswith(f, "benchmark_ERROR"), readdir(RESULTS_DIR))

    @info "Found files" n_results = length(result_files) n_errors = length(error_files)

    all_results = []
    summary_stats = []

    # Process successful results
    for file in result_files
        file_path = joinpath(RESULTS_DIR, file)
        try
            data = JLD2.load(file_path)
            push!(all_results, data)

            if haskey(data, "benchmark_stats")
                push!(summary_stats, data["benchmark_stats"])
            end

            @info "Loaded" file
        catch e
            @warn "Failed to load" file error = string(e)
        end
    end

    # Process errors
    error_info = []
    for file in error_files
        file_path = joinpath(RESULTS_DIR, file)
        try
            data = JLD2.load(file_path)
            push!(error_info, data)
            @info "Loaded error info" file
        catch e
            @warn "Failed to load error file" file error = string(e)
        end
    end

    # Sort summary by model size
    if !isempty(summary_stats)
        sort!(summary_stats, by=x -> x["n_reactions"])
    end

    # Save aggregated results
    JLD2.save(
        OUTPUT_FILE,
        "all_results", all_results,
        "summary_stats", summary_stats,
        "error_info", error_info,
        "aggregation_timestamp", Dates.format(now(), "yyyymmdd_HHMMSS"),
        "n_successful", length(all_results),
        "n_failed", length(error_info)
    )

    @info "Aggregated results saved to" OUTPUT_FILE

    # Print summary
    println("\n=== AGGREGATED BENCHMARK SUMMARY ===")
    for stats in summary_stats
        println("$(stats["model_file"]): $(stats["n_reactions"]) reactions, " *
                "$(round(stats["median_analysis_time_sec"], digits=1))s, " *
                "$(round(stats["peak_memory_mb"], digits=1))MB")
    end

    if !isempty(error_info)
        println("\n=== FAILED BENCHMARKS ===")
        for err in error_info
            println("$(err["model_file"]): $(err["error"])")
        end
    end

    println("=====================================")
end

aggregate_results()
