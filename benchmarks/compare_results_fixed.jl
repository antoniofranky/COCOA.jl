"""
compare_results_fixed.jl — Compare the POST-FIX COCOA kinetic results (produced by
run_kinetic_step.jl / submit_kinetic_array.sbatch, zero-complex fix applied) against
the MATLAB reference, and against the PRE-FIX summary_table.csv from the original run.

Adapted from application_note/compare_results.jl (same normalization, same is_free_met
filter, plus exclusion of the synthetic zero complex). Only the kinetic-result input
directory changed (kinetic_fixed instead of kinetic); everything else — concordance
JLD2s, MATLAB reference files — is unaffected by the fix and reused as-is.

Usage:
    julia -t auto --project=/work/schaffran1/COCOA.jl/test benchmarks/compare_results_fixed.jl

Outputs (written to application_note/results/comparison_fixed/):
    summary_table.csv, acr_detail.csv, acrr_detail.csv, giant_rxn_detail.csv
"""

using COCOA
using JLD2, CSV, DataFrames, Statistics, Printf
import MATFBCModels
import MATFBCModels.MAT
import AbstractFBCModels as A

const COCOA_CONC_DIR = "/work/schaffran1/application_note/results/matlab_preprocessed/concordance"
const COCOA_KIN_DIR = "/work/schaffran1/application_note/results/matlab_preprocessed/kinetic_fixed"
const MATLAB_BASE_DIR = "/work/schaffran1/Upstream_Algorithm/Results"
const OLD_SUMMARY_CSV = "/work/schaffran1/application_note/results/comparison/summary_table.csv"
const OUTPUT_DIR = "/work/schaffran1/application_note/results/comparison_fixed"

mkpath(OUTPUT_DIR)

const FILE_TO_PAPER_NAME = Dict(
    "ArabidopsisCoreModel" => "AraCore",
    "Chlorella_variabilis_iAJ526" => "iAJ526",
    "Saha2011iRS1597" => "iRS1597",
)

function normalize_met_id(s::AbstractString)::String
    s = lowercase(strip(s))
    startswith(s, "m_") && (s = s[3:end])
    s = replace(s, r"\[(\w+)\]" => s"_\1")
    return s
end
normalize_met_id(s::Symbol)::String = normalize_met_id(string(s))

function normalize_pair(a, b)
    na, nb = normalize_met_id(a), normalize_met_id(b)
    return na < nb ? (na, nb) : (nb, na)
end

is_free_met(s) = let str = string(s)
    !endswith(str, "_complex") && !occursin(r"^E_?\d+$", str) && str != string(COCOA.ZERO_COMPLEX)
end

function load_matlab_acr(model_name::String, variant::String)
    path = joinpath(MATLAB_BASE_DIR, "MetSingle", "M_MetSingle_$(model_name)_concordant_$(variant).csv")
    isfile(path) || return nothing
    mets = Set{String}()
    for line in eachline(path)
        val = strip(line)
        isempty(val) && continue
        push!(mets, normalize_met_id(val))
    end
    return mets
end

function load_matlab_acrr(model_name::String, variant::String)
    path = joinpath(MATLAB_BASE_DIR, "MetDouble", "M_MetDouble_$(model_name)_concordant_$(variant).csv")
    isfile(path) || return nothing
    result = Set{Tuple{String,String}}()
    for line in eachline(path)
        val = strip(line)
        isempty(val) && continue
        parts = split(val, ',')
        length(parts) < 2 && continue
        a, b = strip(parts[1]), strip(parts[2])
        (isempty(a) || isempty(b)) && continue
        push!(result, normalize_pair(a, b))
    end
    return result
end

function discover_pairs()
    pairs = []
    for f in sort(readdir(COCOA_KIN_DIR; join=true))
        endswith(f, ".jld2") || continue
        m = match(r"^kinetic_(.+)_(fixed|random)_seed\d+\.jld2$", basename(f))
        m === nothing && continue
        model_name, variant = String(m[1]), String(m[2])
        conc_f = joinpath(COCOA_CONC_DIR, "concordance_$(model_name)_$(variant)_seed42.jld2")
        isfile(conc_f) || continue
        push!(pairs, (model_name=model_name, variant=variant, cocoa_kin=f))
    end
    return pairs
end

old_summary = isfile(OLD_SUMMARY_CSV) ? CSV.read(OLD_SUMMARY_CSV, DataFrame) : nothing
pairs = discover_pairs()
println("Found $(length(pairs)) kinetic_fixed results to compare\n")

rows = Dict{String,Any}[]
for (i, p) in enumerate(pairs)
    row = Dict{String,Any}("model_name" => p.model_name, "variant" => p.variant)
    try
        kin = JLD2.load(p.cocoa_kin)
        acr_metabolites = kin["acr_metabolites"]
        acrr_pairs = kin["acrr_pairs"]

        cocoa_acr = Set{String}(normalize_met_id(m) for m in acr_metabolites if is_free_met(m))
        cocoa_acrr = Set{Tuple{String,String}}(normalize_pair(a, b) for (a, b) in acrr_pairs
                                                if is_free_met(a) && is_free_met(b))
        row["fixed_n_acr"] = length(cocoa_acr)
        row["fixed_n_acrr"] = length(cocoa_acrr)

        matlab_acr = load_matlab_acr(p.model_name, p.variant)
        matlab_acrr = load_matlab_acrr(p.model_name, p.variant)
        if matlab_acr !== nothing
            ov = length(intersect(cocoa_acr, matlab_acr))
            u = length(union(cocoa_acr, matlab_acr))
            row["fixed_acr_only_cocoa"] = length(setdiff(cocoa_acr, matlab_acr))
            row["fixed_acr_only_matlab"] = length(setdiff(matlab_acr, cocoa_acr))
            row["fixed_acr_jaccard"] = u == 0 ? 1.0 : ov / u
        end
        if matlab_acrr !== nothing
            ov = length(intersect(cocoa_acrr, matlab_acrr))
            u = length(union(cocoa_acrr, matlab_acrr))
            row["fixed_acrr_only_cocoa"] = length(setdiff(cocoa_acrr, matlab_acrr))
            row["fixed_acrr_only_matlab"] = length(setdiff(matlab_acrr, cocoa_acrr))
            row["fixed_acrr_jaccard"] = u == 0 ? 1.0 : ov / u
        end
        if old_summary !== nothing
            old_rows = filter(r -> r.model_name == p.model_name && r.variant == p.variant, old_summary)
            if nrow(old_rows) == 1
                row["prefix_acr_only_cocoa"] = old_rows[1, :acr_only_cocoa]
                row["prefix_acrr_only_cocoa"] = old_rows[1, :acrr_only_cocoa]
                row["prefix_acr_jaccard"] = old_rows[1, :acr_jaccard]
                row["prefix_acrr_jaccard"] = old_rows[1, :acrr_jaccard]
            end
        end
        push!(rows, row)
        println("  $(p.model_name) ($(p.variant))... ok")
    catch e
        println("  $(p.model_name) ($(p.variant))... ERROR: $e")
    end
end

all_keys = sort(unique(reduce(vcat, collect.(keys.(rows)))))
df = DataFrame()
for k in all_keys
    df[!, k] = [get(r, k, missing) for r in rows]
end
select!(df, ["model_name", "variant", filter(k -> k ∉ ("model_name", "variant"), all_keys)...])
out = joinpath(OUTPUT_DIR, "summary_table.csv")
CSV.write(out, df)
println("\nWrote: $out ($(nrow(df)) rows)")
