"""
collect_results.jl — summarise Run A (and Run B) kinetic outputs into one CSV.

Reads every kinetic_*.jld2 in OUT_DIR (Run A) and, if present, kineticB_*.jld2 in OUT_DIR_B
(Run B), and writes SUMMARY_CSV with per-model giant-module size, #ACR, #ACRR, and (for
Run B) the reference-concordance match_rate. Env: OUT_DIR, OUT_DIR_B (optional), SUMMARY_CSV.
"""
using JLD2

const OUT_DIR = get(ENV, "OUT_DIR", "/work/schaffran1/kinetic_rerun/out_A")
const OUT_DIR_B = get(ENV, "OUT_DIR_B", "")
const SUMMARY_CSV = get(ENV, "SUMMARY_CSV", "/work/schaffran1/kinetic_rerun/kinetic_summary.csv")

giant(km) = isempty(km) ? 0 : maximum(length, km)

function row_for(path)
    d = JLD2.load(path)
    (giant(d["kinetic_modules"]), length(d["acr_metabolites"]), length(d["acrr_pairs"]),
     get(d, "match_rate", missing))
end

# key "<model>_<variant>" from a filename like kinetic_<model>_<variant>_seed42.jld2
function key_of(fname, prefix)
    s = replace(fname, prefix => "", "_seed42.jld2" => "")
    return s
end

rows = Dict{String,Any}()
for f in readdir(OUT_DIR)
    startswith(f, "kinetic_") && endswith(f, ".jld2") || continue
    g, nacr, nacrr, _ = row_for(joinpath(OUT_DIR, f))
    rows[key_of(f, "kinetic_")] = Dict("giantA"=>g, "acrA"=>nacr, "acrrA"=>nacrr)
end
if !isempty(OUT_DIR_B) && isdir(OUT_DIR_B)
    for f in readdir(OUT_DIR_B)
        startswith(f, "kineticB_") && endswith(f, ".jld2") || continue
        g, nacr, nacrr, mr = row_for(joinpath(OUT_DIR_B, f))
        k = key_of(f, "kineticB_")
        r = get!(rows, k, Dict{String,Any}())
        r["giantB"] = g; r["acrB"] = nacr; r["acrrB"] = nacrr; r["matchrate"] = mr
    end
end

open(SUMMARY_CSV, "w") do io
    println(io, "model_variant,giant_A,acr_A,acrr_A,giant_B,acr_B,acrr_B,ref_match_rate")
    for k in sort(collect(keys(rows)))
        r = rows[k]
        g(x) = get(r, x, "")
        println(io, join([k, g("giantA"), g("acrA"), g("acrrA"),
                          g("giantB"), g("acrB"), g("acrrB"), g("matchrate")], ","))
    end
end
println("Wrote $SUMMARY_CSV  ($(length(rows)) models)")
println("Run A outputs: $(count(r->haskey(r,"giantA"), values(rows)))   ",
        "Run B outputs: $(count(r->haskey(r,"giantB"), values(rows)))")
