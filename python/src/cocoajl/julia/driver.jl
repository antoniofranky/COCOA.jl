# Executes one cocoajl job: `julia --project=<env> driver.jl job.json`.
#
# The Python package writes the job file (all parameters resolved, nothing left to defaults
# here) and starts this script as a separate process; results go to files. The same job file
# can be rerun by hand with the command above, without Python.
#
# Job fields:
#   command      "run" | "sweep" | "preprocess" | "kinetic" | "info"
#   workers      number of worker processes for the LPs (0 = serial)
#   model        model file;   mat_key   model inside a .mat file (or null)
#   preprocess   true: preprocess the model; false: use it as given
#   params       preprocessing + concordance parameters (see cocoajl.api)
#   extra        further keyword arguments for COCOA.activity_concordance_analysis
#   exhaustive, write_acr          kinetic-module options
#   outdir                         run / sweep / kinetic output directory
#   runs         sweep only: [{"dir": ..., "params": {...}}, ...]
#   output       preprocess only: output .xml
#   rundir       kinetic only: directory of an earlier run
using Distributed
import JSON

const JOB_FILE = only(ARGS)
const JOB = JSON.parsefile(JOB_FILE)
const T0 = time()

logmsg(msg) = (println("[", Libc.strftime("%H:%M:%S", time()), "] ", msg); flush(stdout))

function start_workers(n::Integer)
    n <= 0 && return
    have = nprocs() == 1 ? 0 : nworkers()
    n > have && addprocs(n - have; exeflags="--project=$(dirname(Base.active_project()))",
                         env=["JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1"])
    return
end

start_workers(get(JOB, "workers", 0))
logmsg("loading COCOA" * (nprocs() > 1 ? " on $(nworkers()) workers" : ""))
@everywhere import COCOA, HiGHS
include(joinpath(@__DIR__, "CocoaPy.jl"))
using .CocoaPy

pget(p, k) = p[k] === nothing ? nothing : p[k]

function settings_for(p; presolve::Bool)
    return CocoaPy.solver_settings(tol=Float64(p["solver_tol"]),
        time_limit=Float64(p["lp_time_limit"]), seed=Int(p["seed"]), presolve=presolve)
end

function do_preprocess(model, p)
    gw = p["growth_window"]
    return CocoaPy.preprocess(model; binding=String(p["binding"]),
        random_fraction=Float64(p["random_fraction"]), flux_tol=Float64(p["flux_tol"]),
        growth_lo=Float64(gw[1]), growth_hi=Float64(gw[2]), split_seed=Int(p["split_seed"]),
        settings=settings_for(p; presolve=true))
end

function write_json(path, x)
    open(path, "w") do io
        JSON.print(io, CocoaPy._plain(x), 1)
    end
end

function write_kinetic(outdir, kin, write_acr)
    CocoaPy.write_csv(joinpath(outdir, "complexes.csv"), kin.complexes)
    mods = CocoaPy.module_table(kin.complexes)
    CocoaPy.write_csv(joinpath(outdir, "kinetic_modules.csv"), mods)
    if write_acr
        CocoaPy.write_csv(joinpath(outdir, "acr.csv"), kin.acr)
        CocoaPy.write_csv(joinpath(outdir, "acrr.csv"), kin.acrr)
    end
    return Dict{String,Any}("n_kinetic_modules" => length(mods.kinetic_module),
        "largest_kinetic_module" => isempty(mods.n_complexes) ? 0 : first(mods.n_complexes),
        "n_acr" => length(kin.acr.metabolite_id), "n_acrr" => length(kin.acrr.metabolite_1))
end

"""Concordance + kinetic modules of a prepared model into `outdir`."""
function analyse(model, outdir, p, info)
    mkpath(outdir)
    t0 = time()
    CocoaPy.save_model(model, joinpath(outdir, "model_preprocessed.xml"))
    extra = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in get(JOB, "extra", Dict()))
    logmsg("concordance analysis on $(max(nworkers(), 1)) process(es) (the long step)")
    t = time()
    conc = CocoaPy.concordance(model; growth_fraction=pget(p, "growth_fraction"),
        settings=settings_for(p; presolve=false),   # presolve per pair LP costs ~20 %
        sample_size=Int(p["sample_size"]), cv_threshold=Float64(p["cv_threshold"]),
        concordance_tolerance=Float64(p["concordance_tolerance"]),
        balanced_threshold=Float64(p["balanced_threshold"]), seed=UInt(p["seed"]),
        use_transitivity=Bool(p["use_transitivity"]), extra...)
    info["seconds_concordance"] = round(time() - t; digits=1)
    st = CocoaPy.stats(conc)
    logmsg("concordance: $(st["n_concordance_modules"]) modules, " *
           "$(get(st, "n_unknown_pairs", 0)) undecided pairs")
    CocoaPy.save_checkpoint(joinpath(outdir, "concordance.jld2"), conc, model,
                            JSON.json(info["parameters"]))
    t = time()
    kin = CocoaPy.kinetic(conc, model; exhaustive=Bool(JOB["exhaustive"]))
    merge!(st, write_kinetic(outdir, kin, Bool(JOB["write_acr"])))
    st["seconds_kinetic"] = round(time() - t; digits=1)
    info["stats"] = st
    info["seconds_total"] = round(time() - t0; digits=1)
    write_json(joinpath(outdir, "run.json"), info)
    logmsg("done: $(st["n_kinetic_modules"]) kinetic modules, largest " *
           "$(st["largest_kinetic_module"]) complexes → $outdir")
end

function load_input()
    m = CocoaPy.load_model(String(JOB["model"]); mat_key=pget(JOB, "mat_key"))
    sz = CocoaPy.model_size(m)
    logmsg("loaded $(JOB["model"]): $(sz["reactions"]) reactions, $(sz["metabolites"]) metabolites")
    return m
end

function base_info(p)
    return Dict{String,Any}("model" => JOB["model"], "mat_key" => pget(JOB, "mat_key"),
        "parameters" => p, "exhaustive" => JOB["exhaustive"], "extra" => get(JOB, "extra", Dict()),
        "versions" => CocoaPy.versions(), "job" => abspath(JOB_FILE))
end

function prepared(m, p, report_ref)
    if JOB["preprocess"]
        logmsg("preprocessing (blocked reactions, $(p["binding"]) binding)")
        m2, report = do_preprocess(m, p)
        logmsg("preprocessed: $(report["final"]["reactions"]) reactions, growth optimum " *
               "$(report["growth_optimum"])")
        report_ref[] = report
        return m2
    end
    report_ref[] = "skipped (model used as given)"
    return m
end

function cmd_info()
    write_json(String(JOB["output"]), merge(CocoaPy.versions(),
        Dict("example_model" => CocoaPy.example_model_path())))
end

function cmd_preprocess()
    p = JOB["params"]
    m, report = do_preprocess(load_input(), p)
    out = String(JOB["output"])
    CocoaPy.save_model(m, out)
    write_json(splitext(out)[1] * ".report.json", report)
    logmsg("wrote $out ($(report["final"]["reactions"]) reactions)")
end

function cmd_run()
    p = JOB["params"]
    rep = Ref{Any}(nothing)
    m = prepared(load_input(), p, rep)
    info = base_info(p)
    info["preprocessing"] = rep[]
    analyse(m, String(JOB["outdir"]), p, info)
end

function cmd_sweep()
    base = nothing                                  # loaded once, on first need
    cache = Dict{String,Tuple{Any,Any}}()           # preprocessing setting → (model, report)
    runs = JOB["runs"]
    for (i, r) in enumerate(runs)
        dir = String(r["dir"])
        logmsg("sweep $i/$(length(runs)): $(basename(dir))")
        if isfile(joinpath(dir, "run.json"))
            logmsg("already done, skipped")
            continue
        end
        p = r["params"]
        key = String(r["preprocess_key"])
        if !haskey(cache, key)
            base === nothing && (base = load_input())
            rep = Ref{Any}(nothing)
            cache[key] = (prepared(base, p, rep), rep[])
        end
        m, report = cache[key]
        info = base_info(p)
        info["preprocessing"] = report
        analyse(m, dir, p, info)
    end
end

function cmd_kinetic()
    rundir = String(JOB["rundir"])
    conc, m, _ = CocoaPy.load_checkpoint(joinpath(rundir, "concordance.jld2"))
    out = String(JOB["outdir"])
    mkpath(out)
    t = time()
    kin = CocoaPy.kinetic(conc, m; exhaustive=Bool(JOB["exhaustive"]))
    st = merge(CocoaPy.stats(conc), write_kinetic(out, kin, Bool(JOB["write_acr"])))
    st["seconds_kinetic"] = round(time() - t; digits=1)
    info = Dict{String,Any}("from_run" => abspath(rundir), "exhaustive" => JOB["exhaustive"],
                            "stats" => st, "versions" => CocoaPy.versions())
    write_json(joinpath(out, out == rundir ? "kinetic.json" : "run.json"), info)
    logmsg("$(st["n_kinetic_modules"]) kinetic modules, largest $(st["largest_kinetic_module"])" *
           " complexes → $out")
end

const COMMANDS = Dict("info" => cmd_info, "preprocess" => cmd_preprocess, "run" => cmd_run,
                      "sweep" => cmd_sweep, "kinetic" => cmd_kinetic)
haskey(COMMANDS, JOB["command"]) || error("unknown command $(JOB["command"])")
COMMANDS[JOB["command"]]()
logmsg("finished in $(round(time() - T0; digits=1)) s")
