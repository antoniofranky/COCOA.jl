# Julia side of the cocoajl Python package: the pipeline steps that driver.jl runs. Usable
# from plain Julia as well:
#
#     include("CocoaPy.jl"); using .CocoaPy
#     m = CocoaPy.load_model("model.xml")
#     m, report = CocoaPy.preprocess(m; binding="ordered")
module CocoaPy

using Distributed
import COCOA
import COBREXA
import HiGHS
import JLD2
import AbstractFBCModels as A
import SBMLFBCModels
import JSONFBCModels
import MATFBCModels

const C = COBREXA.C

# --------------------------------------------------------------------------------------
# Solver settings
# --------------------------------------------------------------------------------------

"""
HiGHS settings used for every LP. `tol` is the primal/dual feasibility tolerance, the base
numerical tolerance of the whole pipeline; `time_limit` bounds a single LP (seconds).
"""
function solver_settings(; tol::Float64=1e-8, time_limit::Float64=3600.0, seed::Int=42,
                         presolve::Bool=true)
    return [
        COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", tol),
        COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", tol),
        COBREXA.set_optimizer_attribute("random_seed", seed),
        COBREXA.set_optimizer_attribute("time_limit", time_limit),
        COBREXA.set_optimizer_attribute("presolve", presolve ? "on" : "off"),
    ]
end

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

# Make duplicated entries unique (first keeps its name, then _2, _3, ...). Elementary-step
# .mat files with random binding enumerate every binding order and repeat reaction names.
function _dedup!(d::AbstractDict, field::String)
    haskey(d, field) || return 0
    arr = vec(d[field])
    seen = Dict{String,Int}()
    n = 0
    for i in eachindex(arr)
        k = get(seen, arr[i], 0) + 1
        seen[arr[i]] = k
        if k > 1
            arr[i] = "$(arr[i])_$(k)"
            n += 1
        end
    end
    d[field] = arr
    return n
end

"""
Load a model as an `AbstractFBCModels.CanonicalModel.Model`.

The format follows the extension: `.xml`/`.sbml` (SBML), `.json` (COBRApy/BiGG JSON) or
`.mat` (COBRA Toolbox). For a `.mat` file holding several models, `mat_key` selects one;
without it the only model in the file is used. Duplicated reaction ids in a `.mat` file are
made unique.
"""
function load_model(path::String; mat_key::Union{Nothing,String}=nothing)
    isfile(path) || error("model file not found: $path")
    ext = lowercase(splitext(path)[2])
    m = if ext == ".mat"
        f = MATFBCModels.MAT.matopen(path)
        keys_ = collect(keys(f))
        key = something(mat_key, length(keys_) == 1 ? only(keys_) :
            error("$path holds several models ($(join(keys_, ", "))); choose one with mat_key"))
        d = MATFBCModels.MAT.read(f, key)
        MATFBCModels.MAT.close(f)
        _dedup!(d, "rxns")
        _dedup!(d, "rxnNames")
        MATFBCModels.MATFBCModel(key, d)
    elseif ext in (".xml", ".sbml")
        A.load(SBMLFBCModels.SBMLFBCModel, path)
    elseif ext == ".json"
        A.load(JSONFBCModels.JSONFBCModel, path)
    else
        error("unknown model format '$ext' (expected .xml, .sbml, .json or .mat)")
    end
    return convert(A.CanonicalModel.Model, m)
end

# SBMLFBCModels prefixes metabolite ids with M_ on writing. Doing it before the analysis
# makes the in-memory model, the saved SBML and the complex ids in the results identical.
function _sbml_safe_ids!(model::A.CanonicalModel.Model)
    ren = Dict(id => "M_" * id for id in keys(model.metabolites) if !startswith(id, "M_"))
    isempty(ren) && return model
    for (old, new) in ren
        model.metabolites[new] = pop!(model.metabolites, old)
    end
    for rxn in values(model.reactions)
        for old in collect(keys(rxn.stoichiometry))
            haskey(ren, old) && (rxn.stoichiometry[ren[old]] = pop!(rxn.stoichiometry, old))
        end
    end
    return model
end

"""Save a model as SBML."""
save_model(model, path::String) =
    A.save(convert(SBMLFBCModels.SBMLFBCModel, _sbml_safe_ids!(deepcopy(model))), path)

model_size(model) = Dict("reactions" => length(A.reactions(model)),
                         "metabolites" => length(A.metabolites(model)))

function _optimum(model, settings)
    r = COBREXA.flux_balance_analysis(model; optimizer=HiGHS.Optimizer, settings)
    return r === nothing ? NaN : r.objective
end

# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------

"""
Prepare a genome-scale model for concordance analysis.

Steps (as in the COCOA yeast analysis):
 1. remove orphan metabolites, normalise bounds to the flux cone (lb < 0 → -1000, ub > 0 →
    1000, other bounds 0);
 2. remove blocked reactions: a reaction is blocked if its flux range stays below
    `flux_tol` while growth is held in [`growth_lo`, `growth_hi`] × optimum.
    An LP that fails even when re-solved cold stops the run (a kept blocked reaction can
    glue modules together). The removal is checked twice: no blocked reaction may be left,
    and the growth optimum must be unchanged;
 3. remove orphans again;
 4. split reactions into elementary enzyme steps: `binding = "ordered"` or `"random"`
    (fraction `random_fraction` of the enzymes random, the rest ordered), or `"none"` to keep
    the reactions as they are. Then split reversible reactions into two irreversible ones.

Skip all of it with `preprocess=false` in the Python API for a model that is already
prepared (e.g. the elementary-step .mat files of the application note).

Returns `(model, report)`; `report` holds the sizes after each step and the growth optimum
before and after blocked-reaction removal.
"""
function preprocess(model; binding::String="ordered", random_fraction::Float64=1.0,
                    flux_tol::Float64=1e-6, growth_lo::Float64=0.001, growth_hi::Float64=1.0,
                    split_seed::Int=42,
                    settings=solver_settings(), workers=Distributed.workers())
    binding in ("ordered", "random", "none") ||
        error("binding must be \"ordered\", \"random\" or \"none\", got \"$binding\"")
    report = Dict{String,Any}("binding" => binding, "flux_tol" => flux_tol,
                              "growth_window" => [growth_lo, growth_hi])
    report["input"] = model_size(model)
    m = normalize_bounds_and_orphans(model)
    report["normalized"] = model_size(m)

    opt_before = _optimum(m, settings)
    isfinite(opt_before) && opt_before > 0 ||
        error("the model cannot grow after bound normalisation (optimum $opt_before)")
    lo, hi = growth_lo, growth_hi
    kw = (optimizer=HiGHS.Optimizer, settings, workers, flux_tolerance=flux_tol,
          objective_bound=o -> C.Between(lo * o, hi * o), scheduling=:static)
    m = COCOA.remove_blocked_reactions(m; kw..., on_undecided=:error)
    left = COCOA.find_blocked_reactions(m; kw..., on_undecided=:error)
    isempty(left) || error("blocked reactions left after removal: " * join(sort(left), ", "))
    opt_after = _optimum(m, settings)
    isapprox(opt_after, opt_before; rtol=1e-6) ||
        error("blocked-reaction removal changed the growth optimum ($opt_before -> " *
              "$opt_after); flux_tol $flux_tol is too coarse for this model, try 1e-9")
    report["growth_optimum"] = opt_after
    m = COCOA.remove_orphans(m)
    report["unblocked"] = model_size(m)

    if binding != "none"
        frac = binding == "ordered" ? 0.0 : random_fraction
        m = COCOA.split_into_elementary(m; random=frac, seed=UInt(split_seed))
        report["split_seed"] = split_seed
        report["random_fraction"] = frac
    end
    m = COCOA.split_into_irreversible(m)
    _sbml_safe_ids!(m)
    report["final"] = model_size(m)
    return m, report
end

normalize_bounds_and_orphans(model) =
    model |> COCOA.remove_orphans |> COCOA.normalize_bounds

# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------

"""
Concordance analysis over the full steady-state cone (`growth_fraction = nothing`) or with
growth held at ≥ `growth_fraction` × optimum. `kwargs` go to
`COCOA.activity_concordance_analysis` unchanged.
"""
function concordance(model; growth_fraction=nothing, settings=solver_settings(),
                     workers=Distributed.workers(), kwargs...)
    bound = growth_fraction === nothing ? nothing :
            (o -> C.Between(Float64(growth_fraction) * o, o))
    return COCOA.activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, settings,
        workers, objective_bound=bound, kinetic_analysis=false, detailed_results=true,
        kwargs...)
end

"""Kinetic modules (and ACR/ACRR, which come with them) from a concordance result."""
kinetic(result, model; exhaustive::Bool=false) =
    COCOA.kinetic_analysis(result, model; efficient=!exhaustive)

# --------------------------------------------------------------------------------------
# Results → plain data
# --------------------------------------------------------------------------------------

_plain(v::Union{Integer,String,Bool,Nothing}) = v
_plain(v::AbstractFloat) = isfinite(v) ? Float64(v) : (isnan(v) ? nothing : string(v))
_plain(v::Symbol) = String(v)
_plain(v::AbstractVector) = Any[_plain(x) for x in v]
_plain(v::AbstractDict) = Dict{String,Any}(string(k) => _plain(x) for (k, x) in v)
_plain(v) = string(v)

stats(result) = _plain(result.stats)

"""Save / load a concordance result together with the exact model it was computed on."""
save_checkpoint(path::String, result, model, meta::String) =
    JLD2.save(path, "result", result, "model", model, "meta", meta; compress=true)
function load_checkpoint(path::String)
    d = JLD2.load(path)
    return d["result"], d["model"], d["meta"]
end

function versions()
    dir = pkgdir(COCOA)
    commit = try
        strip(read(`git -C $dir rev-parse --short HEAD`, String))
    catch
        ""
    end
    return Dict{String,Any}(
        "julia" => string(VERSION),
        "COCOA" => string(pkgversion(COCOA)),
        "COCOA_path" => dir,
        "COCOA_commit" => commit,
        "workers" => nworkers(),
    )
end

# CSV with minimal quoting (complex ids contain '+' and '_', never commas, but be safe).
_csv(x::Nothing) = ""
_csv(x::Bool) = x ? "true" : "false"
_csv(x::Symbol) = _csv(String(x))
_csv(x::AbstractFloat) = isnan(x) ? "NaN" : string(x)
_csv(x) = string(x)
function _csv(x::AbstractString)
    any(c -> c in (',', '"', '\n', '\r'), x) || return String(x)
    return "\"" * replace(x, "\"" => "\"\"") * "\""
end

"""Write a NamedTuple of equally long column vectors as CSV."""
function write_csv(path::String, nt::NamedTuple)
    open(path, "w") do io
        println(io, join(string.(keys(nt)), ','))
        cols = values(nt)
        n = isempty(cols) ? 0 : length(first(cols))
        for i in 1:n
            println(io, join((_csv(c[i]) for c in cols), ','))
        end
    end
end

"""One row per kinetic module (1 = largest): size and `;`-separated member complexes."""
function module_table(complexes::NamedTuple)
    members = Dict{Int,Vector{String}}()
    for (cid, km) in zip(complexes.complex_id, complexes.kinetic_module)
        km > 0 && push!(get!(members, km, String[]), string(cid))
    end
    order = sort(collect(keys(members)); by=k -> (-length(members[k]), k))
    return (kinetic_module=order, n_complexes=[length(members[k]) for k in order],
            complexes=[join(sort(members[k]), ';') for k in order])
end

example_model_path() = joinpath(pkgdir(COCOA), "test", "e_coli_core.xml")

end # module
