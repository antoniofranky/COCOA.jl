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

# COBRA's `rules` field writes gene rules by index ("x(3) & x(7)"); MATFBCModels reads it
# only as a stringy rule. Translate it via `genes`, or drop it when there is no gene list
# (as in the application note's elementary-step models). Concordance does not use genes.
function _mat_rules!(d::AbstractDict)
    haskey(d, "rules") && !haskey(d, "grRules") || return d
    if !haskey(d, "genes")
        delete!(d, "rules")
        return d
    end
    genes = vec(d["genes"])
    tr(r) = r isa AbstractString ?
        replace(r, r"x\((\d+)\)" => m -> genes[parse(Int, m[3:end-1])], "&" => "and", "|" => "or") : r
    d["grRules"] = map(tr, d["rules"])
    delete!(d, "rules")
    return d
end

# MATFBCModels requires some fields that minimal COBRA structs (e.g. the application note's
# elementary-step models) do not have; fill them with "unknown".
function _mat_optional_fields!(d::AbstractDict)
    nm = length(d["mets"])
    haskey(d, "genes") || (d["genes"] = Matrix{Any}(undef, 0, 1))
    any(haskey(d, k) for k in ("metFormula", "metFormulas")) || (d["metFormulas"] = fill("", nm, 1))
    any(haskey(d, k) for k in ("metCharge", "metCharges")) || (d["metCharges"] = fill(NaN, nm, 1))
    haskey(d, "metNames") || (d["metNames"] = copy(d["mets"]))
    haskey(d, "rxnNames") || (d["rxnNames"] = copy(d["rxns"]))
    haskey(d, "b") || (d["b"] = zeros(nm, 1))
    return d
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
        d = if mat_key === nothing
            # a .mat file is a MATLAB workspace: keep the variables that are COBRA models
            models = Dict{String,Any}()
            for k in keys(f)
                v = MATFBCModels.MAT.read(f, k)
                v isa AbstractDict && haskey(v, "S") && haskey(v, "rxns") && (models[k] = v)
            end
            isempty(models) && error("$path holds no COBRA model (a struct with S and rxns)")
            if length(models) > 1
                MATFBCModels.MAT.close(f)
                list = join(("$k ($(length(models[k]["rxns"])) reactions)" for k in sort(collect(keys(models)))), ", ")
                error("$path holds several models: $list. Choose one with mat_key " *
                      "(CLI: --mat-key NAME)")
            end
            mat_key = only(keys(models))
            only(values(models))
        else
            haskey(f, mat_key) || error("$path has no variable '$mat_key' " *
                                        "(variables: $(join(sort(collect(keys(f))), ", ")))")
            MATFBCModels.MAT.read(f, mat_key)
        end
        MATFBCModels.MAT.close(f)
        key = mat_key
        _mat_rules!(d)
        _mat_optional_fields!(d)
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
    return _sbml_safe_ids!(convert(A.CanonicalModel.Model, m))
end

# SBMLFBCModels prefixes metabolite ids with M_ (reactions with R_) on writing, and SBML ids
# allow only [A-Za-z0-9_]. Doing both at load makes the in-memory model, the saved SBML and
# the ids in the results identical; other characters are escaped as in the yeast GEMs
# ("[c]" -> "__91__c__93__").
_sid_escape(id::AbstractString) =
    replace(id, r"[^A-Za-z0-9_]" => c -> "__$(Int(only(c)))__")   # "[c]" -> "__91__c__93__"

# Results show the ids as the model wrote them: the escapes of `_sid_escape` are undone
# ("__91__c__93__" -> "[c]"). Only characters SBML forbids are decoded, so an id's own
# "__95__" (an allowed '_') stays as it is.
function _sid_unescape(id::AbstractString)
    return replace(id, r"__(\d{1,7})__" => function (s)
        n = parse(Int, s[3:end-2])
        (n <= 0x10ffff && isvalid(Char, n) && !occursin(r"^[A-Za-z0-9_]$", string(Char(n)))) ?
            string(Char(n)) : s
    end)
end

"""The columns of a result table with string ids decoded by `_sid_unescape`."""
readable_ids(nt::NamedTuple) =
    map(c -> eltype(c) <: AbstractString ? _sid_unescape.(c) :
             eltype(c) <: Symbol ? Symbol.(_sid_unescape.(String.(c))) : c, nt)

function _sbml_safe_ids!(model::A.CanonicalModel.Model)
    ren = Dict{String,String}()
    for id in keys(model.metabolites)
        new = _sid_escape(id)
        startswith(new, "M_") || (new = "M_" * new)
        new == id || (ren[id] = new)
    end
    for (old, new) in ren
        model.metabolites[new] = pop!(model.metabolites, old)
    end
    for rxn in values(model.reactions)
        for old in collect(keys(rxn.stoichiometry))
            haskey(ren, old) && (rxn.stoichiometry[ren[old]] = pop!(rxn.stoichiometry, old))
        end
    end
    # reactions get R_ for the same reason; the writer would add it to the reaction but not
    # to the objective's reference, leaving a file that does not load
    for id in collect(keys(model.reactions))
        new = _sid_escape(id)
        startswith(new, "R_") || (new = "R_" * new)
        new == id || (model.reactions[new] = pop!(model.reactions, id))
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
