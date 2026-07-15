#!/usr/bin/env julia
# =====================================================================================
# Reproducibility / robustness experiments for COCOA.jl
# =====================================================================================
#
# Addresses reviewer concerns on determinism and numerical stability:
#   R1 2b1 : seed dependence; "identify and remove the source of disagreement and seed
#            dependence and report whether rounding or tighter tolerances can make
#            results deterministic and reproducible."
#   R2     : sensitivity of concordance modules / ACR / ACRR to the concordance tolerance.
#
# Four controlled experiments (default model: bundled e_coli_core). The BASELINE matches
# the manuscript: the analysis is over the FULL steady-state cone, biomass NOT forced to
# optimal (objective_bound = nothing). Experiment B additionally adds an obj-bound
# variant for contrast.
#
#   A. SEED x TRANSITIVITY     10 seeds x {use_transitivity true,false}.
#   B. BIOMASS ISOLATION       {kept+bound, kept+nobound, removed+nobound} x 3 seeds.
#   C. TOLERANCE SWEEP         concordance/balanced tol in {1e-6,1e-8,1e-10} x 3 seeds.
#   D. BLOCKED-TOLERANCE SWEEP remove_blocked_reactions flux_tolerance in the same set.
#   E. EFFICIENT=FALSE         exhaustive ACR/ACRR via the full matrix/deficiency path
#                              (kinetic_efficient=false); slow, reference-comparable.
#   F. SAMPLE-SIZE SWEEP       sample_size in {1000,5000,20000} x seeds; tests whether more
#                              ACHR sampling removes genome-scale seed-dependence (needs RAM).
#   G. CV-THRESHOLD SWEEP      cv_threshold in {0.01,0.05,0.1} x seeds; higher = more pairs go
#                              to the exact LP test -> less CV-noise-driven seed-dependence.
#                              E/F/G are not in the default set; enable via EXPERIMENTS=E / F / G.
#
# Every run records a LABEL-INVARIANT partition fingerprint of the concordance modules,
# so "same counts" is distinguished from "identical partition".
#
# ------------------------------------------------------------------------------------
# RUN MODES
#   1) Whole grid, one process (small models):
#        NPROCS=15 julia --project=examples/reproducibility robustness_experiments.jl
#   2) SLURM job array (one config per task -> full parallelism, wall-clock ~= one run):
#        COUNT_ONLY=1 julia ... robustness_experiments.jl        # prints N configs
#        SLURM_ARRAY_TASK_ID=<k> julia ...                        # runs config k (1-based)
#      Per-task CSVs are written and later concatenated (see submit_slurm_array.sh).
#
# Environment variables:
#   MODEL       "e_coli_core" (bundled, default), a BiGG id that is fetched from
#               BiGG with SHA-256 verification (iJR904, iAF1260b, iMM904,
#               iAB_RBC_283 — see bigg_models.jl), or an explicit SBML path.
#   MODELTAG    output filename tag (default: basename of MODEL without extension)
#   BIOMASS_IDS comma-separated biomass reaction ids (default: auto-detect)
#   NPROCS      worker processes (default: min(15, nCPU-1))
#   SEEDS       comma-separated (default: 1234,42,7,101,202,303,404,505,606,707)
#   EXPERIMENTS subset of A,B,C,D (default: A,B,C,D)
#   OUTDIR      output dir (default: examples/reproducibility/results)
#   OPTIMIZER   HiGHS (default) or GLPK
#   COUNT_ONLY  if set, print the number of grid configs and exit
#   SLURM_ARRAY_TASK_ID / ARRAY_TASK_ID  run only that 1-based config
# =====================================================================================

using Distributed
import SHA

const MODEL       = get(ENV, "MODEL", "e_coli_core")
const MODELTAG    = get(ENV, "MODELTAG",
                        MODEL == "e_coli_core" ? "e_coli_core" : splitext(basename(MODEL))[1])
const NPROCS      = parse(Int, get(ENV, "NPROCS", string(min(15, max(1, Sys.CPU_THREADS - 1)))))
const SEEDS       = parse.(Int, split(get(ENV, "SEEDS", "1234,42,7,101,202,303,404,505,606,707"), ","))
const EXPERIMENTS = split(get(ENV, "EXPERIMENTS", "A,B,C,D"), ",")
const OUTDIR      = get(ENV, "OUTDIR", joinpath(@__DIR__, "results"))
const OPTNAME     = get(ENV, "OPTIMIZER", "HiGHS")
const COUNT_ONLY  = haskey(ENV, "COUNT_ONLY")
const ARRAY_ID    = let v = get(ENV, "SLURM_ARRAY_TASK_ID", get(ENV, "ARRAY_TASK_ID", ""))
                        isempty(v) ? nothing : parse(Int, v)
                    end
mkpath(OUTDIR)

# ---- build the flat config list (deterministic order) -------------------------------
# obj_mode: :none -> full cone (paper); :rel999 -> objective_bound relative 0.999.
const DEF_BLOCKED = 1e-9

# Config constructor with defaults; every experiment builds configs through this so the
# schema (incl. kinetic_efficient) stays consistent.
_cfg(; label, variant=:with, blocked_tol=DEF_BLOCKED, seed, use_transitivity=true,
       obj_mode=:none, concordance_tolerance=0.01, balanced_threshold=1e-7,
       kinetic_efficient=true, sample_size=1000, cv_threshold=0.01) =
    (; label, variant, blocked_tol, seed, use_transitivity, obj_mode,
       concordance_tolerance, balanced_threshold, kinetic_efficient,
       sample_size, cv_threshold)

function build_configs()
    cfgs = NamedTuple[]
    b_seeds = SEEDS[1:min(3, length(SEEDS))]
    if "A" in EXPERIMENTS
        for seed in SEEDS, ut in (true, false)
            push!(cfgs, _cfg(label="A_seed_transitivity", seed=seed, use_transitivity=ut))
        end
    end
    if "B" in EXPERIMENTS
        for seed in b_seeds
            push!(cfgs, _cfg(label="B_biomass_kept_bound", seed=seed, obj_mode=:rel999))
        end
        for seed in b_seeds
            push!(cfgs, _cfg(label="B_biomass_kept_nobound", seed=seed))
        end
        for seed in b_seeds
            push!(cfgs, _cfg(label="B_biomass_removed_nobound", variant=:without, seed=seed))
        end
    end
    if "C" in EXPERIMENTS
        for tol in (1e-6, 1e-8, 1e-10), seed in b_seeds
            push!(cfgs, _cfg(label="C_tolerance_sweep", seed=seed,
                             concordance_tolerance=tol, balanced_threshold=tol))
        end
    end
    if "D" in EXPERIMENTS
        for btol in (1e-6, 1e-8, 1e-10), seed in b_seeds
            push!(cfgs, _cfg(label="D_blocked_tol_$(btol)", blocked_tol=btol, seed=seed))
        end
    end
    if "E" in EXPERIMENTS
        # Exhaustive ACR/ACRR via the full matrix/deficiency path (kinetic_efficient=false).
        # Full cone, ordered binding. This is the slow, reference-comparable detector;
        # keep the grid small (few seeds) because it is very expensive at genome scale.
        for seed in b_seeds
            push!(cfgs, _cfg(label="E_efficient_false", seed=seed, kinetic_efficient=false))
        end
    end
    if "F" in EXPERIMENTS
        # Sample-size sweep: does more ACHR sampling resolve the CV pre-filter better and
        # remove the genome-scale seed-dependence? Full cone, transitivity on. Higher
        # sample_size needs MORE RAM (samples stored per complex).
        for ss in (1000, 5000, 20000), seed in b_seeds
            push!(cfgs, _cfg(label="F_samplesize_$(ss)", seed=seed, sample_size=ss))
        end
    end
    if "G" in EXPERIMENTS
        # CV-threshold sweep: raising cv_threshold sends MORE pairs to the exact (seed-
        # independent) LP test instead of gating them on the noisy sampled CV, so it
        # should reduce seed-dependence -- at higher compute cost. Full cone, transitivity on.
        for cv in (0.01, 0.05, 0.1), seed in b_seeds
            push!(cfgs, _cfg(label="G_cvthreshold_$(cv)", seed=seed, cv_threshold=cv))
        end
    end
    return cfgs
end

const CONFIGS = build_configs()

if COUNT_ONLY
    println(length(CONFIGS))
    exit(0)
end

# ---- workers ------------------------------------------------------------------------
if NPROCS > 0 && nprocs() <= NPROCS
    addprocs(NPROCS - (nprocs() - 1); exeflags="--project=$(Base.active_project())")
end
@everywhere begin
    using COCOA
    import COBREXA
    import SBMLFBCModels
    import AbstractFBCModels as A
    import HiGHS
    import GLPK
end
import CSV, DataFrames
import SparseArrays
include(joinpath(@__DIR__, "bigg_models.jl"))  # fetch_bigg_model / is_bigg_id

const OPTIMIZER = OPTNAME == "GLPK" ? GLPK.Optimizer : HiGHS.Optimizer

# ---- solver (LP) tolerance -----------------------------------------------------------
# We deliberately use the solver's DEFAULT feasibility tolerance (HiGHS ~1e-7) and set
# no attributes. This was determined empirically: TIGHTENING the LP tolerance (tested at
# 1e-9/1e-10) *degrades* determinism rather than improving it. Metabolic LPs are highly
# degenerate (many alternate optima); a tighter feasibility tolerance makes HiGHS return
# more numerically variable vertices, which propagate through the seed-dependent sampler
# + CV pre-filter + transitivity into module-partition variability. At the default
# tolerance, concordance is deterministic for concordance_tolerance >= ~1e-8; the
# 1e-10 sweep point is below the resolvable precision of these degenerate LPs and cannot
# be stabilised by tightening. If cross-version pinning is ever needed, pin at 1e-7
# (the current default) rather than lower.

# ---- model loading & preprocessing --------------------------------------------------
function load_canon(model_spec::String)
    path = if model_spec == "e_coli_core"
        joinpath(pkgdir(COCOA), "test", "e_coli_core.xml")   # bundled (no download)
    elseif isfile(model_spec)
        model_spec                                            # explicit path wins
    elseif is_bigg_id(model_spec)
        fetch_bigg_model(model_spec)                          # download + SHA-256 verify
    else
        model_spec
    end
    isfile(path) || error("Model file not found: $path")
    return convert(A.CanonicalModel.Model, COBREXA.load_model(path))
end

function detect_biomass_ids(model_canon)
    ids = String[]
    for (rid, rxn) in model_canon.reactions
        oc = getfield(rxn, :objective_coefficient)
        isbio = occursin(r"biomass"i, rid) ||
                (rxn.name !== nothing && occursin(r"biomass"i, rxn.name))
        ((oc !== nothing && oc != 0) || isbio) && push!(ids, rid)
    end
    return ids
end

remove_reactions!(m, rids) = (for r in rids; haskey(m.reactions, r) && delete!(m.reactions, r); end; m)

function preprocess(model_canon; blocked_tol::Float64=DEF_BLOCKED)
    return model_canon |>
        normalize_bounds |>
        m -> remove_blocked_reactions(m; optimizer=OPTIMIZER, flux_tolerance=blocked_tol) |>
        remove_orphans |>
        split_into_elementary |>
        split_into_irreversible
end

# ---- label-invariant partition fingerprint ------------------------------------------
function partition_fingerprint(result)
    groups = Dict{Int,Vector{String}}()
    for (c, m) in zip(result.complexes.complex_id, result.complexes.concordance_module)
        push!(get!(Vector{String}, groups, m), c)
    end
    canon = sort([sort(v) for (_, v) in groups])
    return bytes2hex(SHA.sha256(string(canon)))[1:16]
end

# ---- module-size statistics (giant module size, singleton count, #modules) -----------
# `assignments` is the per-complex Int module id; `include_zero=false` ignores the 0
# label (unassigned complexes for kinetic modules; balanced group is label 0 too but is
# handled explicitly by the caller). Returns (n_modules, giant_size, n_singletons).
function module_size_stats(assignments; include_zero::Bool=true)
    sizes = Dict{Int,Int}()
    for m in assignments
        (!include_zero && m == 0) && continue
        sizes[m] = get(sizes, m, 0) + 1
    end
    isempty(sizes) && return (0, 0, 0)
    vals = collect(values(sizes))
    return (length(vals), maximum(vals), count(==(1), vals))
end

# ---- model cache (per (variant, blocked_tol)) ---------------------------------------
const BASE_CANON  = load_canon(MODEL)
const BIOMASS_IDS = haskey(ENV, "BIOMASS_IDS") ? String.(split(ENV["BIOMASS_IDS"], ",")) :
                    detect_biomass_ids(BASE_CANON)
println("Model=$MODELTAG  biomass ids=$BIOMASS_IDS  nprocs=$(nprocs())  configs=$(length(CONFIGS))")
const MODEL_CACHE = Dict{Tuple{Symbol,Float64},Any}()
function get_model(variant::Symbol, blocked_tol::Float64)
    get!(MODEL_CACHE, (variant, blocked_tol)) do
        canon = deepcopy(BASE_CANON)
        variant == :without && remove_reactions!(canon, BIOMASS_IDS)
        preprocess(canon; blocked_tol=blocked_tol)
    end
end

objbound(mode) = mode == :rel999 ? COBREXA.relative_tolerance_bound(0.999) : nothing

# Persist the FULL result the package returns (not just summary scalars), so any
# measure — per-complex module membership, giant-module composition, exact ACR/
# ACRR sets — is recoverable without re-running. Controlled by SAVE_FULL (default on).
const SAVE_FULL = lowercase(get(ENV, "SAVE_FULL", "true")) in ("1", "true", "yes")
# metabolite composition per complex, from the model's Y matrix (mets x complexes) —
# the canonical multiset key ("1*M_h_c + 1*M_accoa_c") for aligning complexes to the
# reference, whose internal complex labels differ. Formatted like the reference exporter.
function complex_compositions(model)
    Yc, metids, cids = complex_stoichiometry(model; return_ids=true)
    metstr = String.(metids)
    cid2col = Dict(string(c) => j for (j, c) in enumerate(cids))
    fmt(v) = isinteger(v) ? string(Int(v)) : string(v)
    function comp_of(cid)
        j = get(cid2col, string(cid), 0)
        j == 0 && return missing
        idxs, vals = SparseArrays.findnz(Yc[:, j])
        isempty(idxs) && return "(zero)"
        join(sort([fmt(vals[k]) * "*" * metstr[idxs[k]] for k in eachindex(idxs)]), " + ")
    end
    return comp_of
end

function save_full_result(result, run_id::AbstractString, model)
    SAVE_FULL || return
    dir = joinpath(OUTDIR, "full"); mkpath(dir)
    # complexes: complex_id, concordance_module, kinetic_module, classification, [lambda]
    # + composition (metabolite multiset from Y) for exact cross-tool alignment.
    cdf = DataFrames.DataFrame(result.complexes)
    comp_of = complex_compositions(model)
    cdf.composition = [comp_of(c) for c in cdf.complex_id]
    CSV.write(joinpath(dir, "$(run_id)_complexes.csv"), cdf)
    CSV.write(joinpath(dir, "$(run_id)_acr.csv"),       DataFrames.DataFrame(result.acr))
    CSV.write(joinpath(dir, "$(run_id)_acrr.csv"),      DataFrames.DataFrame(result.acrr))
    CSV.write(joinpath(dir, "$(run_id)_stats.csv"),
              DataFrames.DataFrame(key=collect(string.(keys(result.stats))),
                                   value=collect(string.(values(result.stats)))))
end

# ---- single run ---------------------------------------------------------------------
function run_config(cfg; run_id::AbstractString="$(MODELTAG)_$(cfg.label)_seed$(cfg.seed)")
    mp = get_model(cfg.variant, cfg.blocked_tol)
    t0 = time()
    result = activity_concordance_analysis(
        mp;
        optimizer=OPTIMIZER,
        objective_bound=objbound(cfg.obj_mode),
        concordance_tolerance=cfg.concordance_tolerance,
        balanced_threshold=cfg.balanced_threshold,
        cv_threshold=cfg.cv_threshold,
        sample_size=cfg.sample_size,
        seed=UInt(cfg.seed),
        use_transitivity=cfg.use_transitivity,
        kinetic_analysis=true,
        kinetic_efficient=cfg.kinetic_efficient,
    )
    elapsed = time() - t0
    n_acr  = length(Set(result.acr.metabolite_id))
    n_acrr = length(Set(zip(result.acrr.metabolite_1, result.acrr.metabolite_2)))
    # Structural measures — capture ALL the important ones, per run:
    # giant module size (concordance + kinetic), module counts, singletons, sizes.
    conc_n, conc_giant, conc_single = module_size_stats(result.complexes.concordance_module)
    kin_n,  kin_giant,  kin_single  = module_size_stats(result.complexes.kinetic_module; include_zero=false)
    row = (
        model=MODELTAG, label=cfg.label, seed=cfg.seed,
        use_transitivity=cfg.use_transitivity, kinetic_efficient=cfg.kinetic_efficient,
        objective_bound=cfg.obj_mode == :rel999 ? "rel0.999" : "none",
        variant=String(cfg.variant), blocked_tol=cfg.blocked_tol,
        concordance_tolerance=cfg.concordance_tolerance, balanced_threshold=cfg.balanced_threshold,
        cv_threshold=cfg.cv_threshold, sample_size=cfg.sample_size,
        n_complexes=get(result.stats, "n_complexes", missing),
        n_balanced=get(result.stats, "n_balanced", missing),
        n_concordance_modules=get(result.stats, "n_concordance_modules", missing),
        n_concordant_total=get(result.stats, "n_concordant_total", missing),
        giant_concordance_size=conc_giant, n_singleton_concordance=conc_single,
        n_kinetic_modules=kin_n, giant_kinetic_size=kin_giant, n_singleton_kinetic=kin_single,
        n_acr=n_acr, n_acrr=n_acrr,
        partition_fp=partition_fingerprint(result), elapsed_s=round(elapsed, digits=1),
    )
    println("[$(cfg.label) seed=$(cfg.seed) trans=$(cfg.use_transitivity) " *
            "eff=$(cfg.kinetic_efficient) obj=$(row.objective_bound) var=$(row.variant) " *
            "btol=$(cfg.blocked_tol) ctol=$(cfg.concordance_tolerance)] " *
            "modules=$(row.n_concordance_modules) giant_conc=$conc_giant giant_kin=$kin_giant " *
            "acr=$n_acr acrr=$n_acrr fp=$(row.partition_fp) t=$(row.elapsed_s)s")
    flush(stdout)
    save_full_result(result, run_id, mp)   # persist the full package result, not just the summary
    return row
end

# =====================================================================================
if ARRAY_ID !== nothing
    # ---- SLURM array mode: run exactly one config -----------------------------------
    1 <= ARRAY_ID <= length(CONFIGS) || error("ARRAY_ID $ARRAY_ID out of range 1:$(length(CONFIGS))")
    row = run_config(CONFIGS[ARRAY_ID]; run_id="$(MODELTAG)_task_$(lpad(ARRAY_ID, 4, '0'))")
    taskfile = joinpath(OUTDIR, "$(MODELTAG)_task_$(lpad(ARRAY_ID, 4, '0')).csv")
    CSV.write(taskfile, DataFrames.DataFrame([row]))
    println("Wrote $taskfile")
else
    # ---- whole grid in one process --------------------------------------------------
    rows = NamedTuple[]
    for (i, cfg) in enumerate(CONFIGS)
        push!(rows, run_config(cfg; run_id="$(MODELTAG)_run_$(lpad(i, 4, '0'))"))
        CSV.write(joinpath(OUTDIR, "$(MODELTAG)_all_runs.csv"), DataFrames.DataFrame(rows))
    end
    println("\nDone. Wrote $(length(rows)) runs to $(joinpath(OUTDIR, "$(MODELTAG)_all_runs.csv"))")
end
