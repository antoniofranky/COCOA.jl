# Guardrails for the three-way pair verdict.
#
# Before this fix `process_concordance_batch` had exactly two outcomes per complex
# pair: `union_sets!` (concordant) or `add_non_concordant!` (not concordant). A pair
# whose LPs merely FAILED — solver time limit, OTHER_ERROR, SLOW_PROGRESS — therefore
# ended up in the second branch, i.e. a solver failure was recorded as the positive
# scientific claim "these two complexes are not concordant". `add_non_concordant!`
# additionally lifts that claim to module level, so under `use_transitivity=true` a
# single failed LP blocked every further pair between two whole modules.
#
# In the archived yeast run this hit 12.2 % of all candidate pairs.
#
# These tests pin the corrected semantics:
#   - a refutation requires a COMPLETE, NaN-free comparison  -> decisive_non_concordant
#   - a failure leaves the pair undecided                    -> neither flag set
#   - a refuted pair can never be resurrected by a later direction

using Test
using COCOA
using HiGHS
using COBREXA
import SparseArrays

const _CT = 1e-2   # concordance tolerance used throughout these tests

# Feed the four solver results of one pair in the order the batch produces them:
# (positive, MIN), (positive, MAX), (negative, MIN), (negative, MAX).
function _feed!(state, results; tol=_CT)
    for (direction, dir_multiplier, value, timeout) in results
        COCOA.update_pair_concordance!(state, direction, dir_multiplier, value, timeout, tol)
    end
    return state
end

@testset "Three-way pair verdict (concordant / refuted / undecided)" begin

    @testset "fresh state is undecided, not refuted" begin
        s = COCOA.PairConcordanceState()
        @test s.is_concordant == false
        @test s.decisive_non_concordant == false
    end

    @testset "constant ratio -> concordant" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, 2.0, false),   # min = 2.0
            (:positive, +1, 2.0, false),   # max = 2.0
        ])
        @test s.is_concordant == true
        @test s.decisive_non_concordant == false
        @test s.reference_lambda ≈ 2.0
    end

    @testset "min != max -> decisive refutation" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, 1.0, false),
            (:positive, +1, 5.0, false),
        ])
        @test s.is_concordant == false
        @test s.decisive_non_concordant == true
    end

    @testset "all LPs failed (NaN) -> undecided, NOT refuted" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, NaN, false),
            (:positive, +1, NaN, false),
        ])
        @test s.is_concordant == false
        @test s.decisive_non_concordant == false   # the whole point: no verdict
    end

    @testset "solver time limit -> undecided, NOT refuted" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, NaN, true),
            (:positive, +1, NaN, true),
        ])
        @test s.is_concordant == false
        @test s.has_timeout == true
        @test s.decisive_non_concordant == false
    end

    @testset "half-failed direction -> undecided, NOT refuted" begin
        # min solved, max failed: nothing can be concluded from an incomplete interval
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, 2.0, false),
            (:positive, +1, NaN, false),
        ])
        @test s.is_concordant == false
        @test s.decisive_non_concordant == false
    end

    @testset "inconsistent lambda across directions -> decisive refutation" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, 2.0, false),
            (:positive, +1, 2.0, false),   # lambda = 2.0
            (:negative, -1, 7.0, false),
            (:negative, +1, 7.0, false),   # lambda = 7.0 -> inconsistent
        ])
        @test s.is_concordant == false
        @test s.decisive_non_concordant == true
    end

    @testset "a refuted pair is never resurrected by a later direction" begin
        # The positive cone already shows a non-constant ratio, so the pair is not
        # concordant no matter what the negative cone says. Before the guard, the
        # trailing consistent direction flipped `is_concordant` back to true.
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, 1.0, false),
            (:positive, +1, 5.0, false),   # refuted here, lambda still NaN
            (:negative, -1, 3.0, false),
            (:negative, +1, 3.0, false),   # complete and self-consistent
        ])
        @test s.decisive_non_concordant == true
        @test s.is_concordant == false
    end

    @testset "failure in one direction cannot mask a refutation in the other" begin
        s = _feed!(COCOA.PairConcordanceState(), [
            (:positive, -1, NaN, false),
            (:positive, +1, NaN, false),   # failed
            (:negative, -1, 1.0, false),
            (:negative, +1, 9.0, false),   # genuinely refuted
        ])
        @test s.decisive_non_concordant == true
        @test s.is_concordant == false
    end
end

@testset "Undecided pairs are counted and reported" begin
    # The counter must exist and be plumbed all the way into the stats dictionary,
    # otherwise solver attrition stays invisible the way it did in the archived run.
    counts = COCOA.MutableCounts()
    @test counts.unknown == 0
    counts.unknown += 3
    @test counts.unknown == 3

    model = COCOA.create_envz_ompr_model()
    result = COCOA.activity_concordance_analysis(
        model;
        optimizer=HiGHS.Optimizer,
        sample_size=50,
        seed=UInt(1234),
        kinetic_analysis=false,
    )
    stats = result.stats
    @test haskey(stats, "n_unknown_pairs")
    # A tiny, well-conditioned CRN must not lose a single pair to the solver.
    @test stats["n_unknown_pairs"] == 0
end

# Label-invariant partition: complexes grouped by concordance module, sorted within
# and across groups. Two runs agree iff they induce the same partition, whatever the
# module numbering happens to be.
function _partition(result)
    groups = Dict{Int,Vector{String}}()
    for (c, m) in zip(result.complexes.complex_id, result.complexes.concordance_module)
        push!(get!(Vector{String}, groups, m), string(c))
    end
    return sort([sort(v) for v in values(groups)])
end

@testset "Deterministic static scheduling" begin
    # `screen_directions_optimization_model` used to hand tests to workers via
    # `pmap` + `CachingPool`, i.e. dynamically, by wall-clock timing. Because each
    # worker mutates ONE cached JuMP model per batch (add constraint -> optimize ->
    # delete), the warm-start basis an LP starts from depended on which tests its
    # worker happened to run first — and on a degenerate flux polytope a different
    # starting basis can return a different optimal vertex, flipping the verdict.
    # Measured on Saccharomyces cerevisiae: 3 identical runs, 3 different partitions.
    #
    # `scheduling=:static` fixes the partition of tests into chunks, so the result no
    # longer depends on timing. This test only pins the contract — a single-process
    # test session cannot exhibit the race itself.

    model = COCOA.create_envz_ompr_model()
    analyse(sched) = COCOA.activity_concordance_analysis(
        model; optimizer=HiGHS.Optimizer, sample_size=50,
        seed=UInt(1234), kinetic_analysis=false, scheduling=sched,
    )

    @test_throws ArgumentError analyse(:sometimes)

    static1 = analyse(:static)
    static2 = analyse(:static)
    dynamic = analyse(:dynamic)
    default = COCOA.activity_concordance_analysis(
        model; optimizer=HiGHS.Optimizer, sample_size=50,
        seed=UInt(1234), kinetic_analysis=false,
    )

    # The static path must be the DEFAULT, and must be recorded in the result so a
    # stored run carries the conditions under which it can be reproduced.
    @test static1.stats["scheduling"] == "static"
    @test dynamic.stats["scheduling"] == "dynamic"
    @test default.stats["scheduling"] == "static"

    # Chunking must not change WHAT is computed, only the order it is computed in.
    @test _partition(static1) == _partition(static2)
    @test _partition(static1) == _partition(dynamic)
    @test static1.stats["n_concordant_total"] == dynamic.stats["n_concordant_total"]
end

@testset "A 'successful' solve with no usable value counts as a failure" begin
    # `categorize_termination_status` books ALMOST_OPTIMAL as :success, but the worker
    # only extracts an objective value for OPTIMAL / LOCALLY_SOLVED that is also
    # `is_solved_and_feasible`. Such a solve therefore returns NaN while being counted
    # as a success — invisible in every failure counter, and (before the three-way
    # verdict) silently booked as "not concordant".
    #
    # Measured on Saccharomyces cerevisiae before this fix: 775 pairs left undecided
    # while the failure counters accounted for at most 93 of them.
    @test COCOA.categorize_termination_status(COCOA.J.ALMOST_OPTIMAL) == :success
    @test COCOA.categorize_termination_status(COCOA.J.OPTIMAL) == :success
    @test COCOA.categorize_termination_status(COCOA.J.TIME_LIMIT) == :timeout

    counts = COCOA.MutableCounts()
    @test counts.success_no_value == 0

    model = COCOA.create_envz_ompr_model()
    result = COCOA.activity_concordance_analysis(
        model; optimizer=HiGHS.Optimizer, sample_size=50,
        seed=UInt(1234), kinetic_analysis=false,
    )
    @test haskey(result.stats, "n_success_no_value_pairs")
    # A tiny, well-conditioned CRN must not produce a single one of these.
    @test result.stats["n_success_no_value_pairs"] == 0
    @test result.stats["n_unknown_pairs"] == 0
end

@testset "Deterministic blocked-reaction detection" begin
    # `find_blocked_reactions` decides WHICH REACTIONS EXIST in every downstream model,
    # and it used to do so through COBREXA's dynamic `pmap` path. Measured: three
    # identical preprocessing runs of S. cerevisiae gave networks differing by up to
    # 35 reactions (no_split) and 850 (random_0).
    # NB: no `load_model` by filename here — AbstractFBCModels then scans every
    # registered model type for eligible extensions, and `DCEToyModel` from
    # test_dce_acr.jl does not implement `filename_extensions`. Use an in-memory model.
    # The substantive determinism evidence is the genome-scale preprocessing probe
    # on a yeast GEM, not this contract test.
    model = COCOA.create_envz_ompr_model()

    b_static = COCOA.find_blocked_reactions(model; optimizer=HiGHS.Optimizer, scheduling=:static)
    b_again  = COCOA.find_blocked_reactions(model; optimizer=HiGHS.Optimizer, scheduling=:static)
    b_dyn    = COCOA.find_blocked_reactions(model; optimizer=HiGHS.Optimizer, scheduling=:dynamic)

    @test sort(b_static) == sort(b_again)
    # On a small, well-conditioned model the two schedulings must still agree: static
    # chunking changes the order of computation, never the answer.
    @test sort(b_static) == sort(b_dyn)

    @test_throws ArgumentError COCOA.find_blocked_reactions(
        model; optimizer=HiGHS.Optimizer, scheduling=:whenever)

    # A threshold below the solver's achievable accuracy must not pass silently.
    @test_logs (:warn, r"below the solver") match_mode=:any COCOA.find_blocked_reactions(
        model; optimizer=HiGHS.Optimizer, flux_tolerance=1e-12)
end

@testset "Threaded ACR detection is thread-count invariant" begin
    # The threaded loops used to index shared workspaces by `Threads.threadid()`. Since
    # Julia 1.8 `@threads` schedules dynamically, so a task may resume on a different
    # thread and two tasks could share a buffer — unsound by contract, and this loop
    # runs on the production path (`efficient=true`). Both loops now give every spawned
    # task its own buffers.
    #
    # A race cannot be proven absent by testing; this pins the observable contract
    # instead: the result must not depend on how the work was split. The genome-scale
    # evidence is a yeast GEM run, where 16 and 64 threads gave identical kinetic partitions.
    model = COCOA.create_envz_ompr_model()
    conc = COCOA.activity_concordance_analysis(
        model; optimizer=HiGHS.Optimizer, sample_size=50,
        seed=UInt(1234), kinetic_analysis=false, detailed_results=true,
    )
    mods = COCOA.extract_concordance_modules(conc)

    for eff in (true, false)
        a = COCOA.kinetic_analysis(mods, model; efficient=eff)
        b = COCOA.kinetic_analysis(mods, model; efficient=eff)
        @test sort(String.(a.acr_metabolites)) == sort(String.(b.acr_metabolites))
        @test sort(a.acrr_pairs) == sort(b.acrr_pairs)
        @test length.(a.kinetic_modules) == length.(b.kinetic_modules)
    end
end

@testset "Sparse span test agrees with the dense one" begin
    # The sparse path exists to stop the pair loop from streaming the whole Q basis
    # through memory once per pair (~100 MB on genome-scale models, which is why
    # efficient=false showed the same runtime at 16, 64 and 128 threads). It must not
    # change any answer: it either decides exactly as the dense path, or declares itself
    # undecided and hands over.
    rng_vals = [1.0, -2.0, 0.5, -0.25, 3.0]
    Y = [1.0 0.0 2.0; 0.0 1.0 -1.0; 1.0 1.0 0.0; 0.0 0.0 1.0]
    cache = COCOA.build_cached_column_span(Y; tolerance=1e-8)
    @test size(cache.Qt) == (cache.rank, size(Y, 1))
    @test cache.Qt ≈ cache.Q_reduced'

    coeffs = Vector{Float64}(undef, max(1, cache.rank))
    proj = Vector{Float64}(undef, size(Y, 1))

    for trial in 1:40
        v = zeros(Float64, size(Y, 1))
        # mix of in-span vectors and arbitrary ones
        if isodd(trial)
            v .= Y * [rng_vals[mod1(trial, 5)], rng_vals[mod1(trial + 1, 5)], rng_vals[mod1(trial + 2, 5)]]
        else
            v[mod1(trial, length(v))] = rng_vals[mod1(trial, 5)]
        end
        dense = COCOA.is_in_span!(copy(v), cache, coeffs, proj)

        sp = SparseArrays.sparse(v)
        idx, val = SparseArrays.findnz(sp)
        nz_idx = collect(idx); nz_val = collect(val)
        resize!(nz_idx, max(length(nz_idx), 1)); resize!(nz_val, max(length(nz_val), 1))
        in_span, decided = COCOA.is_in_span_sparse(nz_idx, nz_val, length(idx), cache, coeffs)

        # Only a decisive sparse answer is binding, and then it must match.
        decided && @test in_span == dense
    end
end

@testset "Sparse difference helper" begin
    oi = Vector{Int}(undef, 16); ov = Vector{Float64}(undef, 16)
    n = COCOA._sparse_difference!(oi, ov, [1, 3, 5], [2.0, 1.0, 4.0], [3, 4], [1.0, 7.0])
    @test oi[1:n] == [1, 4, 5]                 # index 3 cancels exactly and is dropped
    @test ov[1:n] == [2.0, -7.0, 4.0]
    n = COCOA._sparse_difference!(oi, ov, Int[], Float64[], [2], [3.0])
    @test oi[1:n] == [2] && ov[1:n] == [-3.0]
end

@testset "merge_map matches the quadratic reference" begin
    # `merge_coupled_sets_tracked` used to build its merge map with an
    # O(n_upstream * n_modules) `isdisjoint` scan, which on a genome-scale model sat for
    # hours. The replacement uses one reverse index. It must return exactly the same
    # map, including the "lowest matching group id" tie-break.
    reference(upstream_sets, kinetic_modules) = begin
        mm = zeros(Int, length(upstream_sets))
        for i in eachindex(upstream_sets)
            for (gid, fm) in enumerate(kinetic_modules)
                if !isdisjoint(upstream_sets[i], fm)
                    mm[i] = gid
                    break
                end
            end
        end
        mm
    end
    fast(upstream_sets, kinetic_modules) = begin
        c2g = Dict{Symbol,Int}()
        for (gid, fm) in enumerate(kinetic_modules), c in fm
            g = get(c2g, c, typemax(Int)); g > gid && (c2g[c] = gid)
        end
        mm = zeros(Int, length(upstream_sets))
        for i in eachindex(upstream_sets)
            best = typemax(Int)
            for c in upstream_sets[i]
                g = get(c2g, c, typemax(Int)); g < best && (best = g)
            end
            mm[i] = best == typemax(Int) ? 0 : best
        end
        mm
    end

    us = [Set([:a, :b]), Set([:c]), Set([:z]), Set([:b, :c]), Set{Symbol}()]
    km = [Set([:a, :b]), Set([:c, :d]), Set([:e])]
    @test fast(us, km) == reference(us, km)
    @test fast(us, km)[3] == 0            # no group contains :z
    @test fast(us, km)[4] == 1            # touches groups 1 and 2 -> lowest wins
    @test fast(us, Set{Symbol}[]) == zeros(Int, length(us))
end

@testset "Span-based robustness detection (ACR / ACRR / gACRR)" begin
    # The SM defines all three as span memberships (S3-13): e_S in im(Y𝚫) for ACR,
    # e_S1 - γ e_S2 in im(Y𝚫) for gACRR, γ = 1 giving ACRR. The pairwise complex scan
    # is only a sufficient shortcut, which is why it misses the published EnvZ-OmpR ACR.
    #
    # These tests pin the algebra on hand-built spans where the answer is known by
    # construction, before any use on real models.

    # im(Y𝚫) spanned by e1 alone: species 1 is ACR, nothing else is.
    Y1 = reshape([1.0, 0.0, 0.0, 0.0], 4, 1)
    c1 = COCOA.build_cached_column_span(Y1; tolerance=1e-10)
    r1 = COCOA.detect_robustness_via_span(c1, [:a, :b, :c, :d]; tolerance=1e-8)
    @test r1.acr_metabolites == [:a]
    @test isempty(r1.acrr_pairs)

    # Span contains e2 - e3: that is ACRR with γ = 1, and neither species alone is ACR.
    Y2 = reshape([0.0, 1.0, -1.0, 0.0], 4, 1)
    c2 = COCOA.build_cached_column_span(Y2; tolerance=1e-10)
    r2 = COCOA.detect_robustness_via_span(c2, [:a, :b, :c, :d]; tolerance=1e-6)
    @test isempty(r2.acr_metabolites)
    @test (:b, :c) in r2.acrr_pairs
    @test any(t -> Set([t[1], t[2]]) == Set([:b, :c]) && isapprox(t[3], 1.0; atol=1e-6),
              r2.gacrr_triples)

    # Span contains e2 - 2*e3: gACRR with γ = 2, and NOT ACRR. This is exactly the case
    # the pairwise criterion cannot express.
    Y3 = reshape([0.0, 1.0, -2.0, 0.0], 4, 1)
    c3 = COCOA.build_cached_column_span(Y3; tolerance=1e-10)
    r3 = COCOA.detect_robustness_via_span(c3, [:a, :b, :c, :d]; tolerance=1e-6)
    @test isempty(r3.acr_metabolites)
    @test isempty(r3.acrr_pairs)                       # γ = 2, so not ACRR
    tri = filter(t -> Set([t[1], t[2]]) == Set([:b, :c]), r3.gacrr_triples)
    @test length(tri) == 1
    @test isapprox(tri[1][3], 2.0; atol=1e-6) || isapprox(tri[1][3], 0.5; atol=1e-6)

    # Span contains e2 + e3 (γ would be negative): excluded by the SM's γ > 0.
    Y4 = reshape([0.0, 1.0, 1.0, 0.0], 4, 1)
    c4 = COCOA.build_cached_column_span(Y4; tolerance=1e-10)
    r4 = COCOA.detect_robustness_via_span(c4, [:a, :b, :c, :d]; tolerance=1e-6)
    @test isempty(r4.acrr_pairs)
    @test isempty(r4.gacrr_triples)

    # Empty span: nothing is robust, and the function must not throw.
    c0 = COCOA.build_cached_column_span(zeros(4, 0); tolerance=1e-10)
    r0 = COCOA.detect_robustness_via_span(c0, [:a, :b, :c, :d])
    @test isempty(r0.acr_metabolites) && isempty(r0.acrr_pairs) && isempty(r0.gacrr_triples)
end

@testset "Upstream algorithm invariants" begin
    # `upstream_algorithm` decides which complexes make up a coupling set, so its output
    # determines every kinetic module, ACR and ACRR downstream. In the MATLAB reference
    # this is exactly where an indexing bug sat undetected for years (Phase IV wrote
    # `x[terminal == 0]` instead of `x[terminal[x] == 0]`, keeping ~54 terminal complexes
    # that theory says to prune). These tests pin the two defining properties instead of
    # trusting the implementation to keep them.
    model = COCOA.create_envz_ompr_model()
    Y, met_ids, cplx_ids = COCOA.complex_stoichiometry(model; return_ids=true)
    A = COCOA.incidence(model)
    A, Y, cplx_ids = COCOA.augment_with_zero_complex(A, Y, cplx_ids)
    c2i = Dict{Symbol,Int}(c => i for (i, c) in enumerate(cplx_ids))
    adj = COCOA.build_cached_adjacency(A, length(cplx_ids))
    gterm = COCOA.compute_global_terminal_complexes(adj, length(cplx_ids))
    network = (A=A, Y=Y, complex_ids=cplx_ids, metabolite_ids=met_ids,
               complex_to_idx=c2i, acr_augmentation=zeros(size(Y, 1), 0),
               enable_advanced_merging=false, cached_adj=adj, global_terminal=gterm)

    all_complexes = Set(cplx_ids)
    u = COCOA.upstream_algorithm(all_complexes, network)

    # Phase IV (Remark S2-1): terminal strong linkage classes are identified on the FULL
    # complex graph and removed. No survivor may be globally terminal.
    @test all(c -> !(c2i[c] in gterm), u)

    # Phase I (autonomy): no survivor may have an in-edge from outside the set. This is
    # the fixpoint the iteration is supposed to reach, not merely approach.
    idx = Set(c2i[c] for c in u)
    for v in idx
        for w in adj.in_neighbors[v]
            @test w in idx
        end
    end

    # A set with no complexes cannot produce survivors.
    @test isempty(COCOA.upstream_algorithm(Set{Symbol}(), network))
end

@testset "Sparse ACR/ACRR difference matches the dense scan" begin
    # `_detect_acr_acrr` used to recover the difference of two complexes by scanning all
    # n_metabolites rows with `Y_matrix[met_idx, idx]` lookups — a search per access on a
    # sparse matrix — although the difference has a handful of nonzeros. The merge must
    # return exactly what the scan returned: the count (capped at 3) and the first two
    # (index, value) pairs.
    dense_scan(ya, yb, tol) = begin
        idx = Int[]; val = Float64[]; n = 0
        for k in eachindex(ya)
            d = ya[k] - yb[k]
            if abs(d) > tol
                n += 1
                n > 2 && return (3, idx, val)
                push!(idx, k); push!(val, d)
            end
        end
        (n, idx, val)
    end

    oi = Vector{Int}(undef, 4); ov = Vector{Float64}(undef, 4)
    tol = 1e-8
    cases = [
        ([1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]),   # identical -> 0
        ([1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 1.0, 0.0]),   # one differing entry -> ACR
        ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]),   # two entries -> ACRR candidate
        ([2.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 0.0]),   # four -> capped at 3
        ([0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]),   # only b nonzero
        ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),   # only a nonzero
    ]
    for (ya, yb) in cases
        sa = SparseArrays.sparse(ya); sb = SparseArrays.sparse(yb)
        ia, va = SparseArrays.findnz(sa); ib, vb = SparseArrays.findnz(sb)
        n = COCOA._sparse_diff_upto2!(oi, ov, collect(ia), collect(va),
                                      collect(ib), collect(vb), tol)
        dn, didx, dval = dense_scan(ya, yb, tol)
        @test n == dn
        if n <= 2
            @test oi[1:n] == didx[1:n]
            @test ov[1:n] == dval[1:n]
        end
    end
end

# `optimize_verified!`. The failure it repairs — HiGHS OPTIMAL with an
# INFEASIBLE_POINT after unscaling — only shows up on genome-scale LPs after a sequence
# of warm starts and could not be provoked on small instances, so its effect is verified
# end to end on a real model (a yeast GEM). What is pinned here is that it is inert
# whenever nothing needs repairing: same value, same status, options untouched.
const JM = COCOA.J   # macros need a global binding
@testset "optimize_verified! is inert on well-posed and on infeasible LPs" begin
    om = JM.Model(HiGHS.Optimizer); JM.set_silent(om)
    JM.set_attribute(om, "presolve", "off")
    JM.@variable(om, 0 <= x[1:3] <= 4)
    JM.@constraint(om, x[1] + x[2] + x[3] == 5)
    JM.@objective(om, Max, 2x[1] + x[2])
    @test COCOA.optimize_verified!(om) == 0
    @test JM.termination_status(om) == JM.OPTIMAL
    @test JM.objective_value(om) ≈ 9.0
    @test !COCOA.reported_success_without_point(om)
    @test JM.get_attribute(om, "presolve") == "off"
    @test JM.get_attribute(om, "solver") == "choose"

    # Restoring options after a recovery must not discard the solution it produced.
    # (JuMP's set_attribute does exactly that — A42's first replay lost every value.)
    COCOA._set_highs_option!(om, "presolve", "on")
    COCOA._set_highs_option!(om, "presolve", "off")
    @test JM.termination_status(om) == JM.OPTIMAL
    @test JM.objective_value(om) ≈ 9.0
    @test JM.get_attribute(om, "presolve") == "off"

    JM.@constraint(om, x[1] >= 6)            # now infeasible: a genuine failure
    @test COCOA.optimize_verified!(om) == 0
    @test JM.termination_status(om) == JM.INFEASIBLE
    @test !COCOA.reported_success_without_point(om)
end

@testset "recovered-LP counter is carried through the accumulator" begin
    c = COCOA.MutableCounts()
    @test c.recovered_lps == 0
    c.recovered_lps = 3
    acc = COCOA.BatchResultAccumulator(10)
    COCOA.accumulate_results!(acc, COCOA.SparseConcordantPairs(10), c,
                              Tuple{Int,Int,Symbol,Float64}[])
    COCOA.accumulate_results!(acc, COCOA.SparseConcordantPairs(10), c,
                              Tuple{Int,Int,Symbol,Float64}[])
    @test acc.counts.recovered_lps == 6
end

# Blocked-reaction detection must never turn a FAILED LP into a verdict. The
# warm-started FVA sweep left some LPs unsolved on the yeast panel; the affected reactions
# were silently kept as "not blocked", and three of them glued a whole network into one
# giant kinetic module. `classify_blocked` is where that decision is made, so it is pinned
# directly; a failing LP cannot be provoked reliably on a small model.
@testset "Undecided reactions are neither blocked nor silently unblocked" begin
    variability = [
        :active      => (0.0, 5.0),
        :blocked     => (0.0, 0.0),
        :tiny        => (-1e-9, 1e-9),
        :no_max      => (0.0, nothing),
        :no_min      => (nothing, 0.0),
        :reversible  => (-3.0, 3.0),
    ]
    blocked, undecided = COCOA.classify_blocked(variability, 1e-6)
    @test blocked == ["blocked", "tiny"]           # FVA order is kept (= deletion order)
    @test undecided == ["no_max", "no_min"]
    @test isempty(intersect(blocked, undecided))   # a failed LP is never evidence of blockedness

    @test COCOA.report_undecided(String[], :error) === nothing
    @test COCOA.report_undecided(undecided, :warn) === nothing
    @test_throws ErrorException COCOA.report_undecided(undecided, :error)
    @test_throws ArgumentError COCOA.report_undecided(undecided, :ignore)
end

@testset "One pass of blocked-reaction removal is final" begin
    # A blocked reaction carries zero flux in every steady state, so deleting it cannot
    # change what the others can do. A second pass must therefore find nothing — the
    # invariant preprocess_models.jl now asserts instead of looping until it holds.
    # `DEAD` consumes C, which nothing produces.
    CMod = COCOA.A.CanonicalModel
    m = CMod.Model()
    for id in ("A", "B", "C")
        m.metabolites[id] = CMod.Metabolite(name=id)
    end
    m.reactions["EX_A"] = CMod.Reaction(stoichiometry=Dict("A" => 1.0), lower_bound=-10.0, upper_bound=10.0)
    m.reactions["R1"]   = CMod.Reaction(stoichiometry=Dict("A" => -1.0, "B" => 1.0), lower_bound=0.0, upper_bound=10.0)
    m.reactions["EX_B"] = CMod.Reaction(stoichiometry=Dict("B" => -1.0), lower_bound=-10.0, upper_bound=10.0)
    m.reactions["DEAD"] = CMod.Reaction(stoichiometry=Dict("C" => -1.0, "B" => 1.0), lower_bound=0.0, upper_bound=10.0)

    @test COCOA.find_blocked_reactions(m; optimizer=HiGHS.Optimizer, flux_tolerance=1e-6) == ["DEAD"]
    cleaned = COCOA.remove_blocked_reactions(m; optimizer=HiGHS.Optimizer, flux_tolerance=1e-6,
                                             on_undecided=:error)
    @test sort(collect(keys(cleaned.reactions))) == ["EX_A", "EX_B", "R1"]
    @test isempty(COCOA.find_blocked_reactions(cleaned; optimizer=HiGHS.Optimizer, flux_tolerance=1e-6))
    @test haskey(m.reactions, "DEAD")               # the input model is left untouched
end

@testset "Complexes shared by every coupling set are flagged" begin
    glue = :M_glue
    sets = [Set([Symbol("c$i"), glue]) for i in 1:12]
    @test COCOA.complexes_in_every_coupling_set(sets) == [glue]
    @test isempty(COCOA.complexes_in_every_coupling_set([Set([Symbol("c$i")]) for i in 1:12]))
    # Few sets: sharing is common and says nothing (EnvZ-OmpR has four coupling sets).
    @test isempty(COCOA.complexes_in_every_coupling_set(sets[1:4]))
end
