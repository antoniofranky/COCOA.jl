# Guardrails for the pair LP and its verdict (v1.2):
#   - the decision rule |hi - lo| <= atol + rtol * max(|hi|, |lo|), by default relative
#     1e-5 with an absolute 1e-9 at zero (the old absolute 0.01 stays selectable);
#   - a growth bound is scaled by t like the flux bounds;
#   - the Charnes–Cooper variable t has no upper bound (the old cap of 999 cut off steady
#     states in which the reference activity is small relative to the fluxes);
#   - a pair LP that is PROVEN unbounded refutes concordance instead of failing.

using Test
using COCOA
using HiGHS
using COBREXA
import AbstractFBCModels.CanonicalModel as CM
const J = COCOA.J
const C = COBREXA.C

# ∅ → A, A → B + 10^6 E, B → ∅, E → ∅, and a second source ∅ → 2 E. The drain of E carries
# 10^6 · v_split + 2 · v_source, so it is at least 10^6 times the split, but "E" is NOT
# concordant with anything (the source is free). The complexes "B + 1e6 E" (activity v) and
# "B" (-v) are concordant. Fixing the activity of "B + 1e6 E" to 1 forces the drain to 10^6,
# i.e. t >= 1000 with flux bounds of 1000: beyond the old cap. (The analysis as a whole
# finds this pair without an LP, because B occurs in only these two complexes, so the pair
# LP is tested directly.)
function small_reference_model()
    m = CM.Model()
    m.metabolites = Dict(x => CM.Metabolite(name=x) for x in ("A", "B", "E"))
    m.reactions["in"] = CM.Reaction(stoichiometry=Dict("A" => 1.0), lower_bound=0.0, upper_bound=1000.0)
    m.reactions["split"] = CM.Reaction(stoichiometry=Dict("A" => -1.0, "B" => 1.0, "E" => 1e6),
                                       lower_bound=0.0, upper_bound=1000.0)
    m.reactions["outB"] = CM.Reaction(stoichiometry=Dict("B" => -1.0), lower_bound=0.0, upper_bound=1000.0)
    m.reactions["outE"] = CM.Reaction(stoichiometry=Dict("E" => -1.0), lower_bound=0.0, upper_bound=1000.0)
    m.reactions["srcE"] = CM.Reaction(stoichiometry=Dict("E" => 2.0), lower_bound=0.0, upper_bound=1000.0)
    return m
end

@testset "Pair LP: relative tolerance, no cap, proven unbounded" begin
    tol = COCOA.ConcordanceTolerance(1e-5, 1e-9)        # the default rule

    @testset "default rule: relative 1e-5, absolute 1e-9 at zero" begin
        # solver noise on truly concordant pairs (<= 1e-10 relative) passes
        @test COCOA.within_tolerance(0.0036300002, 0.0036300002 * (1 + 1e-10), tol)
        @test COCOA.within_tolerance(55.3, 55.3 * (1 + 1e-8), tol)
        # the false merges of the absolute 0.01: 6.7 % spread at λ ≈ 0.008, and a ratio
        # that reaches zero
        @test !COCOA.within_tolerance(0.0079875807, 0.0085628169, tol)
        @test !COCOA.within_tolerance(0.0, 0.0058408462, tol)
        # the inositol case: 0.23 % spread at λ ≈ 0.008
        @test !COCOA.within_tolerance(0.0079876, 0.0080062, tol)
        # large ratios: decided on the same relative scale
        @test !COCOA.within_tolerance(500.0, 501.0, tol)
        @test COCOA.within_tolerance(500.0, 500.0 + 1e-4, tol)
        # the verdict does not depend on which complex is the reference (λ vs 1/λ)
        for (lo, hi) in ((0.8, 0.8 + 1e-7), (0.8, 0.82), (2.0, 2.0 + 1e-6), (1e-3, 1.004e-3), (1e-3, 1e-3 + 1e-12))
            @test COCOA.within_tolerance(lo, hi, tol) == COCOA.within_tolerance(1 / hi, 1 / lo, tol)
        end
        # near zero the absolute part decides
        @test COCOA.within_tolerance(0.0, 5e-10, tol)
        @test !COCOA.within_tolerance(0.0, 1e-7, tol)
    end

    @testset "the pre-1.2 absolute 0.01 can still be chosen" begin
        abs01 = COCOA.ConcordanceTolerance(0.0, 0.01)
        @test COCOA.within_tolerance(1.0, 1.009, abs01)
        @test !COCOA.within_tolerance(1.0, 1.011, abs01)
        @test COCOA.within_tolerance(0.0079875807, 0.0085628169, abs01)   # why it was replaced
    end

    @testset "a proven unbounded LP refutes concordance" begin
        for (mult, v) in ((1, Inf), (-1, -Inf))
            s = COCOA.PairConcordanceState()
            COCOA.update_pair_concordance!(s, :positive, mult, v, false, tol)
            @test s.decisive_non_concordant
            @test !s.is_concordant
            # a later finite result cannot resurrect it
            COCOA.update_pair_concordance!(s, :positive, -mult, 1.0, false, tol)
            @test !s.is_concordant
        end
    end

    @testset "no cap on t by default" begin
        cons = COCOA.concordance_constraints(small_reference_model())
        @test cons.charnes_cooper.positive.t.bound.upper == Inf
        @test cons.charnes_cooper.negative.t.bound.lower == -Inf
    end

    # Pair LP as COCOA builds it: Charnes–Cooper template, activity of c2 fixed to 1,
    # activity of c1 maximised.
    function pair_lp(cap, c1, c2)
        cons = COCOA.concordance_constraints(small_reference_model(); cc_scale_bound=cap)
        lp = cons.charnes_cooper.positive * :fix^C.Constraint(cons.activities[c2].value, 1.0)
        om = COBREXA.optimization_model(lp; objective=cons.activities[c1].value,
                                        sense=COBREXA.Maximal, optimizer=HiGHS.Optimizer)
        J.set_silent(om)
        J.optimize!(om)
        return om
    end
    split, b, src = Symbol("B+1000000_E"), :B, Symbol("2_E")

    @testset "the old cap cuts off the only steady states" begin
        om = pair_lp(999.0, b, split)
        @test J.termination_status(om) == J.INFEASIBLE
        om = pair_lp(Inf, b, split)
        @test J.is_solved_and_feasible(om)
        @test J.objective_value(om) ≈ -1.0
    end

    @testset "an unbounded ratio is proven, not failed" begin
        om = pair_lp(Inf, src, split)          # the free source has no bound relative to v
        @test COCOA.proven_unbounded(om)
        @test !COCOA.proven_unbounded(pair_lp(Inf, b, split))
    end

    @testset "the analysis records undecided pairs and the tolerances used" begin
        r = activity_concordance_analysis(small_reference_model(); optimizer=HiGHS.Optimizer,
                                          sample_size=200)
        @test r.stats["n_unknown_pairs"] == 0 && isempty(r.stats["undecided_pairs"])
        @test r.stats["cc_scale_bound"] == Inf
        @test r.stats["concordance_tolerance"] == 1e-9 && r.stats["concordance_rtol"] == 1e-5
    end
end

# The growth bound in the pair LPs. In w = t·v the bound has to be scaled by t like every
# flux bound; before 1.2 it bounded obj(w) itself, i.e. obj(v) = obj(w)/t, which depends on
# the normalisation. With outB as objective the optimum is 1e-3 (outE ≤ 1000 allows a split
# of at most 1e-3); fixing the activity of "B + 1e6 E" to 1 gives t = 1000 and
# obj(w) = 1 ≠ 1e-3, so the unscaled bound made the LP infeasible.
@testset "Pair LP: growth bound scaled by t" begin
    m = small_reference_model()
    m.reactions["outB"].objective_coefficient = 1.0
    split, b = Symbol("B+1000000_E"), :B
    for (direction, fixed, act) in ((:positive, split, 1.0), (:negative, b, 1.0))
        cons = COCOA.concordance_constraints(m)
        opt = COBREXA.optimized_values(cons.balance; objective=cons.balance.objective.value,
                                       output=cons.balance.objective, optimizer=HiGHS.Optimizer)
        @test opt ≈ 1e-3
        COCOA.add_objective_bound!(cons, C.Between(opt, opt))     # growth at the optimum
        lp = getproperty(cons.charnes_cooper, direction) *
             :fix^C.Constraint(cons.activities[fixed].value, act)
        res = COBREXA.optimized_values(lp; objective=lp.objective.value, output=lp,
                                       optimizer=HiGHS.Optimizer)
        @test res !== nothing                                     # infeasible before 1.2
        @test res.objective / res.t ≈ opt                          # obj(v) = obj(w)/t
    end
end
