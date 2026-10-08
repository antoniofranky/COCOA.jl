# Guardrails for the ratio-CV pre-filter on sampled activities.
#
# A pair the pre-filter rejects is never tested by LP, so the CV must not be distorted.
# Two devices did distort it and are gone:
#   - rounding every sample to 0.001: for complexes whose activity is a few thousandths of
#     the flux scale, exactly constant ratios showed CVs of 0.02-0.06 and the pairs were
#     discarded (four growth-coupled uptake complexes of one yeast model ended up in no
#     concordance module);
#   - the ratio (a + 1e-16) / (b + 1e-16), which turns a sample where both complexes are
#     inactive into a ratio of 1.

using Test
using COCOA
using HiGHS

@testset "Ratio CV of sampled activities" begin
    # activities around 1, spread deterministically (golden-ratio sequence)
    growth = 0.5 .+ mod.((1:1000) .* 0.6180339887, 1.0)
    ztol = 1e-7

    @testset "exact ratios keep CV ≈ 0 at any scale" begin
        for lambda in (1e-4, 1.24e-3, 3.63e-3, 0.5, 250.0)
            cv, n = COCOA._pair_cv(lambda .* growth, growth, ztol)
            @test n == 1000
            @test cv < 1e-12
            @test COCOA._pair_cv(growth, lambda .* growth, ztol)[1] < 1e-12
        end
    end

    @testset "regression: the old rounding to 0.001 failed this case" begin
        old = x -> round(x, digits=3)
        cv_old, _ = COCOA._pair_cv(old.(3.63e-3 .* growth), old.(growth), ztol)
        @test cv_old > 0.01
    end

    @testset "samples where both complexes are inactive are skipped" begin
        a = vcat(0.5 .* growth, zeros(200))
        b = vcat(growth, fill(1e-12, 200))
        cv, n = COCOA._pair_cv(a, b, ztol)
        @test n == 1000          # the 200 joint zeros do not count
        @test cv < 1e-12         # and do not pull the ratio towards 1 (the old ε trick)
    end

    @testset "one complex inactive while the other is active is evidence" begin
        a = vcat(0.5 .* growth, zeros(10))      # a = 0 while b is active: ratio 0
        b = vcat(growth, ones(10))
        @test COCOA._pair_cv(a, b, ztol)[1] > 0.01
        # a denominator of exactly 0 gives no finite ratio and is skipped
        cv, n = COCOA._pair_cv(vcat(growth, ones(5)), vcat(growth, zeros(5)), ztol)
        @test n == 1000 && cv < 1e-12
    end

    @testset "zero tolerance is derived from balanced_threshold and recorded" begin
        model = COCOA.create_envz_ompr_model()
        r = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, sample_size=200,
                                          balanced_threshold=1e-6)
        @test r.stats["sample_zero_tol"] == 1e-6
        r = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer, sample_size=200,
                                          sample_zero_tol=1e-9)
        @test r.stats["sample_zero_tol"] == 1e-9
    end
end
