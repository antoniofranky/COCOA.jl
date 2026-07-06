# Determinism / reproducibility guardrails.
#
# These lock in the reproducibility invariants:
#   1. Preprocessing (esp. split_into_elementary) is deterministic across repeated calls
#      and independent of Dict hash order — no rand() seed default, canonical binding
#      order, canonical enzyme numbering.
#   2. The concordance pipeline gives byte-identical partitions on a re-run (same seed).
#
# A regression in any of the ordering / seeding fixes will flip one of these.

using Test
using COCOA
using COBREXA
using HiGHS
import SBMLFBCModels
import AbstractFBCModels as A

# Label-invariant canonical partition: group complex ids by concordance module,
# sort within and across groups. Equal partitions compare equal regardless of module
# numbering or complex ordering.
function _canonical_partition(result)
    groups = Dict{Int,Vector{String}}()
    for (c, m) in zip(result.complexes.complex_id, result.complexes.concordance_module)
        push!(get!(Vector{String}, groups, m), c)
    end
    return sort([sort(v) for (_, v) in groups])
end

_preprocess(canon) = canon |>
    normalize_bounds |>
    m -> remove_blocked_reactions(m; optimizer=HiGHS.Optimizer) |>
    remove_orphans |>
    split_into_elementary |>
    split_into_irreversible

@testset "Reproducibility / determinism" begin

    @testset "Preprocessing is deterministic (canonical order, fixed seed)" begin
        # Typed load: the untyped load_model guesses the type from the filename by
        # iterating ALL registered AbstractFBCModel subtypes, which errors once the
        # test suite has defined toy model types (e.g. DCEToyModel) lacking
        # filename_extensions. Loading with an explicit type bypasses that.
        model = A.load(SBMLFBCModels.SBMLFBCModel, joinpath(pkgdir(COCOA), "test", "e_coli_core.xml"))
        canon = convert(A.CanonicalModel.Model, model)

        mp1 = _preprocess(canon)
        mp2 = _preprocess(canon)

        # Same reactions and metabolites produced, regardless of Dict hash order.
        @test sort(collect(keys(mp1.reactions))) == sort(collect(keys(mp2.reactions)))
        @test sort(collect(keys(mp1.metabolites))) == sort(collect(keys(mp2.metabolites)))

        # Same complex set AND same canonical complex ordering (drives matrix indexing).
        _, ids1 = COCOA.incidence(mp1; return_ids=true)
        _, ids2 = COCOA.incidence(mp2; return_ids=true)
        @test ids1 == ids2
        println("    ✓ preprocessing deterministic: $(length(ids1)) complexes, identical order")
    end

    @testset "Concordance partition is identical on re-run (same seed)" begin
        model = create_envz_ompr_model()
        r1 = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer,
                                           seed=UInt(1234), kinetic_analysis=true)
        r2 = activity_concordance_analysis(model; optimizer=HiGHS.Optimizer,
                                           seed=UInt(1234), kinetic_analysis=true)
        @test _canonical_partition(r1) == _canonical_partition(r2)
        println("    ✓ concordance re-run reproducible on EnvZ-OmpR")
    end

end
