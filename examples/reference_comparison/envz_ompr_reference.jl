# Exact-agreement check on a numerically stable example (reviewer R1, 2b1-ii).
#
# The EnvZ-OmpR two-component system is the canonical absolute-concentration-
# robustness (ACR) example. Its result is analytically known and is reproduced
# by the MATLAB reference implementation, so it is the right "numerically stable
# example" on which to demonstrate *exact* agreement.
#
# Reference answer (kinetic-modules paper, Section S.5.2):
#   * ACR  = {Yp}                              (OmpR-phosphorylated)
#   * ACRR = 15 pairs = C(6,2) over the giant module {X, XD, XDYp, XT, XTYp, XpY}
#
# Notes
# -----
# * Use the exhaustive detector (`kinetic_efficient=false`): the fast default is
#   a lower bound and, as the reviewer observed (2b3), can miss the EnvZ-OmpR ACR.
# * `create_envz_ompr_model` is the paper CRN and is exported by COCOA; we run it
#   directly rather than the SBML-through-preprocessing network (elementary
#   splitting decorates the network with enzyme complexes that mask the ACR).
#
# Run:
#   julia --project examples/reference_comparison/envz_ompr_reference.jl

using COCOA
import HiGHS

const REF_ACR  = Set(["Yp"])
const REF_ACRR = 15
const REF_ACRR_METS = Set(["X", "XD", "XDYp", "XT", "XTYp", "XpY"])

model = create_envz_ompr_model()

result = activity_concordance_analysis(
    model;
    optimizer = HiGHS.Optimizer,
    kinetic_analysis = true,
    kinetic_efficient = false,     # exhaustive ACR/ACRR detection
    use_transitivity = true,
    concordance_tolerance = 0.01,
    cv_threshold = 0.01,
)

acr  = Set(String.(collect(result.acr.metabolite_id)))
acrr_pairs = collect(zip(String.(result.acrr.metabolite_1), String.(result.acrr.metabolite_2)))
acrr_mets  = Set(vcat(first.(acrr_pairs), last.(acrr_pairs)))

println("COCOA ACR       : ", sort(collect(acr)))
println("reference ACR   : ", sort(collect(REF_ACR)))
println("COCOA ACRR count: ", length(acrr_pairs), "   reference: ", REF_ACRR)
println("COCOA ACRR mets : ", sort(collect(acrr_mets)))

ok_acr   = acr == REF_ACR
ok_acrr  = length(acrr_pairs) == REF_ACRR
ok_mets  = acrr_mets == REF_ACRR_METS
println("\nACR  matches reference: ", ok_acr)
println("ACRR matches reference: ", ok_acrr && ok_mets)

if ok_acr && ok_acrr && ok_mets
    println("\nEXACT AGREEMENT with reference on EnvZ-OmpR ✓")
else
    error("EnvZ-OmpR result does NOT match the reference — investigate.")
end
