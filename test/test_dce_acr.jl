# Regression tests for the flux-coupling (DCE) ACR/ACRR detector,
# `identify_acr_acrr_dce`, on the autocatalytic toy CRNs from the application note.
# These networks have species on both sides of reactions (A+D->2A, ...), so they are
# supplied as explicit-complex models.

using Test
using COCOA
import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel as CM

# --- minimal explicit-complex model (carries substrate/product complex per reaction) ---
struct DCEToyModel <: A.AbstractFBCModel
    inner::CM.Model
    complex_compositions::Dict{Symbol,Vector{Tuple{Symbol,Float64}}}
    reaction_complex_map::Dict{Symbol,Tuple{Symbol,Symbol}}
    complex_order::Vector{Symbol}
end
A.reactions(m::DCEToyModel) = A.reactions(m.inner)
A.metabolites(m::DCEToyModel) = A.metabolites(m.inner)
A.n_reactions(m::DCEToyModel) = A.n_reactions(m.inner)
A.n_metabolites(m::DCEToyModel) = A.n_metabolites(m.inner)
A.stoichiometry(m::DCEToyModel) = A.stoichiometry(m.inner)
A.bounds(m::DCEToyModel) = A.bounds(m.inner)
A.objective(m::DCEToyModel) = A.objective(m.inner)
A.reaction_stoichiometry(m::DCEToyModel, rid::String) = A.reaction_stoichiometry(m.inner, rid)
# These toy CRNs are closed (no boundary reactions) and carry explicit autocatalytic
# complexes, so the zero-complex flag is accepted for signature compatibility but ignored.
COCOA._extract_complexes_from_model(m::DCEToyModel; include_zero_complex::Bool=false) =
    (complexes=m.complex_compositions, reaction_complex_map=m.reaction_complex_map,
     complex_order=m.complex_order)

function build_toy(species, reactions)
    inner = CM.Model()
    for s in species
        inner.metabolites[s] = CM.Metabolite(name=s)
    end
    comps = Dict{Symbol,Vector{Tuple{Symbol,Float64}}}()
    order = Symbol[]
    rmap = Dict{Symbol,Tuple{Symbol,Symbol}}()
    cid(comp) = begin
        items = sort([(Symbol(k), v) for (k, v) in comp]; by=x -> x[1])
        id = COCOA.generate_complex_id(items)
        haskey(comps, id) || (comps[id] = items; push!(order, id))
        id
    end
    for (rid, sub, prod) in reactions
        rmap[Symbol(rid)] = (cid(sub), cid(prod))
        net = Dict{String,Float64}()
        for (k, v) in sub; net[k] = get(net, k, 0.0) - v; end
        for (k, v) in prod; net[k] = get(net, k, 0.0) + v; end
        inner.reactions[rid] = CM.Reaction(name=rid, stoichiometry=net, lower_bound=0.0, upper_bound=1000.0)
    end
    DCEToyModel(inner, comps, rmap, order)
end
D(p...) = Dict{String,Float64}(p...)

@testset "DCE flux-coupling ACR/ACRR (toy CRNs)" begin
    # Example 1 (ACR + DCE): ground truth [D] and [B] are ACR.
    ex1 = build_toy(["A","B","C","D"], [
        ("R1", D("A"=>1.0), D("B"=>1.0)),
        ("R2", D("B"=>1.0), D("D"=>1.0)),
        ("R3", D("A"=>1.0,"D"=>1.0), D("A"=>2.0)),
        ("R4", D("C"=>1.0,"D"=>1.0), D("C"=>2.0)),
        ("R5", D("B"=>1.0,"C"=>1.0), D("B"=>2.0)),
    ])
    r1 = identify_acr_acrr_dce(ex1)
    @test Set(r1.acr_metabolites) == Set([:B, :D])
    @test Set(r1.acrr_pairs) == Set([(:B, :D)])

    # Example 2 (ACRR + DCE): ground truth ACRR pair (A,D), no single-metabolite ACR.
    ex2 = build_toy(["A","B","C","D"], [
        ("R1", D("D"=>1.0), D("A"=>1.0)),
        ("R2", D("A"=>1.0), D("B"=>1.0)),
        ("R3", D("B"=>1.0), D("C"=>1.0)),
        ("R4", D("B"=>1.0,"D"=>1.0), D("D"=>2.0)),
        ("R5", D("C"=>1.0,"D"=>1.0), D("D"=>2.0)),
    ])
    r2 = identify_acr_acrr_dce(ex2)
    @test isempty(r2.acr_metabolites)
    @test Set(r2.acrr_pairs) == Set([(:A, :D)])

    # Example 3 (linear + non-linear): ground truth [B] ACR and ACRR pair (A,D).
    ex3 = build_toy(["A","B","C","D","E"], [
        ("R1", D("D"=>1.0), D("A"=>1.0)),
        ("R2", D("A"=>1.0), D("B"=>1.0)),
        ("R3", D("B"=>1.0), D("C"=>1.0)),
        ("R4", D("A"=>1.0,"C"=>1.0), D("A"=>1.0,"B"=>1.0)),
        ("R5", D("A"=>1.0,"B"=>1.0), D("A"=>2.0)),
        ("R6", D("A"=>2.0), D("A"=>1.0,"D"=>1.0)),
        ("R7", D("C"=>1.0,"D"=>1.0), D("B"=>1.0,"D"=>1.0)),
        ("R8", D("A"=>1.0,"E"=>1.0), D("E"=>2.0)),
        ("R9", D("D"=>1.0,"E"=>1.0), D("D"=>2.0)),
    ])
    r3 = identify_acr_acrr_dce(ex3)
    @test Set(r3.acr_metabolites) == Set([:B])
    @test (:A, :D) in Set(r3.acrr_pairs)

    # No-op / propagation sanity: a known ACR can only ADD metabolites.
    r3b = identify_acr_acrr_dce(ex3; known_acr=[:A])
    @test Set([:B]) ⊆ Set(r3b.acr_metabolites)
end
