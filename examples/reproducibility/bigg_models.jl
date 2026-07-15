# Download-on-demand for public BiGG genome-scale models, with SHA-256
# verification. We deliberately do NOT ship these models inside the package
# (they total ~25 MB); instead we fetch them from the public, stable BiGG
# endpoint the first time they are needed and cache them locally.
#
# Provenance: the pinned checksums below are for the canonical BiGG SBML
# (http://bigg.ucsd.edu/static/models/<id>.xml). These files are byte-for-byte
# identical to the `Models/original/*.xml` used by the MATLAB reference
# implementation, so results are directly comparable.

using Downloads, SHA

# id => SHA-256 of the canonical BiGG SBML.
const BIGG_SHA256 = Dict(
    "e_coli_core" => "b4db506aeed0e434c1f5f1fdd35feda0dfe5d82badcfda0e9d1342335ab31116",
    "iAB_RBC_283" => "784511b2ae5c5a1803eef5b896e4afb5d40f75341147b42404b75dc09abb7ffe",
    "iAF1260b"    => "831f72a504a2f38e41556527a54846b09e954fdce8d11842e747b9788d410480",
    "iJR904"      => "a0d9cdc34c45a04599ca136a2938e6d8c42b93de6a07df90379ca61965de9250",
    "iMM904"      => "a4bcff4f8f2228a5b1553264c3bcb1e8a8093907754446007f3f3b529d29d3e1",
)

bigg_url(id::AbstractString) = "http://bigg.ucsd.edu/static/models/$(id).xml"

# Cache dir: $COCOA_MODEL_CACHE, else a `bigg` dir next to this file.
bigg_cache_dir() = get(ENV, "COCOA_MODEL_CACHE", joinpath(@__DIR__, "bigg"))

_sha256_file(path) = open(io -> bytes2hex(SHA.sha256(io)), path)

"""
    fetch_bigg_model(id) -> path

Return a local path to the BiGG SBML for `id`, downloading it from BiGG (with
SHA-256 verification) into the cache on first use. Raises if the checksum does
not match the pinned value.
"""
function fetch_bigg_model(id::AbstractString)
    haskey(BIGG_SHA256, id) ||
        error("Unknown BiGG id '$id'. Known: $(join(sort(collect(keys(BIGG_SHA256))), ", "))")
    dir = bigg_cache_dir(); mkpath(dir)
    path = joinpath(dir, "$(id).xml")
    want = BIGG_SHA256[id]
    if isfile(path) && _sha256_file(path) == want
        return path
    end
    tmp = path * ".tmp"
    @info "Fetching BiGG model" id url=bigg_url(id)
    Downloads.download(bigg_url(id), tmp)
    got = _sha256_file(tmp)
    got == want || (rm(tmp; force=true);
        error("SHA-256 mismatch for $id: got $got, expected $want"))
    mv(tmp, path; force=true)
    return path
end

# True if `spec` should be resolved via BiGG rather than treated as a file path.
is_bigg_id(spec::AbstractString) = haskey(BIGG_SHA256, spec)
