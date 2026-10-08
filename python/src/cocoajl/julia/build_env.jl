# Build the Julia project that cocoajl runs its jobs in. Called by `cocoa-jl setup`. A
# pre-built, precompiled project lets every later job start without any package operation
# and without network access (compute nodes without internet).
#
# Configuration via environment variables (set by cocoa-jl setup):
#   COCOAJL_PROJECT        project directory to create (required)
#   COCOAJL_COCOA_PATH     develop this local COCOA.jl checkout, or
#   COCOAJL_COCOA_URL      add COCOA from this git URL at COCOAJL_COCOA_REV, or
#   COCOAJL_COCOA_VERSION  add this registered version (default: newest)
import Pkg

proj = ENV["COCOAJL_PROJECT"]
mkpath(proj)
Pkg.activate(proj)

path = get(ENV, "COCOAJL_COCOA_PATH", "")
url = get(ENV, "COCOAJL_COCOA_URL", "")
if !isempty(path)
    Pkg.develop(path=path)
elseif !isempty(url)
    Pkg.add(url=url, rev=get(ENV, "COCOAJL_COCOA_REV", "main"))
else
    v = get(ENV, "COCOAJL_COCOA_VERSION", "")
    isempty(v) ? Pkg.add("COCOA") : Pkg.add(name="COCOA", version=v)
end
for p in ("COBREXA", "HiGHS", "AbstractFBCModels", "SBMLFBCModels", "JSONFBCModels",
          "MATFBCModels", "JLD2", "JSON")
    haskey(Pkg.project().dependencies, p) || Pkg.add(p)
end

Pkg.instantiate()
Pkg.precompile()
# Load once so that a broken install fails here, not hours into the first run.
@eval using COCOA, HiGHS, COBREXA, SBMLFBCModels, JSONFBCModels, MATFBCModels, JLD2, JSON
println("COCOAJL_BUILD_DONE ", proj, " COCOA ", pkgversion(COCOA))
