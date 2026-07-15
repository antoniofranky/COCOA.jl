# Build a fully-instantiated Julia project with a NATIVE julia, so that
# juliacall never has to run Pkg operations (resolve / precompile) from inside
# the embedded libjulia runtime.
#
# Why: on some HPC systems, running Pkg.develop / Pkg.add + precompile from
# *inside* juliacall segfaults during package management. Pre-building the
# project with a normal julia and pointing juliacall at it read-only (see
# run_offline.sh) avoids that code path entirely and is also the most
# reproducible way to ship a Python entry point.
#
# Usage (native julia, NOT via juliacall):
#     julia examples/python/build_env.jl [PROJECT_DIR]
#
# PROJECT_DIR defaults to $COCOA_PYENV, else ./cocoa_pyenv.
# Set COCOA_DEV_PATH to develop a local COCOA checkout instead of the
# registered package (e.g. COCOA_DEV_PATH=$(pwd) from the repo root).
import Pkg

proj = get(ENV, "COCOA_PYENV",
    length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "cocoa_pyenv"))
mkpath(proj)
Pkg.activate(proj)

dev = get(ENV, "COCOA_DEV_PATH", "")
if !isempty(dev)
    Pkg.develop(path = dev)
else
    Pkg.add("COCOA")
end
for p in ("COBREXA", "HiGHS", "SBMLFBCModels", "AbstractFBCModels", "PythonCall")
    haskey(Pkg.project().dependencies, p) || Pkg.add(p)
end
Pkg.instantiate()
Pkg.precompile()
println("BUILD_DONE project=", proj)
