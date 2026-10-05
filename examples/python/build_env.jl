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
for p in ("COBREXA", "HiGHS", "SBMLFBCModels", "AbstractFBCModels")
    haskey(Pkg.project().dependencies, p) || Pkg.add(p)
end

# PythonCall.jl must have EXACTLY the version of the Python package `juliacall`, or juliacall
# stops at start-up with "PythonCall.jl did not start properly". Take the version from the
# Python that will run the scripts ($COCOA_PYTHON, default `python3`), or from
# $PYTHONCALL_VERSION if that Python is not available here.
pyver = get(ENV, "PYTHONCALL_VERSION", "")
if isempty(pyver)
    py = get(ENV, "COCOA_PYTHON", "python3")
    pyver = try
        strip(read(`$py -c "import importlib.metadata as m; print(m.version('juliacall'))"`, String))
    catch
        ""
    end
end
if isempty(pyver)
    @warn "Could not determine the juliacall version; installing the latest PythonCall. " *
          "If juliacall reports 'PythonCall.jl did not start properly', rerun with " *
          "PYTHONCALL_VERSION=<output of `pip show juliacall`>."
    Pkg.add("PythonCall")
else
    println("pinning PythonCall to juliacall's version ", pyver)
    Pkg.add(name = "PythonCall", version = pyver)
end
Pkg.instantiate()
Pkg.precompile()
println("BUILD_DONE project=", proj)
