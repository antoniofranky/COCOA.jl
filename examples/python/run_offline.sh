#!/bin/bash
# Run the Python quick-start against a PRE-BUILT Julia project (offline), so
# juliacall performs no Pkg operations at runtime. Build the project first:
#
#     COCOA_PYENV=/path/to/cocoa_pyenv COCOA_DEV_PATH=$(pwd) \
#         julia examples/python/build_env.jl
#
# then run:
#
#     COCOA_PYENV=/path/to/cocoa_pyenv JULIA_BIN=$(which julia) \
#         bash examples/python/run_offline.sh
#
# On HPC you typically need a Python >= 3.10 (e.g. `module load python/3.11`).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${COCOA_PYENV:?set COCOA_PYENV to the project built by build_env.jl}"
: "${JULIA_BIN:=julia}"

export PYTHON_JULIAPKG_EXE="$JULIA_BIN"
export PYTHON_JULIAPKG_PROJECT="$COCOA_PYENV"
export PYTHON_JULIAPKG_OFFLINE=yes
export JULIA_CONDAPKG_BACKEND=Null
export PYTHONUNBUFFERED=1

echo "python: $(python --version 2>&1)  julia: $JULIA_BIN  project: $COCOA_PYENV"
python "$HERE/quickstart.py"
