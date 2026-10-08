"""One-time installation of the Julia side (``cocoa-jl setup``)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from . import _runner

MIN_JULIA = (1, 12)
COCOA_URL = "https://github.com/antoniofranky/COCOA.jl"


def default_project() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "cocoajl" / "julia_env"


def _julia_version(julia: str) -> tuple[int, int]:
    out = subprocess.run([julia, "--startup-file=no", "--version"], capture_output=True,
                         text=True, check=True).stdout
    m = re.search(r"(\d+)\.(\d+)", out)
    if not m:
        raise RuntimeError(f"cannot read the version of {julia}: {out!r}")
    return int(m.group(1)), int(m.group(2))


def setup(project: str | os.PathLike | None = None, julia: str | None = None, *,
          cocoa_path: str | None = None, cocoa_rev: str | None = None,
          cocoa_version: str | None = None) -> Path:
    """Build the Julia project cocoajl runs on and record it as the default.

    Parameters
    ----------
    project
        Directory for the Julia project (default ``~/.local/share/cocoajl/julia_env``).
        Must be on a file system the compute nodes can read.
    julia
        The ``julia`` binary (>= 1.12). Default: ``julia`` on ``PATH``.
    cocoa_path
        Use a local COCOA.jl checkout (developed, so edits take effect).
    cocoa_rev
        Install COCOA from GitHub at this branch, tag or commit (e.g. ``"main"``).
    cocoa_version
        Install this registered version. Default: the newest registered version.

    Packages are downloaded and precompiled, which takes 5-20 minutes. Afterwards no
    network access is needed, so compute nodes without internet work.
    """
    julia = julia or shutil.which("julia")
    if not julia:
        raise RuntimeError(
            "no `julia` found. Install Julia >= 1.12, e.g. with juliaup "
            "(curl -fsSL https://install.julialang.org | sh) or `module load julia` on a "
            "cluster, then rerun, or pass the binary with --julia.")
    # Absolute, but symlinks NOT resolved: juliaup's `julia` is a link to its launcher, which
    # selects the channel from the link name.
    julia = os.path.abspath(os.path.expanduser(julia)) if os.sep in julia else shutil.which(julia)
    v = _julia_version(julia)
    if v < MIN_JULIA:
        raise RuntimeError(f"{julia} is Julia {v[0]}.{v[1]}; COCOA needs >= "
                           f"{MIN_JULIA[0]}.{MIN_JULIA[1]}")
    if sum(x is not None for x in (cocoa_path, cocoa_rev, cocoa_version)) > 1:
        raise ValueError("give at most one of cocoa_path, cocoa_rev, cocoa_version")

    project = Path(project).expanduser().resolve() if project else default_project()
    env = dict(os.environ, COCOAJL_PROJECT=str(project))
    if cocoa_path:
        env["COCOAJL_COCOA_PATH"] = str(Path(cocoa_path).expanduser().resolve())
    elif cocoa_rev:
        env.update(COCOAJL_COCOA_URL=COCOA_URL, COCOAJL_COCOA_REV=cocoa_rev)
    elif cocoa_version:
        env["COCOAJL_COCOA_VERSION"] = cocoa_version
    print(f"building Julia project {project} with {julia} (Julia {v[0]}.{v[1]})", flush=True)
    subprocess.run([julia, "--startup-file=no", "--history-file=no",
                    str(_runner.JULIA_DIR / "build_env.jl")], env=env, check=True)

    cfg = _runner.read_config()
    cfg.update(project=str(project), julia=julia)
    path = _runner.write_config(cfg)
    print(f"recorded in {path}", flush=True)
    return project
