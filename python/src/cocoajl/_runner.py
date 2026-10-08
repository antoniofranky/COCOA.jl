"""Running jobs in Julia.

Every analysis runs as a separate ``julia`` process: the Python side writes a job file
(JSON, every parameter resolved) and starts ``julia --project=<env> driver.jl job.json``.
Output is streamed to the terminal; results are files. Keeping Julia out of the Python
process avoids the problems of embedding it (library clashes such as OpenSSL, version pins,
crashes during package installation on clusters), and any job can be rerun by hand with
the command recorded next to its job file.

The Julia project and binary come from, in this order: :func:`configure`, the environment
variables ``COCOAJL_PROJECT`` / ``COCOAJL_JULIA``, the config file written by
``cocoa-jl setup`` (``$COCOAJL_CONFIG`` or ``~/.config/cocoajl/config.json``).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

JULIA_DIR = Path(__file__).parent / "julia"
_override: dict = {}


class CocoaError(RuntimeError):
    """A Julia job failed; the message names the job file and the command to rerun it."""


def config_path() -> Path:
    if os.environ.get("COCOAJL_CONFIG"):
        return Path(os.environ["COCOAJL_CONFIG"])
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "cocoajl" / "config.json"


def read_config() -> dict:
    p = config_path()
    return json.loads(p.read_text()) if p.is_file() else {}


def write_config(cfg: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1) + "\n")
    return p


def configure(project: str | os.PathLike | None = None, julia: str | None = None) -> None:
    """Use this Julia project / binary for the jobs of this Python process."""
    if project is not None:
        _override["project"] = str(Path(project).expanduser().resolve())
    if julia is not None:
        _override["julia"] = julia


def settings() -> dict:
    """The Julia project and binary that jobs will use, and where each came from."""
    cfg = read_config()
    out = {}
    for key, env in (("project", "COCOAJL_PROJECT"), ("julia", "COCOAJL_JULIA")):
        if _override.get(key):
            out[key], out[key + "_source"] = _override[key], "configure()"
        elif os.environ.get(env):
            out[key], out[key + "_source"] = os.environ[env], env
        elif cfg.get(key):
            out[key], out[key + "_source"] = cfg[key], str(config_path())
        else:
            out[key], out[key + "_source"] = None, "not set"
    return out


def command(job_file: str | os.PathLike) -> list[str]:
    """The command line that executes a job file."""
    s = settings()
    if not s["project"] or not (Path(s["project"]) / "Project.toml").is_file():
        raise CocoaError("the Julia side is not set up: run `cocoa-jl setup` first "
                         f"(project: {s['project']}, from {s['project_source']})")
    julia = s["julia"] or "julia"
    return [julia, f"--project={s['project']}", "--startup-file=no", "--history-file=no",
            str(JULIA_DIR / "driver.jl"), str(job_file)]


def run_job(job: dict, job_file: str | os.PathLike, quiet: bool = False) -> None:
    """Write ``job`` to ``job_file`` and execute it; raise :class:`CocoaError` on failure.

    The job's output (progress messages, warnings) goes to this process's stdout/stderr.
    """
    job_file = Path(job_file)
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(json.dumps(job, indent=1) + "\n")
    cmd = command(job_file)
    env = dict(os.environ, JULIA_NUM_THREADS=os.environ.get("JULIA_NUM_THREADS", "1"))
    out = subprocess.DEVNULL if quiet else None
    rc = subprocess.run(cmd, env=env, stdout=out, stderr=out).returncode
    if rc != 0:
        raise CocoaError(f"Julia job failed (exit code {rc}). Job file: {job_file}\n"
                         f"Rerun by hand with:\n  {shlex.join(cmd)}")
