"""Command line interface: ``cocoa-jl <command> ...`` (``cocoa-jl <command> --help``)."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .api import CONCORDANCE_PARAMS, PREPROCESS_PARAMS

HELP = {
    "binding": "elementary-step mechanism: ordered (default), random, or none (no splitting; "
               "kinetic modules then lack theoretical backing)",
    "random_fraction": "with --binding random: fraction of enzymes split randomly (default 1)",
    "flux_tol": "flux below which a reaction is blocked (default 1e-6; use 1e-9 if the run "
                "stops because removal changed growth)",
    "growth_window": "growth range (× optimum) while blocked reactions are found, "
                     "as LO:HI (default 0.001:1)",
    "split_seed": "seed of the random mechanism assignment (default 42)",
    "sample_size": "flux samples for the CV pre-filter (default 5000)",
    "cv_threshold": "pairs whose sampled activity ratio has a larger coefficient of variation "
                    "are not tested by LP (default 0.01)",
    "concordance_rtol": "maximum relative spread of the activity ratio for two complexes to "
                        "count as concordant (default 1e-5; concordance is exact, this absorbs "
                        "solver error)",
    "concordance_tolerance": "absolute part of the concordance rule, for ratios at zero "
                             "(default 1e-9; 0.01 with concordance_rtol 0 is the pre-1.2 rule)",
    "balanced_threshold": "activity range below which a complex is balanced (default 1e-7)",
    "seed": "sampling seed (default 42)",
    "use_transitivity": "skip pairs whose verdict follows from earlier ones (default on)",
    "growth_fraction": "none (default): full steady-state cone; a number f: growth held at "
                       ">= f × optimum",
    "solver_tol": "HiGHS feasibility tolerance (default 1e-8)",
    "lp_time_limit": "seconds per LP (default 3600)",
}


def parse_value(name: str, text: str):
    """Command-line text → value of the parameter's type."""
    default = {**PREPROCESS_PARAMS, **CONCORDANCE_PARAMS}[name]
    if name == "growth_window":
        lo, hi = text.split(":")
        return (float(lo), float(hi))
    if name == "growth_fraction":
        return None if text.lower() in ("none", "") else float(text)
    if isinstance(default, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name}: expected true/false, got {text!r}")
    if isinstance(default, int):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    return text


def parse_grid(items: list[str]) -> dict:
    grid = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--grid expects NAME=V1,V2,..., got {item!r}")
        name, vals = item.split("=", 1)
        name = name.strip().replace("-", "_")
        if name not in PREPROCESS_PARAMS and name not in CONCORDANCE_PARAMS:
            raise SystemExit(f"--grid: unknown parameter {name!r}; known: "
                             + ", ".join([*PREPROCESS_PARAMS, *CONCORDANCE_PARAMS]))
        grid[name] = [parse_value(name, v.strip()) for v in vals.split(",")]
    return grid


def _add_params(p: argparse.ArgumentParser, which: dict, title: str) -> None:
    g = p.add_argument_group(title)
    for name in which:
        flag = "--" + name.replace("_", "-")
        if name == "use_transitivity":
            g.add_argument("--no-transitivity", dest=name, action="store_false",
                           default=argparse.SUPPRESS, help="test every candidate pair by LP")
            continue
        kw = dict(dest=name, default=argparse.SUPPRESS, help=HELP.get(name))
        if name == "binding":
            kw["choices"] = ["ordered", "random", "none"]
        else:
            kw.update(metavar="X", type=lambda t, n=name: parse_value(n, t))
        g.add_argument(flag, **kw)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", type=int, default=0,
                   help="Julia worker processes for the LPs (default 0: serial). Use the "
                        "number of cores minus one; under SLURM --cpus-per-task minus one")
    p.add_argument("--project", default=None,
                   help="Julia project to use instead of the one recorded by `cocoa-jl setup`")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cocoa-jl",
        description="Concordance and kinetic modules of metabolic networks with COCOA.jl.")
    from . import __version__  # noqa: PLC0415
    ap.add_argument("--version", action="version", version=f"cocoajl {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("setup", help="install the Julia side once (run this first)",
                       description="Build the Julia project with a native julia and record "
                                   "it as the default. Takes 5-20 minutes, once.")
    p.add_argument("--project", default=None,
                   help="directory for the Julia project (default ~/.local/share/cocoajl/julia_env)")
    p.add_argument("--julia", default=None, help="julia binary, >= 1.12 (default: on PATH)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--cocoa-path", help="use this local COCOA.jl checkout")
    src.add_argument("--cocoa-rev", help="install COCOA.jl from GitHub at this branch/tag/commit")
    src.add_argument("--cocoa-version", help="install this registered COCOA.jl version")

    p = sub.add_parser("info", help="show the Julia project, versions and settings in use")
    p.add_argument("--project", default=None)
    p.add_argument("--no-julia", action="store_true",
                   help="only show the configuration, do not start Julia")

    p = sub.add_parser("run", help="model file → kinetic modules",
                       description="Load, preprocess, concordance analysis, kinetic modules. "
                                   "Writes complexes.csv, kinetic_modules.csv, acr.csv, "
                                   "acrr.csv, model_preprocessed.xml, concordance.jld2 and "
                                   "run.json to OUTDIR.")
    p.add_argument("model", help="model file: SBML .xml, COBRApy .json or COBRA .mat")
    p.add_argument("outdir")
    _add_common(p)
    p.add_argument("--mat-key", help="which model of a .mat file holding several")
    p.add_argument("--no-preprocess", action="store_true",
                   help="use the model as given (already split into elementary, irreversible steps)")
    p.add_argument("--exhaustive", action="store_true",
                   help="full matrix-based kinetic algorithm (slower, can merge more modules)")
    p.add_argument("--no-acr", action="store_true", help="do not write acr.csv / acrr.csv")
    _add_params(p, PREPROCESS_PARAMS, "preprocessing")
    _add_params(p, CONCORDANCE_PARAMS, "concordance analysis")

    p = sub.add_parser("preprocess", help="model file → preprocessed model (SBML)")
    p.add_argument("model")
    p.add_argument("output", help="output .xml")
    _add_common(p)
    p.add_argument("--mat-key")
    _add_params(p, PREPROCESS_PARAMS, "preprocessing")
    p.add_argument("--solver-tol", type=float, default=argparse.SUPPRESS, metavar="X",
                   help=HELP["solver_tol"])

    p = sub.add_parser("kinetic", help="recompute kinetic modules of an earlier run",
                       description="Kinetic modules from the concordance.jld2 of a `run` "
                                   "directory, without repeating the concordance analysis.")
    p.add_argument("rundir")
    p.add_argument("--out", default=None, help="output directory (default: RUNDIR)")
    p.add_argument("--project", default=None)
    p.add_argument("--exhaustive", action="store_true")
    p.add_argument("--no-acr", action="store_true")

    p = sub.add_parser("sweep", help="kinetic modules for a grid of parameter values",
                       description="One run per combination of --grid values, each in its own "
                                   "subdirectory of OUTDIR, plus summary.csv. Preprocessing is "
                                   "done once per preprocessing setting; finished combinations "
                                   "are skipped on restart.")
    p.add_argument("model")
    p.add_argument("outdir")
    p.add_argument("--grid", action="append", required=True, metavar="NAME=V1,V2,...",
                   help="parameter and values, e.g. cv_threshold=0.01,0.03 "
                        "(repeat for more parameters; all combinations are run)")
    _add_common(p)
    p.add_argument("--mat-key")
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--exhaustive", action="store_true")
    p.add_argument("--write-acr", action="store_true", help="also write acr.csv / acrr.csv per run")
    _add_params(p, PREPROCESS_PARAMS, "fixed preprocessing parameters")
    _add_params(p, CONCORDANCE_PARAMS, "fixed concordance parameters")
    return ap


def _params(ns: argparse.Namespace) -> dict:
    names = {*PREPROCESS_PARAMS, *CONCORDANCE_PARAMS}
    return {k: v for k, v in vars(ns).items() if k in names}


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    from . import _runner, api, setup_env  # noqa: PLC0415

    if ns.command == "setup":
        setup_env.setup(ns.project, ns.julia, cocoa_path=ns.cocoa_path,
                        cocoa_rev=ns.cocoa_rev, cocoa_version=ns.cocoa_version)
        print("done. Check with: cocoa-jl info")
        return 0
    if getattr(ns, "project", None):
        _runner.configure(project=ns.project)
    try:
        if ns.command == "info":
            print(json.dumps({"config_file": str(_runner.config_path()), **_runner.settings()},
                             indent=1))
            if not ns.no_julia:
                print(json.dumps(api.versions(), indent=1))
        elif ns.command == "run":
            api.run(ns.model, ns.outdir, preprocess_model=not ns.no_preprocess,
                    mat_key=ns.mat_key, workers=ns.workers or 0, exhaustive=ns.exhaustive,
                    write_acr=not ns.no_acr, **_params(ns))
        elif ns.command == "preprocess":
            report = api.preprocess(ns.model, ns.output, workers=ns.workers or 0,
                                    mat_key=ns.mat_key, **_params(ns))
            print(f"wrote {ns.output} ({report['final']['reactions']} reactions)")
        elif ns.command == "kinetic":
            k = api.kinetic_from_run(ns.rundir, ns.out, exhaustive=ns.exhaustive,
                                     write_acr=not ns.no_acr)
            print(f"{len(k.modules)} kinetic modules, largest {k.largest_module} complexes")
        elif ns.command == "sweep":
            s = api.sweep(ns.model, ns.outdir, parse_grid(ns.grid), mat_key=ns.mat_key,
                          preprocess_model=not ns.no_preprocess, workers=ns.workers or 0,
                          exhaustive=ns.exhaustive, write_acr=ns.write_acr, **_params(ns))
            print(f"{len(s)} runs; summary in {os.path.join(ns.outdir, 'summary.csv')}")
    except _runner.CocoaError as e:
        print(f"cocoa-jl: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
