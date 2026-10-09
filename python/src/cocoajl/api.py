"""Python API: run the pipeline, sweep parameters, read results."""
from __future__ import annotations

import csv
import itertools
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from . import _runner

__all__ = [
    "Table", "Result", "run", "preprocess", "kinetic_from_run", "sweep", "load_result",
    "example_model", "versions", "PREPROCESS_PARAMS", "CONCORDANCE_PARAMS", "resolve_params",
]

# Every tunable parameter with its default, by stage. A sweep preprocesses again only when
# a preprocessing parameter changes.
PREPROCESS_PARAMS: dict[str, Any] = dict(
    binding="ordered",           # "ordered", "random" or "none" (no elementary steps)
    random_fraction=1.0,         # with binding="random": fraction of enzymes split randomly
    flux_tol=1e-6,               # flux below which a reaction counts as blocked
    growth_window=(0.001, 1.0),  # growth held in this range × optimum while finding blocked
    split_seed=42,               # seed of the random mechanism assignment
)
CONCORDANCE_PARAMS: dict[str, Any] = dict(
    sample_size=5000,            # flux samples for the CV pre-filter
    cv_threshold=0.01,           # pairs whose sampled ratio varies more are not LP-tested
    concordance_rtol=1e-5,       # max relative spread of the activity ratio for "concordant"
    concordance_tolerance=1e-9,  # absolute part of that rule, for ratios at zero
    balanced_threshold=1e-7,     # activity range below which a complex counts as balanced
    seed=42,                     # sampling seed
    use_transitivity=True,       # skip pairs whose verdict follows from earlier ones
    growth_fraction=None,        # None: full steady-state cone; f: growth >= f × optimum
    solver_tol=1e-8,             # HiGHS primal/dual feasibility tolerance
    lp_time_limit=3600.0,        # seconds per LP
)
ALL_PARAMS = {**PREPROCESS_PARAMS, **CONCORDANCE_PARAMS}


def resolve_params(params: Mapping[str, Any]) -> dict:
    """Defaults overlaid with ``params``; rejects unknown names."""
    unknown = set(params) - set(ALL_PARAMS)
    if unknown:
        raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}. Known: "
                        + ", ".join(ALL_PARAMS) + ". Expert keywords of "
                        "activity_concordance_analysis go in extra={...}.")
    p = {**ALL_PARAMS, **params}
    p["growth_window"] = [float(x) for x in p["growth_window"]]
    if p["binding"] not in ("ordered", "random", "none"):
        raise ValueError(f"binding must be 'ordered', 'random' or 'none', got {p['binding']!r}")
    return p


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------

def _convert(text: str):
    if text == "":
        return None
    if text in ("true", "false"):
        return text == "true"
    for f in (int, float):
        try:
            return f(text)
        except ValueError:
            pass
    return text


@dataclass
class Table:
    """A column table (dict of equally long lists) with CSV and pandas export."""

    columns: dict[str, list] = field(default_factory=dict)

    @classmethod
    def read_csv(cls, path: str | os.PathLike) -> "Table":
        with open(path, newline="") as f:
            r = csv.reader(f)
            names = next(r, [])
            cols: dict[str, list] = {n: [] for n in names}
            for row in r:
                for n, v in zip(names, row):
                    cols[n].append(_convert(v))
        return cls(cols)

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()), []))

    def __getitem__(self, name: str) -> list:
        return self.columns[name]

    def rows(self) -> Iterator[dict]:
        for vals in zip(*self.columns.values()):
            yield dict(zip(self.columns, vals))

    def to_pandas(self):
        """As a ``pandas.DataFrame`` (requires pandas)."""
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame(self.columns)

    def to_csv(self, path: str | os.PathLike) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.names)
            # same conventions as the CSV files Julia writes: true/false, empty for missing
            fmt = lambda v: ("true" if v else "false") if isinstance(v, bool) else v  # noqa: E731
            w.writerows([fmt(v) for v in row] for row in zip(*self.columns.values()))

    def __repr__(self) -> str:
        return f"Table({len(self)} rows: {', '.join(self.names)})"


@dataclass
class Result:
    """The results of one run, read from its output directory.

    ``modules``: one row per kinetic module (``kinetic_module`` 1 = largest,
    ``n_complexes``, ``complexes``). ``complexes``: one row per complex with its
    ``concordance_module`` (0 = balanced, -1 = in no module), ``kinetic_module`` (0 = none),
    classification and activity range. ``acr`` / ``acrr``: ACR metabolites and ACRR pairs
    (None if not written). ``info``: the run's ``run.json`` (parameters, preprocessing
    report, versions, timings); ``stats`` its counts.
    """

    outdir: Path
    modules: Table
    complexes: Table
    acr: Table | None
    acrr: Table | None
    info: dict

    @property
    def stats(self) -> dict:
        return self.info.get("stats", {})

    @property
    def largest_module(self) -> int:
        return self.modules["n_complexes"][0] if len(self.modules) else 0

    def __repr__(self) -> str:
        return (f"Result({self.outdir}: {len(self.modules)} kinetic modules, largest "
                f"{self.largest_module} complexes)")


def load_result(outdir: str | os.PathLike) -> Result:
    """Read the results of a finished run (or ``cocoa-jl kinetic``) from its directory."""
    d = Path(outdir)
    info_file = d / "run.json" if (d / "run.json").is_file() else d / "kinetic.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"{d} holds no finished run (no run.json)")
    opt = lambda name: Table.read_csv(d / name) if (d / name).is_file() else None  # noqa: E731
    return Result(d, Table.read_csv(d / "kinetic_modules.csv"),
                  Table.read_csv(d / "complexes.csv"), opt("acr.csv"), opt("acrr.csv"),
                  json.loads(info_file.read_text()))


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------

def _abs(p) -> str:
    return str(Path(p).expanduser().resolve())


def _job(command: str, *, model=None, mat_key=None, preprocess_model=True, params=None,
         extra=None, workers=0, exhaustive=False, write_acr=True, **more) -> dict:
    return dict(command=command, model=_abs(model) if model else None, mat_key=mat_key,
                preprocess=bool(preprocess_model), params=params, extra=dict(extra or {}),
                workers=int(workers or 0), exhaustive=bool(exhaustive),
                write_acr=bool(write_acr), **more)


def run(model: str | os.PathLike, outdir: str | os.PathLike, *, workers: int = 0,
        preprocess_model: bool = True, mat_key: str | None = None, exhaustive: bool = False,
        write_acr: bool = True, extra: Mapping[str, Any] | None = None,
        quiet: bool = False, **params) -> Result:
    """Model file → kinetic modules. Runs in Julia; returns the results read from ``outdir``.

    Steps: load → preprocess (skip with ``preprocess_model=False`` for a model that is
    already split into elementary, irreversible steps) → concordance analysis → kinetic
    modules. ``params``: any of :data:`PREPROCESS_PARAMS` and :data:`CONCORDANCE_PARAMS`;
    ``extra``: further keyword arguments for ``COCOA.activity_concordance_analysis``.
    ``workers``: Julia worker processes for the LPs (0 = serial; use cores - 1).

    Files in ``outdir``: ``kinetic_modules.csv``, ``complexes.csv``, ``acr.csv``,
    ``acrr.csv``, ``model_preprocessed.xml``, ``concordance.jld2`` (for
    :func:`kinetic_from_run`), ``run.json`` (parameters, preprocessing report, versions,
    counts, timings) and ``job.json`` (the job; rerun by hand as printed on failure).
    """
    if not preprocess_model and set(params) & set(PREPROCESS_PARAMS):
        raise TypeError("preprocessing parameters given with preprocess_model=False: "
                        + ", ".join(sorted(set(params) & set(PREPROCESS_PARAMS))))
    outdir = Path(outdir)
    job = _job("run", model=model, mat_key=mat_key, preprocess_model=preprocess_model,
               params=resolve_params(params), extra=extra, workers=workers,
               exhaustive=exhaustive, write_acr=write_acr, outdir=_abs(outdir))
    _runner.run_job(job, outdir / "job.json", quiet=quiet)
    return load_result(outdir)


def preprocess(model: str | os.PathLike, output: str | os.PathLike, *, workers: int = 0,
               mat_key: str | None = None, quiet: bool = False, **params) -> dict:
    """Preprocess a model and write it as SBML; returns the preprocessing report.

    1. Remove orphan metabolites; normalise bounds to the flux cone (negative lower bounds →
       -1000, positive upper bounds → 1000, all others 0; the objective forward).
    2. Remove blocked reactions: flux range below ``flux_tol`` while growth is held within
       ``growth_window`` × optimum. Stops if an LP cannot be solved, a blocked reaction is
       left afterwards, or the growth optimum changed (then ``flux_tol`` is too coarse for
       the model; biomass built from nested pools can need 1e-9).
    3. Remove orphans again.
    4. Split into elementary steps (``binding="ordered"``/``"random"``; ``"none"`` keeps
       the reactions) and reversible reactions into two irreversible ones.

    The report (``<output stem>.report.json``) has the model size after each step and the
    growth optimum. Only :data:`PREPROCESS_PARAMS` (and ``solver_tol``) apply.
    """
    bad = set(params) - set(PREPROCESS_PARAMS) - {"solver_tol", "lp_time_limit"}
    if bad:
        raise TypeError(f"not a preprocessing parameter: {', '.join(sorted(bad))}")
    output = Path(output)
    job = _job("preprocess", model=model, mat_key=mat_key, params=resolve_params(params),
               workers=workers, output=_abs(output))
    _runner.run_job(job, output.with_suffix(".job.json"), quiet=quiet)
    return json.loads(output.with_suffix(".report.json").read_text())


def kinetic_from_run(rundir: str | os.PathLike, outdir: str | os.PathLike | None = None, *,
                     exhaustive: bool = False, write_acr: bool = True,
                     quiet: bool = False) -> Result:
    """Recompute the kinetic modules of an earlier :func:`run` without redoing concordance.

    Reads ``concordance.jld2`` (result and the exact model) from ``rundir``. With
    ``outdir`` the outputs go there; without, the kinetic outputs in ``rundir`` are
    overwritten (``run.json`` is kept; the new summary goes to ``kinetic.json``).
    """
    rundir = Path(rundir)
    out = Path(outdir) if outdir else rundir
    job = _job("kinetic", exhaustive=exhaustive, write_acr=write_acr, rundir=_abs(rundir),
               outdir=_abs(out))
    _runner.run_job(job, out / "kinetic_job.json", quiet=quiet)
    return load_result(out) if outdir else Result(
        out, Table.read_csv(out / "kinetic_modules.csv"), Table.read_csv(out / "complexes.csv"),
        Table.read_csv(out / "acr.csv") if write_acr else None,
        Table.read_csv(out / "acrr.csv") if write_acr else None,
        json.loads((out / "kinetic.json").read_text()))


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)):
        return "-".join(_fmt(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def run_label(params: Mapping[str, Any]) -> str:
    """Directory name of a sweep combination, e.g. ``cv_threshold=0.03__binding=random``."""
    return "__".join(f"{k}={_fmt(v)}" for k, v in params.items()) or "default"


def sweep(model: str | os.PathLike, outdir: str | os.PathLike,
          grid: Mapping[str, Iterable], *, workers: int = 0, preprocess_model: bool = True,
          mat_key: str | None = None, exhaustive: bool = False, write_acr: bool = False,
          extra: Mapping[str, Any] | None = None, quiet: bool = False, **fixed) -> Table:
    """Kinetic modules for every combination of the parameter values in ``grid``.

    ``grid`` maps parameter names (any of :data:`PREPROCESS_PARAMS` and
    :data:`CONCORDANCE_PARAMS`) to lists of values; ``fixed`` sets others for all
    combinations. Each combination is a run in its own subdirectory of ``outdir``. All runs
    happen in one Julia process: the model is loaded once, preprocessed once per distinct
    preprocessing setting, and the workers are reused. A combination whose ``run.json``
    exists is skipped, so an interrupted sweep continues where it stopped.

    Writes and returns ``summary.csv``: per combination its parameters, number of complexes,
    concordance and kinetic modules, size of the largest kinetic module, undecided pairs and
    runtime.
    """
    outdir = Path(outdir)
    names = list(grid)
    resolve_params({**fixed, **{n: next(iter(grid[n])) for n in names}})  # validate names
    runs = []
    for values in itertools.product(*(list(grid[n]) for n in names)):
        combo = dict(zip(names, values))
        p = resolve_params({**fixed, **combo})
        pre_key = run_label({k: p[k] for k in PREPROCESS_PARAMS}) if preprocess_model else "as-given"
        runs.append(dict(dir=_abs(outdir / run_label(combo)), params=p, preprocess_key=pre_key,
                         combo=combo))
    job = _job("sweep", model=model, mat_key=mat_key, preprocess_model=preprocess_model,
               params=None, extra=extra, workers=workers, exhaustive=exhaustive,
               write_acr=write_acr, outdir=_abs(outdir),
               runs=[{k: v for k, v in r.items() if k != "combo"} for r in runs])
    _runner.run_job(job, outdir / "sweep_job.json", quiet=quiet)

    rows = []
    for r in runs:
        info = json.loads((Path(r["dir"]) / "run.json").read_text())
        s = info.get("stats", {})
        rows.append({**{n: _fmt(v) if isinstance(v, (list, tuple)) else v
                        for n, v in r["combo"].items()},
                     "run": Path(r["dir"]).name,
                     "n_complexes": s.get("n_complexes"),
                     "n_concordance_modules": s.get("n_concordance_modules"),
                     "n_kinetic_modules": s.get("n_kinetic_modules"),
                     "largest_kinetic_module": s.get("largest_kinetic_module"),
                     "n_undecided_pairs": s.get("n_unknown_pairs"),
                     "seconds": info.get("seconds_total")})
    summary = Table({k: [row[k] for row in rows] for k in rows[0]}) if rows else Table()
    summary.to_csv(outdir / "summary.csv")
    return summary


def versions(quiet: bool = True) -> dict:
    """Versions of Julia and COCOA.jl (path, git commit) in the configured project, and the
    path of the bundled example model. Starts Julia briefly."""
    from . import __version__  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "info.json"
        _runner.run_job(_job("info", output=str(out)), Path(d) / "job.json", quiet=quiet)
        v = json.loads(out.read_text())
    v.pop("workers", None)
    v["cocoajl"] = __version__
    return v


def example_model() -> str:
    """Path of the small ``e_coli_core`` model shipped with COCOA.jl."""
    return versions()["example_model"]
