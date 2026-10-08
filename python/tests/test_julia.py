"""End-to-end tests on e_coli_core. Need the Julia side (``cocoa-jl setup``); run with
``pytest -m julia``. Several minutes, mostly Julia start-up and compilation.
"""
import json
import subprocess

import pytest

import cocoajl as cj
from cocoajl import _runner

pytestmark = pytest.mark.julia
SMALL = dict(flux_tol=1e-9, sample_size=1000)


@pytest.fixture(scope="module")
def ecoli():
    return cj.example_model()


@pytest.fixture(scope="module")
def serial(ecoli, tmp_path_factory):
    return cj.run(ecoli, tmp_path_factory.mktemp("serial"), **SMALL)


def test_versions():
    v = cj.versions()
    assert v["COCOA"] and v["julia"].startswith("1.") and v["example_model"].endswith(".xml")


def test_run_outputs(serial):
    d = serial.outdir
    for f in ("complexes.csv", "kinetic_modules.csv", "acr.csv", "acrr.csv", "run.json",
              "job.json", "model_preprocessed.xml", "concordance.jld2"):
        assert (d / f).is_file(), f
    s = serial.stats
    assert s["n_unknown_pairs"] == 0
    assert len(serial.complexes) == s["n_complexes"]
    assert len(serial.modules) == s["n_kinetic_modules"] > 0
    assert serial.modules["n_complexes"] == sorted(serial.modules["n_complexes"], reverse=True)
    assert sum(serial.modules["n_complexes"]) == \
        sum(k > 0 for k in serial.complexes["kinetic_module"])
    assert serial.info["parameters"]["flux_tol"] == 1e-9
    assert serial.info["preprocessing"]["growth_optimum"] > 0


def test_same_as_plain_cocoa(serial, tmp_path):
    """The partition equals plain COCOA.jl on the same preprocessed model and settings."""
    script = tmp_path / "ref.jl"
    script.write_text(f'''
import COCOA, HiGHS, COBREXA, SBMLFBCModels, AbstractFBCModels as A
m = convert(A.CanonicalModel.Model, A.load(SBMLFBCModels.SBMLFBCModel,
            "{serial.outdir / 'model_preprocessed.xml'}"))
s = [COBREXA.set_optimizer_attribute("primal_feasibility_tolerance", 1e-8),
     COBREXA.set_optimizer_attribute("dual_feasibility_tolerance", 1e-8),
     COBREXA.set_optimizer_attribute("random_seed", 42),
     COBREXA.set_optimizer_attribute("time_limit", 3600.0),
     COBREXA.set_optimizer_attribute("presolve", "off")]
r = COCOA.activity_concordance_analysis(m; optimizer=HiGHS.Optimizer, settings=s,
        sample_size=1000, seed=UInt(42), objective_bound=nothing)
open("{tmp_path / 'ref.tsv'}", "w") do io
    for (c, k) in zip(r.complexes.complex_id, r.complexes.concordance_module)
        println(io, c, "\\t", k)
    end
end
''')
    s = _runner.settings()
    subprocess.run([s["julia"] or "julia", f"--project={s['project']}", "--startup-file=no",
                    str(script)], check=True)
    ref = dict(line.split("\t") for line in (tmp_path / "ref.tsv").read_text().splitlines())
    got = dict(zip(serial.complexes["complex_id"], serial.complexes["concordance_module"]))
    assert {k: int(v) for k, v in ref.items()} == got


def test_kinetic_from_run(serial, tmp_path):
    k = cj.kinetic_from_run(serial.outdir, tmp_path / "again")
    assert k.modules.columns == serial.modules.columns


def test_preprocess(ecoli, tmp_path):
    rep = cj.preprocess(ecoli, tmp_path / "pre.xml", flux_tol=1e-9, binding="none")
    assert (tmp_path / "pre.xml").is_file()
    assert rep["binding"] == "none" and rep["final"]["reactions"] > 0


def test_sweep_and_restart(ecoli, tmp_path):
    grid = {"concordance_tolerance": [0.01, 0.001]}
    s = cj.sweep(ecoli, tmp_path / "sw", grid, **SMALL)
    assert len(s) == 2 and (tmp_path / "sw" / "summary.csv").is_file()
    stamp = (tmp_path / "sw" / s["run"][0] / "run.json").stat().st_mtime
    s2 = cj.sweep(ecoli, tmp_path / "sw", grid, **SMALL)   # finished runs are skipped
    assert s2.columns == s.columns
    assert (tmp_path / "sw" / s["run"][0] / "run.json").stat().st_mtime == stamp


def test_workers_same_result(serial, ecoli, tmp_path):
    par = cj.run(ecoli, tmp_path / "par", workers=2, **SMALL)
    assert par.modules.columns == serial.modules.columns


def test_failure_reports_rerun_command(tmp_path):
    with pytest.raises(cj.CocoaError, match="Rerun by hand"):
        cj.run(tmp_path / "missing.xml", tmp_path / "out", quiet=True)


def test_cli_run(ecoli, tmp_path):
    from cocoajl.cli import main
    assert main(["run", ecoli, str(tmp_path / "cli"), "--flux-tol", "1e-9",
                 "--sample-size", "1000", "--no-acr"]) == 0
    assert not (tmp_path / "cli" / "acr.csv").exists()
    info = json.loads((tmp_path / "cli" / "run.json").read_text())
    assert info["parameters"]["sample_size"] == 1000
