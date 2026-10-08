"""Fast tests that need no Julia: argument parsing, parameters, tables, config, job files."""
import json

import pytest

import cocoajl as cj
from cocoajl import _runner
from cocoajl.api import Table, run_label
from cocoajl.cli import build_parser, parse_grid, parse_value


def test_parse_value_types():
    assert parse_value("cv_threshold", "0.05") == 0.05
    assert parse_value("sample_size", "1e4") == 10000
    assert parse_value("use_transitivity", "off") is False
    assert parse_value("growth_fraction", "none") is None
    assert parse_value("growth_fraction", "0.5") == 0.5
    assert parse_value("growth_window", "0.001:1") == (0.001, 1.0)
    assert parse_value("binding", "random") == "random"
    with pytest.raises(ValueError):
        parse_value("use_transitivity", "maybe")


def test_parse_grid():
    g = parse_grid(["concordance-tolerance=0.01,0.001", "binding=ordered,random"])
    assert g == {"concordance_tolerance": [0.01, 0.001], "binding": ["ordered", "random"]}
    with pytest.raises(SystemExit):
        parse_grid(["no_such_parameter=1"])
    with pytest.raises(SystemExit):
        parse_grid(["cv_threshold"])


def test_run_flags_only_explicit():
    ns = build_parser().parse_args(["run", "m.xml", "out", "--cv-threshold", "0.05",
                                    "--binding", "random", "--no-transitivity"])
    given = {k: v for k, v in vars(ns).items() if k in cj.api.ALL_PARAMS}
    # parameters not on the command line are absent, so the API defaults apply
    assert given == {"cv_threshold": 0.05, "binding": "random", "use_transitivity": False}


def test_resolve_params():
    p = cj.resolve_params({"binding": "random", "growth_window": (0.01, 1)})
    assert p["binding"] == "random" and p["growth_window"] == [0.01, 1.0]
    assert p["concordance_tolerance"] == cj.CONCORDANCE_PARAMS["concordance_tolerance"]
    with pytest.raises(TypeError):
        cj.resolve_params({"cc_scale_bound": 5.0})
    with pytest.raises(ValueError):
        cj.resolve_params({"binding": "both"})


def test_run_label():
    assert run_label({"concordance_tolerance": 0.001, "binding": "ordered"}) == \
        "concordance_tolerance=0.001__binding=ordered"
    assert run_label({"growth_window": (0.001, 1.0)}) == "growth_window=0.001-1"
    assert run_label({}) == "default"


def test_table_csv_roundtrip(tmp_path):
    t = Table({"complex_id": ["a+b", "x,y"], "kinetic_module": [2, 0],
               "lambda": [0.5, None], "trivially_balanced": [True, False]})
    t.to_csv(tmp_path / "t.csv")
    back = Table.read_csv(tmp_path / "t.csv")
    assert back.columns == t.columns
    assert list(back.rows())[0] == {"complex_id": "a+b", "kinetic_module": 2, "lambda": 0.5,
                                    "trivially_balanced": True}


def test_config_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("COCOAJL_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("COCOAJL_PROJECT", raising=False)
    monkeypatch.delenv("COCOAJL_JULIA", raising=False)
    monkeypatch.setattr(_runner, "_override", {})
    assert _runner.settings()["project"] is None
    _runner.write_config({"project": "/from/config", "julia": "/bin/julia"})
    assert _runner.settings()["project"] == "/from/config"
    monkeypatch.setenv("COCOAJL_PROJECT", "/from/env")
    assert _runner.settings()["project"] == "/from/env"
    cj.configure(project=tmp_path)
    assert _runner.settings()["project"] == str(tmp_path.resolve())
    assert json.loads((tmp_path / "config.json").read_text())["julia"] == "/bin/julia"


def test_not_set_up(tmp_path, monkeypatch):
    monkeypatch.setenv("COCOAJL_CONFIG", str(tmp_path / "none.json"))
    monkeypatch.delenv("COCOAJL_PROJECT", raising=False)
    monkeypatch.setattr(_runner, "_override", {})
    with pytest.raises(cj.CocoaError, match="cocoa-jl setup"):
        cj.run("model.xml", tmp_path / "out")


def test_job_file(tmp_path, monkeypatch):
    """The job written for a run carries every resolved parameter."""
    seen = {}
    monkeypatch.setattr(_runner, "run_job", lambda job, f, quiet=False: seen.update(job))
    monkeypatch.setattr(cj.api, "load_result", lambda d: None)
    cj.run(tmp_path / "m.xml", tmp_path / "out", workers=3, cv_threshold=0.05,
           extra={"cc_scale_bound": 1e6})
    assert seen["command"] == "run" and seen["workers"] == 3
    assert seen["params"]["cv_threshold"] == 0.05
    assert set(seen["params"]) == set(cj.api.ALL_PARAMS)
    assert seen["extra"] == {"cc_scale_bound": 1e6}
