"""cocoajl: concordance and kinetic modules of metabolic networks, from Python.

A Python front end to `COCOA.jl <https://github.com/antoniofranky/COCOA.jl>`_. Every analysis
runs as a separate Julia process (set up once with ``cocoa-jl setup``); results come back as
files and tables.

Typical use::

    import cocoajl as cj
    res = cj.run("model.xml", "out/", workers=31)   # files in out/
    res.modules.to_pandas()                          # one row per kinetic module

Parameter sweeps::

    cj.sweep("model.xml", "sweep/", {"cv_threshold": [0.01, 0.03]}, workers=31)

See the README for installation and the command line interface (``cocoa-jl --help``).
"""
from ._runner import CocoaError, configure
from .api import (CONCORDANCE_PARAMS, PREPROCESS_PARAMS, Result, Table, example_model,
                  kinetic_from_run, load_result, preprocess, resolve_params, run, sweep,
                  versions)
from .setup_env import setup

__version__ = "0.1.0"

__all__ = [
    "run", "sweep", "preprocess", "kinetic_from_run", "load_result",
    "example_model", "versions", "setup", "configure",
    "Result", "Table", "CocoaError", "resolve_params",
    "PREPROCESS_PARAMS", "CONCORDANCE_PARAMS", "__version__",
]
