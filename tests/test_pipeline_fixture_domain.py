# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Full-pipeline integration test on a minimal lumped GR4JCN fixture domain.

Builds the SYMFLUENCE model-ready store the plugin reads (canonical CFIF forcing,
grouped attributes, preprocessed streamflow observations) entirely in code under a
tmp data dir, then drives the four pipeline steps via the public SYMFLUENCE API:

    model_specific_preprocessing -> run_model -> postprocess_results -> calibrate_model

Asserts the plugin's real-domain wiring: the ``.rv*`` config is written into
``settings/RAVEN``, ``run_model`` produces ``Hydrographs.csv``, postprocess writes a
standardized ``results/*.csv``, and a short DDS loop completes with finite KGE scores
that vary across iterations (i.e. observations are actually loaded and the objective
responds to parameters).

Skipped automatically when RavenPy or the raven binary is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ravenpy = pytest.importorskip("ravenpy", reason="RavenPy not installed")

try:
    from ravenpy._raven import RAVEN_EXEC_PATH
except Exception:  # noqa: BLE001 -- binary missing => skip
    RAVEN_EXEC_PATH = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RAVEN_EXEC_PATH or not Path(str(RAVEN_EXEC_PATH)).exists(),
        reason="raven binary not available",
    ),
]

DOMAIN = "FIXTURE"
N_DAYS = 3 * 365
START = "2000-01-01"
LAT, LON, ELEV, AREA_M2 = 51.0, -115.0, 1500.0, 100.0e6


def _build_forcing(forcings_dir: Path) -> pd.DatetimeIndex:
    """Canonical CFIF daily forcing (precipitation_flux kg m-2 s-1, air_temperature K)."""
    import xarray as xr

    forcings_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    time = pd.date_range(START, periods=N_DAYS, freq="D")
    precip_flux = (np.abs(rng.gamma(1.0, 4.0, N_DAYS)) / 86400.0).astype("f4")  # mm/d -> kg/m2/s
    seasonal = 8.0 + 12.0 * np.sin(np.arange(N_DAYS) / 365.0 * 2 * np.pi)
    temp_k = (seasonal + rng.normal(0.0, 2.0, N_DAYS) + 273.15).astype("f4")

    ds = xr.Dataset(
        {
            "precipitation_flux": (("time", "hru"), precip_flux[:, None],
                                   {"units": "kg m-2 s-1", "standard_name": "precipitation_flux"}),
            "air_temperature": (("time", "hru"), temp_k[:, None],
                                {"units": "K", "standard_name": "air_temperature"}),
        },
        coords={"time": time, "hru": [0]},
    )
    ds.attrs.update({"Conventions": "CF-1.8", "timestep_seconds": 86400.0,
                     "forcing_vocabulary": "SYMFLUENCE-canonical"})
    enc = {"time": {"units": "days since 1970-01-01", "calendar": "gregorian", "dtype": "float64"}}
    ds.to_netcdf(forcings_dir / f"{DOMAIN}_forcing.nc", engine="netcdf4", encoding=enc)
    ds.close()
    return time


def _build_attributes(attrs_dir: Path) -> None:
    """Grouped attributes NetCDF: hru_identity + terrain (the groups the plugin reads)."""
    import netCDF4

    attrs_dir.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(attrs_dir / f"{DOMAIN}_attributes.nc"), "w", format="NETCDF4") as root:
        root.Conventions = "CF-1.8"
        root.domain_name = DOMAIN
        hi = root.createGroup("hru_identity")
        hi.createDimension("hru", 1)
        hi.createVariable("hru_id", str, ("hru",))[0] = "1"
        hi.createVariable("hru_area", "f8", ("hru",))[:] = [AREA_M2]
        hi.createVariable("latitude", "f8", ("hru",))[:] = [LAT]
        hi.createVariable("longitude", "f8", ("hru",))[:] = [LON]
        te = root.createGroup("terrain")
        te.createDimension("hru", 1)
        te.createVariable("hru_id", str, ("hru",))[0] = "1"
        te.createVariable("hru_area", "f8", ("hru",))[:] = [AREA_M2]
        te.createVariable("elev_mean", "f8", ("hru",))[:] = [ELEV]


def _build_observations(obs_preproc_dir: Path, time: pd.DatetimeIndex) -> None:
    """Synthetic daily streamflow CSV (the worker/evaluator observation source)."""
    rng = np.random.default_rng(11)
    base = 5.0 + 3.0 * np.sin((np.arange(len(time)) / 365.0 + 0.25) * 2 * np.pi)
    q = np.clip(base + rng.normal(0, 0.5, len(time)), 0.05, None)
    obs_preproc_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(q, index=time, name="discharge_cms").to_frame().to_csv(
        obs_preproc_dir / f"{DOMAIN}_streamflow_processed.csv")


def _write_config(cfg_path: Path, data_dir: Path, code_dir: Path, raven_exe: str) -> None:
    cfg_path.write_text(
        f'SYMFLUENCE_DATA_DIR: "{data_dir}"\n'
        f'SYMFLUENCE_CODE_DIR: "{code_dir}"\n'
        f'DOMAIN_NAME: "{DOMAIN}"\n'
        'EXPERIMENT_ID: "fixture_run"\n'
        'HYDROLOGICAL_MODEL: "RAVEN"\n'
        'RAVEN_MODEL_TEMPLATE: "GR4JCN"\n'
        'RAVEN_SPATIAL_MODE: "lumped"\n'
        f'RAVEN_EXE: "{raven_exe}"\n'
        f'RAVEN_INSTALL_PATH: "{Path(raven_exe).parent}"\n'
        'DOMAIN_DEFINITION_METHOD: "lumped"\n'
        'SUB_GRID_DISCRETIZATION: "elevation"\n'
        'FORCING_DATASET: "ERA5"\n'
        'EXPERIMENT_TIME_START: "2000-01-01 00:00"\n'
        'EXPERIMENT_TIME_END: "2002-12-30 00:00"\n'
        'CALIBRATION_PERIOD: "2000-06-01, 2001-12-31"\n'
        'EVALUATION_PERIOD: "2002-01-01, 2002-12-30"\n'
        'OPTIMIZATION_METHODS: ["iteration"]\n'
        'ITERATIVE_OPTIMIZATION_ALGORITHM: "DDS"\n'
        'OPTIMIZATION_METRIC: "KGE"\n'
        'NUMBER_OF_ITERATIONS: 8\n'
        'NUM_PROCESSES: 1\n'
        'RAVEN_PARAMS_TO_CALIBRATE: "default"\n'
        'FORCE_RUN_ALL_STEPS: true\n'
    )


def test_full_pipeline_on_fixture_domain(tmp_path, monkeypatch):
    """End-to-end: build a fixture store, run the 4 steps, assert outputs + calibration."""
    monkeypatch.setenv("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))
    logging.getLogger("raven_symfluence").setLevel(logging.INFO)

    code_dir = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data_root"
    project_dir = data_dir / f"domain_{DOMAIN}"
    mr = project_dir / "data" / "model_ready"

    time = _build_forcing(mr / "forcings")
    _build_attributes(mr / "attributes")
    _build_observations(project_dir / "data" / "observations" / "streamflow" / "preprocessed", time)

    cfg_path = data_dir / "config_fixture.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_config(cfg_path, data_dir, code_dir, str(RAVEN_EXEC_PATH))

    from symfluence import SYMFLUENCE

    sym = SYMFLUENCE(cfg_path)
    sym.run_individual_steps(
        ["model_specific_preprocessing", "run_model", "postprocess_results", "calibrate_model"])

    # --- preprocessing wrote the .rv* config into settings/RAVEN ---
    settings = project_dir / "settings" / "RAVEN"
    rv_files = {p.suffix for p in settings.glob("*.rv*")}
    assert {".rvi", ".rvp", ".rvh", ".rvt", ".rvc"}.issubset(rv_files), \
        f"missing .rv* config in {settings}: {sorted(rv_files)}"

    # --- run_model produced Hydrographs.csv ---
    from raven_symfluence.postprocessor import find_hydrographs_file
    sim = project_dir / "simulations" / "fixture_run" / "RAVEN"
    assert find_hydrographs_file(sim) is not None, f"no Hydrographs.csv under {sim}"

    # --- postprocess wrote a standardized results CSV ---
    results = list((project_dir / "results").rglob("*.csv"))
    assert results, "postprocess did not write any results/*.csv"

    # --- calibration completed with finite, varying KGE scores ---
    iter_csv = next((project_dir / "optimization").rglob("*iteration_results.csv"), None)
    assert iter_csv is not None, "no calibration iteration_results.csv written"
    df = pd.read_csv(iter_csv)
    assert (df["crash_count"] == 0).all(), "every DDS trial crashed (model never ran)"
    scores = df["score"].to_numpy()
    assert np.isfinite(scores).all(), f"non-finite calibration scores: {scores}"
    assert (scores > -9000).all(), f"penalty scores present (obs not loaded?): {scores}"
    assert df["score"].nunique() > 1, f"calibration score never varied: {scores}"
