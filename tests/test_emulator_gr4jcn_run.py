# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""End-to-end GR4JCN integration test against the real RavenPy + raven binary.

This test exercises the full emulator path without a SYMFLUENCE domain: synthesize
~2 years of daily forcing and a single lumped HRU, drive the same NetCDF/Gauge/config
build that :func:`raven_symfluence.emulator.build_and_run_emulator` uses, RUN raven, and
assert a non-empty, finite simulated streamflow series comes back.

Skipped automatically when RavenPy or the raven binary is unavailable.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

# Skip the whole module unless RavenPy *and* the raven binary are importable/usable.
ravenpy = pytest.importorskip("ravenpy", reason="RavenPy not installed")

try:
    from ravenpy._raven import RAVEN_EXEC_PATH
except Exception:  # noqa: BLE001 -- binary missing => skip
    RAVEN_EXEC_PATH = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RAVEN_EXEC_PATH or not os.path.exists(str(RAVEN_EXEC_PATH)),
        reason="raven binary not available",
    ),
]


def _synthesize_forcing(n_days: int = 730, start: str = "2000-01-01") -> pd.DataFrame:
    """~2 years of plausible daily precip (mm/d) and temperature (degC)."""
    rng = np.random.default_rng(42)
    time = pd.date_range(start, periods=n_days, freq="D")
    precip = np.abs(rng.gamma(1.0, 3.0, n_days))  # mm/d, non-negative
    seasonal = 10.0 + 10.0 * np.sin(np.arange(n_days) / 365.0 * 2 * np.pi)
    temp = seasonal + rng.normal(0.0, 2.0, n_days)  # degC
    return pd.DataFrame({"PRECIP": precip, "TEMP_AVE": temp}, index=time)


def test_gr4jcn_emulator_runs_and_returns_streamflow(tmp_path, caplog):
    """Build the GR4JCN config from forcing + HRU + params, run raven, read q_sim."""
    from raven_symfluence import emulator as em

    # Ensure RavenPy can find the engine regardless of PATH.
    os.environ.setdefault("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))

    import logging

    logger = logging.getLogger("raven_test")

    forcing_df = _synthesize_forcing()
    hru = {
        "area_km2": 100.0,
        "elevation": 1500.0,
        "latitude": 51.0,
        "longitude": -115.0,
    }
    # GR4JCN parameter vector in the verified RavenPy order
    # (GR4J_X1..X4, CEMANEIGE_X1, CEMANEIGE_X2).
    params = {
        "GR4J_X1": 0.529,
        "GR4J_X2": -3.396,
        "GR4J_X3": 407.29,
        "GR4J_X4": 1.072,
        "CEMANEIGE_X1": 16.9,
        "CEMANEIGE_X2": 0.947,
    }
    param_vector = em.params_to_vector("GR4JCN", params, logger)
    assert param_vector == [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]

    settings_dir = tmp_path / "settings"
    output_dir = tmp_path / "output"
    settings_dir.mkdir()
    output_dir.mkdir()

    # --- Write the CF-compliant forcing NetCDF the Gauge reads. ---
    forcing_nc = settings_dir / "test_raven_forcing.nc"
    em._write_forcing_netcdf(forcing_df, hru, forcing_nc)
    assert forcing_nc.exists()

    # --- Build the RavenPy objects exactly as build_and_run_emulator does. ---
    from ravenpy import Emulator
    from ravenpy.config import commands as rc
    from ravenpy.config import emulators

    gauge = em._build_gauge(rc, forcing_nc, hru, logger)

    config_model = emulators.GR4JCN(
        params=param_vector,
        HRUs=[{
            "area": hru["area_km2"],
            "elevation": hru["elevation"],
            "latitude": hru["latitude"],
            "longitude": hru["longitude"],
            "hru_type": "land",
        }],
        Gauge=[gauge],
        StartDate=dt.datetime(2000, 1, 1),
        EndDate=dt.datetime(2001, 12, 30),
        RunName="test_run",
        WriteNetcdfFormat=False,
    )

    # --- Run raven for real. ---
    out = Emulator(config=config_model, workdir=str(output_dir), overwrite=True).run()

    # With WriteNetcdfFormat=False Raven writes the CSV hydrograph (not NetCDF), which is
    # exactly what the SYMFLUENCE postprocessor consumes — read it back through the plugin.
    from raven_symfluence.postprocessor import find_hydrographs_file, read_raven_streamflow

    hydro_csv = find_hydrographs_file(out.path)
    assert hydro_csv is not None, f"no Hydrographs.csv written in {out.path}"

    streamflow = read_raven_streamflow(hydro_csv, config={}, logger=logger)
    assert streamflow is not None and not streamflow.empty
    q = streamflow.to_numpy()
    assert q.size > 700, f"expected ~730 daily steps, got {q.size}"
    assert np.isfinite(q).all(), "simulated streamflow contains non-finite values"
    assert (q >= 0).all(), "simulated streamflow should be non-negative"
    assert q.sum() > 0, "simulated streamflow is identically zero"
