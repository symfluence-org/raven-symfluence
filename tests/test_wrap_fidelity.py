# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wrap-fidelity check: SYMFLUENCE-driven Raven == standalone RavenPy.

The whole point of the plugin is to *wrap* Raven without distorting it. This test
proves that by running ONE identical set of inputs through two paths and asserting the
simulated streamflow series are numerically identical:

Path 1 (plugin): drive ``raven_symfluence.emulator.build_and_run_emulator`` — the exact
    function ``RavenRunner`` calls — against a self-contained model-ready store built in
    a tmp dir. This exercises the plugin's CFIF->Raven forcing conversion, HRU resolution
    from the attributes store, parameter ordering, NetCDF/Gauge encoding, and the run.

Path 2 (raw RavenPy): build the SAME ``ravenpy.config.emulators.GR4JCN`` config directly
    from the forcing NetCDF the plugin wrote, the same param vector, and the same lumped
    HRU spec, then ``Emulator().run()``.

Because both paths feed Raven byte-identical forcing, an identical HRU, and the identical
parameter vector, Raven must return the identical hydrograph. ``np.allclose`` at a tight
tolerance proves the wrap adds no numerical distortion; any mismatch is a plugin bug
(forcing encoding, parameter mapping, or HRU spec) to be fixed here.

Skipped automatically when RavenPy or the raven binary is unavailable.
"""
from __future__ import annotations

import datetime as dt
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

DOMAIN = "WRAPFIDELITY"
N_DAYS = 2 * 365
START = "2000-01-01"
LAT, LON, ELEV, AREA_M2 = 51.0, -115.0, 1500.0, 100.0e6

# One shared GR4JCN parameter vector (verified RavenPy 0.21 positional order:
# GR4J_X1..X4, CEMANEIGE_X1, CEMANEIGE_X2).
PARAMS = {
    "GR4J_X1": 0.529,
    "GR4J_X2": -3.396,
    "GR4J_X3": 407.29,
    "GR4J_X4": 1.072,
    "CEMANEIGE_X1": 16.9,
    "CEMANEIGE_X2": 0.947,
}


def _build_forcing(forcings_dir: Path) -> pd.DatetimeIndex:
    """Canonical CFIF daily forcing (precipitation_flux kg m-2 s-1, air_temperature K)."""
    import xarray as xr

    forcings_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
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
    """Grouped attributes NetCDF the plugin's HRU resolver reads (hru_identity + terrain)."""
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


def test_wrap_fidelity_plugin_matches_raw_ravenpy(tmp_path, monkeypatch):
    """SYMFLUENCE-driven Raven streamflow == standalone RavenPy streamflow (allclose)."""
    monkeypatch.setenv("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))
    logger = logging.getLogger("wrap_fidelity")

    data_dir = tmp_path / "data_root"
    project_dir = data_dir / f"domain_{DOMAIN}"
    mr = project_dir / "data" / "model_ready"
    _build_forcing(mr / "forcings")
    _build_attributes(mr / "attributes")

    config = {
        "SYMFLUENCE_DATA_DIR": str(data_dir),
        "DOMAIN_NAME": DOMAIN,
        "RAVEN_MODEL_TEMPLATE": "GR4JCN",
        "RAVEN_RUN_NAME": "wrap_fidelity",
        "EXPERIMENT_TIME_START": "2000-01-01 00:00",
        "EXPERIMENT_TIME_END": "2001-12-30 00:00",
        "RAVEN_EXE": str(RAVEN_EXEC_PATH),
        "RAVEN_INSTALL_PATH": str(Path(str(RAVEN_EXEC_PATH)).parent),
    }

    # =====================================================================
    # Path 1 -- the plugin: build_and_run_emulator (exactly what the runner calls).
    # =====================================================================
    from raven_symfluence import emulator as em

    settings_dir = project_dir / "settings" / "RAVEN"
    plugin_out = project_dir / "simulations" / "wrap" / "RAVEN"
    ok = em.build_and_run_emulator(
        config=config,
        settings_dir=settings_dir,
        output_dir=plugin_out,
        params=PARAMS,
        logger=logger,
    )
    assert ok, "plugin build_and_run_emulator failed"

    from raven_symfluence.postprocessor import find_hydrographs_file, read_raven_streamflow

    plugin_csv = find_hydrographs_file(plugin_out)
    assert plugin_csv is not None, f"plugin path produced no Hydrographs.csv under {plugin_out}"
    q_plugin = read_raven_streamflow(plugin_csv, config={}, logger=logger)
    assert q_plugin is not None and not q_plugin.empty

    # The plugin wrote the forcing NetCDF it fed Raven; Path 2 reuses that exact file so
    # the only thing under test is the wrap, not forcing-synthesis differences.
    forcing_nc = settings_dir / f"{DOMAIN}_raven_forcing.nc"
    assert forcing_nc.exists(), "plugin did not persist its Raven forcing NetCDF"

    # =====================================================================
    # Path 2 -- raw RavenPy: same forcing NetCDF, same params, same HRU spec.
    # =====================================================================
    from ravenpy import Emulator
    from ravenpy.config import commands as rc
    from ravenpy.config import emulators

    param_vector = em.params_to_vector("GR4JCN", PARAMS, logger)
    hru = {"area_km2": AREA_M2 / 1e6, "elevation": ELEV, "latitude": LAT, "longitude": LON}
    gauge = rc.Gauge.from_nc(
        str(forcing_nc),
        data_type=["PRECIP", "TEMP_AVE"],
        station_idx=1,
        data_kwds={"ALL": {"elevation": float(hru["elevation"])}},
    )
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
        RunName="wrap_fidelity",
        WriteNetcdfFormat=False,
    )
    raw_out = tmp_path / "raw_ravenpy"
    out = Emulator(config=config_model, workdir=str(raw_out), overwrite=True).run(overwrite=True)
    raw_csv = find_hydrographs_file(out.path)
    assert raw_csv is not None, "raw RavenPy path produced no Hydrographs.csv"
    q_raw = read_raven_streamflow(raw_csv, config={}, logger=logger)
    assert q_raw is not None and not q_raw.empty

    # =====================================================================
    # Fidelity assertion: identical series.
    # =====================================================================
    # Align on the common date index, then compare element-wise.
    aligned = pd.concat([q_plugin.rename("plugin"), q_raw.rename("raw")], axis=1, join="inner")
    assert len(aligned) > 700, f"too little overlap to compare: {len(aligned)} steps"
    assert aligned["plugin"].notna().all() and aligned["raw"].notna().all()

    a = aligned["plugin"].to_numpy(dtype=float)
    b = aligned["raw"].to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(a - b)))
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9), (
        f"plugin vs raw RavenPy streamflow differ (max abs diff {max_abs:.3e}); "
        "the wrap is distorting Raven inputs -- check forcing encoding / params / HRU spec"
    )
