# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""End-to-end run test across all four wired Raven emulator templates.

Parametrizes over ``GR4JCN``, ``HBVEC``, ``HMETS``, ``MOHYSE``: for each template it
builds a self-contained model-ready store (canonical CFIF forcing + grouped attributes),
drives the *plugin's* :func:`raven_symfluence.emulator.build_and_run_emulator` — the exact
function ``RavenRunner``/``RavenWorker`` call — against it, RUNS the real raven binary, and
asserts a non-empty, finite, non-negative, non-zero simulated streamflow series comes back.

This proves each template is wired through the template-aware emulator path (right RavenPy
class, parameter ordering, and the minimal per-template extras: RAINSNOW_DINGMAN override
for HMETS/MOHYSE, monthly-ave climatology for HBVEC).

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

N_DAYS = 2 * 365
START = "2000-01-01"
LAT, LON, ELEV, AREA_M2 = 51.0, -115.0, 1500.0, 100.0e6

# Sensible per-template default parameter sets (RavenPy emulator example values, in each
# emulator's verified positional order). Bounds *midpoints* are hydrologically degenerate for
# some templates (e.g. HBVEC yields all-zero flow), so canonical defaults are used to assert a
# non-zero hydrograph. Order matches ravenpy.config.emulators.{...}.P exactly.
TEMPLATE_PARAMS = {
    "GR4JCN": {
        "GR4J_X1": 0.529, "GR4J_X2": -3.396, "GR4J_X3": 407.29,
        "GR4J_X4": 1.072, "CEMANEIGE_X1": 16.9, "CEMANEIGE_X2": 0.947,
    },
    # HBVEC X01..X21 (RavenPy HBV-EC Salmon River example).
    "HBVEC": dict(zip(
        [f"X{i:02d}" for i in range(1, 22)],
        [0.05984519, 4.072232, 2.001574, 0.03473693, 0.09985144, 0.5060520,
         3.438486, 38.32455, 0.4606565, 0.06303738, 2.277781, 4.873686,
         0.5718813, 0.04505643, 0.877607, 18.94145, 2.036937, 0.4452843,
         0.6771759, 1.141608, 1.024278],
    )),
    # HMETS (RavenPy HMETS Salmon River example), named params in P order.
    "HMETS": {
        "GAMMA_SHAPE": 9.5019, "GAMMA_SCALE": 0.2774, "GAMMA_SHAPE2": 6.3942,
        "GAMMA_SCALE2": 0.6884, "MIN_MELT_FACTOR": 1.2875, "MAX_MELT_FACTOR": 5.4134,
        "DD_MELT_TEMP": 2.3641, "DD_AGGRADATION": 0.0973, "SNOW_SWI_MIN": 0.0464,
        "SNOW_SWI_MAX": 0.1998, "SWI_REDUCT_COEFF": 0.0222, "DD_REFREEZE_TEMP": -1.0919,
        "REFREEZE_FACTOR": 2.6851, "REFREEZE_EXP": 0.3740, "PET_CORRECTION": 1.0000,
        "HMETS_RUNOFF_COEFF": 0.4739, "PERC_COEFF": 0.0114, "BASEFLOW_COEFF_1": 0.0243,
        "BASEFLOW_COEFF_2": 0.0069, "TOPSOIL": 310.74, "PHREATIC": 916.20,
    },
    # MOHYSE X01..X10 (RavenPy MOHYSE example).
    "MOHYSE": dict(zip(
        [f"X{i:02d}" for i in range(1, 11)],
        [1.0, 0.0468, 4.2952, 2.658, 0.4038, 0.0621, 0.0273, 0.0453, 0.9039, 5.6167],
    )),
}


def _build_forcing(forcings_dir: Path, domain: str) -> None:
    """Canonical CFIF daily forcing (precipitation_flux kg m-2 s-1, air_temperature K)."""
    import xarray as xr

    forcings_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    time = pd.date_range(START, periods=N_DAYS, freq="D")
    precip_flux = (np.abs(rng.gamma(1.0, 5.0, N_DAYS)) / 86400.0).astype("f4")  # mm/d -> kg/m2/s
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
    ds.to_netcdf(forcings_dir / f"{domain}_forcing.nc", engine="netcdf4", encoding=enc)
    ds.close()


def _build_attributes(attrs_dir: Path, domain: str) -> None:
    """Grouped attributes NetCDF the plugin's HRU resolver reads (hru_identity + terrain)."""
    import netCDF4

    attrs_dir.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(attrs_dir / f"{domain}_attributes.nc"), "w", format="NETCDF4") as root:
        root.Conventions = "CF-1.8"
        root.domain_name = domain
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


@pytest.mark.parametrize("template", ["GR4JCN", "HBVEC", "HMETS", "MOHYSE"])
def test_emulator_template_runs_and_returns_streamflow(template, tmp_path, monkeypatch):
    """Each Raven template runs the real binary via the plugin and returns finite streamflow."""
    monkeypatch.setenv("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))
    logger = logging.getLogger(f"raven_run_{template}")

    domain = f"TMPL{template}"
    data_dir = tmp_path / "data_root"
    project_dir = data_dir / f"domain_{domain}"
    mr = project_dir / "data" / "model_ready"
    _build_forcing(mr / "forcings", domain)
    _build_attributes(mr / "attributes", domain)

    config = {
        "SYMFLUENCE_DATA_DIR": str(data_dir),
        "DOMAIN_NAME": domain,
        "RAVEN_MODEL_TEMPLATE": template,
        "RAVEN_RUN_NAME": f"run_{template}",
        "EXPERIMENT_TIME_START": "2000-01-01 00:00",
        "EXPERIMENT_TIME_END": "2001-12-30 00:00",
        "RAVEN_EXE": str(RAVEN_EXEC_PATH),
        "RAVEN_INSTALL_PATH": str(Path(str(RAVEN_EXEC_PATH)).parent),
    }

    from raven_symfluence import emulator as em

    # The plugin orders the vector in the template's verified RavenPy positional order.
    param_vector = em.params_to_vector(template, TEMPLATE_PARAMS[template], logger)
    assert len(param_vector) == len(TEMPLATE_PARAMS[template])

    settings_dir = project_dir / "settings" / "RAVEN"
    output_dir = project_dir / "simulations" / "run" / "RAVEN"
    ok = em.build_and_run_emulator(
        config=config,
        settings_dir=settings_dir,
        output_dir=output_dir,
        params=TEMPLATE_PARAMS[template],
        logger=logger,
    )
    assert ok, f"plugin build_and_run_emulator failed for template {template}"

    from raven_symfluence.postprocessor import find_hydrographs_file, read_raven_streamflow

    hydro_csv = find_hydrographs_file(output_dir)
    assert hydro_csv is not None, f"{template}: no Hydrographs.csv written under {output_dir}"

    streamflow = read_raven_streamflow(hydro_csv, config={}, logger=logger)
    assert streamflow is not None and not streamflow.empty, f"{template}: empty streamflow"
    q = streamflow.to_numpy(dtype=float)
    assert q.size > 700, f"{template}: expected ~730 daily steps, got {q.size}"
    assert np.isfinite(q).all(), f"{template}: streamflow contains non-finite values"
    assert (q >= 0).all(), f"{template}: streamflow should be non-negative"
    assert q.sum() > 0, f"{template}: simulated streamflow is identically zero"
