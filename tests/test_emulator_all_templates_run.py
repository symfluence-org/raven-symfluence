# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""End-to-end run test across all eight wired Raven hydrologic emulator templates.

Parametrizes over ``GR4JCN``, ``HBVEC``, ``HMETS``, ``MOHYSE``, ``BLENDED``,
``CANADIANSHIELD``, ``HYPR``, ``SACSMA``: for each template it builds a self-contained
model-ready store (canonical CFIF forcing + grouped attributes), drives the *plugin's*
:func:`raven_symfluence.emulator.build_and_run_emulator` — the exact function
``RavenRunner``/``RavenWorker`` call — against it, RUNS the real raven binary, and asserts a
non-empty, finite, non-negative, non-zero simulated streamflow series comes back.

This proves each template is wired through the template-aware emulator path (right RavenPy
class, parameter ordering, and the minimal per-template extras driven by PRECIP + TEMP_AVE
only: RAINSNOW_DINGMAN override for HMETS/MOHYSE/SACSMA, monthly-ave climatology for HBVEC/HYPR,
the PET_OUDIN override + organic/bedrock HRU pair for CanadianShield).

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
    # BLENDED X01..X35 then R01..R08 (RavenPy Blended example params, tests/emulators.py).
    "BLENDED": dict(zip(
        [f"X{i:02d}" for i in range(1, 36)] + [f"R{i:02d}" for i in range(1, 9)],
        [2.930702e-02, 2.211166e00, 2.166229e00, 0.0002254976, 2.173976e01, 1.565091e00,
         6.211146e00, 9.313578e-01, 3.486263e-02, 0.251835, 0.0002279250, 1.214339e00,
         4.736668e-02, 0.2070342, 7.806324e-02, -1.336429e00, 2.189741e-01, 3.845617e00,
         2.950022e-01, 4.827523e-01, 4.099820e00, 1.283144e01, 5.937894e-01, 1.651588e00,
         1.705806, 3.719308e-01, 7.121015e-02, 1.906440e-02, 4.080660e-01, 9.415693e-01,
         -1.856108e00, 2.356995e00, 1.0e00, 1.0e00, 7.510967e-03, 5.321608e-01, 2.891977e-02,
         9.605330e-01, 6.128669e-01, 9.558293e-01, 1.008196e-01, 9.275730e-02, 7.469583e-01],
    )),
    # CANADIANSHIELD X01..X34 (RavenPy CanadianShield example params, tests/emulators.py).
    "CANADIANSHIELD": dict(zip(
        [f"X{i:02d}" for i in range(1, 35)],
        [4.72304300e-01, 8.16392200e-01, 9.86197600e-02, 3.92699900e-03, 4.69073600e-02,
         4.95528400e-01, 6.803492000e00, 4.33050200e-03, 1.01425900e-05, 1.823470000e00,
         5.12215400e-01, 9.017555000e00, 3.077103000e01, 5.094095000e01, 1.69422700e-01,
         8.23412200e-02, 2.34595300e-01, 7.30904000e-02, 1.284052000e00, 3.653415000e00,
         2.306515000e01, 2.402183000e00, 2.522095000e00, 5.80344900e-01, 1.614157000e00,
         6.031781000e00, 3.11129800e-01, 6.71695100e-02, 5.83759500e-05, 9.824723000e00,
         9.00747600e-01, 8.04057300e-01, 1.179003000e00, 7.98001300e-01],
    )),
    # HYPR X01..X21 (RavenPy HYPR example params, tests/emulators.py).
    "HYPR": dict(zip(
        [f"X{i:02d}" for i in range(1, 22)],
        [-1.856410e-01, 2.92301100e00, 3.1194200e-02, 4.3982810e-01, 4.6509760e-01,
         1.1770040e-01, 1.31236800e01, 4.0417950e-01, 1.21225800e00, 5.91273900e01,
         1.6612030e-01, 4.10501500e00, 8.2296110e-01, 4.15635200e01, 5.85111700e00,
         6.9090140e-01, 9.2459950e-01, 1.64358800e00, 1.59920500e00, 2.51938100e00,
         1.14820100e00],
    )),
    # SACSMA X01..X21 (RavenPy SACSMA example params, tests/emulators.py). X01..X03 are log10
    # quantities in the canonical example; values are used verbatim in P order.
    "SACSMA": dict(zip(
        [f"X{i:02d}" for i in range(1, 22)],
        [0.0100000, 0.0500000, 0.3000000, 0.0500000, 0.0500000, 0.1300000, 0.0250000,
         0.0600000, 0.0600000, 1.0000000, 40.000000, 0.0000000, 0.0000000, 0.1000000,
         0.0000000, 0.0100000, 1.5000000, 0.4827523, 4.0998200, 1.0000000, 1.0000000],
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


@pytest.mark.parametrize("template", [
    "GR4JCN", "HBVEC", "HMETS", "MOHYSE",
    "BLENDED", "CANADIANSHIELD", "HYPR", "SACSMA",
])
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
