# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Self-contained fixture domain builder for the Phase-1 multi-engine benchmark.

Builds the SYMFLUENCE model-ready store the engine plugins read -- canonical CFIF
forcing, grouped catchment attributes, and a synthetic-but-reasonable observed
streamflow series so a short DDS calibration has a real target. Everything is written
under a caller-supplied directory (use a tmp dir); nothing touches the user's real
domains.

This is a benchmark-harness extension of the test fixture builders in
``tests/test_pipeline_fixture_domain.py`` (kept in sync intentionally): same single
lumped HRU, same CFIF vocabulary, same observation layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# A single lumped HRU resembling a small mountain headwater (Bow-at-Banff-ish).
LAT, LON, ELEV, AREA_M2 = 51.0, -115.0, 1500.0, 100.0e6
START = "2000-01-01"
N_DAYS = 3 * 365


@dataclass
class FixtureDomain:
    """Paths + window describing a built fixture domain."""

    domain_name: str
    data_dir: Path
    project_dir: Path
    config_path: Path
    time: pd.DatetimeIndex
    calibration_period: str
    evaluation_period: str


def _build_forcing(forcings_dir: Path, domain: str) -> pd.DatetimeIndex:
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
    ds.to_netcdf(forcings_dir / f"{domain}_forcing.nc", engine="netcdf4", encoding=enc)
    ds.close()
    return time


def _build_attributes(attrs_dir: Path, domain: str) -> None:
    """Grouped attributes NetCDF: hru_identity + terrain (the groups the plugins read)."""
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


def _build_observations(obs_preproc_dir: Path, time: pd.DatetimeIndex, domain: str) -> None:
    """Synthetic but reasonable daily observed streamflow (the calibration target)."""
    rng = np.random.default_rng(11)
    base = 5.0 + 3.0 * np.sin((np.arange(len(time)) / 365.0 + 0.25) * 2 * np.pi)
    q = np.clip(base + rng.normal(0, 0.5, len(time)), 0.05, None)
    obs_preproc_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(q, index=time, name="discharge_cms").to_frame().to_csv(
        obs_preproc_dir / f"{domain}_streamflow_processed.csv")


def _write_config(fx: "FixtureDomain", code_dir: Path, engine: str, binaries: dict) -> None:
    """Write a per-engine SYMFLUENCE config against the shared store.

    ``binaries`` maps an engine name to its resolved binary path (or None). Only the
    keys relevant to *engine* are emitted; the rest of the store is engine-agnostic.
    """
    lines = [
        f'SYMFLUENCE_DATA_DIR: "{fx.data_dir}"',
        f'SYMFLUENCE_CODE_DIR: "{code_dir}"',
        f'DOMAIN_NAME: "{fx.domain_name}"',
        f'EXPERIMENT_ID: "bench_{engine.lower()}"',
        f'HYDROLOGICAL_MODEL: "{engine}"',
        'DOMAIN_DEFINITION_METHOD: "lumped"',
        'SUB_GRID_DISCRETIZATION: "elevation"',
        'FORCING_DATASET: "ERA5"',
        'EXPERIMENT_TIME_START: "2000-01-01 00:00"',
        'EXPERIMENT_TIME_END: "2002-12-30 00:00"',
        f'CALIBRATION_PERIOD: "{fx.calibration_period}"',
        f'EVALUATION_PERIOD: "{fx.evaluation_period}"',
        'OPTIMIZATION_METHODS: ["iteration"]',
        'ITERATIVE_OPTIMIZATION_ALGORITHM: "DDS"',
        'OPTIMIZATION_METRIC: "KGE"',
        'NUMBER_OF_ITERATIONS: 6',
        'NUM_PROCESSES: 1',
        'FORCE_RUN_ALL_STEPS: true',
    ]
    if engine == "RAVEN":
        raven = binaries.get("RAVEN")
        lines += [
            'RAVEN_MODEL_TEMPLATE: "GR4JCN"',
            'RAVEN_SPATIAL_MODE: "lumped"',
            f'RAVEN_EXE: "{raven}"',
            f'RAVEN_INSTALL_PATH: "{Path(raven).parent}"',
            'RAVEN_PARAMS_TO_CALIBRATE: "default"',
        ]
    elif engine == "FUSE":
        fuse = binaries.get("FUSE")
        lines += [
            f'FUSE_EXE: "{Path(fuse).name}"',
            f'FUSE_INSTALL_PATH: "{Path(fuse).parent}"',
        ]
    elif engine == "SUMMA":
        summa = binaries.get("SUMMA")
        lines += [
            f'SUMMA_EXE: "{Path(summa).name if summa else "summa.exe"}"',
            f'SUMMA_INSTALL_PATH: "{Path(summa).parent if summa else "default"}"',
        ]
    fx.config_path.write_text("\n".join(lines) + "\n")


def build_fixture_domain(
    data_dir: Path,
    code_dir: Path,
    domain_name: str = "BENCHFIX",
) -> FixtureDomain:
    """Build the shared, engine-agnostic model-ready store under *data_dir*.

    Returns a :class:`FixtureDomain`. Per-engine configs are written later via
    :func:`write_engine_config` so every engine reads the same forcing/attrs/obs.
    """
    data_dir = Path(data_dir)
    project_dir = data_dir / f"domain_{domain_name}"
    mr = project_dir / "data" / "model_ready"
    data_dir.mkdir(parents=True, exist_ok=True)

    time = _build_forcing(mr / "forcings", domain_name)
    _build_attributes(mr / "attributes", domain_name)
    _build_observations(
        project_dir / "data" / "observations" / "streamflow" / "preprocessed", time, domain_name)

    fx = FixtureDomain(
        domain_name=domain_name,
        data_dir=data_dir,
        project_dir=project_dir,
        config_path=data_dir / "config_placeholder.yaml",
        time=time,
        calibration_period="2000-06-01, 2001-12-31",
        evaluation_period="2002-01-01, 2002-12-30",
    )
    return fx


def write_engine_config(
    fx: FixtureDomain, code_dir: Path, engine: str, binaries: dict
) -> Path:
    """Write (and return the path to) a per-engine config against the shared store."""
    cfg = fx.data_dir / f"config_{engine.lower()}.yaml"
    fx.config_path = cfg
    _write_config(fx, code_dir, engine, binaries)
    return cfg


def observed_streamflow(fx: FixtureDomain) -> pd.Series:
    """Read back the synthetic observed streamflow series (m3/s, daily)."""
    csv = (fx.project_dir / "data" / "observations" / "streamflow" / "preprocessed"
           / f"{fx.domain_name}_streamflow_processed.csv")
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    return df.iloc[:, 0]
