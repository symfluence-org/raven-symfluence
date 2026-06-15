# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the Raven multi-gauge bridge: per-reach Hydrographs -> routed NetCDF.

Validates that Raven's per-subbasin output is read correctly and converted into the
mizuRoute-style NetCDF (seg/segId/IRFroutedRunoff) that SYMFLUENCE's MultiGaugeMetrics
consumes — exercising the whole chain without RavenPy or the raven binary.
"""
from __future__ import annotations

import logging

import pandas as pd

from raven_symfluence.postprocessor import read_raven_per_reach, write_routed_netcdf

logger = logging.getLogger("test_multigauge_routing")


def _write_hydrographs(path, n_days=400):
    """Fabricate a Raven Hydrographs.csv: date + three gauged-subbasin columns."""
    t = pd.date_range("2002-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "time": range(1, n_days + 1),
        "date": t.strftime("%Y-%m-%d"),
        "hour": "12:00:00",
        "precip [mm/day]": 2.0,
        "sub_1 [m3/s]": 3.0,    # outlet (largest)
        "sub_2 [m3/s]": 2.0,
        "sub_3 [m3/s]": 1.0,    # headwater
    })
    df.to_csv(path, index=False)
    return path


def test_read_per_reach_parses_subbasin_columns(tmp_path):
    hp = _write_hydrographs(tmp_path / "Hydrographs.csv")
    per_reach = read_raven_per_reach(hp, logger)
    assert per_reach is not None
    assert set(per_reach.columns) == {1, 2, 3}        # integer subbasin ids
    assert abs(per_reach[1].mean() - 3.0) < 1e-9
    assert "precip" not in [str(c).lower() for c in per_reach.columns]


def test_write_routed_netcdf_layout(tmp_path):
    hp = _write_hydrographs(tmp_path / "Hydrographs.csv")
    per_reach = read_raven_per_reach(hp, logger)
    nc = write_routed_netcdf(per_reach, tmp_path / "routed.nc", logger)
    assert nc is not None and nc.exists()

    import xarray as xr

    ds = xr.open_dataset(nc)
    try:
        assert "IRFroutedRunoff" in ds.data_vars
        assert "segId" in ds.variables
        assert set(ds.dims) >= {"time", "seg"}
        assert list(ds["segId"].values) == [1, 2, 3]
        assert ds["IRFroutedRunoff"].attrs.get("units") == "m3/s"
    finally:
        ds.close()


def test_multigauge_metrics_consumes_routed_netcdf(tmp_path):
    """The routed NetCDF + a gauge->segment mapping feeds MultiGaugeMetrics end-to-end."""
    import pytest

    metrics_mod = pytest.importorskip(
        "symfluence.optimization.multi_gauge.metrics",
        reason="symfluence multi_gauge not importable")
    from _fixtures import write_multigauge_obs

    # Simulated routed flow from Raven (3 reaches).
    hp = _write_hydrographs(tmp_path / "Hydrographs.csv")
    per_reach = read_raven_per_reach(hp, logger)
    routed_nc = write_routed_netcdf(per_reach, tmp_path / "routed.nc", logger)

    # Two gauges mapped to subbasins 1 and 2.
    mapping = tmp_path / "gauge_segment_mapping.csv"
    pd.DataFrame({
        "id": [101, 102],
        "name": ["outlet", "mid"],
        "nearest_segment": [1, 2],
        "distance_to_segment": [0.0, 0.0],
    }).to_csv(mapping, index=False)
    obs_dir = write_multigauge_obs(tmp_path / "obs", [101, 102])

    mg = metrics_mod.MultiGaugeMetrics(
        gauge_segment_mapping_path=mapping, obs_data_dir=obs_dir, logger=logger)
    result = mg.calculate_multi_gauge_metrics(
        mizuroute_output_path=routed_nc, min_gauges=1,
        aggregation="mean", min_overlap_days=10)

    assert result["n_valid_gauges"] >= 1
    assert "kge" in result and result["kge"] is not None
