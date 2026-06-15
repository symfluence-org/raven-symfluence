# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for Raven parameter regionalization (transfer functions across subbasins).

Unit-covers the attribute loader, the transfer-function strategy (coefficients ->
per-subbasin values), and the parameter manager's coefficient exposure. The @integration
test runs a regionalized distributed model on the real raven binary.
"""
from __future__ import annotations

import logging
import os
import tempfile

import pytest

from _fixtures import (
    build_distributed_attributes_store,
    write_canonical_forcing,
    write_observations,
)
from raven_symfluence.distributed import read_distributed_topology

logger = logging.getLogger("test_regionalization")

# The shared regionalization framework must be present.
pytest.importorskip("symfluence.optimization.regionalization.strategies",
                    reason="SYMFLUENCE regionalization framework not available")


def _domain(tmp_path, n=5):
    """Synthetic distributed domain: n subbasins in a chain, varying elevation."""
    proj = tmp_path / "domain_reg"
    ids = [str(i) for i in range(1, n + 1)]
    downstream = [str(i + 1) for i in range(1, n)] + [str(n)]   # 1->2->...->n(outlet)
    build_distributed_attributes_store(
        proj, "reg", ids=ids, downstream=downstream,
        elevations=[1000.0 + 300.0 * i for i in range(n)],
        river_length_m=[5000.0] * (n - 1) + [0.0])
    return proj


def test_load_subbasin_attributes(tmp_path):
    from raven_symfluence.calibration.raven_regionalization import load_raven_subbasin_attributes

    proj = _domain(tmp_path)
    net = read_distributed_topology(proj, "reg", logger)
    attrs = load_raven_subbasin_attributes(proj, "reg", net, logger)
    assert len(attrs) == net.n_subbasins
    assert "elev_m" in attrs.columns
    assert attrs["elev_m"].nunique() > 1                      # elevation varies across subbasins


def test_transfer_function_varies_with_attribute(tmp_path):
    import pandas as pd

    from raven_symfluence.calibration.raven_regionalization import create_raven_regionalization

    # 4 subbasins with increasing elevation; regionalize one param on elev_m.
    attrs = pd.DataFrame({
        "subbasin_id": [1, 2, 3, 4],
        "elev_m": [1000.0, 1500.0, 2000.0, 2500.0],
        "area_km2": [10.0, 10.0, 10.0, 10.0],
        "latitude": [51.0, 51.1, 51.2, 51.3],
        "longitude": [-115.0, -115.1, -115.2, -115.3],
    })
    bounds = {"CEMANEIGE_X1": {"min": 0.0, "max": 20.0, "transform": "linear"}}
    cfg = {"RAVEN_REGIONALIZATION_PARAM_CONFIG": {
        "CEMANEIGE_X1": {"attribute": "elev_m", "calibrate_b": True}}}
    strategy, resolved = create_raven_regionalization(
        method="transfer_function", param_bounds=bounds, n_units=4,
        attributes=attrs, model_template="GR4JCN", config=cfg, logger=logger)
    assert strategy is not None and "CEMANEIGE_X1" in resolved

    coeffs = strategy.get_calibration_parameters()
    assert "CEMANEIGE_X1_a" in coeffs and "CEMANEIGE_X1_b" in coeffs

    # Positive slope -> per-subbasin value increases with elevation.
    values, names = strategy.to_distributed({"CEMANEIGE_X1_a": 5.0, "CEMANEIGE_X1_b": 10.0})
    assert names == ["CEMANEIGE_X1"]
    col = [values[i, 0] for i in range(4)]
    assert col[0] < col[-1]                                    # varies monotonically with attr
    assert all(0.0 <= v <= 20.0 for v in col)                 # clipped to bounds


def test_param_manager_exposes_coefficients(tmp_path):
    from raven_symfluence.calibration.parameter_manager import RavenParameterManager

    _domain(tmp_path)                                          # creates the on-disk store
    cfg = {
        "DOMAIN_NAME": "reg",
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "RAVEN_MODEL_TEMPLATE": "GR4JCN",
        "RAVEN_SPATIAL_MODE": "distributed",
        "PARAMETER_REGIONALIZATION": "transfer_function",
    }
    pm = RavenParameterManager(cfg, logger, tmp_path / "settings")
    names = pm.all_param_names
    # Regionalized params become _a/_b coefficients; the rest stay raw scalars.
    assert any(n.endswith("_a") for n in names)
    assert any(n.endswith("_b") for n in names)
    assert "GR4J_X2" in names                                  # not regionalized -> raw
    for c in names:
        b = pm.param_bounds[c]
        assert b["min"] <= b["max"]


def test_param_manager_lumped_is_raw(tmp_path):
    """Without regionalization, the manager calibrates the raw template parameters."""
    from raven_symfluence.calibration.parameter_manager import RavenParameterManager

    cfg = {"DOMAIN_NAME": "x", "SYMFLUENCE_DATA_DIR": str(tmp_path),
           "RAVEN_MODEL_TEMPLATE": "GR4JCN", "RAVEN_SPATIAL_MODE": "lumped"}
    pm = RavenParameterManager(cfg, logger, tmp_path / "s")
    assert "GR4J_X1" in pm.all_param_names
    assert not any(n.endswith("_a") for n in pm.all_param_names)


# --- integration: regionalized run on the real raven binary ---------------------------------

try:
    from ravenpy._raven import RAVEN_EXEC_PATH
except Exception:  # noqa: BLE001
    RAVEN_EXEC_PATH = None


@pytest.mark.integration
@pytest.mark.skipif(not RAVEN_EXEC_PATH or not os.path.exists(str(RAVEN_EXEC_PATH)),
                    reason="raven binary not available")
def test_regionalized_distributed_run(tmp_path):
    pytest.importorskip("ravenpy")
    from raven_symfluence import emulator as em
    from raven_symfluence.calibration.parameter_manager import RavenParameterManager
    from raven_symfluence.postprocessor import find_hydrographs_file, read_raven_per_reach

    os.environ.setdefault("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))
    proj = _domain(tmp_path, n=4)
    write_canonical_forcing(proj, "reg")
    write_observations(proj, "reg")
    cfg = {
        "DOMAIN_NAME": "reg", "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "RAVEN_MODEL_TEMPLATE": "GR4JCN", "RAVEN_SPATIAL_MODE": "distributed",
        "PARAMETER_REGIONALIZATION": "transfer_function", "RAVEN_RUN_NAME": "reg",
        "EXPERIMENT_TIME_START": "2002-01-01", "EXPERIMENT_TIME_END": "2003-12-31",
    }
    pm = RavenParameterManager(cfg, logger, tempfile.mkdtemp())
    ok = em.build_and_run_emulator(
        config=cfg, settings_dir=tmp_path / "s", output_dir=tmp_path / "o",
        params=pm.get_initial_parameters(), logger=logger)
    assert ok, "regionalized distributed run failed"
    per_reach = read_raven_per_reach(find_hydrographs_file(tmp_path / "o"), logger)
    assert per_reach is not None and len(per_reach.columns) == 4
