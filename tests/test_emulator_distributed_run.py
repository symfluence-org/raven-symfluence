# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""End-to-end distributed Raven run against the real RavenPy + raven binary.

Builds a synthetic 3-subbasin SYMFLUENCE domain (attributes store with a routing
``topology`` group + CFIF forcing + observations), drives
:func:`raven_symfluence.emulator.build_and_run_emulator` in distributed mode, and asserts
a routed multi-reach Hydrographs.csv comes back with discharge accumulating downstream.

Skipped automatically when RavenPy or the raven binary is unavailable.
"""
from __future__ import annotations

import logging
import os

import pytest

from _fixtures import (
    build_distributed_attributes_store,
    write_canonical_forcing,
    write_observations,
)

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

logger = logging.getLogger("test_distributed_run")


def _build_domain(tmp_path):
    """Synthetic 3-subbasin chain 1->2->3(outlet) with forcing + observations."""
    data_dir = tmp_path / "data"
    domain = "distrib"
    proj = data_dir / f"domain_{domain}"
    build_distributed_attributes_store(
        proj, domain, ids=["1", "2", "3"], downstream=["2", "3", "3"],
        areas_m2=[3.0e8, 3.0e8, 3.0e8], river_length_m=[20000.0, 20000.0, 0.0],
        river_slope=[0.01, 0.01, 0.0])
    write_canonical_forcing(proj, domain)
    write_observations(proj, domain)
    return data_dir, domain


def test_distributed_run_produces_routed_multireach_output(tmp_path):
    from raven_symfluence import emulator as em
    from raven_symfluence.postprocessor import find_hydrographs_file, read_raven_per_reach

    os.environ.setdefault("RAVENPY_RAVEN_BINARY_PATH", str(RAVEN_EXEC_PATH))
    data_dir, domain = _build_domain(tmp_path)

    config = {
        "DOMAIN_NAME": domain,
        "SYMFLUENCE_DATA_DIR": str(data_dir),
        "RAVEN_MODEL_TEMPLATE": "GR4JCN",
        "RAVEN_SPATIAL_MODE": "distributed",
        "RAVEN_RUN_NAME": "distrib",
        "EXPERIMENT_TIME_START": "2002-01-01",
        "EXPERIMENT_TIME_END": "2003-12-31",
    }
    settings_dir = tmp_path / "settings"
    output_dir = tmp_path / "out"

    ok = em.build_and_run_emulator(
        config=config,
        settings_dir=settings_dir,
        output_dir=output_dir,
        params={"GR4J_X1": 0.7, "GR4J_X2": -3.4, "GR4J_X3": 40.0,
                "GR4J_X4": 0.5, "CEMANEIGE_X1": 7.0, "CEMANEIGE_X2": 0.3},
        logger=logger,
    )
    assert ok, "distributed Raven run failed"

    hydro = find_hydrographs_file(output_dir)
    assert hydro is not None, "Hydrographs.csv not produced"
    per_reach = read_raven_per_reach(hydro, logger)
    assert per_reach is not None
    # Three gauged subbasins -> three reach columns.
    assert set(per_reach.columns) == {1, 2, 3}
    means = {c: float(per_reach[c].mean()) for c in per_reach.columns}
    assert all(v >= 0 for v in means.values())
    # Chain is 1->2->3(outlet): flow accumulates downstream, so the outlet (sub 3) carries
    # the most, then mid (sub 2), then the headwater (sub 1).
    assert means[3] >= means[2] >= means[1] > 0
