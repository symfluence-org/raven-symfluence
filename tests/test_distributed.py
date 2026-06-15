# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the distributed Raven network builder (no RavenPy required).

Covers topology reading from a synthetic attributes store, downstream/outlet resolution,
upstream-area accumulation, hydraulic-geometry channel sizing, and the RavenPy-object
construction (with a lightweight fake ``rc`` so RavenPy need not be installed).
"""
from __future__ import annotations

import logging

from _fixtures import build_distributed_attributes_store

from raven_symfluence.distributed import (
    RAVEN_OUTLET_ID,
    build_distributed_objects,
    estimate_bankfull_geometry,
    read_distributed_topology,
)

logger = logging.getLogger("test_distributed")


def test_estimate_bankfull_geometry_monotonic_and_floored():
    w_small, d_small = estimate_bankfull_geometry(0.0)
    w_big, d_big = estimate_bankfull_geometry(5000.0)
    assert w_small >= 1.0 and d_small >= 0.3          # floors
    assert w_big > w_small and d_big > d_small         # grows with drainage area


def test_read_topology_chain(tmp_path):
    """A 3-subbasin chain 1->2->3(outlet) parses with correct connectivity + areas."""
    proj = tmp_path / "domain_synd"
    build_distributed_attributes_store(
        proj, "synd", ids=["1", "2", "3"], downstream=["2", "3", "3"],
        areas_m2=[1e7, 1e7, 1e7], river_length_m=[5000.0, 5000.0, 0.0],
        river_slope=[0.01, 0.02, 0.0])

    net = read_distributed_topology(proj, "synd", logger)
    assert net is not None and net.n_subbasins == 3

    by_id = {s.subbasin_id: s for s in net.subbasins}
    assert by_id[1].downstream_id == 2
    assert by_id[2].downstream_id == 3
    assert by_id[3].downstream_id == RAVEN_OUTLET_ID          # self-loop -> outlet
    # Upstream-accumulated area increases downstream: 10, 20, 30 km2.
    assert by_id[1].accumulated_area_km2 == 10.0
    assert by_id[2].accumulated_area_km2 == 20.0
    assert by_id[3].accumulated_area_km2 == 30.0
    # Outlet reach has no channel (length 0); slope floored to a positive value.
    assert by_id[3].reach_length_km == 0.0
    assert by_id[3].river_slope > 0


def test_read_topology_lumped_returns_none(tmp_path):
    """A single-subbasin (or downstream-less) store yields None -> lumped fallback."""
    proj = tmp_path / "domain_one"
    build_distributed_attributes_store(
        proj, "one", ids=["1"], downstream=["1"])
    assert read_distributed_topology(proj, "one", logger) is None


def test_read_topology_no_store_returns_none(tmp_path):
    assert read_distributed_topology(tmp_path / "domain_missing", "missing", logger) is None


class _FakeChannelProfile:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeSubBasin:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeRC:
    """Minimal stand-in for ravenpy.config.commands (SubBasin/ChannelProfile)."""

    SubBasin = _FakeSubBasin
    ChannelProfile = _FakeChannelProfile


def test_build_distributed_objects_shapes(tmp_path):
    proj = tmp_path / "domain_synd"
    build_distributed_attributes_store(
        proj, "synd", ids=["1", "2", "3"], downstream=["2", "3", "3"],
        river_length_m=[5000.0, 5000.0, 0.0])
    net = read_distributed_topology(proj, "synd", logger)

    subs, hrus, channels, gauged = build_distributed_objects(_FakeRC, net, logger)
    assert len(subs) == 3 and len(hrus) == 3
    # Every subbasin is gauged so each reach is an output column.
    assert gauged == [1, 2, 3]
    assert all(s.gauged for s in subs)
    # One HRU per subbasin, mapped by subbasin_id.
    assert {h["subbasin_id"] for h in hrus} == {1, 2, 3}
    assert all(h["hru_type"] == "land" for h in hrus)
    # Every subbasin gets a valid channel profile (Raven rejects "NONE" under routing).
    assert len(channels) == 3
    assert {c.name for c in channels} == {"chn_1", "chn_2", "chn_3"}
    # Channel profiles are valid trapezoids (8 survey points, 3 roughness zones).
    for c in channels:
        assert len(c.survey_points) == 8
        assert len(c.roughness_zones) == 3
        assert c.bed_slope > 0
