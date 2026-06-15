# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven calibration bounds are well-formed."""
from __future__ import annotations

import pytest


def test_gr4jcn_bounds_wellformed():
    from raven_symfluence.calibration.bounds import get_raven_bounds

    bounds = get_raven_bounds("GR4JCN")
    # Names/order match ravenpy.config.emulators.gr4jcn.P exactly.
    assert {"GR4J_X1", "GR4J_X2", "GR4J_X3", "GR4J_X4",
            "CEMANEIGE_X1", "CEMANEIGE_X2"} == set(bounds)
    for name, spec in bounds.items():
        assert spec["min"] < spec["max"], f"{name}: min !< max"
        assert spec.get("transform", "linear") in ("linear", "log")


@pytest.mark.parametrize("template,n", [("HBVEC", 21), ("HMETS", 21), ("MOHYSE", 10)])
def test_phase2_template_bounds(template, n):
    from raven_symfluence.calibration.bounds import get_raven_bounds

    bounds = get_raven_bounds(template)
    assert len(bounds) == n, f"{template}: expected {n} params, got {len(bounds)}"
    for name, spec in bounds.items():
        assert spec["min"] < spec["max"], f"{template}.{name}: min !< max"
        assert spec.get("transform", "linear") in ("linear", "log")


def test_unknown_template_falls_back():
    from raven_symfluence.calibration.bounds import get_raven_bounds

    # Unknown template should not raise — falls back to a sane default set.
    bounds = get_raven_bounds("NOPE")
    assert isinstance(bounds, dict) and bounds
