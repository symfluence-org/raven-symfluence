# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR4JCN calibration bounds are well-formed."""
from __future__ import annotations


def test_gr4jcn_bounds_wellformed():
    from raven_symfluence.calibration.bounds import get_raven_bounds

    bounds = get_raven_bounds("GR4JCN")
    assert {"X1", "X2", "X3", "X4"} <= set(bounds)
    for name, spec in bounds.items():
        assert spec["min"] < spec["max"], f"{name}: min !< max"
        assert spec.get("transform", "linear") in ("linear", "log")


def test_unknown_template_falls_back():
    from raven_symfluence.calibration.bounds import get_raven_bounds

    # Unknown template should not raise — falls back to a sane default set.
    bounds = get_raven_bounds("NOPE")
    assert isinstance(bounds, dict) and bounds
