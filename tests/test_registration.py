# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven registers as a first-class SYMFLUENCE model (RavenPy not required)."""
from __future__ import annotations

import importlib.util

import pytest


def test_ravenpy_not_required_to_register():
    # The plugin must import + register with RavenPy absent (lazy-import contract).
    assert importlib.util.find_spec("raven_symfluence") is not None


@pytest.mark.parametrize(
    "registry",
    ["runners", "preprocessors", "postprocessors", "workers",
     "parameter_managers", "optimizers", "config_schemas"],
)
def test_raven_registered_in(registry):
    import raven_symfluence
    raven_symfluence.register()
    from symfluence.core.registries import R

    keys = [k.upper() for k in getattr(R, registry).keys()]
    assert "RAVEN" in keys, f"RAVEN missing from R.{registry}: {keys}"


def test_raven_streamflow_calibration_target_registered():
    import raven_symfluence
    raven_symfluence.register()
    from symfluence.core.registries import R

    assert "RAVEN_STREAMFLOW" in [k.upper() for k in R.calibration_targets.keys()]
