# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RavenConfig + adapter: defaults, schema wiring, validation, key recognition."""
from __future__ import annotations

import pytest


def test_config_defaults():
    from raven_symfluence.config import RavenConfig

    cfg = RavenConfig()
    assert cfg.model_template == "GR4JCN"
    assert cfg.spatial_mode == "lumped"
    assert cfg.exe == "raven"


def test_adapter_returns_schema():
    from raven_symfluence.config import RavenConfig, RavenConfigAdapter

    assert RavenConfigAdapter().get_config_schema() is RavenConfig


def test_adapter_rejects_bad_template():
    from raven_symfluence.config import RavenConfigAdapter

    with pytest.raises(ValueError):
        RavenConfigAdapter().validate({"model_template": "NOT_A_MODEL"})


def test_raven_keys_recognized_by_symfluence():
    import raven_symfluence
    raven_symfluence.register()
    from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP

    recognized = {k for k in FLAT_TO_NESTED_MAP if k.startswith("RAVEN_")}
    assert "RAVEN_MODEL_TEMPLATE" in recognized
    assert "RAVEN_SPATIAL_MODE" in recognized
