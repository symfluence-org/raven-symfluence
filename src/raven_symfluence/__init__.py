# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""raven-symfluence — Raven hydrological modelling framework as a SYMFLUENCE model.

Registers Raven as a first-class SYMFLUENCE model via the ``symfluence.plugins``
entry point. The plugin feeds Raven from SYMFLUENCE's model-ready datastore
(canonical CFIF forcing + catchment attributes), runs it through RavenPy (which
writes the ``.rvi/.rvp/.rvh/.rvt/.rvc`` files and drives the ``raven`` binary),
and exposes it to the standard run + calibration pipeline.

RavenPy and the Raven binary are imported/resolved lazily (inside methods), so
``register()`` succeeds even when they are not installed.
"""
from __future__ import annotations

from ._version import __version__

__all__ = ["__version__", "register"]


def register() -> None:
    """Register Raven components with the SYMFLUENCE registry (entry-point hook)."""
    from symfluence.core.registry import model_manifest

    from .calibration.optimizer import RavenModelOptimizer
    from .calibration.parameter_manager import RavenParameterManager
    from .calibration.worker import RavenWorker
    from .config import RavenConfig, RavenConfigAdapter
    from .extractor import RavenResultExtractor
    from .postprocessor import RavenPostProcessor
    from .preprocessor import RavenPreProcessor
    from .runner import RavenRunner

    # Importing the calibration targets registers them via their @R.calibration_targets.add decorators.
    from .calibration import targets as _targets  # noqa: F401

    model_manifest(
        "RAVEN",
        config_adapter=RavenConfigAdapter,
        config_schema=RavenConfig,
        preprocessor=RavenPreProcessor,
        runner=RavenRunner,
        runner_method="run_raven",
        postprocessor=RavenPostProcessor,
        result_extractor=RavenResultExtractor,
        optimizer=RavenModelOptimizer,
        worker=RavenWorker,
        parameter_manager=RavenParameterManager,
        build_instructions_module="raven_symfluence.build_instructions",
    )
