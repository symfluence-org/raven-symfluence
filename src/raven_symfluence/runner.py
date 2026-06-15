# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Model Runner.

Executes the Raven hydrological modelling framework. RavenPy generates the
``.rv*`` files and drives the ``raven`` binary; the runner resolves the binary
(so SYMFLUENCE's bundled build is used), points RavenPy at it, and runs the
emulator built from the model-ready store.

RavenPy and the binary are resolved lazily (inside ``run_raven``), so the runner
imports/registers cleanly without RavenPy installed (the SYMFLUENCE convention).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.models.base import BaseModelRunner


class RavenRunner(BaseModelRunner):  # type: ignore[misc]
    """Runner for the Raven hydrological modelling framework (RavenPy-driven)."""

    MODEL_NAME = "RAVEN"

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ):
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up Raven-specific paths (settings dir holds the generated .rv* files)."""
        self.setup_dir = self.project_dir / 'settings' / self.model_name

    def _get_expected_outputs(self) -> list:
        """Raven writes Hydrographs.csv as its primary streamflow output."""
        return ['Hydrographs.csv']

    def run_raven(self, **kwargs) -> Optional[Path]:
        """Run the Raven model: resolve the binary, build the emulator, execute it.

        Returns the output directory on success, None on failure.
        """
        self.logger.info(f"Starting Raven run for domain: {self.domain_name}")

        with symfluence_error_handler(
            "Raven model execution",
            self.logger,
            error_type=ModelExecutionError,
        ):
            output_dir = self._get_output_dir()
            self.ensure_dir(output_dir)

            # Resolve the Raven binary so RavenPy uses SYMFLUENCE's bundled build.
            raven_exe = self.get_model_executable(
                install_path_key='RAVEN_INSTALL_PATH',
                default_install_subpath='installs/raven/bin',
                exe_name_key='RAVEN_EXE',
                default_exe_name='raven',
            )
            if raven_exe.exists():
                # RavenPy locates the engine via RAVENPY_RAVEN_BINARY_PATH.
                os.environ['RAVENPY_RAVEN_BINARY_PATH'] = str(raven_exe)
                self.logger.debug(f"Using Raven binary: {raven_exe}")
            else:
                self.logger.warning(
                    f"Raven binary not found at {raven_exe}; "
                    "relying on RavenPy's bundled raven-hydro if available")

            # Lazy import: RavenPy is only needed at run time.
            try:
                from .emulator import build_and_run_emulator
            except ImportError as e:
                self.logger.error(f"RavenPy not available: {e}")
                return None

            params = self._initial_parameters()
            ok = build_and_run_emulator(
                config=self.config_dict,
                settings_dir=self.setup_dir,
                output_dir=output_dir,
                params=params,
                logger=self.logger,
            )
            if not ok:
                self.logger.error("Raven run failed")
                return None

            if self._get_expected_outputs():
                # Hydrographs.csv may land in a RavenPy output/ subdir; verify loosely.
                from .postprocessor import find_hydrographs_file
                if find_hydrographs_file(output_dir) is None:
                    self.logger.error(
                        f"Raven completed but no Hydrographs.csv found under {output_dir}")
                    return None

            self.logger.info(f"Raven run completed. Output: {output_dir}")
            return output_dir

    def _initial_parameters(self) -> Dict[str, float]:
        """Initial parameters for the run (from the persisted sidecar or bounds midpoint)."""
        # Prefer parameters persisted by a prior preprocessing/calibration step.
        try:
            import json

            sidecar = self.setup_dir / 'raven_params.json'
            if sidecar.exists():
                with open(sidecar, encoding='utf-8') as fh:
                    payload = json.load(fh)
                params = payload.get('params')
                if params:
                    return {k: float(v) for k, v in params.items()}
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.debug(f"Could not read raven_params.json: {e}")

        try:
            from .calibration.parameter_manager import RavenParameterManager

            pm = RavenParameterManager(self.config, self.logger, self.setup_dir)
            return pm.get_initial_parameters() or {}
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.debug(f"Could not resolve initial Raven parameters: {e}")
            return {}


__all__ = ['RavenRunner']
