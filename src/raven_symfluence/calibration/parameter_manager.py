# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Parameter Manager.

Calibrates the scalar parameters of a Raven emulator structure (Phase 1: the six
GR4JCN parameters X1..X4 + CemaNeige CN1/CN2). The package owns its parameter
bounds (see :mod:`.bounds`), matching the dRoute/JAX-model plugin pattern.

Parameters are applied in-memory: RavenPy regenerates the ``.rvp`` (and the rest
of the ``.rv*`` set) from the parameter vector each iteration, so
``update_model_files`` records the trial parameters in a small JSON sidecar
(``raven_params.json``) that the worker reads when it rebuilds the model. There is
no fixed-width Fortran parameter file to patch.

Configuration keys:
    RAVEN_MODEL_TEMPLATE        : emulator structure (GR4JCN default)
    RAVEN_PARAMS_TO_CALIBRATE   : comma-separated params; 'default' = full set for the template
    RAVEN_PARAM_BOUNDS          : optional {param: [min, max]} / {param: {min, max}} overrides
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

from .bounds import get_raven_bounds


class RavenParameterManager(BaseParameterManager):
    """Parameter manager for Raven emulator calibration (lumped scalar parameters)."""

    def __init__(self, config: Any, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        self.model_template = str(self._get_config_value(
            lambda: self.config.model.raven.model_template,
            default='GR4JCN', dict_key='RAVEN_MODEL_TEMPLATE')).upper()

        params_str = self._get_config_value(
            lambda: self.config.model.raven.params_to_calibrate,
            default='default', dict_key='RAVEN_PARAMS_TO_CALIBRATE')
        if params_str is None or str(params_str).strip().lower() in ('', 'default', 'all'):
            self._explicit_params: Optional[List[str]] = None
        else:
            self._explicit_params = [p.strip() for p in str(params_str).split(',') if p.strip()]

    # ---- helpers ----------------------------------------------------------------------------
    def _get(self, default: Any, key: str) -> Any:
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    # ---- BaseParameterManager contract ------------------------------------------------------
    def _get_parameter_names(self) -> List[str]:
        registry_bounds = get_raven_bounds(self.model_template)
        if self._explicit_params is not None:
            # Honour the user's ordering; warn (not fail) on unknown names so registration
            # stays resilient and the user sees the typo.
            for p in self._explicit_params:
                if p not in registry_bounds:
                    self.logger.warning(
                        f"Raven param '{p}' has no bounds for template "
                        f"'{self.model_template}'; using [0, 1]")
            return list(self._explicit_params)
        return list(registry_bounds.keys())

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        registry_bounds = get_raven_bounds(self.model_template)
        bounds: Dict[str, Dict[str, Any]] = {}
        for param in self.all_param_names:
            if param in registry_bounds:
                bounds[param] = registry_bounds[param]
            else:
                bounds[param] = {'min': 0.0, 'max': 1.0, 'transform': 'linear'}

        config_bounds = self._get(None, 'RAVEN_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        bounds = self.param_bounds
        initial: Dict[str, float] = {}
        for param in self.all_param_names:
            if param not in bounds:
                continue
            b = bounds[param]
            if b.get('transform') == 'log' and b['min'] > 0:
                initial[param] = math.sqrt(b['min'] * b['max'])
            else:
                initial[param] = (b['min'] + b['max']) / 2.0
        return initial if initial else None

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """Record the trial parameters for the worker to rebuild the Raven model from.

        RavenPy regenerates the ``.rv*`` files from the parameter vector each iteration,
        so we persist the trial params to a JSON sidecar rather than patching files.
        """
        try:
            self.settings_dir.mkdir(parents=True, exist_ok=True)
            out = Path(self.settings_dir) / 'raven_params.json'
            payload = {
                'model_template': self.model_template,
                'param_names': list(self.all_param_names),
                'params': {k: float(v) for k, v in params.items()},
            }
            with open(out, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
            self.logger.debug(f"Wrote Raven trial parameters -> {out}")
            return True
        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error writing Raven parameters: {e}")
            return False


__all__ = ['RavenParameterManager']
