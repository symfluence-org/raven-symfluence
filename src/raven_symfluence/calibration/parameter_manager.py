# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Parameter Manager.

Calibrates the scalar parameters of a Raven emulator structure (Phase 1: the six
GR4JCN parameters GR4J_X1..GR4J_X4 + CEMANEIGE_X1/CEMANEIGE_X2, matching
ravenpy.config.emulators.gr4jcn.P). The package owns its parameter bounds
(see :mod:`.bounds`), matching the dRoute/JAX-model plugin pattern.

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

        # Regionalization (Phase 4): when on (+ distributed), the optimizer calibrates
        # transfer-function coefficients instead of the raw regionalized parameters.
        self._reg_initialized = False
        self._reg_strategy: Any = None
        self._reg_params: Dict[str, Any] = {}      # resolved {param: {attribute, ...}}
        self._reg_coeff_bounds: Dict[str, tuple] = {}

    # ---- helpers ----------------------------------------------------------------------------
    def _get(self, default: Any, key: str) -> Any:
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    # ---- Regionalization (Phase 4) ----------------------------------------------------------
    def _raw_parameter_names(self) -> List[str]:
        """The raw (non-regionalized) parameter set: explicit list or the full template set."""
        registry_bounds = get_raven_bounds(self.model_template)
        if self._explicit_params is not None:
            for p in self._explicit_params:
                if p not in registry_bounds:
                    self.logger.warning(
                        f"Raven param '{p}' has no bounds for template "
                        f"'{self.model_template}'; using [0, 1]")
            return list(self._explicit_params)
        return list(registry_bounds.keys())

    def _ensure_regionalization(self) -> None:
        """Build the regionalization strategy once (distributed + non-lumped only)."""
        if self._reg_initialized:
            return
        self._reg_initialized = True

        method = str(self._get('lumped', 'PARAMETER_REGIONALIZATION')).lower()
        spatial = str(self._get('lumped', 'RAVEN_SPATIAL_MODE')).lower()
        if method == 'lumped' or spatial != 'distributed':
            return
        try:
            from ..distributed import read_distributed_topology
            from .raven_regionalization import (
                create_raven_regionalization,
                load_raven_subbasin_attributes,
            )

            domain = self._get('unknown', 'DOMAIN_NAME')
            data_dir = Path(self._get('.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{domain}"
            network = read_distributed_topology(project_dir, domain, self.logger)
            if network is None:
                self.logger.info("Regionalization requested but no routable topology; "
                                 "calibrating raw (global) parameters")
                return
            attributes = load_raven_subbasin_attributes(project_dir, domain, network, self.logger)
            cfg = self.config if isinstance(self.config, dict) else None
            strategy, resolved = create_raven_regionalization(
                method=method, param_bounds=get_raven_bounds(self.model_template),
                n_units=network.n_subbasins, attributes=attributes,
                model_template=self.model_template, config=cfg, logger=self.logger)
            if strategy is None or not resolved:
                return
            self._reg_strategy = strategy
            self._reg_params = resolved
            self._reg_coeff_bounds = strategy.get_calibration_parameters()
        except Exception as e:  # noqa: BLE001 -- fall back to raw params on any failure
            self.logger.warning(f"Regionalization init failed ({e}); calibrating raw params")

    # ---- BaseParameterManager contract ------------------------------------------------------
    def _get_parameter_names(self) -> List[str]:
        self._ensure_regionalization()
        if self._reg_strategy is not None:
            # Calibrate transfer-function coefficients for regionalized params + raw bounds
            # for the rest (parameters not regionalized stay global/scalar).
            coeffs = list(self._reg_coeff_bounds.keys())
            raw_kept = [p for p in self._raw_parameter_names() if p not in self._reg_params]
            return coeffs + raw_kept
        return self._raw_parameter_names()

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_regionalization()
        registry_bounds = get_raven_bounds(self.model_template)
        bounds: Dict[str, Dict[str, Any]] = {}
        for param in self.all_param_names:
            if param in self._reg_coeff_bounds:
                lo, hi = self._reg_coeff_bounds[param]
                bounds[param] = {'min': float(lo), 'max': float(hi), 'transform': 'linear'}
            elif param in registry_bounds:
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
