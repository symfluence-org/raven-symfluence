# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Model Optimizer.

Thin subclass of :class:`BaseModelOptimizer` (mirrors dRoute/FUSE). Delegates
execution to :class:`RavenWorker` and parameter handling to
:class:`RavenParameterManager`. Provides the standard evolutionary algorithm
entry points (``run_dds``, ``run_pso``, ``run_sce``, ``run_de``) via the base class.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.registries import R
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer

from .parameter_manager import RavenParameterManager  # noqa: F401 - trigger registration
from .worker import RavenWorker  # noqa: F401 - trigger registration


@R.optimizers.add('RAVEN')
class RavenModelOptimizer(BaseModelOptimizer):
    """Raven-specific optimizer using the unified BaseModelOptimizer framework."""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        from symfluence.core.config.coercion import coerce_config
        _cd = coerce_config(config, warn=False)
        self.data_dir = Path(_cd.get('SYMFLUENCE_DATA_DIR', '.'))
        self.domain_name = _cd.get('DOMAIN_NAME', 'unknown')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.raven_settings_dir = self.project_dir / 'settings' / 'RAVEN'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager)

    def _get_model_name(self) -> str:
        return 'RAVEN'

    def _create_parameter_manager(self):
        """Create the Raven parameter manager."""
        return RavenParameterManager(
            self.config_dict, self.logger, self.raven_settings_dir
        )

    def _check_routing_needed(self) -> bool:
        """Phase 1 is lumped GR4JCN; Raven handles its own (in-binary) routing."""
        return False

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best parameters to the worker for the final evaluation."""
        worker = getattr(self, 'worker', None)
        if worker is not None:
            worker._current_params = best_params
        return True

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run Raven with the best parameters for final evaluation."""
        worker = getattr(self, 'worker', None)
        if worker is not None:
            return worker.run_model(
                self.config_dict,
                self.raven_settings_dir,
                output_dir,
                sim_dir=output_dir,
                params=getattr(worker, '_current_params', {}),
            )
        return False

    def _get_final_file_manager_path(self) -> Path:
        """Return the path used for the final evaluation file manager record."""
        return self.raven_settings_dir / 'raven_params.json'

    def _setup_parallel_dirs(self) -> None:
        """Set up parallel directories for Raven calibration."""
        n_processors = int(self._get_config_value(
            lambda: self.config.system.number_of_processors, default=1))
        for i in range(n_processors):
            proc_dir = self.project_dir / 'simulations' / 'calibration' / f'proc_{i}' / 'RAVEN'
            proc_dir.mkdir(parents=True, exist_ok=True)


__all__ = ['RavenModelOptimizer']
