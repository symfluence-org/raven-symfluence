# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Calibration Targets.

Provides the streamflow calibration target for Raven output. Subclasses the
centralized :class:`StreamflowEvaluator` and overrides only the two model-specific
hooks: locating ``Hydrographs.csv`` and extracting the gauge column as a discharge
series (m3/s). Registered via ``@R.calibration_targets.add('RAVEN_STREAMFLOW')``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from symfluence.core.registries import R
from symfluence.evaluation.evaluators import StreamflowEvaluator


@R.calibration_targets.add('RAVEN_STREAMFLOW')
class RavenStreamflowTarget(StreamflowEvaluator):
    """Raven-specific streamflow evaluator (reads Hydrographs.csv -> m3/s)."""

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Find Raven's ``Hydrographs.csv`` (Raven writes it directly or under output/)."""
        sim_dir = Path(sim_dir)
        search_dirs = [sim_dir, sim_dir / 'output']
        candidates: List[Path] = []
        for d in search_dirs:
            if not d.exists():
                continue
            # Raven names it "<run_name>_Hydrographs.csv" or "Hydrographs.csv".
            candidates.extend(sorted(d.glob('*Hydrographs.csv')))
        if not candidates:
            # Fall back to a recursive search for robustness across RavenPy layouts.
            candidates = sorted(sim_dir.rglob('*Hydrographs.csv'))
        return [candidates[0]] if candidates else []

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract the simulated discharge (m3/s) from Hydrographs.csv.

        Raven's Hydrographs.csv has a leading date/time block then one column per
        subbasin gauge in m3/s (header like ``sub<ID> [m3/s]``). For the lumped
        outlet we take the gauge column with the highest mean discharge.
        """
        if not sim_files:
            return pd.Series(dtype=float)

        from ..postprocessor import read_raven_streamflow
        series = read_raven_streamflow(sim_files[0], self.config, self.logger)
        if series is None:
            return pd.Series(dtype=float)
        return series


__all__ = ['RavenStreamflowTarget']
