# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Calibration Targets.

Provides the streamflow, SWE (snow) and ET calibration targets for Raven output. Each
subclasses the centralized SYMFLUENCE evaluator for that variable and overrides only the
model-specific hooks (locating Raven's output file + extracting the series); the base
class handles obs loading, time alignment, spinup removal, resampling, and metric calc.
Registered via ``@R.calibration_targets.add('RAVEN_STREAMFLOW' | 'RAVEN_SWE' | 'RAVEN_ET')``
so the multi-objective framework (OBJECTIVE_WEIGHTS) can combine them.

SWE/ET come from Raven ``:CustomOutput`` CSVs (SNOW in mm = SWE; AET in mm/day), which the
emulator emits when a SWE/ET objective is configured (see emulator._build_custom_outputs).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from symfluence.core.registries import R
from symfluence.evaluation.evaluators import StreamflowEvaluator

try:  # Snow/ET evaluators are present in current SYMFLUENCE; guard for older cores.
    from symfluence.evaluation.evaluators.snow import SnowEvaluator
    from symfluence.evaluation.evaluators.et import ETEvaluator
    _SNOW_ET_AVAILABLE = True
except Exception:  # noqa: BLE001 -- older SYMFLUENCE without snow/ET evaluators
    SnowEvaluator = object  # type: ignore
    ETEvaluator = object  # type: ignore
    _SNOW_ET_AVAILABLE = False


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


if _SNOW_ET_AVAILABLE:

    @R.calibration_targets.add('RAVEN_SWE')
    class RavenSnowTarget(SnowEvaluator):
        """Raven SWE evaluator (reads the SNOW :CustomOutput CSV -> SWE in mm)."""

        def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
            super().__init__(config, project_dir, logger)

        def get_simulation_files(self, sim_dir: Path) -> List[Path]:
            from ..postprocessor import find_custom_output_file

            f = find_custom_output_file(Path(sim_dir), 'SNOW')
            return [f] if f is not None else []

        def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
            """Raven SNOW custom output is snow water equivalent in mm (== kg/m2 for SWE)."""
            if not sim_files:
                return pd.Series(dtype=float)
            from ..postprocessor import read_raven_custom_output

            series = read_raven_custom_output(sim_files[0], self.logger)
            return series if series is not None else pd.Series(dtype=float)

    @R.calibration_targets.add('RAVEN_ET')
    class RavenETTarget(ETEvaluator):
        """Raven ET evaluator (reads the AET :CustomOutput CSV -> ET in mm/day)."""

        def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
            super().__init__(config, project_dir, logger)

        def get_simulation_files(self, sim_dir: Path) -> List[Path]:
            from ..postprocessor import find_custom_output_file

            f = find_custom_output_file(Path(sim_dir), 'AET')
            return [f] if f is not None else []

        def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
            """Raven AET custom output is actual evapotranspiration in mm/day."""
            if not sim_files:
                return pd.Series(dtype=float)
            from ..postprocessor import read_raven_custom_output

            series = read_raven_custom_output(sim_files[0], self.logger)
            return series if series is not None else pd.Series(dtype=float)

    __all__ = ['RavenStreamflowTarget', 'RavenSnowTarget', 'RavenETTarget']
else:  # pragma: no cover - exercised only on older SYMFLUENCE cores
    __all__ = ['RavenStreamflowTarget']
