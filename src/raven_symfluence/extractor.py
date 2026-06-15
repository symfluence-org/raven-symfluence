# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Result Extractor.

Extracts simulated streamflow from Raven's ``Hydrographs.csv`` output. Mirrors
dRoute's result extractor: encapsulates the Raven-specific file patterns, column
naming, and outlet identification behind the :class:`ModelResultExtractor` contract.
Raven discharge is already in m3/s, so no unit conversion is needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from symfluence.models.base import ModelResultExtractor

logger = logging.getLogger(__name__)


class RavenResultExtractor(ModelResultExtractor):
    """Raven-specific result extraction (Hydrographs.csv -> outlet discharge series)."""

    def __init__(self):
        super().__init__('RAVEN')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """File patterns for Raven outputs."""
        return {
            'streamflow': [
                'RAVEN/*Hydrographs.csv',
                'RAVEN/output/*Hydrographs.csv',
                '*Hydrographs.csv',
                'output/*Hydrographs.csv',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Raven Hydrographs.csv has per-subbasin discharge columns (no fixed var name)."""
        return ['discharge'] if variable_type == 'streamflow' else [variable_type]

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract the outlet discharge time series from a Raven Hydrographs.csv.

        Args:
            output_file: Path to the Raven Hydrographs.csv.
            variable_type: Must be 'streamflow'.

        Returns:
            Time series of outlet discharge (m3/s).

        Raises:
            ValueError: If variable_type is unsupported or no discharge is found.
        """
        if variable_type != 'streamflow':
            raise ValueError(
                f"Raven extractor only supports 'streamflow', got '{variable_type}'")

        from .postprocessor import read_raven_streamflow

        series = read_raven_streamflow(output_file, {}, logger)
        if series is None:
            raise ValueError(f"No discharge data found in {output_file}")
        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Raven Hydrographs.csv discharge is already in m3/s."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """Raven outlet is the highest-mean-discharge subbasin column."""
        return 'outlet_selection'


__all__ = ['RavenResultExtractor']
