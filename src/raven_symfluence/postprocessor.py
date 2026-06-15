# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven model postprocessor.

Parses Raven's ``Hydrographs.csv`` into standardized streamflow (m3/s). Subclasses
:class:`StandardModelPostProcessor`; the custom CSV layout (a date/time block then
one column per subbasin gauge in m3/s) is handled by overriding ``extract_streamflow``
and shared helpers (also reused by the runner, worker, and calibration target).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from symfluence.models.base import StandardModelPostProcessor

# Raven Hydrographs.csv non-discharge columns (the leading date/time block).
_NON_FLOW_COLUMNS = {'time', 'date', 'hour', 'precip', 'precip [mm/day]'}

# Raven names each gauged subbasin column like "sub_3 [m3/s]" or "sub3 [m3/s]"; capture the id.
_SUBBASIN_COL_RE = re.compile(r'sub_?(\d+)\b')


def find_hydrographs_file(sim_dir: Path) -> Optional[Path]:
    """Locate Raven's ``Hydrographs.csv`` (RavenPy may nest it under output/)."""
    sim_dir = Path(sim_dir)
    for d in (sim_dir, sim_dir / 'output'):
        if not d.exists():
            continue
        matches = sorted(d.glob('*Hydrographs.csv'))
        if matches:
            return matches[0]
    matches = sorted(sim_dir.rglob('*Hydrographs.csv'))
    return matches[0] if matches else None


def read_raven_streamflow(
    hydrographs_file: Path,
    config: Any,
    logger: logging.Logger,
) -> Optional[pd.Series]:
    """Read the outlet discharge (m3/s) from a Raven Hydrographs.csv.

    Raven's Hydrographs.csv columns: a leading date/time block, then one column per
    subbasin gauge labelled like ``sub<ID> [m3/s]``. For a lumped/outlet series we
    select the gauge column with the highest mean discharge (matches the outlet-by-max
    convention used across SYMFLUENCE evaluators).
    """
    try:
        df = pd.read_csv(hydrographs_file, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]

        # Build the datetime index from Raven's date (+ optional hour) columns.
        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
        if date_col is None:
            logger.error(f"No 'date' column in {hydrographs_file}")
            return None
        hour_col = next((c for c in df.columns if c.lower() == 'hour'), None)
        if hour_col is not None:
            idx = pd.to_datetime(
                df[date_col].astype(str) + ' ' + df[hour_col].astype(str),
                errors='coerce')
        else:
            idx = pd.to_datetime(df[date_col], errors='coerce')
        df.index = idx

        # Candidate discharge columns: numeric, not part of the date/precip block.
        flow_cols = []
        for c in df.columns:
            cl = c.lower()
            if cl in _NON_FLOW_COLUMNS or cl in (date_col.lower(),):
                continue
            if hour_col is not None and cl == hour_col.lower():
                continue
            if 'precip' in cl:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                flow_cols.append(c)

        if not flow_cols:
            logger.error(f"No discharge columns found in {hydrographs_file}")
            return None

        # Prefer an observed-aware outlet column ([m3/s]) with the highest mean flow.
        means = df[flow_cols].mean(numeric_only=True)
        outlet = str(means.idxmax())
        series = df[outlet].astype(float)
        series = series[series.index.notna()]
        logger.debug(f"Read Raven streamflow from column '{outlet}' ({len(series)} steps)")
        return series
    except Exception as e:  # noqa: BLE001 -- model execution resilience
        logger.error(f"Error reading Raven Hydrographs.csv: {e}")
        return None


def _hydrographs_datetime_index(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    """Build a datetime index from a Raven Hydrographs.csv frame's date(+hour) columns."""
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col is None:
        return None
    hour_col = next((c for c in df.columns if c.lower() == 'hour'), None)
    if hour_col is not None:
        return pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[hour_col].astype(str), errors='coerce')
    return pd.to_datetime(df[date_col], errors='coerce')


def read_raven_per_reach(
    hydrographs_file: Path,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Read every gauged subbasin's discharge (m3/s) from a Raven Hydrographs.csv.

    Returns a DataFrame indexed by datetime whose integer columns are subbasin ids
    (parsed from the ``sub<ID> [m3/s]`` headers) — the per-reach routed flow the
    multi-gauge objective needs. Returns None if no subbasin columns are present.
    """
    try:
        df = pd.read_csv(hydrographs_file, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        idx = _hydrographs_datetime_index(df)
        if idx is None:
            logger.error(f"No 'date' column in {hydrographs_file}")
            return None
        df.index = idx

        cols: dict = {}
        for c in df.columns:
            if 'precip' in c.lower() or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            m = _SUBBASIN_COL_RE.search(c.lower())
            if m:
                cols[int(m.group(1))] = df[c].astype(float)
        if not cols:
            logger.debug(f"No 'sub<ID>' reach columns in {hydrographs_file}")
            return None
        out = pd.DataFrame(cols)
        out = out[out.index.notna()]
        logger.debug(f"Read {len(out.columns)} reach hydrographs from {hydrographs_file.name}")
        return out
    except Exception as e:  # noqa: BLE001 -- model execution resilience
        logger.error(f"Error reading per-reach Raven Hydrographs.csv: {e}")
        return None


def write_routed_netcdf(
    per_reach: pd.DataFrame,
    out_path: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Write per-reach flow to a mizuRoute-style NetCDF for ``MultiGaugeMetrics``.

    SYMFLUENCE's :class:`MultiGaugeMetrics` reads routed flow from a NetCDF with a ``seg``
    dimension, a ``segId`` variable, and an ``IRFroutedRunoff(time, seg)`` variable in
    m3/s. We emit exactly that layout from Raven's per-reach columns so the same
    multi-gauge objective serves Raven, SUMMA/mizuRoute, and FUSE without special-casing.
    """
    try:
        import numpy as np
        import xarray as xr

        seg_ids = list(per_reach.columns)
        data = per_reach[seg_ids].to_numpy(dtype='float64')  # (time, seg)
        ds = xr.Dataset(
            {
                'IRFroutedRunoff': (('time', 'seg'), data, {'units': 'm3/s'}),
                'segId': (('seg',), np.asarray(seg_ids, dtype='int64')),
            },
            coords={'time': pd.to_datetime(per_reach.index)},
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path)
        ds.close()
        logger.debug(f"Wrote routed NetCDF ({len(seg_ids)} reaches) -> {out_path.name}")
        return out_path
    except Exception as e:  # noqa: BLE001 -- model execution resilience
        logger.error(f"Error writing routed NetCDF: {e}")
        return None


class RavenPostProcessor(StandardModelPostProcessor):
    """Postprocessor for the Raven hydrological modelling framework."""

    model_name = "RAVEN"
    output_file_pattern = "Hydrographs.csv"
    streamflow_variable = "discharge"
    streamflow_unit = "cms"  # Raven Hydrographs.csv is already in m3/s

    def _get_model_name(self) -> str:
        return "RAVEN"

    def _setup_model_specific_paths(self) -> None:
        """Raven outputs to the standard experiment simulation directory."""
        self.raven_output_dir = (
            self.project_dir / 'simulations' / self.experiment_id / 'RAVEN')

    def _get_output_dir(self) -> Path:
        return self.project_dir / 'simulations' / self.experiment_id / 'RAVEN'

    def extract_streamflow(self) -> Optional[Path]:
        """Extract outlet streamflow (m3/s) from Raven's Hydrographs.csv."""
        self.logger.info("Extracting streamflow from Raven output")

        output_dir = self._get_output_dir()
        hydro_file = find_hydrographs_file(output_dir)
        if hydro_file is None:
            self.logger.error(f"Raven Hydrographs.csv not found in {output_dir}")
            return None

        streamflow = read_raven_streamflow(hydro_file, self.config, self.logger)
        if streamflow is None or streamflow.empty:
            return None

        # Already m3/s; apply resampling if configured (e.g. hourly -> daily).
        if self.resample_frequency:
            streamflow = streamflow.resample(self.resample_frequency).mean()

        return self.save_streamflow_to_results(
            streamflow, model_column_name='RAVEN_discharge_cms')


__all__ = [
    'RavenPostProcessor',
    'find_hydrographs_file',
    'read_raven_streamflow',
    'read_raven_per_reach',
    'write_routed_netcdf',
]
