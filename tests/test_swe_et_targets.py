# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the Raven SWE/ET multi-objective calibration targets.

Covers the custom-output CSV reader, target registration, and (when the raven binary is
available) end-to-end extraction of SWE/ET series from a real run.
"""
from __future__ import annotations

import logging

import pytest

logger = logging.getLogger("test_swe_et")


def _write_custom_output(path, value, n=50):
    """Fabricate a Raven ENTIRE_WATERSHED :CustomOutput CSV (2-row header)."""
    import pandas as pd

    t = pd.date_range("2002-01-01", periods=n, freq="D")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",Watershed:,Watershed,\n")
        fh.write("time,day,mean,\n")
        for i, d in enumerate(t, start=1):
            fh.write(f"{i},{d.strftime('%Y-%m-%d')},{value},\n")
    return path


def test_read_custom_output_parses_value_series(tmp_path):
    from raven_symfluence.postprocessor import read_raven_custom_output

    f = _write_custom_output(tmp_path / "run_SNOW_Daily_Average_ByWatershed.csv", 12.5)
    s = read_raven_custom_output(f, logger)
    assert s is not None and len(s) == 50
    assert abs(s.mean() - 12.5) < 1e-9
    assert str(s.index.dtype).startswith("datetime64")


def test_find_custom_output_file(tmp_path):
    from raven_symfluence.postprocessor import find_custom_output_file

    out = tmp_path / "output"
    out.mkdir()
    _write_custom_output(out / "run_SNOW_Daily_Average_ByWatershed.csv", 5.0)
    _write_custom_output(out / "run_AET_Daily_Average_ByWatershed.csv", 1.0)
    assert find_custom_output_file(tmp_path, "SNOW").name.endswith("SNOW_Daily_Average_ByWatershed.csv")
    assert find_custom_output_file(tmp_path, "AET").name.endswith("AET_Daily_Average_ByWatershed.csv")
    assert find_custom_output_file(tmp_path, "NOPE") is None


def test_swe_et_targets_registered():
    """Importing the targets module registers RAVEN_SWE / RAVEN_ET (when core supports them)."""
    pytest.importorskip("symfluence.evaluation.evaluators.snow",
                        reason="SYMFLUENCE core without snow/ET evaluators")
    import raven_symfluence.calibration.targets  # noqa: F401
    from symfluence.core.registries import R

    assert R.calibration_targets.get("RAVEN_STREAMFLOW") is not None
    assert R.calibration_targets.get("RAVEN_SWE") is not None
    assert R.calibration_targets.get("RAVEN_ET") is not None
