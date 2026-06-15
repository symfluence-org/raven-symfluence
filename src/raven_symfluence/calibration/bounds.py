# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration parameter bounds owned by the raven-symfluence package.

Following the JAX/dRoute plugin pattern, the package owns its own parameter
bounds rather than relying on the SYMFLUENCE shared bounds registry.

Bounds are keyed by Raven emulator structure (``model_template``). Each entry is
``{'min': float, 'max': float, 'transform': 'linear'|'log'}``.

All eight hydrologic emulators are fully specified: GR4JCN, HBVEC, HMETS, MOHYSE,
Blended, CanadianShield, HYPR, SACSMA. Parameter ordering is verified against each
emulator's ``params`` dataclass in RavenPy 0.21; bounds are the canonical RavenPy
calibration ranges from the official calibration notebook (RavenPy ``docs/notebooks/
06_Raven_calibration.ipynb``, the "List of Model-Boundaries" section).
"""
from __future__ import annotations

from typing import Any, Dict

# --- GR4JCN (CemaNeige GR4J) -------------------------------------------------------------------
# Parameter names/order match ravenpy.config.emulators.gr4jcn.P exactly (verified against
# RavenPy 0.21): GR4J_X1, GR4J_X2, GR4J_X3, GR4J_X4, CEMANEIGE_X1, CEMANEIGE_X2.
#   GR4J_X1: production store capacity (mm); GR4J_X2: inter-catchment exchange (mm/d, signed);
#   GR4J_X3: routing store capacity (mm); GR4J_X4: unit-hydrograph time base (days);
#   CEMANEIGE_X1: CemaNeige average annual snow (mm); CEMANEIGE_X2: CemaNeige melt coefficient.
GR4JCN_BOUNDS: Dict[str, Dict[str, Any]] = {
    'GR4J_X1':      {'min': 0.01,  'max': 2.5,    'transform': 'linear'},  # production store (mm*1000)
    'GR4J_X2':      {'min': -15.0, 'max': 10.0,   'transform': 'linear'},  # exchange coefficient (mm/d)
    'GR4J_X3':      {'min': 10.0,  'max': 700.0,  'transform': 'linear'},  # routing store capacity (mm)
    'GR4J_X4':      {'min': 0.5,   'max': 7.0,    'transform': 'linear'},  # UH time base (days)
    'CEMANEIGE_X1': {'min': 0.0,   'max': 1000.0, 'transform': 'linear'},  # avg annual snow (mm)
    'CEMANEIGE_X2': {'min': 0.0,   'max': 1.0,    'transform': 'linear'},  # melt coefficient
}

def _zip_bounds(order, low, high, transform: str = 'linear') -> Dict[str, Dict[str, Any]]:
    """Build a {name: {min, max, transform}} dict from ordered low/high tuples."""
    assert len(order) == len(low) == len(high), 'param order / low / high length mismatch'
    return {name: {'min': float(lo), 'max': float(hi), 'transform': transform}
            for name, lo, hi in zip(order, low, high)}


# --- HBVEC (HBV-EC, 21 parameters) ------------------------------------------------------------
# Order matches ravenpy.config.emulators.hbvec.P (X01..X21). Bounds are the canonical RavenPy
# calibration ranges (RavenPy docs, notebook 06 — Calibration). Param ordering verified vs P.
HBVEC_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 22)]
HBVEC_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    HBVEC_PARAM_ORDER,
    (-3.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.01, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.05, 0.8, 0.8),
    (3.0, 8.0, 8.0, 0.1, 1.0, 1.0, 7.0, 100.0, 1.0, 0.1, 6.0, 5.0, 5.0, 0.2, 1.0, 30.0, 3.0, 2.0, 1.0, 1.5, 1.5),
)

# --- HMETS (21 parameters) --------------------------------------------------------------------
# Order matches ravenpy.config.emulators.hmets.P. Bounds = canonical RavenPy calibration ranges
# (RavenPy docs, notebook 06; Martel et al. 2017). TOPSOIL/PHREATIC are in metres (RavenPy units).
HMETS_PARAM_ORDER = [
    'GAMMA_SHAPE', 'GAMMA_SCALE', 'GAMMA_SHAPE2', 'GAMMA_SCALE2', 'MIN_MELT_FACTOR',
    'MAX_MELT_FACTOR', 'DD_MELT_TEMP', 'DD_AGGRADATION', 'SNOW_SWI_MIN', 'SNOW_SWI_MAX',
    'SWI_REDUCT_COEFF', 'DD_REFREEZE_TEMP', 'REFREEZE_FACTOR', 'REFREEZE_EXP',
    'PET_CORRECTION', 'HMETS_RUNOFF_COEFF', 'PERC_COEFF', 'BASEFLOW_COEFF_1',
    'BASEFLOW_COEFF_2', 'TOPSOIL', 'PHREATIC',
]
HMETS_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    HMETS_PARAM_ORDER,
    (0.3, 0.01, 0.5, 0.15, 0.0, 0.0, -2.0, 0.01, 0.0, 0.01, 0.005, -5.0, 0.0, 0.0, 0.0, 0.0, 0.00001, 0.0, 0.00001, 0.0, 0.0),
    (20.0, 5.0, 13.0, 1.5, 20.0, 20.0, 3.0, 0.2, 0.1, 0.3, 0.1, 2.0, 5.0, 1.0, 3.0, 1.0, 0.02, 0.1, 0.01, 0.5, 2.0),
)

# --- MOHYSE (10 parameters) -------------------------------------------------------------------
# Order matches ravenpy.config.emulators.mohyse.P (X01..X10). Bounds = canonical RavenPy ranges.
MOHYSE_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 11)]
MOHYSE_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    MOHYSE_PARAM_ORDER,
    (0.01, 0.01, 0.01, -5.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    (20.0, 1.0, 20.0, 5.0, 0.5, 1.0, 1.0, 1.0, 15.0, 15.0),
)

# --- Blended (43 parameters) ------------------------------------------------------------------
# Order matches ravenpy.config.emulators.blended.P (X01..X35 then R01..R08); verified vs the
# params dataclass. Bounds = canonical RavenPy calibration ranges (notebook 06; Mai et al. 2020,
# "blended model" structure-and-parameter calibration).
BLENDED_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 36)] + [f'R{i:02d}' for i in range(1, 9)]
BLENDED_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    BLENDED_PARAM_ORDER,
    (0.0, 0.1, 0.5, -5.0, 0.0, 0.5, 5.0, 0.0, 0.0, 0.0, -5.0, 0.5, 0.0, 0.01, 0.005, -5.0, 0.0,
     0.0, 0.0, 0.3, 0.01, 0.5, 0.15, 1.5, 0.0, -1.0, 0.01, 0.00001, 0.0, 0.0, -3.0, 0.5, 0.8, 0.8,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 3.0, 3.0, -1.0, 100.0, 2.0, 10.0, 3.0, 0.05, 0.45, -2.0, 2.0, 0.1, 0.3, 0.1, 2.0, 1.0,
     5.0, 0.4, 20.0, 5.0, 13.0, 1.5, 3.0, 5.0, 1.0, 0.2, 0.02, 0.5, 2.0, 3.0, 4.0, 1.2, 1.2,
     0.02, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)

# --- CanadianShield (34 parameters) -----------------------------------------------------------
# Order matches ravenpy.config.emulators.canadianshield.P (X01..X34); verified vs the params
# dataclass. Bounds = canonical RavenPy calibration ranges (notebook 06).
CANADIANSHIELD_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 35)]
CANADIANSHIELD_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    CANADIANSHIELD_PARAM_ORDER,
    (0.01, 0.01, 0.01, 0.0, 0.0, 0.05, 0.0, -5.0, -5.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01,
     0.005, -3.0, 0.5, 5.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0),
    (0.5, 2.0, 3.0, 3.0, 0.05, 0.45, 7.0, -1.0, -1.0, 2.0, 2.0, 100.0, 100.0, 100.0, 0.4, 0.1,
     0.3, 0.1, 3.0, 4.0, 500.0, 5.0, 5.0, 1.0, 8.0, 20.0, 1.5, 0.2, 0.2, 10.0, 10.0, 1.2, 1.2, 1.0),
)

# --- HYPR (21 parameters) ---------------------------------------------------------------------
# Order matches ravenpy.config.emulators.hypr.P (X01..X21); verified vs the params dataclass.
# Bounds = canonical RavenPy calibration ranges (notebook 06).
HYPR_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 22)]
HYPR_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    HYPR_PARAM_ORDER,
    (-1.0, -3.0, 0.0, 0.3, -1.3, -2.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
     1.5, 0.0, 0.0, 0.8),
    (1.0, 3.0, 0.8, 1.0, 0.3, 0.0, 30.0, 0.8, 2.0, 100.0, 0.5, 5.0, 1.0, 1000.0, 6.0, 7.0, 8.0,
     3.0, 5.0, 5.0, 1.2),
)

# --- SACSMA (21 parameters) -------------------------------------------------------------------
# Order matches ravenpy.config.emulators.sacsma.P (X01..X21); verified vs the params dataclass.
# Bounds = canonical RavenPy calibration ranges (notebook 06). Several SAC-SMA parameters are
# calibrated in log10 space (X01..X03 lows/highs are negative powers-of-ten exponents); these
# bounds are reproduced verbatim from the RavenPy notebook.
SACSMA_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 22)]
SACSMA_BOUNDS: Dict[str, Dict[str, Any]] = _zip_bounds(
    SACSMA_PARAM_ORDER,
    (-3.0, -1.52287874, -0.69897, 0.025, 0.01, 0.075, 0.015, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.3, 0.01, 0.8, 0.8),
    (-1.82390874, -0.69897, -0.30102999, 0.125, 0.075, 0.3, 0.3, 0.6, 0.5, 3.0, 80.0, 0.8, 0.05,
     0.2, 0.1, 0.4, 8.0, 20.0, 5.0, 1.2, 1.2),
)

# Registry of per-template bounds, keyed by the canonical RAVEN_MODEL_TEMPLATE value.
RAVEN_TEMPLATE_BOUNDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'GR4JCN': GR4JCN_BOUNDS,
    'HBVEC':  HBVEC_BOUNDS,
    'HMETS':  HMETS_BOUNDS,
    'MOHYSE': MOHYSE_BOUNDS,
    'BLENDED': BLENDED_BOUNDS,
    'CANADIANSHIELD': CANADIANSHIELD_BOUNDS,
    'HYPR': HYPR_BOUNDS,
    'SACSMA': SACSMA_BOUNDS,
}


def get_raven_bounds(model_template: str = 'GR4JCN') -> Dict[str, Dict[str, Any]]:
    """Return the package-owned calibration bounds for a Raven emulator structure.

    Args:
        model_template: Raven emulator name (``GR4JCN``, ``HBVEC``, ``HMETS``, ``MOHYSE``,
            ``BLENDED``, ``CANADIANSHIELD``, ``HYPR``, ``SACSMA``).

    Returns:
        ``{param: {'min', 'max', 'transform'}}`` (a deep-ish copy so callers can
        mutate freely). Unknown/unsupported templates fall back to GR4JCN so a
        parameter manager always has a usable, non-empty bound set.
    """
    template = str(model_template or 'GR4JCN').upper()
    bounds = RAVEN_TEMPLATE_BOUNDS.get(template) or RAVEN_TEMPLATE_BOUNDS['GR4JCN']
    return {k: dict(v) for k, v in bounds.items()}
