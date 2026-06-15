# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration parameter bounds owned by the raven-symfluence package.

Following the JAX/dRoute plugin pattern, the package owns its own parameter
bounds rather than relying on the SYMFLUENCE shared bounds registry.

Bounds are keyed by Raven emulator structure (``model_template``). Each entry is
``{'min': float, 'max': float, 'transform': 'linear'|'log'}``.

Phase 1 fully specifies GR4JCN; HBVEC/HMETS/MOHYSE are provided as stubs with the
canonical RavenPy parameter ordering so later phases can extend them in place.
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

# --- HBVEC (HBV-EC, 21 parameters) -- stub for a later phase -----------------------------------
# Parameter ordering verified against ravenpy.config.emulators.hbvec.P (RavenPy 0.21):
#   X01..X21 (21 ordered slots). Bounds intentionally unpopulated: GR4JCN is the only
#   Phase 1 template wired through the runner; fill these in when enabling HBVEC calibration.
HBVEC_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 22)]
HBVEC_BOUNDS: Dict[str, Dict[str, Any]] = {}

# --- HMETS (21 parameters) -- stub for a later phase ------------------------------------------
# Parameter ordering verified against ravenpy.config.emulators.hmets.P (RavenPy 0.21).
# Bounds intentionally unpopulated (Phase 1 = GR4JCN only).
HMETS_PARAM_ORDER = [
    'GAMMA_SHAPE', 'GAMMA_SCALE', 'GAMMA_SHAPE2', 'GAMMA_SCALE2', 'MIN_MELT_FACTOR',
    'MAX_MELT_FACTOR', 'DD_MELT_TEMP', 'DD_AGGRADATION', 'SNOW_SWI_MIN', 'SNOW_SWI_MAX',
    'SWI_REDUCT_COEFF', 'DD_REFREEZE_TEMP', 'REFREEZE_FACTOR', 'REFREEZE_EXP',
    'PET_CORRECTION', 'HMETS_RUNOFF_COEFF', 'PERC_COEFF', 'BASEFLOW_COEFF_1',
    'BASEFLOW_COEFF_2', 'TOPSOIL', 'PHREATIC',
]
HMETS_BOUNDS: Dict[str, Dict[str, Any]] = {}

# --- MOHYSE (10 parameters) -- stub for a later phase -----------------------------------------
# Parameter ordering verified against ravenpy.config.emulators.mohyse.P (RavenPy 0.21):
#   X01..X10. Bounds intentionally unpopulated (Phase 1 = GR4JCN only).
MOHYSE_PARAM_ORDER = [f'X{i:02d}' for i in range(1, 11)]
MOHYSE_BOUNDS: Dict[str, Dict[str, Any]] = {}

# Registry of per-template bounds, keyed by the canonical RAVEN_MODEL_TEMPLATE value.
RAVEN_TEMPLATE_BOUNDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'GR4JCN': GR4JCN_BOUNDS,
    'HBVEC':  HBVEC_BOUNDS,
    'HMETS':  HMETS_BOUNDS,
    'MOHYSE': MOHYSE_BOUNDS,
}


def get_raven_bounds(model_template: str = 'GR4JCN') -> Dict[str, Dict[str, Any]]:
    """Return the package-owned calibration bounds for a Raven emulator structure.

    Args:
        model_template: Raven emulator name (``GR4JCN``, ``HBVEC``, ``HMETS``, ``MOHYSE``).

    Returns:
        ``{param: {'min', 'max', 'transform'}}`` (a deep-ish copy so callers can
        mutate freely). Unknown/unsupported templates fall back to GR4JCN so a
        parameter manager always has a usable, non-empty bound set.
    """
    template = str(model_template or 'GR4JCN').upper()
    bounds = RAVEN_TEMPLATE_BOUNDS.get(template) or RAVEN_TEMPLATE_BOUNDS['GR4JCN']
    return {k: dict(v) for k, v in bounds.items()}
