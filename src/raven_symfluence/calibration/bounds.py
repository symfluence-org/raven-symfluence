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
# X1: production store capacity (mm); X2: inter-catchment exchange (mm/d, can be negative);
# X3: routing store capacity (mm); X4: unit-hydrograph time base (days);
# CN1: CemaNeige snow cold-content factor; CN2: CemaNeige degree-day melt factor.
# TODO(ravenpy): verify GR4JCN parameter ordering/names against the installed RavenPy
#   (ravenpy.config.emulators.GR4JCN expects an ordered 6-tuple GR4J_X1..X4 + CEMANEIGE_X1/X2).
GR4JCN_BOUNDS: Dict[str, Dict[str, Any]] = {
    'X1':  {'min': 10.0,  'max': 2000.0, 'transform': 'log'},     # production store capacity (mm)
    'X2':  {'min': -15.0, 'max': 10.0,   'transform': 'linear'},  # exchange coefficient (mm/d)
    'X3':  {'min': 1.0,   'max': 500.0,  'transform': 'log'},     # routing store capacity (mm)
    'X4':  {'min': 0.5,   'max': 15.0,   'transform': 'linear'},  # UH time base (days)
    'CN1': {'min': 0.0,   'max': 1.0,    'transform': 'linear'},  # CemaNeige cold-content factor
    'CN2': {'min': 1.0,   'max': 30.0,   'transform': 'linear'},  # CemaNeige melt factor (mm/degC/d)
}

# --- HBVEC (HBV-EC, 21 parameters) -- stub for a later phase -----------------------------------
# TODO(ravenpy): populate the full HBVEC bounds + parameter ordering against
#   ravenpy.config.emulators.HBVEC before enabling HBVEC calibration.
HBVEC_BOUNDS: Dict[str, Dict[str, Any]] = {}

# --- HMETS (21 parameters) -- stub for a later phase ------------------------------------------
# TODO(ravenpy): populate the full HMETS bounds + parameter ordering against
#   ravenpy.config.emulators.HMETS before enabling HMETS calibration.
HMETS_BOUNDS: Dict[str, Dict[str, Any]] = {}

# --- MOHYSE (10 parameters) -- stub for a later phase -----------------------------------------
# TODO(ravenpy): populate the full MOHYSE bounds + parameter ordering against
#   ravenpy.config.emulators.MOHYSE before enabling MOHYSE calibration.
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
