# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven parameter regionalization (transfer functions across subbasins).

Phase 4 of the Raven plugin. Instead of one global parameter set for the whole basin,
regionalization makes selected Raven parameters vary **per subbasin** as a function of
physical attributes (elevation, climate, area): ``param_i = a + b * attr_norm_i``. The
optimizer then calibrates the transfer-function coefficients (``a``, ``b``) — a handful of
numbers that generalise across the network — instead of one value per subbasin.

This wraps SYMFLUENCE's shared regionalization framework
(:mod:`symfluence.optimization.regionalization.strategies`): we build the per-subbasin
attribute table and hand it to ``RegionalizationFactory``, which returns a
``ParameterRegionalization`` whose ``get_calibration_parameters()`` gives coefficient
bounds to the optimizer and whose ``to_distributed()`` expands a coefficient set into the
per-subbasin parameter values. The Raven worker applies those values via
``SBGroupPropertyMultiplier`` (one subbasin group per subbasin) — see
:func:`raven_symfluence.emulator.build_and_run_emulator`.

Only the attribute table + parameter→attribute mapping are Raven-specific; the transfer
functions, normalization, and coefficient bookkeeping are the shared framework's.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default parameter -> attribute mapping for GR4JCN. Deliberately limited to parameters
# whose calibration range is strictly positive, so the per-subbasin SBGroupPropertyMultiplier
# (value_i / reference) stays positive and well-defined. Each entry maps a Raven parameter to
# a physical attribute and whether the slope coefficient b is calibrated.
#   GR4J_X1     (production store, mm)      ~ precipitation  (wetter -> larger store)
#   GR4J_X3     (routing store, mm)         ~ elevation      (steeper/higher -> routing change)
#   CEMANEIGE_X1(mean annual snow, mm)      ~ elevation      (higher -> more snow storage)
RAVEN_DEFAULT_PARAM_CONFIG: Dict[str, Dict[str, Dict[str, Any]]] = {
    'GR4JCN': {
        'GR4J_X1': {'attribute': 'precip_mm_yr', 'calibrate_b': True, 'fallback': 'elev_m'},
        'GR4J_X3': {'attribute': 'elev_m', 'calibrate_b': True},
        'CEMANEIGE_X1': {'attribute': 'elev_m', 'calibrate_b': True, 'b_sign': 'positive'},
    },
}

# Attributes always available from the distributed network geometry (no store needed).
_GEOMETRY_ATTRS = ('elev_m', 'area_km2', 'latitude', 'longitude')


def load_raven_subbasin_attributes(
    project_dir: Path,
    domain_name: str,
    network: Any,
    logger: logging.Logger,
) -> "Any":
    """Build a per-subbasin attribute DataFrame (one row per subbasin, network order).

    Always includes the geometry attributes carried by the distributed network
    (``elev_m``, ``area_km2``, ``latitude``, ``longitude``). When the model-ready
    attributes store has a ``climate`` group, its per-HRU scalars (e.g. ``precip_mm_yr``,
    ``aridity``, ``temp_C``) are joined on by subbasin id, so transfer functions can use
    climate gradients. Rows are ordered to match ``network.subbasins``.
    """
    import pandas as pd

    rows: List[Dict[str, float]] = []
    for s in network.subbasins:
        rows.append({
            'subbasin_id': s.subbasin_id,
            'elev_m': float(s.elevation),
            'area_km2': float(s.area_km2),
            'latitude': float(s.latitude),
            'longitude': float(s.longitude),
        })
    df = pd.DataFrame(rows)

    climate = _load_climate_attributes(project_dir, domain_name, network, logger)
    if climate is not None and not climate.empty:
        df = df.merge(climate, on='subbasin_id', how='left')
    # Constant columns break min-max normalization; drop all-equal non-geometry columns.
    for col in list(df.columns):
        if col in ('subbasin_id',) or col in _GEOMETRY_ATTRS:
            continue
        if df[col].nunique(dropna=True) <= 1:
            df = df.drop(columns=[col])
    logger.debug(f"Raven subbasin attributes: {len(df)} rows, columns {list(df.columns)}")
    return df


def _load_climate_attributes(project_dir: Path, domain_name: str, network: Any,
                             logger: logging.Logger):
    """Join the attributes-store ``climate`` group onto subbasins by id (or position)."""
    try:
        import numpy as np
        import pandas as pd
        from symfluence.data.model_ready.attributes_reader import open_canonical_attributes

        reader = open_canonical_attributes(project_dir, domain_name)
        if reader is None or not reader.has_group('climate'):
            return None
        with reader.group('climate') as ds:
            scalar_vars = [v for v in ds.data_vars
                           if ds[v].ndim == 1 and np.issubdtype(ds[v].dtype, np.number)]
            if not scalar_vars:
                return None
            hru_order = reader.hru_ids('climate') or reader.hru_ids('hru_identity') or []
            data = {v: [float(x) for x in np.atleast_1d(ds[v].values)] for v in scalar_vars}
        sub_ids = [str(s.subbasin_id) for s in network.subbasins]
        n = len(network.subbasins)
        out = {'subbasin_id': [s.subbasin_id for s in network.subbasins]}
        for v in scalar_vars:
            vals = data[v]
            # Strip the 'climate.' group prefix some stores use in variable names.
            col = v.split('.')[-1]
            # Skip climate columns that duplicate the authoritative network geometry
            # (e.g. the climate group's own 'elev_m') to avoid a merge-suffix collision.
            if col in _GEOMETRY_ATTRS:
                continue
            if hru_order and len(hru_order) == len(vals):
                lookup = dict(zip([str(h) for h in hru_order], vals))
                out[col] = [lookup.get(sid, float('nan')) for sid in sub_ids]
            elif len(vals) == n:
                out[col] = vals
        return pd.DataFrame(out) if len(out) > 1 else None
    except Exception as e:  # noqa: BLE001 -- climate attrs are optional
        logger.debug(f"No climate attributes joined ({e})")
        return None


def create_raven_regionalization(
    method: str,
    param_bounds: Dict[str, Dict[str, Any]],
    n_units: int,
    attributes: Any,
    model_template: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    """Build a ``ParameterRegionalization`` for Raven via the shared factory.

    Returns ``(strategy, param_config)``. ``param_config`` (the resolved parameter ->
    attribute mapping) is returned so the caller knows which parameters are regionalized.
    Falls back gracefully: any configured parameter whose attribute is absent from
    *attributes* is dropped (with a warning) rather than failing the run.
    """
    from symfluence.optimization.regionalization.strategies import RegionalizationFactory

    logger = logger or logging.getLogger(__name__)
    cfg = dict(config or {})

    param_config = cfg.get('RAVEN_REGIONALIZATION_PARAM_CONFIG') or cfg.get(
        'TRANSFER_FUNCTION_PARAM_CONFIG')
    if not param_config:
        param_config = RAVEN_DEFAULT_PARAM_CONFIG.get(str(model_template).upper(), {})

    available = set(getattr(attributes, 'columns', []))
    resolved: Dict[str, Dict[str, Any]] = {}
    for pname, pcfg in param_config.items():
        if pname not in param_bounds:
            logger.warning(f"Regionalized param '{pname}' has no bounds; skipping")
            continue
        attr = pcfg.get('attribute')
        fb = pcfg.get('fallback')
        chosen = attr if attr in available else (fb if fb in available else None)
        if chosen is None:
            logger.warning(
                f"Regionalized param '{pname}' attribute '{attr}' "
                f"(fallback '{fb}') not in attributes; skipping")
            continue
        resolved[pname] = {**pcfg, 'attribute': chosen}

    if not resolved:
        logger.warning("No regionalizable parameters resolved; falling back to lumped")
        return None, {}

    tuple_bounds = {p: (param_bounds[p]['min'], param_bounds[p]['max']) for p in resolved}
    cfg['TRANSFER_FUNCTION_PARAM_CONFIG'] = resolved
    strategy = RegionalizationFactory.create(
        method=method if method != 'lumped' else 'transfer_function',
        param_bounds=tuple_bounds,
        n_units=n_units,
        config=cfg,
        attributes=attributes,
        logger=logger,
    )
    logger.info(
        f"Raven regionalization ({method}): {len(resolved)} parameters "
        f"({', '.join(resolved)}) over {n_units} subbasins")
    return strategy, resolved


__all__ = [
    'RAVEN_DEFAULT_PARAM_CONFIG',
    'load_raven_subbasin_attributes',
    'create_raven_regionalization',
]
