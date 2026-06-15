# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared test fixtures for the Raven plugin (synthetic distributed domain).

Builds a self-contained SYMFLUENCE domain on disk — a model-ready attributes store with a
multi-subbasin ``topology`` group, canonical CFIF forcing, and a preprocessed observation
series — so the distributed Raven path can be exercised without any real data acquisition.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr


def build_distributed_attributes_store(
    project_dir: Path,
    domain_name: str,
    ids: List[str],
    downstream: List[str],
    *,
    areas_m2: Optional[List[float]] = None,
    elevations: Optional[List[float]] = None,
    river_length_m: Optional[List[float]] = None,
    river_slope: Optional[List[float]] = None,
) -> Path:
    """Write a model-ready attributes store with hru_identity/terrain/topology groups.

    ``ids``/``downstream`` define the routing graph (one HRU per subbasin). Returns the
    store path: ``{project_dir}/data/model_ready/attributes/{domain_name}_attributes.nc``.
    """
    n = len(ids)
    areas_m2 = areas_m2 or [1.0e7] * n
    elevations = elevations or [1300.0 + 100.0 * i for i in range(n)]
    river_length_m = river_length_m or [5000.0] * n
    river_slope = river_slope or [0.01] * n

    adir = project_dir / 'data' / 'model_ready' / 'attributes'
    adir.mkdir(parents=True, exist_ok=True)
    store = adir / f'{domain_name}_attributes.nc'

    id_arr = np.array(ids, dtype=object)
    xr.Dataset({
        'hru_id': (('hru',), id_arr),
        'hru_area': (('hru',), np.array(areas_m2, dtype='float64')),
        'latitude': (('hru',), np.array([51.0 + 0.1 * i for i in range(n)], dtype='float64')),
        'longitude': (('hru',), np.array([-115.0 - 0.1 * i for i in range(n)], dtype='float64')),
    }).to_netcdf(store, group='hru_identity', mode='w')
    xr.Dataset({
        'hru_id': (('hru',), id_arr),
        'elev_mean': (('hru',), np.array(elevations, dtype='float64')),
    }).to_netcdf(store, group='terrain', mode='a')
    xr.Dataset({
        'gru_id': (('gru',), id_arr),
        'downstream_id': (('gru',), np.array(downstream, dtype=object)),
        'gru_area': (('gru',), np.array(areas_m2, dtype='float64')),
        'river_length': (('gru',), np.array(river_length_m, dtype='float64')),
        'river_slope': (('gru',), np.array(river_slope, dtype='float64')),
    }).to_netcdf(store, group='topology', mode='a')
    return store


def build_delineation_shapefiles(
    project_dir: Path,
    links: List[int],
    downstream: List[int],
    *,
    length_m: Optional[List[float]] = None,
    slope: Optional[List[float]] = None,
    contrib_area_m2: Optional[List[float]] = None,
    gru_area_m2: Optional[List[float]] = None,
) -> None:
    """Write TauDEM-style river_network + river_basins shapefiles (1 GRU per segment).

    Mirrors a SYMFLUENCE delineated domain: river_network carries LINKNO/DSLINKNO/Length/
    Slope/DSContArea; river_basins carries GRU_ID/GRU_area/gru_to_seg (GRU id == segment).
    A DSLINKNO with no matching LINKNO marks the outlet.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    n = len(links)
    length_m = length_m or [5000.0] * n
    slope = slope or [0.01] * n
    contrib_area_m2 = contrib_area_m2 or [(i + 1) * 1e7 for i in range(n)]
    gru_area_m2 = gru_area_m2 or [1e7] * n

    rn = gpd.GeoDataFrame({
        'LINKNO': links,
        'DSLINKNO': downstream,
        'Length': length_m,
        'Slope': slope,
        'DSContArea': contrib_area_m2,
        'geometry': [LineString([(i, 0), (i + 1, 0)]) for i in range(n)],
    }, crs='EPSG:4326')
    rb = gpd.GeoDataFrame({
        'GRU_ID': links,
        'GRU_area': gru_area_m2,
        'gru_to_seg': links,
        'geometry': [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(n)],
    }, crs='EPSG:4326')

    rn_dir = project_dir / 'shapefiles' / 'river_network'
    rb_dir = project_dir / 'shapefiles' / 'river_basins'
    rn_dir.mkdir(parents=True, exist_ok=True)
    rb_dir.mkdir(parents=True, exist_ok=True)
    rn.to_file(rn_dir / 'river_network.shp')
    rb.to_file(rb_dir / 'river_basins.shp')


def write_canonical_forcing(
    project_dir: Path,
    domain_name: str,
    start: str = '2002-01-01',
    n_days: int = 730,
) -> Path:
    """Write a basin-mean CFIF forcing file (precipitation_flux, air_temperature)."""
    fdir = project_dir / 'data' / 'model_ready' / 'forcings'
    fdir.mkdir(parents=True, exist_ok=True)
    t = pd.date_range(start, periods=n_days, freq='D')
    rng = np.arange(n_days)
    # precipitation_flux kg m-2 s-1 (~2 mm/d mean), air_temperature K (seasonal).
    precip_mm_d = (2.0 + 2.0 * np.sin(rng / 30.0)).clip(0)
    precip_flux = precip_mm_d / 86400.0
    temp_k = 278.15 + 12.0 * np.sin((rng - 80) / 58.0)
    out = fdir / f'{domain_name}_forcing.nc'
    xr.Dataset(
        {
            'precipitation_flux': (('time',), precip_flux.astype('float64'),
                                   {'units': 'kg m-2 s-1', 'standard_name': 'precipitation_flux'}),
            'air_temperature': (('time',), temp_k.astype('float64'),
                                {'units': 'K', 'standard_name': 'air_temperature'}),
        },
        coords={'time': t},
        attrs={'timestep_seconds': 86400},
    ).to_netcdf(out)
    return out


def write_observations(
    project_dir: Path,
    domain_name: str,
    start: str = '2002-01-01',
    n_days: int = 730,
) -> Path:
    """Write a preprocessed daily streamflow observation CSV (m3/s)."""
    obs_dir = project_dir / 'data' / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True, exist_ok=True)
    t = pd.date_range(start, periods=n_days, freq='D')
    rng = np.arange(n_days)
    q = (3.0 + 2.0 * np.sin((rng - 100) / 58.0)).clip(0.1)
    out = obs_dir / f'{domain_name}_streamflow_processed.csv'
    pd.DataFrame({'discharge_cms': q}, index=t).to_csv(out)
    return out


def write_multigauge_obs(
    obs_dir: Path,
    gauge_ids: List[int],
    start: str = '2002-01-01',
    n_days: int = 730,
) -> Path:
    """Write per-gauge LaMAH-style ID_<id>.csv files (semicolon, YYYY;MM;DD;qobs)."""
    obs_dir.mkdir(parents=True, exist_ok=True)
    t = pd.date_range(start, periods=n_days, freq='D')
    rng = np.arange(n_days)
    for k, gid in enumerate(gauge_ids):
        q = (1.0 + 0.5 * k + 1.5 * np.sin((rng - 100) / 58.0)).clip(0.05)
        df = pd.DataFrame({
            'YYYY': t.year, 'MM': t.month, 'DD': t.day, 'qobs': np.round(q, 4),
        })
        df.to_csv(obs_dir / f'ID_{gid}.csv', sep=';', index=False)
    return obs_dir
