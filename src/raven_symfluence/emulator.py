# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RavenPy emulator construction + execution (the single RavenPy integration point).

Everything that touches RavenPy lives here so the rest of the plugin can import
cleanly without RavenPy installed. RavenPy is imported lazily inside the functions
(the SYMFLUENCE heavy-dependency convention), so importing this module never pulls
in RavenPy at module-parse time.

The plugin feeds Raven from SYMFLUENCE's model-ready datastore:
- canonical CFIF forcing (precipitation_flux kg m-2 s-1, air_temperature K, ...)
  via :func:`symfluence.data.model_ready.forcing_reader.open_canonical_forcing`
- per-HRU catchment attributes (hru_area, elev_mean, latitude/longitude, topology)
  via :func:`symfluence.data.model_ready.attributes_reader.open_canonical_attributes`,
  falling back to the catchment shapefile when the attributes store is absent.

RavenPy builds an emulator config (e.g. ``ravenpy.config.emulators.GR4JCN``) from the
forcing + HRU attributes + parameters, writes the ``.rvi/.rvp/.rvh/.rvt/.rvc`` files,
and drives the ``raven`` binary; the result is ``Hydrographs.csv`` in *output_dir*.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Positional parameter order for GR4JCN, matching ravenpy.config.emulators.gr4jcn.P
# exactly (verified against RavenPy 0.21): GR4J_X1..GR4J_X4 then the two CemaNeige snow
# parameters CEMANEIGE_X1 (avg annual snow, mm) and CEMANEIGE_X2 (melt coefficient).
GR4JCN_PARAM_ORDER: List[str] = [
    'GR4J_X1', 'GR4J_X2', 'GR4J_X3', 'GR4J_X4', 'CEMANEIGE_X1', 'CEMANEIGE_X2',
]


# Per-template emulator metadata (verified against RavenPy 0.21 emulator classes).
#
# Each Raven emulator class (ravenpy.config.emulators.{GR4JCN,HBVEC,HMETS,Mohyse}) already
# carries its own soil/vegetation/land-use classes, hydrologic processes, soil model, and PET
# method. The plugin therefore only has to (a) name the right class, (b) order the parameter
# vector, and (c) supply the *minimal extra inputs* each emulator's options imply for a single
# lumped HRU driven by basin-mean PRECIP + TEMP_AVE:
#
#   class_name          : attribute name on ravenpy.config.emulators (Mohyse is not all-caps).
#   rain_snow_fraction  : override for the emulator's RainSnowFraction option, or None to keep
#                         the emulator default. HMETS and MOHYSE default to RAINSNOW_DATA, which
#                         makes Raven *require* a separate snow time series; overriding to the
#                         temperature-based RAINSNOW_DINGMAN (the same split GR4JCN uses) lets
#                         Raven do the rain/snow partition internally from PRECIP + TEMP_AVE, so
#                         no extra forcing is needed.
#   needs_monthly_aves  : HBVEC/HYPR use PET_FROMMONTHLY, so each gauge must carry
#                         :MonthlyAveTemperature and :MonthlyAveEvaporation (climatologies). We
#                         attach a generic Northern-Hemisphere climatology to the gauge; this is
#                         the canonical minimal construction (RavenPy HBV/HYPR examples).
#   evaporation         : override for the emulator's Evaporation (PET) option, or None to keep
#                         the emulator default. CanadianShield defaults to PET_HARGREAVES_1985,
#                         which makes Raven *require* a daily temperature range (TEMP_MIN/TEMP_MAX).
#                         Overriding to PET_OUDIN (temperature + latitude, the same PET GR4JCN and
#                         Blended use) lets PRECIP + TEMP_AVE alone suffice. ``ow_evaporation`` is
#                         True when the open-water PET option (OW_Evaporation) must match.
#   hru_kind            : how the lumped HRU(s) are built. 'land' (default) -> one generic land
#                         HRU. 'canadianshield' -> CanadianShield is validated to require exactly
#                         two HRUs (one organic SOILP_ORG, one bedrock SOILP_BEDROCK); we build the
#                         emulator's OrganicHRU + BedRockHRU, splitting the catchment area equally.
#
# Forcing: every spec is driven by PRECIP + TEMP_AVE only. No template needs TEMP_MIN/TEMP_MAX,
# an explicit PET series, or a separate snow series under these option overrides (the rain/snow
# split and PET are computed internally from temperature, and SACSMA's default RAINSNOW_DATA is
# overridden to the temperature-based RAINSNOW_DINGMAN just like HMETS/MOHYSE).
RAVEN_TEMPLATE_SPECS: Dict[str, Dict[str, Any]] = {
    'GR4JCN': {'class_name': 'GR4JCN', 'rain_snow_fraction': None, 'needs_monthly_aves': False,
               'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'HBVEC':  {'class_name': 'HBVEC',  'rain_snow_fraction': None, 'needs_monthly_aves': True,
               'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'HMETS':  {'class_name': 'HMETS',  'rain_snow_fraction': 'RAINSNOW_DINGMAN', 'needs_monthly_aves': False,
               'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'MOHYSE': {'class_name': 'Mohyse', 'rain_snow_fraction': 'RAINSNOW_DINGMAN', 'needs_monthly_aves': False,
               'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'BLENDED': {'class_name': 'Blended', 'rain_snow_fraction': None, 'needs_monthly_aves': False,
                'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'CANADIANSHIELD': {'class_name': 'CanadianShield', 'rain_snow_fraction': None,
                       'needs_monthly_aves': False, 'evaporation': 'PET_OUDIN',
                       'ow_evaporation': True, 'hru_kind': 'canadianshield'},
    'HYPR':   {'class_name': 'HYPR',   'rain_snow_fraction': None, 'needs_monthly_aves': True,
               'evaporation': None, 'ow_evaporation': False, 'hru_kind': 'land'},
    'SACSMA': {'class_name': 'SACSMA', 'rain_snow_fraction': 'RAINSNOW_DINGMAN',
               'needs_monthly_aves': False, 'evaporation': None, 'ow_evaporation': False,
               'hru_kind': 'land'},
}

# Generic Northern-Hemisphere monthly climatology (Jan..Dec) for the PET_FROMMONTHLY PET method
# used by HBVEC and HYPR. Temperatures in degC, potential evaporation in mm/month. These are
# gauge-level climatological inputs Raven requires for the from-monthly PET method; they are
# deliberately generic (the gauge's actual sub-daily forcing still drives the water balance).
MONTHLY_AVE_TEMPERATURE: List[float] = [-10.0, -8.0, -2.0, 5.0, 10.0, 15.0,
                                        18.0, 16.0, 11.0, 5.0, -2.0, -8.0]
MONTHLY_AVE_EVAPORATION: List[float] = [10.0, 15.0, 30.0, 55.0, 85.0, 100.0,
                                        110.0, 95.0, 60.0, 35.0, 15.0, 8.0]

# Backwards-compatible aliases (the constants were HBVEC-specific in earlier phases).
HBVEC_MONTHLY_AVE_TEMPERATURE = MONTHLY_AVE_TEMPERATURE
HBVEC_MONTHLY_AVE_EVAPORATION = MONTHLY_AVE_EVAPORATION


def _resolve_template_spec(model_template: str) -> Dict[str, Any]:
    """Return the emulator spec for a template, defaulting to GR4JCN for unknown names."""
    return RAVEN_TEMPLATE_SPECS.get(str(model_template or 'GR4JCN').upper(),
                                    RAVEN_TEMPLATE_SPECS['GR4JCN'])


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """Flat-dict-first config accessor (workers/preprocessors pass flat dicts)."""
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, 'get'):
        return config.get(key, default)
    return default


# =============================================================================
# Data assembly (model-ready store -> RavenPy inputs)
# =============================================================================

def load_forcing(forcing_basin_path: Path, logger: logging.Logger):
    """Open the canonical CFIF forcing dataset for Raven.

    Returns an xarray Dataset under canonical CF names (precipitation_flux,
    air_temperature, ...) with ``ds.attrs['timestep_seconds']`` set, or None if no
    forcing files are present.
    """
    from symfluence.data.model_ready.forcing_reader import open_canonical_forcing

    files = sorted(Path(forcing_basin_path).glob('*.nc'))
    if not files:
        logger.error(f"No forcing NetCDF files found in {forcing_basin_path}")
        return None
    return open_canonical_forcing(files)


def build_raven_forcing_frame(ds, logger: logging.Logger) -> pd.DataFrame:
    """Convert a canonical CFIF forcing Dataset to a daily Raven forcing DataFrame.

    Columns produced (Raven-native units):
        PRECIP    : mm/d   (from precipitation_flux kg m-2 s-1)
        TEMP_AVE  : degC   (from air_temperature K)

    Areal/basin-averaged forcing is assumed (lumped GR4JCN). Sub-daily forcing is
    aggregated to daily (sum for precip rate -> depth, mean for temperature).
    """
    from symfluence.data.preprocessing.cfif.units import UnitConverter

    converter = UnitConverter()

    # Reduce any spatial dimension to a basin mean (lumped).
    def _basin_mean(var):
        spatial = [d for d in var.dims if d != 'time']
        return var.mean(dim=spatial) if spatial else var

    precip_flux = _basin_mean(ds['precipitation_flux'])  # kg m-2 s-1
    temp_k = _basin_mean(ds['air_temperature'])          # K

    precip_series = precip_flux.to_pandas()
    temp_series = temp_k.to_pandas()

    # CFIF -> Raven units: precip kg m-2 s-1 -> mm/d; temp K -> degC.
    precip_mm_d = converter.convert('precipitation', precip_series, 'kg m-2 s-1', 'mm/day')
    temp_c = converter.convert('temperature', temp_series, 'K', 'degC')

    df = pd.DataFrame({'PRECIP': precip_mm_d, 'TEMP_AVE': temp_c})
    df.index = pd.to_datetime(df.index)

    # Daily aggregation: precip is a depth/day rate (mean preserves the daily total
    # once it is already daily; sub-daily depth-rate samples average to the daily rate),
    # temperature averages.
    daily = df.resample('D').agg({'PRECIP': 'mean', 'TEMP_AVE': 'mean'})
    logger.debug(f"Built Raven daily forcing frame: {len(daily)} days")
    return daily


def resolve_hru_attributes(
    project_dir: Path,
    domain_name: str,
    catchment_shapefile: Optional[Path],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Resolve the single (lumped) HRU's attributes for the Raven emulator.

    Prefers the model-ready attributes store (per-HRU hru_area, elev_mean,
    latitude/longitude); falls back to the catchment shapefile when the store is
    absent (the "open returns None => use legacy" contract).

    Returns a dict with keys: area_km2, elevation, latitude, longitude.
    """
    from symfluence.data.model_ready.attributes_reader import open_canonical_attributes

    attrs = open_canonical_attributes(project_dir, domain_name)
    if attrs is not None:
        try:
            # Areas are stored per-HRU in m2 under the terrain group; sum for the
            # lumped catchment, average elevation/centroid.
            area_m2 = attrs.variable('terrain', 'hru_area', reduce='mean')
            elev = attrs.find_variable('terrain', ['elev_mean', 'elevation'], reduce='mean')
            lat = attrs.find_variable('hru_identity', ['latitude', 'lat'], reduce='mean')
            lon = attrs.find_variable('hru_identity', ['longitude', 'lon'], reduce='mean')
            if area_m2 is not None:
                logger.debug("Resolved Raven HRU attributes from model-ready store")
                return {
                    'area_km2': float(area_m2) / 1e6,
                    'elevation': float(elev) if elev is not None else 1000.0,
                    'latitude': float(lat) if lat is not None else 0.0,
                    'longitude': float(lon) if lon is not None else 0.0,
                }
        except Exception as e:  # noqa: BLE001 -- absent/partial store => shapefile fallback
            logger.debug(f"Attributes store incomplete ({e}); falling back to shapefile")

    return _hru_attributes_from_shapefile(catchment_shapefile, logger)


def _hru_attributes_from_shapefile(
    catchment_shapefile: Optional[Path], logger: logging.Logger
) -> Dict[str, Any]:
    """Derive the lumped HRU attributes from the catchment shapefile."""
    defaults = {'area_km2': 1.0, 'elevation': 1000.0, 'latitude': 0.0, 'longitude': 0.0}
    if not catchment_shapefile or not Path(catchment_shapefile).exists():
        logger.warning("No catchment shapefile available; using default HRU attributes")
        return defaults
    try:
        import geopandas as gpd

        gdf = gpd.read_file(catchment_shapefile)
        if gdf.crs and gdf.crs.is_geographic:
            centroid = gdf.geometry.unary_union.centroid
            defaults['latitude'] = float(centroid.y)
            defaults['longitude'] = float(centroid.x)
            area_m2 = gdf.to_crs(gdf.estimate_utm_crs()).geometry.area.sum()
        else:
            centroid = gdf.to_crs(4326).geometry.unary_union.centroid
            defaults['latitude'] = float(centroid.y)
            defaults['longitude'] = float(centroid.x)
            area_m2 = gdf.geometry.area.sum()
        defaults['area_km2'] = float(area_m2) / 1e6
        # Optional elevation column.
        for col in ('elev_mean', 'elevation', 'Elevation', 'ELEV_MEAN'):
            if col in gdf.columns:
                defaults['elevation'] = float(gdf[col].mean())
                break
        logger.debug(f"Resolved Raven HRU attributes from shapefile: area={defaults['area_km2']:.2f} km2")
    except Exception as e:  # noqa: BLE001 -- model execution resilience
        logger.warning(f"Could not read catchment shapefile ({e}); using defaults")
    return defaults


# =============================================================================
# Parameter vector
# =============================================================================

def params_to_vector(model_template: str, params: Dict[str, float],
                     logger: logging.Logger) -> List[float]:
    """Order a parameter dict into the RavenPy positional parameter vector.

    Missing parameters fall back to the bounds midpoint so a partial calibration
    set (e.g. calibrating only X1..X4) still produces a runnable model.
    """
    from .calibration.bounds import get_raven_bounds

    template = str(model_template or 'GR4JCN').upper()
    bounds = get_raven_bounds(template)
    # The bounds dict for each template is declared in the exact positional order of that
    # emulator's ``params`` dataclass (verified against RavenPy 0.21: GR4JCN GR4J_X1..CEMANEIGE_X2,
    # HBVEC X01..X21, HMETS named, MOHYSE X01..X10, BLENDED X01..X35+R01..R08, CANADIANSHIELD
    # X01..X34, HYPR X01..X21, SACSMA X01..X21), so the bounds key order *is* the RavenPy
    # parameter-vector order. GR4JCN keeps its explicit constant for clarity / the fidelity test.
    order = GR4JCN_PARAM_ORDER if template == 'GR4JCN' else list(bounds.keys())

    vector: List[float] = []
    for name in order:
        if name in params:
            vector.append(float(params[name]))
        elif name in bounds:
            b = bounds[name]
            vector.append(float((b['min'] + b['max']) / 2.0))
        else:
            vector.append(0.0)
    logger.debug(f"Raven {template} parameter vector: {vector}")
    return vector


# =============================================================================
# RavenPy build + run (lazy import)
# =============================================================================

def build_and_run_emulator(
    config: Any,
    settings_dir: Path,
    output_dir: Path,
    params: Dict[str, float],
    logger: logging.Logger,
    forcing_basin_path: Optional[Path] = None,
    catchment_shapefile: Optional[Path] = None,
) -> bool:
    """Build a Raven emulator from forcing + HRU attributes + params and run it.

    Writes the ``.rvi/.rvp/.rvh/.rvt/.rvc`` files into *settings_dir* and the model
    outputs (incl. ``Hydrographs.csv``) into *output_dir*. Returns True on success.

    RavenPy is imported lazily so this module imports without it installed.
    """
    # Lazy, isolated import — never at module top level. Importing ravenpy itself
    # requires the raven binary to be discoverable; point RavenPy at the resolved
    # binary (env var) before the import so it works regardless of PATH.
    _ensure_raven_binary_env(config, logger)
    try:
        import ravenpy  # noqa: F401
        from ravenpy import Emulator
        from ravenpy.config import commands as rc
        from ravenpy.config import emulators
    except ImportError as e:
        logger.error(f"RavenPy is not installed; cannot run Raven: {e}")
        return False
    except RuntimeError as e:
        # ravenpy raises RuntimeError at import if the raven binary is missing.
        logger.error(f"RavenPy could not locate the raven binary: {e}")
        return False

    domain_name = _cfg(config, 'DOMAIN_NAME', 'unknown')
    data_dir = Path(_cfg(config, 'SYMFLUENCE_DATA_DIR', '.'))
    project_dir = data_dir / f"domain_{domain_name}"
    model_template = str(_cfg(config, 'RAVEN_MODEL_TEMPLATE', 'GR4JCN')).upper()
    run_name = str(_cfg(config, 'RAVEN_RUN_NAME', 'raven_run'))

    if forcing_basin_path is None:
        forcing_basin_path = _resolve_forcing_basin_path(project_dir)
    if catchment_shapefile is None:
        catchment_shapefile = _resolve_catchment_shapefile(project_dir)

    ds = load_forcing(forcing_basin_path, logger)
    if ds is None:
        return False

    try:
        forcing_df = build_raven_forcing_frame(ds, logger)
        hru = resolve_hru_attributes(project_dir, domain_name, catchment_shapefile, logger)
        start, end = _resolve_time_window(config, forcing_df)
        param_vector = params_to_vector(model_template, params, logger)

        settings_dir = Path(settings_dir)
        output_dir = Path(output_dir)
        settings_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Persist the daily forcing to a CF-compliant NetCDF that RavenPy's
        # Gauge.from_nc can introspect (CF names pr/tas, a station dimension, and
        # lat/lon/elevation station coordinates).
        forcing_nc = settings_dir / f"{domain_name}_raven_forcing.nc"
        _write_forcing_netcdf(forcing_df, hru, forcing_nc)

        spec = _resolve_template_spec(model_template)
        emulator_cls = _resolve_emulator_class(emulators, model_template, logger)
        if emulator_cls is None:
            return False

        # HRU(s) for the lumped catchment. Most emulators take a single generic land HRU;
        # CanadianShield is validated to require exactly two (organic + bedrock), so it gets a
        # dedicated builder that splits the catchment area between its two HRU subclasses.
        hru_list = _build_hrus(emulators, hru, spec, logger)
        gauge = _build_gauge(rc, forcing_nc, hru, spec, logger)

        # GR4JCN defaults to NetCDF output (WriteNetcdfFormat=True); force CSV so the
        # postprocessor's Hydrographs.csv reader (read_raven_streamflow) finds its input.
        # Templates whose default rain/snow split reads a separate snow series (HMETS/MOHYSE/
        # SACSMA use RAINSNOW_DATA) are overridden to a temperature-based split, and templates
        # whose default PET needs a temperature range (CanadianShield -> HARGREAVES_1985) are
        # overridden to PET_OUDIN, so PRECIP + TEMP_AVE alone suffice (see RAVEN_TEMPLATE_SPECS).
        config_kwargs: Dict[str, Any] = {
            'params': param_vector,
            'HRUs': hru_list,
            'Gauge': [gauge],
            'StartDate': _to_datetime(start),
            'EndDate': _to_datetime(end),
            'RunName': run_name,
            'WriteNetcdfFormat': False,
        }
        if spec['rain_snow_fraction']:
            config_kwargs['RainSnowFraction'] = spec['rain_snow_fraction']
        # PET override (e.g. CanadianShield: HARGREAVES_1985 -> OUDIN so no TEMP_MIN/MAX needed).
        if spec.get('evaporation'):
            config_kwargs['Evaporation'] = spec['evaporation']
            if spec.get('ow_evaporation'):
                config_kwargs['OW_Evaporation'] = spec['evaporation']
        config_model = emulator_cls(**config_kwargs)

        emulator = Emulator(config=config_model, workdir=str(output_dir), overwrite=True)
        # run() has its own overwrite flag (defaults to False); without it RavenPy
        # refuses to re-run into a workdir that already holds outputs — which every
        # calibration iteration does, since the worker reuses one sim_dir per process.
        output = emulator.run(overwrite=True)
        logger.info(
            f"Raven {model_template} run completed -> {output.path} "
            f"(hydrograph: {output.files.get('hydrograph')})")
        return True
    except Exception as e:  # noqa: BLE001 -- model execution resilience
        logger.error(f"RavenPy emulator build/run failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    finally:
        try:
            ds.close()
        except Exception:  # noqa: BLE001 -- best-effort close
            pass


def _resolve_emulator_class(emulators, model_template: str, logger: logging.Logger):
    """Look up the RavenPy emulator class for a template.

    Maps the SYMFLUENCE template name (always upper-case, e.g. ``MOHYSE``) to RavenPy's
    actual class attribute (``Mohyse``/``Blended``/``CanadianShield`` are not all-caps) via
    :data:`RAVEN_TEMPLATE_SPECS`. Supports all eight hydrologic emulators: GR4JCN, HBVEC, HMETS,
    MOHYSE, BLENDED, CANADIANSHIELD, HYPR, SACSMA.
    """
    spec = RAVEN_TEMPLATE_SPECS.get(str(model_template or '').upper())
    class_name = spec['class_name'] if spec else str(model_template)
    cls = getattr(emulators, class_name, None)
    if cls is None:
        logger.error(
            f"RavenPy has no emulator '{class_name}' for template '{model_template}'. "
            f"Supported templates: {', '.join(sorted(RAVEN_TEMPLATE_SPECS))}.")
    return cls


def _build_hrus(emulators, hru: Dict[str, Any], spec: Dict[str, Any],
                logger: logging.Logger) -> List[Any]:
    """Build the HRU list for an emulator from the resolved lumped catchment attributes.

    Most emulators take a single generic land HRU (``hru_type='land'`` lets RavenPy pick the
    Land/Forest HRU subclass). CanadianShield is validated by RavenPy to require *exactly two*
    HRUs — one organic (``SOILP_ORG``) and one bedrock (``SOILP_BEDROCK``) — so for it we build
    the emulator module's ``OrganicHRU`` + ``BedRockHRU`` and split the catchment area equally
    between them (the two HRUs share the lumped centroid/elevation).
    """
    if spec.get('hru_kind') == 'canadianshield':
        from ravenpy.config.emulators.canadianshield import BedRockHRU, OrganicHRU

        half_area = float(hru['area_km2']) / 2.0
        common = {
            'area': half_area,
            'elevation': float(hru['elevation']),
            'latitude': float(hru['latitude']),
            'longitude': float(hru['longitude']),
        }
        logger.debug("Built CanadianShield organic + bedrock HRUs (area split equally)")
        return [OrganicHRU(**common), BedRockHRU(**common)]

    return [{
        'area': hru['area_km2'],
        'elevation': hru['elevation'],
        'latitude': hru['latitude'],
        'longitude': hru['longitude'],
        'hru_type': 'land',
    }]


def _build_gauge(rc, forcing_nc: Path, hru: Dict[str, Any],
                 spec: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
    """Build a RavenPy ``Gauge`` from the written forcing NetCDF.

    Uses :meth:`ravenpy.config.commands.Gauge.from_nc` (the canonical RavenPy 0.21
    recipe): the NetCDF carries CF-named variables (``pr`` for PRECIP, ``tas`` for
    TEMP_AVE) plus station lat/lon/elevation coordinates, so RavenPy can emit the
    ``:ReadFromNetCDF`` block for each forcing. ``station_idx`` is 1-based.

    For templates that need them (``spec['needs_monthly_aves']`` -> HBVEC, which uses
    PET_FROMMONTHLY), the gauge is given monthly average temperature/evaporation
    climatologies; Raven errors out without them.

    ``spec`` defaults to the GR4JCN spec (no monthly aves). A logger passed positionally in
    the legacy ``_build_gauge(rc, nc, hru, logger)`` form is tolerated for backward
    compatibility.
    """
    # Backward-compat: legacy callers passed the logger as the 4th positional argument.
    if isinstance(spec, logging.Logger):
        logger, spec = spec, None
    if spec is None:
        spec = RAVEN_TEMPLATE_SPECS['GR4JCN']
    if logger is None:
        logger = logging.getLogger(__name__)

    gauge = rc.Gauge.from_nc(
        str(forcing_nc),
        data_type=['PRECIP', 'TEMP_AVE'],
        station_idx=1,
        # Elevation is supplied explicitly so the gauge always has a vertical datum
        # even when the NetCDF elevation coordinate is not picked up via CF.
        data_kwds={'ALL': {'elevation': float(hru['elevation'])}},
    )
    if spec.get('needs_monthly_aves'):
        gauge.monthly_ave_temperature = list(MONTHLY_AVE_TEMPERATURE)
        gauge.monthly_ave_evaporation = list(MONTHLY_AVE_EVAPORATION)
        logger.debug("Attached monthly ave temperature/evaporation climatology to gauge")
    logger.debug(
        f"Built Raven gauge from {forcing_nc.name}: "
        f"lat={gauge.latitude}, lon={gauge.longitude}, elev={gauge.elevation}")
    return gauge


def _write_forcing_netcdf(forcing_df: pd.DataFrame, hru: Dict[str, Any], out: Path) -> None:
    """Write the daily Raven forcing frame to a CF-compliant NetCDF for ``Gauge.from_nc``.

    RavenPy's ``Gauge.from_nc`` introspects the file via cf_xarray, so the layout must be
    CF-discoverable:
      - variables named ``pr`` (precipitation_flux) and ``tas`` (air_temperature) with a
        ``(time, station)`` shape — the station dimension is required for ``station_idx``;
      - station coordinates ``lat``/``lon``/``elevation`` with CF standard names;
      - the time coordinate is written with an explicit ``days since`` float64 encoding,
        which Raven's NetCDF reader requires (a string/native datetime encoding triggers
        "Attempt to convert between text & numbers").

    Units are carried through to Raven: precip mm/d, temperature degC.
    """
    import xarray as xr

    time = pd.to_datetime(forcing_df.index)
    precip = forcing_df['PRECIP'].astype(np.float32).values[:, None]
    temp = forcing_df['TEMP_AVE'].astype(np.float32).values[:, None]
    lat = float(hru['latitude'])
    lon = float(hru['longitude'])
    elev = float(hru['elevation'])

    ds = xr.Dataset(
        {
            'pr': (('time', 'station'), precip,
                   {'units': 'mm/d', 'standard_name': 'precipitation_flux',
                    'long_name': 'daily precipitation'}),
            'tas': (('time', 'station'), temp,
                    {'units': 'degC', 'standard_name': 'air_temperature',
                     'long_name': 'daily mean air temperature'}),
        },
        coords={
            'time': time,
            'lat': (('station',), [lat],
                    {'units': 'degrees_north', 'standard_name': 'latitude'}),
            'lon': (('station',), [lon],
                    {'units': 'degrees_east', 'standard_name': 'longitude'}),
            'elevation': (('station',), [elev],
                          {'units': 'm', 'standard_name': 'height',
                           'positive': 'up', 'axis': 'Z'}),
        },
    )
    encoding = {
        'time': {
            'units': 'days since 1900-01-01 00:00:00',
            'calendar': 'proleptic_gregorian',
            'dtype': 'float64',
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, engine='netcdf4', encoding=encoding)
    ds.close()


def _to_datetime(value: Any):
    """Coerce a date/datetime/str window bound to a ``datetime.datetime`` for RavenPy."""
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    return pd.to_datetime(value).to_pydatetime()


def _ensure_raven_binary_env(config: Any, logger: logging.Logger) -> None:
    """Make the raven binary discoverable to RavenPy via RAVENPY_RAVEN_BINARY_PATH.

    RavenPy resolves the engine from ``RAVENPY_RAVEN_BINARY_PATH`` (else PATH) *at import
    time*, so this must run before ``import ravenpy``. If the env var is already set (e.g.
    by :class:`RavenRunner`) it is left untouched; otherwise we fall back to a config-
    supplied path or ``shutil.which('raven')``.
    """
    import os
    import shutil

    if os.environ.get('RAVENPY_RAVEN_BINARY_PATH'):
        return
    candidate = _cfg(config, 'RAVEN_INSTALL_PATH') or _cfg(config, 'RAVEN_EXE')
    resolved = None
    if candidate:
        p = Path(candidate)
        if p.is_dir():
            p = p / 'raven'
        if p.exists():
            resolved = str(p)
    if resolved is None:
        resolved = shutil.which('raven')
    if resolved:
        os.environ['RAVENPY_RAVEN_BINARY_PATH'] = resolved
        logger.debug(f"Set RAVENPY_RAVEN_BINARY_PATH={resolved}")
    else:
        logger.debug("No raven binary resolved; relying on RavenPy's own discovery")


def _resolve_time_window(config: Any, forcing_df: pd.DataFrame) -> Tuple[Any, Any]:
    """Resolve the (start, end) simulation window from config, clipped to the forcing."""
    start = _cfg(config, 'EXPERIMENT_TIME_START') or _cfg(config, 'TIME_START')
    end = _cfg(config, 'EXPERIMENT_TIME_END') or _cfg(config, 'TIME_END')
    f_start = forcing_df.index.min()
    f_end = forcing_df.index.max()
    start = pd.to_datetime(start).date() if start and str(start) != 'default' else f_start.date()
    end = pd.to_datetime(end).date() if end and str(end) != 'default' else f_end.date()
    return start, end


def _resolve_forcing_basin_path(project_dir: Path) -> Path:
    """Prefer the model-ready forcings store; fall back to basin_averaged_data."""
    model_ready = project_dir / 'data' / 'model_ready' / 'forcings'
    if model_ready.exists() and any(model_ready.glob('*.nc')):
        return model_ready
    return project_dir / 'forcing' / 'basin_averaged_data'


def _resolve_catchment_shapefile(project_dir: Path) -> Optional[Path]:
    """Best-effort lookup of a catchment shapefile for HRU attribute fallback."""
    for sub in ('catchment', 'river_basins'):
        d = project_dir / 'shapefiles' / sub
        if d.exists():
            shps = sorted(d.rglob('*.shp'))
            if shps:
                return shps[0]
    return None


__all__ = [
    'build_and_run_emulator',
    'load_forcing',
    'build_raven_forcing_frame',
    'resolve_hru_attributes',
    'params_to_vector',
    'GR4JCN_PARAM_ORDER',
    'RAVEN_TEMPLATE_SPECS',
    'MONTHLY_AVE_TEMPERATURE',
    'MONTHLY_AVE_EVAPORATION',
]
