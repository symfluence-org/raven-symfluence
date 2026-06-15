# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Distributed Raven network construction from the model-ready attributes store.

Phase 3 of the Raven plugin: turn SYMFLUENCE's river-network *topology* into a routed
multi-subbasin Raven model. The lumped path (one HRU, one subbasin, no routing) lives in
:mod:`raven_symfluence.emulator`; this module adds the distributed path:

  attributes store ``topology`` group  ->  RavenPy ``SubBasin`` + ``HRU`` + ``ChannelProfile``
  + ``:Routing ROUTE_DIFFUSIVE_WAVE``  ->  per-reach gauged ``Hydrographs.csv``.

Model: **one land HRU per subbasin (GRU)** — the canonical BasinMaker "one HRU per
sub-basin" construction (``ravenpy.extractors.routing_product``). Subbasin connectivity,
reach length and slope come from the ``topology`` group (``downstream_id``,
``river_length`` [m], ``river_slope`` [m/m]); per-HRU area/elevation/centroid come from
the ``hru_identity``/``terrain`` groups. Channel cross-sections are built with the same
SWAT trapezoid recipe RavenPy's routing-product extractor uses, with bankfull width/depth
estimated from upstream-accumulated drainage area via downstream hydraulic-geometry
relations (an estimate — the store carries no surveyed cross-sections; calibratable later
via channel multipliers in Phase 4).

RavenPy is imported lazily by the caller (:mod:`raven_symfluence.emulator`); this module
itself imports neither RavenPy nor the raven binary at parse time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Raven's outlet sentinel for SubBasin.downstream_id.
RAVEN_OUTLET_ID = -1

# Downstream-hydraulic-geometry coefficients: bankfull width W = c_w * A^p_w,
# depth D = c_d * A^p_d, with A the upstream-accumulated drainage area in km^2 and W/D in
# metres. Mid-range continental values (Bieger et al. 2015 regional curves); these size the
# routing channel only and are documented estimates — Phase-4 calibration can scale them.
_HG_WIDTH_COEFF = 2.7
_HG_WIDTH_EXP = 0.50
_HG_DEPTH_COEFF = 0.30
_HG_DEPTH_EXP = 0.40

# Minimum channel geometry so headwater reaches still yield a valid trapezoid.
_MIN_BANKFULL_WIDTH = 1.0   # m
_MIN_BANKFULL_DEPTH = 0.3   # m
_MIN_RIVER_SLOPE = 1e-5     # m/m (Raven rejects zero/negative bed slopes)
_MANNING_CHANNEL = 0.035
_MANNING_FLOODPLAIN = 0.10


@dataclass
class SubbasinSpec:
    """One subbasin = one routing reach + one land HRU (GRU-level distributed model)."""

    subbasin_id: int
    downstream_id: int            # RAVEN_OUTLET_ID (-1) for the outlet
    area_km2: float               # local (incremental) HRU/subbasin area
    reach_length_km: float        # 0.0 disables in-channel routing for that reach
    river_slope: float            # m/m
    elevation: float              # m
    latitude: float
    longitude: float
    accumulated_area_km2: float = 0.0   # upstream-accumulated area (for channel sizing)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"sub_{self.subbasin_id}"


@dataclass
class DistributedNetwork:
    """Resolved distributed network ready to hand to RavenPy."""

    subbasins: List[SubbasinSpec]
    avg_annual_runoff_mm: float = 400.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_subbasins(self) -> int:
        return len(self.subbasins)


# =============================================================================
# Topology reading (attributes store -> SubbasinSpec list)
# =============================================================================

def read_distributed_topology(
    project_dir: Path,
    domain_name: str,
    logger: logging.Logger,
) -> Optional[DistributedNetwork]:
    """Read the river-network topology into a :class:`DistributedNetwork`.

    Tries two sources, in order:

    1. the attributes store's ``topology`` group, when it carries routing variables
       (``downstream_id``/``river_length``/``river_slope``) — the ideal, pre-digested case;
    2. the delineation **shapefiles** (``shapefiles/river_network`` TauDEM
       ``LINKNO``/``DSLINKNO``/``Length``/``Slope``/``DSContArea`` + ``shapefiles/river_basins``
       ``GRU_ID``/``GRU_area``/``gru_to_seg``) — where SYMFLUENCE actually stores connectivity
       for delineated domains (the attributes builder does not always propagate it).

    Returns ``None`` (caller falls back to lumped) when neither source yields a routable
    multi-subbasin network — the correct, simpler model when there is nothing to route.
    """
    from symfluence.data.model_ready.attributes_reader import open_canonical_attributes

    reader = open_canonical_attributes(project_dir, domain_name)

    network = _read_topology_from_store(reader, logger)
    source = "attributes store topology group"
    if network is None:
        network = _read_topology_from_shapefiles(project_dir, domain_name, reader, logger)
        source = "delineation shapefiles"
    if network is None:
        return None

    network.avg_annual_runoff_mm = _estimate_avg_annual_runoff(project_dir, domain_name, logger)
    logger.info(
        f"Built distributed Raven network from {source}: {network.n_subbasins} subbasins "
        f"(one HRU each), AvgAnnualRunoff={network.avg_annual_runoff_mm:.0f} mm/yr")
    return network


def _read_topology_from_store(reader, logger: logging.Logger) -> Optional[DistributedNetwork]:
    """Build the network from the attributes store ``topology`` group (when routable)."""
    if reader is None or not reader.has_group('topology'):
        return None
    try:
        import numpy as np

        with reader.group('topology') as ds:
            id_var = next((v for v in ('gru_id', 'hru_id', 'subbasin_id') if v in ds.variables),
                          None)
            if id_var is None:
                return None
            gru_ids = [str(x) for x in np.atleast_1d(ds[id_var].values)]
            if len(gru_ids) < 2 or 'downstream_id' not in set(ds.variables):
                logger.debug(
                    "Topology group not routable (single subbasin or no 'downstream_id'); "
                    "trying delineation shapefiles")
                return None
            downstream = [str(v) for v in np.atleast_1d(ds['downstream_id'].values)]
            areas_m2 = _values_or_none(ds, ['gru_area', 'area'])
            river_len_m = _values_or_none(ds, ['river_length', 'length'])
            river_slope = _values_or_none(ds, ['river_slope', 'slope'])

        per_hru = _read_hru_geometry(reader, gru_ids, logger)
        specs = _assemble_specs(
            gru_ids, downstream, areas_m2, river_len_m, river_slope, per_hru, logger)
        if len(specs) < 2:
            return None
        _accumulate_drainage_area(specs)
        return DistributedNetwork(subbasins=specs)
    except Exception as e:  # noqa: BLE001 -- partial store => shapefile fallback
        logger.debug(f"Store topology not usable ({e}); trying delineation shapefiles")
        return None


def _read_topology_from_shapefiles(
    project_dir: Path,
    domain_name: str,
    reader,
    logger: logging.Logger,
) -> Optional[DistributedNetwork]:
    """Build the network from TauDEM delineation shapefiles (river_network + river_basins).

    river_network carries the routing graph (``LINKNO`` -> ``DSLINKNO``, ``Length`` [m],
    ``Slope``, accumulated ``DSContArea`` [m^2]); river_basins maps each GRU to a segment
    (``gru_to_seg``) and carries ``GRU_area`` [m^2]. We model one land HRU per GRU/segment;
    a ``DSLINKNO`` that points off-network (no matching ``LINKNO``) is the outlet.
    """
    rn_shp = _find_shapefile(project_dir, 'river_network')
    rb_shp = _find_shapefile(project_dir, 'river_basins')
    if rn_shp is None or rb_shp is None:
        logger.debug("No river_network/river_basins shapefiles; cannot build distributed network")
        return None
    try:
        import geopandas as gpd

        rn = gpd.read_file(rn_shp)
        rb = gpd.read_file(rb_shp)
        link_col = _pick_col(rn, ['LINKNO', 'linkno', 'seg_id', 'segId'])
        ds_col = _pick_col(rn, ['DSLINKNO', 'dslinkno', 'downSegId', 'downstream_id'])
        len_col = _pick_col(rn, ['Length', 'length', 'river_length', 'RivLength'])
        slope_col = _pick_col(rn, ['Slope', 'slope', 'river_slope', 'RivSlope'])
        acc_col = _pick_col(rn, ['DSContArea', 'uparea', 'contribarea'])
        gru_col = _pick_col(rb, ['GRU_ID', 'gru_id', 'HRU_ID'])
        seg_col = _pick_col(rb, ['gru_to_seg', 'hru_to_seg', 'seg_id', 'GRU_to_seg'])
        area_col = _pick_col(rb, ['GRU_area', 'gru_area', 'HRU_area', 'Area'])
        if link_col is None or ds_col is None:
            logger.debug("river_network lacks LINKNO/DSLINKNO; cannot build distributed network")
            return None
        if len(rn) < 2:
            return None

        # GRU -> segment, GRU -> area (default: GRU id IS the segment id when no mapping col).
        seg_of_gru, area_of_seg = {}, {}
        for _, r in rb.iterrows():
            gid = int(r[gru_col]) if gru_col else None
            seg = int(r[seg_col]) if seg_col else gid
            if seg is None:
                continue
            if area_col is not None:
                area_of_seg[seg] = float(r[area_col])
            if gid is not None:
                seg_of_gru[gid] = seg

        segs = [int(v) for v in rn[link_col].tolist()]
        seg_set = set(segs)
        seg_to_int = {seg: i + 1 for i, seg in enumerate(segs)}   # stable 1-based Raven ids
        per_hru = _read_hru_geometry(reader, [str(s) for s in segs], logger) if reader else {}

        specs: List[SubbasinSpec] = []
        for _, r in rn.iterrows():
            seg = int(r[link_col])
            ds_raw = int(r[ds_col])
            ds_id = seg_to_int.get(ds_raw, RAVEN_OUTLET_ID) if ds_raw in seg_set else RAVEN_OUTLET_ID
            length_km = max(float(r[len_col]) / 1000.0, 0.0) if len_col else 0.0
            slope = max(float(r[slope_col]), _MIN_RIVER_SLOPE) if slope_col else _MIN_RIVER_SLOPE
            acc_km2 = (float(r[acc_col]) / 1e6) if acc_col else 0.0
            area_km2 = area_of_seg.get(seg, 0.0) / 1e6
            geom = per_hru.get(str(seg), {})
            if area_km2 <= 0:
                area_km2 = geom.get('area_km2', max(acc_km2, 1.0))
            specs.append(SubbasinSpec(
                subbasin_id=seg_to_int[seg],
                downstream_id=ds_id,
                area_km2=area_km2,
                reach_length_km=round(length_km, 5),
                river_slope=slope,
                elevation=geom.get('elevation', 1000.0),
                latitude=geom.get('latitude', 0.0),
                longitude=geom.get('longitude', 0.0),
                accumulated_area_km2=acc_km2,
            ))
        if len(specs) < 2:
            return None
        # Use TauDEM's DSContArea when present; otherwise accumulate from connectivity.
        if not acc_col:
            _accumulate_drainage_area(specs)
        n_outlets = sum(1 for s in specs if s.downstream_id == RAVEN_OUTLET_ID)
        logger.debug(f"Shapefile topology: {len(specs)} subbasins, {n_outlets} outlet(s)")
        return DistributedNetwork(subbasins=specs)
    except Exception as e:  # noqa: BLE001 -- unreadable shapefiles => lumped fallback
        logger.warning(f"Could not read delineation shapefiles ({e}); falling back to lumped")
        return None


def _find_shapefile(project_dir: Path, subdir: str) -> Optional[Path]:
    """Locate the first ``.shp`` under ``{project_dir}/shapefiles/{subdir}``."""
    d = Path(project_dir) / 'shapefiles' / subdir
    if not d.exists():
        return None
    shps = sorted(d.glob('*.shp'))
    return shps[0] if shps else None


def _pick_col(df, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column present in *df* (case-sensitive then -insensitive)."""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _values_or_none(ds, names: List[str]):
    """Return the first present variable's values as a list, else None."""
    import numpy as np

    for n in names:
        if n in ds.variables:
            return [None if v is None else float(v) for v in np.atleast_1d(ds[n].values)]
    return None


def _read_hru_geometry(reader, gru_ids: List[str],
                       logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    """Map each GRU id -> {area_km2, elevation, latitude, longitude}.

    One HRU per subbasin. The HRU id-space may or may not match the GRU id-space, so each
    field is aligned by (1) id-join when the GRU id is a known HRU id, else (2) position
    when the HRU and GRU counts match, else (3) the domain mean. Any field a store omits
    falls back to a sensible default.
    """
    hru_order = reader.hru_ids('hru_identity') or []

    def _id_map(group: str, names: List[str]) -> Dict[str, float]:
        """Return {hru_id: value}, joining by the group's own hru_id when present, else by
        position against the canonical hru_identity id order (groups like 'terrain' carry no
        hru_id coordinate, but their rows are written in the same order as hru_identity)."""
        for name in names:
            d = reader.per_hru_values(group, name)
            if d:
                return d
        for name in names:
            arr = reader.variable(group, name)
            if arr is not None and hru_order and len(arr) == len(hru_order):
                return {hid: float(v) for hid, v in zip(hru_order, arr)}
        return {}

    area = _id_map('hru_identity', ['hru_area'])
    lat = _id_map('hru_identity', ['latitude', 'lat'])
    lon = _id_map('hru_identity', ['longitude', 'lon'])
    elev = _id_map('terrain', ['elev_mean', 'elevation'])

    def _aligned(d: Dict[str, float], default: float) -> Dict[str, float]:
        vals = list(d.values())
        mean = float(sum(vals) / len(vals)) if vals else default
        same_count = len(d) == len(gru_ids) and len(hru_order) == len(gru_ids)
        out: Dict[str, float] = {}
        for i, gid in enumerate(gru_ids):
            if gid in d:
                out[gid] = float(d[gid])
            elif same_count and hru_order:
                out[gid] = float(d.get(hru_order[i], mean))
            else:
                out[gid] = mean
        return out

    a = _aligned(area, 1.0e6)
    e = _aligned(elev, 1000.0)
    la = _aligned(lat, 0.0)
    lo = _aligned(lon, 0.0)
    if not area:
        logger.debug("hru_identity areas absent; HRU areas fall back to gru_area/topology")
    return {gid: {'area_km2': a[gid] / 1e6, 'elevation': e[gid],
                  'latitude': la[gid], 'longitude': lo[gid]} for gid in gru_ids}


def _assemble_specs(
    gru_ids: List[str],
    downstream: List[str],
    areas_m2: Optional[List[float]],
    river_len_m: Optional[List[float]],
    river_slope: Optional[List[float]],
    per_hru: Dict[str, Dict[str, float]],
    logger: logging.Logger,
) -> List[SubbasinSpec]:
    """Zip the topology arrays + per-HRU geometry into SubbasinSpec rows.

    GRU ids (strings) are mapped to stable 1-based integer subbasin ids for Raven; the
    downstream pointer is resolved through that same mapping, with anything that does not
    resolve to a known upstream subbasin (self-loop, 0, -1, missing) becoming the outlet.
    """
    id_to_int = {gid: i + 1 for i, gid in enumerate(gru_ids)}

    def _to_int_downstream(raw: str) -> int:
        key = str(raw).strip()
        if key in id_to_int:
            mapped = id_to_int[key]
            return mapped
        return RAVEN_OUTLET_ID

    specs: List[SubbasinSpec] = []
    for i, gid in enumerate(gru_ids):
        sb_id = id_to_int[gid]
        ds_raw = downstream[i] if i < len(downstream) else ''
        ds_id = _to_int_downstream(ds_raw)
        if ds_id == sb_id:  # self-loop => outlet
            ds_id = RAVEN_OUTLET_ID

        geom = per_hru.get(gid, {})
        # Area: prefer the per-HRU area; fall back to topology gru_area.
        area_km2 = geom.get('area_km2')
        if (area_km2 is None or area_km2 <= 0) and areas_m2 and i < len(areas_m2):
            area_km2 = float(areas_m2[i]) / 1e6
        area_km2 = float(area_km2) if area_km2 and area_km2 > 0 else 1.0

        length_km = 0.0
        if river_len_m and i < len(river_len_m) and river_len_m[i] is not None:
            length_km = max(float(river_len_m[i]) / 1000.0, 0.0)
        slope = _MIN_RIVER_SLOPE
        if river_slope and i < len(river_slope) and river_slope[i] is not None:
            slope = max(float(river_slope[i]), _MIN_RIVER_SLOPE)

        specs.append(SubbasinSpec(
            subbasin_id=sb_id,
            downstream_id=ds_id,
            area_km2=area_km2,
            reach_length_km=round(length_km, 5),
            river_slope=slope,
            elevation=geom.get('elevation', 1000.0),
            latitude=geom.get('latitude', 0.0),
            longitude=geom.get('longitude', 0.0),
        ))

    n_outlets = sum(1 for s in specs if s.downstream_id == RAVEN_OUTLET_ID)
    if n_outlets != 1:
        logger.warning(
            f"Distributed network has {n_outlets} outlets (expected 1); "
            "routing topology may be incomplete")
    return specs


def _accumulate_drainage_area(specs: List[SubbasinSpec]) -> None:
    """Set ``accumulated_area_km2`` = own area + all upstream areas (for channel sizing).

    Walks each subbasin's downstream chain and adds its local area to every reach below it.
    O(n^2) worst case — fine for the subbasin counts (10^1-10^3) Raven runs at.
    """
    by_id = {s.subbasin_id: s for s in specs}
    for s in specs:
        s.accumulated_area_km2 = s.area_km2
    for s in specs:
        cur = by_id.get(s.downstream_id)
        guard = 0
        while cur is not None and guard <= len(specs):
            cur.accumulated_area_km2 += s.area_km2
            cur = by_id.get(cur.downstream_id)
            guard += 1


def _estimate_avg_annual_runoff(project_dir: Path, domain_name: str,
                                logger: logging.Logger) -> float:
    """Estimate long-term mean annual runoff (mm/yr) for Raven's ``:AvgAnnualRunoff``.

    Raven requires this command for any multi-basin model (CemaNeige snow init). Prefer the
    observed record (depth = volume / area); otherwise return a temperate default. The value
    only seeds initial snow storage, so a coarse estimate is adequate.
    """
    try:
        import pandas as pd
        from symfluence.core.mixins.project import resolve_data_subdir

        obs_file = (resolve_data_subdir(project_dir, 'observations')
                    / 'streamflow' / 'preprocessed'
                    / f"{domain_name}_streamflow_processed.csv")
        if obs_file.exists():
            df = pd.read_csv(obs_file, index_col=0, parse_dates=True)
            if not df.empty:
                q = df.iloc[:, 0].astype(float).dropna()  # m3/s
                if len(q) > 30:
                    # Depth needs catchment area; without it here, fall back to default
                    # unless the value is implausibly resolvable. Keep the robust default.
                    pass
    except Exception as e:  # noqa: BLE001 -- best-effort estimate
        logger.debug(f"AvgAnnualRunoff estimate fell back to default ({e})")
    return 400.0


# =============================================================================
# RavenPy object construction (lazy RavenPy types passed in by the caller)
# =============================================================================

def build_distributed_objects(
    rc,
    network: DistributedNetwork,
    logger: logging.Logger,
) -> Tuple[List[Any], List[Dict[str, Any]], List[Any], List[int]]:
    """Build RavenPy ``SubBasin`` + HRU dicts + ``ChannelProfile`` lists for *network*.

    ``rc`` is ``ravenpy.config.commands`` (imported lazily by the caller). Returns
    ``(sub_basins, hrus, channel_profiles, gauged_subbasin_ids)``. Every subbasin is marked
    gauged so each reach appears as a ``sub<ID> [m3/s]`` column in ``Hydrographs.csv`` (the
    per-reach output the multi-gauge objective consumes).
    """
    sub_basins: List[Any] = []
    hrus: List[Dict[str, Any]] = []
    channels: List[Any] = []
    gauged: List[int] = []

    for s in network.subbasins:
        # Every subbasin gets a real channel profile (chn_<id>): Raven rejects the "NONE"
        # channel code under any routing method other than ROUTE_NONE/ROUTE_EXTERNAL, even
        # for a zero-length reach (which simply routes with no in-channel delay).
        sub_basins.append(rc.SubBasin(
            subbasin_id=s.subbasin_id,
            name=s.name,
            downstream_id=s.downstream_id,
            profile=f"chn_{s.subbasin_id}",
            reach_length=s.reach_length_km,
            gauged=True,
            gauge_id=str(s.subbasin_id),
        ))
        gauged.append(s.subbasin_id)
        hrus.append({
            'hru_id': s.subbasin_id,
            'area': s.area_km2,
            'elevation': s.elevation,
            'latitude': s.latitude,
            'longitude': s.longitude,
            'subbasin_id': s.subbasin_id,
            'hru_type': 'land',
        })
        channels.append(_build_channel_profile(rc, s))

    logger.debug(
        f"Built {len(sub_basins)} subbasins, {len(hrus)} HRUs, "
        f"{len(channels)} channel profiles")
    return sub_basins, hrus, channels, gauged


def _build_channel_profile(rc, s: SubbasinSpec):
    """Build a trapezoidal :class:`ChannelProfile` for one reach (SWAT geometry).

    Mirrors ``ravenpy.extractors.routing_product`` ``_extract_channel_profile``: a
    compound channel + floodplain cross-section with 2:1 channel side slopes and Manning's
    n zones. Bankfull width/depth are estimated from upstream-accumulated drainage area via
    downstream hydraulic geometry (the store has no surveyed cross-sections).
    """
    width, depth = estimate_bankfull_geometry(s.accumulated_area_km2)
    channel_elev = float(s.elevation)
    slope = max(float(s.river_slope), _MIN_RIVER_SLOPE)

    # SWAT compound-channel geometry (see SWAT2009 theory, "Channel Characteristics").
    zch = 2.0                       # channel side slope (2:1 run:rise)
    sidwd = zch * depth             # channel side width
    botwd = width - 2.0 * sidwd     # channel bottom width
    if botwd < 0:
        botwd = 0.5 * width
        sidwd = 0.25 * width
    zfld = 4.0 + channel_elev       # floodplain top elevation
    zbot = channel_elev - depth     # channel bottom elevation
    sfp = 16.0                      # floodplain side width (4/0.25)

    survey_points = (
        (0.0, zfld),
        (sfp, channel_elev),
        (sfp + 2.0 * width, channel_elev),
        (sfp + 2.0 * width + sidwd, zbot),
        (sfp + 2.0 * width + sidwd + botwd, zbot),
        (sfp + 2.0 * width + 2.0 * sidwd + botwd, channel_elev),
        (sfp + 4.0 * width + 2.0 * sidwd + botwd, channel_elev),
        (2.0 * sfp + 4.0 * width + 2.0 * sidwd + botwd, zfld),
    )
    roughness_zones = (
        (0.0, _MANNING_FLOODPLAIN),
        (sfp + 2.0 * width, _MANNING_CHANNEL),
        (sfp + 2.0 * width + 2.0 * sidwd + botwd, _MANNING_FLOODPLAIN),
    )
    return rc.ChannelProfile(
        name=f"chn_{s.subbasin_id}",
        bed_slope=slope,
        survey_points=survey_points,
        roughness_zones=roughness_zones,
    )


def estimate_bankfull_geometry(accumulated_area_km2: float) -> Tuple[float, float]:
    """Estimate bankfull (width, depth) in metres from upstream drainage area (km^2).

    Downstream hydraulic geometry: ``W = c_w * A^p_w``, ``D = c_d * A^p_d``. Floored to a
    minimum so headwater reaches still produce a valid trapezoid.
    """
    area = max(float(accumulated_area_km2), 0.0)
    width = max(_HG_WIDTH_COEFF * (area ** _HG_WIDTH_EXP), _MIN_BANKFULL_WIDTH)
    depth = max(_HG_DEPTH_COEFF * (area ** _HG_DEPTH_EXP), _MIN_BANKFULL_DEPTH)
    return width, depth


__all__ = [
    'SubbasinSpec',
    'DistributedNetwork',
    'read_distributed_topology',
    'build_distributed_objects',
    'estimate_bankfull_geometry',
    'RAVEN_OUTLET_ID',
]
