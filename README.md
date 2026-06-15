# raven-symfluence

A [SYMFLUENCE](https://symfluence.readthedocs.io) plugin that runs the
[Raven](http://raven.uwaterloo.ca/) hydrological modelling framework as a SYMFLUENCE model.

The plugin registers Raven via the `symfluence.plugins` entry point. It feeds Raven from
SYMFLUENCE's model-ready datastore (canonical CFIF forcing + catchment attributes), drives
it through [RavenPy](https://github.com/Ouranos-Hydro/RavenPy) (which writes the
`.rvi/.rvp/.rvh/.rvt/.rvc` files and runs the `raven` binary), and connects it to
SYMFLUENCE's run + calibration pipeline.

> **Status: early / experimental.** The pieces below run end-to-end and are covered by unit
> and integration tests, but they have only been exercised on synthetic fixtures and a small
> number of real basins with short calibration budgets. Treat results as preliminary and
> validate on your own domains. Issues and PRs welcome.

## Install

```bash
pip install raven-symfluence            # plugin + SYMFLUENCE
pip install "raven-symfluence[raven]"   # also pulls RavenPy
symfluence binary install raven         # build the Raven engine from source
```

## Use

Set `HYDROLOGICAL_MODEL: RAVEN` and pick a structure with `RAVEN_MODEL_TEMPLATE`. The eight
RavenPy emulators are wired: GR4JCN, HBVEC, HMETS, MOHYSE, BLENDED, CANADIANSHIELD, HYPR,
SACSMA. Then run the normal workflow:

```bash
symfluence list models            # RAVEN appears once installed
symfluence workflow run --config my_config.yaml
```

### Lumped vs. distributed

`RAVEN_SPATIAL_MODE: lumped` (default) runs a single HRU/subbasin. `RAVEN_SPATIAL_MODE:
distributed` reads the river-network topology (from the model-ready attributes store, or the
TauDEM delineation shapefiles) and builds a routed multi-subbasin model — one land HRU per
subbasin, in-channel routing via `RAVEN_ROUTING_METHOD` (default `ROUTE_DIFFUSIVE_WAVE`).
Each subbasin is gauged, so a per-reach hydrograph is produced, which allows multi-gauge
calibration:

```yaml
HYDROLOGICAL_MODEL: RAVEN
RAVEN_MODEL_TEMPLATE: HMETS
RAVEN_SPATIAL_MODE: distributed
MULTI_GAUGE_CALIBRATION: true
GAUGE_SEGMENT_MAPPING: /path/to/gauge_segment_mapping.csv   # gauge id -> subbasin id
MULTI_GAUGE_OBS_DIR:   /path/to/obs                          # ID_<id>.csv per gauge
```

Distributed mode falls back to lumped for undelineated or single-subbasin domains.
CanadianShield needs two HRUs per subbasin and currently runs lumped only.

Channel cross-sections for routing are estimated from drainage area via hydraulic-geometry
relations (the attributes store carries no surveyed cross-sections), so routed timing is
approximate.

### Regionalization and multi-objective calibration

In distributed mode, `PARAMETER_REGIONALIZATION: transfer_function` lets selected parameters
vary per subbasin as `param_i = a + b·attr_norm_i`; the optimizer calibrates the
transfer-function coefficients rather than a value per subbasin, and values are applied via
Raven's `SBGroupPropertyMultiplier`. The default GR4JCN mapping ties the production store to
precipitation and the routing/snow stores to elevation; override with
`RAVEN_REGIONALIZATION_PARAM_CONFIG`. Regionalization is a regularization aimed at
ungauged/multi-gauge prediction — it will not always improve a single-outlet fit.

Calibration can also target SWE and ET alongside streamflow. Raven emits the matching
`:CustomOutput` (SNOW for SWE, AET) when a SWE/ET objective is configured, and SYMFLUENCE's
multi-objective framework combines them:

```yaml
RAVEN_SPATIAL_MODE: distributed
PARAMETER_REGIONALIZATION: transfer_function
OBJECTIVE_WEIGHTS: {STREAMFLOW: 0.7, SWE: 0.2, ET: 0.1}
```

See the SYMFLUENCE docs for the full configuration reference.

## Tests

```bash
pip install -e ".[raven]"
pytest                       # unit tests; integration tests need the raven binary
```

## License

MIT (see [LICENSE](LICENSE)). Raven is developed at the University of Waterloo; RavenPy is
developed by Ouranos; SYMFLUENCE is a separate project under its own license. This plugin is
an independent integration and is not affiliated with or endorsed by any of them; install
and use each under its own terms.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
