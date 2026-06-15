# raven-symfluence

[Raven](http://raven.uwaterloo.ca/) hydrological modelling framework as a first-class
[SYMFLUENCE](https://symfluence.readthedocs.io) model.

This plugin registers Raven as a SYMFLUENCE model via the `symfluence.plugins` entry
point. It feeds Raven from SYMFLUENCE's model-ready datastore (canonical CFIF forcing +
catchment attributes), drives it through [RavenPy](https://github.com/Ouranos-Hydro/RavenPy)
(which writes the `.rvi/.rvp/.rvh/.rvt/.rvc` files and runs the `raven` binary), and exposes
it to the standard SYMFLUENCE run + calibration pipeline (DDS/PSO/…, multi-gauge,
regionalization).

## Install

```bash
pip install raven-symfluence            # plugin + SYMFLUENCE
pip install "raven-symfluence[raven]"   # also pulls RavenPy
symfluence binary install raven         # build the Raven engine from source
```

## Use

Set `HYDROLOGICAL_MODEL: RAVEN` and pick a structure with `RAVEN_MODEL_TEMPLATE`. All eight
RavenPy emulators are supported: **GR4JCN, HBVEC, HMETS, MOHYSE, BLENDED, CANADIANSHIELD,
HYPR, SACSMA**. Then run the normal workflow:

```bash
symfluence list models            # RAVEN appears once installed
symfluence workflow run --config my_config.yaml
```

### Lumped vs. distributed

`RAVEN_SPATIAL_MODE: lumped` (default) runs a single HRU/subbasin. `RAVEN_SPATIAL_MODE:
distributed` reads the river-network *topology* from the model-ready attributes store and
builds a routed multi-subbasin model (one land HRU per subbasin, in-channel routing via
`RAVEN_ROUTING_METHOD`, default `ROUTE_DIFFUSIVE_WAVE`). Each subbasin is gauged, so a
per-reach hydrograph is produced — which enables **multi-gauge calibration** against the
whole network:

```yaml
HYDROLOGICAL_MODEL: RAVEN
RAVEN_MODEL_TEMPLATE: HMETS
RAVEN_SPATIAL_MODE: distributed
MULTI_GAUGE_CALIBRATION: true
GAUGE_SEGMENT_MAPPING: /path/to/gauge_segment_mapping.csv   # id -> subbasin id
MULTI_GAUGE_OBS_DIR:   /path/to/obs                          # ID_<id>.csv per gauge
```

Distributed mode falls back to lumped automatically for undelineated/single-subbasin
domains. (CanadianShield needs two HRUs per subbasin and runs lumped only for now.)

### Regionalization + multi-objective calibration

In distributed mode, `PARAMETER_REGIONALIZATION: transfer_function` makes selected
parameters vary **per subbasin** as a function of physical attributes
(`param_i = a + b·attr_norm_i`). The optimizer then calibrates the transfer-function
coefficients (a handful of numbers that generalise across the network) instead of one
value per subbasin; values are applied per subbasin via Raven's `SBGroupPropertyMultiplier`.
The default mapping (GR4JCN) regionalizes the production store on precipitation and the
routing/snow stores on elevation; override with `RAVEN_REGIONALIZATION_PARAM_CONFIG`.

Calibration can also target **SWE and ET** alongside streamflow. Raven emits the matching
`:CustomOutput` (SNOW = SWE, AET) automatically, and the multi-objective framework combines
them:

```yaml
RAVEN_SPATIAL_MODE: distributed
PARAMETER_REGIONALIZATION: transfer_function
OBJECTIVE_WEIGHTS: {STREAMFLOW: 0.7, SWE: 0.2, ET: 0.1}
```

See the SYMFLUENCE docs for the full configuration reference.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
