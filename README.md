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

Set `HYDROLOGICAL_MODEL: RAVEN` and `RAVEN_MODEL_TEMPLATE: GR4JCN` (or HBVEC / HMETS /
MOHYSE / BLENDED) in your config, then run the normal workflow:

```bash
symfluence list models            # RAVEN appears once installed
symfluence workflow run --config my_config.yaml
```

See the SYMFLUENCE docs for the full configuration reference.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
