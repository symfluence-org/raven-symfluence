# Phase-1 multi-engine benchmark (CAMELS-bound)

A reproducible harness that runs **multiple hydrological engines through the same
SYMFLUENCE pipeline on the same store**, calibrates each with a short DDS loop, and
compares them on KGE/NSE. Phase-1 is the *plumbing + Raven validation*; the multi-basin
CAMELS sweep is Phase-2/3.

## What Phase-1 is (and isn't)

- **Is:** one self-contained fixture domain (a single lumped HRU with synthetic CFIF
  forcing and a synthetic-but-reasonable observed streamflow target), driven through
  `model_specific_preprocessing → run_model → postprocess_results → calibrate_model`
  for every engine whose binary is available, with KGE/NSE scored on the evaluation
  period and written to `results.csv` + `comparison.png`.
- **Isn't:** a real CAMELS basin, real forcing/attributes acquisition, or a full
  multi-basin/cross-engine skill comparison. Those are Phase-2/3 (see *Status* below).

The harness **never writes into your real SYMFLUENCE domains** — it builds the fixture
store under a fresh temp dir (or `--workdir`).

## How to run

The harness has two modes:

**1. Synthetic-fixture mode** (default, no args) — fabricates per-engine synthetic
discharge against a synthetic observation series and scores KGE/NSE. Needs no model
binaries and no real data; safe to run anywhere (CI self-check).

```bash
python benchmark/camels_phase1/run_benchmark.py            # all engines, synthetic
python benchmark/camels_phase1/run_benchmark.py --out ./bench_out
```

**2. Real-domain mode** (`--domain <path>`) — drives each first-class SYMFLUENCE model
through the *standard* workflow (`model_specific_preprocessing → run_model →
postprocess_results → calibrate_model`) on a real domain, using the per-engine config
`configs/config_<engine>.yaml`, then reads each engine's
`optimization/<MODEL>/*_dds_final_evaluation.json` into `results.csv` + `comparison.png`.

```bash
# Copy the example configs and fill in your absolute paths first:
cp configs/config_raven.example.yaml configs/config_raven.yaml   # then edit paths
# (repeat for fuse / summa)

# Run the engines end-to-end on an isolated domain:
python benchmark/camels_phase1/run_benchmark.py \
    --domain /path/to/SYMFLUENCE_data/domain_multimodel_benchmark \
    --engines raven fuse summa \
    --out benchmark/camels_phase1/results

# Already calibrated? Just assemble the comparison from existing runs (no re-run):
python benchmark/camels_phase1/run_benchmark.py \
    --domain /path/to/SYMFLUENCE_data/domain_multimodel_benchmark \
    --engines raven fuse summa --no-run
```

Engine names are lowercase (`raven`, `fuse`, `summa`). The actual `config_<engine>.yaml`
files hold machine-specific absolute paths and are git-ignored; only the
`*.example.yaml` templates are tracked. Binaries are resolved from each config's
`*_INSTALL_PATH`/`*_EXE` keys (a bare `summa` on macOS PATH is an unrelated coreutil and
is not used). The harness **never writes into a real domain** unless you point `--domain`
at one — use an isolated copy (e.g. `domain_multimodel_benchmark` with symlinked inputs).

## Outputs

- `results.csv` — one row per engine: `status`, `kge_eval`, `nse_eval`,
  `kge_calib_best` (best calibration-period KGE from the DDS log), `n_eval_days`, `note`.
- `comparison.png` — KGE bar chart for engines that ran (matplotlib; imported lazily and
  guarded, so it is not a hard dependency — the CSV is always written).

## Pointing it at a real CAMELS basin

The fixture config mirrors the keys SYMFLUENCE's
`resources/config_templates/camelsspat_template.yaml` uses
(`DOMAIN_NAME`, `HYDROLOGICAL_MODEL`, `CALIBRATION_PERIOD`, `EVALUATION_PERIOD`,
`FORCING_DATASET`, the per-engine `*_EXE` / `*_INSTALL_PATH` blocks, …). To run a real
basin instead of the fixture:

1. Acquire a real CAMELS(-SPAT) domain with SYMFLUENCE (`define_domain`,
   `acquire_attributes`, `acquire_forcing`, `discretize_domain`) so the model-ready store
   and catchment shapefiles exist.
2. Copy `camelsspat_template.yaml`, set `HYDROLOGICAL_MODEL` per engine, point the
   `*_INSTALL_PATH`/`*_EXE` keys at your built binaries.
3. Run each engine's config through the same four steps the harness uses
   (`run_individual_steps([...])`) and score with `symfluence.evaluation.metrics_core`.

Phase-2 will fold steps 1–3 into the harness for a named CAMELS basin; Phase-3 sweeps
the multi-basin CAMELS set.

## Validated real-domain result (Bow at Banff, lumped)

All three engines were driven through the standard SYMFLUENCE workflow
(`model_specific_preprocessing → run_model → postprocess_results → calibrate_model`) on
one real lumped basin sharing a single byte-identical model-ready store, then assembled
with `--from-domain`. All cold-started (DDS from each engine's default parameters):

| Engine | DDS iters | start KGE | KGE (calib) | **KGE (eval)** | NSE (eval) |
|--------|----------:|----------:|------------:|---------------:|-----------:|
| **Raven** (GR4JCN, 6 params) | 20   | 0.70   | 0.792 | **0.700** | 0.589 |
| **SUMMA** (v4.0.0)           | 20   | —      | 0.745 | **0.734** | 0.451 |
| **FUSE** (13 params)         | 1000 | −0.151 | 0.613 | **0.665** | 0.291 |

The point holds: feeding all three engines from the same model-agnostic store via the
normal workflow yields a like-for-like comparison, and **the Raven wrap is fully
competitive** (best NSE, second KGE) — driven entirely through SYMFLUENCE.

### Why FUSE looks weakest here — and why it isn't a structural result

An earlier smoke run scored FUSE at eval KGE 0.49 (21 iters). Digging in: on **this exact
basin** (byte-identical forcing/obs/period — `precip` mean 1.823, `temp` −2.922,
`q_obs` 1.392) a known-good FUSE run reaches **calib 0.874 / eval 0.87+** with the *same*
13 parameters and the *same* 1001-iter DDS budget. The entire difference is the **DDS
starting point**:

- known-good run: DDS *starts* at KGE **0.449** → 0.874 (its `para_def.nc` had been
  pre-optimised by a prior SCE-UA pass — `para_def.nc.bak_sce` is the receipt; a
  **warm start**);
- this benchmark: DDS *starts* at KGE **−0.151** from FUSE's **raw defaults**, which are
  pathological on a snowmelt basin like Bow → 0.613 even at the full 1000 iters.

DDS is a perturbation search around its seed, so a 0.6-KGE-worse start lands in a worse
local basin regardless of budget. **FUSE is therefore not structurally weak on Bow and
not under-iterated — it is starting-point sensitive in its 13-D space.** The honest
cold-start number is 0.665; warm-started (or with a global pre-search / multi-start DDS)
it matches the others. The fair-comparison fix for Phase-2: give every engine the same
seeding policy (all cold from defaults, *or* all warm from a short global pre-search) —
not a fixed iteration count, which silently favours low-dimensional structures like
GR4JCN's 6 params.

## Status

- **Raven — validated** end-to-end (fixture + real domain). `tests/test_wrap_fidelity.py`
  shows SYMFLUENCE-driven Raven is identical to standalone RavenPy at `rtol=atol=1e-9`.
- **FUSE — validated on the real domain.** Its `model_specific_preprocessing` builds the
  FUSE-native store from the shared model-ready store (the "first-class model" claim). The
  *synthetic fixture* leg stays skipped (the fixture doesn't build FUSE-native inputs); use
  `--from-domain` / a real domain for FUSE.
- **SUMMA — validated on the real domain** (SUMMA-hydro v4.0.0). Skips in the synthetic
  fixture (no SUMMA-hydro binary + needs discretized shapefiles).

### What a publication-grade run still needs

1. **A uniform seeding + budget policy across engines** — either all cold-start from
   defaults or all warm-start from a short global pre-search (SCE-UA / multi-start),
   with a per-engine budget scaled to parameter dimension rather than a flat iteration
   count (a flat count favours GR4JCN's 6 params over FUSE's 13). See the FUSE note above.
2. **Many basins** — a CAMELS-scale sweep across gauges/regimes, not one basin.
3. **Performance** — cache the daily-aggregated forcing; Raven currently rebuilds forcing
   per DDS trial (~73 s/trial) and FUSE's hourly `.load()` is slow.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
