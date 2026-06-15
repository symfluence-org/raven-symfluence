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

```bash
# All engines, fixture store in a temp dir, outputs in ./bench_out
python benchmark/camels_phase1/run_benchmark.py -v

# Subset of engines
python benchmark/camels_phase1/run_benchmark.py --engines RAVEN

# Keep the fixture store + choose an output dir
python benchmark/camels_phase1/run_benchmark.py \
    --workdir /path/to/scratch --outdir /path/to/out

# Assemble the comparison from a REAL domain's completed standard-workflow runs
# (reads optimization/<ENGINE>/*_dds_final_evaluation.json) — how the result below was made
python benchmark/camels_phase1/run_benchmark.py \
    --from-domain /path/to/domain_multimodel_benchmark \
    --engines RAVEN SUMMA FUSE --outdir benchmark/camels_phase1/results
```

Binaries are auto-resolved (skipped engines are reported, not errored):

| Engine | Resolution | Override |
|--------|-----------|----------|
| Raven  | RavenPy's `RAVEN_EXEC_PATH` / `RAVENPY_RAVEN_BINARY_PATH` / PATH | `RAVENPY_RAVEN_BINARY_PATH` |
| FUSE   | `FUSE_INSTALL_PATH`/`FUSE_EXE`, else `fuse.exe` on PATH | `FUSE_INSTALL_PATH` + `FUSE_EXE` |
| SUMMA  | SUMMA-hydro-named binary on PATH, else `SUMMA_INSTALL_PATH`/`SUMMA_EXE` | `SUMMA_INSTALL_PATH` + `SUMMA_EXE` |

A bare `summa` on PATH is intentionally **not** accepted (it's usually an unrelated
coreutil on macOS); SUMMA skips until a real SUMMA-hydro build is provided.

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
one real lumped basin sharing a single model-ready store, then assembled with
`--from-domain`. Short DDS budgets (15–20 iters — a smoke run, not publication-grade):

| Engine | KGE (calib) | **KGE (eval)** | NSE (eval) |
|--------|------------:|---------------:|-----------:|
| **Raven** (GR4JCN) | 0.792 | **0.700** | 0.589 |
| **SUMMA** (v4.0.0) | 0.745 | **0.734** | 0.451 |
| **FUSE**           | 0.420 | **0.493** | −0.053 |

The point holds: feeding all three engines from the same model-agnostic store via the
normal workflow yields a like-for-like comparison, and **the Raven wrap is fully
competitive** (best NSE, second KGE) — driven entirely through SYMFLUENCE.

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

1. **Bigger DDS budgets** (500–2000+ iters with a proper spin-up/warmup hold-out — the
   numbers above are a smoke run; FUSE's eval NSE is still negative).
2. **Many basins** — a CAMELS-scale sweep across gauges/regimes, not one basin.
3. **Performance** — cache the daily-aggregated forcing; Raven currently rebuilds forcing
   per DDS trial (~73 s/trial) and FUSE's hourly `.load()` is slow.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
