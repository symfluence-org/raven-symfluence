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

## Honest status

- **Raven — validated.** Runs end-to-end through this harness on the fixture store and
  produces a real KGE row. (See `tests/test_wrap_fidelity.py`: SYMFLUENCE-driven Raven is
  byte-for-byte identical to standalone RavenPy at `rtol=atol=1e-9`.)
- **FUSE — wired but skipped.** The FUSE binary is detected, but the FUSE preprocessor
  needs a FUSE-native forcing + elevation-band store this synthetic lumped fixture does
  not yet build. Wiring the FUSE `forcing_adapter` against the CFIF fixture store is the
  one remaining TODO for the FUSE leg (intentionally not rabbit-holed in Phase-1).
- **SUMMA — skipped.** No SUMMA-hydro binary is built in this environment. SUMMA also
  needs intersected catchment shapefiles (`acquire`/`discretize` outputs) the lumped
  fixture does not provide.
- **Eval-period KGE** is computed from the postprocessed `results/*.csv` of the run leg.
  `kge_calib_best` reflects the (stochastic, unseeded) DDS loop and will vary run-to-run.

### What a real single-CAMELS-basin 3-engine run still needs

1. **SUMMA build** — compile SUMMA-hydro (`summa_sundials.exe`) and point
   `SUMMA_INSTALL_PATH` at it.
2. **FUSE config** — a FUSE-native forcing/elevation-band store (FUSE `forcing_adapter`)
   built from the same domain so the FUSE leg can preprocess.
3. **Data acquisition** — a real CAMELS(-SPAT) domain (forcing, attributes, catchment +
   river shapefiles) instead of the synthetic fixture.
4. **Compute** — real basins + meaningful DDS budgets (hundreds of iterations × 3
   engines) are far heavier than the fixture's 6-iteration smoke loop.

---
Repo and code assistance from [Claude](https://claude.ai) (Anthropic).
