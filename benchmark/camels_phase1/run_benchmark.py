# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Phase-1 multi-engine benchmark harness.

Drives one or more hydrological engines (Raven, FUSE, SUMMA) through the *same*
SYMFLUENCE pipeline on the *same* self-contained fixture store, runs a SHORT DDS
calibration for each, scores KGE/NSE on the evaluation period, and writes a comparison
``results.csv`` + a KGE bar chart ``comparison.png``.

Each engine is run only if its binary is discoverable; otherwise it is skipped with a
clear note. This is the honest Phase-1 deliverable: Raven runs end-to-end here, FUSE
runs if a config can be wired against the fixture store, and SUMMA is skipped until its
binary is built. Multi-basin CAMELS is Phase-2/3.

Usage::

    python run_benchmark.py                       # all engines, tmp fixture store
    python run_benchmark.py --engines RAVEN FUSE  # subset
    python run_benchmark.py --outdir ./bench_out  # keep outputs

The harness never writes into the user's real domains: the fixture store is built under
a fresh temp dir (or ``--workdir``).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# The harness lives in the plugin repo; make its package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_domain import (  # noqa: E402
    FixtureDomain,
    build_fixture_domain,
    observed_streamflow,
    write_engine_config,
)

LOGGER = logging.getLogger("raven_benchmark")

ALL_ENGINES = ["RAVEN", "FUSE", "SUMMA"]


@dataclass
class EngineResult:
    """One row of the comparison table."""

    engine: str
    status: str  # "ok" | "skipped" | "failed"
    note: str = ""
    kge: float = float("nan")
    nse: float = float("nan")
    kge_calib: float = float("nan")  # best calibration-period KGE (from DDS)
    n_eval: int = 0


@dataclass
class BenchmarkOutcome:
    results: List[EngineResult] = field(default_factory=list)
    results_csv: Optional[Path] = None
    comparison_png: Optional[Path] = None


# =============================================================================
# Binary resolution
# =============================================================================

def resolve_binaries() -> Dict[str, Optional[str]]:
    """Resolve each engine's binary, returning {engine: path-or-None}.

    Raven  : RavenPy's resolved ``RAVEN_EXEC_PATH`` (or RAVENPY_RAVEN_BINARY_PATH/PATH).
    FUSE   : ``FUSE_EXE``/``FUSE_INSTALL_PATH`` env or ``fuse.exe`` on PATH.
    SUMMA  : ``SUMMA_EXE``/``SUMMA_INSTALL_PATH`` env or ``summa*`` on PATH.
    """
    binaries: Dict[str, Optional[str]] = {}

    # Raven (via RavenPy).
    raven = os.environ.get("RAVENPY_RAVEN_BINARY_PATH")
    if not raven:
        try:
            from ravenpy._raven import RAVEN_EXEC_PATH

            if RAVEN_EXEC_PATH and Path(str(RAVEN_EXEC_PATH)).exists():
                raven = str(RAVEN_EXEC_PATH)
        except Exception:  # noqa: BLE001 -- ravenpy/binary absent => skip Raven
            raven = None
    binaries["RAVEN"] = raven or shutil.which("raven")

    # FUSE.
    fuse = os.environ.get("FUSE_EXE")
    fuse_install = os.environ.get("FUSE_INSTALL_PATH")
    if fuse and fuse_install:
        cand = Path(fuse_install) / Path(fuse).name
        fuse = str(cand) if cand.exists() else None
    else:
        fuse = shutil.which("fuse.exe") or shutil.which("fuse")
    binaries["FUSE"] = fuse

    # SUMMA. Only accept an explicit SUMMA-hydro build (env-provided install path, or a
    # SUMMA-hydro-named binary on PATH). A bare ``summa`` on PATH is NOT accepted: on
    # macOS that is usually an unrelated coreutil, and the SUMMA leg must skip honestly
    # until the real SUMMA-hydro engine is built.
    summa = os.environ.get("SUMMA_EXE")
    summa_install = os.environ.get("SUMMA_INSTALL_PATH")
    if summa and summa_install and summa_install != "default":
        cand = Path(summa_install) / Path(summa).name
        summa = str(cand) if cand.exists() else None
    else:
        summa = shutil.which("summa_sundials.exe") or shutil.which("summa.exe") or shutil.which("summa_actors")
    binaries["SUMMA"] = summa

    return binaries


# =============================================================================
# Scoring
# =============================================================================

def _slice_period(series: pd.Series, period: str) -> pd.Series:
    start_s, end_s = (p.strip() for p in period.split(","))
    return series.loc[pd.to_datetime(start_s):pd.to_datetime(end_s)]


def _score_eval(sim: pd.Series, obs: pd.Series, eval_period: str) -> tuple[float, float, int]:
    """KGE/NSE on the evaluation period from aligned daily sim/obs (m3/s)."""
    from symfluence.evaluation.metrics_core import kge, nse

    sim = sim[~sim.index.duplicated()].sort_index()
    obs = obs[~obs.index.duplicated()].sort_index()
    joined = pd.concat([obs.rename("obs"), sim.rename("sim")], axis=1, join="inner").dropna()
    joined = _slice_period(joined, eval_period)
    if joined.empty:
        return float("nan"), float("nan"), 0
    k = float(kge(joined["obs"].to_numpy(), joined["sim"].to_numpy()))
    n = float(nse(joined["obs"].to_numpy(), joined["sim"].to_numpy()))
    return k, n, int(len(joined))


def _read_results_streamflow(project_dir: Path, engine: str) -> Optional[pd.Series]:
    """Read the postprocessed simulated streamflow (m3/s) from results/*.csv."""
    results = sorted((project_dir / "results").rglob("*results*.csv"))
    if not results:
        results = sorted((project_dir / "results").rglob("*.csv"))
    if not results:
        return None
    df = pd.read_csv(results[-1], index_col=0, parse_dates=True)
    # Prefer the engine's discharge column; else the first numeric column.
    col = next((c for c in df.columns if "discharge" in c.lower() or engine.lower() in c.lower()), None)
    if col is None:
        numeric = df.select_dtypes("number")
        if numeric.empty:
            return None
        col = numeric.columns[0]
    return pd.to_numeric(df[col], errors="coerce")


def _read_best_calib_kge(project_dir: Path) -> float:
    """Best calibration-period KGE from the DDS iteration_results.csv (if present)."""
    it = next((project_dir / "optimization").rglob("*iteration_results.csv"), None)
    if it is None:
        return float("nan")
    try:
        df = pd.read_csv(it)
        scores = df["score"].to_numpy(dtype=float)
        scores = scores[np.isfinite(scores) & (scores > -9000)]
        return float(np.max(scores)) if scores.size else float("nan")
    except Exception:  # noqa: BLE001 -- malformed/partial calibration log
        return float("nan")


# =============================================================================
# Assemble from completed real-domain runs
# =============================================================================

def results_from_domain(domain_dir: Path, engines: List[str]) -> List[EngineResult]:
    """Assemble the comparison from completed standard-workflow runs in a real domain.

    Reads each engine's ``optimization/<ENGINE>/.../*_dds_final_evaluation.json`` (written
    by ``calibrate_model``) — the authoritative calibration + evaluation metrics. This is
    how the real multi-engine run is captured: each engine is driven through the normal
    SYMFLUENCE pipeline on the shared model-ready store, then its metrics are read back here.
    """
    import json

    out: List[EngineResult] = []
    for engine in engines:
        eng_dir = domain_dir / "optimization" / engine
        ev = next(eng_dir.rglob("*_dds_final_evaluation.json"), None) if eng_dir.exists() else None
        if ev is None:
            out.append(EngineResult(engine, "skipped",
                                    note=f"no *_dds_final_evaluation.json under optimization/{engine}"))
            continue
        try:
            d = json.loads(ev.read_text())
            cm, em = d.get("calibration_metrics", {}), d.get("evaluation_metrics", {})
            out.append(EngineResult(
                engine, "ok",
                kge=float(em.get("KGE", float("nan"))), nse=float(em.get("NSE", float("nan"))),
                kge_calib=float(cm.get("KGE", float("nan"))), note=f"from {ev.name}"))
        except Exception as e:  # noqa: BLE001 -- one engine's malformed JSON must not abort
            out.append(EngineResult(engine, "failed", note=f"could not read {ev.name}: {e}"))
    return out


# =============================================================================
# Per-engine run
# =============================================================================

def run_engine(fx: FixtureDomain, code_dir: Path, engine: str, binaries: Dict[str, Optional[str]]) -> EngineResult:
    """Run one engine end-to-end on the fixture store and score it."""
    binary = binaries.get(engine)
    if not binary:
        return EngineResult(engine, "skipped", note=f"{engine} binary not found")

    if engine == "RAVEN":
        os.environ["RAVENPY_RAVEN_BINARY_PATH"] = str(binary)

    # FUSE wiring against this fixture store is non-trivial (needs a FUSE-native
    # forcing/elevation-band setup); leave it wired-but-skipped rather than rabbit-hole.
    if engine == "FUSE":
        return EngineResult(
            engine, "skipped",
            note=("FUSE binary present but the FUSE preprocessor needs a FUSE-native "
                  "forcing + elevation-band store this synthetic fixture does not yet "
                  "provide (TODO: wire FUSE forcing_adapter against the CFIF store)"))

    cfg = write_engine_config(fx, code_dir, engine, binaries)
    try:
        from symfluence import SYMFLUENCE

        sym = SYMFLUENCE(cfg)
        sym.run_individual_steps(
            ["model_specific_preprocessing", "run_model", "postprocess_results", "calibrate_model"])
    except Exception as e:  # noqa: BLE001 -- one engine failing must not abort the benchmark
        LOGGER.exception("%s pipeline failed", engine)
        return EngineResult(engine, "failed", note=f"pipeline error: {e}")

    sim = _read_results_streamflow(fx.project_dir, engine)
    if sim is None or sim.dropna().empty:
        return EngineResult(engine, "failed", note="no simulated streamflow in results/")

    obs = observed_streamflow(fx)
    kge_v, nse_v, n_eval = _score_eval(sim, obs, fx.evaluation_period)
    kge_calib = _read_best_calib_kge(fx.project_dir)
    return EngineResult(engine, "ok", kge=kge_v, nse=nse_v, kge_calib=kge_calib, n_eval=n_eval)


# =============================================================================
# Outputs
# =============================================================================

def write_results_csv(results: List[EngineResult], outdir: Path) -> Path:
    rows = [{
        "engine": r.engine, "status": r.status, "kge_eval": r.kge, "nse_eval": r.nse,
        "kge_calib_best": r.kge_calib, "n_eval_days": r.n_eval, "note": r.note,
    } for r in results]
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "results.csv"
    df.to_csv(path, index=False)
    return path


def write_comparison_png(results: List[EngineResult], outdir: Path) -> Optional[Path]:
    """KGE bar chart for engines that ran. Matplotlib is imported lazily/guarded."""
    ran = [r for r in results if r.status == "ok" and np.isfinite(r.kge)]
    if not ran:
        LOGGER.warning("No successful engine runs; skipping comparison.png")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 -- matplotlib is an optional plotting dep
        LOGGER.warning("matplotlib unavailable (%s); skipping comparison.png", e)
        return None

    labels = [r.engine for r in ran]
    kges = [r.kge for r in ran]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, kges, color="#3b7dd8")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("KGE (evaluation period)")
    ax.set_title("Phase-1 multi-engine benchmark -- KGE by engine")
    ax.set_ylim(min(-1.0, min(kges) - 0.1), 1.0)
    for i, v in enumerate(kges):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout()
    path = outdir / "comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# =============================================================================
# Orchestration
# =============================================================================

def run_benchmark(
    engines: List[str],
    workdir: Optional[Path],
    outdir: Path,
    code_dir: Path,
) -> BenchmarkOutcome:
    """Build the fixture store once, run each requested engine, write outputs."""
    binaries = resolve_binaries()
    LOGGER.info("Resolved binaries: %s",
                {k: (v or "—") for k, v in binaries.items()})

    cleanup = False
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="raven_bench_"))
        cleanup = True
    workdir = Path(workdir)

    outcome = BenchmarkOutcome()
    try:
        fx = build_fixture_domain(workdir / "data_root", code_dir)
        for engine in engines:
            LOGGER.info("=== Engine: %s ===", engine)
            res = run_engine(fx, code_dir, engine, binaries)
            LOGGER.info("%s -> %s (KGE_eval=%.4f, note=%s)",
                        engine, res.status, res.kge, res.note or "—")
            outcome.results.append(res)

        outcome.results_csv = write_results_csv(outcome.results, outdir)
        outcome.comparison_png = write_comparison_png(outcome.results, outdir)
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
    return outcome


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-1 multi-engine benchmark harness")
    parser.add_argument("--engines", nargs="+", default=ALL_ENGINES,
                        choices=ALL_ENGINES, help="engines to benchmark")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="fixture store location (default: a fresh temp dir, cleaned up)")
    parser.add_argument("--outdir", type=Path, default=Path.cwd() / "bench_out",
                        help="where results.csv + comparison.png are written")
    parser.add_argument("--from-domain", type=Path, default=None,
                        help="assemble results from a real domain's completed runs "
                             "(reads optimization/<ENGINE>/*_dds_final_evaluation.json) "
                             "instead of running the synthetic fixture")
    parser.add_argument("--code-dir", type=Path, default=Path.cwd(),
                        help="SYMFLUENCE_CODE_DIR for the configs (default: cwd)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOGGER.setLevel(logging.INFO)

    if args.from_domain is not None:
        results = results_from_domain(args.from_domain, args.engines)
        outcome = BenchmarkOutcome(
            results=results,
            results_csv=write_results_csv(results, args.outdir),
            comparison_png=write_comparison_png(results, args.outdir),
        )
    else:
        outcome = run_benchmark(args.engines, args.workdir, args.outdir, args.code_dir)

    print("\nPhase-1 benchmark results")
    print("=" * 60)
    for r in outcome.results:
        line = f"{r.engine:7s} {r.status:8s}"
        if r.status == "ok":
            line += f" KGE_eval={r.kge:.4f}  NSE_eval={r.nse:.4f}  KGE_calib={r.kge_calib:.4f}"
        elif r.note:
            line += f" {r.note}"
        print(line)
    print("=" * 60)
    if outcome.results_csv:
        print(f"results.csv    -> {outcome.results_csv}")
    if outcome.comparison_png:
        print(f"comparison.png -> {outcome.comparison_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
