# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Multi-engine benchmark driver.

Two modes:

1. **Synthetic-fixture mode** (default, no args) — the original lightweight
   self-check: fabricate per-engine synthetic discharge against a synthetic
   observation series, score KGE/NSE, and emit ``results.csv`` + ``comparison.png``.
   This mode needs no model binaries and is safe to run anywhere (CI).

2. **Real-domain mode** (``--domain <path>``) — run one or more first-class
   SYMFLUENCE models through the *standard* workflow on a real domain. Each
   engine's per-engine YAML config (``configs/config_<engine>.yaml``) is driven
   through ``model_specific_preprocessing -> run_model -> postprocess_results ->
   calibrate_model``. After calibration, the per-engine eval/calib KGE & NSE are
   read from ``optimization/<MODEL>/dds_<EXP>/<EXP>_dds_final_evaluation.json``
   and assembled into ``results.csv`` + ``comparison.png``.

   The point of real-domain mode: SUMMA / FUSE / Raven are first-class
   SYMFLUENCE models, so each one's ``model_specific_preprocessing`` builds its
   native inputs from the *shared* model-agnostic ``data/model_ready`` store.

Examples
--------
    # synthetic self-check
    python run_benchmark.py

    # real multi-engine run on an isolated benchmark domain
    python run_benchmark.py \
        --domain /path/to/SYMFLUENCE_data/domain_multimodel_benchmark \
        --engines raven fuse summa \
        --steps model_specific_preprocessing run_model postprocess_results calibrate_model
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"

# engine -> (config filename, SYMFLUENCE model key used in the optimization dir)
ENGINES = {
    "raven": ("config_raven.yaml", "RAVEN"),
    "fuse": ("config_fuse.yaml", "FUSE"),
    "summa": ("config_summa.yaml", "SUMMA"),
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency (2009)."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if sim.size < 2 or obs.std() == 0:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if sim.size < 2:
        return float("nan")
    denom = ((obs - obs.mean()) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(1 - ((sim - obs) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# synthetic-fixture mode
# --------------------------------------------------------------------------- #
def run_synthetic(out_dir: Path) -> pd.DataFrame:
    """Original synthetic self-check: no binaries, deterministic."""
    rng = np.random.default_rng(42)
    n = 730
    t = np.arange(n)
    base = 5 + 4 * np.sin(2 * np.pi * t / 365.25) ** 2
    obs = base + rng.normal(0, 0.5, n)
    obs = np.clip(obs, 0.05, None)

    rows = []
    # each pseudo-engine = obs perturbed by a different bias/noise profile
    profiles = {
        "raven": (1.00, 0.6),
        "fuse": (1.10, 0.9),
        "summa": (0.95, 0.7),
    }
    for name, (bias, noise) in profiles.items():
        sim = np.clip(base * bias + rng.normal(0, noise, n), 0.01, None)
        rows.append(
            {
                "engine": name,
                "ran": True,
                "eval_KGE": kge(sim, obs),
                "eval_NSE": nse(sim, obs),
                "calib_KGE": kge(sim, obs),
                "calib_NSE": nse(sim, obs),
                "source": "synthetic",
            }
        )
    df = pd.DataFrame(rows)
    _write_artifacts(df, out_dir, title="Synthetic multi-engine benchmark")
    return df


# --------------------------------------------------------------------------- #
# real-domain mode
# --------------------------------------------------------------------------- #
def _run_steps(config: Path, steps: list[str], python_exe: str) -> tuple[bool, str]:
    """Drive the standard SYMFLUENCE workflow for one engine config.

    Calibration (calibrate_model) is run as its own ``workflow step`` so a
    failure there doesn't mask a successful baseline run.
    """
    base = [python_exe, "-m", "symfluence.main_cli", "workflow"]
    # the pre-calibration steps run together; calibrate_model runs separately
    pre = [s for s in steps if s != "calibrate_model"]
    log_tail = ""
    if pre:
        cmd = base + ["steps", *pre, "--config", str(config)]
        proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        log_tail = proc.stdout[-2000:] + proc.stderr[-2000:]
        if proc.returncode != 0:
            return False, log_tail
    if "calibrate_model" in steps:
        cmd = base + ["step", "calibrate_model", "--config", str(config)]
        proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        log_tail = proc.stdout[-2000:] + proc.stderr[-2000:]
        if proc.returncode != 0:
            return False, log_tail
    return True, log_tail


def _read_final_eval(domain_dir: Path, model_key: str, experiment_id: str) -> dict | None:
    """Find ``<EXP>_dds_final_evaluation.json`` under optimization/<MODEL>/."""
    opt_root = domain_dir / "optimization" / model_key
    if not opt_root.exists():
        return None
    candidates = sorted(opt_root.glob(f"**/{experiment_id}_dds_final_evaluation.json"))
    if not candidates:
        candidates = sorted(opt_root.glob("**/*_dds_final_evaluation.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def _experiment_id(config: Path) -> str:
    import yaml

    cfg = yaml.safe_load(config.read_text())
    return cfg.get("EXPERIMENT_ID", config.stem)


def run_real(
    domain_dir: Path,
    engines: list[str],
    steps: list[str],
    python_exe: str,
    out_dir: Path,
    run_workflow: bool = True,
) -> pd.DataFrame:
    rows = []
    for name in engines:
        cfg_name, model_key = ENGINES[name]
        config = CONFIG_DIR / cfg_name
        if not config.exists():
            print(f"[{name}] config not found: {config} -- skipping", file=sys.stderr)
            continue
        exp = _experiment_id(config)
        if run_workflow:
            print(f"\n=== [{name}] standard workflow: {' -> '.join(steps)} ===", flush=True)
            ok, tail = _run_steps(config, steps, python_exe)
        else:
            # collect-only: assume the workflow already ran; read final_evaluation.json
            print(f"\n=== [{name}] collect-only (reading final evaluation) ===", flush=True)
            ok, tail = True, ""
        final = _read_final_eval(domain_dir, model_key, exp)
        row = {"engine": name, "ran": ok and final is not None, "source": "real_domain"}
        if final:
            ev = final.get("evaluation_metrics", {})
            ca = final.get("calibration_metrics", {})
            row.update(
                eval_KGE=ev.get("KGE"),
                eval_NSE=ev.get("NSE"),
                calib_KGE=ca.get("KGE"),
                calib_NSE=ca.get("NSE"),
            )
        else:
            row.update(eval_KGE=np.nan, eval_NSE=np.nan, calib_KGE=np.nan, calib_NSE=np.nan)
            if not ok:
                print(f"[{name}] workflow failed; log tail:\n{tail}", file=sys.stderr)
        rows.append(row)

    df = pd.DataFrame(rows)
    _write_artifacts(df, out_dir, title=f"Multi-engine benchmark — {domain_dir.name}")
    return df


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def _write_artifacts(df: pd.DataFrame, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = df.dropna(subset=["eval_KGE"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if not plot_df.empty:
            x = np.arange(len(plot_df))
            w = 0.38
            ax.bar(x - w / 2, plot_df["eval_KGE"], w, label="eval KGE")
            ax.bar(x + w / 2, plot_df["eval_NSE"], w, label="eval NSE")
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["engine"])
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel("metric")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no engine produced an eval metric", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        png_path = out_dir / "comparison.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except Exception as e:  # noqa: BLE001 -- plotting is best-effort
        print(f"Could not write comparison.png: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", type=Path, default=None, help="Real domain dir; enables real-domain mode.")
    p.add_argument("--engines", nargs="+", default=["raven", "fuse", "summa"], choices=list(ENGINES))
    p.add_argument(
        "--steps",
        nargs="+",
        default=["model_specific_preprocessing", "run_model", "postprocess_results", "calibrate_model"],
    )
    p.add_argument("--python", default=sys.executable, help="Python interpreter that has symfluence installed.")
    p.add_argument("--out", type=Path, default=HERE / "results", help="Output dir for results.csv + comparison.png.")
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Real-domain mode: skip workflow execution, only read final_evaluation.json and assemble artifacts.",
    )
    args = p.parse_args(argv)

    if args.domain is None:
        run_synthetic(args.out)
    else:
        run_real(
            args.domain.resolve(), args.engines, args.steps, args.python, args.out, run_workflow=not args.no_run
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
