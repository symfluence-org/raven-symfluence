# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Calibration Worker.

Worker implementation for Raven emulator calibration. Mirrors the dRoute/FUSE
workers: ``apply_parameters`` -> ``run_model`` -> ``calculate_metrics``.

Each evaluation rebuilds the Raven model with the trial parameters via RavenPy
(which writes the ``.rvi/.rvp/.rvh/.rvt/.rvc`` set and drives the ``raven`` binary)
and reads ``Hydrographs.csv`` for the objective. RavenPy is imported lazily inside
the methods so the worker (and the plugin) register without it installed.

Multi-gauge calibration (``MULTI_GAUGE_CALIBRATION``) is delegated to
``symfluence.optimization.multi_gauge.metrics.MultiGaugeMetrics`` exactly like FUSE.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.constants import ModelDefaults
from symfluence.evaluation.metrics import kge, nse
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask


class RavenWorker(BaseWorker):
    """Worker for Raven emulator calibration (Phase 1: lumped GR4JCN, single-gauge KGE)."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        self._current_params: Dict[str, float] = {}

    # =========================================================================
    # Parameter application
    # =========================================================================

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """Persist the trial parameters via the parameter manager.

        RavenPy rebuilds the ``.rv*`` files from the parameter vector at run time, so
        we only record the trial params (raven_params.json) and stash them in-memory.
        """
        try:
            config = kwargs.get('config', self.config)
            self._current_params = dict(params)
            from .parameter_manager import RavenParameterManager
            pm = RavenParameterManager(config, self.logger, Path(settings_dir))
            return pm.update_model_files(params)
        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error applying Raven parameters: {e}")
            return False

    # =========================================================================
    # Model execution
    # =========================================================================

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """Rebuild + run the Raven emulator for the current trial parameters via RavenPy."""
        try:
            sim_dir = kwargs.get('sim_dir') or output_dir
            sim_dir = Path(sim_dir)
            sim_dir.mkdir(parents=True, exist_ok=True)

            params = kwargs.get('params') or getattr(self, '_current_params', {})

            # Lazy import: RavenPy must not be required to register the worker.
            try:
                from ..emulator import build_and_run_emulator
            except ImportError as e:
                self.logger.error(f"RavenPy not available for calibration run: {e}")
                return False

            ok = build_and_run_emulator(
                config=config,
                settings_dir=Path(settings_dir),
                output_dir=sim_dir,
                params=params,
                logger=self.logger,
            )
            if not ok:
                self.logger.error("RavenPy emulator run failed")
                return False
            return True
        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error running Raven: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    # =========================================================================
    # Metrics
    # =========================================================================

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Compute KGE/NSE from Hydrographs.csv against preprocessed observations."""
        if config.get('MULTI_GAUGE_CALIBRATION', False):
            return self._calculate_multi_gauge_metrics(output_dir, config, **kwargs)

        try:
            import numpy as np

            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            sim_dir = Path(kwargs.get('sim_dir') or output_dir)
            from ..postprocessor import find_hydrographs_file, read_raven_streamflow
            hydro_file = find_hydrographs_file(sim_dir)
            if hydro_file is None:
                self.logger.error(f"Raven Hydrographs.csv not found in {sim_dir}")
                return {'kge': self.penalty_score}

            simulated = read_raven_streamflow(hydro_file, config, self.logger)
            if simulated is None or simulated.empty:
                return {'kge': self.penalty_score}

            observed = self._load_observations(project_dir, domain_name)
            if observed is None or observed.empty:
                self.logger.error("No observations available for Raven metrics")
                return {'kge': self.penalty_score}

            # Align on a common daily index.
            obs = observed.resample('D').mean()
            sim = simulated.resample('D').mean()
            joined = obs.to_frame('obs').join(sim.to_frame('sim'), how='inner')
            joined = joined.dropna()
            if len(joined) < 10:
                return {'kge': self.penalty_score, 'error': 'Insufficient overlap'}

            obs_v = joined['obs'].values
            sim_v = joined['sim'].values
            kge_val = kge(obs_v, sim_v, transfo=1)
            nse_val = nse(obs_v, sim_v, transfo=1)
            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {'kge': float(kge_val), 'nse': float(nse_val), 'n_points': int(len(joined))}

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error calculating Raven metrics: {e}")
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def _load_observations(self, project_dir: Path, domain_name: str):
        """Load preprocessed streamflow observations (m3/s) as a pandas Series.

        Resolves the preprocessed CSV through ``resolve_data_subdir`` so it works
        whether observations live under ``data/observations/`` (the model-ready
        layout) or the legacy ``observations/`` location — matching every other
        SYMFLUENCE worker/evaluator (inmemory_worker, summa metrics, streamflow
        evaluator). The previous hard-coded ``project_dir/observations`` path
        silently returned None on any domain using the ``data/`` layout.
        """
        import pandas as pd
        from symfluence.core.mixins.project import resolve_data_subdir

        obs_file = (resolve_data_subdir(project_dir, 'observations')
                    / 'streamflow' / 'preprocessed'
                    / f"{domain_name}_streamflow_processed.csv")
        if not obs_file.exists():
            return None
        df = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        if df.empty:
            return None
        # First column is discharge (matches the dRoute worker convention).
        return df.iloc[:, 0].astype(float)

    # =========================================================================
    # Multi-gauge (delegates to MultiGaugeMetrics, like FUSE)
    # =========================================================================

    def _calculate_multi_gauge_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Multi-gauge KGE objective via the shared MultiGaugeMetrics helper.

        Phase 1 targets lumped GR4JCN (single outlet), so multi-gauge is wired but
        gated: it requires GAUGE_SEGMENT_MAPPING + MULTI_GAUGE_OBS_DIR and a routed
        per-reach Hydrographs output, available once distributed Raven lands.
        """
        try:
            from symfluence.optimization.multi_gauge.metrics import MultiGaugeMetrics

            gauge_mapping_path = config.get('GAUGE_SEGMENT_MAPPING')
            obs_dir = config.get('MULTI_GAUGE_OBS_DIR')
            if not gauge_mapping_path or not obs_dir:
                self.logger.error(
                    "Multi-gauge Raven calibration requires GAUGE_SEGMENT_MAPPING "
                    "and MULTI_GAUGE_OBS_DIR")
                return {'kge': self.penalty_score}

            sim_dir = Path(kwargs.get('sim_dir') or output_dir)
            from ..postprocessor import find_hydrographs_file
            hydro_file = find_hydrographs_file(sim_dir)
            if hydro_file is None:
                return {'kge': self.penalty_score}

            calib_period = config.get('CALIBRATION_PERIOD', '')
            start_date = end_date = None
            if calib_period and ',' in str(calib_period):
                start_date, end_date = (s.strip() for s in str(calib_period).split(',')[:2])

            multi_gauge = MultiGaugeMetrics(
                gauge_segment_mapping_path=Path(gauge_mapping_path),
                obs_data_dir=Path(obs_dir),
                logger=self.logger,
            )
            results = multi_gauge.calculate_multi_gauge_metrics(
                mizuroute_output_path=hydro_file,
                gauge_ids=config.get('MULTI_GAUGE_IDS'),
                start_date=start_date,
                end_date=end_date,
                aggregation=config.get('MULTI_GAUGE_AGGREGATION', 'mean'),
                min_overlap_days=int(config.get('MULTI_GAUGE_MIN_OVERLAP_DAYS', 10)),
            )
            return {
                'kge': results['kge'],
                'n_gauges': results.get('n_valid_gauges', 0),
            }
        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error in Raven multi-gauge metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score}

    # =========================================================================
    # Static worker function (process pool)
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_raven_parameters_worker(task_data)


def _evaluate_raven_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    try:
        worker = RavenWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 -- calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'Raven worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }


__all__ = ['RavenWorker']
