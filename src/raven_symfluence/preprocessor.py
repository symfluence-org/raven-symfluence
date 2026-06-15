# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Raven Model Preprocessor.

Converts SYMFLUENCE's model-ready datastore (canonical CFIF forcing + per-HRU
catchment attributes) into Raven-native inputs and writes the ``.rvi/.rvp/.rvh/.rvt/.rvc``
files (via RavenPy) into ``settings/RAVEN``.

Follows the SYMFLUENCE BaseModelPreProcessor template: ``run_preprocessing`` delegates
to ``run_preprocessing_template`` (which calls ``_prepare_forcing`` then
``_create_model_configs``). RavenPy and other heavy deps are imported lazily inside
the hooks, so the class imports/registers without RavenPy installed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Union

from symfluence.models.base import BaseModelPreProcessor


class RavenPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Preprocessor for the Raven hydrological modelling framework (RavenPy-driven)."""

    MODEL_NAME = "RAVEN"

    def __init__(self, config: Union[Dict[str, Any], Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.logger.debug(f"RavenPreProcessor initialized. setup_dir: {self.setup_dir}")
        # The basin-averaged daily Raven forcing is written here during _prepare_forcing.
        self._raven_forcing_frame = None
        self._raven_forcing_dataset = None

    def run_preprocessing(self) -> bool:
        """Run the standard preprocessing template (forcing prep + config generation)."""
        return self.run_preprocessing_template()

    # =========================================================================
    # Template hooks
    # =========================================================================

    def _prepare_forcing(self) -> None:
        """Read canonical CFIF forcing and convert it to a daily Raven forcing frame.

        The frame (PRECIP mm/d, TEMP_AVE degC) is cached on the instance so
        ``_create_model_configs`` can hand it to RavenPy. The canonical dataset is
        read via the model-ready forcing reader (CARRA/ERA5-agnostic, CF names).
        """
        from .emulator import build_raven_forcing_frame, load_forcing

        ds = load_forcing(self.forcing_basin_path, self.logger)
        if ds is None:
            self.logger.warning(
                f"No Raven forcing found in {self.forcing_basin_path}; "
                "config generation will be skipped")
            return
        try:
            self._raven_forcing_dataset = ds
            self._raven_forcing_frame = build_raven_forcing_frame(ds, self.logger)
            self.logger.info(
                f"Prepared Raven forcing: {len(self._raven_forcing_frame)} daily steps")
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.error(f"Failed to prepare Raven forcing: {e}")
            self._raven_forcing_frame = None

    def _create_model_configs(self) -> None:
        """Write the Raven ``.rv*`` configuration files via RavenPy.

        Resolves the (lumped) HRU attributes from the model-ready attributes store
        (falling back to the catchment shapefile), assembles initial parameters, and
        builds the emulator config into ``setup_dir``. RavenPy is imported lazily by
        :func:`raven_symfluence.emulator.build_and_run_emulator`.
        """
        if self._raven_forcing_frame is None:
            self.logger.warning("Skipping Raven config generation (no forcing prepared)")
            return

        from .emulator import build_and_run_emulator

        # Use the parameter manager's initial parameters so the generated config is
        # immediately runnable (and matches what calibration will perturb).
        params = self._initial_parameters()

        # build_and_run_emulator both writes the .rv* files and performs the initial
        # default run; the run output lands under the experiment simulation dir.
        sim_dir = (self.project_dir / 'simulations'
                   / self.experiment_id / self.model_name)
        try:
            ok = build_and_run_emulator(
                config=self.config_dict,
                settings_dir=self.setup_dir,
                output_dir=sim_dir,
                params=params,
                logger=self.logger,
                forcing_basin_path=self.forcing_basin_path,
                catchment_shapefile=self._catchment_shapefile(),
            )
            if ok:
                # RavenPy writes the .rv* set into the emulator workdir (the sim dir),
                # not the settings dir. Mirror them into settings/RAVEN so the
                # documented contract holds and the generated config is inspectable
                # there alongside the forcing NetCDF + raven_params.json sidecar.
                self._mirror_rv_files(sim_dir)
                self.logger.info(f"Raven .rv* files written to {self.setup_dir}")
            else:
                self.logger.warning(
                    "Raven config generation did not complete "
                    "(RavenPy missing or build failed)")
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.error(f"Failed to generate Raven configs: {e}")

    def _post_setup(self) -> None:
        """Close the cached forcing dataset."""
        if self._raven_forcing_dataset is not None:
            try:
                self._raven_forcing_dataset.close()
            except Exception:  # noqa: BLE001 -- best-effort close
                pass
            self._raven_forcing_dataset = None

    # =========================================================================
    # Helpers
    # =========================================================================

    def _initial_parameters(self) -> Dict[str, float]:
        """Initial parameter values from the Raven parameter manager (bounds midpoint)."""
        try:
            from .calibration.parameter_manager import RavenParameterManager

            pm = RavenParameterManager(self.config, self.logger, self.setup_dir)
            initial = pm.get_initial_parameters()
            return initial or {}
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.debug(f"Could not resolve initial Raven parameters: {e}")
            return {}

    def _mirror_rv_files(self, sim_dir) -> None:
        """Copy the RavenPy-generated ``.rv*`` files from the run dir into settings/RAVEN.

        RavenPy emits the ``.rvi/.rvp/.rvh/.rvt/.rvc/.rve`` set into the emulator
        workdir; SYMFLUENCE's convention is that ``settings/<MODEL>`` holds the model
        config. Copies (not symlinks) so the settings copy survives a cleaned sim dir.
        """
        import shutil
        from pathlib import Path

        sim_dir = Path(sim_dir)
        rv_files = sorted(p for p in sim_dir.glob('*.rv*') if p.is_file())
        if not rv_files:
            self.logger.debug(f"No .rv* files found in {sim_dir} to mirror into settings")
            return
        self.setup_dir.mkdir(parents=True, exist_ok=True)
        for src in rv_files:
            try:
                shutil.copy2(str(src), str(self.setup_dir / src.name))
            except Exception as e:  # noqa: BLE001 -- best-effort mirror
                self.logger.debug(f"Could not mirror {src.name} into settings: {e}")

    def _catchment_shapefile(self):
        """Resolve the catchment shapefile (used as the HRU-attribute fallback)."""
        try:
            return self.get_catchment_path()
        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.debug(f"No catchment shapefile resolved: {e}")
            return None


__all__ = ['RavenPreProcessor']
