"""WaterSplatting trainer compatibility helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch

from nerfstudio.engine.trainer import Trainer
from nerfstudio.utils.rich_utils import CONSOLE


class WaterSplattingTrainer(Trainer):
    """Trainer with checkpoint compatibility for archived research branches.

    Some archived BND checkpoints contain optimizer/scheduler state for optimizer
    groups that are no longer present on the clean M1+BND branch. Filtering only
    absent groups preserves current-group resume semantics while allowing those
    checkpoints to be used as read-only continuation sources.
    """

    @staticmethod
    def _filter_state_dicts(loaded_state: Dict[str, Any], current_keys: set[str], field: str) -> Dict[str, Any]:
        state = loaded_state.get(field, {})
        extra = sorted(set(state.keys()) - current_keys)
        missing = sorted(current_keys - set(state.keys()))
        if extra:
            CONSOLE.log(f"Ignoring checkpoint {field} groups not present in current model: {extra}")
        if missing:
            CONSOLE.log(f"Checkpoint is missing current {field} groups; they will keep initialized state: {missing}")
        return {key: value for key, value in state.items() if key in current_keys}

    def _load_checkpoint(self) -> None:
        """Load pipeline, optimizer, scheduler, and scaler state with group filtering."""

        load_path: Path | None = None
        if self.config.load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                load_step = sorted(
                    int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(self.config.load_dir)
                )[-1]
            load_path = self.config.load_dir / f"step-{load_step:09d}.ckpt"
        elif self.config.load_checkpoint is not None:
            load_path = self.config.load_checkpoint

        if load_path is None:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
            return

        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        self._start_step = loaded_state["step"] + 1
        self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        # Loading splat checkpoints can replace Gaussian Parameter objects to
        # match the saved population size. Rebuild optimizers after pipeline load
        # so optimizer param_groups point at the live model parameters.
        self.optimizers = self.setup_optimizers()

        optimizer_keys = set(self.optimizers.optimizers.keys())
        self.optimizers.load_optimizers(self._filter_state_dicts(loaded_state, optimizer_keys, "optimizers"))
        if "schedulers" in loaded_state and self.config.load_scheduler:
            scheduler_keys = set(self.optimizers.schedulers.keys())
            self.optimizers.load_schedulers(self._filter_state_dicts(loaded_state, scheduler_keys, "schedulers"))
        self.grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
