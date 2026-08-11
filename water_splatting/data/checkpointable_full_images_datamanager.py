"""Full-image datamanager with checkpointed camera sampling state."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Type

import numpy as np
import torch

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE


def _object_to_uint8_tensor(value: Any) -> torch.Tensor:
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return torch.tensor(list(data), dtype=torch.uint8)


def _uint8_tensor_to_object(value: torch.Tensor) -> Any:
    data = bytes(value.detach().cpu().numpy().astype(np.uint8).tolist())
    return pickle.loads(data)


@dataclass
class CheckpointableFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    """Config for the checkpointable full-image datamanager."""

    _target: Type = field(default_factory=lambda: CheckpointableFullImageDatamanager)
    load_depths: bool = False
    """Load dataparser depth images into batches. Disabled preserves historical WaterSplatting behavior."""


class CheckpointableFullImageDatamanager(FullImageDatamanager[InputDataset]):
    """Persist full-image camera sampler state across Nerfstudio checkpoints.

    Nerfstudio's stock ``FullImageDatamanager`` stores the random camera sampler
    as Python-side state, so resumed training restarts the camera permutation
    sequence from the seed instead of continuing the saved trajectory. This
    subclass serializes the sampler state into the module state_dict so
    checkpoint/resume experiments can share the same post-checkpoint image order.
    """

    _STATE_PREFIX = "_ws_full_image_sampler_"

    @cached_property
    def dataset_type(self) -> Type[InputDataset]:
        """Select the standard image dataset unless a diagnostic explicitly asks for depth images."""

        if self.config.load_depths:
            return DepthDataset
        return super().dataset_type

    def _load_images(self, split: str, cache_images_device: str) -> List[Dict[str, torch.Tensor]]:
        """Load full-image caches and keep optional depth tensors on the same cache device."""

        caches = super()._load_images(split, cache_images_device)
        if cache_images_device == "gpu":
            for cache in caches:
                if "depth_image" in cache:
                    cache["depth_image"] = cache["depth_image"].to(self.device)
        elif cache_images_device == "cpu":
            for cache in caches:
                if "depth_image" in cache:
                    cache["depth_image"] = cache["depth_image"].pin_memory()
        return caches

    def _save_to_state_dict(
        self,
        destination: Dict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + self._STATE_PREFIX + "train_unseen_cameras"] = torch.as_tensor(
            list(getattr(self, "train_unseen_cameras", [])),
            dtype=torch.long,
        )
        destination[prefix + self._STATE_PREFIX + "eval_unseen_cameras"] = torch.as_tensor(
            list(getattr(self, "eval_unseen_cameras", [])),
            dtype=torch.long,
        )
        if hasattr(self, "random_generator"):
            destination[prefix + self._STATE_PREFIX + "random_generator_state"] = _object_to_uint8_tensor(
                self.random_generator.getstate()
            )
        if hasattr(self, "train_unsampled_epoch_count"):
            destination[prefix + self._STATE_PREFIX + "train_unsampled_epoch_count"] = torch.as_tensor(
                self.train_unsampled_epoch_count,
                dtype=torch.float64,
            )

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        train_unseen_key = prefix + self._STATE_PREFIX + "train_unseen_cameras"
        eval_unseen_key = prefix + self._STATE_PREFIX + "eval_unseen_cameras"
        rng_key = prefix + self._STATE_PREFIX + "random_generator_state"
        fps_key = prefix + self._STATE_PREFIX + "train_unsampled_epoch_count"

        train_unseen = state_dict.pop(train_unseen_key, None)
        eval_unseen = state_dict.pop(eval_unseen_key, None)
        rng_state = state_dict.pop(rng_key, None)
        fps_state = state_dict.pop(fps_key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if train_unseen is not None:
            self.train_unseen_cameras = [int(v) for v in train_unseen.detach().cpu().reshape(-1).tolist()]
        if eval_unseen is not None:
            self.eval_unseen_cameras = [int(v) for v in eval_unseen.detach().cpu().reshape(-1).tolist()]
        if rng_state is not None:
            self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            self.random_generator.setstate(_uint8_tensor_to_object(rng_state))
        if fps_state is not None:
            self.train_unsampled_epoch_count = fps_state.detach().cpu().numpy().astype(np.float64)

        if train_unseen is not None or rng_state is not None or fps_state is not None:
            CONSOLE.log(
                "Restored full-image datamanager sampler state "
                f"({len(getattr(self, 'train_unseen_cameras', []))} unseen train cameras)"
            )
