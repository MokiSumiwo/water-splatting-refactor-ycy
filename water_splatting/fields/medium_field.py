"""Medium field helpers.

This module intentionally does not own trainable modules.  The model keeps the
original top-level ``direction_encoding`` and ``medium_mlp`` attributes so
existing checkpoints remain loadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras


@dataclass
class MediumFieldOutput:
    """Per-pixel medium parameters used by the underwater rasterizer."""

    rgb: Tensor
    bs: Tensor
    attn: Tensor
    directions: Tensor
    b_inf: Optional[Tensor] = None


MediumContextMode = Literal[
    "dir_only",
    "dir_xy",
    "dir_xy_camera",
]


def get_medium_context_extra_dim(mode: MediumContextMode) -> int:
    """Return additional non-direction feature channels for a medium mode."""

    if mode == "dir_only":
        return 0
    if mode == "dir_xy":
        return 3
    if mode == "dir_xy_camera":
        return 6
    raise ValueError(f"Unknown medium_context_mode: {mode}")


class DirectionConditionedMediumField:
    """Original WaterSplatting direction-conditioned medium field wrapper."""

    def __init__(
        self,
        direction_encoding: nn.Module,
        medium_mlp: nn.Module,
        colour_activation: nn.Module,
        sigma_activation: nn.Module,
    ) -> None:
        self.direction_encoding = direction_encoding
        self.medium_mlp = medium_mlp
        self.colour_activation = colour_activation
        self.sigma_activation = sigma_activation

    def __call__(
        self,
        camera: Cameras,
        rotation_world_from_camera: Tensor,
        height: int,
        width: int,
        cx: float,
        cy: float,
        density_bias: float,
        mlp_type: Literal["tcnn", "torch"],
        zero_medium: bool,
        context_mode: MediumContextMode = "dir_only",
        camera_center: Optional[Tensor] = None,
        scene_center: Optional[Tensor] = None,
        scene_scale: Optional[Union[Tensor, float]] = None,
        camera_context_scale: float = 1.0,
        camera_context_dropout: float = 0.0,
        training: bool = False,
        b_inf_mode: Literal["implicit", "tied"] = "implicit",
    ) -> MediumFieldOutput:
        y = torch.linspace(0.0, height, height, device=rotation_world_from_camera.device)
        x = torch.linspace(0.0, width, width, device=rotation_world_from_camera.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ rotation_world_from_camera.T

        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)
        image_y = torch.linspace(-1.0, 1.0, height, device=rotation_world_from_camera.device, dtype=directions.dtype)
        image_x = torch.linspace(-1.0, 1.0, width, device=rotation_world_from_camera.device, dtype=directions.dtype)
        image_yy, image_xx = torch.meshgrid(image_y, image_x, indexing="ij")
        mlp_input = self._append_context(
            directions_encoded=directions_encoded,
            image_xx=image_xx,
            image_yy=image_yy,
            height=height,
            width=width,
            mode=context_mode,
            camera_center=camera_center,
            scene_center=scene_center,
            scene_scale=scene_scale,
            camera_context_scale=camera_context_scale,
            camera_context_dropout=camera_context_dropout,
            training=training,
        )
        outputs_shape = directions.shape[:-1]

        if mlp_type == "tcnn":
            medium_base_out = self.medium_mlp(mlp_input)
        else:
            medium_base_out = self.medium_mlp(mlp_input.float())

        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3:6] + density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )
        medium_attn = (
            self.sigma_activation(medium_base_out[..., 6:9] + density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )

        if zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        b_inf = None
        if b_inf_mode == "implicit":
            b_inf = None
        elif b_inf_mode == "tied":
            b_inf = medium_rgb
        else:
            raise ValueError(f"Unknown b_inf_mode: {b_inf_mode}")

        return MediumFieldOutput(
            rgb=medium_rgb,
            bs=medium_bs,
            attn=medium_attn,
            directions=directions,
            b_inf=b_inf,
        )

    def _append_context(
        self,
        *,
        directions_encoded: Tensor,
        image_xx: Tensor,
        image_yy: Tensor,
        height: int,
        width: int,
        mode: MediumContextMode,
        camera_center: Optional[Tensor],
        scene_center: Optional[Tensor],
        scene_scale: Optional[Union[Tensor, float]],
        camera_context_scale: float,
        camera_context_dropout: float,
        training: bool,
    ) -> Tensor:
        if mode == "dir_only":
            return directions_encoded

        r = torch.sqrt(image_xx.square() + image_yy.square())
        features = [directions_encoded, torch.stack([image_xx, image_yy, r], dim=-1).view(-1, 3)]

        if "camera" in mode:
            if camera_center is None:
                camera_feature = torch.zeros(3, device=image_xx.device, dtype=image_xx.dtype)
            else:
                camera_feature = camera_center.reshape(-1, 3)[0].to(device=image_xx.device, dtype=image_xx.dtype)
                if scene_center is not None:
                    scene_center = scene_center.to(device=image_xx.device, dtype=image_xx.dtype)
                    if scene_scale is None:
                        scene_scale = torch.tensor(1.0, device=image_xx.device, dtype=image_xx.dtype)
                    elif not isinstance(scene_scale, Tensor):
                        scene_scale = torch.tensor(scene_scale, device=image_xx.device, dtype=image_xx.dtype)
                    else:
                        scene_scale = scene_scale.to(device=image_xx.device, dtype=image_xx.dtype)
                    camera_feature = (camera_feature - scene_center) / (scene_scale + 1e-6)
                camera_feature = camera_feature * camera_context_scale
            if camera_context_dropout > 0.0:
                camera_feature = F.dropout(camera_feature, p=camera_context_dropout, training=training)
            features.append(camera_feature.expand(height * width, 3))

        return torch.cat(features, dim=-1)
