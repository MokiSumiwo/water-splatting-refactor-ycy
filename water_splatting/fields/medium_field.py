"""Medium field helpers.

This module intentionally does not own trainable modules.  The model keeps the
original top-level ``direction_encoding`` and ``medium_mlp`` attributes so
existing checkpoints remain loadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

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
    b_inf_residual: Optional[Tensor] = None


MediumContextMode = Literal[
    "dir_only",
    "dir_xy",
    "dir_xy_depth",
    "dir_xy_camera",
    "dir_xy_depth_camera",
]


def get_medium_context_extra_dim(mode: MediumContextMode) -> int:
    """Return additional non-direction feature channels for a medium mode."""

    if mode == "dir_only":
        return 0
    if mode == "dir_xy":
        return 3
    if mode == "dir_xy_depth":
        return 4
    if mode == "dir_xy_camera":
        return 6
    if mode == "dir_xy_depth_camera":
        return 7
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
        depth_context: Optional[Tensor] = None,
        enable_b_inf: bool = False,
        b_inf_mode: Literal["implicit", "tied", "bounded_residual", "independent"] = "implicit",
        b_inf_residual_scale: float = 0.02,
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
            depth_context=depth_context,
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
        b_inf_raw = medium_base_out[..., 9:12] if enable_b_inf else None

        if zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        b_inf = None
        b_inf_residual = None
        if b_inf_mode == "implicit":
            b_inf = None
        elif b_inf_mode == "tied":
            b_inf = medium_rgb
        elif b_inf_mode == "independent":
            if b_inf_raw is None:
                raise RuntimeError("b_inf_mode='independent' requires enable_b_inf=True")
            b_inf = self.colour_activation(b_inf_raw).view(*outputs_shape, -1).to(directions)
        elif b_inf_mode == "bounded_residual":
            if b_inf_raw is None:
                raise RuntimeError("b_inf_mode='bounded_residual' requires enable_b_inf=True")
            b_inf_residual = torch.tanh(b_inf_raw).view(*outputs_shape, -1).to(directions)
            rgb_logit = torch.logit(medium_rgb.clamp(1e-4, 1.0 - 1e-4))
            b_inf = torch.sigmoid(rgb_logit + float(b_inf_residual_scale) * b_inf_residual)
        else:
            raise ValueError(f"Unknown b_inf_mode: {b_inf_mode}")

        return MediumFieldOutput(
            rgb=medium_rgb,
            bs=medium_bs,
            attn=medium_attn,
            directions=directions,
            b_inf=b_inf,
            b_inf_residual=b_inf_residual,
        )

    def query_points(
        self,
        *,
        directions: Tensor,
        image_xy: Tensor,
        camera_centers: Optional[Tensor],
        density_bias: float,
        mlp_type: Literal["tcnn", "torch"],
        zero_medium: bool,
        context_mode: MediumContextMode = "dir_only",
        scene_center: Optional[Tensor] = None,
        scene_scale: Optional[Union[Tensor, float]] = None,
        camera_context_scale: float = 1.0,
        camera_context_dropout: float = 0.0,
        training: bool = False,
        depth_context: Optional[Tensor] = None,
        enable_b_inf: bool = False,
        b_inf_mode: Literal["implicit", "tied", "bounded_residual", "independent"] = "implicit",
        b_inf_residual_scale: float = 0.02,
    ) -> MediumFieldOutput:
        """Query medium parameters for arbitrary rays without rasterizing a full image.

        ``image_xy`` is normalized to the same ``[-1, 1]`` convention used by the
        full image path: ``x = 2 * pixel_x / (width - 1) - 1`` and likewise for y.
        """

        if directions.ndim != 2 or directions.shape[-1] != 3:
            raise ValueError(f"directions must be Nx3, got {tuple(directions.shape)}")
        if image_xy.ndim != 2 or image_xy.shape[-1] != 2:
            raise ValueError(f"image_xy must be Nx2 normalized coordinates, got {tuple(image_xy.shape)}")
        if directions.shape[0] != image_xy.shape[0]:
            raise ValueError("directions and image_xy must have the same row count")

        directions = directions.to(dtype=image_xy.dtype)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)
        directions_encoded = self.direction_encoding(directions)
        mlp_input = self._append_point_context(
            directions_encoded=directions_encoded,
            image_xy=image_xy,
            mode=context_mode,
            camera_centers=camera_centers,
            scene_center=scene_center,
            scene_scale=scene_scale,
            camera_context_scale=camera_context_scale,
            camera_context_dropout=camera_context_dropout,
            training=training,
            depth_context=depth_context,
        )

        if mlp_type == "tcnn":
            medium_base_out = self.medium_mlp(mlp_input)
        else:
            medium_base_out = self.medium_mlp(mlp_input.float())

        medium_rgb = self.colour_activation(medium_base_out[..., :3]).to(directions)
        medium_bs = self.sigma_activation(medium_base_out[..., 3:6] + density_bias).to(directions)
        medium_attn = self.sigma_activation(medium_base_out[..., 6:9] + density_bias).to(directions)
        b_inf_raw = medium_base_out[..., 9:12] if enable_b_inf else None

        if zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        b_inf = None
        b_inf_residual = None
        if b_inf_mode == "implicit":
            b_inf = None
        elif b_inf_mode == "tied":
            b_inf = medium_rgb
        elif b_inf_mode == "independent":
            if b_inf_raw is None:
                raise RuntimeError("b_inf_mode='independent' requires enable_b_inf=True")
            b_inf = self.colour_activation(b_inf_raw).to(directions)
        elif b_inf_mode == "bounded_residual":
            if b_inf_raw is None:
                raise RuntimeError("b_inf_mode='bounded_residual' requires enable_b_inf=True")
            b_inf_residual = torch.tanh(b_inf_raw).to(directions)
            rgb_logit = torch.logit(medium_rgb.clamp(1e-4, 1.0 - 1e-4))
            b_inf = torch.sigmoid(rgb_logit + float(b_inf_residual_scale) * b_inf_residual)
        else:
            raise ValueError(f"Unknown b_inf_mode: {b_inf_mode}")

        return MediumFieldOutput(
            rgb=medium_rgb,
            bs=medium_bs,
            attn=medium_attn,
            directions=directions,
            b_inf=b_inf,
            b_inf_residual=b_inf_residual,
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
        depth_context: Optional[Tensor],
    ) -> Tensor:
        if mode == "dir_only":
            return directions_encoded

        r = torch.sqrt(image_xx.square() + image_yy.square())
        features = [directions_encoded, torch.stack([image_xx, image_yy, r], dim=-1).view(-1, 3)]

        if "depth" in mode:
            if depth_context is None:
                depth_feature = torch.zeros(height, width, 1, device=image_xx.device, dtype=image_xx.dtype)
            else:
                depth_feature = depth_context.to(device=image_xx.device, dtype=image_xx.dtype)
                if depth_feature.ndim == 2:
                    depth_feature = depth_feature[..., None]
            features.append(depth_feature.view(-1, 1))

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

    def _append_point_context(
        self,
        *,
        directions_encoded: Tensor,
        image_xy: Tensor,
        mode: MediumContextMode,
        camera_centers: Optional[Tensor],
        scene_center: Optional[Tensor],
        scene_scale: Optional[Union[Tensor, float]],
        camera_context_scale: float,
        camera_context_dropout: float,
        training: bool,
        depth_context: Optional[Tensor],
    ) -> Tensor:
        if mode == "dir_only":
            return directions_encoded

        n = int(image_xy.shape[0])
        image_xy = image_xy.to(device=directions_encoded.device, dtype=directions_encoded.dtype)
        image_x = image_xy[:, 0]
        image_y = image_xy[:, 1]
        r = torch.sqrt(image_x.square() + image_y.square())
        features = [directions_encoded, torch.stack([image_x, image_y, r], dim=-1)]

        if "depth" in mode:
            if depth_context is None:
                depth_feature = torch.zeros(n, 1, device=directions_encoded.device, dtype=directions_encoded.dtype)
            else:
                depth_feature = depth_context.to(device=directions_encoded.device, dtype=directions_encoded.dtype)
                if depth_feature.ndim == 1:
                    depth_feature = depth_feature[:, None]
            features.append(depth_feature.reshape(n, 1))

        if "camera" in mode:
            if camera_centers is None:
                camera_feature = torch.zeros(n, 3, device=directions_encoded.device, dtype=directions_encoded.dtype)
            else:
                camera_feature = camera_centers.to(device=directions_encoded.device, dtype=directions_encoded.dtype)
                if camera_feature.ndim == 1:
                    camera_feature = camera_feature[None, :].expand(n, 3)
                else:
                    camera_feature = camera_feature.reshape(n, 3)
                if scene_center is not None:
                    scene_center = scene_center.to(device=directions_encoded.device, dtype=directions_encoded.dtype)
                    if scene_scale is None:
                        scene_scale = torch.tensor(1.0, device=directions_encoded.device, dtype=directions_encoded.dtype)
                    elif not isinstance(scene_scale, Tensor):
                        scene_scale = torch.tensor(scene_scale, device=directions_encoded.device, dtype=directions_encoded.dtype)
                    else:
                        scene_scale = scene_scale.to(device=directions_encoded.device, dtype=directions_encoded.dtype)
                    camera_feature = (camera_feature - scene_center.reshape(1, 3)) / (scene_scale + 1e-6)
                camera_feature = camera_feature * camera_context_scale
            if camera_context_dropout > 0.0:
                camera_feature = F.dropout(camera_feature, p=camera_context_dropout, training=training)
            features.append(camera_feature)

        return torch.cat(features, dim=-1)
