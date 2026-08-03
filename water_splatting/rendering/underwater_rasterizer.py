"""Thin wrappers around the original projection and underwater rasterization calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from water_splatting.project_gaussians import project_gaussians
from water_splatting.rasterize import rasterize_gaussians


@dataclass
class UnderwaterRenderOutput:
    """Rendered outputs produced by the original underwater compositor."""

    rgb: Tensor
    rgb_object: Tensor
    j_raw: Tensor
    j_gaussian: Tensor
    rgb_clear: Tensor
    rgb_clear_clamp: Tensor
    rgb_medium: Tensor
    depth: Tensor
    accumulation: Tensor
    depth_second_moment: Tensor
    depth_variance: Tensor
    depth_std_relative: Tensor
    first_depth: Tensor
    last_depth: Tensor
    final_transmittance: Tensor


class UnderwaterRasterizer:
    """Original WaterSplatting rasterizer orchestration."""

    block_width: int = 16

    def project(
        self,
        *,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        viewmat: Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        height: int,
        width: int,
        clip_thresh: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return project_gaussians(
            means,
            torch.exp(scales),
            1,
            quats / quats.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            self.block_width,
            clip_thresh=clip_thresh,
        )

    def rasterize(
        self,
        *,
        xys: Tensor,
        xys_grad_abs: Tensor,
        depths: Tensor,
        radii: Tensor,
        conics: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        opacities: Tensor,
        medium_rgb: Tensor,
        medium_bs: Tensor,
        medium_attn: Tensor,
        height: int,
        width: int,
        background: Optional[Tensor],
        step: int,
        force_white_background: bool = True,
        igaf_coeffs: Optional[Tensor] = None,
        igaf_screen_to_uv: Optional[Tensor] = None,
        igaf_gate: Optional[Tensor] = None,
        igaf_frequency: float = 1.5,
        igaf_amplitude_max: float = 0.10,
        igaf_coordinate_clamp: float = 3.0,
    ) -> UnderwaterRenderOutput:
        (
            rgb_object,
            rgb_clear_raw,
            rgb_medium,
            depth_im,
            alpha,
            depth2_im,
            first_depth,
            last_depth,
            final_transmittance,
        ) = rasterize_gaussians(
            xys,
            xys_grad_abs,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            height,
            width,
            self.block_width,
            igaf_coeffs=igaf_coeffs,
            igaf_screen_to_uv=igaf_screen_to_uv,
            igaf_gate=igaf_gate,
            igaf_frequency=igaf_frequency,
            igaf_amplitude_max=igaf_amplitude_max,
            igaf_coordinate_clamp=igaf_coordinate_clamp,
            background=background,
            return_alpha=True,
            step=step,
            return_hit_stats=True,
            force_white_background=force_white_background,
        )

        rgb = rgb_object + rgb_medium
        j_gaussian = torch.clamp(rgb_clear_raw, 0.0, 1.0)
        rgb_clear = rgb_clear_raw / (rgb_clear_raw + 1.0)

        depth_im = depth_im[..., None]
        depth2_im = depth2_im[..., None]
        alpha = alpha[..., None]
        first_depth = first_depth[..., None]
        last_depth = last_depth[..., None]
        final_transmittance = final_transmittance[..., None]
        depth_expected = torch.where(alpha > 0, depth_im / alpha.clamp_min(1e-6), depth_im.detach().max())
        depth_second_moment = torch.where(
            alpha > 0,
            depth2_im / alpha.clamp_min(1e-6),
            torch.zeros_like(depth2_im),
        )
        depth_variance = torch.clamp(depth_second_moment - depth_expected.square(), min=0.0)
        depth_std_relative = torch.sqrt(depth_variance + 1e-8) / depth_expected.clamp_min(1e-6)

        return UnderwaterRenderOutput(
            rgb=rgb,
            rgb_object=rgb_object,
            j_raw=rgb_clear_raw,
            j_gaussian=j_gaussian,
            rgb_clear=rgb_clear,
            rgb_clear_clamp=j_gaussian,
            rgb_medium=rgb_medium,
            depth=depth_expected,
            accumulation=alpha,
            depth_second_moment=depth_second_moment,
            depth_variance=depth_variance,
            depth_std_relative=depth_std_relative,
            first_depth=first_depth,
            last_depth=last_depth,
            final_transmittance=final_transmittance,
        )

    def rasterize_clear_proxy(
        self,
        *,
        xys: Tensor,
        xys_grad_abs: Tensor,
        depths: Tensor,
        radii: Tensor,
        conics: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        opacities: Tensor,
        height: int,
        width: int,
        step: int,
    ) -> UnderwaterRenderOutput:
        zeros = colors.new_zeros(height, width, colors.shape[-1])
        black = colors.new_zeros(colors.shape[-1])
        return self.rasterize(
            xys=xys,
            xys_grad_abs=xys_grad_abs,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors,
            opacities=opacities,
            medium_rgb=zeros,
            medium_bs=zeros,
            medium_attn=zeros,
            height=height,
            width=width,
            background=black,
            step=step,
            force_white_background=False,
        )
