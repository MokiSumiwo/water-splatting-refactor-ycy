"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import water_splatting.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    xys_grad_abs: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    medium_rgb: Float[Tensor, "*height width channels"],
    medium_bs: Float[Tensor, "*height width channels"],
    medium_attn: Float[Tensor, "*height width channels"],
    img_height: int,
    img_width: int,
    block_width: int,
    igaf_coeffs: Optional[Float[Tensor, "*batch 4 3"]] = None,
    igaf_screen_to_uv: Optional[Float[Tensor, "*batch 4"]] = None,
    igaf_gate: Optional[Float[Tensor, "*batch 5"]] = None,
    igaf_frequency: float = 1.5,
    igaf_amplitude_max: float = 0.10,
    igaf_coordinate_clamp: float = 3.0,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
    step: Optional[int] = None,
    return_hit_stats: Optional[bool] = False,
    force_white_background: Optional[bool] = True,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        xys_grad_abs (Tensor): absolute value of the gradient to be edited.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        medium_rgb (Tensor): RGB color of the medium.
        medium_bs (Tensor): Scattering coefficients of the medium.
        medium_attn (Tensor): Attenuation coefficients of the medium.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive
        background (Tensor): background color of shape (channels,)
        return_alpha (bool): whether to return alpha channel
        force_white_background (bool): preserve legacy behavior by replacing
            the provided background with white. Set False only for diagnostics
            that explicitly need a trainable black-background clear proxy.

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output object.
        - **out_clr** (Tensor): N-dimensional rendered output clear object.
        - **out_medium** (Tensor): N-dimensional rendered output medium.
        - **depth_im** (Tensor): N-dimensional rendered output depth image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if force_white_background or background is None:
        background = torch.ones(colors.shape[-1], dtype=torch.float32, device=colors.device)
    else:
        if background.ndim != 1 or background.shape[0] != colors.shape[-1]:
            raise ValueError(
                "background must be a single color tensor with shape "
                f"({colors.shape[-1]},); got {tuple(background.shape)}"
            )
        background = background.to(device=colors.device, dtype=torch.float32)

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    use_igaf = (
        colors.shape[-1] == 3
        and igaf_coeffs is not None
        and igaf_screen_to_uv is not None
        and igaf_gate is not None
        and abs(float(igaf_amplitude_max)) > 0.0
    )
    if use_igaf:
        assert igaf_coeffs is not None
        assert igaf_screen_to_uv is not None
        assert igaf_gate is not None
        igaf_coeffs = igaf_coeffs.to(device=colors.device, dtype=torch.float32).contiguous()
        igaf_screen_to_uv = igaf_screen_to_uv.to(device=colors.device, dtype=torch.float32).contiguous()
        igaf_gate = igaf_gate.to(device=colors.device, dtype=torch.float32).contiguous()
    else:
        igaf_coeffs = colors.new_empty((0, 4, 3))
        igaf_screen_to_uv = colors.new_empty((0, 4))
        igaf_gate = colors.new_empty((0, 5))

    return _RasterizeGaussians.apply(
        xys.contiguous(),
        xys_grad_abs.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        igaf_coeffs,
        igaf_screen_to_uv,
        igaf_gate,
        float(igaf_frequency),
        float(igaf_amplitude_max),
        float(igaf_coordinate_clamp),
        bool(use_igaf),
        opacity.contiguous(),
        medium_rgb.contiguous(),
        medium_bs.contiguous(),
        medium_attn.contiguous(),
        img_height,
        img_width,
        block_width,
        background.contiguous(),
        return_alpha,
        step,
        return_hit_stats,
        force_white_background,
    )


class _RasterizeGaussians(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        xys_grad_abs: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        igaf_coeffs: Float[Tensor, "*batch 4 3"],
        igaf_screen_to_uv: Float[Tensor, "*batch 4"],
        igaf_gate: Float[Tensor, "*batch 5"],
        igaf_frequency: float,
        igaf_amplitude_max: float,
        igaf_coordinate_clamp: float,
        use_igaf: bool,
        opacity: Float[Tensor, "*batch 1"],
        medium_rgb: Float[Tensor, "*height width channels"],
        medium_bs: Float[Tensor, "*height width channels"],
        medium_attn: Float[Tensor, "*height width channels"],
        img_height: int,
        img_width: int,
        block_width: int,
        background: Optional[Float[Tensor, "channels"]] = None,
        return_alpha: Optional[bool] = False,
        step: Optional[int] = None,
        return_hit_stats: Optional[bool] = False,
        force_white_background: Optional[bool] = True,
    ) -> Tensor:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
            first_idx = torch.zeros(img_height, img_width, device=xys.device)
            depth_im = torch.zeros(img_height, img_width, device=xys.device)
            out_clr = torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            out_medium = torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            depth2_im = torch.zeros(img_height, img_width, device=xys.device)
            first_depth_im = torch.zeros(img_height, img_width, device=xys.device)
            last_depth_im = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            if use_igaf and colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_forward_igaf
            elif colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_forward
            else:
                rasterize_fn = _C.nd_rasterize_forward

            if use_igaf and colors.shape[-1] == 3:
                rasterized = rasterize_fn(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    colors,
                    igaf_coeffs,
                    igaf_screen_to_uv,
                    igaf_gate,
                    float(igaf_frequency),
                    float(igaf_amplitude_max),
                    float(igaf_coordinate_clamp),
                    opacity,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    depths,
                    background,
                )
            else:
                rasterized = rasterize_fn(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    colors,
                    opacity,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    depths,
                    background,
                )
            if colors.shape[-1] == 3:
                (
                    out_img,
                    out_clr,
                    out_medium,
                    depth_im,
                    final_Ts,
                    final_idx,
                    first_idx,
                    depth2_im,
                    first_depth_im,
                    last_depth_im,
                ) = rasterized
            else:
                out_img, out_clr, out_medium, depth_im = rasterized
                final_Ts = torch.zeros(img_height, img_width, device=xys.device)
                final_idx = torch.zeros(img_height, img_width, device=xys.device)
                first_idx = torch.zeros(img_height, img_width, device=xys.device)
                depth2_im = torch.zeros(img_height, img_width, device=xys.device)
                first_depth_im = torch.zeros(img_height, img_width, device=xys.device)
                last_depth_im = torch.zeros(img_height, img_width, device=xys.device)

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.use_igaf = bool(use_igaf)
        ctx.igaf_frequency = float(igaf_frequency)
        ctx.igaf_amplitude_max = float(igaf_amplitude_max)
        ctx.igaf_coordinate_clamp = float(igaf_coordinate_clamp)
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            xys_grad_abs,
            conics,
            colors,
            igaf_coeffs,
            igaf_screen_to_uv,
            igaf_gate,
            opacity,
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            background,
            final_Ts,
            final_idx,
            first_idx,
        )
        
        if return_alpha:
            out_alpha = 1 - final_Ts
            if return_hit_stats:
                return (
                    out_img,
                    out_clr,
                    out_medium,
                    depth_im,
                    out_alpha,
                    depth2_im,
                    first_depth_im,
                    last_depth_im,
                    final_Ts,
                )
            return out_img, out_clr, out_medium, depth_im, out_alpha
        if return_hit_stats:
            return out_img, out_clr, out_medium, depth_im, depth2_im, first_depth_im, last_depth_im, final_Ts
        else:
            return out_img, out_clr, out_medium, depth_im

    @staticmethod
    def backward(
        ctx,
        v_out_img,
        v_out_clr,
        v_out_medium,
        v_depth_im,
        v_out_alpha=None,
        v_depth2_im=None,
        v_first_depth_im=None,
        v_last_depth_im=None,
        v_final_Ts=None,
    ):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            xys_grad_abs,
            conics,
            colors,
            igaf_coeffs,
            igaf_screen_to_uv,
            igaf_gate,
            opacity,
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            background,
            final_Ts,
            final_idx,
            first_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_igaf_coeffs = torch.zeros_like(igaf_coeffs)
            v_opacity = torch.zeros_like(opacity)
            v_medium_rgb = torch.zeros_like(medium_rgb)
            v_medium_bs = torch.zeros_like(medium_bs)
            v_medium_attn = torch.zeros_like(medium_attn)

        else:
            if ctx.use_igaf and colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_backward_igaf
            elif colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_backward
            else:
                rasterize_fn = _C.nd_rasterize_backward
            if ctx.use_igaf and colors.shape[-1] == 3:
                (
                    v_xy,
                    v_conic,
                    v_colors,
                    v_igaf_coeffs,
                    v_opacity,
                    v_medium_rgb,
                    v_medium_bs,
                    v_medium_attn,
                ) = rasterize_fn(
                    img_height,
                    img_width,
                    ctx.block_width,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    xys_grad_abs,
                    conics,
                    colors,
                    igaf_coeffs,
                    igaf_screen_to_uv,
                    igaf_gate,
                    ctx.igaf_frequency,
                    ctx.igaf_amplitude_max,
                    ctx.igaf_coordinate_clamp,
                    opacity,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    depths,
                    background,
                    final_Ts,
                    final_idx,
                    first_idx,
                    v_out_img,
                    v_out_medium,
                    v_out_alpha,
                )
            else:
                v_igaf_coeffs = torch.zeros_like(igaf_coeffs)
                v_xy, v_conic, v_colors, v_opacity, v_medium_rgb, v_medium_bs, v_medium_attn = rasterize_fn(
                    img_height,
                    img_width,
                    ctx.block_width,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    xys_grad_abs,
                    conics,
                    colors,
                    opacity,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    depths,
                    background,
                    final_Ts,
                    final_idx,
                    first_idx,
                    v_out_img,
                    v_out_medium,
                    v_out_alpha,
                )
            
        return (
            v_xy,  # xys
            None,  # xys_grad_abs
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_igaf_coeffs,  # igaf_coeffs
            None,  # igaf_screen_to_uv
            None,  # igaf_gate
            None,  # igaf_frequency
            None,  # igaf_amplitude_max
            None,  # igaf_coordinate_clamp
            None,  # use_igaf
            v_opacity,  # opacity
            v_medium_rgb,  # medium_rgb
            v_medium_bs,  # medium_bs
            v_medium_attn,  # medium_attn
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # background
            None,  # return_alpha
            None,  # step
            None,  # return_hit_stats
            None,  # force_white_background
        )
