"""Gaussian appearance helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from water_splatting.sh import spherical_harmonics


@dataclass
class GaussianColorOutput:
    """Current-view Gaussian colors and optional bounded-color diagnostics."""

    rgb: Tensor
    logits: Tensor | None = None
    sigmoid_derivative: Tensor | None = None
    dc_rgb: Tensor | None = None
    dc_logits: Tensor | None = None
    sh_residual: Tensor | None = None
    color_residual: Tensor | None = None
    positive_utilization: Tensor | None = None
    negative_utilization: Tensor | None = None


def resolve_headroom_alpha(step: int, *, target: float = 1.0,
                           schedule: str = "constant", start_step: int = 3000,
                           end_step: int = 8000) -> float:
    """Deterministic, mode-independent continuation from BND to FSB."""
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("headroom target must be finite and in [0, 1]")
    if schedule == "constant":
        return float(target)
    if schedule != "smoothstep" or end_step <= start_step:
        raise ValueError("expected smoothstep with end_step > start_step")
    t = min(1.0, max(0.0, (int(step) - start_step) / (end_step - start_step)))
    return float(target) * t * t * (3.0 - 2.0 * t)


def blend_bounded_headroom_colors(base: GaussianColorOutput,
                                  head: GaussianColorOutput, alpha: float) -> GaussianColorOutput:
    """Convex RGB mixture with diagnostics describing the actual mixed RGB."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if alpha == 0.0:
        return base
    if alpha == 1.0:
        return head
    rgb = (1.0 - alpha) * base.rgb + alpha * head.rgb
    dc = base.dc_rgb
    residual = rgb - dc
    return GaussianColorOutput(
        rgb=rgb, logits=base.logits, dc_rgb=dc, dc_logits=base.dc_logits,
        sh_residual=base.sh_residual, color_residual=residual,
        positive_utilization=residual.clamp_min(0.0) / (1.0 - dc).clamp_min(1e-8),
        negative_utilization=(-residual).clamp_min(0.0) / dc.clamp_min(1e-8),
    )


def _sh_viewdirs(means: Tensor, camera_position: Tensor) -> Tensor:
    viewdirs = means.detach() - camera_position.detach()
    return viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def compute_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> Tensor:
    """Compute the original WaterSplatting Gaussian RGB values."""

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        rgbs = spherical_harmonics(active_sh_degree, viewdirs, colors)
        return torch.clamp(rgbs + 0.5, min=0.0)

    return torch.sigmoid(colors[:, 0, :])


def compute_bounded_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> GaussianColorOutput:
    """Compute bounded full-SH Gaussian RGB values.

    For SH>0, the active SH evaluation is interpreted as RGB logits and then
    passed through sigmoid. The legacy ``+0.5`` offset is not applied.
    """

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        logits = colors[:, 0, :]
        dc_logits = logits
    rgb = torch.sigmoid(logits)
    return GaussianColorOutput(
        rgb=rgb,
        logits=logits,
        sigmoid_derivative=rgb * (1.0 - rgb),
        dc_rgb=torch.sigmoid(dc_logits),
        dc_logits=dc_logits,
        sh_residual=logits - dc_logits,
        color_residual=rgb - torch.sigmoid(dc_logits),
    )


def compute_bounded_headroom_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> GaussianColorOutput:
    """Compute bounded headroom-SH Gaussian RGB values.

    The degree-0 SH contribution is interpreted as the base logit ``s0``.
    The active non-DC SH contribution ``r`` is mapped through asymmetric
    positive/negative RGB headroom while preserving the BND-v1 first-order
    Jacobian at ``r=0``.
    """

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        full_logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        full_logits = colors[:, 0, :]
        dc_logits = full_logits

    c0 = torch.sigmoid(dc_logits)
    residual = full_logits - dc_logits
    positive_rgb = c0 + (1.0 - c0) * torch.tanh(c0 * residual)
    negative_rgb = c0 + c0 * torch.tanh((1.0 - c0) * residual)
    rgb = torch.where(residual >= 0.0, positive_rgb, negative_rgb)
    color_residual = rgb - c0
    positive_utilization = torch.where(
        color_residual > 0.0,
        color_residual / (1.0 - c0).clamp_min(1e-8),
        torch.zeros_like(color_residual),
    )
    negative_utilization = torch.where(
        color_residual < 0.0,
        (-color_residual) / c0.clamp_min(1e-8),
        torch.zeros_like(color_residual),
    )
    return GaussianColorOutput(
        rgb=rgb,
        logits=full_logits,
        dc_rgb=c0,
        dc_logits=dc_logits,
        sh_residual=residual,
        color_residual=color_residual,
        positive_utilization=positive_utilization,
        negative_utilization=negative_utilization,
    )


def compute_bounded_degreewise_headroom_gaussian_colors(
    *, means: Tensor, features_dc: Tensor, features_rest: Tensor,
    camera_position: Tensor, sh_degree: int, active_sh_degree: int,
    degree_alphas: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GaussianColorOutput:
    """Headroom mapping with independent weights for SH degrees 1, 2, 3.

    The DC term remains unchanged.  Degree increments are accumulated in logit
    space before the same asymmetric headroom map is applied, preserving RGB
    bounds and the underwater renderer semantics.
    """
    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewdirs = _sh_viewdirs(means, camera_position)
    dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    degree_alphas = tuple(float(x) for x in degree_alphas)
    weighted = dc_logits
    previous = dc_logits
    for degree in range(1, min(int(active_sh_degree), 3) + 1):
        current = spherical_harmonics(degree, viewdirs, colors)
        increment = current - previous
        weighted = weighted + degree_alphas[degree - 1] * increment
        previous = current
    c0 = torch.sigmoid(dc_logits)
    residual = weighted - dc_logits
    positive_rgb = c0 + (1.0 - c0) * torch.tanh(c0 * residual)
    negative_rgb = c0 + c0 * torch.tanh((1.0 - c0) * residual)
    rgb = torch.where(residual >= 0.0, positive_rgb, negative_rgb)
    color_residual = rgb - c0
    return GaussianColorOutput(
        rgb=rgb, logits=weighted, dc_rgb=c0, dc_logits=dc_logits,
        sh_residual=residual, color_residual=color_residual,
        positive_utilization=torch.where(color_residual > 0.0, color_residual / (1.0 - c0).clamp_min(1e-8), torch.zeros_like(c0)),
        negative_utilization=torch.where(color_residual < 0.0, -color_residual / c0.clamp_min(1e-8), torch.zeros_like(c0)),
    )


def compute_bounded_selective_gaussian_colors(
    *, means: Tensor, features_dc: Tensor, features_rest: Tensor,
    camera_position: Tensor, sh_degree: int, active_sh_degree: int,
    saturation_threshold: float = 0.055, temperature: float = 0.015,
) -> GaussianColorOutput:
    """Selective BND relaxation for saturated Gaussian appearance.

    The ordinary sigmoid BND mapping is retained for well-conditioned colors.
    A smooth, detached-free gate activates the headroom mapping only when the
    full-SH sigmoid derivative is small, preserving bounded RGB while restoring
    gradient capacity near saturation.
    """
    base = compute_bounded_gaussian_colors(
        means=means, features_dc=features_dc, features_rest=features_rest,
        camera_position=camera_position, sh_degree=sh_degree,
        active_sh_degree=active_sh_degree)
    head = compute_bounded_headroom_gaussian_colors(
        means=means, features_dc=features_dc, features_rest=features_rest,
        camera_position=camera_position, sh_degree=sh_degree,
        active_sh_degree=active_sh_degree)
    deriv = base.sigmoid_derivative
    gate = torch.sigmoid((float(saturation_threshold) - deriv) /
                         max(float(temperature), 1e-6))
    rgb = (1.0 - gate) * base.rgb + gate * head.rgb
    return GaussianColorOutput(
        rgb=rgb, logits=base.logits, sigmoid_derivative=deriv,
        dc_rgb=base.dc_rgb, dc_logits=base.dc_logits,
        sh_residual=base.sh_residual, color_residual=rgb - base.dc_rgb,
        positive_utilization=head.positive_utilization * gate,
        negative_utilization=head.negative_utilization * gate)


def compute_brash_gaussian_colors(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    brash_gate_logit: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
    residual_scale: float = 1.0,
    gate_mode: str = "learned",
) -> GaussianColorOutput:
    """Compute BRASH's bounded DC/base and view-residual mixture.

    Both branches are sigmoid-bounded before mixing, so the convex mixture is
    bounded independently of the learned gate. ``fixed_one`` is the exact
    full-SH bounded path when ``residual_scale`` is one.
    """

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if sh_degree > 0:
        viewdirs = _sh_viewdirs(means, camera_position)
        full_logits = spherical_harmonics(active_sh_degree, viewdirs, colors)
        dc_logits = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    else:
        full_logits = colors[:, 0, :]
        dc_logits = full_logits
    residual_logits = full_logits - dc_logits
    view_logits = dc_logits + float(residual_scale) * residual_logits
    base_rgb = torch.sigmoid(dc_logits)
    view_rgb = torch.sigmoid(view_logits)
    if gate_mode == "fixed_one":
        gate = torch.ones_like(base_rgb)
    elif gate_mode == "learned":
        gate = torch.sigmoid(brash_gate_logit).expand_as(base_rgb)
    else:
        raise ValueError(f"Unknown BRASH gate mode: {gate_mode}")
    rgb = (1.0 - gate) * base_rgb + gate * view_rgb
    return GaussianColorOutput(
        rgb=rgb,
        logits=view_logits,
        sigmoid_derivative=view_rgb * (1.0 - view_rgb),
        dc_rgb=base_rgb,
        dc_logits=dc_logits,
        sh_residual=residual_logits,
        color_residual=rgb - base_rgb,
    )


def compute_gaussian_sh_residual(
    *,
    means: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    camera_position: Tensor,
    sh_degree: int,
    active_sh_degree: int,
) -> Tensor:
    """Compute the non-DC SH color residual for the current view."""

    if sh_degree <= 0 or active_sh_degree <= 0 or features_rest.numel() == 0:
        return torch.zeros_like(features_dc)

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewdirs = _sh_viewdirs(means, camera_position)
    active_color = spherical_harmonics(active_sh_degree, viewdirs, colors)
    dc_color = spherical_harmonics(0, viewdirs, colors[:, :1, :])
    return active_color - dc_color


def compute_taba_allocation(
    transmission: Tensor,
    support: Tensor | None = None,
    *,
    valid_mask: Tensor | None = None,
    min_support: int = 3,
    support_scale: float = 3.0,
    transmission_mid: float = 0.5,
    transmission_temperature: float = 0.15,
    luma_transmission_mid: float = 0.35,
    luma_transmission_temperature: float = 0.20,
    chroma_floor: float = 0.20,
    luma_floor: float = 0.60,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached allocation gates from renderer statistics.

    With no support argument this is chromatic-only TABA: luminance is
    unchanged, and invalid transmission falls back to a unit chromatic gate.
    The explicit support argument preserves the historical diagnostic API.
    """

    if support is None:
        if not 0.0 <= chroma_floor <= 1.0:
            raise ValueError("chroma_floor must be in [0, 1]")
        if not 0.0 <= transmission_mid <= 1.0 or not 0.0 < transmission_temperature < float("inf"):
            raise ValueError("require transmission_mid in [0, 1] and a finite positive temperature")
        t = transmission.detach()
        valid = torch.isfinite(t) & (t >= 0.0) & (t <= 1.0)
        if valid_mask is not None:
            valid = valid & valid_mask.detach().bool()
        t = torch.where(valid, t, torch.ones_like(t))
        gate = chroma_floor + (1.0 - chroma_floor) * torch.sigmoid(
            (t - transmission_mid) / transmission_temperature
        )
        gate = torch.where(valid, gate, torch.ones_like(gate))
        return torch.ones_like(gate), gate.detach(), torch.where(valid, t, torch.zeros_like(t)).detach()

    if not 0.0 <= chroma_floor <= luma_floor <= 1.0:
        raise ValueError("require 0 <= chroma_floor <= luma_floor <= 1")
    if support_scale <= 0.0 or transmission_temperature <= 0.0 or luma_transmission_temperature <= 0.0:
        raise ValueError("support scale and temperatures must be positive")
    t = transmission.detach().float().clamp(0.0, 1.0)
    s = support.detach().float()
    reliability = ((s - float(min_support)) / float(support_scale)).clamp(0.0, 1.0)
    g_chroma = chroma_floor + (1.0 - chroma_floor) * torch.sigmoid((t - transmission_mid) / transmission_temperature) * reliability
    g_luma = luma_floor + (1.0 - luma_floor) * torch.sigmoid((t - luma_transmission_mid) / luma_transmission_temperature) * reliability
    return g_luma.detach(), g_chroma.detach(), (reliability * t).detach()


def safe_sigmoid(logits: Tensor) -> Tensor:
    """Keep floating-point sigmoid results strictly inside the RGB interval."""
    limits = torch.finfo(logits.dtype)
    return torch.sigmoid(logits).clamp(min=limits.tiny, max=1.0 - limits.eps)


def apply_taba_to_bounded_color(color: GaussianColorOutput, gate: Tensor) -> GaussianColorOutput:
    """Allocate only the zero-mean chromatic SH residual with a detached gate."""
    if color.logits is None or color.dc_logits is None:
        raise ValueError("TABA requires bounded SH logits and DC logits")
    if gate.shape != color.logits.shape[:-1] + (1,):
        raise ValueError("TABA requires one scalar gate per Gaussian")
    residual = color.logits - color.dc_logits
    chroma = residual - residual.mean(dim=-1, keepdim=True)
    # Algebraically s_base + delta_luma + gate * delta_chroma; a unit gate
    # retains the original logits exactly, including invalid-depth fallback.
    logits = color.logits + (gate.detach() - 1.0) * chroma
    rgb = safe_sigmoid(logits)
    return GaussianColorOutput(
        rgb=rgb,
        logits=logits,
        sigmoid_derivative=rgb * (1.0 - rgb),
        dc_rgb=color.dc_rgb,
        dc_logits=color.dc_logits,
        sh_residual=logits - color.dc_logits,
        color_residual=rgb - color.dc_rgb,
    )
