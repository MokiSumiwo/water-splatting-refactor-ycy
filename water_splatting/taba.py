"""Detached transmission sampling and bounded-appearance diagnostics."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor

from water_splatting.fields.gaussian_appearance import compute_taba_allocation, safe_sigmoid


@torch.no_grad()
def sample_taba_transmission(
    medium_attn: Tensor, depth: Tensor, accumulation: Tensor, xys: Tensor, radii: Tensor,
    *, rgb_object: Tensor | None = None, rgb_medium: Tensor | None = None,
    tau_D: Tensor | None = None, transmission_rgb: Tensor | None = None,
) -> Dict[str, Tensor]:
    """Sample mean RGB transmission at rounded projected Gaussian centers.

    Depth is the renderer's alpha-normalized expected depth, independent of
    Gaussian color. Out-of-frame, invisible, empty, or nonfinite samples are
    invalid and receive a unit gate at allocation time.
    """
    attn, depth, accumulation = medium_attn.detach(), depth.detach(), accumulation.detach()
    xy = xys.detach()
    height, width = depth.shape[:2]
    valid_pixel = (
        torch.isfinite(depth) & (depth > 0.0) & torch.isfinite(accumulation) & (accumulation > 0.0)
        & torch.isfinite(attn).all(dim=-1, keepdim=True)
        & (attn >= 0.0).all(dim=-1, keepdim=True)
    )
    tau = (attn * depth).clamp_min(0.0) if tau_D is None else tau_D.detach().clamp_min(0.0)
    if transmission_rgb is None:
        transmission_rgb = torch.exp(-tau)
    else:
        transmission_rgb = transmission_rgb.detach()
        valid_pixel &= torch.isfinite(transmission_rgb).all(-1, keepdim=True)
        valid_pixel &= ((transmission_rgb >= 0) & (transmission_rgb <= 1)).all(-1, keepdim=True)
    valid_pixel = valid_pixel & torch.isfinite(tau).all(dim=-1, keepdim=True)
    transmission = transmission_rgb.mean(dim=-1, keepdim=True)
    rounded = torch.nan_to_num(xy, nan=-1.0, posinf=-1.0, neginf=-1.0).round().long()
    x, y = rounded[:, 0], rounded[:, 1]
    valid = (
        torch.isfinite(xy).all(dim=-1) & (radii.detach().reshape(-1) > 0)
        & (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
    )
    x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
    valid = valid[:, None] & valid_pixel[y, x]
    result = {
        "transmission": torch.where(valid, transmission[y, x], torch.ones_like(transmission[y, x])),
        "transmission_rgb": torch.where(valid, transmission_rgb[y, x], torch.ones_like(transmission_rgb[y, x])),
        "valid": valid,
        "depth": torch.where(valid, depth[y, x], torch.zeros_like(depth[y, x])),
        "tau_D": torch.where(valid, tau[y, x], torch.zeros_like(tau[y, x])),
    }
    if rgb_object is not None and rgb_medium is not None:
        result["rgb_object"] = rgb_object.detach()[y, x]
        result["rgb_medium"] = rgb_medium.detach()[y, x]
    return result


def compute_taba_dcu(
    s_full: Tensor, s_base: Tensor, tau_D: Tensor, rgb_object: Tensor | None = None,
    rgb_medium: Tensor | None = None, *, mode: str = "directional_uncertainty",
    transmission: Tensor | None = None, valid_mask: Tensor | None = None,
    transmission_mid: float = 0.50, transmission_temperature: float = 0.15,
    chroma_floor: float = 0.20, uncertainty_lambda: float = 0.50,
    water_axis_eps: float = 1e-4, eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """Allocate SH chroma using detached RGB transmission and water direction.

    Statistics reduce over the three RGB channels of each Gaussian sample.
    The prescribed raw RGB axis is not recentered: directional allocation can
    change mean logits. Only rgb and s_adjusted retain appearance gradients.
    """
    if mode not in ("disabled", "chromatic", "directional", "directional_uncertainty"):
        raise ValueError(f"unknown TABA mode: {mode}")
    if not 0 <= uncertainty_lambda <= 1 or not 0 < water_axis_eps < float("inf") or not 0 < eps < float("inf"):
        raise ValueError("require finite uncertainty_lambda in [0,1] and positive axis/normalization epsilons")
    if s_full.shape != s_base.shape or s_full.shape[-1] != 3 or tau_D.shape != s_full.shape:
        raise ValueError("logits, base logits and RGB tau_D must have matching (...,3) shapes")
    with torch.no_grad():
        tau = tau_D.detach()
        valid = torch.isfinite(tau).all(-1, keepdim=True) & (tau >= 0).all(-1, keepdim=True)
        if transmission is None:
            t = torch.exp(-torch.where(valid, tau, torch.zeros_like(tau)))
        else:
            if transmission.shape != tau.shape:
                raise ValueError("transmission must have the same RGB shape as tau_D")
            t = transmission.detach()
        valid &= torch.isfinite(t).all(-1, keepdim=True) & ((t >= 0) & (t <= 1)).all(-1, keepdim=True)
        if valid_mask is not None:
            valid &= valid_mask.detach().bool()
        t = torch.where(valid, t, torch.ones_like(t))
        t_mean = t.mean(-1, keepdim=True)
        t_std = t.var(-1, keepdim=True, correction=0).sqrt()
        t_cv = (t_std / (t_mean + eps)).clamp(0, 1)
        q = (t_mean * (1 - uncertainty_lambda * t_cv)).clamp(0, 1) if mode == "directional_uncertainty" else t_mean
        q = torch.where(valid, q, torch.ones_like(q))
        _, directional_gate, _ = compute_taba_allocation(
            q, valid_mask=valid, transmission_mid=transmission_mid,
            transmission_temperature=transmission_temperature, chroma_floor=chroma_floor,
        )
        _, fallback_gate, _ = compute_taba_allocation(
            t_mean, valid_mask=valid, transmission_mid=transmission_mid,
            transmission_temperature=transmission_temperature, chroma_floor=chroma_floor,
        )
        raw_axis = torch.zeros_like(t)
        axis_valid = torch.zeros_like(valid)
        if rgb_object is not None and rgb_medium is not None:
            if rgb_object.shape != t.shape or rgb_medium.shape != t.shape:
                raise ValueError("object and medium RGB must match the Gaussian sample shape")
            raw_axis = rgb_medium.detach() - rgb_object.detach()
            axis_valid = torch.isfinite(raw_axis).all(-1, keepdim=True)
            raw_axis = torch.where(axis_valid, raw_axis, torch.zeros_like(raw_axis))
        strength = raw_axis.norm(dim=-1, keepdim=True)
        axis_valid &= torch.isfinite(strength) & (strength >= water_axis_eps)
        strength = torch.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)
        axis = torch.where(axis_valid, raw_axis / (strength + eps), torch.zeros_like(raw_axis))
        fallback = ~axis_valid
        gate = torch.where(fallback, fallback_gate, directional_gate)
        if mode == "chromatic":
            gate = fallback_gate
        elif mode == "disabled":
            gate = torch.ones_like(gate)
    delta = s_full - s_base
    chroma = delta - delta.mean(-1, keepdim=True)
    parallel = (chroma * axis).sum(-1, keepdim=True) * axis
    orthogonal = chroma - parallel
    allocation = chroma if mode == "chromatic" else torch.where(fallback, chroma, parallel)
    adjusted = s_full if mode == "disabled" else s_full + (gate - 1) * allocation
    return {
        "rgb": torch.sigmoid(s_full) if mode == "disabled" else safe_sigmoid(adjusted),
        "s_adjusted": adjusted, "gate": gate.detach(), "transmission": t.detach(),
        "T_mean": t_mean.detach(), "T_std": t_std.detach(), "T_cv": t_cv.detach(),
        "q_T": q.detach(), "water_axis": axis.detach(), "water_axis_strength": strength.detach(),
        "delta_parallel": parallel.detach(), "delta_orthogonal": orthogonal.detach(),
        "fallback_mask": fallback.detach(), "valid": valid.detach(),
        "g_fallback": fallback_gate.detach(), "g_directional": directional_gate.detach(),
        "luminance_logit_change": (adjusted - s_full).mean(-1, keepdim=True).detach(),
    }


@torch.no_grad()
def taba_statistics(outputs: Mapping[str, Tensor], *, xys: Tensor | None = None, radii: Tensor | None = None) -> Dict[str, float]:
    """Summarize raw J, intrinsic colors and the actual applied allocation."""
    j = outputs["J_gaussian_raw"].detach().float()
    intrinsic = outputs["gaussian_view_rgb"].detach().float()
    clear = outputs["rgb_clear"].detach().float()
    transmission = outputs["transmission"].detach().float().mean(dim=-1)
    residual = outputs["gaussian_sh_residual"].detach().float()
    visible = outputs["gaussian_visible_mask"].detach().reshape(-1).bool()
    residual = residual[visible]
    luma = residual.mean(dim=-1, keepdim=True)
    gate = outputs.get("taba_g_chroma", torch.ones_like(intrinsic[:, :1])).detach().float()[visible]
    valid = outputs.get("taba_valid", torch.ones_like(intrinsic[:, :1], dtype=torch.bool)).detach()[visible]
    stats = {
        "mean_transmission": float(transmission.mean()),
        "low_transmission_fraction": float((transmission < 0.33).float().mean()),
        "middle_transmission_fraction": float(((transmission >= 0.33) & (transmission < 0.66)).float().mean()),
        "high_transmission_fraction": float((transmission >= 0.66).float().mean()),
        "g_chroma_mean": float(gate.mean()) if gate.numel() else 1.0,
        "invalid_gate_fraction": float((~valid).float().mean()) if valid.numel() else 1.0,
        "chromatic_residual_magnitude": float((residual - luma).norm(dim=-1).mean()) if residual.numel() else 0.0,
        "luminance_residual_magnitude": float(luma.abs().mean()) if luma.numel() else 0.0,
        "clear_rgb_min": float(clear.min()), "clear_rgb_max": float(clear.max()),
        "intrinsic_rgb_min": float(intrinsic.min()), "intrinsic_rgb_max": float(intrinsic.max()),
        "P_J_gt_1": float((j > 1.0).float().mean()),
        "P_intrinsic_rgb_gt_1": float((intrinsic > 1.0).float().mean()),
        "J_p99": float(torch.quantile(j.reshape(-1), 0.99)),
        "tau_D_p90": float(torch.quantile(outputs["tau_D"].detach().float().reshape(-1), 0.90)),
    }
    for name, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        stats["g_chroma_" + name] = float(torch.quantile(gate.reshape(-1), quantile)) if gate.numel() else 1.0
    if "taba_transmission" in outputs:
        selected = outputs["taba_transmission"].detach().float()[visible][valid]
        stats["gaussian_proxy_mean_transmission"] = float(selected.mean()) if selected.numel() else 1.0
    dcu = {key[len("taba_dcu_"):]: value.detach() for key, value in outputs.items() if key.startswith("taba_dcu_")}
    if not dcu and xys is not None and radii is not None:
        proxy = sample_taba_transmission(outputs["medium_attn"], outputs["depth"], outputs["accumulation"], xys, radii,
                                         rgb_object=outputs["rgb_object"], rgb_medium=outputs["rgb_medium"])
        dcu = compute_taba_dcu(outputs["gaussian_view_logits"], outputs["gaussian_view_dc_logits"], proxy["tau_D"],
                               proxy["rgb_object"], proxy["rgb_medium"], mode="disabled",
                               transmission=proxy["transmission_rgb"], valid_mask=proxy["valid"])
    if dcu:
        selected_valid = visible & dcu["valid"].reshape(-1)
        for label, values, default in (("transmission", dcu["T_mean"][selected_valid], 1.0),
                                       ("q_T", dcu["q_T"][visible], 1.0)):
            stats[label + "_mean"] = float(values.mean()) if values.numel() else default
            stats[label + "_std"] = float(values.std(correction=0)) if values.numel() else 0.0
            for name, quantile in (("p10", .1), ("p50", .5), ("p90", .9)):
                stats[label + "_" + name] = float(torch.quantile(values.float().reshape(-1), quantile)) if values.numel() else default
        for label, values in (("transmission_channel_std", dcu["T_std"][selected_valid]),
                               ("transmission_cv", dcu["T_cv"][selected_valid]),
                               ("water_axis_strength", dcu["water_axis_strength"][visible]),
                               ("directional_fallback_ratio", dcu["fallback_mask"][visible].float()),
                               ("delta_parallel_magnitude", dcu["delta_parallel"][visible].norm(dim=-1)),
                               ("delta_orthogonal_magnitude", dcu["delta_orthogonal"][visible].norm(dim=-1)),
                               ("luminance_logit_change_abs", dcu["luminance_logit_change"][visible].abs())):
            stats[label] = float(values.mean()) if values.numel() else 0.0
    return stats
