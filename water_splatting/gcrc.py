"""Detached Gaussian cross-view residual contamination reweighting."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from water_splatting.fields.gaussian_appearance import safe_sigmoid


GCRC_FIELDS = ("water_direction_prototype", "water_direction_initialized",
               "residual_direction_prototype", "residual_direction_initialized", "visibility_count")
GCRC_CONFIG = {
    "enabled": False, "mode": "disabled", "diagnostic_only": False,
    "beta_water_direction": .95, "beta_residual_direction": .95,
    "residual_eps": 1e-5, "water_axis_eps": 1e-4, "water_strength_scale": .10,
    "visibility_scale": 8., "min_visibility_count": 4,
    "reliability_start_step": 3000, "max_correction": .50,
}
GCRC_QUANTITIES = ("transmission", "low_transmission", "water_strength", "water_axis_consistency",
                   "water_alignment", "residual_consistency", "residual_instability",
                   "visibility_count", "visibility_support", "contamination", "alpha")
GCRC_STATS = tuple("gcrc_" + key + "_" + stat for key in GCRC_QUANTITIES
                   for stat in ("mean", "std", "p10", "p50", "p90")) + (
    "gcrc_active_fraction", "gcrc_fallback_fraction", "gcrc_corrected_fraction",
    "gcrc_history_available", "gcrc_delta_parallel_magnitude", "gcrc_delta_orthogonal_magnitude",
    "gcrc_luminance_logit_change_abs", "gcrc_decomposition_error",
    "gcrc_corrected_intrinsic_min", "gcrc_corrected_intrinsic_max",
    "gcrc_zero_alpha_intrinsic_endpoint_fraction")


def config_values(config):
    return {key: getattr(config, "gcrc_" + key, default) for key, default in GCRC_CONFIG.items()}


def validate_config(config):
    c = config_values(config)
    if c["mode"] not in ("disabled", "gcrc", "taba", "gcrc_taba", "grac"):
        raise ValueError("unknown GCRC mode")
    for key in ("beta_water_direction", "beta_residual_direction"):
        if not 0 <= c[key] < 1:
            raise ValueError("invalid GCRC " + key)
    for key in ("residual_eps", "water_axis_eps", "visibility_scale", "water_strength_scale"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("invalid GCRC " + key)
    if c["water_strength_scale"] <= c["water_axis_eps"]:
        raise ValueError("GCRC water_strength_scale must exceed water_axis_eps")
    if not 0 <= c["max_correction"] <= 1:
        raise ValueError("GCRC max_correction must be in [0,1]")
    for key in ("min_visibility_count", "reliability_start_step"):
        if not isinstance(c[key], int) or c[key] < 0:
            raise ValueError("invalid GCRC " + key)


class GaussianContaminationState(nn.Module):
    """Persistent O(N) history in the same order as the Gaussian parameters."""

    def __init__(self, reference: Tensor):
        super().__init__()
        n = reference.shape[0]
        for prefix in ("water", "residual"):
            self.register_buffer(prefix + "_direction_prototype", reference.new_zeros(n, 3))
            self.register_buffer(prefix + "_direction_initialized",
                                 torch.zeros(n, 1, device=reference.device, dtype=torch.bool))
        self.register_buffer("visibility_count", torch.zeros(n, 1, device=reference.device, dtype=torch.int64))
        self.register_buffer("last_update_step", torch.tensor(-1, device=reference.device, dtype=torch.int64))

    def check_shape(self, count):
        for key in GCRC_FIELDS:
            expected = (count, 3 if key.endswith("prototype") else 1)
            if getattr(self, key).shape != expected:
                raise RuntimeError(f"GCRC state/Gaussian mismatch: {key}, expected {expected}")

    def values(self, indices=None):
        return {key: getattr(self, key) if indices is None else getattr(self, key)[indices] for key in GCRC_FIELDS}

    @torch.no_grad()
    def update(self, chroma: Tensor, water_raw: Tensor, visible: Tensor, sample_valid: Tensor,
               *, step: int, beta_water_direction=.95, beta_residual_direction=.95,
               residual_eps=1e-5, water_axis_eps=1e-4):
        self.check_shape(chroma.shape[0])
        if water_raw.shape != chroma.shape or visible.numel() != chroma.shape[0] or sample_valid.numel() != chroma.shape[0]:
            raise ValueError("GCRC observation shape mismatch")
        if step <= int(self.last_update_step):
            return
        visible = visible.detach().reshape(-1, 1).bool()
        self.visibility_count.add_(visible.long())
        for prefix, raw, beta, threshold, support in (
            ("water", water_raw, beta_water_direction, water_axis_eps, visible & sample_valid.detach().reshape(-1, 1).bool()),
            ("residual", chroma, beta_residual_direction, residual_eps, visible),
        ):
            raw = raw.detach()
            norm = raw.norm(dim=-1, keepdim=True)
            valid = support & torch.isfinite(raw).all(-1, keepdim=True) & torch.isfinite(norm) & (norm > threshold)
            unit = torch.where(valid, raw / (norm + 1e-8), torch.zeros_like(raw))
            old = getattr(self, prefix + "_direction_prototype")
            proposal = beta * old + (1-beta) * unit
            length = proposal.norm(dim=-1, keepdim=True)
            valid &= torch.isfinite(proposal).all(-1, keepdim=True) & torch.isfinite(length) & (length > 1e-8)
            old.copy_(torch.where(valid, proposal / length.clamp_min(1e-8), old))
            getattr(self, prefix + "_direction_initialized").logical_or_(valid)
        self.last_update_step.fill_(step)

    @torch.no_grad()
    def densify(self, split_mask: Tensor, samples: int, clone_mask: Tensor):
        self.check_shape(split_mask.numel())
        for key in GCRC_FIELDS:
            value = getattr(self, key)
            setattr(self, key, torch.cat((value, value[split_mask].repeat(samples, 1), value[clone_mask]), 0))

    @torch.no_grad()
    def prune(self, removed: Tensor):
        self.check_shape(removed.numel())
        for key in GCRC_FIELDS:
            setattr(self, key, getattr(self, key)[~removed])


@torch.no_grad()
def compute_gcrc_contamination(chroma: Tensor, water_raw: Tensor, transmission: Tensor,
                               sample_valid: Tensor, state: Mapping[str, Tensor], *, step: int,
                               residual_eps=1e-5, water_axis_eps=1e-4, water_strength_scale=.10,
                               visibility_scale=8., min_visibility_count=4,
                               reliability_start_step=3000, max_correction=.50):
    if chroma.shape != water_raw.shape or chroma.shape != transmission.shape or chroma.shape[-1] != 3:
        raise ValueError("GCRC residual, water and transmission require matching (N,3) shapes")
    d, w, t = chroma.detach(), water_raw.detach(), transmission.detach()
    dn, wn = d.norm(dim=-1, keepdim=True), w.norm(dim=-1, keepdim=True)
    dvalid = torch.isfinite(d).all(-1, keepdim=True) & torch.isfinite(dn) & (dn > residual_eps)
    wvalid = torch.isfinite(w).all(-1, keepdim=True) & torch.isfinite(wn) & (wn > water_axis_eps)
    tvalid = torch.isfinite(t).all(-1, keepdim=True) & ((t >= 0) & (t <= 1)).all(-1, keepdim=True)
    tvalid &= sample_valid.detach().bool().reshape(-1, 1)
    tmean = torch.where(tvalid, t.mean(-1, keepdim=True), torch.ones_like(dn))
    low = (1-tmean).clamp(0, 1)
    strength = torch.where(wvalid, ((wn-water_axis_eps)/(water_strength_scale-water_axis_eps)).clamp(0, 1), torch.zeros_like(wn))
    du = torch.where(dvalid, d/(dn+1e-8), torch.zeros_like(d))
    wu = torch.where(wvalid, w/(wn+1e-8), torch.zeros_like(w))
    wp, rp = (state[key].detach() for key in ("water_direction_prototype", "residual_direction_prototype"))
    wpn, rpn = wp.norm(dim=-1, keepdim=True), rp.norm(dim=-1, keepdim=True)
    wpvalid = torch.isfinite(wp).all(-1, keepdim=True) & torch.isfinite(wpn) & (wpn > 1e-8)
    rpvalid = torch.isfinite(rp).all(-1, keepdim=True) & torch.isfinite(rpn) & (rpn > 1e-8)
    wi = state["water_direction_initialized"].detach() & wpvalid
    ri = state["residual_direction_initialized"].detach() & rpvalid
    axis = torch.where(wi, wp/wpn.clamp_min(1e-8), torch.zeros_like(wp))
    water_consistency = torch.where(wi & wvalid, ((1+(wu*axis).sum(-1, keepdim=True))/2).clamp(0, 1), torch.zeros_like(dn))
    residual_consistency = torch.where(ri & dvalid, ((1+(du*rp).sum(-1, keepdim=True))/2).clamp(0, 1), torch.ones_like(dn))
    instability = 1-residual_consistency
    alignment = (du*axis).sum(-1, keepdim=True).abs().clamp(0, 1)
    count = state["visibility_count"].detach()
    support = (-torch.expm1(-count.to(d.dtype).clamp_min(0)/visibility_scale)).clamp(0, 1)
    active = dvalid & wvalid & tvalid & wi & ri & (count >= min_visibility_count) & (step >= reliability_start_step)
    contamination = torch.where(active, low*strength*water_consistency*alignment*instability*support, torch.zeros_like(dn)).clamp(0, 1)
    alpha = max_correction*contamination
    result = {"transmission": tmean, "low_transmission": low, "water_strength": strength,
              "water_axis_consistency": water_consistency, "water_alignment": alignment,
              "residual_consistency": residual_consistency, "residual_instability": instability,
              "visibility_count": count, "visibility_support": support, "contamination": contamination,
              "alpha": alpha, "water_dir": axis, "active": active, "fallback": ~active,
              "corrected": alpha > 0, "water_valid": wvalid, "transmission_valid": tvalid}
    return {key: torch.nan_to_num(value.detach(), nan=0., posinf=0., neginf=0.) if value.is_floating_point() else value.detach()
            for key, value in result.items()}


def apply_gcrc(s_full: Tensor, s_base: Tensor, contamination: Mapping[str, Tensor], *, taba=None):
    """Correct along the raw RGB prototype; alpha=0 preserves baseline bits.

    The prescribed raw RGB axis can change mean logits despite leaving the
    original luminance term unscaled. Combined mode applies GCRC then TABA.
    """
    residual = s_full-s_base
    chroma = residual-residual.mean(-1, keepdim=True)
    axis, alpha = contamination["water_dir"].detach(), contamination["alpha"].detach()
    # Mask invalid observations before arithmetic: zero times NaN is not zero.
    finite = torch.isfinite(chroma).all(-1, keepdim=True)
    clean = torch.where(finite, chroma, torch.zeros_like(chroma))
    parallel = (clean*axis).sum(-1, keepdim=True)*axis
    orthogonal = clean-parallel
    corrected = (alpha > 0) & finite
    gcrc_logits = torch.where(corrected, s_full-alpha*parallel, s_full)
    adjusted = gcrc_logits
    baseline_rgb = torch.sigmoid(s_full)
    if taba is not None:
        changed_delta = gcrc_logits-s_base
        changed_chroma = changed_delta-changed_delta.mean(-1, keepdim=True)
        ta = taba["water_axis"].detach()
        ta_parallel = (changed_chroma*ta).sum(-1, keepdim=True)*ta
        allocation = torch.where(taba["fallback_mask"].detach(), changed_chroma, ta_parallel)
        adjusted = gcrc_logits+(taba["gate"].detach()-1)*allocation
        adjusted = torch.where(corrected, adjusted, taba["s_adjusted"])
        baseline_rgb = taba["rgb"]
    rgb = torch.where(corrected, safe_sigmoid(adjusted), baseline_rgb)
    return {"rgb": rgb, "s_adjusted": adjusted,
            "delta_parallel_magnitude": parallel.norm(dim=-1, keepdim=True).detach(),
            "delta_orthogonal_magnitude": orthogonal.norm(dim=-1, keepdim=True).detach(),
            "luminance_logit_change": (gcrc_logits-s_full).mean(-1, keepdim=True).detach(),
            "decomposition_error": (parallel+orthogonal-clean).abs().amax(-1, keepdim=True).detach()}


@torch.no_grad()
def gcrc_statistics(outputs):
    visible = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
    reference = outputs["gaussian_view_rgb"]
    present = "gcrc_alpha" in outputs
    stats = {"gcrc_history_available": float(present)}
    for key in GCRC_QUANTITIES:
        default = 1. if key in ("transmission", "residual_consistency") else 0.
        value = outputs.get("gcrc_"+key, reference.new_full((visible.numel(), 1), default)).detach()[visible].float().reshape(-1)
        stats["gcrc_"+key+"_mean"] = float(value.mean()) if value.numel() else default
        stats["gcrc_"+key+"_std"] = float(value.std(correction=0)) if value.numel() else 0.
        for label, q in (("p10", .1), ("p50", .5), ("p90", .9)):
            stats["gcrc_"+key+"_"+label] = float(torch.quantile(value, q)) if value.numel() else default
    for key, source, default in (("active_fraction", "active", 0.), ("fallback_fraction", "fallback", 1.),
        ("corrected_fraction", "corrected", 0.), ("delta_parallel_magnitude", "delta_parallel_magnitude", 0.),
        ("delta_orthogonal_magnitude", "delta_orthogonal_magnitude", 0.),
        ("luminance_logit_change_abs", "luminance_logit_change", 0.), ("decomposition_error", "decomposition_error", 0.)):
        value = outputs.get("gcrc_"+source, reference.new_full((visible.numel(), 1), default)).detach()[visible].float().abs()
        stats["gcrc_"+key] = float(value.mean()) if value.numel() else default
    corrected = outputs.get("gcrc_corrected", torch.zeros_like(reference[:, :1], dtype=torch.bool)).reshape(-1)
    values = reference.detach()[corrected]
    stats["gcrc_corrected_intrinsic_min"] = float(values.min()) if values.numel() else .5
    stats["gcrc_corrected_intrinsic_max"] = float(values.max()) if values.numel() else .5
    values = reference.detach()[~corrected]
    stats["gcrc_zero_alpha_intrinsic_endpoint_fraction"] = float(((values == 0) | (values == 1)).float().mean()) if values.numel() else 0.
    return stats
