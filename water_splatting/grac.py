"""Detached per-Gaussian appearance history and reliability allocation."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from water_splatting.fields.gaussian_appearance import safe_sigmoid


GRAC_FIELDS = ("direction_prototype", "magnitude_ema", "magnitude_sq_ema", "visibility_count",
               "reliability_initialized_mask", "magnitude_count")
GRAC_CONFIG = {
    "enabled": False, "mode": "disabled", "diagnostic_only": False,
    "beta_direction": .95, "beta_magnitude": .95, "residual_eps": 1e-5,
    "visibility_scale": 8.0, "min_visibility_count": 4, "reliability_start_step": 3000,
    "weight_direction": .50, "weight_magnitude": .30, "weight_visibility": .20,
    "chroma_floor": .35,
}
GRAC_QUANTITIES = ("visibility_count", "direction_consistency", "magnitude_stability",
                   "visibility_confidence", "pseudo_reliability", "gate", "final_parallel_gate")
GRAC_STATS = tuple("grac_" + key + "_" + stat for key in GRAC_QUANTITIES
                   for stat in ("mean", "p10", "p50", "p90")) + (
    "grac_active_fraction", "grac_fallback_fraction", "grac_initialized_fraction",
    "grac_history_available", "grac_variance_mean", "grac_luminance_logit_change_abs")


def config_values(config):
    return {key: getattr(config, "grac_" + key, default) for key, default in GRAC_CONFIG.items()}


def validate_config(config):
    c = config_values(config)
    if c["mode"] not in ("disabled", "grac", "taba", "grac_taba"):
        raise ValueError("unknown GRAC mode")
    for key in ("beta_direction", "beta_magnitude", "chroma_floor"):
        if not 0 <= c[key] <= 1 or (key.startswith("beta") and c[key] == 1):
            raise ValueError("invalid GRAC " + key)
    for key in ("residual_eps", "visibility_scale"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("invalid GRAC " + key)
    weights = [c["weight_" + key] for key in ("direction", "magnitude", "visibility")]
    if not all(math.isfinite(x) and x >= 0 for x in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("GRAC reliability weights must be nonnegative and sum to one")
    if c["min_visibility_count"] < 0 or c["reliability_start_step"] < 0:
        raise ValueError("GRAC count and start step must be nonnegative")


class GaussianReliabilityState(nn.Module):
    """O(N) non-parameter buffers; topology follows Gaussian parameter order."""

    def __init__(self, reference: Tensor):
        super().__init__()
        n = reference.shape[0]
        self.register_buffer("direction_prototype", reference.new_zeros(n, 3))
        self.register_buffer("magnitude_ema", reference.new_zeros(n, 1))
        self.register_buffer("magnitude_sq_ema", reference.new_zeros(n, 1))
        self.register_buffer("visibility_count", torch.zeros(n, 1, device=reference.device, dtype=torch.int64))
        self.register_buffer("reliability_initialized_mask", torch.zeros(n, 1, device=reference.device, dtype=torch.bool))
        self.register_buffer("magnitude_count", torch.zeros(n, 1, device=reference.device, dtype=torch.int64))
        self.register_buffer("last_update_step", torch.tensor(-1, device=reference.device, dtype=torch.int64))

    def check_shape(self, count):
        for key in GRAC_FIELDS:
            expected = (count, 3 if key == "direction_prototype" else 1)
            if getattr(self, key).shape != expected:
                raise RuntimeError(f"GRAC state/Gaussian mismatch: {key}, expected {expected}")

    def values(self, indices=None):
        return {key: getattr(self, key) if indices is None else getattr(self, key)[indices] for key in GRAC_FIELDS}

    @torch.no_grad()
    def update(self, chroma: Tensor, visible: Tensor, *, step: int, beta_direction=.95,
               beta_magnitude=.95, residual_eps=1e-5):
        self.check_shape(chroma.shape[0])
        if step <= int(self.last_update_step):
            return
        visible = visible.detach().reshape(-1, 1).bool()
        self.visibility_count.add_(visible.to(torch.int64))
        d = chroma.detach()
        a = d.norm(dim=-1, keepdim=True)
        valid = visible & torch.isfinite(d).all(-1, keepdim=True) & torch.isfinite(a) & torch.isfinite(a.square())
        update_direction = valid & (a > residual_eps)
        u = torch.nan_to_num(d / (a + 1e-8))
        proposal = beta_direction * self.direction_prototype + (1 - beta_direction) * u
        norm = proposal.norm(dim=-1, keepdim=True)
        update_direction &= torch.isfinite(proposal).all(-1, keepdim=True) & (norm > residual_eps)
        self.direction_prototype.copy_(torch.where(update_direction, proposal / norm.clamp_min(residual_eps), self.direction_prototype))
        self.reliability_initialized_mask.logical_or_(update_direction)
        first = self.magnitude_count == 0
        mean = torch.where(first, a, beta_magnitude * self.magnitude_ema + (1-beta_magnitude) * a)
        second = torch.where(first, a.square(), beta_magnitude * self.magnitude_sq_ema + (1-beta_magnitude) * a.square())
        self.magnitude_ema.copy_(torch.where(valid, mean, self.magnitude_ema))
        self.magnitude_sq_ema.copy_(torch.where(valid, second, self.magnitude_sq_ema))
        self.magnitude_count.add_(valid.to(torch.int64))
        self.last_update_step.fill_(step)

    @torch.no_grad()
    def densify(self, split_mask: Tensor, samples: int, clone_mask: Tensor):
        self.check_shape(split_mask.numel())
        for key in GRAC_FIELDS:
            value = getattr(self, key)
            setattr(self, key, torch.cat((value, value[split_mask].repeat(samples, 1), value[clone_mask]), dim=0))

    @torch.no_grad()
    def prune(self, removed: Tensor):
        self.check_shape(removed.numel())
        for key in GRAC_FIELDS:
            setattr(self, key, getattr(self, key)[~removed])


@torch.no_grad()
def compute_grac_reliability(chroma: Tensor, state: Mapping[str, Tensor], *, step: int,
                             residual_eps=1e-5, visibility_scale=8.0, min_visibility_count=4,
                             reliability_start_step=3000, weight_direction=.50,
                             weight_magnitude=.30, weight_visibility=.20, chroma_floor=.35):
    d = chroma.detach()
    a = d.norm(dim=-1, keepdim=True)
    p, m, m2 = (state[key].detach() for key in ("direction_prototype", "magnitude_ema", "magnitude_sq_ema"))
    count = state["visibility_count"].detach()
    initialized = state["reliability_initialized_mask"].detach()
    valid = (torch.isfinite(d).all(-1, keepdim=True) & torch.isfinite(a)
             & torch.isfinite(p).all(-1, keepdim=True) & torch.isfinite(m) & torch.isfinite(m2)
             & (m >= 0) & (m2 >= 0) & (count >= 0) & (state["magnitude_count"] >= 0))
    valid &= ~initialized | (p.norm(dim=-1, keepdim=True) > residual_eps)
    unit = torch.nan_to_num(d / (a + 1e-8))
    direction = ((1 + (unit * p).sum(-1, keepdim=True)) / 2).clamp(0, 1)
    direction = torch.where(initialized & (a > residual_eps), direction, torch.ones_like(direction))
    variance = (m2 - m.square()).clamp_min(0)
    cv = variance.sqrt() / (m.abs() + residual_eps)
    stability = (1 / (1 + cv)).clamp(0, 1)
    stability = torch.where((state["magnitude_count"] >= 2) & (m > residual_eps), stability, torch.ones_like(stability))
    confidence = -torch.expm1(-count.to(d.dtype) / visibility_scale)
    reliability = (weight_direction * direction + weight_magnitude * stability + weight_visibility * confidence).clamp(0, 1)
    valid &= torch.isfinite(reliability) & torch.isfinite(variance)
    active = valid & (count >= min_visibility_count) & (step >= reliability_start_step)
    gate = torch.where(active, chroma_floor + (1-chroma_floor) * reliability, torch.ones_like(reliability))
    result = {"visibility_count": count, "direction_consistency": direction,
              "magnitude_stability": stability, "visibility_confidence": confidence,
              "pseudo_reliability": reliability, "gate": gate, "variance": variance,
              "active": active, "fallback": ~active, "initialized": initialized, "valid": valid}
    return {key: torch.nan_to_num(value.detach(), nan=1., posinf=1., neginf=0.) if value.is_floating_point() else value.detach()
            for key, value in result.items()}


def apply_grac(s_full: Tensor, s_base: Tensor, reliability: Mapping[str, Tensor], *,
               taba=None, chroma_floor=.35):
    residual = s_full - s_base
    chroma = residual - residual.mean(-1, keepdim=True)
    gate = reliability["gate"].detach()
    adjusted = s_full + (gate - 1) * chroma
    final = gate
    if taba is not None:
        axis = taba["water_axis"].detach()
        parallel = (chroma * axis).sum(-1, keepdim=True) * axis
        allocation = torch.where(taba["fallback_mask"], chroma, parallel)
        tg = taba["gate"].detach()
        final = torch.where(reliability["active"], (tg * gate).clamp(chroma_floor, 1), tg)
        adjusted = adjusted + (final - gate) * allocation
    return {"rgb": safe_sigmoid(adjusted), "s_adjusted": adjusted,
            "final_parallel_gate": final.detach(),
            "luminance_logit_change": (adjusted-s_full).mean(-1, keepdim=True).detach()}


@torch.no_grad()
def grac_statistics(outputs):
    visible = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
    n = visible.numel()
    reference = outputs["gaussian_view_rgb"]
    present = "grac_gate" in outputs
    stats = {"grac_history_available": float(present)}
    for key in GRAC_QUANTITIES:
        default = 0.0 if key == "visibility_count" else 1.0
        value = outputs.get("grac_" + key, reference.new_full((n, 1), default)).detach()[visible].float().reshape(-1)
        stats["grac_" + key + "_mean"] = float(value.mean()) if value.numel() else default
        for label, q in (("p10", .1), ("p50", .5), ("p90", .9)):
            stats["grac_" + key + "_" + label] = float(torch.quantile(value, q)) if value.numel() else default
    for key, source, default in (("active_fraction", "active", 0.), ("fallback_fraction", "fallback", 1.),
                                 ("initialized_fraction", "initialized", 0.), ("variance_mean", "variance", 0.),
                                 ("luminance_logit_change_abs", "luminance_logit_change", 0.)):
        value = outputs.get("grac_" + source, reference.new_full((n, 1), default)).detach()[visible].float().abs()
        stats["grac_" + key] = float(value.mean()) if value.numel() else default
    return stats
