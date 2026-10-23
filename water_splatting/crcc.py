"""Detached clear-chroma history and an appearance-only consistency loss."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


CRCC_CONFIG = {
    "enabled": False, "mode": "disabled", "diagnostic_only": False,
    "beta_ema": .95, "charbonnier_eps": 1e-3, "support_scale": 4.,
    "min_observations": 2, "medium_weight_floor": .25,
    "start_step": 3000, "ramp_steps": 2000, "lambda_max": .02,
}
CRCC_FIELDS = ("crcc_chroma_ema", "crcc_initialized", "crcc_observation_count")
CRCC_QUANTITIES = ("observation_count", "support", "medium_weight")
CRCC_STATS = tuple("crcc_" + key + "_" + stat for key in CRCC_QUANTITIES
                   for stat in ("mean", "p10", "p50", "p90")) + (
    "crcc_loss_raw", "crcc_loss_weighted", "crcc_lambda", "crcc_effective_count",
    "crcc_visible_count", "crcc_initialized_ratio", "crcc_weight_sum",
    "crcc_chroma_magnitude", "crcc_consistency_residual_magnitude",
    "crcc_invalid_transmission_fraction", "crcc_history_available", "crcc_state_mismatch",
    "crcc_intrinsic_min", "crcc_intrinsic_max",
)


def config_values(config):
    return {key: getattr(config, "crcc_" + key, value) for key, value in CRCC_CONFIG.items()}


def validate_config(config):
    c = config_values(config)
    if c["mode"] not in ("disabled", "crcc", "taba", "crcc_taba"):
        raise ValueError("unknown CRCC mode")
    if not 0 <= c["beta_ema"] < 1 or not 0 <= c["medium_weight_floor"] <= 1:
        raise ValueError("invalid CRCC EMA or medium floor")
    for key in ("charbonnier_eps", "support_scale"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("invalid CRCC " + key)
    if not math.isfinite(c["lambda_max"]) or c["lambda_max"] < 0:
        raise ValueError("invalid CRCC lambda_max")
    if c["min_observations"] < 2 or c["start_step"] < 0 or c["ramp_steps"] < 0:
        raise ValueError("invalid CRCC observation or schedule configuration")


def loss_coefficient(step, *, start_step=3000, ramp_steps=2000, lambda_max=.02):
    ramp = float(step >= start_step) if ramp_steps == 0 else min(max((step-start_step)/ramp_steps, 0.), 1.)
    return float(lambda_max) * ramp


class ClearRadianceHistory(nn.Module):
    def __init__(self, reference: Tensor):
        super().__init__()
        n = reference.shape[0]
        self.register_buffer("crcc_chroma_ema", reference.new_zeros(n, 3))
        self.register_buffer("crcc_initialized", torch.zeros(n, 1, device=reference.device, dtype=torch.bool))
        self.register_buffer("crcc_observation_count", torch.zeros(n, 1, device=reference.device, dtype=torch.int64))
        self.register_buffer("last_update_step", torch.tensor(-1, device=reference.device, dtype=torch.int64))

    def check_shape(self, count):
        for key in CRCC_FIELDS:
            if getattr(self, key).shape != (count, 3 if key == "crcc_chroma_ema" else 1):
                raise RuntimeError("CRCC state/Gaussian mismatch: " + key)

    def values(self, indices=None):
        return {key: getattr(self, key) if indices is None else getattr(self, key)[indices] for key in CRCC_FIELDS}

    @torch.no_grad()
    def update(self, chroma, visible, valid, *, step, beta_ema=.95):
        self.check_shape(chroma.shape[0])
        if step <= int(self.last_update_step):
            return
        c = chroma.detach()
        accepted = visible.detach().reshape(-1, 1).bool() & valid.detach().reshape(-1, 1).bool()
        accepted &= torch.isfinite(c).all(-1, keepdim=True)
        proposal = torch.where(self.crcc_initialized,
                               beta_ema*self.crcc_chroma_ema + (1-beta_ema)*c, c)
        self.crcc_chroma_ema.copy_(torch.where(accepted, proposal, self.crcc_chroma_ema))
        self.crcc_initialized.logical_or_(accepted)
        self.crcc_observation_count.add_(accepted.long())
        self.last_update_step.fill_(step)

    @torch.no_grad()
    def densify(self, split_mask, samples, clone_mask):
        self.check_shape(split_mask.numel())
        for key in CRCC_FIELDS:
            value = getattr(self, key)
            setattr(self, key, torch.cat((value, value[split_mask].repeat(samples, 1), value[clone_mask]), dim=0))

    @torch.no_grad()
    def prune(self, removed):
        self.check_shape(removed.numel())
        for key in CRCC_FIELDS:
            setattr(self, key, getattr(self, key)[~removed])


def clear_consistency(rgb, state, transmission, visible, sample_valid, *,
                      charbonnier_eps=1e-3, support_scale=4., min_observations=2,
                      medium_weight_floor=.25, alpha=None):
    """Use previous completed-step history; invalid auxiliaries have zero weight."""
    n = rgb.shape[0]
    current_valid = torch.isfinite(rgb.detach()).all(-1, keepdim=True)
    clean_rgb = torch.where(current_valid, rgb, torch.zeros_like(rgb))
    chroma = clean_rgb-clean_rgb.mean(-1, keepdim=True)
    shapes_ok = all(state[key].shape == (n, 3 if key == "crcc_chroma_ema" else 1) for key in CRCC_FIELDS)
    with torch.no_grad():
        visible = visible.detach().reshape(n, 1).bool()
        t = transmission.detach()
        valid_t = torch.isfinite(t).all(-1, keepdim=True) & ((t >= 0) & (t <= 1)).all(-1, keepdim=True)
        valid_t &= sample_valid.detach().reshape(n, 1).bool()
        medium_weight = medium_weight_floor + (1-medium_weight_floor)*(1-t.mean(-1, keepdim=True)).clamp(0, 1)
        medium_weight = torch.where(valid_t, medium_weight, torch.full_like(medium_weight, medium_weight_floor))
        count = state["crcc_observation_count"].detach() if shapes_ok else torch.zeros(n, 1, device=rgb.device, dtype=torch.int64)
        initialized = state["crcc_initialized"].detach() if shapes_ok else torch.zeros_like(visible)
        target = state["crcc_chroma_ema"].detach() if shapes_ok else torch.zeros_like(rgb)
        valid_history = torch.isfinite(target).all(-1, keepdim=True) & (target.abs() <= 1).all(-1, keepdim=True) & (count >= 0)
        target = torch.where(valid_history, target, torch.zeros_like(target))
        support = -torch.expm1(-count.clamp_min(0).to(rgb.dtype)/support_scale)
        active = visible & current_valid & valid_t & valid_history & initialized & (count >= min_observations)
        alpha_weight = torch.ones_like(support) if alpha is None else torch.nan_to_num(alpha.detach().reshape(n, 1), nan=0., posinf=0., neginf=0.).clamp(0, 1)
        weight = (active.to(rgb.dtype)*support*medium_weight*alpha_weight).clamp(0, 1)
        update_valid = current_valid & valid_t
    residual = chroma-target
    penalty = (residual.square()+charbonnier_eps**2).sqrt().mean(-1, keepdim=True)
    loss = (weight*penalty).sum()/(weight.sum()+1e-8)
    return {"loss": loss, "chroma": chroma.detach(), "residual": residual.detach(),
            "weight": weight, "support": support, "medium_weight": medium_weight,
            "observation_count": count, "initialized": initialized, "active": weight > 0,
            "valid_transmission": valid_t, "update_valid": update_valid,
            "state_mismatch": rgb.new_tensor(float(not shapes_ok)), "intrinsic": clean_rgb.detach()}


@torch.no_grad()
def crcc_statistics(outputs):
    visible = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
    reference = outputs["gaussian_view_rgb"]
    n = visible.numel()
    stats = {}
    present = "crcc_loss" in outputs
    for key in CRCC_QUANTITIES:
        value = outputs.get("crcc_"+key, reference.new_zeros(n, 1)).detach()[visible].float().flatten()
        stats["crcc_"+key+"_mean"] = float(value.mean()) if value.numel() else 0.
        for label,q in (("p10",.1),("p50",.5),("p90",.9)):
            stats["crcc_"+key+"_"+label] = float(torch.quantile(value,q)) if value.numel() else 0.
    raw = float(outputs.get("crcc_loss", reference.new_zeros(())).detach())
    coefficient = float(outputs.get("crcc_lambda", reference.new_zeros(())))
    stats.update(crcc_loss_raw=raw, crcc_loss_weighted=raw*coefficient, crcc_lambda=coefficient,
                 crcc_visible_count=int(visible.sum()), crcc_history_available=float(present),
                 crcc_state_mismatch=float(outputs.get("crcc_state_mismatch", reference.new_zeros(()))))
    for key,source in (("effective_count","active"),("initialized_ratio","initialized"),
                       ("weight_sum","weight"),("invalid_transmission_fraction","valid_transmission")):
        value = outputs.get("crcc_"+source, reference.new_zeros(n, 1)).detach()[visible].float()
        if source == "valid_transmission":
            value = 1-value if present else torch.zeros_like(value)
        stats["crcc_"+key] = float(value.sum() if key in ("effective_count","weight_sum") else value.mean()) if value.numel() else 0.
    for key,source in (("chroma_magnitude","chroma"),("consistency_residual_magnitude","residual")):
        value = outputs.get("crcc_"+source, torch.zeros_like(reference)).detach()[visible]
        stats["crcc_"+key] = float(value.norm(dim=-1).mean()) if value.numel() else 0.
    value = outputs.get("crcc_intrinsic", reference.new_full((n,3),.5)).detach()[visible]
    stats["crcc_intrinsic_min"] = float(value.min()) if value.numel() else .5
    stats["crcc_intrinsic_max"] = float(value.max()) if value.numel() else .5
    return stats
