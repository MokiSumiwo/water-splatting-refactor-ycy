#!/usr/bin/env python
"""Offline checks for MCMC relocation math.

This script avoids Nerfstudio initialization. It validates the pure-Torch port
of the 3DGS-MCMC relocation transform and a minimal Adam-state reset pattern
for relocated rows.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List

import torch

from water_splatting.density_control import alpha_after_coincident_split, alpha_to_logit, relocation_alpha_and_scale


def _official_denominator(alpha: torch.Tensor, copies: int) -> torch.Tensor:
    denom = torch.zeros_like(alpha.float())
    for i in range(1, copies + 1):
        for k in range(i):
            coeff = ((-1.0) ** k) * torch.tensor(float(math.comb(i - 1, k)), device=alpha.device)
            coeff = coeff / torch.sqrt(torch.tensor(float(k + 1), device=alpha.device))
            denom = denom + coeff * alpha.pow(k + 1)
    return denom


def _projected_alpha_mass(alpha: torch.Tensor, scale: torch.Tensor, copies: int) -> torch.Tensor:
    """Analytic 1D integral of coincident alpha-composited Gaussian copies."""

    denom = _official_denominator(alpha, copies)
    return torch.sqrt(torch.tensor(2.0 * math.pi, device=alpha.device)) * scale * denom


def density_preservation_cases(device: torch.device) -> List[Dict[str, float]]:
    counts = [1, 2, 3, 5, 10, 25, 50]
    opacities = [0.01, 0.1, 0.5, 0.9]
    scales = [0.03, 0.1, 0.4]
    rows: List[Dict[str, float]] = []
    for count in counts:
        for opacity in opacities:
            alpha = torch.tensor([opacity], device=device)
            for scale in scales:
                scale_tensor = torch.tensor([[scale, scale, scale]], device=device)
                new_alpha, new_scale = relocation_alpha_and_scale(
                    alpha,
                    scale_tensor,
                    torch.tensor([count], device=device),
                    min_output_alpha=0.0,
                )
                before = torch.sqrt(torch.tensor(2.0 * math.pi, device=device)) * alpha * scale
                after = _projected_alpha_mass(new_alpha, new_scale[:, 0], count)
                err = (after - before).abs()
                rows.append(
                    {
                        "copies": float(count),
                        "opacity": float(opacity),
                        "scale": float(scale),
                        "new_opacity": float(new_alpha[0].item()),
                        "new_scale": float(new_scale[0, 0].item()),
                        "mean_abs_error": float(err.mean().item()),
                        "max_abs_error": float(err.max().item()),
                    }
                )
    return rows


def optimizer_state_case(device: torch.device) -> Dict[str, object]:
    param = torch.nn.Parameter(torch.arange(18, device=device, dtype=torch.float32).reshape(6, 3) / 100.0)
    optimizer = torch.optim.Adam([param], lr=1e-3)
    loss = param.square().sum()
    loss.backward()
    optimizer.step()
    state = optimizer.state[param]
    before_exp_avg = state["exp_avg"].detach().clone()
    before_exp_avg_sq = state["exp_avg_sq"].detach().clone()
    reset_indices = torch.tensor([1, 4], device=device, dtype=torch.long)
    state["exp_avg"][reset_indices] = 0
    state["exp_avg_sq"][reset_indices] = 0
    untouched = torch.tensor([0, 2, 3, 5], device=device, dtype=torch.long)
    return {
        "reset_exp_avg_zero": bool((state["exp_avg"][reset_indices] == 0).all().item()),
        "reset_exp_avg_sq_zero": bool((state["exp_avg_sq"][reset_indices] == 0).all().item()),
        "untouched_exp_avg_same": bool(torch.equal(state["exp_avg"][untouched], before_exp_avg[untouched])),
        "untouched_exp_avg_sq_same": bool(torch.equal(state["exp_avg_sq"][untouched], before_exp_avg_sq[untouched])),
        "param_group_identity_ok": optimizer.param_groups[0]["params"][0] is param,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mean-threshold", type=float, default=1e-4)
    parser.add_argument("--max-threshold", type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device(args.device)

    rows = density_preservation_cases(device)
    max_mean = max(row["mean_abs_error"] for row in rows)
    max_abs = max(row["max_abs_error"] for row in rows)
    optimizer_result = optimizer_state_case(device)
    alpha_roundtrip = torch.sigmoid(alpha_to_logit(torch.tensor([0.01, 0.5, 0.9], device=device)))
    exact_split = alpha_after_coincident_split(torch.tensor([0.01, 0.5, 0.9], device=device), 5)
    result = {
        "density_cases": len(rows),
        "max_mean_abs_error": max_mean,
        "max_abs_error": max_abs,
        "optimizer_state": optimizer_result,
        "alpha_roundtrip": [float(x) for x in alpha_roundtrip.cpu()],
        "exact_split": [float(x) for x in exact_split.cpu()],
    }
    print(json.dumps(result, indent=2))
    if max_mean >= args.mean_threshold or max_abs >= args.max_threshold:
        raise SystemExit("density preservation threshold failed")
    if not all(bool(v) for v in optimizer_result.values()):
        raise SystemExit("optimizer state threshold failed")


if __name__ == "__main__":
    main()
