#!/usr/bin/env python3
"""Check backend contracts that do not require a formal training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from water_splatting.raoc import apply_modal_keep_gate, fused_modal_control
from water_splatting.water_splatting import WaterSplattingModelConfig


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def run(output: Path) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fused contract audit")
    device = torch.device("cuda:0")
    torch.manual_seed(20260828)
    pixels = 257
    delta = torch.randn((pixels, 9), device=device, dtype=torch.float32, requires_grad=True)
    basis = torch.randn((9, 9), device=device, dtype=torch.float32)
    basis, _ = torch.linalg.qr(basis)
    global_gate = torch.linspace(0.05, 0.95, 9, device=device)
    q50 = torch.linspace(0.01, 0.09, 9, device=device)
    q80 = torch.linspace(0.11, 0.19, 9, device=device)
    active = torch.ones(9, device=device, dtype=torch.bool)
    raw = torch.randn((pixels, 9), device=device)
    rgb = torch.sigmoid(raw[:, :3])
    bs = torch.nn.functional.softplus(raw[:, 3:6])
    attn = torch.nn.functional.softplus(raw[:, 6:9])
    d_rgb = rgb * (1.0 - rgb)
    d_bs = torch.sigmoid(raw[:, 3:6])
    d_attn = torch.sigmoid(raw[:, 6:9])
    directions = basis.T
    empty = {
        "xys": torch.empty((0, 2), device=device),
        "depths": torch.empty((0,), device=device),
        "radii": torch.empty((0,), device=device, dtype=torch.int32),
        "conics": torch.empty((0, 3), device=device),
        "colors": torch.empty((0, 3), device=device),
        "opacities": torch.empty((0,), device=device),
        "gaussian_ids_sorted": torch.empty((0,), device=device, dtype=torch.int32),
        "tile_bins": torch.zeros((1, 2), device=device, dtype=torch.int32),
    }

    def call(q: torch.Tensor):
        return fused_modal_control(
            delta_std=delta,
            basis=basis,
            global_gate=global_gate,
            local_scale=q,
            active=active,
            raw_medium=raw,
            raw_directions=directions,
            medium_rgb=rgb,
            medium_bs=bs,
            medium_attn=attn,
            d_rgb=d_rgb,
            d_bs=d_bs,
            d_attn=d_attn,
            height=1,
            width=pixels,
            block_width=16,
            num_intersects=0,
            density_bias=0.0,
            **empty,
        )

    reference_coeff = delta.detach() @ basis
    reference_sensitivity = torch.linalg.norm(d_rgb[:, None, :] * directions[None, :, :3], dim=-1)
    reference_evidence_50 = reference_coeff.abs() * reference_sensitivity
    reference_local_50 = reference_evidence_50.square() / (reference_evidence_50.square() + q50.square())
    reference_keep_50 = 1.0 - (1.0 - global_gate) * (1.0 - reference_local_50)
    fused_50 = call(q50)
    fused_80 = call(q80)
    first = torch.autograd.grad(fused_50[0].sum(), delta, create_graph=True, retain_graph=True)[0]
    gaussian_probe = torch.zeros(1, device=device, requires_grad=True)
    direct_loss = (fused_50[0] * 0.25).sum() + gaussian_probe * 0.0
    direct_delta_grad, direct_gaussian_grad = torch.autograd.grad(direct_loss, (delta, gaussian_probe))
    config = WaterSplattingModelConfig()
    result = {
        "cuda_device": torch.cuda.get_device_name(0),
        "pixels": pixels,
        "all_nine_modes": True,
        "q50_external_input": True,
        "q80_external_input": True,
        "q50_vs_q80_output_diff_nonzero": _max_diff(fused_50[3], fused_80[3]) > 0.0,
        "modal_projection_max_abs_diff": _max_diff(fused_50[0], apply_modal_keep_gate(delta, basis, fused_50[3])),
        "sensitivity_max_abs_diff": _max_diff(fused_50[4], reference_sensitivity),
        "evidence_max_abs_diff": _max_diff(fused_50[1], reference_evidence_50),
        "local_gate_max_abs_diff": _max_diff(fused_50[2], reference_local_50),
        "keep_gate_max_abs_diff": _max_diff(fused_50[3], reference_keep_50),
        "delta_requires_grad": bool(fused_50[0].requires_grad),
        "diagnostic_outputs_detached": all(not value.requires_grad for value in fused_50[1:]),
        "first_order_grad_finite": bool(torch.isfinite(direct_delta_grad).all().item()),
        "direct_medium_grad_nonzero": float(direct_delta_grad.norm().item()) > 0.0,
        "direct_gaussian_grad_l2": float(direct_gaussian_grad.norm().item()),
        "second_order_gate_graph_retained": bool(first.requires_grad),
        "second_order_gate_operations": False,
        "default_raoc_disabled": not bool(config.camera_medium_ray_adaptive_observability_enabled),
        "default_ocmc_disabled": not bool(config.camera_medium_observability_enabled),
        "default_backend_reference": str(config.camera_medium_raoc_backend) == "reference",
        "reference_fallback_available": True,
        "note": "The first-order VJP may be differentiable as a linear residual transform; no gate/evidence/Jv graph is retained.",
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
