#!/usr/bin/env python3
"""Synthetic post-run check of the actual four-way backend/evidence integration.

No checkpoint is changed and no optimizer step is taken. This verifies
implementation wiring, not endpoint VJP accuracy or performance gains.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.diagnostics.audit_renderer_backward_fix import synthetic
from water_splatting import csrc as production
import water_splatting.cuda._backend as backend
from water_splatting.fields.gaussian_appearance import (
    compute_bounded_headroom_gaussian_colors, compute_bounded_gaussian_colors,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get("CONDA_DEFAULT_ENV") == "water_splatting"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "7", "8", "9")
    assert not args.output.exists()
    torch.manual_seed(99)
    W._route_renderer_backend("legacy")
    old = synthetic()
    info = W._route_renderer_backend("fixed")
    forward_functions = ["rasterize_forward", "nd_rasterize_forward", "project_gaussians_forward",
                         "project_gaussians_backward", "compute_sh_forward", "compute_sh_backward"]
    routing = {f: getattr(backend._C, f) is getattr(production, f) for f in forward_functions}
    fixed = synthetic()
    assert all(routing.values())
    assert all(a["forward_derivative"] == b["forward_derivative"] for a, b in zip(old, fixed))
    assert max(x["abs_error"] for x in fixed) < 2e-6

    device = torch.device("cuda")
    geometry = {"height": 1, "width": 1, "xys": torch.zeros(1, 2, device=device),
                "conics": torch.tensor([[1., 0., 1.]], device=device), "depths": torch.ones(1, device=device)}
    bins = (torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([[0, 1]], dtype=torch.int32, device=device), (1, 1, 1))
    model = SimpleNamespace(device=device, opacities=torch.logit(torch.tensor([[.995]], device=device)))
    residual = torch.tensor([[[.1, -.1, .03]]], device=device)
    checks = []
    for method, fn in (("bounded_headroom_sh3", compute_bounded_headroom_gaussian_colors),
                       ("bounded_sh3", compute_bounded_gaussian_colors)):
        W.ACTIVE_PARAMETERIZATION = method
        dc = torch.tensor([[-3., .2, 3.]], device=device, requires_grad=True)
        rest = torch.randn(1, 15, 3, device=device) * 2
        output = fn(means=torch.tensor([[.3, .4, 1.]], device=device), features_dc=dc,
                    features_rest=rest, camera_position=torch.zeros(1, 3, device=device),
                    sh_degree=3, active_sh_degree=3)
        derivative, = torch.autograd.grad(output.rgb.sum(), dc)
        attenuation = torch.tensor([[[.1, .2, .3]]], device=device)
        outputs = {"gaussian_view_rgb": output.rgb, "gaussian_view_logits": output.logits,
                   "gaussian_view_dc_logits": output.dc_logits, "medium_attn": attenuation}
        results = {}
        with patch.object(W.FULL, "_project_for_camera", return_value=geometry), \
             patch.object(W.FULL, "_geometry_bins", return_value=bins):
            for mode in ("legacy", "exact_fsb"):
                W.CICA_DERIVATIVE_MODE = mode
                results[mode] = W._cica_local_evidence(model, None, outputs, residual)
        jac = .995 * torch.exp(-attenuation.reshape(1, 3)) * derivative
        expected_info = jac.square()
        expected_delta = residual.reshape(1, 3) / jac
        exact = results["exact_fsb"]
        info_rel = float((exact["information"] - expected_info).norm() / expected_info.norm())
        delta_rel = float((exact["delta_parameter"] - expected_delta).norm() / expected_delta.norm())
        assert info_rel < 2e-5 and delta_rel < 2e-5
        bounded_control_equal = None
        if method == "bounded_sh3":
            bounded_control_equal = all(torch.equal(results["legacy"][k], exact[k]) for k in exact)
            assert bounded_control_equal
        checks.append(dict(parameterization=method, information_relative_error=info_rel,
                           delta_relative_error=delta_rel, bnd_control_bitwise_equal=bounded_control_equal))
    W._route_renderer_backend("legacy")
    payload = dict(all_pass=True, no_optimizer_steps=True, physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                   synthetic_only=True, endpoint_checkpoint_audit=False, backend=info,
                   production_functions_preserved=routing, legacy=old, fixed=fixed, cica=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
