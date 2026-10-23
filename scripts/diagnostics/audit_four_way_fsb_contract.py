#!/usr/bin/env python3
"""Post-run verification of the backward-only backend used by the four arms."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.diagnostics import audit_renderer_backward_fix as A
from scripts.experiments import run_direct_mdrr_cica_scene as W
from water_splatting.rasterize import rasterize_gaussians
import water_splatting.cuda._backend as backend


def synthetic(native, fixed):
    rows = []
    cases = {
        "single_high_alpha": ([.995], 0.),
        "multiple_high_alpha": ([.995, .8, .4], 0.),
        "early_termination": ([.99, .99, .99, .9], 0.),
        "medium_and_threshold": ([.9, .6, .001, .8], .4),
        "capped_alpha": ([.9999, .8], 0.),
    }
    for name, (alpha, attenuation) in cases.items():
        n = len(alpha)
        colors = torch.full((n, 3), .4, device="cuda")
        opacity = torch.tensor(alpha, device="cuda")

        def render(ext, c):
            backend._C = ext
            return rasterize_gaussians(
                torch.zeros(n, 2, device="cuda"), torch.zeros(n, 2, device="cuda"),
                torch.arange(1, n + 1, device="cuda", dtype=torch.float32),
                torch.full((n,), 3, device="cuda", dtype=torch.int32),
                torch.tensor([[1., 0., 1.]], device="cuda").repeat(n, 1),
                torch.ones(n, device="cuda", dtype=torch.int32), c, opacity,
                torch.full((1, 1, 3), .2, device="cuda"),
                torch.full((1, 1, 3), .1, device="cuda"),
                torch.full((1, 1, 3), attenuation, device="cuda"),
                1, 1, 16, return_alpha=True, step=14999, return_hit_stats=True)[0]

        # Color is affine at fixed geometry/opacity, so basis renders give an
        # independent derivative without using either backward implementation.
        with torch.no_grad():
            zero = render(native, torch.zeros_like(colors))
            ref = torch.zeros_like(colors)
            for i in range(n):
                basis = torch.zeros_like(colors)
                basis[i] = 1.
                ref[i] = (render(native, basis) - zero)[0, 0]
        values = {}
        images = {}
        for label, ext in (("native", native), ("fixed", fixed)):
            c = colors.clone().requires_grad_(True)
            im = render(ext, c)
            grad, = torch.autograd.grad(im.sum(), (c,))
            values[label + "_color_max_abs_error"] = float((grad - ref).abs().max())
            images[label] = im.detach()
        rows.append({"case": name, **values,
                     "forward_equal": torch.equal(images["native"], images["fixed"]),
                     "forward_max_abs": float((images["native"] - images["fixed"]).abs().max()),
                     "reference_color_derivatives": ref[:, 0].tolist()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, choices=tuple(W.SCENE_GPUS))
    p.add_argument("--gpu", required=True, choices=("6", "7", "8", "9"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
    assert os.environ.get("CONDA_DEFAULT_ENV") == "water_splatting"
    args.output.mkdir(parents=True, exist_ok=False)
    W._route_renderer_backend("legacy")
    native = backend._C
    route = W._route_renderer_backend("fixed")
    fixed = backend._C
    binaries = {"production": Path(native.__file__),
                "fixed": Path(route["build"]) / (route["extension"] + ".so")}
    before = {k: W._sha256(v) for k, v in binaries.items()}
    checks = {name: getattr(fixed, name) is getattr(native, name)
              for name in dir(native) if not name.startswith("_") and callable(getattr(native, name))
              and name not in route["changed_functions"]}
    rows = synthetic(native, fixed)
    real = A.real_scene(args.scene, args.gpu, native, fixed)
    result = {
        "scene": args.scene, "gpu": args.gpu, "route": route,
        "timing": "post-training supplemental audit; not a preregistered training gate",
        "checkpoint_scope": "historical BND+CICA and FSB retained checkpoints; v4 endpoints were deleted",
        "no_optimizer_steps": True, "unchanged_function_identity": checks,
        "binary_sha256": before,
        "binaries_unchanged": before == {k: W._sha256(v) for k, v in binaries.items()},
        "synthetic": rows, "real_scene": real,
        "all_forward_exact": all(checks.values()) and all(r["forward_equal"] for r in rows)
                             and all(r["forward_max_abs"] == 0 for r in real),
        "synthetic_color_pass": all(r["fixed_color_max_abs_error"] < 2e-6 for r in rows),
        "real_color_pass": all(r["fixed_reference_relative_l2"] < .02 for r in real),
        "scope_limit": "RGB color VJP only; not a proof of opacity, geometry, medium or ND backward correctness",
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("scene", "all_forward_exact", "synthetic_color_pass", "real_color_pass")}))
    if not all(result[k] for k in ("all_forward_exact", "synthetic_color_pass", "real_color_pass", "binaries_unchanged")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
