#!/usr/bin/env python3
"""Rebuild the CUDA extension with a forward/backward alpha-cap fix and audit it.

This diagnostic never overwrites ``water_splatting/csrc.so``.  It loads a
separately named JIT extension, temporarily routes the Python bindings to it,
and runs synthetic and frozen real-view VJP checks.  No model update occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.cpp_extension import _get_build_directory, load


def build_fixed_extension():
    import glob
    srcdir = ROOT / "water_splatting" / "cuda" / "csrc"
    sources = sorted(glob.glob(str(srcdir / "*.cu")) + glob.glob(str(srcdir / "*.cpp")))
    name = "water_splatting_cuda_fsb_fix_20260915"
    build = Path(_get_build_directory(name, verbose=False))
    ext = load(name=name, sources=sources, extra_cflags=["-O3"],
               extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
               extra_include_paths=[str(srcdir / "third_party" / "glm")], verbose=False)
    return ext, build


def route_extension(ext):
    import water_splatting.cuda._backend as backend
    import water_splatting.cuda as cuda
    old = backend._C
    backend._C = ext
    # cuda's lazy functions resolve backend._C at call time; no other patch.
    return old


def synthetic():
    from water_splatting.rasterize import rasterize_gaussians
    rows = []
    for opacity in (.5, .99, .995, .999):
        color = torch.full((1, 3), .4, device="cuda", requires_grad=True)
        output = rasterize_gaussians(
            torch.zeros(1, 2, device="cuda"), torch.zeros(1, 2, device="cuda"),
            torch.ones(1, device="cuda"), torch.tensor([3], device="cuda", dtype=torch.int32),
            torch.tensor([[1., 0., 1.]], device="cuda"), torch.ones(1, device="cuda", dtype=torch.int32),
            color, torch.tensor([[opacity]], device="cuda"), torch.zeros(1, 1, 3, device="cuda"),
            torch.zeros(1, 1, 3, device="cuda"), torch.zeros(1, 1, 3, device="cuda"),
            1, 1, 16, return_alpha=True, step=14999, return_hit_stats=True)[0]
        grad, = torch.autograd.grad(output.sum(), (color,))
        rows.append(dict(opacity=opacity, forward_derivative=float(output[0, 0, 0].detach()/.4),
                         backward_derivative=float(grad[0, 0]),
                         abs_error=float((grad[0, 0]-output[0, 0, 0].detach()/.4).abs())))
    return rows


def real_scene(scene: str, gpu: str, native_ext, fixed_ext):
    """Compare color VJPs on all six CICA bank cameras of both checkpoints."""
    from scripts.diagnostics.evaluate_fsb_degreewise_scene import W, run_paths, configure
    from water_splatting.rasterize import rasterize_gaussians
    import water_splatting.cuda._backend as backend
    paths = run_paths(scene, ROOT / "unused")
    records = []
    for method in ("BND_CICA", "FSB"):
        path = paths[method]
        configure(path)
        W._seed_all(42)
        branch = W._new_branch(scene)
        try:
            payload = W._load_snapshot(branch, scene, path / "checkpoints/step-000014999.ckpt")
            model = branch.pipeline.model
            bank = json.loads((path / "cica_camera_bank.json").read_text())
            train = W._train_records(branch)
            for ordinal, index in enumerate(bank["camera_indices"]):
                _, view, camera, batch = train[index]
                camera = camera.to(model.device)
                # Use the original forward path to obtain frozen geometry and medium.
                old_backend = native_ext
                backend._C = old_backend
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera)
                geometry = W.FULL._project_for_camera(model, camera)
                color0 = outputs["gaussian_view_rgb"].detach().float()
                opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)
                medium_rgb = outputs["medium_rgb"].detach().float()
                medium_bs = outputs["medium_bs"].detach().float()
                medium_attn = outputs["medium_attn"].detach().float()
                gt = W._gt(model, W._batch_to_device(batch, model.device), outputs["background"]).detach().float()
                cotangent = gt - outputs["pred_image"].detach().float()

                def direct(ext):
                    backend._C = ext
                    colors = color0.clone().requires_grad_(True)
                    image = rasterize_gaussians(
                        geometry["xys"].detach(), torch.zeros_like(geometry["xys"]),
                        geometry["depths"].detach(), geometry["radii"].detach(), geometry["conics"].detach(),
                        geometry["num_tiles_hit"].detach(), colors, opacity, medium_rgb, medium_bs, medium_attn,
                        geometry["height"], geometry["width"], 16, return_alpha=True, step=model.step,
                        return_hit_stats=True)[0]
                    forward = (image * cotangent).sum()
                    grad, = torch.autograd.grad(forward, (colors,))
                    return image.detach(), grad.detach()

                native_image, native_grad = direct(old_backend)
                fixed_image, fixed_grad = direct(fixed_ext)
                ids, bins, bounds = W.FULL._geometry_bins(geometry)
                cn, _ = W._cica_extension().cica_forward(
                    torch.tensor(bounds, dtype=torch.int32, device=model.device),
                    torch.tensor([geometry["width"], geometry["height"], 1], dtype=torch.int32, device=model.device),
                    ids.contiguous(), bins.contiguous(), geometry["xys"].float().contiguous(),
                    geometry["conics"].float().contiguous(), color0.float().contiguous(), opacity.float().contiguous(),
                    medium_attn.float().contiguous(), geometry["depths"].reshape(-1).float().contiguous(),
                    cotangent.reshape(-1, 3).float().contiguous(), 16, 1)
                reference_grad = cn / W.SH_C0
                records.append(dict(scene=scene, method=method, view_id=view, ordinal=ordinal,
                    forward_max_abs=float((fixed_image-native_image).abs().max()),
                    forward_relative_l2=float((fixed_image.double()-native_image.double()).norm() /
                                              native_image.double().norm().clamp_min(1e-30)),
                    native_grad_norm=float(native_grad.norm()), fixed_grad_norm=float(fixed_grad.norm()),
                    reference_grad_norm=float(reference_grad.norm()),
                    native_reference_relative_l2=float((native_grad.double()-reference_grad.double()).norm() /
                                                       reference_grad.double().norm().clamp_min(1e-30)),
                    fixed_reference_relative_l2=float((fixed_grad.double()-reference_grad.double()).norm() /
                                                      reference_grad.double().norm().clamp_min(1e-30)),
                    vjp_relative_l2=float((fixed_grad.double()-native_grad.double()).norm() /
                                          native_grad.double().norm().clamp_min(1e-30)),
                    vjp_cosine=float((fixed_grad.double()*native_grad.double()).sum() /
                                     (fixed_grad.double().norm()*native_grad.double().norm()).clamp_min(1e-30))))
                del outputs, geometry, color0, opacity, medium_rgb, medium_bs, medium_attn, gt, cotangent, cn, reference_grad
        finally:
            W._release(branch)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene", choices=("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama"))
    p.add_argument("--gpu", choices=("6", "7", "8", "9"))
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "water_splatting" / "cuda" / "csrc" / "backward.cu"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    import water_splatting.cuda._backend as backend
    native_ext = backend._C
    ext, build = build_fixed_extension()
    route_extension(ext)
    rows = synthetic()
    torch.cuda.synchronize()
    source_hash_after = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = dict(source_backward_sha256=before, source_unchanged=before == source_hash_after,
                   build_directory=str(build), extension_name="water_splatting_cuda_fsb_fix_20260915",
                   synthetic=rows, synthetic_all_pass=all(r["abs_error"] < 2e-6 for r in rows),
                   no_optimizer_steps=True, production_csrc_so_untouched=True)
    if args.scene:
        assert args.gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
        payload["real_scene"] = real_scene(args.scene, args.gpu, native_ext, ext)
        payload["real_scene_all_pass"] = all(r["forward_max_abs"] < 2e-6 and r["fixed_reference_relative_l2"] < .02
                                              for r in payload["real_scene"])
    (out / "synthetic.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
