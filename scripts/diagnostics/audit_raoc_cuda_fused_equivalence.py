#!/usr/bin/env python3
"""Forward/backward equivalence audit for the standalone RAOC CUDA backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC
from water_splatting.raoc import fused_modal_control


ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "equivalence"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf8")


def _default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else float(value.detach().cpu().item())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _max_diff(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item()) if left.numel() else 0.0


def _geometry(holder: Any, camera: Any) -> Mapping[str, Tensor]:
    model = holder.pipeline.model
    raw_full, _raw_base, height, width = RAOC._raw_pair(model, camera)
    geom = RAOC._geometry(model, camera, height, width)
    from water_splatting.utils import bin_and_sort_gaussians, compute_cumulative_intersects
    num_intersects, cumulative = compute_cumulative_intersects(geom["num_tiles_hit"])
    tiles = ((width + 15) // 16, (height + 15) // 16, 1)
    if num_intersects:
        _a, _b, _c, ids, bins = bin_and_sort_gaussians(
            geom["xys"].shape[0], num_intersects, geom["xys"], geom["depths"], geom["radii"], cumulative, tiles, 16
        )
    else:
        ids = torch.empty(0, device=model.device, dtype=torch.int32)
        bins = torch.zeros((tiles[0] * tiles[1], 2), device=model.device, dtype=torch.int32)
    return {**geom, "raw_full": raw_full.reshape(-1, 9), "height": height, "width": width, "ids": ids, "bins": bins, "num_intersects": num_intersects}


def audit(scene: str, gpu: str, step: int, output: Path) -> Dict[str, Any]:
    holder = RAOC._setup_branch(REPO_ROOT, RAOC.SCENES[scene], "C1")
    try:
        ckpt_path = ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt"
        ckpt = RAOC._load_checkpoint(holder, ckpt_path)
        model = holder.pipeline.model
        _idx, _view, camera, _batch = RAOC._train_records(holder.pipeline)[0]
        geom = _geometry(holder, camera)
        state = ckpt["raoc_state"]
        scale = state["standardization_scale"].to(model.device, dtype=torch.float32).clamp_min(1e-6)
        basis = state["basis"].to(model.device, dtype=torch.float32)
        delta = (geom["raw_full"] - RAOC._raw_pair(model, camera)[1].reshape(-1, 9)) / scale
        raw_control = geom["raw_full"].float()
        medium_rgb = torch.sigmoid(raw_control[:, :3])
        medium_bs = torch.nn.functional.softplus(raw_control[:, 3:6] + float(model.medium_density_bias))
        medium_attn = torch.nn.functional.softplus(raw_control[:, 6:9] + float(model.medium_density_bias))
        d_rgb = medium_rgb * (1.0 - medium_rgb)
        d_bs = torch.sigmoid(raw_control[:, 3:6] + float(model.medium_density_bias))
        d_attn = torch.sigmoid(raw_control[:, 6:9] + float(model.medium_density_bias))
        common = dict(delta_std=delta, basis=basis, global_gate=state["global_gate"], local_scale=state["local_scale"], active=state["active"], raw_medium=geom["raw_full"], raw_directions=basis.T * scale.reshape(1, 9), medium_rgb=medium_rgb, medium_bs=medium_bs, medium_attn=medium_attn, d_rgb=d_rgb, d_bs=d_bs, d_attn=d_attn, xys=geom["xys"], depths=geom["depths"], radii=geom["radii"], conics=geom["conics"], colors=geom["colors"], opacities=geom["opacities"], gaussian_ids_sorted=geom["ids"], tile_bins=geom["bins"], height=geom["height"], width=geom["width"], block_width=16, num_intersects=geom["num_intersects"], density_bias=float(model.medium_density_bias))
        reference = RAOC._raoc_controls(model, camera, geom["raw_full"], RAOC._raw_pair(model, camera)[1].reshape_as(geom["raw_full"]), geom["height"], geom["width"], torch.arange(geom["height"] * geom["width"], device=model.device), state_override=state)
        fused = fused_modal_control(**common)
        rows = []
        for key, index in (("sensitivity", 4), ("evidence", 1), ("local_gate", 2), ("keep_gate", 3)):
            rows.append({"quantity": key, "max_abs_diff": _max_diff(fused[index], reference[key]), "mean_abs_diff": float((fused[index] - reference[key]).abs().mean().item())})
        rows.append({"quantity": "delta_raoc_std", "max_abs_diff": _max_diff(fused[0], RAOC.apply_modal_keep_gate(common["delta_std"], basis, reference["keep_gate"])), "mean_abs_diff": float((fused[0] - RAOC.apply_modal_keep_gate(common["delta_std"], basis, reference["keep_gate"])).abs().mean().item())})
        probe = common["delta_std"].detach().clone().requires_grad_(True)
        common_probe = {**common, "delta_std": probe}
        fused_probe = fused_modal_control(**common_probe)[0]
        cot = torch.randn_like(fused_probe)
        fused_grad = torch.autograd.grad((fused_probe * cot).sum(), probe)[0]
        keep = reference["keep_gate"]
        expected_grad = RAOC.apply_modal_keep_gate(cot, basis, keep)
        result = {"scene": scene, "gpu": gpu, "step": step, "forward_rows": rows, "backward_max_abs_diff": _max_diff(fused_grad, expected_grad), "backward_mean_abs_diff": float((fused_grad - expected_grad).abs().mean().item()), "forward_pass": all(row["max_abs_diff"] <= 1e-5 for row in rows), "backward_pass": _max_diff(fused_grad, expected_grad) <= 1e-6, "second_order_graph": str(fused_probe.grad_fn)}
        _write(output, result)
        return result
    finally:
        RAOC._release(holder)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="IUI3-RedSea", choices=sorted(RAOC.SCENES))
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.scene, args.gpu, args.step, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
