#!/usr/bin/env python3
"""Profile frozen RAOC reference/fused execution on archived checkpoints.

This is an engineering benchmark only.  It never trains, changes a
checkpoint, or creates a formal causal experiment.  Each invocation exposes
exactly one allowed physical GPU and uses CUDA events for stage timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC
from water_splatting.raoc import (
    apply_modal_keep_gate,
    local_keep_gates,
    modal_coefficients,
    ray_keep_gates,
)
from water_splatting.rendering.medium_jacobian import analytic_medium_jacobian_actions
from water_splatting.utils import bin_and_sort_gaussians, compute_cumulative_intersects


OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828"
ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else float(value.detach().cpu().item())
    return str(value)


def _stats(values: Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    } if flat.numel() else {key: float("nan") for key in ("mean", "p50", "p90", "p95", "p99", "max")}


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _geometry(model: Any, camera: Any, height: int, width: int) -> Dict[str, Tensor]:
    return RAOC._geometry(model, camera, height, width)


def _prepare_geometry(model: Any, camera: Any, height: int, width: int) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    geom = _geometry(model, camera, height, width)
    num_intersects, cumulative = compute_cumulative_intersects(geom["num_tiles_hit"].detach())
    if num_intersects:
        tiles_x = (width + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        tiles_y = (height + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        _iu, _gu, _is, ids, bins = bin_and_sort_gaussians(
            geom["xys"].shape[0], num_intersects, geom["xys"].detach(), geom["depths"].detach(), geom["radii"].detach(),
            cumulative, (tiles_x, tiles_y, 1), model.underwater_rasterizer.block_width,
        )
    else:
        tiles_x = (width + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        tiles_y = (height + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        ids = torch.empty(0, device=model.device, dtype=torch.int32)
        bins = torch.zeros((tiles_x * tiles_y, 2), device=model.device, dtype=torch.int32)
    geom["ids"] = ids
    geom["bins"] = bins
    return geom, torch.tensor(num_intersects, device=model.device), cumulative


def _reference_control(model: Any, raw_full: Tensor, raw_base: Tensor, geom: Mapping[str, Tensor], state: Mapping[str, Tensor], height: int, width: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    scale = state["standardization_scale"].to(model.device, dtype=torch.float32).clamp_min(1e-6)
    basis = state["basis"].to(model.device, dtype=torch.float32)
    directions = basis.T * scale.reshape(1, 9)
    actions = analytic_medium_jacobian_actions(
        xys=geom["xys"], depths=geom["depths"], radii=geom["radii"], conics=geom["conics"], colors=geom["colors"],
        opacities=geom["opacities"], num_tiles_hit=geom["num_tiles_hit"], height=height, width=width,
        block_width=model.underwater_rasterizer.block_width, raw_medium=raw_full.reshape(-1, 9), raw_directions=directions,
        density_bias=float(model.medium_density_bias),
    )
    delta_std = (raw_full.reshape(-1, 9) - raw_base.reshape(-1, 9)) / scale.reshape(1, 9)
    coeff = modal_coefficients(delta_std.detach(), basis)
    sensitivity = torch.linalg.norm(actions, dim=-1)
    evidence = coeff.abs() * sensitivity
    q = state["local_scale"].to(model.device, dtype=torch.float32)
    local = local_keep_gates(evidence, q, state["active"].to(model.device))
    keep = ray_keep_gates(state["global_gate"].to(model.device), local)
    delta_out = apply_modal_keep_gate(delta_std, basis, keep)
    delta_out = torch.where((keep == 1).all(dim=1, keepdim=True), delta_std, delta_out)
    return delta_out, evidence, local, keep, sensitivity


def _profile_one(model: Any, camera: Any, batch: Mapping[str, Any], state: Mapping[str, Tensor], backend: str, warmup: int, repeats: int) -> Dict[str, Any]:
    if backend == "ocmc":
        model.config.camera_medium_raoc_backend = "reference"
        model.config.camera_medium_observability_enabled = True
        model.config.camera_medium_ray_adaptive_observability_enabled = False
        projector = state["basis"].float() @ torch.diag(state["global_gate"].float()) @ state["basis"].float().T
        model.set_camera_medium_observability_projector(projector, state["spectrum"], state["standardization_scale"])
    else:
        model.config.camera_medium_observability_enabled = False
        model.config.camera_medium_ray_adaptive_observability_enabled = True
        model.config.camera_medium_raoc_backend = backend
    model.eval()
    with torch.no_grad():
        raw_full, raw_base, height, width = RAOC._raw_pair(model, camera)
        geom, intersects, _ = _prepare_geometry(model, camera, height, width)
        raw_full = raw_full.reshape(-1, 9)
        raw_base = raw_base.reshape(-1, 9)
        scale = state["standardization_scale"].to(model.device, dtype=torch.float32).clamp_min(1e-6)
        delta = (raw_full - raw_base) / scale.reshape(1, 9)
        basis = state["basis"].to(model.device, dtype=torch.float32)
        dirs = basis.T * scale.reshape(1, 9)
        medium_rgb = torch.sigmoid(raw_full[:, :3])
        medium_bs = torch.nn.functional.softplus(raw_full[:, 3:6] + float(model.medium_density_bias))
        medium_attn = torch.nn.functional.softplus(raw_full[:, 6:9] + float(model.medium_density_bias))
        d_rgb = medium_rgb * (1.0 - medium_rgb)
        d_bs = torch.sigmoid(raw_full[:, 3:6] + float(model.medium_density_bias))
        d_attn = torch.sigmoid(raw_full[:, 6:9] + float(model.medium_density_bias))
        args = dict(delta_std=delta, basis=basis, global_gate=state["global_gate"], local_scale=state["local_scale"], active=state["active"], raw_medium=raw_full, raw_directions=dirs, medium_rgb=medium_rgb, medium_bs=medium_bs, medium_attn=medium_attn, d_rgb=d_rgb, d_bs=d_bs, d_attn=d_attn, xys=geom["xys"], depths=geom["depths"], radii=geom["radii"], conics=geom["conics"], colors=geom["colors"], opacities=geom["opacities"], gaussian_ids_sorted=geom["ids"], tile_bins=geom["bins"], height=height, width=width, block_width=model.underwater_rasterizer.block_width, num_intersects=int(intersects.item()), density_bias=float(model.medium_density_bias))
        for _ in range(warmup):
            if backend == "ocmc":
                continue
            _reference_control(model, raw_full, raw_base, geom, state, height, width) if backend == "reference" else __import__("water_splatting.raoc", fromlist=["fused_modal_control"]).fused_modal_control(**args)
        torch.cuda.synchronize()
        stage_rows: List[Dict[str, Any]] = []
        for _ in range(repeats):
            if backend == "ocmc":
                break
            if backend == "reference":
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record(); _reference_control(model, raw_full, raw_base, geom, state, height, width); end.record()
                stage_rows.append({"stage": "raoc_control", "ms": _event_ms(start, end)})
            else:
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record(); __import__("water_splatting.raoc", fromlist=["fused_modal_control"]).fused_modal_control(**args); end.record()
                stage_rows.append({"stage": "raoc_control", "ms": _event_ms(start, end)})
    # Measure a complete differentiable model step separately.  The RAOC
    # control signal remains detached, while the gated residual is attached to
    # the medium MLP graph exactly as it is during training.
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = model.get_outputs(camera)
    end.record(); render_ms = _event_ms(start, end)
    allocated_after_render = torch.cuda.memory_allocated()
    losses = model.get_loss_dict(outputs, batch, {})
    loss = sum(losses.values())
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); loss.backward(); end.record(); backward_ms = _event_ms(start, end)
    allocated_after_backward = torch.cuda.memory_allocated()
    control_ms = float(torch.tensor([r["ms"] for r in stage_rows]).median().item()) if stage_rows else float("nan")
    row = {
        "backend": backend, "height": height, "width": width, "gaussian_count": int(model.means.shape[0]),
        "num_intersects": int(intersects.item()), "raoc_control_ms_median": control_ms,
        "renderer_forward_ms": render_ms, "backward_ms": backward_ms,
        "full_train_like_step_ms": render_ms + backward_ms,
        "loss": float(loss.detach().cpu().item()),
        "loss_terms": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
        "allocated_before_MB": allocated_before / 2**20, "allocated_after_render_MB": allocated_after_render / 2**20,
        "allocated_after_backward_MB": allocated_after_backward / 2**20,
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
        "stage_samples": stage_rows,
    }
    del outputs, losses, loss, raw_full, raw_base, geom, args
    gc.collect(); torch.cuda.empty_cache()
    return row


def profile_scene(scene: str, gpu: str, steps: Sequence[int], repeats: int, warmup: int, output_dir: Path, backends: Sequence[str]) -> Dict[str, Any]:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {gpu} is outside allowed set {sorted(ALLOWED_GPUS)}")
    scene_cfg = RAOC.SCENES[scene]
    rows: List[Dict[str, Any]] = []
    for step in steps:
        for branch in ("C1",):
            state_path = ARCHIVE_ROOT / scene / "checkpoints" / branch / f"step-{step:09d}.ckpt"
            if not state_path.is_file():
                continue
            holder = RAOC._setup_branch(REPO_ROOT, scene_cfg, branch)
            try:
                ckpt = RAOC._load_checkpoint(holder, state_path)
                state = ckpt["raoc_state"]
                _index, _view_id, camera, batch = RAOC._train_records(holder.pipeline)[0]
                batch = {key: value.to(holder.pipeline.model.device) if isinstance(value, Tensor) else value for key, value in batch.items()}
                for backend in backends:
                    rows.append({"scene": scene, "step": step, **_profile_one(holder.pipeline.model, camera, batch, state, backend, warmup, repeats)})
            finally:
                RAOC._release(holder)
    payload = {"scene": scene, "gpu": gpu, "rows": rows, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    _write(output_dir / f"profile_{scene}.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(RAOC.SCENES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--steps", default="3000,8000,13000")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backends", default="reference,cuda_fused")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "runtime_profiles")
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.steps.split(",") if value)
    backends = tuple(value.strip() for value in args.backends.split(",") if value.strip())
    unknown = set(backends) - {"reference", "cuda_fused", "ocmc"}
    if unknown:
        raise ValueError(f"unknown backend(s): {sorted(unknown)}")
    payload = profile_scene(args.scene, args.gpu, steps, args.repeats, args.warmup, args.output_dir, backends)
    print(json.dumps({"scene": args.scene, "rows": len(payload["rows"]), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
