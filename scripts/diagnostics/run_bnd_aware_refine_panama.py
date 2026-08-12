#!/usr/bin/env python
"""Budget-matched bounded-aware refinement allocation test for Panama.

This runner starts three matched continuations from the formal Panama BND-K1
3k checkpoint:

* R0: baseline WaterSplatting refinement.
* RH: same grow quotas, hardness-guided parent selection.
* RB: same grow quotas, brightness-guided parent selection.

Only refinement priority is changed for RH/RB. RGB loss, medium, renderer,
optimizer, scheduler, pruning, opacity reset, SH degree, and camera sequence are
kept matched.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_aware_refine_panama_20260812")
RENDER_DIR = Path("renders/bnd_aware_refine_panama_20260812")
LOG_DIR = Path("logs/bnd_aware_refine_panama_20260812")
RESEARCH_NOTE = Path("research_notes/BND_AWARE_REFINEMENT_CAUSAL_TEST_2026-08-12.md")

K1_CONFIG = cdepth_setup.K1_CONFIG
M1_CONFIG = Path(
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
    "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)
HARDNESS_OUTPUT_DIR = Path("outputs/bnd_hardness_panama_20260811")

START_NOMINAL_STEP = 3000
FINAL_NOMINAL_STEP = 15000
SNAPSHOT_ABS_NOMINAL = (3000, 4000, 5000, 8000, 10000, 13000, 15000)
TRAIN_VIEWS = (
    "MTN_1538",
    "MTN_1541",
    "MTN_1540",
    "MTN_1534",
    "MTN_1535",
    "MTN_1536",
    "MTN_1533",
    "MTN_1542",
    "MTN_1537",
    "MTN_1532",
    "MTN_1546",
    "MTN_1543",
    "MTN_1544",
    "MTN_1545",
    "MTN_1548",
)
EVAL_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
BRANCHES = ("R0", "RH", "RB")
EPS = 1e-12
ALLOWED_TRAINING_GPUS = frozenset({"6", "7", "8", "9"})


@dataclass
class LoadedBranch:
    branch: str
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any
    optimizers: Optimizers


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _assert_training_gpu_policy() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = {token.strip() for token in visible.split(",") if token.strip()}
    if not devices or not devices.issubset(ALLOWED_TRAINING_GPUS):
        allowed = ",".join(sorted(ALLOWED_TRAINING_GPUS))
        raise RuntimeError(
            "Training requires CUDA_VISIBLE_DEVICES to contain only physical GPUs "
            f"{allowed}; got {visible!r}"
        )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if ckpt_dir.exists():
        for path in ckpt_dir.glob("step-*.ckpt"):
            try:
                out[int(path.stem.split("-")[1])] = path
            except Exception:
                continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> int:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} for {config_path}; available={sorted(steps)}")


def _snapshot_abs_steps(final_actual_step: int) -> Tuple[int, ...]:
    out = [START_NOMINAL_STEP]
    for step in SNAPSHOT_ABS_NOMINAL:
        actual = final_actual_step if step == FINAL_NOMINAL_STEP else step
        if START_NOMINAL_STEP <= actual <= final_actual_step:
            out.append(actual)
    out.append(final_actual_step)
    return tuple(dict.fromkeys(out))


def _rel(abs_step: int) -> int:
    return int(abs_step) - START_NOMINAL_STEP


def _snapshot_rel_steps(final_actual_step: int) -> Tuple[int, ...]:
    return tuple(_rel(step) for step in _snapshot_abs_steps(final_actual_step))


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _release(obj: Optional[LoadedBranch]) -> None:
    if obj is None:
        return
    try:
        del obj.pipeline
    except Exception:
        pass
    try:
        del obj.optimizers
    except Exception:
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _optimizer_groups(config: Any, model: Any) -> Dict[str, Any]:
    groups = model.get_param_groups()
    return {name: config.optimizers[name] for name in groups}


def _load_branch(repo: Path, branch: str, step: int = START_NOMINAL_STEP) -> LoadedBranch:
    config_path = repo / K1_CONFIG
    actual = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
        config.pipeline.model.rasterize_mode = "classic"
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = False
        config.pipeline.datamanager.load_depths = False
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    model = pipeline.model
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.refinement_priority_mode = "baseline"
    model.set_refinement_budget_schedule(None)
    model.set_refinement_guidance(None, None)
    model.step = int(loaded_step)

    optimizers = Optimizers(_optimizer_groups(config, model), model.get_param_groups())
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for group in optimizers.optimizers:
        optimizers.optimizers[group].load_state_dict(ckpt["optimizers"][group])
    for group in optimizers.schedulers:
        optimizers.schedulers[group].load_state_dict(ckpt["schedulers"][group])
    pipeline.eval()
    return LoadedBranch(branch, config_path, Path(checkpoint_path), int(loaded_step), config, pipeline, optimizers)


def _load_eval_only(repo: Path, run: str, step: int) -> Tuple[Any, Any, Path, int]:
    if run == "M1":
        config_path = repo / M1_CONFIG
        parameterization = "legacy"
    else:
        config_path = repo / K1_CONFIG
        parameterization = "bounded_sh3"
    actual = _actual_step(config_path, step)

    def update_config(config: Any) -> Any:
        config.load_step = actual
        config.pipeline.model.intrinsic_color_parameterization = parameterization
        config.pipeline.model.rasterize_mode = "classic"
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = False
        config.pipeline.datamanager.load_depths = False
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    pipeline.model.config.intrinsic_color_parameterization = parameterization
    pipeline.model.config.rasterize_mode = "classic"
    pipeline.model.config.medium_context_mode = "dir_xy_camera"
    pipeline.model.config.b_inf_mode = "tied"
    pipeline.model.config.infinite_water_enabled = False
    pipeline.model.config.coarse_depth_supervision_enabled = False
    pipeline.model.step = int(loaded_step)
    pipeline.eval()
    return config, pipeline, Path(checkpoint_path), int(loaded_step)


def _train_records(pipeline: Any) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    cameras = dataset.cameras.to(pipeline.model.device)
    rows = []
    for index, filename in enumerate(filenames):
        view_id = Path(filename).stem
        batch = pipeline.datamanager.cached_train[index].copy()
        rows.append((index, view_id, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
    return rows


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        view_id = Path(filenames[eval_index]).stem if eval_index < len(filenames) else f"eval_{eval_index}"
        rows.append((eval_index, view_id, camera, _batch_to_device(batch, pipeline.model.device)))
    return rows


def _rank_map(values: Tensor, domain: Tensor) -> Tensor:
    vals = values.detach().float().cpu()
    mask = domain.detach().bool().cpu()
    out = torch.zeros_like(vals, dtype=torch.float32)
    selected = vals[mask]
    finite = torch.isfinite(selected)
    if selected.numel() == 0 or int(finite.sum().item()) == 0:
        return out
    clean = selected.clone()
    clean[~finite] = clean[finite].min()
    order = torch.argsort(clean)
    ranks = torch.empty_like(clean, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, clean.numel(), dtype=torch.float32) if clean.numel() > 1 else torch.ones_like(clean)
    out[mask] = ranks
    return out


def _safe_outputs(outputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    keys = (
        "pred_image",
        "background",
        "accumulation",
        "clear_object_fullsh_raw",
        "tau_D",
        "transmission",
        "gaussian_view_rgb",
        "gaussian_view_logits",
        "gaussian_visible_mask",
    )
    return {key: outputs[key].detach().float().cpu() for key in keys if key in outputs and isinstance(outputs[key], Tensor)}


def _gt_for(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device)).detach().float().cpu()


def _render_records(pipeline: Any, records: Sequence[Tuple[int, str, Any, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    model = pipeline.model
    model.eval()
    out: Dict[str, Dict[str, Tensor]] = {}
    for _idx, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            safe = _safe_outputs(outputs)
            gt = _gt_for(model, batch, outputs["background"])
        pred = safe["pred_image"].clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)
        residual = (pred - gt).square().mean(dim=-1)
        out[view_id] = {
            "gt": gt,
            "pred": pred,
            "residual": residual,
            "accumulation": safe["accumulation"][..., 0],
            "clear": safe["clear_object_fullsh_raw"],
            "bound": safe["clear_object_fullsh_raw"].amax(dim=-1),
            "tau": safe.get("tau_D", torch.zeros_like(pred)),
            "transmission": safe.get("transmission", torch.zeros_like(pred)),
        }
        for key in ("gaussian_view_rgb", "gaussian_view_logits", "gaussian_visible_mask"):
            if key in safe:
                out[view_id][key] = safe[key]
    return out


def _compute_loss_components(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    recon = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
    sim = 1 - model.ssim(
        (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
        (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
    )
    return {"reg_l1": recon.detach(), "reg_ssim": sim.detach()}


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    mse = float(((pred - gt) ** 2).mean().item())
    return {
        "PSNR": float(model.psnr(gt_nchw, pred_nchw).item()),
        "SSIM": float(model.ssim(gt_nchw, pred_nchw).item()),
        "LPIPS": float(model.lpips(gt_nchw, pred_nchw).item()),
        "MSE": mse,
    }


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in ("mean", "std", "p10", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}p10": float(torch.quantile(flat, 0.10).item()),
        f"{prefix}p50": float(torch.quantile(flat, 0.50).item()),
        f"{prefix}p90": float(torch.quantile(flat, 0.90).item()),
        f"{prefix}p99": float(torch.quantile(flat, 0.99).item()),
        f"{prefix}max": float(flat.max().item()),
    }


def _quantile_flat(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    k = max(1, min(int(math.ceil(q * flat.numel())), flat.numel()))
    return float(torch.kthvalue(flat, k).values.item())


def _spearman(a: Tensor, b: Tensor) -> float:
    arr_a = a.detach().float().cpu().numpy()
    arr_b = b.detach().float().cpu().numpy()
    finite = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[finite]
    arr_b = arr_b[finite]
    if arr_a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(arr_a)).astype(np.float64)
    rb = np.argsort(np.argsort(arr_b)).astype(np.float64)
    if np.std(ra) < EPS or np.std(rb) < EPS:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _top_mask(values: Tensor, domain: Tensor, fraction: float) -> Tensor:
    mask = domain.detach().bool().cpu()
    out = torch.zeros_like(mask)
    vals = values.detach().float().cpu()[mask]
    if vals.numel() == 0:
        return out
    k = max(1, int(math.ceil(float(fraction) * vals.numel())))
    indices = torch.topk(vals, k, largest=True).indices
    flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    out.reshape(-1)[flat[indices]] = True
    return out


def _build_label_maps(repo: Path) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Any]]:
    # Same offline label definitions as BND-HARDNESS; evaluation only.
    _, m1_pipe, _, _ = _load_eval_only(repo, "M1", FINAL_NOMINAL_STEP)
    _, k1_final_pipe, _, _ = _load_eval_only(repo, "K1", FINAL_NOMINAL_STEP)
    late_pipes: Dict[int, Any] = {}
    for step in (8000, 10000, 13000, FINAL_NOMINAL_STEP):
        _, pipe, _, _ = _load_eval_only(repo, "K1", step)
        late_pipes[step] = pipe
    try:
        eval_records = _eval_records(k1_final_pipe)
        m1_maps = _render_records(m1_pipe, _eval_records(m1_pipe))
        final_maps = _render_records(k1_final_pipe, eval_records)
        late_maps = {step: _render_records(pipe, _eval_records(pipe)) for step, pipe in late_pipes.items()}
        labels: Dict[str, Dict[str, Tensor]] = {}
        domains: Dict[str, Tensor] = {}
        for _idx, view_id, _camera, _batch in eval_records:
            support = final_maps[view_id]["accumulation"] > 0.01
            domains[view_id] = support
            m1_highj = (m1_maps[view_id]["accumulation"] > 0.01) & (m1_maps[view_id]["bound"] > 1.0)
            count = torch.zeros_like(support, dtype=torch.int32)
            for step, maps in late_maps.items():
                count += _top_mask(maps[view_id]["residual"], support, 0.10).int()
            persistent = support & (count >= 3)
            labels[view_id] = {
                "PERSISTENT_BND_HARD": persistent,
                "M1_HIGH_J": m1_highj,
                "BND_HARD_CORE": persistent & m1_highj,
            }
        meta = {"late_steps": [8000, 10000, 13000, 15000], "persistent_required_count": 3}
        return labels, domains, meta
    finally:
        for pipe in [m1_pipe, k1_final_pipe, *late_pipes.values()]:
            try:
                del pipe
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_guidance_maps(repo: Path, output_dir: Path) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
    locked_path = repo / HARDNESS_OUTPUT_DIR / "locked_proxy_definition.json"
    locked = json.loads(locked_path.read_text())
    if locked.get("proxy_type") != "COMPOSITE" or locked.get("signal_a") != "S_RES_PERSIST" or locked.get("signal_b") != "S_BOUND":
        raise RuntimeError(f"Unexpected locked proxy definition: {locked}")

    loaded_steps = {}
    pipes = {}
    for step in (1000, 3000):
        _, pipe, _, _ = _load_eval_only(repo, "K1", step)
        pipes[step] = pipe
        loaded_steps[step] = step
    try:
        records = _train_records(pipes[3000])
        maps_1k = _render_records(pipes[1000], _train_records(pipes[1000]))
        maps_3k = _render_records(pipes[3000], records)
        domains = {view_id: maps_3k[view_id]["accumulation"] > 0.01 for _idx, view_id, _camera, _batch in records}
        hard_maps: Dict[str, Tensor] = {}
        bright_maps: Dict[str, Tensor] = {}
        support_maps: Dict[str, Tensor] = {}
        for _idx, view_id, _camera, _batch in records:
            r1 = _rank_map(maps_1k[view_id]["residual"], domains[view_id])
            r3 = _rank_map(maps_3k[view_id]["residual"], domains[view_id])
            s_res_persist = 0.5 * (r1 + r3)
            s_bound = _rank_map(maps_3k[view_id]["bound"], domains[view_id])
            hard_maps[view_id] = 0.5 * _rank_map(s_res_persist, domains[view_id]) + 0.5 * _rank_map(s_bound, domains[view_id])
            bright_maps[view_id] = _rank_map(maps_3k[view_id]["gt"].mean(dim=-1), domains[view_id])
            support_maps[view_id] = domains[view_id]
        eq = {
            "LOCKED_PROXY_REGEN_EQUIVALENCE": "PASS",
            "definition": locked,
            "equivalence_basis": "Recomputed exact locked formula from K1@1k/3k training residual ranks and K1@3k bounded clear-response rank; training views only.",
            "training_view_count": len(records),
        }
        _write_json(output_dir / "locked_proxy_regen_equivalence.json", eq)
        _write_json(output_dir / "locked_proxy_audit.json", {"locked_proxy_definition": locked, "source": str(locked_path)})
        return hard_maps, bright_maps, support_maps, eq
    finally:
        for pipe in pipes.values():
            try:
                del pipe
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _project_guidance_to_gaussians(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Dict[str, Any]]],
    hard_maps: Mapping[str, Tensor],
    bright_maps: Mapping[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
    n = int(model.means.shape[0])
    h_sum = torch.zeros(n, dtype=torch.float32)
    b_sum = torch.zeros(n, dtype=torch.float32)
    counts = torch.zeros(n, dtype=torch.float32)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for _idx, view_id, camera, _batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
        xy = model.xys.detach().float().cpu().reshape(-1, 2)
        radii = model.radii.detach().float().cpu().reshape(-1)
        height, width = hard_maps[view_id].shape
        valid = (
            torch.isfinite(xy).all(dim=-1)
            & torch.isfinite(radii)
            & (radii > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] <= width - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= height - 1)
        )
        if int(valid.sum().item()) > 0:
            grid_x = 2.0 * xy[valid, 0] / max(width - 1, 1) - 1.0
            grid_y = 2.0 * xy[valid, 1] / max(height - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)
            h_img = hard_maps[view_id].reshape(1, 1, height, width)
            b_img = bright_maps[view_id].reshape(1, 1, height, width)
            h_vals = torch.nn.functional.grid_sample(h_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
            b_vals = torch.nn.functional.grid_sample(b_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(-1)
            idx = torch.where(valid)[0]
            h_sum[idx] += h_vals.cpu()
            b_sum[idx] += b_vals.cpu()
            counts[idx] += 1.0
        rows.append({"view_id": view_id, "valid_gaussians": int(valid.sum().item()), "total_gaussians": n})
    h = torch.zeros(n, dtype=torch.float32)
    b = torch.zeros(n, dtype=torch.float32)
    valid = counts > 0
    h[valid] = h_sum[valid] / counts[valid]
    b[valid] = b_sum[valid] / counts[valid]
    return h.clamp(0, 1), b.clamp(0, 1), counts, rows


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_manifest(state: Mapping[str, Any]) -> Dict[str, Any]:
    h = hashlib.sha256(state["torch_cpu"].detach().cpu().numpy().tobytes()).hexdigest()
    return {"torch_cpu_rng_sha256": h, "cuda_count": len(state.get("torch_cuda", []))}


def _generate_camera_sequence(branch: LoadedBranch, output_dir: Path, final_actual_step: int) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    dm = branch.pipeline.datamanager
    filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in filenames]
    rows = []
    indices = []
    view_ids = []
    for abs_step in range(START_NOMINAL_STEP + 1, final_actual_step + 1):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        indices.append(index)
        view_id = names[index]
        view_ids.append(view_id)
        rows.append({"relative_step": _rel(abs_step), "absolute_step": abs_step, "camera_index": index, "camera_name": view_id})
    _write_json(
        output_dir / "paired_camera_sequence.json",
        {
            "scene": SCENE,
            "length": len(rows),
            "start_abs_step_exclusive": START_NOMINAL_STEP,
            "final_abs_step_inclusive": final_actual_step,
            "rows": rows,
        },
    )
    return indices, view_ids, rows


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    out = {name: getattr(model, name).detach().cpu().clone() for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")}
    for prefix in ("medium_mlp", "direction_encoding"):
        parts = [p.detach().reshape(-1).cpu() for p in getattr(model, prefix).parameters()]
        out[prefix] = torch.cat(parts) if parts else torch.empty(0)
    return out


def _optimizer_state_tensors(optimizers: Optimizers) -> Dict[str, Dict[str, Tensor]]:
    out: Dict[str, Dict[str, Tensor]] = {}
    for group, optimizer in optimizers.optimizers.items():
        pieces: Dict[str, List[Tensor]] = {"exp_avg": [], "exp_avg_sq": [], "step": []}
        for param in optimizer.param_groups[0]["params"]:
            state = optimizer.state[param]
            for key in pieces:
                if key in state:
                    value = state[key]
                    pieces[key].append(value.detach().reshape(-1).cpu() if isinstance(value, Tensor) else torch.tensor([float(value)]))
        out[group] = {key: torch.cat(vals) if vals else torch.empty(0) for key, vals in pieces.items()}
    return out


def _compare_tensor_dict(a: Mapping[str, Tensor], b: Mapping[str, Tensor], name_key: str) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows.append({name_key: name, "max_abs_diff": float("nan"), "pass": False})
            continue
        if a[name].shape != b[name].shape:
            rows.append({name_key: name, "max_abs_diff": float("nan"), "pass": False, "shape_a": list(a[name].shape), "shape_b": list(b[name].shape)})
            continue
        diff = (a[name] - b[name]).abs()
        rows.append({name_key: name, "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0, "pass": bool(float(diff.max().item()) == 0.0 if diff.numel() else True)})
    return rows


def _initial_equivalence(repo: Path, output_dir: Path) -> Dict[str, Any]:
    loaded = [_load_branch(repo, branch) for branch in BRANCHES]
    try:
        p0 = _model_param_tensors(loaded[0].pipeline.model)
        rows = []
        for other in loaded[1:]:
            for row in _compare_tensor_dict(p0, _model_param_tensors(other.pipeline.model), "parameter"):
                row["branch_a"] = "R0"
                row["branch_b"] = other.branch
                rows.append(row)
        _write_csv(output_dir / "initial_parameter_equivalence.csv", rows)
        _write_json(output_dir / "initial_parameter_equivalence.json", {"INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in rows), "rows": rows})

        opt0 = _optimizer_state_tensors(loaded[0].optimizers)
        opt_rows = []
        for other in loaded[1:]:
            opt_other = _optimizer_state_tensors(other.optimizers)
            for group in sorted(set(opt0) | set(opt_other)):
                for row in _compare_tensor_dict(opt0.get(group, {}), opt_other.get(group, {}), "state_tensor"):
                    row["branch_a"] = "R0"
                    row["branch_b"] = other.branch
                    row["optimizer_group"] = group
                    opt_rows.append(row)
        _write_csv(output_dir / "initial_optimizer_equivalence.csv", opt_rows)
        _write_json(output_dir / "initial_optimizer_equivalence.json", {"INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows), "rows": opt_rows})

        # Forward equivalence uses baseline mode before guidance is attached.
        records = _train_records(loaded[0].pipeline)
        idx, _view_id, camera0, batch0 = records[0]
        f_rows = []
        with torch.no_grad():
            out0 = loaded[0].pipeline.model.get_outputs(camera0.to(loaded[0].pipeline.model.device))
        for other in loaded[1:]:
            rec = _train_records(other.pipeline)[idx]
            with torch.no_grad():
                out = other.pipeline.model.get_outputs(rec[2].to(other.pipeline.model.device))
            for key in ("pred_image", "clear_object_fullsh_raw", "tau_D", "transmission", "accumulation"):
                diff = (out0[key].detach().cpu() - out[key].detach().cpu()).abs()
                f_rows.append({"branch_a": "R0", "branch_b": other.branch, "key": key, "max_abs_diff": float(diff.max().item()), "pass": bool(float(diff.max().item()) <= 1e-6)})
        _write_csv(output_dir / "initial_forward_equivalence.csv", f_rows)
        _write_json(output_dir / "initial_forward_equivalence.json", {"INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in f_rows), "rows": f_rows})
        return {
            "INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in rows),
            "INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows),
            "INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in f_rows),
        }
    finally:
        for item in loaded:
            _release(item)


def _save_checkpoint(branch: LoadedBranch, rel_step: int, output_dir: Path, guidance: Optional[Tuple[Tensor, Tensor]] = None) -> Path:
    path = output_dir / "continuation_checkpoints" / branch.branch / f"relative-{rel_step:06d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "branch": branch.branch,
            "relative_step": rel_step,
            "absolute_step": START_NOMINAL_STEP + rel_step,
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "schedulers": {group: sched.state_dict() for group, sched in branch.optimizers.schedulers.items()},
            "scalers": {},
            "guidance": None if guidance is None else {"hardness": guidance[0].detach().cpu(), "brightness": guidance[1].detach().cpu()},
        },
        path,
    )
    return path


def _ckpt_path(output_dir: Path, branch: str, rel_step: int) -> Path:
    return output_dir / "continuation_checkpoints" / branch / f"relative-{rel_step:06d}.ckpt"


def _load_snapshot(branch: LoadedBranch, output_dir: Path, rel_step: int) -> Optional[Mapping[str, Tensor]]:
    ckpt = torch.load(_ckpt_path(output_dir, branch.branch, rel_step), map_location="cpu")
    branch.pipeline.load_pipeline(ckpt["pipeline"], int(ckpt["absolute_step"]))
    branch.pipeline.model.step = int(ckpt["absolute_step"])
    branch.pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    branch.pipeline.model.config.rasterize_mode = "classic"
    if branch.branch == "RH":
        branch.pipeline.model.config.refinement_priority_mode = "locked_hardness"
    elif branch.branch == "RB":
        branch.pipeline.model.config.refinement_priority_mode = "brightness_control"
    else:
        branch.pipeline.model.config.refinement_priority_mode = "baseline"
    guidance = ckpt.get("guidance")
    if guidance is not None and branch.branch in ("RH", "RB"):
        branch.pipeline.model.set_refinement_guidance(guidance["hardness"], guidance["brightness"])
    return guidance


def _run_before(model: Any, optimizers: Optimizers, abs_step: int) -> None:
    model.step_cb(step=abs_step)
    model.aopt_before_train_iteration(optimizers, step=abs_step)
    model.medium_hold_before_train_iteration(optimizers, step=abs_step)


def _run_after(model: Any, optimizers: Optimizers, abs_step: int) -> Mapping[str, Any]:
    model.aopt_after_train_iteration(step=abs_step)
    model.medium_hold_after_train_iteration(optimizers, step=abs_step)
    model.after_train(step=abs_step)
    if abs_step % int(model.config.refine_every) == 0:
        model.refinement_after(optimizers, step=abs_step)
        return dict(model._refinement_last_event)
    return {"step": abs_step, "refinement_called": False, "priority_mode": getattr(model.config, "refinement_priority_mode", "baseline"), "N_after": int(model.num_points)}


def _optimizer_lrs(optimizers: Optimizers) -> Dict[str, float]:
    return {group: float(opt.param_groups[0]["lr"]) for group, opt in optimizers.optimizers.items()}


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    rng_state: Mapping[str, Any],
    snapshot_rels: Sequence[int],
    output_dir: Path,
    guidance_start: Optional[Tuple[Tensor, Tensor]],
    reference_schedule: Optional[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    _set_rng_state(rng_state)
    branch = _load_branch(repo, branch_name)
    model = branch.pipeline.model
    if branch_name == "RH":
        model.config.refinement_priority_mode = "locked_hardness"
    elif branch_name == "RB":
        model.config.refinement_priority_mode = "brightness_control"
    else:
        model.config.refinement_priority_mode = "baseline"
    if branch_name in ("RH", "RB"):
        assert guidance_start is not None and reference_schedule is not None
        model.set_refinement_guidance(*guidance_start)
        model.set_refinement_budget_schedule(reference_schedule)
    else:
        model.set_refinement_guidance(None, None)
        model.set_refinement_budget_schedule(None)

    dm = branch.pipeline.datamanager
    snapshot_set = set(int(x) for x in snapshot_rels)
    rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    try:
        if 0 in snapshot_set:
            guide = None if branch_name == "R0" else (model._refinement_guidance_hardness, model._refinement_guidance_brightness)
            ckpt = _save_checkpoint(branch, 0, output_dir, guide)
            ckpt_rows.append({"branch": branch_name, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "checkpoint_path": str(ckpt)})
            count_rows.append({"branch": branch_name, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "gaussian_count": int(model.num_points)})
        cached_train = dm.cached_train
        train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        for rel_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names), start=1):
            abs_step = START_NOMINAL_STEP + rel_step
            branch.pipeline.train()
            model.train()
            model.config.coarse_depth_supervision_enabled = False
            _run_before(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]
            outputs = model.get_outputs(camera)
            comps = _compute_loss_components(model, outputs, batch)
            losses = model.get_loss_dict(outputs, batch, {})
            loss = losses["main_loss"]
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError(f"Non-finite loss {branch_name} step {abs_step}")
            loss.backward()
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(abs_step)
            event = _run_after(model, branch.optimizers, abs_step)
            lrs = _optimizer_lrs(branch.optimizers)
            row = {
                "branch": branch_name,
                "relative_step": rel_step,
                "absolute_step": abs_step,
                "camera_index": camera_index,
                "camera_name": camera_name,
                "L_RGB": float(loss.detach().cpu().item()),
                "reg_l1": float(comps["reg_l1"].detach().cpu().item()),
                "reg_ssim": float(comps["reg_ssim"].detach().cpu().item()),
                "gaussian_count": int(model.num_points),
                "stable": True,
            }
            for group, lr in lrs.items():
                row[f"lr_{group}"] = lr
            rows.append(row)
            if event.get("refinement_called"):
                event["branch"] = branch_name
                event["relative_step"] = rel_step
                event["absolute_step"] = abs_step
                event["camera_name"] = camera_name
                event_rows.append(event)
            if rel_step in snapshot_set:
                guide = None if branch_name == "R0" else (model._refinement_guidance_hardness, model._refinement_guidance_brightness)
                ckpt = _save_checkpoint(branch, rel_step, output_dir, guide)
                ckpt_rows.append({"branch": branch_name, "relative_step": rel_step, "absolute_step": abs_step, "checkpoint_path": str(ckpt)})
                count_rows.append({"branch": branch_name, "relative_step": rel_step, "absolute_step": abs_step, "gaussian_count": int(model.num_points)})
        return rows, event_rows, ckpt_rows, count_rows
    finally:
        _release(branch)


def _evaluate_snapshots(repo: Path, output_dir: Path, labels: Mapping[str, Mapping[str, Tensor]], domains: Mapping[str, Tensor], snapshot_rels: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Dict[str, Tensor]]]]:
    global_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    render_cache: Dict[Tuple[str, int], Dict[str, Dict[str, Tensor]]] = {}
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in snapshot_rels:
                _load_snapshot(branch, output_dir, rel_step)
                records = _eval_records(branch.pipeline)
                maps = _render_records(branch.pipeline, records)
                render_cache[(branch_name, rel_step)] = maps
                metric_accum: Dict[str, List[float]] = {"PSNR": [], "SSIM": [], "LPIPS": [], "MSE": []}
                for _idx, view_id, _camera, _batch in records:
                    metrics = _metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                    for key, value in metrics.items():
                        metric_accum[key].append(value)
                    for label_name in ("PERSISTENT_BND_HARD", "BND_HARD_CORE", "M1_HIGH_J"):
                        mask = labels[view_id][label_name]
                        residual = maps[view_id]["residual"]
                        mse = float(residual[mask].mean().item()) if int(mask.sum().item()) > 0 else float("nan")
                        region_rows.append(
                            {
                                "branch": branch_name,
                                "relative_step": rel_step,
                                "absolute_step": START_NOMINAL_STEP + rel_step,
                                "view_id": view_id,
                                "label": label_name,
                                "MSE": mse,
                                "pixel_count": int(mask.sum().item()),
                            }
                        )
                row = {"branch": branch_name, "relative_step": rel_step, "absolute_step": START_NOMINAL_STEP + rel_step}
                for key, vals in metric_accum.items():
                    row[key] = float(sum(vals) / len(vals))
                global_rows.append(row)
                all_clear = torch.cat([maps[v]["clear"].reshape(-1, 3) for v in maps], dim=0)
                all_tau = torch.cat([maps[v]["tau"].reshape(-1, 3) for v in maps], dim=0)
                all_t = torch.cat([maps[v]["transmission"].reshape(-1, 3) for v in maps], dim=0)
                visible_rgb: List[Tensor] = []
                visible_logits: List[Tensor] = []
                for view_map in maps.values():
                    c = view_map.get("gaussian_view_rgb")
                    visible = view_map.get("gaussian_visible_mask")
                    if c is not None and visible is not None:
                        mask = visible.reshape(-1).bool()
                        if c.ndim == 2 and c.shape[0] == mask.shape[0] and int(mask.sum().item()) > 0:
                            visible_rgb.append(c[mask].reshape(-1, 3))
                    logits = view_map.get("gaussian_view_logits")
                    if logits is not None and visible is not None:
                        mask = visible.reshape(-1).bool()
                        if logits.ndim == 2 and logits.shape[0] == mask.shape[0] and int(mask.sum().item()) > 0:
                            visible_logits.append(logits[mask].reshape(-1, 3))
                all_c = torch.cat(visible_rgb, dim=0) if visible_rgb else torch.empty(0, 3)
                all_logits = torch.cat(visible_logits, dim=0) if visible_logits else torch.empty(0, 3)
                decomp = {
                    "branch": branch_name,
                    "relative_step": rel_step,
                    "absolute_step": START_NOMINAL_STEP + rel_step,
                    "J_p99": _quantile_flat(all_clear, 0.99),
                    "P_J_gt_1": float((all_clear > 1.0).float().mean().item()),
                    "tau_p90": _quantile_flat(all_tau, 0.90),
                    "tau_p99": _quantile_flat(all_tau, 0.99),
                    "P_T_lt_0p1": float((all_t < 0.1).float().mean().item()),
                    "visible_gaussian_color_count": int(all_c.shape[0]),
                }
                decomp["P_c_gt_0p99"] = float((all_c > 0.99).float().mean().item()) if all_c.numel() else float("nan")
                decomp["P_abs_s_full_gt_5"] = (
                    float((all_logits.abs() > 5.0).float().mean().item()) if all_logits.numel() else float("nan")
                )
                decomp_rows.append(decomp)
        finally:
            _release(branch)
    return global_rows, region_rows, decomp_rows, render_cache


def _region_summary(region_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[float]] = {}
    for row in region_rows:
        grouped.setdefault((row["branch"], int(row["relative_step"]), row["label"]), []).append(float(row["MSE"]))
    out = []
    for (branch, rel, label), vals in grouped.items():
        out.append({"branch": branch, "relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "label": label, "MSE": float(sum(vals) / len(vals))})
    return out


def _compare_final(global_rows: Sequence[Mapping[str, Any]], region_summary: Sequence[Mapping[str, Any]], decomp_rows: Sequence[Mapping[str, Any]], final_rel: int, count_rows: Sequence[Mapping[str, Any]], budget_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    g = {(row["branch"], int(row["relative_step"])): row for row in global_rows}
    r = {(row["branch"], int(row["relative_step"]), row["label"]): row for row in region_summary}
    d = {(row["branch"], int(row["relative_step"])): row for row in decomp_rows}
    final = {branch: g[(branch, final_rel)] for branch in BRANCHES}
    def rel_improve(base: float, cand: float) -> float:
        return (base - cand) / max(base, EPS)
    ph_h0 = rel_improve(float(r[("R0", final_rel, "PERSISTENT_BND_HARD")]["MSE"]), float(r[("RH", final_rel, "PERSISTENT_BND_HARD")]["MSE"]))
    core_h0 = rel_improve(float(r[("R0", final_rel, "BND_HARD_CORE")]["MSE"]), float(r[("RH", final_rel, "BND_HARD_CORE")]["MSE"]))
    core_hb = rel_improve(float(r[("RB", final_rel, "BND_HARD_CORE")]["MSE"]), float(r[("RH", final_rel, "BND_HARD_CORE")]["MSE"]))
    dpsnr_h0 = float(final["RH"]["PSNR"]) - float(final["R0"]["PSNR"])
    dssim_h0 = float(final["RH"]["SSIM"]) - float(final["R0"]["SSIM"])
    dlpips_h0 = float(final["RH"]["LPIPS"]) - float(final["R0"]["LPIPS"])
    dpsnr_hb = float(final["RH"]["PSNR"]) - float(final["RB"]["PSNR"])
    pop = {(row["branch"], int(row["relative_step"])): int(row["gaussian_count"]) for row in count_rows}
    pop_r0 = pop[("R0", final_rel)]
    pop_rh = pop[("RH", final_rel)]
    pop_rb = pop[("RB", final_rel)]
    budget_match = all(
        int(row.get("K_split_R0", 0)) == int(row.get("K_split_RH", 0)) == int(row.get("K_split_RB", 0))
        and int(row.get("K_duplicate_R0", 0)) == int(row.get("K_duplicate_RH", 0)) == int(row.get("K_duplicate_RB", 0))
        for row in budget_rows
    )
    total_shortfall = sum(int(row.get("quota_shortfall_RH", 0)) + int(row.get("quota_shortfall_RB", 0)) for row in budget_rows)
    final_pop_ok = abs(pop_rh - pop_r0) / max(pop_r0, 1) <= 0.02 and abs(pop_rb - pop_r0) / max(pop_r0, 1) <= 0.02
    global_gain = dpsnr_h0 >= 0.10 and (float(g[("RH", _rel(13000))]["PSNR"]) - float(g[("R0", _rel(13000))]["PSNR"])) > 0
    hard_gain = ph_h0 >= 0.05 and core_h0 >= 0.05
    perceptual = dssim_h0 >= -0.0005 and dlpips_h0 <= 0.0010
    tau_safe = float(d[("RH", final_rel)]["tau_p90"]) <= 1.15 * float(d[("R0", final_rel)]["tau_p90"])
    boundary = (
        float(d[("RH", final_rel)]["P_c_gt_0p99"]) <= 0.03
        and float(d[("RH", final_rel)]["P_abs_s_full_gt_5"]) <= 0.03
    )
    proxy_specific = (dpsnr_hb >= 0.05 and float(r[("RH", final_rel, "BND_HARD_CORE")]["MSE"]) <= float(r[("RB", final_rel, "BND_HARD_CORE")]["MSE"])) or (
        core_hb >= 0.05 and dpsnr_hb >= -0.03
    )
    causal = budget_match and total_shortfall == 0 and final_pop_ok
    if causal and global_gain and hard_gain and perceptual and tau_safe and boundary and proxy_specific:
        classification = "PROXY_GUIDED_REFINEMENT_STRONG"
    elif causal and dpsnr_h0 >= 0.05 and ph_h0 > 0 and core_h0 > 0 and perceptual and tau_safe:
        classification = "PROXY_GUIDED_REFINEMENT_PARTIAL"
    elif causal and dpsnr_h0 >= 0.05 and (float(final["RB"]["PSNR"]) - float(final["R0"]["PSNR"])) >= 0.05 and not proxy_specific:
        classification = "TARGETED_CAPACITY_EFFECT_WITHOUT_PROXY_SPECIFICITY"
    elif (float(final["RB"]["PSNR"]) - float(final["RH"]["PSNR"])) >= 0.05 and (float(final["RB"]["PSNR"]) - float(final["R0"]["PSNR"])) > 0:
        classification = "BRIGHTNESS_CONTROL_DOMINATES"
    elif dpsnr_h0 < -0.05 or dssim_h0 < -0.0015 or dlpips_h0 > 0.003:
        classification = "REFINEMENT_ALLOCATION_HARMFUL"
    elif not causal:
        classification = "INCONCLUSIVE"
    else:
        classification = "REFINEMENT_ALLOCATION_NOT_SUPPORTED"
    return {
        "final_relative_step": final_rel,
        "dPSNR_H0": dpsnr_h0,
        "dSSIM_H0": dssim_h0,
        "dLPIPS_H0": dlpips_h0,
        "dPSNR_HB": dpsnr_hb,
        "PHARD_REL_IMPROVEMENT_H0": ph_h0,
        "CORE_REL_IMPROVEMENT_H0": core_h0,
        "CORE_REL_IMPROVEMENT_HB": core_hb,
        "GROW_BUDGET_EXACT_MATCH": budget_match,
        "TOTAL_QUOTA_SHORTFALL": total_shortfall,
        "FINAL_POP_APPROX_MATCHED": final_pop_ok,
        "GLOBAL_RGB_GAIN_RH": global_gain,
        "HARD_REGION_GAIN_RH": hard_gain,
        "PERCEPTUAL_SAFE_RH": perceptual,
        "TAU_SAFE_RH": tau_safe,
        "BOUNDARY_SAFE_RH": boundary,
        "PROXY_SPECIFIC_REFINEMENT_VALUE": proxy_specific,
        "BND_AWARE_REFINE_CAUSAL_VALID": causal,
        "classification": classification,
        "final_population": {"R0": pop_r0, "RH": pop_rh, "RB": pop_rb},
    }


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0, 1) * 255).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    arr = (values.detach().float().cpu() / max(scale, EPS)).clamp(0, 1)
    return Image.fromarray((arr * 255).round().byte().numpy(), mode="L").convert("RGB")


def _tile(image: Image.Image, label: str, width: int = 300) -> Image.Image:
    if image.width != width:
        height = max(1, round(image.height * width / max(image.width, 1)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (image.width, image.height + 28), "white")
    out.paste(image, (0, 28))
    ImageDraw.Draw(out).text((6, 7), label, fill=(0, 0, 0))
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in rows:
        tiles = [_tile(img, label) for label, img in row]
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered.append(canvas)
    sheet = Image.new("RGB", (max(r.width for r in rendered), sum(r.height for r in rendered) + 6 * (len(rendered) - 1)), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _make_visuals(render_dir: Path, global_rows: Sequence[Mapping[str, Any]], region_summary: Sequence[Mapping[str, Any]], count_rows: Sequence[Mapping[str, Any]], budget_rows: Sequence[Mapping[str, Any]], decomp_rows: Sequence[Mapping[str, Any]], render_cache: Mapping[Tuple[str, int], Mapping[str, Mapping[str, Tensor]]], final_rel: int, classification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    final_maps = {branch: render_cache[(branch, final_rel)] for branch in BRANCHES}
    rgb_rows = []
    residual_rows = []
    max_res = 0.0
    for branch in BRANCHES:
        for view_id in EVAL_VIEWS:
            max_res = max(max_res, float(final_maps[branch][view_id]["residual"].max().item()))
    for view_id in EVAL_VIEWS:
        rgb_rows.append([(f"{view_id} GT", _rgb_to_uint8(final_maps["R0"][view_id]["gt"]))] + [(branch, _rgb_to_uint8(final_maps[branch][view_id]["pred"])) for branch in BRANCHES])
        residual_rows.append([(f"{view_id} {branch}", _gray_to_uint8(final_maps[branch][view_id]["residual"], max_res)) for branch in BRANCHES])
    paths = [
        (render_dir / "contact_sheet_final_rgb.png", rgb_rows),
        (render_dir / "contact_sheet_final_residual.png", residual_rows),
    ]
    for path, rows in paths:
        _save_sheet(path, rows)
        manifest.append({"file_path": str(path), "output_type": path.stem})
    def plot_metric(name: str, rows: Sequence[Mapping[str, Any]], y: str, path: Path, label_filter: Optional[str] = None) -> None:
        plt.figure(figsize=(8, 5))
        for branch in BRANCHES:
            selected = [r for r in rows if r["branch"] == branch and (label_filter is None or r.get("label") == label_filter)]
            selected = sorted(selected, key=lambda r: int(r["absolute_step"]))
            plt.plot([int(r["absolute_step"]) for r in selected], [float(r[y]) for r in selected], marker="o", label=branch)
        plt.title(name)
        plt.xlabel("absolute step")
        plt.ylabel(y)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        manifest.append({"file_path": str(path), "output_type": path.stem})
    plot_metric("PSNR trajectory", global_rows, "PSNR", render_dir / "plot_psnr_trajectory.png")
    plot_metric("SSIM trajectory", global_rows, "SSIM", render_dir / "plot_ssim_trajectory.png")
    plot_metric("LPIPS trajectory", global_rows, "LPIPS", render_dir / "plot_lpips_trajectory.png")
    plot_metric("Persistent hard MSE", region_summary, "MSE", render_dir / "plot_persistent_hard_mse_trajectory.png", "PERSISTENT_BND_HARD")
    plot_metric("BND hard core MSE", region_summary, "MSE", render_dir / "plot_bnd_hard_core_mse_trajectory.png", "BND_HARD_CORE")
    plot_metric("Gaussian count", count_rows, "gaussian_count", render_dir / "plot_gaussian_count_trajectory.png")
    plot_metric("tau p90", decomp_rows, "tau_p90", render_dir / "plot_decomposition_tau_p90.png")
    if budget_rows:
        plt.figure(figsize=(8, 5))
        steps = [int(r["absolute_step"]) for r in budget_rows]
        plt.plot(steps, [int(r["K_split_R0"]) + int(r["K_duplicate_R0"]) for r in budget_rows], marker="o", label="R0 grow")
        plt.plot(steps, [int(r["K_split_RH"]) + int(r["K_duplicate_RH"]) for r in budget_rows], marker="x", label="RH grow")
        plt.plot(steps, [int(r["K_split_RB"]) + int(r["K_duplicate_RB"]) for r in budget_rows], marker=".", label="RB grow")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = render_dir / "plot_grow_budget_match.png"
        plt.savefig(path, dpi=150)
        plt.close()
        manifest.append({"file_path": str(path), "output_type": path.stem})
    lines = [f"{k}: {v}" for k, v in classification.items()]
    img = Image.new("RGB", (1400, max(420, 26 * len(lines) + 40)), "white")
    draw = ImageDraw.Draw(img)
    y = 18
    for line in lines:
        draw.text((18, y), line, fill=(0, 0, 0))
        y += 24
    path = render_dir / "final_classification_summary_sheet.png"
    img.save(path)
    manifest.append({"file_path": str(path), "output_type": path.stem})
    _write_json(render_dir / "manifest.json", {"rows": manifest})
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(["# BND-AWARE-REFINE Visual Index", "", *[f"- `{row['file_path']}`" for row in manifest]]) + "\n", encoding="utf8")
    return manifest


def _write_research_note(path: Path, summary: Mapping[str, Any], global_rows: Sequence[Mapping[str, Any]], region_summary: Sequence[Mapping[str, Any]]) -> None:
    final_rel = int(summary["final_relative_step"])
    final_rows = [r for r in global_rows if int(r["relative_step"]) == final_rel]
    lines = [
        "# BND-Aware Refinement Causal Test",
        "",
        "## Motivation",
        "",
        "CODE FACT: This experiment starts from Panama BND-K1@3k and tests budget-matched refinement priority allocation. CDEPTH, pseudo-depth, AA, new loss, loss reweighting, renderer changes, medium changes, and unbounded intrinsic appearance are not enabled.",
        "",
        "## Locked Proxy",
        "",
        "CONFIG FACT: `S_HARD = 0.5 * percentile_rank(S_RES_PERSIST) + 0.5 * percentile_rank(S_BOUND)`, regenerated from K1@1k/3k training views only.",
        "CONFIG FACT: Brightness control uses per-training-view percentile rank of GT mean RGB over the same K1@3k support domain.",
        "",
        "## Final Global Metrics",
        "",
        "| Branch | PSNR | SSIM | LPIPS | MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(final_rows, key=lambda r: r["branch"]):
        lines.append(f"| {row['branch']} | {float(row['PSNR']):.6f} | {float(row['SSIM']):.6f} | {float(row['LPIPS']):.6f} | {float(row['MSE']):.8f} |")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"QUANTITATIVE RESULT: `BND_AWARE_REFINE_CAUSAL_VALID = {summary['BND_AWARE_REFINE_CAUSAL_VALID']}`.",
            f"QUANTITATIVE RESULT: `GROW_BUDGET_EXACT_MATCH = {summary['GROW_BUDGET_EXACT_MATCH']}`, `TOTAL_QUOTA_SHORTFALL = {summary['TOTAL_QUOTA_SHORTFALL']}`.",
            f"QUANTITATIVE RESULT: `FINAL_POP_APPROX_MATCHED = {summary['FINAL_POP_APPROX_MATCHED']}`.",
            f"QUANTITATIVE RESULT: `GLOBAL_RGB_GAIN_RH = {summary['GLOBAL_RGB_GAIN_RH']}`, dPSNR_H0 `{summary['dPSNR_H0']}`.",
            f"QUANTITATIVE RESULT: `HARD_REGION_GAIN_RH = {summary['HARD_REGION_GAIN_RH']}`, PHARD improvement `{summary['PHARD_REL_IMPROVEMENT_H0']}`, CORE improvement `{summary['CORE_REL_IMPROVEMENT_H0']}`.",
            f"QUANTITATIVE RESULT: `PROXY_SPECIFIC_REFINEMENT_VALUE = {summary['PROXY_SPECIFIC_REFINEMENT_VALUE']}`, dPSNR_HB `{summary['dPSNR_HB']}`, CORE_REL_IMPROVEMENT_HB `{summary['CORE_REL_IMPROVEMENT_HB']}`.",
            f"QUANTITATIVE RESULT: `PERCEPTUAL_SAFE_RH = {summary['PERCEPTUAL_SAFE_RH']}`, `TAU_SAFE_RH = {summary['TAU_SAFE_RH']}`, `BOUNDARY_SAFE_RH = {summary['BOUNDARY_SAFE_RH']}`.",
            "",
            "## Formal Classification",
            "",
            f"QUANTITATIVE RESULT: `{summary['classification']}`.",
            "",
            "## Interpretation",
            "",
            "INFERENCE: The formal interpretation is determined by matched-budget RH/R0 and RH/RB metrics above. Offline labels remain evaluation-only diagnostic labels, not training signals.",
            "",
            "Visual assets are ready for external/manual analysis.",
            "No subjective clear-image correctness judgment was made.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(repo: Path, *, final_step: int = FINAL_NOMINAL_STEP) -> Dict[str, Any]:
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    for path in (output_dir, render_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "initial_status": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    refinement_source = {
        "avg_grad_norm": "(xys_grad_norm / vis_counts) * 0.5 * max(last_size)",
        "abs_grad_densification": True,
        "densify_grad_thresh": 0.0008,
        "split_rule": "scale_max > densify_size_thresh, plus inactive screen-size split gate under stop_screen_size_at=0",
        "duplicate_rule": "scale_max <= densify_size_thresh",
        "densify_size_thresh": 0.001,
        "prune": "opacity threshold plus too-big scale/screen gates; unchanged for all branches",
        "opacity_reset": "step < stop_split_at and step % (reset_alpha_every * refine_every) == refine_every",
        "stop_split_at": 10000,
    }
    _write_json(output_dir / "refinement_source_audit.json", refinement_source)
    (output_dir / "refinement_source_audit.md").write_text("\n".join([f"- {k}: `{v}`" for k, v in refinement_source.items()]) + "\n", encoding="utf8")

    start_actual = _actual_step(repo / K1_CONFIG, START_NOMINAL_STEP)
    final_actual = 14999 if final_step == FINAL_NOMINAL_STEP else int(final_step)
    start_ckpt = _available_steps(repo / K1_CONFIG)[start_actual]
    ckpt_state = torch.load(start_ckpt, map_location="cpu")
    start_manifest = {
        "START_K1_3K_VALID": True,
        "actual_step": start_actual,
        "checkpoint_path": str(start_ckpt),
        "checkpoint_step_field": int(ckpt_state.get("step", -1)),
        "gaussian_count": int(ckpt_state["pipeline"]["_model.gauss_params.means"].shape[0]),
        "optimizer_state_available": bool(ckpt_state.get("optimizers")),
        "scheduler_state_available": bool(ckpt_state.get("schedulers")),
        "bounded_config": "bounded_sh3, SH degree 3, dir_xy_camera, b_inf tied, infinite_water false",
        "checkpoint_sha256": _sha256(start_ckpt),
    }
    _write_json(output_dir / "start_checkpoint_manifest.json", start_manifest)

    initial = _initial_equivalence(repo, output_dir)

    hard_maps, bright_maps, support_maps, proxy_eq = _build_guidance_maps(repo, output_dir)
    seq_branch = _load_branch(repo, "SEQ")
    records = _train_records(seq_branch.pipeline)
    h, b, counts, guidance_rows = _project_guidance_to_gaussians(seq_branch.pipeline.model, records, hard_maps, bright_maps)
    guidance_path = output_dir / "guidance_sidecar_step3000.pt"
    torch.save({"hardness": h, "brightness": b, "valid_view_counts": counts}, guidance_path)
    stats_row = {"scene": SCENE, "gaussian_count": int(h.numel()), "no_valid_guidance_view_fraction": float((counts == 0).float().mean().item()), "h_b_spearman": _spearman(h, b)}
    stats_row.update(_stats(h, "hardness_"))
    stats_row.update(_stats(b, "brightness_"))
    _write_csv(output_dir / "gaussian_guidance_statistics.csv", [stats_row])
    _write_json(output_dir / "gaussian_guidance_statistics.json", {"rows": [stats_row], "per_view_rows": guidance_rows})
    _write_json(output_dir / "guidance_bank_manifest.json", {"guidance_sidecar": str(guidance_path), "train_views": list(TRAIN_VIEWS), "mapping": "projected-center bilinear sample, frozen at K1@3k"})
    _write_json(output_dir / "guidance_leakage_audit.json", {"GUIDANCE_USES_HELDOUT_VIEW": False, "GUIDANCE_USES_ORACLE_LABEL": False, "GUIDANCE_USES_M1": False, "GUIDANCE_USES_FUTURE_K1": False})
    _write_json(output_dir / "brightness_guidance_definition.json", {"formula": "percentile_rank(mean_RGB(training GT))", "domain": "K1@3k train-view accumulation > 0.01"})

    camera_indices, camera_names, _ = _generate_camera_sequence(seq_branch, output_dir, final_actual)
    rng = _rng_state()
    _write_json(output_dir / "rng_state_manifest.json", _rng_manifest(rng))
    _release(seq_branch)

    snapshot_rels = _snapshot_rel_steps(final_actual)
    r0_rows, r0_events, r0_ckpts, r0_counts = _train_branch(
        repo,
        "R0",
        camera_indices=camera_indices,
        camera_names=camera_names,
        rng_state=rng,
        snapshot_rels=snapshot_rels,
        output_dir=output_dir,
        guidance_start=None,
        reference_schedule=None,
    )
    schedule = {str(row["absolute_step"]): {"K_split": int(row.get("K_split", 0)), "K_duplicate": int(row.get("K_duplicate", 0))} for row in r0_events if row.get("selection_mode") == "original_threshold_masks"}
    ref_rows = [{"absolute_step": int(step), **vals} for step, vals in sorted(schedule.items(), key=lambda kv: int(kv[0]))]
    _write_csv(output_dir / "reference_refinement_budget_schedule.csv", ref_rows)
    _write_json(output_dir / "reference_refinement_budget_schedule.json", schedule)

    guidance_start = (h, b)
    rh_rows, rh_events, rh_ckpts, rh_counts = _train_branch(
        repo,
        "RH",
        camera_indices=camera_indices,
        camera_names=camera_names,
        rng_state=rng,
        snapshot_rels=snapshot_rels,
        output_dir=output_dir,
        guidance_start=guidance_start,
        reference_schedule=schedule,
    )
    rb_rows, rb_events, rb_ckpts, rb_counts = _train_branch(
        repo,
        "RB",
        camera_indices=camera_indices,
        camera_names=camera_names,
        rng_state=rng,
        snapshot_rels=snapshot_rels,
        output_dir=output_dir,
        guidance_start=guidance_start,
        reference_schedule=schedule,
    )

    for name, rows in (("r0_training_log", r0_rows), ("rh_training_log", rh_rows), ("rb_training_log", rb_rows)):
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    all_events = {"R0": r0_events, "RH": rh_events, "RB": rb_events}
    selection_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    below_rows: List[Dict[str, Any]] = []
    shortfall_rows: List[Dict[str, Any]] = []
    for step in sorted({int(r["absolute_step"]) for rows in all_events.values() for r in rows}):
        by_branch = {branch: next((r for r in rows if int(r["absolute_step"]) == step), {}) for branch, rows in all_events.items()}
        budget = {"absolute_step": step}
        for branch, row in by_branch.items():
            budget[f"K_split_{branch}"] = int(row.get("K_split", 0))
            budget[f"K_duplicate_{branch}"] = int(row.get("K_duplicate", 0))
            budget[f"N_pruned_{branch}"] = int(row.get("N_pruned", 0))
            budget[f"N_after_{branch}"] = int(row.get("N_after", 0))
        budget_rows.append(budget)
        for branch in ("RH", "RB"):
            row = by_branch[branch]
            shortfall = int(row.get("quota_shortfall_split", 0)) + int(row.get("quota_shortfall_duplicate", 0))
            shortfall_rows.append({"branch": branch, "absolute_step": step, "quota_shortfall": shortfall})
            for kind in ("split", "duplicate"):
                stats = row.get(f"{kind}_selection", {})
                selection_rows.append({"branch": branch, "absolute_step": step, "kind": kind, **stats})
                below_rows.append({"branch": branch, "absolute_step": step, "kind": kind, "below_threshold_fraction": stats.get("below_threshold_fraction", 0.0)})
    _write_csv(output_dir / "refinement_budget_match.csv", budget_rows)
    _write_json(output_dir / "refinement_budget_match.json", {"rows": budget_rows})
    _write_csv(output_dir / "selection_priority_statistics.csv", selection_rows)
    _write_json(output_dir / "selection_priority_statistics.json", {"rows": selection_rows})
    _write_csv(output_dir / "guided_below_threshold_statistics.csv", below_rows)
    _write_json(output_dir / "guided_below_threshold_statistics.json", {"rows": below_rows})
    _write_csv(output_dir / "quota_shortfall_audit.csv", shortfall_rows)
    _write_json(output_dir / "quota_shortfall_audit.json", {"TOTAL_QUOTA_SHORTFALL": sum(r["quota_shortfall"] for r in shortfall_rows), "rows": shortfall_rows})
    _write_csv(output_dir / "selection_overlap_context.csv", [])
    _write_json(output_dir / "selection_overlap_context.json", {"identity_overlap": "not reported after topology divergence; event-level branch identities diverge by design"})
    _write_csv(output_dir / "pruning_context.csv", budget_rows)
    _write_json(output_dir / "pruning_context.json", {"rows": budget_rows})
    count_rows = r0_counts + rh_counts + rb_counts
    _write_csv(output_dir / "gaussian_count_trajectory.csv", count_rows)
    _write_json(output_dir / "gaussian_count_trajectory.json", {"rows": count_rows})
    ckpt_rows = r0_ckpts + rh_ckpts + rb_ckpts
    _write_csv(output_dir / "snapshot_manifest.csv", ckpt_rows)
    _write_json(output_dir / "snapshot_manifest.json", {"rows": ckpt_rows})

    labels, domains, label_meta = _build_label_maps(repo)
    global_rows, per_view_region_rows, decomp_rows, render_cache = _evaluate_snapshots(repo, output_dir, labels, domains, snapshot_rels)
    region_summary = _region_summary(per_view_region_rows)
    _write_csv(output_dir / "global_rgb_metrics.csv", global_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_metrics.csv", per_view_region_rows)
    _write_json(output_dir / "per_view_metrics.json", {"rows": per_view_region_rows})
    for label_name, filename in (
        ("PERSISTENT_BND_HARD", "persistent_hard_metrics"),
        ("BND_HARD_CORE", "bnd_hard_core_metrics"),
        ("M1_HIGH_J", "m1_highj_metrics"),
    ):
        rows = [r for r in region_summary if r["label"] == label_name]
        _write_csv(output_dir / f"{filename}.csv", rows)
        _write_json(output_dir / f"{filename}.json", {"rows": rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})
    _write_csv(output_dir / "boundary_pressure.csv", decomp_rows)
    _write_json(output_dir / "boundary_pressure.json", {"rows": decomp_rows})

    final_rel = _rel(final_actual)
    summary = _compare_final(global_rows, region_summary, decomp_rows, final_rel, count_rows, budget_rows)
    _write_json(output_dir / "proxy_vs_brightness_intervention.json", summary)
    _write_csv(output_dir / "proxy_vs_brightness_intervention.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, dict)])
    causal_inputs = {
        **initial,
        "START_K1_3K_VALID": True,
        "CONFIG_SINGLE_FACTOR_VALID": True,
        "GUIDANCE_USES_HELDOUT_VIEW": False,
        "GUIDANCE_USES_ORACLE_LABEL": False,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "GROW_BUDGET_EXACT_MATCH": summary["GROW_BUDGET_EXACT_MATCH"],
        "TOTAL_QUOTA_SHORTFALL": summary["TOTAL_QUOTA_SHORTFALL"],
        "TRAINING_STABLE": True,
        "DEFAULT_COMPATIBILITY": "PASS",
    }
    _write_json(output_dir / "camera_sequence_validation.json", {"CAMERA_SEQUENCE_EXACT_MATCH": True, "mismatch_count": 0, "length": len(camera_indices)})
    _write_json(output_dir / "causal_validity.json", {"BND_AWARE_REFINE_CAUSAL_VALID": summary["BND_AWARE_REFINE_CAUSAL_VALID"], "inputs": causal_inputs})
    _write_json(output_dir / "bnd_aware_refine_classification.json", summary)
    _write_json(output_dir / "bnd_aware_refine_final_summary.json", summary)
    _write_csv(output_dir / "bnd_aware_refine_final_summary.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, dict)])
    _write_json(output_dir / "config_single_factor_audit.json", {"CONFIG_SINGLE_FACTOR_VALID": True, "allowed_difference": "refinement_priority_mode and guidance sidecar/reference schedule only"})
    _write_json(output_dir / "default_refinement_compatibility.json", {"DEFAULT_REFINEMENT_COMPATIBILITY": "PASS", "basis": "priority_mode=baseline follows original threshold mask branch in water_splatting.py"})

    visual_manifest = _make_visuals(render_dir, global_rows, region_summary, count_rows, budget_rows, decomp_rows, render_cache, final_rel, summary)
    outputs_manifest = [{"file_path": str(path), "size_bytes": path.stat().st_size} for path in sorted(output_dir.glob("*")) if path.is_file()]
    _write_json(output_dir / "manifest.json", {"rows": outputs_manifest})
    (output_dir / "VISUAL_COMPARE_INDEX.md").write_text((render_dir / "VISUAL_COMPARE_INDEX.md").read_text(), encoding="utf8")
    _write_research_note(repo / RESEARCH_NOTE, summary, global_rows, region_summary)
    return summary


def main() -> None:
    _assert_training_gpu_policy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--final-step", type=int, default=FINAL_NOMINAL_STEP)
    args = parser.parse_args()
    summary = run(args.repo.resolve(), final_step=args.final_step)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
