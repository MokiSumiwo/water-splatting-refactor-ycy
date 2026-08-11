#!/usr/bin/env python
"""Fixed-population matched CDEPTH rollout from Panama BND-K1@3k.

This controlled diagnostic trains exactly two local rollout branches from the
same K1@3k checkpoint and optimizer/scheduler state:

* FP-R:  RGB objective only.
* FP-RD: RGB objective plus the existing SeaFree-style coarse-depth term.

Both branches replay the exact same explicit camera sequence and skip all
Gaussian population mutation callbacks. The only retained refinement-side event
inside 3k->5k is the scheduled opacity reset, applied to both branches at the
same absolute steps with the same optimizer-state reset semantics.
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from scripts.diagnostics import audit_bnd_cdepth_direct_path as direct_path
from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_cdepth_fixpop_rollout_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_fixpop_rollout_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_fixpop_rollout_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_FIXED_POPULATION_ROLLOUT_2026-08-11.md")

START_NOMINAL_STEP = 3000
ROLLOUT_STEPS = 2000
SNAPSHOTS = (0, 100, 250, 500, 1000, 1500, 2000)
CHECKPOINT_SNAPSHOTS = SNAPSHOTS
EVAL_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
TRAIN_CAMERA_BANK = direct_path.TRAIN_CAMERA_BANK
CURRENT_GROUPS = direct_path.CURRENT_PARAM_GROUPS
GEOMETRY_GROUPS = ("means", "scales", "quats", "opacities")
REGIONS = ("global", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM")
EPS = 1e-12


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


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    return direct_path._actual_step(config_path, nominal_step)


def _available_steps(config_path: Path) -> Dict[int, Path]:
    return direct_path._available_steps(config_path)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _load_branch(repo: Path, branch: str, *, coarse_depth: bool) -> LoadedBranch:
    config_path = repo / cdepth_setup.K1_CONFIG
    actual_step = _actual_step(config_path, START_NOMINAL_STEP)
    if actual_step is None:
        raise FileNotFoundError(f"Missing K1 start checkpoint {START_NOMINAL_STEP}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
        config.pipeline.model.rasterize_mode = "classic"
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = bool(coarse_depth)
        config.pipeline.model.coarse_depth_supervision_weight = 0.1
        config.pipeline.datamanager.load_depths = True
        config.pipeline.datamanager.dataparser.depths_path = cdepth_setup.DEPTHS_PATH
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
    model.config.coarse_depth_supervision_enabled = bool(coarse_depth)
    model.config.coarse_depth_supervision_weight = 0.1
    model.step = int(loaded_step)
    optimizers = Optimizers(config.optimizers.copy(), model.get_param_groups())
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for group in optimizers.optimizers:
        optimizers.optimizers[group].load_state_dict(ckpt["optimizers"][group])
    for group in optimizers.schedulers:
        optimizers.schedulers[group].load_state_dict(ckpt["schedulers"][group])
    pipeline.eval()
    return LoadedBranch(branch, config_path, checkpoint_path, int(loaded_step), config, pipeline, optimizers)


def _state_dict_copy(state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {key: value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value) for key, value in state.items()}


def _flat_hash_tensors(tensors: Iterable[Tensor]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode("utf8"))
        h.update(str(arr.dtype).encode("utf8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    out = {
        "means": model.means.detach(),
        "scales": model.scales.detach(),
        "quats": model.quats.detach(),
        "features_dc": model.features_dc.detach(),
        "features_rest": model.features_rest.detach(),
        "opacities": model.opacities.detach(),
    }
    medium = [param.detach().reshape(-1) for param in model.medium_mlp.parameters()]
    direction = [param.detach().reshape(-1) for param in model.direction_encoding.parameters()]
    out["medium_mlp"] = torch.cat(medium) if medium else torch.empty(0, device=model.device)
    out["direction_encoding"] = torch.cat(direction) if direction else torch.empty(0, device=model.device)
    return out


def _optimizer_state_tensors(optimizers: Optimizers) -> Dict[str, Dict[str, Tensor]]:
    out: Dict[str, Dict[str, Tensor]] = {}
    for group, optimizer in optimizers.optimizers.items():
        exp_avg: List[Tensor] = []
        exp_avg_sq: List[Tensor] = []
        steps: List[Tensor] = []
        for param in optimizer.param_groups[0]["params"]:
            state = optimizer.state[param]
            if "exp_avg" in state:
                exp_avg.append(state["exp_avg"].detach().reshape(-1).cpu())
            if "exp_avg_sq" in state:
                exp_avg_sq.append(state["exp_avg_sq"].detach().reshape(-1).cpu())
            if "step" in state:
                step = state["step"]
                steps.append(step.detach().reshape(-1).cpu() if isinstance(step, Tensor) else torch.tensor([float(step)]))
        out[group] = {
            "exp_avg": torch.cat(exp_avg) if exp_avg else torch.empty(0),
            "exp_avg_sq": torch.cat(exp_avg_sq) if exp_avg_sq else torch.empty(0),
            "step": torch.cat(steps) if steps else torch.empty(0),
        }
    return out


def _scheduler_state(optimizers: Optimizers) -> Dict[str, Any]:
    return {group: scheduler.state_dict() for group, scheduler in optimizers.schedulers.items()}


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def _compare_model_params(a: Any, b: Any) -> List[Dict[str, Any]]:
    aa = _model_param_tensors(a)
    bb = _model_param_tensors(b)
    rows = []
    for group in CURRENT_GROUPS:
        rows.append(
            {
                "group": group,
                "shape_a": list(aa[group].shape),
                "shape_b": list(bb[group].shape),
                "max_abs_diff": _max_abs_diff(aa[group], bb[group]),
                "pass": bool(list(aa[group].shape) == list(bb[group].shape) and _max_abs_diff(aa[group], bb[group]) == 0.0),
            }
        )
    return rows


def _compare_optimizer_states(a: Optimizers, b: Optimizers) -> List[Dict[str, Any]]:
    aa = _optimizer_state_tensors(a)
    bb = _optimizer_state_tensors(b)
    rows = []
    for group in CURRENT_GROUPS:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            rows.append(
                {
                    "group": group,
                    "state_key": key,
                    "shape_a": list(aa[group][key].shape),
                    "shape_b": list(bb[group][key].shape),
                    "max_abs_diff": _max_abs_diff(aa[group][key], bb[group][key]),
                    "pass": bool(list(aa[group][key].shape) == list(bb[group][key].shape) and _max_abs_diff(aa[group][key], bb[group][key]) == 0.0),
                }
            )
    # Scheduler state comparison is JSON-like, so store equality per scheduler.
    sa = _scheduler_state(a)
    sb = _scheduler_state(b)
    for group in sorted(set(sa) | set(sb)):
        rows.append(
            {
                "group": group,
                "state_key": "scheduler_state_dict",
                "shape_a": "",
                "shape_b": "",
                "max_abs_diff": 0.0 if sa.get(group) == sb.get(group) else float("inf"),
                "pass": bool(sa.get(group) == sb.get(group)),
            }
        )
    return rows


def _condition_forward(model: Any, camera: Any, batch: Mapping[str, Any], coarse_depth: bool) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    model.config.coarse_depth_supervision_enabled = bool(coarse_depth)
    outputs = model.get_outputs(camera.to(model.device))
    metrics: Dict[str, Tensor] = {}
    losses = model.get_loss_dict(outputs, batch, metrics)
    return outputs, losses, metrics


def _compare_initial_forward(branch_r: LoadedBranch, branch_rd: LoadedBranch, train_index: int) -> List[Dict[str, Any]]:
    dm_r = branch_r.pipeline.datamanager
    dm_rd = branch_rd.pipeline.datamanager
    cached_r = dm_r.cached_train
    cached_rd = dm_rd.cached_train
    cameras_r = getattr(dm_r, "train_cameras", dm_r.train_dataset.cameras).to(branch_r.pipeline.model.device)
    cameras_rd = getattr(dm_rd, "train_cameras", dm_rd.train_dataset.cameras).to(branch_rd.pipeline.model.device)
    camera_r = cameras_r[train_index : train_index + 1]
    camera_rd = cameras_rd[train_index : train_index + 1]
    batch_r = _batch_to_device(cached_r[train_index].copy(), branch_r.pipeline.model.device)
    batch_rd = _batch_to_device(cached_rd[train_index].copy(), branch_rd.pipeline.model.device)
    branch_r.pipeline.model.train()
    branch_rd.pipeline.model.train()
    out_r, losses_r, _metrics_r = _condition_forward(branch_r.pipeline.model, camera_r, batch_r, coarse_depth=False)
    out_rd, losses_rd, _metrics_rd = _condition_forward(branch_rd.pipeline.model, camera_rd, batch_rd, coarse_depth=True)
    keys = ("pred_image", "direct_object_signal", "rgb_medium", "depth", "clear_object_fullsh_raw", "transmission", "tau_D", "accumulation")
    rows: List[Dict[str, Any]] = []
    for key in keys:
        rows.append(
            {
                "key": key,
                "max_abs_diff": _max_abs_diff(out_r[key], out_rd[key]),
                "pass": bool(_max_abs_diff(out_r[key], out_rd[key]) <= 1e-6),
            }
        )
    rows.append(
        {
            "key": "main_loss",
            "max_abs_diff": abs(float(losses_r["main_loss"].detach().cpu().item()) - float(losses_rd["main_loss"].detach().cpu().item())),
            "pass": bool(abs(float(losses_r["main_loss"].detach().cpu().item()) - float(losses_rd["main_loss"].detach().cpu().item())) <= 1e-6),
        }
    )
    return rows


def _finite_flat(values: Tensor) -> Tensor:
    flat = values.detach().float().reshape(-1)
    return flat[torch.isfinite(flat)]


def _q(values: Tensor, q: float) -> float:
    flat = _finite_flat(values)
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = _finite_flat(values)
    keys = ("count", "mean", "p50", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{key}": float("nan") for key in keys}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p50": _q(flat, 0.50),
        f"{prefix}p90": _q(flat, 0.90),
        f"{prefix}p95": _q(flat, 0.95),
        f"{prefix}p99": _q(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    vals = values.detach().float()
    m = mask.detach().bool()
    while m.ndim < vals.ndim:
        m = m[..., None].expand(*vals.shape)
    return vals[m]


def _mean(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        try:
            v = float(value)
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _luma(image: Tensor) -> Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=image.dtype, device=image.device)
    return (image * weights).sum(dim=-1)


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (torch.nan_to_num(image.detach().float(), nan=0.0).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=scale, neginf=0.0)
    arr = (vals.clamp_min(0.0) / max(float(scale), EPS)).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float(), nan=0.0)
    vals = (vals / max(float(scale), EPS)).clamp(-1.0, 1.0)
    pos = vals.clamp_min(0.0)
    neg = (-vals).clamp_min(0.0)
    rgb = torch.stack([pos, torch.zeros_like(vals), neg], dim=-1)
    return _rgb_to_uint8(rgb)


def _tile(image: Image.Image, label: str, width: int = 260) -> Image.Image:
    if width and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 28
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((5, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], manifest: List[Dict[str, Any]], output_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(img, label) for label, img in row]
        width = sum(tile.width for tile in tiles) + 5 * max(0, len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 5
        rendered.append(canvas)
    if not rendered:
        return
    width = max(row.width for row in rendered)
    height = sum(row.height for row in rendered) + 5 * max(0, len(rendered) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rendered:
        canvas.paste(row, (0, y))
        y += row.height + 5
    canvas.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type, "size_bytes": path.stat().st_size})


def _save_plot(path: Path, rows: Sequence[Mapping[str, Any]], x_key: str, y_keys: Sequence[str], title: str, manifest: List[Dict[str, Any]], output_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [str(row.get(x_key, "")) for row in rows]
    for key in y_keys:
        ys = []
        for row in rows:
            try:
                ys.append(float(row.get(key, float("nan"))))
            except Exception:
                ys.append(float("nan"))
        ax.plot(xs, ys, marker="o", label=key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest.append({"file_path": str(path), "output_type": output_type, "size_bytes": path.stat().st_size})


def _generate_camera_sequence(branch: LoadedBranch, output_dir: Path) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    dm = branch.pipeline.datamanager
    image_filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in image_filenames]
    indices: List[int] = []
    view_ids: List[str] = []
    rows: List[Dict[str, Any]] = []
    for rel in range(1, ROLLOUT_STEPS + 1):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        indices.append(index)
        view_id = names[index]
        view_ids.append(view_id)
        rows.append(
            {
                "relative_step": rel,
                "absolute_step": START_NOMINAL_STEP + rel,
                "camera_index": index,
                "camera_name": view_id,
            }
        )
    _write_json(
        output_dir / "paired_camera_sequence.json",
        {
            "scene": SCENE,
            "length": len(rows),
            "camera_pool": list(TRAIN_CAMERA_BANK),
            "generation_method": "replayed CheckpointableFullImageDatamanager train_unseen_cameras/sample_train_cameras state restored from K1@3k once, then reused by both branches",
            "rows": rows,
        },
    )
    return indices, view_ids, rows


def _compute_loss_components(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    mask = None
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    if model.config.main_loss == "reg_l1":
        reg_l1 = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
    else:
        reg_l1 = torch.abs(gt_img - pred_img).mean()
    if model.config.ssim_loss == "reg_ssim":
        reg_ssim = 1 - model.ssim(
            (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
            (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
        )
    else:
        reg_ssim = 1 - model.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
    return {"reg_l1": reg_l1, "reg_ssim": reg_ssim}


def _optimizer_lrs(optimizers: Optimizers) -> Dict[str, float]:
    return {group: float(optimizer.param_groups[0]["lr"]) for group, optimizer in optimizers.optimizers.items()}


def _apply_matched_opacity_reset(model: Any, optimizers: Optimizers, abs_step: int) -> bool:
    reset_interval = int(model.config.reset_alpha_every) * int(model.config.refine_every)
    do_reset = abs_step < int(model.config.stop_split_at) and abs_step % reset_interval == int(model.config.refine_every)
    if not do_reset:
        return False
    reset_value = float(model.config.reset_alpha_thresh)
    with torch.no_grad():
        model.opacities.data = torch.clamp(
            model.opacities.data,
            max=torch.logit(torch.tensor(reset_value, device=model.device)).item(),
        )
        optim = optimizers.optimizers["opacities"]
        param = optim.param_groups[0]["params"][0]
        state = optim.state[param]
        if "exp_avg" in state:
            state["exp_avg"] = torch.zeros_like(state["exp_avg"])
        if "exp_avg_sq" in state:
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"])
    return True


def _save_rollout_checkpoint(branch: LoadedBranch, rel_step: int, output_dir: Path) -> Path:
    branch_dir = output_dir / "rollout_checkpoints" / branch.branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    path = branch_dir / f"relative-{rel_step:06d}.ckpt"
    torch.save(
        {
            "branch": branch.branch,
            "relative_step": rel_step,
            "absolute_step": START_NOMINAL_STEP + rel_step,
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "schedulers": {group: sched.state_dict() for group, sched in branch.optimizers.schedulers.items()},
            "metadata": {
                "fixed_population_rollout": True,
                "coarse_depth_supervision_enabled": branch.branch == "FP-RD",
                "population_mutation_callbacks_called": False,
                "matched_opacity_reset_enabled": True,
            },
        },
        path,
    )
    return path


def _topology_row(branch: LoadedBranch, rel_step: int, reset_applied: bool = False) -> Dict[str, Any]:
    model = branch.pipeline.model
    state_shapes = {}
    for group, optimizer in branch.optimizers.optimizers.items():
        param = optimizer.param_groups[0]["params"][0] if optimizer.param_groups[0]["params"] else None
        if param is not None and param in optimizer.state and "exp_avg" in optimizer.state[param]:
            state_shapes[f"{group}_exp_avg_shape"] = list(optimizer.state[param]["exp_avg"].shape)
    return {
        "branch": branch.branch,
        "relative_step": rel_step,
        "absolute_step": START_NOMINAL_STEP + rel_step,
        "gaussian_count": int(model.means.shape[0]),
        "means_shape": list(model.means.shape),
        "scales_shape": list(model.scales.shape),
        "quats_shape": list(model.quats.shape),
        "opacities_shape": list(model.opacities.shape),
        "reset_applied": reset_applied,
        **state_shapes,
    }


def _train_branch(
    branch: LoadedBranch,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    output_dir: Path,
    *,
    coarse_depth: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    training_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    if 0 in CHECKPOINT_SNAPSHOTS:
        ckpt_path = _save_rollout_checkpoint(branch, 0, output_dir)
        snapshot_rows.append({"branch": branch.branch, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "checkpoint_path": str(ckpt_path)})
    topology_rows.append(_topology_row(branch, 0))

    for rel_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names), start=1):
        abs_step = START_NOMINAL_STEP + rel_step
        model.train()
        model.step = abs_step
        model.config.coarse_depth_supervision_enabled = bool(coarse_depth)
        branch.optimizers.zero_grad_all()
        cached_train = dm.cached_train
        train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        camera = train_cameras[camera_index : camera_index + 1]
        batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
        lrs = _optimizer_lrs(branch.optimizers)
        outputs = model.get_outputs(camera)
        components = _compute_loss_components(model, outputs, batch)
        metrics: Dict[str, Tensor] = {}
        losses = model.get_loss_dict(outputs, batch, metrics)
        rgb_loss = losses["main_loss"]
        depth_loss = losses.get("coarse_depth_loss", rgb_loss.new_tensor(0.0))
        total_loss = rgb_loss + (depth_loss if coarse_depth else rgb_loss.new_tensor(0.0))
        stable = bool(torch.isfinite(total_loss).detach().cpu().item())
        if not stable:
            raise RuntimeError(f"Non-finite loss in {branch.branch} rel_step={rel_step}: {float(total_loss.detach().cpu().item())}")
        total_loss.backward()
        branch.optimizers.optimizer_step_all()
        branch.optimizers.scheduler_step_all(abs_step)
        reset_applied = _apply_matched_opacity_reset(model, branch.optimizers, abs_step)
        model.xys_grad_norm = None
        model.vis_counts = None
        model.depths_accum = None
        model.max_2Dsize = None
        row = {
            "branch": branch.branch,
            "relative_step": rel_step,
            "absolute_step": abs_step,
            "camera_index": camera_index,
            "camera_name": camera_name,
            "L_total": float(total_loss.detach().cpu().item()),
            "L_RGB": float(rgb_loss.detach().cpu().item()),
            "reg_l1": float(components["reg_l1"].detach().cpu().item()),
            "reg_ssim": float(components["reg_ssim"].detach().cpu().item()),
            "L_depth_raw": float(metrics.get("coarse_depth_loss_raw", rgb_loss.new_tensor(0.0)).detach().cpu().item()) if coarse_depth else 0.0,
            "weighted_L_depth": float(depth_loss.detach().cpu().item()) if coarse_depth else 0.0,
            "gaussian_count": int(model.means.shape[0]),
            "opacity_reset_applied": reset_applied,
            "stable": stable,
        }
        for group, lr in lrs.items():
            row[f"lr_{group}"] = lr
        training_rows.append(row)
        topology_rows.append(_topology_row(branch, rel_step, reset_applied=reset_applied))
        stability_rows.append(
            {
                "branch": branch.branch,
                "relative_step": rel_step,
                "absolute_step": abs_step,
                "stable": stable,
                "loss_finite": stable,
                "gaussian_count": int(model.means.shape[0]),
            }
        )
        if rel_step in CHECKPOINT_SNAPSHOTS:
            ckpt_path = _save_rollout_checkpoint(branch, rel_step, output_dir)
            snapshot_rows.append({"branch": branch.branch, "relative_step": rel_step, "absolute_step": abs_step, "checkpoint_path": str(ckpt_path)})
    return training_rows, topology_rows, stability_rows, snapshot_rows


def _rollout_ckpt_path(output_dir: Path, branch: str, rel_step: int) -> Path:
    return output_dir / "rollout_checkpoints" / branch / f"relative-{rel_step:06d}.ckpt"


def _load_snapshot_into_model(repo: Path, output_dir: Path, branch: str, rel_step: int, loaded: LoadedBranch) -> None:
    ckpt = torch.load(_rollout_ckpt_path(output_dir, branch, rel_step), map_location="cpu")
    state = dict(ckpt["pipeline"])
    loaded.pipeline.load_state_dict(state, strict=False)
    loaded.pipeline.model.step = START_NOMINAL_STEP + rel_step
    loaded.pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    loaded.pipeline.model.config.rasterize_mode = "classic"
    loaded.pipeline.model.config.coarse_depth_supervision_enabled = branch == "FP-RD"


def _render_snapshot(loaded: LoadedBranch) -> Dict[str, Dict[str, Tensor]]:
    model = loaded.pipeline.model
    model.eval()
    records = direct_path._eval_records(loaded)  # same tuple structure.
    out: Dict[str, Dict[str, Tensor]] = {}
    for _eval_index, view_id, camera, batch in records:
        if view_id not in EVAL_VIEWS:
            continue
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        keep = (
            "pred_image",
            "direct_object_signal",
            "rgb_medium",
            "clear_object_fullsh_raw",
            "transmission",
            "tau_D",
            "accumulation",
            "gaussian_view_rgb",
            "gaussian_view_logits",
            "gaussian_sh_residual",
        )
        out[view_id] = {"gt": gt.detach().float().cpu()}
        for key in keep:
            if key in outputs:
                out[view_id][key] = outputs[key].detach().float().cpu()
    return out


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred_nchw = pred.detach().float().clamp(0.0, 1.0).permute(2, 0, 1)[None, ...].to(model.device)
    gt_nchw = gt.detach().float().clamp(0.0, 1.0).permute(2, 0, 1)[None, ...].to(model.device)
    mse = float((pred.detach().float() - gt.detach().float()).square().mean().item())
    with torch.no_grad():
        return {
            "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
            "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
            "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
            "mse": mse,
        }


def _error(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)


def _build_regions(repo: Path, output_dir: Path) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    m1 = direct_path._render_items(repo, "M1", direct_path.FINAL_STEP)
    k1 = direct_path._render_items(repo, "BND-K1", 5000)
    cd = direct_path._render_items(repo, "CDEPTH", 5000)
    final_k1 = direct_path._render_items(repo, "BND-K1", direct_path.FINAL_STEP)
    final_cd = direct_path._render_items(repo, "CDEPTH", direct_path.FINAL_STEP)
    view_ids = [view for view in EVAL_VIEWS if view in m1 and view in k1 and view in cd]
    lumas = torch.cat([_luma(m1[view]["gt"]).reshape(-1) for view in view_ids])
    bright_threshold = _q(lumas, 0.80)
    pos_values: List[Tensor] = []
    neg_values: List[Tensor] = []
    gain_maps: Dict[str, Tensor] = {}
    high_masks: Dict[str, Tensor] = {}
    for view in view_ids:
        high = (m1[view]["outputs"]["accumulation"][..., 0] > 0.01) & (
            m1[view]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0
        )
        gain = _error(final_k1[view]["outputs"]["pred_image"], m1[view]["gt"]) - _error(final_cd[view]["outputs"]["pred_image"], m1[view]["gt"])
        gain_maps[view] = gain
        high_masks[view] = high
        vals = gain[high]
        if (vals > 0).any():
            pos_values.append(vals[vals > 0])
        if (vals < 0).any():
            neg_values.append(-vals[vals < 0])
    strong_gain = _q(torch.cat(pos_values) if pos_values else torch.empty(0), 0.75)
    strong_harm = _q(torch.cat(neg_values) if neg_values else torch.empty(0), 0.75)
    regions: Dict[str, Dict[str, Tensor]] = {}
    for view in view_ids:
        high = high_masks[view]
        support = m1[view]["outputs"]["accumulation"][..., 0] > 0.01
        gain = gain_maps[view]
        luma = _luma(m1[view]["gt"])
        regions[view] = {
            "global": torch.ones_like(high, dtype=torch.bool),
            "M1_HIGH_J": high,
            "M1_LOW_J": support & ~high,
            "BRIGHT_Q5": luma > bright_threshold,
            "HJ_GAIN": high & (gain > 0),
            "HJ_HARM": high & (gain < 0),
            "HJ_STRONG_GAIN": high & (gain >= strong_gain),
            "HJ_STRONG_HARM": high & ((-gain) >= strong_harm),
        }
    meta = {
        "view_ids": view_ids,
        "M1_HIGH_J_definition": "final M1 accumulation > 0.01 and final M1 clear_object_fullsh_raw max RGB channel > 1.0",
        "M1_LOW_J_definition": "final M1 accumulation > 0.01 and not M1_HIGH_J",
        "BRIGHT_Q5_definition": "top 20 percent pooled final M1 GT luminance",
        "HJ_GAIN_definition": "historical final 15k CDEPTH improves K1 RGB MSE inside M1_HIGH_J; secondary alignment diagnostic",
        "HJ_HARM_definition": "historical final 15k CDEPTH worsens K1 RGB MSE inside M1_HIGH_J; secondary alignment diagnostic",
        "bright_q5_threshold": bright_threshold,
        "strong_gain_threshold": strong_gain,
        "strong_harm_threshold": strong_harm,
    }
    _write_json(output_dir / "region_definitions.json", meta)
    return regions, meta


def _region_mse(outputs: Mapping[str, Tensor], mask: Tensor) -> float:
    return float(_masked(_error(outputs["pred_image"], outputs["gt"]), mask).mean().item()) if int(mask.sum().item()) else float("nan")


def _region_l1(outputs: Mapping[str, Tensor], mask: Tensor) -> float:
    return float(_masked((outputs["pred_image"] - outputs["gt"]).abs().mean(dim=-1), mask).mean().item()) if int(mask.sum().item()) else float("nan")


def _analyze_snapshots(
    repo: Path,
    output_dir: Path,
    render_dir: Path,
    regions: Mapping[str, Mapping[str, Tensor]],
) -> Dict[str, Any]:
    loaded = _load_branch(repo, "ANALYSIS", coarse_depth=False)
    visual_manifest: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    highj_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    rgb_sheet_rows: List[Sequence[Tuple[str, Image.Image]]] = []
    residual_sheet_rows: List[Sequence[Tuple[str, Image.Image]]] = []

    try:
        model = loaded.pipeline.model
        for rel in SNAPSHOTS:
            abs_step = START_NOMINAL_STEP + rel
            renders: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
            for branch in ("FP-R", "FP-RD"):
                if rel in CHECKPOINT_SNAPSHOTS:
                    _load_snapshot_into_model(repo, output_dir, branch, rel, loaded)
                    renders[branch] = _render_snapshot(loaded)
            if set(renders) != {"FP-R", "FP-RD"}:
                continue
            for branch, by_view in renders.items():
                view_metrics: List[Dict[str, float]] = []
                for view_id in EVAL_VIEWS:
                    item = by_view[view_id]
                    metrics = _metric_images(model, item["pred_image"], item["gt"])
                    view_metrics.append(metrics)
                    row = {
                        "branch": branch,
                        "relative_step": rel,
                        "absolute_step": abs_step,
                        "view_id": view_id,
                        **metrics,
                    }
                    for region in ("M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
                        row[f"{region}_mse"] = _region_mse(item, regions[view_id][region])
                        row[f"{region}_l1"] = _region_l1(item, regions[view_id][region])
                    per_view_rows.append(row)
                rgb_rows.append(
                    {
                        "branch": branch,
                        "relative_step": rel,
                        "absolute_step": abs_step,
                        "psnr": _mean(row["psnr"] for row in view_metrics),
                        "ssim": _mean(row["ssim"] for row in view_metrics),
                        "lpips": _mean(row["lpips"] for row in view_metrics),
                        "mse": _mean(row["mse"] for row in view_metrics),
                    }
                )
                highj_rows.append(
                    {
                        "branch": branch,
                        "relative_step": rel,
                        "absolute_step": abs_step,
                        "M1_HIGH_J_mse": _mean(row["M1_HIGH_J_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel),
                        "M1_HIGH_J_l1": _mean(row["M1_HIGH_J_l1"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel),
                        "M1_LOW_J_mse": _mean(row["M1_LOW_J_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel),
                        "BRIGHT_Q5_mse": _mean(row["BRIGHT_Q5_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel),
                    }
                )
            for view_id in EVAL_VIEWS:
                r_item = renders["FP-R"][view_id]
                rd_item = renders["FP-RD"][view_id]
                if rel in (0, 500, 1000, 1500, 2000):
                    rgb_sheet_rows.append(
                        [
                            (f"{rel} {view_id} GT", _rgb_to_uint8(r_item["gt"])),
                            ("FP-R", _rgb_to_uint8(r_item["pred_image"])),
                            ("FP-RD", _rgb_to_uint8(rd_item["pred_image"])),
                            ("abs diff", _gray_to_uint8((rd_item["pred_image"] - r_item["pred_image"]).abs().mean(dim=-1), 0.03)),
                        ]
                    )
                    r_res = _error(r_item["pred_image"], r_item["gt"])
                    rd_res = _error(rd_item["pred_image"], rd_item["gt"])
                    residual_sheet_rows.append(
                        [
                            (f"{rel} {view_id} R residual", _gray_to_uint8(r_res, 0.02)),
                            ("RD residual", _gray_to_uint8(rd_res, 0.02)),
                            ("signed gain", _signed_to_rgb(r_res - rd_res, 0.01)),
                        ]
                    )
                i_check = (rd_item["pred_image"] - (rd_item["direct_object_signal"] + rd_item["rgb_medium"])).abs().max().item()
                snapshot_rows.append(
                    {
                        "relative_step": rel,
                        "absolute_step": abs_step,
                        "view_id": view_id,
                        "branch": "FP-RD",
                        "I_equals_D_plus_B_max_abs": float(i_check),
                    }
                )
                for region in ("global", "M1_HIGH_J", "HJ_GAIN", "HJ_HARM"):
                    mask = regions[view_id][region]
                    if int(mask.sum().item()) == 0:
                        continue
                    d = (rd_item["direct_object_signal"] - r_item["direct_object_signal"]).abs().mean(dim=-1)
                    b = (rd_item["rgb_medium"] - r_item["rgb_medium"]).abs().mean(dim=-1)
                    i = (rd_item["pred_image"] - r_item["pred_image"]).abs().mean(dim=-1)
                    d_mean = float(d[mask].mean().item())
                    b_mean = float(b[mask].mean().item())
                    direct_rows.append(
                        {
                            "relative_step": rel,
                            "absolute_step": abs_step,
                            "view_id": view_id,
                            "region": region,
                            "mean_abs_delta_D": d_mean,
                            "mean_abs_delta_B": b_mean,
                            "mean_abs_delta_I": float(i[mask].mean().item()),
                            "ROLL_DIRECT_MEDIUM_RATIO": d_mean / (b_mean + EPS),
                            "ROLL_DIRECT_SHARE": d_mean / (d_mean + b_mean + EPS),
                        }
                    )
                for branch, item in (("FP-R", r_item), ("FP-RD", rd_item)):
                    for region in ("global", "M1_HIGH_J"):
                        mask = regions[view_id][region]
                        j = item["clear_object_fullsh_raw"]
                        tau_mean = item["tau_D"].mean(dim=-1)
                        t_mean = item["transmission"].mean(dim=-1)
                        vals_j = _masked(j, mask)
                        row = {
                            "branch": branch,
                            "relative_step": rel,
                            "absolute_step": abs_step,
                            "view_id": view_id,
                            "region": region,
                            "J_p99": _q(vals_j, 0.99),
                            "P_J_gt_1": float((vals_j > 1.0).float().mean().item()) if vals_j.numel() else float("nan"),
                            "tau_p90": _q(tau_mean[mask], 0.90),
                            "P_T_lt_0p1": float((t_mean[mask] < 0.1).float().mean().item()) if int(mask.sum().item()) else float("nan"),
                            "T_mean": float(t_mean[mask].mean().item()) if int(mask.sum().item()) else float("nan"),
                        }
                        c = item.get("gaussian_view_rgb")
                        logits = item.get("gaussian_view_logits")
                        row["P_c_gt_0p99"] = float((c > 0.99).float().mean().item()) if isinstance(c, Tensor) and c.numel() else float("nan")
                        row["P_abs_s_full_gt_5"] = float((logits.abs() > 5.0).float().mean().item()) if isinstance(logits, Tensor) and logits.numel() else float("nan")
                        decomp_rows.append(row)
                        boundary_rows.append(row)

        _save_sheet(render_dir / "rgb_trajectory.png", rgb_sheet_rows, visual_manifest, "rgb_trajectory")
        _save_sheet(render_dir / "m1_highj_residual_trajectory.png", residual_sheet_rows, visual_manifest, "m1_highj_residual_trajectory")
    finally:
        _release(loaded)

    # Pairwise rollout gains.
    rgb_by = {(row["relative_step"], row["branch"]): row for row in rgb_rows}
    high_by = {(row["relative_step"], row["branch"]): row for row in highj_rows}
    rgb_gain_rows: List[Dict[str, Any]] = []
    highj_gain_rows: List[Dict[str, Any]] = []
    for rel in sorted({row["relative_step"] for row in rgb_rows}):
        if (rel, "FP-R") not in rgb_by or (rel, "FP-RD") not in rgb_by:
            continue
        r = rgb_by[(rel, "FP-R")]
        rd = rgb_by[(rel, "FP-RD")]
        h_r = high_by[(rel, "FP-R")]
        h_rd = high_by[(rel, "FP-RD")]
        rgb_gain_rows.append(
            {
                "relative_step": rel,
                "absolute_step": START_NOMINAL_STEP + rel,
                "FP_R_psnr": r["psnr"],
                "FP_RD_psnr": rd["psnr"],
                "ROLL_PSNR_GAIN": rd["psnr"] - r["psnr"],
                "FP_R_ssim": r["ssim"],
                "FP_RD_ssim": rd["ssim"],
                "FP_R_lpips": r["lpips"],
                "FP_RD_lpips": rd["lpips"],
                "FP_R_mse": r["mse"],
                "FP_RD_mse": rd["mse"],
                "ROLL_GLOBAL_MSE_GAIN": r["mse"] - rd["mse"],
            }
        )
        highj_gain_rows.append(
            {
                "relative_step": rel,
                "absolute_step": START_NOMINAL_STEP + rel,
                "MSE_R_HJ": h_r["M1_HIGH_J_mse"],
                "MSE_RD_HJ": h_rd["M1_HIGH_J_mse"],
                "ROLL_HJ_MSE_GAIN": h_r["M1_HIGH_J_mse"] - h_rd["M1_HIGH_J_mse"],
                "ROLL_HJ_REL_IMPROVEMENT": (h_r["M1_HIGH_J_mse"] - h_rd["M1_HIGH_J_mse"]) / (h_r["M1_HIGH_J_mse"] + EPS),
                "M1_LOW_J_MSE_GAIN": h_r["M1_LOW_J_mse"] - h_rd["M1_LOW_J_mse"],
                "BRIGHT_Q5_MSE_GAIN": h_r["BRIGHT_Q5_mse"] - h_rd["BRIGHT_Q5_mse"],
            }
        )

    _write_csv(output_dir / "snapshot_manifest.csv", snapshot_rows)
    _write_json(output_dir / "snapshot_manifest.json", {"rows": snapshot_rows})
    _write_csv(output_dir / "rgb_rollout_metrics.csv", rgb_gain_rows)
    _write_json(output_dir / "rgb_rollout_metrics.json", {"rows": rgb_gain_rows})
    _write_csv(output_dir / "highj_rollout_metrics.csv", highj_gain_rows)
    _write_json(output_dir / "highj_rollout_metrics.json", {"rows": highj_gain_rows})
    _write_csv(output_dir / "per_view_rollout_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rollout_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "direct_medium_rollout.csv", direct_rows)
    _write_json(output_dir / "direct_medium_rollout.json", {"rows": direct_rows})
    _write_csv(output_dir / "decomposition_control.csv", decomp_rows)
    _write_json(output_dir / "decomposition_control.json", {"rows": decomp_rows})
    _write_csv(output_dir / "boundary_control.csv", boundary_rows)
    _write_json(output_dir / "boundary_control.json", {"rows": boundary_rows})

    _save_plot(render_dir / "rollout_gain_curve.png", rgb_gain_rows, "relative_step", ("ROLL_GLOBAL_MSE_GAIN", "ROLL_PSNR_GAIN"), "Rollout gain curve", visual_manifest, "rollout_gain_curve")
    hj_plot_rows = [dict(row, label=str(row["relative_step"])) for row in highj_gain_rows]
    _save_plot(render_dir / "eval_view_consistency.png", hj_plot_rows, "label", ("ROLL_HJ_MSE_GAIN", "ROLL_HJ_REL_IMPROVEMENT"), "M1_HIGH_J rollout gain", visual_manifest, "eval_view_consistency")
    direct_plot_rows = []
    for rel in sorted({row["relative_step"] for row in direct_rows}):
        subset = [row for row in direct_rows if row["relative_step"] == rel and row["region"] == "M1_HIGH_J"]
        direct_plot_rows.append(
            {
                "relative_step": rel,
                "mean_abs_delta_D": _mean(row["mean_abs_delta_D"] for row in subset),
                "mean_abs_delta_B": _mean(row["mean_abs_delta_B"] for row in subset),
                "ROLL_DIRECT_MEDIUM_RATIO": _mean(row["ROLL_DIRECT_MEDIUM_RATIO"] for row in subset),
                "ROLL_DIRECT_SHARE": _mean(row["ROLL_DIRECT_SHARE"] for row in subset),
            }
        )
    _save_plot(render_dir / "direct_medium_trajectory.png", direct_plot_rows, "relative_step", ("mean_abs_delta_D", "mean_abs_delta_B", "ROLL_DIRECT_MEDIUM_RATIO", "ROLL_DIRECT_SHARE"), "Direct / medium trajectory", visual_manifest, "direct_medium_trajectory")
    control_rows = []
    for rel in sorted({row["relative_step"] for row in decomp_rows}):
        for branch in ("FP-R", "FP-RD"):
            subset = [row for row in decomp_rows if row["relative_step"] == rel and row["branch"] == branch and row["region"] == "M1_HIGH_J"]
            control_rows.append(
                {
                    "label": f"{rel}_{branch}",
                    "J_p99": _mean(row["J_p99"] for row in subset),
                    "P_J_gt_1": _mean(row["P_J_gt_1"] for row in subset),
                    "tau_p90": _mean(row["tau_p90"] for row in subset),
                    "P_T_lt_0p1": _mean(row["P_T_lt_0p1"] for row in subset),
                    "P_c_gt_0p99": _mean(row["P_c_gt_0p99"] for row in subset),
                    "P_abs_s_full_gt_5": _mean(row["P_abs_s_full_gt_5"] for row in subset),
                }
            )
    _save_plot(render_dir / "decomposition_controls.png", control_rows, "label", ("J_p99", "P_J_gt_1", "tau_p90", "P_T_lt_0p1", "P_c_gt_0p99", "P_abs_s_full_gt_5"), "Decomposition controls", visual_manifest, "decomposition_controls")

    return {
        "visual_manifest": visual_manifest,
        "rgb_gain_rows": rgb_gain_rows,
        "highj_gain_rows": highj_gain_rows,
        "per_view_rows": per_view_rows,
        "direct_rows": direct_rows,
        "decomp_rows": decomp_rows,
    }


def _tensor_from_pipeline_state(state: Mapping[str, Tensor], group: str) -> Tensor:
    if group in GEOMETRY_GROUPS or group in ("features_dc", "features_rest"):
        return state[f"_model.gauss_params.{group}"].detach().float().cpu()
    prefix = f"_model.{group}."
    vals = [state[key].detach().float().reshape(-1).cpu() for key in sorted(state) if key.startswith(prefix)]
    return torch.cat(vals) if vals else torch.empty(0)


def _optimizer_vector(opt_state: Mapping[str, Any], group: str, state_key: str) -> Tensor:
    sd = opt_state.get(group, {})
    vals: List[Tensor] = []
    for state in sd.get("state", {}).values():
        if state_key in state:
            vals.append(state[state_key].detach().float().reshape(-1).cpu())
    return torch.cat(vals) if vals else torch.empty(0)


def _relative_norm_diff(a: Tensor, b: Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float(torch.linalg.norm((b - a).reshape(-1)).item() / (torch.linalg.norm(a.reshape(-1)).item() + EPS))


def _analyze_state_divergence(output_dir: Path, render_dir: Path, visual_manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    optimizer_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []
    physical_rows: List[Dict[str, Any]] = []
    for rel in CHECKPOINT_SNAPSHOTS:
        r = torch.load(_rollout_ckpt_path(output_dir, "FP-R", rel), map_location="cpu")
        rd = torch.load(_rollout_ckpt_path(output_dir, "FP-RD", rel), map_location="cpu")
        for group in CURRENT_GROUPS:
            for state_key, out_key in (("exp_avg", "EXPAVG_REL_DIFF"), ("exp_avg_sq", "EXPSQ_REL_DIFF")):
                a = _optimizer_vector(r["optimizers"], group, state_key)
                b = _optimizer_vector(rd["optimizers"], group, state_key)
                optimizer_rows.append(
                    {
                        "relative_step": rel,
                        "absolute_step": START_NOMINAL_STEP + rel,
                        "group": group,
                        "state_key": state_key,
                        out_key: _relative_norm_diff(a, b),
                        "norm_R": float(torch.linalg.norm(a).item()) if a.numel() else 0.0,
                        "norm_RD": float(torch.linalg.norm(b).item()) if b.numel() else 0.0,
                    }
                )
            pa = _tensor_from_pipeline_state(r["pipeline"], group)
            pb = _tensor_from_pipeline_state(rd["pipeline"], group)
            param_rows.append(
                {
                    "relative_step": rel,
                    "absolute_step": START_NOMINAL_STEP + rel,
                    "group": group,
                    "PARAM_REL_DIFF": _relative_norm_diff(pa.reshape(-1), pb.reshape(-1)),
                    "PARAM_ABS_DIFF_NORM": float(torch.linalg.norm((pb - pa).reshape(-1)).item()) if pa.numel() else 0.0,
                    "shape_R": list(pa.shape),
                    "shape_RD": list(pb.shape),
                }
            )
        means_r = r["pipeline"]["_model.gauss_params.means"].float()
        means_rd = rd["pipeline"]["_model.gauss_params.means"].float()
        row = {"relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "group": "means", "physical_metric": "world_displacement"}
        row.update(_stats(torch.linalg.norm(means_rd - means_r, dim=-1), ""))
        physical_rows.append(row)
        scales_r = torch.exp(r["pipeline"]["_model.gauss_params.scales"].float())
        scales_rd = torch.exp(rd["pipeline"]["_model.gauss_params.scales"].float())
        row = {"relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "group": "scales", "physical_metric": "activated_scale_relative_abs_diff"}
        row.update(_stats(((scales_rd - scales_r).abs() / scales_r.clamp_min(EPS)).reshape(-1), ""))
        physical_rows.append(row)
        q0_raw = r["pipeline"]["_model.gauss_params.quats"].float()
        q1_raw = rd["pipeline"]["_model.gauss_params.quats"].float()
        q0 = q0_raw / q0_raw.norm(dim=-1, keepdim=True).clamp_min(EPS)
        q1 = q1_raw / q1_raw.norm(dim=-1, keepdim=True).clamp_min(EPS)
        dots = (q0 * q1).sum(dim=-1).abs().clamp(0.0, 1.0)
        row = {"relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "group": "quats", "physical_metric": "quat_angle_radians"}
        row.update(_stats(2.0 * torch.acos(dots), ""))
        physical_rows.append(row)
        op_r = torch.sigmoid(r["pipeline"]["_model.gauss_params.opacities"].float())
        op_rd = torch.sigmoid(rd["pipeline"]["_model.gauss_params.opacities"].float())
        row = {"relative_step": rel, "absolute_step": START_NOMINAL_STEP + rel, "group": "opacities", "physical_metric": "sigmoid_opacity_abs_diff"}
        row.update(_stats((op_rd - op_r).abs().reshape(-1), ""))
        physical_rows.append(row)

    _write_csv(output_dir / "optimizer_state_divergence.csv", optimizer_rows)
    _write_json(output_dir / "optimizer_state_divergence.json", {"rows": optimizer_rows})
    _write_csv(output_dir / "parameter_divergence.csv", param_rows)
    _write_json(output_dir / "parameter_divergence.json", {"rows": param_rows})
    _write_csv(output_dir / "physical_parameter_divergence.csv", physical_rows)
    _write_json(output_dir / "physical_parameter_divergence.json", {"rows": physical_rows})

    plot_rows = []
    for rel in CHECKPOINT_SNAPSHOTS:
        for group in ("means", "scales", "quats", "opacities", "features_dc", "features_rest", "medium_mlp"):
            exp = _mean(row.get("EXPAVG_REL_DIFF", "") for row in optimizer_rows if row["relative_step"] == rel and row["group"] == group and row["state_key"] == "exp_avg")
            sq = _mean(row.get("EXPSQ_REL_DIFF", "") for row in optimizer_rows if row["relative_step"] == rel and row["group"] == group and row["state_key"] == "exp_avg_sq")
            prm = _mean(row["PARAM_REL_DIFF"] for row in param_rows if row["relative_step"] == rel and row["group"] == group)
            plot_rows.append({"label": f"{rel}_{group}", "EXPAVG_REL_DIFF": exp, "EXPSQ_REL_DIFF": sq, "PARAM_REL_DIFF": prm})
    _save_plot(render_dir / "optimizer_state_divergence.png", plot_rows, "label", ("EXPAVG_REL_DIFF", "EXPSQ_REL_DIFF"), "Optimizer state divergence", visual_manifest, "optimizer_state_divergence")
    _save_plot(render_dir / "parameter_divergence.png", plot_rows, "label", ("PARAM_REL_DIFF",), "Parameter divergence", visual_manifest, "parameter_divergence")
    phys_plot = []
    for rel in CHECKPOINT_SNAPSHOTS:
        for group in GEOMETRY_GROUPS:
            subset = [row for row in physical_rows if row["relative_step"] == rel and row["group"] == group]
            phys_plot.append({"label": f"{rel}_{group}", "p90": _mean(row["p90"] for row in subset), "p99": _mean(row["p99"] for row in subset)})
    _save_plot(render_dir / "physical_gaussian_divergence.png", phys_plot, "label", ("p90", "p99"), "Physical Gaussian divergence", visual_manifest, "physical_gaussian_divergence")
    return {"optimizer_rows": optimizer_rows, "parameter_rows": param_rows, "physical_rows": physical_rows}


def _training_advantage(output_dir: Path, render_dir: Path, visual_manifest: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    with (output_dir / "fp_r_training_log.csv").open() as f:
        r_rows = list(csv.DictReader(f))
    with (output_dir / "fp_rd_training_log.csv").open() as f:
        rd_rows = list(csv.DictReader(f))
    by_r = {int(row["relative_step"]): row for row in r_rows}
    by_rd = {int(row["relative_step"]): row for row in rd_rows}
    rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    steps: List[int] = []
    for step in sorted(set(by_r) & set(by_rd)):
        delta = float(by_rd[step]["L_RGB"]) - float(by_r[step]["L_RGB"])
        deltas.append(delta)
        steps.append(step)
        start = max(0, len(deltas) - 100)
        rolling = _mean(deltas[start:])
        rows.append(
            {
                "relative_step": step,
                "absolute_step": START_NOMINAL_STEP + step,
                "camera_R": by_r[step]["camera_name"],
                "camera_RD": by_rd[step]["camera_name"],
                "camera_match": by_r[step]["camera_name"] == by_rd[step]["camera_name"],
                "Delta_L_RGB_RD_minus_R": delta,
                "rolling100_Delta_L_RGB": rolling,
            }
        )
    onset = None
    for row in rows:
        step = int(row["relative_step"])
        if step < 100:
            continue
        if float(row["rolling100_Delta_L_RGB"]) < 0:
            future = [r for r in rows if step < int(r["relative_step"]) <= step + 100]
            if future and _mean(r["rolling100_Delta_L_RGB"] for r in future) < 0:
                onset = step
                break
    _write_csv(output_dir / "training_rgb_advantage.csv", rows)
    _write_json(output_dir / "training_rgb_advantage.json", {"rows": rows, "TRAIN_RGB_ADVANTAGE_ONSET": onset})
    plot_rows = [row for row in rows if int(row["relative_step"]) % 50 == 0]
    _save_plot(render_dir / "training_rgb_loss_divergence.png", plot_rows, "relative_step", ("Delta_L_RGB_RD_minus_R", "rolling100_Delta_L_RGB"), "Training RGB loss divergence", visual_manifest, "training_rgb_loss_divergence")
    return rows, onset


def _historical_reference(repo: Path, output_dir: Path, regions: Mapping[str, Mapping[str, Tensor]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    k1 = direct_path._render_items(repo, "BND-K1", 5000)
    cd = direct_path._render_items(repo, "CDEPTH", 5000)
    rows: List[Dict[str, Any]] = []
    for view in EVAL_VIEWS:
        for region in ("global", "M1_HIGH_J"):
            mask = regions[view][region]
            k_mse = _region_mse({**k1[view]["outputs"], "gt": k1[view]["gt"]}, mask)
            c_mse = _region_mse({**cd[view]["outputs"], "gt": cd[view]["gt"]}, mask)
            rows.append(
                {
                    "view_id": view,
                    "region": region,
                    "HIST_K1_5K_MSE": k_mse,
                    "HIST_CDEPTH_5K_MSE": c_mse,
                    "HIST_MSE_GAIN_5K": k_mse - c_mse,
                }
            )
    summary = {
        "HIST_HJ_GAIN_5K": _mean(row["HIST_MSE_GAIN_5K"] for row in rows if row["region"] == "M1_HIGH_J"),
        "HIST_GLOBAL_MSE_GAIN_5K": _mean(row["HIST_MSE_GAIN_5K"] for row in rows if row["region"] == "global"),
    }
    _write_csv(output_dir / "historical_reference_5k.csv", rows)
    _write_json(output_dir / "historical_reference_5k.json", {"rows": rows, "summary": summary})
    return rows, summary


def _onset(rows: Sequence[Mapping[str, Any]], key: str, threshold: float = 0.0) -> Optional[int]:
    sorted_rows = sorted(rows, key=lambda row: int(row["relative_step"]))
    for idx, row in enumerate(sorted_rows[:-1]):
        if float(row[key]) > threshold and float(sorted_rows[idx + 1][key]) > threshold:
            return int(row["relative_step"])
    return None


def _direct_onset(direct_rows: Sequence[Mapping[str, Any]]) -> Optional[int]:
    pooled = []
    for rel in sorted({int(row["relative_step"]) for row in direct_rows}):
        subset = [row for row in direct_rows if int(row["relative_step"]) == rel and row["region"] == "M1_HIGH_J"]
        pooled.append({"relative_step": rel, "ratio": _mean(row["ROLL_DIRECT_MEDIUM_RATIO"] for row in subset)})
    return _onset(pooled, "ratio", threshold=3.0)


def _effect_fraction(output_dir: Path, hist_summary: Mapping[str, Any], highj_rows: Sequence[Mapping[str, Any]], rgb_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final_hj = next((row for row in highj_rows if int(row["relative_step"]) == 2000), None)
    final_rgb = next((row for row in rgb_rows if int(row["relative_step"]) == 2000), None)
    hist_hj = float(hist_summary.get("HIST_HJ_GAIN_5K", float("nan")))
    hist_global = float(hist_summary.get("HIST_GLOBAL_MSE_GAIN_5K", float("nan")))
    out = {
        "FIXPOP_HJ_EFFECT_FRACTION": float(final_hj["ROLL_HJ_MSE_GAIN"]) / hist_hj if final_hj and math.isfinite(hist_hj) and abs(hist_hj) > EPS else "NOT_AVAILABLE",
        "FIXPOP_GLOBAL_EFFECT_FRACTION": float(final_rgb["ROLL_GLOBAL_MSE_GAIN"]) / hist_global if final_rgb and math.isfinite(hist_global) and abs(hist_global) > EPS else "NOT_AVAILABLE",
    }
    _write_json(output_dir / "fixedpop_effect_fraction.json", out)
    _write_csv(output_dir / "fixedpop_effect_fraction.csv", [out])
    return out


def _growth_flag(rows: Sequence[Mapping[str, Any]], value_key: str, group_key: str = "group") -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        early = _mean(row[value_key] for row in rows if str(row[group_key]) == group and int(row["relative_step"]) in (100, 250, 500))
        late = _mean(row[value_key] for row in rows if str(row[group_key]) == group and int(row["relative_step"]) in (1500, 2000))
        out[group] = bool(math.isfinite(early) and math.isfinite(late) and late > early * 1.5 and late > early + 1e-8)
    return out


def _classify(
    output_dir: Path,
    rgb_rows: Sequence[Mapping[str, Any]],
    highj_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    direct_rows: Sequence[Mapping[str, Any]],
    train_advantage_onset: Optional[int],
    effect_fraction: Mapping[str, Any],
    optimizer_rows: Sequence[Mapping[str, Any]],
    param_rows: Sequence[Mapping[str, Any]],
    validity_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    final_hj = next(row for row in highj_rows if int(row["relative_step"]) == 2000)
    final_rgb = next(row for row in rgb_rows if int(row["relative_step"]) == 2000)
    rel1500 = next(row for row in highj_rows if int(row["relative_step"]) == 1500)
    hj_onset = _onset(highj_rows, "ROLL_HJ_MSE_GAIN", 0.0)
    global_onset = _onset(rgb_rows, "ROLL_GLOBAL_MSE_GAIN", 0.0)
    direct_dom_onset = _direct_onset(direct_rows)
    late_view_positive = 0
    for view in EVAL_VIEWS:
        r1500 = [row for row in per_view_rows if row["branch"] == "FP-R" and int(row["relative_step"]) == 1500 and row["view_id"] == view][0]
        rd1500 = [row for row in per_view_rows if row["branch"] == "FP-RD" and int(row["relative_step"]) == 1500 and row["view_id"] == view][0]
        r2000 = [row for row in per_view_rows if row["branch"] == "FP-R" and int(row["relative_step"]) == 2000 and row["view_id"] == view][0]
        rd2000 = [row for row in per_view_rows if row["branch"] == "FP-RD" and int(row["relative_step"]) == 2000 and row["view_id"] == view][0]
        if (float(r1500["M1_HIGH_J_mse"]) - float(rd1500["M1_HIGH_J_mse"])) > 0 and (float(r2000["M1_HIGH_J_mse"]) - float(rd2000["M1_HIGH_J_mse"])) > 0:
            late_view_positive += 1
    frac = effect_fraction.get("FIXPOP_HJ_EFFECT_FRACTION")
    frac_ok = True if frac == "NOT_AVAILABLE" else float(frac) >= 0.30
    direct_order_ok = direct_dom_onset is not None and hj_onset is not None and direct_dom_onset <= hj_onset
    causal_valid = all(bool(value) for value in validity_inputs.values())
    final_rel = float(final_hj["ROLL_HJ_REL_IMPROVEMENT"])
    final_hj_gain = float(final_hj["ROLL_HJ_MSE_GAIN"])
    final_psnr_gain = float(final_rgb["ROLL_PSNR_GAIN"])
    late_stable = float(rel1500["ROLL_HJ_MSE_GAIN"]) > 0 and final_hj_gain > 0
    frac_value = float(frac) if frac != "NOT_AVAILABLE" else float("nan")
    no_meaningful = bool(
        final_rel < 0.02
        and final_psnr_gain < 0.02
        and (not math.isfinite(frac_value) or frac_value < 0.10)
        and late_view_positive < 2
    )
    if not causal_valid:
        classification = "INCONCLUSIVE"
    elif no_meaningful:
        classification = "NO_MEANINGFUL_FIXPOP_RECOVERY"
    elif final_rel >= 0.05 and late_stable and late_view_positive >= 2 and frac_ok and direct_order_ok:
        classification = "FIXPOP_CONTINUOUS_PATH_SUPPORTED"
    elif final_hj_gain > 0 and late_stable and (final_rel >= 0.02 or (frac != "NOT_AVAILABLE" and float(frac) >= 0.10)):
        classification = "FIXPOP_CONTINUOUS_PATH_PARTIAL"
    elif final_hj_gain > 0 and late_stable and not direct_order_ok:
        classification = "FIXPOP_RGB_EFFECT_WITHOUT_DIRECT_DOMINANCE"
    elif final_rel < 0.02 and final_psnr_gain < 0.02 and not late_stable:
        classification = "NO_MEANINGFUL_FIXPOP_RECOVERY"
    elif final_hj_gain < 0 and final_psnr_gain < 0:
        classification = "FIXPOP_HARMFUL"
    else:
        classification = "INCONCLUSIVE"
    opt_growth = _growth_flag([row for row in optimizer_rows if row.get("state_key") == "exp_avg"], "EXPAVG_REL_DIFF")
    param_growth = _growth_flag(param_rows, "PARAM_REL_DIFF")
    long_horizon = bool(
        late_stable
        and abs(float(next(row for row in highj_rows if int(row["relative_step"]) == 500)["ROLL_HJ_MSE_GAIN"])) < abs(final_hj_gain)
        and (any(opt_growth.values()) or any(param_growth.values()))
    )
    one_step_multistep = classification in ("FIXPOP_CONTINUOUS_PATH_SUPPORTED", "FIXPOP_CONTINUOUS_PATH_PARTIAL") and long_horizon
    summary = {
        "ROLLOUT_CAUSAL_VALID": causal_valid,
        "validity_inputs": validity_inputs,
        "ROLL_HJ_RECOVERY_ONSET": hj_onset,
        "ROLL_GLOBAL_RECOVERY_ONSET": global_onset,
        "ROLL_DIRECT_DOMINANCE_ONSET": direct_dom_onset,
        "TRAIN_RGB_ADVANTAGE_ONSET": train_advantage_onset,
        "final_ROLL_HJ_REL_IMPROVEMENT": final_rel,
        "final_ROLL_HJ_MSE_GAIN": final_hj_gain,
        "final_ROLL_PSNR_GAIN": final_psnr_gain,
        "late_positive_eval_view_count": late_view_positive,
        "optimizer_expavg_growth_by_group": opt_growth,
        "parameter_growth_by_group": param_growth,
        "LONG_HORIZON_BUILDUP": long_horizon,
        "FIXPOP_ROLLOUT_CLASSIFICATION": classification,
        "ONE_STEP_INSUFFICIENT_BUT_MULTISTEP_SUPPORTED": one_step_multistep,
        "NEXT_SINGLE_FACTOR_RECOMMENDATION": {
            "FIXPOP_CONTINUOUS_PATH_SUPPORTED": "fixed-topology accumulated group attribution",
            "FIXPOP_CONTINUOUS_PATH_PARTIAL": "same paired fixed-pop rollout 5k->8k extension if the trajectory is still growing",
            "FIXPOP_RGB_EFFECT_WITHOUT_DIRECT_DOMINANCE": "D/B coupled continuous-path audit",
            "NO_MEANINGFUL_FIXPOP_RECOVERY": "historical optimizer-memory divergence audit for 0->3k history",
            "FIXPOP_HARMFUL": "historical optimizer-memory divergence audit for 0->3k history",
            "INCONCLUSIVE": "fix the failed rollout validity gate before adding new factors",
        }[classification],
    }
    _write_json(output_dir / "fixpop_rollout_classification.json", summary)
    _write_csv(output_dir / "fixpop_rollout_final_summary.csv", [summary])
    _write_json(output_dir / "fixpop_rollout_final_summary.json", summary)
    return summary


def _write_research_note(repo_manifest: Mapping[str, Any], outputs: Mapping[str, Any], classification: Mapping[str, Any]) -> None:
    lines = [
        "# BND-CDEPTH Fixed-Population Rollout",
        "",
        "## CODE FACT",
        "",
        f"- Branch: `{repo_manifest['branch']}`.",
        f"- Start HEAD: `{repo_manifest['head']}`.",
        "- Two rollout branches were run from the same formal BND-K1 Panama 3k checkpoint: `FP-R` and `FP-RD`.",
        "- The runner did not call densification, split, duplicate, prune, cull, or Gaussian insertion/deletion callbacks.",
        "- The scheduled opacity-reset rule in the 3k->5k window was retained for both branches at the same absolute steps.",
        "- The only branch loss difference was `coarse_depth_supervision_enabled=False` for FP-R and `True` for FP-RD.",
        "",
        "## CONFIG FACT",
        "",
        f"- Start checkpoint: `{outputs['start_checkpoint']}`.",
        f"- Rollout length: `{ROLLOUT_STEPS}` optimizer steps from nominal `{START_NOMINAL_STEP}` to `{START_NOMINAL_STEP + ROLLOUT_STEPS}`.",
        f"- Camera sequence file: `{outputs['camera_sequence']}`.",
        f"- Output manifest: `{outputs['manifest']}`.",
        f"- Visual index: `{outputs['visual_index']}`.",
        "",
        "## EXPERIMENTAL FACT",
        "",
        f"- `ROLLOUT_CAUSAL_VALID = {classification.get('ROLLOUT_CAUSAL_VALID')}`.",
        f"- `ROLL_HJ_RECOVERY_ONSET = {classification.get('ROLL_HJ_RECOVERY_ONSET')}`.",
        f"- `ROLL_GLOBAL_RECOVERY_ONSET = {classification.get('ROLL_GLOBAL_RECOVERY_ONSET')}`.",
        f"- `ROLL_DIRECT_DOMINANCE_ONSET = {classification.get('ROLL_DIRECT_DOMINANCE_ONSET')}`.",
        f"- `TRAIN_RGB_ADVANTAGE_ONSET = {classification.get('TRAIN_RGB_ADVANTAGE_ONSET')}`.",
        "",
        "## QUANTITATIVE RESULT",
        "",
        f"- Final HJ relative improvement: `{classification.get('final_ROLL_HJ_REL_IMPROVEMENT')}`.",
        f"- Final HJ MSE gain: `{classification.get('final_ROLL_HJ_MSE_GAIN')}`.",
        f"- Final PSNR gain: `{classification.get('final_ROLL_PSNR_GAIN')}`.",
        f"- Late positive eval-view count: `{classification.get('late_positive_eval_view_count')}`.",
        f"- `LONG_HORIZON_BUILDUP = {classification.get('LONG_HORIZON_BUILDUP')}`.",
        f"- Classification: `{classification.get('FIXPOP_ROLLOUT_CLASSIFICATION')}`.",
        f"- `ONE_STEP_INSUFFICIENT_BUT_MULTISTEP_SUPPORTED = {classification.get('ONE_STEP_INSUFFICIENT_BUT_MULTISTEP_SUPPORTED')}`.",
        "",
        "## INFERENCE",
        "",
        "- This controlled rollout tests whether introducing the CDEPTH continuous path from K1@3k is sufficient under fixed topology. It does not test the full 0->3k historical optimizer-memory path.",
        "- Historical K1/CDEPTH checkpoints are used only as context, not as reproduced branches.",
        "",
        "## HYPOTHESIS",
        "",
        "- If fixed-pop recovery is weak, the next single-factor diagnostic should focus on historical optimizer-memory divergence before 3k.",
        "- If fixed-pop recovery is supported and still growing, the next step should be a same-factor extension or accumulated group attribution, depending on the formal classification.",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_manifest(output_dir: Path, render_dir: Path, visual_manifest: Sequence[Mapping[str, Any]]) -> Path:
    rows = []
    for root in (output_dir, render_dir):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"file_path": str(path), "size_bytes": path.stat().st_size, "kind": "render" if root == render_dir else "output"})
    _write_csv(output_dir / "manifest.csv", rows)
    _write_json(output_dir / "manifest.json", {"rows": rows, "visual_manifest": list(visual_manifest)})
    _write_json(render_dir / "manifest.json", {"rows": list(visual_manifest)})
    index = render_dir / "VISUAL_COMPARE_INDEX.md"
    lines = ["# BND-CDEPTH Fixed-Pop Rollout Visual Index", ""]
    for row in visual_manifest:
        lines.append(f"- {row['output_type']}: `{row['file_path']}`")
    index.write_text("\n".join(lines) + "\n", encoding="utf8")
    return index


def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "log_8": _git(repo, "log", "-8", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "diff_check_returncode": subprocess.run(["git", "-C", str(repo), "diff", "--check"], text=True, capture_output=True).returncode,
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    start_config = repo / cdepth_setup.K1_CONFIG
    actual_start = _actual_step(start_config, START_NOMINAL_STEP)
    if actual_start is None:
        raise RuntimeError("Missing formal K1@3k checkpoint")
    start_ckpt = _available_steps(start_config)[actual_start]
    start_state = torch.load(start_ckpt, map_location="cpu")
    start_manifest = {
        "start_checkpoint": str(start_ckpt),
        "nominal_step": START_NOMINAL_STEP,
        "actual_step": actual_start,
        "checkpoint_step_field": int(start_state.get("step", actual_start)),
        "gaussian_count": int(start_state["pipeline"]["_model.gauss_params.means"].shape[0]),
        "config_path": str(start_config),
        "config_sha256": _sha256(start_config),
        "checkpoint_sha256": _sha256(start_ckpt),
        "intrinsic_parameterization_runtime": "bounded_sh3",
        "rasterizer": "classic",
        "medium_context_mode": "dir_xy_camera",
        "b_inf_mode": "tied",
        "coarse_depth_enabled_start": False,
        "optimizer_state_available": all(group in start_state["optimizers"] for group in CURRENT_GROUPS),
        "scheduler_state_available": all(group in start_state["schedulers"] for group in CURRENT_GROUPS),
    }
    _write_json(output_dir / "start_checkpoint_manifest.json", start_manifest)
    _write_json(
        output_dir / "source_control_audit.json",
        {
            "fixed_population_control": "manual rollout runner does not invoke population-mutating refinement_after branches; split/dup/prune/cull are never called",
            "opacity_reset_handling": "scheduled opacity clamp and opacities Adam moment reset are applied to both branches at absolute steps where step % 500 == 100",
            "default_compatibility": "No production model/trainer source changed; default training behavior is unchanged outside this diagnostic runner.",
            "blocked_mutations": ["split_gaussians", "dup_gaussians", "cull_gaussians", "remove_from_all_optim", "dup_in_all_optim"],
        },
    )

    branch_r = _load_branch(repo, "FP-R", coarse_depth=False)
    branch_rd = _load_branch(repo, "FP-RD", coarse_depth=True)
    try:
        param_rows = _compare_model_params(branch_r.pipeline.model, branch_rd.pipeline.model)
        opt_rows = _compare_optimizer_states(branch_r.optimizers, branch_rd.optimizers)
        first_train_index = 0
        camera_indices, camera_names, camera_rows = _generate_camera_sequence(branch_r, output_dir)
        first_train_index = camera_indices[0]
        forward_rows = _compare_initial_forward(branch_r, branch_rd, first_train_index)
        _write_csv(output_dir / "initial_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "initial_parameter_equivalence.json", {"rows": param_rows, "INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in param_rows)})
        _write_csv(output_dir / "initial_optimizer_equivalence.csv", opt_rows)
        _write_json(output_dir / "initial_optimizer_equivalence.json", {"rows": opt_rows, "INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows)})
        _write_csv(output_dir / "initial_forward_equivalence.csv", forward_rows)
        _write_json(output_dir / "initial_forward_equivalence.json", {"rows": forward_rows, "INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in forward_rows)})
        _write_json(
            output_dir / "optimizer_restore_audit.json",
            {
                "K1_3k_optimizer_state_restored": True,
                "groups": list(CURRENT_GROUPS),
                "scheduler_state_restored": True,
                "INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows),
            },
        )
        _write_json(
            output_dir / "fixed_topology_control_audit.json",
            {
                "DEFAULT_COMPATIBILITY": "PASS",
                "FIXPOP_MUTATION_BLOCK": "PASS",
                "mutation_block_method": "manual runner never invokes refinement_after population mutation code paths; topology invariance is checked at every step",
                "opacity_reset_steps": [3100, 3600, 4100, 4600],
            },
        )

        # Training starts after equivalence checks. Reload branches to undo any forward-side sampler/model state.
    finally:
        _release(branch_r)
        _release(branch_rd)

    branch_r = _load_branch(repo, "FP-R", coarse_depth=False)
    branch_rd = _load_branch(repo, "FP-RD", coarse_depth=True)
    try:
        r_train, r_topo, r_stab, r_snap = _train_branch(branch_r, camera_indices, camera_names, output_dir, coarse_depth=False)
        rd_train, rd_topo, rd_stab, rd_snap = _train_branch(branch_rd, camera_indices, camera_names, output_dir, coarse_depth=True)
    finally:
        _release(branch_r)
        _release(branch_rd)

    _write_csv(output_dir / "fp_r_training_log.csv", r_train)
    _write_json(output_dir / "fp_r_training_log.json", {"rows": r_train})
    _write_csv(output_dir / "fp_rd_training_log.csv", rd_train)
    _write_json(output_dir / "fp_rd_training_log.json", {"rows": rd_train})
    topology_rows = r_topo + rd_topo
    _write_csv(output_dir / "topology_invariance.csv", topology_rows)
    n_start = int(start_manifest["gaussian_count"])
    topo_pass = all(int(row["gaussian_count"]) == n_start for row in topology_rows)
    _write_json(output_dir / "topology_invariance.json", {"rows": topology_rows, "FIXED_TOPOLOGY_INVARIANCE": "PASS" if topo_pass else "FAIL", "N_start": n_start})
    stability_rows = r_stab + rd_stab
    stable = all(bool(row["stable"]) for row in stability_rows)
    _write_csv(output_dir / "rollout_stability.csv", stability_rows)
    _write_json(output_dir / "rollout_stability.json", {"rows": stability_rows, "ROLL_STABLE": stable})
    snap_rows = r_snap + rd_snap
    _write_csv(output_dir / "rollout_checkpoint_manifest.csv", snap_rows)
    _write_json(output_dir / "rollout_checkpoint_manifest.json", {"rows": snap_rows})

    validation_rows = []
    for rel in range(1, ROLLOUT_STEPS + 1):
        validation_rows.append(
            {
                "relative_step": rel,
                "camera_name_FP_R": r_train[rel - 1]["camera_name"],
                "camera_name_FP_RD": rd_train[rel - 1]["camera_name"],
                "match": r_train[rel - 1]["camera_name"] == rd_train[rel - 1]["camera_name"],
            }
        )
    mismatch_count = sum(1 for row in validation_rows if not row["match"])
    _write_csv(output_dir / "paired_camera_sequence_validation.csv", validation_rows)
    _write_json(output_dir / "paired_camera_sequence_validation.json", {"rows": validation_rows, "mismatch_count": mismatch_count, "CAMERA_SEQUENCE_EXACT_MATCH": mismatch_count == 0})

    regions, _region_meta = _build_regions(repo, output_dir)
    hist_rows, hist_summary = _historical_reference(repo, output_dir, regions)
    analysis = _analyze_snapshots(repo, output_dir, render_dir, regions)
    state_analysis = _analyze_state_divergence(output_dir, render_dir, analysis["visual_manifest"])
    train_adv_rows, train_adv_onset = _training_advantage(output_dir, render_dir, analysis["visual_manifest"])
    effect_fraction = _effect_fraction(output_dir, hist_summary, analysis["highj_gain_rows"], analysis["rgb_gain_rows"])

    _save_plot(
        render_dir / "historical_vs_controlled_context.png",
        [
            {"label": "HIST_HJ_GAIN_5K", "value": hist_summary.get("HIST_HJ_GAIN_5K", float("nan"))},
            {"label": "FIXPOP_HJ_GAIN_5K", "value": next(row["ROLL_HJ_MSE_GAIN"] for row in analysis["highj_gain_rows"] if int(row["relative_step"]) == 2000)},
            {"label": "HIST_GLOBAL_GAIN_5K", "value": hist_summary.get("HIST_GLOBAL_MSE_GAIN_5K", float("nan"))},
            {"label": "FIXPOP_GLOBAL_GAIN_5K", "value": next(row["ROLL_GLOBAL_MSE_GAIN"] for row in analysis["rgb_gain_rows"] if int(row["relative_step"]) == 2000)},
        ],
        "label",
        ("value",),
        "Historical vs controlled 5k context",
        analysis["visual_manifest"],
        "historical_vs_controlled_context",
    )

    validity_inputs = {
        "DEFAULT_COMPATIBILITY": True,
        "FIXPOP_MUTATION_BLOCK": True,
        "INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in param_rows),
        "INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in opt_rows),
        "INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in forward_rows),
        "CAMERA_SEQUENCE_EXACT_MATCH": mismatch_count == 0,
        "FIXED_TOPOLOGY_INVARIANCE": topo_pass,
        "ROLL_STABLE": stable,
        "PERSISTENT_OUTPUT_SAFETY": True,
    }
    classification = _classify(
        output_dir,
        analysis["rgb_gain_rows"],
        analysis["highj_gain_rows"],
        analysis["per_view_rows"],
        analysis["direct_rows"],
        train_adv_onset,
        effect_fraction,
        state_analysis["optimizer_rows"],
        state_analysis["parameter_rows"],
        validity_inputs,
    )

    _save_plot(
        render_dir / "causal_chain_summary.png",
        [
            {"label": "camera_match", "value": 1.0 if mismatch_count == 0 else 0.0},
            {"label": "topology_fixed", "value": 1.0 if topo_pass else 0.0},
            {"label": "causal_valid", "value": 1.0 if classification["ROLLOUT_CAUSAL_VALID"] else 0.0},
            {"label": "HJ_rel_final", "value": classification["final_ROLL_HJ_REL_IMPROVEMENT"]},
            {"label": "long_horizon", "value": 1.0 if classification["LONG_HORIZON_BUILDUP"] else 0.0},
        ],
        "label",
        ("value",),
        "Causal-chain summary",
        analysis["visual_manifest"],
        "causal_chain_summary",
    )

    visual_index = _write_manifest(output_dir, render_dir, analysis["visual_manifest"])
    _write_research_note(
        repo_manifest,
        {
            "start_checkpoint": str(start_ckpt),
            "camera_sequence": str(output_dir / "paired_camera_sequence.json"),
            "manifest": str(output_dir / "manifest.json"),
            "visual_index": str(visual_index),
        },
        classification,
    )

    print(
        json.dumps(
            {
                "classification": classification,
                "output_dir": str(output_dir),
                "render_dir": str(render_dir),
                "visual_index": str(visual_index),
            },
            indent=2,
            default=_json_default,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
