#!/usr/bin/env python
"""Read-only CDEPTH densification-trigger response audit for Panama.

This diagnostic loads existing BND-K1 / BND-CDEPTH checkpoints, reconstructs
the WaterSplatting densification eligibility statistic on a fixed training
camera bank, and compares RGB-only loss against RGB + SeaFree-style coarse
depth loss. It never calls optimizer.step(), scheduler.step(), densification,
split, duplicate, prune, opacity reset, or checkpoint writes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
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

from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup
from scripts.diagnostics import audit_bnd_cdepth_optimization_path as path_audit


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_cdepth_densify_trigger_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_densify_trigger_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_densify_trigger_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_DENSIFICATION_TRIGGER_AUDIT_2026-08-11.md")
TARGET_STEPS = (1000, 3000, 5000)
CONTROL_STEPS = (8000,)
FINAL_STEP = 15000
EPS = 1e-8


@dataclass(frozen=True)
class RunSpec:
    run: str
    config_relpath: Path
    intrinsic_color_parameterization: str = "bounded_sh3"
    rasterize_mode: str = "classic"


@dataclass
class LoadedRun:
    spec: RunSpec
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any


RUNS: Dict[str, RunSpec] = {
    "BND-K1": RunSpec("BND-K1", cdepth_setup.K1_CONFIG),
    "CDEPTH": RunSpec(
        "CDEPTH",
        Path(
            "outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/"
            "water-splatting/20260811_bnd_cdepth/config.yml"
        ),
    ),
}

EVAL_OUTCOME_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
REGION_NAMES = ("M1_HIGH_J", "HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM")
GROUP_NAMES = ("DEPTH_ADDED", "DEPTH_REMOVED", "UNCHANGED_ELIGIBLE", "UNCHANGED_INELIGIBLE")


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


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    if not steps:
        return None
    nearest = min(steps, key=lambda step: abs(step - nominal_step))
    if abs(nearest - nominal_step) <= 1:
        return nearest
    return None


def _release(loaded: Optional[LoadedRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_run(repo: Path, run: str, nominal_step: int, *, load_depths: bool = True) -> LoadedRun:
    spec = RUNS[run]
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"{run} missing checkpoint for nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        config.pipeline.model.intrinsic_color_parameterization = spec.intrinsic_color_parameterization
        config.pipeline.model.rasterize_mode = spec.rasterize_mode
        config.pipeline.model.medium_context_mode = "dir_xy_camera"
        config.pipeline.model.b_inf_mode = "tied"
        config.pipeline.model.infinite_water_enabled = False
        config.pipeline.model.coarse_depth_supervision_enabled = False
        config.pipeline.model.coarse_depth_supervision_weight = 0.1
        config.pipeline.datamanager.load_depths = bool(load_depths)
        if load_depths:
            config.pipeline.datamanager.dataparser.depths_path = cdepth_setup.DEPTHS_PATH
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    model = pipeline.model
    model.config.intrinsic_color_parameterization = spec.intrinsic_color_parameterization
    model.config.rasterize_mode = spec.rasterize_mode
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.coarse_depth_supervision_weight = 0.1
    pipeline.eval()
    return LoadedRun(spec, config_path, checkpoint_path, int(loaded_step), config, pipeline)


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
    keys = ("count", "mean", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{key}": float("nan") for key in keys}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p10": _q(flat, 0.10),
        f"{prefix}p25": _q(flat, 0.25),
        f"{prefix}p50": _q(flat, 0.50),
        f"{prefix}p75": _q(flat, 0.75),
        f"{prefix}p90": _q(flat, 0.90),
        f"{prefix}p95": _q(flat, 0.95),
        f"{prefix}p99": _q(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _masked_stats(values: Tensor, mask: Tensor, prefix: str = "") -> Dict[str, Any]:
    m = mask.detach().bool().reshape(-1)
    vals = values.detach().float().reshape(-1)
    n = min(vals.numel(), m.numel())
    return _stats(vals[:n][m[:n]], prefix)


def _parameter_snapshot(model: Any) -> Dict[str, Tensor]:
    out = {
        "means": model.means.detach().cpu().clone(),
        "scales": model.scales.detach().cpu().clone(),
        "quats": model.quats.detach().cpu().clone(),
        "opacities": model.opacities.detach().cpu().clone(),
        "features_dc": model.features_dc.detach().cpu().clone(),
        "features_rest": model.features_rest.detach().cpu().clone(),
    }
    for name, param in model.medium_mlp.named_parameters():
        out[f"medium_mlp.{name}"] = param.detach().cpu().clone()
    for name, param in model.direction_encoding.named_parameters():
        out[f"direction_encoding.{name}"] = param.detach().cpu().clone()
    return out


def _parameter_delta_rows(before: Mapping[str, Tensor], model: Any, run: str, nominal_step: int) -> List[Dict[str, Any]]:
    after = _parameter_snapshot(model)
    rows: List[Dict[str, Any]] = []
    for name, tensor in before.items():
        diff = (after[name] - tensor).abs()
        rows.append(
            {
                "scene": SCENE,
                "run": run,
                "nominal_step": nominal_step,
                "parameter": name,
                "shape": list(tensor.shape),
                "max_abs_delta": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs_delta": float(diff.mean().item()) if diff.numel() else 0.0,
            }
        )
    return rows


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _train_records(pipeline: Any, count: int = 0) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    total = len(dataset)
    if count <= 0 or count >= total:
        indices = list(range(total))
        selection_rule = "all training views in dataset order"
    else:
        indices = np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
        selection_rule = f"linspace over training dataset, count={len(indices)}"
    cameras = dataset.cameras.to(pipeline.model.device)
    rows: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    for index in indices:
        filename = image_filenames[index] if index < len(image_filenames) else Path(f"train_{index}")
        batch = pipeline.datamanager.cached_train[index].copy()
        batch = _batch_to_device(batch, pipeline.model.device)
        batch["_fixed_bank_selection_rule"] = selection_rule
        rows.append((index, Path(filename).stem, cameras[index : index + 1], batch))
    return rows


def _eval_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Any, Mapping[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    vals = torch.nan_to_num(image.detach().float(), nan=0.0).clamp(0.0, 1.0)
    arr = (vals * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: Optional[float] = None) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float().cpu(), nan=0.0)
    if scale is None:
        scale = float(vals.max().item()) if vals.numel() else 1.0
    scale = max(float(scale), EPS)
    arr = (vals.clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _label_tile(image: Image.Image, label: str, tile_width: int = 290) -> Image.Image:
    ratio = tile_width / max(image.width, 1)
    resized = image.resize((tile_width, max(1, int(round(image.height * ratio)))), Image.BILINEAR)
    label_h = 30
    canvas = Image.new("RGB", (tile_width, resized.height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), label, fill=(0, 0, 0))
    canvas.paste(resized, (0, label_h))
    return canvas


def _save_sheet(
    path: Path,
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    manifest: List[Dict[str, Any]],
    output_type: str,
    view_ids: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_label_tile(img, label) for label, img in row]
        canvas = Image.new(
            "RGB",
            (sum(tile.width for tile in tiles) + 6 * max(len(tiles) - 1, 0), max(tile.height for tile in tiles)),
            "white",
        )
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rendered_rows), sum(row.height for row in rendered_rows) + 6 * max(len(rendered_rows) - 1, 0)),
        "white",
    )
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "scene": SCENE,
            "file_path": str(path),
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _source_audit(output_dir: Path) -> Dict[str, Any]:
    audit = {
        "DENSIFICATION_TRIGGER_SOURCE_AUDITED": True,
        "source_files": [
            "water_splatting/water_splatting.py",
            "water_splatting/rasterize.py",
            "water_splatting/rendering/underwater_rasterizer.py",
        ],
        "strategy_class_function": {
            "callbacks": "WaterSplattingModel.get_training_callbacks registers after_train and refinement_after after each train iteration.",
            "accumulator": "WaterSplattingModel.after_train(step)",
            "mutation": "WaterSplattingModel.refinement_after(optimizers, step)",
            "split": "WaterSplattingModel.split_gaussians",
            "duplicate": "WaterSplattingModel.dup_gaussians",
            "prune": "WaterSplattingModel.cull_gaussians",
        },
        "trigger_statistic": {
            "formula": "avg_grad_norm_i = (xys_grad_norm_i / vis_counts_i) * 0.5 * max(last_size)",
            "primary_score": "screen-space projected Gaussian xys absolute gradient norm, accumulated over train iterations",
            "input_when_abs_grad_densification_true": "self.xys_grad_abs.detach().norm(dim=-1)",
            "input_when_abs_grad_densification_false": "self.xys.grad.detach().norm(dim=-1)",
            "current_config_abs_grad_densification": True,
            "visibility_normalization": "vis_counts increments only for radii>0 on subsequent images; first accumulator initialization uses ones_like for all Gaussians.",
            "threshold": 0.0008,
        },
        "gates": {
            "high_grad": "avg_grad_norm > densify_grad_thresh",
            "split_scale3d": "exp(scales).max(dim=-1) > densify_size_thresh",
            "duplicate_scale3d": "exp(scales).max(dim=-1) <= densify_size_thresh",
            "densify_size_thresh": 0.001,
            "split_screen_size": "Only active if step < stop_screen_size_at; default stop_screen_size_at=0, therefore inactive for positive training steps.",
            "opacity_in_grow_eligibility": False,
            "split_mask": "high_grad and split_scale3d, plus inactive screen-size split gate under current default.",
            "duplicate_mask": "high_grad and duplicate_scale3d",
        },
        "pruning": {
            "pre_stop_alpha": "sigmoid(opacities) < cull_alpha_thresh, threshold 0.5 before stop_split_at",
            "post_stop_alpha": "sigmoid(opacities) < cull_alpha_thresh_post, threshold 0.1 at/after stop_split_at",
            "too_big_3d": "after step > refine_every * reset_alpha_every, cull exp(scales).max > cull_scale_thresh=10.0",
            "too_big_screen": "only active if step < stop_screen_size_at; inactive by default",
            "split_original_prune": "original split Gaussians are passed as extra_cull_mask after children are created",
            "current_loss_can_change_prune_in_fixed_state": False,
        },
        "timing": {
            "warmup_length": 500,
            "refine_every": 100,
            "reset_alpha_every": 5,
            "reset_interval": 500,
            "stop_split_at": 10000,
            "do_densification": "step < stop_split_at and step % reset_interval > num_train_data + refine_every",
            "opacity_reset": "when step < stop_split_at and step % reset_interval == refine_every, opacity logits are clamped to logit(0.5)",
        },
        "depth_gradient_routing_code_fact": {
            "coarse_depth_loss": "outputs['depth'] is transformed as 1/(10*depth+1) and correlated with normalized batch['depth_image']; weight 0.1.",
            "depth_output": "UnderwaterRasterizer.depth = depth_im / alpha for alpha>0, else a detached fill value.",
            "rasterize_backward_note": "Current Python binding passes v_out_img, v_out_medium, and v_out_alpha to CUDA backward. v_depth_im is not forwarded directly, but depth_expected depends on alpha, so coarse-depth can reach rasterization through the alpha denominator path if no-step autograd produces xys/xys_grad_abs changes.",
            "must_measure_directly": "Parameter-group gradients are not a substitute for the true densification trigger statistic; this audit measures xys_grad_abs / xys.grad directly.",
        },
        "line_references": {
            "config": "water_splatting/water_splatting.py:115-172",
            "after_train": "water_splatting/water_splatting.py:504-536",
            "refinement_after": "water_splatting/water_splatting.py:545-648",
            "cull": "water_splatting/water_splatting.py:650-683",
            "split_duplicate": "water_splatting/water_splatting.py:685-732",
            "rasterizer_xys_grad_abs": "water_splatting/water_splatting.py:1431-1435",
            "loss": "water_splatting/water_splatting.py:1567-1654",
            "rasterize_backward": "water_splatting/rasterize.py:269-356",
        },
    }
    _write_json(output_dir / "densification_source_audit.json", audit)
    lines = [
        "# WaterSplatting Densification Source Audit",
        "",
        f"- DENSIFICATION_TRIGGER_SOURCE_AUDITED: `{audit['DENSIFICATION_TRIGGER_SOURCE_AUDITED']}`",
        "- Trigger statistic: `avg_grad_norm_i = (xys_grad_norm_i / vis_counts_i) * 0.5 * max(last_size)`.",
        "- Current accumulator input: `self.xys_grad_abs.detach().norm(dim=-1)` because `abs_grad_densification=True`.",
        "- Eligibility: high trigger score plus a 3D scale gate; large Gaussians split, small Gaussians duplicate.",
        "- Opacity is not part of grow eligibility; it is used by pruning and opacity reset.",
        "- Fixed-state pruning eligibility is independent of the current loss gradient.",
        "- Coarse-depth gradient reachability is measured by this audit on the exact trigger input, not inferred from parameter-group gradients.",
    ]
    (output_dir / "densification_source_audit.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return audit


def _parse_train_log(path: Path, run: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    split_re = re.compile(r"Splitting\s+([0-9.eE+-]+)\s+gaussians:\s+(\d+)/(\d+)")
    dup_re = re.compile(r"Duplicating\s+([0-9.eE+-]+)\s+gaussians:\s+(\d+)/(\d+)")
    cull_re = re.compile(r"Culled(?: step=(\d+))?\s+(\d+)\s+gaussians\s+\((\d+)\s+below alpha thresh,\s+(\d+)\s+too bigs,\s+(\d+)")
    rows: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    for line in path.read_text(errors="ignore").splitlines():
        split = split_re.search(line)
        if split:
            pending = {
                "scene": SCENE,
                "run": run,
                "event_type": "densify",
                "split_fraction": float(split.group(1)),
                "split_count": int(split.group(2)),
                "pre_count": int(split.group(3)),
            }
            continue
        dup = dup_re.search(line)
        if dup and pending:
            pending["duplicate_fraction"] = float(dup.group(1))
            pending["duplicate_count"] = int(dup.group(2))
            pending["duplicate_pre_count"] = int(dup.group(3))
            continue
        cull = cull_re.search(line)
        if cull:
            if not pending:
                pending = {"scene": SCENE, "run": run, "event_type": "cull_only"}
            pending.update(
                {
                    "step": int(cull.group(1)) if cull.group(1) else "",
                    "cull_count": int(cull.group(2)),
                    "below_alpha_count": int(cull.group(3)),
                    "too_big_count": int(cull.group(4)),
                    "remaining_count": int(cull.group(5)),
                }
            )
            rows.append(pending)
            pending = {}
    return rows


def _historical_state_availability(repo: Path, output_dir: Path, checkpoint_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ckpt_rows: List[Dict[str, Any]] = []
    all_acc_available = True
    for row in checkpoint_rows:
        ckpt_path = Path(row["checkpoint_path"])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pipe = ckpt.get("pipeline", {})
        keys = list(pipe.keys())
        matches = {
            "xys_grad_norm": [key for key in keys if "xys_grad_norm" in key],
            "vis_counts": [key for key in keys if "vis_counts" in key],
            "max_2Dsize": [key for key in keys if "max_2Dsize" in key],
            "depths_accum": [key for key in keys if "depths_accum" in key],
            "radii": [key for key in keys if "radii" in key],
            "split_dup_prune_bookkeeping": [
                key
                for key in keys
                if any(needle in key.lower() for needle in ("split", "dup", "prune", "cull"))
            ],
            "sampler_state": [key for key in keys if "_ws_full_image_sampler" in key],
        }
        exact_acc = bool(matches["xys_grad_norm"] and matches["vis_counts"] and matches["max_2Dsize"])
        all_acc_available = all_acc_available and exact_acc
        ckpt_rows.append(
            {
                "scene": SCENE,
                "run": row["run"],
                "nominal_step": row["nominal_step"],
                "actual_step": row["actual_step"],
                "checkpoint_path": str(ckpt_path),
                "pipeline_key_count": len(keys),
                "historical_accumulators_available": exact_acc,
                "visibility_counter_available": bool(matches["vis_counts"]),
                "max_2d_size_available": bool(matches["max_2Dsize"]),
                "split_dup_prune_bookkeeping_available": bool(matches["split_dup_prune_bookkeeping"]),
                "sampler_state_available": bool(matches["sampler_state"]),
                "matched_keys": json.dumps(matches, sort_keys=True),
            }
        )
        del ckpt

    k1_log = repo / "logs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/train.log"
    event_rows = _parse_train_log(k1_log, "BND-K1")
    _write_csv(output_dir / "historical_event_log_counts.csv", event_rows)
    _write_json(output_dir / "historical_event_log_counts.json", {"rows": event_rows})
    _write_csv(output_dir / "historical_state_availability.csv", ckpt_rows)
    summary = {
        "HISTORICAL_TRIGGER_STATE_AVAILABLE": bool(all_acc_available),
        "HISTORICAL_VISIBILITY_ACCUMULATOR_AVAILABLE": bool(all(row["visibility_counter_available"] for row in ckpt_rows)),
        "HISTORICAL_SPLIT_COUNTS_AVAILABLE": bool(event_rows),
        "HISTORICAL_PRUNE_COUNTS_AVAILABLE": bool(event_rows),
        "HISTORICAL_EXACT_ELIGIBILITY_AVAILABLE": False,
        "HISTORICAL_TRIGGER_CORROBORATION": "NOT_AVAILABLE",
        "note": "Checkpoint exact accumulators are unavailable; K1 log event counts exist, but CDEPTH matching logs were not found under logs/.",
        "rows": ckpt_rows,
    }
    _write_json(output_dir / "historical_state_availability.json", summary)
    return summary


def _checkpoint_manifest(repo: Path, steps: Sequence[int], runs: Sequence[str], output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        spec = RUNS[run]
        config_path = repo / spec.config_relpath
        for nominal_step in steps:
            actual_step = _actual_step(config_path, nominal_step)
            ckpt_path = _available_steps(config_path).get(actual_step) if actual_step is not None else None
            rows.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": actual_step if actual_step is not None else "",
                    "config_path": str(config_path),
                    "checkpoint_path": str(ckpt_path) if ckpt_path else "",
                    "checkpoint_exists": bool(ckpt_path and ckpt_path.exists()),
                }
            )
    _write_csv(output_dir / "checkpoint_manifest.csv", rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": rows})
    return rows


def _compute_loss(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any], condition: str) -> Tuple[Tensor, Dict[str, float]]:
    metrics: Dict[str, Tensor] = {}
    if condition == "R":
        model.config.coarse_depth_supervision_enabled = False
        losses = model.get_loss_dict(outputs, batch, metrics)
        loss = losses["main_loss"]
    elif condition == "RD":
        model.config.coarse_depth_supervision_enabled = True
        model.config.coarse_depth_supervision_weight = 0.1
        losses = model.get_loss_dict(outputs, batch, metrics)
        loss = losses["main_loss"] + losses["coarse_depth_loss"]
    elif condition == "D_ONLY":
        model.config.coarse_depth_supervision_enabled = True
        model.config.coarse_depth_supervision_weight = 0.1
        losses = model.get_loss_dict(outputs, batch, metrics)
        loss = losses["coarse_depth_loss"]
    else:
        raise ValueError(condition)
    vals = {f"loss_{key}": float(value.detach().cpu().item()) for key, value in losses.items()}
    vals.update({f"metric_{key}": float(value.detach().cpu().item()) for key, value in metrics.items()})
    return loss, vals


def _accumulate_trigger_condition(
    model: Any,
    train_records: Sequence[Tuple[int, str, Any, Mapping[str, Any]]],
    condition: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = int(model.means.shape[0])
    xys_grad_sum: Optional[Tensor] = None
    signed_grad_sum: Optional[Tensor] = None
    vis_counts: Optional[Tensor] = None
    depths_accum: Optional[Tensor] = None
    max_2d_size = torch.zeros(n, dtype=torch.float32)
    visible_any = torch.zeros(n, dtype=torch.bool)
    rows: List[Dict[str, Any]] = []
    last_size: Tuple[int, int] = (1, 1)
    model.train()
    for order, (train_index, view_id, camera, batch) in enumerate(train_records):
        model.zero_grad(set_to_none=True)
        if getattr(model, "xys_grad_abs", None) is not None:
            model.xys_grad_abs = None
        outputs = model.get_outputs(camera.to(model.device))
        loss, loss_values = _compute_loss(model, outputs, batch, condition)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {condition} {view_id}: {loss}")
        loss.backward()

        visible = (model.radii.detach().reshape(-1) > 0)
        visible_cpu = visible.cpu()
        visible_any |= visible_cpu
        abs_grad = model.xys_grad_abs.detach().norm(dim=-1).reshape(-1).float().cpu()
        if getattr(model.xys, "grad", None) is not None:
            signed_grad = model.xys.grad.detach().norm(dim=-1).reshape(-1).float().cpu()
        else:
            signed_grad = torch.zeros_like(abs_grad)
        depths = model.depths.detach().reshape(-1).float().cpu()
        radii = model.radii.detach().reshape(-1).float().cpu()
        last_size = (int(model.last_size[0]), int(model.last_size[1]))
        if xys_grad_sum is None:
            xys_grad_sum = abs_grad.clone()
            signed_grad_sum = signed_grad.clone()
            vis_counts = torch.ones_like(abs_grad)
            depths_accum = depths.clone()
        else:
            assert signed_grad_sum is not None and vis_counts is not None and depths_accum is not None
            vis_counts[visible_cpu] += 1.0
            xys_grad_sum[visible_cpu] += abs_grad[visible_cpu]
            signed_grad_sum[visible_cpu] += signed_grad[visible_cpu]
            depths_accum[visible_cpu] += depths[visible_cpu]
        max_2d_size[visible_cpu] = torch.maximum(max_2d_size[visible_cpu], radii[visible_cpu] / float(max(last_size)))
        row = {
            "condition": condition,
            "bank_order": order,
            "train_index": train_index,
            "view_id": view_id,
            "visible_count": int(visible_cpu.sum().item()),
            "last_size_h": last_size[0],
            "last_size_w": last_size[1],
            "loss_total": float(loss.detach().cpu().item()),
        }
        row.update(loss_values)
        row.update(_stats(abs_grad[visible_cpu], "abs_xys_grad_visible_"))
        row.update(_stats(signed_grad[visible_cpu], "signed_xys_grad_visible_"))
        rows.append(row)
        model.zero_grad(set_to_none=True)

    if xys_grad_sum is None or vis_counts is None or depths_accum is None or signed_grad_sum is None:
        raise RuntimeError("No training records were processed")
    scale = 0.5 * float(max(last_size))
    avg_grad_norm = (xys_grad_sum / vis_counts.clamp_min(1.0)) * scale
    avg_signed_grad_norm = (signed_grad_sum / vis_counts.clamp_min(1.0)) * scale
    mean_depth = depths_accum / vis_counts.clamp_min(1.0)
    result = {
        "score": avg_grad_norm,
        "signed_score": avg_signed_grad_norm,
        "vis_counts": vis_counts,
        "visible_any": visible_any,
        "max_2d_size": max_2d_size,
        "mean_depth": mean_depth,
        "last_size": last_size,
        "score_scale": scale,
        "view_rows": rows,
    }
    return result, rows


def _eligibility(model: Any, trigger: Mapping[str, Any]) -> Dict[str, Tensor]:
    score = trigger["score"].detach().cpu()
    high_grad = score > float(model.config.densify_grad_thresh)
    scale_max = model.scales.detach().cpu().float().exp().max(dim=-1).values.reshape(-1)
    split_scale3d = scale_max > float(model.config.densify_size_thresh)
    duplicate_scale3d = scale_max <= float(model.config.densify_size_thresh)
    if int(model.step) < int(model.config.stop_screen_size_at):
        split_screen = trigger["max_2d_size"] > float(model.config.split_screen_size)
        split_gate = split_scale3d | split_screen
    else:
        split_screen = torch.zeros_like(split_scale3d)
        split_gate = split_scale3d
    splits = high_grad & split_gate
    duplicates = high_grad & duplicate_scale3d
    eligible = splits | duplicates
    opacity = torch.sigmoid(model.opacities.detach().cpu().float()).reshape(-1)
    if int(model.step) < int(model.config.stop_split_at):
        opacity_culls = opacity < float(model.config.cull_alpha_thresh)
    else:
        opacity_culls = opacity < float(model.config.cull_alpha_thresh_post)
    too_big = torch.zeros_like(opacity_culls)
    if int(model.step) > int(model.config.refine_every * model.config.reset_alpha_every):
        too_big = scale_max > float(model.config.cull_scale_thresh)
        if int(model.step) < int(model.config.stop_screen_size_at):
            too_big = too_big | (trigger["max_2d_size"] > float(model.config.cull_screen_size))
    prune = opacity_culls | too_big
    return {
        "high_grad": high_grad,
        "split_scale3d": split_scale3d,
        "duplicate_scale3d": duplicate_scale3d,
        "split_screen": split_screen,
        "split": splits,
        "duplicate": duplicates,
        "eligible": eligible,
        "prune": prune,
        "scale_max": scale_max,
        "opacity": opacity,
        "anisotropy": scale_max / model.scales.detach().cpu().float().exp().min(dim=-1).values.reshape(-1).clamp_min(EPS),
    }


def _group_masks(eligible_r: Tensor, eligible_rd: Tensor) -> Dict[str, Tensor]:
    return {
        "DEPTH_ADDED": (~eligible_r) & eligible_rd,
        "DEPTH_REMOVED": eligible_r & (~eligible_rd),
        "UNCHANGED_ELIGIBLE": eligible_r & eligible_rd,
        "UNCHANGED_INELIGIBLE": (~eligible_r) & (~eligible_rd),
    }


def _trigger_comparison_rows(
    run: str,
    nominal_step: int,
    actual_step: int,
    model: Any,
    trig_r: Mapping[str, Any],
    trig_rd: Mapping[str, Any],
    trig_d: Mapping[str, Any],
    elig_r: Mapping[str, Tensor],
    elig_rd: Mapping[str, Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Tensor]]:
    visible = trig_r["visible_any"] | trig_rd["visible_any"]
    groups = _group_masks(elig_r["eligible"], elig_rd["eligible"])
    delta_score = trig_rd["score"] - trig_r["score"]
    delta_signed = trig_rd["signed_score"] - trig_r["signed_score"]
    theta = float(model.config.densify_grad_thresh)
    near = visible & (trig_r["score"] / theta >= 0.8) & (trig_r["score"] / theta <= 1.2)
    rows: List[Dict[str, Any]] = []
    base = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "actual_step": actual_step}
    for condition, trigger, elig in (("R", trig_r, elig_r), ("RD", trig_rd, elig_rd), ("D_ONLY", trig_d, _eligibility(model, trig_d))):
        row = {**base, "condition": condition, "metric_group": "trigger_distribution"}
        row.update(_masked_stats(trigger["score"], visible, "score_"))
        row.update(_masked_stats(trigger["signed_score"], visible, "signed_score_"))
        row.update(
            {
                "N_total": int(trigger["score"].numel()),
                "N_visible": int(visible.sum().item()),
                "N_high_grad": int(elig["high_grad"][visible].sum().item()),
                "N_eligible": int(elig["eligible"][visible].sum().item()),
                "N_split": int(elig["split"][visible].sum().item()),
                "N_duplicate": int(elig["duplicate"][visible].sum().item()),
                "N_prune": int(elig["prune"][visible].sum().item()),
                "score_threshold": theta,
                "score_scale": float(trigger["score_scale"]),
            }
        )
        rows.append(row)

    flip_row = {**base, "condition": "RD_MINUS_R", "metric_group": "eligibility_flips"}
    flip_row.update(_masked_stats(delta_score, visible, "delta_score_"))
    flip_row.update(_masked_stats(delta_signed, visible, "delta_signed_score_"))
    flip_row.update(
        {
            "positive_delta_fraction": float((delta_score[visible] > 0).float().mean().item()) if visible.any() else float("nan"),
            "negative_delta_fraction": float((delta_score[visible] < 0).float().mean().item()) if visible.any() else float("nan"),
            "N_visible": int(visible.sum().item()),
            "N_eligible_R": int(elig_r["eligible"][visible].sum().item()),
            "N_eligible_RD": int(elig_rd["eligible"][visible].sum().item()),
            "N_depth_added": int(groups["DEPTH_ADDED"][visible].sum().item()),
            "N_depth_removed": int(groups["DEPTH_REMOVED"][visible].sum().item()),
            "ADDED_RATE": float(groups["DEPTH_ADDED"][visible].float().mean().item()) if visible.any() else float("nan"),
            "REMOVED_RATE": float(groups["DEPTH_REMOVED"][visible].float().mean().item()) if visible.any() else float("nan"),
            "N_depth_added_over_eligible_R": int(groups["DEPTH_ADDED"][visible].sum().item())
            / max(int(elig_r["eligible"][visible].sum().item()), 1),
            "ELIGIBLE_COUNT_RATIO": int(elig_rd["eligible"][visible].sum().item()) / max(int(elig_r["eligible"][visible].sum().item()), 1),
            "near_threshold_count": int(near.sum().item()),
            "near_threshold_median_abs_shift_over_threshold": (
                float((delta_score[near].abs() / theta).median().item()) if near.any() else float("nan")
            ),
            "near_threshold_mean_shift_over_threshold": (
                float((delta_score[near] / theta).mean().item()) if near.any() else float("nan")
            ),
            "DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC": bool((_finite_flat(trig_d["score"][visible]).abs().max().item() > 0.0) if visible.any() else False),
        }
    )
    rows.append(flip_row)

    type_row = {**base, "condition": "RD_MINUS_R", "metric_group": "candidate_types"}
    type_row.update(
        {
            "DEPTH_ADDED_SPLIT": int((groups["DEPTH_ADDED"] & elig_rd["split"] & visible).sum().item()),
            "DEPTH_ADDED_DUPLICATE": int((groups["DEPTH_ADDED"] & elig_rd["duplicate"] & visible).sum().item()),
            "DEPTH_REMOVED_SPLIT": int((groups["DEPTH_REMOVED"] & elig_r["split"] & visible).sum().item()),
            "DEPTH_REMOVED_DUPLICATE": int((groups["DEPTH_REMOVED"] & elig_r["duplicate"] & visible).sum().item()),
        }
    )
    rows.append(type_row)

    summary = {
        **base,
        "N_visible": int(visible.sum().item()),
        "N_eligible_R": int(elig_r["eligible"][visible].sum().item()),
        "N_eligible_RD": int(elig_rd["eligible"][visible].sum().item()),
        "N_depth_added": int(groups["DEPTH_ADDED"][visible].sum().item()),
        "N_depth_removed": int(groups["DEPTH_REMOVED"][visible].sum().item()),
        "ELIGIBLE_COUNT_RATIO": int(elig_rd["eligible"][visible].sum().item()) / max(int(elig_r["eligible"][visible].sum().item()), 1),
        "ADDED_RATE": float(groups["DEPTH_ADDED"][visible].float().mean().item()) if visible.any() else float("nan"),
        "REMOVED_RATE": float(groups["DEPTH_REMOVED"][visible].float().mean().item()) if visible.any() else float("nan"),
        "near_threshold_median_abs_shift_over_threshold": flip_row["near_threshold_median_abs_shift_over_threshold"],
        "DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC": flip_row["DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC"],
    }
    return rows, summary, groups


def _candidate_attribute_rows(
    run: str,
    nominal_step: int,
    actual_step: int,
    masks: Mapping[str, Tensor],
    visible: Tensor,
    trigger: Mapping[str, Any],
    elig: Mapping[str, Tensor],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = {
        "scale_max": elig["scale_max"],
        "anisotropy": elig["anisotropy"],
        "opacity": elig["opacity"],
        "visibility_count": trigger["vis_counts"],
        "projected_radius_max2d_fraction": trigger["max_2d_size"],
        "mean_projected_depth": trigger["mean_depth"],
        "trigger_score": trigger["score"],
    }
    for group, mask in masks.items():
        group_mask = mask & visible
        for metric_name, values in metrics.items():
            row = {
                "scene": SCENE,
                "run": run,
                "nominal_step": nominal_step,
                "actual_step": actual_step,
                "candidate_group": group,
                "metric": metric_name,
                "N": int(group_mask.sum().item()),
            }
            row.update(_masked_stats(values, group_mask))
            rows.append(row)
    return rows


def _load_final_items(repo: Path, run: str) -> Dict[str, Dict[str, Any]]:
    loaded = None
    try:
        loaded = path_audit._load_run(repo, run, FINAL_STEP)
        model = loaded.pipeline.model
        model.eval()
        items: Dict[str, Dict[str, Any]] = {}
        for eval_index, view_id, camera, batch in path_audit._view_records(loaded):
            if view_id not in EVAL_OUTCOME_VIEWS:
                continue
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            keep = ("pred_image", "clear_object_fullsh_raw", "accumulation")
            items[view_id] = {
                "eval_index": eval_index,
                "view_id": view_id,
                "gt": gt.detach().float().cpu(),
                "outputs": {key: outputs[key].detach().float().cpu() for key in keep},
            }
        return items
    finally:
        path_audit._release(loaded)


def _error(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)


def _build_outcome_regions(repo: Path, output_dir: Path) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    m1 = _load_final_items(repo, "M1")
    k1 = _load_final_items(repo, "BND-K1")
    cd = _load_final_items(repo, "CDEPTH")
    view_ids = [view for view in EVAL_OUTCOME_VIEWS if view in m1 and view in k1 and view in cd]
    pos_values: List[Tensor] = []
    neg_values: List[Tensor] = []
    gain_maps: Dict[str, Tensor] = {}
    high_masks: Dict[str, Tensor] = {}
    for view_id in view_ids:
        gain = _error(k1[view_id]["outputs"]["pred_image"], m1[view_id]["gt"]) - _error(
            cd[view_id]["outputs"]["pred_image"], m1[view_id]["gt"]
        )
        high = (m1[view_id]["outputs"]["accumulation"][..., 0] > 0.01) & (
            m1[view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0
        )
        gain_maps[view_id] = gain
        high_masks[view_id] = high
        vals = gain[high]
        pos = vals[vals > 0]
        neg = -vals[vals < 0]
        if pos.numel():
            pos_values.append(pos)
        if neg.numel():
            neg_values.append(neg)
    strong_gain_threshold = _q(torch.cat(pos_values) if pos_values else torch.empty(0), 0.75)
    strong_harm_threshold = _q(torch.cat(neg_values) if neg_values else torch.empty(0), 0.75)
    regions: Dict[str, Dict[str, Tensor]] = {}
    for view_id in view_ids:
        gain = gain_maps[view_id]
        high = high_masks[view_id]
        regions[view_id] = {
            "M1_HIGH_J": high,
            "HJ_GAIN": high & (gain > 0),
            "HJ_HARM": high & (gain < 0),
            "HJ_STRONG_GAIN": high & (gain >= strong_gain_threshold),
            "HJ_STRONG_HARM": high & ((-gain) >= strong_harm_threshold),
        }
    meta = {
        "view_ids": view_ids,
        "M1_HIGH_J_definition": "final M1 accumulation > 0.01 and final M1 clear_object_fullsh_raw max RGB channel > 1.0",
        "HJ_GAIN_definition": "M1_HIGH_J and final RGB MSE(K1) - RGB MSE(CDEPTH) > 0; post-hoc future-outcome mask",
        "HJ_HARM_definition": "M1_HIGH_J and final RGB MSE(K1) - RGB MSE(CDEPTH) < 0; post-hoc future-outcome mask",
        "HJ_STRONG_GAIN_definition": "pooled top quartile of positive HJ_GAIN values",
        "HJ_STRONG_HARM_definition": "pooled top quartile of negative HJ_HARM magnitude",
        "strong_gain_threshold": strong_gain_threshold,
        "strong_harm_threshold": strong_harm_threshold,
        "post_hoc_warning": "Outcome masks are defined from final 15k K1/CDEPTH results and are not online training signals.",
    }
    _write_json(output_dir / "outcome_region_definitions.json", meta)
    return regions, meta


def _draw_candidate_centers(
    height: int,
    width: int,
    xys: Tensor,
    radii: Tensor,
    mask: Tensor,
    base_regions: Optional[Mapping[str, Tensor]] = None,
    strong: bool = False,
) -> Image.Image:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if base_regions:
        if "M1_HIGH_J" in base_regions:
            canvas[base_regions["M1_HIGH_J"].cpu().numpy()] = np.array([64, 64, 64], dtype=np.uint8)
        gain_name = "HJ_STRONG_GAIN" if strong else "HJ_GAIN"
        harm_name = "HJ_STRONG_HARM" if strong else "HJ_HARM"
        if gain_name in base_regions:
            canvas[base_regions[gain_name].cpu().numpy()] = np.array([0, 120, 0], dtype=np.uint8)
        if harm_name in base_regions:
            canvas[base_regions[harm_name].cpu().numpy()] = np.array([140, 0, 140], dtype=np.uint8)
    xy = xys.detach().float().cpu().reshape(-1, 2)
    r = radii.detach().float().cpu().reshape(-1)
    m = mask.detach().bool().cpu().reshape(-1)
    valid = (
        m
        & torch.isfinite(xy).all(dim=-1)
        & torch.isfinite(r)
        & (r > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if valid.any():
        coords = xy[valid].round().long()
        coords[:, 0].clamp_(0, width - 1)
        coords[:, 1].clamp_(0, height - 1)
        canvas[coords[:, 1].numpy(), coords[:, 0].numpy()] = np.array([255, 220, 0], dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def _spatial_projection(
    repo: Path,
    run: str,
    nominal_step: int,
    actual_step: int,
    masks: Mapping[str, Tensor],
    regions: Mapping[str, Mapping[str, Tensor]],
    render_manifest: List[Dict[str, Any]],
    map_rows: List[Sequence[Tuple[str, Image.Image]]],
    strong_rows: List[Sequence[Tuple[str, Image.Image]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    loaded = None
    try:
        loaded = _load_run(repo, run, nominal_step, load_depths=False)
        model = loaded.pipeline.model
        model.eval()
        for _eval_index, view_id, camera, _batch in _eval_records(loaded):
            if view_id not in regions:
                continue
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            height, width = outputs["pred_image"].shape[:2]
            xy = model.xys.detach().float().cpu().reshape(-1, 2)
            radii = model.radii.detach().float().cpu().reshape(-1)
            visible = (
                torch.isfinite(xy).all(dim=-1)
                & torch.isfinite(radii)
                & (radii > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < height)
            )
            xi = xy[:, 0].round().long().clamp(0, width - 1)
            yi = xy[:, 1].round().long().clamp(0, height - 1)
            for group, group_mask in masks.items():
                candidate = group_mask.cpu().bool() & visible
                for region_name, region_mask in regions[view_id].items():
                    visible_in = region_mask[yi[visible], xi[visible]]
                    candidate_in = region_mask[yi[candidate], xi[candidate]]
                    visible_frac = float(visible_in.float().mean().item()) if visible.any() else float("nan")
                    cand_frac = float(candidate_in.float().mean().item()) if candidate.any() else float("nan")
                    rows.append(
                        {
                            "scene": SCENE,
                            "run": run,
                            "nominal_step": nominal_step,
                            "actual_step": actual_step,
                            "view_id": view_id,
                            "candidate_group": group,
                            "region": region_name,
                            "visible_projected_count": int(visible.sum().item()),
                            "candidate_projected_count": int(candidate.sum().item()),
                            "P_region_given_visible_center": visible_frac,
                            "P_region_given_candidate_center": cand_frac,
                            "CENTER_ENRICHMENT": cand_frac / max(visible_frac, EPS) if math.isfinite(cand_frac) and math.isfinite(visible_frac) else float("nan"),
                            "support_proxy_status": "NOT_COMPUTED_CENTER_PROXY_ONLY",
                        }
                    )
            added = masks["DEPTH_ADDED"].cpu().bool()
            removed = masks["DEPTH_REMOVED"].cpu().bool()
            map_rows.append(
                [
                    (f"{nominal_step} {view_id} masks", _draw_candidate_centers(height, width, xy, radii, torch.zeros_like(added), regions[view_id])),
                    (f"{nominal_step} {view_id} ADDED", _draw_candidate_centers(height, width, xy, radii, added, regions[view_id])),
                    (f"{nominal_step} {view_id} REMOVED", _draw_candidate_centers(height, width, xy, radii, removed, regions[view_id])),
                ]
            )
            strong_rows.append(
                [
                    (f"{nominal_step} {view_id} strong masks", _draw_candidate_centers(height, width, xy, radii, torch.zeros_like(added), regions[view_id], strong=True)),
                    (f"{nominal_step} {view_id} ADDED strong", _draw_candidate_centers(height, width, xy, radii, added, regions[view_id], strong=True)),
                    (f"{nominal_step} {view_id} REMOVED strong", _draw_candidate_centers(height, width, xy, radii, removed, regions[view_id], strong=True)),
                ]
            )
        return rows
    finally:
        _release(loaded)


def _plot_score_distribution(path: Path, rows_by_step: Mapping[int, Mapping[str, Tensor]], manifest: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(rows_by_step), 1, figsize=(8, 3.2 * len(rows_by_step)), squeeze=False)
    for ax, (step, by_cond) in zip(axes[:, 0], rows_by_step.items()):
        for condition, values in by_cond.items():
            flat = _finite_flat(values)
            if flat.numel() == 0:
                continue
            sample = flat
            if sample.numel() > 200000:
                idx = torch.linspace(0, sample.numel() - 1, 200000).long()
                sample = sample[idx]
            ax.hist(sample.numpy(), bins=80, alpha=0.45, label=condition)
        ax.axvline(0.0008, color="black", linestyle="--", linewidth=1.0, label="threshold")
        ax.set_title(f"K1 fixed-state trigger score distribution @ {step}")
        ax.set_xlabel("avg_grad_norm")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": "trigger_score_distributions", "view_ids": "fixed_train_bank"})


def _plot_counts(path: Path, summaries: Sequence[Mapping[str, Any]], manifest: List[Dict[str, Any]]) -> None:
    k1 = [row for row in summaries if row["run"] == "BND-K1"]
    if not k1:
        return
    steps = [int(row["nominal_step"]) for row in k1]
    labels = ["N_eligible_R", "N_eligible_RD", "N_depth_added", "N_depth_removed"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.18
    xs = np.arange(len(steps))
    for idx, label in enumerate(labels):
        ax.bar(xs + (idx - 1.5) * width, [float(row[label]) for row in k1], width=width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_xlabel("nominal step")
    ax.set_ylabel("Gaussian count")
    ax.set_title("K1 fixed-state eligibility counts")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": "eligibility_counts", "view_ids": "fixed_train_bank"})


def _plot_candidate_type(path: Path, type_rows: Sequence[Mapping[str, Any]], manifest: List[Dict[str, Any]]) -> None:
    rows = [row for row in type_rows if row["run"] == "BND-K1" and row.get("metric_group") == "candidate_types"]
    if not rows:
        return
    steps = [int(row["nominal_step"]) for row in rows]
    labels = ("DEPTH_ADDED_SPLIT", "DEPTH_ADDED_DUPLICATE", "DEPTH_REMOVED_SPLIT", "DEPTH_REMOVED_DUPLICATE")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.18
    xs = np.arange(len(steps))
    for idx, label in enumerate(labels):
        ax.bar(xs + (idx - 1.5) * width, [float(row.get(label, 0)) for row in rows], width=width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_xlabel("nominal step")
    ax.set_ylabel("Gaussian count")
    ax.set_title("K1 fixed-state candidate type counts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": "candidate_type_distributions", "view_ids": "fixed_train_bank"})


def _text_visual(path: Path, title: str, lines: Sequence[str], manifest: List[Dict[str, Any]], output_type: str) -> None:
    width = 1800
    height = max(200, 30 * (len(lines) + 2))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((12, 54 + idx * 30), line, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": output_type, "view_ids": "fixed_train_bank", "width": width, "height": height})


def _classify(
    summaries: Sequence[Mapping[str, Any]],
    spatial_rows: Sequence[Mapping[str, Any]],
    source_depth_reaches: bool,
) -> Dict[str, Any]:
    primary = [row for row in summaries if row["run"] == "BND-K1" and int(row["nominal_step"]) in (1000, 3000)]
    nontrivial_by_step: Dict[int, bool] = {}
    for row in primary:
        ratio = abs(float(row["ELIGIBLE_COUNT_RATIO"]) - 1.0)
        added_over = float(row["N_depth_added"]) / max(float(row["N_eligible_R"]), 1.0)
        near_shift = float(row["near_threshold_median_abs_shift_over_threshold"])
        nontrivial_by_step[int(row["nominal_step"])] = bool(ratio >= 0.05 or added_over >= 0.05 or (math.isfinite(near_shift) and near_shift >= 0.05))

    view_directions: Dict[str, bool] = {}
    for step in (1000, 3000):
        for view_id in EVAL_OUTCOME_VIEWS:
            gain = [
                row
                for row in spatial_rows
                if row["run"] == "BND-K1"
                and int(row["nominal_step"]) == step
                and row["view_id"] == view_id
                and row["candidate_group"] == "DEPTH_ADDED"
                and row["region"] == "HJ_GAIN"
            ]
            harm = [
                row
                for row in spatial_rows
                if row["run"] == "BND-K1"
                and int(row["nominal_step"]) == step
                and row["view_id"] == view_id
                and row["candidate_group"] == "DEPTH_ADDED"
                and row["region"] == "HJ_HARM"
            ]
            if gain and harm:
                g = float(gain[0]["CENTER_ENRICHMENT"])
                h = float(harm[0]["CENTER_ENRICHMENT"])
                if math.isfinite(g) and math.isfinite(h):
                    view_directions[f"{step}:{view_id}"] = g > h
    consistent_steps = []
    for step in (1000, 3000):
        vals = [ok for key, ok in view_directions.items() if key.startswith(f"{step}:")]
        if vals and sum(vals) >= 2:
            consistent_steps.append(step)
    pre_recovery = bool(any(nontrivial_by_step.get(step, False) and step in consistent_steps for step in (1000, 3000)))
    cross_view_consistent = bool(consistent_steps)

    if not source_depth_reaches:
        classification = "DENSIFICATION_TRIGGER_NOT_SUPPORTED"
    elif pre_recovery and cross_view_consistent:
        classification = "DENSIFICATION_TRIGGER_SUPPORTED"
    elif any(nontrivial_by_step.values()):
        classification = "DENSIFICATION_TRIGGER_PARTIAL"
    elif cross_view_consistent:
        classification = "HJ_ALIGNMENT_WITH_WEAK_TRIGGER_RESPONSE"
    else:
        classification = "DENSIFICATION_TRIGGER_NOT_SUPPORTED"

    k1_types = [row for row in summaries if row["run"] == "BND-K1"]
    added_split = 0
    added_dup = 0
    for row in k1_types:
        added_split += int(row.get("DEPTH_ADDED_SPLIT", 0) or 0)
        added_dup += int(row.get("DEPTH_ADDED_DUPLICATE", 0) or 0)
    if added_split == 0 and added_dup == 0:
        candidate_type = "NO_CLEAR_CANDIDATE_TYPE"
    elif added_split > 2 * added_dup:
        candidate_type = "LARGE_SCALE_SPLIT_BIASED"
    elif added_dup > 2 * added_split:
        candidate_type = "SMALL_SCALE_DUPLICATION_BIASED"
    else:
        candidate_type = "MIXED_CANDIDATE_REDISTRIBUTION"
    return {
        "DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC": bool(source_depth_reaches),
        "PRE_RECOVERY_TRIGGER_REDISTRIBUTION": pre_recovery,
        "SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT": cross_view_consistent,
        "nontrivial_trigger_shift_by_step": nontrivial_by_step,
        "view_direction_gain_gt_harm": view_directions,
        "Mechanism Classification": classification,
        "Candidate Type": candidate_type,
        "Next Single-Factor Experiment": (
            "BND-DTRIG depth-guided densification-trigger only"
            if classification == "DENSIFICATION_TRIGGER_SUPPORTED"
            else "DIRECT-OBJECT CONTINUOUS OPTIMIZATION PATH AUDIT"
        ),
    }


def _write_visual_index(path: Path, render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# BND-CDEPTH Densification Trigger Visual Index",
        "",
        "No subjective visual-quality judgment is made here.",
        "",
    ]
    for row in manifest:
        lines.append(f"- {row.get('output_type', 'file')}: `{row['file_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(
    path: Path,
    repo_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    historical: Mapping[str, Any],
    camera_bank: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
    manifest_paths: Mapping[str, str],
) -> None:
    k1_rows = [row for row in summaries if row["run"] == "BND-K1"]
    lines = [
        "# BND-CDEPTH Densification Trigger Audit",
        "",
        "## Motivation",
        "",
        "HYPOTHESIS: CDEPTH may alter early Gaussian refinement eligibility before the later high-J RGB recovery appears.",
        "",
        "## Code Fact",
        "",
        f"- Branch: `{repo_manifest['branch']}`.",
        f"- Start HEAD: `{repo_manifest['head']}`.",
        "- This was a read-only audit: no optimizer step, scheduler step, densification mutation, split, duplicate, prune, opacity reset, checkpoint write, or training run was executed.",
        f"- Source audit complete: `{source['DENSIFICATION_TRIGGER_SOURCE_AUDITED']}`.",
        "- WaterSplatting trigger statistic is `avg_grad_norm = (xys_grad_norm / vis_counts) * 0.5 * max(last_size)`.",
        "- Current trigger input is `xys_grad_abs.detach().norm(dim=-1)`, a screen-space absolute gradient statistic.",
        "- Split/duplicate eligibility is high trigger score plus a 3D scale gate; opacity is only a pruning/reset lifecycle condition.",
        "- CDEPTH loss uses normalized pseudo depth and `1/(10*outputs['depth']+1)` with weight 0.1.",
        "",
        "## Historical State Availability",
        "",
        f"- HISTORICAL_TRIGGER_STATE_AVAILABLE: `{historical['HISTORICAL_TRIGGER_STATE_AVAILABLE']}`.",
        f"- HISTORICAL_SPLIT_COUNTS_AVAILABLE: `{historical['HISTORICAL_SPLIT_COUNTS_AVAILABLE']}`.",
        f"- HISTORICAL_PRUNE_COUNTS_AVAILABLE: `{historical['HISTORICAL_PRUNE_COUNTS_AVAILABLE']}`.",
        "- Exact historical eligibility is not available from checkpoint accumulators, so fixed-bank outputs are labeled `FIXED_BANK_TRIGGER_RESPONSE`.",
        "",
        "## Fixed-State Intervention Design",
        "",
        f"- FIXED_TRIGGER_CAMERA_BANK count: `{camera_bank['count']}`.",
        f"- Selection rule: `{camera_bank['selection_rule']}`.",
        f"- Training camera names: `{';'.join(camera_bank['view_ids'])}`.",
        "- Conditions: `R = formal BND-K1 RGB objective`; `RD = R + exact CDEPTH coarse-depth term`.",
        "",
        "## Quantitative Result",
        "",
        "| Step | Eligible R | Eligible RD | Depth Added | Depth Removed | Eligible Ratio | Added Rate | Near-threshold median abs shift/theta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in k1_rows:
        lines.append(
            f"| {row['nominal_step']} | {row['N_eligible_R']} | {row['N_eligible_RD']} | "
            f"{row['N_depth_added']} | {row['N_depth_removed']} | "
            f"{float(row['ELIGIBLE_COUNT_RATIO']):.6f} | {float(row['ADDED_RATE']):.6f} | "
            f"{float(row['near_threshold_median_abs_shift_over_threshold']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"- DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC: `{classification['DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC']}`.",
            f"- PRE_RECOVERY_TRIGGER_REDISTRIBUTION: `{classification['PRE_RECOVERY_TRIGGER_REDISTRIBUTION']}`.",
            f"- SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT: `{classification['SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT']}`.",
            f"- Mechanism Classification: `{classification['Mechanism Classification']}`.",
            f"- Candidate Type: `{classification['Candidate Type']}`.",
            "",
            "INFERENCE: The fixed-state trigger response is an association-compatible audit, not a causal training proof.",
            "",
            "## Next Single-Factor Recommendation",
            "",
            f"- `{classification['Next Single-Factor Experiment']}`.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path_str in manifest_paths.items():
        lines.append(f"- {name}: `{path_str}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run_audit(repo: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "log_8": _git(repo, "log", "-8", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "diff_check_at_start": _git(repo, "diff", "--check"),
        "untracked_gmvc_scripts": [
            str(path)
            for path in (
                repo / "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
                repo / "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
            )
            if path.exists()
        ],
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    source = _source_audit(output_dir)
    steps = tuple(args.steps)
    runs = tuple(args.runs)
    checkpoint_rows = _checkpoint_manifest(repo, steps, runs, output_dir)
    historical = _historical_state_availability(repo, output_dir, checkpoint_rows)

    outcome_regions, outcome_meta = _build_outcome_regions(repo, output_dir)
    _write_json(output_dir / "fixed_state_equivalence.json", {"semantic": "same checkpoint, same Gaussians, same training camera bank, independent forward/backward per condition"})

    render_manifest: List[Dict[str, Any]] = []
    all_trigger_rows: List[Dict[str, Any]] = []
    candidate_flip_rows: List[Dict[str, Any]] = []
    candidate_type_rows: List[Dict[str, Any]] = []
    candidate_attr_rows: List[Dict[str, Any]] = []
    near_rows: List[Dict[str, Any]] = []
    spatial_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    score_plot_inputs: Dict[int, Dict[str, Tensor]] = {}
    projected_map_rows: List[Sequence[Tuple[str, Image.Image]]] = []
    strong_map_rows: List[Sequence[Tuple[str, Image.Image]]] = []
    camera_bank: Dict[str, Any] = {}
    source_depth_reaches = False

    for run in runs:
        for nominal_step in steps:
            actual_step = next((row["actual_step"] for row in checkpoint_rows if row["run"] == run and row["nominal_step"] == nominal_step), None)
            if actual_step in (None, ""):
                continue
            loaded = None
            try:
                loaded = _load_run(repo, run, nominal_step, load_depths=True)
                model = loaded.pipeline.model
                model.step = int(loaded.loaded_step)
                train_records = _train_records(loaded.pipeline, args.camera_count)
                if not camera_bank:
                    camera_bank = {
                        "scene": SCENE,
                        "count": len(train_records),
                        "selection_rule": train_records[0][3].get("_fixed_bank_selection_rule", "all training views in dataset order"),
                        "view_ids": [view_id for _idx, view_id, _camera, _batch in train_records],
                        "train_indices": [idx for idx, _view_id, _camera, _batch in train_records],
                        "pseudo_depth_mapping": "batch['depth_image'] from DepthDataset; dataparser depths_path=depthAnything_u16; per-image max normalization inside loss.",
                    }
                    _write_json(output_dir / "fixed_trigger_camera_bank.json", camera_bank)
                before = _parameter_snapshot(model)
                trig_r, r_view_rows = _accumulate_trigger_condition(model, train_records, "R")
                trig_rd, rd_view_rows = _accumulate_trigger_condition(model, train_records, "RD")
                trig_d, d_view_rows = _accumulate_trigger_condition(model, train_records, "D_ONLY")
                delta_rows = _parameter_delta_rows(before, model, run, nominal_step)
                safety_rows.extend(delta_rows)
                if any(float(row["max_abs_delta"]) != 0.0 for row in delta_rows):
                    raise RuntimeError(f"Parameter mutation detected for {run}@{nominal_step}")
                elig_r = _eligibility(model, trig_r)
                elig_rd = _eligibility(model, trig_rd)
                rows, summary, masks = _trigger_comparison_rows(
                    run,
                    nominal_step,
                    int(actual_step),
                    model,
                    trig_r,
                    trig_rd,
                    trig_d,
                    elig_r,
                    elig_rd,
                )
                source_depth_reaches = source_depth_reaches or bool(summary["DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC"])
                all_trigger_rows.extend(rows)
                summaries.append(summary)
                for row in rows:
                    if row.get("metric_group") == "eligibility_flips":
                        candidate_flip_rows.append(row)
                        near_rows.append(row)
                    if row.get("metric_group") == "candidate_types":
                        candidate_type_rows.append(row)
                        summary.update(
                            {
                                "DEPTH_ADDED_SPLIT": row.get("DEPTH_ADDED_SPLIT", 0),
                                "DEPTH_ADDED_DUPLICATE": row.get("DEPTH_ADDED_DUPLICATE", 0),
                                "DEPTH_REMOVED_SPLIT": row.get("DEPTH_REMOVED_SPLIT", 0),
                                "DEPTH_REMOVED_DUPLICATE": row.get("DEPTH_REMOVED_DUPLICATE", 0),
                            }
                        )
                visible = trig_r["visible_any"] | trig_rd["visible_any"]
                candidate_attr_rows.extend(_candidate_attribute_rows(run, nominal_step, int(actual_step), masks, visible, trig_rd, elig_rd))
                if run == "BND-K1":
                    score_plot_inputs[nominal_step] = {"R": trig_r["score"][visible], "RD": trig_rd["score"][visible]}
                    spatial_rows.extend(
                        _spatial_projection(
                            repo,
                            run,
                            nominal_step,
                            int(actual_step),
                            masks,
                            outcome_regions,
                            render_manifest,
                            projected_map_rows,
                            strong_map_rows,
                        )
                    )
                step_payload = {
                    "summary": summary,
                    "trigger_rows": rows,
                    "view_rows": {"R": r_view_rows, "RD": rd_view_rows, "D_ONLY": d_view_rows},
                    "parameter_safety": delta_rows,
                }
                _write_json(output_dir / f"trigger_response_{nominal_step//1000}k_{run.lower().replace('-', '_')}.json", step_payload)
                _write_csv(output_dir / f"trigger_response_{nominal_step//1000}k_{run.lower().replace('-', '_')}.csv", rows)
            finally:
                _release(loaded)

    _write_csv(output_dir / "trigger_response_all.csv", all_trigger_rows)
    _write_json(output_dir / "trigger_response_all.json", {"rows": all_trigger_rows})
    for step in steps:
        step_rows = [row for row in all_trigger_rows if row.get("run") == "BND-K1" and int(row.get("nominal_step", -1)) == int(step)]
        _write_csv(output_dir / f"trigger_response_{step//1000}k.csv", step_rows)
        _write_json(output_dir / f"trigger_response_{step//1000}k.json", {"rows": step_rows})
    _write_csv(output_dir / "candidate_flip_counts.csv", candidate_flip_rows)
    _write_json(output_dir / "candidate_flip_counts.json", {"rows": candidate_flip_rows})
    _write_csv(output_dir / "candidate_type_metrics.csv", candidate_type_rows)
    _write_json(output_dir / "candidate_type_metrics.json", {"rows": candidate_type_rows})
    _write_csv(output_dir / "candidate_attribute_metrics.csv", candidate_attr_rows)
    _write_json(output_dir / "candidate_attribute_metrics.json", {"rows": candidate_attr_rows})
    _write_csv(output_dir / "near_threshold_metrics.csv", near_rows)
    _write_json(output_dir / "near_threshold_metrics.json", {"rows": near_rows})
    _write_csv(output_dir / "candidate_spatial_enrichment.csv", spatial_rows)
    _write_json(output_dir / "candidate_spatial_enrichment.json", {"rows": spatial_rows, "outcome_region_meta": outcome_meta})

    cross_view_rows: List[Dict[str, Any]] = []
    for step in steps:
        for view_id in EVAL_OUTCOME_VIEWS:
            gain = next(
                (
                    row
                    for row in spatial_rows
                    if row["run"] == "BND-K1"
                    and int(row["nominal_step"]) == int(step)
                    and row["view_id"] == view_id
                    and row["candidate_group"] == "DEPTH_ADDED"
                    and row["region"] == "HJ_GAIN"
                ),
                None,
            )
            harm = next(
                (
                    row
                    for row in spatial_rows
                    if row["run"] == "BND-K1"
                    and int(row["nominal_step"]) == int(step)
                    and row["view_id"] == view_id
                    and row["candidate_group"] == "DEPTH_ADDED"
                    and row["region"] == "HJ_HARM"
                ),
                None,
            )
            if gain and harm:
                g = float(gain["CENTER_ENRICHMENT"])
                h = float(harm["CENTER_ENRICHMENT"])
                cross_view_rows.append(
                    {
                        "scene": SCENE,
                        "nominal_step": step,
                        "view_id": view_id,
                        "DEPTH_ADDED_HJ_GAIN_CENTER_ENRICHMENT": g,
                        "DEPTH_ADDED_HJ_HARM_CENTER_ENRICHMENT": h,
                        "ADDED_GAIN_VS_HARM_RATIO": g / max(h, EPS) if math.isfinite(g) and math.isfinite(h) else float("nan"),
                        "direction_gain_gt_harm": bool(g > h) if math.isfinite(g) and math.isfinite(h) else "",
                    }
                )
    _write_csv(output_dir / "candidate_cross_view_metrics.csv", cross_view_rows)
    _write_json(output_dir / "candidate_cross_view_metrics.json", {"rows": cross_view_rows})
    _write_csv(output_dir / "parameter_safety.csv", safety_rows)
    _write_json(output_dir / "parameter_safety.json", {"rows": safety_rows, "AUDIT_PARAMETER_SAFETY": "PASS"})

    pruning_context = {
        "fixed_state_current_loss_can_affect_pruning_gate": False,
        "reason": "Pruning gates use opacity, 3D scale, optional screen size, and optional extra split mask; no current loss gradient is read.",
        "historical_prune_differences_available": historical["HISTORICAL_PRUNE_COUNTS_AVAILABLE"],
    }
    _write_json(output_dir / "pruning_context.json", pruning_context)
    _write_csv(output_dir / "pruning_context.csv", [pruning_context])
    _write_json(output_dir / "historical_trigger_corroboration.json", {"HISTORICAL_TRIGGER_CORROBORATION": "NOT_AVAILABLE", "reason": historical["note"]})
    _write_csv(output_dir / "historical_trigger_corroboration.csv", [{"HISTORICAL_TRIGGER_CORROBORATION": "NOT_AVAILABLE", "reason": historical["note"]}])

    classification = _classify(summaries, spatial_rows, source_depth_reaches)
    _write_json(output_dir / "mechanism_classification.json", classification)
    temporal_summary = {
        "HIGHJ_RECOVERY_ONSET_STEP": 5000,
        "GLOBAL_RECOVERY_ONSET_STEP": 8000,
        "summaries": summaries,
        **classification,
    }
    _write_json(output_dir / "trigger_temporal_summary.json", temporal_summary)
    _write_csv(output_dir / "trigger_temporal_summary.csv", summaries)
    _write_json(output_dir / "densify_trigger_final_summary.json", {**classification, "summaries": summaries})
    _write_csv(output_dir / "densify_trigger_final_summary.csv", summaries)

    _plot_score_distribution(render_dir / "plot_trigger_score_distributions.png", score_plot_inputs, render_manifest)
    _plot_counts(render_dir / "plot_eligibility_counts.png", summaries, render_manifest)
    _plot_candidate_type(render_dir / "plot_candidate_type_counts.png", all_trigger_rows, render_manifest)
    if projected_map_rows:
        _save_sheet(render_dir / "contact_sheet_projected_candidate_maps.png", projected_map_rows, render_manifest, "projected_candidate_maps", EVAL_OUTCOME_VIEWS)
    if strong_map_rows:
        _save_sheet(render_dir / "contact_sheet_strong_gain_overlay.png", strong_map_rows, render_manifest, "strong_gain_overlay", EVAL_OUTCOME_VIEWS)
    temporal_lines = [
        f"{row['run']}@{row['nominal_step']}: eligible R={row['N_eligible_R']} RD={row['N_eligible_RD']} "
        f"added={row['N_depth_added']} removed={row['N_depth_removed']} ratio={float(row['ELIGIBLE_COUNT_RATIO']):.6f}"
        for row in summaries
        if row["run"] == "BND-K1"
    ]
    temporal_lines.extend(
        [
            f"PRE_RECOVERY_TRIGGER_REDISTRIBUTION={classification['PRE_RECOVERY_TRIGGER_REDISTRIBUTION']}",
            f"SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT={classification['SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT']}",
            f"Mechanism Classification={classification['Mechanism Classification']}",
        ]
    )
    _text_visual(render_dir / "contact_sheet_temporal_summary.png", "CDEPTH-DENSIFY-TRIGGER temporal summary", temporal_lines, render_manifest, "temporal_summary")
    causal_lines = [
        "Association chain only; causality is not claimed in this audit.",
        f"Source gradient reaches trigger statistic: {classification['DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC']}",
        f"Pre-recovery trigger redistribution: {classification['PRE_RECOVERY_TRIGGER_REDISTRIBUTION']}",
        "High-J recovery onset reference: 5000",
        "Global recovery onset reference: 8000",
    ]
    _text_visual(render_dir / "contact_sheet_compact_causal_chain.png", "Compact trigger-response chain", causal_lines, render_manifest, "compact_causal_chain")

    _write_csv(render_dir / "manifest.csv", render_manifest)
    _write_json(render_dir / "manifest.json", {"rows": render_manifest})
    _write_visual_index(render_dir / "VISUAL_COMPARE_INDEX.md", render_dir, render_manifest)
    output_manifest = {
        "repo_manifest": str(output_dir / "repo_manifest.json"),
        "source_audit": str(output_dir / "densification_source_audit.json"),
        "historical_state": str(output_dir / "historical_state_availability.json"),
        "camera_bank": str(output_dir / "fixed_trigger_camera_bank.json"),
        "trigger_all": str(output_dir / "trigger_response_all.json"),
        "candidate_flip_counts": str(output_dir / "candidate_flip_counts.json"),
        "candidate_type_metrics": str(output_dir / "candidate_type_metrics.json"),
        "candidate_attribute_metrics": str(output_dir / "candidate_attribute_metrics.json"),
        "near_threshold_metrics": str(output_dir / "near_threshold_metrics.json"),
        "candidate_spatial_enrichment": str(output_dir / "candidate_spatial_enrichment.json"),
        "candidate_cross_view_metrics": str(output_dir / "candidate_cross_view_metrics.json"),
        "trigger_temporal_summary": str(output_dir / "trigger_temporal_summary.json"),
        "mechanism_classification": str(output_dir / "mechanism_classification.json"),
        "render_manifest": str(render_dir / "manifest.json"),
        "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
    }
    _write_json(output_dir / "manifest.json", output_manifest)
    _write_csv(output_dir / "manifest.csv", [{"name": key, "path": value} for key, value in output_manifest.items()])

    _write_research_note(
        repo / RESEARCH_NOTE,
        repo_manifest,
        source,
        historical,
        camera_bank,
        summaries,
        classification,
        output_manifest,
    )
    return {
        "repo_manifest": repo_manifest,
        "source": source,
        "historical": historical,
        "camera_bank": camera_bank,
        "summaries": summaries,
        "classification": classification,
        "manifest": output_manifest,
        "render_manifest": render_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=list(TARGET_STEPS))
    parser.add_argument("--runs", nargs="+", default=["BND-K1", "CDEPTH"], choices=sorted(RUNS))
    parser.add_argument("--camera-count", type=int, default=0, help="0 means all Panama training views.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.repo.resolve(), args)
    print(json.dumps(result["classification"], indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
