#!/usr/bin/env python
"""Bounded hard-region observability audit for Panama.

This diagnostic is intentionally read-only. It loads existing Panama M1 and
BND-K1 checkpoints, builds offline labels, evaluates training-deployable
candidate signals, locks at most one proxy using training views only, and then
reports held-out metrics. It never calls optimizer.step(), scheduler.step(),
refinement_after(), split, duplicate, prune, or opacity reset.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_densify_trigger as densify_audit
from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_hardness_panama_20260811")
RENDER_DIR = Path("renders/bnd_hardness_panama_20260811")
LOG_DIR = Path("logs/bnd_hardness_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_HARD_REGION_OBSERVABILITY_2026-08-11.md")

M1_CONFIG = Path(
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
    "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)
K1_CONFIG = cdepth_setup.K1_CONFIG

EARLY_STEPS = (1000, 3000, 5000)
LATE_STEPS = (8000, 10000, 13000, 15000)
ALL_K1_STEPS = tuple(dict.fromkeys((*EARLY_STEPS, *LATE_STEPS)))
FINAL_NOMINAL_STEP = 15000

TRAIN_DEVELOPMENT_VIEWS = (
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
HELDOUT_EVAL_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")

LABELS = ("PERSISTENT_BND_HARD", "M1_HIGH_J", "BND_HARD_CORE")
TOPK_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12


@dataclass
class LoadedRun:
    run: str
    nominal_step: int
    actual_step: int
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any

    @property
    def model(self) -> Any:
        return self.pipeline.model


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
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


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
    if nominal_step == FINAL_NOMINAL_STEP and 14999 in steps:
        return 14999
    if not steps:
        return None
    nearest = min(steps, key=lambda step: abs(step - nominal_step))
    if abs(nearest - nominal_step) <= 1:
        return nearest
    return None


def _checkpoint_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    if run == "M1":
        config_path = repo / M1_CONFIG
        param_mode = "legacy"
    elif run == "BND-K1":
        config_path = repo / K1_CONFIG
        param_mode = "bounded_sh3"
    else:
        raise ValueError(run)
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing {run} checkpoint for nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        config.pipeline.model.intrinsic_color_parameterization = param_mode
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
    model.config.intrinsic_color_parameterization = param_mode
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.step = int(loaded_step)
    pipeline.eval()
    return LoadedRun(run, nominal_step, int(actual_step), config_path, Path(checkpoint_path), int(loaded_step), config, pipeline)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _train_records(pipeline: Any) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = pipeline.datamanager.train_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    cameras = dataset.cameras.to(pipeline.model.device)
    out: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    for index in range(len(dataset)):
        filename = image_filenames[index] if index < len(image_filenames) else Path(f"train_{index}")
        view_id = Path(filename).stem
        batch = pipeline.datamanager.cached_train[index].copy()
        out.append((index, view_id, cameras[index : index + 1], _batch_to_device(batch, pipeline.model.device)))
    return out


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    out: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        view_id = Path(filename).stem
        out.append((eval_index, view_id, camera, _batch_to_device(batch, pipeline.model.device)))
    return out


def _safe_outputs(outputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    keys = ("pred_image", "background", "accumulation", "clear_object_fullsh_raw", "tau_D", "transmission")
    return {key: outputs[key].detach().float().cpu() for key in keys if key in outputs and isinstance(outputs[key], Tensor)}


def _gt_for(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device)).detach().float().cpu()


def _render_records(loaded: LoadedRun, records: Sequence[Tuple[int, str, Any, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    model = loaded.model
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
            "brightness": gt.mean(dim=-1),
            "tau": safe.get("tau_D", torch.zeros_like(pred)).mean(dim=-1),
            "transmission": safe.get("transmission", torch.zeros_like(pred)).mean(dim=-1),
        }
    return out


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None


def _formal_loss(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Tensor:
    losses = model.get_loss_dict(outputs, batch, {})
    return losses["main_loss"]


def _responsibility_maps(loaded: LoadedRun, records: Sequence[Tuple[int, str, Any, Dict[str, Any]]]) -> Dict[str, Tensor]:
    model = loaded.model
    model.eval()
    maps: Dict[str, Tensor] = {}
    for _idx, view_id, camera, batch in records:
        _zero_grad(model)
        outputs = model.get_outputs(camera.to(model.device))
        pred = outputs["pred_image"]
        loss = _formal_loss(model, outputs, batch)
        grad = torch.autograd.grad(loss, pred, retain_graph=False, allow_unused=False)[0]
        maps[view_id] = torch.linalg.norm(grad.detach().float(), dim=-1).cpu()
        _zero_grad(model)
        del outputs, pred, loss, grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return maps


def _dc_clear_maps(loaded: LoadedRun, records: Sequence[Tuple[int, str, Any, Dict[str, Any]]]) -> Dict[str, Tensor]:
    model = loaded.model
    model.eval()
    saved = model.features_rest.detach().clone()
    maps: Dict[str, Tensor] = {}
    try:
        with torch.no_grad():
            model.features_rest.zero_()
            for _idx, view_id, camera, _batch in records:
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                maps[view_id] = outputs["clear_object_fullsh_raw"].detach().float().cpu()
    finally:
        with torch.no_grad():
            model.features_rest.copy_(saved)
        del saved
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return maps


def _parameter_snapshot(model: Any) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        out[name] = param.detach().cpu().clone()
    return out


def _parameter_delta_rows(before: Mapping[str, Tensor], model: Any, run: str, step: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    after_params = dict(model.named_parameters())
    for name, tensor in before.items():
        if name not in after_params:
            rows.append(
                {
                    "run": run,
                    "nominal_step": step,
                    "parameter": name,
                    "max_abs_delta": float("nan"),
                    "mean_abs_delta": float("nan"),
                    "missing_after": True,
                }
            )
            continue
        after = after_params[name].detach().cpu()
        diff = (after - tensor).abs()
        rows.append(
            {
                "run": run,
                "nominal_step": step,
                "parameter": name,
                "max_abs_delta": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs_delta": float(diff.mean().item()) if diff.numel() else 0.0,
                "missing_after": False,
            }
        )
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
    min_value = clean[finite].min()
    clean[~finite] = min_value
    order = torch.argsort(clean)
    ranks = torch.empty_like(clean, dtype=torch.float32)
    if clean.numel() == 1:
        ranks[0] = 1.0
    else:
        ranks[order] = torch.linspace(0.0, 1.0, clean.numel(), dtype=torch.float32)
    out[mask] = ranks
    return out


def _top_mask(values: Tensor, domain: Tensor, fraction: float) -> Tensor:
    mask = domain.detach().bool().cpu()
    out = torch.zeros_like(mask, dtype=torch.bool)
    vals = values.detach().float().cpu()[mask]
    if vals.numel() == 0:
        return out
    k = max(1, int(math.ceil(float(fraction) * vals.numel())))
    indices = torch.topk(vals, k, largest=True).indices
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    out.reshape(-1)[flat_indices[indices]] = True
    return out


def _ap_auc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n = len(scores)
    pos = int(labels.sum())
    neg = n - pos
    if n == 0 or pos == 0:
        return float("nan"), float("nan")
    order_desc = np.argsort(-scores, kind="mergesort")
    y = labels[order_desc].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, n + 1, dtype=np.float64)
    ap = float((precision * y).sum() / max(pos, 1))
    if neg == 0:
        return ap, float("nan")
    order_asc = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order_asc]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order_asc[start:end]] = avg_rank
        start = end
    sum_pos_ranks = float(ranks[labels].sum())
    auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return ap, float(auc)


def _metric_from_arrays(scores: np.ndarray, labels: np.ndarray, top_fracs: Sequence[float]) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n = int(labels.size)
    positives = int(labels.sum())
    prevalence = positives / max(n, 1)
    if n == 0 or positives == 0:
        order = np.asarray([], dtype=np.int64)
        ap = float("nan")
        auc = float("nan")
    else:
        order = np.argsort(-scores, kind="mergesort")
        y = labels[order].astype(np.float64)
        tp = np.cumsum(y)
        precision_curve = tp / np.arange(1, n + 1, dtype=np.float64)
        ap = float((precision_curve * y).sum() / max(positives, 1))
        negatives = n - positives
        if negatives == 0:
            auc = float("nan")
        else:
            neg_seen = np.cumsum(1.0 - y)
            pos_neg_below = float(((negatives - neg_seen) * y).sum())
            auc = pos_neg_below / max(float(positives * negatives), 1.0)
    row: Dict[str, Any] = {
        "pixel_count": n,
        "positive_count": positives,
        "label_prevalence": prevalence,
        "AUPRC": ap,
        "AUROC": auc,
        "AP_LIFT": ap / max(prevalence, EPS) if math.isfinite(ap) else float("nan"),
    }
    if n == 0:
        for frac in top_fracs:
            suffix = f"{int(frac * 100):02d}"
            row[f"precision_at_{suffix}"] = float("nan")
            row[f"recall_at_{suffix}"] = float("nan")
            row[f"enrichment_at_{suffix}"] = float("nan")
        return row
    for frac in top_fracs:
        suffix = f"{int(frac * 100):02d}"
        k = max(1, int(math.ceil(frac * n)))
        chosen = labels[order[:k]]
        precision = float(chosen.mean()) if k else float("nan")
        recall = float(chosen.sum() / max(positives, 1))
        row[f"precision_at_{suffix}"] = precision
        row[f"recall_at_{suffix}"] = recall
        row[f"enrichment_at_{suffix}"] = precision / max(prevalence, EPS)
    return row


def _metrics_for_signal(
    split: str,
    step: int,
    signal: str,
    label_name: str,
    signal_maps: Mapping[str, Tensor],
    labels: Mapping[str, Mapping[str, Tensor]],
    domains: Mapping[str, Tensor],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled_scores: List[np.ndarray] = []
    pooled_labels: List[np.ndarray] = []
    for view_id in view_ids:
        domain = domains[view_id].detach().bool().cpu()
        ranked = _rank_map(signal_maps[view_id], domain)
        label = labels[view_id][label_name].detach().bool().cpu()
        scores = ranked[domain].numpy()
        target = label[domain].numpy()
        row = {
            "split": split,
            "nominal_step": step,
            "signal": signal,
            "label": label_name,
            "view_id": view_id,
        }
        row.update(_metric_from_arrays(scores, target, TOPK_FRACTIONS))
        rows.append(row)
        pooled_scores.append(scores)
        pooled_labels.append(target)
    if pooled_scores:
        pooled_score = np.concatenate(pooled_scores)
        pooled_label = np.concatenate(pooled_labels)
    else:
        pooled_score = np.asarray([], dtype=np.float64)
        pooled_label = np.asarray([], dtype=np.bool_)
    row = {
        "split": split,
        "nominal_step": step,
        "signal": signal,
        "label": label_name,
        "view_id": "ALL",
    }
    row.update(_metric_from_arrays(pooled_score, pooled_label, TOPK_FRACTIONS))
    rows.append(row)
    return rows


def _metrics_for_signal_labels(
    split: str,
    step: int,
    signal: str,
    signal_maps: Mapping[str, Tensor],
    labels: Mapping[str, Mapping[str, Tensor]],
    domains: Mapping[str, Tensor],
    view_ids: Sequence[str],
    label_names: Sequence[str] = LABELS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pooled_scores: Dict[str, List[np.ndarray]] = {label_name: [] for label_name in label_names}
    pooled_labels: Dict[str, List[np.ndarray]] = {label_name: [] for label_name in label_names}
    for view_id in view_ids:
        domain = domains[view_id].detach().bool().cpu()
        ranked = _rank_map(signal_maps[view_id], domain)
        scores = ranked[domain].numpy()
        for label_name in label_names:
            label = labels[view_id][label_name].detach().bool().cpu()
            target = label[domain].numpy()
            row = {
                "split": split,
                "nominal_step": step,
                "signal": signal,
                "label": label_name,
                "view_id": view_id,
            }
            row.update(_metric_from_arrays(scores, target, TOPK_FRACTIONS))
            rows.append(row)
            pooled_scores[label_name].append(scores)
            pooled_labels[label_name].append(target)
    for label_name in label_names:
        pooled_score = np.concatenate(pooled_scores[label_name]) if pooled_scores[label_name] else np.asarray([], dtype=np.float64)
        pooled_label = np.concatenate(pooled_labels[label_name]) if pooled_labels[label_name] else np.asarray([], dtype=np.bool_)
        row = {
            "split": split,
            "nominal_step": step,
            "signal": signal,
            "label": label_name,
            "view_id": "ALL",
        }
        row.update(_metric_from_arrays(pooled_score, pooled_label, TOPK_FRACTIONS))
        rows.append(row)
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    ar = _rank_np(a)
    br = _rank_np(b)
    if np.std(ar) < EPS or np.std(br) < EPS:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _rank_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _accumulate_fixed_bank_refinement(loaded: LoadedRun, train_records: Sequence[Tuple[int, str, Any, Dict[str, Any]]]) -> Dict[str, Any]:
    trigger, _rows = densify_audit._accumulate_trigger_condition(loaded.model, train_records, "R")
    loaded.model.eval()
    eligible = densify_audit._eligibility(loaded.model, trigger)
    return {
        "score": trigger["score"].detach().float().cpu(),
        "eligible": eligible["eligible"].detach().bool().cpu(),
        "visible_any": trigger["visible_any"].detach().bool().cpu(),
        "vis_counts": trigger["vis_counts"].detach().float().cpu(),
        "source": "FIXED_BANK_REFINEMENT_SCORE",
    }


def _projected_center_window_map(model: Any, camera: Any, gaussian_score: Tensor, window: int = 16) -> Tuple[Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
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
    score_clean = torch.nan_to_num(gaussian_score.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    score_rank = _rank_np(score_clean.numpy())
    if score_rank.size > 1:
        score_rank = score_rank / max(score_rank.size - 1, 1)
    score_rank_t = torch.from_numpy(score_rank.astype(np.float32))
    xi = xy[:, 0].round().long().clamp(0, width - 1)
    yi = xy[:, 1].round().long().clamp(0, height - 1)
    flat_index = yi[visible] * width + xi[visible]
    score_image = torch.zeros(height * width, dtype=torch.float32)
    count_image = torch.zeros(height * width, dtype=torch.float32)
    score_image.scatter_add_(0, flat_index, score_rank_t[visible])
    count_image.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.float32))
    score_image = score_image.reshape(1, 1, height, width)
    count_image = count_image.reshape(1, 1, height, width)
    kernel = torch.ones(1, 1, window, window, dtype=torch.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    score_sum = F.conv2d(F.pad(score_image, (pad_left, pad_right, pad_left, pad_right), mode="constant", value=0.0), kernel)
    count_sum = F.conv2d(F.pad(count_image, (pad_left, pad_right, pad_left, pad_right), mode="constant", value=0.0), kernel)
    mean_score = score_sum[0, 0] / count_sum[0, 0].clamp_min(1.0)
    return mean_score, count_sum[0, 0]


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (torch.nan_to_num(image.detach().float(), nan=0.0).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: Optional[float] = None) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float().cpu(), nan=0.0)
    if scale is None:
        scale = 1.0
    arr = (vals / max(float(scale), EPS)).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _mask_to_rgb(mask: Tensor, color: Tuple[int, int, int] = (255, 80, 40)) -> Image.Image:
    mask_np = mask.detach().bool().cpu().numpy()
    arr = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    arr[mask_np] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _overlay_mask(base: Tensor, mask: Tensor, color: Tuple[int, int, int]) -> Image.Image:
    image = _gray_to_uint8(base, 1.0).convert("RGB")
    arr = np.array(image).astype(np.float32)
    mask_np = mask.detach().bool().cpu().numpy()
    arr[mask_np] = 0.45 * arr[mask_np] + 0.55 * np.array(color, dtype=np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if image.width != width:
        height = max(1, round(image.height * width / max(1, image.width)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_h = 30
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image, (0, label_h))
    ImageDraw.Draw(out).text((6, 8), label, fill=(0, 0, 0))
    return out


def _save_sheet(
    path: Path,
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    manifest: List[Dict[str, Any]],
    output_type: str,
    view_ids: Sequence[str],
    tile_width: int = 260,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 6 * max(0, len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered.append(canvas)
    if not rendered:
        return
    sheet = Image.new("RGB", (max(row.width for row in rendered), sum(row.height for row in rendered) + 6 * max(0, len(rendered) - 1)), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type, "view_ids": ";".join(view_ids), "width": sheet.width, "height": sheet.height})


def _plot_lines(path: Path, rows: Sequence[Mapping[str, Any]], title: str, y_key: str, group_key: str = "signal") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    groups = sorted({str(row[group_key]) for row in rows if row.get("view_id") == "ALL" and row.get("label") == "PERSISTENT_BND_HARD"})
    for group in groups:
        selected = [row for row in rows if row.get("view_id") == "ALL" and row.get("label") == "PERSISTENT_BND_HARD" and str(row[group_key]) == group]
        selected = sorted(selected, key=lambda row: int(row["nominal_step"]))
        if selected:
            plt.plot([int(r["nominal_step"]) for r in selected], [float(r[y_key]) for r in selected], marker="o", label=group)
    plt.title(title)
    plt.xlabel("K1 nominal step")
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_text_sheet(path: Path, title: str, lines: Sequence[str], manifest: List[Dict[str, Any]], output_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1300
    height = max(400, 46 + 22 * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill=(0, 0, 0))
    y = 54
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 22
    image.save(path)
    manifest.append({"file_path": str(path), "output_type": output_type, "width": width, "height": height})


def _write_manifest(output_dir: Path, render_dir: Path, render_manifest: Sequence[Mapping[str, Any]]) -> None:
    output_rows = [
        {"file_path": str(path), "kind": "output_file", "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("*"))
        if path.is_file()
    ]
    _write_json(output_dir / "manifest.json", {"rows": output_rows})
    _write_json(render_dir / "manifest.json", {"rows": list(render_manifest)})
    lines = ["# BND-HARDNESS Visual Compare Index", ""]
    for row in render_manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _label_prevalence_rows(
    labels: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    domains: Mapping[str, Mapping[str, Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, view_map in labels.items():
        pooled: Dict[str, float] = {name: 0.0 for name in LABELS}
        pooled_total = 0.0
        for view_id, label_map in view_map.items():
            domain = domains[split][view_id].bool()
            total = float(domain.sum().item())
            pooled_total += total
            for label_name in LABELS:
                count = float((label_map[label_name] & domain).sum().item())
                pooled[label_name] += count
                rows.append(
                    {
                        "split": split,
                        "view_id": view_id,
                        "label": label_name,
                        "pixel_count": int(count),
                        "domain_pixel_count": int(total),
                        "prevalence": count / max(total, EPS),
                        "sparse": bool(label_name == "BND_HARD_CORE" and count / max(total, EPS) < 0.005),
                    }
                )
        for label_name in LABELS:
            rows.append(
                {
                    "split": split,
                    "view_id": "ALL",
                    "label": label_name,
                    "pixel_count": int(pooled[label_name]),
                    "domain_pixel_count": int(pooled_total),
                    "prevalence": pooled[label_name] / max(pooled_total, EPS),
                    "sparse": bool(label_name == "BND_HARD_CORE" and pooled[label_name] / max(pooled_total, EPS) < 0.005),
                }
            )
    return rows


def _view_count_rows(metric_rows: Sequence[Mapping[str, Any]], split: str, step: int, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    signals = sorted({str(row["signal"]) for row in metric_rows if row["split"] == split and int(row["nominal_step"]) == step and row["label"] == label})
    for signal in signals:
        selected = [
            row
            for row in metric_rows
            if row["split"] == split
            and int(row["nominal_step"]) == step
            and row["label"] == label
            and row["signal"] == signal
            and row["view_id"] != "ALL"
        ]
        vals = [float(row["enrichment_at_10"]) for row in selected if math.isfinite(float(row["enrichment_at_10"]))]
        rows.append(
            {
                "split": split,
                "nominal_step": step,
                "label": label,
                "signal": signal,
                "n_views": len(vals),
                "n_views_enrich_gt_1": sum(v > 1.0 for v in vals),
                "n_views_enrich_ge_1p5": sum(v >= 1.5 for v in vals),
                "n_views_enrich_ge_2": sum(v >= 2.0 for v in vals),
            }
        )
    return rows


def _lookup_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    step: int,
    signal: str,
    label: str,
    view_id: str = "ALL",
) -> Optional[Mapping[str, Any]]:
    return next(
        (
            row
            for row in rows
            if row["split"] == split
            and int(row["nominal_step"]) == step
            and row["signal"] == signal
            and row["label"] == label
            and row["view_id"] == view_id
        ),
        None,
    )


def _build_scorecard(metric_rows: Sequence[Mapping[str, Any]], signal_availability: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    score_rows: List[Dict[str, Any]] = []
    count_rows = _view_count_rows(metric_rows, "train", 3000, "PERSISTENT_BND_HARD")
    count_lookup = {row["signal"]: row for row in count_rows}
    for signal, availability in signal_availability.items():
        if availability["availability"] == "NOT_EVALUABLE":
            score_rows.append({"signal": signal, "training_score": "NOT_EVALUABLE", "reason": availability.get("reason", "")})
            continue
        row3 = _lookup_metric(metric_rows, split="train", step=3000, signal=signal, label="PERSISTENT_BND_HARD")
        row5 = _lookup_metric(metric_rows, split="train", step=5000, signal=signal, label="PERSISTENT_BND_HARD")
        counts = count_lookup.get(signal, {})
        enrich3 = float(row3["enrichment_at_10"]) if row3 else float("nan")
        aplift3 = float(row3["AP_LIFT"]) if row3 else float("nan")
        enrich5 = float(row5["enrichment_at_10"]) if row5 else float("nan")
        aplift5 = float(row5["AP_LIFT"]) if row5 else float("nan")
        views_gt1 = int(counts.get("n_views_enrich_gt_1", 0))
        if enrich3 >= 2.0 and aplift3 >= 1.75 and views_gt1 >= 10 and enrich5 > 1.0 and aplift5 > 1.0:
            category = "STRONG_TRAIN"
        elif enrich3 >= 1.5 and aplift3 >= 1.35 and views_gt1 >= 9:
            category = "MODERATE_TRAIN"
        elif enrich3 < 1.0 and aplift3 < 1.0:
            category = "EVIDENCE_AGAINST"
        else:
            category = "WEAK_TRAIN"
        score_rows.append(
            {
                "signal": signal,
                "training_score": category,
                "train_3k_enrich_at_10": enrich3,
                "train_3k_AP_LIFT": aplift3,
                "train_5k_enrich_at_10": enrich5,
                "train_5k_AP_LIFT": aplift5,
                "train_3k_views_enrich_gt_1": views_gt1,
                "EARLY_PREDICTIVE": bool(enrich3 >= 1.5 and enrich5 > 1.0),
            }
        )
    return score_rows


def _pooled_rank_values(
    signal_maps: Mapping[str, Tensor],
    domains: Mapping[str, Tensor],
    view_ids: Sequence[str],
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for view_id in view_ids:
        domain = domains[view_id].bool()
        chunks.append(_rank_map(signal_maps[view_id], domain)[domain].numpy())
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)


def _signal_redundancy_rows(
    signal_maps_by_step_split: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Tensor]]]],
    domains: Mapping[str, Mapping[str, Tensor]],
    signals: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for signal in signals:
        maps = signal_maps_by_step_split[3000]["train"].get(signal)
        if maps:
            arrays[signal] = _pooled_rank_values(maps, domains["train"], TRAIN_DEVELOPMENT_VIEWS)
    for a in signals:
        for b in signals:
            rho = _spearman(arrays[a], arrays[b]) if a in arrays and b in arrays else float("nan")
            rows.append({"nominal_step": 3000, "signal_a": a, "signal_b": b, "spearman_rho": rho, "strongly_redundant": bool(math.isfinite(rho) and abs(rho) >= 0.8 and a != b)})
    return rows


def _select_proxy(
    scorecard: Sequence[Mapping[str, Any]],
    redundancy_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    signal_availability: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = [row for row in scorecard if row.get("training_score") in ("STRONG_TRAIN", "MODERATE_TRAIN")]
    candidates = sorted(candidates, key=lambda row: (float(row.get("train_3k_enrich_at_10", -1)), float(row.get("train_3k_AP_LIFT", -1))), reverse=True)
    rho_lookup = {(row["signal_a"], row["signal_b"]): float(row["spearman_rho"]) for row in redundancy_rows}
    if len(candidates) >= 2:
        for idx, first in enumerate(candidates):
            for second in candidates[idx + 1 :]:
                if signal_availability[first["signal"]]["family"] == signal_availability[second["signal"]]["family"]:
                    continue
                rho = rho_lookup.get((first["signal"], second["signal"]), float("nan"))
                if math.isfinite(rho) and abs(rho) >= 0.8:
                    continue
                return {
                    "COMPOSITE_AVAILABLE": True,
                    "proxy_type": "COMPOSITE",
                    "signal_a": first["signal"],
                    "signal_b": second["signal"],
                    "formula": f"0.5 * percentile_rank({first['signal']}) + 0.5 * percentile_rank({second['signal']})",
                    "selection_stage": "training views only, K1@3k, PERSISTENT_BND_HARD target",
                    "selection_used_heldout": False,
                }
    best = candidates[0] if candidates else None
    if best is None:
        available = [row for row in scorecard if row.get("training_score") != "NOT_EVALUABLE"]
        available = sorted(available, key=lambda row: (float(row.get("train_3k_enrich_at_10", -1)), float(row.get("train_3k_AP_LIFT", -1))), reverse=True)
        best = available[0] if available else None
    if best is None:
        return {
            "COMPOSITE_AVAILABLE": False,
            "proxy_type": "NONE",
            "selection_stage": "training views only",
            "selection_used_heldout": False,
            "reason": "No evaluable training-deployable signal.",
        }
    return {
        "COMPOSITE_AVAILABLE": False,
        "proxy_type": "SINGLE_SIGNAL",
        "signal": best["signal"],
        "formula": f"percentile_rank({best['signal']})",
        "selection_stage": "training views only, K1@3k, PERSISTENT_BND_HARD target",
        "selection_used_heldout": False,
    }


def _add_locked_proxy_maps(
    locked: Mapping[str, Any],
    signal_maps_by_step_split: MutableMapping[int, MutableMapping[str, MutableMapping[str, Dict[str, Tensor]]]],
    domains: Mapping[str, Mapping[str, Tensor]],
) -> None:
    for step in EARLY_STEPS:
        for split in ("train", "eval"):
            if locked["proxy_type"] == "COMPOSITE":
                a = locked["signal_a"]
                b = locked["signal_b"]
                if a not in signal_maps_by_step_split[step][split] or b not in signal_maps_by_step_split[step][split]:
                    continue
                maps: Dict[str, Tensor] = {}
                view_ids = TRAIN_DEVELOPMENT_VIEWS if split == "train" else HELDOUT_EVAL_VIEWS
                for view_id in view_ids:
                    domain = domains[split][view_id]
                    maps[view_id] = 0.5 * _rank_map(signal_maps_by_step_split[step][split][a][view_id], domain) + 0.5 * _rank_map(signal_maps_by_step_split[step][split][b][view_id], domain)
                signal_maps_by_step_split[step][split]["S_LOCKED_PROXY"] = maps
            elif locked["proxy_type"] == "SINGLE_SIGNAL":
                signal = locked["signal"]
                if signal in signal_maps_by_step_split[step][split]:
                    signal_maps_by_step_split[step][split]["S_LOCKED_PROXY"] = signal_maps_by_step_split[step][split][signal]


def _heldout_cross_view(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == "eval"
        and int(row["nominal_step"]) == 3000
        and row["signal"] == "S_LOCKED_PROXY"
        and row["label"] == "PERSISTENT_BND_HARD"
        and row["view_id"] != "ALL"
    ]
    vals = [float(row["enrichment_at_10"]) for row in selected]
    pooled = _lookup_metric(rows, split="eval", step=3000, signal="S_LOCKED_PROXY", label="PERSISTENT_BND_HARD")
    pooled_enrich = float(pooled["enrichment_at_10"]) if pooled else float("nan")
    return {
        "HELDOUT_CROSS_VIEW_CONSISTENT": bool(sum(v > 1.0 for v in vals) >= 2 and pooled_enrich > 1.0),
        "n_views": len(vals),
        "n_views_enrich_gt_1": sum(v > 1.0 for v in vals),
        "n_views_enrich_ge_1p5": sum(v >= 1.5 for v in vals),
        "n_views_enrich_ge_2": sum(v >= 2.0 for v in vals),
        "heldout_pooled_enrich_at_10": pooled_enrich,
    }


def _compute_cost_for_proxy(locked: Mapping[str, Any], signal_availability: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    order = {
        "LOW_COST_ONLINE": 0,
        "MODERATE_ONLINE": 1,
        "PERIODIC_TRAIN_BANK": 2,
        "HIGH_COST_DIAGNOSTIC_ONLY": 3,
    }
    signals: List[str] = []
    if locked["proxy_type"] == "COMPOSITE":
        signals = [locked["signal_a"], locked["signal_b"]]
    elif locked["proxy_type"] == "SINGLE_SIGNAL":
        signals = [locked["signal"]]
    if not signals:
        return {"DEPLOYABLE_PROXY_AVAILABLE": False, "compute_cost_class": "NOT_EVALUABLE"}
    costs = [signal_availability[s]["compute_cost_class"] for s in signals]
    cost = max(costs, key=lambda item: order.get(item, 99))
    return {
        "DEPLOYABLE_PROXY_AVAILABLE": bool(cost != "HIGH_COST_DIAGNOSTIC_ONLY"),
        "compute_cost_class": cost,
        "requires_gt": any(signal_availability[s]["requires_gt"] for s in signals),
        "requires_backward": any(signal_availability[s]["requires_backward"] for s in signals),
        "requires_history": any(signal_availability[s]["requires_history"] for s in signals),
        "requires_train_bank_pass": any(signal_availability[s]["requires_train_bank_pass"] for s in signals),
        "requires_extra_render": any(signal_availability[s]["requires_extra_render"] for s in signals),
        "uses_M1": False,
        "uses_future_K1": False,
        "uses_eval_GT_for_training_trigger": False,
        "uses_pseudo_depth": False,
        "uses_CDEPTH": False,
        "uses_unbounded_J": False,
    }


def _classify_proxy(
    locked: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    cross: Mapping[str, Any],
    deploy: Mapping[str, Any],
) -> Dict[str, Any]:
    if locked["proxy_type"] == "NONE":
        return {"Final": "NOT_EVALUABLE", "reason": "No locked proxy."}
    train3 = _lookup_metric(metric_rows, split="train", step=3000, signal="S_LOCKED_PROXY", label="PERSISTENT_BND_HARD")
    held3 = _lookup_metric(metric_rows, split="eval", step=3000, signal="S_LOCKED_PROXY", label="PERSISTENT_BND_HARD")
    held5 = _lookup_metric(metric_rows, split="eval", step=5000, signal="S_LOCKED_PROXY", label="PERSISTENT_BND_HARD")
    if train3 is None or held3 is None:
        return {"Final": "NOT_EVALUABLE", "reason": "Missing train or held-out locked proxy metrics."}
    train_enrich = float(train3["enrichment_at_10"])
    train_aplift = float(train3["AP_LIFT"])
    held_enrich = float(held3["enrichment_at_10"])
    held_aplift = float(held3["AP_LIFT"])
    held5_enrich = float(held5["enrichment_at_10"]) if held5 else float("nan")
    strong = (
        bool(deploy.get("DEPLOYABLE_PROXY_AVAILABLE"))
        and train_enrich >= 2.0
        and train_aplift >= 1.75
        and held_enrich >= 1.75
        and int(cross.get("n_views_enrich_ge_1p5", 0)) >= 2
        and held_aplift >= 1.5
        and held5_enrich >= 1.5
        and bool(cross.get("HELDOUT_CROSS_VIEW_CONSISTENT"))
    )
    moderate = (
        bool(deploy.get("DEPLOYABLE_PROXY_AVAILABLE"))
        and held_enrich >= 1.25
        and held_aplift >= 1.20
        and int(cross.get("n_views_enrich_gt_1", 0)) >= 2
    )
    if strong:
        final = "DEPLOYABLE_HARDNESS_PROXY_STRONG"
    elif moderate:
        final = "DEPLOYABLE_HARDNESS_PROXY_MODERATE"
    elif held_enrich <= 1.05 and held_aplift <= 1.05:
        final = "NO_USEFUL_DEPLOYABLE_PROXY"
    else:
        final = "DEPLOYABLE_HARDNESS_PROXY_WEAK"
    m1 = _lookup_metric(metric_rows, split="eval", step=3000, signal="S_LOCKED_PROXY", label="M1_HIGH_J")
    core = _lookup_metric(metric_rows, split="eval", step=3000, signal="S_LOCKED_PROXY", label="BND_HARD_CORE")
    m1_enrich = float(m1["enrichment_at_10"]) if m1 else float("nan")
    core_enrich = float(core["enrichment_at_10"]) if core else float("nan")
    if held_enrich > 1.25 and m1_enrich > 1.25 and core_enrich > 1.25:
        semantic = "BOUND_SPECIFIC_HARDNESS_PROXY"
    elif held_enrich > 1.25:
        semantic = "GENERAL_RECONSTRUCTION_HARDNESS_PROXY"
    elif m1_enrich > 1.25 and held_enrich <= 1.25:
        semantic = "COMPENSATION_OBSERVABLE_ONLY"
    else:
        semantic = "NO_CLEAR_SEMANTIC_ALIGNMENT"
    return {
        "Final": final,
        "semantic": semantic,
        "train_3k_enrich_at_10": train_enrich,
        "train_3k_AP_LIFT": train_aplift,
        "heldout_3k_enrich_at_10": held_enrich,
        "heldout_3k_AP_LIFT": held_aplift,
        "heldout_5k_enrich_at_10": held5_enrich,
        "heldout_M1_HIGH_J_enrich_at_10": m1_enrich,
        "heldout_BND_HARD_CORE_enrich_at_10": core_enrich,
    }


def _overlap_rows(
    locked_maps: Mapping[str, Tensor],
    labels: Mapping[str, Mapping[str, Tensor]],
    domains: Mapping[str, Tensor],
    split: str,
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in view_ids:
        domain = domains[view_id]
        ranked = _rank_map(locked_maps[view_id], domain)
        top = _top_mask(ranked, domain, 0.10)
        for label_name in LABELS:
            label = labels[view_id][label_name] & domain
            inter = int((top & label).sum().item())
            union = int((top | label).sum().item())
            top_count = int(top.sum().item())
            label_count = int(label.sum().item())
            rows.append(
                {
                    "split": split,
                    "view_id": view_id,
                    "label": label_name,
                    "top10_count": top_count,
                    "label_count": label_count,
                    "intersection": inter,
                    "union": union,
                    "IoU": inter / max(union, 1),
                    "precision": inter / max(top_count, 1),
                    "recall": inter / max(label_count, 1),
                }
            )
    return rows


def _spatial_bias_rows(
    locked_maps: Mapping[str, Tensor],
    k1_maps: Mapping[str, Mapping[str, Tensor]],
    domains: Mapping[str, Tensor],
    split: str,
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in view_ids:
        domain = domains[view_id].bool()
        ranked = _rank_map(locked_maps[view_id], domain)
        top = _top_mask(ranked, domain, 0.10)
        h, w = top.shape
        yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
        brightness = k1_maps[view_id]["brightness"]
        bright = _top_mask(brightness, torch.ones_like(domain, dtype=torch.bool), 0.20)
        dark = _top_mask(-brightness, torch.ones_like(domain, dtype=torch.bool), 0.20)
        total = float(top.sum().item())
        row = {
            "split": split,
            "view_id": view_id,
            "top10_count": int(total),
            "fraction_top20_band": float((top & (yy <= 0.2)).sum().item()) / max(total, EPS),
            "fraction_bottom20_band": float((top & (yy >= 0.8)).sum().item()) / max(total, EPS),
            "fraction_bright_top20": float((top & bright).sum().item()) / max(total, EPS),
            "fraction_dark_top20": float((top & dark).sum().item()) / max(total, EPS),
            "fraction_final_object_support": float((top & domain).sum().item()) / max(total, EPS),
            "fraction_background": float((top & (~domain)).sum().item()) / max(total, EPS),
        }
        row["SPATIAL_BIAS_WARNING"] = bool(
            row["fraction_bottom20_band"] >= 0.80
            or row["fraction_top20_band"] >= 0.80
            or row["fraction_bright_top20"] >= 0.80
            or row["fraction_background"] >= 0.50
        )
        rows.append(row)
    return rows


def _proxy_specificity_audit(metric_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    weak = False
    for split in ("train", "eval"):
        proxy = _lookup_metric(
            metric_rows,
            split=split,
            step=3000,
            signal="S_LOCKED_PROXY",
            label="PERSISTENT_BND_HARD",
        )
        bright = _lookup_metric(
            metric_rows,
            split=split,
            step=3000,
            signal="S_BRIGHT_CONTROL",
            label="PERSISTENT_BND_HARD",
        )
        if proxy is None or bright is None:
            continue
        proxy_enrich = float(proxy["enrichment_at_10"])
        bright_enrich = float(bright["enrichment_at_10"])
        proxy_aplift = float(proxy["AP_LIFT"])
        bright_aplift = float(bright["AP_LIFT"])
        split_weak = proxy_enrich <= bright_enrich or proxy_aplift <= bright_aplift
        weak = weak or split_weak
        rows.append(
            {
                "split": split,
                "nominal_step": 3000,
                "label": "PERSISTENT_BND_HARD",
                "locked_proxy_enrich_at_10": proxy_enrich,
                "brightness_enrich_at_10": bright_enrich,
                "delta_enrich_at_10": proxy_enrich - bright_enrich,
                "locked_proxy_AP_LIFT": proxy_aplift,
                "brightness_AP_LIFT": bright_aplift,
                "delta_AP_LIFT": proxy_aplift - bright_aplift,
                "PROXY_SPECIFICITY_WEAK": split_weak,
            }
        )
    return {
        "PROXY_SPECIFICITY_WEAK": weak,
        "definition": "True when the locked proxy is not better than the simple brightness control on ENRICH@10 or AP_LIFT at K1@3k for PERSISTENT_BND_HARD.",
        "rows": rows,
    }


def _make_visuals(
    render_dir: Path,
    render_manifest: List[Dict[str, Any]],
    k1_maps: Mapping[int, Mapping[str, Mapping[str, Dict[str, Tensor]]]],
    labels: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    domains: Mapping[str, Mapping[str, Tensor]],
    signal_maps_by_step_split: Mapping[int, Mapping[str, Mapping[str, Dict[str, Tensor]]]],
    metric_rows: Sequence[Mapping[str, Any]],
    scorecard: Sequence[Mapping[str, Any]],
    redundancy_rows: Sequence[Mapping[str, Any]],
    locked: Mapping[str, Any],
    classification: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)

    # Label prevalence plot.
    pooled = [row for row in label_rows if row["view_id"] == "ALL"]
    plt.figure(figsize=(8, 4))
    names = [f"{row['split']} {row['label']}" for row in pooled]
    vals = [float(row["prevalence"]) for row in pooled]
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=45, ha="right")
    plt.ylabel("prevalence in final K1 object support")
    plt.tight_layout()
    path = render_dir / "plot_label_prevalence_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    render_manifest.append({"file_path": str(path), "output_type": "label_prevalence_summary"})

    # Signal sheets on held-out views.
    for step in EARLY_STEPS:
        rows = []
        for view_id in HELDOUT_EVAL_VIEWS:
            domain = domains["eval"][view_id]
            cols: List[Tuple[str, Image.Image]] = [(f"{view_id} GT", _rgb_to_uint8(k1_maps[step]["eval"][view_id]["gt"]))]
            for signal in ("S_RES_CURRENT", "S_RES_PERSIST", "S_SH", "S_BOUND", "S_RESP", "S_REFINE"):
                maps = signal_maps_by_step_split[step]["eval"].get(signal)
                if maps and view_id in maps:
                    cols.append((signal, _gray_to_uint8(_rank_map(maps[view_id], domain), 1.0)))
            rows.append(cols)
        _save_sheet(render_dir / f"contact_sheet_{step//1000}k_signal_maps.png", rows, render_manifest, f"{step//1000}k_signal_maps", HELDOUT_EVAL_VIEWS)

    # Label maps and core maps.
    label_sheet_rows = []
    core_sheet_rows = []
    for view_id in HELDOUT_EVAL_VIEWS:
        label_sheet_rows.append(
            [
                (f"{view_id} persistent", _mask_to_rgb(labels["eval"][view_id]["PERSISTENT_BND_HARD"], (255, 70, 40))),
                ("M1_HIGH_J", _mask_to_rgb(labels["eval"][view_id]["M1_HIGH_J"], (60, 140, 255))),
                ("BND_HARD_CORE", _mask_to_rgb(labels["eval"][view_id]["BND_HARD_CORE"], (255, 220, 40))),
            ]
        )
        core_sheet_rows.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(k1_maps[FINAL_NOMINAL_STEP]["eval"][view_id]["gt"])),
                ("persistent", _mask_to_rgb(labels["eval"][view_id]["PERSISTENT_BND_HARD"], (255, 70, 40))),
                ("core", _mask_to_rgb(labels["eval"][view_id]["BND_HARD_CORE"], (255, 220, 40))),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_persistent_hard_label_maps.png", label_sheet_rows, render_manifest, "persistent_hard_label_maps", HELDOUT_EVAL_VIEWS)
    _save_sheet(render_dir / "contact_sheet_bnd_hard_core_maps.png", core_sheet_rows, render_manifest, "bnd_hard_core_maps", HELDOUT_EVAL_VIEWS)

    _plot_lines(render_dir / "plot_signal_enrich10_trajectories.png", [r for r in metric_rows if r["split"] == "train"], "Training ENRICH@10 vs persistent hard", "enrichment_at_10")
    render_manifest.append({"file_path": str(render_dir / "plot_signal_enrich10_trajectories.png"), "output_type": "signal_enrich10_trajectories"})
    _plot_lines(render_dir / "plot_signal_aplift_trajectories.png", [r for r in metric_rows if r["split"] == "train"], "Training AP_LIFT vs persistent hard", "AP_LIFT")
    render_manifest.append({"file_path": str(render_dir / "plot_signal_aplift_trajectories.png"), "output_type": "signal_aplift_trajectories"})

    # Scorecard text.
    score_lines = [
        f"{row.get('signal')}: {row.get('training_score')} enrich3={row.get('train_3k_enrich_at_10')} AP3={row.get('train_3k_AP_LIFT')}"
        for row in scorecard
    ]
    _save_text_sheet(render_dir / "training_view_scorecard_sheet.png", "Training View Scorecard", score_lines, render_manifest, "training_view_scorecard")

    # Spearman matrix.
    signals = sorted({row["signal_a"] for row in redundancy_rows})
    matrix = np.full((len(signals), len(signals)), np.nan)
    lookup = {(row["signal_a"], row["signal_b"]): float(row["spearman_rho"]) for row in redundancy_rows}
    for i, a in enumerate(signals):
        for j, b in enumerate(signals):
            matrix[i, j] = lookup.get((a, b), np.nan)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(signals)), signals, rotation=45, ha="right")
    plt.yticks(range(len(signals)), signals)
    plt.tight_layout()
    path = render_dir / "plot_signal_spearman_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    render_manifest.append({"file_path": str(path), "output_type": "signal_spearman_matrix"})

    # Locked proxy and held-out per-view comparisons.
    if "S_LOCKED_PROXY" in signal_maps_by_step_split[3000]["eval"]:
        locked_maps = signal_maps_by_step_split[3000]["eval"]["S_LOCKED_PROXY"]
        rows = []
        for view_id in HELDOUT_EVAL_VIEWS:
            domain = domains["eval"][view_id]
            proxy_rank = _rank_map(locked_maps[view_id], domain)
            top = _top_mask(proxy_rank, domain, 0.10)
            rows.append(
                [
                    (f"{view_id} GT", _rgb_to_uint8(k1_maps[3000]["eval"][view_id]["gt"])),
                    ("K1@3k", _rgb_to_uint8(k1_maps[3000]["eval"][view_id]["pred"])),
                    ("locked proxy", _gray_to_uint8(proxy_rank, 1.0)),
                    ("proxy top10 + persistent", _overlay_mask(proxy_rank, labels["eval"][view_id]["PERSISTENT_BND_HARD"], (255, 70, 40))),
                    ("proxy top10", _mask_to_rgb(top, (255, 220, 40))),
                    ("BND_HARD_CORE", _mask_to_rgb(labels["eval"][view_id]["BND_HARD_CORE"], (255, 220, 40))),
                ]
            )
            _save_sheet(
                render_dir / f"heldout_{view_id}_comparison.png",
                [rows[-1]],
                render_manifest,
                f"heldout_{view_id}_comparison",
                [view_id],
            )
        _save_sheet(render_dir / "contact_sheet_locked_proxy_maps.png", rows, render_manifest, "locked_proxy_maps", HELDOUT_EVAL_VIEWS)
        overlay_rows = []
        for view_id in HELDOUT_EVAL_VIEWS:
            domain = domains["eval"][view_id]
            proxy_rank = _rank_map(locked_maps[view_id], domain)
            top = _top_mask(proxy_rank, domain, 0.10)
            overlay_rows.append(
                [
                    (f"{view_id} proxy", _gray_to_uint8(proxy_rank, 1.0)),
                    ("top10", _mask_to_rgb(top, (255, 220, 40))),
                    ("persistent", _mask_to_rgb(labels["eval"][view_id]["PERSISTENT_BND_HARD"], (255, 70, 40))),
                    ("M1_HIGH_J", _mask_to_rgb(labels["eval"][view_id]["M1_HIGH_J"], (60, 140, 255))),
                    ("core", _mask_to_rgb(labels["eval"][view_id]["BND_HARD_CORE"], (255, 80, 220))),
                ]
            )
        _save_sheet(render_dir / "contact_sheet_selected_proxy_top10_vs_labels.png", overlay_rows, render_manifest, "selected_proxy_top10_vs_labels", HELDOUT_EVAL_VIEWS)

    # Brightness control and final sheets.
    bright_rows = [row for row in metric_rows if row.get("signal") == "S_BRIGHT_CONTROL" and row.get("view_id") == "ALL" and row.get("label") == "PERSISTENT_BND_HARD"]
    _plot_lines(render_dir / "plot_brightness_control_comparison.png", bright_rows, "Brightness control ENRICH@10", "enrichment_at_10")
    render_manifest.append({"file_path": str(render_dir / "plot_brightness_control_comparison.png"), "output_type": "brightness_control_comparison"})

    temporal_lines = [
        f"{row.get('signal')}: early={row.get('EARLY_PREDICTIVE')} score={row.get('training_score')} 3k_enrich={row.get('train_3k_enrich_at_10')} 5k_enrich={row.get('train_5k_enrich_at_10')}"
        for row in scorecard
    ]
    _save_text_sheet(render_dir / "temporal_prediction_summary_sheet.png", "Temporal Prediction Summary", temporal_lines, render_manifest, "temporal_prediction_summary")
    final_lines = [
        f"Locked proxy: {locked}",
        f"Final classification: {classification.get('Final')}",
        f"Semantic classification: {classification.get('semantic')}",
        f"Held-out enrich@10: {classification.get('heldout_3k_enrich_at_10')}",
        f"Held-out AP_LIFT: {classification.get('heldout_3k_AP_LIFT')}",
    ]
    _save_text_sheet(render_dir / "final_deployability_next_step_summary_sheet.png", "BND-HARDNESS Final Summary", final_lines, render_manifest, "final_deployability_next_step_summary")


def _write_research_note(
    path: Path,
    repo_manifest: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    signal_availability: Mapping[str, Mapping[str, Any]],
    scorecard: Sequence[Mapping[str, Any]],
    locked: Mapping[str, Any],
    cross: Mapping[str, Any],
    deploy: Mapping[str, Any],
    classification: Mapping[str, Any],
    proxy_specificity: Mapping[str, Any],
) -> None:
    def pooled(split: str, label: str) -> str:
        row = next((r for r in label_rows if r["split"] == split and r["view_id"] == "ALL" and r["label"] == label), None)
        return "NA" if row is None else f"{float(row['prevalence']):.6f}"

    lines = [
        "# BND Hard-Region Observability Audit",
        "",
        "## Motivation",
        "",
        "CODE FACT: This is a read-only diagnostic audit. No training, optimizer step, scheduler step, checkpoint mutation, densification, split, duplicate, prune, opacity reset, new loss, or new model was executed.",
        "",
        "INFERENCE: CDEPTH is closed for the current study. The active question is bounded representation capacity under `bounded_sh3`, not another CDEPTH variant.",
        "",
        "## Repository State",
        "",
        f"- Branch: `{repo_manifest.get('branch')}`",
        f"- Start HEAD: `{repo_manifest.get('start_head')}`",
        f"- Initial status: `{repo_manifest.get('initial_status')}`",
        "",
        "## Checkpoints And Camera Split",
        "",
        "CONFIG FACT: BND-K1 checkpoints were audited at nominal 1k, 3k, 5k, 8k, 10k, 13k, and 15k. M1 final was used only for offline labels/context.",
        "",
        "| Run | Nominal | Actual | Exists |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in checkpoint_rows:
        lines.append(f"| {row['run']} | {row['nominal_step']} | {row['actual_step']} | {row['checkpoint_exists']} |")
    lines.extend(
        [
            "",
            f"CONFIG FACT: Training-development views: `{';'.join(TRAIN_DEVELOPMENT_VIEWS)}`.",
            f"CONFIG FACT: Held-out eval views: `{';'.join(HELDOUT_EVAL_VIEWS)}`.",
            "CONFIG FACT: `HELD_OUT_SELECTION_LEAKAGE = FALSE`; signal selection and proxy locking used training views only.",
            "",
            "## Offline Labels",
            "",
            "- `M1_HIGH_J`: M1 final accumulation > 0.01 and max RGB of M1 final `clear_object_fullsh_raw` > 1.0. Oracle diagnostic label only.",
            "- `PERSISTENT_BND_HARD`: K1 late residual top 10% inside final K1 object support for at least 75% of available late checkpoints. Future-outcome diagnostic label only.",
            "- `BND_HARD_CORE`: `PERSISTENT_BND_HARD AND M1_HIGH_J`. Oracle plus future diagnostic label only.",
            "",
            f"QUANTITATIVE RESULT: train pooled prevalence M1_HIGH_J `{pooled('train', 'M1_HIGH_J')}`, PERSISTENT_BND_HARD `{pooled('train', 'PERSISTENT_BND_HARD')}`, BND_HARD_CORE `{pooled('train', 'BND_HARD_CORE')}`.",
            f"QUANTITATIVE RESULT: held-out pooled prevalence M1_HIGH_J `{pooled('eval', 'M1_HIGH_J')}`, PERSISTENT_BND_HARD `{pooled('eval', 'PERSISTENT_BND_HARD')}`, BND_HARD_CORE `{pooled('eval', 'BND_HARD_CORE')}`.",
            "",
            "## Signal Availability",
            "",
            "| Signal | Semantics | Availability | Deployability | Compute cost |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for signal, row in signal_availability.items():
        lines.append(f"| {signal} | {row['semantics']} | {row['availability']} | {row['deployability_class']} | {row['compute_cost_class']} |")
    lines.extend(["", "## Training-Bank Prediction Results", "", "| Signal | Score | 3k ENRICH@10 | 3k AP_LIFT | 5k ENRICH@10 | Early predictive |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for row in scorecard:
        lines.append(
            f"| {row.get('signal')} | {row.get('training_score')} | {row.get('train_3k_enrich_at_10')} | {row.get('train_3k_AP_LIFT')} | {row.get('train_5k_enrich_at_10')} | {row.get('EARLY_PREDICTIVE')} |"
        )
    lines.extend(
        [
            "",
            "## Locked Proxy",
            "",
            f"CONFIG FACT: Locked proxy definition: `{locked}`.",
            "CONFIG FACT: Held-out views were not used for signal selection, formula choice, direction choice, or threshold tuning.",
            "",
            "## Held-Out Evaluation",
            "",
            f"QUANTITATIVE RESULT: `HELDOUT_CROSS_VIEW_CONSISTENT = {cross.get('HELDOUT_CROSS_VIEW_CONSISTENT')}`.",
            f"QUANTITATIVE RESULT: held-out views with ENRICH@10 > 1: `{cross.get('n_views_enrich_gt_1')}`; >=1.5: `{cross.get('n_views_enrich_ge_1p5')}`; >=2: `{cross.get('n_views_enrich_ge_2')}`.",
            "",
            "## Deployability",
            "",
            f"CONFIG FACT: `DEPLOYABLE_PROXY_AVAILABLE = {deploy.get('DEPLOYABLE_PROXY_AVAILABLE')}`.",
            f"CONFIG FACT: compute cost class `{deploy.get('compute_cost_class')}`.",
            f"CONFIG FACT: uses M1 `{deploy.get('uses_M1')}`, uses future K1 `{deploy.get('uses_future_K1')}`, uses eval GT for training trigger `{deploy.get('uses_eval_GT_for_training_trigger')}`, uses CDEPTH `{deploy.get('uses_CDEPTH')}`.",
            f"QUANTITATIVE RESULT: `PROXY_SPECIFICITY_WEAK = {proxy_specificity.get('PROXY_SPECIFICITY_WEAK')}` from the brightness-control comparison.",
            "",
            "## Formal Classification",
            "",
            f"QUANTITATIVE RESULT: `{classification.get('Final')}`.",
            f"QUANTITATIVE RESULT: semantic alignment `{classification.get('semantic')}`.",
            "",
            "## Scientific Interpretation",
            "",
            "INFERENCE: A deployable signal is only considered suitable for the next refinement stage if it predicts late persistent K1 error on held-out views and does not depend on M1 oracle labels or future K1 labels.",
        ]
    )
    final = str(classification.get("Final"))
    if final == "DEPLOYABLE_HARDNESS_PROXY_STRONG":
        next_step = "BND-AWARE-REFINE single-factor proxy-guided refinement causal test."
    elif final == "DEPLOYABLE_HARDNESS_PROXY_MODERATE":
        next_step = "One proxy robustness audit on a fixed additional validation setting; no training yet."
    else:
        next_step = "AA mechanism refinement, because deployable hard-region observability was not strong enough to justify targeted refinement."
    lines.extend(
        [
            "",
            "## Next Single-Factor Decision",
            "",
            f"PROPOSED NEXT STEP: {next_step}",
            "",
            "Visual assets are ready for external/manual analysis.",
            "No subjective clear-image correctness judgment was made.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(repo: Path) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "initial_status": _git(repo, "status", "--short"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "tracked_output_paths": _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints"),
        "tracked_output_count": len(
            [
                line
                for line in _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints").splitlines()
                if line.strip()
            ]
        ),
    }
    _write_json(OUTPUT_DIR / "repo_manifest.json", repo_manifest)

    checkpoint_rows: List[Dict[str, Any]] = []
    checkpoint_before: Dict[str, Dict[str, Any]] = {}
    for run_name, config_rel, steps in (
        ("M1", M1_CONFIG, (FINAL_NOMINAL_STEP,)),
        ("BND-K1", K1_CONFIG, ALL_K1_STEPS),
    ):
        config_path = repo / config_rel
        for nominal in steps:
            actual = _actual_step(config_path, nominal)
            ckpt_path = _available_steps(config_path).get(actual) if actual is not None else None
            row = {
                "scene": SCENE,
                "run": run_name,
                "nominal_step": nominal,
                "actual_step": actual if actual is not None else "",
                "config_path": str(config_path),
                "checkpoint_path": str(ckpt_path) if ckpt_path else "",
                "checkpoint_exists": bool(ckpt_path and ckpt_path.exists()),
            }
            checkpoint_rows.append(row)
            if ckpt_path and ckpt_path.exists():
                checkpoint_before[str(ckpt_path)] = _checkpoint_fingerprint(ckpt_path)
    _write_csv(OUTPUT_DIR / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(OUTPUT_DIR / "checkpoint_manifest.json", {"rows": checkpoint_rows})

    _write_json(
        OUTPUT_DIR / "camera_split_manifest.json",
        {
            "scene": SCENE,
            "train_development_views": list(TRAIN_DEVELOPMENT_VIEWS),
            "heldout_eval_views": list(HELDOUT_EVAL_VIEWS),
            "HELD_OUT_SELECTION_LEAKAGE": False,
        },
    )
    _write_json(
        OUTPUT_DIR / "label_definitions.json",
        {
            "M1_HIGH_J": "M1 final accumulation > 0.01 AND max_rgb(M1 final clear_object_fullsh_raw) > 1.0; oracle diagnostic label only.",
            "PERSISTENT_BND_HARD": "Per-view K1 late residual top 10 percent inside final K1 object support for at least 75 percent of available late checkpoints; future-outcome diagnostic label only.",
            "BND_HARD_CORE": "PERSISTENT_BND_HARD AND M1_HIGH_J; oracle plus future diagnostic label only.",
        },
    )
    print("[BND-HARDNESS] repo/checkpoint manifests written", flush=True)

    m1_maps: Dict[str, Dict[str, Dict[str, Tensor]]] = {"train": {}, "eval": {}}
    k1_maps: Dict[int, Dict[str, Dict[str, Dict[str, Tensor]]]] = {}
    signal_maps: Dict[int, Dict[str, Dict[str, Dict[str, Tensor]]]] = {step: {"train": {}, "eval": {}} for step in EARLY_STEPS}
    parameter_delta_rows: List[Dict[str, Any]] = []

    # M1 final labels/context.
    loaded = _load_run(repo, "M1", FINAL_NOMINAL_STEP)
    try:
        before = _parameter_snapshot(loaded.model)
        m1_maps["train"] = _render_records(loaded, _train_records(loaded.pipeline))
        m1_maps["eval"] = _render_records(loaded, _eval_records(loaded.pipeline))
        parameter_delta_rows.extend(_parameter_delta_rows(before, loaded.model, "M1", FINAL_NOMINAL_STEP))
    finally:
        _release(loaded)
    print("[BND-HARDNESS] M1 final maps rendered", flush=True)

    # K1 maps and early signals.
    for step in ALL_K1_STEPS:
        loaded = _load_run(repo, "BND-K1", step)
        try:
            before = _parameter_snapshot(loaded.model)
            train_recs = _train_records(loaded.pipeline)
            eval_recs = _eval_records(loaded.pipeline)
            k1_maps[step] = {
                "train": _render_records(loaded, train_recs),
                "eval": _render_records(loaded, eval_recs),
            }
            if step in EARLY_STEPS:
                all_recs = train_recs + eval_recs
                dc_maps = _dc_clear_maps(loaded, all_recs)
                resp_maps = _responsibility_maps(loaded, all_recs)
                refine = _accumulate_fixed_bank_refinement(loaded, train_recs)
                gaussian_score = refine["score"]
                refine_maps: Dict[str, Tensor] = {}
                refine_counts: Dict[str, Tensor] = {}
                for _idx, view_id, camera, _batch in all_recs:
                    fmap, count = _projected_center_window_map(loaded.model, camera, gaussian_score, window=16)
                    refine_maps[view_id] = fmap
                    refine_counts[view_id] = count
                for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
                    signal_maps[step][split]["S_RES_CURRENT"] = {view_id: k1_maps[step][split][view_id]["residual"] for view_id in view_ids}
                    signal_maps[step][split]["S_SH"] = {
                        view_id: (k1_maps[step][split][view_id]["clear"] - dc_maps[view_id]).abs().mean(dim=-1)
                        for view_id in view_ids
                    }
                    signal_maps[step][split]["S_BOUND"] = {view_id: k1_maps[step][split][view_id]["bound"] for view_id in view_ids}
                    signal_maps[step][split]["S_RESP"] = {view_id: resp_maps[view_id] for view_id in view_ids}
                    signal_maps[step][split]["S_REFINE"] = {view_id: refine_maps[view_id] for view_id in view_ids}
                    signal_maps[step][split]["S_BRIGHT_CONTROL"] = {view_id: k1_maps[step][split][view_id]["brightness"] for view_id in view_ids}
            parameter_delta_rows.extend(_parameter_delta_rows(before, loaded.model, "BND-K1", step))
        finally:
            _release(loaded)
        print(f"[BND-HARDNESS] BND-K1 nominal {step} maps/signals ready", flush=True)

    # Label construction.
    final_maps = k1_maps[FINAL_NOMINAL_STEP]
    domains: Dict[str, Dict[str, Tensor]] = {"train": {}, "eval": {}}
    labels: Dict[str, Dict[str, Dict[str, Tensor]]] = {"train": {}, "eval": {}}
    late_available = [step for step in LATE_STEPS if step in k1_maps]
    persistent_required = int(math.ceil(0.75 * len(late_available)))
    for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
        for view_id in view_ids:
            final_support = final_maps[split][view_id]["accumulation"] > 0.01
            domains[split][view_id] = final_support
            m1_support = m1_maps[split][view_id]["accumulation"] > 0.01
            m1_highj = m1_support & (m1_maps[split][view_id]["bound"] > 1.0)
            late_count = torch.zeros_like(final_support, dtype=torch.int32)
            for late_step in late_available:
                late_res = k1_maps[late_step][split][view_id]["residual"]
                late_hard = _top_mask(late_res, final_support, 0.10)
                late_count += late_hard.int()
            persistent = final_support & (late_count >= persistent_required)
            core = persistent & m1_highj
            labels[split][view_id] = {
                "M1_HIGH_J": m1_highj,
                "PERSISTENT_BND_HARD": persistent,
                "BND_HARD_CORE": core,
            }
    label_rows = _label_prevalence_rows(labels, domains)
    _write_csv(OUTPUT_DIR / "label_prevalence.csv", label_rows)
    _write_json(OUTPUT_DIR / "label_prevalence.json", {"late_available_steps": late_available, "persistent_required": persistent_required, "rows": label_rows})
    _write_json(OUTPUT_DIR / "m1_highj_labels_manifest.json", {"definition": "M1 final accumulation > 0.01 and max clear_object_fullsh_raw > 1.0", "rows": [r for r in label_rows if r["label"] == "M1_HIGH_J"]})
    _write_json(OUTPUT_DIR / "persistent_hard_labels_manifest.json", {"late_available_steps": late_available, "persistent_required": persistent_required, "rows": [r for r in label_rows if r["label"] == "PERSISTENT_BND_HARD"]})
    _write_json(OUTPUT_DIR / "bnd_hard_core_manifest.json", {"definition": "PERSISTENT_BND_HARD AND M1_HIGH_J", "rows": [r for r in label_rows if r["label"] == "BND_HARD_CORE"]})
    print("[BND-HARDNESS] offline labels constructed", flush=True)

    # Residual persistence after labels are available because it needs final-support rank domains.
    for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
        history: Dict[str, List[Tensor]] = {view_id: [] for view_id in view_ids}
        for step in EARLY_STEPS:
            maps: Dict[str, Tensor] = {}
            for view_id in view_ids:
                rank = _rank_map(signal_maps[step][split]["S_RES_CURRENT"][view_id], domains[split][view_id])
                history[view_id].append(rank)
                maps[view_id] = torch.stack(history[view_id]).mean(dim=0)
            signal_maps[step][split]["S_RES_PERSIST"] = maps

    signal_availability: Dict[str, Dict[str, Any]] = {
        "S_RES_CURRENT": {
            "family": "RESIDUAL_PERSISTENCE",
            "semantics": "current K1 RGB residual MSE rank inside final K1 object support",
            "availability": "AVAILABLE",
            "deployability_class": "ONLINE_TRAINING_DEPLOYABLE",
            "compute_cost_class": "LOW_COST_ONLINE",
            "requires_gt": True,
            "requires_backward": False,
            "requires_history": False,
            "requires_train_bank_pass": False,
            "requires_extra_render": False,
        },
        "S_RES_PERSIST": {
            "family": "RESIDUAL_PERSISTENCE",
            "semantics": "mean of current and past per-view residual percentile ranks",
            "availability": "AVAILABLE",
            "deployability_class": "ONLINE_TRAINING_DEPLOYABLE",
            "compute_cost_class": "MODERATE_ONLINE",
            "requires_gt": True,
            "requires_backward": False,
            "requires_history": True,
            "requires_train_bank_pass": False,
            "requires_extra_render": False,
        },
        "S_SH": {
            "family": "SH_VIEW_DEPENDENT_APPEARANCE_PRESSURE",
            "semantics": "mean RGB abs difference between bounded full-SH clear render and bounded DC-only clear render",
            "availability": "AVAILABLE",
            "deployability_class": "TRAINING_DEPLOYABLE",
            "compute_cost_class": "MODERATE_ONLINE",
            "requires_gt": False,
            "requires_backward": False,
            "requires_history": False,
            "requires_train_bank_pass": False,
            "requires_extra_render": True,
        },
        "S_BOUND": {
            "family": "BOUNDED_COLOR_PRESSURE",
            "semantics": "max RGB channel of current bounded full-SH clear_object_fullsh_raw",
            "availability": "AVAILABLE",
            "deployability_class": "ONLINE_TRAINING_DEPLOYABLE",
            "compute_cost_class": "LOW_COST_ONLINE",
            "requires_gt": False,
            "requires_backward": False,
            "requires_history": False,
            "requires_train_bank_pass": False,
            "requires_extra_render": False,
        },
        "S_RESP": {
            "family": "RGB_LOSS_RESPONSIBILITY",
            "semantics": "L2 RGB norm of d formal K1 RGB loss / d pred_image",
            "availability": "AVAILABLE",
            "deployability_class": "ONLINE_TRAINING_DEPLOYABLE",
            "compute_cost_class": "MODERATE_ONLINE",
            "requires_gt": True,
            "requires_backward": True,
            "requires_history": False,
            "requires_train_bank_pass": False,
            "requires_extra_render": False,
        },
        "S_REFINE": {
            "family": "REFINEMENT_PRESSURE",
            "semantics": "projected-center 16px-window proxy from fixed 15-view RGB-only densification trigger score",
            "availability": "AVAILABLE",
            "deployability_class": "TRAINING_BANK_DEPLOYABLE",
            "compute_cost_class": "PERIODIC_TRAIN_BANK",
            "requires_gt": True,
            "requires_backward": True,
            "requires_history": False,
            "requires_train_bank_pass": True,
            "requires_extra_render": True,
        },
    }
    _write_json(OUTPUT_DIR / "signal_availability.json", signal_availability)
    _write_json(
        OUTPUT_DIR / "signal_direction_rules.json",
        {
            "common_rule": "higher signal rank means more predicted difficulty",
            "S_RES_CURRENT": "higher residual MSE",
            "S_RES_PERSIST": "higher mean historical residual percentile rank",
            "S_SH": "higher full-vs-DC bounded clear difference",
            "S_BOUND": "higher max RGB bounded clear response",
            "S_RESP": "higher formal RGB-loss image-gradient L2 norm",
            "S_REFINE": "higher fixed-bank projected-center refinement score",
            "S_BRIGHT_CONTROL": "higher GT RGB mean brightness; control only",
        },
    )
    print("[BND-HARDNESS] signal definitions written; computing metrics", flush=True)

    metric_rows: List[Dict[str, Any]] = []
    signals_for_metrics = ("S_RES_CURRENT", "S_RES_PERSIST", "S_SH", "S_BOUND", "S_RESP", "S_REFINE")
    for step in EARLY_STEPS:
        for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
            for signal in (*signals_for_metrics, "S_BRIGHT_CONTROL"):
                if signal not in signal_maps[step][split]:
                    continue
                metric_rows.extend(
                    _metrics_for_signal_labels(
                        split,
                        step,
                        signal,
                        signal_maps[step][split][signal],
                        labels[split],
                        domains[split],
                        view_ids,
                    )
                )
        print(f"[BND-HARDNESS] metrics complete for nominal {step}", flush=True)

    _write_csv(OUTPUT_DIR / "temporal_prediction_metrics.csv", metric_rows)
    _write_json(OUTPUT_DIR / "temporal_prediction_metrics.json", {"rows": metric_rows})
    for filename, prefix in (
        ("residual_signal_metrics", ("S_RES_CURRENT", "S_RES_PERSIST")),
        ("sh_pressure_metrics", ("S_SH",)),
        ("bound_pressure_metrics", ("S_BOUND",)),
        ("responsibility_metrics", ("S_RESP",)),
        ("refinement_pressure_metrics", ("S_REFINE",)),
    ):
        rows = [row for row in metric_rows if row["signal"] in prefix]
        _write_csv(OUTPUT_DIR / f"{filename}.csv", rows)
        _write_json(OUTPUT_DIR / f"{filename}.json", {"rows": rows})
    _write_csv(OUTPUT_DIR / "topk_metrics.csv", metric_rows)
    _write_json(OUTPUT_DIR / "topk_metrics.json", {"rows": metric_rows})
    _write_csv(OUTPUT_DIR / "auprc_metrics.csv", metric_rows)
    _write_json(OUTPUT_DIR / "auprc_metrics.json", {"rows": metric_rows})

    scorecard = _build_scorecard(metric_rows, signal_availability)
    _write_csv(OUTPUT_DIR / "training_view_scorecard.csv", scorecard)
    _write_json(OUTPUT_DIR / "training_view_scorecard.json", {"rows": scorecard})
    print("[BND-HARDNESS] training scorecard written", flush=True)

    redundancy_rows = _signal_redundancy_rows(signal_maps, domains, list(signals_for_metrics))
    _write_csv(OUTPUT_DIR / "signal_redundancy.csv", redundancy_rows)
    _write_json(OUTPUT_DIR / "signal_redundancy.json", {"rows": redundancy_rows})

    locked = _select_proxy(scorecard, redundancy_rows, metric_rows, signal_availability)
    _write_json(OUTPUT_DIR / "composite_selection.json", locked)
    _add_locked_proxy_maps(locked, signal_maps, domains)
    _write_json(OUTPUT_DIR / "locked_proxy_definition.json", locked)
    print(f"[BND-HARDNESS] locked proxy: {locked}", flush=True)

    # Metrics for locked proxy, after lock.
    locked_rows: List[Dict[str, Any]] = []
    if locked["proxy_type"] != "NONE":
        for step in EARLY_STEPS:
            for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
                if "S_LOCKED_PROXY" not in signal_maps[step][split]:
                    continue
                locked_rows.extend(
                    _metrics_for_signal_labels(
                        split,
                        step,
                        "S_LOCKED_PROXY",
                        signal_maps[step][split]["S_LOCKED_PROXY"],
                        labels[split],
                        domains[split],
                        view_ids,
                    )
                )
    all_metric_rows = metric_rows + locked_rows
    heldout_rows = [row for row in locked_rows if row["split"] == "eval"]
    _write_csv(OUTPUT_DIR / "heldout_proxy_metrics.csv", heldout_rows)
    _write_json(OUTPUT_DIR / "heldout_proxy_metrics.json", {"rows": heldout_rows})

    cross = _heldout_cross_view(all_metric_rows)
    _write_csv(OUTPUT_DIR / "heldout_cross_view_metrics.csv", [cross])
    _write_json(OUTPUT_DIR / "heldout_cross_view_metrics.json", cross)

    brightness_rows = [row for row in metric_rows if row["signal"] == "S_BRIGHT_CONTROL"]
    _write_csv(OUTPUT_DIR / "brightness_control.csv", brightness_rows)
    _write_json(OUTPUT_DIR / "brightness_control.json", {"rows": brightness_rows})
    proxy_specificity = _proxy_specificity_audit(all_metric_rows)
    _write_csv(OUTPUT_DIR / "proxy_specificity_audit.csv", proxy_specificity["rows"])
    _write_json(OUTPUT_DIR / "proxy_specificity_audit.json", proxy_specificity)

    overlap_rows: List[Dict[str, Any]] = []
    spatial_rows: List[Dict[str, Any]] = []
    if locked["proxy_type"] != "NONE":
        for split, view_ids in (("train", TRAIN_DEVELOPMENT_VIEWS), ("eval", HELDOUT_EVAL_VIEWS)):
            if "S_LOCKED_PROXY" in signal_maps[3000][split]:
                overlap_rows.extend(_overlap_rows(signal_maps[3000][split]["S_LOCKED_PROXY"], labels[split], domains[split], split, view_ids))
                spatial_rows.extend(_spatial_bias_rows(signal_maps[3000][split]["S_LOCKED_PROXY"], k1_maps[3000][split], domains[split], split, view_ids))
    _write_csv(OUTPUT_DIR / "selected_proxy_overlap.csv", overlap_rows)
    _write_json(OUTPUT_DIR / "selected_proxy_overlap.json", {"rows": overlap_rows})
    _write_csv(OUTPUT_DIR / "spatial_bias_audit.csv", spatial_rows)
    _write_json(OUTPUT_DIR / "spatial_bias_audit.json", {"SPATIAL_BIAS_WARNING": any(bool(row["SPATIAL_BIAS_WARNING"]) for row in spatial_rows), "rows": spatial_rows})

    deploy = _compute_cost_for_proxy(locked, signal_availability)
    _write_json(OUTPUT_DIR / "deployability_audit.json", deploy)
    classification = _classify_proxy(locked, all_metric_rows, cross, deploy)
    _write_json(OUTPUT_DIR / "semantic_alignment_classification.json", {"semantic_alignment": classification.get("semantic"), **classification})
    _write_json(OUTPUT_DIR / "hardness_proxy_classification.json", classification)
    print(f"[BND-HARDNESS] classification: {classification}", flush=True)
    summary = {
        "HELD_OUT_SELECTION_LEAKAGE": False,
        "AUDIT_PARAMETER_SAFETY": all(float(row["max_abs_delta"]) == 0.0 for row in parameter_delta_rows),
        "late_available_steps": late_available,
        "persistent_required_count": persistent_required,
        "locked_proxy": locked,
        "heldout_cross_view": cross,
        "deployability": deploy,
        "proxy_specificity": proxy_specificity,
        "classification": classification,
    }
    _write_json(OUTPUT_DIR / "bnd_hardness_final_summary.json", summary)
    _write_csv(OUTPUT_DIR / "bnd_hardness_final_summary.csv", [{"key": key, "value": value} for key, value in summary.items() if key != "locked_proxy"])

    checkpoint_after = {path: _checkpoint_fingerprint(Path(path)) for path in checkpoint_before}
    checkpoint_safety = {
        "CHECKPOINT_SAFETY": all(checkpoint_before[path] == checkpoint_after[path] for path in checkpoint_before),
        "before": checkpoint_before,
        "after": checkpoint_after,
    }
    _write_json(OUTPUT_DIR / "checkpoint_safety.json", checkpoint_safety)
    _write_csv(OUTPUT_DIR / "parameter_safety.csv", parameter_delta_rows)
    _write_json(
        OUTPUT_DIR / "parameter_safety.json",
        {
            "AUDIT_PARAMETER_SAFETY": all(float(row["max_abs_delta"]) == 0.0 for row in parameter_delta_rows),
            "rows": parameter_delta_rows,
        },
    )

    render_manifest: List[Dict[str, Any]] = []
    _make_visuals(
        RENDER_DIR,
        render_manifest,
        k1_maps,
        labels,
        domains,
        signal_maps,
        all_metric_rows,
        scorecard,
        redundancy_rows,
        locked,
        classification,
        label_rows,
    )
    _write_manifest(OUTPUT_DIR, RENDER_DIR, render_manifest)
    print("[BND-HARDNESS] visual assets and manifests written", flush=True)
    _write_research_note(
        RESEARCH_NOTE,
        repo_manifest,
        checkpoint_rows,
        label_rows,
        signal_availability,
        scorecard,
        locked,
        cross,
        deploy,
        classification,
        proxy_specificity,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    summary = run(args.repo.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
