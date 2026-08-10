#!/usr/bin/env python
"""Setup and eligibility audit for Panama BND-UNORM.

This script is read-only with respect to checkpoints. It audits the existing
BND-K1 checkpoint, compares relative-prediction and absolute photometric losses
at fixed state, checks train-view responsibility replication, and validates
scratch initialization equivalence before any BND-UNORM training run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.utils.eval_utils import eval_setup


DIAGNOSTIC_DIR = Path(__file__).resolve().parent
if str(DIAGNOSTIC_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_DIR))

import audit_bnd_loss_responsibility as lossresp  # noqa: E402


SCENE = "Panama"
EPS = 1e-12
FINAL_STEP = 15000
CHANNELS = ("r", "g", "b")
REGIONS = (
    "M1_HIGH_J",
    "M1_LOW_J",
    "BRIGHT_Q5",
    "DARK_BOTTOM_QUINTILE",
    "BOTTOM20",
    "EDGE_TOP20",
    "LOW_TRANSMISSION",
)
PARAM_GROUPS = ("features_dc", "features_rest", "means", "scales", "opacities", "medium_mlp")
CONFLICT_GROUPS = ("features_dc", "features_rest", "means", "medium_mlp")

M1_CONFIG = (
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
    "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)
K1_CONFIG = (
    "outputs/dewater_bounded_sh3_cross_scene_20260808/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
    "config.yml"
)


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


def _safe_quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(float(q) * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _luma(rgb: Tensor) -> Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return (rgb.detach().float() * weights).sum(dim=-1)


def _gradient_luma(image: Tensor) -> Tensor:
    lum = _luma(image).float()
    dx = torch.zeros_like(lum)
    dy = torch.zeros_like(lum)
    dx[:, 1:] = lum[:, 1:] - lum[:, :-1]
    dy[1:, :] = lum[1:, :] - lum[:-1, :]
    return torch.sqrt(dx.square() + dy.square() + EPS)


def _actual_step(config_path: Path, nominal_step: int) -> int:
    step = lossresp._actual_step(config_path, nominal_step)
    if step is None:
        raise FileNotFoundError(f"missing nominal step {nominal_step}: {config_path}")
    return step


def _load_checkpoint(repo: Path, config_relpath: str, parameterization: str, mode: str = "relative_pred_detached") -> Any:
    config_path = repo / config_relpath
    actual_step = _actual_step(config_path, FINAL_STEP)

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    pipeline.model.config.intrinsic_color_parameterization = parameterization
    pipeline.model.config.rasterize_mode = "classic"
    pipeline.model.config.photometric_normalization_mode = mode
    pipeline.eval()
    return {
        "config": config,
        "pipeline": pipeline,
        "checkpoint_path": checkpoint_path,
        "loaded_step": loaded_step,
        "config_path": config_path,
    }


def _release(obj: Optional[Any]) -> None:
    if obj is not None:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_view_records(pipeline: Any) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dm = pipeline.datamanager
    train_dataset = dm.train_dataset
    cached = dm.cached_train
    image_filenames = list(getattr(train_dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Any, Mapping[str, Any]]] = []
    for index, batch in enumerate(cached):
        filename = image_filenames[index] if index < len(image_filenames) else Path(f"train_{index}")
        rows.append((index, Path(filename).stem, train_dataset.cameras[index : index + 1], batch))
    return rows


def _mode_loss_terms(model: Any, pred: Tensor, gt: Tensor, mode: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == "relative_pred_detached":
        denom = pred.detach() + 1e-3
        per_channel = torch.abs((gt - pred) / denom)
        recon = per_channel.mean()
        sim = 1 - model.ssim((gt / denom).permute(2, 0, 1)[None, ...], (pred / denom).permute(2, 0, 1)[None, ...])
    elif mode == "absolute":
        per_channel = torch.abs(gt - pred)
        recon = per_channel.mean()
        sim = 1 - model.ssim(gt.permute(2, 0, 1)[None, ...], pred.permute(2, 0, 1)[None, ...])
    else:
        raise ValueError(mode)
    total = (1.0 - model.config.ssim_lambda) * recon + model.config.ssim_lambda * sim
    return per_channel, recon, sim, total


def _forward_loss_grad(model: Any, camera: Any, batch: Mapping[str, Any], mode: str) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    model.zero_grad(set_to_none=True)
    model.config.photometric_normalization_mode = mode
    outputs = model.get_outputs(camera.to(model.device))
    gt, pred = lossresp._formal_gt_pred(model, outputs, batch)
    per_channel, recon, sim, total = _mode_loss_terms(model, pred, gt, mode)
    grad = torch.autograd.grad(total, outputs["pred_image"], retain_graph=True, allow_unused=False)[0]
    grad_map = torch.linalg.norm(grad.detach().float(), dim=-1)
    return outputs, gt, pred, per_channel, recon, sim, total, grad_map


def _manual_legacy_loss(model: Any, pred: Tensor, gt: Tensor) -> Tensor:
    denom = pred.detach() + 1e-3
    recon = torch.abs((gt - pred) / denom).mean()
    sim = 1 - model.ssim((gt / denom).permute(2, 0, 1)[None, ...], (pred / denom).permute(2, 0, 1)[None, ...])
    return 0.8 * recon + 0.2 * sim


def _object_support(outputs: Mapping[str, Tensor]) -> Tensor:
    return outputs["accumulation"].detach().float().cpu()[..., 0] > 0.01


def _make_regions(m1_outputs: Mapping[str, Tensor], gt: Tensor, bright_threshold: float, dark_threshold: float) -> Dict[str, Tensor]:
    gt_cpu = gt.detach().float().cpu()
    support = _object_support(m1_outputs)
    clear = m1_outputs["clear_object_fullsh_raw"].detach().float().cpu()
    jmax = clear.amax(dim=-1)
    height, width = jmax.shape
    yy = torch.linspace(0.0, 1.0, height).reshape(height, 1).expand(height, width)
    luma = _luma(gt_cpu)
    edge = _gradient_luma(gt_cpu)
    edge_threshold = _safe_quantile(edge, 0.80)
    transmission = m1_outputs.get("transmission")
    if transmission is None:
        low_t = torch.zeros((height, width), dtype=torch.bool)
    else:
        t = transmission.detach().float().cpu()
        low_t = support & (t.amin(dim=-1) < 0.1)
    return {
        "M1_HIGH_J": support & (jmax > 1.0),
        "M1_LOW_J": support & (jmax <= 1.0),
        "BRIGHT_Q5": luma > bright_threshold,
        "DARK_BOTTOM_QUINTILE": luma < dark_threshold,
        "BOTTOM20": yy >= 0.8,
        "EDGE_TOP20": edge >= edge_threshold,
        "LOW_TRANSMISSION": low_t,
    }


def _empty_region_accum() -> Dict[str, Dict[str, float]]:
    return {
        region: {
            "pixels": 0.0,
            "total_pixels": 0.0,
            "mse": 0.0,
            "mse_total": 0.0,
            "loss": 0.0,
            "loss_total": 0.0,
            "grad": 0.0,
            "grad_total": 0.0,
        }
        for region in REGIONS
    }


def _accumulate_region(
    accum: MutableMapping[str, Dict[str, float]],
    regions: Mapping[str, Tensor],
    mse_map: Tensor,
    loss_map: Tensor,
    grad_map: Tensor,
) -> None:
    mse_cpu = mse_map.detach().float().cpu()
    loss_cpu = loss_map.detach().float().cpu()
    grad_cpu = grad_map.detach().float().cpu()
    total_pixels = float(mse_cpu.numel())
    mse_total = float(mse_cpu.sum().item())
    loss_total = float(loss_cpu.sum().item())
    grad_total = float(grad_cpu.sum().item())
    for region in REGIONS:
        mask = regions[region].detach().bool().cpu()
        pix = float(mask.sum().item())
        accum[region]["pixels"] += pix
        accum[region]["total_pixels"] += total_pixels
        accum[region]["mse"] += float(mse_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["mse_total"] += mse_total
        accum[region]["loss"] += float(loss_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["loss_total"] += loss_total
        accum[region]["grad"] += float(grad_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["grad_total"] += grad_total


def _accum_rows(accum: Mapping[str, Mapping[str, float]], mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region in REGIONS:
        value = accum[region]
        pixel_fraction = value["pixels"] / max(value["total_pixels"], EPS)
        mse_share = value["mse"] / max(value["mse_total"], EPS)
        loss_share = value["loss"] / max(value["loss_total"], EPS)
        grad_share = value["grad"] / max(value["grad_total"], EPS)
        rows.append(
            {
                "scene": SCENE,
                "view_id": "TRAIN_ALL",
                "photometric_normalization_mode": mode,
                "region": region,
                "pixel_fraction": pixel_fraction,
                "mse_error_share": mse_share,
                "mse_enrichment": mse_share / max(pixel_fraction, EPS),
                "formal_l1_loss_share": loss_share,
                "formal_l1_loss_enrichment": loss_share / max(pixel_fraction, EPS),
                "total_image_gradient_share": grad_share,
                "total_image_gradient_enrichment": grad_share / max(pixel_fraction, EPS),
                "responsibility_ratio": grad_share / max(mse_share, EPS),
            }
        )
    return rows


def _lookup(rows: Sequence[Mapping[str, Any]], mode: str, region: str, key: str) -> float:
    for row in rows:
        if row.get("photometric_normalization_mode") == mode and row.get("region") == region:
            return float(row[key])
    return float("nan")


def _train_m1_items(repo: Path) -> Tuple[Dict[str, Dict[str, Any]], float, float, List[str]]:
    m1 = _load_checkpoint(repo, M1_CONFIG, "legacy")
    pipeline = m1["pipeline"]
    records = _train_view_records(pipeline)
    items: Dict[str, Dict[str, Any]] = {}
    lumas: List[Tensor] = []
    view_ids: List[str] = []
    model = pipeline.model
    model.eval()
    with torch.no_grad():
        for _, view_id, camera, batch in records:
            outputs = model.get_outputs_for_camera(camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            outputs_cpu = lossresp._safe_cpu_outputs(outputs, ("clear_object_fullsh_raw", "transmission", "accumulation", "pred_image"))
            gt_cpu = gt.detach().float().cpu()
            items[view_id] = {"gt": gt_cpu, "outputs": outputs_cpu}
            lumas.append(_luma(gt_cpu).reshape(-1))
            view_ids.append(view_id)
    bright_threshold = _safe_quantile(torch.cat(lumas), 0.80)
    dark_threshold = _safe_quantile(torch.cat(lumas), 0.20)
    _release(pipeline)
    return items, bright_threshold, dark_threshold, view_ids


def _param_groups(model: Any) -> Dict[str, List[torch.nn.Parameter]]:
    groups = model.get_param_groups()
    return {name: [p for p in groups.get(name, []) if p.requires_grad] for name in PARAM_GROUPS}


def _flatten_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[Optional[Tensor]]) -> Tensor:
    chunks: List[Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            chunks.append(grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)


def _cosine_from_stats(dot: float, norm_a_sq: float, norm_b_sq: float) -> float:
    denom = math.sqrt(max(norm_a_sq, 0.0)) * math.sqrt(max(norm_b_sq, 0.0))
    if denom <= EPS:
        return float("nan")
    return max(-1.0, min(1.0, dot / denom))


def _selected_subset(view_ids: Sequence[str], count: int) -> List[str]:
    ordered = sorted(view_ids)
    if len(ordered) <= count:
        return ordered
    if count <= 1:
        return [ordered[0]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    out: List[str] = []
    for idx in indices:
        if ordered[idx] not in out:
            out.append(ordered[idx])
    return out


def _snapshot(model: Any) -> Dict[str, Tensor]:
    out = {
        "features_dc": model.features_dc.detach().cpu().clone(),
        "features_rest": model.features_rest.detach().cpu().clone(),
        "means": model.means.detach().cpu().clone(),
        "opacities": model.opacities.detach().cpu().clone(),
    }
    medium = [p.detach().cpu().reshape(-1) for p in model.medium_mlp.parameters()]
    out["medium_mlp_flat"] = torch.cat(medium) if medium else torch.empty(0)
    return out


def _delta(model: Any, before: Mapping[str, Tensor]) -> Dict[str, float]:
    out = {
        "features_dc": float((model.features_dc.detach().cpu() - before["features_dc"]).abs().max().item()),
        "features_rest": float((model.features_rest.detach().cpu() - before["features_rest"]).abs().max().item()),
        "means": float((model.means.detach().cpu() - before["means"]).abs().max().item()),
        "opacities": float((model.opacities.detach().cpu() - before["opacities"]).abs().max().item()),
    }
    medium = [p.detach().cpu().reshape(-1) for p in model.medium_mlp.parameters()]
    current = torch.cat(medium) if medium else torch.empty(0)
    out["medium_mlp_flat"] = float((current - before["medium_mlp_flat"]).abs().max().item()) if current.numel() else 0.0
    return out


def loss_semantics_audit(repo: Path) -> Dict[str, Any]:
    return {
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "current_k1_formula": "0.8*mean(abs((GT-pred)/(stopgrad(pred)+1e-3))) + 0.2*(1-SSIM(GT/(stopgrad(pred)+1e-3), pred/(stopgrad(pred)+1e-3)))",
        "candidate_unorm_formula": "0.8*mean(abs(GT-pred)) + 0.2*(1-SSIM(GT, pred))",
        "epsilon": 1e-3,
        "denominator": "per-channel pred.detach()+1e-3",
        "pred_clamp_before_loss": False,
        "gt_range": "dataset RGB tensors in [0,1]",
        "ssim": "pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=3)",
        "extra_training_losses_in_get_loss_dict": False,
        "m1_and_bnd_k1_same_loss": True,
        "only_variable_for_bnd_unorm": "photometric_normalization_mode relative_pred_detached -> absolute",
    }


def default_loss_equivalence(repo: Path, m1_items: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    k1 = _load_checkpoint(repo, K1_CONFIG, "bounded_sh3", "relative_pred_detached")
    model = k1["pipeline"].model
    rows: List[Dict[str, Any]] = []
    for _, view_id, camera, batch in _train_view_records(k1["pipeline"]):
        outputs = model.get_outputs_for_camera(camera)
        gt, pred = lossresp._formal_gt_pred(model, outputs, batch)
        code_loss = model.get_loss_dict(outputs, batch)["main_loss"]
        manual = _manual_legacy_loss(model, pred, gt)
        rows.append(
            {
                "scene": SCENE,
                "view_id": view_id,
                "code_relative_loss": float(code_loss.detach().item()),
                "manual_legacy_relative_loss": float(manual.detach().item()),
                "abs_diff": float((code_loss - manual).abs().detach().item()),
            }
        )
        del outputs, gt, pred, code_loss, manual
    max_diff = max(float(row["abs_diff"]) for row in rows)
    summary = {
        "DEFAULT_LOSS_EQUIVALENCE": max_diff <= 1e-7,
        "max_abs_loss_diff": max_diff,
        "definition": "current relative_pred_detached loss path versus explicit pre-change legacy formula",
    }
    _release(k1["pipeline"])
    return rows, summary


def responsibility_audit(repo: Path, m1_items: Mapping[str, Mapping[str, Any]], bright_threshold: float, dark_threshold: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]]:
    k1 = _load_checkpoint(repo, K1_CONFIG, "bounded_sh3", "relative_pred_detached")
    pipeline = k1["pipeline"]
    model = pipeline.model
    records = _train_view_records(pipeline)
    view_by_id = {view_id: (camera, batch) for _, view_id, camera, batch in records}
    subset = _selected_subset(list(view_by_id), 8)
    rel_accum = _empty_region_accum()
    abs_accum = _empty_region_accum()
    grad_values: Dict[str, List[Tensor]] = {"relative_pred_detached": [], "absolute": []}
    image_norm_rows: List[Dict[str, Any]] = []
    before = _snapshot(model)

    for _, view_id, camera, batch in records:
        for mode, accum in (("relative_pred_detached", rel_accum), ("absolute", abs_accum)):
            outputs, gt, pred, per_channel, recon, sim, total, grad_map = _forward_loss_grad(model, camera, batch, mode)
            regions = _make_regions(m1_items[view_id]["outputs"], gt.detach().cpu(), bright_threshold, dark_threshold)
            mse_map = (pred.detach().float().cpu() - gt.detach().float().cpu()).square().mean(dim=-1)
            loss_map = per_channel.detach().float().cpu().mean(dim=-1)
            _accumulate_region(accum, regions, mse_map, loss_map, grad_map.cpu())
            grad_values[mode].append(grad_map.detach().float().cpu().reshape(-1))
            del outputs, gt, pred, per_channel, recon, sim, total, grad_map
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = _accum_rows(rel_accum, "relative_pred_detached") + _accum_rows(abs_accum, "absolute")
    for mode, chunks in grad_values.items():
        flat = torch.cat(chunks)
        image_norm_rows.append(
            {
                "scene": SCENE,
                "view_id": "TRAIN_ALL",
                "photometric_normalization_mode": mode,
                "image_grad_l1_sum": float(flat.sum().item()),
                "image_grad_l2": float(torch.linalg.norm(flat).item()),
                "image_grad_mean_abs": float(flat.abs().mean().item()),
                "image_grad_p50": _safe_quantile(flat, 0.50),
                "image_grad_p90": _safe_quantile(flat, 0.90),
                "image_grad_p99": _safe_quantile(flat, 0.99),
            }
        )

    highj_rel = next(r for r in rows if r["photometric_normalization_mode"] == "relative_pred_detached" and r["region"] == "M1_HIGH_J")
    highj_abs = next(r for r in rows if r["photometric_normalization_mode"] == "absolute" and r["region"] == "M1_HIGH_J")
    train_rep = highj_rel["mse_enrichment"] >= 2.0 and highj_rel["responsibility_ratio"] <= 0.60
    highj_grad_gain = highj_abs["total_image_gradient_share"] / max(highj_rel["total_image_gradient_share"], EPS)
    resp_gain = highj_abs["responsibility_ratio"] / max(highj_rel["responsibility_ratio"], EPS)

    # Parameter-space REL/ABS global gradient direction and scale.
    groups = _param_groups(model)
    stats = {
        group: {"rel_sq": 0.0, "abs_sq": 0.0, "dot": 0.0, "rel_abs_sum": 0.0, "abs_abs_sum": 0.0, "numel": 0}
        for group in PARAM_GROUPS
    }
    conflict = {
        group: {"dot": 0.0, "high_sq": 0.0, "low_sq": 0.0}
        for group in CONFLICT_GROUPS
    }
    for view_id in subset:
        camera, batch = view_by_id[view_id]
        rel_vecs = _param_grad_vectors(model, camera, batch, "relative_pred_detached", groups)
        abs_vecs = _param_grad_vectors(model, camera, batch, "absolute", groups)
        for group in PARAM_GROUPS:
            rv = rel_vecs.get(group, torch.empty(0))
            av = abs_vecs.get(group, torch.empty(0))
            if rv.numel() == 0 or av.numel() == 0:
                continue
            stats[group]["rel_sq"] += float(rv.square().sum().item())
            stats[group]["abs_sq"] += float(av.square().sum().item())
            stats[group]["dot"] += float(torch.dot(rv, av).item())
            stats[group]["rel_abs_sum"] += float(rv.abs().sum().item())
            stats[group]["abs_abs_sum"] += float(av.abs().sum().item())
            stats[group]["numel"] += rv.numel()

        regions = _make_regions(m1_items[view_id]["outputs"], m1_items[view_id]["gt"], bright_threshold, dark_threshold)
        high_vecs, low_vecs = _abs_region_grad_vectors(model, camera, batch, regions["M1_HIGH_J"], regions["M1_LOW_J"], groups)
        for group in CONFLICT_GROUPS:
            hv = high_vecs.get(group, torch.empty(0))
            lv = low_vecs.get(group, torch.empty(0))
            if hv.numel() == 0 or lv.numel() == 0:
                continue
            conflict[group]["dot"] += float(torch.dot(hv, lv).item())
            conflict[group]["high_sq"] += float(hv.square().sum().item())
            conflict[group]["low_sq"] += float(lv.square().sum().item())

    scale_rows: List[Dict[str, Any]] = []
    direction_rows: List[Dict[str, Any]] = []
    for group, value in stats.items():
        rel_l2 = math.sqrt(value["rel_sq"])
        abs_l2 = math.sqrt(value["abs_sq"])
        scale_rows.append(
            {
                "scene": SCENE,
                "view_id": ";".join(subset),
                "parameter_group": group,
                "rel_grad_l2": rel_l2,
                "abs_grad_l2": abs_l2,
                "GLOBAL_GRAD_NORM_RATIO_group": abs_l2 / max(rel_l2, EPS),
                "rel_grad_mean_abs": value["rel_abs_sum"] / max(value["numel"], 1),
                "abs_grad_mean_abs": value["abs_abs_sum"] / max(value["numel"], 1),
            }
        )
        direction_rows.append(
            {
                "scene": SCENE,
                "view_id": ";".join(subset),
                "parameter_group": group,
                "rel_vs_abs_cosine": _cosine_from_stats(value["dot"], value["rel_sq"], value["abs_sq"]),
            }
        )

    conflict_rows: List[Dict[str, Any]] = []
    abs_conflict = False
    for group, value in conflict.items():
        high_l2 = math.sqrt(value["high_sq"])
        low_l2 = math.sqrt(value["low_sq"])
        # Use absolute total norm for non-negligibility.
        total_abs_l2 = next((r["abs_grad_l2"] for r in scale_rows if r["parameter_group"] == group), 0.0)
        cosine = _cosine_from_stats(value["dot"], value["high_sq"], value["low_sq"])
        high_norm_to_total = high_l2 / max(total_abs_l2, EPS)
        low_norm_to_total = low_l2 / max(total_abs_l2, EPS)
        if group in ("features_dc", "features_rest") and cosine <= -0.20 and high_norm_to_total >= 0.01 and low_norm_to_total >= 0.01:
            abs_conflict = True
        conflict_rows.append(
            {
                "scene": SCENE,
                "view_id": ";".join(subset),
                "photometric_normalization_mode": "absolute",
                "region_pair": "M1_HIGH_J_vs_M1_LOW_J",
                "parameter_group": group,
                "cosine": cosine,
                "high_norm_to_abs_total": high_norm_to_total,
                "low_norm_to_abs_total": low_norm_to_total,
            }
        )

    delta = _delta(model, before)
    safety = {
        "AUDIT_PARAMETER_SAFETY": all(v == 0.0 for v in delta.values()),
        **{f"max_abs_delta_{k}": v for k, v in delta.items()},
    }
    eligibility = {
        "TRAIN_HIGHJ_UNDEREMPHASIS_REPLICATED": train_rep,
        "HIGHJ_GRAD_SHARE_GAIN": highj_grad_gain,
        "RESP_RATIO_GAIN": resp_gain,
        "ABS_HIGHJ_GRADIENT_CONFLICT": abs_conflict,
        "BND_UNORM_TRAINING_ELIGIBLE": train_rep and (highj_grad_gain >= 1.50 or resp_gain >= 1.50) and not abs_conflict,
        "parameter_gradient_subset_view_ids": subset,
        **safety,
    }
    _release(pipeline)
    return rows, image_norm_rows, eligibility, scale_rows, direction_rows, conflict_rows, subset, safety


def _param_grad_vectors(model: Any, camera: Any, batch: Mapping[str, Any], mode: str, groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> Dict[str, Tensor]:
    model.zero_grad(set_to_none=True)
    model.config.photometric_normalization_mode = mode
    outputs = model.get_outputs(camera.to(model.device))
    gt, pred = lossresp._formal_gt_pred(model, outputs, batch)
    _, _, _, total = _mode_loss_terms(model, pred, gt, mode)
    all_params: List[torch.nn.Parameter] = []
    offsets: Dict[str, Tuple[int, int, Sequence[torch.nn.Parameter]]] = {}
    for group in PARAM_GROUPS:
        params = list(groups.get(group, []))
        start = len(all_params)
        all_params.extend(params)
        offsets[group] = (start, len(all_params), params)
    grads = torch.autograd.grad(total, all_params, retain_graph=False, allow_unused=True)
    out: Dict[str, Tensor] = {}
    for group, (start, end, params) in offsets.items():
        out[group] = _flatten_grads(params, grads[start:end])
    del outputs, gt, pred, total
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _abs_region_grad_vectors(
    model: Any,
    camera: Any,
    batch: Mapping[str, Any],
    high_mask_cpu: Tensor,
    low_mask_cpu: Tensor,
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    model.zero_grad(set_to_none=True)
    model.config.photometric_normalization_mode = "absolute"
    outputs = model.get_outputs(camera.to(model.device))
    gt, pred = lossresp._formal_gt_pred(model, outputs, batch)
    per_channel, _, _, _ = _mode_loss_terms(model, pred, gt, "absolute")
    loss_map = per_channel.mean(dim=-1)
    high_mask = high_mask_cpu.to(device=pred.device)
    low_mask = low_mask_cpu.to(device=pred.device)
    high_loss = (loss_map * high_mask.float()).sum() / loss_map.numel()
    low_loss = (loss_map * low_mask.float()).sum() / loss_map.numel()
    all_params: List[torch.nn.Parameter] = []
    offsets: Dict[str, Tuple[int, int, Sequence[torch.nn.Parameter]]] = {}
    for group in CONFLICT_GROUPS:
        params = list(groups.get(group, []))
        start = len(all_params)
        all_params.extend(params)
        offsets[group] = (start, len(all_params), params)
    high_grads = torch.autograd.grad(high_loss, all_params, retain_graph=True, allow_unused=True)
    low_grads = torch.autograd.grad(low_loss, all_params, retain_graph=False, allow_unused=True)
    high_out: Dict[str, Tensor] = {}
    low_out: Dict[str, Tensor] = {}
    for group, (start, end, params) in offsets.items():
        high_out[group] = _flatten_grads(params, high_grads[start:end])
        low_out[group] = _flatten_grads(params, low_grads[start:end])
    del outputs, gt, pred, per_channel, loss_map
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return high_out, low_out


def scratch_pipeline(repo: Path, mode: str) -> Any:
    config_path = repo / K1_CONFIG
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.pipeline.model.intrinsic_color_parameterization = "bounded_sh3"
    config.pipeline.model.rasterize_mode = "classic"
    config.pipeline.model.medium_context_mode = "dir_xy_camera"
    config.pipeline.model.b_inf_mode = "tied"
    config.pipeline.model.infinite_water_enabled = False
    config.pipeline.model.appearance_lr_scale = 1.0
    config.pipeline.model.photometric_normalization_mode = mode
    _set_random_seed(config.machine.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(pipeline, Pipeline)
    pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    pipeline.model.config.rasterize_mode = "classic"
    pipeline.model.config.photometric_normalization_mode = mode
    pipeline.eval()
    return pipeline


def _parameter_snapshot_for_equivalence(model: Any) -> Dict[str, Tensor]:
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
    return out


def initialization_equivalence(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rel = scratch_pipeline(repo, "relative_pred_detached")
    rel_model = rel.model
    rel_records = _train_view_records(rel)
    _, view_id, camera, batch = rel_records[0]
    rel_snap = _parameter_snapshot_for_equivalence(rel_model)
    with torch.no_grad():
        rel_out = rel_model.get_outputs_for_camera(camera)
        rel_cpu = lossresp._safe_cpu_outputs(rel_out, ("pred_image", "direct_object_signal", "rgb_medium", "depth", "accumulation", "clear_object_fullsh_raw", "transmission", "tau_D"))
    _release(rel)

    abs_pipe = scratch_pipeline(repo, "absolute")
    abs_model = abs_pipe.model
    abs_records = _train_view_records(abs_pipe)
    _, abs_view_id, abs_camera, _ = abs_records[0]
    abs_snap = _parameter_snapshot_for_equivalence(abs_model)
    with torch.no_grad():
        abs_out = abs_model.get_outputs_for_camera(abs_camera)
        abs_cpu = lossresp._safe_cpu_outputs(abs_out, ("pred_image", "direct_object_signal", "rgb_medium", "depth", "accumulation", "clear_object_fullsh_raw", "transmission", "tau_D"))

    param_rows: List[Dict[str, Any]] = []
    max_param = 0.0
    for key, rel_tensor in rel_snap.items():
        diff = abs_snap[key] - rel_tensor
        value = float(diff.abs().max().item()) if diff.numel() else 0.0
        max_param = max(max_param, value)
        param_rows.append({"parameter": key, "max_abs_diff": value, "shape": list(rel_tensor.shape)})

    forward_rows: List[Dict[str, Any]] = []
    max_forward = 0.0
    for key, rel_tensor in rel_cpu.items():
        diff = abs_cpu[key] - rel_tensor
        value = float(diff.abs().max().item()) if diff.numel() else 0.0
        max_forward = max(max_forward, value)
        forward_rows.append({"view_id_relative": view_id, "view_id_absolute": abs_view_id, "output": key, "max_abs_diff": value, "shape": list(rel_tensor.shape)})

    _release(abs_pipe)
    summary = {
        "INIT_PARAMETER_EQUIVALENCE": max_param == 0.0,
        "INIT_FORWARD_EQUIVALENCE": max_forward <= 1e-7,
        "max_parameter_abs_diff": max_param,
        "max_forward_abs_diff": max_forward,
    }
    return param_rows, forward_rows, summary


def write_research_note(path: Path, repo_manifest: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    lines = [
        "# BND-UNORM Photometric Loss - 2026-08-10",
        "",
        "## 1. Motivation",
        "",
        "HYPOTHESIS: Removing inverse-prediction photometric normalization may increase optimization responsibility for localized bright / legacy-high-J Panama failure regions.",
        "",
        "## 5. Exact Current Loss Semantics",
        "",
        "CODE FACT: Current K1 uses `0.8*reg_l1 + 0.2*reg_ssim`, with per-channel denominator `stopgrad(pred)+1e-3` in both terms.",
        "",
        "## 6. BND-UNORM Formulation",
        "",
        "CODE FACT: BND-UNORM uses `0.8*mean(abs(GT-pred)) + 0.2*(1-SSIM(GT,pred))`. No foreground, residual, brightness, pseudo-depth, or SeaFree weighting is added.",
        "",
        "## 7. Single-Factor Experimental Design",
        "",
        "CODE FACT: The only intended training variable is `photometric_normalization_mode: relative_pred_detached -> absolute` under `bounded_sh3`, SH3, classic rasterization, M1 medium settings, and seed 42.",
        "",
        "## 8. Default Behavior Equivalence",
        "",
        f"QUANTITATIVE RESULT: `DEFAULT_LOSS_EQUIVALENCE = {gate.get('DEFAULT_LOSS_EQUIVALENCE')}`, max diff `{gate.get('default_loss_max_abs_diff')}`.",
        "",
        "## 9. Train-View Responsibility Replication",
        "",
        f"QUANTITATIVE RESULT: `TRAIN_HIGHJ_UNDEREMPHASIS_REPLICATED = {gate.get('TRAIN_HIGHJ_UNDEREMPHASIS_REPLICATED')}`.",
        f"M1_HIGH_J train pixel fraction `{gate.get('train_highj_pixel_fraction')}`, MSE share `{gate.get('train_highj_mse_share')}`, gradient share `{gate.get('train_highj_rel_grad_share')}`, responsibility ratio `{gate.get('train_highj_rel_resp_ratio')}`.",
        "",
        "## 10. Fixed-State REL-vs-ABS Responsibility Shift",
        "",
        f"QUANTITATIVE RESULT: HIGHJ_GRAD_SHARE_GAIN `{gate.get('HIGHJ_GRAD_SHARE_GAIN')}`, RESP_RATIO_GAIN `{gate.get('RESP_RATIO_GAIN')}`.",
        "",
        "## 11. Global Gradient Scale Audit",
        "",
        "EXPERIMENTAL FACT: Parameter-group REL/ABS scale and direction are stored in `fixed_state_gradient_scale.csv` and `fixed_state_gradient_direction.csv`.",
        "",
        "## 13. High-J/Low-J Conflict Audit",
        "",
        f"QUANTITATIVE RESULT: `ABS_HIGHJ_GRADIENT_CONFLICT = {gate.get('ABS_HIGHJ_GRADIENT_CONFLICT')}`.",
        "",
        "## 14. Initialization Equivalence",
        "",
        f"QUANTITATIVE RESULT: `INIT_PARAMETER_EQUIVALENCE = {gate.get('INIT_PARAMETER_EQUIVALENCE')}`, `INIT_FORWARD_EQUIVALENCE = {gate.get('INIT_FORWARD_EQUIVALENCE')}`.",
        "",
        "## Eligibility",
        "",
        f"QUANTITATIVE RESULT: `BND_UNORM_TRAINING_ELIGIBLE = {gate.get('BND_UNORM_TRAINING_ELIGIBLE')}`.",
        "",
        "Later sections will be filled after the eligible training run and final summary, if training is executed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_unorm_panama_20260810"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short").splitlines(),
        "m1_config": str(repo / M1_CONFIG),
        "k1_config": str(repo / K1_CONFIG),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    semantics = loss_semantics_audit(repo)
    _write_json(output_dir / "loss_semantics_audit.json", semantics)

    m1_items, bright_threshold, dark_threshold, train_view_ids = _train_m1_items(repo)
    default_rows, default_summary = default_loss_equivalence(repo, m1_items)
    responsibility_rows, image_norm_rows, eligibility, scale_rows, direction_rows, conflict_rows, subset, safety = responsibility_audit(
        repo, m1_items, bright_threshold, dark_threshold
    )
    init_param_rows, init_forward_rows, init_summary = initialization_equivalence(repo)

    gate = {
        **default_summary,
        **eligibility,
        **init_summary,
        "train_view_ids": train_view_ids,
        "bright_q5_threshold": bright_threshold,
        "dark_bottom_quintile_threshold": dark_threshold,
    }
    gate["default_loss_max_abs_diff"] = default_summary["max_abs_loss_diff"]
    highj_rel = next(r for r in responsibility_rows if r["photometric_normalization_mode"] == "relative_pred_detached" and r["region"] == "M1_HIGH_J")
    gate.update(
        {
            "train_highj_pixel_fraction": highj_rel["pixel_fraction"],
            "train_highj_mse_share": highj_rel["mse_error_share"],
            "train_highj_mse_enrichment": highj_rel["mse_enrichment"],
            "train_highj_rel_grad_share": highj_rel["total_image_gradient_share"],
            "train_highj_rel_resp_ratio": highj_rel["responsibility_ratio"],
        }
    )
    gate["BND_UNORM_TRAINING_ELIGIBLE"] = (
        bool(gate["DEFAULT_LOSS_EQUIVALENCE"])
        and bool(gate["TRAIN_HIGHJ_UNDEREMPHASIS_REPLICATED"])
        and (float(gate["HIGHJ_GRAD_SHARE_GAIN"]) >= 1.50 or float(gate["RESP_RATIO_GAIN"]) >= 1.50)
        and not bool(gate["ABS_HIGHJ_GRADIENT_CONFLICT"])
        and bool(gate["AUDIT_PARAMETER_SAFETY"])
        and bool(gate["INIT_PARAMETER_EQUIVALENCE"])
        and bool(gate["INIT_FORWARD_EQUIVALENCE"])
    )

    _write_csv(output_dir / "default_loss_equivalence.csv", default_rows)
    _write_json(output_dir / "default_loss_equivalence.json", {"rows": default_rows, "summary": default_summary})
    _write_csv(output_dir / "train_view_responsibility_replication.csv", [r for r in responsibility_rows if r["photometric_normalization_mode"] == "relative_pred_detached"])
    _write_json(output_dir / "train_view_responsibility_replication.json", [r for r in responsibility_rows if r["photometric_normalization_mode"] == "relative_pred_detached"])
    _write_csv(output_dir / "fixed_state_rel_vs_abs_responsibility.csv", responsibility_rows)
    _write_json(output_dir / "fixed_state_rel_vs_abs_responsibility.json", responsibility_rows)
    _write_csv(output_dir / "fixed_state_gradient_scale.csv", image_norm_rows + scale_rows)
    _write_json(output_dir / "fixed_state_gradient_scale.json", {"image": image_norm_rows, "parameter": scale_rows})
    _write_csv(output_dir / "fixed_state_gradient_direction.csv", direction_rows)
    _write_json(output_dir / "fixed_state_gradient_direction.json", direction_rows)
    _write_csv(output_dir / "fixed_state_gradient_conflict.csv", conflict_rows)
    _write_json(output_dir / "fixed_state_gradient_conflict.json", conflict_rows)
    _write_csv(output_dir / "initialization_parameter_equivalence.csv", init_param_rows)
    _write_json(output_dir / "initialization_parameter_equivalence.json", {"rows": init_param_rows, "summary": init_summary})
    _write_csv(output_dir / "initialization_forward_equivalence.csv", init_forward_rows)
    _write_json(output_dir / "initialization_forward_equivalence.json", {"rows": init_forward_rows, "summary": init_summary})
    _write_json(output_dir / "bnd_unorm_setup_gate.json", gate)

    manifest = {
        "outputs": sorted(str(p) for p in output_dir.rglob("*") if p.is_file()),
        "gate": gate,
        "parameter_gradient_subset_view_ids": subset,
        "train_view_ids": train_view_ids,
    }
    _write_json(output_dir / "manifest.json", manifest)
    write_research_note(repo / "research_notes/BND_UNORM_PHOTOMETRIC_LOSS_2026-08-10.md", repo_manifest, gate)
    print(json.dumps({"output_dir": str(output_dir), "gate": gate}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
