#!/usr/bin/env python
"""Summarize and render the Panama BND-CDEPTH single-factor run."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_seafree_panama_legal_solution as sea


SCENE = "Panama"
FINAL_STEP = 15000
TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
OUTPUT_DIR = Path("outputs/bnd_cdepth_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_SEAFREE_COARSE_DEPTH_PANAMA_2026-08-11.md")
SEAFREE_EXPORT = Path("outputs/seafree_legal_panama_20260810/seafree_export_step_000029999.pt")
CDEPTH_CONFIG = (
    "outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/"
    "water-splatting/20260811_bnd_cdepth/config.yml"
)
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=sea._json_default) + "\n", encoding="utf8")


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


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_quantile(values: Tensor, q: float) -> float:
    return sea._safe_quantile(values, q)


def _luma(rgb: Tensor) -> Tensor:
    return (rgb.detach().float() * LUMA_WEIGHTS).sum(dim=-1)


def _rgb_l2(image: Tensor) -> Tensor:
    return torch.linalg.norm(image.detach().float(), dim=-1)


def _object_support(item: Mapping[str, Any]) -> Tensor:
    return sea._object_support(item)


def _masked_values(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim == mask.ndim:
        return values[mask]
    while mask.ndim < values.ndim:
        mask = mask[..., None].expand(*values.shape)
    return values[mask]


def _load_runs(repo: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    sea.WS_RUNS["CDEPTH"] = sea.WsRunSpec(
        run="CDEPTH",
        config_relpath=CDEPTH_CONFIG,
        parameterization="bounded_sh3",
        rasterize_mode="classic",
        nominal_step=FINAL_STEP,
    )
    cached: Dict[str, List[Dict[str, Any]]] = {}
    metadata: Dict[str, Any] = {}
    for run in ("M1", "BND-K1", "CDEPTH"):
        items, meta = sea._cache_ws_outputs(repo, run)
        cached[run] = items
        metadata[run] = meta
    sea_export = torch.load(repo / SEAFREE_EXPORT, map_location="cpu")
    sf_items = []
    for item in sea_export["items"]:
        fixed = dict(item)
        fixed["run"] = "SeaFree"
        sf_items.append(fixed)
    cached["SeaFree"] = sf_items
    metadata["SeaFree"] = sea_export.get("metadata", {})
    return cached, {"runs": metadata, "seafree_export": sea_export}


def _view_map(cached: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[Dict[str, Dict[str, Mapping[str, Any]]], List[str]]:
    maps = {run: {str(item["view_id"]): item for item in items} for run, items in cached.items()}
    common = [
        view_id
        for view_id in sea.COMMON_VIEW_IDS
        if all(view_id in maps[run] for run in ("M1", "BND-K1", "CDEPTH", "SeaFree"))
    ]
    if not common:
        raise RuntimeError("No common eval views found across M1, BND-K1, CDEPTH, and SeaFree.")
    return maps, common


def _metric_summary(items: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    metrics = [item["metrics"] for item in items]
    return {
        "scene": SCENE,
        "run": run,
        "num_views": len(items),
        "view_ids": ";".join(str(item["view_id"]) for item in items),
        "psnr": _mean(float(item["psnr"]) for item in metrics),
        "ssim": _mean(float(item["ssim"]) for item in metrics),
        "lpips": _mean(float(item["lpips"]) for item in metrics),
        "mse": _mean(float(item["mse"]) for item in metrics),
    }


def rgb_metrics(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    per_view: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for run in ("M1", "BND-K1", "CDEPTH", "SeaFree"):
        items = [by_run_view[run][view_id] for view_id in view_ids]
        rows.append(_metric_summary(items, run))
        for item in items:
            per_view.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "view_id": item["view_id"],
                    "psnr": item["metrics"]["psnr"],
                    "ssim": item["metrics"]["ssim"],
                    "lpips": item["metrics"]["lpips"],
                    "mse": item["metrics"]["mse"],
                }
            )
    keyed = {row["run"]: row for row in rows}
    for row in per_view:
        if row["run"] == "CDEPTH":
            k1 = by_run_view["BND-K1"][row["view_id"]]["metrics"]
            row["delta_psnr_vs_K1"] = float(row["psnr"]) - float(k1["psnr"])
            row["delta_ssim_vs_K1"] = float(row["ssim"]) - float(k1["ssim"])
            row["delta_lpips_vs_K1"] = float(row["lpips"]) - float(k1["lpips"])
    cdepth = keyed["CDEPTH"]
    k1 = keyed["BND-K1"]
    m1 = keyed["M1"]
    summary = {
        "CDEPTH_PSNR_GAIN": float(cdepth["psnr"]) - float(k1["psnr"]),
        "CDEPTH_SSIM_DELTA": float(cdepth["ssim"]) - float(k1["ssim"]),
        "CDEPTH_LPIPS_DELTA": float(cdepth["lpips"]) - float(k1["lpips"]),
        "GLOBAL_MSE_GAP_RECOVERY": (float(k1["mse"]) - float(cdepth["mse"])) / max(float(k1["mse"]) - float(m1["mse"]), EPS),
        "RGB_SAFETY": bool(
            float(cdepth["psnr"]) - float(m1["psnr"]) >= -0.15
            and float(cdepth["ssim"]) - float(m1["ssim"]) >= -0.0015
            and float(cdepth["lpips"]) - float(m1["lpips"]) <= 0.003
        ),
    }
    c_rows = [row for row in per_view if row["run"] == "CDEPTH"]
    summary["CDEPTH_vs_K1_views_improved"] = sum(1 for row in c_rows if float(row.get("delta_psnr_vs_K1", 0.0)) > 0.0)
    summary["CDEPTH_vs_K1_views_degraded"] = sum(1 for row in c_rows if float(row.get("delta_psnr_vs_K1", 0.0)) < 0.0)
    deltas = [float(row.get("delta_psnr_vs_K1", float("nan"))) for row in c_rows]
    summary["CDEPTH_delta_psnr_mean"] = _mean(deltas)
    summary["CDEPTH_delta_psnr_median"] = float(np.median(deltas)) if deltas else float("nan")
    summary["CDEPTH_delta_psnr_min"] = min(deltas) if deltas else float("nan")
    summary["CDEPTH_delta_psnr_max"] = max(deltas) if deltas else float("nan")
    return rows, per_view, summary


def region_metrics(
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "BRIGHT_NOT_Q5"):
        for run in ("M1", "BND-K1", "CDEPTH", "SeaFree"):
            mse_vals: List[Tensor] = []
            l1_vals: List[Tensor] = []
            pixels = 0
            total = 0
            for view_id in view_ids:
                mask = regions[view_id][region]
                gt = by_run_view["M1"][view_id]["gt"]
                pred = by_run_view[run][view_id]["outputs"]["pred_image"]
                mse = (pred - gt).detach().float().square().mean(dim=-1)
                l1 = (pred - gt).detach().float().abs().mean(dim=-1)
                mse_vals.append(mse[mask])
                l1_vals.append(l1[mask])
                pixels += int(mask.sum().item())
                total += int(mask.numel())
            joined_mse = torch.cat([v.reshape(-1) for v in mse_vals if v.numel()], dim=0)
            joined_l1 = torch.cat([v.reshape(-1) for v in l1_vals if v.numel()], dim=0)
            mean_mse = float(joined_mse.mean().item()) if joined_mse.numel() else float("nan")
            rows.append(
                {
                    "scene": SCENE,
                    "region": region,
                    "run": run,
                    "pixels": pixels,
                    "total_pixels": total,
                    "pixel_fraction": pixels / max(total, 1),
                    "mse": mean_mse,
                    "l1": float(joined_l1.mean().item()) if joined_l1.numel() else float("nan"),
                    "psnr_like": -10.0 * math.log10(max(mean_mse, EPS)) if math.isfinite(mean_mse) else float("nan"),
                }
            )
    keyed = {(row["region"], row["run"]): row for row in rows}
    high_rec = (float(keyed[("M1_HIGH_J", "BND-K1")]["mse"]) - float(keyed[("M1_HIGH_J", "CDEPTH")]["mse"])) / max(
        float(keyed[("M1_HIGH_J", "BND-K1")]["mse"]) - float(keyed[("M1_HIGH_J", "M1")]["mse"]), EPS
    )
    low_damage = float(keyed[("M1_LOW_J", "CDEPTH")]["mse"]) - float(keyed[("M1_LOW_J", "BND-K1")]["mse"])
    bright_rec = (float(keyed[("BRIGHT_Q5", "BND-K1")]["mse"]) - float(keyed[("BRIGHT_Q5", "CDEPTH")]["mse"])) / max(
        float(keyed[("BRIGHT_Q5", "BND-K1")]["mse"]) - float(keyed[("BRIGHT_Q5", "M1")]["mse"]), EPS
    )
    summary = {
        "M1_HIGH_J_pixel_fraction": keyed[("M1_HIGH_J", "M1")]["pixel_fraction"],
        "M1_HIGH_J_MSE_M1": keyed[("M1_HIGH_J", "M1")]["mse"],
        "M1_HIGH_J_MSE_K1": keyed[("M1_HIGH_J", "BND-K1")]["mse"],
        "M1_HIGH_J_MSE_CDEPTH": keyed[("M1_HIGH_J", "CDEPTH")]["mse"],
        "M1_HIGH_J_MSE_SeaFree": keyed[("M1_HIGH_J", "SeaFree")]["mse"],
        "HIGH_J_MSE_GAP_RECOVERY": high_rec,
        "LOW_J_DAMAGE": low_damage,
        "BRIGHT_Q5_GAP_RECOVERY": bright_rec,
    }
    return rows, summary


def geometry_metrics(
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
        for run in ("BND-K1", "CDEPTH", "SeaFree"):
            per = []
            for view_id in view_ids:
                pseudo = by_run_view["SeaFree"][view_id]["outputs"]["pseudo_depth"]
                pred_depth = by_run_view[run][view_id]["outputs"]["depth"]
                per.append(sea._depth_metrics_for_region(pred_depth, pseudo, regions[view_id][region]))
            row: Dict[str, Any] = {
                "scene": SCENE,
                "run": run,
                "region": region,
                "pseudo_depth_source": "depthAnything_u16 normalized per image",
            }
            for key in ("spearman", "pearson", "aligned_mae", "aligned_rmse", "gradient_pearson"):
                row[key] = _mean(float(item[key]) for item in per)
            row["depth_quantity"] = "SeaFree-style approximate disparity = 1/(rendered_depth*10+1)"
            rows.append(row)
    high = {(row["run"]): row for row in rows if row["region"] == "M1_HIGH_J"}
    k1 = high["BND-K1"]
    cd = high["CDEPTH"]
    rmse_improve = (float(k1["aligned_rmse"]) - float(cd["aligned_rmse"])) / max(float(k1["aligned_rmse"]), EPS)
    grad_gain = float(cd["gradient_pearson"]) - float(k1["gradient_pearson"])
    spearman_gain = float(cd["spearman"]) - float(k1["spearman"])
    pearson_gain = float(cd["pearson"]) - float(k1["pearson"])
    same_direction = sum(
        [
            float(cd["aligned_rmse"]) < float(k1["aligned_rmse"]),
            float(cd["aligned_mae"]) < float(k1["aligned_mae"]),
            float(cd["gradient_pearson"]) > float(k1["gradient_pearson"]),
            float(cd["spearman"]) > float(k1["spearman"]),
            float(cd["pearson"]) > float(k1["pearson"]),
        ]
    )
    improved = bool((rmse_improve >= 0.10 or grad_gain >= 0.05) and same_direction >= 2)
    return rows, {
        "HIGHJ_DEPTH_RMSE_IMPROVEMENT": rmse_improve,
        "HIGHJ_DEPTH_GRAD_GAIN": grad_gain,
        "HIGHJ_SPEARMAN_GAIN": spearman_gain,
        "HIGHJ_PEARSON_GAIN": pearson_gain,
        "GEOMETRY_TARGET_IMPROVED": improved,
    }


def canonical_decomposition(
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("M1", "BND-K1", "CDEPTH"):
        items = [by_run_view[run][view_id] for view_id in view_ids]
        support_vals: Dict[str, List[Tensor]] = {"tau": [], "J": [], "T": []}
        for item in items:
            mask = _object_support(item)
            support_vals["tau"].append(item["outputs"]["tau_D"][mask].reshape(-1, 3))
            support_vals["J"].append(item["outputs"]["clear_object_fullsh_raw"][mask].reshape(-1, 3))
            support_vals["T"].append(item["outputs"]["transmission"][mask].reshape(-1, 3))
        tau = torch.cat(support_vals["tau"], dim=0)
        j = torch.cat(support_vals["J"], dim=0)
        t = torch.cat(support_vals["T"], dim=0)
        row = {
            "scene": SCENE,
            "run": run,
            "tau_p90": _mean(_safe_quantile(tau[:, idx], 0.90) for idx in range(3)),
            "tau_p50": _mean(_safe_quantile(tau[:, idx], 0.50) for idx in range(3)),
            "tau_p99": _mean(_safe_quantile(tau[:, idx], 0.99) for idx in range(3)),
            "T_mean": float(t.mean().item()),
            "P_T_lt_0.1": _mean(float((t[:, idx] < 0.1).float().mean().item()) for idx in range(3)),
            "P_T_lt_0.05": _mean(float((t[:, idx] < 0.05).float().mean().item()) for idx in range(3)),
            "J_p95": _mean(_safe_quantile(j[:, idx], 0.95) for idx in range(3)),
            "J_p99": _mean(_safe_quantile(j[:, idx], 0.99) for idx in range(3)),
            "P_J_gt_1": _mean(float((j[:, idx] > 1.0).float().mean().item()) for idx in range(3)),
            "P_J_gt_1.5": _mean(float((j[:, idx] > 1.5).float().mean().item()) for idx in range(3)),
            "P_J_gt_2": _mean(float((j[:, idx] > 2.0).float().mean().item()) for idx in range(3)),
        }
        rows.append(row)
    keyed = {row["run"]: row for row in rows}
    retention = (float(keyed["M1"]["tau_p90"]) - float(keyed["CDEPTH"]["tau_p90"])) / max(
        float(keyed["M1"]["tau_p90"]) - float(keyed["BND-K1"]["tau_p90"]), EPS
    )
    return rows, {"TAU_BENEFIT_RETENTION": retention}


def boundary_metrics(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], sea_export: Mapping[str, Any], view_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for run in ("BND-K1", "CDEPTH"):
        row = sea._boundary_stats_from_ws_items([by_run_view[run][view_id] for view_id in view_ids], run)
        p99 = float(row.get("c_all_gt_0.99", row.get("c_all_gaussians_all_gt_0.99", 0.0)))
        logit = float(row.get("logit_all_abs_gt_5", row.get("logit_all_gt_5", 0.0))) if "logit_all_abs_gt_5" in row else 0.0
        row["P_c_gt_0.99_for_gate"] = p99
        row["P_abs_s_gt_5_for_gate"] = logit
        row["BOUNDARY_ESCAPE"] = bool(p99 > 0.05 or logit > 0.05)
        rows.append(row)
    sf = dict(sea_export.get("boundary_stats", {}))
    sf.update({"scene": SCENE, "run": "SeaFree"})
    rows.append(sf)
    cd = next(row for row in rows if row["run"] == "CDEPTH")
    return rows, {"BOUNDARY_ESCAPE": bool(cd.get("BOUNDARY_ESCAPE", False))}


def population_rows(repo: Path) -> List[Dict[str, Any]]:
    configs = {
        "BND-K1": repo / sea.WS_RUNS["BND-K1"].config_relpath,
        "CDEPTH": repo / CDEPTH_CONFIG,
    }
    rows: List[Dict[str, Any]] = []
    for run, config_path in configs.items():
        ckpt_dir = config_path.parent / "nerfstudio_models"
        for nominal in TRAJECTORY_STEPS:
            actual = nominal
            path = ckpt_dir / f"step-{actual:09d}.ckpt"
            if nominal == 15000 and not path.exists():
                actual = 14999
                path = ckpt_dir / f"step-{actual:09d}.ckpt"
            row: Dict[str, Any] = {"scene": SCENE, "run": run, "nominal_step": nominal, "actual_step": actual, "checkpoint_path": str(path), "available": path.exists()}
            if path.exists():
                ckpt = torch.load(path, map_location="cpu")
                pipe = ckpt["pipeline"]
                means = pipe.get("_model.gauss_params.means")
                opacities = pipe.get("_model.gauss_params.opacities")
                scales = pipe.get("_model.gauss_params.scales")
                row["gaussian_count"] = int(means.shape[0]) if isinstance(means, Tensor) else ""
                if isinstance(opacities, Tensor):
                    row.update(sea._stats(torch.sigmoid(opacities.float()), "opacity_"))
                if isinstance(scales, Tensor):
                    row.update(sea._stats(torch.exp(scales.float()).reshape(-1), "scale_"))
            rows.append(row)
    return rows


def _event_accumulator(run_dir: Path) -> Any:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def _last_scalar_before(accumulator: Any, tag: str, step: int) -> Optional[float]:
    if accumulator is None or tag not in accumulator.Tags().get("scalars", []):
        return None
    events = [event for event in accumulator.Scalars(tag) if int(event.step) <= step]
    if not events:
        return None
    return float(events[-1].value)


def loss_trajectory_rows(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_dir = repo / Path(CDEPTH_CONFIG).parent
    accumulator = _event_accumulator(run_dir)
    if accumulator is None:
        unavailable = {
            "scene": SCENE,
            "run": "CDEPTH",
            "status": "unavailable",
            "reason": f"No TensorBoard event file found in {run_dir}.",
        }
        return [unavailable], [unavailable]

    tags = set(accumulator.Tags().get("scalars", []))
    step_source = "Train Loss" if "Train Loss" in tags else "Train Loss Dict/main_loss"
    steps = sorted({int(event.step) for event in accumulator.Scalars(step_source)})
    fields = {
        "L_total": "Train Loss",
        "L_rgb_total": "Train Loss Dict/main_loss",
        "L_coarse_depth_raw": "Train Metrics Dict/coarse_depth_loss_raw",
        "lambda_depth_times_L_coarse_depth": "Train Loss Dict/coarse_depth_loss",
        "coarse_depth_loss_weighted_metric": "Train Metrics Dict/coarse_depth_loss_weighted",
        "train_psnr": "Train Metrics Dict/psnr",
        "gaussian_count": "Train Metrics Dict/gaussian_count",
    }
    for channel in range(3):
        fields[f"medium_attn_{channel}"] = f"Train Metrics Dict/medium_attn_{channel}"
        fields[f"medium_bs_{channel}"] = f"Train Metrics Dict/medium_bs_{channel}"
        fields[f"medium_rgb_{channel}"] = f"Train Metrics Dict/medium_rgb_{channel}"

    rows: List[Dict[str, Any]] = []
    for step in steps:
        row: Dict[str, Any] = {"scene": SCENE, "run": "CDEPTH", "step": step}
        for name, tag in fields.items():
            value = _last_scalar_before(accumulator, tag, step)
            if value is not None:
                row[name] = value
        row["reg_l1"] = ""
        row["reg_ssim"] = ""
        row["reg_l1_reg_ssim_status"] = "unavailable_not_logged_as_separate_training_scalars"
        rows.append(row)

    finite = True
    nonfinite_tags: List[str] = []
    for tag in sorted(tags):
        vals = [float(event.value) for event in accumulator.Scalars(tag)]
        if any(not math.isfinite(val) for val in vals):
            finite = False
            nonfinite_tags.append(tag)
    stability = [
        {
            "scene": SCENE,
            "run": "CDEPTH",
            "event_file": str(sorted(run_dir.glob("events.out.tfevents.*"))[-1]),
            "num_scalar_tags": len(tags),
            "num_loss_rows": len(rows),
            "NaN_or_Inf_in_logged_scalars": not finite,
            "nonfinite_tags": ";".join(nonfinite_tags),
            "final_logged_step": max(steps) if steps else "",
            "final_L_total": _last_scalar_before(accumulator, "Train Loss", max(steps)) if steps else "",
            "final_L_rgb_total": _last_scalar_before(accumulator, "Train Loss Dict/main_loss", max(steps)) if steps else "",
            "final_L_coarse_depth_raw": _last_scalar_before(accumulator, "Train Metrics Dict/coarse_depth_loss_raw", max(steps)) if steps else "",
            "final_lambda_depth_times_L_coarse_depth": _last_scalar_before(accumulator, "Train Loss Dict/coarse_depth_loss", max(steps)) if steps else "",
        }
    ]
    return rows, stability


def forward_closure_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for run in ("M1", "BND-K1", "CDEPTH"):
        vals = []
        for view_id in view_ids:
            out = by_run_view[run][view_id]["outputs"]
            vals.append((out["pred_image"] - (out["direct_object_signal"] + out["rgb_medium"])).abs().reshape(-1))
        joined = torch.cat(vals)
        rows.append(
            {
                "scene": SCENE,
                "run": run,
                "closure_definition": "pred_image - (direct_object_signal + rgb_medium)",
                "mean_abs": float(joined.mean().item()),
                "p99_abs": _safe_quantile(joined, 0.99),
                "max_abs": float(joined.max().item()),
            }
        )
    return rows


def recomposition_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J"):
        for run in ("BND-K1", "CDEPTH"):
            d_vals = []
            b_vals = []
            cross_vals = []
            for view_id in view_ids:
                mask = regions[view_id][region]
                d = by_run_view[run][view_id]["outputs"]["direct_object_signal"] - by_run_view["M1"][view_id]["outputs"]["direct_object_signal"]
                b = by_run_view[run][view_id]["outputs"]["rgb_medium"] - by_run_view["M1"][view_id]["outputs"]["rgb_medium"]
                d_vals.append(d[mask].square().mean(dim=-1))
                b_vals.append(b[mask].square().mean(dim=-1))
                cross_vals.append((2.0 * d * b)[mask].mean(dim=-1))
            d_join = torch.cat(d_vals)
            b_join = torch.cat(b_vals)
            c_join = torch.cat(cross_vals)
            rows.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "region": region,
                    "C_direct": float(d_join.mean().item()) if d_join.numel() else float("nan"),
                    "C_medium": float(b_join.mean().item()) if b_join.numel() else float("nan"),
                    "C_cross": float(c_join.mean().item()) if c_join.numel() else float("nan"),
                    "RECOMP_EFFICIENCY": float((d_join + b_join + c_join).mean().item()) if d_join.numel() else float("nan"),
                }
            )
    return rows


def coverage_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for run in ("BND-K1", "CDEPTH"):
        for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
            vals = []
            depths = []
            for view_id in view_ids:
                mask = regions[view_id][region]
                vals.append(by_run_view[run][view_id]["outputs"]["accumulation"][..., 0][mask])
                depths.append(by_run_view[run][view_id]["outputs"]["depth"][..., 0][mask])
            acc = torch.cat(vals)
            dep = torch.cat(depths)
            row = {"scene": SCENE, "run": run, "region": region}
            row.update(sea._stats(acc, "alpha_"))
            row.update(sea._stats(dep, "depth_"))
            rows.append(row)
    return rows


def _tile(image: Image.Image, label: str, tile_width: int) -> Image.Image:
    ratio = tile_width / max(image.width, 1)
    size = (tile_width, max(1, int(round(image.height * ratio))))
    resized = image.resize(size, Image.BILINEAR)
    label_h = 28
    canvas = Image.new("RGB", (tile_width, resized.height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(0, 0, 0))
    canvas.paste(resized, (0, label_h))
    return canvas


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str], tile_width: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in rows:
        tiles = [_tile(img, label, tile_width) for label, img in row]
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
    manifest.append(
        {
            "file_path": str(path),
            "scene": SCENE,
            "runs": "M1;BND-K1;CDEPTH;SeaFree",
            "step": FINAL_STEP,
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def render_visuals(
    render_dir: Path,
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
    final_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    residual_scale = 0.02
    depth_resid_scale = 0.5
    grad_scale = 0.2
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "BND-K1", "CDEPTH", "SeaFree"):
            residual_scale = max(residual_scale, float(_rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt).max().item()))

    rows_underwater = []
    rows_highj = []
    rows_lowj = []
    rows_bright = []
    rows_depth = []
    rows_depth_resid = []
    rows_depth_grad = []
    rows_clear = []
    rows_boundary = []
    rows_direct_medium = []
    rows_alpha = []
    for view_id in view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        rows_underwater.append(
            [(f"{view_id} GT", sea._rgb_to_uint8(gt))]
            + [(run, sea._rgb_to_uint8(by_run_view[run][view_id]["outputs"]["pred_image"])) for run in ("M1", "BND-K1", "CDEPTH", "SeaFree")]
        )
        residuals = []
        for run in ("M1", "BND-K1", "CDEPTH", "SeaFree"):
            resid = _rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt)
            residuals.append((f"{run} residual", sea._gray_to_uint8(resid, residual_scale)))
        high = regions[view_id]["M1_HIGH_J"]
        low = regions[view_id]["M1_LOW_J"]
        bright = regions[view_id]["BRIGHT_Q5"]
        rows_highj.append([(f"{view_id} M1_HIGH_J", sea._mask_to_rgb(high))] + [(label, sea._overlay_mask(img, high)) for label, img in residuals])
        rows_lowj.append([(f"{view_id} M1_LOW_J", sea._mask_to_rgb(low))] + [(label, sea._overlay_mask(img, low, (40, 120, 255))) for label, img in residuals])
        rows_bright.append([(f"{view_id} Bright Q5", sea._mask_to_rgb(bright))] + [(label, sea._overlay_mask(img, bright, (40, 200, 120))) for label, img in residuals])

        pseudo = by_run_view["SeaFree"][view_id]["outputs"]["pseudo_depth"]
        if pseudo.ndim == 3:
            pseudo = pseudo[..., 0]
        aligned = {}
        gradients = {"pseudo": sea._gradient_magnitude(pseudo)}
        for run in ("BND-K1", "CDEPTH", "SeaFree"):
            disp = 1.0 / (by_run_view[run][view_id]["outputs"]["depth"][..., 0] * 10.0 + 1.0)
            aligned[run], _, _ = sea._scale_shift_align(disp, pseudo)
            depth_resid_scale = max(depth_resid_scale, float((aligned[run] - pseudo).abs().max().item()))
            gradients[run] = sea._gradient_magnitude(disp)
            grad_scale = max(grad_scale, float(gradients[run].max().item()), float(gradients["pseudo"].max().item()))
        rows_depth.append(
            [(f"{view_id} pseudo", sea._gray_to_uint8(pseudo, 1.0))]
            + [(f"{run} aligned", sea._gray_to_uint8(aligned[run], 1.0)) for run in ("BND-K1", "CDEPTH", "SeaFree")]
        )
        rows_depth_resid.append(
            [(f"{view_id} K1 residual", sea._signed_to_rgb(aligned["BND-K1"] - pseudo, depth_resid_scale))]
            + [(f"{run} residual", sea._signed_to_rgb(aligned[run] - pseudo, depth_resid_scale)) for run in ("CDEPTH", "SeaFree")]
        )
        rows_depth_grad.append(
            [(f"{view_id} pseudo grad", sea._gray_to_uint8(gradients["pseudo"], grad_scale))]
            + [(f"{run} grad", sea._gray_to_uint8(gradients[run], grad_scale)) for run in ("BND-K1", "CDEPTH", "SeaFree")]
        )
        rows_clear.append(
            [
                (f"{view_id} M1 clear", sea._rgb_to_uint8(by_run_view["M1"][view_id]["outputs"]["clear_object_fullsh_raw"])),
                ("K1 clear", sea._rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["clear_object_fullsh_raw"])),
                ("CDEPTH clear", sea._rgb_to_uint8(by_run_view["CDEPTH"][view_id]["outputs"]["clear_object_fullsh_raw"])),
                ("SeaFree intrinsic", sea._rgb_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["intrinsic_color_render"])),
            ]
        )
        rows_boundary.append(
            [
                (f"{view_id} K1 c>0.99", sea._mask_to_rgb(by_run_view["BND-K1"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 0.99)),
                ("CDEPTH c>0.99", sea._mask_to_rgb(by_run_view["CDEPTH"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 0.99)),
                ("SeaFree c>0.99", sea._mask_to_rgb(by_run_view["SeaFree"][view_id]["outputs"]["intrinsic_color_render"].amax(dim=-1) > 0.99)),
                ("M1_HIGH_J", sea._mask_to_rgb(high)),
            ]
        )
        rows_direct_medium.append(
            [
                (f"{view_id} K1 direct", sea._rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["direct_object_signal"])),
                ("CDEPTH direct", sea._rgb_to_uint8(by_run_view["CDEPTH"][view_id]["outputs"]["direct_object_signal"])),
                ("abs direct delta", sea._rgb_to_uint8((by_run_view["CDEPTH"][view_id]["outputs"]["direct_object_signal"] - by_run_view["BND-K1"][view_id]["outputs"]["direct_object_signal"]).abs() / 0.1)),
                ("K1 medium", sea._rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["rgb_medium"])),
                ("CDEPTH medium", sea._rgb_to_uint8(by_run_view["CDEPTH"][view_id]["outputs"]["rgb_medium"])),
                ("abs medium delta", sea._rgb_to_uint8((by_run_view["CDEPTH"][view_id]["outputs"]["rgb_medium"] - by_run_view["BND-K1"][view_id]["outputs"]["rgb_medium"]).abs() / 0.1)),
            ]
        )
        rows_alpha.append(
            [
                (f"{view_id} K1 alpha", sea._gray_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["accumulation"][..., 0], 1.0)),
                ("CDEPTH alpha", sea._gray_to_uint8(by_run_view["CDEPTH"][view_id]["outputs"]["accumulation"][..., 0], 1.0)),
                ("M1_HIGH_J", sea._mask_to_rgb(high)),
                ("K1 overlay", sea._overlay_mask(sea._gray_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["accumulation"][..., 0], 1.0), high)),
                ("CDEPTH overlay", sea._overlay_mask(sea._gray_to_uint8(by_run_view["CDEPTH"][view_id]["outputs"]["accumulation"][..., 0], 1.0), high)),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_underwater_rgb.png", rows_underwater, manifest, "underwater_rgb", view_ids)
    _save_sheet(render_dir / "contact_sheet_fixed_m1_high_j_residual.png", rows_highj, manifest, "fixed_m1_high_j_residual", view_ids)
    _save_sheet(render_dir / "contact_sheet_m1_low_j_control.png", rows_lowj, manifest, "m1_low_j_control", view_ids)
    _save_sheet(render_dir / "contact_sheet_brightness_q5.png", rows_bright, manifest, "brightness_q5", view_ids)
    _save_sheet(render_dir / "contact_sheet_pseudo_depth_diagnostic.png", rows_depth, manifest, "pseudo_depth_diagnostic", view_ids)
    _save_sheet(render_dir / "contact_sheet_depth_residual.png", rows_depth_resid, manifest, "depth_residual", view_ids)
    _save_sheet(render_dir / "contact_sheet_depth_gradient_structure.png", rows_depth_grad, manifest, "depth_gradient_structure", view_ids)
    _save_sheet(render_dir / "contact_sheet_clear_raw.png", rows_clear, manifest, "clear_raw", view_ids)
    _save_sheet(render_dir / "contact_sheet_boundary_usage.png", rows_boundary, manifest, "boundary_usage", view_ids)
    _save_sheet(render_dir / "contact_sheet_direct_medium.png", rows_direct_medium, manifest, "direct_medium", view_ids)
    _save_sheet(render_dir / "contact_sheet_alpha_coverage.png", rows_alpha, manifest, "alpha_coverage", view_ids)

    lines = ["BND-CDEPTH Panama compact summary", ""]
    for key in (
        "CDEPTH_PSNR_GAIN",
        "GLOBAL_MSE_GAP_RECOVERY",
        "HIGH_J_MSE_GAP_RECOVERY",
        "GEOMETRY_TARGET_IMPROVED",
        "TAU_BENEFIT_RETENTION",
        "P_J_gt_1",
        "BOUNDARY_ESCAPE",
        "Hypothesis",
    ):
        if key in final_summary:
            lines.append(f"{key}: {final_summary[key]}")
    text_path = render_dir / "contact_sheet_training_trajectory_summary.png"
    img = Image.new("RGB", (1500, max(120, 28 * len(lines) + 20)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((10, 10 + i * 28), line, fill=(0, 0, 0))
    img.save(text_path)
    manifest.append({"file_path": str(text_path), "scene": SCENE, "runs": "CDEPTH", "step": FINAL_STEP, "output_type": "trajectory_compact_summary", "view_ids": ";".join(view_ids), "width": img.width, "height": img.height})
    return manifest


def classify(
    rgb: Mapping[str, Any],
    region: Mapping[str, Any],
    geometry: Mapping[str, Any],
    decomp: Mapping[str, Any],
    boundary: Mapping[str, Any],
    final_cd: Mapping[str, Any],
) -> Dict[str, Any]:
    psnr_gain = float(rgb["CDEPTH_PSNR_GAIN"])
    global_recovery = float(rgb["GLOBAL_MSE_GAP_RECOVERY"])
    high_recovery = float(region["HIGH_J_MSE_GAP_RECOVERY"])
    low_damage = float(region["LOW_J_DAMAGE"])
    tau_retention = float(decomp["TAU_BENEFIT_RETENTION"])
    p_j_gt_1 = float(final_cd["P_J_gt_1"])
    boundary_escape = bool(boundary["BOUNDARY_ESCAPE"])
    ssim_drop = -float(rgb["CDEPTH_SSIM_DELTA"])
    lpips_worse = float(rgb["CDEPTH_LPIPS_DELTA"])
    geometry_improved = bool(geometry["GEOMETRY_TARGET_IMPROVED"])
    strong = bool(
        (psnr_gain >= 0.20 or global_recovery >= 0.30)
        and high_recovery >= 0.25
        and geometry_improved
        and tau_retention >= 0.75
        and p_j_gt_1 == 0.0
        and not boundary_escape
        and low_damage <= 0.000010
        and ssim_drop <= 0.0005
        and lpips_worse <= 0.0015
    )
    partial = bool(
        (psnr_gain >= 0.08 or global_recovery >= 0.15)
        and high_recovery > 0.0
        and geometry_improved
        and tau_retention >= 0.75
        and p_j_gt_1 == 0.0
        and not boundary_escape
        and low_damage <= 0.000020
        and lpips_worse <= 0.003
    )
    geometry_only = bool(geometry_improved and abs(psnr_gain) < 0.05 and high_recovery < 0.10)
    rgb_only = bool(psnr_gain >= 0.10 and not geometry_improved)
    no_recovery = bool(abs(psnr_gain) < 0.05 and high_recovery < 0.10 and not geometry_improved)
    harmful = bool(psnr_gain <= -0.10 or ssim_drop >= 0.002 or lpips_worse >= 0.003)
    decomp_fail = bool(tau_retention < 0.75 or p_j_gt_1 > 0.0 or boundary_escape)
    pareto_closed = bool(rgb["RGB_SAFETY"] and tau_retention >= 0.75 and p_j_gt_1 == 0.0 and not boundary_escape and low_damage <= 0.000010 and high_recovery > 0.0)
    if decomp_fail:
        label = "DECOMPOSITION_FAILURE"
    elif strong:
        label = "SUPPORTED"
    elif partial:
        label = "PARTIALLY_SUPPORTED"
    elif geometry_only:
        label = "GEOMETRY_EFFECT_NOT_PERFORMANCE_LIMITING"
    elif rgb_only:
        label = "PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE"
    elif harmful and not geometry_improved:
        label = "CONTRADICTED"
    elif no_recovery:
        label = "NOT_SUPPORTED"
    else:
        label = "GRAY_ZONE"
    return {
        "STRONG_CDEPTH_RECOVERY": strong,
        "PARTIAL_CDEPTH_RECOVERY": partial,
        "GEOMETRY_ONLY_POSITIVE": geometry_only,
        "RGB_ONLY_POSITIVE": rgb_only,
        "NO_CDEPTH_RECOVERY": no_recovery,
        "CDEPTH_HARMFUL": harmful,
        "CDEPTH_DECOMPOSITION_REGRESSION": decomp_fail,
        "PANAMA_PARETO_CLOSED": pareto_closed,
        "Hypothesis": label,
    }


def write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND-CDEPTH Panama Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    lines.append("")
    lines.append("No subjective clear-image correctness judgment is included.")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def append_research_note(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    marker = "## BND-CDEPTH Final Summary"
    lines = [
        "",
        marker,
        "",
        "### Experimental Fact",
        "",
        f"- CDEPTH run config: `{outputs.get('cdepth_config')}`.",
        f"- Summary outputs: `{outputs.get('summary_json')}`.",
        f"- Visual manifest: `{outputs.get('visual_manifest')}`.",
        "",
        "### Quantitative Result",
        "",
        f"- CDEPTH PSNR gain vs K1: `{summary.get('CDEPTH_PSNR_GAIN')}`.",
        f"- Global MSE gap recovery: `{summary.get('GLOBAL_MSE_GAP_RECOVERY')}`.",
        f"- High-J MSE gap recovery: `{summary.get('HIGH_J_MSE_GAP_RECOVERY')}`.",
        f"- High-J geometry target improved: `{summary.get('GEOMETRY_TARGET_IMPROVED')}`.",
        f"- Tau benefit retention: `{summary.get('TAU_BENEFIT_RETENTION')}`.",
        f"- P(J>1): `{summary.get('P_J_gt_1')}`.",
        f"- Boundary escape: `{summary.get('BOUNDARY_ESCAPE')}`.",
        f"- Formal hypothesis label: `{summary.get('Hypothesis')}`.",
        "",
        "### Inference",
        "",
        "- The label above follows the pre-registered quantitative gates. Pseudo-depth remains a diagnostic/coarse cue, not depth GT.",
        "",
        "### Hypothesis",
        "",
        "- The next single-factor recommendation is recorded in `bnd_cdepth_final_summary.json`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists():
        text = path.read_text(encoding="utf8")
        prefix = text.split(marker, 1)[0].rstrip()
    with path.open("w", encoding="utf8") as handle:
        if prefix:
            handle.write(prefix + "\n")
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--render-dir", type=Path, default=RENDER_DIR)
    parser.add_argument("--research-note", type=Path, default=RESEARCH_NOTE)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    note_path = (repo / args.research_note).resolve() if not args.research_note.is_absolute() else args.research_note
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    cached, metadata = _load_runs(repo)
    by_run_view, view_ids = _view_map(cached)
    regions, bright_threshold = sea._make_regions(by_run_view["M1"], view_ids)
    rgb_rows, per_view_rows, rgb_summary = rgb_metrics(by_run_view, view_ids)
    region_rows, region_summary = region_metrics(by_run_view, regions, view_ids)
    geom_rows, geom_summary = geometry_metrics(by_run_view, regions, view_ids)
    decomp_rows, decomp_summary = canonical_decomposition(by_run_view, view_ids)
    boundary_rows, boundary_summary = boundary_metrics(by_run_view, metadata["seafree_export"], view_ids)
    pop_rows = population_rows(repo)
    loss_rows, stability_rows = loss_trajectory_rows(repo)
    closure_rows = forward_closure_rows(by_run_view, view_ids)
    recomp_rows = recomposition_rows(by_run_view, regions, view_ids)
    coverage = coverage_rows(by_run_view, regions, view_ids)
    keyed_decomp = {row["run"]: row for row in decomp_rows}
    classification = classify(rgb_summary, region_summary, geom_summary, decomp_summary, boundary_summary, keyed_decomp["CDEPTH"])
    final_summary = {
        "scene": SCENE,
        "view_ids": ";".join(view_ids),
        "bright_q5_threshold": bright_threshold,
        **rgb_summary,
        **region_summary,
        **geom_summary,
        **decomp_summary,
        **boundary_summary,
        "P_J_gt_1": keyed_decomp["CDEPTH"]["P_J_gt_1"],
        "J_p99": keyed_decomp["CDEPTH"]["J_p99"],
        "tau_p90": keyed_decomp["CDEPTH"]["tau_p90"],
        **classification,
    }
    if classification["Hypothesis"] in ("SUPPORTED", "PARTIALLY_SUPPORTED") and classification["PANAMA_PARETO_CLOSED"]:
        final_summary["Next Single-Factor Experiment"] = "Curasao / IUI3 lightweight transfer eligibility audit"
    elif classification["Hypothesis"] == "GEOMETRY_EFFECT_NOT_PERFORMANCE_LIMITING":
        final_summary["Next Single-Factor Experiment"] = "Gaussian population / densification mechanism audit"
    elif classification["Hypothesis"] == "PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE":
        final_summary["Next Single-Factor Experiment"] = "depth-regularization optimization-path diagnostic"
    elif classification["Hypothesis"] == "NOT_SUPPORTED":
        final_summary["Next Single-Factor Experiment"] = "Gaussian population / densification / coverage audit"
    elif classification["Hypothesis"] == "DECOMPOSITION_FAILURE":
        final_summary["Next Single-Factor Experiment"] = "diagnose whether depth supervision reopens object-medium compensation"
    else:
        final_summary["Next Single-Factor Experiment"] = "resolve the most limiting missing/gray-zone quantitative evidence"

    visual_manifest = render_visuals(render_dir, by_run_view, regions, view_ids, final_summary)
    write_visual_index(render_dir, visual_manifest)

    outputs = {
        "final_rgb_metrics": rgb_rows,
        "per_view_metrics": per_view_rows,
        "high_j_region_metrics": [row for row in region_rows if row["region"] == "M1_HIGH_J"],
        "low_j_control": [row for row in region_rows if row["region"] == "M1_LOW_J"],
        "brightness_q5_metrics": [row for row in region_rows if row["region"] in ("BRIGHT_Q5", "BRIGHT_NOT_Q5")],
        "geometry_metrics_global": [row for row in geom_rows if row["region"] == "WHOLE_IMAGE"],
        "geometry_metrics_high_j": [row for row in geom_rows if row["region"] == "M1_HIGH_J"],
        "geometry_metrics_low_j": [row for row in geom_rows if row["region"] == "M1_LOW_J"],
        "canonical_decomposition_metrics": decomp_rows,
        "boundary_metrics": boundary_rows,
        "gaussian_population": pop_rows,
        "coverage_alpha_metrics": coverage,
        "recomposition_metrics": recomp_rows,
        "forward_closure": closure_rows,
        "training_trajectory": [row for row in pop_rows if row["run"] == "CDEPTH"],
        "loss_trajectory": loss_rows,
        "training_stability": stability_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    _write_csv(output_dir / "bnd_cdepth_final_summary.csv", [final_summary])
    _write_json(output_dir / "bnd_cdepth_final_summary.json", final_summary)
    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_csv(render_dir / "manifest.csv", visual_manifest)
    _write_json(
        output_dir / "manifest.json",
        {
            "scene": SCENE,
            "repo": str(repo),
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "cdepth_config": str(repo / CDEPTH_CONFIG),
            "seafree_export": str(repo / SEAFREE_EXPORT),
            "summary_json": str(output_dir / "bnd_cdepth_final_summary.json"),
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
            "view_ids": view_ids,
            "outputs": {name: str(output_dir / f"{name}.json") for name in outputs},
        },
    )
    append_research_note(
        note_path,
        final_summary,
        {
            "cdepth_config": str(repo / CDEPTH_CONFIG),
            "summary_json": str(output_dir / "bnd_cdepth_final_summary.json"),
            "visual_manifest": str(render_dir / "manifest.json"),
        },
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(final_summary, indent=2, sort_keys=True, default=sea._json_default))


if __name__ == "__main__":
    main()
