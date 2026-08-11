#!/usr/bin/env python
"""Optimizer-aware CDEPTH direct-object path audit for Panama.

This diagnostic is read-only with respect to formal training artifacts. It
loads existing BND-K1 / BND-CDEPTH checkpoints, inspects saved optimizer state,
computes RGB-only versus RGB+coarse-depth gradients on fixed training cameras,
constructs analytic Adam one-step virtual updates, and renders throwaway local
counterfactuals for eval views. It never calls persistent optimizer.step(),
scheduler.step(), densification, pruning, opacity reset, or checkpoint writes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
from PIL import Image
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_densify_trigger as densify_audit
from scripts.diagnostics import audit_bnd_cdepth_optimization_path as path_audit
from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_cdepth_direct_path_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_direct_path_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_direct_path_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_DIRECT_OBJECT_OPTIMIZATION_PATH_2026-08-11.md")

HISTORICAL_STEPS = (1000, 3000, 5000, 8000)
INTERVENTION_STEPS = (1000, 3000, 5000)
FINAL_STEP = 15000
EVAL_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
TRAIN_CAMERA_BANK = (
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
CURRENT_PARAM_GROUPS = (
    "means",
    "scales",
    "quats",
    "features_dc",
    "features_rest",
    "opacities",
    "medium_mlp",
    "direction_encoding",
)
GEOMETRY_GROUPS = ("means", "scales", "quats", "opacities")
REGIONS = ("global", "M1_HIGH_J", "HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM")
EPS = 1e-12


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
    "M1": RunSpec(
        "M1",
        Path(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        intrinsic_color_parameterization="legacy",
    ),
    "BND-K1": RunSpec("BND-K1", cdepth_setup.K1_CONFIG),
    "CDEPTH": RunSpec(
        "CDEPTH",
        Path(
            "outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/"
            "water-splatting/20260811_bnd_cdepth/config.yml"
        ),
    ),
}


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


def _file_state(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
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


def _load_run(repo: Path, run: str, nominal_step: int, *, load_depths: bool = False) -> LoadedRun:
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
    model.step = int(loaded_step)
    pipeline.eval()
    return LoadedRun(spec, config_path, checkpoint_path, int(loaded_step), config, pipeline)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _eval_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        view_id = Path(filename).stem
        if view_id in EVAL_VIEWS:
            rows.append((eval_index, view_id, camera, _batch_to_device(batch, loaded.pipeline.model.device)))
    return rows


def _train_bank_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    dataset = loaded.pipeline.datamanager.train_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    stem_to_index = {Path(path).stem: index for index, path in enumerate(image_filenames)}
    missing = [view_id for view_id in TRAIN_CAMERA_BANK if view_id not in stem_to_index]
    if missing:
        raise RuntimeError(f"Missing train views in loaded dataset: {missing}")
    cameras = dataset.cameras.to(loaded.pipeline.model.device)
    rows: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    for view_id in TRAIN_CAMERA_BANK:
        index = stem_to_index[view_id]
        batch = loaded.pipeline.datamanager.cached_train[index].copy()
        rows.append((index, view_id, cameras[index : index + 1], _batch_to_device(batch, loaded.pipeline.model.device)))
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
    names = ("count", "mean", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
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


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    vals = values.detach().float()
    m = mask.detach().bool()
    while m.ndim < vals.ndim:
        m = m[..., None].expand(*vals.shape)
    return vals[m]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _norm(parts: Sequence[Optional[Tensor]]) -> float:
    total = 0.0
    for part in parts:
        if part is None:
            continue
        val = part.detach().float()
        total += float((val * val).sum().item())
    return math.sqrt(total)


def _dot(xs: Sequence[Optional[Tensor]], ys: Sequence[Optional[Tensor]]) -> float:
    total = 0.0
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        total += float((x.detach().float() * y.detach().float()).sum().item())
    return total


def _cos(xs: Sequence[Optional[Tensor]], ys: Sequence[Optional[Tensor]]) -> float:
    nx = _norm(xs)
    ny = _norm(ys)
    if nx <= EPS or ny <= EPS:
        return float("nan")
    return float(_dot(xs, ys) / (nx * ny))


def _safe_cpu(value: Tensor) -> Tensor:
    return value.detach().float().cpu()


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (torch.nan_to_num(image.detach().float(), nan=0.0).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=scale, neginf=0.0)
    arr = (vals.clamp_min(0.0) / max(float(scale), EPS)).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _write_plot(path: Path, title: str, rows: Sequence[Mapping[str, Any]], x_key: str, y_keys: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Image.new("RGB", (640, 360), (255, 255, 255)).save(path)
        return
    xs = [str(row.get(x_key, "")) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(xs)), 4.8))
    for key in y_keys:
        ys = []
        for row in rows:
            try:
                ys.append(float(row.get(key, float("nan"))))
            except Exception:
                ys.append(float("nan"))
        ax.plot(xs, ys, marker="o", label=key)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_bar(path: Path, title: str, rows: Sequence[Mapping[str, Any]], label_key: str, value_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(label_key, "")) for row in rows]
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(value_key, float("nan"))))
        except Exception:
            vals.append(float("nan"))
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * max(1, len(labels))), 4.8))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(value_key)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _source_audit(repo: Path) -> Dict[str, Any]:
    config_text = (repo / "water_splatting/water_splatting_config.py").read_text(encoding="utf8")
    model_text = (repo / "water_splatting/water_splatting.py").read_text(encoding="utf8")
    trainer_path = Path("/opt/anaconda3/envs/water_splatting/lib/python3.8/site-packages/nerfstudio/engine/trainer.py")
    optim_path = Path("/opt/anaconda3/envs/water_splatting/lib/python3.8/site-packages/nerfstudio/engine/optimizers.py")
    sched_path = Path("/opt/anaconda3/envs/water_splatting/lib/python3.8/site-packages/nerfstudio/engine/schedulers.py")
    rows = []
    config = path_audit.RUNS["BND-K1"]
    config_path = repo / config.config_relpath
    actual = _actual_step(config_path, 1000)
    ckpt_path = _available_steps(config_path)[actual] if actual is not None else None
    opt_state = torch.load(ckpt_path, map_location="cpu")["optimizers"] if ckpt_path is not None else {}
    for group in CURRENT_PARAM_GROUPS:
        sd = opt_state.get(group, {})
        pg = (sd.get("param_groups") or [{}])[0]
        rows.append(
            {
                "group": group,
                "optimizer": "torch.optim.Adam",
                "betas": str(pg.get("betas", (0.9, 0.999))),
                "eps": pg.get("eps", ""),
                "weight_decay": pg.get("weight_decay", ""),
                "amsgrad": pg.get("amsgrad", ""),
                "initial_lr_config": {
                    "means": 1.6e-4,
                    "features_dc": 0.0025,
                    "features_rest": 0.000125,
                    "opacities": 0.05,
                    "scales": 0.005,
                    "quats": 0.001,
                    "medium_mlp": 0.001,
                    "direction_encoding": 0.001,
                }.get(group, ""),
                "current_lr_1k": pg.get("lr", ""),
                "scheduler": "ExponentialDecayScheduler/LambdaLR",
                "max_norm": 0.001 if group in ("medium_mlp", "direction_encoding") else "",
                "trainable_in_current_model": True,
            }
        )
    for step in (3000, 5000, 8000):
        actual_step = _actual_step(config_path, step)
        if actual_step is None:
            continue
        ckpt = torch.load(_available_steps(config_path)[actual_step], map_location="cpu")
        for row in rows:
            sd = ckpt["optimizers"].get(row["group"], {})
            pg = (sd.get("param_groups") or [{}])[0]
            row[f"current_lr_{step//1000}k"] = pg.get("lr", "")
    return {
        "source_files": {
            "water_splatting_config.py": "defines AdamOptimizerConfig and ExponentialDecaySchedulerConfig per group",
            "water_splatting.py": "get_param_groups returns Gaussian groups plus medium_mlp and direction_encoding; camera_opt is not returned",
            str(trainer_path): "training step zeroes grads, forwards loss, backward, optimizer step, then scheduler step if scaler did not drop",
            str(optim_path): "Nerfstudio optimizer wrapper clips max_norm per group before optimizer.step",
            str(sched_path): "ExponentialDecayScheduler uses exponential interpolation between initial lr and lr_final",
        },
        "group_rows": rows,
        "current_param_groups": list(CURRENT_PARAM_GROUPS),
        "camera_opt_status": "configured in method config but not returned by WaterSplattingModel.get_param_groups()",
        "checkpoint_extra_groups": sorted(set(opt_state.keys()) - set(CURRENT_PARAM_GROUPS)),
        "training_loop_update_semantics": "current optimizer param_group lr is used for the update; scheduler_step_all(step) runs after optimizer_scaler_step_some",
        "zero_grad_semantics": "Trainer.zero_grad_some(needs_zero) before forward/backward; this audit independently zeroes gradients per condition.",
        "excerpts": {
            "config_contains_adam_groups": "AdamOptimizerConfig(lr=..., eps=1e-15) per optimizer group in water_splatting_config.py",
            "model_get_param_groups": "WaterSplattingModel.get_param_groups returns get_gaussian_param_groups(), medium_mlp, direction_encoding",
            "trainer_order": "zero_grad -> pipeline.get_train_loss_dict -> backward -> optimizer_scaler_step_some -> scheduler_step_all",
        },
        "source_text_sha256": {
            "water_splatting_config.py": hashlib.sha256(config_text.encode("utf8")).hexdigest(),
            "water_splatting.py": hashlib.sha256(model_text.encode("utf8")).hexdigest(),
            str(trainer_path): _sha256(trainer_path),
            str(optim_path): _sha256(optim_path),
            str(sched_path): _sha256(sched_path),
        },
    }


def _checkpoint_manifest(repo: Path, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("BND-K1", "CDEPTH"):
        config_path = repo / RUNS[run].config_relpath
        for nominal_step in HISTORICAL_STEPS:
            actual = _actual_step(config_path, nominal_step)
            ckpt_path = _available_steps(config_path).get(actual) if actual is not None else None
            row = {
                "scene": SCENE,
                "run": run,
                "nominal_step": nominal_step,
                "actual_step": actual if actual is not None else "",
                "config_path": str(config_path),
                "checkpoint_path": str(ckpt_path) if ckpt_path else "",
                "checkpoint_exists": bool(ckpt_path and ckpt_path.exists()),
            }
            if ckpt_path and ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                row["checkpoint_step_field"] = int(state.get("step", actual))
                row["gaussian_count"] = int(state["pipeline"]["_model.gauss_params.means"].shape[0])
            rows.append(row)
    _write_csv(output_dir / "checkpoint_manifest.csv", rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": rows})
    return rows


def _optimizer_state_availability(repo: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("BND-K1", "CDEPTH"):
        config_path = repo / RUNS[run].config_relpath
        for nominal_step in HISTORICAL_STEPS:
            actual = _actual_step(config_path, nominal_step)
            if actual is None:
                rows.append({"run": run, "nominal_step": nominal_step, "tier": "NO_OPTIMIZER_STATE"})
                continue
            ckpt_path = _available_steps(config_path)[actual]
            ckpt = torch.load(ckpt_path, map_location="cpu")
            opt = ckpt.get("optimizers", {})
            sched = ckpt.get("schedulers", {})
            for group in CURRENT_PARAM_GROUPS:
                sd = opt.get(group)
                pg = (sd.get("param_groups") or [{}])[0] if isinstance(sd, Mapping) else {}
                state = sd.get("state", {}) if isinstance(sd, Mapping) else {}
                param_ids = list(pg.get("params", []))
                has_scheduler = group in sched
                exact = bool(
                    sd is not None
                    and param_ids
                    and all(pid in state for pid in param_ids)
                    and all(
                        all(name in state[pid] for name in ("step", "exp_avg", "exp_avg_sq"))
                        for pid in param_ids
                    )
                    and "lr" in pg
                    and has_scheduler
                )
                rows.append(
                    {
                        "scene": SCENE,
                        "run": run,
                        "nominal_step": nominal_step,
                        "actual_step": actual,
                        "group": group,
                        "tier": "EXACT_OPTIMIZER_STATE" if exact else "PARTIAL_OPTIMIZER_STATE",
                        "param_ids": ";".join(str(pid) for pid in param_ids),
                        "state_count": len(state),
                        "has_scheduler_state": has_scheduler,
                        "current_lr": pg.get("lr", ""),
                        "optimizer_step": float(next(iter(state.values())).get("step", torch.tensor(float("nan"))).item())
                        if state
                        else "",
                    }
                )
            extra = sorted(set(opt.keys()) - set(CURRENT_PARAM_GROUPS))
            rows.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": actual,
                    "group": "__checkpoint_extra_groups__",
                    "tier": "INFO_ONLY",
                    "extra_groups": ";".join(extra),
                }
            )
    by_step: Dict[Tuple[str, int], str] = {}
    for run in ("BND-K1", "CDEPTH"):
        for step in HISTORICAL_STEPS:
            subset = [row for row in rows if row.get("run") == run and row.get("nominal_step") == step and row.get("group") in CURRENT_PARAM_GROUPS]
            if subset and all(row.get("tier") == "EXACT_OPTIMIZER_STATE" for row in subset):
                by_step[(run, step)] = "EXACT_OPTIMIZER_STATE"
            elif subset:
                by_step[(run, step)] = "PARTIAL_OPTIMIZER_STATE"
            else:
                by_step[(run, step)] = "NO_OPTIMIZER_STATE"
    summary = {
        f"{run}_{step}": tier for (run, step), tier in sorted(by_step.items())
    }
    summary["OPTIMIZER_AWARE_VIRTUAL_STEP_VALID"] = (
        by_step.get(("BND-K1", 1000)) == "EXACT_OPTIMIZER_STATE"
        and by_step.get(("BND-K1", 3000)) == "EXACT_OPTIMIZER_STATE"
    )
    _write_csv(output_dir / "optimizer_state_availability.csv", rows)
    _write_json(output_dir / "optimizer_state_availability.json", {"rows": rows, "summary": summary})
    return rows, summary


def _render_items(repo: Path, run: str, nominal_step: int) -> Dict[str, Dict[str, Any]]:
    loaded = None
    try:
        loaded = _load_run(repo, run, nominal_step, load_depths=False)
        model = loaded.pipeline.model
        model.eval()
        model.step = loaded.loaded_step
        items: Dict[str, Dict[str, Any]] = {}
        for eval_index, view_id, camera, batch in _eval_records(loaded):
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
                "depth",
            )
            items[view_id] = {
                "eval_index": eval_index,
                "view_id": view_id,
                "gt": _safe_cpu(gt),
                "outputs": {key: _safe_cpu(outputs[key]) for key in keep if key in outputs},
            }
        return items
    finally:
        _release(loaded)


def _error(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)


def _build_outcome_regions(repo: Path, output_dir: Path) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Any]]:
    m1 = _render_items(repo, "M1", FINAL_STEP)
    k1 = _render_items(repo, "BND-K1", FINAL_STEP)
    cd = _render_items(repo, "CDEPTH", FINAL_STEP)
    view_ids = [view_id for view_id in EVAL_VIEWS if view_id in m1 and view_id in k1 and view_id in cd]
    pos_values: List[Tensor] = []
    neg_values: List[Tensor] = []
    gains: Dict[str, Tensor] = {}
    highs: Dict[str, Tensor] = {}
    for view_id in view_ids:
        gain = _error(k1[view_id]["outputs"]["pred_image"], m1[view_id]["gt"]) - _error(
            cd[view_id]["outputs"]["pred_image"], m1[view_id]["gt"]
        )
        high = (m1[view_id]["outputs"]["accumulation"][..., 0] > 0.01) & (
            m1[view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 1.0
        )
        gains[view_id] = gain
        highs[view_id] = high
        vals = gain[high]
        if (vals > 0).any():
            pos_values.append(vals[vals > 0])
        if (vals < 0).any():
            neg_values.append(-vals[vals < 0])
    strong_gain_threshold = _q(torch.cat(pos_values) if pos_values else torch.empty(0), 0.75)
    strong_harm_threshold = _q(torch.cat(neg_values) if neg_values else torch.empty(0), 0.75)
    regions: Dict[str, Dict[str, Tensor]] = {}
    for view_id in view_ids:
        gain = gains[view_id]
        high = highs[view_id]
        regions[view_id] = {
            "global": torch.ones_like(high, dtype=torch.bool),
            "M1_HIGH_J": high,
            "HJ_GAIN": high & (gain > 0),
            "HJ_HARM": high & (gain < 0),
            "HJ_STRONG_GAIN": high & (gain >= strong_gain_threshold),
            "HJ_STRONG_HARM": high & ((-gain) >= strong_harm_threshold),
        }
    meta = {
        "scene": SCENE,
        "view_ids": view_ids,
        "M1_HIGH_J_definition": "final M1 accumulation > 0.01 and final M1 clear_object_fullsh_raw max RGB channel > 1.0",
        "HJ_GAIN_definition": "M1_HIGH_J and final RGB MSE(K1) - RGB MSE(CDEPTH) > 0; post-hoc future-outcome mask",
        "HJ_HARM_definition": "M1_HIGH_J and final RGB MSE(K1) - RGB MSE(CDEPTH) < 0; post-hoc future-outcome mask",
        "HJ_STRONG_GAIN_definition": "pooled top quartile of positive HJ_GAIN values",
        "HJ_STRONG_HARM_definition": "pooled top quartile of negative HJ_HARM magnitude",
        "strong_gain_threshold": strong_gain_threshold,
        "strong_harm_threshold": strong_harm_threshold,
        "post_hoc_warning": "Outcome masks are diagnostics only; they are never training signals.",
    }
    _write_json(output_dir / "outcome_region_definitions.json", meta)
    return regions, meta


def _historical_trajectory(repo: Path, regions: Mapping[str, Mapping[str, Tensor]], output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    for nominal_step in HISTORICAL_STEPS:
        common = all(_actual_step(repo / RUNS[run].config_relpath, nominal_step) is not None for run in ("BND-K1", "CDEPTH"))
        if not common:
            continue
        k1 = _render_items(repo, "BND-K1", nominal_step)
        cd = _render_items(repo, "CDEPTH", nominal_step)
        for view_id in EVAL_VIEWS:
            if view_id not in k1 or view_id not in cd or view_id not in regions:
                continue
            for region in REGIONS:
                mask = regions[view_id][region]
                n = int(mask.sum().item())
                if n == 0:
                    continue
                gt = k1[view_id]["gt"]
                k_pred = k1[view_id]["outputs"]["pred_image"]
                c_pred = cd[view_id]["outputs"]["pred_image"]
                rgb_mse_k1 = float(_masked(_error(k_pred, gt), mask).mean().item())
                rgb_mse_cd = float(_masked(_error(c_pred, gt), mask).mean().item())
                delta_d = (cd[view_id]["outputs"]["direct_object_signal"] - k1[view_id]["outputs"]["direct_object_signal"]).abs().mean(dim=-1)
                delta_b = (cd[view_id]["outputs"]["rgb_medium"] - k1[view_id]["outputs"]["rgb_medium"]).abs().mean(dim=-1)
                delta_i = (c_pred - k_pred).abs().mean(dim=-1)
                delta_j = (cd[view_id]["outputs"]["clear_object_fullsh_raw"] - k1[view_id]["outputs"]["clear_object_fullsh_raw"]).abs().mean(dim=-1)
                delta_t = (cd[view_id]["outputs"]["transmission"] - k1[view_id]["outputs"]["transmission"]).abs().mean(dim=-1)
                delta_tau = (cd[view_id]["outputs"]["tau_D"] - k1[view_id]["outputs"]["tau_D"]).abs().mean(dim=-1)
                d_mean = float(delta_d[mask].mean().item())
                b_mean = float(delta_b[mask].mean().item())
                row = {
                    "scene": SCENE,
                    "nominal_step": nominal_step,
                    "view_id": view_id,
                    "region": region,
                    "pixels": n,
                    "K1_rgb_mse": rgb_mse_k1,
                    "CDEPTH_rgb_mse": rgb_mse_cd,
                    "CDEPTH_minus_K1_rgb_mse_gain": rgb_mse_k1 - rgb_mse_cd,
                    "mean_abs_delta_D_DIRECT": d_mean,
                    "mean_abs_delta_B_MEDIUM": b_mean,
                    "mean_abs_delta_I_PRED": float(delta_i[mask].mean().item()),
                    "mean_abs_delta_J_CLEAR": float(delta_j[mask].mean().item()),
                    "mean_abs_delta_T": float(delta_t[mask].mean().item()),
                    "mean_abs_delta_tau": float(delta_tau[mask].mean().item()),
                    "DIRECT_MEDIUM_RESPONSE_RATIO": d_mean / (b_mean + EPS),
                }
                rows.append(row)
        for region in ("HJ_GAIN", "HJ_HARM"):
            pooled = [row for row in rows if row["nominal_step"] == nominal_step and row["region"] == region]
            if pooled:
                direct_rows.append(
                    {
                        "scene": SCENE,
                        "nominal_step": nominal_step,
                        "region": region,
                        "RGB_MSE_GAIN_mean": _mean(row["CDEPTH_minus_K1_rgb_mse_gain"] for row in pooled),
                        "mean_abs_delta_D_DIRECT": _mean(row["mean_abs_delta_D_DIRECT"] for row in pooled),
                        "mean_abs_delta_B_MEDIUM": _mean(row["mean_abs_delta_B_MEDIUM"] for row in pooled),
                        "DIRECT_MEDIUM_RESPONSE_RATIO": _mean(row["DIRECT_MEDIUM_RESPONSE_RATIO"] for row in pooled),
                    }
                )
    _write_csv(output_dir / "historical_direct_trajectory.csv", rows)
    _write_json(output_dir / "historical_direct_trajectory.json", {"rows": rows})
    _write_csv(output_dir / "direct_medium_trajectory.csv", direct_rows)
    _write_json(output_dir / "direct_medium_trajectory.json", {"rows": direct_rows})
    return rows, direct_rows


def _param_groups(model: Any) -> Dict[str, List[Tensor]]:
    groups = model.get_param_groups()
    return {name: list(groups.get(name, [])) for name in CURRENT_PARAM_GROUPS}


def _zero_grads(params: Mapping[str, Sequence[Tensor]]) -> None:
    for plist in params.values():
        for param in plist:
            param.grad = None


def _clone_grads(params: Mapping[str, Sequence[Tensor]]) -> Dict[str, List[Tensor]]:
    out: Dict[str, List[Tensor]] = {}
    for group, plist in params.items():
        out[group] = []
        for param in plist:
            if param.grad is None:
                out[group].append(torch.zeros_like(param.detach()))
            else:
                out[group].append(param.grad.detach().clone())
    return out


def _forward_loss(model: Any, camera: Any, batch: Mapping[str, Any], condition: str) -> Tuple[Dict[str, Tensor], Tensor, Dict[str, Any]]:
    outputs = model.get_outputs(camera.to(model.device))
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
    else:
        raise ValueError(condition)
    row = {f"loss_{key}": float(value.detach().cpu().item()) for key, value in losses.items()}
    row.update({f"metric_{key}": float(value.detach().cpu().item()) for key, value in metrics.items()})
    return outputs, loss, row


def _condition_grads(model: Any, camera: Any, batch: Mapping[str, Any], condition: str, params: Mapping[str, Sequence[Tensor]]) -> Tuple[Dict[str, List[Tensor]], Dict[str, Any]]:
    model.train()
    _zero_grads(params)
    outputs, loss, loss_row = _forward_loss(model, camera, batch, condition)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite {condition} loss: {float(loss.detach().cpu().item())}")
    loss.backward()
    grads = _clone_grads(params)
    _zero_grads(params)
    model.config.coarse_depth_supervision_enabled = False
    del outputs, loss
    return grads, loss_row


def _pre_backward_equivalence(model: Any, camera: Any, batch: Mapping[str, Any], params: Mapping[str, Sequence[Tensor]]) -> Dict[str, Any]:
    model.train()
    _zero_grads(params)
    model.config.coarse_depth_supervision_enabled = False
    outputs_r = model.get_outputs(camera.to(model.device))
    model.config.coarse_depth_supervision_enabled = True
    model.config.coarse_depth_supervision_weight = 0.1
    outputs_rd = model.get_outputs(camera.to(model.device))
    keys = ("pred_image", "direct_object_signal", "rgb_medium", "depth", "clear_object_fullsh_raw", "transmission", "tau_D", "accumulation")
    row: Dict[str, Any] = {}
    max_seen = 0.0
    for key in keys:
        if key in outputs_r and key in outputs_rd:
            diff = float((outputs_r[key].detach() - outputs_rd[key].detach()).abs().max().cpu().item())
            row[f"{key}_max_abs_diff"] = diff
            max_seen = max(max_seen, diff)
        else:
            row[f"{key}_max_abs_diff"] = ""
    row["overall_max_abs_diff"] = max_seen
    row["pass"] = bool(max_seen <= 1e-6)
    model.config.coarse_depth_supervision_enabled = False
    _zero_grads(params)
    del outputs_r, outputs_rd
    return row


def _clip_group_grads(grads: Sequence[Tensor], max_norm: Optional[float]) -> List[Tensor]:
    if max_norm is None:
        return [g.detach().clone() for g in grads]
    total = math.sqrt(sum(float((g.detach().float() * g.detach().float()).sum().item()) for g in grads))
    if total <= max_norm or total <= EPS:
        return [g.detach().clone() for g in grads]
    scale = float(max_norm) / (total + 1e-6)
    return [g.detach().clone() * scale for g in grads]


def _optimizer_state_for_group(optimizer_state: Mapping[str, Any], group: str) -> Tuple[Dict[str, Any], List[Any], List[Mapping[str, Tensor]]]:
    sd = optimizer_state[group]
    pg = sd["param_groups"][0]
    param_ids = list(pg["params"])
    states = [sd["state"][pid] for pid in param_ids]
    return pg, param_ids, states


def _adam_delta(param: Tensor, grad: Tensor, state: Mapping[str, Tensor], param_group: Mapping[str, Any]) -> Tensor:
    if grad is None:
        return torch.zeros_like(param)
    lr = float(param_group["lr"])
    beta1, beta2 = param_group.get("betas", (0.9, 0.999))
    eps = float(param_group.get("eps", 1e-8))
    weight_decay = float(param_group.get("weight_decay", 0.0))
    if bool(param_group.get("amsgrad", False)):
        raise RuntimeError("amsgrad=True is not implemented in this diagnostic")
    g = grad.detach()
    if weight_decay != 0.0:
        g = g.add(param.detach(), alpha=weight_decay)
    exp_avg = state["exp_avg"].to(device=param.device, dtype=param.dtype)
    exp_avg_sq = state["exp_avg_sq"].to(device=param.device, dtype=param.dtype)
    step = float(state["step"].detach().cpu().item()) + 1.0
    exp_avg_new = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
    exp_avg_sq_new = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    step_size = lr / bias_correction1
    denom = exp_avg_sq_new.sqrt().div(math.sqrt(bias_correction2)).add(eps)
    return exp_avg_new.div(denom).mul(-step_size)


def _virtual_updates(
    params: Mapping[str, Sequence[Tensor]],
    optimizer_state: Mapping[str, Any],
    grads_r: Mapping[str, Sequence[Tensor]],
    grads_rd: Mapping[str, Sequence[Tensor]],
) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
    delta_r: Dict[str, List[Tensor]] = {}
    delta_rd: Dict[str, List[Tensor]] = {}
    delta_depth: Dict[str, List[Tensor]] = {}
    for group, plist in params.items():
        if group not in optimizer_state:
            delta_r[group] = [torch.zeros_like(param) for param in plist]
            delta_rd[group] = [torch.zeros_like(param) for param in plist]
            delta_depth[group] = [torch.zeros_like(param) for param in plist]
            continue
        pg, _param_ids, states = _optimizer_state_for_group(optimizer_state, group)
        max_norm = 0.001 if group in ("medium_mlp", "direction_encoding") else None
        clipped_r = _clip_group_grads(list(grads_r[group]), max_norm)
        clipped_rd = _clip_group_grads(list(grads_rd[group]), max_norm)
        drs: List[Tensor] = []
        drds: List[Tensor] = []
        dds: List[Tensor] = []
        for param, grad_r, grad_rd, state in zip(plist, clipped_r, clipped_rd, states):
            d_r = _adam_delta(param, grad_r, state, pg)
            d_rd = _adam_delta(param, grad_rd, state, pg)
            drs.append(d_r)
            drds.append(d_rd)
            dds.append(d_rd - d_r)
        delta_r[group] = drs
        delta_rd[group] = drds
        delta_depth[group] = dds
    return delta_r, delta_rd, delta_depth


def _adam_equivalence_check(
    param: Tensor,
    grad: Tensor,
    state: Mapping[str, Tensor],
    param_group: Mapping[str, Any],
) -> Dict[str, Any]:
    count = min(1024, param.shape[0]) if param.ndim > 0 else 1
    p0 = param.detach()[:count].clone() if param.ndim > 0 else param.detach().clone()
    g = grad.detach()[:count].clone() if grad.ndim > 0 else grad.detach().clone()
    exp_avg = state["exp_avg"][:count].clone() if state["exp_avg"].ndim > 0 else state["exp_avg"].clone()
    exp_avg_sq = state["exp_avg_sq"][:count].clone() if state["exp_avg_sq"].ndim > 0 else state["exp_avg_sq"].clone()
    p = torch.nn.Parameter(p0.clone())
    opt = torch.optim.Adam(
        [p],
        lr=float(param_group["lr"]),
        betas=param_group.get("betas", (0.9, 0.999)),
        eps=float(param_group.get("eps", 1e-8)),
        weight_decay=float(param_group.get("weight_decay", 0.0)),
        amsgrad=bool(param_group.get("amsgrad", False)),
    )
    opt.state[p]["step"] = state["step"].detach().clone().to(p.device)
    opt.state[p]["exp_avg"] = exp_avg.to(p.device)
    opt.state[p]["exp_avg_sq"] = exp_avg_sq.to(p.device)
    p.grad = g.to(p.device)
    opt.step()
    real_delta = p.detach() - p0.to(p.device)
    sliced_state = {
        "step": state["step"].detach().clone().to(p.device),
        "exp_avg": exp_avg.to(p.device),
        "exp_avg_sq": exp_avg_sq.to(p.device),
    }
    analytic_delta = _adam_delta(p0.to(p.device), g.to(p.device), sliced_state, param_group)
    max_abs = float((real_delta - analytic_delta).abs().max().cpu().item())
    tolerance = 1e-6
    return {
        "group": "means",
        "sample_count": count,
        "max_abs_difference": max_abs,
        "tolerance": tolerance,
        "pass": bool(max_abs <= tolerance),
        "semantics": "isolated tensor torch.optim.Adam.step compared with analytic Adam delta; no model/checkpoint mutation",
    }


def _gradient_rows(
    nominal_step: int,
    train_view: str,
    grads_r: Mapping[str, Sequence[Tensor]],
    grads_rd: Mapping[str, Sequence[Tensor]],
    loss_r: Mapping[str, Any],
    loss_rd: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in CURRENT_PARAM_GROUPS:
        gd = [rd - r for r, rd in zip(grads_r[group], grads_rd[group])]
        nr = _norm(grads_r[group])
        nrd = _norm(grads_rd[group])
        nd = _norm(gd)
        rows.append(
            {
                "scene": SCENE,
                "nominal_step": nominal_step,
                "train_view_id": train_view,
                "group": group,
                "grad_R_norm": nr,
                "grad_RD_norm": nrd,
                "grad_DEPTH_INCREMENT_norm": nd,
                "GRAD_INCREMENT_RATIO": nd / (nr + EPS),
                "cos_grad_R_depth_increment": _cos(grads_r[group], gd),
                "loss_R_main": loss_r.get("loss_main_loss", ""),
                "loss_RD_main": loss_rd.get("loss_main_loss", ""),
                "loss_RD_coarse_depth": loss_rd.get("loss_coarse_depth_loss", ""),
                "metric_coarse_depth_loss_raw": loss_rd.get("metric_coarse_depth_loss_raw", ""),
            }
        )
    return rows


def _update_rows(
    nominal_step: int,
    train_view: str,
    delta_r: Mapping[str, Sequence[Tensor]],
    delta_rd: Mapping[str, Sequence[Tensor]],
    delta_depth: Mapping[str, Sequence[Tensor]],
    grad_rows_for_camera: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grad_by_group = {row["group"]: row for row in grad_rows_for_camera}
    rows: List[Dict[str, Any]] = []
    for group in CURRENT_PARAM_GROUPS:
        nr = _norm(delta_r[group])
        nrd = _norm(delta_rd[group])
        nd = _norm(delta_depth[group])
        grad_ratio = float(grad_by_group.get(group, {}).get("GRAD_INCREMENT_RATIO", float("nan")))
        rows.append(
            {
                "scene": SCENE,
                "nominal_step": nominal_step,
                "train_view_id": train_view,
                "group": group,
                "Delta_theta_R_norm": nr,
                "Delta_theta_RD_norm": nrd,
                "Delta_theta_DEPTH_norm": nd,
                "UPDATE_INCREMENT_RATIO": nd / (nr + EPS),
                "cos_update_R_depth_increment": _cos(delta_r[group], delta_depth[group]),
                "OPTIMIZER_AMPLIFICATION": (nd / (nr + EPS)) / (grad_ratio + EPS) if math.isfinite(grad_ratio) else float("nan"),
            }
        )
    return rows


def _physical_rows(
    nominal_step: int,
    train_view: str,
    params: Mapping[str, Sequence[Tensor]],
    delta_depth: Mapping[str, Sequence[Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    means = params["means"][0].detach()
    d_means = delta_depth["means"][0].detach()
    row = {"scene": SCENE, "nominal_step": nominal_step, "train_view_id": train_view, "group": "means", "physical_metric": "world_space_displacement_norm"}
    row.update(_stats(torch.linalg.norm(d_means.float(), dim=-1), ""))
    rows.append(row)

    scales = params["scales"][0].detach()
    d_scales = delta_depth["scales"][0].detach()
    scale_before = torch.exp(scales.float())
    scale_after = torch.exp((scales + d_scales).float())
    rel = (scale_after - scale_before).abs() / scale_before.clamp_min(EPS)
    row = {"scene": SCENE, "nominal_step": nominal_step, "train_view_id": train_view, "group": "scales", "physical_metric": "activated_exp_scale_relative_abs_delta"}
    row.update(_stats(rel.reshape(-1), ""))
    rows.append(row)

    quats = params["quats"][0].detach().float()
    d_quats = delta_depth["quats"][0].detach().float()
    q0 = quats / quats.norm(dim=-1, keepdim=True).clamp_min(EPS)
    q1 = (quats + d_quats) / (quats + d_quats).norm(dim=-1, keepdim=True).clamp_min(EPS)
    dots = (q0 * q1).sum(dim=-1).abs().clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dots)
    row = {"scene": SCENE, "nominal_step": nominal_step, "train_view_id": train_view, "group": "quats", "physical_metric": "normalized_quaternion_angle_radians"}
    row.update(_stats(angles, ""))
    rows.append(row)

    opacities = params["opacities"][0].detach().float()
    d_opacity = delta_depth["opacities"][0].detach().float()
    o_delta = (torch.sigmoid(opacities + d_opacity) - torch.sigmoid(opacities)).abs()
    row = {"scene": SCENE, "nominal_step": nominal_step, "train_view_id": train_view, "group": "opacities", "physical_metric": "sigmoid_opacity_abs_delta"}
    row.update(_stats(o_delta.reshape(-1), ""))
    rows.append(row)
    return rows


def _momentum_rows(
    nominal_step: int,
    train_view: str,
    optimizer_state: Mapping[str, Any],
    grads_r: Mapping[str, Sequence[Tensor]],
    grads_rd: Mapping[str, Sequence[Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in CURRENT_PARAM_GROUPS:
        if group not in optimizer_state:
            continue
        pg, _ids, states = _optimizer_state_for_group(optimizer_state, group)
        del pg
        exp = [state["exp_avg"].to(grads_r[group][idx].device, dtype=grads_r[group][idx].dtype) for idx, state in enumerate(states)]
        gd = [rd - r for r, rd in zip(grads_r[group], grads_rd[group])]
        rows.append(
            {
                "scene": SCENE,
                "nominal_step": nominal_step,
                "train_view_id": train_view,
                "group": group,
                "cos_depth_increment_gradient_vs_adam_exp_avg": _cos(gd, exp),
                "exp_avg_norm": _norm(exp),
                "depth_increment_grad_norm": _norm(gd),
            }
        )
    return rows


def _save_param_state(params: Mapping[str, Sequence[Tensor]]) -> Dict[str, List[Tensor]]:
    return {group: [param.detach().clone() for param in plist] for group, plist in params.items()}


def _copy_state(params: Mapping[str, Sequence[Tensor]], state: Mapping[str, Sequence[Tensor]]) -> None:
    with torch.no_grad():
        for group, plist in params.items():
            for param, value in zip(plist, state[group]):
                param.copy_(value)


def _apply_state_plus(
    params: Mapping[str, Sequence[Tensor]],
    base: Mapping[str, Sequence[Tensor]],
    delta_r: Mapping[str, Sequence[Tensor]],
    delta_depth: Optional[Mapping[str, Sequence[Tensor]]] = None,
    depth_groups: Optional[Sequence[str]] = None,
) -> None:
    depth_groups_set = set(depth_groups or CURRENT_PARAM_GROUPS)
    with torch.no_grad():
        for group, plist in params.items():
            for idx, param in enumerate(plist):
                value = base[group][idx] + delta_r[group][idx]
                if delta_depth is not None and group in depth_groups_set:
                    value = value + delta_depth[group][idx]
                param.copy_(value)


def _model_restore_delta(params: Mapping[str, Sequence[Tensor]], base: Mapping[str, Sequence[Tensor]]) -> float:
    max_delta = 0.0
    with torch.no_grad():
        for group, plist in params.items():
            for idx, param in enumerate(plist):
                param.copy_(base[group][idx])
                diff = float((param.detach() - base[group][idx]).abs().max().cpu().item()) if param.numel() else 0.0
                max_delta = max(max_delta, diff)
    return max_delta


def _render_eval_outputs(model: Any, eval_records: Sequence[Tuple[int, str, Any, Mapping[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    model.eval()
    out: Dict[str, Dict[str, Tensor]] = {}
    for _eval_index, view_id, camera, batch in eval_records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        keep = ("pred_image", "direct_object_signal", "rgb_medium", "clear_object_fullsh_raw", "transmission", "tau_D", "accumulation")
        out[view_id] = {"gt": _safe_cpu(gt)}
        for key in keep:
            if key in outputs:
                out[view_id][key] = _safe_cpu(outputs[key])
    return out


def _virtual_region_rows(
    nominal_step: int,
    train_view: str,
    branch: str,
    base_render: Mapping[str, Mapping[str, Tensor]],
    branch_render: Mapping[str, Mapping[str, Tensor]],
    regions: Mapping[str, Mapping[str, Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in EVAL_VIEWS:
        if view_id not in base_render or view_id not in branch_render or view_id not in regions:
            continue
        for region in REGIONS:
            mask = regions[view_id][region]
            n = int(mask.sum().item())
            if n == 0:
                continue
            gt = base_render[view_id]["gt"]
            i_r = base_render[view_id]["pred_image"]
            i_b = branch_render[view_id]["pred_image"]
            d_resp = (branch_render[view_id]["direct_object_signal"] - base_render[view_id]["direct_object_signal"]).abs().mean(dim=-1)
            b_resp = (branch_render[view_id]["rgb_medium"] - base_render[view_id]["rgb_medium"]).abs().mean(dim=-1)
            i_resp = (i_b - i_r).abs().mean(dim=-1)
            j_resp = (branch_render[view_id]["clear_object_fullsh_raw"] - base_render[view_id]["clear_object_fullsh_raw"]).abs().mean(dim=-1)
            t_resp = (branch_render[view_id]["transmission"] - base_render[view_id]["transmission"]).abs().mean(dim=-1)
            tau_resp = (branch_render[view_id]["tau_D"] - base_render[view_id]["tau_D"]).mean(dim=-1)
            d_mean = float(d_resp[mask].mean().item())
            b_mean = float(b_resp[mask].mean().item())
            row = {
                "scene": SCENE,
                "nominal_step": nominal_step,
                "train_view_id": train_view,
                "eval_view_id": view_id,
                "branch": branch,
                "region": region,
                "pixels": n,
                "VIRTUAL_RGB_GAIN": float(_masked(_error(i_r, gt), mask).mean().item() - _masked(_error(i_b, gt), mask).mean().item()),
                "mean_abs_I_RESPONSE": float(i_resp[mask].mean().item()),
                "mean_abs_D_RESPONSE": d_mean,
                "mean_abs_B_RESPONSE": b_mean,
                "mean_abs_J_RESPONSE": float(j_resp[mask].mean().item()),
                "mean_abs_T_RESPONSE": float(t_resp[mask].mean().item()),
                "mean_delta_tau": float(tau_resp[mask].mean().item()),
                "VIRTUAL_DIRECT_MEDIUM_RATIO": d_mean / (b_mean + EPS),
            }
            rows.append(row)
    return rows


def _boundary_rows(
    nominal_step: int,
    train_view: str,
    branch: str,
    renders: Mapping[str, Mapping[str, Tensor]],
    regions: Mapping[str, Mapping[str, Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in EVAL_VIEWS:
        if view_id not in renders or view_id not in regions:
            continue
        for region in ("global", "HJ_GAIN", "HJ_HARM"):
            mask = regions[view_id][region]
            if int(mask.sum().item()) == 0:
                continue
            j = renders[view_id]["clear_object_fullsh_raw"]
            tau = renders[view_id]["tau_D"].mean(dim=-1)
            t = renders[view_id]["transmission"].mean(dim=-1)
            alpha = renders[view_id]["accumulation"][..., 0]
            vals = _masked(j, mask)
            rows.append(
                {
                    "scene": SCENE,
                    "nominal_step": nominal_step,
                    "train_view_id": train_view,
                    "eval_view_id": view_id,
                    "branch": branch,
                    "region": region,
                    "P_J_gt_0p99": float((vals > 0.99).float().mean().item()) if vals.numel() else float("nan"),
                    "P_J_gt_1p0": float((vals > 1.0).float().mean().item()) if vals.numel() else float("nan"),
                    "tau_mean": float(tau[mask].mean().item()),
                    "T_mean": float(t[mask].mean().item()),
                    "alpha_mean": float(alpha[mask].mean().item()),
                }
            )
    return rows


def _group_additivity_rows(
    nominal_step: int,
    train_view: str,
    base_render: Mapping[str, Mapping[str, Tensor]],
    full_render: Mapping[str, Mapping[str, Tensor]],
    group_renders: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    regions: Mapping[str, Mapping[str, Tensor]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in EVAL_VIEWS:
        if view_id not in base_render or view_id not in full_render or view_id not in regions:
            continue
        i_full = full_render[view_id]["pred_image"] - base_render[view_id]["pred_image"]
        d_full = full_render[view_id]["direct_object_signal"] - base_render[view_id]["direct_object_signal"]
        i_sum = torch.zeros_like(i_full)
        d_sum = torch.zeros_like(d_full)
        for group in GEOMETRY_GROUPS:
            if group not in group_renders or view_id not in group_renders[group]:
                continue
            i_sum = i_sum + (group_renders[group][view_id]["pred_image"] - base_render[view_id]["pred_image"])
            d_sum = d_sum + (group_renders[group][view_id]["direct_object_signal"] - base_render[view_id]["direct_object_signal"])
        for region in REGIONS:
            mask = regions[view_id][region]
            if int(mask.sum().item()) == 0:
                continue
            i_num = _masked((i_full - i_sum).abs().mean(dim=-1), mask).mean()
            i_den = _masked(i_full.abs().mean(dim=-1), mask).mean().clamp_min(EPS)
            d_num = _masked((d_full - d_sum).abs().mean(dim=-1), mask).mean()
            d_den = _masked(d_full.abs().mean(dim=-1), mask).mean().clamp_min(EPS)
            rows.append(
                {
                    "scene": SCENE,
                    "nominal_step": nominal_step,
                    "train_view_id": train_view,
                    "eval_view_id": view_id,
                    "region": region,
                    "GROUP_RESPONSE_NONADDITIVITY_I": float((i_num / i_den).item()),
                    "GROUP_RESPONSE_NONADDITIVITY_D": float((d_num / d_den).item()),
                    "GROUP_ATTRIBUTION_APPROX_ADDITIVE": bool((i_num / i_den).item() <= 0.30),
                }
            )
    return rows


def _run_interventions(
    repo: Path,
    regions: Mapping[str, Mapping[str, Tensor]],
    output_dir: Path,
) -> Dict[str, Any]:
    pre_rows: List[Dict[str, Any]] = []
    raw_grad_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    physical_rows: List[Dict[str, Any]] = []
    momentum_rows: List[Dict[str, Any]] = []
    virtual_rows: List[Dict[str, Any]] = []
    virtual_full_rows: List[Dict[str, Any]] = []
    virtual_group_rows: List[Dict[str, Any]] = []
    additivity_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    tau_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    equivalence: Optional[Dict[str, Any]] = None
    checkpoint_safety_before: List[Dict[str, Any]] = []
    checkpoint_safety_after: List[Dict[str, Any]] = []

    for nominal_step in INTERVENTION_STEPS:
        loaded = None
        try:
            loaded = _load_run(repo, "BND-K1", nominal_step, load_depths=True)
            ckpt_path = Path(loaded.checkpoint_path)
            checkpoint_safety_before.append(_file_state(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            optimizer_state = ckpt["optimizers"]
            model = loaded.pipeline.model
            model.step = loaded.loaded_step
            params = _param_groups(model)
            base_state = _save_param_state(params)
            eval_records = _eval_records(loaded)
            train_records = _train_bank_records(loaded)
            for _train_index, train_view, camera, batch in train_records:
                model.step = loaded.loaded_step
                _copy_state(params, base_state)
                pre = _pre_backward_equivalence(model, camera, batch, params)
                pre.update({"scene": SCENE, "nominal_step": nominal_step, "train_view_id": train_view})
                pre_rows.append(pre)
                if not pre["pass"]:
                    raise RuntimeError(f"Pre-backward forward equivalence failed at {nominal_step} {train_view}: {pre}")
                grads_r, loss_r = _condition_grads(model, camera, batch, "R", params)
                grads_rd, loss_rd = _condition_grads(model, camera, batch, "RD", params)
                camera_grad_rows = _gradient_rows(nominal_step, train_view, grads_r, grads_rd, loss_r, loss_rd)
                raw_grad_rows.extend(camera_grad_rows)
                delta_r, delta_rd, delta_depth = _virtual_updates(params, optimizer_state, grads_r, grads_rd)
                if equivalence is None:
                    pg, _ids, states = _optimizer_state_for_group(optimizer_state, "means")
                    equivalence = _adam_equivalence_check(params["means"][0], grads_r["means"][0], states[0], pg)
                update_rows.extend(_update_rows(nominal_step, train_view, delta_r, delta_rd, delta_depth, camera_grad_rows))
                physical_rows.extend(_physical_rows(nominal_step, train_view, params, delta_depth))
                momentum_rows.extend(_momentum_rows(nominal_step, train_view, optimizer_state, grads_r, grads_rd))

                # theta_R baseline.
                _apply_state_plus(params, base_state, delta_r)
                theta_r_render = _render_eval_outputs(model, eval_records)
                boundary_rows.extend(_boundary_rows(nominal_step, train_view, "theta_R", theta_r_render, regions))

                # Full theta_RD.
                _apply_state_plus(params, base_state, delta_r, delta_depth, depth_groups=CURRENT_PARAM_GROUPS)
                theta_rd_render = _render_eval_outputs(model, eval_records)
                full_rows = _virtual_region_rows(nominal_step, train_view, "FULL_RD_MINUS_R", theta_r_render, theta_rd_render, regions)
                virtual_rows.extend(full_rows)
                virtual_full_rows.extend(full_rows)
                boundary_rows.extend(_boundary_rows(nominal_step, train_view, "theta_RD", theta_rd_render, regions))

                # Group-isolated geometry depth increments.
                group_renders: Dict[str, Dict[str, Mapping[str, Tensor]]] = {}
                for group in GEOMETRY_GROUPS:
                    _apply_state_plus(params, base_state, delta_r, delta_depth, depth_groups=(group,))
                    branch_render = _render_eval_outputs(model, eval_records)
                    group_renders[group] = branch_render
                    grow = _virtual_region_rows(nominal_step, train_view, f"GROUP_{group}", theta_r_render, branch_render, regions)
                    virtual_rows.extend(grow)
                    virtual_group_rows.extend(grow)

                additivity_rows.extend(
                    _group_additivity_rows(nominal_step, train_view, theta_r_render, theta_rd_render, group_renders, regions)
                )

                # Tau/T controls for full response are duplicated in virtual rows; keep a compact table.
                for row in full_rows:
                    if row["region"] in ("HJ_GAIN", "HJ_HARM", "global"):
                        tau_rows.append(
                            {
                                "scene": SCENE,
                                "nominal_step": nominal_step,
                                "train_view_id": train_view,
                                "eval_view_id": row["eval_view_id"],
                                "region": row["region"],
                                "mean_delta_tau": row["mean_delta_tau"],
                                "mean_abs_T_RESPONSE": row["mean_abs_T_RESPONSE"],
                                "mean_abs_J_RESPONSE": row["mean_abs_J_RESPONSE"],
                            }
                        )
                restore_delta = _model_restore_delta(params, base_state)
                safety_rows.append(
                    {
                        "scene": SCENE,
                        "nominal_step": nominal_step,
                        "train_view_id": train_view,
                        "model_restore_max_abs_delta": restore_delta,
                        "PERSISTENT_MODEL_SAFETY": bool(restore_delta == 0.0),
                        "semantics": "throwaway in-memory virtual parameter copies restored to checkpoint-loaded state; no optimizer/scheduler/checkpoint writes",
                    }
                )
                del grads_r, grads_rd, delta_r, delta_rd, delta_depth, theta_r_render, theta_rd_render, group_renders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            checkpoint_safety_after.append(_file_state(ckpt_path))
        finally:
            _release(loaded)

    if equivalence is None:
        equivalence = {"pass": False, "reason": "no intervention run"}
    _write_csv(output_dir / "pre_backward_equivalence.csv", pre_rows)
    _write_json(output_dir / "pre_backward_equivalence.json", {"rows": pre_rows})
    _write_csv(output_dir / "raw_gradient_increment_metrics.csv", raw_grad_rows)
    _write_json(output_dir / "raw_gradient_increment_metrics.json", {"rows": raw_grad_rows})
    _write_csv(output_dir / "optimizer_update_increment_metrics.csv", update_rows)
    _write_json(output_dir / "optimizer_update_increment_metrics.json", {"rows": update_rows, "VIRTUAL_UPDATE_EQUIVALENCE": equivalence})
    _write_csv(output_dir / "physical_update_metrics.csv", physical_rows)
    _write_json(output_dir / "physical_update_metrics.json", {"rows": physical_rows})
    _write_csv(output_dir / "adam_momentum_alignment.csv", momentum_rows)
    _write_json(output_dir / "adam_momentum_alignment.json", {"rows": momentum_rows})
    _write_csv(output_dir / "virtual_full_response.csv", virtual_full_rows)
    _write_json(output_dir / "virtual_full_response.json", {"rows": virtual_full_rows})
    _write_csv(output_dir / "virtual_full_region_metrics.csv", virtual_full_rows)
    _write_json(output_dir / "virtual_full_region_metrics.json", {"rows": virtual_full_rows})
    _write_csv(output_dir / "virtual_group_response.csv", virtual_group_rows)
    _write_json(output_dir / "virtual_group_response.json", {"rows": virtual_group_rows})
    _write_csv(output_dir / "virtual_group_region_metrics.csv", virtual_group_rows)
    _write_json(output_dir / "virtual_group_region_metrics.json", {"rows": virtual_group_rows})
    _write_csv(output_dir / "group_additivity_metrics.csv", additivity_rows)
    _write_json(output_dir / "group_additivity_metrics.json", {"rows": additivity_rows})
    _write_csv(output_dir / "boundary_control.csv", boundary_rows)
    _write_json(output_dir / "boundary_control.json", {"rows": boundary_rows})
    _write_csv(output_dir / "tau_transmission_control.csv", tau_rows)
    _write_json(output_dir / "tau_transmission_control.json", {"rows": tau_rows})
    _write_csv(output_dir / "persistent_model_safety.csv", safety_rows)
    _write_json(
        output_dir / "persistent_model_safety.json",
        {
            "rows": safety_rows,
            "checkpoint_before": checkpoint_safety_before,
            "checkpoint_after": checkpoint_safety_after,
            "checkpoint_files_unchanged": checkpoint_safety_before == checkpoint_safety_after,
            "PERSISTENT_MODEL_SAFETY": bool(all(row["PERSISTENT_MODEL_SAFETY"] for row in safety_rows) and checkpoint_safety_before == checkpoint_safety_after),
        },
    )
    return {
        "pre_rows": pre_rows,
        "raw_grad_rows": raw_grad_rows,
        "update_rows": update_rows,
        "physical_rows": physical_rows,
        "momentum_rows": momentum_rows,
        "virtual_full_rows": virtual_full_rows,
        "virtual_group_rows": virtual_group_rows,
        "additivity_rows": additivity_rows,
        "boundary_rows": boundary_rows,
        "tau_rows": tau_rows,
        "safety_rows": safety_rows,
        "equivalence": equivalence,
        "checkpoint_safety_before": checkpoint_safety_before,
        "checkpoint_safety_after": checkpoint_safety_after,
    }


def _pooled(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any], value_key: str) -> float:
    vals = []
    for row in rows:
        ok = True
        for key, value in filters.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            try:
                vals.append(float(row[value_key]))
            except Exception:
                pass
    return _mean(vals)


def _robustness_tables(
    virtual_full_rows: Sequence[Mapping[str, Any]],
    virtual_group_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for step in INTERVENTION_STEPS:
        for branch_rows, branch_name in ((virtual_full_rows, "FULL_RD_MINUS_R"),):
            camera_positive = 0
            for train_view in TRAIN_CAMERA_BANK:
                gain = _pooled(branch_rows, {"nominal_step": step, "train_view_id": train_view, "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
                positive = bool(math.isfinite(gain) and gain > 0)
                camera_positive += int(positive)
                train_rows.append(
                    {
                        "scene": SCENE,
                        "nominal_step": step,
                        "branch": branch_name,
                        "train_view_id": train_view,
                        "HJ_GAIN_virtual_rgb_gain_mean_over_eval": gain,
                        "HJ_positive": positive,
                    }
                )
            rate = camera_positive / max(1, len(TRAIN_CAMERA_BANK))
            train_rows.append(
                {
                    "scene": SCENE,
                    "nominal_step": step,
                    "branch": branch_name,
                    "train_view_id": "POOLED",
                    "HJ_positive_count": camera_positive,
                    "num_train_cameras": len(TRAIN_CAMERA_BANK),
                    "TRAIN_CAMERA_HJ_POSITIVE_RATE": rate,
                    "TRAIN_CAMERA_ROBUST": bool(rate >= 0.60),
                }
            )
            for eval_view in EVAL_VIEWS:
                gain = _pooled(branch_rows, {"nominal_step": step, "eval_view_id": eval_view, "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
                harm = _pooled(branch_rows, {"nominal_step": step, "eval_view_id": eval_view, "region": "HJ_HARM"}, "VIRTUAL_RGB_GAIN")
                aligned = bool(math.isfinite(gain) and math.isfinite(harm) and gain > 0 and gain > harm)
                eval_rows.append(
                    {
                        "scene": SCENE,
                        "nominal_step": step,
                        "branch": branch_name,
                        "eval_view_id": eval_view,
                        "HJ_GAIN_virtual_rgb_gain_mean_over_train": gain,
                        "HJ_HARM_virtual_rgb_gain_mean_over_train": harm,
                        "ALIGNED": aligned,
                    }
                )
    for step in INTERVENTION_STEPS:
        for group in GEOMETRY_GROUPS:
            rows = [row for row in virtual_group_rows if row.get("nominal_step") == step and row.get("branch") == f"GROUP_{group}" and row.get("region") == "HJ_GAIN"]
            camera_positive = 0
            for train_view in TRAIN_CAMERA_BANK:
                gain = _pooled(rows, {"train_view_id": train_view}, "VIRTUAL_RGB_GAIN")
                camera_positive += int(math.isfinite(gain) and gain > 0)
            train_rows.append(
                {
                    "scene": SCENE,
                    "nominal_step": step,
                    "branch": f"GROUP_{group}",
                    "train_view_id": "POOLED",
                    "HJ_positive_count": camera_positive,
                    "num_train_cameras": len(TRAIN_CAMERA_BANK),
                    "GROUP_HJ_POSITIVE_RATE": camera_positive / max(1, len(TRAIN_CAMERA_BANK)),
                }
            )
    summary: Dict[str, Any] = {}
    for step in (1000, 3000):
        aligned = [row for row in eval_rows if row["nominal_step"] == step and row["branch"] == "FULL_RD_MINUS_R" and row["ALIGNED"]]
        pooled_gain = _pooled(virtual_full_rows, {"nominal_step": step, "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
        pooled_harm = _pooled(virtual_full_rows, {"nominal_step": step, "region": "HJ_HARM"}, "VIRTUAL_RGB_GAIN")
        summary[f"future_hj_alignment_{step}"] = {
            "aligned_eval_views": len(aligned),
            "pooled_HJ_GAIN": pooled_gain,
            "pooled_HJ_HARM": pooled_harm,
            "pooled_aligned": bool(math.isfinite(pooled_gain) and math.isfinite(pooled_harm) and pooled_gain > 0 and pooled_gain > pooled_harm),
        }
    summary["FUTURE_HJ_GAIN_ALIGNMENT"] = any(
        item["aligned_eval_views"] >= 2 and item["pooled_aligned"] for item in summary.values() if isinstance(item, Mapping)
    )
    _write_csv(output_dir / "train_camera_robustness.csv", train_rows)
    _write_json(output_dir / "train_camera_robustness.json", {"rows": train_rows})
    _write_csv(output_dir / "eval_view_robustness.csv", eval_rows)
    _write_json(output_dir / "eval_view_robustness.json", {"rows": eval_rows, "summary": summary})
    return train_rows, eval_rows, summary


def _classification(
    state_summary: Mapping[str, Any],
    historical_rows: Sequence[Mapping[str, Any]],
    virtual_full_rows: Sequence[Mapping[str, Any]],
    virtual_group_rows: Sequence[Mapping[str, Any]],
    additivity_rows: Sequence[Mapping[str, Any]],
    train_robustness: Sequence[Mapping[str, Any]],
    eval_summary: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> Dict[str, Any]:
    optimizer_valid = bool(state_summary.get("OPTIMIZER_AWARE_VIRTUAL_STEP_VALID")) and bool(equivalence.get("pass"))
    pre_rows = [row for row in virtual_full_rows if row.get("nominal_step") in (1000, 3000) and row.get("region") == "HJ_GAIN"]
    ratios = [float(row.get("VIRTUAL_DIRECT_MEDIUM_RATIO", float("nan"))) for row in pre_rows]
    ratios = [v for v in ratios if math.isfinite(v)]
    direct_ratio = _mean(ratios)
    direct_dominant = bool(math.isfinite(direct_ratio) and direct_ratio >= 3.0)
    robust = any(
        row.get("branch") == "FULL_RD_MINUS_R"
        and row.get("train_view_id") == "POOLED"
        and bool(row.get("TRAIN_CAMERA_ROBUST"))
        and row.get("nominal_step") in (1000, 3000)
        for row in train_robustness
    )
    historical_pre = [
        row
        for row in historical_rows
        if row.get("nominal_step") in (1000, 3000, 5000)
        and row.get("region") == "HJ_GAIN"
        and math.isfinite(float(row.get("DIRECT_MEDIUM_RESPONSE_RATIO", float("nan"))))
    ]
    historical_compatible = bool(historical_pre and _mean(row["DIRECT_MEDIUM_RESPONSE_RATIO"] for row in historical_pre) >= 3.0)
    future_alignment = bool(eval_summary.get("FUTURE_HJ_GAIN_ALIGNMENT"))

    full_gain = _pooled(virtual_full_rows, {"region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
    if optimizer_valid and future_alignment and direct_dominant and robust and historical_compatible:
        classification = "CONTINUOUS_OBJECT_PATH_SUPPORTED"
    elif optimizer_valid and future_alignment:
        classification = "CONTINUOUS_OBJECT_PATH_PARTIAL"
    elif optimizer_valid and math.isfinite(full_gain):
        classification = "LOCAL_RESPONSE_WITHOUT_FUTURE_ALIGNMENT"
    elif historical_compatible:
        classification = "LONG_HORIZON_CONTINUOUS_EFFECT"
    elif optimizer_valid:
        classification = "CONTINUOUS_OBJECT_PATH_NOT_SUPPORTED"
    else:
        classification = "NOT_EVALUABLE"

    add_rows = [row for row in additivity_rows if row.get("region") == "HJ_GAIN" and row.get("nominal_step") in (1000, 3000)]
    nonadd = _mean(row["GROUP_RESPONSE_NONADDITIVITY_I"] for row in add_rows)
    additive = bool(math.isfinite(nonadd) and nonadd <= 0.30)
    group_scores = []
    full = _pooled(virtual_full_rows, {"region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
    for group in GEOMETRY_GROUPS:
        gain = _pooled(virtual_group_rows, {"branch": f"GROUP_{group}", "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN")
        frac = gain / (full + EPS) if math.isfinite(gain) and math.isfinite(full) else float("nan")
        group_scores.append({"group": group, "gain": gain, "fraction": frac})
    dominant_group = "NOT_EVALUABLE"
    if future_alignment and additive and math.isfinite(full) and abs(full) > EPS:
        sorted_scores = sorted(group_scores, key=lambda row: row["fraction"] if math.isfinite(row["fraction"]) else -1e9, reverse=True)
        best = sorted_scores[0]
        if best["fraction"] >= 0.50:
            dominant_group = {
                "means": "MEANS_DOMINANT",
                "scales": "SCALES_DOMINANT",
                "quats": "QUATS_DOMINANT",
                "opacities": "OPACITY_DOMINANT",
            }[best["group"]]
        elif all(math.isfinite(row["fraction"]) for row in group_scores):
            sq = sum(row["fraction"] for row in group_scores if row["group"] in ("scales", "quats"))
            geom = sum(row["fraction"] for row in group_scores if row["group"] in ("means", "scales", "quats"))
            if sq >= 0.50:
                dominant_group = "SCALE_QUAT_MIXED"
            elif geom >= 0.50:
                dominant_group = "GEOMETRY_SHAPE_MIXED"
            else:
                dominant_group = "MIXED_CONTINUOUS_UPDATE"
        else:
            dominant_group = "NO_CLEAR_GROUP"
    elif not additive:
        dominant_group = "MIXED_CONTINUOUS_UPDATE" if future_alignment else "NOT_EVALUABLE"

    if classification == "CONTINUOUS_OBJECT_PATH_SUPPORTED" and dominant_group not in ("MIXED_CONTINUOUS_UPDATE", "NOT_EVALUABLE", "NO_CLEAR_GROUP"):
        next_experiment = "BND-CDEPTH-GROUP group-restricted depth-gradient causal test"
    elif classification == "CONTINUOUS_OBJECT_PATH_SUPPORTED":
        next_experiment = "CDEPTH-RGBTRIG continuous-path isolation training"
    elif classification == "LONG_HORIZON_CONTINUOUS_EFFECT":
        next_experiment = "CDEPTH-OPT-MEM optimizer momentum / accumulated-update audit"
    elif classification == "CONTINUOUS_OBJECT_PATH_PARTIAL":
        next_experiment = "CDEPTH-RGBTRIG continuous-path isolation training"
    else:
        next_experiment = "read-only optimization-basin diagnostic"

    return {
        "OPTIMIZER_AWARE_VIRTUAL_STEP_VALID": optimizer_valid,
        "FUTURE_HJ_GAIN_ALIGNMENT": future_alignment,
        "VIRTUAL_DIRECT_MEDIUM_RATIO_pre_recovery_HJ_GAIN_mean": direct_ratio,
        "DIRECT_RESPONSE_DOMINANT": direct_dominant,
        "TRAIN_CAMERA_ROBUST": robust,
        "historical_direct_trajectory_compatible": historical_compatible,
        "GROUP_RESPONSE_NONADDITIVITY_HJ_GAIN_pre_recovery_mean": nonadd,
        "GROUP_ATTRIBUTION_APPROX_ADDITIVE": additive,
        "group_scores": group_scores,
        "DOMINANT_GROUP": dominant_group,
        "CONTINUOUS_PATH_CLASSIFICATION": classification,
        "NEXT_SINGLE_FACTOR_RECOMMENDATION": next_experiment,
        "classification_definitions": [
            "CONTINUOUS_OBJECT_PATH_SUPPORTED",
            "CONTINUOUS_OBJECT_PATH_PARTIAL",
            "LOCAL_RESPONSE_WITHOUT_FUTURE_ALIGNMENT",
            "LONG_HORIZON_CONTINUOUS_EFFECT",
            "CONTINUOUS_OBJECT_PATH_NOT_SUPPORTED",
            "NOT_EVALUABLE",
        ],
    }


def _write_visuals(
    render_dir: Path,
    historical_direct_rows: Sequence[Mapping[str, Any]],
    raw_grad_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    momentum_rows: Sequence[Mapping[str, Any]],
    virtual_full_rows: Sequence[Mapping[str, Any]],
    virtual_group_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Path]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    def add(path: Path, output_type: str, description: str) -> None:
        manifest.append(
            {
                "scene": SCENE,
                "file_path": str(path),
                "output_type": output_type,
                "description": description,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    hist_plot_rows = [
        row
        for row in historical_direct_rows
        if row.get("region") == "HJ_GAIN"
    ]
    for row in hist_plot_rows:
        row["label"] = f"{row['nominal_step']}_{row['region']}"
    path = render_dir / "historical_direct_trajectory.png"
    _write_plot(path, "Historical HJ_GAIN Direct/Medium Trajectory", hist_plot_rows, "label", ("mean_abs_delta_D_DIRECT", "mean_abs_delta_B_MEDIUM", "RGB_MSE_GAIN_mean"))
    add(path, "historical_direct_trajectory", "K1 vs CDEPTH HJ_GAIN |Delta D|, |Delta B|, and RGB MSE gain by step.")

    def pooled_group(rows: Sequence[Mapping[str, Any]], key: str, region_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for step in INTERVENTION_STEPS:
            for group in GEOMETRY_GROUPS:
                subset = [r for r in rows if r.get("nominal_step") == step and r.get("group") == group]
                if region_filter:
                    subset = [r for r in subset if r.get("region") == region_filter]
                out.append({"label": f"{step}_{group}", key: _mean(float(r.get(key, float("nan"))) for r in subset)})
        return out

    path = render_dir / "raw_gradient_ratios.png"
    _write_bar(path, "Raw Depth-Increment Gradient Ratio", pooled_group(raw_grad_rows, "GRAD_INCREMENT_RATIO"), "label", "GRAD_INCREMENT_RATIO")
    add(path, "raw_gradient_ratios", "Mean ||g_RD-g_R||/||g_R|| for geometry groups.")

    path = render_dir / "optimizer_update_ratios.png"
    _write_bar(path, "Optimizer-Aware Update Increment Ratio", pooled_group(update_rows, "UPDATE_INCREMENT_RATIO"), "label", "UPDATE_INCREMENT_RATIO")
    add(path, "optimizer_update_ratios", "Mean ||Delta_DEPTH||/||Delta_R|| after Adam state.")

    path = render_dir / "adam_amplification_momentum.png"
    amp_rows = pooled_group(update_rows, "OPTIMIZER_AMPLIFICATION")
    _write_bar(path, "Adam Amplification", amp_rows, "label", "OPTIMIZER_AMPLIFICATION")
    add(path, "adam_amplification", "Norm-level optimizer amplification of the raw gradient ratio.")

    full_rows = []
    for step in INTERVENTION_STEPS:
        for region in ("HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM"):
            full_rows.append(
                {
                    "label": f"{step}_{region}",
                    "VIRTUAL_RGB_GAIN": _pooled(virtual_full_rows, {"nominal_step": step, "region": region}, "VIRTUAL_RGB_GAIN"),
                    "mean_abs_D_RESPONSE": _pooled(virtual_full_rows, {"nominal_step": step, "region": region}, "mean_abs_D_RESPONSE"),
                    "mean_abs_B_RESPONSE": _pooled(virtual_full_rows, {"nominal_step": step, "region": region}, "mean_abs_B_RESPONSE"),
                }
            )
    path = render_dir / "full_virtual_rgb_gain.png"
    _write_bar(path, "Full Virtual RGB Gain", full_rows, "label", "VIRTUAL_RGB_GAIN")
    add(path, "full_virtual_rgb_gain", "RGB gain from theta_RD versus theta_R by future diagnostic region.")

    path = render_dir / "full_direct_medium_response.png"
    _write_plot(path, "Full Virtual Direct/Medium Response", full_rows, "label", ("mean_abs_D_RESPONSE", "mean_abs_B_RESPONSE"))
    add(path, "full_direct_medium_response", "Full virtual response split into renderer direct_object_signal and rgb_medium magnitudes.")

    group_rows = []
    for step in INTERVENTION_STEPS:
        for group in GEOMETRY_GROUPS:
            group_rows.append(
                {
                    "label": f"{step}_{group}",
                    "HJ_GAIN_RGB_GAIN": _pooled(virtual_group_rows, {"nominal_step": step, "branch": f"GROUP_{group}", "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN"),
                    "HJ_HARM_RGB_GAIN": _pooled(virtual_group_rows, {"nominal_step": step, "branch": f"GROUP_{group}", "region": "HJ_HARM"}, "VIRTUAL_RGB_GAIN"),
                }
            )
    path = render_dir / "group_isolated_hj_response.png"
    _write_plot(path, "Group-Isolated HJ Response", group_rows, "label", ("HJ_GAIN_RGB_GAIN", "HJ_HARM_RGB_GAIN"))
    add(path, "group_isolated_hj_response", "Group-isolated depth increment response on HJ_GAIN and HJ_HARM.")

    strong_rows = []
    for step in INTERVENTION_STEPS:
        for group in GEOMETRY_GROUPS:
            strong_rows.append(
                {
                    "label": f"{step}_{group}",
                    "STRONG_GAIN": _pooled(virtual_group_rows, {"nominal_step": step, "branch": f"GROUP_{group}", "region": "HJ_STRONG_GAIN"}, "VIRTUAL_RGB_GAIN"),
                    "STRONG_HARM": _pooled(virtual_group_rows, {"nominal_step": step, "branch": f"GROUP_{group}", "region": "HJ_STRONG_HARM"}, "VIRTUAL_RGB_GAIN"),
                }
            )
    path = render_dir / "strong_hj_subset_response.png"
    _write_plot(path, "Strong HJ Subset Response", strong_rows, "label", ("STRONG_GAIN", "STRONG_HARM"))
    add(path, "strong_hj_subset_response", "Strong HJ_GAIN/HJ_HARM subset group response.")

    pooled_train = [row for row in train_rows if row.get("train_view_id") == "POOLED"]
    for row in pooled_train:
        row["label"] = f"{row.get('nominal_step')}_{row.get('branch')}"
        if "TRAIN_CAMERA_HJ_POSITIVE_RATE" not in row and "GROUP_HJ_POSITIVE_RATE" in row:
            row["TRAIN_CAMERA_HJ_POSITIVE_RATE"] = row["GROUP_HJ_POSITIVE_RATE"]
    path = render_dir / "camera_robustness.png"
    _write_bar(path, "Training-Camera HJ Positive Rate", pooled_train, "label", "TRAIN_CAMERA_HJ_POSITIVE_RATE")
    add(path, "camera_robustness", "Fraction of training camera branches with positive HJ_GAIN virtual RGB gain.")

    for row in eval_rows:
        row["label"] = f"{row.get('nominal_step')}_{row.get('eval_view_id')}"
    path = render_dir / "eval_view_robustness.png"
    _write_plot(path, "Eval-View Future HJ Alignment", eval_rows, "label", ("HJ_GAIN_virtual_rgb_gain_mean_over_train", "HJ_HARM_virtual_rgb_gain_mean_over_train"))
    add(path, "eval_view_robustness", "Eval-view pooled HJ_GAIN versus HJ_HARM virtual gains.")

    temporal_rows = []
    for step in INTERVENTION_STEPS:
        temporal_rows.append(
            {
                "label": str(step),
                "UPDATE_scales": _pooled(update_rows, {"nominal_step": step, "group": "scales"}, "UPDATE_INCREMENT_RATIO"),
                "UPDATE_quats": _pooled(update_rows, {"nominal_step": step, "group": "quats"}, "UPDATE_INCREMENT_RATIO"),
                "FULL_HJ_GAIN": _pooled(virtual_full_rows, {"nominal_step": step, "region": "HJ_GAIN"}, "VIRTUAL_RGB_GAIN"),
            }
        )
    path = render_dir / "temporal_causal_chain_summary.png"
    _write_plot(path, "Temporal Summary", temporal_rows, "label", ("UPDATE_scales", "UPDATE_quats", "FULL_HJ_GAIN"))
    add(path, "temporal_causal_chain_summary", "Association-compatible chronology of optimizer update ratios and HJ_GAIN virtual response.")

    factor_rows = [
        {"label": "direct_ratio", "value": classification.get("VIRTUAL_DIRECT_MEDIUM_RATIO_pre_recovery_HJ_GAIN_mean", float("nan"))},
        {"label": "nonadditivity", "value": classification.get("GROUP_RESPONSE_NONADDITIVITY_HJ_GAIN_pre_recovery_mean", float("nan"))},
        {"label": "future_alignment", "value": 1.0 if classification.get("FUTURE_HJ_GAIN_ALIGNMENT") else 0.0},
        {"label": "camera_robust", "value": 1.0 if classification.get("TRAIN_CAMERA_ROBUST") else 0.0},
    ]
    path = render_dir / "final_factor_summary.png"
    _write_bar(path, "Final Factor Summary", factor_rows, "label", "value")
    add(path, "final_factor_summary", "Compact numeric summary for the continuous path classification gates.")

    _write_json(render_dir / "manifest.json", {"rows": manifest})
    index_path = render_dir / "VISUAL_COMPARE_INDEX.md"
    lines = ["# BND-CDEPTH Direct Path Visual Index", ""]
    for row in manifest:
        lines.append(f"- {row['output_type']}: `{row['file_path']}`")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return manifest, index_path


def _write_manifest(output_dir: Path, render_dir: Path, visual_manifest: Sequence[Mapping[str, Any]], visual_index: Path) -> None:
    rows = []
    for root in (output_dir, render_dir):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "file_path": str(path),
                        "kind": "render" if str(path).startswith(str(render_dir)) else "output",
                        "size_bytes": path.stat().st_size,
                    }
                )
    _write_json(output_dir / "manifest.json", {"rows": rows, "visual_manifest": list(visual_manifest), "visual_index": str(visual_index)})
    _write_csv(output_dir / "manifest.csv", rows)


def _write_research_note(repo: Path, outputs: Mapping[str, Any], classification: Mapping[str, Any]) -> None:
    lines = [
        "# BND-CDEPTH Continuous Direct-Object Optimization Path Audit",
        "",
        "## CODE FACT",
        "",
        f"- Repo branch at audit start: `{outputs['repo_manifest']['branch']}`.",
        f"- Start HEAD: `{outputs['repo_manifest']['head']}`.",
        "- This audit is read-only with respect to training artifacts: no persistent optimizer step, scheduler step, densification, pruning, opacity reset, checkpoint save, or renderer physics change was used.",
        "- Current trainable groups from `WaterSplattingModel.get_param_groups()` are `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities`, `medium_mlp`, and `direction_encoding`.",
        "- `camera_opt` is configured in the method config but is not returned by the current model param-group implementation.",
        "- Nerfstudio training order is zero-grad, forward/loss, backward, optimizer step, then scheduler step.",
        "- Analytic Adam virtual updates use checkpoint current LR and saved `step`, `exp_avg`, and `exp_avg_sq`; medium groups apply the configured `max_norm=0.001` clipping.",
        "",
        "## CONFIG FACT",
        "",
        "- Formal Panama train camera bank: `"
        + "`, `".join(TRAIN_CAMERA_BANK)
        + "`.",
        "- Formal eval views: `"
        + "`, `".join(EVAL_VIEWS)
        + "`.",
        "- Outcome masks are post-hoc diagnostics: `M1_HIGH_J` is final M1 accumulation > 0.01 and final M1 clear-object full-SH raw max RGB > 1.0; `HJ_GAIN/HJ_HARM` are defined from final K1 vs CDEPTH RGB MSE change inside `M1_HIGH_J`.",
        "",
        "## EXPERIMENTAL FACT",
        "",
        f"- Optimizer-aware virtual step valid: `{classification.get('OPTIMIZER_AWARE_VIRTUAL_STEP_VALID')}`.",
        f"- Virtual update equivalence: `{outputs['intervention']['equivalence']}`.",
        f"- Future HJ_GAIN alignment: `{classification.get('FUTURE_HJ_GAIN_ALIGNMENT')}`.",
        f"- Direct response dominant: `{classification.get('DIRECT_RESPONSE_DOMINANT')}` with pre-recovery HJ_GAIN ratio `{classification.get('VIRTUAL_DIRECT_MEDIUM_RATIO_pre_recovery_HJ_GAIN_mean')}`.",
        f"- Training camera robust: `{classification.get('TRAIN_CAMERA_ROBUST')}`.",
        f"- Group additivity approximate: `{classification.get('GROUP_ATTRIBUTION_APPROX_ADDITIVE')}` with mean nonadditivity `{classification.get('GROUP_RESPONSE_NONADDITIVITY_HJ_GAIN_pre_recovery_mean')}`.",
        "",
        "## QUANTITATIVE RESULT",
        "",
        f"- Continuous path classification: `{classification.get('CONTINUOUS_PATH_CLASSIFICATION')}`.",
        f"- Dominant group classification: `{classification.get('DOMINANT_GROUP')}`.",
        f"- Group scores: `{classification.get('group_scores')}`.",
        "",
        "## INFERENCE",
        "",
        "- The local optimizer-aware measurements are evidence about one-step response from fixed K1 checkpoints. They do not by themselves prove the final 15k RGB gain causal mechanism.",
        "- Renderer branch responses use true `direct_object_signal` and `rgb_medium` outputs; no direct-object GT is assumed.",
        "",
        "## HYPOTHESIS",
        "",
        "- If the local response and historical trajectory are compatible, the next causal test should isolate the continuous path in training rather than sweep depth weights.",
        f"- Next single-factor recommendation: `{classification.get('NEXT_SINGLE_FACTOR_RECOMMENDATION')}`.",
        "",
        "## Artifacts",
        "",
        f"- Output manifest: `{outputs['output_manifest']}`.",
        f"- Visual manifest: `{outputs['visual_manifest']}`.",
        f"- Visual index: `{outputs['visual_index']}`.",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf8")


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
        "diff_check_at_start": subprocess.run(["git", "-C", str(repo), "diff", "--check"], text=True, capture_output=True).returncode,
        "historical_untracked_files_protected": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    source_audit = _source_audit(repo)
    _write_json(output_dir / "optimizer_source_audit.json", source_audit)
    _write_csv(output_dir / "optimizer_source_audit.csv", source_audit["group_rows"])
    (output_dir / "optimizer_source_audit.md").write_text(
        "# Optimizer Source Audit\n\n"
        + "\n".join(
            f"- `{row['group']}`: {row['optimizer']}, lr@1k={row['current_lr_1k']}, scheduler={row['scheduler']}, max_norm={row['max_norm']}"
            for row in source_audit["group_rows"]
        )
        + "\n",
        encoding="utf8",
    )

    _checkpoint_manifest(repo, output_dir)
    _state_rows, state_summary = _optimizer_state_availability(repo, output_dir)
    regions, region_meta = _build_outcome_regions(repo, output_dir)
    historical_rows, historical_direct_rows = _historical_trajectory(repo, regions, output_dir)
    intervention = _run_interventions(repo, regions, output_dir)
    train_rows, eval_rows, eval_summary = _robustness_tables(intervention["virtual_full_rows"], intervention["virtual_group_rows"], output_dir)
    classification = _classification(
        state_summary,
        historical_rows,
        intervention["virtual_full_rows"],
        intervention["virtual_group_rows"],
        intervention["additivity_rows"],
        train_rows,
        eval_summary,
        intervention["equivalence"],
    )
    _write_json(output_dir / "continuous_path_classification.json", classification)

    summary_rows = [
        {
            "scene": SCENE,
            "classification": classification["CONTINUOUS_PATH_CLASSIFICATION"],
            "dominant_group": classification["DOMINANT_GROUP"],
            "optimizer_valid": classification["OPTIMIZER_AWARE_VIRTUAL_STEP_VALID"],
            "future_hj_alignment": classification["FUTURE_HJ_GAIN_ALIGNMENT"],
            "direct_response_dominant": classification["DIRECT_RESPONSE_DOMINANT"],
            "train_camera_robust": classification["TRAIN_CAMERA_ROBUST"],
            "next_single_factor_recommendation": classification["NEXT_SINGLE_FACTOR_RECOMMENDATION"],
        }
    ]
    _write_csv(output_dir / "direct_path_final_summary.csv", summary_rows)
    _write_json(output_dir / "direct_path_final_summary.json", {"rows": summary_rows, "classification": classification})

    visual_manifest, visual_index = _write_visuals(
        render_dir,
        historical_direct_rows,
        intervention["raw_grad_rows"],
        intervention["update_rows"],
        intervention["momentum_rows"],
        intervention["virtual_full_rows"],
        intervention["virtual_group_rows"],
        train_rows,
        eval_rows,
        classification,
    )
    _write_manifest(output_dir, render_dir, visual_manifest, visual_index)

    outputs = {
        "repo_manifest": repo_manifest,
        "region_meta": region_meta,
        "intervention": intervention,
        "output_manifest": str(output_dir / "manifest.json"),
        "visual_manifest": str(render_dir / "manifest.json"),
        "visual_index": str(visual_index),
    }
    _write_research_note(repo, outputs, classification)
    print(json.dumps({"classification": classification, "output_dir": str(output_dir), "render_dir": str(render_dir)}, indent=2, default=_json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
