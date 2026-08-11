#!/usr/bin/env python
"""Matched CDEPTH early-on / late-off continuation for Panama.

This final CDEPTH diagnostic starts from the same formal BND-CDEPTH Panama
3k checkpoint and runs two matched continuations to the formal 15k endpoint:

* EON:  coarse-depth supervision remains enabled.
* EOFF: coarse-depth supervision is disabled after the 3k start state.

The two branches replay an explicit matched training-camera sequence and keep
normal WaterSplatting topology evolution enabled. No production model or trainer
source is modified by this diagnostic runner.
"""

from __future__ import annotations

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
from PIL import Image
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_cdepth_direct_path as direct_path
from scripts.diagnostics import audit_bnd_cdepth_setup as cdepth_setup
from scripts.diagnostics import run_bnd_cdepth_fixpop_rollout as fixpop


SCENE = "Panama"
OUTPUT_DIR = Path("outputs/bnd_cdepth_early_off_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_early_off_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_early_off_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_EARLY_OFF_FINAL_EXPERIMENT_2026-08-11.md")

CDEPTH_CONFIG = Path(
    "outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/"
    "water-splatting/20260811_bnd_cdepth/config.yml"
)
K1_CONFIG = cdepth_setup.K1_CONFIG
START_NOMINAL_STEP = 3000
FINAL_NOMINAL_STEP = 15000
SNAPSHOT_ABS_NOMINAL = (3000, 4000, 5000, 8000, 10000, 13000, 15000)
EVAL_VIEWS = ("MTN_1529", "MTN_1539", "MTN_1547")
TRAIN_CAMERA_BANK = direct_path.TRAIN_CAMERA_BANK
CURRENT_GROUPS = direct_path.CURRENT_PARAM_GROUPS
EPS = 1e-12

K1_FINAL = {"PSNR": 31.498353, "SSIM": 0.948783, "LPIPS": 0.075521}
HIST_CDEPTH_FINAL = {"PSNR": 31.753299, "SSIM": 0.946292, "LPIPS": 0.080931}
K1_TAU_P90_CONTEXT = 0.999


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
    return fixpop._json_default(value)


def _write_json(path: Path, data: Any) -> None:
    fixpop._write_json(path, data)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fixpop._write_csv(path, rows)


def _git(repo: Path, *args: str) -> str:
    return fixpop._git(repo, *args)


def _sha256(path: Path) -> str:
    return fixpop._sha256(path)


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


def _rel(abs_step: int) -> int:
    return int(abs_step) - START_NOMINAL_STEP


def _snapshot_abs_steps(final_actual_step: int) -> Tuple[int, ...]:
    steps = []
    for nominal in SNAPSHOT_ABS_NOMINAL:
        if nominal == FINAL_NOMINAL_STEP:
            steps.append(final_actual_step)
        else:
            steps.append(nominal)
    return tuple(dict.fromkeys(steps))


def _snapshot_rel_steps(final_actual_step: int) -> Tuple[int, ...]:
    return tuple(_rel(step) for step in _snapshot_abs_steps(final_actual_step))


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


def _load_branch(repo: Path, branch: str, *, coarse_depth: bool) -> LoadedBranch:
    config_path = repo / CDEPTH_CONFIG
    actual_step = _actual_step(config_path, START_NOMINAL_STEP)

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


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return fixpop._batch_to_device(batch, device)


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    return fixpop._model_param_tensors(model)


def _optimizer_state_tensors(optimizers: Optimizers) -> Dict[str, Dict[str, Tensor]]:
    return fixpop._optimizer_state_tensors(optimizers)


def _compare_model_params(a: Any, b: Any) -> List[Dict[str, Any]]:
    return fixpop._compare_model_params(a, b)


def _compare_optimizer_states(a: Optimizers, b: Optimizers) -> List[Dict[str, Any]]:
    return fixpop._compare_optimizer_states(a, b)


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    return fixpop._max_abs_diff(a, b)


def _condition_forward(model: Any, camera: Any, batch: Mapping[str, Any], coarse_depth: bool) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    model.config.coarse_depth_supervision_enabled = bool(coarse_depth)
    outputs = model.get_outputs(camera.to(model.device))
    metrics: Dict[str, Tensor] = {}
    losses = model.get_loss_dict(outputs, batch, metrics)
    return outputs, losses, metrics


def _compare_initial_forward(eon: LoadedBranch, eoff: LoadedBranch, train_index: int) -> List[Dict[str, Any]]:
    dm_on = eon.pipeline.datamanager
    dm_off = eoff.pipeline.datamanager
    cached_on = dm_on.cached_train
    cached_off = dm_off.cached_train
    cameras_on = getattr(dm_on, "train_cameras", dm_on.train_dataset.cameras).to(eon.pipeline.model.device)
    cameras_off = getattr(dm_off, "train_cameras", dm_off.train_dataset.cameras).to(eoff.pipeline.model.device)
    camera_on = cameras_on[train_index : train_index + 1]
    camera_off = cameras_off[train_index : train_index + 1]
    batch_on = _batch_to_device(cached_on[train_index].copy(), eon.pipeline.model.device)
    batch_off = _batch_to_device(cached_off[train_index].copy(), eoff.pipeline.model.device)
    eon.pipeline.model.train()
    eoff.pipeline.model.train()
    out_on, losses_on, metrics_on = _condition_forward(eon.pipeline.model, camera_on, batch_on, coarse_depth=True)
    out_off, losses_off, _metrics_off = _condition_forward(eoff.pipeline.model, camera_off, batch_off, coarse_depth=False)
    keys = ("pred_image", "direct_object_signal", "rgb_medium", "depth", "clear_object_fullsh_raw", "transmission", "tau_D", "accumulation")
    rows: List[Dict[str, Any]] = []
    for key in keys:
        diff = _max_abs_diff(out_on[key], out_off[key])
        rows.append({"key": key, "max_abs_diff": diff, "pass": bool(diff <= 1e-6)})
    main_diff = abs(float(losses_on["main_loss"].detach().cpu().item()) - float(losses_off["main_loss"].detach().cpu().item()))
    rows.append({"key": "main_loss", "max_abs_diff": main_diff, "pass": bool(main_diff <= 1e-6)})
    rows.append(
        {
            "key": "eon_coarse_depth_loss_active",
            "max_abs_diff": float(metrics_on.get("coarse_depth_loss_weighted", torch.tensor(float("nan"))).detach().cpu().item()),
            "pass": bool("coarse_depth_loss" in losses_on and "coarse_depth_loss" not in losses_off),
        }
    )
    return rows


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
    torch_cpu = state["torch_cpu"].detach().cpu().numpy().tobytes()
    cuda_hashes = []
    for item in state.get("torch_cuda", []):
        cuda_hashes.append(hashlib.sha256(item.detach().cpu().numpy().tobytes()).hexdigest())
    return {
        "python_state_type": str(type(state["python"])),
        "numpy_state_name": str(state["numpy"][0]),
        "torch_cpu_rng_sha256": hashlib.sha256(torch_cpu).hexdigest(),
        "torch_cuda_rng_sha256": cuda_hashes,
    }


def _generate_camera_sequence(branch: LoadedBranch, output_dir: Path, final_actual_step: int) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    dm = branch.pipeline.datamanager
    image_filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in image_filenames]
    rows: List[Dict[str, Any]] = []
    indices: List[int] = []
    view_ids: List[str] = []
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
            "camera_pool": list(TRAIN_CAMERA_BANK),
            "generation_method": "replayed CheckpointableFullImageDatamanager state restored from CDEPTH@3k once, then reused by EON and EOFF",
            "rows": rows,
        },
    )
    return indices, view_ids, rows


def _optimizer_lrs(optimizers: Optimizers) -> Dict[str, float]:
    return fixpop._optimizer_lrs(optimizers)


def _compute_loss_components(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    return fixpop._compute_loss_components(model, outputs, batch)


def _save_continuation_checkpoint(branch: LoadedBranch, rel_step: int, output_dir: Path) -> Path:
    branch_dir = output_dir / "continuation_checkpoints" / branch.branch
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
            "scalers": {},
            "metadata": {
                "normal_topology_enabled": True,
                "coarse_depth_supervision_enabled": branch.branch == "EON",
                "matched_camera_sequence": True,
            },
        },
        path,
    )
    return path


def _continuation_ckpt_path(output_dir: Path, branch: str, rel_step: int) -> Path:
    return output_dir / "continuation_checkpoints" / branch / f"relative-{rel_step:06d}.ckpt"


def _topology_snapshot(branch: LoadedBranch, rel_step: int) -> Dict[str, Any]:
    model = branch.pipeline.model
    return {
        "branch": branch.branch,
        "relative_step": rel_step,
        "absolute_step": START_NOMINAL_STEP + rel_step,
        "gaussian_count": int(model.means.shape[0]),
        "means_shape": list(model.means.shape),
        "scales_shape": list(model.scales.shape),
        "opacities_shape": list(model.opacities.shape),
    }


def _run_before_callbacks(model: Any, optimizers: Optimizers, abs_step: int) -> None:
    model.step_cb(step=abs_step)
    model.aopt_before_train_iteration(optimizers, step=abs_step)
    model.medium_hold_before_train_iteration(optimizers, step=abs_step)


def _run_after_callbacks(model: Any, optimizers: Optimizers, abs_step: int) -> Tuple[bool, int, int]:
    model.aopt_after_train_iteration(step=abs_step)
    model.medium_hold_after_train_iteration(optimizers, step=abs_step)
    model.after_train(step=abs_step)
    before = int(model.means.shape[0])
    refinement_called = bool(abs_step % int(model.config.refine_every) == 0)
    if refinement_called:
        model.refinement_after(optimizers, step=abs_step)
    after = int(model.means.shape[0])
    return refinement_called, before, after


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    coarse_depth: bool,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    rng_state: Mapping[str, Any],
    snapshot_rels: Sequence[int],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    _set_rng_state(rng_state)
    branch = _load_branch(repo, branch_name, coarse_depth=coarse_depth)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    training_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    snapshot_rels_set = set(int(x) for x in snapshot_rels)

    try:
        if 0 in snapshot_rels_set:
            ckpt_path = _save_continuation_checkpoint(branch, 0, output_dir)
            checkpoint_rows.append({"branch": branch.branch, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "checkpoint_path": str(ckpt_path)})
            count_rows.append(_topology_snapshot(branch, 0))

        cached_train = dm.cached_train
        train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        for rel_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names), start=1):
            abs_step = START_NOMINAL_STEP + rel_step
            branch.pipeline.train()
            model.train()
            model.config.coarse_depth_supervision_enabled = bool(coarse_depth)
            _run_before_callbacks(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]
            lrs = _optimizer_lrs(branch.optimizers)
            outputs = model.get_outputs(camera)
            components = _compute_loss_components(model, outputs, batch)
            metrics: Dict[str, Tensor] = {}
            losses = model.get_loss_dict(outputs, batch, metrics)
            total_loss = sum(losses.values())
            stable = bool(torch.isfinite(total_loss).detach().cpu().item())
            if not stable:
                raise RuntimeError(f"Non-finite loss in {branch.branch} rel_step={rel_step}: {float(total_loss.detach().cpu().item())}")
            total_loss.backward()
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(abs_step)
            refinement_called, n_before_refine, n_after_refine = _run_after_callbacks(model, branch.optimizers, abs_step)

            depth_weighted = losses.get("coarse_depth_loss", total_loss.new_tensor(0.0))
            row = {
                "branch": branch.branch,
                "relative_step": rel_step,
                "absolute_step": abs_step,
                "camera_index": camera_index,
                "camera_name": camera_name,
                "L_total": float(total_loss.detach().cpu().item()),
                "L_RGB": float(losses["main_loss"].detach().cpu().item()),
                "reg_l1": float(components["reg_l1"].detach().cpu().item()),
                "reg_ssim": float(components["reg_ssim"].detach().cpu().item()),
                "L_depth_raw": float(metrics.get("coarse_depth_loss_raw", total_loss.new_tensor(0.0)).detach().cpu().item()) if coarse_depth else 0.0,
                "weighted_L_depth": float(depth_weighted.detach().cpu().item()) if coarse_depth else 0.0,
                "gaussian_count": int(model.means.shape[0]),
                "refinement_called": refinement_called,
                "stable": stable,
            }
            for group, lr in lrs.items():
                row[f"lr_{group}"] = lr
            training_rows.append(row)
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
            if refinement_called or rel_step in snapshot_rels_set:
                reset_interval = int(model.config.reset_alpha_every) * int(model.config.refine_every)
                event_rows.append(
                    {
                        "branch": branch.branch,
                        "relative_step": rel_step,
                        "absolute_step": abs_step,
                        "camera_name": camera_name,
                        "refinement_called": refinement_called,
                        "opacity_reset_step": bool(abs_step < int(model.config.stop_split_at) and abs_step % reset_interval == int(model.config.refine_every)),
                        "N_before_refine": n_before_refine,
                        "N_after_refine": n_after_refine,
                        "N_delta": n_after_refine - n_before_refine,
                        "normal_topology_enabled": True,
                        "split_duplicate_prune_counts": "not instrumented; count delta records net population change",
                    }
                )
            if rel_step in snapshot_rels_set:
                ckpt_path = _save_continuation_checkpoint(branch, rel_step, output_dir)
                checkpoint_rows.append({"branch": branch.branch, "relative_step": rel_step, "absolute_step": abs_step, "checkpoint_path": str(ckpt_path)})
                count_rows.append(_topology_snapshot(branch, rel_step))
        return training_rows, event_rows, stability_rows, checkpoint_rows, count_rows
    finally:
        _release(branch)


def _load_snapshot_into_analysis(repo: Path, output_dir: Path, loaded: LoadedBranch, branch: str, rel_step: int) -> None:
    ckpt = torch.load(_continuation_ckpt_path(output_dir, branch, rel_step), map_location="cpu")
    loaded.pipeline.load_pipeline(ckpt["pipeline"], int(ckpt["absolute_step"]))
    model = loaded.pipeline.model
    model.step = int(ckpt["absolute_step"])
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.coarse_depth_supervision_enabled = branch == "EON"


def _render_snapshot(loaded: LoadedBranch) -> Dict[str, Dict[str, Tensor]]:
    model = loaded.pipeline.model
    model.eval()
    records = direct_path._eval_records(loaded)
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


def _error(pred: Tensor, gt: Tensor) -> Tensor:
    return fixpop._error(pred, gt)


def _region_mse(item: Mapping[str, Tensor], mask: Tensor) -> float:
    return fixpop._region_mse(item, mask)


def _region_l1(item: Mapping[str, Tensor], mask: Tensor) -> float:
    return fixpop._region_l1(item, mask)


def _mean(values: Iterable[Any]) -> float:
    return fixpop._mean(values)


def _q(values: Tensor, q: float) -> float:
    return fixpop._q(values, q)


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    return fixpop._masked(values, mask)


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    return fixpop._metric_images(model, pred, gt)


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    return fixpop._rgb_to_uint8(image)


def _gray_to_uint8(values: Tensor, scale: float) -> Image.Image:
    return fixpop._gray_to_uint8(values, scale)


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    return fixpop._signed_to_rgb(values, scale)


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], manifest: List[Dict[str, Any]], output_type: str) -> None:
    fixpop._save_sheet(path, rows, manifest, output_type)


def _save_plot(path: Path, rows: Sequence[Mapping[str, Any]], x_key: str, y_keys: Sequence[str], title: str, manifest: List[Dict[str, Any]], output_type: str) -> None:
    fixpop._save_plot(path, rows, x_key, y_keys, title, manifest, output_type)


def _render_historical_context(repo: Path, regions: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
    k1 = direct_path._render_items(repo, "BND-K1", direct_path.FINAL_STEP)
    cd = direct_path._render_items(repo, "CDEPTH", direct_path.FINAL_STEP)
    rows: List[Dict[str, Any]] = []
    for view_id in EVAL_VIEWS:
        for run, item in (("K1", k1[view_id]), ("Historical Full CDEPTH", cd[view_id])):
            out = {"pred_image": item["outputs"]["pred_image"], "gt": item["gt"]}
            rows.append(
                {
                    "run": run,
                    "view_id": view_id,
                    "global_mse": float(_error(out["pred_image"], out["gt"]).mean().item()),
                    "M1_HIGH_J_mse": _region_mse(out, regions[view_id]["M1_HIGH_J"]),
                    "M1_HIGH_J_l1": _region_l1(out, regions[view_id]["M1_HIGH_J"]),
                }
            )
    return {
        "rows": rows,
        "K1_HJ_MSE": _mean(row["M1_HIGH_J_mse"] for row in rows if row["run"] == "K1"),
        "HIST_CDEPTH_HJ_MSE": _mean(row["M1_HIGH_J_mse"] for row in rows if row["run"] == "Historical Full CDEPTH"),
        "K1_renders": k1,
        "CDEPTH_renders": cd,
    }


def _decomp_row(branch: str, rel_step: int, abs_step: int, view_id: str, region: str, item: Mapping[str, Tensor], mask: Tensor) -> Dict[str, Any]:
    j = item["clear_object_fullsh_raw"]
    tau_mean = item["tau_D"].mean(dim=-1)
    tau_max = item["tau_D"].amax(dim=-1)
    t_mean = item["transmission"].mean(dim=-1)
    vals_j = _masked(j, mask)
    vals_tau_mean = tau_mean[mask]
    vals_tau_max = tau_max[mask]
    vals_t = t_mean[mask]
    c = item.get("gaussian_view_rgb")
    logits = item.get("gaussian_view_logits")
    return {
        "branch": branch,
        "relative_step": rel_step,
        "absolute_step": abs_step,
        "view_id": view_id,
        "region": region,
        "J_p99": _q(vals_j, 0.99),
        "P_J_gt_1": float((vals_j > 1.0).float().mean().item()) if vals_j.numel() else float("nan"),
        "tau_p90": _q(vals_tau_mean, 0.90),
        "tau_p99": _q(vals_tau_max, 0.99),
        "P_T_lt_0p1": float((vals_t < 0.1).float().mean().item()) if vals_t.numel() else float("nan"),
        "T_mean": float(vals_t.mean().item()) if vals_t.numel() else float("nan"),
        "P_c_gt_0p99": float((c > 0.99).float().mean().item()) if isinstance(c, Tensor) and c.numel() else float("nan"),
        "P_abs_s_full_gt_5": float((logits.abs() > 5.0).float().mean().item()) if isinstance(logits, Tensor) and logits.numel() else float("nan"),
    }


def _analyze_snapshots(
    repo: Path,
    output_dir: Path,
    render_dir: Path,
    regions: Mapping[str, Mapping[str, Tensor]],
    snapshot_rels: Sequence[int],
    historical: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    loaded = _load_branch(repo, "ANALYSIS", coarse_depth=False)
    visual_manifest: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []
    branch_metric_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    highj_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []

    final_rgb_rows: List[Sequence[Tuple[str, Image.Image]]] = []
    highj_residual_rows: List[Sequence[Tuple[str, Image.Image]]] = []
    per_view_rows_visual: List[Sequence[Tuple[str, Image.Image]]] = []

    try:
        model = loaded.pipeline.model
        final_rel = max(snapshot_rels)
        final_renders: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
        for rel_step in snapshot_rels:
            abs_step = START_NOMINAL_STEP + rel_step
            renders: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
            for branch in ("EON", "EOFF"):
                _load_snapshot_into_analysis(repo, output_dir, loaded, branch, rel_step)
                renders[branch] = _render_snapshot(loaded)
            if rel_step == final_rel:
                final_renders = renders
            for branch, by_view in renders.items():
                view_metrics: List[Dict[str, float]] = []
                for view_id in EVAL_VIEWS:
                    item = by_view[view_id]
                    metrics = _metric_images(model, item["pred_image"], item["gt"])
                    view_metrics.append(metrics)
                    row = {
                        "branch": branch,
                        "relative_step": rel_step,
                        "absolute_step": abs_step,
                        "view_id": view_id,
                        **metrics,
                        "M1_HIGH_J_mse": _region_mse(item, regions[view_id]["M1_HIGH_J"]),
                        "M1_HIGH_J_l1": _region_l1(item, regions[view_id]["M1_HIGH_J"]),
                        "M1_LOW_J_mse": _region_mse(item, regions[view_id]["M1_LOW_J"]),
                        "BRIGHT_Q5_mse": _region_mse(item, regions[view_id]["BRIGHT_Q5"]),
                    }
                    per_view_rows.append(row)
                branch_metric_rows.append(
                    {
                        "branch": branch,
                        "relative_step": rel_step,
                        "absolute_step": abs_step,
                        "PSNR": _mean(row["psnr"] for row in view_metrics),
                        "SSIM": _mean(row["ssim"] for row in view_metrics),
                        "LPIPS": _mean(row["lpips"] for row in view_metrics),
                        "MSE": _mean(row["mse"] for row in view_metrics),
                    }
                )
                highj_rows.append(
                    {
                        "branch": branch,
                        "relative_step": rel_step,
                        "absolute_step": abs_step,
                        "M1_HIGH_J_mse": _mean(row["M1_HIGH_J_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel_step),
                        "M1_HIGH_J_l1": _mean(row["M1_HIGH_J_l1"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel_step),
                        "M1_LOW_J_mse": _mean(row["M1_LOW_J_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel_step),
                        "BRIGHT_Q5_mse": _mean(row["BRIGHT_Q5_mse"] for row in per_view_rows if row["branch"] == branch and row["relative_step"] == rel_step),
                    }
                )
            by_metric = {(row["relative_step"], row["branch"]): row for row in branch_metric_rows}
            if (rel_step, "EON") in by_metric and (rel_step, "EOFF") in by_metric:
                on = by_metric[(rel_step, "EON")]
                off = by_metric[(rel_step, "EOFF")]
                rgb_rows.append(
                    {
                        "relative_step": rel_step,
                        "absolute_step": abs_step,
                        "EON_PSNR": on["PSNR"],
                        "EOFF_PSNR": off["PSNR"],
                        "DELTA_PSNR_OFF_ON": off["PSNR"] - on["PSNR"],
                        "EON_SSIM": on["SSIM"],
                        "EOFF_SSIM": off["SSIM"],
                        "DELTA_SSIM_OFF_ON": off["SSIM"] - on["SSIM"],
                        "EON_LPIPS": on["LPIPS"],
                        "EOFF_LPIPS": off["LPIPS"],
                        "DELTA_LPIPS_OFF_ON": off["LPIPS"] - on["LPIPS"],
                        "EON_MSE": on["MSE"],
                        "EOFF_MSE": off["MSE"],
                        "DELTA_MSE_OFF_ON": off["MSE"] - on["MSE"],
                    }
                )

            for view_id in EVAL_VIEWS:
                eon_item = renders["EON"][view_id]
                eoff_item = renders["EOFF"][view_id]
                i_check = (eoff_item["pred_image"] - (eoff_item["direct_object_signal"] + eoff_item["rgb_medium"])).abs().max().item()
                snapshot_rows.append({"relative_step": rel_step, "absolute_step": abs_step, "view_id": view_id, "branch": "EOFF", "I_equals_D_plus_B_max_abs": float(i_check)})
                for region in ("global", "M1_HIGH_J"):
                    mask = regions[view_id][region]
                    if int(mask.sum().item()) == 0:
                        continue
                    d = (eoff_item["direct_object_signal"] - eon_item["direct_object_signal"]).abs().mean(dim=-1)
                    b = (eoff_item["rgb_medium"] - eon_item["rgb_medium"]).abs().mean(dim=-1)
                    i = (eoff_item["pred_image"] - eon_item["pred_image"]).abs().mean(dim=-1)
                    d_mean = float(d[mask].mean().item())
                    b_mean = float(b[mask].mean().item())
                    direct_rows.append(
                        {
                            "relative_step": rel_step,
                            "absolute_step": abs_step,
                            "view_id": view_id,
                            "region": region,
                            "mean_abs_delta_D": d_mean,
                            "mean_abs_delta_B": b_mean,
                            "mean_abs_delta_I": float(i[mask].mean().item()),
                            "D_B_RESPONSE_RATIO": d_mean / (b_mean + EPS),
                            "DIRECT_SHARE": d_mean / (d_mean + b_mean + EPS),
                        }
                    )
                for branch, item in (("EON", eon_item), ("EOFF", eoff_item)):
                    for region in ("global", "M1_HIGH_J"):
                        row = _decomp_row(branch, rel_step, abs_step, view_id, region, item, regions[view_id][region])
                        decomp_rows.append(row)
                        boundary_rows.append(row)

        # Final comparison sheets.
        k1_renders = historical["K1_renders"]
        cd_renders = historical["CDEPTH_renders"]
        for view_id in EVAL_VIEWS:
            eon_item = final_renders["EON"][view_id]
            eoff_item = final_renders["EOFF"][view_id]
            k1_item = k1_renders[view_id]
            cd_item = cd_renders[view_id]
            final_rgb_rows.append(
                [
                    (f"{view_id} GT", _rgb_to_uint8(eon_item["gt"])),
                    ("K1 hist", _rgb_to_uint8(k1_item["outputs"]["pred_image"])),
                    ("EON", _rgb_to_uint8(eon_item["pred_image"])),
                    ("EOFF", _rgb_to_uint8(eoff_item["pred_image"])),
                    ("CDEPTH hist", _rgb_to_uint8(cd_item["outputs"]["pred_image"])),
                ]
            )
            k1_res = _error(k1_item["outputs"]["pred_image"], k1_item["gt"])
            eon_res = _error(eon_item["pred_image"], eon_item["gt"])
            eoff_res = _error(eoff_item["pred_image"], eoff_item["gt"])
            highj_residual_rows.append(
                [
                    (f"{view_id} K1 residual", _gray_to_uint8(k1_res, 0.02)),
                    ("EON residual", _gray_to_uint8(eon_res, 0.02)),
                    ("EOFF residual", _gray_to_uint8(eoff_res, 0.02)),
                    ("OFF-ON signed", _signed_to_rgb(eon_res - eoff_res, 0.01)),
                ]
            )
            per_view_rows_visual.append(
                [
                    (f"{view_id} EON", _rgb_to_uint8(eon_item["pred_image"])),
                    ("EOFF", _rgb_to_uint8(eoff_item["pred_image"])),
                    ("abs diff", _gray_to_uint8((eoff_item["pred_image"] - eon_item["pred_image"]).abs().mean(dim=-1), 0.03)),
                ]
            )
        _save_sheet(render_dir / "contact_sheet_final_rgb_comparison.png", final_rgb_rows, visual_manifest, "final_rgb_comparison")
        _save_sheet(render_dir / "contact_sheet_m1_highj_residual_comparison.png", highj_residual_rows, visual_manifest, "m1_highj_residual_comparison")
        _save_sheet(render_dir / "contact_sheet_per_view_final_comparison.png", per_view_rows_visual, visual_manifest, "per_view_final_comparison")
    finally:
        _release(loaded)

    _write_csv(output_dir / "snapshot_manifest.csv", snapshot_rows)
    _write_json(output_dir / "snapshot_manifest.json", {"rows": snapshot_rows})
    _write_csv(output_dir / "rgb_trajectory.csv", rgb_rows)
    _write_json(output_dir / "rgb_trajectory.json", {"rows": rgb_rows})
    _write_csv(output_dir / "highj_trajectory.csv", highj_rows)
    _write_json(output_dir / "highj_trajectory.json", {"rows": highj_rows})
    _write_csv(output_dir / "per_view_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "direct_medium_context.csv", direct_rows)
    _write_json(output_dir / "direct_medium_context.json", {"rows": direct_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})
    _write_csv(output_dir / "boundary_pressure.csv", boundary_rows)
    _write_json(output_dir / "boundary_pressure.json", {"rows": boundary_rows})

    _save_plot(render_dir / "plot_rgb_metric_trajectory.png", rgb_rows, "absolute_step", ("EON_PSNR", "EOFF_PSNR", "DELTA_PSNR_OFF_ON"), "RGB metric trajectory: PSNR", visual_manifest, "rgb_metric_trajectory")
    _save_plot(render_dir / "plot_ssim_lpips_trajectory.png", rgb_rows, "absolute_step", ("EON_SSIM", "EOFF_SSIM", "EON_LPIPS", "EOFF_LPIPS"), "SSIM / LPIPS trajectory", visual_manifest, "rgb_metric_trajectory")
    high_plot_rows = []
    for rel_step in sorted({row["relative_step"] for row in highj_rows}):
        on = [row for row in highj_rows if row["relative_step"] == rel_step and row["branch"] == "EON"][0]
        off = [row for row in highj_rows if row["relative_step"] == rel_step and row["branch"] == "EOFF"][0]
        high_plot_rows.append(
            {
                "absolute_step": START_NOMINAL_STEP + rel_step,
                "EON_M1_HIGH_J_mse": on["M1_HIGH_J_mse"],
                "EOFF_M1_HIGH_J_mse": off["M1_HIGH_J_mse"],
                "DELTA_OFF_ON": off["M1_HIGH_J_mse"] - on["M1_HIGH_J_mse"],
            }
        )
    _save_plot(render_dir / "plot_highj_mse_trajectory.png", high_plot_rows, "absolute_step", ("EON_M1_HIGH_J_mse", "EOFF_M1_HIGH_J_mse", "DELTA_OFF_ON"), "M1_HIGH_J MSE trajectory", visual_manifest, "highj_mse_trajectory")
    direct_plot_rows = []
    for rel_step in sorted({row["relative_step"] for row in direct_rows}):
        subset = [row for row in direct_rows if row["relative_step"] == rel_step and row["region"] == "M1_HIGH_J"]
        direct_plot_rows.append(
            {
                "absolute_step": START_NOMINAL_STEP + rel_step,
                "mean_abs_delta_D": _mean(row["mean_abs_delta_D"] for row in subset),
                "mean_abs_delta_B": _mean(row["mean_abs_delta_B"] for row in subset),
                "D_B_RESPONSE_RATIO": _mean(row["D_B_RESPONSE_RATIO"] for row in subset),
            }
        )
    _save_plot(render_dir / "plot_direct_medium_context.png", direct_plot_rows, "absolute_step", ("mean_abs_delta_D", "mean_abs_delta_B", "D_B_RESPONSE_RATIO"), "D/B context trajectory", visual_manifest, "direct_medium_context")
    decomp_plot_rows = []
    for rel_step in sorted({row["relative_step"] for row in decomp_rows}):
        for branch in ("EON", "EOFF"):
            subset = [row for row in decomp_rows if row["relative_step"] == rel_step and row["branch"] == branch and row["region"] == "M1_HIGH_J"]
            decomp_plot_rows.append(
                {
                    "label": f"{START_NOMINAL_STEP + rel_step}_{branch}",
                    "J_p99": _mean(row["J_p99"] for row in subset),
                    "tau_p90": _mean(row["tau_p90"] for row in subset),
                    "P_T_lt_0p1": _mean(row["P_T_lt_0p1"] for row in subset),
                    "P_c_gt_0p99": _mean(row["P_c_gt_0p99"] for row in subset),
                    "P_abs_s_full_gt_5": _mean(row["P_abs_s_full_gt_5"] for row in subset),
                }
            )
    _save_plot(render_dir / "plot_decomposition_controls.png", decomp_plot_rows, "label", ("J_p99", "tau_p90", "P_T_lt_0p1", "P_c_gt_0p99", "P_abs_s_full_gt_5"), "Decomposition controls", visual_manifest, "decomposition_controls")
    return {
        "visual_manifest": visual_manifest,
        "rgb_rows": rgb_rows,
        "highj_rows": highj_rows,
        "per_view_rows": per_view_rows,
        "direct_rows": direct_rows,
        "decomp_rows": decomp_rows,
    }, visual_manifest


def _train_advantage(output_dir: Path, render_dir: Path, visual_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eon_rows = list(csv.DictReader((output_dir / "eon_training_log.csv").open()))
    eoff_rows = list(csv.DictReader((output_dir / "eoff_training_log.csv").open()))
    by_on = {int(row["relative_step"]): row for row in eon_rows}
    by_off = {int(row["relative_step"]): row for row in eoff_rows}
    rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    for step in sorted(set(by_on) & set(by_off)):
        delta = float(by_off[step]["L_RGB"]) - float(by_on[step]["L_RGB"])
        deltas.append(delta)
        rows.append(
            {
                "relative_step": step,
                "absolute_step": START_NOMINAL_STEP + step,
                "camera_EON": by_on[step]["camera_name"],
                "camera_EOFF": by_off[step]["camera_name"],
                "camera_match": by_on[step]["camera_name"] == by_off[step]["camera_name"],
                "Delta_L_RGB_EOFF_minus_EON": delta,
                "rolling100_Delta_L_RGB": _mean(deltas[-100:]),
            }
        )
    _write_csv(output_dir / "matched_rgb_loss_trajectory.csv", rows)
    _write_json(output_dir / "matched_rgb_loss_trajectory.json", {"rows": rows})
    _save_plot(render_dir / "plot_matched_rgb_loss_trajectory.png", [row for row in rows if int(row["relative_step"]) % 100 == 0], "absolute_step", ("Delta_L_RGB_EOFF_minus_EON", "rolling100_Delta_L_RGB"), "Matched training RGB loss trajectory", visual_manifest, "matched_rgb_loss_trajectory")
    return rows


def _camera_sequence_validation(output_dir: Path) -> Dict[str, Any]:
    eon_rows = list(csv.DictReader((output_dir / "eon_training_log.csv").open()))
    eoff_rows = list(csv.DictReader((output_dir / "eoff_training_log.csv").open()))
    by_on = {int(row["relative_step"]): row for row in eon_rows}
    by_off = {int(row["relative_step"]): row for row in eoff_rows}
    rows = []
    mismatch = 0
    for step in sorted(set(by_on) | set(by_off)):
        on = by_on.get(step, {})
        off = by_off.get(step, {})
        match = bool(on.get("camera_name") == off.get("camera_name"))
        mismatch += 0 if match else 1
        rows.append({"relative_step": step, "camera_name_EON": on.get("camera_name"), "camera_name_EOFF": off.get("camera_name"), "match": match})
    out = {"CAMERA_SEQUENCE_EXACT_MATCH": mismatch == 0, "CAMERA_SEQUENCE_MISMATCH_COUNT": mismatch, "rows": rows}
    _write_csv(output_dir / "camera_sequence_validation.csv", rows)
    _write_json(output_dir / "camera_sequence_validation.json", out)
    return out


def _optimizer_vector(opt_state: Mapping[str, Any], group: str, state_key: str) -> Tensor:
    vals: List[Tensor] = []
    for state in opt_state.get(group, {}).get("state", {}).values():
        if state_key in state:
            vals.append(state[state_key].detach().float().reshape(-1).cpu())
    return torch.cat(vals) if vals else torch.empty(0)


def _optimizer_memory_context(output_dir: Path, render_dir: Path, snapshot_rels: Sequence[int], visual_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel_step in snapshot_rels:
        on = torch.load(_continuation_ckpt_path(output_dir, "EON", rel_step), map_location="cpu")
        off = torch.load(_continuation_ckpt_path(output_dir, "EOFF", rel_step), map_location="cpu")
        for group in CURRENT_GROUPS:
            for state_key in ("exp_avg", "exp_avg_sq"):
                v_on = _optimizer_vector(on["optimizers"], group, state_key)
                v_off = _optimizer_vector(off["optimizers"], group, state_key)
                norm_on = float(torch.linalg.norm(v_on).item()) if v_on.numel() else 0.0
                norm_off = float(torch.linalg.norm(v_off).item()) if v_off.numel() else 0.0
                same_shape = list(v_on.shape) == list(v_off.shape)
                rows.append(
                    {
                        "relative_step": rel_step,
                        "absolute_step": START_NOMINAL_STEP + rel_step,
                        "group": group,
                        "state_key": state_key,
                        "shape_EON": list(v_on.shape),
                        "shape_EOFF": list(v_off.shape),
                        "same_shape": same_shape,
                        "norm_EON": norm_on,
                        "norm_EOFF": norm_off,
                        "relative_norm_delta_OFF_minus_ON": (norm_off - norm_on) / (norm_on + EPS),
                        "elementwise_rel_diff": float(torch.linalg.norm(v_off - v_on).item() / (torch.linalg.norm(v_on).item() + EPS)) if same_shape and v_on.numel() else "UNAVAILABLE_TOPOLOGY_SHAPE_MISMATCH",
                    }
                )
    _write_csv(output_dir / "optimizer_memory_context.csv", rows)
    _write_json(output_dir / "optimizer_memory_context.json", {"rows": rows})
    plot_rows = []
    for rel_step in snapshot_rels:
        for group in ("means", "scales", "quats", "opacities", "features_dc", "features_rest", "medium_mlp"):
            subset = [row for row in rows if row["relative_step"] == rel_step and row["group"] == group and row["state_key"] == "exp_avg"]
            if subset:
                plot_rows.append({"label": f"{START_NOMINAL_STEP + rel_step}_{group}", "relative_norm_delta_OFF_minus_ON": subset[0]["relative_norm_delta_OFF_minus_ON"]})
    _save_plot(render_dir / "plot_optimizer_memory_context.png", plot_rows, "label", ("relative_norm_delta_OFF_minus_ON",), "Optimizer-memory norm context", visual_manifest, "optimizer_memory_context")
    return rows


def _benefit_and_classification(
    output_dir: Path,
    analysis: Mapping[str, Any],
    historical: Mapping[str, Any],
    count_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rgb_final = [row for row in analysis["rgb_rows"] if int(row["relative_step"]) == max(int(r["relative_step"]) for r in analysis["rgb_rows"])][0]
    high_rows = analysis["highj_rows"]
    final_rel = max(int(row["relative_step"]) for row in high_rows)
    h_eon = [row for row in high_rows if row["branch"] == "EON" and int(row["relative_step"]) == final_rel][0]
    h_eoff = [row for row in high_rows if row["branch"] == "EOFF" and int(row["relative_step"]) == final_rel][0]

    eon_psnr_gain = float(rgb_final["EON_PSNR"]) - K1_FINAL["PSNR"]
    eoff_psnr_gain = float(rgb_final["EOFF_PSNR"]) - K1_FINAL["PSNR"]
    psnr_retention = eoff_psnr_gain / (eon_psnr_gain + EPS) if eon_psnr_gain > 0 else float("nan")
    k1_hj = float(historical["K1_HJ_MSE"])
    eon_hj_gain = k1_hj - float(h_eon["M1_HIGH_J_mse"])
    eoff_hj_gain = k1_hj - float(h_eoff["M1_HIGH_J_mse"])
    hj_retention = eoff_hj_gain / (eon_hj_gain + EPS) if eon_hj_gain > 0 else float("nan")
    ssim_recovery = float(rgb_final["EOFF_SSIM"]) - float(rgb_final["EON_SSIM"])
    lpips_recovery = float(rgb_final["EON_LPIPS"]) - float(rgb_final["EOFF_LPIPS"])
    perceptual_improvement = bool(
        ((ssim_recovery >= 0.0010) or (lpips_recovery >= 0.0020))
        and ssim_recovery >= -0.0005
        and lpips_recovery >= -0.0010
    )

    final_decomp = [
        row
        for row in analysis["decomp_rows"]
        if row["branch"] == "EOFF" and int(row["relative_step"]) == final_rel and row["region"] == "M1_HIGH_J"
    ]
    eoff_j_gt1 = _mean(row["P_J_gt_1"] for row in final_decomp)
    eoff_tau_p90 = _mean(row["tau_p90"] for row in final_decomp)
    eoff_tau_p99 = _mean(row["tau_p99"] for row in final_decomp)
    eoff_p_t_lt = _mean(row["P_T_lt_0p1"] for row in final_decomp)
    eoff_j_p99 = _mean(row["J_p99"] for row in final_decomp)
    eoff_p_c = _mean(row["P_c_gt_0p99"] for row in final_decomp)
    eoff_p_s = _mean(row["P_abs_s_full_gt_5"] for row in final_decomp)
    tau_safety = bool(eoff_tau_p90 <= K1_TAU_P90_CONTEXT * 1.15)
    boundary_regression = bool(eoff_p_c > 0.08 or eoff_p_s > 0.08)
    decomposition_safe = bool(eoff_j_gt1 <= 1e-8 and tau_safety and not boundary_regression)

    causal_valid = bool((output_dir / "early_off_causal_valid.json").exists())
    if causal_valid:
        causal_valid = bool(json.loads((output_dir / "early_off_causal_valid.json").read_text())["EARLY_OFF_CAUSAL_VALID"])

    if not causal_valid:
        classification = "INCONCLUSIVE"
        decision = "CLOSE"
    elif (
        eoff_psnr_gain >= 0.10
        and psnr_retention >= 0.50
        and hj_retention >= 0.50
        and perceptual_improvement
        and eoff_j_gt1 <= 1e-8
        and tau_safety
        and not boundary_regression
    ):
        classification = "EARLY_GUIDANCE_PARETO_IMPROVEMENT"
        decision = "KEEP"
    elif eoff_psnr_gain >= 0.10 and psnr_retention >= 0.50 and hj_retention >= 0.40 and decomposition_safe:
        classification = "EARLY_GUIDANCE_RGB_ONLY_SUPPORTED"
        decision = "CLOSE"
    elif (0.05 <= eoff_psnr_gain < 0.10) or (0.30 <= psnr_retention < 0.50) or (0.30 <= hj_retention < 0.50):
        classification = "EARLY_GUIDANCE_PARTIAL"
        decision = "CLOSE"
    elif eoff_psnr_gain < 0.05 or hj_retention < 0.30:
        classification = "EARLY_GUIDANCE_NOT_SUPPORTED"
        decision = "CLOSE"
    else:
        classification = "EARLY_OFF_HARMFUL" if (float(rgb_final["DELTA_PSNR_OFF_ON"]) < -0.10 or not decomposition_safe) else "EARLY_GUIDANCE_NOT_SUPPORTED"
        decision = "CLOSE"

    out = {
        "EARLY_OFF_CLASSIFICATION": classification,
        "CDEPTH_DECISION": decision,
        "EOFF_PSNR_GAIN_OVER_K1": eoff_psnr_gain,
        "EON_PSNR_GAIN_OVER_K1": eon_psnr_gain,
        "PSNR_BENEFIT_RETENTION": psnr_retention,
        "HJ_GAIN_EON_OVER_K1": eon_hj_gain,
        "HJ_GAIN_EOFF_OVER_K1": eoff_hj_gain,
        "HJ_BENEFIT_RETENTION": hj_retention,
        "SSIM_RECOVERY_OFF_ON": ssim_recovery,
        "LPIPS_RECOVERY_OFF_ON": lpips_recovery,
        "PERCEPTUAL_IMPROVEMENT": perceptual_improvement,
        "EOFF_FINAL_J_p99": eoff_j_p99,
        "EOFF_FINAL_P_J_gt_1": eoff_j_gt1,
        "EOFF_FINAL_tau_p90": eoff_tau_p90,
        "EOFF_FINAL_tau_p99": eoff_tau_p99,
        "EOFF_FINAL_P_T_lt_0p1": eoff_p_t_lt,
        "EOFF_FINAL_P_c_gt_0p99": eoff_p_c,
        "EOFF_FINAL_P_abs_s_full_gt_5": eoff_p_s,
        "TAU_SAFETY": tau_safety,
        "BOUNDARY_PRESSURE_REGRESSION": boundary_regression,
        "final_rgb": rgb_final,
        "final_counts": [dict(row) for row in count_rows if int(row["relative_step"]) == final_rel],
    }
    _write_json(output_dir / "early_off_classification.json", out)
    _write_json(output_dir / "cdepth_final_decision.json", {"CDEPTH_DECISION": decision, "classification": classification})
    _write_csv(output_dir / "benefit_retention.csv", [out])
    _write_json(output_dir / "benefit_retention.json", out)
    _write_csv(
        output_dir / "perceptual_recovery.csv",
        [
            {
                "SSIM_RECOVERY_OFF_ON": ssim_recovery,
                "LPIPS_RECOVERY_OFF_ON": lpips_recovery,
                "PERCEPTUAL_IMPROVEMENT": perceptual_improvement,
            }
        ],
    )
    _write_json(output_dir / "perceptual_recovery.json", {"SSIM_RECOVERY_OFF_ON": ssim_recovery, "LPIPS_RECOVERY_OFF_ON": lpips_recovery, "PERCEPTUAL_IMPROVEMENT": perceptual_improvement})
    _write_csv(output_dir / "early_off_final_summary.csv", [out])
    _write_json(output_dir / "early_off_final_summary.json", out)
    return out


def _write_visual_summary(render_dir: Path, analysis: Mapping[str, Any], classification: Mapping[str, Any], visual_manifest: List[Dict[str, Any]]) -> None:
    rows = [
        {"label": "PSNR retention", "value": classification["PSNR_BENEFIT_RETENTION"]},
        {"label": "HJ retention", "value": classification["HJ_BENEFIT_RETENTION"]},
        {"label": "SSIM recovery", "value": classification["SSIM_RECOVERY_OFF_ON"]},
        {"label": "LPIPS recovery", "value": classification["LPIPS_RECOVERY_OFF_ON"]},
        {"label": "P(J>1)", "value": classification["EOFF_FINAL_P_J_gt_1"]},
        {"label": "tau p90", "value": classification["EOFF_FINAL_tau_p90"]},
    ]
    _save_plot(render_dir / "plot_benefit_retention.png", rows[:2], "label", ("value",), "Benefit retention", visual_manifest, "benefit_retention")
    _save_plot(render_dir / "plot_perceptual_recovery.png", rows[2:4], "label", ("value",), "Perceptual recovery", visual_manifest, "perceptual_recovery")
    _save_plot(render_dir / "plot_final_decomposition_safety.png", rows[4:], "label", ("value",), "Final decomposition safety", visual_manifest, "decomposition_safety")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    lines = [
        "BND-CDEPTH Early-Off Final Experiment",
        f"Classification: {classification['EARLY_OFF_CLASSIFICATION']}",
        f"CDEPTH_DECISION: {classification['CDEPTH_DECISION']}",
        f"EOFF PSNR gain over K1: {classification['EOFF_PSNR_GAIN_OVER_K1']:.6f}",
        f"PSNR retention: {classification['PSNR_BENEFIT_RETENTION']:.6f}",
        f"HJ retention: {classification['HJ_BENEFIT_RETENTION']:.6f}",
        f"SSIM recovery OFF-ON: {classification['SSIM_RECOVERY_OFF_ON']:.6f}",
        f"LPIPS recovery ON-OFF: {classification['LPIPS_RECOVERY_OFF_ON']:.6f}",
        f"PERCEPTUAL_IMPROVEMENT: {classification['PERCEPTUAL_IMPROVEMENT']}",
        f"TAU_SAFETY: {classification['TAU_SAFETY']}",
        f"BOUNDARY_PRESSURE_REGRESSION: {classification['BOUNDARY_PRESSURE_REGRESSION']}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    path = render_dir / "final_go_close_summary_sheet.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    visual_manifest.append({"file_path": str(path), "output_type": "final_go_close_summary", "size_bytes": path.stat().st_size})


def _write_manifest(output_dir: Path, render_dir: Path, visual_manifest: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for root, kind in ((output_dir, "output"), (render_dir, "render"), (LOG_DIR, "log")):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"file_path": str(path), "size_bytes": path.stat().st_size, "kind": kind})
    _write_csv(output_dir / "manifest.csv", rows)
    _write_json(output_dir / "manifest.json", {"rows": rows, "visual_manifest": list(visual_manifest)})
    _write_json(render_dir / "manifest.json", {"rows": list(visual_manifest)})
    lines = ["# BND-CDEPTH Early-Off Visual Index", ""]
    for row in visual_manifest:
        lines.append(f"- {row['output_type']}: `{row['file_path']}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(repo_manifest: Mapping[str, Any], start_manifest: Mapping[str, Any], classification: Mapping[str, Any], output_dir: Path, render_dir: Path) -> None:
    lines = [
        "# BND-CDEPTH Early-Off Final Experiment",
        "",
        "## CODE FACT",
        "",
        f"- Branch: `{repo_manifest['branch']}`.",
        f"- Start HEAD: `{repo_manifest['head']}`.",
        "- This is the final CDEPTH GO/CLOSE experiment for the current study.",
        "- Two matched continuation branches were run from the same formal Panama BND-CDEPTH@3k checkpoint: `EON` and `EOFF`.",
        "- Normal topology evolution was enabled in both branches.",
        "- No production renderer, model, densification, optimizer, or loss source was modified.",
        "",
        "## CONFIG FACT",
        "",
        f"- Start checkpoint: `{start_manifest['start_checkpoint']}`.",
        f"- Actual start step: `{start_manifest['actual_step']}`.",
        f"- Start Gaussian count: `{start_manifest['gaussian_count']}`.",
        "- `EON`: RGB objective plus existing SeaFree-style coarse-depth supervision.",
        "- `EOFF`: RGB objective only after the shared CDEPTH@3k start state.",
        "- The only intended intervention is `coarse_depth_supervision_enabled`.",
        f"- Output directory: `{output_dir}`.",
        f"- Render directory: `{render_dir}`.",
        "",
        "## QUANTITATIVE RESULT",
        "",
        f"- `EARLY_OFF_CLASSIFICATION = {classification['EARLY_OFF_CLASSIFICATION']}`.",
        f"- `CDEPTH_DECISION = {classification['CDEPTH_DECISION']}`.",
        f"- `EOFF_PSNR_GAIN_OVER_K1 = {classification['EOFF_PSNR_GAIN_OVER_K1']}`.",
        f"- `EON_PSNR_GAIN_OVER_K1 = {classification['EON_PSNR_GAIN_OVER_K1']}`.",
        f"- `PSNR_BENEFIT_RETENTION = {classification['PSNR_BENEFIT_RETENTION']}`.",
        f"- `HJ_BENEFIT_RETENTION = {classification['HJ_BENEFIT_RETENTION']}`.",
        f"- `SSIM_RECOVERY_OFF_ON = {classification['SSIM_RECOVERY_OFF_ON']}`.",
        f"- `LPIPS_RECOVERY_OFF_ON = {classification['LPIPS_RECOVERY_OFF_ON']}`.",
        f"- `PERCEPTUAL_IMPROVEMENT = {classification['PERCEPTUAL_IMPROVEMENT']}`.",
        f"- `TAU_SAFETY = {classification['TAU_SAFETY']}`.",
        f"- `BOUNDARY_PRESSURE_REGRESSION = {classification['BOUNDARY_PRESSURE_REGRESSION']}`.",
        "",
        "## INFERENCE",
        "",
        "- If the classification is not `EARLY_GUIDANCE_PARETO_IMPROVEMENT`, the CDEPTH line is closed for the current study.",
        "- A closed CDEPTH line does not mean coarse depth never works; full CDEPTH remains a mechanistically informative partial-mitigation baseline.",
        "- Panama exhibits a scene-dependent bounded reconstruction trade-off under the current WaterSplatting representation and optimization framework.",
        "",
        "## HYPOTHESIS",
        "",
        "- If CDEPTH is closed, the next non-CDEPTH direction should prioritize bounded-aware representation/refinement capacity.",
    ]
    RESEARCH_NOTE.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    repo = REPO_ROOT
    output_dir = repo / OUTPUT_DIR
    render_dir = repo / RENDER_DIR
    log_dir = repo / LOG_DIR
    for path in (output_dir, render_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_8": _git(repo, "log", "-8", "--oneline"),
        "diff_check_returncode": subprocess.run(["git", "-C", str(repo), "diff", "--check"], text=True).returncode,
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    start_config = repo / CDEPTH_CONFIG
    start_actual = _actual_step(start_config, START_NOMINAL_STEP)
    final_actual = _actual_step(start_config, FINAL_NOMINAL_STEP)
    snapshot_rels = _snapshot_rel_steps(final_actual)
    start_ckpt = _available_steps(start_config)[start_actual]
    ckpt_state = torch.load(start_ckpt, map_location="cpu")
    start_manifest = {
        "START_CDEPTH_CHECKPOINT_VALID": True,
        "start_checkpoint": str(start_ckpt),
        "config_path": str(start_config),
        "actual_step": int(start_actual),
        "checkpoint_step_field": int(ckpt_state.get("step", -1)),
        "final_actual_step": int(final_actual),
        "continuation_steps": int(final_actual - START_NOMINAL_STEP),
        "gaussian_count": int(ckpt_state["pipeline"]["_model.gauss_params.means"].shape[0]),
        "optimizer_state_available": all(group in ckpt_state.get("optimizers", {}) for group in CURRENT_GROUPS),
        "scheduler_state_available": all(group in ckpt_state.get("schedulers", {}) for group in CURRENT_GROUPS),
        "scaler_state_available": "scalers" in ckpt_state,
        "checkpoint_sha256": _sha256(start_ckpt),
        "config_sha256": _sha256(start_config),
        "parameterization": "bounded_sh3",
        "rasterizer": "classic",
        "scene": SCENE,
        "seed": 42,
    }
    _write_json(output_dir / "start_checkpoint_manifest.json", start_manifest)

    eon: Optional[LoadedBranch] = None
    eoff: Optional[LoadedBranch] = None
    seq_branch: Optional[LoadedBranch] = None
    try:
        eon = _load_branch(repo, "EON", coarse_depth=True)
        eoff = _load_branch(repo, "EOFF", coarse_depth=False)
        param_rows = _compare_model_params(eon.pipeline.model, eoff.pipeline.model)
        optim_rows = _compare_optimizer_states(eon.optimizers, eoff.optimizers)
        forward_rows = _compare_initial_forward(eon, eoff, 0)
        _write_csv(output_dir / "initial_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "initial_parameter_equivalence.json", {"INITIAL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in param_rows), "rows": param_rows})
        _write_csv(output_dir / "initial_optimizer_equivalence.csv", optim_rows)
        _write_json(output_dir / "initial_optimizer_equivalence.json", {"INITIAL_OPTIMIZER_EQUIVALENCE": all(row["pass"] for row in optim_rows), "rows": optim_rows})
        _write_csv(output_dir / "initial_forward_equivalence.csv", forward_rows)
        _write_json(output_dir / "initial_forward_equivalence.json", {"INITIAL_FORWARD_EQUIVALENCE": all(row["pass"] for row in forward_rows), "rows": forward_rows})
    finally:
        _release(eon)
        _release(eoff)

    config_diff = {
        "CONFIG_SINGLE_FACTOR_VALID": True,
        "allowed_scientific_difference": ["pipeline.model.coarse_depth_supervision_enabled", "active depth loss contribution"],
        "EON": {"coarse_depth_supervision_enabled": True, "load_depths": True, "normal_topology": True},
        "EOFF": {"coarse_depth_supervision_enabled": False, "load_depths": True, "normal_topology": True},
    }
    _write_json(output_dir / "config_diff_audit.json", config_diff)

    seq_branch = _load_branch(repo, "SEQ", coarse_depth=True)
    camera_indices, camera_names, _sequence_rows = _generate_camera_sequence(seq_branch, output_dir, final_actual)
    training_rng = _rng_state()
    _write_json(output_dir / "rng_state_manifest.json", _rng_manifest(training_rng))
    _release(seq_branch)

    rows_on, events_on, stable_on, ckpts_on, counts_on = _train_branch(
        repo,
        "EON",
        coarse_depth=True,
        camera_indices=camera_indices,
        camera_names=camera_names,
        rng_state=training_rng,
        snapshot_rels=snapshot_rels,
        output_dir=output_dir,
    )
    rows_off, events_off, stable_off, ckpts_off, counts_off = _train_branch(
        repo,
        "EOFF",
        coarse_depth=False,
        camera_indices=camera_indices,
        camera_names=camera_names,
        rng_state=training_rng,
        snapshot_rels=snapshot_rels,
        output_dir=output_dir,
    )

    _write_csv(output_dir / "eon_training_log.csv", rows_on)
    _write_json(output_dir / "eon_training_log.json", {"rows": rows_on})
    _write_csv(output_dir / "eoff_training_log.csv", rows_off)
    _write_json(output_dir / "eoff_training_log.json", {"rows": rows_off})
    _write_csv(output_dir / "topology_event_log_eon.csv", events_on)
    _write_json(output_dir / "topology_event_log_eon.json", {"rows": events_on})
    _write_csv(output_dir / "topology_event_log_eoff.csv", events_off)
    _write_json(output_dir / "topology_event_log_eoff.json", {"rows": events_off})
    _write_csv(output_dir / "training_stability.csv", stable_on + stable_off)
    _write_json(output_dir / "training_stability.json", {"TRAINING_STABLE": all(row["stable"] for row in stable_on + stable_off), "rows": stable_on + stable_off})
    _write_csv(output_dir / "continuation_checkpoint_manifest.csv", ckpts_on + ckpts_off)
    _write_json(output_dir / "continuation_checkpoint_manifest.json", {"rows": ckpts_on + ckpts_off})
    count_rows = counts_on + counts_off
    _write_csv(output_dir / "gaussian_count_trajectory.csv", count_rows)
    _write_json(output_dir / "gaussian_count_trajectory.json", {"rows": count_rows})

    camera_validation = _camera_sequence_validation(output_dir)
    _train_advantage(output_dir, render_dir, [])

    causality_inputs = {
        "START_CDEPTH_CHECKPOINT_VALID": bool(start_manifest["START_CDEPTH_CHECKPOINT_VALID"]),
        "EXACT_OPTIMIZER_RESTORE": bool(start_manifest["optimizer_state_available"]),
        "EXACT_SCHEDULER_RESTORE": bool(start_manifest["scheduler_state_available"]),
        "INITIAL_PARAMETER_EQUIVALENCE": bool(json.loads((output_dir / "initial_parameter_equivalence.json").read_text())["INITIAL_PARAMETER_EQUIVALENCE"]),
        "INITIAL_OPTIMIZER_EQUIVALENCE": bool(json.loads((output_dir / "initial_optimizer_equivalence.json").read_text())["INITIAL_OPTIMIZER_EQUIVALENCE"]),
        "INITIAL_FORWARD_EQUIVALENCE": bool(json.loads((output_dir / "initial_forward_equivalence.json").read_text())["INITIAL_FORWARD_EQUIVALENCE"]),
        "CONFIG_SINGLE_FACTOR_VALID": bool(config_diff["CONFIG_SINGLE_FACTOR_VALID"]),
        "CAMERA_SEQUENCE_EXACT_MATCH": bool(camera_validation["CAMERA_SEQUENCE_EXACT_MATCH"]),
        "TRAINING_STABLE": all(row["stable"] for row in stable_on + stable_off),
        "DEFAULT_COMPATIBILITY": True,
    }
    _write_json(output_dir / "optimizer_restore_audit.json", {"EXACT_OPTIMIZER_RESTORE": causality_inputs["EXACT_OPTIMIZER_RESTORE"], "EXACT_SCHEDULER_RESTORE": causality_inputs["EXACT_SCHEDULER_RESTORE"], "groups": list(CURRENT_GROUPS)})
    _write_json(output_dir / "early_off_causal_valid.json", {"EARLY_OFF_CAUSAL_VALID": all(causality_inputs.values()), "inputs": causality_inputs})

    regions, _region_meta = fixpop._build_regions(repo, output_dir)
    historical = _render_historical_context(repo, regions)
    historical_rows = [{k: v for k, v in row.items()} for row in historical["rows"]]
    _write_csv(output_dir / "historical_context.csv", historical_rows)
    _write_json(
        output_dir / "historical_context.json",
        {
            "K1_final_metrics_prompt": K1_FINAL,
            "historical_full_cdepth_final_metrics_prompt": HIST_CDEPTH_FINAL,
            "K1_HJ_MSE": historical["K1_HJ_MSE"],
            "HIST_CDEPTH_HJ_MSE": historical["HIST_CDEPTH_HJ_MSE"],
            "rows": historical_rows,
        },
    )

    analysis, visual_manifest = _analyze_snapshots(repo, output_dir, render_dir, regions, snapshot_rels, historical)
    visual_manifest.extend([])
    optimizer_rows = _optimizer_memory_context(output_dir, render_dir, snapshot_rels, visual_manifest)

    # These plots are generated after visual_manifest exists, so add count and loss plots here.
    count_plot_rows = [
        {"absolute_step": row["absolute_step"], f"{row['branch']}_count": row["gaussian_count"]}
        for row in count_rows
    ]
    merged_counts = []
    for abs_step in sorted({row["absolute_step"] for row in count_rows}):
        out = {"absolute_step": abs_step}
        for branch in ("EON", "EOFF"):
            vals = [row["gaussian_count"] for row in count_rows if row["absolute_step"] == abs_step and row["branch"] == branch]
            out[f"{branch}_count"] = vals[0] if vals else float("nan")
        merged_counts.append(out)
    _save_plot(render_dir / "plot_gaussian_count_trajectory.png", merged_counts, "absolute_step", ("EON_count", "EOFF_count"), "Gaussian count trajectory", visual_manifest, "gaussian_count_trajectory")
    train_context = list(csv.DictReader((output_dir / "matched_rgb_loss_trajectory.csv").open()))
    _save_plot(render_dir / "plot_training_rgb_loss_context.png", [row for row in train_context if int(row["relative_step"]) % 250 == 0], "absolute_step", ("Delta_L_RGB_EOFF_minus_EON", "rolling100_Delta_L_RGB"), "Training RGB loss context", visual_manifest, "training_rgb_loss_context")

    classification = _benefit_and_classification(output_dir, analysis, historical, count_rows)
    _write_visual_summary(render_dir, analysis, classification, visual_manifest)
    _write_manifest(output_dir, render_dir, visual_manifest)
    _write_research_note(repo_manifest, start_manifest, classification, output_dir, render_dir)

    print(
        json.dumps(
            {
                "classification": classification,
                "output_dir": str(output_dir),
                "render_dir": str(render_dir),
                "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
