#!/usr/bin/env python3
"""Read-only causal audit of OCMC projector temporal mismatch on IUI3.

This diagnostic compares the stale projector saved in each C1 checkpoint with a
fresh projector recomputed from the current checkpoint state, then evaluates the
same C1 weights under three forward conditions:

* IDENTITY: no projector.
* STALE: projector bundle saved with the checkpoint.
* FRESH: projector recomputed from the current checkpoint state.

No training, optimizer steps, checkpoint mutation, or projector redesign occur
in this script.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_ocmc_causal_iui3 as OCMC
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC


PW = MI.PW

EXPERIMENT = "OCMC-PROJECTOR-TEMPORAL-MISMATCH-AUDIT"
SCENE = "IUI3-RedSea"
SOURCE_OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
OUTPUT_DIR = Path("outputs/ocmc_projector_temporal_mismatch_iui3_20260825")
LOG_DIR = Path("logs/ocmc_projector_temporal_mismatch_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/OCMC_PROJECTOR_TEMPORAL_MISMATCH_IUI3_2026-08-25.md")
PRIMARY_STEPS = (8000, 13000, 14999)
FINAL_STEP = 14999
TRAIN_POPULATIONS = ("GENERAL", "M_SAFE")
PROJECTOR_POPULATION = OCMC.PROJECTOR_POPULATION
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
SAMPLES_PER_VIEW = MI.SAMPLES_PER_VIEW
SAMPLE_SEED = 20260825
EPS = 1e-12


@dataclass
class ConditionBundle:
    name: str
    bundle: Optional[Mapping[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _object_hash(value: Any) -> str:
    import pickle

    return _sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _assert_runtime_policy() -> Dict[str, Any]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {conda_env!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly one physical GPU in "
            f"{sorted(ALLOWED_PHYSICAL_GPUS)}; got {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this diagnostic.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one torch-visible GPU after masking, got {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": 0,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def _environment_manifest(gpu_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu_manifest),
    }


def _repo_manifest(repo: Path) -> Dict[str, Any]:
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "historical_untracked_files_preserved_not_modified_or_committed": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return float("nan")
    vals.sort()
    return vals[len(vals) // 2]


def _rms(delta: Tensor) -> float:
    if delta.numel() == 0:
        return float("nan")
    return float(torch.sqrt(delta.detach().float().square().mean()).item())


def _condition_projector_bundle(condition: str, bundles: Mapping[str, Optional[Mapping[str, Any]]]) -> Optional[Mapping[str, Any]]:
    if condition not in bundles:
        raise KeyError(condition)
    return bundles[condition]


def _load_checkpoint_manifest(repo: Path, source_output_dir: Path, steps: Sequence[int]) -> List[Dict[str, Any]]:
    path = source_output_dir / "checkpoint_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf8"))
    rows = obj.get("rows", [])
    out: List[Dict[str, Any]] = []
    wanted = {int(step) for step in steps}
    for row in rows:
        if row.get("branch") != "C1":
            continue
        step = int(row["absolute_step"])
        if step not in wanted:
            continue
        ckpt_path = _repo_path(Path(row["checkpoint_path"]), repo)
        out.append(
            {
                "requested_step": step,
                "actual_step": step,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_exists": bool(ckpt_path.exists()),
                "ocmc_projector_refresh_step": row.get("ocmc_projector_refresh_step", ""),
            }
        )
    out.sort(key=lambda row: int(row["requested_step"]))
    return out


def _load_swap_bank(repo: Path, source_output_dir: Path, train_view_ids: Sequence[str]) -> Dict[str, Any]:
    path = source_output_dir / "real_camera_swap_bank.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf8"))
        rows = obj.get("rows", {})
        return {
            "seed": int(obj.get("seed", 0)),
            "alternatives_per_source": int(obj.get("alternatives_per_source", 0)),
            "rows": {str(k): list(v) for k, v in rows.items()},
            "source_path": str(path),
            "hash": _sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf8")),
            "reused_existing_bank": True,
        }

    seed = 202608255
    rng = random.Random(seed)
    rows: Dict[str, List[str]] = {}
    for view_id in train_view_ids:
        candidates = [v for v in train_view_ids if v != view_id]
        local = candidates[:]
        rng.shuffle(local)
        rows[view_id] = local[:8]
    out = {
        "seed": seed,
        "alternatives_per_source": 8,
        "rows": rows,
        "source_path": "deterministic_reconstruction",
        "hash": _sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf8")),
        "reused_existing_bank": False,
    }
    _write_json(source_output_dir / "real_camera_swap_bank.json", out)
    return out


def _build_samples(repo: Path, output_dir: Path) -> Tuple[Dict[str, MI.ViewSample], Dict[str, Any], List[Dict[str, Any]], str]:
    samples, meta, rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    rows_hash = _sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf8"))
    manifest = {
        "scene": SCENE,
        "sample_seed": SAMPLE_SEED,
        "samples_per_view": SAMPLES_PER_VIEW,
        "train_view_count": len(samples),
        "train_views": list(samples.keys()),
        "rows_hash": rows_hash,
        "meta_hash": _object_hash(meta),
        "source_meta": meta,
        "reused_registered_deterministic_bank": True,
    }
    _write_json(output_dir / "sample_bank_manifest.json", manifest)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", meta)
    return samples, meta, rows, rows_hash


def _camera_records(branch: Any) -> Tuple[Dict[str, Tuple[int, Any, Dict[str, Any]]], Dict[str, Tuple[int, Any, Dict[str, Any]]]]:
    train_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in OCMC._train_records(branch.pipeline)}
    eval_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in OCMC._eval_records(branch.pipeline)}
    return train_records, eval_records


def _natural_medium_state(model: Any, camera: Any) -> Tuple[Tensor, Tensor, Tensor, int, int, Dict[str, Tensor], Dict[str, Tensor]]:
    raw_full, height, width, features_full = CAM._medium_raw_for_camera(model, camera, force_real_camera_context=True)
    zero_context = torch.zeros(3, device=model.device, dtype=raw_full.dtype)
    raw_base, _, _, features_base = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero_context)
    natural_delta = raw_full - raw_base
    return raw_full, raw_base, natural_delta, int(height), int(width), features_full, features_base


def _natural_eff_delta(natural_delta: Tensor, bundle: Optional[Mapping[str, Any]]) -> Tensor:
    if bundle is None:
        return natural_delta
    return OCMC._apply_projector_to_delta(natural_delta, bundle)


def _effective_raw_from_natural(raw_base: Tensor, natural_delta: Tensor, bundle: Optional[Mapping[str, Any]]) -> Tensor:
    return raw_base + _natural_eff_delta(natural_delta, bundle)


def _metric_rows_from_outputs(
    model: Any,
    out_correct: Mapping[str, Tensor],
    gt_correct: Tensor,
    out_swap: Mapping[str, Tensor],
    gt_swap: Tensor,
    flat: Tensor,
    source_view_id: str,
    swapped_view_id: str,
    condition: str,
    checkpoint_step: int,
    population: str,
) -> Dict[str, Any]:
    pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
    pred_swap = out_swap["pred_image"].reshape(-1, 3).detach().float().cpu()
    err_correct = (pred_correct[flat] - gt_correct[flat]).square().mean(dim=-1)
    err_swap = (pred_swap[flat] - gt_swap[flat]).square().mean(dim=-1)
    delta_e = err_swap - err_correct
    return {
        "branch": "C1",
        "condition": condition,
        "absolute_step": int(checkpoint_step),
        "population": population,
        "source_view_id": source_view_id,
        "swapped_view_id": swapped_view_id,
        "sampled_rays": int(flat.numel()),
        "E_correct_mean": float(err_correct.mean().item()),
        "E_swap_mean": float(err_swap.mean().item()),
        "Delta_E_swap_mean": float(delta_e.mean().item()),
        "Delta_E_swap_median": float(torch.quantile(delta_e.float(), 0.5).item()),
        "fraction_Delta_E_swap_gt_0": float((delta_e > 0).float().mean().item()),
        "rgb_change_mean_abs": float((pred_swap[flat] - pred_correct[flat]).abs().mean().item()),
    }


def _render_condition(
    model: Any,
    camera: Any,
    raw_base: Tensor,
    natural_delta: Tensor,
    bundle: Optional[Mapping[str, Any]],
    height: int,
    width: int,
) -> Dict[str, Tensor]:
    raw_eff = _effective_raw_from_natural(raw_base, natural_delta, bundle)
    return CAM._render_from_raw(model, camera, raw_eff, height, width)


def _natural_projection_rows(
    model: Any,
    checkpoint_step: int,
    train_records: Mapping[str, Tuple[int, Any, Dict[str, Any]]],
    samples: Mapping[str, MI.ViewSample],
    swap_bank_rows: Mapping[str, Sequence[str]],
    fresh_bundle: Mapping[str, Any],
    condition_name: str,
    condition_bundle: Optional[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    utility_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    fresh_scale = fresh_bundle["scale"]
    fresh_vmin = fresh_bundle["v_min"]

    context_bank = {
        view_id: CAM._camera_context_for(model, camera.to(model.device), neutral=False).detach()
        for view_id, (_idx, camera, _batch) in train_records.items()
    }

    for source_view_id, sample in samples.items():
        _idx, camera, batch = train_records[source_view_id]
        camera = camera.to(model.device)
        raw_full, raw_base, natural_delta, height, width, features_full, _features_base = _natural_medium_state(model, camera)
        raw_eff_correct = _effective_raw_from_natural(raw_base, natural_delta, condition_bundle)
        out_correct = CAM._render_from_raw(model, camera, raw_eff_correct, height, width)
        gt_correct = PW._get_gt(model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
        pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
        depth_correct = out_correct["depth"].reshape(-1, 1).detach().float().cpu()
        tau_correct = out_correct["tau_D"].reshape(-1, 3).detach().float().cpu().mean(dim=-1, keepdim=True)

        flat_general = sample.flat_for("GENERAL")
        flat_safe = sample.flat_for("M_SAFE")

        for population, flat in (("GENERAL", flat_general), ("M_SAFE", flat_safe)):
            if flat.numel() == 0:
                continue
            flat_dev = flat.to(model.device)
            eff_delta = raw_eff_correct.reshape(-1, 9)[flat_dev].detach().float().cpu() - raw_base.reshape(-1, 9)[flat_dev].detach().float().cpu()
            nat_delta = natural_delta.reshape(-1, 9)[flat_dev].detach().float().cpu()
            delta_std, weak_std, orth_std, proj = CAM._projection_stats(eff_delta, fresh_scale.double(), fresh_vmin.double())
            raw_rms = _rms(nat_delta)
            eff_rms = _rms(eff_delta)
            supp_rms = _rms(nat_delta - eff_delta)
            capacity_rows.append(
                {
                    "branch": "C1",
                    "condition": condition_name,
                    "absolute_step": int(checkpoint_step),
                    "population": population,
                    "source_view_id": source_view_id,
                    "sampled_rays": int(flat.numel()),
                    "camera_context_x": float(features_full["camera_context"][0, 0].detach().cpu().item()),
                    "camera_context_y": float(features_full["camera_context"][0, 1].detach().cpu().item()),
                    "camera_context_z": float(features_full["camera_context"][0, 2].detach().cpu().item()),
                    "raw_delta_rms_9d": raw_rms,
                    "effective_delta_rms_9d": eff_rms,
                    "suppressed_delta_rms_9d": supp_rms,
                    "suppressed_over_full": supp_rms / max(raw_rms, EPS),
                    "effective_over_full": eff_rms / max(raw_rms, EPS),
                    "camera_residual_std_rms": float(proj["delta_std_rms"]),
                    "weak_std_rms": float(proj["weak_std_rms"]),
                    "orth_std_rms": float(proj["orth_std_rms"]),
                    "weak_energy_fraction_mean": float(proj["weak_energy_fraction_mean"]),
                    "weak_energy_fraction_median": float(proj["weak_energy_fraction_median"]),
                    "weak_projection_over_random_1over9": float(proj["weak_projection_over_random_1over9"]),
                }
            )
            rgb_pred = pred_correct[flat]
            rgb_gt = gt_correct[flat]
            mse = (rgb_pred - rgb_gt).square().mean().item()
            mae = (rgb_pred - rgb_gt).abs().mean().item()
            psnr = float("nan") if mse <= 0 else float(-10.0 * math.log10(max(mse, EPS)))
            rgb_rows.append(
                {
                    "branch": "C1",
                    "condition": condition_name,
                    "absolute_step": int(checkpoint_step),
                    "population": population,
                    "source_view_id": source_view_id,
                    "sampled_rays": int(flat.numel()),
                    "sampled_rgb_mse": float(mse),
                    "sampled_rgb_mae": float(mae),
                    "sampled_rgb_psnr": psnr,
                }
            )

        nat = natural_delta.reshape(-1, 9).detach().float().cpu()
        fresh_proj_nat = _natural_eff_delta(natural_delta, fresh_bundle)
        stale_proj_nat = _natural_eff_delta(natural_delta, condition_bundle)
        action_delta = fresh_proj_nat - stale_proj_nat
        for population, flat in (("GENERAL", flat_general), ("M_SAFE", flat_safe)):
            if flat.numel() == 0:
                continue
            flat_dev = flat.to(model.device)
            flat_cpu = flat.cpu()
            action_flat = action_delta.reshape(-1, 9)[flat_dev].detach().float().cpu()
            fresh_flat = fresh_proj_nat.reshape(-1, 9)[flat_dev].detach().float().cpu()
            nat_flat = nat[flat_cpu]
            action_rows.append(
                {
                    "branch": "C1",
                    "condition": condition_name,
                    "absolute_step": int(checkpoint_step),
                    "population": population,
                    "source_view_id": source_view_id,
                    "sampled_rays": int(flat.numel()),
                    "projector_action_rms_9d": _rms(action_flat),
                    "projector_action_over_full": _rms(action_flat) / max(_rms(nat_flat), EPS),
                    "projector_action_over_fresh_projection": _rms(action_flat) / max(_rms(fresh_flat), EPS),
                    "projector_action_mean_abs_9d": float(action_flat.abs().mean().item()),
                }
            )

        raw_ref_sum = torch.zeros_like(raw_eff_correct.detach())
        ref_count = 0
        for swapped_view_id in swap_bank_rows.get(source_view_id, []):
            swapped_context = context_bank[swapped_view_id]
            raw_swap_full, h2, w2, _features_swap = CAM._medium_raw_for_camera(
                model,
                camera,
                camera_context_override=swapped_context,
            )
            zero_context = torch.zeros(3, device=model.device, dtype=raw_swap_full.dtype)
            raw_swap_base, _, _, _ = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero_context)
            swapped_delta = raw_swap_full - raw_swap_base
            raw_swap_eff = _effective_raw_from_natural(raw_swap_base, swapped_delta, condition_bundle)
            out_swap = CAM._render_from_raw(model, camera, raw_swap_eff, h2, w2)
            gt_swap = PW._get_gt(model, batch, out_swap["background"]).reshape(-1, 3).detach().float().cpu()
            pred_swap = out_swap["pred_image"].reshape(-1, 3).detach().float().cpu()
            raw_ref_sum += raw_swap_eff.detach()
            ref_count += 1

            for population, flat in (("GENERAL", flat_general), ("M_SAFE", flat_safe)):
                if flat.numel() == 0:
                    continue
                utility_rows.append(
                    _metric_rows_from_outputs(
                        model,
                        out_correct,
                        gt_correct,
                        out_swap,
                        gt_swap,
                        flat,
                        source_view_id,
                        swapped_view_id,
                        condition_name,
                        checkpoint_step,
                        population,
                    )
                )

        raw_ref = raw_ref_sum / max(ref_count, 1)
        out_ref = CAM._render_from_raw(model, camera, raw_ref, height, width)
        pred_ref = out_ref["pred_image"].reshape(-1, 3).detach().float().cpu()

        for population, flat in (("GENERAL", flat_general), ("M_SAFE", flat_safe)):
            if flat.numel() == 0:
                continue
            flat_dev = flat.to(model.device)
            flat_cpu = flat.cpu()
            raw_delta_swap = raw_eff_correct.reshape(-1, 9)[flat_dev].detach().float().cpu() - raw_ref.reshape(-1, 9)[flat_dev].detach().float().cpu()
            delta_std, weak_std, orth_std, proj = CAM._projection_stats(raw_delta_swap, fresh_bundle["scale"].double(), fresh_bundle["v_min"].double())
            target_rms = float(torch.sqrt(delta_std.square().mean()).item())
            weak_rms = float(torch.sqrt(weak_std.square().mean()).item())
            orth_rms = float(torch.sqrt(orth_std.square().mean()).item())
            weak_gain = target_rms / max(weak_rms, EPS)
            orth_gain = target_rms / max(orth_rms, EPS)
            raw_ref_flat = raw_ref.reshape(-1, 9).detach().clone()
            weak_map = raw_ref_flat.clone()
            orth_map = raw_ref_flat.clone()
            removed_map = raw_eff_correct.reshape(-1, 9).detach().clone()
            map_dtype = raw_ref_flat.dtype
            scale_dev = fresh_bundle["scale"].to(device=model.device, dtype=map_dtype).reshape(1, 9)
            weak_component = weak_std.to(device=model.device, dtype=map_dtype) * scale_dev
            orth_component = orth_std.to(device=model.device, dtype=map_dtype) * scale_dev
            weak_map[flat_cpu] = raw_ref_flat[flat_cpu] + weak_component * weak_gain
            orth_map[flat_cpu] = raw_ref_flat[flat_cpu] + orth_component * orth_gain
            removed_map[flat_cpu] = raw_ref_flat[flat_cpu] + orth_component
            weak_map = weak_map.view(height, width, 9)
            orth_map = orth_map.view(height, width, 9)
            removed_map = removed_map.view(height, width, 9)
            out_weak = CAM._render_from_raw(model, camera, weak_map, height, width)
            out_orth = CAM._render_from_raw(model, camera, orth_map, height, width)
            out_removed = CAM._render_from_raw(model, camera, removed_map, height, width)
            weak_pred = out_weak["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
            orth_pred = out_orth["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
            removed_pred = out_removed["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
            ref_pred = pred_ref[flat]
            correct_pred = pred_correct[flat]
            target = gt_correct[flat]
            err_correct = (correct_pred - target).square().mean(dim=-1)
            err_removed = (removed_pred - target).square().mean(dim=-1)
            removal_rows.append(
                {
                    "branch": "C1",
                    "condition": condition_name,
                    "absolute_step": int(checkpoint_step),
                    "population": population,
                    "source_view_id": source_view_id,
                    "reference": "mean_of_fixed_swap_bank",
                    "sampled_rays": int(flat.numel()),
                    "E_correct_mean": float(err_correct.mean().item()),
                    "E_remove_weak_mean": float(err_removed.mean().item()),
                    "Delta_E_remove_weak_mean": float((err_removed - err_correct).mean().item()),
                    "Delta_E_remove_weak_median": float(torch.quantile((err_removed - err_correct).float(), 0.5).item()),
                    "fraction_remove_weak_improves_or_equal": float((err_removed <= err_correct).float().mean().item()),
                    "target_delta_std_rms": target_rms,
                    "weak_component_gain_for_matched_rms": weak_gain,
                    "orth_component_gain_for_matched_rms": orth_gain,
                    "weak_component_rgb_change_mean_abs": float((weak_pred - ref_pred).abs().mean().item()),
                    "orth_component_rgb_change_mean_abs": float((orth_pred - ref_pred).abs().mean().item()),
                    "weak_over_orth_rgb_change": float((weak_pred - ref_pred).abs().mean().item()) / max(float((orth_pred - ref_pred).abs().mean().item()), EPS),
                    "camera_residual_std_rms": float(delta_std.float().square().mean().sqrt().item()),
                    "weak_energy_fraction_mean": float(((delta_std.float() @ fresh_vmin.float()).square() / delta_std.float().square().sum(dim=-1).clamp_min(EPS)).mean().item()),
                }
            )

            for basis_name, values in (
                ("depth", depth_correct[flat, 0]),
                ("tau", tau_correct[flat, 0]),
            ):
                q1 = torch.quantile(values.float(), 1.0 / 3.0)
                q2 = torch.quantile(values.float(), 2.0 / 3.0)
                for stratum, mask in (
                    ("near_or_low", values <= q1),
                    ("middle", (values > q1) & (values <= q2)),
                    ("far_or_high", values > q2),
                ):
                    if int(mask.sum().item()) == 0:
                        continue
                    strata_rows.append(
                        {
                            "branch": "C1",
                            "condition": condition_name,
                            "absolute_step": int(checkpoint_step),
                            "population": population,
                            "source_view_id": source_view_id,
                            "stratification_basis": basis_name,
                            "stratum": stratum,
                            "sampled_rays": int(mask.sum().item()),
                            "utility_mean": float((err_removed - err_correct)[mask].mean().item()),
                            "rgb_mse_mean": float(err_correct[mask].mean().item()),
                            "weak_energy_fraction_mean": float(((delta_std.float()[mask] @ fresh_vmin.float()).square() / delta_std.float()[mask].square().sum(dim=-1).clamp_min(EPS)).mean().item()),
                            "weak_component_rgb_change_mean_abs": float((weak_pred[mask] - ref_pred[mask]).abs().mean().item()),
                            "orth_component_rgb_change_mean_abs": float((orth_pred[mask] - ref_pred[mask]).abs().mean().item()),
                            "Delta_E_remove_weak_mean": float((err_removed[mask] - err_correct[mask]).mean().item()),
                        }
                    )

    return utility_rows, capacity_rows, rgb_rows, action_rows, removal_rows, strata_rows


def _aggregate_utility_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["absolute_step"]), str(row["population"]), str(row["source_view_id"]))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (condition, step, population, source_view_id), items in sorted(grouped.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "condition": condition,
                "absolute_step": step,
                "population": population,
                "source_view_id": source_view_id,
                "pair_count": len(items),
                "Delta_E_swap_mean": _mean(items, "Delta_E_swap_mean"),
                "Delta_E_swap_median": _median(items, "Delta_E_swap_mean"),
                "fraction_Delta_E_swap_gt_0": _mean(items, "fraction_Delta_E_swap_gt_0"),
                "E_correct_mean": _mean(items, "E_correct_mean"),
                "E_swap_mean": _mean(items, "E_swap_mean"),
            }
        )
    return out


def _aggregate_condition_step(rows: Sequence[Mapping[str, Any]], value_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["absolute_step"]), str(row["population"]))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (condition, step, population), items in sorted(grouped.items(), key=lambda kv: kv[0]):
        item = {
            "condition": condition,
            "absolute_step": step,
            "population": population,
            "source_count": len({str(row["source_view_id"]) for row in items}),
            "sampled_rays": int(sum(int(row.get("sampled_rays", 0)) for row in items)),
        }
        for key in value_keys:
            item[key] = _mean(items, key)
        out.append(item)
    return out


def _aggregate_per_camera_14999(rows: Sequence[Mapping[str, Any]], rgb_rows: Sequence[Mapping[str, Any]], removal_rows: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[Mapping[str, Any]]]] = {}
    for row in rows:
        if int(row["absolute_step"]) != FINAL_STEP:
            continue
        key = (str(row["source_view_id"]), str(row["population"]))
        grouped.setdefault(key, {"utility": [], "capacity": [], "removal": [], "action": [], "rgb": []})["utility"].append(row)
    for row in rgb_rows:
        if int(row["absolute_step"]) != FINAL_STEP:
            continue
        key = (str(row["source_view_id"]), str(row["population"]))
        grouped.setdefault(key, {"utility": [], "capacity": [], "removal": [], "action": [], "rgb": []})["rgb"].append(row)
    for row in removal_rows:
        if int(row["absolute_step"]) != FINAL_STEP:
            continue
        key = (str(row["source_view_id"]), str(row["population"]))
        grouped.setdefault(key, {"utility": [], "capacity": [], "removal": [], "action": [], "rgb": []})["removal"].append(row)
    for row in action_rows:
        if int(row["absolute_step"]) != FINAL_STEP:
            continue
        key = (str(row["source_view_id"]), str(row["population"]))
        grouped.setdefault(key, {"utility": [], "capacity": [], "removal": [], "action": [], "rgb": []})["action"].append(row)

    out: List[Dict[str, Any]] = []
    for (source_view_id, population), groups in sorted(grouped.items(), key=lambda kv: kv[0]):
        item = {
            "absolute_step": FINAL_STEP,
            "source_view_id": source_view_id,
            "population": population,
            "utility_identity_mean": float("nan"),
            "utility_stale_mean": float("nan"),
            "utility_fresh_mean": float("nan"),
            "weak_energy_identity_mean": float("nan"),
            "weak_energy_stale_mean": float("nan"),
            "weak_energy_fresh_mean": float("nan"),
            "camera_residual_identity_rms_9d": float("nan"),
            "camera_residual_stale_rms_9d": float("nan"),
            "camera_residual_fresh_rms_9d": float("nan"),
            "projector_action_rms_9d": float("nan"),
            "projector_action_over_full": float("nan"),
            "projector_action_over_fresh_projection": float("nan"),
            "train_rgb_mse_identity": float("nan"),
            "train_rgb_mse_stale": float("nan"),
            "train_rgb_mse_fresh": float("nan"),
            "train_rgb_mae_identity": float("nan"),
            "train_rgb_mae_stale": float("nan"),
            "train_rgb_mae_fresh": float("nan"),
            "train_rgb_psnr_identity": float("nan"),
            "train_rgb_psnr_stale": float("nan"),
            "train_rgb_psnr_fresh": float("nan"),
            "remove_weak_delta_identity": float("nan"),
            "remove_weak_delta_stale": float("nan"),
            "remove_weak_delta_fresh": float("nan"),
            "remove_weak_improves_identity": float("nan"),
            "remove_weak_improves_stale": float("nan"),
            "remove_weak_improves_fresh": float("nan"),
        }
        for row in groups["utility"]:
            if str(row["condition"]) == "IDENTITY":
                item["utility_identity_mean"] = float(row["Delta_E_swap_mean"])
            elif str(row["condition"]) == "STALE":
                item["utility_stale_mean"] = float(row["Delta_E_swap_mean"])
            elif str(row["condition"]) == "FRESH":
                item["utility_fresh_mean"] = float(row["Delta_E_swap_mean"])
        for row in groups["capacity"]:
            if str(row["condition"]) == "IDENTITY":
                item["weak_energy_identity_mean"] = float(row["weak_energy_fraction_mean"])
                item["camera_residual_identity_rms_9d"] = float(row["effective_delta_rms_9d"])
            elif str(row["condition"]) == "STALE":
                item["weak_energy_stale_mean"] = float(row["weak_energy_fraction_mean"])
                item["camera_residual_stale_rms_9d"] = float(row["effective_delta_rms_9d"])
            elif str(row["condition"]) == "FRESH":
                item["weak_energy_fresh_mean"] = float(row["weak_energy_fraction_mean"])
                item["camera_residual_fresh_rms_9d"] = float(row["effective_delta_rms_9d"])
        for row in groups["action"]:
            item["projector_action_rms_9d"] = float(row["projector_action_rms_9d"])
            item["projector_action_over_full"] = float(row["projector_action_over_full"])
            item["projector_action_over_fresh_projection"] = float(row["projector_action_over_fresh_projection"])
        for row in groups["rgb"]:
            if str(row["condition"]) == "IDENTITY":
                item["train_rgb_mse_identity"] = float(row["sampled_rgb_mse"])
                item["train_rgb_mae_identity"] = float(row["sampled_rgb_mae"])
                item["train_rgb_psnr_identity"] = float(row["sampled_rgb_psnr"])
            elif str(row["condition"]) == "STALE":
                item["train_rgb_mse_stale"] = float(row["sampled_rgb_mse"])
                item["train_rgb_mae_stale"] = float(row["sampled_rgb_mae"])
                item["train_rgb_psnr_stale"] = float(row["sampled_rgb_psnr"])
            elif str(row["condition"]) == "FRESH":
                item["train_rgb_mse_fresh"] = float(row["sampled_rgb_mse"])
                item["train_rgb_mae_fresh"] = float(row["sampled_rgb_mae"])
                item["train_rgb_psnr_fresh"] = float(row["sampled_rgb_psnr"])
        for row in groups["removal"]:
            if str(row["condition"]) == "IDENTITY":
                item["remove_weak_delta_identity"] = float(row["Delta_E_remove_weak_mean"])
                item["remove_weak_improves_identity"] = float(row["fraction_remove_weak_improves_or_equal"])
            elif str(row["condition"]) == "STALE":
                item["remove_weak_delta_stale"] = float(row["Delta_E_remove_weak_mean"])
                item["remove_weak_improves_stale"] = float(row["fraction_remove_weak_improves_or_equal"])
            elif str(row["condition"]) == "FRESH":
                item["remove_weak_delta_fresh"] = float(row["Delta_E_remove_weak_mean"])
                item["remove_weak_improves_fresh"] = float(row["fraction_remove_weak_improves_or_equal"])
        out.append(item)
    return out


def _aggregate_eval_view_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["absolute_step"]) != FINAL_STEP:
            continue
        key = (str(row["condition"]), str(row["view_id"]))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (condition, view_id), items in sorted(grouped.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "absolute_step": FINAL_STEP,
                "condition": condition,
                "view_id": view_id,
                "PSNR": _mean(items, "PSNR"),
                "SSIM": _mean(items, "SSIM"),
                "LPIPS": _mean(items, "LPIPS"),
                "MSE": _mean(items, "MSE"),
            }
        )
    return out


def _condition_eval_metrics(
    model: Any,
    train_records: Mapping[str, Tuple[int, Any, Dict[str, Any]]],
    eval_records: Mapping[str, Tuple[int, Any, Dict[str, Any]]],
    conditions: Mapping[str, Optional[Mapping[str, Any]]],
    step: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for condition_name, bundle in conditions.items():
        for view_id, (_idx, camera, batch) in eval_records.items():
            camera = camera.to(model.device)
            raw_full, raw_base, natural_delta, height, width, _features_full, _features_base = _natural_medium_state(model, camera)
            raw_eff = _effective_raw_from_natural(raw_base, natural_delta, bundle)
            out = CAM._render_from_raw(model, camera, raw_eff, height, width)
            gt = PW._get_gt(model, batch, out["background"]).detach().float()
            metrics = MIC._metric_images(model, out["pred_image"], gt)
            rows.append(
                {
                    "branch": "C1",
                    "absolute_step": int(step),
                    "condition": condition_name,
                    "view_id": view_id,
                    **metrics,
                }
            )
    return rows


def _condition_summary(
    condition_rows: Sequence[Mapping[str, Any]],
    rgb_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    removal_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    condition: str,
) -> Dict[str, Any]:
    rows = [row for row in condition_rows if str(row["condition"]) == condition]
    rgb = [row for row in rgb_rows if str(row["condition"]) == condition]
    action = [row for row in action_rows if str(row["condition"]) == condition]
    removal = [row for row in removal_rows if str(row["condition"]) == condition]
    eval_cond = [row for row in eval_rows if str(row["condition"]) == condition]
    return {
        "condition": condition,
        "utility_mean": _mean(rows, "Delta_E_swap_mean"),
        "utility_median": _median(rows, "Delta_E_swap_mean"),
        "utility_fraction_gt_0": _mean(rows, "fraction_Delta_E_swap_gt_0"),
        "rgb_mse_mean": _mean(rgb, "sampled_rgb_mse"),
        "rgb_mae_mean": _mean(rgb, "sampled_rgb_mae"),
        "rgb_psnr_mean": _mean(rgb, "sampled_rgb_psnr"),
        "weak_energy_fraction_mean": _mean([row for row in removal if "weak_energy_fraction_mean" in row], "weak_energy_fraction_mean"),
        "projector_action_rms_9d_mean": _mean(action, "projector_action_rms_9d"),
        "projector_action_over_full_mean": _mean(action, "projector_action_over_full"),
        "projector_action_over_fresh_projection_mean": _mean(action, "projector_action_over_fresh_projection"),
        "remove_weak_delta_mean": _mean(removal, "Delta_E_remove_weak_mean"),
        "remove_weak_fraction_improves_mean": _mean(removal, "fraction_remove_weak_improves_or_equal"),
        "eval_PSNR_mean": _mean(eval_cond, "PSNR"),
        "eval_SSIM_mean": _mean(eval_cond, "SSIM"),
        "eval_LPIPS_mean": _mean(eval_cond, "LPIPS"),
        "eval_MSE_mean": _mean(eval_cond, "MSE"),
    }


def _classification_from_summary(summary: Mapping[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    checkpoints = summary["checkpoints"]
    utility_recovery = []
    weak_preservation = []
    rgb_non_worsening = []
    for row in checkpoints:
        if not math.isfinite(float(row["utility_identity_mean"])) or not math.isfinite(float(row["utility_stale_mean"])) or not math.isfinite(float(row["utility_fresh_mean"])):
            continue
        denom = float(row["utility_identity_mean"]) - float(row["utility_stale_mean"]) + EPS
        utility_recovery.append((float(row["utility_fresh_mean"]) - float(row["utility_stale_mean"])) / denom)
        stale_weak = float(row["weak_energy_stale_mean"])
        fresh_weak = float(row["weak_energy_fresh_mean"])
        if math.isfinite(stale_weak) and stale_weak != 0.0:
            weak_preservation.append(fresh_weak / stale_weak)
        rgb_non_worsening.append(float(row["train_rgb_mse_fresh"]) <= float(row["train_rgb_mse_stale"]) + 1e-6)

    mean_recovery = float(sum(utility_recovery) / len(utility_recovery)) if utility_recovery else float("nan")
    mean_weak_ratio = float(sum(weak_preservation) / len(weak_preservation)) if weak_preservation else float("nan")
    eval_delta = summary["final_eval_delta"]
    final_eval_improved = bool(float(eval_delta["PSNR"]) >= 0.0 and float(eval_delta["LPIPS"]) <= 0.0)
    stale_worse_than_identity = summary["final_condition_gap"]["utility_identity_minus_stale"] > 0.0
    fresh_better_than_stale = summary["final_condition_gap"]["fresh_minus_stale"] > 0.0
    fresh_close_to_stale = abs(summary["final_condition_gap"]["fresh_minus_stale"]) <= abs(summary["final_condition_gap"]["identity_minus_stale"]) * 0.5
    weak_preserved = math.isfinite(mean_weak_ratio) and mean_weak_ratio >= 0.8
    utility_recovers = math.isfinite(mean_recovery) and mean_recovery >= 0.35
    utility_partially_recovers = math.isfinite(mean_recovery) and mean_recovery >= 0.15
    rgb_safe = all(rgb_non_worsening) and final_eval_improved

    if utility_recovers and fresh_better_than_stale and weak_preserved and rgb_safe:
        classification = "TEMPORAL_PROJECTOR_MISMATCH_SUPPORTED"
        secondary = "OCMC_PROJECTION_PRINCIPLE_SUPPORTED"
    elif stale_worse_than_identity and not fresh_better_than_stale and weak_preserved:
        classification = "OBSERVABILITY_PROJECTION_OVER_SUPPRESSION_SUPPORTED"
        secondary = "OCMC_PROJECTION_PRINCIPLE_TENTATIVE"
    elif utility_partially_recovers and fresh_close_to_stale and weak_preserved:
        classification = "MIXED_TEMPORAL_AND_SUPPRESSION_EFFECT"
        secondary = "OCMC_PROJECTION_PRINCIPLE_TENTATIVE"
    else:
        classification = "PROJECTOR_FAILURE_CAUSE_NOT_RESOLVED"
        secondary = "OCMC_PROJECTION_PRINCIPLE_NOT_SUPPORTED"

    evidence = {
        "mean_recovery_ratio": mean_recovery,
        "mean_weak_preservation_ratio": mean_weak_ratio,
        "final_eval_improved": final_eval_improved,
        "rgb_safe_across_conditions": all(rgb_non_worsening) if rgb_non_worsening else False,
        "stale_worse_than_identity": stale_worse_than_identity,
        "fresh_better_than_stale": fresh_better_than_stale,
        "fresh_close_to_stale": fresh_close_to_stale,
    }
    return classification, secondary, evidence


def _final_summary(
    repo: Path,
    source_output_dir: Path,
    output_dir: Path,
    env: Mapping[str, Any],
    gpu: Mapping[str, Any],
    sample_bank: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    stale_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    capacity_rows: Sequence[Mapping[str, Any]],
    rgb_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    removal_rows: Sequence[Mapping[str, Any]],
    strata_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    per_camera_rows: Sequence[Mapping[str, Any]],
    per_eval_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    condition_summaries = {
        condition: _condition_summary(utility_rows, rgb_rows, action_rows, removal_rows, eval_rows, condition)
        for condition in ("IDENTITY", "STALE", "FRESH")
    }
    final_condition_gap = {
        "identity_minus_stale": float(condition_summaries["IDENTITY"]["utility_mean"]) - float(condition_summaries["STALE"]["utility_mean"]),
        "identity_minus_fresh": float(condition_summaries["IDENTITY"]["utility_mean"]) - float(condition_summaries["FRESH"]["utility_mean"]),
        "fresh_minus_stale": float(condition_summaries["FRESH"]["utility_mean"]) - float(condition_summaries["STALE"]["utility_mean"]),
        "utility_identity_minus_stale": float(condition_summaries["IDENTITY"]["utility_mean"]) - float(condition_summaries["STALE"]["utility_mean"]),
        "utility_identity_minus_fresh": float(condition_summaries["IDENTITY"]["utility_mean"]) - float(condition_summaries["FRESH"]["utility_mean"]),
        "utility_fresh_minus_stale": float(condition_summaries["FRESH"]["utility_mean"]) - float(condition_summaries["STALE"]["utility_mean"]),
    }
    final_eval = {
        "PSNR": _mean([row for row in per_eval_rows if int(row["absolute_step"]) == FINAL_STEP and str(row["condition"]) == "FRESH"], "PSNR"),
        "SSIM": _mean([row for row in per_eval_rows if int(row["absolute_step"]) == FINAL_STEP and str(row["condition"]) == "FRESH"], "SSIM"),
        "LPIPS": _mean([row for row in per_eval_rows if int(row["absolute_step"]) == FINAL_STEP and str(row["condition"]) == "FRESH"], "LPIPS"),
        "MSE": _mean([row for row in per_eval_rows if int(row["absolute_step"]) == FINAL_STEP and str(row["condition"]) == "FRESH"], "MSE"),
    }
    stale_fresh_rows = [row for row in temporal_rows if int(row["requested_step"]) in PRIMARY_STEPS]
    classification, secondary, evidence = _classification_from_summary(
        {
            "checkpoints": [row for row in per_camera_rows if int(row["absolute_step"]) == FINAL_STEP],
            "final_eval_delta": {
                "PSNR": float(condition_summaries["FRESH"]["eval_PSNR_mean"]) - float(condition_summaries["STALE"]["eval_PSNR_mean"]),
                "SSIM": float(condition_summaries["FRESH"]["eval_SSIM_mean"]) - float(condition_summaries["STALE"]["eval_SSIM_mean"]),
                "LPIPS": float(condition_summaries["FRESH"]["eval_LPIPS_mean"]) - float(condition_summaries["STALE"]["eval_LPIPS_mean"]),
                "MSE": float(condition_summaries["FRESH"]["eval_MSE_mean"]) - float(condition_summaries["STALE"]["eval_MSE_mean"]),
            },
            "final_condition_gap": final_condition_gap,
        }
    )
    summary = {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "repo": str(repo),
        "source_output_dir": str(source_output_dir),
        "output_dir": str(output_dir),
        "CONDA_ENV": env["CONDA_ENV"],
        "PYTHON_PATH": env["PYTHON_PATH"],
        "TORCH_VERSION": env["TORCH_VERSION"],
        "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "gpu": dict(gpu),
        "sample_bank": dict(sample_bank),
        "checkpoint_rows": list(checkpoint_rows),
        "stale_projector_rows": list(stale_rows),
        "fresh_projector_rows": list(fresh_rows),
        "projector_temporal_difference_rows": list(temporal_rows),
        "utility_summary": condition_summaries,
        "final_condition_gap": final_condition_gap,
        "final_eval_delta": {
            "PSNR": float(condition_summaries["FRESH"]["eval_PSNR_mean"]) - float(condition_summaries["STALE"]["eval_PSNR_mean"]),
            "SSIM": float(condition_summaries["FRESH"]["eval_SSIM_mean"]) - float(condition_summaries["STALE"]["eval_SSIM_mean"]),
            "LPIPS": float(condition_summaries["FRESH"]["eval_LPIPS_mean"]) - float(condition_summaries["STALE"]["eval_LPIPS_mean"]),
            "MSE": float(condition_summaries["FRESH"]["eval_MSE_mean"]) - float(condition_summaries["STALE"]["eval_MSE_mean"]),
        },
        "per_camera_14999_count": len([row for row in per_camera_rows if int(row["absolute_step"]) == FINAL_STEP]),
        "per_eval_view_14999_count": len([row for row in per_eval_rows if int(row["absolute_step"]) == FINAL_STEP]),
        "classification_candidate": classification,
        "secondary_classification_candidate": secondary,
        "classification_evidence": evidence,
    }
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, (dict, list))])
    _write_json(output_dir / "final_classification.json", {
        "classification": classification,
        "secondary_classification": secondary,
        "evidence": evidence,
        "final_condition_gap": final_condition_gap,
        "final_eval_delta": summary["final_eval_delta"],
    })
    return summary


def _write_note(repo: Path, output_dir: Path, summary: Mapping[str, Any]) -> None:
    utility = summary["utility_summary"]
    lines = [
        "# OCMC Projector Temporal Mismatch Audit",
        "",
        "## CODE FACT",
        "OCMC remains the detached 9-D camera-conditioned medium projector implemented in `water_splatting/fields/medium_field.py` and installed through `water_splatting/water_splatting.py`.",
        "This task does not alter the projector equation or the medium MLP.",
        "",
        "## CONFIG FACT",
        f"Diagnostic checkpoints: `{PRIMARY_STEPS}`.",
        f"Projector population: `{PROJECTOR_POPULATION}` with the registered GENERAL train-ray bank.",
        "Identity means projector disabled at forward time. Stale means the checkpoint-saved bundle. Fresh means the projector recomputed from the current checkpoint state with the same estimator.",
        "",
        "## EXPERIMENTAL FACT",
        f"Outputs: `{output_dir}`.",
        f"Checkpoint mapping rows: `{len(summary['checkpoint_rows'])}`.",
        f"Per-camera final rows: `{summary['per_camera_14999_count']}`.",
        f"Per-eval-view final rows: `{summary['per_eval_view_14999_count']}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"Identity utility mean: `{utility['IDENTITY']['utility_mean']:.6f}`.",
        f"Stale utility mean: `{utility['STALE']['utility_mean']:.6f}`.",
        f"Fresh utility mean: `{utility['FRESH']['utility_mean']:.6f}`.",
        f"Fresh-vs-stale utility gap: `{summary['final_condition_gap']['fresh_minus_stale']:.6f}`.",
        f"Final eval fresh-vs-stale delta: `{summary['final_eval_delta']}`.",
        "",
        "## INFERENCE",
        f"Candidate classification: `{summary['classification_candidate']}`.",
        f"Secondary classification: `{summary['secondary_classification_candidate']}`.",
        "No training, optimizer step, or projector redesign was performed.",
        "",
    ]
    path = repo / RESEARCH_NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _prepare_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise RuntimeError(f"Output directory exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _checkpoint_bundle_summary(
    repo: Path,
    output_dir: Path,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    stale_bundles: Sequence[Mapping[str, Any]],
    fresh_bundles: Sequence[Mapping[str, Any]],
    sample_hash: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    checkpoint_mapping_rows: List[Dict[str, Any]] = []
    stale_rows: List[Dict[str, Any]] = []
    fresh_rows: List[Dict[str, Any]] = []
    for checkpoint_row, stale_bundle, fresh_bundle in zip(checkpoint_rows, stale_bundles, fresh_bundles):
        requested_step = int(checkpoint_row["requested_step"])
        checkpoint_mapping_rows.append(
            {
                "requested_step": requested_step,
                "actual_step": int(checkpoint_row["actual_step"]),
                "checkpoint_path": checkpoint_row["checkpoint_path"],
                "checkpoint_exists": bool(checkpoint_row["checkpoint_exists"]),
                "ocmc_projector_refresh_step": checkpoint_row["ocmc_projector_refresh_step"],
                "stale_bundle_step": int(stale_bundle["step"]),
                "fresh_bundle_step": int(fresh_bundle["step"]),
            }
        )
        stale_summary = {
            "requested_step": requested_step,
            "checkpoint_path": checkpoint_row["checkpoint_path"],
            "stale_refresh_step": int(stale_bundle["step"]),
            "source": str(stale_bundle.get("source", "")),
            "population": str(stale_bundle.get("population", "")),
            "sample_hash": sample_hash,
        }
        stale_summary.update(OCMC._bundle_summary_row(stale_bundle))
        stale_rows.append(stale_summary)
        fresh_summary = {
            "requested_step": requested_step,
            "checkpoint_path": checkpoint_row["checkpoint_path"],
            "loaded_step": int(fresh_bundle["step"]),
            "source": str(fresh_bundle.get("source", "")),
            "population": str(fresh_bundle.get("population", "")),
            "sample_hash": sample_hash,
            "analysis_loaded_step": int(fresh_bundle["step"]),
        }
        fresh_summary.update(OCMC._bundle_summary_row(fresh_bundle))
        fresh_rows.append(fresh_summary)
    _write_json(output_dir / "checkpoint_mapping.json", {"rows": checkpoint_mapping_rows})
    _write_csv(output_dir / "checkpoint_mapping.csv", checkpoint_mapping_rows)
    _write_json(output_dir / "stale_projector_provenance.json", {"rows": stale_rows})
    _write_csv(output_dir / "stale_projector_provenance.csv", stale_rows)
    _write_json(output_dir / "fresh_projector_summary.json", {"rows": fresh_rows})
    _write_csv(output_dir / "fresh_projector_summary.csv", fresh_rows)
    return checkpoint_mapping_rows, stale_rows, fresh_rows


def _projector_temporal_difference_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    stale_bundles: Sequence[Mapping[str, Any]],
    fresh_bundles: Sequence[Mapping[str, Any]],
    sample_hash: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for checkpoint_row, stale_bundle, fresh_bundle in zip(checkpoint_rows, stale_bundles, fresh_bundles):
        pair = OCMC._projector_pair_row(stale_bundle, fresh_bundle)
        pair.update(
            {
                "requested_step": int(checkpoint_row["requested_step"]),
                "stale_refresh_step": int(stale_bundle["step"]),
                "fresh_loaded_step": int(fresh_bundle["step"]),
                "sample_hash": sample_hash,
                "stale_trace": float(stale_bundle["trace"]),
                "fresh_trace": float(fresh_bundle["trace"]),
            }
        )
        rows.append(pair)
    return rows


def _load_branch_for_checkpoint(repo: Path, source_output_dir: Path, requested_step: int) -> Tuple[Any, Mapping[str, Any]]:
    branch = OCMC._setup_branch(repo, "C1")
    bundle = OCMC._load_snapshot(branch, source_output_dir, int(requested_step))
    return branch, bundle


def run(repo: Path, source_output_dir: Path, output_dir: Path, allow_existing_output: bool) -> Dict[str, Any]:
    gpu = _assert_runtime_policy()
    repo = repo.resolve()
    source_output_dir = _repo_path(source_output_dir, repo)
    output_dir = _repo_path(output_dir, repo)
    _prepare_output_dir(output_dir, allow_existing_output)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = _environment_manifest(gpu)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "repo_manifest.json", _repo_manifest(repo))
    _write_json(output_dir / "source_output_manifest.json", {"source_output_dir": str(source_output_dir), "exists": bool(source_output_dir.exists())})

    print("[OCMC-AUDIT] building deterministic sample bank", flush=True)
    samples, sample_meta, sample_rows, sample_hash = _build_samples(repo, output_dir)
    train_view_ids = list(samples.keys())
    swap_bank = _load_swap_bank(repo, source_output_dir, train_view_ids)
    _write_json(output_dir / "swap_bank_manifest.json", swap_bank)

    checkpoint_rows = _load_checkpoint_manifest(repo, source_output_dir, PRIMARY_STEPS)
    if len(checkpoint_rows) != len(PRIMARY_STEPS):
        missing = sorted(set(PRIMARY_STEPS) - {int(row["requested_step"]) for row in checkpoint_rows})
        raise FileNotFoundError(f"Missing C1 checkpoints for requested steps: {missing}")

    stale_bundles: List[Mapping[str, Any]] = []
    fresh_bundles: List[Mapping[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    final_model: Optional[Any] = None
    final_train_records: Optional[Mapping[str, Tuple[int, Any, Dict[str, Any]]]] = None
    final_eval_records: Optional[Mapping[str, Tuple[int, Any, Dict[str, Any]]]] = None
    final_stale_bundle: Optional[Mapping[str, Any]] = None
    final_fresh_bundle: Optional[Mapping[str, Any]] = None

    for checkpoint_row in checkpoint_rows:
        step = int(checkpoint_row["requested_step"])
        print(f"[OCMC-AUDIT] loading checkpoint step={step}", flush=True)
        branch, stale_bundle = _load_branch_for_checkpoint(repo, source_output_dir, step)
        try:
            train_records, eval_records = _camera_records(branch)
            model = branch.pipeline.model
            analyses, meta = CAM._analyse_loaded_branch(branch, samples)
            fresh_analysis = analyses[PROJECTOR_POPULATION]
            fresh_bundle = OCMC._projector_bundle_from_analysis(
                fresh_analysis,
                branch="C1",
                step=int(step),
                population=PROJECTOR_POPULATION,
                source="fresh_current_checkpoint_state",
            )
            stale_bundle = OCMC._projector_bundle_for_checkpoint(stale_bundle)
            fresh_bundle = OCMC._projector_bundle_for_checkpoint(fresh_bundle)
            stale_bundles.append(stale_bundle)
            fresh_bundles.append(fresh_bundle)

            temporal_rows.extend(_projector_temporal_difference_rows([checkpoint_row], [stale_bundle], [fresh_bundle], sample_hash))

            bundles = {"IDENTITY": None, "STALE": stale_bundle, "FRESH": fresh_bundle}
            for condition_name in ("IDENTITY", "STALE", "FRESH"):
                util_rows, cap_rows, rgb_cond_rows, action_cond_rows, removal_cond_rows, strata_cond_rows = _natural_projection_rows(
                    model,
                    step,
                    train_records,
                    samples,
                    swap_bank["rows"],
                    fresh_bundle,
                    condition_name,
                    bundles[condition_name],
                )
                utility_rows.extend(util_rows)
                capacity_rows.extend(cap_rows)
                rgb_rows.extend(rgb_cond_rows)
                action_rows.extend(action_cond_rows)
                removal_rows.extend(removal_cond_rows)
                strata_rows.extend(strata_cond_rows)
            if step == FINAL_STEP:
                final_model = model
                final_train_records = train_records
                final_eval_records = eval_records
                final_stale_bundle = stale_bundle
                final_fresh_bundle = fresh_bundle
        finally:
            OCMC._release(branch)

    checkpoint_mapping_rows, stale_rows, fresh_rows = _checkpoint_bundle_summary(
        repo,
        output_dir,
        checkpoint_rows,
        stale_bundles,
        fresh_bundles,
        sample_hash,
    )
    utility_summary_rows = _aggregate_utility_rows(utility_rows)
    weak_capacity_summary_rows = _aggregate_condition_step(
        [
            {
                **row,
                "weak_energy_fraction_mean": float(row["weak_energy_fraction_mean"]),
            }
            for row in capacity_rows
        ],
        ("weak_energy_fraction_mean", "weak_projection_over_random_1over9", "suppressed_over_full", "effective_over_full", "camera_residual_std_rms"),
    )
    rgb_summary_rows = _aggregate_condition_step(rgb_rows, ("sampled_rgb_mse", "sampled_rgb_mae", "sampled_rgb_psnr"))
    removal_summary_rows = _aggregate_condition_step(removal_rows, ("Delta_E_remove_weak_mean", "fraction_remove_weak_improves_or_equal", "weak_component_rgb_change_mean_abs", "orth_component_rgb_change_mean_abs"))
    action_summary_rows = _aggregate_condition_step(action_rows, ("projector_action_rms_9d", "projector_action_over_full", "projector_action_over_fresh_projection"))

    if final_model is None or final_train_records is None or final_eval_records is None or final_stale_bundle is None or final_fresh_bundle is None:
        raise RuntimeError("Missing final checkpoint state for eval audit.")
    eval_rows = _condition_eval_metrics(
        final_model,
        final_train_records,
        final_eval_records,
        {"IDENTITY": None, "STALE": final_stale_bundle, "FRESH": final_fresh_bundle},
        FINAL_STEP,
    )
    per_eval_rows = _aggregate_eval_view_summary(eval_rows)
    per_camera_rows = _aggregate_per_camera_14999(utility_rows, rgb_rows, removal_rows, action_rows)

    # Write raw outputs.
    _write_csv(output_dir / "projector_temporal_difference.csv", temporal_rows)
    _write_json(output_dir / "projector_temporal_difference.json", {"rows": temporal_rows})
    _write_csv(output_dir / "correct_context_utility.csv", utility_rows)
    _write_json(output_dir / "correct_context_utility.json", {"rows": utility_rows})
    _write_csv(output_dir / "correct_context_utility_summary.csv", utility_summary_rows)
    _write_json(output_dir / "correct_context_utility_summary.json", {"rows": utility_summary_rows})
    _write_csv(output_dir / "train_rgb_error.csv", rgb_rows)
    _write_json(output_dir / "train_rgb_error.json", {"rows": rgb_rows})
    _write_csv(output_dir / "train_rgb_error_summary.csv", rgb_summary_rows)
    _write_json(output_dir / "train_rgb_error_summary.json", {"rows": rgb_summary_rows})
    _write_csv(output_dir / "weak_capacity.csv", capacity_rows)
    _write_json(output_dir / "weak_capacity.json", {"rows": capacity_rows})
    _write_csv(output_dir / "weak_capacity_summary.csv", weak_capacity_summary_rows)
    _write_json(output_dir / "weak_capacity_summary.json", {"rows": weak_capacity_summary_rows})
    _write_csv(output_dir / "projector_action_difference.csv", action_rows)
    _write_json(output_dir / "projector_action_difference.json", {"rows": action_rows})
    _write_csv(output_dir / "projector_action_difference_summary.csv", action_summary_rows)
    _write_json(output_dir / "projector_action_difference_summary.json", {"rows": action_summary_rows})
    _write_csv(output_dir / "weak_removal_counterfactual.csv", removal_rows)
    _write_json(output_dir / "weak_removal_counterfactual.json", {"rows": removal_rows})
    _write_csv(output_dir / "weak_removal_counterfactual_summary.csv", removal_summary_rows)
    _write_json(output_dir / "weak_removal_counterfactual_summary.json", {"rows": removal_summary_rows})
    _write_csv(output_dir / "depth_tau_strata.csv", strata_rows)
    _write_json(output_dir / "depth_tau_strata.json", {"rows": strata_rows})
    _write_csv(output_dir / "per_camera_summary.csv", per_camera_rows)
    _write_json(output_dir / "per_camera_summary.json", {"rows": per_camera_rows})
    _write_csv(output_dir / "per_eval_view_summary.csv", per_eval_rows)
    _write_json(output_dir / "per_eval_view_summary.json", {"rows": per_eval_rows})

    summary = _final_summary(
        repo,
        source_output_dir,
        output_dir,
        env,
        gpu,
        {"rows_hash": sample_hash, "sample_seed": SAMPLE_SEED, "samples_per_view": SAMPLES_PER_VIEW, "train_view_count": len(train_view_ids), "swap_bank_hash": swap_bank["hash"], "source_swap_bank": swap_bank["source_path"]},
        checkpoint_rows,
        stale_rows,
        fresh_rows,
        temporal_rows,
        utility_rows,
        capacity_rows,
        rgb_rows,
        action_rows,
        removal_rows,
        strata_rows,
        eval_rows,
        per_camera_rows,
        per_eval_rows,
    )
    _write_note(repo, output_dir, summary)
    return summary


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-output-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.repo, args.source_output_dir, args.output_dir, bool(args.allow_existing_output))
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
