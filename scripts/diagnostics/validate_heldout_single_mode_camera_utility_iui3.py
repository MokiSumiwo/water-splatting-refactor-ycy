#!/usr/bin/env python3
"""Read-only Phase A for held-out single-mode camera utility validation on IUI3.

This diagnostic freezes one bottom-3 GENERAL mode from the existing 14999 mode
wise audit, reconstructs a new deterministic held-out ray bank, and evaluates
the selected mode under the current C1 checkpoint without any training or
checkpoint mutation.
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

from nerfstudio.cameras.cameras import Cameras

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_bnd_pw_iui3 as PW
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_ocmc_causal_iui3 as OCMC


EXPERIMENT = "HELDOUT-SINGLE-MODE-UTILITY-VALIDATION"
SCENE = "IUI3-RedSea"
SELECTION_OUTPUT_DIR = Path("outputs/modewise_camera_utility_observability_iui3_20260826")
SOURCE_OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
OUTPUT_DIR = Path("outputs/heldout_single_mode_camera_utility_iui3_20260826")
LOG_DIR = Path("logs/heldout_single_mode_camera_utility_iui3_20260826")
RESEARCH_NOTE = Path("research_notes/HELDOUT_SINGLE_MODE_CAMERA_UTILITY_IUI3_2026-08-26.md")
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
FINAL_STEP = 14999
SAMPLES_PER_VIEW = 1024
SELECTION_SEED = 20260825
HELDOUT_SEED = 20260826
SWAP_SEED = 20260826
SWAP_COUNT = 8
EPS = 1e-12


@dataclass
class BankView:
    view_id: str
    height: int
    width: int
    selection_general: Tensor
    heldout_general: Tensor
    selection_safe: Tensor
    heldout_safe: Tensor


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


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, default=_json_default).encode("utf8"))


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


def _environment_manifest(gpu: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu),
    }


def _source_semantics(selection_dir: Path) -> Dict[str, Any]:
    return {
        "CODE_FACT": True,
        "selection_source_dir": str(selection_dir),
        "medium_context_mode": "dir_xy_camera",
        "medium_mlp_input_dimension": 22,
        "medium_mlp_input_parts": {
            "direction_encoding": 16,
            "xy_context": 3,
            "camera_context": 3,
        },
        "raw_output_channels": {
            "0:3": "B_inf / medium_rgb logits -> sigmoid",
            "3:6": "beta_B / medium_bs logits -> softplus",
            "6:9": "beta_D / medium_attn logits -> softplus",
        },
        "current_effective_raw_source": "OCMC._effective_raw_for_context on the formal C1 checkpoint",
        "selected_mode_removal_source": "frozen 14999 GENERAL modewise audit from the archived mode-removal summary",
        "heldout_general_bank_rule": "25 train cameras; 1024 valid GENERAL rays per camera; new seed 20260826; exclude frozen selection GENERAL pixels.",
        "heldout_swap_bank_rule": "8 deterministic real-camera alternatives per source view from a new seed tied to 20260826.",
        "scale_rule": "recovered from the frozen selection bank via the existing structured medium analysis; no held-out basis refit.",
    }


def _source_semantics_md(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Held-Out Single-Mode Camera Utility Validation",
            "",
            "## CODE FACT",
            "The selected mode comes from the archived 14999 GENERAL mode-wise audit.",
            "The held-out evaluation reuses the current C1 checkpoint and the frozen 14999 basis.",
            "",
            "## CONFIG FACT",
            f"Selection source: `{summary['selection_source_dir']}`.",
            f"Held-out bank seed: `{HELDOUT_SEED}`.",
            f"Swap seed: `{SWAP_SEED}`.",
            "",
            "## INFERENCE",
            "This diagnostic only measures whether the frozen low-observability mode carries reproducible camera-specific utility on disjoint rays.",
        ]
    ) + "\n"


def _sample_flat(mask: Tensor, max_pixels: int, seed: int) -> Tensor:
    flat = torch.nonzero(mask.reshape(-1).detach().bool().cpu(), as_tuple=False).reshape(-1)
    if flat.numel() <= max_pixels:
        return flat.long()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(flat.numel(), generator=generator)[:max_pixels]
    return flat[order].long()


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.std(ddof=0))


def _fraction_positive(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float((arr > 0.0).mean())


def _rms(values: Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.sqrt(values.detach().float().square().mean()).item())


def _load_archived_selection(selection_dir: Path) -> Dict[str, Any]:
    basis = json.loads((selection_dir / "modewise_basis.json").read_text(encoding="utf8"))["rows"]
    removal = json.loads((selection_dir / "mode_removal_summary.json").read_text(encoding="utf8"))["rows"]
    removal_raw = json.loads((selection_dir / "mode_removal_counterfactual.json").read_text(encoding="utf8"))["rows"]
    basis_summary = json.loads((selection_dir / "modewise_basis_summary.json").read_text(encoding="utf8"))["rows"]
    final = json.loads((selection_dir / "final_summary.json").read_text(encoding="utf8"))

    basis_14999 = {
        int(row["canonical_mode_index"]): row
        for row in basis
        if int(row["step"]) == FINAL_STEP and row["population"] == "GENERAL"
    }
    removal_14999 = {
        int(row["canonical_mode_index"]): row
        for row in removal
        if int(row["absolute_step"]) == FINAL_STEP and row["population"] == "GENERAL"
    }
    basis_summary_14999 = next(row for row in basis_summary if int(row["step"]) == FINAL_STEP)
    sigma_values = [float(basis_14999[i]["sigma_per_sqrt_ray"]) for i in range(9)]
    sigma_ranks = sorted(range(9), key=lambda idx: sigma_values[idx])
    median_sigma = float(basis_summary_14999["sigma_median"])

    candidates = []
    for idx in sigma_ranks[:3]:
        sigma = float(basis_14999[idx]["sigma_per_sqrt_ray"])
        g_obs = float(sigma * sigma / (sigma * sigma + median_sigma * median_sigma))
        c_utility = float(removal_14999[idx]["Delta_E_remove_mode_mean"])
        c_rgb = float(removal_14999[idx]["Delta_E_remove_mode_mean"])
        mode_row = basis_14999[idx]
        raw_rows = [
            row
            for row in removal_raw
            if int(row["absolute_step"]) == FINAL_STEP
            and row["population"] == "GENERAL"
            and int(row["canonical_mode_index"]) == idx
        ]
        amplitude_proxy = float("nan")
        if raw_rows:
            energy_weighted = [
                float(row["delta_std_rms"]) ** 2 * float(row["mode_energy_fraction_mean"])
                for row in raw_rows
            ]
            amplitude_proxy = math.sqrt(max(_mean(energy_weighted), 0.0))
        utility_per_energy_proxy = c_utility / max(amplitude_proxy, EPS) if math.isfinite(amplitude_proxy) else float("nan")
        candidates.append(
            {
                "mode_index": idx,
                "mode_label": f"mode_{idx:02d}",
                "sigma_rank": int(basis_14999[idx]["sigma_rank_in_step"]),
                "sigma": sigma,
                "g_obs": g_obs,
                "C_utility_proxy": c_utility,
                "C_rgb_proxy": c_rgb,
                "mode_energy_fraction_mean": float(removal_14999[idx]["mode_energy_fraction_mean"]),
                "mode_projection_over_random_1over9": float(removal_14999[idx]["mode_projection_over_random_1over9"]),
                "natural_mode_amplitude_proxy": amplitude_proxy,
                "utility_per_energy_proxy": utility_per_energy_proxy,
                "mode_vector": [float(mode_row[f"v_{k}"]) for k in range(9)],
                "selection_score": float((1.0 - g_obs) * max(c_utility, 0.0)),
            }
        )

    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    top_score = float(candidates[0]["selection_score"])
    tied = [row for row in candidates if abs(float(row["selection_score"]) - top_score) <= 1e-12]
    tied.sort(
        key=lambda row: (
            float(row["utility_per_energy_proxy"])
            if math.isfinite(float(row["utility_per_energy_proxy"]))
            else float("-inf"),
            -float(row["sigma"]),
        ),
        reverse=True,
    )
    selected = tied[0]

    archived = {
        "archived_final_summary_classification": final.get("classification"),
        "archived_final_summary_pairwise_warranted": final.get("pairwise_warranted"),
        "basis_summary": basis_summary_14999,
        "basis_rows": basis_14999,
        "removal_rows": removal_14999,
        "median_sigma": median_sigma,
        "candidates": candidates,
        "selected_mode": selected,
        "selected_mode_index": int(selected["mode_index"]),
        "selected_mode_label": selected["mode_label"],
        "source_files": {
            name: _sha256_bytes((selection_dir / name).read_bytes())
            for name in (
                "modewise_basis.json",
                "mode_removal_summary.json",
                "mode_removal_counterfactual.json",
                "modewise_basis_summary.json",
                "final_summary.json",
            )
        },
    }
    return archived


def _build_banks(repo: Path) -> Tuple[Dict[str, BankView], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    step3_maps, step3_meta = PW._render_split_maps(repo, "BND", 3000)
    masks, mask_meta = PW._build_masks(step3_maps)
    train_views = list(step3_maps["train"].keys())

    banks: Dict[str, BankView] = {}
    rows: List[Dict[str, Any]] = []
    selection_total = 0
    heldout_total = 0
    overlap_total = 0

    for view_index, view_id in enumerate(train_views):
        acc = PW._scalar_map(step3_maps["train"][view_id]["accumulation"])
        height, width = int(acc.shape[0]), int(acc.shape[1])
        all_valid = torch.ones((height, width), dtype=torch.bool)
        selection_general = _sample_flat(all_valid, SAMPLES_PER_VIEW, SELECTION_SEED + 17 * view_index)
        selection_mask = torch.zeros(height * width, dtype=torch.bool)
        selection_mask[selection_general] = True
        heldout_general = _sample_flat((~selection_mask).view(height, width), SAMPLES_PER_VIEW, HELDOUT_SEED + 17 * view_index)
        safe_mask = masks["train"][view_id]["M_SAFE"].bool()
        safe_cap = min(SAMPLES_PER_VIEW, int(safe_mask.sum().item()))
        selection_safe = _sample_flat(safe_mask, safe_cap, SELECTION_SEED + 1000 + 17 * view_index)
        heldout_safe = _sample_flat(safe_mask, safe_cap, HELDOUT_SEED + 1000 + 17 * view_index)
        banks[view_id] = BankView(view_id, height, width, selection_general, heldout_general, selection_safe, heldout_safe)
        selection_total += int(selection_general.numel())
        heldout_total += int(heldout_general.numel())
        overlap = len(set(selection_general.tolist()) & set(heldout_general.tolist()))
        overlap_total += overlap
        rows.append(
            {
                "view_index": view_index,
                "view_id": view_id,
                "height": height,
                "width": width,
                "selection_general_count": int(selection_general.numel()),
                "heldout_general_count": int(heldout_general.numel()),
                "selection_safe_count": int(selection_safe.numel()),
                "heldout_safe_count": int(heldout_safe.numel()),
                "selection_general_hash": _sha256_json(selection_general.tolist()),
                "heldout_general_hash": _sha256_json(heldout_general.tolist()),
                "selection_safe_hash": _sha256_json(selection_safe.tolist()),
                "heldout_safe_hash": _sha256_json(heldout_safe.tolist()),
                "selection_general_flat": selection_general.tolist(),
                "heldout_general_flat": heldout_general.tolist(),
                "selection_safe_flat": selection_safe.tolist(),
                "heldout_safe_flat": heldout_safe.tolist(),
                "overlap_count": overlap,
            }
        )

    bank_meta = {
        "scene": SCENE,
        "train_view_count": len(train_views),
        "train_views": train_views,
        "samples_per_view": SAMPLES_PER_VIEW,
        "selection_seed": SELECTION_SEED,
        "heldout_seed": HELDOUT_SEED,
        "step3000_render_meta": step3_meta,
        "mask_meta": mask_meta,
        "rows_hash": _sha256_json(rows),
        "selection_total_rays": selection_total,
        "heldout_total_rays": heldout_total,
        "overlap_total": overlap_total,
    }
    overlap_report = {
        "selection_ray_count": selection_total,
        "heldout_ray_count": heldout_total,
        "pixel_overlap_count": overlap_total,
        "pixel_overlap_fraction": float(overlap_total / max(selection_total, 1)),
        "strict_zero_overlap": bool(overlap_total == 0),
    }
    selection_rows = [
        {
            "view_index": row["view_index"],
            "view_id": row["view_id"],
            "height": row["height"],
            "width": row["width"],
            "selection_general_count": row["selection_general_count"],
            "selection_safe_count": row["selection_safe_count"],
        }
        for row in rows
    ]
    return banks, bank_meta, overlap_report, {"rows": rows}, {"rows": selection_rows}


def _build_swap_bank(train_view_ids: Sequence[str]) -> Dict[str, Any]:
    rng = random.Random(SWAP_SEED)
    rows: Dict[str, List[str]] = {}
    for view_id in train_view_ids:
        candidates = [alt for alt in train_view_ids if alt != view_id]
        rng.shuffle(candidates)
        rows[view_id] = candidates[:SWAP_COUNT]
    payload = {
        "seed": SWAP_SEED,
        "alternatives_per_source": SWAP_COUNT,
        "rows": rows,
        "hash": _sha256_json(rows),
        "reused_existing_bank": False,
    }
    return payload


def _camera_records(branch: Any) -> Tuple[Dict[str, Tuple[int, Cameras, Dict[str, Any]]], Dict[str, Tuple[int, Cameras, Dict[str, Any]]]]:
    train_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in OCMC._train_records(branch.pipeline)}
    eval_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in OCMC._eval_records(branch.pipeline)}
    return train_records, eval_records


def _load_c1_branch(repo: Path, source_output_dir: Path) -> Tuple[Any, Mapping[str, Any]]:
    branch = OCMC._setup_branch(repo, "C1")
    bundle = OCMC._load_snapshot(branch, source_output_dir, FINAL_STEP)
    if bundle is None:
        raise RuntimeError(f"Missing projector bundle in {SOURCE_OUTPUT_DIR}")
    branch.pipeline.eval()
    branch.pipeline.model.eval()
    return branch, bundle


def _mode_vector(selection: Mapping[str, Any], mode_index: int) -> Tensor:
    row = selection["basis_rows"][mode_index]
    vec = torch.tensor([float(row[f"v_{k}"]) for k in range(9)], dtype=torch.float64)
    return vec


def _selected_mode_projection(
    delta_raw: Tensor,
    basis_vector: Tensor,
    scale: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    scale = scale.reshape(1, 9).to(device=delta_raw.device, dtype=delta_raw.dtype).clamp_min(EPS)
    basis = basis_vector.reshape(9).to(device=delta_raw.device, dtype=delta_raw.dtype)
    delta_std = delta_raw.reshape(-1, 9) / scale
    coeff = delta_std @ basis
    removed_std = delta_std - coeff[:, None] * basis[None, :]
    removed_raw = (removed_std * scale).reshape_as(delta_raw)
    return delta_std, coeff, removed_raw


def _camera_context(model: Any, camera: Cameras) -> Tensor:
    return CAM._camera_context_for(model, camera.to(model.device), neutral=False).detach()


@torch.no_grad()
def _natural_and_effective_raw(
    model: Any,
    camera: Cameras,
    projector_bundle: Mapping[str, Any],
    *,
    camera_context_override: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, int, int]:
    raw_natural, height, width, _features = CAM._medium_raw_for_camera(
        model,
        camera,
        camera_context_override=camera_context_override,
        force_real_camera_context=camera_context_override is None,
    )
    zero_context = torch.zeros(3, device=model.device, dtype=raw_natural.dtype)
    raw_base, _, _, _ = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero_context)
    raw_effective, _, _, _ = OCMC._effective_raw_for_context(
        model,
        camera,
        camera_context_override=camera_context_override,
        projector_bundle=projector_bundle,
    )
    return raw_base, raw_natural, raw_effective, int(height), int(width)


@torch.no_grad()
def _render_from_raw(model: Any, camera: Cameras, raw: Tensor, height: int, width: int) -> Dict[str, Tensor]:
    return CAM._render_from_raw(model, camera, raw, height, width)


def _gt_for(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return PW._get_gt(model, batch, background)


@torch.no_grad()
def _evaluate_bank(
    *,
    model: Any,
    train_records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]],
    banks: Mapping[str, BankView],
    swap_bank: Mapping[str, Sequence[str]],
    selected_mode_index: int,
    selection: Mapping[str, Any],
    projector_bundle: Mapping[str, Any],
    population: str,
    compute_swaps: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    per_ray_rows: List[Dict[str, Any]] = []
    per_camera_rows: List[Dict[str, Any]] = []
    coeff_nat_all: List[Tensor] = []
    coeff_eff_all: List[Tensor] = []
    u_full_all: List[Tensor] = []
    u_minus_all: List[Tensor] = []
    c_utility_all: List[Tensor] = []
    c_rgb_all: List[Tensor] = []

    basis_vector = _mode_vector(selection, selected_mode_index).to(model.device)
    scale = selection["scale"].to(model.device)
    mode_label = f"mode_{selected_mode_index:02d}"

    context_bank = {view_id: _camera_context(model, camera) for view_id, (_idx, camera, _batch) in train_records.items()}

    for source_view_id, bank in banks.items():
        _idx, camera, batch = train_records[source_view_id]
        camera = camera.to(model.device)
        flat_general = bank.selection_general if population == "SELECTION" else bank.heldout_general
        flat_safe = bank.selection_safe if population == "SELECTION_SAFE" else bank.heldout_safe
        if population == "GENERAL":
            flat = flat_general
        elif population == "M_SAFE":
            flat = flat_safe
        elif population == "SELECTION":
            flat = flat_general
        elif population == "SELECTION_SAFE":
            flat = flat_safe
        else:
            raise ValueError(population)

        flat_dev = flat.to(model.device)
        raw_base, raw_natural, raw_effective, height, width = _natural_and_effective_raw(
            model,
            camera,
            projector_bundle,
            camera_context_override=None,
        )
        raw_effective_minus = None
        raw_natural_delta = raw_natural - raw_base
        raw_effective_delta = raw_effective - raw_base
        nat_std, nat_coeff, _nat_removed_raw = _selected_mode_projection(raw_natural_delta, basis_vector, scale)
        eff_std, eff_coeff, raw_effective_minus = _selected_mode_projection(raw_effective_delta, basis_vector, scale)
        raw_effective_minus = raw_base + raw_effective_minus

        out_correct_full = _render_from_raw(model, camera, raw_effective, height, width)
        out_correct_minus = _render_from_raw(model, camera, raw_effective_minus, height, width)
        gt_correct_full = _gt_for(model, batch, out_correct_full["background"]).reshape(-1, 3)
        gt_correct_minus = _gt_for(model, batch, out_correct_minus["background"]).reshape(-1, 3)
        pred_correct_full = out_correct_full["pred_image"].reshape(-1, 3)
        pred_correct_minus = out_correct_minus["pred_image"].reshape(-1, 3)
        err_correct_full = (pred_correct_full[flat_dev] - gt_correct_full[flat_dev]).square().mean(dim=-1)
        err_correct_minus = (pred_correct_minus[flat_dev] - gt_correct_minus[flat_dev]).square().mean(dim=-1)

        if compute_swaps:
            err_swap_full_sum = torch.zeros_like(err_correct_full)
            err_swap_minus_sum = torch.zeros_like(err_correct_full)
            for swapped_view_id in swap_bank[source_view_id]:
                swap_context = context_bank[swapped_view_id]
                raw_base_swap, raw_natural_swap, raw_effective_swap, height_swap, width_swap = _natural_and_effective_raw(
                    model,
                    camera,
                    projector_bundle,
                    camera_context_override=swap_context,
                )
                swap_effective_minus = raw_base_swap + _selected_mode_projection(
                    raw_effective_swap - raw_base_swap,
                    basis_vector,
                    scale,
                )[2]
                out_swap_full = _render_from_raw(model, camera, raw_effective_swap, height_swap, width_swap)
                out_swap_minus = _render_from_raw(model, camera, swap_effective_minus, height_swap, width_swap)
                gt_swap_full = _gt_for(model, batch, out_swap_full["background"]).reshape(-1, 3)
                gt_swap_minus = _gt_for(model, batch, out_swap_minus["background"]).reshape(-1, 3)
                pred_swap_full = out_swap_full["pred_image"].reshape(-1, 3)
                pred_swap_minus = out_swap_minus["pred_image"].reshape(-1, 3)
                err_swap_full_sum += (pred_swap_full[flat_dev] - gt_swap_full[flat_dev]).square().mean(dim=-1)
                err_swap_minus_sum += (pred_swap_minus[flat_dev] - gt_swap_minus[flat_dev]).square().mean(dim=-1)
                del raw_base_swap, raw_natural_swap, raw_effective_swap, swap_effective_minus
                del out_swap_full, out_swap_minus, gt_swap_full, gt_swap_minus, pred_swap_full, pred_swap_minus
            err_swap_full = err_swap_full_sum / float(len(swap_bank[source_view_id]))
            err_swap_minus = err_swap_minus_sum / float(len(swap_bank[source_view_id]))
        else:
            err_swap_full = torch.full_like(err_correct_full, float("nan"))
            err_swap_minus = torch.full_like(err_correct_full, float("nan"))

        u_full = err_swap_full - err_correct_full
        u_minus = err_swap_minus - err_correct_minus
        c_utility = u_full - u_minus
        c_rgb = err_correct_minus - err_correct_full

        coeff_nat_cpu = nat_coeff[flat_dev].detach().float().cpu()
        coeff_eff_cpu = eff_coeff[flat_dev].detach().float().cpu()
        err_correct_full_cpu = err_correct_full.detach().float().cpu()
        err_correct_minus_cpu = err_correct_minus.detach().float().cpu()
        err_swap_full_cpu = err_swap_full.detach().float().cpu()
        err_swap_minus_cpu = err_swap_minus.detach().float().cpu()
        u_full_cpu = u_full.detach().float().cpu()
        u_minus_cpu = u_minus.detach().float().cpu()
        c_utility_cpu = c_utility.detach().float().cpu()
        c_rgb_cpu = c_rgb.detach().float().cpu()
        nat_energy_frac = (
            nat_coeff[flat_dev].square() / nat_std[flat_dev].square().sum(dim=-1).clamp_min(EPS)
        ).detach().float().cpu()
        eff_energy_frac = (
            eff_coeff[flat_dev].square() / eff_std[flat_dev].square().sum(dim=-1).clamp_min(EPS)
        ).detach().float().cpu()

        coeff_nat_all.append(coeff_nat_cpu)
        coeff_eff_all.append(coeff_eff_cpu)
        if compute_swaps:
            u_full_all.append(u_full_cpu)
            u_minus_all.append(u_minus_cpu)
            c_utility_all.append(c_utility_cpu)
            c_rgb_all.append(c_rgb_cpu)

        for ray_idx in range(int(flat.numel())):
            row = {
                "population": population,
                "source_view_id": source_view_id,
                "ray_index_within_bank": ray_idx,
                "flat_index": int(flat[ray_idx].item()),
                "selected_mode_index": int(selected_mode_index),
                "selected_mode_label": mode_label,
                "A_natural_coeff": float(coeff_nat_cpu[ray_idx].item()),
                "A_effective_coeff": float(coeff_eff_cpu[ray_idx].item()),
                "A_natural_energy_fraction": float(nat_energy_frac[ray_idx].item()),
                "A_effective_energy_fraction": float(eff_energy_frac[ray_idx].item()),
                "E_correct_full": float(err_correct_full_cpu[ray_idx].item()),
                "E_correct_minus": float(err_correct_minus_cpu[ray_idx].item()),
            }
            if compute_swaps:
                row.update(
                    {
                        "E_swap_full": float(err_swap_full_cpu[ray_idx].item()),
                        "E_swap_minus": float(err_swap_minus_cpu[ray_idx].item()),
                        "U_full": float(u_full_cpu[ray_idx].item()),
                        "U_minus": float(u_minus_cpu[ray_idx].item()),
                        "C_utility_heldout": float(c_utility_cpu[ray_idx].item()),
                        "C_rgb_heldout": float(c_rgb_cpu[ray_idx].item()),
                    }
                )
            per_ray_rows.append(row)

        camera_row = {
            "population": population,
            "source_view_id": source_view_id,
            "sampled_rays": int(flat.numel()),
            "A_natural_rms": _rms(coeff_nat_cpu),
            "A_effective_rms": _rms(coeff_eff_cpu),
            "suppression_ratio_effective_over_natural": _rms(coeff_eff_cpu) / max(_rms(coeff_nat_cpu), EPS),
            "suppression_fraction_estimate": 1.0 - (_rms(coeff_eff_cpu) / max(_rms(coeff_nat_cpu), EPS)),
            "E_correct_full_mean": float(err_correct_full_cpu.mean().item()),
            "E_correct_minus_mean": float(err_correct_minus_cpu.mean().item()),
            "selected_mode_coeff_nat_mean": float(coeff_nat_cpu.mean().item()),
            "selected_mode_coeff_eff_mean": float(coeff_eff_cpu.mean().item()),
            "selected_mode_coeff_nat_std": float(coeff_nat_cpu.std(unbiased=False).item()) if coeff_nat_cpu.numel() > 1 else 0.0,
            "selected_mode_coeff_eff_std": float(coeff_eff_cpu.std(unbiased=False).item()) if coeff_eff_cpu.numel() > 1 else 0.0,
        }
        if compute_swaps:
            camera_row.update(
                {
                    "E_swap_full_mean": float(err_swap_full_cpu.mean().item()),
                    "E_swap_minus_mean": float(err_swap_minus_cpu.mean().item()),
                    "U_full_mean": float(u_full_cpu.mean().item()),
                    "U_minus_mean": float(u_minus_cpu.mean().item()),
                    "C_utility_mean": float(c_utility_cpu.mean().item()),
                    "C_rgb_mean": float(c_rgb_cpu.mean().item()),
                    "C_utility_median": float(torch.median(c_utility_cpu).item()),
                    "C_rgb_median": float(torch.median(c_rgb_cpu).item()),
                    "C_utility_fraction_positive": float((c_utility_cpu > 0).float().mean().item()),
                    "C_rgb_fraction_positive": float((c_rgb_cpu > 0).float().mean().item()),
                    "U_full_median": float(torch.median(u_full_cpu).item()),
                    "U_full_fraction_positive": float((u_full_cpu > 0).float().mean().item()),
                    "U_minus_median": float(torch.median(u_minus_cpu).item()),
                    "U_minus_fraction_positive": float((u_minus_cpu > 0).float().mean().item()),
                    "relative_utility_contribution": float(c_utility_cpu.mean().item())
                    / max(abs(float(u_full_cpu.mean().item())), EPS),
                    "utility_per_energy": float(c_utility_cpu.mean().item()) / max(_rms(coeff_nat_cpu), EPS),
                    "rgb_cost_per_energy": float(c_rgb_cpu.mean().item()) / max(_rms(coeff_nat_cpu), EPS),
                }
            )
        per_camera_rows.append(camera_row)

        del raw_base, raw_natural, raw_effective, raw_natural_delta, raw_effective_delta, nat_std, nat_coeff, eff_std, eff_coeff
        del out_correct_full, out_correct_minus, gt_correct_full, gt_correct_minus
        del pred_correct_full, pred_correct_minus, err_correct_full, err_correct_minus
        del err_swap_full, err_swap_minus, u_full, u_minus, c_utility, c_rgb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "population": population,
        "sampled_rays": int(sum(int(r["sampled_rays"]) for r in per_camera_rows if r["population"] == population)),
        "A_natural_rms": _rms(torch.cat(coeff_nat_all)) if coeff_nat_all else float("nan"),
        "A_effective_rms": _rms(torch.cat(coeff_eff_all)) if coeff_eff_all else float("nan"),
        "suppression_ratio_effective_over_natural": (
            _rms(torch.cat(coeff_eff_all)) / max(_rms(torch.cat(coeff_nat_all)), EPS)
            if coeff_nat_all
            else float("nan")
        ),
    }
    if compute_swaps:
        u_full_all_cat = torch.cat(u_full_all) if u_full_all else torch.empty(0)
        u_minus_all_cat = torch.cat(u_minus_all) if u_minus_all else torch.empty(0)
        c_utility_all_cat = torch.cat(c_utility_all) if c_utility_all else torch.empty(0)
        c_rgb_all_cat = torch.cat(c_rgb_all) if c_rgb_all else torch.empty(0)
        summary.update(
            {
                "U_full_mean": float(u_full_all_cat.mean().item()) if u_full_all_cat.numel() else float("nan"),
                "U_full_median": float(torch.median(u_full_all_cat).item()) if u_full_all_cat.numel() else float("nan"),
                "U_full_std": float(u_full_all_cat.std(unbiased=False).item()) if u_full_all_cat.numel() > 1 else 0.0,
                "U_full_fraction_positive": float((u_full_all_cat > 0).float().mean().item()) if u_full_all_cat.numel() else float("nan"),
                "U_minus_mean": float(u_minus_all_cat.mean().item()) if u_minus_all_cat.numel() else float("nan"),
                "U_minus_median": float(torch.median(u_minus_all_cat).item()) if u_minus_all_cat.numel() else float("nan"),
                "U_minus_std": float(u_minus_all_cat.std(unbiased=False).item()) if u_minus_all_cat.numel() > 1 else 0.0,
                "U_minus_fraction_positive": float((u_minus_all_cat > 0).float().mean().item()) if u_minus_all_cat.numel() else float("nan"),
                "C_utility_mean": float(c_utility_all_cat.mean().item()) if c_utility_all_cat.numel() else float("nan"),
                "C_utility_median": float(torch.median(c_utility_all_cat).item()) if c_utility_all_cat.numel() else float("nan"),
                "C_utility_std": float(c_utility_all_cat.std(unbiased=False).item()) if c_utility_all_cat.numel() > 1 else 0.0,
                "C_utility_fraction_positive": float((c_utility_all_cat > 0).float().mean().item()) if c_utility_all_cat.numel() else float("nan"),
                "C_rgb_mean": float(c_rgb_all_cat.mean().item()) if c_rgb_all_cat.numel() else float("nan"),
                "C_rgb_median": float(torch.median(c_rgb_all_cat).item()) if c_rgb_all_cat.numel() else float("nan"),
                "C_rgb_std": float(c_rgb_all_cat.std(unbiased=False).item()) if c_rgb_all_cat.numel() > 1 else 0.0,
                "C_rgb_fraction_positive": float((c_rgb_all_cat > 0).float().mean().item()) if c_rgb_all_cat.numel() else float("nan"),
            }
        )
    return per_ray_rows, per_camera_rows, summary


def _aggregate_per_camera(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["population"]), str(row["source_view_id"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (population, source_view_id), group in sorted(grouped.items(), key=lambda kv: kv[0]):
        item = {
            "population": population,
            "source_view_id": source_view_id,
            "sampled_rays": int(sum(int(r.get("sampled_rays", 0)) for r in group) if group and "sampled_rays" in group[0] else len(group)),
        }
        for key in keys:
            item[key] = _mean([float(r[key]) for r in group if key in r])
        out.append(item)
    return out


def _fraction_positive_camera(rows: Sequence[Mapping[str, Any]], key: str, population: str) -> float:
    vals = [float(r[key]) for r in rows if r["population"] == population and key in r]
    if not vals:
        return float("nan")
    return float(sum(v > 0.0 for v in vals) / len(vals))


def _dominance_share(rows: Sequence[Mapping[str, Any]], key: str, population: str) -> float:
    vals = [abs(float(r[key])) for r in rows if r["population"] == population and key in r and math.isfinite(float(r[key]))]
    if not vals:
        return float("nan")
    total = sum(vals)
    if total <= 0.0:
        return float("nan")
    return float(max(vals) / total)


def _classification(summary_general: Mapping[str, Any], summary_safe: Mapping[str, Any], selected: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    general_mean = float(summary_general.get("C_utility_mean", float("nan")))
    general_median = float(summary_general.get("C_utility_median", float("nan")))
    general_frac = float(summary_general.get("C_utility_fraction_positive", float("nan")))
    general_rgb = float(summary_general.get("C_rgb_mean", float("nan")))
    general_rel = float(summary_general.get("relative_utility_contribution", float("nan")))
    general_energy = float(summary_general.get("utility_per_energy", float("nan")))
    general_dominance = float(summary_general.get("dominance_share", float("nan")))
    safe_mean = float(summary_safe.get("C_utility_mean", float("nan")))
    safe_median = float(summary_safe.get("C_utility_median", float("nan")))
    safe_frac = float(summary_safe.get("C_utility_fraction_positive", float("nan")))
    safe_rgb = float(summary_safe.get("C_rgb_mean", float("nan")))
    safe_rel = float(summary_safe.get("relative_utility_contribution", float("nan")))

    hard_fail = (
        not math.isfinite(general_mean)
        or general_mean <= 0.0
        or not math.isfinite(general_median)
        or general_median <= 0.0
        or not math.isfinite(general_frac)
        or general_frac < 0.40
        or not math.isfinite(general_rgb)
        or general_rgb < 0.0
        or not math.isfinite(general_energy)
        or general_energy <= 0.0
        or float(selected["g_obs"]) >= 0.5
        or not math.isfinite(general_dominance)
        or general_dominance > 0.5
    )
    validated = (
        not hard_fail
        and general_frac >= 0.60
        and general_rel >= 0.10
        and general_mean > 0.0
        and general_median > 0.0
        and general_rgb >= 0.0
        and general_energy > 0.0
        and float(selected["g_obs"]) < 0.5
        and general_dominance <= 0.5
        and math.isfinite(safe_mean)
        and safe_mean >= 0.0
        and math.isfinite(safe_median)
        and safe_median >= 0.0
        and math.isfinite(safe_frac)
        and safe_frac >= 0.40
        and math.isfinite(safe_rgb)
        and safe_rgb >= -0.5 * abs(general_rgb if math.isfinite(general_rgb) else 0.0)
    )
    tentative = (
        not hard_fail
        and not validated
        and general_mean > 0.0
        and general_frac >= 0.40
    )
    if validated:
        label = "SINGLE_MODE_CONTEXT_UTILITY_VALIDATED"
    elif tentative:
        label = "SINGLE_MODE_CONTEXT_UTILITY_TENTATIVE"
    else:
        label = "SINGLE_MODE_CONTEXT_UTILITY_NOT_VALIDATED"
    evidence = {
        "general": dict(summary_general),
        "safe": dict(summary_safe),
        "selected_mode": dict(selected),
        "hard_fail": bool(hard_fail),
        "validated": bool(validated),
        "tentative": bool(tentative),
    }
    return label, evidence


@torch.no_grad()
def _selection_set_metrics(
    *,
    model: Any,
    train_records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]],
    banks: Mapping[str, BankView],
    selected_mode_index: int,
    selection: Mapping[str, Any],
    projector_bundle: Mapping[str, Any],
    population: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    per_ray_rows: List[Dict[str, Any]] = []
    per_camera_rows: List[Dict[str, Any]] = []
    coeff_nat_all: List[Tensor] = []
    coeff_eff_all: List[Tensor] = []
    basis_vector = _mode_vector(selection, selected_mode_index).to(model.device)
    scale = selection["scale"].to(model.device)
    mode_label = f"mode_{selected_mode_index:02d}"
    for source_view_id, bank in banks.items():
        _idx, camera, batch = train_records[source_view_id]
        camera = camera.to(model.device)
        flat = bank.selection_general if population == "GENERAL" else bank.selection_safe
        flat_dev = flat.to(model.device)
        raw_base, raw_natural, raw_effective, height, width = _natural_and_effective_raw(
            model,
            camera,
            projector_bundle,
            camera_context_override=None,
        )
        nat_std, nat_coeff, _ = _selected_mode_projection(raw_natural - raw_base, basis_vector, scale)
        eff_std, eff_coeff, _ = _selected_mode_projection(raw_effective - raw_base, basis_vector, scale)
        nat_coeff_sample = nat_coeff[flat_dev].detach().float().cpu()
        eff_coeff_sample = eff_coeff[flat_dev].detach().float().cpu()
        nat_std_sample = nat_std[flat_dev].detach().float().cpu()
        eff_std_sample = eff_std[flat_dev].detach().float().cpu()
        coeff_nat_all.append(nat_coeff_sample)
        coeff_eff_all.append(eff_coeff_sample)
        per_camera_rows.append(
            {
                "population": population,
                "source_view_id": source_view_id,
                "sampled_rays": int(flat.numel()),
                "A_natural_rms": _rms(nat_coeff_sample),
                "A_effective_rms": _rms(eff_coeff_sample),
                "suppression_ratio_effective_over_natural": _rms(eff_coeff_sample) / max(_rms(nat_coeff_sample), EPS),
            }
        )
        for ray_idx in range(int(flat.numel())):
            per_ray_rows.append(
                {
                    "population": population,
                    "source_view_id": source_view_id,
                    "ray_index_within_bank": ray_idx,
                    "flat_index": int(flat[ray_idx].item()),
                    "selected_mode_index": int(selected_mode_index),
                    "selected_mode_label": mode_label,
                    "A_natural_coeff": float(nat_coeff_sample[ray_idx].item()),
                    "A_effective_coeff": float(eff_coeff_sample[ray_idx].item()),
                    "A_natural_energy_fraction": float(
                        (nat_coeff_sample.square() / nat_std_sample.square().sum(dim=-1).clamp_min(EPS))[ray_idx].item()
                    ),
                    "A_effective_energy_fraction": float(
                        (eff_coeff_sample.square() / eff_std_sample.square().sum(dim=-1).clamp_min(EPS))[ray_idx].item()
                    ),
                }
            )
        del raw_base, raw_natural, raw_effective, nat_std, nat_coeff, eff_std, eff_coeff
        gc.collect()
    summary = {
        "population": population,
        "sampled_rays": int(sum(int(r["sampled_rays"]) for r in per_camera_rows)),
        "A_natural_rms": _rms(torch.cat(coeff_nat_all)) if coeff_nat_all else float("nan"),
        "A_effective_rms": _rms(torch.cat(coeff_eff_all)) if coeff_eff_all else float("nan"),
        "suppression_ratio_effective_over_natural": (
            _rms(torch.cat(coeff_eff_all)) / max(_rms(torch.cat(coeff_nat_all)), EPS) if coeff_nat_all else float("nan")
        ),
        "suppression_fraction_estimate": (
            1.0 - (_rms(torch.cat(coeff_eff_all)) / max(_rms(torch.cat(coeff_nat_all)), EPS)) if coeff_nat_all else float("nan")
        ),
    }
    return per_ray_rows, per_camera_rows, summary


@torch.no_grad()
def _eval_replication(
    *,
    model: Any,
    train_records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]],
    eval_records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]],
    selected_mode_index: int,
    selection: Mapping[str, Any],
    projector_bundle: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    basis_vector = _mode_vector(selection, selected_mode_index).to(model.device)
    scale = selection["scale"].to(model.device)
    rows: List[Dict[str, Any]] = []
    per_view_psnr_full: List[float] = []
    per_view_psnr_minus: List[float] = []
    for view_id, (_idx, camera, batch) in eval_records.items():
        camera = camera.to(model.device)
        raw_base, raw_natural, raw_effective, height, width = _natural_and_effective_raw(
            model,
            camera,
            projector_bundle,
            camera_context_override=None,
        )
        raw_effective_minus = raw_base + _selected_mode_projection(raw_effective - raw_base, basis_vector, scale)[2]
        out_full = _render_from_raw(model, camera, raw_effective, height, width)
        out_minus = _render_from_raw(model, camera, raw_effective_minus, height, width)
        gt_full = _gt_for(model, batch, out_full["background"])
        gt_minus = _gt_for(model, batch, out_minus["background"])
        metrics_full = MIC._metric_images(model, out_full["pred_image"], gt_full)
        metrics_minus = MIC._metric_images(model, out_minus["pred_image"], gt_minus)
        row = {
            "view_id": view_id,
            "PSNR_full": float(metrics_full["PSNR"]),
            "SSIM_full": float(metrics_full["SSIM"]),
            "LPIPS_full": float(metrics_full["LPIPS"]),
            "MSE_full": float(metrics_full["MSE"]),
            "PSNR_minus": float(metrics_minus["PSNR"]),
            "SSIM_minus": float(metrics_minus["SSIM"]),
            "LPIPS_minus": float(metrics_minus["LPIPS"]),
            "MSE_minus": float(metrics_minus["MSE"]),
            "delta_PSNR_minus_full": float(metrics_minus["PSNR"] - metrics_full["PSNR"]),
            "delta_SSIM_minus_full": float(metrics_minus["SSIM"] - metrics_full["SSIM"]),
            "delta_LPIPS_minus_full": float(metrics_minus["LPIPS"] - metrics_full["LPIPS"]),
            "delta_MSE_minus_full": float(metrics_minus["MSE"] - metrics_full["MSE"]),
        }
        rows.append(row)
        per_view_psnr_full.append(float(metrics_full["PSNR"]))
        per_view_psnr_minus.append(float(metrics_minus["PSNR"]))
        del raw_base, raw_natural, raw_effective, raw_effective_minus, out_full, out_minus, gt_full, gt_minus
        gc.collect()
    mean_row = {
        "view_id": "MEAN",
        "PSNR_full": _mean([r["PSNR_full"] for r in rows]),
        "SSIM_full": _mean([r["SSIM_full"] for r in rows]),
        "LPIPS_full": _mean([r["LPIPS_full"] for r in rows]),
        "MSE_full": _mean([r["MSE_full"] for r in rows]),
        "PSNR_minus": _mean([r["PSNR_minus"] for r in rows]),
        "SSIM_minus": _mean([r["SSIM_minus"] for r in rows]),
        "LPIPS_minus": _mean([r["LPIPS_minus"] for r in rows]),
        "MSE_minus": _mean([r["MSE_minus"] for r in rows]),
    }
    rows.append(mean_row)
    return rows, mean_row


def _final_classification(
    *,
    selected_mode: Mapping[str, Any],
    general_summary: Mapping[str, Any],
    safe_summary: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    eval_summary: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    general_mean = float(general_summary.get("C_utility_mean", float("nan")))
    general_median = float(general_summary.get("C_utility_median", float("nan")))
    general_frac = float(general_summary.get("camera_positive_fraction", float("nan")))
    general_rel = float(general_summary.get("relative_utility_contribution", float("nan")))
    general_rgb = float(general_summary.get("C_rgb_mean", float("nan")))
    general_energy = float(general_summary.get("utility_per_energy", float("nan")))
    dominance = float(general_summary.get("dominance_share", float("nan")))
    safe_mean = float(safe_summary.get("C_utility_mean", float("nan")))
    safe_median = float(safe_summary.get("C_utility_median", float("nan")))
    safe_frac = float(safe_summary.get("camera_positive_fraction", float("nan")))
    safe_rgb = float(safe_summary.get("C_rgb_mean", float("nan")))
    selected_sigma_rank = int(selected_mode["sigma_rank"])
    is_bottom3 = selected_sigma_rank <= 3

    validated = (
        is_bottom3
        and general_mean > 0.0
        and general_median > 0.0
        and general_frac >= 0.60
        and general_rel >= 0.10
        and general_rgb >= 0.0
        and general_energy > 0.0
        and float(selected_mode["g_obs"]) < 0.5
        and dominance <= 0.5
        and safe_mean >= 0.0
        and safe_median >= 0.0
        and safe_frac >= 0.40
        and safe_rgb >= -0.5 * abs(general_rgb if math.isfinite(general_rgb) else 0.0)
    )
    tentative = (
        is_bottom3
        and general_mean > 0.0
        and general_frac >= 0.40
        and general_rel > 0.0
    )
    if validated:
        label = "SINGLE_MODE_CONTEXT_UTILITY_VALIDATED"
    elif tentative:
        label = "SINGLE_MODE_CONTEXT_UTILITY_TENTATIVE"
    else:
        label = "SINGLE_MODE_CONTEXT_UTILITY_NOT_VALIDATED"
    evidence = {
        "selected_mode": dict(selected_mode),
        "general": dict(general_summary),
        "safe": dict(safe_summary),
        "selection_set": dict(selection_summary),
        "eval": dict(eval_summary) if eval_summary is not None else None,
        "validated": bool(validated),
        "tentative": bool(tentative),
        "bottom3_selected": bool(is_bottom3),
    }
    return label, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--selection-output-dir", type=Path, default=SELECTION_OUTPUT_DIR)
    parser.add_argument("--source-output-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args()

    repo = args.repo.resolve()
    selection_dir = args.selection_output_dir.resolve()
    source_output_dir = args.source_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    environment_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    source_semantics = _source_semantics(selection_dir)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "environment_manifest.json", environment_manifest)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "source_semantics.json", source_semantics)
    (output_dir / "source_semantics.md").write_text(_source_semantics_md(source_semantics), encoding="utf8")

    archived = _load_archived_selection(selection_dir)
    selection_mode = archived["selected_mode"]
    selected_mode_index = int(selection_mode["mode_index"])

    banks, bank_meta, overlap_report, bank_rows, selection_bank_rows = _build_banks(repo)
    train_view_ids = bank_meta["train_views"]
    swap_bank = _build_swap_bank(train_view_ids)

    _write_json(output_dir / "heldout_ray_bank.json", {**bank_meta, "rows": bank_rows["rows"]})
    _write_json(output_dir / "heldout_swap_bank.json", swap_bank)
    _write_json(output_dir / "heldout_overlap_report.json", overlap_report)
    _write_json(
        output_dir / "selection_mode.json",
        {
            **selection_mode,
            "selected_mode_index": selected_mode_index,
            "frozen": True,
            "archived_source_dir": str(selection_dir),
        },
    )

    branch, projector_bundle = _load_c1_branch(repo, source_output_dir)
    try:
        train_records, eval_records = _camera_records(branch)
        selection_samples = {
            view_id: MI.ViewSample(
                view_id=view_id,
                height=bank.height,
                width=bank.width,
                general_flat=bank.selection_general,
                safe_flat=bank.selection_safe,
                safe_available_pixels=int(banks[view_id].selection_safe.numel()),
            )
            for view_id, bank in banks.items()
        }
        analyses, meta = CAM._analyse_loaded_branch(branch, selection_samples)
        selection_general = analyses["GENERAL"]
        selection_scale = selection_general.scale.detach().float().cpu()
        selection_basis_row = selection_mode.copy()
        selection_basis_row["scale"] = [float(v) for v in selection_scale.tolist()]
        selection_basis_row["basis_source"] = "recovered_from_frozen_selection_bank"
        selection_basis_row["checkpoint_path"] = str(source_output_dir / "checkpoints/C1/step-000014999.ckpt")
        selection_basis_row["checkpoint_refresh_step"] = 10000
        selection_basis_row["selection_basis_meta"] = meta
        _write_json(
            output_dir / "selection_mode.json",
            {
                **selection_basis_row,
                "selected_mode_index": selected_mode_index,
                "frozen": True,
                "archived_source_dir": str(selection_dir),
            },
        )

        selection_rows, selection_camera_rows, selection_summary = _selection_set_metrics(
            model=branch.pipeline.model,
            train_records=train_records,
            banks=banks,
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
            population="GENERAL",
        )
        selection_safe_rows, selection_safe_camera_rows, selection_safe_summary = _selection_set_metrics(
            model=branch.pipeline.model,
            train_records=train_records,
            banks=banks,
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
            population="SELECTION_SAFE",
        )
        selection_summary.update(
            {
                "A_natural_rms": _rms(torch.cat([torch.tensor([r["A_natural_coeff"] for r in selection_rows], dtype=torch.float32)])) if selection_rows else float("nan"),
                "A_effective_rms": _rms(torch.cat([torch.tensor([r["A_effective_coeff"] for r in selection_rows], dtype=torch.float32)])) if selection_rows else float("nan"),
            }
        )

        # Rebuild the bank summary explicitly for the frozen selection set.
        selection_bank_summary = {
            "selection_seed": SELECTION_SEED,
            "selection_general_ray_count": int(sum(row["selection_general_count"] for row in selection_bank_rows["rows"])),
            "selection_safe_ray_count": int(sum(row["selection_safe_count"] for row in selection_bank_rows["rows"])),
            "selection_general_rows": selection_bank_rows["rows"],
            "selection_general_hash": _sha256_json([row["selection_general_hash"] for row in bank_rows["rows"]]),
            "selection_safe_hash": _sha256_json([row["selection_safe_hash"] for row in bank_rows["rows"]]),
            "scale": [float(v) for v in selection_scale.tolist()],
            "median_sigma": archived["median_sigma"],
            "selected_mode": selection_basis_row,
        }
        _write_json(output_dir / "selection_set_summary.json", selection_bank_summary)

        general_rows, general_camera_rows, general_summary = _evaluate_bank(
            model=branch.pipeline.model,
            train_records=train_records,
            banks=banks,
            swap_bank=swap_bank["rows"],
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
            population="GENERAL",
            compute_swaps=True,
        )
        msafe_rows, msafe_camera_rows, msafe_summary = _evaluate_bank(
            model=branch.pipeline.model,
            train_records=train_records,
            banks=banks,
            swap_bank=swap_bank["rows"],
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
            population="M_SAFE",
            compute_swaps=True,
        )

        # Selection bank amplitude control: use the same basis on the frozen selection GENERAL rays.
        selection_amp_rows, selection_amp_camera_rows, selection_amp_summary = _selection_set_metrics(
            model=branch.pipeline.model,
            train_records=train_records,
            banks=banks,
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
            population="GENERAL",
        )
        selection_bank_summary["A_natural_rms"] = selection_amp_summary["A_natural_rms"]
        selection_bank_summary["A_effective_rms"] = selection_amp_summary["A_effective_rms"]
        selection_bank_summary["suppression_ratio_effective_over_natural"] = selection_amp_summary["suppression_ratio_effective_over_natural"]
        selection_bank_summary["suppression_fraction_estimate"] = selection_amp_summary["suppression_fraction_estimate"]
        selection_bank_summary["selection_set_utility_per_energy"] = float(
            selection_mode["C_utility_proxy"] / max(selection_amp_summary["A_natural_rms"], EPS)
        )
        _write_json(output_dir / "selection_set_summary.json", selection_bank_summary)

        # Write row-level outputs.
        _write_csv(output_dir / "heldout_full_utility.csv", general_rows)
        _write_json(output_dir / "heldout_full_utility.json", {"rows": general_rows, "summary": general_summary})
        _write_csv(output_dir / "heldout_selected_mode_removal.csv", general_rows)
        _write_json(output_dir / "heldout_selected_mode_removal.json", {"rows": general_rows, "summary": general_summary})
        _write_csv(output_dir / "heldout_per_camera_utility.csv", general_camera_rows)
        _write_json(output_dir / "heldout_per_camera_utility.json", {"rows": general_camera_rows, "summary": general_summary})
        _write_csv(output_dir / "heldout_msafe_replication.csv", msafe_rows)
        _write_json(output_dir / "heldout_msafe_replication.json", {"rows": msafe_rows, "summary": msafe_summary})

        # Add explicit row-level deltas for the held-out GENERAL bank.
        per_ray_general: List[Dict[str, Any]] = []
        for row in general_rows:
            per_ray_general.append(
                {
                    **row,
                    "U_full": float(row["U_full"]),
                    "U_minus": float(row["U_minus"]),
                    "C_utility_heldout": float(row["C_utility_heldout"]),
                    "C_rgb_heldout": float(row["C_rgb_heldout"]),
                }
            )
        _write_csv(output_dir / "heldout_full_utility.csv", [
            {
                "population": row["population"],
                "source_view_id": row["source_view_id"],
                "ray_index_within_bank": row["ray_index_within_bank"],
                "flat_index": row["flat_index"],
                "A_natural_coeff": row["A_natural_coeff"],
                "A_effective_coeff": row["A_effective_coeff"],
                "A_natural_energy_fraction": row["A_natural_energy_fraction"],
                "A_effective_energy_fraction": row["A_effective_energy_fraction"],
                "E_correct_full": row["E_correct_full"],
                "E_swap_full": row["E_swap_full"],
                "U_full": row["U_full"],
            }
            for row in per_ray_general
        ])
        _write_json(
            output_dir / "heldout_full_utility.json",
            {"rows": per_ray_general, "summary": general_summary},
        )
        _write_csv(
            output_dir / "heldout_selected_mode_removal.csv",
            [
                {
                    "population": row["population"],
                    "source_view_id": row["source_view_id"],
                    "ray_index_within_bank": row["ray_index_within_bank"],
                    "flat_index": row["flat_index"],
                    "A_natural_coeff": row["A_natural_coeff"],
                    "A_effective_coeff": row["A_effective_coeff"],
                    "A_natural_energy_fraction": row["A_natural_energy_fraction"],
                    "A_effective_energy_fraction": row["A_effective_energy_fraction"],
                    "E_correct_full": row["E_correct_full"],
                    "E_swap_full": row["E_swap_full"],
                    "U_full": row["U_full"],
                    "E_correct_minus": row["E_correct_minus"],
                    "E_swap_minus": row["E_swap_minus"],
                    "U_minus": row["U_minus"],
                    "C_utility_heldout": row["C_utility_heldout"],
                    "C_rgb_heldout": row["C_rgb_heldout"],
                }
                for row in per_ray_general
            ],
        )
        _write_json(
            output_dir / "heldout_selected_mode_removal.json",
            {"rows": per_ray_general, "summary": general_summary},
        )

        # Add the per-camera utility aggregates and normalize summary fields.
        general_camera_summary = {
            "population": "GENERAL",
            "C_utility_mean": general_summary["C_utility_mean"],
            "C_utility_median": general_summary["C_utility_median"],
            "C_utility_std": general_summary["C_utility_std"],
            "C_utility_fraction_positive": general_summary["C_utility_fraction_positive"],
            "C_rgb_mean": general_summary["C_rgb_mean"],
            "C_rgb_median": general_summary["C_rgb_median"],
            "C_rgb_std": general_summary["C_rgb_std"],
            "C_rgb_fraction_positive": general_summary["C_rgb_fraction_positive"],
            "U_full_mean": general_summary["U_full_mean"],
            "U_full_median": general_summary["U_full_median"],
            "U_full_std": general_summary["U_full_std"],
            "U_full_fraction_positive": general_summary["U_full_fraction_positive"],
            "U_minus_mean": general_summary["U_minus_mean"],
            "U_minus_median": general_summary["U_minus_median"],
            "U_minus_std": general_summary["U_minus_std"],
            "U_minus_fraction_positive": general_summary["U_minus_fraction_positive"],
            "A_natural_rms": general_summary["A_natural_rms"],
            "A_effective_rms": general_summary["A_effective_rms"],
            "suppression_ratio_effective_over_natural": general_summary["suppression_ratio_effective_over_natural"],
            "relative_utility_contribution": general_summary["C_utility_mean"] / max(abs(general_summary["U_full_mean"]), EPS),
            "utility_per_energy": general_summary["C_utility_mean"] / max(general_summary["A_natural_rms"], EPS),
            "rgb_cost_per_energy": general_summary["C_rgb_mean"] / max(general_summary["A_natural_rms"], EPS),
            "dominance_share": _dominance_share(general_camera_rows, "C_utility_mean", "GENERAL"),
            "camera_positive_fraction": _fraction_positive_camera(general_camera_rows, "C_utility_mean", "GENERAL"),
        }
        safe_camera_summary = {
            "population": "M_SAFE",
            "C_utility_mean": msafe_summary["C_utility_mean"],
            "C_utility_median": msafe_summary["C_utility_median"],
            "C_utility_std": msafe_summary["C_utility_std"],
            "C_utility_fraction_positive": msafe_summary["C_utility_fraction_positive"],
            "C_rgb_mean": msafe_summary["C_rgb_mean"],
            "C_rgb_median": msafe_summary["C_rgb_median"],
            "C_rgb_std": msafe_summary["C_rgb_std"],
            "C_rgb_fraction_positive": msafe_summary["C_rgb_fraction_positive"],
            "U_full_mean": msafe_summary["U_full_mean"],
            "U_full_median": msafe_summary["U_full_median"],
            "U_full_std": msafe_summary["U_full_std"],
            "U_full_fraction_positive": msafe_summary["U_full_fraction_positive"],
            "U_minus_mean": msafe_summary["U_minus_mean"],
            "U_minus_median": msafe_summary["U_minus_median"],
            "U_minus_std": msafe_summary["U_minus_std"],
            "U_minus_fraction_positive": msafe_summary["U_minus_fraction_positive"],
            "A_natural_rms": msafe_summary["A_natural_rms"],
            "A_effective_rms": msafe_summary["A_effective_rms"],
            "suppression_ratio_effective_over_natural": msafe_summary["suppression_ratio_effective_over_natural"],
            "relative_utility_contribution": msafe_summary["C_utility_mean"] / max(abs(msafe_summary["U_full_mean"]), EPS),
            "utility_per_energy": msafe_summary["C_utility_mean"] / max(msafe_summary["A_natural_rms"], EPS),
            "rgb_cost_per_energy": msafe_summary["C_rgb_mean"] / max(msafe_summary["A_natural_rms"], EPS),
            "dominance_share": _dominance_share(msafe_camera_rows, "C_utility_mean", "M_SAFE"),
            "camera_positive_fraction": _fraction_positive_camera(msafe_camera_rows, "C_utility_mean", "M_SAFE"),
        }

        _write_csv(output_dir / "heldout_per_camera_utility.csv", general_camera_rows + msafe_camera_rows)
        _write_json(
            output_dir / "heldout_per_camera_utility.json",
            {"rows": general_camera_rows + msafe_camera_rows, "summary": {"GENERAL": general_camera_summary, "M_SAFE": safe_camera_summary}},
        )
        _write_csv(output_dir / "heldout_msafe_replication.csv", msafe_rows)
        _write_json(output_dir / "heldout_msafe_replication.json", {"rows": msafe_rows, "summary": safe_camera_summary})

        eval_rows, eval_summary = _eval_replication(
            model=branch.pipeline.model,
            train_records=train_records,
            eval_records=eval_records,
            selected_mode_index=selected_mode_index,
            selection={**archived, "scale": selection_scale},
            projector_bundle=projector_bundle,
        )
        _write_csv(output_dir / "heldout_eval_replication.csv", eval_rows)
        _write_json(output_dir / "heldout_eval_replication.json", {"rows": eval_rows, "summary": eval_summary})

        # Phase A classification.
        selection_summary.update(
            {
                "selection_mode_index": selected_mode_index,
                "selection_mode_label": f"mode_{selected_mode_index:02d}",
                "selection_mode_sigma_rank": int(selection_mode["sigma_rank"]),
                "selection_mode_sigma": float(selection_mode["sigma"]),
                "selection_mode_g_obs": float(selection_mode["g_obs"]),
                "selection_mode_selection_score": float(selection_mode["selection_score"]),
                "selection_mode_utility_per_energy": float(
                    selection_mode["C_utility_proxy"] / max(selection_bank_summary["A_natural_rms"], EPS)
                ),
                "selection_mode_suppression_fraction": float(selection_bank_summary["suppression_fraction_estimate"]),
            }
        )
        phase_a_label, phase_a_evidence = _final_classification(
            selected_mode=selection_mode,
            general_summary=general_camera_summary,
            safe_summary=safe_camera_summary,
            selection_summary=selection_summary,
            eval_summary=eval_summary,
        )
        _write_json(
            output_dir / "phase_a_classification.json",
            {
                "classification": phase_a_label,
                "selected_mode": selection_mode,
                "selection_summary": selection_summary,
                "general_summary": general_camera_summary,
                "safe_summary": safe_camera_summary,
                "eval_summary": eval_summary,
                "evidence": phase_a_evidence,
            },
        )

        # Persist the summary artifacts after classification is frozen.
        selection_bank_summary["selection_mode"] = selection_mode
        selection_bank_summary["selection_mode_utility_per_energy"] = float(selection_mode["C_utility_proxy"] / max(selection_bank_summary["A_natural_rms"], EPS))
        selection_bank_summary["selection_mode_rgb_cost_per_energy"] = float(selection_mode["C_rgb_proxy"] / max(selection_bank_summary["A_natural_rms"], EPS))
        _write_json(output_dir / "selection_set_summary.json", selection_bank_summary)
        _write_json(
            output_dir / "selection_mode.json",
            {
                **selection_mode,
                "scale": [float(v) for v in selection_scale.tolist()],
                "basis_source": "frozen_selection_bank_reconstruction",
                "checkpoint_path": str(source_output_dir / "checkpoints/C1/step-000014999.ckpt"),
                "checkpoint_refresh_step": 10000,
                "frozen": True,
            },
        )
        _write_json(
            output_dir / "checkpoint_manifest.json",
            {
                "checkpoint_path": str(source_output_dir / "checkpoints/C1/step-000014999.ckpt"),
                "checkpoint_refresh_step": 10000,
                "checkpoint_sha256": _sha256_bytes((source_output_dir / "checkpoints/C1/step-000014999.ckpt").read_bytes()),
                "source_output_dir": str(source_output_dir),
                "selection_output_dir": str(selection_dir),
                "loaded_step": FINAL_STEP,
                "projector_bundle_keys": sorted([key for key in projector_bundle.keys()]),
                "projector_population": "GENERAL",
            },
        )
        print(json.dumps({"classification": phase_a_label, "selected_mode": selection_mode["mode_label"]}, sort_keys=True), flush=True)
    finally:
        OCMC._release(branch)

    _write_research_note(
        RESEARCH_NOTE,
        {
            "classification": phase_a_label,
            "selection_mode": selection_mode,
            "selection_summary": selection_summary,
            "general_summary": general_camera_summary,
            "safe_summary": safe_camera_summary,
            "eval_summary": eval_summary,
            "environment": environment_manifest,
            "gpu": gpu_manifest,
            "repo": repo_manifest,
        },
    )


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# HELD-OUT SINGLE-MODE CAMERA UTILITY VALIDATION",
        "",
        "## CODE FACT",
        "The selected mode is frozen from the archived 14999 GENERAL mode-wise audit.",
        "The held-out evaluation uses the current formal C1 checkpoint and a disjoint deterministic ray bank.",
        "",
        "## CONFIG FACT",
        f"Selection source: `{SELECTION_OUTPUT_DIR}`.",
        f"Source checkpoint: `{SOURCE_OUTPUT_DIR / 'checkpoints/C1/step-000014999.ckpt'}`.",
        f"Output dir: `{OUTPUT_DIR}`.",
        "",
        "## EXPERIMENTAL FACT",
        f"CONDA_ENV: `{summary['environment']['CONDA_ENV']}`.",
        f"CUDA_VISIBLE_DEVICES: `{summary['environment']['CUDA_VISIBLE_DEVICES']}`.",
        f"GPU: `{summary['gpu']['gpu_name']}`.",
        f"Selected mode: `{summary['selection_mode']['mode_label']}`.",
        f"Frozen selection label: `{summary['classification']}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"GENERAL C_utility mean: `{summary['general_summary']['C_utility_mean']}`.",
        f"GENERAL C_utility median: `{summary['general_summary']['C_utility_median']}`.",
        f"GENERAL camera-positive fraction: `{summary['general_summary']['camera_positive_fraction']}`.",
        f"GENERAL relative utility contribution: `{summary['general_summary']['relative_utility_contribution']}`.",
        f"GENERAL utility-per-energy: `{summary['general_summary']['utility_per_energy']}`.",
        f"M_SAFE C_utility mean: `{summary['safe_summary']['C_utility_mean']}`.",
        f"Eval PSNR full: `{summary['eval_summary']['PSNR_full']}`.",
        f"Eval PSNR minus: `{summary['eval_summary']['PSNR_minus']}`.",
        "",
        "## INFERENCE",
        "The task is read-only. Phase B is only justified if the frozen single-mode utility remains positive and not sample-specific on the held-out bank.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
