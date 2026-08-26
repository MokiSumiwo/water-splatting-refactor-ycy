#!/usr/bin/env python3
"""Read-only mode-wise camera utility vs observability audit for IUI3.

The script loads the formal C1 checkpoints at 5000, 10000, and 14999 from the
existing OCMC run, recomputes the current GENERAL observability basis at each
checkpoint, applies that basis to M_SAFE camera-utility counterfactuals, and
records mode removal as a single-factor diagnostic.

No training, optimizer step, checkpoint mutation, or projector redesign is
performed.
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
from scipy.optimize import linear_sum_assignment
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_ocmc_causal_iui3 as OCMC


PW = MI.PW

EXPERIMENT = "MODE-WISE-CAMERA-UTILITY-VS-OBSERVABILITY-AUDIT"
SCENE = "IUI3-RedSea"
SOURCE_OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
OUTPUT_DIR = Path("outputs/modewise_camera_utility_observability_iui3_20260826")
LOG_DIR = Path("logs/modewise_camera_utility_observability_iui3_20260826")
RESEARCH_NOTE = Path("research_notes/MODEWISE_CAMERA_UTILITY_OBSERVABILITY_IUI3_2026-08-26.md")
CHECKPOINT_STEPS = (5000, 10000, 14999)
CANONICAL_STEP = 5000
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
SAMPLES_PER_VIEW = MI.SAMPLES_PER_VIEW
SAMPLE_SEED = 20260825
SWAP_SEED = 202608255
EPS = 1e-12


@dataclass
class StepBasis:
    step: int
    eigvecs: Tensor
    singular_values: Tensor
    scale: Tensor
    canonical_to_current: List[int]
    abs_cosines: List[float]
    signed_cosines: List[float]
    aligned_eigvecs: Tensor
    aligned_singular_values: Tensor


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


def _environment_manifest(gpu_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu_manifest),
    }


def _source_semantics(repo: Path) -> Dict[str, Any]:
    return {
        "CODE_FACT": True,
        "medium_context_mode": "dir_xy_camera",
        "medium_mlp_input_dimension": 22,
        "medium_mlp_input_parts": {
            "direction_encoding": 16,
            "xy_context": 3,
            "camera_context": 3,
        },
        "camera_context_formula": "(camera_center - scene_center) / (scene_scale + 1e-6) * medium_camera_context_scale",
        "raw_output_channels": {
            "0:3": "B_inf / medium_rgb logits -> sigmoid",
            "3:6": "beta_B / medium_bs logits -> softplus",
            "6:9": "beta_D / medium_attn logits -> softplus",
        },
        "forward_semantics": {
            "clear_object_fullsh_raw": "render.j_raw",
            "direct_object_signal": "render.rgb_object",
            "rgb_object": "render.rgb_object",
            "transmission": "exp(-tau_D.clamp_min(0)).clamp(0,1)",
            "tau_D": "medium_attn * depth",
        },
        "source_hits": {
            "medium_field": _shell_capture(
                repo,
                [
                    "rg",
                    "-n",
                    "camera_context|medium_base_out|medium_rgb|medium_bs|medium_attn|b_inf",
                    "water_splatting/fields/medium_field.py",
                ],
            ),
            "model_forward": _shell_capture(
                repo,
                [
                    "rg",
                    "-n",
                    "clear_object_fullsh_raw|rgb_object|direct_object_signal|transmission|tau_D|rgb_tail",
                    "water_splatting/water_splatting.py",
                ],
            ),
        },
    }


def _source_semantics_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Mode-wise Camera Utility vs Observability",
        "",
        "## CODE FACT",
        "The medium MLP uses `dir_xy_camera` with 16-D direction encoding, 3-D XY/r context, and 3-D camera context.",
        "The raw 9-D output is split into `B_inf/medium_rgb`, `beta_B/medium_bs`, and `beta_D/medium_attn`.",
        "The forward path exposes `clear_object_fullsh_raw` as `render.j_raw` and `rgb_object` / `direct_object_signal` as the attenuated object branch.",
        "",
        "## CONFIG FACT",
        "This audit reuses the formal C1 checkpoints at 5000, 10000, and 14999 from the existing OCMC run.",
        "The observability basis is recomputed from GENERAL rays at each checkpoint and then applied unchanged to M_SAFE utility counterfactuals.",
        "",
        "## EXPERIMENTAL FACT",
        f"Source hits: `{summary.get('source_hits', {})}`.",
        "",
        "## INFERENCE",
        "This file only records code semantics needed to interpret the diagnostic outputs.",
    ]
    return "\n".join(lines) + "\n"


def _shell_capture(repo: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _build_samples(repo: Path, output_dir: Path) -> Tuple[Dict[str, MI.ViewSample], Dict[str, Any], List[Dict[str, Any]]]:
    samples, meta, rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", meta)
    return samples, meta, rows


def _load_swap_bank(source_output_dir: Path, output_dir: Path) -> Dict[str, List[str]]:
    path = source_output_dir / "real_camera_swap_bank.json"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf8"))
    rows = obj.get("rows", {})
    _write_json(output_dir / "real_camera_swap_bank.json", obj)
    return {str(key): [str(item) for item in value] for key, value in rows.items()}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
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


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return CAM._train_records(pipeline)


def _camera_context_bank(model: Any, records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]]) -> Dict[str, Tensor]:
    bank: Dict[str, Tensor] = {}
    for view_id, (_idx, camera, _batch) in records.items():
        bank[view_id] = CAM._camera_context_for(model, camera.to(model.device), neutral=False).detach()
    return bank


def _basis_bundle_from_analysis(analysis: Any) -> Dict[str, Tensor]:
    eigvecs = analysis.eigvecs.detach().float().cpu()
    singular = analysis.singular_values_per_sqrt_ray.detach().float().cpu()
    scale = analysis.scale.detach().float().cpu()
    return {
        "eigvecs": eigvecs,
        "singular_values": singular,
        "scale": scale,
        "v_min": eigvecs[:, 0],
        "v_max": eigvecs[:, -1],
    }


def _match_modes(reference: Tensor, current: Tensor) -> Tuple[List[int], List[float], List[float], Tensor]:
    ref = reference.detach().double()
    cur = current.detach().double()
    sim = torch.abs(ref.T @ cur).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-sim)
    mapping = {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}
    perm = [mapping[idx] for idx in range(ref.shape[1])]
    abs_cosines: List[float] = []
    signed_cosines: List[float] = []
    aligned_cols: List[Tensor] = []
    for ref_idx, cur_idx in enumerate(perm):
        ref_vec = ref[:, ref_idx]
        cur_vec = cur[:, cur_idx]
        dot = float(torch.dot(ref_vec, cur_vec).item())
        abs_cosines.append(abs(dot))
        signed_cosines.append(dot)
        aligned_cols.append(cur_vec * (1.0 if dot >= 0.0 else -1.0))
    aligned = torch.stack(aligned_cols, dim=1).float().cpu()
    return perm, abs_cosines, signed_cosines, aligned


def _mode_ranks(values: Sequence[float], descending: bool = False) -> List[int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=descending)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _basis_rows(
    step: int,
    basis: StepBasis,
    canonical_step: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sigma_ranks = _mode_ranks([float(v) for v in basis.aligned_singular_values.tolist()], descending=False)
    for mode_idx in range(9):
        row = {
            "step": int(step),
            "canonical_step": int(canonical_step),
            "population": "GENERAL",
            "canonical_mode_index": mode_idx,
            "matched_current_mode_index": int(basis.canonical_to_current[mode_idx]),
            "sigma_per_sqrt_ray": float(basis.aligned_singular_values[mode_idx].item()),
            "sigma_rank_in_step": int(sigma_ranks[mode_idx]),
            "abs_cosine_to_canonical": float(basis.abs_cosines[mode_idx]),
            "signed_cosine_to_canonical": float(basis.signed_cosines[mode_idx]),
            "canonical_label": f"mode_{mode_idx:02d}",
        }
        row.update({f"v_{idx}": float(basis.aligned_eigvecs[idx, mode_idx].item()) for idx in range(9)})
        rows.append(row)
    return rows


def _basis_summary_row(step: int, basis: StepBasis) -> Dict[str, Any]:
    return {
        "step": int(step),
        "canonical_step": int(CANONICAL_STEP),
        "mean_abs_cosine_to_canonical": float(sum(basis.abs_cosines) / len(basis.abs_cosines)),
        "min_abs_cosine_to_canonical": float(min(basis.abs_cosines)),
        "mean_signed_cosine_to_canonical": float(sum(basis.signed_cosines) / len(basis.signed_cosines)),
        "sigma_min": float(basis.aligned_singular_values.min().item()),
        "sigma_median": float(torch.median(basis.aligned_singular_values).item()),
        "sigma_max": float(basis.aligned_singular_values.max().item()),
    }


def _project_delta(delta_raw: Tensor, basis_vecs: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    scale = scale.reshape(1, 9).clamp_min(EPS).to(device=delta_raw.device, dtype=delta_raw.dtype)
    basis = basis_vecs.to(device=delta_raw.device, dtype=delta_raw.dtype)
    delta_std = delta_raw.detach().double() / scale.double()
    coeff = delta_std @ basis.double()
    total_energy = delta_std.square().sum(dim=-1, keepdim=True).clamp_min(EPS)
    energy_fractions = coeff.square() / total_energy
    return delta_std, coeff, energy_fractions, scale


def _utility_rows_for_source(
    *,
    branch: Any,
    model: Any,
    projector_bundle: Mapping[str, Any],
    basis: StepBasis,
    sample: MI.ViewSample,
    source_view: str,
    camera: Cameras,
    batch: Mapping[str, Any],
    context_bank: Mapping[str, Tensor],
    swap_bank: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[int, float]]]:
    utility_rows: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    single_mode_delta_means: Dict[str, Dict[int, float]] = {"GENERAL": {}, "M_SAFE": {}}
    with torch.no_grad():
        raw_correct, height, width, _features = OCMC._effective_raw_for_context(
            model,
            camera,
            projector_bundle=projector_bundle,
        )
        out_correct = CAM._render_from_raw(model, camera, raw_correct, height, width)
        pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
        gt_correct = PW._get_gt(model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
        raw_ref_sum = torch.zeros_like(raw_correct.detach())
        ref_count = 0
        for alt_view in swap_bank:
            raw_swap, _h2, _w2, _features_swap = OCMC._effective_raw_for_context(
                model,
                camera,
                camera_context_override=context_bank[alt_view],
                projector_bundle=projector_bundle,
            )
            raw_ref_sum += raw_swap.detach()
            ref_count += 1
            out_swap = CAM._render_from_raw(model, camera, raw_swap, height, width)
            pred_swap = out_swap["pred_image"].reshape(-1, 3).detach().float().cpu()
            gt_swap = PW._get_gt(model, batch, out_swap["background"]).reshape(-1, 3).detach().float().cpu()
            for population in ("GENERAL", "M_SAFE"):
                flat = sample.flat_for(population)
                if flat.numel() == 0:
                    continue
                err_correct = (pred_correct[flat] - gt_correct[flat]).square().mean(dim=-1)
                err_swap = (pred_swap[flat] - gt_swap[flat]).square().mean(dim=-1)
                delta_e = err_swap - err_correct
                utility_rows.append(
                    {
                        "branch": branch.branch,
                        "absolute_step": int(model.step),
                        "population": population,
                        "source_view_id": source_view,
                        "swapped_view_id": alt_view,
                        "sampled_rays": int(flat.numel()),
                        "E_correct_mean": float(err_correct.mean().item()),
                        "E_swap_mean": float(err_swap.mean().item()),
                        "Delta_E_swap_mean": float(delta_e.mean().item()),
                        "Delta_E_swap_median": float(torch.quantile(delta_e.float(), 0.5).item()),
                        "fraction_Delta_E_swap_gt_0": float((delta_e > 0).float().mean().item()),
                        "rgb_change_mean_abs": float((pred_swap[flat] - pred_correct[flat]).abs().mean().item()),
                    }
                )
        raw_ref = raw_ref_sum / max(ref_count, 1)
        raw_correct_flat = raw_correct.reshape(-1, 9).detach()
        raw_ref_flat = raw_ref.reshape(-1, 9).detach()
        for population in ("GENERAL", "M_SAFE"):
            flat = sample.flat_for(population)
            if flat.numel() == 0:
                continue
            flat_dev = flat.to(model.device)
            delta_raw = raw_correct_flat[flat_dev] - raw_ref_flat[flat_dev]
            delta_std, coeff, energy_fractions, scale = _project_delta(
                delta_raw,
                basis.aligned_eigvecs,
                basis.scale,
            )
            total_rms = _rms(delta_std)
            mode_energy_mean = energy_fractions.mean(dim=0)
            basis_vecs = basis.aligned_eigvecs.to(device=model.device, dtype=torch.double)
            for mode_idx in range(9):
                component_std = coeff[:, mode_idx : mode_idx + 1] * basis_vecs[:, mode_idx].reshape(1, 9)
                removed_std = delta_std - component_std
                removed_raw = raw_ref_flat.clone()
                removed_raw[flat_dev] = raw_ref_flat[flat_dev] + (removed_std * scale.double()).to(dtype=raw_ref_flat.dtype)
                with torch.no_grad():
                    out_removed = CAM._render_from_raw(model, camera, removed_raw, height, width)
                pred_removed = out_removed["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                gt_removed = gt_correct[flat]
                err_correct = (pred_correct[flat] - gt_correct[flat]).square().mean(dim=-1)
                err_removed = (pred_removed - gt_removed).square().mean(dim=-1)
                removal_rows.append(
                    {
                        "branch": branch.branch,
                        "absolute_step": int(model.step),
                        "population": population,
                        "source_view_id": source_view,
                        "reference": "mean_of_8_real_swapped_camera_contexts",
                        "sampled_rays": int(flat.numel()),
                        "canonical_mode_index": mode_idx,
                        "canonical_mode_label": f"mode_{mode_idx:02d}",
                        "E_correct_mean": float(err_correct.mean().item()),
                        "E_remove_mode_mean": float(err_removed.mean().item()),
                        "Delta_E_remove_mode_mean": float((err_removed - err_correct).mean().item()),
                        "Delta_E_remove_mode_median": float(torch.quantile((err_removed - err_correct).float(), 0.5).item()),
                        "fraction_remove_mode_improves_or_equal": float((err_removed <= err_correct).float().mean().item()),
                        "delta_std_rms": total_rms,
                        "mode_energy_fraction_mean": float(mode_energy_mean[mode_idx].item()),
                        "mode_energy_fraction_median": float(torch.quantile(energy_fractions[:, mode_idx].float().cpu(), 0.5).item()),
                        "mode_projection_over_random_1over9": float(mode_energy_mean[mode_idx].item() / (1.0 / 9.0)),
                    }
                )
                single_mode_delta_means[population][mode_idx] = float((err_removed - err_correct).mean().item())
                del out_removed, pred_removed, removed_raw
    return utility_rows, removal_rows, single_mode_delta_means


def _pairwise_rows_for_source(
    *,
    branch: Any,
    model: Any,
    sample: MI.ViewSample,
    source_view: str,
    camera: Cameras,
    batch: Mapping[str, Any],
    projector_bundle: Mapping[str, Any],
    context_bank: Mapping[str, Tensor],
    swap_bank: Sequence[str],
    mode_a: int,
    mode_b: int,
    basis: StepBasis,
    single_mode_delta_means: Mapping[str, Mapping[int, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        raw_correct, height, width, _features = OCMC._effective_raw_for_context(
            model,
            camera,
            projector_bundle=projector_bundle,
        )
        out_correct = CAM._render_from_raw(model, camera, raw_correct, height, width)
        pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
        gt_correct = PW._get_gt(model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
        raw_ref_sum = torch.zeros_like(raw_correct.detach())
        ref_count = 0
        for alt_view in swap_bank:
            raw_swap, _h2, _w2, _features_swap = OCMC._effective_raw_for_context(
                model,
                camera,
                camera_context_override=context_bank[alt_view],
                projector_bundle=projector_bundle,
            )
            raw_ref_sum += raw_swap.detach()
            ref_count += 1
        raw_ref = raw_ref_sum / max(ref_count, 1)
        raw_correct_flat = raw_correct.reshape(-1, 9).detach()
        raw_ref_flat = raw_ref.reshape(-1, 9).detach()
        for population in ("GENERAL", "M_SAFE"):
            flat = sample.flat_for(population)
            if flat.numel() == 0:
                continue
            flat_dev = flat.to(model.device)
            delta_raw = raw_correct_flat[flat_dev] - raw_ref_flat[flat_dev]
            delta_std, coeff, energy_fractions, scale = _project_delta(
                delta_raw,
                basis.aligned_eigvecs,
                basis.scale,
            )
            removed_std = delta_std - coeff[:, mode_a : mode_a + 1] * basis.aligned_eigvecs.to(device=model.device, dtype=torch.double)[:, mode_a].reshape(1, 9)
            removed_std = removed_std - coeff[:, mode_b : mode_b + 1] * basis.aligned_eigvecs.to(device=model.device, dtype=torch.double)[:, mode_b].reshape(1, 9)
            removed_raw = raw_ref_flat.clone()
            removed_raw[flat_dev] = raw_ref_flat[flat_dev] + (removed_std * scale.double()).to(dtype=raw_ref_flat.dtype)
            with torch.no_grad():
                out_removed = CAM._render_from_raw(model, camera, removed_raw, height, width)
            pred_removed = out_removed["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
            err_correct = (pred_correct[flat] - gt_correct[flat]).square().mean(dim=-1)
            err_removed = (pred_removed - gt_correct[flat]).square().mean(dim=-1)
            rows.append(
                {
                    "branch": branch.branch,
                    "absolute_step": int(model.step),
                    "population": population,
                    "source_view_id": source_view,
                    "reference": "mean_of_8_real_swapped_camera_contexts",
                    "sampled_rays": int(flat.numel()),
                    "mode_a": int(mode_a),
                    "mode_b": int(mode_b),
                    "pair_label": f"mode_{mode_a:02d}+mode_{mode_b:02d}",
                    "E_correct_mean": float(err_correct.mean().item()),
                    "E_remove_pair_mean": float(err_removed.mean().item()),
                    "Delta_E_remove_pair_mean": float((err_removed - err_correct).mean().item()),
                    "Delta_E_remove_pair_median": float(torch.quantile((err_removed - err_correct).float(), 0.5).item()),
                    "fraction_remove_pair_improves_or_equal": float((err_removed <= err_correct).float().mean().item()),
                    "interaction_excess_mean": float(
                        (err_removed - err_correct).mean().item()
                        - float(single_mode_delta_means.get(population, {}).get(mode_a, float("nan")))
                        - float(single_mode_delta_means.get(population, {}).get(mode_b, float("nan")))
                    ),
                    "mode_energy_fraction_mean_a": float(energy_fractions[:, mode_a].mean().item()),
                    "mode_energy_fraction_mean_b": float(energy_fractions[:, mode_b].mean().item()),
                }
            )
            del out_removed, pred_removed, removed_raw
    return rows


def _aggregate_basis_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return list(rows)


def _aggregate_utility(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["absolute_step"]), str(row["population"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (step, population), group in sorted(grouped.items()):
        out.append(
            {
                "absolute_step": step,
                "population": population,
                "row_count": len(group),
                "Delta_E_swap_mean": _mean(group, "Delta_E_swap_mean"),
                "Delta_E_swap_median": _median(group, "Delta_E_swap_mean"),
                "fraction_Delta_E_swap_gt_0": _mean(group, "fraction_Delta_E_swap_gt_0"),
                "rgb_change_mean_abs": _mean(group, "rgb_change_mean_abs"),
            }
        )
    return out


def _aggregate_mode_removal(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["absolute_step"]), str(row["population"]), int(row["canonical_mode_index"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (step, population, mode_idx), group in sorted(grouped.items()):
        out.append(
            {
                "absolute_step": step,
                "population": population,
                "canonical_mode_index": mode_idx,
                "canonical_mode_label": f"mode_{mode_idx:02d}",
                "row_count": len(group),
                "Delta_E_remove_mode_mean": _mean(group, "Delta_E_remove_mode_mean"),
                "Delta_E_remove_mode_median": _median(group, "Delta_E_remove_mode_mean"),
                "fraction_remove_mode_improves_or_equal": _mean(group, "fraction_remove_mode_improves_or_equal"),
                "mode_energy_fraction_mean": _mean(group, "mode_energy_fraction_mean"),
                "mode_energy_fraction_median": _median(group, "mode_energy_fraction_mean"),
                "mode_projection_over_random_1over9": _mean(group, "mode_projection_over_random_1over9"),
            }
        )
    return out


def _mode_summary_rows(
    basis_rows: Sequence[Mapping[str, Any]],
    mode_removal_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped_basis: Dict[int, List[Mapping[str, Any]]] = {}
    grouped_mode: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    grouped_utility: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in basis_rows:
        grouped_basis.setdefault(int(row["step"]), []).append(row)
    for row in mode_removal_rows:
        grouped_mode.setdefault((int(row["absolute_step"]), str(row["population"])), []).append(row)
    for row in utility_rows:
        grouped_utility.setdefault((int(row["absolute_step"]), str(row["population"])), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    for (step, population), group in sorted(grouped_mode.items()):
        basis = grouped_basis[step]
        sigma_by_mode = {int(row["canonical_mode_index"]): float(row["sigma_per_sqrt_ray"]) for row in basis}
        utility_by_mode = {int(row["canonical_mode_index"]): float(row["Delta_E_remove_mode_mean"]) for row in group}
        sigma_vals = [sigma_by_mode[idx] for idx in range(9)]
        utility_vals = [utility_by_mode[idx] for idx in range(9)]
        sigma_ranks = _mode_ranks(sigma_vals, descending=False)
        utility_ranks = _mode_ranks(utility_vals, descending=True)
        low_sigma_top3_overlap = sum(1 for idx in sorted(range(9), key=lambda i: utility_vals[i], reverse=True)[:3] if sigma_ranks[idx] <= 4)
        high_sigma_top3_overlap = sum(1 for idx in sorted(range(9), key=lambda i: utility_vals[i], reverse=True)[:3] if sigma_ranks[idx] >= 6)
        low_obs_high_util = sum(1 for idx in range(9) if sigma_ranks[idx] <= 4 and utility_ranks[idx] <= 3)
        high_obs_high_util = sum(1 for idx in range(9) if sigma_ranks[idx] >= 6 and utility_ranks[idx] <= 3)
        spearman = _spearman(sigma_vals, utility_vals)
        pearson = _pearson(sigma_vals, utility_vals)
        utility_signal = max(abs(v) for v in utility_vals) if utility_vals else float("nan")
        full_util = _mean([row for row in utility_rows if int(row["absolute_step"]) == step and str(row["population"]) == population], "Delta_E_swap_mean")
        summary_rows.append(
            {
                "absolute_step": step,
                "population": population,
                "mode_count": 9,
                "full_utility_mean": full_util,
                "mode_removal_mean": _mean(group, "Delta_E_remove_mode_mean"),
                "mode_removal_median": _median(group, "Delta_E_remove_mode_mean"),
                "utility_signal_strength": utility_signal,
                "spearman_sigma_vs_utility": spearman,
                "pearson_sigma_vs_utility": pearson,
                "mean_sigma": float(sum(sigma_vals) / len(sigma_vals)),
                "mean_utility": float(sum(utility_vals) / len(utility_vals)),
                "top3_utility_low4_sigma_overlap": int(low_sigma_top3_overlap),
                "top3_utility_high3_sigma_overlap": int(high_sigma_top3_overlap),
                "low_observability_high_utility_count": int(low_obs_high_util),
                "high_observability_high_utility_count": int(high_obs_high_util),
                "utility_positive_modes": int(sum(1 for v in utility_vals if v > 0.0)),
                "utility_negative_modes": int(sum(1 for v in utility_vals if v < 0.0)),
            }
        )
        if step == int(CHECKPOINT_STEPS[-1]):
            evidence_rows.append(
                {
                    "absolute_step": step,
                    "population": population,
                    "sigma_ranks": sigma_ranks,
                    "utility_ranks": utility_ranks,
                    "sigma_values": sigma_vals,
                    "utility_values": utility_vals,
                }
            )
    return summary_rows, evidence_rows


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return float("nan")
    ar = _rank_array(aa)
    br = _rank_array(bb)
    if np.std(ar) < EPS or np.std(br) < EPS:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _rank_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
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
    if values.size > 1:
        ranks = ranks / float(values.size - 1)
    return ranks


def _classify(summary_rows: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {"GENERAL": [], "M_SAFE": []}
    for row in summary_rows:
        grouped.setdefault(str(row["population"]), []).append(row)

    def _metrics(pop: str) -> Dict[str, Any]:
        rows = grouped.get(pop, [])
        corr = [float(r["spearman_sigma_vs_utility"]) for r in rows if math.isfinite(float(r["spearman_sigma_vs_utility"]))]
        utility = [abs(float(r["mode_removal_mean"])) for r in rows if math.isfinite(float(r["mode_removal_mean"]))]
        full_utility = [abs(float(r["full_utility_mean"])) for r in rows if math.isfinite(float(r["full_utility_mean"]))]
        low_overlap = [float(r["top3_utility_low4_sigma_overlap"]) for r in rows]
        high_overlap = [float(r["top3_utility_high3_sigma_overlap"]) for r in rows]
        support_steps = sum(
            1
            for r in rows
            if math.isfinite(float(r["spearman_sigma_vs_utility"]))
            and float(r["spearman_sigma_vs_utility"]) <= -0.25
            and float(r["top3_utility_low4_sigma_overlap"]) >= 2.0
        )
        align_steps = sum(
            1
            for r in rows
            if math.isfinite(float(r["spearman_sigma_vs_utility"]))
            and float(r["spearman_sigma_vs_utility"]) >= 0.25
            and float(r["top3_utility_high3_sigma_overlap"]) >= 2.0
        )
        return {
            "median_spearman": float(np.median(corr)) if corr else float("nan"),
            "mean_abs_mode_removal": float(np.mean(utility)) if utility else float("nan"),
            "mean_abs_full_utility": float(np.mean(full_utility)) if full_utility else float("nan"),
            "mean_low_overlap": float(np.mean(low_overlap)) if low_overlap else float("nan"),
            "mean_high_overlap": float(np.mean(high_overlap)) if high_overlap else float("nan"),
            "support_steps": int(support_steps),
            "align_steps": int(align_steps),
        }

    safe = _metrics("M_SAFE")
    general = _metrics("GENERAL")
    signal = max(safe["mean_abs_mode_removal"], safe["mean_abs_full_utility"], general["mean_abs_mode_removal"], general["mean_abs_full_utility"])
    if not math.isfinite(signal) or signal < 1e-6:
        classification = "MODE_UTILITY_STRUCTURE_NOT_RESOLVED"
    elif safe["support_steps"] >= 2 and general["support_steps"] >= 2 and safe["align_steps"] == 0 and general["align_steps"] == 0:
        classification = "LOW_OBSERVABILITY_HIGH_UTILITY_MODES_SUPPORTED"
    elif safe["align_steps"] >= 2 and general["align_steps"] >= 2 and safe["support_steps"] == 0 and general["support_steps"] == 0:
        classification = "OBSERVABILITY_AND_CAMERA_UTILITY_ALIGNED"
    elif (
        (safe["support_steps"] > 0 and safe["align_steps"] > 0)
        or (general["support_steps"] > 0 and general["align_steps"] > 0)
        or (math.isfinite(safe["median_spearman"]) and abs(safe["median_spearman"]) < 0.25)
        or (math.isfinite(general["median_spearman"]) and abs(general["median_spearman"]) < 0.25)
    ):
        classification = "MIXED_MODE_UTILITY_STRUCTURE"
    else:
        classification = "MODE_UTILITY_STRUCTURE_NOT_RESOLVED"
    evidence = {"M_SAFE": safe, "GENERAL": general, "signal_strength": signal}
    return classification, evidence


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# MODE-WISE CAMERA UTILITY VS OBSERVABILITY AUDIT",
        "",
        "## CODE FACT",
        "The audit reuses the existing OCMC C1 checkpoints and the established 9-D raw-medium basis.",
        "GENERAL rays provide the observability basis; the same basis is then applied to M_SAFE utility counterfactuals.",
        "",
        "## CONFIG FACT",
        f"Checkpoints: `{CHECKPOINT_STEPS}`.",
        f"Source output dir: `{SOURCE_OUTPUT_DIR}`.",
        f"Output dir: `{OUTPUT_DIR}`.",
        "",
        "## EXPERIMENTAL FACT",
        f"CONDA_ENV: `{summary['environment']['CONDA_ENV']}`.",
        f"CUDA_VISIBLE_DEVICES: `{summary['environment']['CUDA_VISIBLE_DEVICES']}`.",
        f"GPU: `{summary['gpu']['gpu_name']}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"Classification: `{summary['classification']}`.",
        f"M_SAFE median Spearman(sigma, utility): `{summary['classification_evidence']['M_SAFE']['median_spearman']}`.",
        f"GENERAL median Spearman(sigma, utility): `{summary['classification_evidence']['GENERAL']['median_spearman']}`.",
        f"M_SAFE signal strength: `{summary['classification_evidence']['M_SAFE']['mean_abs_mode_removal']}`.",
        f"GENERAL signal strength: `{summary['classification_evidence']['GENERAL']['mean_abs_mode_removal']}`.",
        "",
        "## INFERENCE",
        "This diagnostic only tests whether camera utility tracks observability in a stable mode-wise way.",
        "No training, optimizer step, or projector redesign was performed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-output-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    source_output_dir = args.source_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.resolve().mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    environment_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    source_semantics = _source_semantics(repo)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "environment_manifest.json", environment_manifest)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "source_semantics.json", source_semantics)
    (output_dir / "source_semantics.md").write_text(_source_semantics_md(source_semantics), encoding="utf8")

    samples, sample_meta, sample_rows = _build_samples(repo, output_dir)
    swap_bank = _load_swap_bank(source_output_dir, output_dir)

    _write_csv(output_dir / "sampling_rows.csv", sample_rows)
    _write_json(output_dir / "sampling_meta.json", sample_meta)

    branch = OCMC._setup_branch(repo, "C1")
    basis_rows: List[Dict[str, Any]] = []
    basis_summary_rows: List[Dict[str, Any]] = []
    mode_match_rows: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, Any]] = []
    mode_removal_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    step_summaries: List[Dict[str, Any]] = []
    canonical_basis: Optional[Tensor] = None
    canonical_bundle: Optional[StepBasis] = None
    final_pairs: List[Tuple[int, int]] = []

    try:
        for step in CHECKPOINT_STEPS:
            print(f"[modewise-audit] loading checkpoint step={step}", flush=True)
            projector_bundle = OCMC._load_snapshot(branch, source_output_dir, int(step))
            analyses, meta = CAM._analyse_loaded_branch(branch, samples)
            general = analyses["GENERAL"]
            basis = _basis_bundle_from_analysis(general)
            current_vecs = basis["eigvecs"]
            current_sigma = basis["singular_values"]
            current_scale = basis["scale"]
            if canonical_basis is None:
                canonical_basis = current_vecs
                perm = list(range(9))
                abs_cos = [1.0 for _ in range(9)]
                signed_cos = [1.0 for _ in range(9)]
                aligned = current_vecs.clone()
                aligned_sigma = current_sigma.clone()
            else:
                perm, abs_cos, signed_cos, aligned = _match_modes(canonical_basis, current_vecs)
                aligned_sigma = torch.stack([current_sigma[idx] for idx in perm], dim=0).float()
            basis_step = StepBasis(
                step=int(step),
                eigvecs=current_vecs,
                singular_values=current_sigma,
                scale=current_scale,
                canonical_to_current=perm,
                abs_cosines=abs_cos,
                signed_cosines=signed_cos,
                aligned_eigvecs=aligned,
                aligned_singular_values=aligned_sigma,
            )
            if canonical_bundle is None:
                canonical_bundle = basis_step
            basis_rows.extend(_basis_rows(step, basis_step, CANONICAL_STEP))
            basis_summary_rows.append(_basis_summary_row(step, basis_step))
            if step != CANONICAL_STEP:
                for canonical_idx, current_idx in enumerate(perm):
                    mode_match_rows.append(
                        {
                            "reference_step": int(CANONICAL_STEP),
                            "current_step": int(step),
                            "reference_mode_index": int(canonical_idx),
                            "current_mode_index": int(current_idx),
                            "abs_cosine": float(abs_cos[canonical_idx]),
                            "signed_cosine": float(signed_cos[canonical_idx]),
                        }
                    )
            records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
            context_bank = _camera_context_bank(branch.pipeline.model, records)
            for source_view, sample in samples.items():
                _idx, camera, batch = records[source_view]
                camera = camera.to(branch.pipeline.model.device)
                swap_list = swap_bank.get(source_view, [])
                util_rows, rem_rows, single_mode_delta_means = _utility_rows_for_source(
                    branch=branch,
                    model=branch.pipeline.model,
                    projector_bundle=projector_bundle,
                    basis=basis_step,
                    sample=sample,
                    source_view=source_view,
                    camera=camera,
                    batch=batch,
                    context_bank=context_bank,
                    swap_bank=swap_list,
                )
                utility_rows.extend(util_rows)
                mode_removal_rows.extend(rem_rows)
                if step == CHECKPOINT_STEPS[-1]:
                    top_modes = _select_pairwise_modes(rem_rows)
                    if top_modes is not None:
                        final_pairs.extend(top_modes)
                        for mode_a, mode_b in top_modes:
                            pairwise_rows.extend(
                                _pairwise_rows_for_source(
                                    branch=branch,
                                    model=branch.pipeline.model,
                                    sample=sample,
                                    source_view=source_view,
                                    camera=camera,
                                    batch=batch,
                                    projector_bundle=projector_bundle,
                                    context_bank=context_bank,
                                    swap_bank=swap_list,
                                    mode_a=mode_a,
                                    mode_b=mode_b,
                                    basis=basis_step,
                                    single_mode_delta_means=single_mode_delta_means,
                                )
                            )
                del camera, batch
                gc.collect()
                torch.cuda.empty_cache()
            checkpoint_rows.append(
                {
                    "branch": branch.branch,
                    "absolute_step": int(step),
                    "loaded_step": int(branch.pipeline.model.step),
                    "gaussian_count": int(branch.pipeline.model.means.shape[0]),
                    "projector_bundle_step": int(projector_bundle["step"]) if isinstance(projector_bundle, Mapping) and "step" in projector_bundle else int(step),
                    "analysis_branch": meta.get("branch", ""),
                    "analysis_loaded_step": int(meta.get("loaded_step", branch.pipeline.model.step)),
                    "analysis_gaussian_count": int(meta.get("gaussian_count", branch.pipeline.model.means.shape[0])),
                    "general_mean_abs_cosine_to_canonical": float(sum(abs_cos) / len(abs_cos)),
                    "general_min_abs_cosine_to_canonical": float(min(abs_cos)),
                }
            )
            del analyses, meta
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        OCMC._release(branch)

    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_csv(output_dir / "modewise_basis.csv", basis_rows)
    _write_json(output_dir / "modewise_basis.json", {"rows": basis_rows})
    _write_csv(output_dir / "modewise_basis_summary.csv", basis_summary_rows)
    _write_json(output_dir / "modewise_basis_summary.json", {"rows": basis_summary_rows})
    _write_csv(output_dir / "mode_matching.csv", mode_match_rows)
    _write_json(output_dir / "mode_matching.json", {"rows": mode_match_rows})
    _write_csv(output_dir / "correct_context_utility.csv", utility_rows)
    _write_json(output_dir / "correct_context_utility.json", {"rows": utility_rows})
    _write_csv(output_dir / "mode_removal_counterfactual.csv", mode_removal_rows)
    _write_json(output_dir / "mode_removal_counterfactual.json", {"rows": mode_removal_rows})
    _write_csv(output_dir / "pairwise_mode_interaction.csv", pairwise_rows)
    _write_json(output_dir / "pairwise_mode_interaction.json", {"rows": pairwise_rows})

    utility_summary = _aggregate_utility(utility_rows)
    mode_removal_summary = _aggregate_mode_removal(mode_removal_rows)
    mode_summary_rows, evidence_rows = _mode_summary_rows(basis_rows, mode_removal_rows, utility_rows)
    classification, classification_evidence = _classify(mode_summary_rows)

    _write_csv(output_dir / "utility_summary.csv", utility_summary)
    _write_json(output_dir / "utility_summary.json", {"rows": utility_summary})
    _write_csv(output_dir / "mode_removal_summary.csv", mode_removal_summary)
    _write_json(output_dir / "mode_removal_summary.json", {"rows": mode_removal_summary})
    _write_csv(output_dir / "mode_observability_utility_summary.csv", mode_summary_rows)
    _write_json(output_dir / "mode_observability_utility_summary.json", {"rows": mode_summary_rows})

    pairwise_warranted = bool(
        mode_summary_rows
        and any(
            row["population"] == "M_SAFE"
            and math.isfinite(float(row["full_utility_mean"]))
            and math.isfinite(float(row["mode_removal_mean"]))
            and abs(float(row["mode_removal_mean"])) > 1e-5
            and float(row["low_observability_high_utility_count"]) >= 2
            for row in mode_summary_rows
            if int(row["absolute_step"]) == CHECKPOINT_STEPS[-1]
        )
    )
    if not pairwise_warranted:
        pairwise_rows = []
        final_pairs = []
        _write_csv(output_dir / "pairwise_mode_interaction.csv", pairwise_rows)
        _write_json(output_dir / "pairwise_mode_interaction.json", {"rows": pairwise_rows})
    unique_final_pairs = sorted(set(final_pairs))
    final_summary = {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "repo": str(repo),
        "source_output_dir": str(source_output_dir),
        "output_dir": str(output_dir),
        "CONDA_ENV": environment_manifest["CONDA_ENV"],
        "PYTHON_PATH": environment_manifest["PYTHON_PATH"],
        "TORCH_VERSION": environment_manifest["TORCH_VERSION"],
        "CUDA_VISIBLE_DEVICES": environment_manifest["CUDA_VISIBLE_DEVICES"],
        "gpu": dict(gpu_manifest),
        "source_semantics": source_semantics,
        "sample_bank": {
            "sample_seed": sample_meta["sample_seed"],
            "samples_per_view": sample_meta["samples_per_view"],
            "train_view_count": sample_meta["train_view_count"],
            "train_views": sample_meta["train_views"],
            "sha256": _sha256_bytes(json.dumps(sample_meta, sort_keys=True, default=_json_default).encode("utf8")),
        },
        "checkpoint_rows": checkpoint_rows,
        "utility_summary": utility_summary,
        "mode_removal_summary": mode_removal_summary,
        "mode_observability_utility_summary": mode_summary_rows,
        "classification": classification,
        "classification_evidence": classification_evidence,
        "pairwise_warranted": pairwise_warranted,
        "pairwise_mode_pairs_considered": unique_final_pairs,
    }
    _write_json(output_dir / "final_summary.json", final_summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": k, "value": v} for k, v in final_summary.items() if not isinstance(v, (dict, list))])
    _write_json(
        output_dir / "classification.json",
        {
            "classification": classification,
            "classification_evidence": classification_evidence,
            "pairwise_warranted": pairwise_warranted,
        },
    )

    _write_research_note(RESEARCH_NOTE, {
        "environment": environment_manifest,
        "gpu": gpu_manifest,
        "classification": classification,
        "classification_evidence": classification_evidence,
    })
    print(json.dumps({"classification": classification, "pairwise_warranted": pairwise_warranted}, sort_keys=True), flush=True)


def _select_pairwise_modes(mode_rows: Sequence[Mapping[str, Any]]) -> Optional[List[Tuple[int, int]]]:
    grouped: Dict[int, float] = {}
    for row in mode_rows:
        idx = int(row["canonical_mode_index"])
        grouped[idx] = float(row["Delta_E_remove_mode_mean"])
    positives = [idx for idx, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True) if value > 0.0]
    if len(positives) < 2:
        return None
    selected = positives[:3]
    if len(selected) < 2:
        return None
    pairs: List[Tuple[int, int]] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairs.append((selected[i], selected[j]))
    return pairs


if __name__ == "__main__":
    main()
