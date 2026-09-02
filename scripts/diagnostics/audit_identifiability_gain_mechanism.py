#!/usr/bin/env python3
"""Attribute the frozen identifiability causal experiment's RGB gain.

This is a read-only checkpoint audit.  C0 training visibility freezes the
Gaussian sample, the registered ambiguity score, the module tangent, and the
heldout projection boxes before heldout ground truth is accessed.  C1 is used
only as the causal comparison.  No training, backward pass, optimizer step,
checkpoint write, or render write is performed.

The causal runs did not retain split/prune lineage.  Consequently population
statistics and frozen-C0 image regions are primary evidence.  Nearest-geometry
parameter comparisons are retained as an explicitly non-identifying proxy and
cannot, by construction, qualify the strongest mechanism classification.
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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from scipy.spatial import cKDTree
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_gaussian_parameter_identifiability_ocmc as IDENT_AUDIT
from scripts.diagnostics import audit_gaussian_view_consistency_ocmc as VC_AUDIT
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.diagnostics import preflight_gaussian_identifiability_module as MODULE
from scripts.experiments import run_identifiability_module_causal_scene as CAUSAL_RUN
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


EXPERIMENT = "IDENTIFIABILITY_GAIN_MECHANISM_ATTRIBUTION_AUDIT"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "8c587d643585e3822143e7e8e2737502d179187b"
SOURCE_ROOT = REPO_ROOT / "outputs" / "identifiability_module_causal_iui3_20260902"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "identifiability_gain_mechanism_audit_20260902"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "IDENTIFIABILITY_GAIN_MECHANISM_AUDIT_2026-09-02.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
STEPS = (5000, 8000, 10000, 13000, 14999)
FINAL_STEP = 14999
TEMPORAL_SAMPLE_COUNT = 256
FINAL_SAMPLE_COUNT = 512
MATCH_MAX_NORMALIZED_DISTANCE = 0.5
MATCH_MIN_MUTUAL_FRACTION = 0.75
MIN_LOCALIZATION_COUNT = 128
EPS = 1e-12

LEVEL_A = "IDENTIFIABILITY_MECHANISM_SUPPORTED"
LEVEL_B = "POSITIVE_EFFECT_MECHANISM_TENTATIVE"
LEVEL_C = "GENERIC_EFFECT_MORE_LIKELY"
DATA_INSUFFICIENT = "DATA_INSUFFICIENT"

PROTECTED_FILES = (
    "water_splatting/water_splatting.py",
    "water_splatting/_torch_impl.py",
    "water_splatting/fields/gaussian_appearance.py",
    "scripts/diagnostics/audit_gaussian_parameter_identifiability_ocmc.py",
    "scripts/diagnostics/preflight_gaussian_identifiability_module.py",
    "scripts/experiments/run_identifiability_module_causal_scene.py",
)
HISTORICAL_UNTRACKED_HASHES = dict(CAUSAL_RUN.HISTORICAL_UNTRACKED_HASHES)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Tensor):
        cpu = value.detach().cpu()
        return _sanitize(cpu.item() if cpu.numel() == 1 else cpu.tolist())
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            writer.writerow({key: _sanitize(source.get(key, "")) for key in fields})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _ratio(after: float, before: float) -> float:
    return float(after) / max(abs(float(before)), EPS)


def _relative(after: float, before: float) -> float:
    return (float(after) - float(before)) / max(abs(float(before)), EPS)


def _rho(left: Sequence[float], right: Sequence[float], minimum: int = 12) -> Tuple[float, float, int]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    count = int(valid.sum())
    if count < minimum or np.ptp(a[valid]) <= 0.0 or np.ptp(b[valid]) <= 0.0:
        return float("nan"), float("nan"), count
    result = scipy.stats.spearmanr(a[valid], b[valid])
    return float(result.statistic), float(result.pvalue), count


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: float("nan") for key in ("q10", "q25", "q50", "q75", "q90")}
    points = np.quantile(finite, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {key: float(value) for key, value in zip(("q10", "q25", "q50", "q75", "q90"), points)}


def _checkpoint(scene: str, arm: str, step: int) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / arm / f"step-{step:09d}.ckpt"


def _source_checkpoint_rows(scene: str, steps: Sequence[int]) -> List[Dict[str, Any]]:
    registered = {
        (row["arm"], int(row["absolute_step"])): row
        for row in _read_csv(SOURCE_ROOT / scene / "checkpoint_manifest.csv")
    }
    rows = []
    for arm in ("C0", "C1"):
        for step in steps:
            path = _checkpoint(scene, arm, step)
            row = registered[(arm, step)]
            actual = _sha256(path)
            if int(row["size_bytes"]) != path.stat().st_size or row["sha256"] != actual:
                raise RuntimeError(f"source checkpoint provenance mismatch: {path}")
            rows.append(
                {
                    "scene": scene,
                    "arm": arm,
                    "absolute_step": step,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": actual,
                }
            )
    return rows


def _strict_repo() -> Dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected repository state: {branch}@{head}")
    protected = {relative: _sha256(REPO_ROOT / relative) for relative in PROTECTED_FILES}
    historical = {}
    for relative, expected in HISTORICAL_UNTRACKED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"historical protected script changed: {relative}")
        historical[relative] = actual
    return {
        "branch": branch,
        "starting_head": head,
        "git_status": _git("status", "--short"),
        "protected_hashes": protected,
        "historical_untracked_hashes": historical,
    }


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose physical GPU {gpu} only")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {
        "scene": scene,
        "physical_gpu": gpu,
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }


def _model_tensor(payload: Mapping[str, Any], name: str) -> Tensor:
    return payload["model"][f"gauss_params.{name}"].detach().float().cpu()


def _per_item(value: Tensor) -> np.ndarray:
    if value.numel() == 0:
        return np.empty(0, dtype=np.float64)
    return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=-1).numpy()


def _parameter_row(
    scene: str,
    step: int,
    group: str,
    c0: Tensor,
    c1: Tensor,
    direct_target: bool,
    matched_c0: Optional[Tensor] = None,
    matched_c1: Optional[Tensor] = None,
    matched_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    item0 = _per_item(c0)
    item1 = _per_item(c1)
    mean0 = float(item0.mean()) if item0.size else float("nan")
    mean1 = float(item1.mean()) if item1.size else float("nan")
    std0 = float(item0.std()) if item0.size else float("nan")
    std1 = float(item1.std()) if item1.size else float("nan")
    pooled = math.sqrt(0.5 * (std0 * std0 + std1 * std1))
    rms0 = float(torch.sqrt(c0.square().mean())) if c0.numel() else float("nan")
    rms1 = float(torch.sqrt(c1.square().mean())) if c1.numel() else float("nan")
    row: Dict[str, Any] = {
        "scene": scene,
        "absolute_step": step,
        "parameter_group": group,
        "direct_module_target": direct_target,
        "comparison_basis": "population_distribution; no persistent Gaussian lineage",
        "C0_count": int(c0.shape[0]) if c0.ndim else int(c0.numel()),
        "C1_count": int(c1.shape[0]) if c1.ndim else int(c1.numel()),
        "C0_element_rms": rms0,
        "C1_element_rms": rms1,
        "element_rms_ratio_C1_over_C0": _ratio(rms1, rms0),
        "C0_mean_item_norm": mean0,
        "C1_mean_item_norm": mean1,
        "mean_item_norm_relative_delta": _relative(mean1, mean0),
        "standardized_population_shift": (mean1 - mean0) / max(pooled, EPS),
        **{f"C0_{key}": value for key, value in _quantiles(item0).items()},
        **{f"C1_{key}": value for key, value in _quantiles(item1).items()},
    }
    if matched_c0 is not None and matched_c1 is not None and matched_mask is not None:
        valid = torch.from_numpy(matched_mask.astype(np.bool_))
        delta = matched_c1[valid] - matched_c0[valid]
        denominator = torch.linalg.vector_norm(matched_c0[valid].reshape(int(valid.sum()), -1), dim=-1)
        numer = torch.linalg.vector_norm(delta.reshape(int(valid.sum()), -1), dim=-1)
        row.update(
            {
                "nearest_proxy_match_count": int(valid.sum()),
                "nearest_proxy_delta_rms": float(torch.sqrt(delta.square().mean())) if delta.numel() else float("nan"),
                "nearest_proxy_median_relative_item_change": float(
                    torch.median(numer / denominator.clamp_min(EPS))
                ) if delta.numel() else float("nan"),
                "nearest_proxy_is_identity_evidence": False,
            }
        )
    if group in ("medium_mlp", "direction_encoding", "OCMC_projector") and c0.shape == c1.shape:
        delta = c1 - c0
        row.update(
            {
                "exact_coordinate_alignment": True,
                "exact_delta_rms": float(torch.sqrt(delta.square().mean())) if delta.numel() else 0.0,
                "exact_delta_max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                "exact_relative_delta_l2": float(
                    torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(c0).clamp_min(EPS)
                ) if delta.numel() else 0.0,
            }
        )
    return row


def _optimizer_stats(payload: Mapping[str, Any], group: str) -> Dict[str, float]:
    state = payload["optimizers"][group]["state"]
    entries = list(state.values())
    first = entries[0] if entries else {}
    avg = first.get("exp_avg", torch.empty(0)).detach().float()
    avg_sq = first.get("exp_avg_sq", torch.empty(0)).detach().float()
    lr = float(payload["optimizers"][group]["param_groups"][0]["lr"])
    normalized = avg / torch.sqrt(avg_sq).clamp_min(EPS) if avg.numel() else avg
    return {
        "exp_avg_rms": float(torch.sqrt(avg.square().mean())) if avg.numel() else 0.0,
        "exp_avg_sq_mean": float(avg_sq.mean()) if avg_sq.numel() else 0.0,
        "adam_normalized_update_rms": float(torch.sqrt(normalized.square().mean())) if normalized.numel() else 0.0,
        "lr": lr,
        "estimated_update_rms": lr * float(torch.sqrt(normalized.square().mean())) if normalized.numel() else 0.0,
    }


def _spatial_smoothness(means: Tensor, rest: Tensor, selected: Tensor) -> float:
    points = means.numpy()
    ids = selected.numpy()
    distances, neighbors = cKDTree(points).query(points[ids], k=2, workers=-1)
    neighbor_ids = torch.from_numpy(neighbors[:, 1].astype(np.int64))
    delta = rest[selected] - rest[neighbor_ids]
    valid = np.isfinite(distances[:, 1])
    if not bool(valid.any()):
        return float("nan")
    return float(torch.linalg.vector_norm(delta[torch.from_numpy(valid)].reshape(int(valid.sum()), -1), dim=-1).mean())


def _population_metrics(means: Tensor, rest: Tensor, selected: Tensor) -> Dict[str, float]:
    degree1 = rest[:, :3]
    degree2 = rest[:, 3:8]
    degree3 = rest[:, 8:15]
    return {
        "population_count": int(rest.shape[0]),
        "nonDC_energy": float(rest.square().sum(dim=(1, 2)).mean()),
        "degree1_energy": float(degree1.square().sum(dim=(1, 2)).mean()),
        "degree2_energy": float(degree2.square().sum(dim=(1, 2)).mean()),
        "degree3_energy": float(degree3.square().sum(dim=(1, 2)).mean()),
        "high_order_degree2plus3_energy": float(rest[:, 3:].square().sum(dim=(1, 2)).mean()),
        "coefficient_variance": float(torch.var(rest, unbiased=False)),
        "spatial_neighbor_difference": _spatial_smoothness(means, rest, selected),
    }


def _view_response_metrics(response: Tensor, observations: Mapping[str, Tensor]) -> Dict[str, float]:
    mask = observations["mask"]
    visible = response[mask]
    variation = []
    coefficient_of_variation = []
    for index in range(response.shape[0]):
        local = response[index, mask[index]]
        if local.shape[0] < 2:
            continue
        variation.append(float(torch.sqrt((local - local.mean(dim=0)).square().mean())))
        norms = torch.linalg.vector_norm(local, dim=-1)
        coefficient_of_variation.append(
            float(norms.std(unbiased=False) / norms.mean().clamp_min(EPS))
        )
    return {
        "view_dependent_response_rms": float(torch.sqrt(visible.square().mean())) if visible.numel() else float("nan"),
        "mean_cross_view_response_variation": _mean(variation),
        "mean_cross_view_response_norm_cv": _mean(coefficient_of_variation),
    }


def _strata(scores: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    q1, q2 = np.quantile(scores[np.isfinite(scores)], (1.0 / 3.0, 2.0 / 3.0))
    labels = np.full(scores.shape, "middle", dtype=object)
    labels[scores <= q1] = "low"
    labels[scores > q2] = "high"
    return labels, {"low_middle_boundary": float(q1), "middle_high_boundary": float(q2)}


def _local_c0_state(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Mapping[str, Any]]],
    selected: Tensor,
) -> List[Dict[str, Any]]:
    selected_gpu = selected.to(model.device)
    rows: List[Dict[str, Any]] = []
    for _index, view_id, camera, batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        local = torch.nonzero(visible, as_tuple=False).reshape(-1)
        global_ids = selected_gpu[local]
        gt = IDENT_AUDIT.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        rows.append(
            {
                "view_id": view_id,
                "local": local.cpu(),
                "xys": model.xys.detach()[global_ids].float().cpu(),
                "radii": model.radii.detach().reshape(-1)[global_ids].float().cpu(),
                "c0_residual": (pred - gt).square().mean(dim=-1).cpu(),
                "c0_pred": pred.cpu(),
            }
        )
        del outputs, gt, pred
    return rows


def _local_c1_comparison(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Mapping[str, Any]]],
    frozen: Sequence[Mapping[str, Any]],
    count: int,
) -> Dict[str, np.ndarray]:
    error0 = torch.zeros(count, dtype=torch.float64)
    error1 = torch.zeros(count, dtype=torch.float64)
    rgb_change = torch.zeros(count, dtype=torch.float64)
    observed = torch.zeros(count, dtype=torch.int16)
    by_view = {str(row["view_id"]): row for row in frozen}
    for _index, view_id, camera, batch in records:
        source = by_view[str(view_id)]
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = IDENT_AUDIT.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        residual1 = (pred - gt).square().mean(dim=-1)
        pred_change = (pred - source["c0_pred"].to(model.device)).square().mean(dim=-1)
        local = source["local"].long()
        if local.numel():
            xys = source["xys"].to(model.device)
            radii = source["radii"].to(model.device)
            zeros = torch.zeros_like(residual1, dtype=torch.bool)
            c0_box = VC_AUDIT._box_statistics(source["c0_residual"].to(model.device), zeros, xys, radii)[0]
            c1_box = VC_AUDIT._box_statistics(residual1, zeros, xys, radii)[0]
            change_box = VC_AUDIT._box_statistics(pred_change, zeros, xys, radii)[0]
            error0[local] += c0_box.cpu()
            error1[local] += c1_box.cpu()
            rgb_change[local] += change_box.cpu()
            observed[local] += 1
        del outputs, gt, pred, residual1, pred_change
    seen = observed > 0
    result: Dict[str, np.ndarray] = {"observed": observed.numpy(), "seen": seen.numpy()}
    for name, total in (("C0_error", error0), ("C1_error", error1), ("rgb_change_mse", rgb_change)):
        value = torch.full((count,), float("nan"), dtype=torch.float64)
        value[seen] = total[seen] / observed[seen].double()
        result[name] = value.numpy()
    result["error_improvement"] = result["C0_error"] - result["C1_error"]
    return result


def _match_geometry(
    c0_means: Tensor,
    c1_means: Tensor,
    c0_scales: Tensor,
    selected: Tensor,
) -> Dict[str, Any]:
    points0 = c0_means.numpy()
    points1 = c1_means.numpy()
    selected_np = selected.numpy()
    distance, matched = cKDTree(points1).query(points0[selected_np], k=1, workers=-1)
    reverse = cKDTree(points0).query(points1[matched], k=1, workers=-1)[1]
    mutual = reverse == selected_np
    scale = torch.exp(c0_scales[selected]).amax(dim=-1).numpy()
    normalized = distance / np.maximum(scale, EPS)
    reliable = mutual & np.isfinite(normalized) & (normalized <= MATCH_MAX_NORMALIZED_DISTANCE)
    return {
        "matched_ids": matched.astype(np.int64),
        "distance": distance.astype(np.float64),
        "normalized_distance": normalized.astype(np.float64),
        "mutual": mutual,
        "reliable": reliable,
        "mutual_fraction": float(mutual.mean()),
        "reliable_fraction": float(reliable.mean()),
        "lineage_available": False,
        "identity_quality_pass": bool(mutual.mean() >= MATCH_MIN_MUTUAL_FRACTION),
    }


def _group_summary(values: np.ndarray, labels: np.ndarray, group: str) -> Tuple[float, int]:
    selected = (labels == group) & np.isfinite(values)
    return (float(np.mean(values[selected])) if bool(selected.any()) else float("nan"), int(selected.sum()))


def _gradient_rows(scene: str) -> List[Dict[str, Any]]:
    scene_dir = SOURCE_ROOT / scene
    source = {
        arm: {
            (int(row["absolute_step"]), row["parameter_group"]): row
            for row in _read_csv(scene_dir / f"{arm}_gradient_progress.csv")
        }
        for arm in ("C0", "C1")
    }
    rows = []
    for step in (3001,) + STEPS:
        groups = sorted(
            set(group for local_step, group in source["C0"] if local_step == step)
            & set(group for local_step, group in source["C1"] if local_step == step)
        )
        for group in groups:
            c0 = source["C0"][(step, group)]
            c1 = source["C1"][(step, group)]
            grad0 = float(c0["total_loss_gradient_l2"])
            grad1 = float(c1["total_loss_gradient_l2"])
            rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "row_type": "saved_gradient",
                    "parameter_group": group,
                    "direct_module_target": group == "features_rest",
                    "C0_total_loss_gradient_l2": grad0,
                    "C1_total_loss_gradient_l2": grad1,
                    "total_loss_gradient_ratio_C1_over_C0": _ratio(grad1, grad0),
                    "C0_total_loss_gradient_max_abs": float(c0["total_loss_gradient_max_abs"]),
                    "C1_total_loss_gradient_max_abs": float(c1["total_loss_gradient_max_abs"]),
                    "C1_module_direct_gradient_l2": float(c1["module_direct_gradient_l2"]),
                    "C1_module_to_total_gradient_ratio": _ratio(
                        float(c1["module_direct_gradient_l2"]), grad1
                    ),
                    "source": (
                        f"{scene_dir / 'C0_gradient_progress.csv'};"
                        f"{scene_dir / 'C1_gradient_progress.csv'}"
                    ),
                }
            )
    progress = {
        arm: {int(row["absolute_step"]): row for row in _read_csv(scene_dir / f"{arm}_training_progress.csv")}
        for arm in ("C0", "C1")
    }
    for step in sorted(set(progress["C0"]) & set(progress["C1"])):
        c0, c1 = progress["C0"][step], progress["C1"][step]
        base0, base1 = float(c0["L_base"]), float(c1["L_base"])
        rows.append(
            {
                "scene": scene,
                "absolute_step": step,
                "row_type": "saved_training_trajectory",
                "parameter_group": "base_loss",
                "C0_L_base": base0,
                "C1_L_base": base1,
                "L_base_delta_C1_minus_C0": base1 - base0,
                "L_base_ratio_C1_over_C0": _ratio(base1, base0),
                "C1_L_ident_raw": float(c1["L_ident_raw"]),
                "C0_iteration_seconds": float(c0["iteration_seconds"]),
                "C1_iteration_seconds": float(c1["iteration_seconds"]),
                "source": (
                    f"{scene_dir / 'C0_training_progress.csv'};"
                    f"{scene_dir / 'C1_training_progress.csv'}"
                ),
            }
        )
    return rows


def _load_snapshot(branch: Any, scene: str, arm: str, step: int) -> Mapping[str, Any]:
    payload = CAUSAL_RUN._load_snapshot(branch, scene, arm, step, SOURCE_ROOT / scene)
    if (
        payload.get("experiment") != CAUSAL_RUN.EXPERIMENT
        or payload.get("branch") != arm
        or int(payload.get("absolute_step", -1)) != step
        or payload.get("ocmc_bundle") is None
        or payload.get("raoc_state") is not None
    ):
        raise RuntimeError(f"condition provenance mismatch: {scene}/{arm}/{step}")
    return payload


@torch.no_grad()
def worker(
    scene: str,
    gpu: str,
    output_root: Path = OUTPUT_ROOT,
    steps: Sequence[int] = STEPS,
    sample_count: Optional[int] = None,
) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    scene_dir = output_root / "workers" / scene
    if scene_dir.exists() and any(scene_dir.iterdir()):
        raise RuntimeError(f"non-empty worker output: {scene_dir}")
    scene_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    repo_before = _strict_repo()
    worker_script_hash = _sha256(Path(__file__).resolve())
    checkpoint_rows = _source_checkpoint_rows(scene, steps)
    branch = FORMAL._setup_branch(REPO_ROOT, CAUSAL._scene_config(scene), "C0")
    parameter_rows: List[Dict[str, Any]] = []
    tangent_rows: List[Dict[str, Any]] = []
    ambiguity_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    regularization_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    gradient_rows = _gradient_rows(scene)
    scene_source_summary = _read_json(SOURCE_ROOT / "final_summary.json")
    source_scene = next(row for row in scene_source_summary["scene_rows"] if row["scene"] == scene)
    evaluation = {
        (row["arm"], int(row["absolute_step"]), row["split"]): row
        for row in _read_csv(SOURCE_ROOT / scene / "evaluation_metrics.csv")
    }
    mechanism = {
        (row["arm"], int(row["absolute_step"])): row
        for row in _read_csv(SOURCE_ROOT / scene / "mechanism_metrics.csv")
    }
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        for step in steps:
            c0_payload = _load_snapshot(branch, scene, "C0", step)
            c0_state_before = CAUSAL._model_state_hash(model)
            c0_projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            support = VC_AUDIT._support_counts(model, train_records)
            requested = sample_count if sample_count is not None else (
                FINAL_SAMPLE_COUNT if step == FINAL_STEP else TEMPORAL_SAMPLE_COUNT
            )
            selected, _sampling = IDENT_AUDIT._sample_gaussians(scene, step, support, requested)
            ambiguity, _overlap = IDENT_AUDIT._training_metrics(model, train_records, selected, support)
            observations = MODULE._collect_observations(model, train_records, selected)
            selected_gpu = selected.to(model.device)
            rest0 = model.features_rest.detach()[selected_gpu].float().clone()
            dc0 = model.features_dc.detach()[selected_gpu].float().clone()
            opacity0 = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float().clone()
            directions, targets, gates, _direction_summary = MODULE._directions_and_gates(
                rest0, dc0, opacity0, observations
            )
            response0, tangent0, _color0, _transmission0 = MODULE._responses(
                rest0, dc0, opacity0, observations
            )
            frozen_heldout = _local_c0_state(model, eval_records, selected)
            c0_state_after = CAUSAL._model_state_hash(model)
            c0_projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            if c0_state_before != c0_state_after or c0_projector_before != c0_projector_after:
                raise RuntimeError("C0 model or OCMC changed during read-only diagnostics")

            # Retain CPU tensors before loading C1 into the same model object.
            c0_tensors = {
                name: _model_tensor(c0_payload, name)
                for name in ("means", "features_dc", "features_rest", "scales", "quats", "opacities")
            }
            c0_medium = c0_payload["model"]["medium_mlp.tcnn_encoding.params"].detach().float().cpu()
            c0_direction = c0_payload["model"]["direction_encoding.tcnn_encoding.params"].detach().float().cpu()
            c0_ocmc_hash = _tensor_hash(c0_payload["ocmc_bundle"]["projector"])

            c1_payload = _load_snapshot(branch, scene, "C1", step)
            c1_state_before = CAUSAL._model_state_hash(model)
            c1_projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            localization = _local_c1_comparison(model, eval_records, frozen_heldout, int(selected.numel()))
            c1_state_after = CAUSAL._model_state_hash(model)
            c1_projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            if c1_state_before != c1_state_after or c1_projector_before != c1_projector_after:
                raise RuntimeError("C1 model or OCMC changed during read-only diagnostics")
            c1_tensors = {
                name: _model_tensor(c1_payload, name)
                for name in ("means", "features_dc", "features_rest", "scales", "quats", "opacities")
            }
            c1_medium = c1_payload["model"]["medium_mlp.tcnn_encoding.params"].detach().float().cpu()
            c1_direction = c1_payload["model"]["direction_encoding.tcnn_encoding.params"].detach().float().cpu()
            c1_ocmc_hash = _tensor_hash(c1_payload["ocmc_bundle"]["projector"])
            if c0_ocmc_hash != c1_ocmc_hash:
                raise RuntimeError("C0/C1 OCMC projector mismatch")
            if not CAUSAL_RUN._nested_equal(c0_payload["ocmc_bundle"], c1_payload["ocmc_bundle"]):
                raise RuntimeError("C0/C1 complete OCMC bundle mismatch")

            match = _match_geometry(
                c0_tensors["means"], c1_tensors["means"], c0_tensors["scales"], selected
            )
            matched = torch.from_numpy(match["matched_ids"])
            reliable = match["reliable"]
            match_rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "sample_count": int(selected.numel()),
                    "lineage_available": False,
                    "matching_method": "C0-to-C1 nearest mean with reciprocal check",
                    "maximum_normalized_distance": MATCH_MAX_NORMALIZED_DISTANCE,
                    "minimum_mutual_fraction_gate": MATCH_MIN_MUTUAL_FRACTION,
                    "mutual_fraction": match["mutual_fraction"],
                    "reliable_fraction": match["reliable_fraction"],
                    "identity_quality_pass": match["identity_quality_pass"],
                    **{f"distance_{key}": value for key, value in _quantiles(match["distance"]).items()},
                    **{
                        f"normalized_distance_{key}": value
                        for key, value in _quantiles(match["normalized_distance"]).items()
                    },
                }
            )

            rest1 = c1_tensors["features_rest"][matched].to(model.device)
            dc1 = c1_tensors["features_dc"][matched].to(model.device)
            opacity1 = torch.sigmoid(c1_tensors["opacities"][matched]).reshape(-1).to(model.device)
            response1, _tangent1, _color1, _transmission1 = MODULE._responses(
                rest1, dc1, opacity1, observations
            )
            projection0 = (rest0 * directions).sum(dim=(1, 2))
            projection1 = (rest1 * directions).sum(dim=(1, 2))
            delta_rest = rest1 - rest0
            tangent_delta = (delta_rest * directions).sum(dim=(1, 2))
            orth_delta = delta_rest - tangent_delta[:, None, None] * directions
            orth0 = rest0 - projection0[:, None, None] * directions
            orth1 = rest1 - projection1[:, None, None] * directions
            flat_tangent = tangent0.reshape(int(selected.numel()), -1)
            response_coordinate0 = (
                (response0.reshape(int(selected.numel()), -1) * flat_tangent).sum(-1)
                / torch.linalg.vector_norm(flat_tangent, dim=-1).clamp_min(EPS)
            ).abs()
            response_coordinate1 = (
                (response1.reshape(int(selected.numel()), -1) * flat_tangent).sum(-1)
                / torch.linalg.vector_norm(flat_tangent, dim=-1).clamp_min(EPS)
            ).abs()
            scores = np.asarray([float(row["ambiguity_score"]) for row in ambiguity], dtype=np.float64)
            strata, boundaries = _strata(scores)

            group_tensors = (
                ("features_rest", c0_tensors["features_rest"], c1_tensors["features_rest"], True),
                ("features_dc", c0_tensors["features_dc"], c1_tensors["features_dc"], False),
                ("opacity_probability", torch.sigmoid(c0_tensors["opacities"]), torch.sigmoid(c1_tensors["opacities"]), False),
                ("means", c0_tensors["means"], c1_tensors["means"], False),
                ("log_scales", c0_tensors["scales"], c1_tensors["scales"], False),
                ("quats", c0_tensors["quats"], c1_tensors["quats"], False),
            )
            for group, tensor0, tensor1, target in group_tensors:
                source0 = tensor0[selected]
                source1 = tensor1[matched]
                parameter_rows.append(
                    _parameter_row(
                        scene, step, group, tensor0, tensor1, target,
                        source0, source1, reliable,
                    )
                )
            parameter_rows.append(_parameter_row(scene, step, "medium_mlp", c0_medium, c1_medium, False))
            parameter_rows.append(
                _parameter_row(scene, step, "direction_encoding", c0_direction, c1_direction, False)
            )
            parameter_rows.append(
                _parameter_row(
                    scene,
                    step,
                    "OCMC_projector",
                    c0_payload["ocmc_bundle"]["projector"].detach().float().cpu(),
                    c1_payload["ocmc_bundle"]["projector"].detach().float().cpu(),
                    False,
                )
            )

            for group in ("all", "low", "middle", "high"):
                selector = np.ones(len(strata), dtype=np.bool_) if group == "all" else strata == group
                strict = selector & reliable
                count = int(strict.sum())
                tangent_rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "ambiguity_stratum": group,
                        "sample_count": int(selector.sum()),
                        "reliable_match_count": count,
                        "identity_quality_pass": match["identity_quality_pass"],
                        "comparison_is_nearest_geometry_proxy": True,
                        "mean_abs_tangent_coefficient_change": float(tangent_delta.detach().cpu().abs().numpy()[strict].mean()) if count else float("nan"),
                        "mean_orthogonal_coefficient_change_norm": float(torch.linalg.vector_norm(orth_delta.detach().cpu()[torch.from_numpy(strict)].reshape(count, -1), dim=-1).mean()) if count else float("nan"),
                        "mean_tangent_response_C0": float(response_coordinate0.detach().cpu().numpy()[strict].mean()) if count else float("nan"),
                        "mean_tangent_response_C1_proxy": float(response_coordinate1.detach().cpu().numpy()[strict].mean()) if count else float("nan"),
                        "tangent_response_ratio_C1_over_C0": _ratio(
                            float(response_coordinate1.detach().cpu().numpy()[strict].mean()) if count else float("nan"),
                            float(response_coordinate0.detach().cpu().numpy()[strict].mean()) if count else float("nan"),
                        ),
                        "orthogonal_energy_ratio_C1_over_C0": _ratio(
                            float(orth1.detach().cpu().square().sum(dim=(1, 2)).numpy()[strict].mean()) if count else float("nan"),
                            float(orth0.detach().cpu().square().sum(dim=(1, 2)).numpy()[strict].mean()) if count else float("nan"),
                        ),
                    }
                )

            correction = (projection0 - targets).abs() - (projection1 - targets).abs()
            for index, source in enumerate(ambiguity):
                ambiguity_rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "C0_gaussian_id": int(selected[index]),
                        "C1_nearest_gaussian_id": int(match["matched_ids"][index]),
                        "ambiguity_score": scores[index],
                        "ambiguity_stratum": strata[index],
                        **boundaries,
                        "SH_opacity_overlap": float(source["overlap_sh_opacity"]),
                        "module_gate": float(gates[index]),
                        "support_count": int(source["support_count"]),
                        "C0_train_depth_mean": float(source["train_depth_mean"]),
                        "C0_train_footprint_mean": float(source["train_footprint_mean"]),
                        "C0_train_transmission_mean": float(source["train_transmission_mean"]),
                        "C0_train_ocmc_active_magnitude_mean": float(
                            source["train_ocmc_active_magnitude_mean"]
                        ),
                        "nearest_distance": float(match["distance"][index]),
                        "nearest_normalized_distance": float(match["normalized_distance"][index]),
                        "mutual_nearest": bool(match["mutual"][index]),
                        "reliable_proxy_match": bool(reliable[index]),
                        "lineage_available": False,
                        "absolute_tangent_coefficient_change_proxy": float(abs(tangent_delta[index])),
                        "orthogonal_coefficient_change_norm_proxy": float(torch.linalg.vector_norm(orth_delta[index])),
                        "target_residual_reduction_proxy": float(correction[index]),
                        "tangent_response_reduction_proxy": float(response_coordinate0[index] - response_coordinate1[index]),
                        "parameter_change_is_identity_evidence": False,
                    }
                )
                rgb_rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "C0_gaussian_id": int(selected[index]),
                        "ambiguity_score": scores[index],
                        "ambiguity_stratum": strata[index],
                        "heldout_visible_views": int(localization["observed"][index]),
                        "C0_frozen_footprint_MSE": float(localization["C0_error"][index]),
                        "C1_on_C0_frozen_footprint_MSE": float(localization["C1_error"][index]),
                        "error_improvement_C0_minus_C1": float(localization["error_improvement"][index]),
                        "C0_C1_prediction_change_MSE": float(localization["rgb_change_mse"][index]),
                        "localization_basis": "C0 projected box fixed before heldout GT; overlapping-box association",
                        "is_additive_per_gaussian_contribution": False,
                        "uses_C1_geometry_for_region_selection": False,
                    }
                )

            population0 = _population_metrics(
                c0_tensors["means"], c0_tensors["features_rest"], selected
            )
            # Use a deterministic support-stratified C1 sample of equal size for population metrics.
            c1_support = VC_AUDIT._support_counts(model, train_records)
            selected1, _sampling1 = IDENT_AUDIT._sample_gaussians(scene, step, c1_support, requested)
            population1 = _population_metrics(
                c1_tensors["means"], c1_tensors["features_rest"], selected1
            )
            observations1 = MODULE._collect_observations(model, train_records, selected1)
            ids1 = selected1.to(model.device)
            response1_local, _tangent1_local, _color1_local, _transmission1_local = MODULE._responses(
                model.features_rest.detach()[ids1].float(),
                model.features_dc.detach()[ids1].float(),
                torch.sigmoid(model.opacities.detach()).reshape(-1)[ids1].float(),
                observations1,
            )
            view_response0 = _view_response_metrics(response0, observations)
            view_response1 = _view_response_metrics(response1_local, observations1)
            regularization_rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "sampling": (
                        "SH energy/variance use complete arm populations; spatial and view-response metrics "
                        "use arm-local deterministic support-stratified samples"
                    ),
                    **{f"C0_{key}": value for key, value in population0.items()},
                    **{f"C1_{key}": value for key, value in population1.items()},
                    **{
                        f"{key}_ratio_C1_over_C0": _ratio(population1[key], population0[key])
                        for key in population0
                    },
                    **{f"C0_{key}": value for key, value in view_response0.items()},
                    **{f"C1_{key}": value for key, value in view_response1.items()},
                    **{
                        f"{key}_ratio_C1_over_C0": _ratio(view_response1[key], view_response0[key])
                        for key in view_response0
                    },
                    "C0_sample_SH_response_rms": float(torch.sqrt(response0.square().mean())),
                    "C1_nearest_proxy_on_C0_observations_SH_response_rms": float(torch.sqrt(response1.square().mean())),
                    "nearest_proxy_SH_response_ratio": _ratio(
                        float(torch.sqrt(response1.square().mean())),
                        float(torch.sqrt(response0.square().mean())),
                    ),
                    "nearest_proxy_is_identity_evidence": False,
                }
            )
            del observations1, response1_local

            for optimizer_group in (
                "means", "scales", "quats", "features_dc", "features_rest", "opacities", "medium_mlp"
            ):
                stats0 = _optimizer_stats(c0_payload, optimizer_group)
                stats1 = _optimizer_stats(c1_payload, optimizer_group)
                optimizer_row = {
                    "scene": scene,
                    "absolute_step": step,
                    "row_type": "saved_optimizer_state",
                    "parameter_group": optimizer_group,
                    "direct_module_target": optimizer_group == "features_rest",
                    "comparison_basis": "saved Adam state population",
                    **{f"C0_{key}": value for key, value in stats0.items()},
                    **{f"C1_{key}": value for key, value in stats1.items()},
                    **{
                        f"{key}_ratio_C1_over_C0": _ratio(stats1[key], stats0[key])
                        for key in stats0
                        if key != "lr"
                    },
                }
                gradient_rows.append(optimizer_row)
                parameter_rows.append(
                    {
                        **optimizer_row,
                        "parameter_group": f"optimizer:{optimizer_group}",
                    }
                )

            eval0 = evaluation[("C0", step, "eval")]
            eval1 = evaluation[("C1", step, "eval")]
            mech0 = mechanism[("C0", step)]
            mech1 = mechanism[("C1", step)]
            high_improvement, high_count = _group_summary(localization["error_improvement"], strata, "high")
            low_improvement, low_count = _group_summary(localization["error_improvement"], strata, "low")
            reliable_correction = correction.detach().cpu().numpy().copy()
            reliable_correction[~reliable] = np.nan
            high_correction, _ = _group_summary(reliable_correction, strata, "high")
            low_correction, _ = _group_summary(reliable_correction, strata, "low")
            rho_action_rgb = _rho(
                np.where(reliable, np.abs(tangent_delta.detach().cpu().numpy()), np.nan),
                localization["error_improvement"],
            )
            temporal_rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "heldout_PSNR_C0": float(eval0["PSNR"]),
                    "heldout_PSNR_C1": float(eval1["PSNR"]),
                    "heldout_PSNR_delta_C1_minus_C0": float(eval1["PSNR"]) - float(eval0["PSNR"]),
                    "shared_response_energy_C0": float(mech0["SH_opacity_shared_response_energy"]),
                    "shared_response_energy_C1": float(mech1["SH_opacity_shared_response_energy"]),
                    "shared_response_energy_ratio_C1_over_C0": _ratio(
                        float(mech1["SH_opacity_shared_response_energy"]),
                        float(mech0["SH_opacity_shared_response_energy"]),
                    ),
                    "median_tangent_overlap_C0": float(mech0["median_tangent_overlap"]),
                    "median_tangent_overlap_C1": float(mech1["median_tangent_overlap"]),
                    "median_tangent_overlap_delta": float(mech1["median_tangent_overlap"]) - float(mech0["median_tangent_overlap"]),
                    "high_ambiguity_mean_error_improvement": high_improvement,
                    "low_ambiguity_mean_error_improvement": low_improvement,
                    "high_minus_low_error_improvement": high_improvement - low_improvement,
                    "high_localization_count": high_count,
                    "low_localization_count": low_count,
                    "high_ambiguity_mean_target_residual_reduction_proxy": high_correction,
                    "low_ambiguity_mean_target_residual_reduction_proxy": low_correction,
                    "high_minus_low_target_residual_reduction_proxy": high_correction - low_correction,
                    "rho_parameter_action_proxy_vs_RGB_improvement": rho_action_rgb[0],
                    "rho_parameter_action_pvalue": rho_action_rgb[1],
                    "rho_parameter_action_count": rho_action_rgb[2],
                    "parameter_action_uses_reliable_nearest_proxy_only": True,
                    "identity_quality_pass": match["identity_quality_pass"],
                    "OCMC_projector_C0_C1_equal": True,
                    "OCMC_complete_bundle_C0_C1_equal": True,
                }
            )
            print(
                f"[{scene}] {step}: n={len(ambiguity)} match={match['mutual_fraction']:.3f}/"
                f"{match['reliable_fraction']:.3f} dPSNR={float(eval1['PSNR']) - float(eval0['PSNR']):+.6f} "
                f"high-low={high_improvement - low_improvement:+.3e}",
                flush=True,
            )
            del (
                c0_payload, c1_payload, support, selected, ambiguity, observations,
                rest0, dc0, opacity0, directions, targets, gates, response0, tangent0,
                c0_tensors, c1_tensors, rest1, dc1, opacity1, response1, frozen_heldout,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        FORMAL._release(branch)

    _write_csv(scene_dir / "parameter_change.csv", parameter_rows)
    _write_csv(scene_dir / "tangent_change.csv", tangent_rows)
    _write_csv(scene_dir / "ambiguity_conditioned_effect.csv", ambiguity_rows)
    _write_csv(scene_dir / "rgb_improvement_localization.csv", rgb_rows)
    _write_csv(scene_dir / "sh_regularization_analysis.csv", regularization_rows)
    _write_csv(scene_dir / "gradient_update_analysis.csv", gradient_rows)
    _write_csv(scene_dir / "temporal_analysis.csv", temporal_rows)
    _write_csv(scene_dir / "matching_quality.csv", match_rows)
    _write_json(scene_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    result = {
        "experiment": EXPERIMENT,
        "scene": scene,
        "runtime": runtime,
        "steps": list(steps),
        "sample_count_override": sample_count,
        "source_final_heldout_PSNR_delta": source_scene["final_heldout_delta_C1_minus_C0"]["PSNR"],
        "rows": {
            "parameter_change": len(parameter_rows),
            "tangent_change": len(tangent_rows),
            "ambiguity_conditioned_effect": len(ambiguity_rows),
            "rgb_improvement_localization": len(rgb_rows),
            "sh_regularization_analysis": len(regularization_rows),
            "gradient_update_analysis": len(gradient_rows),
            "temporal_analysis": len(temporal_rows),
            "matching_quality": len(match_rows),
        },
        "frozen_forward_only": True,
        "backward_calls": 0,
        "jvp_calls": 0,
        "vjp_calls": 0,
        "optimizer_step_calls": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
        "lineage_available": False,
        "final_localization_finite_count": sum(
            int(row["absolute_step"]) == FINAL_STEP
            and math.isfinite(float(row["error_improvement_C0_minus_C1"]))
            for row in rgb_rows
        ),
        "checkpoint_provenance_valid": len(checkpoint_rows) == 2 * len(steps),
        "worker_script_sha256": worker_script_hash,
        "worker_script_unchanged": _sha256(Path(__file__).resolve()) == worker_script_hash,
        "protected_hashes_unchanged": _strict_repo()["protected_hashes"] == repo_before["protected_hashes"],
        "historical_untracked_hashes_unchanged": (
            _strict_repo()["historical_untracked_hashes"] == repo_before["historical_untracked_hashes"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(scene_dir / "worker_summary.json", result)
    return result


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _scene_attribution(
    scene: str,
    source: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    matching = [row for row in tables["matching_quality"] if int(row["absolute_step"]) == FINAL_STEP][0]
    regularization = [
        row for row in tables["sh_regularization_analysis"] if int(row["absolute_step"]) == FINAL_STEP
    ][0]
    parameters = {
        row["parameter_group"]: row
        for row in tables["parameter_change"]
        if int(row["absolute_step"]) == FINAL_STEP
    }
    localization = [
        row for row in tables["rgb_improvement_localization"] if int(row["absolute_step"]) == FINAL_STEP
    ]
    ambiguity = [
        row for row in tables["ambiguity_conditioned_effect"] if int(row["absolute_step"]) == FINAL_STEP
    ]
    gradient = [
        row for row in tables["gradient_update_analysis"]
        if row["row_type"] == "saved_gradient" and int(row["absolute_step"]) == FINAL_STEP
    ]
    trajectory = [
        row for row in tables["gradient_update_analysis"]
        if row["row_type"] == "saved_training_trajectory"
    ]

    scores = np.asarray([_as_float(row["ambiguity_score"]) for row in localization])
    improvements = np.asarray([_as_float(row["error_improvement_C0_minus_C1"]) for row in localization])
    labels = np.asarray([row["ambiguity_stratum"] for row in localization], dtype=object)
    high_improvement, high_count = _group_summary(improvements, labels, "high")
    low_improvement, low_count = _group_summary(improvements, labels, "low")
    rho_ambiguity_rgb = _rho(scores, improvements)
    action = np.asarray([_as_float(row["absolute_tangent_coefficient_change_proxy"]) for row in ambiguity])
    target_reduction = np.asarray([_as_float(row["target_residual_reduction_proxy"]) for row in ambiguity])
    reliable = np.asarray([_as_bool(row["reliable_proxy_match"]) for row in ambiguity])
    action[~reliable] = np.nan
    target_reduction[~reliable] = np.nan
    high_action, high_action_count = _group_summary(action, labels, "high")
    low_action, low_action_count = _group_summary(action, labels, "low")
    high_target, _ = _group_summary(target_reduction, labels, "high")
    low_target, _ = _group_summary(target_reduction, labels, "low")
    rho_action_rgb = _rho(action, improvements)

    non_target = ("features_dc", "opacity_probability", "means", "log_scales", "quats", "medium_mlp")
    shifts = {
        group: abs(_as_float(parameters[group]["standardized_population_shift"]))
        for group in non_target
        if group in parameters
    }
    rest_shift = abs(_as_float(parameters["features_rest"]["standardized_population_shift"]))
    dominant_group = max({"features_rest": rest_shift, **shifts}, key={"features_rest": rest_shift, **shifts}.get)
    non_target_gradient_ratios = [
        _as_float(row["total_loss_gradient_ratio_C1_over_C0"])
        for row in gradient
        if row["parameter_group"] not in ("features_rest", "direction_encoding")
    ]
    trajectory_relative = [
        abs(_as_float(row["L_base_delta_C1_minus_C0"])) / max(abs(_as_float(row["C0_L_base"])), EPS)
        for row in trajectory
    ]
    temporal_scene = tables["temporal_analysis"]
    high_low_positive_count = sum(
        _as_float(row["high_minus_low_error_improvement"]) > 0.0 for row in temporal_scene
    )
    shared_decrease_count = sum(
        _as_float(row["shared_response_energy_ratio_C1_over_C0"]) < 1.0 for row in temporal_scene
    )

    return {
        "scene": scene,
        "heldout_PSNR_delta": source["final_heldout_delta_C1_minus_C0"]["PSNR"],
        "positive_heldout_PSNR": source["final_heldout_delta_C1_minus_C0"]["PSNR"] >= 0.0,
        "median_C0_ambiguity": float(np.nanmedian(scores)),
        "mean_C0_support": _mean(_as_float(row["support_count"]) for row in ambiguity),
        "mean_C0_depth": _mean(_as_float(row["C0_train_depth_mean"]) for row in ambiguity),
        "mean_C0_footprint": _mean(_as_float(row["C0_train_footprint_mean"]) for row in ambiguity),
        "gaussian_count_C0": source["gaussian_count_C0"],
        "gaussian_count_C1": source["gaussian_count_C1"],
        "features_rest_population_shift_abs_standardized": rest_shift,
        "largest_non_target_population_shift_abs_standardized": max(shifts.values()),
        "population_shift_dominant_group": dominant_group,
        "features_rest_is_largest_population_shift": dominant_group == "features_rest",
        "population_shift_does_not_establish_parameter_identity": True,
        "opacity_mean_ratio_C1_over_C0": source["opacity_mean_ratio_C1_over_C0"],
        "medium_mlp_element_rms_ratio_C1_over_C0": _as_float(
            parameters["medium_mlp"]["element_rms_ratio_C1_over_C0"]
        ),
        "mutual_nearest_fraction": _as_float(matching["mutual_fraction"]),
        "strict_reliable_match_fraction": _as_float(matching["reliable_fraction"]),
        "lineage_identity_quality_pass": _as_bool(matching["identity_quality_pass"]),
        "shared_response_energy_relative_delta": source["shared_energy_relative_delta"],
        "shared_response_energy_decreased": source["shared_energy_decreased"],
        "tangent_overlap_delta": source["tangent_overlap_delta"],
        "tangent_overlap_decreased": source["tangent_overlap_decreased"],
        "SH_nonDC_ratio_C1_over_C0": source["SH_nonDC_ratio_C1_over_C0"],
        "SH_orthogonal_ratio_C1_over_C0": source["SH_orthogonal_ratio_C1_over_C0"],
        "orthogonal_capacity_preserved": source["SH_capacity_preserved"],
        "high_ambiguity_mean_action_proxy": high_action,
        "low_ambiguity_mean_action_proxy": low_action,
        "high_minus_low_action_proxy": high_action - low_action,
        "high_action_count": high_action_count,
        "low_action_count": low_action_count,
        "high_ambiguity_mean_target_reduction_proxy": high_target,
        "low_ambiguity_mean_target_reduction_proxy": low_target,
        "high_minus_low_target_reduction_proxy": high_target - low_target,
        "ambiguity_action_interpretation_allowed": _as_bool(matching["identity_quality_pass"]),
        "high_ambiguity_mean_RGB_error_improvement": high_improvement,
        "low_ambiguity_mean_RGB_error_improvement": low_improvement,
        "high_minus_low_RGB_error_improvement": high_improvement - low_improvement,
        "high_RGB_count": high_count,
        "low_RGB_count": low_count,
        "rho_ambiguity_vs_RGB_improvement": rho_ambiguity_rgb[0],
        "pvalue_ambiguity_vs_RGB_improvement": rho_ambiguity_rgb[1],
        "rho_ambiguity_RGB_count": rho_ambiguity_rgb[2],
        "rho_action_proxy_vs_RGB_improvement": rho_action_rgb[0],
        "pvalue_action_proxy_vs_RGB_improvement": rho_action_rgb[1],
        "rho_action_RGB_count": rho_action_rgb[2],
        "nonDC_energy_ratio_C1_over_C0": _as_float(
            regularization["nonDC_energy_ratio_C1_over_C0"]
        ),
        "high_order_energy_ratio_C1_over_C0": _as_float(
            regularization["high_order_degree2plus3_energy_ratio_C1_over_C0"]
        ),
        "spatial_neighbor_difference_ratio_C1_over_C0": _as_float(
            regularization["spatial_neighbor_difference_ratio_C1_over_C0"]
        ),
        "view_dependent_response_rms_ratio_C1_over_C0": _as_float(
            regularization["view_dependent_response_rms_ratio_C1_over_C0"]
        ),
        "cross_view_response_variation_ratio_C1_over_C0": _as_float(
            regularization["mean_cross_view_response_variation_ratio_C1_over_C0"]
        ),
        "cross_view_response_norm_cv_ratio_C1_over_C0": _as_float(
            regularization["mean_cross_view_response_norm_cv_ratio_C1_over_C0"]
        ),
        "features_rest_total_gradient_ratio_C1_over_C0": _as_float(
            next(row for row in gradient if row["parameter_group"] == "features_rest")[
                "total_loss_gradient_ratio_C1_over_C0"
            ]
        ),
        "median_non_target_gradient_ratio_C1_over_C0": float(np.median(non_target_gradient_ratios)),
        "mean_absolute_relative_base_loss_trajectory_delta": _mean(trajectory_relative),
        "temporal_high_minus_low_RGB_positive_count": high_low_positive_count,
        "temporal_shared_energy_decrease_count": shared_decrease_count,
        "OCMC_independent": source["ocmc_independent"],
    }


def _cross_scene_rho(scene_rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _rho(
        [float(row[key]) for row in scene_rows],
        [float(row["heldout_PSNR_delta"]) for row in scene_rows],
        minimum=4,
    )[0]


def _classify(scene_rows: Sequence[Mapping[str, Any]], workers: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source = _read_json(SOURCE_ROOT / "final_summary.json")
    rgb_supported = bool(
        source["RGB_classification"] == "SUPPORTED"
        and source["positive_heldout_PSNR_scene_count"] >= 3
        and source["mean_heldout_PSNR_delta"] > 0.0
    )
    quality = bool(
        len(scene_rows) == 4
        and all(
            row["checkpoint_provenance_valid"]
            and row["worker_script_unchanged"]
            and row["protected_hashes_unchanged"]
            and row["historical_untracked_hashes_unchanged"]
            and row["final_localization_finite_count"] >= MIN_LOCALIZATION_COUNT
            for row in workers
        )
    )
    shared_count = sum(bool(row["shared_response_energy_decreased"]) for row in scene_rows)
    overlap_count = sum(bool(row["tangent_overlap_decreased"]) for row in scene_rows)
    capacity_count = sum(bool(row["orthogonal_capacity_preserved"]) for row in scene_rows)
    lineage_count = sum(bool(row["lineage_identity_quality_pass"]) for row in scene_rows)
    high_action_count = sum(
        bool(row["ambiguity_action_interpretation_allowed"])
        and float(row["high_minus_low_action_proxy"]) > 0.0
        for row in scene_rows
    )
    high_rgb_count = sum(float(row["high_minus_low_RGB_error_improvement"]) > 0.0 for row in scene_rows)
    positive_ambiguity_rho_count = sum(float(row["rho_ambiguity_vs_RGB_improvement"]) > 0.0 for row in scene_rows)
    significant_positive_ambiguity_rho_count = sum(
        float(row["rho_ambiguity_vs_RGB_improvement"]) > 0.0
        and float(row["pvalue_ambiguity_vs_RGB_improvement"]) < 0.05
        for row in scene_rows
    )
    temporal_rgb_count = sum(int(row["temporal_high_minus_low_RGB_positive_count"]) >= 3 for row in scene_rows)
    shrink_count = sum(float(row["nonDC_energy_ratio_C1_over_C0"]) < 0.95 for row in scene_rows)
    response_suppression_count = sum(
        float(row["view_dependent_response_rms_ratio_C1_over_C0"]) < 0.75 for row in scene_rows
    )
    selective = bool(shared_count >= 3 and overlap_count >= 3 and capacity_count >= 3 and lineage_count >= 3)
    ambiguity_chain = bool(
        high_action_count >= 3 and high_rgb_count >= 3 and significant_positive_ambiguity_rho_count >= 3
    )
    temporal = bool(temporal_rgb_count >= 3)
    generic_more_likely = bool(
        rgb_supported
        and overlap_count <= 2
        and high_rgb_count <= 2
        and significant_positive_ambiguity_rho_count <= 1
        and shrink_count >= 3
        and response_suppression_count >= 3
    )
    if not quality:
        classification = DATA_INSUFFICIENT
    elif rgb_supported and selective and ambiguity_chain and temporal:
        classification = LEVEL_A
    elif generic_more_likely:
        classification = LEVEL_C
    elif rgb_supported:
        classification = LEVEL_B
    else:
        classification = DATA_INSUFFICIENT
    reasons = []
    if not selective:
        reasons.append(
            f"selective correction is not established: shared energy {shared_count}/4, tangent overlap "
            f"{overlap_count}/4, capacity {capacity_count}/4, lineage-quality {lineage_count}/4"
        )
    if not ambiguity_chain:
        reasons.append(
            f"ambiguity-to-action-to-RGB chain is incomplete: interpretable high-action {high_action_count}/4, "
            f"high-vs-low RGB localization {high_rgb_count}/4, significant positive ambiguity/RGB rho "
            f"{significant_positive_ambiguity_rho_count}/4"
        )
    if not temporal:
        reasons.append(f"high-vs-low RGB localization is temporally consistent in {temporal_rgb_count}/4 scenes")
    if shrink_count:
        reasons.append(f"non-DC SH energy falls by more than 5% in {shrink_count}/4 scenes")
    if response_suppression_count:
        reasons.append(
            f"view-dependent SH response RMS is below 75% of C0 in {response_suppression_count}/4 scenes"
        )
    reasons.append(
        "causal checkpoints contain no persistent split/prune lineage; nearest-geometry parameter changes are proxy evidence only"
    )
    return {
        "classification": classification,
        "RGB_effect_supported": rgb_supported,
        "analysis_quality_pass": quality,
        "selective_correction_supported": selective,
        "ambiguity_action_RGB_chain_supported": ambiguity_chain,
        "temporal_mechanism_supported": temporal,
        "generic_effect_more_likely_gate": generic_more_likely,
        "shared_energy_decrease_scene_count": shared_count,
        "tangent_overlap_decrease_scene_count": overlap_count,
        "orthogonal_capacity_preserved_scene_count": capacity_count,
        "lineage_identity_quality_scene_count": lineage_count,
        "high_ambiguity_stronger_action_scene_count": high_action_count,
        "high_ambiguity_better_RGB_scene_count": high_rgb_count,
        "positive_ambiguity_RGB_rho_scene_count": positive_ambiguity_rho_count,
        "significant_positive_ambiguity_RGB_rho_scene_count": significant_positive_ambiguity_rho_count,
        "temporally_consistent_RGB_localization_scene_count": temporal_rgb_count,
        "systematic_SH_shrink_scene_count": shrink_count,
        "view_response_suppression_scene_count": response_suppression_count,
        "classification_reasons": reasons,
        "decision_rule": {
            LEVEL_A: "quality and positive RGB; >=3/4 selective response/capacity/lineage, ambiguity-action-RGB, and temporal support",
            LEVEL_B: "quality and positive RGB, but neither complete Level-A attribution nor Level-C generic pattern",
            LEVEL_C: (
                "positive RGB; overlap improves <=2/4, high-ambiguity RGB improves <=2/4, significant "
                "ambiguity/RGB rho occurs <=1/4, and SH energy/response suppress in >=3/4"
            ),
            DATA_INSUFFICIENT: "artifact/provenance/sample quality fails, or positive RGB effect is absent",
        },
    }


def _required_answers(
    classification: Mapping[str, Any], scene_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    dominant_rest = sum(bool(row["features_rest_is_largest_population_shift"]) for row in scene_rows)
    overlap = int(classification["tangent_overlap_decrease_scene_count"])
    capacity = int(classification["orthogonal_capacity_preserved_scene_count"])
    high_action = int(classification["high_ambiguity_stronger_action_scene_count"])
    high_rgb = int(classification["high_ambiguity_better_RGB_scene_count"])
    shrink = int(classification["systematic_SH_shrink_scene_count"])
    temporal = int(classification["temporally_consistent_RGB_localization_scene_count"])
    positive_action_rgb = sum(float(row["rho_action_proxy_vs_RGB_improvement"]) > 0.0 for row in scene_rows)
    positive_scenes = sum(bool(row["positive_heldout_PSNR"]) for row in scene_rows)
    final = str(classification["classification"])
    return [
        {
            "question": 1,
            "answer": "yes at direct-gradient and population-distribution level, not at Gaussian identity level",
            "evidence": (
                f"features_rest has the largest standardized population shift in {dominant_rest}/4 scenes, "
                "and it is the only direct module-gradient target; however split/prune lineage was not retained"
            ),
        },
        {
            "question": 2,
            "answer": "partially",
            "evidence": (
                f"shared response energy decreases in {classification['shared_energy_decrease_scene_count']}/4, "
                f"while formal tangent overlap decreases in only {overlap}/4 scenes"
            ),
        },
        {
            "question": 3,
            "answer": "not consistently",
            "evidence": f"registered orthogonal/non-DC capacity gate passes in {capacity}/4 scenes",
        },
        {
            "question": 4,
            "answer": "not identifiable from these checkpoints",
            "evidence": (
                f"the high-ambiguity nearest-match action proxy is interpretable in {high_action}/4 scenes; "
                f"reliable lineage quality passes in {classification['lineage_identity_quality_scene_count']}/4"
            ),
        },
        {
            "question": 5,
            "answer": "not consistently",
            "evidence": f"frozen-C0 high-ambiguity regions improve more than low ambiguity in {high_rgb}/4 scenes",
        },
        {
            "question": 6,
            "answer": "yes, generic SH regularization is more likely than the claimed selective mechanism",
            "evidence": (
                f"non-DC energy falls by more than 5% in {shrink}/4 scenes, view-dependent response RMS is "
                f"below 75% of C0 in {classification['view_response_suppression_scene_count']}/4, while tangent "
                f"overlap is directionally favorable in only {overlap}/4"
            ),
        },
        {
            "question": 7,
            "answer": "yes, optimization trajectory is a credible co-explanation",
            "evidence": (
                "the direct gradient is routed only to features_rest, but saved total gradients, Adam states, "
                "base-loss trajectories, topology, and non-target parameters diverge downstream"
            ),
        },
        {
            "question": 8,
            "answer": "not reliably",
            "evidence": (
                f"nearest-match action/RGB Spearman direction is positive in {positive_action_rgb}/4 scenes, "
                "but no scene passes the pre-fixed identity-quality gate"
            ),
        },
        {
            "question": 9,
            "answer": "no",
            "evidence": f"heldout PSNR improves in {positive_scenes}/4 scenes and Curasao is negative",
        },
        {
            "question": 10,
            "answer": "no",
            "evidence": f"ambiguity-conditioned RGB localization is directionally stable in {temporal}/4 scenes",
        },
        {
            "question": 11,
            "answer": "only at intervention and projector-state level, not final learned-state level",
            "evidence": (
                "OCMC projector bundles are equal and direct module gradients exclude medium/OCMC, but medium_mlp "
                "follows the changed optimization trajectory (final exact relative L2 drift 0.155-0.271)"
            ),
        },
        {
            "question": 12,
            "answer": "no",
            "evidence": (
                "the +0.036275 dB mean causal gain is real, but the ambiguity -> selective correction -> "
                "representation change -> novel-view improvement chain does not close"
            ),
        },
        {
            "question": 13,
            "answer": "selective ambiguity-conditioned correction and causal RGB localization",
            "evidence": (
                "persistent lineage is absent, tangent overlap is inconsistent, and frozen projected boxes are "
                "associative rather than additive per-Gaussian render attribution"
            ),
        },
        {
            "question": 14,
            "answer": "no, not as an identifiability module",
            "evidence": (
                f"classification is {final}; archive the intervention's small positive RGB effect, but reject and "
                "do not tune or present the identifiability attribution"
            ),
        },
        {
            "question": 15,
            "answer": "no",
            "evidence": "the task explicitly forbids a third module; resolve attribution before new mechanism search",
        },
    ]


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    rows = summary["scene_rows"]
    lines = [
        "# Identifiability Gain Mechanism Attribution Audit",
        "",
        "Date: 2026-09-02",
        f"Classification: `{classification['classification']}`",
        "",
        "## Objective And Boundary",
        "",
        "This read-only audit asks whether the completed C1 RGB gain comes from the claimed SH-opacity identifiability correction or from a generic regularization/optimization side effect. It uses only the completed matched C0/C1 checkpoints and saved trajectories. OCMC stays on, RAOC stays off, and no model training, backward pass, optimizer step, checkpoint write, render write, strength sweep, or module change is performed.",
        "",
        "## Evidence Levels",
        "",
        "Arm-level RGB and saved distribution metrics are causal because C0/C1 used matched starts, cameras, and updates. Ambiguity-conditioned RGB localization freezes C0 training-only ambiguity, sample, and projected heldout boxes before heldout GT, but overlapping projected boxes are associative and not additive per-Gaussian render decomposition. Parameter differences after topology divergence use reciprocal nearest geometry only; they are explicitly proxy evidence because persistent split/prune lineage was not saved.",
        "",
        "## Scene Results",
        "",
        "| Scene | dPSNR | shared rel | overlap d | nonDC | orth | mutual/strict match | high-low RGB MSE improvement | rho(A, improvement) | SH shrink |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['heldout_PSNR_delta']:+.6f} | "
            f"{row['shared_response_energy_relative_delta']:+.6f} | {row['tangent_overlap_delta']:+.6f} | "
            f"{row['SH_nonDC_ratio_C1_over_C0']:.3f} | {row['SH_orthogonal_ratio_C1_over_C0']:.3f} | "
            f"{row['mutual_nearest_fraction']:.3f}/{row['strict_reliable_match_fraction']:.3f} | "
            f"{row['high_minus_low_RGB_error_improvement']:+.3e} | "
            f"{row['rho_ambiguity_vs_RGB_improvement']:+.3f} | "
            f"{row['nonDC_energy_ratio_C1_over_C0']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Attribution Result",
            "",
            f"The causal RGB effect remains supported. Ambiguity/improvement rank direction is positive in {classification['positive_ambiguity_RGB_rho_scene_count']}/4 scenes but significant in {classification['significant_positive_ambiguity_RGB_rho_scene_count']}/4; high-ambiguity regions improve more than low-ambiguity regions in only {classification['high_ambiguity_better_RGB_scene_count']}/4. Shared SH-opacity response energy decreases in {classification['shared_energy_decrease_scene_count']}/4 scenes, while formal tangent overlap decreases in only {classification['tangent_overlap_decrease_scene_count']}/4 and orthogonal capacity is preserved in {classification['orthogonal_capacity_preserved_scene_count']}/4.",
            "",
            f"No scene passes the fixed Gaussian identity-quality gate. Non-DC SH energy falls by more than 5% in {classification['systematic_SH_shrink_scene_count']}/4 scenes, and arm-local view-dependent SH response RMS falls below 75% of C0 in {classification['view_response_suppression_scene_count']}/4. Final medium MLP states also drift by 15%-27% relative L2 despite identical OCMC projector bundles. This systematic capacity suppression together with absent selective localization makes generic regularization/trajectory effects more likely than the claimed identifiability mechanism. Level C does not identify one unique generic cause; it rejects H1 as the reasonable attribution for the gain. The defensible result is `{classification['classification']}`.",
            "",
            "## Cross-Scene Interpretation",
            "",
            f"Across the four scenes, Spearman rho between heldout dPSNR and final median C0 ambiguity is {summary['cross_scene_correlations']['median_C0_ambiguity_vs_dPSNR']:+.3f}; rho with high-minus-low localized improvement is {summary['cross_scene_correlations']['high_minus_low_RGB_vs_dPSNR']:+.3f}; rho with non-DC energy ratio is {summary['cross_scene_correlations']['nonDC_energy_ratio_vs_dPSNR']:+.3f}. With n=4 these are descriptive only. Curasao's negative dPSNR cannot be uniquely explained by ambiguity, Gaussian count, view coverage, tangent overlap, or SH energy from this sample; no post-hoc scene-specific mechanism is selected.",
            "",
            "## Required Answers",
            "",
        ]
    )
    for item in summary["required_answers"]:
        lines.append(f"{item['question']}. **{item['answer']}** {item['evidence']}.")
    lines.extend(
        [
            "",
            "## Minimum Next Diagnostic",
            "",
            "Do not tune or retain this intervention as an identifiability module, and do not search for a third module. Archive the small RGB effect as generic/unattributed. If H1 must be revisited despite this classification, the minimum decisive diagnostic is one exact protocol replication that records immutable Gaussian parent/descendant IDs through every split, duplicate, and prune, then reports ambiguity-stratified tangent/orthogonal updates and additive heldout contribution deltas for those lineages. Without lineage, further nearest-neighbor checkpoint analysis cannot recover the missing causal link.",
            "",
            "## Integrity",
            "",
            f"The audit read {summary['checkpoint_count']} hashed causal checkpoints and produced {summary['total_localization_rows']} frozen-region rows. All workers report zero backward, JVP, VJP, optimizer, training, checkpoint-write, and render-write calls. OCMC projector equality and protected-source hashes pass in every scene.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    source = _read_json(SOURCE_ROOT / "final_summary.json")
    workers = [_read_json(output_root / "workers" / scene / "worker_summary.json") for scene in SCENES]
    if not all(
        row["frozen_forward_only"]
        and row["backward_calls"] == 0
        and row["jvp_calls"] == 0
        and row["vjp_calls"] == 0
        and row["optimizer_step_calls"] == 0
        and row["training_steps"] == 0
        and row["checkpoint_writes"] == 0
        and row["render_writes"] == 0
        and row["checkpoint_provenance_valid"]
        and row["worker_script_unchanged"]
        and row["protected_hashes_unchanged"]
        and row["historical_untracked_hashes_unchanged"]
        for row in workers
    ):
        raise RuntimeError("worker integrity gate failed")
    names = (
        "parameter_change",
        "tangent_change",
        "ambiguity_conditioned_effect",
        "rgb_improvement_localization",
        "sh_regularization_analysis",
        "gradient_update_analysis",
        "temporal_analysis",
        "matching_quality",
    )
    tables: Dict[str, List[Dict[str, str]]] = {name: [] for name in names}
    checkpoint_rows: List[Dict[str, Any]] = []
    scene_tables: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for scene in SCENES:
        local = {}
        for name in names:
            rows = _read_csv(output_root / "workers" / scene / f"{name}.csv")
            local[name] = rows
            tables[name].extend(rows)
        scene_tables[scene] = local
        checkpoint_rows.extend(_read_json(output_root / "workers" / scene / "checkpoint_manifest.json")["rows"])
    source_rows = {row["scene"]: row for row in source["scene_rows"]}
    scene_rows = [_scene_attribution(scene, source_rows[scene], scene_tables[scene]) for scene in SCENES]
    classification = _classify(scene_rows, workers)
    cross_scene = {
        "median_C0_ambiguity_vs_dPSNR": _cross_scene_rho(scene_rows, "median_C0_ambiguity"),
        "high_minus_low_RGB_vs_dPSNR": _cross_scene_rho(scene_rows, "high_minus_low_RGB_error_improvement"),
        "nonDC_energy_ratio_vs_dPSNR": _cross_scene_rho(scene_rows, "nonDC_energy_ratio_C1_over_C0"),
        "tangent_overlap_delta_vs_dPSNR": _cross_scene_rho(scene_rows, "tangent_overlap_delta"),
        "gaussian_count_C0_vs_dPSNR": _cross_scene_rho(scene_rows, "gaussian_count_C0"),
        "mean_support_vs_dPSNR": _cross_scene_rho(scene_rows, "mean_C0_support"),
        "mean_footprint_vs_dPSNR": _cross_scene_rho(scene_rows, "mean_C0_footprint"),
    }
    answers = _required_answers(classification, scene_rows)
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "source_causal_classification": source["classification"],
        "source_RGB_classification": source["RGB_classification"],
        "source_mean_heldout_PSNR_delta": source["mean_heldout_PSNR_delta"],
        "source_positive_heldout_PSNR_scene_count": source["positive_heldout_PSNR_scene_count"],
        "scene_rows": scene_rows,
        "cross_scene_correlations": cross_scene,
        "required_answers": answers,
        "checkpoint_count": len(checkpoint_rows),
        "total_localization_rows": len(tables["rgb_improvement_localization"]),
        "worker_summaries": workers,
        "evidence_boundary": {
            "arm_level_metrics": "causal matched C0/C1 evidence",
            "RGB_localization": "frozen-C0 overlapping-box association; not additive Gaussian contribution",
            "parameter_change": "population evidence plus nearest-geometry proxy; no persistent lineage",
        },
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
        "next_unique_task": (
            "PRESERVE_UNRESOLVED_IDENTIFIABILITY_EFFECT; NO_THIRD_MODULE; lineage-aware exact replication only if attribution is required"
        ),
    }
    cross_rows: List[Dict[str, Any]] = list(scene_rows)
    cross_rows.append(
        {
            "scene": "POOLED_DESCRIPTIVE",
            "heldout_PSNR_delta": source["mean_heldout_PSNR_delta"],
            "positive_heldout_PSNR": f"{source['positive_heldout_PSNR_scene_count']}/4",
            **{f"rho_{key}": value for key, value in cross_scene.items()},
            "classification": classification["classification"],
        }
    )
    for name in names:
        _write_csv(output_root / f"{name}.csv", tables[name])
    _write_csv(output_root / "cross_scene_summary.csv", cross_rows)
    _write_json(output_root / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_json(output_root / "classification.json", classification)
    _write_json(output_root / "final_summary.json", summary)
    _research_note(summary)
    return summary


def preflight(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    repo = _strict_repo()
    sources = []
    for scene in SCENES:
        scene_manifest = _source_checkpoint_rows(scene, STEPS)
        sources.extend(scene_manifest)
    result = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "source_root": str(SOURCE_ROOT),
        "source_checkpoints": sources,
        "source_final_summary": str(SOURCE_ROOT / "final_summary.json"),
        "source_final_summary_sha256": _sha256(SOURCE_ROOT / "final_summary.json"),
        "worker_scene_gpu": SCENE_GPUS,
        "steps": list(STEPS),
        "sample_count": {"temporal": TEMPORAL_SAMPLE_COUNT, "final": FINAL_SAMPLE_COUNT},
        "frozen_definition": {
            "ambiguity": "registered maximum pairwise overlap times total response sensitivity",
            "ambiguity_source_function": "audit_gaussian_parameter_identifiability_ocmc._training_metrics",
            "module_tangent_source_function": "preflight_gaussian_identifiability_module._directions_and_gates",
            "sample": "C0 training support-stratified sample, support>=2",
            "strata": "C0 empirical ambiguity tertiles",
            "heldout_region": "C0 projected bounding box, frozen before heldout GT access",
        },
        "matching_quality_gate": {
            "method": "C0-to-C1 nearest mean with reciprocal check",
            "maximum_normalized_distance": MATCH_MAX_NORMALIZED_DISTANCE,
            "minimum_mutual_fraction": MATCH_MIN_MUTUAL_FRACTION,
            "lineage_available": False,
            "nearest_matches_are_identity_evidence": False,
        },
        "classification_labels": [LEVEL_A, LEVEL_B, LEVEL_C, DATA_INSUFFICIENT],
        "disk_before": subprocess.check_output(["df", "-B1", str(REPO_ROOT)], text=True).splitlines()[-1],
        "disk_cleanup": {"deleted": [], "reason": "new audit writes only small CSV/JSON/log artifacts"},
        "frozen_forward_only": True,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_json(output_root / "preflight.json", result)
    return result


def launch(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    before = preflight(output_root)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for scene, gpu in SCENE_GPUS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["CONDA_DEFAULT_ENV"] = "water_splatting"
        command = [
            str(PYTHON), str(Path(__file__).resolve()), "--worker", "--scene", scene,
            "--gpu", gpu, "--output-root", str(output_root),
        ]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
        processes.append((scene, gpu, process, handle))
    failures = []
    statuses = {}
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        statuses[scene] = code
        if code != 0:
            failures.append({"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")})
    _write_json(output_root / "worker_status.json", statuses)
    if failures:
        _write_json(output_root / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"mechanism attribution workers failed: {failures}")
    result = aggregate(output_root)
    return {"preflight": before, "summary": result}


def _parse_steps(value: str) -> Tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item)
    if not steps or any(step not in STEPS for step in steps):
        raise argparse.ArgumentTypeError(f"steps must be comma-separated members of {STEPS}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    parser.add_argument("--steps", type=_parse_steps, default=STEPS)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        if args.sample_count is not None and args.sample_count < 12:
            parser.error("--sample-count must be at least 12")
        result = worker(args.scene, args.gpu, output_root, args.steps, args.sample_count)
    elif args.preflight:
        result = preflight(output_root)
    elif args.aggregate:
        result = aggregate(output_root)
    else:
        result = launch(output_root)
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
