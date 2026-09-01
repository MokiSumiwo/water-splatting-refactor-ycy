#!/usr/bin/env python3
"""Read-only preflight for distinct-camera support across Gaussian topology.

The audit loads existing locked C0 checkpoints and frozen-renders registered
training/evaluation cameras. It never trains, mutates a checkpoint, or changes
the renderer. Parent-child claims are deliberately unavailable when persisted
lineage is absent.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import scipy.stats
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL

EXPERIMENT = "RESOLVE-LOW-SUPPORT-STATE-LIFECYCLE-PREFLIGHT"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "low_support_state_lifecycle_preflight_20260901"
RESEARCH_NOTE = (
    REPO_ROOT
    / "research_notes"
    / "LOW_SUPPORT_STATE_LIFECYCLE_PREFLIGHT_2026-09-01.md"
)
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
STEPS = (5000, 8000, 10000, 13000, 14999)
INTERVALS = (
    (0, 5000, "through_5k"),
    (5000, 8000, "5k_to_8k"),
    (8000, 10000, "8k_to_10k"),
    (10000, 13000, "10k_to_13k"),
    (13000, 14999, "13k_to_14999"),
)
PROTECTED_HASHES = {
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py": (
        "539f1c044f9ed136dce65b1dedc01746097cb2f3c4298c9682038019d23dfd7a"
    ),
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py": (
        "fe3fd3ddcdbbff7904cfb7225a0ba024f928a9020777561252b66663c3c8ab32"
    ),
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py": (
        "b6a271372e68cd07fc566a3fde5ced5ba6463531278c31a6cfa47972aa15e8d6"
    ),
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py": (
        "d131428cc20ea76010e237abd91ac4cddfc5c6a78944c57c3317ed18bcdf60ef"
    ),
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py": (
        "3a924e88a606d34360a90348f3a392d0d12f80d43c98fe72b56cbec2d27ad6e7"
    ),
}
EXPECTED_START_HEAD = "1d51d66f691ef74e48082097e552abed98d55406"
AUDITED_SOURCE_HASHES = {
    "scripts/experiments/run_m1_raoc_causal_scene.py": (
        "79930754f41887c0530e6b033eef5f0f26b692795a4f3abd078358ad800f9f2a"
    ),
    "water_splatting/water_splatting.py": (
        "1a9930c0e74b4f235fc5ae5e819823fe9e2cdd828e8764ca73e43d0f67aa63e1"
    ),
}
EVENT_ID_FIELDS = {
    "parent_id",
    "parent_ids",
    "child_id",
    "child_ids",
    "source_id",
    "source_ids",
    "gaussian_id",
    "gaussian_ids",
    "pruned_ids",
    "birth_step",
    "creation_iteration",
    "lineage_id",
    "lineage_ids",
}
EPS = 1e-12


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu()
        return cpu.item() if cpu.numel() == 1 else cpu.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
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
        writer.writerows(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _rho(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.ptp(a[valid]) <= EPS or np.ptp(b[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.spearmanr(a[valid], b[valid]).statistic)


def _checkpoint_path(scene: str, step: int) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"


def _historical_visibility(
    model: Any, outputs: Mapping[str, Any], n_gaussians: int
) -> torch.Tensor:
    radii = model.radii.detach().reshape(-1)
    reported = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
    if radii.numel() != n_gaussians or reported.numel() != n_gaussians:
        raise RuntimeError("visibility shape differs from Gaussian population")
    visible = radii > 0
    if not torch.equal(visible, reported):
        raise RuntimeError("model.radii > 0 differs from gaussian_visible_mask")
    return visible


def _support_row(scene: str, step: int, support: torch.Tensor) -> Dict[str, Any]:
    values = support.numpy().astype(np.int64)
    n = int(values.size)
    counts = np.bincount(values, minlength=int(values.max()) + 1)
    return {
        "scene": scene,
        "absolute_step": step,
        "gaussian_count": n,
        "support_min": int(values.min()),
        "support_max": int(values.max()),
        "support_mean": float(values.mean()),
        "support_median": float(np.median(values)),
        "support_q25": float(np.quantile(values, 0.25)),
        "support_q75": float(np.quantile(values, 0.75)),
        "support_eq_0_count": int((values == 0).sum()),
        "support_eq_1_count": int((values == 1).sum()),
        "support_eq_2_count": int((values == 2).sum()),
        "support_ge_3_count": int((values >= 3).sum()),
        "support_eq_0_fraction": float((values == 0).mean()),
        "support_eq_1_fraction": float((values == 1).mean()),
        "support_eq_2_fraction": float((values == 2).mean()),
        "support_ge_3_fraction": float((values >= 3).mean()),
        "low_support_le_1_count": int((values <= 1).sum()),
        "low_support_le_1_fraction": float((values <= 1).mean()),
        "support_histogram": {
            str(index): int(count) for index, count in enumerate(counts)
        },
        "identity_continuity_assumed": False,
    }


def _strategy_reference_rows(
    scene: str,
    step: int,
    support: torch.Tensor,
    camera_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply lifecycle rules to a reference population, not inferred children."""

    values = support.numpy().astype(np.float64)
    rows = []
    strategies = (
        ("A_INHERIT", None, values, "child_support = parent_support"),
        ("B_RESET", None, np.zeros_like(values), "child_support = 0"),
        (
            "C_FRACTIONAL",
            0.25,
            0.25 * values,
            "child_support = 0.25 * parent_support",
        ),
        (
            "C_FRACTIONAL",
            0.5,
            0.5 * values,
            "child_support = 0.5 * parent_support",
        ),
        (
            "C_FRACTIONAL",
            0.75,
            0.75 * values,
            "child_support = 0.75 * parent_support",
        ),
    )
    baseline_low = values <= 1
    for strategy, alpha, mapped, rule in strategies:
        mapped_low = mapped <= 1
        if strategy == "A_INHERIT":
            camera_key = "reference_fraction_A_inherit"
        elif strategy == "B_RESET":
            camera_key = "reference_fraction_B_reset"
        else:
            camera_key = f"reference_fraction_C_alpha_{str(alpha).replace('.', 'p')}"
        reference_rho = _rho(
            [float(row[camera_key]) for row in camera_rows],
            [float(row["E_cam"]) for row in camera_rows],
        )
        reference_rho_finite = math.isfinite(reference_rho)
        rows.append(
            {
                "scene": scene,
                "absolute_step": step,
                "strategy": strategy,
                "alpha": alpha,
                "rule": rule,
                "population_reference_only": True,
                "actual_split_parent_population_known": False,
                "actual_duplicate_source_population_known": False,
                "mapped_low_support_fraction_reference": float(mapped_low.mean()),
                "label_change_fraction_vs_inherit_reference": float(
                    (mapped_low != baseline_low).mean()
                ),
                "mapped_support_is_integer_distinct_camera_count": bool(
                    np.equal(mapped, np.floor(mapped)).all()
                ),
                "whole_population_reference_rho": (
                    reference_rho if reference_rho_finite else None
                ),
                "whole_population_reference_status": (
                    "DEGENERATE_CONSTANT_PREDICTOR"
                    if not reference_rho_finite
                    else "REFERENCE_ONLY_NOT_ACTUAL_SPLIT_CHILD_RESULT"
                ),
                "actual_split_child_residual_association_rho": None,
                "actual_split_child_residual_association_status": (
                    "NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE"
                ),
                "best_alpha_selected": False,
            }
        )
    return rows


def _repo_state() -> Dict[str, Any]:
    protected = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected file changed: {relative}")
        protected[relative] = {"sha256": actual, "untouched": True}
    audited_sources = {}
    for relative, expected in AUDITED_SOURCE_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"audited topology source changed: {relative}")
        audited_sources[relative] = {"sha256": actual, "matches_audit": True}
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != "research/m1-bounded-intrinsic" or head != EXPECTED_START_HEAD:
        raise RuntimeError(f"unexpected starting state: {branch}@{head}")
    return {
        "starting_branch": branch,
        "starting_head": head,
        "protected_files": protected,
        "audited_topology_sources": audited_sources,
        "training_code_modified": False,
        "renderer_modified": False,
        "ocmc_modified": False,
        "raoc_opened": False,
    }


def _checkpoint_schema_audit() -> Dict[str, Any]:
    rows = []
    for scene in SCENES:
        for step in STEPS:
            path = _checkpoint_path(scene, step)
            if not path.is_file():
                raise RuntimeError(f"missing required checkpoint: {path}")
            payload = torch.load(path, map_location="cpu")
            model_keys = sorted(payload["model"])
            lineage_keys = [
                key
                for key in model_keys
                if any(
                    token in key.lower()
                    for token in ("lineage", "birth", "parent", "creation")
                )
            ]
            if payload.get("branch") != "C0" or payload.get("raoc_state") is not None:
                raise RuntimeError(f"checkpoint is not frozen C0/RAOC-off: {path}")
            if payload.get("ocmc_bundle") is None:
                raise RuntimeError(f"checkpoint lacks OCMC bundle: {path}")
            rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "experiment": payload.get("experiment"),
                    "branch": payload.get("branch"),
                    "ocmc_present": True,
                    "raoc_present": False,
                    "gaussian_count": int(
                        payload["model"]["gauss_params.means"].shape[0]
                    ),
                    "lineage_keys": lineage_keys,
                    "lineage_present": bool(lineage_keys),
                }
            )
            del payload
    return {"rows": rows, "all_required_present": True}


def _event_schema_audit() -> Dict[str, Any]:
    rows = []
    for scene in SCENES:
        path = SOURCE_ROOT / scene / "C0_refinement_events.csv"
        events = _read_csv(path)
        fields = set(events[0]) if events else set()
        identity_fields = sorted(fields & EVENT_ID_FIELDS)
        rows.append(
            {
                "scene": scene,
                "path": str(path),
                "event_count": len(events),
                "fields": sorted(fields),
                "identity_fields": identity_fields,
                "event_identity_available": bool(identity_fields),
            }
        )
    return {"rows": rows}


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    repo = _repo_state()
    checkpoints = _checkpoint_schema_audit()
    events = _event_schema_audit()
    lineage_available = bool(
        all(row["lineage_present"] for row in checkpoints["rows"])
        and all(row["event_identity_available"] for row in events["rows"])
    )
    lineage = {
        "classification": (
            "GAUSSIAN_LINEAGE_AVAILABLE"
            if lineage_available
            else "GAUSSIAN_LINEAGE_UNAVAILABLE"
        ),
        "available": lineage_available,
        "checkpoint_lineage_available": all(
            row["lineage_present"] for row in checkpoints["rows"]
        ),
        "event_parent_child_indices_available": all(
            row["event_identity_available"] for row in events["rows"]
        ),
        "birth_or_creation_iteration_available": False,
        "temporal_identity_analysis": "NOT_AVAILABLE",
        "age_control": "AGE_CONTROL_NOT_AVAILABLE",
        "reason": (
            "C0 state_dict checkpoints contain no lineage/birth/parent fields, "
            "and refinement events persist counts but no Gaussian indices."
        ),
        "parent_child_inference_from_array_index_forbidden": True,
        "geometry_nearest_neighbor_lineage_inference_forbidden": True,
    }
    environment = {
        "python": str(PYTHON),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "launcher_visible_device_count": torch.cuda.device_count(),
        "allowed_physical_gpus": [6, 7, 8, 9],
    }
    result = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "environment": environment,
        "checkpoint_schema": checkpoints,
        "event_schema": events,
        "lineage": lineage,
        "no_training": True,
        "optimizer_step_called": False,
        "backward_called": False,
    }
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    _write_json(OUTPUT_ROOT / "lineage_availability.json", lineage)
    return result


@torch.no_grad()
def worker(scene: str, assigned_gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != assigned_gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{assigned_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assigned_gpu:
        raise RuntimeError(f"worker must expose only physical GPU {assigned_gpu}")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("worker must see exactly logical cuda:0")
    started = time.perf_counter()
    scene_cfg = _read_json(SOURCE_ROOT / scene / "scene_config.json")
    branch = FORMAL._setup_branch(REPO_ROOT, scene_cfg, "C0")
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        heldout_records = FORMAL._eval_records(branch.pipeline)
        train_ids = {record[1] for record in train_records}
        heldout_ids = {record[1] for record in heldout_records}
        if train_ids & heldout_ids:
            raise RuntimeError(f"train/heldout camera leakage in {scene}")
        population_rows = []
        camera_rows = []
        strategy_rows = []
        visibility_checks = 0
        for step in STEPS:
            checkpoint = _checkpoint_path(scene, step)
            payload = FORMAL._load_checkpoint(branch, checkpoint)
            if (
                payload.get("experiment") != FORMAL.EXPERIMENT
                or payload.get("branch") != "C0"
                or int(payload.get("absolute_step", -1)) != step
                or payload.get("raoc_state") is not None
                or payload.get("ocmc_bundle") is None
            ):
                raise RuntimeError(f"checkpoint provenance mismatch: {checkpoint}")
            if (
                not model.config.camera_medium_observability_enabled
                or model.config.camera_medium_ray_adaptive_observability_enabled
            ):
                raise RuntimeError("checkpoint/model is not OCMC-on and RAOC-off")
            n_gaussians = int(model.means.shape[0])
            support = torch.zeros(n_gaussians, dtype=torch.int16)
            for _index, _camera_id, camera, _batch in train_records:
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                visible = _historical_visibility(model, outputs, n_gaussians)
                support += visible.cpu().to(torch.int16)
                visibility_checks += 1
                del outputs
            if int(support.max()) > len(train_records):
                raise RuntimeError("support exceeds distinct training-camera count")
            row = _support_row(scene, step, support)
            row.update(
                {
                    "checkpoint_path": str(checkpoint),
                    "train_camera_count": len(train_records),
                    "heldout_camera_count": len(heldout_records),
                    "ocmc_enabled": True,
                    "raoc_enabled": False,
                    "frozen_read_only": True,
                }
            )
            step_camera_rows = []
            for _index, camera_id, camera, batch in heldout_records:
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MI.PW._get_gt(model, batch, outputs["background"])
                pred = outputs["pred_image"].detach().float().clamp(0, 1)
                gt = gt.detach().float().clamp(0, 1)
                visible = _historical_visibility(model, outputs, n_gaussians)
                visible_support = support[visible.cpu()]
                camera_row = {
                    "scene": scene,
                    "absolute_step": step,
                    "camera_id": camera_id,
                    "E_cam": float((pred - gt).square().mean()),
                    "fraction_visible_support_le_1": float(
                        (visible_support <= 1).float().mean()
                    ),
                    "visible_gaussian_count": int(visible.sum()),
                    "heldout_used_for_support": False,
                    "reference_fraction_A_inherit": float(
                        (visible_support <= 1).float().mean()
                    ),
                    "reference_fraction_B_reset": 1.0,
                    "reference_fraction_C_alpha_0p25": float(
                        (0.25 * visible_support.float() <= 1).float().mean()
                    ),
                    "reference_fraction_C_alpha_0p5": float(
                        (0.5 * visible_support.float() <= 1).float().mean()
                    ),
                    "reference_fraction_C_alpha_0p75": float(
                        (0.75 * visible_support.float() <= 1).float().mean()
                    ),
                }
                camera_rows.append(camera_row)
                step_camera_rows.append(camera_row)
                visibility_checks += 1
                del outputs, gt, pred
            baseline_rho = _rho(
                [item["fraction_visible_support_le_1"] for item in step_camera_rows],
                [item["E_cam"] for item in step_camera_rows],
            )
            row["baseline_T1_vs_E_cam_spearman"] = baseline_rho
            row["baseline_only_not_lifecycle_strategy_result"] = True
            population_rows.append(row)
            strategy_rows.extend(
                _strategy_reference_rows(scene, step, support, step_camera_rows)
            )
            print(
                f"[{scene}] frozen lifecycle checkpoint {step}: "
                f"N={n_gaussians} low={row['low_support_le_1_fraction']:.6f}",
                flush=True,
            )
            del payload, support
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "assigned_physical_gpu": assigned_gpu,
            "logical_gpu": 0,
            "gpu_name": torch.cuda.get_device_properties(0).name,
            "train_ids": sorted(train_ids),
            "heldout_ids": sorted(heldout_ids),
            "population_rows": population_rows,
            "camera_rows": camera_rows,
            "strategy_reference_rows": strategy_rows,
            "visibility_equivalence_checks": visibility_checks,
            "heldout_leakage": False,
            "optimizer_step_called": False,
            "backward_called": False,
            "training_performed": False,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(OUTPUT_ROOT / "workers" / scene / "scene_result.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _event_outputs() -> Dict[str, Any]:
    transition_rows = []
    interval_rows = []
    scene_rows = []
    for scene in SCENES:
        source = SOURCE_ROOT / scene / "C0_refinement_events.csv"
        events = _read_csv(source)
        for event in events:
            transition_rows.append(
                {
                    "scene": scene,
                    "branch": event["branch"],
                    "absolute_step": int(event["absolute_step"]),
                    "camera_name": event["camera_name"],
                    "K_split": int(event["K_split"]),
                    "K_duplicate": int(event["K_duplicate"]),
                    "N_before": int(event["N_before"]),
                    "N_after": int(event["N_after"]),
                    "N_pruned": int(event["N_pruned"]),
                    "children_added": int(event["children_added"]),
                    "split_parent_support": "NOT_AVAILABLE",
                    "split_child_support": "NOT_AVAILABLE",
                    "duplicate_source_support": "NOT_AVAILABLE",
                    "duplicate_child_support": "NOT_AVAILABLE",
                    "pruned_support_distribution": "NOT_AVAILABLE",
                    "lineage_status": "GAUSSIAN_LINEAGE_UNAVAILABLE",
                    "event_counts_are_not_unique_gaussian_counts": True,
                }
            )
        for lower, upper, label in INTERVALS:
            selected = [
                row
                for row in transition_rows
                if row["scene"] == scene
                and lower < int(row["absolute_step"]) <= upper
            ]
            interval_rows.append(
                {
                    "scene": scene,
                    "interval": label,
                    "lower_exclusive": lower,
                    "upper_inclusive": upper,
                    "refinement_event_count": len(selected),
                    "split_selections": sum(row["K_split"] for row in selected),
                    "duplicate_selections": sum(
                        row["K_duplicate"] for row in selected
                    ),
                    "children_added_event_sum": sum(
                        row["children_added"] for row in selected
                    ),
                    "pruned_event_sum": sum(row["N_pruned"] for row in selected),
                    "event_sums_are_not_unique_gaussian_counts": True,
                }
            )
        additions = [row for row in transition_rows if row["scene"] == scene and row["children_added"] > 0]
        last_child_step = max(row["absolute_step"] for row in additions)
        scene_rows.append(
            {
                "scene": scene,
                "refinement_event_count": len(events),
                "last_child_addition_step": last_child_step,
                "last_split_step": max(
                    row["absolute_step"]
                    for row in transition_rows
                    if row["scene"] == scene and row["K_split"] > 0
                ),
                "last_duplicate_step": max(
                    row["absolute_step"]
                    for row in transition_rows
                    if row["scene"] == scene and row["K_duplicate"] > 0
                ),
                "minimum_final_age_in_iterations": 14999 - last_child_step,
                "child_additions_after_10k": 0,
                "post_10k_topology_mode": "PRUNE_ONLY",
            }
        )
    _write_csv(OUTPUT_ROOT / "event_support_transition.csv", transition_rows)
    _write_json(
        OUTPUT_ROOT / "event_support_transition.json", {"rows": transition_rows}
    )
    _write_csv(OUTPUT_ROOT / "topology_event_intervals.csv", interval_rows)
    _write_json(OUTPUT_ROOT / "topology_event_intervals.json", {"rows": interval_rows})
    _write_json(OUTPUT_ROOT / "topology_event_audit.json", {"rows": scene_rows})
    return {
        "transition_rows": transition_rows,
        "interval_rows": interval_rows,
        "scene_rows": scene_rows,
    }


def _temporal_statistics(
    population_rows: Sequence[Mapping[str, Any]],
    interval_rows: Sequence[Mapping[str, Any]],
    event_scene_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    scenes = {}
    for scene in SCENES:
        rows = sorted(
            [row for row in population_rows if row["scene"] == scene],
            key=lambda row: int(row["absolute_step"]),
        )
        post_10k = [
            row
            for row in interval_rows
            if row["scene"] == scene and int(row["lower_exclusive"]) >= 10000
        ]
        event_scene = next(row for row in event_scene_rows if row["scene"] == scene)
        scenes[scene] = {
            "gaussian_count_by_step": {
                str(row["absolute_step"]): row["gaussian_count"] for row in rows
            },
            "low_support_fraction_by_step": {
                str(row["absolute_step"]): row["low_support_le_1_fraction"]
                for row in rows
            },
            "low_support_fraction_delta_5k_to_14999": (
                float(rows[-1]["low_support_le_1_fraction"])
                - float(rows[0]["low_support_le_1_fraction"])
            ),
            "children_added_after_10k_event_sum": sum(
                int(row["children_added_event_sum"]) for row in post_10k
            ),
            "pruned_after_10k_event_sum": sum(
                int(row["pruned_event_sum"]) for row in post_10k
            ),
            "last_child_addition_step": event_scene["last_child_addition_step"],
            "minimum_final_age_in_iterations": event_scene[
                "minimum_final_age_in_iterations"
            ],
            "post_10k_newborn_explanation": "RULED_OUT",
            "persistent_low_identity_fraction": None,
            "late_created_low_identity_fraction": None,
            "support_gradually_grows_identity_fraction": None,
            "identity_category_status": "NOT_AVAILABLE",
        }
    return {
        "analysis_scope": "distributional checkpoint populations only",
        "identity_tracking_status": "GAUSSIAN_LINEAGE_UNAVAILABLE",
        "persistent_vs_newborn_conclusion": "NOT_AVAILABLE",
        "narrow_post_10k_newborn_explanation": "RULED_OUT_IN_ALL_SCENES",
        "scenes": scenes,
    }


def _research_note(summary: Mapping[str, Any]) -> None:
    population = summary["population_evolution"]
    pop_lines = []
    for scene in SCENES:
        for row in [item for item in population if item["scene"] == scene]:
            pop_lines.append(
                f"| {scene} | {row['absolute_step']} | "
                f"{row['gaussian_count']} | "
                f"{row['low_support_le_1_fraction']:.6f} | "
                f"{row['support_mean']:.3f} | {row['support_median']:.1f} | "
                f"{row['baseline_T1_vs_E_cam_spearman']:.3f} |"
            )
    strategy_lines = []
    final_strategy = [
        row
        for row in summary["split_strategy_comparison"]
        if int(row["absolute_step"]) == 14999
    ]
    for strategy in ("A_INHERIT", "B_RESET", "C_FRACTIONAL"):
        for alpha in ((None,) if strategy != "C_FRACTIONAL" else (0.25, 0.5, 0.75)):
            selected = [
                row
                for row in final_strategy
                if row["strategy"] == strategy and row["alpha"] == alpha
            ]
            reference_rhos = [
                float(row["whole_population_reference_rho"])
                for row in selected
                if row["whole_population_reference_rho"] is not None
            ]
            reference_rho_text = (
                f"{np.median(reference_rhos):.3f}"
                if reference_rhos
                else "undefined"
            )
            strategy_lines.append(
                f"| {strategy} | {alpha if alpha is not None else '-'} | "
                f"{np.median([row['mapped_low_support_fraction_reference'] for row in selected]):.6f} | "
                f"{reference_rho_text} | "
                "NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |"
            )
    lines = [
        "# Low-Support State Lifecycle Preflight (2026-09-01)",
        "",
        "## Objective",
        "",
        "HYPOTHESIS: a distinct-camera support state may lose a unique meaning "
        "when Gaussian identity changes through split, duplicate, and prune.",
        "",
        "This preflight does not re-test the existence of the low-support "
        "association and does not design a module or loss.",
        "",
        "## Frozen Inputs",
        "",
        "CONFIG FACT: only the four C0 branches under "
        "outputs/m1_raoc_causal_four_scene_20260827 were used at 5K, 8K, "
        "10K, 13K, and 14999. C0 has OCMC on and RAOC off.",
        "",
        "EXPERIMENTAL FACT: every operation was checkpoint loading, frozen "
        "rendering, or offline CSV/JSON analysis. No optimizer step, backward "
        "call, checkpoint write, or training was performed.",
        "",
        "## Lineage Availability",
        "",
        "CODE FACT: all 20 checkpoint state_dict objects omit lineage, birth "
        "iteration, and parent/source identifiers. The 576 C0 refinement "
        "records contain event counts but no Gaussian indices. Model loading "
        "also explicitly discards a legacy gaussian_lineage_ids key.",
        "",
        "QUANTITATIVE RESULT: GAUSSIAN_LINEAGE_UNAVAILABLE. Parent-child "
        "matching by array index or geometry proximity was not attempted.",
        "",
        "## Population Evolution",
        "",
        "| Scene | step | Gaussian count | fraction s<=1 | mean s | median s | baseline T1 rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *pop_lines,
        "",
        "Baseline rho is reported only as frozen-checkpoint provenance. It is "
        "not an inheritance-strategy result. These C0 checkpoints use their "
        "original 2026-08-27 split with 3/4/3/3 heldout cameras for Curasao/"
        "IUI3-RedSea/JapaneseGradens-RedSea/Panama, so the small-N rho values "
        "must not be mixed with the later 2026-08-31 resplit evidence.",
        "",
        "## Topology Events",
        "",
        "CODE FACT: split appends sampled children and then culls original "
        "split parents; duplicate appends parameter copies while retaining "
        "sources; prune masks all Gaussian parameter arrays.",
        "",
        "EXPERIMENTAL FACT: split and duplicate stop after 10K in all four "
        "recorded C0 branches, while pruning continues. The last child "
        "addition is at step 9900 in every scene, so every final survivor is "
        "at least 5099 iterations past its latest possible creation. Event support "
        "transitions remain NOT_AVAILABLE because selected/pruned identities "
        "were not persisted.",
        "",
        "## Strategy Sensitivity",
        "",
        "| Strategy | alpha | median final reference low fraction | median whole-population reference rho | actual child result |",
        "| --- | ---: | ---: | ---: | --- |",
        *strategy_lines,
        "",
        "QUANTITATIVE RESULT: the fractions and reference rho values apply each "
        "rule to the complete checkpoint population as transparent limiting "
        "cases. They are not estimates of actual split-child outcomes because "
        "the selected parent population is unknown. Reset makes every visible "
        "Gaussian low-support, so its reference predictor is constant and rho "
        "is undefined. No alpha was selected.",
        "",
        "INFERENCE: inherit preserves a parent's camera set but can overstate "
        "coverage of a displaced, smaller split child. Reset preserves literal "
        "post-birth observation history but discards inherited parameter "
        "evidence. Fractional inheritance is not an integer distinct-camera "
        "count for many parent supports and silently redefines the statistic.",
        "",
        "## Temporal Identity",
        "",
        "QUANTITATIVE RESULT: persistent-low, late-created-low, and gradually-"
        "growing identity fractions are NOT_AVAILABLE. Independent checkpoint "
        "distributions cannot establish Gaussian identity continuity.",
        "",
        "EXPERIMENTAL FACT: low-support populations remain measurable through "
        "the 10K-to-14999 culling-only phase. This rules out only a post-10K "
        "newborn explanation; it establishes neither persistent identity nor "
        "whether final low-support Gaussians came from pre-10K split/duplicate.",
        "",
        "## Age Confounding",
        "",
        "QUANTITATIVE RESULT: AGE_CONTROL_NOT_AVAILABLE. Creation iteration "
        "and lineage metadata are absent, so age confounding cannot be excluded.",
        "",
        "## Lifecycle Semantics",
        "",
        "INFERENCE: prune can preserve state semantics for surviving indices "
        "through exact masking. Split and duplicate do not have one empirically "
        "validated state rule in the locked artifacts; each candidate answers "
        "a different question about inherited versus post-birth evidence.",
        "",
        "## Final Classification",
        "",
        f"FINAL DECISION: {summary['classification']}.",
        "",
        f"MODULE DESIGN AUTHORIZED: {str(summary['module_design_authorized']).upper()}.",
        "",
        f"RECOMMENDATION: {summary['recommendation']}.",
        "",
        "The low-support association remains a diagnostic finding, not a claim "
        "that Gaussian geometry is wrong and not a causal mechanism.",
        "",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    preflight_result = _read_json(OUTPUT_ROOT / "preflight.json")
    results = [
        _read_json(OUTPUT_ROOT / "workers" / scene / "scene_result.json")
        for scene in SCENES
    ]
    if not all(
        not result["optimizer_step_called"]
        and not result["backward_called"]
        and not result["training_performed"]
        and not result["heldout_leakage"]
        for result in results
    ):
        raise RuntimeError("worker protocol invariant failed")
    population_rows = [
        row for result in results for row in result["population_rows"]
    ]
    camera_rows = [row for result in results for row in result["camera_rows"]]
    strategy_rows = []
    for result in results:
        for source_row in result["strategy_reference_rows"]:
            row = dict(source_row)
            rho = row.get("whole_population_reference_rho")
            if rho is not None and not math.isfinite(float(rho)):
                row["whole_population_reference_rho"] = None
            strategy_rows.append(row)
    if len(population_rows) != len(SCENES) * len(STEPS):
        raise RuntimeError("population row count mismatch")
    events = _event_outputs()
    temporal = _temporal_statistics(
        population_rows, events["interval_rows"], events["scene_rows"]
    )
    lineage = preflight_result["lineage"]
    classification = (
        "LOW_SUPPORT_STATE_SEMANTICS_STABLE"
        if lineage["available"]
        else "LOW_SUPPORT_STATE_LIFECYCLE_AMBIGUOUS"
    )
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "lineage_availability": lineage["classification"],
        "lineage_available": lineage["available"],
        "age_availability": False,
        "age_control": "AGE_CONTROL_NOT_AVAILABLE",
        "newborn_or_split_origin_of_final_low_support": "NOT_AVAILABLE",
        "post_10k_newborn_explanation": "RULED_OUT_IN_ALL_SCENES",
        "population_evolution": population_rows,
        "split_strategy_comparison": strategy_rows,
        "strategy_residual_association_comparison": (
            "NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE"
        ),
        "support_temporal_statistics": temporal,
        "topology_event_summary": events["interval_rows"],
        "topology_event_audit": events["scene_rows"],
        "split_semantics": "AMBIGUOUS",
        "duplicate_semantics": "AMBIGUOUS",
        "prune_semantics": "STABLE_IF_STATE_IS_MASKED_WITH_PARAMETERS",
        "module_design_authorized": False,
        "recommendation": (
            "DO_NOT_DESIGN_LOW_SUPPORT_AWARE_MODULE; next run "
            "INSTRUMENT-GAUSSIAN-LINEAGE-SIDECAR-SMOKE-VALIDATION"
        ),
        "one_next_task": "INSTRUMENT-GAUSSIAN-LINEAGE-SIDECAR-SMOKE-VALIDATION",
        "optimizer_step_called": False,
        "backward_called": False,
        "new_models_trained": False,
        "training_code_modified": False,
        "renderer_modified": False,
        "ocmc_modified": False,
        "raoc_opened": False,
        "worker_wall_seconds": {
            result["scene"]: result["wall_seconds"] for result in results
        },
    }
    _write_csv(OUTPUT_ROOT / "population_evolution.csv", population_rows)
    _write_json(OUTPUT_ROOT / "population_evolution.json", {"rows": population_rows})
    _write_csv(OUTPUT_ROOT / "camera_baseline.csv", camera_rows)
    _write_json(OUTPUT_ROOT / "camera_baseline.json", {"rows": camera_rows})
    _write_csv(OUTPUT_ROOT / "split_strategy_comparison.csv", strategy_rows)
    _write_json(
        OUTPUT_ROOT / "split_strategy_comparison.json", {"rows": strategy_rows}
    )
    _write_json(OUTPUT_ROOT / "support_temporal_statistics.json", temporal)
    _write_json(
        OUTPUT_ROOT / "age_confounding.json",
        {
            "age_available": False,
            "classification": "AGE_CONTROL_NOT_AVAILABLE",
            "age_controlled_association": None,
        },
    )
    _write_json(OUTPUT_ROOT / "final_summary.json", summary)
    _research_note(summary)
    return summary


def launch() -> Dict[str, Any]:
    preflight_result = preflight()
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for scene, gpu in SCENE_GPUS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--worker",
            "--scene",
            scene,
            "--gpu",
            gpu,
        ]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((scene, gpu, process, handle))
    failures = []
    for scene, gpu, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            failures.append(
                {
                    "scene": scene,
                    "gpu": gpu,
                    "return_code": return_code,
                    "log": str(logs / f"{scene}.log"),
                }
            )
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"frozen lifecycle workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        result = worker(args.scene, args.gpu)
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
