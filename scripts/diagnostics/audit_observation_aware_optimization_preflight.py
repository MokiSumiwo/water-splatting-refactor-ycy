#!/usr/bin/env python3
"""Frozen parameter-sensitivity audit for observation-imbalanced Gaussians.

This diagnostic loads the registered C0 checkpoints and perturbs detached
render inputs only. It never mutates model parameters, trains, writes a
checkpoint, or uses held-out ground truth to select Gaussians.
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
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_local_contextual_support_predictor_iui3 as LOCAL
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


EXPERIMENT = "OBSERVATION-AWARE-OPTIMIZATION-PREFLIGHT"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "e73b9cb64bd72fdadbc356673ff65bccc8686e6f"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "observation_aware_optimization_preflight_20260901"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "OBSERVATION_AWARE_OPTIMIZATION_PREFLIGHT_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
PARAMETERS = ("opacity", "features_dc", "scale")
EPSILONS = (0.01, 0.05)
PRIMARY_EPSILON = 0.01
GROUPS = ("T1", "T2", "MIDDLE", "HIGH")
SAMPLES_PER_GROUP = 32
SEED = 42
EPS = 1e-12
EQUIVALENCE_ATOL = 2e-6
CONTROLS = ("depth", "opacity", "scale", "footprint")
PROTECTED_HASHES = dict(CAUSAL.PROTECTED_HASHES)
AUDITED_SOURCE_HASHES = {
    "scripts/diagnostics/audit_low_support_causal_intervention.py": "92a45a7d17621b6f44b882e919ea2d65f9916669a0e94d75c8c72d03249d0ee3",
    "scripts/diagnostics/audit_local_contextual_support_predictor_iui3.py": "2f88afc2174f5753ee6cee494041b1f793529a4ea13742c425ad2928023a3479",
    "scripts/experiments/run_m1_raoc_causal_scene.py": "79930754f41887c0530e6b033eef5f0f26b692795a4f3abd078358ad800f9f2a",
    "water_splatting/water_splatting.py": "1a9930c0e74b4f235fc5ae5e819823fe9e2cdd828e8764ca73e43d0f67aa63e1",
    "water_splatting/rendering/underwater_rasterizer.py": "04e6d1c6d136ee46ea32ea2abd666e688d6a35c00c787e31f17aa5f5ba17beba",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        cpu = value.detach().cpu()
        return cpu.item() if cpu.numel() == 1 else cpu.tolist()
    return str(value)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize(value),
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
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(row.get(key), float)
                    and not math.isfinite(float(row[key]))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(value: Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _stable_seed(*parts: str) -> int:
    payload = ":".join((str(SEED),) + tuple(parts)).encode("utf8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _strict_repo_files() -> Dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected repo state: {branch}@{head}")
    hashes: Dict[str, str] = {}
    for relative, expected in {**PROTECTED_HASHES, **AUDITED_SOURCE_HASHES}.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected/audited source changed: {relative}")
        hashes[relative] = actual
    return {
        "branch": branch,
        "head": head,
        "status_short": _run_text(["git", "status", "--short"]),
        "source_hashes": hashes,
    }


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose exactly physical GPU {gpu}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "physical_gpu": gpu,
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    repo = _strict_repo_files()
    sources: List[Dict[str, Any]] = []
    for scene in SCENES:
        config = REPO_ROOT / CAUSAL._scene_config(scene)["source_config"]
        checkpoint = CAUSAL._checkpoint(scene)
        sequence = CAUSAL.SOURCE_ROOT / scene / "camera_sequence.json"
        actual = {
            "checkpoint_sha256": _sha256(checkpoint),
            "source_config_sha256": _sha256(config),
            "camera_sequence_sha256": _sha256(sequence),
        }
        expected = {
            "checkpoint_sha256": CAUSAL.EXPECTED_CHECKPOINT_HASHES[scene],
            "source_config_sha256": CAUSAL.EXPECTED_CONFIG_HASHES[scene],
            "camera_sequence_sha256": CAUSAL.EXPECTED_CAMERA_SEQUENCE_HASHES[scene],
        }
        if actual != expected:
            raise RuntimeError(f"checkpoint/config/camera-sequence drift for {scene}")
        sources.append(
            {
                "scene": scene,
                "checkpoint": str(checkpoint),
                "source_config": str(config),
                "camera_sequence": str(sequence),
                **actual,
            }
        )
    result = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "sources": sources,
        "frozen_forward_only": True,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "renderer_changes": 0,
        "ocmc_changes": 0,
        "raoc_enabled": False,
        "sampling": {
            "groups": {
                "T1": "s <= 1",
                "T2": "s <= 2",
                "MIDDLE": "2 < s < median(s)",
                "HIGH": "s >= median(s)",
            },
            "samples_per_group": SAMPLES_PER_GROUP,
            "seed": SEED,
            "eligible_population": "heldout-visible Gaussians; GT excluded",
        },
        "perturbation": {
            "parameters": list(PARAMETERS),
            "epsilon_values": list(EPSILONS),
            "opacity": "physical sigmoid opacity increased additively by epsilon without parameter-level clipping; the unchanged renderer applies its existing per-pixel alpha cap",
            "features_dc": "features_dc parameter vector multiplied by 1+epsilon",
            "scale": "physical scale multiplied by 1+epsilon via raw log-scale addition",
            "one_gaussian_at_a_time": True,
        },
        "primary_rule": {
            "epsilon": PRIMARY_EPSILON,
            "composite": "L2 norm of scene-wise rank fractions for opacity/color/scale image-RMS finite-difference sensitivities",
            "scene_support": (
                "median composite sensitivity(T1) > median composite sensitivity(HIGH), "
                "all four single-control support-sensitivity partial rank correlations > 0, "
                "and sensitivity-vs-projected-residual Spearman > 0"
            ),
            "experiment_support": "scene_support in at least 3/4 scenes",
        },
    }
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    return result


def _camera_split(
    train_records: Sequence[Tuple[int, str, Any, Any]],
    eval_records: Sequence[Tuple[int, str, Any, Any]],
) -> Dict[str, Any]:
    train_ids = [str(row[1]) for row in train_records]
    eval_ids = [str(row[1]) for row in eval_records]
    if len(train_ids) != len(set(train_ids)) or len(eval_ids) != len(set(eval_ids)):
        raise RuntimeError("duplicate camera ID within split")
    if set(train_ids) & set(eval_ids):
        raise RuntimeError("train/eval camera leakage")
    payload = {"train_ids": train_ids, "eval_ids": eval_ids}
    return {
        **payload,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "camera_split_sha256": _hash_json(payload),
    }


@torch.no_grad()
def _support_counts(model: Any, records: Sequence[Tuple[int, str, Any, Any]]) -> Tensor:
    support = torch.zeros(int(model.means.shape[0]), dtype=torch.int16)
    for _index, _camera_id, camera, _batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = model.radii.detach().reshape(-1) > 0
        reported = outputs["gaussian_visible_mask"].detach().reshape(-1).bool()
        if not torch.equal(visible, reported) or visible.numel() != support.numel():
            raise RuntimeError("training visibility alias mismatch")
        support += visible.cpu().to(torch.int16)
        del outputs
    if int(support.max()) > len(records):
        raise RuntimeError("support exceeds distinct training-camera count")
    return support


@torch.no_grad()
def _heldout_geometry_stats(
    model: Any, records: Sequence[Tuple[int, str, Any, Any]], n_gaussians: int
) -> Dict[str, Tensor]:
    visibility = torch.zeros(n_gaussians, dtype=torch.int16)
    radius_sum = torch.zeros(n_gaussians, dtype=torch.float64)
    depth_sum = torch.zeros(n_gaussians, dtype=torch.float64)
    for _index, _camera_id, camera, _batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1).bool()
        radii = model.radii.detach().reshape(-1)
        depths = outputs["projected_gaussian_depths"].detach().reshape(-1)
        if visible.numel() != n_gaussians or not torch.equal(visible, radii > 0):
            raise RuntimeError("heldout visibility alias mismatch")
        cpu_visible = visible.cpu()
        visibility += cpu_visible.to(torch.int16)
        radius_sum += torch.where(cpu_visible, radii.cpu().double(), 0.0)
        depth_sum += torch.where(cpu_visible, depths.cpu().double(), 0.0)
        del outputs
    seen = visibility > 0
    mean_radius = torch.full((n_gaussians,), float("nan"), dtype=torch.float64)
    mean_depth = torch.full((n_gaussians,), float("nan"), dtype=torch.float64)
    mean_radius[seen] = radius_sum[seen] / visibility[seen].double()
    mean_depth[seen] = depth_sum[seen] / visibility[seen].double()
    return {
        "visibility": visibility,
        "mean_radius": mean_radius,
        "mean_depth": mean_depth,
    }


def _sample_groups(
    scene: str,
    support: Tensor,
    heldout_visibility: Tensor,
    count: int,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    median = float(torch.median(support.float()).item())
    eligible = heldout_visibility > 0
    pools = {
        "T1": eligible & (support <= 1),
        "T2": eligible & (support <= 2),
        "MIDDLE": eligible & (support > 2) & (support.float() < median),
        "HIGH": eligible & (support.float() >= median),
    }
    selections: Dict[str, Tensor] = {}
    rows: List[Dict[str, Any]] = []
    for group, pool in pools.items():
        indices = torch.nonzero(pool, as_tuple=False).reshape(-1)
        if int(indices.numel()) < count:
            raise RuntimeError(f"{scene}/{group} has only {indices.numel()} eligible Gaussians")
        generator = torch.Generator(device="cpu")
        seed = _stable_seed(scene, group)
        generator.manual_seed(seed)
        selected = indices[torch.randperm(int(indices.numel()), generator=generator)[:count]].sort().values
        selections[group] = selected
        rows.append(
            {
                "group": group,
                "pool_count": int(indices.numel()),
                "sample_count": int(selected.numel()),
                "seed": seed,
                "selected_ids_sha256": _hash_tensor(selected),
            }
        )
    return selections, {
        "support_median_torch_lower": median,
        "heldout_visible_count": int(eligible.sum()),
        "rows": rows,
    }


def _memberships(selections: Mapping[str, Tensor]) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    for group, indices in selections.items():
        for index in indices.tolist():
            result.setdefault(int(index), []).append(group)
    return result


def _view_projection_inputs(model: Any, camera: Any) -> Tuple[Any, Tensor, int, int, float]:
    camera = camera.to(model.device)
    downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / downscale)
    rotation = camera.camera_to_worlds[0, :3, :3]
    translation = camera.camera_to_worlds[0, :3, 3:4]
    edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=rotation.dtype))
    rotation = rotation @ edit
    inverse = rotation.T
    viewmat = torch.eye(4, device=model.device, dtype=rotation.dtype)
    viewmat[:3, :3] = inverse
    viewmat[:3, 3:4] = -inverse @ translation
    return camera, viewmat, int(camera.height.item()), int(camera.width.item()), float(downscale)


@torch.no_grad()
def _scale_geometry(
    model: Any,
    camera: Any,
    base: Mapping[str, Tensor],
    gaussian_id: int,
    epsilon: float,
) -> Mapping[str, Tensor]:
    camera, viewmat, height, width, downscale = _view_projection_inputs(model, camera)
    scales = model.scales.detach().clone()
    scales[gaussian_id] += math.log1p(float(epsilon))
    try:
        xys, depths, radii, conics, _comp, num_tiles_hit, _cov3d = model.underwater_rasterizer.project(
            means=model.means.detach(),
            scales=scales,
            quats=model.quats.detach(),
            viewmat=viewmat,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=float(camera.cx.item()),
            cy=float(camera.cy.item()),
            height=height,
            width=width,
            clip_thresh=model.config.clip_thresh,
        )
    finally:
        camera.rescale_output_resolution(downscale)
    return {
        **base,
        "xys": xys,
        "depths": depths,
        "radii": radii,
        "conics": conics,
        "num_tiles_hit": num_tiles_hit,
    }


@torch.no_grad()
def _render_arrays(
    model: Any,
    geometry: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    *,
    opacities: Optional[Tensor] = None,
    colors: Optional[Tensor] = None,
) -> Tensor:
    height, width = (int(item) for item in geometry["size"].tolist())
    medium_rgb = outputs["medium_rgb"].detach()
    medium_bs = outputs["medium_bs"].detach()
    medium_attn = outputs["medium_attn"].detach()
    render = model.underwater_rasterizer.rasterize(
        xys=geometry["xys"],
        xys_grad_abs=torch.zeros_like(geometry["xys"]),
        depths=geometry["depths"],
        radii=geometry["radii"],
        conics=geometry["conics"],
        num_tiles_hit=geometry["num_tiles_hit"],
        colors=geometry["colors"] if colors is None else colors,
        opacities=geometry["opacities"] if opacities is None else opacities,
        medium_rgb=medium_rgb,
        medium_bs=medium_bs,
        medium_attn=medium_attn,
        height=height,
        width=width,
        background=medium_rgb,
        step=model.step,
    )
    rgb_medium = render.rgb_medium
    if model._effective_b_inf_mode() == "tied":
        tail_weight = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
        rgb_medium = (
            rgb_medium
            - tail_weight * medium_rgb
            + tail_weight * outputs["b_inf"].detach()
        )
        pred = render.rgb_object + rgb_medium
    else:
        pred = render.rgb
    if not bool(torch.isfinite(pred).all()):
        raise RuntimeError("non-finite perturbed render")
    return pred.detach()


@torch.no_grad()
def _perturbed_render(
    model: Any,
    camera: Any,
    base_geometry: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    gaussian_id: int,
    parameter: str,
    epsilon: float,
) -> Tuple[Tensor, float, Dict[str, Any]]:
    if parameter == "opacity":
        opacities = base_geometry["opacities"].clone()
        original = float(opacities[gaussian_id].item())
        perturbed = original + epsilon
        opacities[gaussian_id] = perturbed
        actual = perturbed - original
        pred = _render_arrays(model, base_geometry, outputs, opacities=opacities)
        return pred, actual, {"physical_before": original, "physical_after": perturbed}
    if parameter == "features_dc":
        colors = base_geometry["colors"].clone()
        base_dc = model.features_dc[gaussian_id : gaussian_id + 1].detach()
        perturbed_dc = base_dc * (1.0 + epsilon)
        new_color = MI._colors_for_current_parameterization(
            model,
            camera,
            model.means[gaussian_id : gaussian_id + 1].detach(),
            perturbed_dc,
            model.features_rest[gaussian_id : gaussian_id + 1].detach(),
        ).detach()[0]
        colors[gaussian_id] = new_color
        base_norm = float(torch.linalg.vector_norm(base_dc).item())
        delta_norm = float(torch.linalg.vector_norm(perturbed_dc - base_dc).item())
        actual = delta_norm / max(base_norm, EPS)
        pred = _render_arrays(model, base_geometry, outputs, colors=colors)
        return pred, actual, {
            "physical_before": base_norm,
            "physical_after": float(torch.linalg.vector_norm(perturbed_dc).item()),
        }
    if parameter == "scale":
        geometry = _scale_geometry(model, camera, base_geometry, gaussian_id, epsilon)
        pred = _render_arrays(model, geometry, outputs)
        physical = torch.exp(model.scales[gaussian_id].detach())
        return pred, float(epsilon), {
            "physical_before": float(physical.amax()),
            "physical_after": float((physical * (1.0 + epsilon)).amax()),
            "perturbed_projected_radius": float(geometry["radii"][gaussian_id]),
        }
    raise ValueError(parameter)


def _projected_residual(residual: Tensor, xy: Tensor, radius: Tensor) -> Tuple[float, int]:
    height, width = residual.shape
    x = float(xy[0])
    y = float(xy[1])
    r = max(float(radius), 0.5)
    x0 = max(0, int(math.floor(x - r)))
    x1 = min(width - 1, int(math.ceil(x + r)))
    y0 = max(0, int(math.floor(y - r)))
    y1 = min(height - 1, int(math.ceil(y + r)))
    if x0 > x1 or y0 > y1:
        return float("nan"), 0
    ys = torch.arange(y0, y1 + 1, device=residual.device, dtype=torch.float32)
    xs = torch.arange(x0, x1 + 1, device=residual.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    mask = (grid_x - x).square() + (grid_y - y).square() <= r * r
    values = residual[y0 : y1 + 1, x0 : x1 + 1][mask]
    if not values.numel():
        return float("nan"), 0
    return float(values.mean()), int(values.numel())


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _median(values: Iterable[float]) -> float:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(np.median(finite)) if finite.size else float("nan")


def _rms(values: Iterable[float]) -> float:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def _rho(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 4 or np.ptp(a[valid]) <= EPS or np.ptp(b[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.spearmanr(a[valid], b[valid]).statistic)


def _partial_rank(
    predictor: Sequence[float], target: Sequence[float], controls: Sequence[Sequence[float]]
) -> float:
    arrays = [np.asarray(predictor, dtype=np.float64), np.asarray(target, dtype=np.float64)]
    arrays.extend(np.asarray(control, dtype=np.float64) for control in controls)
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if int(valid.sum()) < max(12, len(controls) + 4):
        return float("nan")
    ranked = [scipy.stats.rankdata(array[valid]) for array in arrays]
    if np.ptp(ranked[0]) <= EPS or np.ptp(ranked[1]) <= EPS:
        return float("nan")
    design = np.column_stack([np.ones(int(valid.sum()))] + ranked[2:])
    predictor_residual = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    target_residual = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    return _rho(predictor_residual, target_residual)


def _primary_population(rows: Sequence[Mapping[str, Any]], scene: str) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["scene"] == scene and row["support_group"] in ("T1", "MIDDLE", "HIGH")
    ]


def _aggregate_gaussians(
    scene: str,
    perturbations: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Tensor],
    support: Tensor,
    model: Any,
) -> List[Dict[str, Any]]:
    projection_lookup: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in projection_rows:
        projection_lookup.setdefault((str(row["support_group"]), int(row["gaussian_id"])), []).append(row)
    output: List[Dict[str, Any]] = []
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1).cpu()
    scale = torch.exp(model.scales.detach()).amax(dim=-1).cpu()
    dc_norm = torch.linalg.vector_norm(model.features_dc.detach(), dim=-1).cpu()
    for group, indices in selections.items():
        for gaussian_id in indices.tolist():
            rows = [
                row
                for row in perturbations
                if row["support_group"] == group and int(row["gaussian_id"]) == gaussian_id
            ]
            projected = projection_lookup[(group, gaussian_id)]
            row: Dict[str, Any] = {
                "scene": scene,
                "support_group": group,
                "gaussian_id": gaussian_id,
                "support_count": int(support[gaussian_id]),
                "underconstraint_score": -int(support[gaussian_id]),
                "heldout_visible_view_count": len(projected),
                "depth": _mean(float(item["projected_depth"]) for item in projected),
                "footprint": _mean(float(item["projected_radius"]) for item in projected),
                "projected_residual": _mean(float(item["projected_residual"]) for item in projected),
                "projected_residual_pixel_count": sum(int(item["projected_residual_pixel_count"]) for item in projected),
                "opacity": float(opacity[gaussian_id]),
                "scale": float(scale[gaussian_id]),
                "features_dc_norm": float(dc_norm[gaussian_id]),
                "sample_selection_used_gt": False,
                "residual_uses_heldout_gt": True,
            }
            primary_sensitivities: List[float] = []
            for parameter in PARAMETERS:
                for epsilon in EPSILONS:
                    subset = [
                        item
                        for item in rows
                        if item["parameter"] == parameter
                        and math.isclose(float(item["epsilon_requested"]), epsilon)
                    ]
                    suffix = f"{parameter}_{int(round(epsilon * 100))}pct"
                    sensitivity = _rms(float(item["image_rms_sensitivity"]) for item in subset)
                    row[f"{suffix}_sensitivity"] = sensitivity
                    row[f"{suffix}_render_delta_rms"] = _rms(float(item["render_delta_rms"]) for item in subset)
                    row[f"{suffix}_dPSNR"] = _mean(float(item["delta_PSNR"]) for item in subset)
                    row[f"{suffix}_dLPIPS"] = _mean(float(item["delta_LPIPS"]) for item in subset)
                    row[f"{suffix}_dMSE"] = _mean(float(item["delta_MSE"]) for item in subset)
                    row[f"{suffix}_view_count"] = len(subset)
                    if epsilon == PRIMARY_EPSILON:
                        primary_sensitivities.append(sensitivity)
                one = float(row[f"{parameter}_1pct_sensitivity"])
                five = float(row[f"{parameter}_5pct_sensitivity"])
                row[f"{parameter}_finite_difference_linearity_ratio"] = five / max(one, EPS)
            row["raw_composite_sensitivity_l2"] = float(np.linalg.norm(primary_sensitivities))
            output.append(row)
    for parameter in PARAMETERS:
        key = f"{parameter}_1pct_sensitivity"
        values = np.asarray([float(row[key]) for row in output], dtype=np.float64)
        ranks = scipy.stats.rankdata(values, method="average")
        rank_fractions = (ranks - 0.5) / max(len(output), 1)
        for row, rank_fraction in zip(output, rank_fractions):
            row[f"{parameter}_1pct_sensitivity_rank_fraction"] = float(rank_fraction)
    for row in output:
        row["composite_sensitivity"] = float(
            np.linalg.norm(
                [
                    float(row[f"{parameter}_1pct_sensitivity_rank_fraction"])
                    for parameter in PARAMETERS
                ]
            )
        )
    return output


def _group_comparisons(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for scene in SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        summaries: Dict[str, Dict[str, Any]] = {}
        for group in GROUPS:
            subset = [row for row in scene_rows if row["support_group"] == group]
            summary = {
                "scene": scene,
                "support_group": group,
                "sample_count": len(subset),
                "composite_sensitivity_mean": _mean(float(row["composite_sensitivity"]) for row in subset),
                "composite_sensitivity_median": _median(float(row["composite_sensitivity"]) for row in subset),
                "projected_residual_mean": _mean(float(row["projected_residual"]) for row in subset),
            }
            for parameter in PARAMETERS:
                summary[f"{parameter}_1pct_sensitivity_median"] = _median(
                    float(row[f"{parameter}_1pct_sensitivity"]) for row in subset
                )
                summary[f"{parameter}_5pct_sensitivity_median"] = _median(
                    float(row[f"{parameter}_5pct_sensitivity"]) for row in subset
                )
            summaries[group] = summary
            output.append(summary)
        low = summaries["T1"]["composite_sensitivity_median"]
        high = summaries["HIGH"]["composite_sensitivity_median"]
        output.append(
            {
                "scene": scene,
                "support_group": "T1_VS_HIGH_PRIMARY",
                "sample_count": summaries["T1"]["sample_count"],
                "composite_sensitivity_mean": float("nan"),
                "composite_sensitivity_median": float("nan"),
                "projected_residual_mean": float("nan"),
                "low_median": low,
                "high_median": high,
                "low_over_high_ratio": low / max(high, EPS),
                "low_greater_than_high": bool(low > high),
            }
        )
    return output


def _control_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for scene in SCENES:
        population = _primary_population(rows, scene)
        underconstraint = [float(row["underconstraint_score"]) for row in population]
        sensitivity = [float(row["composite_sensitivity"]) for row in population]
        output.append(
            {
                "scene": scene,
                "analysis": "underconstraint_vs_composite_sensitivity",
                "control": "NONE",
                "sample_count": len(population),
                "rank_residualized_spearman": _rho(underconstraint, sensitivity),
                "positive_after_control": _rho(underconstraint, sensitivity) > 0.0,
            }
        )
        for control in CONTROLS:
            value = _partial_rank(
                underconstraint,
                sensitivity,
                [[float(row[control]) for row in population]],
            )
            output.append(
                {
                    "scene": scene,
                    "analysis": "underconstraint_vs_composite_sensitivity",
                    "control": control,
                    "sample_count": len(population),
                    "rank_residualized_spearman": value,
                    "positive_after_control": bool(math.isfinite(value) and value > 0.0),
                }
            )
        joint = _partial_rank(
            underconstraint,
            sensitivity,
            [[float(row[control]) for row in population] for control in CONTROLS],
        )
        output.append(
            {
                "scene": scene,
                "analysis": "underconstraint_vs_composite_sensitivity",
                "control": "JOINT_DEPTH_OPACITY_SCALE_FOOTPRINT",
                "sample_count": len(population),
                "rank_residualized_spearman": joint,
                "positive_after_control": bool(math.isfinite(joint) and joint > 0.0),
                "joint_descriptive_not_primary": True,
            }
        )
    return output


def _residual_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    sensitivity_keys = ["composite_sensitivity"] + [f"{parameter}_1pct_sensitivity" for parameter in PARAMETERS]
    for scene in SCENES:
        population = _primary_population(rows, scene)
        residual = [float(row["projected_residual"]) for row in population]
        for key in sensitivity_keys:
            sensitivity = [float(row[key]) for row in population]
            output.append(
                {
                    "scene": scene,
                    "sensitivity": key,
                    "control": "NONE",
                    "sample_count": len(population),
                    "spearman_sensitivity_vs_projected_residual": _rho(sensitivity, residual),
                    "positive": _rho(sensitivity, residual) > 0.0,
                }
            )
            for control in CONTROLS:
                value = _partial_rank(
                    sensitivity,
                    residual,
                    [[float(row[control]) for row in population]],
                )
                output.append(
                    {
                        "scene": scene,
                        "sensitivity": key,
                        "control": control,
                        "sample_count": len(population),
                        "spearman_sensitivity_vs_projected_residual": value,
                        "positive": bool(math.isfinite(value) and value > 0.0),
                    }
                )
    return output


def _quality_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scene_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        subset = [row for row in rows if row["scene"] == scene]
        finite = all(
            math.isfinite(float(row[key]))
            for row in subset
            for key in (
                "composite_sensitivity",
                *[f"{parameter}_1pct_sensitivity" for parameter in PARAMETERS],
                *[f"{parameter}_5pct_sensitivity" for parameter in PARAMETERS],
            )
        )
        nonzero_fraction = _mean(
            float(float(row["raw_composite_sensitivity_l2"]) > EPS) for row in subset
        )
        ratios: Dict[str, float] = {}
        for parameter in PARAMETERS:
            usable = [
                float(row[f"{parameter}_finite_difference_linearity_ratio"])
                for row in subset
                if float(row[f"{parameter}_1pct_sensitivity"]) > EPS
                and math.isfinite(float(row[f"{parameter}_finite_difference_linearity_ratio"]))
            ]
            ratios[parameter] = _median(usable)
        ratio_pass = all(
            math.isfinite(value) and 0.8 <= value <= 1.2
            for value in ratios.values()
        )
        passed = bool(
            len(subset) == len(GROUPS) * SAMPLES_PER_GROUP
            and finite
            and nonzero_fraction >= 0.90
            and ratio_pass
        )
        scene_rows.append(
            {
                "scene": scene,
                "gaussian_count": len(subset),
                "all_primary_values_finite": finite,
                "nonzero_composite_fraction": nonzero_fraction,
                "median_5pct_over_1pct_sensitivity": ratios,
                "linearity_medians_in_0p8_to_1p2": ratio_pass,
                "quality_pass": passed,
            }
        )
    return {
        "required_gaussians_per_scene": len(GROUPS) * SAMPLES_PER_GROUP,
        "minimum_nonzero_composite_fraction": 0.90,
        "linearity_median_range": [0.8, 1.2],
        "scene_rows": scene_rows,
        "all_scenes_pass": all(bool(row["quality_pass"]) for row in scene_rows),
    }


def _classification(
    comparisons: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    residuals: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> Dict[str, Any]:
    scene_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        comparison = next(
            row
            for row in comparisons
            if row["scene"] == scene and row["support_group"] == "T1_VS_HIGH_PRIMARY"
        )
        controlled = [
            row
            for row in controls
            if row["scene"] == scene and row["control"] in CONTROLS
        ]
        residual = next(
            row
            for row in residuals
            if row["scene"] == scene
            and row["sensitivity"] == "composite_sensitivity"
            and row["control"] == "NONE"
        )
        all_controls_positive = len(controlled) == len(CONTROLS) and all(
            bool(row["positive_after_control"]) for row in controlled
        )
        supported = bool(
            comparison["low_greater_than_high"]
            and all_controls_positive
            and residual["positive"]
        )
        scene_rows.append(
            {
                "scene": scene,
                "low_over_high_ratio": float(comparison["low_over_high_ratio"]),
                "low_sensitivity_greater_than_high": bool(comparison["low_greater_than_high"]),
                "single_factor_controls_all_positive": all_controls_positive,
                "control_rhos": {
                    str(row["control"]): float(row["rank_residualized_spearman"])
                    for row in controlled
                },
                "sensitivity_residual_spearman": float(
                    residual["spearman_sensitivity_vs_projected_residual"]
                ),
                "sensitivity_residual_positive": bool(residual["positive"]),
                "scene_supported": supported,
            }
        )
    supported_count = sum(bool(row["scene_supported"]) for row in scene_rows)
    if not bool(quality["all_scenes_pass"]):
        label = "INCONCLUSIVE"
    elif supported_count >= 3:
        label = "OBSERVATION_UNDERCONSTRAINED_SUPPORTED"
    else:
        label = "OBSERVATION_UNDERCONSTRAINED_NOT_SUPPORTED"
    return {
        "experiment": EXPERIMENT,
        "classification": label,
        "required_scene_count": 3,
        "supported_scene_count": supported_count,
        "module_design_authorized": label == "OBSERVATION_UNDERCONSTRAINED_SUPPORTED",
        "scene_rows": scene_rows,
        "inconclusive": label == "INCONCLUSIVE",
        "quality_gate_passed": bool(quality["all_scenes_pass"]),
    }


@torch.no_grad()
def worker(scene: str, gpu: str, sample_count: int = SAMPLES_PER_GROUP) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    started = time.perf_counter()
    scene_dir = OUTPUT_ROOT / "workers" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    branch = FORMAL._setup_branch(REPO_ROOT, CAUSAL._scene_config(scene), "C0")
    try:
        model = branch.pipeline.model
        payload = FORMAL._load_checkpoint(branch, CAUSAL._checkpoint(scene))
        if (
            payload.get("experiment") != FORMAL.EXPERIMENT
            or payload.get("branch") != "C0"
            or int(payload.get("absolute_step", -1)) != 14999
            or payload.get("ocmc_bundle") is None
            or payload.get("raoc_state") is not None
        ):
            raise RuntimeError("checkpoint condition provenance drift")
        if (
            not model.config.camera_medium_observability_enabled
            or model.config.camera_medium_ray_adaptive_observability_enabled
            or model.config.intrinsic_color_parameterization != "bounded_sh3"
            or model.config.rasterize_mode != "classic"
            or model.config.medium_context_mode != "dir_xy_camera"
            or int(model.config.sh_degree) != 3
        ):
            raise RuntimeError("locked C0 model configuration drift")
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        split = _camera_split(train_records, eval_records)
        support = _support_counts(model, train_records)
        heldout = _heldout_geometry_stats(model, eval_records, int(support.numel()))
        selections, selection_manifest = _sample_groups(
            scene, support, heldout["visibility"], sample_count
        )
        memberships = _memberships(selections)
        selected_ids = sorted(memberships)
        state_before = CAUSAL._model_state_hash(model)
        projector_before = _hash_tensor(model._camera_medium_observability_projector)
        perturbations: List[Dict[str, Any]] = []
        projection_rows: List[Dict[str, Any]] = []
        equivalence_rows: List[Dict[str, Any]] = []
        ocmc_rows: List[Dict[str, Any]] = []

        for _index, camera_id, camera, batch in eval_records:
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            baseline = outputs["pred_image"].detach().float()
            baseline_metrics = MIC._metric_images(model, baseline, gt)
            residual = (baseline.clamp(0, 1) - gt).square().mean(dim=-1)
            geometry = CAUSAL._geometry(model, camera, int(baseline.shape[0]), int(baseline.shape[1]))
            if int(geometry["opacities"].shape[0]) != int(support.numel()):
                raise RuntimeError("crop/global Gaussian indexing mismatch")
            visible = geometry["radii"] > 0
            if not torch.equal(visible, outputs["gaussian_visible_mask"].reshape(-1)):
                raise RuntimeError("reprojected heldout visibility mismatch")
            no_op = _render_arrays(model, geometry, outputs)
            difference = (no_op.float() - baseline.float()).abs()
            equivalent = bool(torch.allclose(no_op, baseline, atol=EQUIVALENCE_ATOL, rtol=0.0))
            equivalence_rows.append(
                {
                    "scene": scene,
                    "camera_id": camera_id,
                    "max_abs_pixel_difference": float(difference.max()),
                    "mean_abs_pixel_difference": float(difference.mean()),
                    "atol": EQUIVALENCE_ATOL,
                    "allclose": equivalent,
                }
            )
            if not equivalent:
                raise RuntimeError(f"detached no-op render differs from FULL: {equivalence_rows[-1]}")
            ocmc_before = CAUSAL._ocmc_view_state(outputs, model)

            for gaussian_id in selected_ids:
                if not bool(visible[gaussian_id]):
                    continue
                projected_residual, pixel_count = _projected_residual(
                    residual,
                    geometry["xys"][gaussian_id],
                    geometry["radii"][gaussian_id],
                )
                for group in memberships[gaussian_id]:
                    projection_rows.append(
                        {
                            "scene": scene,
                            "camera_id": camera_id,
                            "support_group": group,
                            "gaussian_id": gaussian_id,
                            "support_count": int(support[gaussian_id]),
                            "projected_depth": float(geometry["depths"][gaussian_id]),
                            "projected_radius": float(geometry["radii"][gaussian_id]),
                            "projected_residual": projected_residual,
                            "projected_residual_pixel_count": pixel_count,
                            "heldout_gt_used": True,
                            "sample_selection_used_gt": False,
                        }
                    )
                for parameter in PARAMETERS:
                    for epsilon in EPSILONS:
                        pred, actual_step, parameter_meta = _perturbed_render(
                            model,
                            camera,
                            geometry,
                            outputs,
                            gaussian_id,
                            parameter,
                            epsilon,
                        )
                        metrics = MIC._metric_images(model, pred, gt)
                        delta = pred.float() - baseline.float()
                        delta_mse = float(delta.square().mean())
                        delta_rms = math.sqrt(max(delta_mse, 0.0))
                        sensitivity = delta_rms / max(abs(actual_step), EPS)
                        base_row = {
                            "scene": scene,
                            "camera_id": camera_id,
                            "gaussian_id": gaussian_id,
                            "support_count": int(support[gaussian_id]),
                            "parameter": parameter,
                            "epsilon_requested": epsilon,
                            "epsilon_actual_parameter_step": actual_step,
                            "render_delta_mae": float(delta.abs().mean()),
                            "render_delta_mse": delta_mse,
                            "render_delta_rms": delta_rms,
                            "render_delta_l2": float(torch.linalg.vector_norm(delta)),
                            "render_delta_psnr": -10.0 * math.log10(max(delta_mse, EPS)),
                            "image_rms_sensitivity": sensitivity,
                            "delta_PSNR": float(metrics["PSNR"] - baseline_metrics["PSNR"]),
                            "delta_SSIM": float(metrics["SSIM"] - baseline_metrics["SSIM"]),
                            "delta_LPIPS": float(metrics["LPIPS"] - baseline_metrics["LPIPS"]),
                            "delta_MSE": float(metrics["MSE"] - baseline_metrics["MSE"]),
                            "projected_residual": projected_residual,
                            "projected_radius": float(geometry["radii"][gaussian_id]),
                            "projected_depth": float(geometry["depths"][gaussian_id]),
                            "baseline_visible": True,
                            "heldout_gt_used_for_metrics": True,
                            "sample_selection_used_gt": False,
                            **parameter_meta,
                        }
                        for group in memberships[gaussian_id]:
                            perturbations.append({**base_row, "support_group": group})
                        del pred, delta
            ocmc_after = CAUSAL._ocmc_view_state(outputs, model)
            ocmc_rows.append(
                {
                    "scene": scene,
                    "camera_id": camera_id,
                    "delta_sha256_before": ocmc_before["delta_sha256"],
                    "delta_sha256_after": ocmc_after["delta_sha256"],
                    "projector_sha256_before": ocmc_before["projector_sha256"],
                    "projector_sha256_after": ocmc_after["projector_sha256"],
                    "exactly_equal": ocmc_before == ocmc_after,
                }
            )
            if ocmc_before != ocmc_after:
                raise RuntimeError("OCMC changed during detached Gaussian perturbation")
            del outputs, gt, baseline, residual, geometry, no_op, difference

        gaussian_rows = _aggregate_gaussians(
            scene, perturbations, projection_rows, selections, support, model
        )
        state_after = CAUSAL._model_state_hash(model)
        projector_after = _hash_tensor(model._camera_medium_observability_projector)
        if state_before != state_after or projector_before != projector_after:
            raise RuntimeError("frozen model/OCMC state mutated")
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "runtime": runtime,
            "checkpoint": str(CAUSAL._checkpoint(scene)),
            "checkpoint_sha256": _sha256(CAUSAL._checkpoint(scene)),
            "camera_split": split,
            "selection": selection_manifest,
            "sample_count_per_group": sample_count,
            "unique_sampled_gaussian_count": len(selected_ids),
            "parameter_perturbation_rows": len(perturbations),
            "per_gaussian_rows": len(gaussian_rows),
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "ocmc_projector_sha256_before": projector_before,
            "ocmc_projector_sha256_after": projector_after,
            "model_state_unchanged": state_before == state_after,
            "ocmc_projector_unchanged": projector_before == projector_after,
            "no_op_allclose": all(bool(row["allclose"]) for row in equivalence_rows),
            "no_op_max_abs_pixel_difference": max(float(row["max_abs_pixel_difference"]) for row in equivalence_rows),
            "ocmc_all_exact": all(bool(row["exactly_equal"]) for row in ocmc_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "backward_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_writes": 0,
        }
        suffix = f"_n{sample_count}" if sample_count != SAMPLES_PER_GROUP else ""
        _write_csv(scene_dir / f"parameter_perturbation{suffix}.csv", perturbations)
        _write_csv(scene_dir / f"per_gaussian_sensitivity{suffix}.csv", gaussian_rows)
        _write_csv(scene_dir / f"projected_residual{suffix}.csv", projection_rows)
        _write_csv(scene_dir / f"no_op_equivalence{suffix}.csv", equivalence_rows)
        _write_csv(scene_dir / f"ocmc_independence{suffix}.csv", ocmc_rows)
        _write_json(scene_dir / f"worker_summary{suffix}.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _coerce_gaussian_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    integer = {
        "gaussian_id",
        "support_count",
        "underconstraint_score",
        "heldout_visible_view_count",
        "projected_residual_pixel_count",
    }
    text = {"scene", "support_group", "sample_selection_used_gt", "residual_uses_heldout_gt"}
    output: List[Dict[str, Any]] = []
    for source in rows:
        row: Dict[str, Any] = {}
        for key, value in source.items():
            if key in text:
                row[key] = value
            elif key in integer or key.endswith("_view_count"):
                row[key] = int(value)
            else:
                row[key] = float(value) if value != "" else float("nan")
        output.append(row)
    return output


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    comparisons = [
        row
        for row in summary["support_group_comparison"]
        if row["support_group"] == "T1_VS_HIGH_PRIMARY"
    ]
    lines = [
        "# Observation-Aware Optimization Preflight",
        "",
        "Date: 2026-09-01",
        f"Experiment: `{EXPERIMENT}`",
        f"Classification: `{classification['classification']}`",
        "",
        "## Frozen Protocol",
        "",
        "The audit uses only the registered step-14999 C0 checkpoints with OCMC on and RAOC off. It samples heldout-visible Gaussians from T1, T2, middle, and high-support populations without using GT. Each sampled Gaussian receives isolated physical-opacity `+0.01/+0.05` without parameter-level clipping, relative `features_dc` `+1%/+5%`, and relative physical-scale `+1%/+5%` perturbations in detached render inputs. Scale perturbations are reprojected. The unchanged renderer retains its existing per-pixel alpha cap. No parameter, topology, optimizer, renderer, OCMC state, or checkpoint is changed.",
        "",
        "Primary sensitivity is the L2 norm of scene-wise rank fractions for the three 1% image-RMS finite-difference sensitivities. Rank normalization prevents opacity, color, and scale coordinate units from dominating the composite. A scene passes only when T1 median sensitivity exceeds high-support median sensitivity, all single-factor rank controls for depth/opacity/scale/footprint retain a positive underconstraint association, and sensitivity correlates positively with projected heldout residual.",
        "",
        "## Primary Results",
        "",
        "| Scene | T1/high sensitivity | low > high | all controls positive | sensitivity-residual rho | scene pass |",
        "|---|---:|:---:|:---:|---:|:---:|",
    ]
    by_scene = {row["scene"]: row for row in classification["scene_rows"]}
    for comparison in comparisons:
        row = by_scene[comparison["scene"]]
        lines.append(
            f"| {row['scene']} | {row['low_over_high_ratio']:.6f} | {'yes' if row['low_sensitivity_greater_than_high'] else 'no'} | {'yes' if row['single_factor_controls_all_positive'] else 'no'} | {row['sensitivity_residual_spearman']:.6f} | {'yes' if row['scene_supported'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Control Analysis",
            "",
            "| Scene | depth | opacity | scale | footprint |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in classification["scene_rows"]:
        values = row["control_rhos"]
        lines.append(
            f"| {row['scene']} | {values['depth']:.6f} | {values['opacity']:.6f} | {values['scale']:.6f} | {values['footprint']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Numerical Quality",
            "",
            "| Scene | finite | nonzero composite | opacity 5%/1% | color 5%/1% | scale 5%/1% | pass |",
            "|---|:---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in summary["quality_audit"]["scene_rows"]:
        ratios = row["median_5pct_over_1pct_sensitivity"]
        lines.append(
            f"| {row['scene']} | {'yes' if row['all_primary_values_finite'] else 'no'} | {row['nonzero_composite_fraction']:.6f} | {ratios['opacity']:.6f} | {ratios['features_dc']:.6f} | {ratios['scale']:.6f} | {'yes' if row['quality_pass'] else 'no'} |"
        )
    authorized = bool(classification["module_design_authorized"])
    lines.extend(
        [
            "",
            "## Scientific Interpretation",
            "",
            f"The numerical quality gate {'passes' if classification['quality_gate_passed'] else 'fails'} and the preregistered scientific criterion passes in {classification['supported_scene_count']}/4 scenes. "
            + (
                "This supports observation imbalance as a parameter-sensitivity mechanism distinct from simple Gaussian suppression."
                if authorized
                else "The frozen sensitivity evidence is not stable enough to establish observation-underconstrained Gaussian optimization as the second failure mechanism."
            ),
            "",
            ("Observation-aware Gaussian Optimization module design is authorized as a separate next phase." if authorized else "Observation-aware optimization module design is not authorized. Treat low support as a difficult-region indicator and search for another failure mechanism."),
            "",
            "## Integrity",
            "",
            f"All detached no-op renders reproduce FULL within `{EQUIVALENCE_ATOL}` absolute tolerance. Every worker reports hash-identical model and OCMC projector state, exact OCMC forward-state equality, zero backward calls, zero optimizer steps, and zero checkpoint writes.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    worker_summaries = [
        _read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json")
        for scene in SCENES
    ]
    if not all(
        row["model_state_unchanged"]
        and row["ocmc_projector_unchanged"]
        and row["no_op_allclose"]
        and row["ocmc_all_exact"]
        and row["backward_calls"] == 0
        and row["optimizer_step_calls"] == 0
        and row["checkpoint_writes"] == 0
        for row in worker_summaries
    ):
        raise RuntimeError("worker integrity gate failed")
    gaussian_raw: List[Dict[str, str]] = []
    perturbations: List[Dict[str, str]] = []
    for scene in SCENES:
        scene_dir = OUTPUT_ROOT / "workers" / scene
        gaussian_raw.extend(_load_csv(scene_dir / "per_gaussian_sensitivity.csv"))
        perturbations.extend(_load_csv(scene_dir / "parameter_perturbation.csv"))
    gaussian_rows = _coerce_gaussian_rows(gaussian_raw)
    comparisons = _group_comparisons(gaussian_rows)
    controls = _control_rows(gaussian_rows)
    residuals = _residual_rows(gaussian_rows)
    quality = _quality_audit(gaussian_rows)
    classification = _classification(comparisons, controls, residuals, quality)
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "quality_audit": quality,
        "support_group_comparison": comparisons,
        "control_analysis": controls,
        "residual_correlation": residuals,
        "worker_summaries": worker_summaries,
        "per_gaussian_sensitivity_rows": len(gaussian_rows),
        "parameter_perturbation_rows": len(perturbations),
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "module_design_started": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
    }
    _write_csv(OUTPUT_ROOT / "per_scene_sensitivity.csv", gaussian_rows)
    _write_csv(OUTPUT_ROOT / "support_group_comparison.csv", comparisons)
    _write_csv(OUTPUT_ROOT / "parameter_perturbation.csv", perturbations)
    _write_csv(OUTPUT_ROOT / "residual_correlation.csv", residuals)
    _write_csv(OUTPUT_ROOT / "control_analysis.csv", controls)
    _write_json(OUTPUT_ROOT / "classification.json", classification)
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    _research_note(summary)
    return summary


def launch() -> Dict[str, Any]:
    preflight_result = preflight()
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: List[Tuple[str, str, subprocess.Popen[Any], Any]] = []
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
    failures: List[Dict[str, Any]] = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append(
                {
                    "scene": scene,
                    "gpu": gpu,
                    "exit_code": code,
                    "log": str(logs / f"{scene}.log"),
                }
            )
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"frozen sensitivity workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    parser.add_argument("--sample-count", type=int, default=SAMPLES_PER_GROUP)
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        if args.sample_count < 1 or args.sample_count > SAMPLES_PER_GROUP:
            parser.error(f"--sample-count must be in [1,{SAMPLES_PER_GROUP}]")
        result = worker(args.scene, args.gpu, int(args.sample_count))
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True, default=_json_default, allow_nan=False))


if __name__ == "__main__":
    main()
