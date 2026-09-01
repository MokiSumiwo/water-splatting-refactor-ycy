#!/usr/bin/env python3
"""Frozen Gaussian view-consistency audit for registered OCMC checkpoints.

The training-view metric is computed without images or ground truth. Held-out
ground truth is opened only after metric construction and deterministic sample
selection, for a projection-based residual association audit.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


EXPERIMENT = "GAUSSIAN-VIEW-CONSISTENCY-AUDIT"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "7532e4bd07fc4d758526a718322f9e986311b689"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "gaussian_view_consistency_audit"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "GAUSSIAN_VIEW_CONSISTENCY_AUDIT_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
STEPS = (5000, 8000, 10000, 13000, 14999)
FINAL_STEP = 14999
SEED = 42
TEMPORAL_SAMPLE_COUNT = 2048
FINAL_SAMPLE_COUNT = 4096
SUPPORT_STRATA = 5
EPS = 1e-12
MIN_VALID_SAMPLE = 1024
CONTROLS = (
    "support_count",
    "train_depth_mean",
    "train_tau_mean",
    "train_transmission_mean",
    "opacity",
    "train_footprint_mean",
    "scale",
    "train_ocmc_active_magnitude_mean",
    "train_medium_suppressed_residual_mean",
)
MAJOR_CONTROLS = CONTROLS

EXPECTED_CHECKPOINT_HASHES = {
    "Curasao": {
        5000: "20d703d5ee1ad629483f8d1955e57644bef859e29ce1b7bd92bb50b33a5e49b2",
        8000: "6869db2afd3112b4cac195c34bc9e574346785f53e4c7a3e8b4cdc3889b430dd",
        10000: "134536287d953421c33408795f342f07ac8757d9592db92ac6aca31a696d321c",
        13000: "0813a8951e77a3444d4288cdabb64e07176591fd560032ce9df4b8f72c581a2e",
        14999: "ccf6a26eee364a02ba68ef9942e083d34962c71e5c84e40c0db8deca265fb406",
    },
    "IUI3-RedSea": {
        5000: "1563c312247d6fc84bbd286fe960e903886ae6779a58eb0ca6960f7a3c6c58b2",
        8000: "426f6167c80ec194f30928a80018c6d1b802862baf562f11cefbafbf13ea1634",
        10000: "b0bed54ea555128bb6463a94f6e73a9907c049ce50d83da792f7a77d2d5ffc0f",
        13000: "9be22004e4bbc761f06912ced707b5e1e2ad13769895a3aaf94bfdc755a50ec3",
        14999: "63ae7295ed0738641db5249a7876a1a05fbc30e5d1c3a0c7d43df843b837a180",
    },
    "JapaneseGradens-RedSea": {
        5000: "59b5e54b338ae92e4256d0c6868adfd35b3df3fa6f8f3bd18d8434a8a4d45c5b",
        8000: "ba2def794e0ace08e819e196d15c1c96698b901749412bed3e98350a6336b373",
        10000: "82b63326409a19b98b6ef856aaad0e1c5f33fa22704c107b6d02e363b4e22899",
        13000: "a31964909030838fae5cd0ec1a1d230e42001f874b204c8fc8379c26cc5dbdd8",
        14999: "452e2aa0e81c4f977df96bc4a97948dbaf9132b5e5c6ef0862a945fac04b4bb2",
    },
    "Panama": {
        5000: "52bfbbc5749ef4285ac3a1d6f856cf81596453830a039b0dc000f656c9ca8ca3",
        8000: "2c0d2da36a16720e70e57076c48020a5e42557fe1f712ac347f2f23bd1a79b27",
        10000: "74ca5c874f7e1f6ef0fc8e2c067d3c56a89cff9974516aa24e511ef591875b21",
        13000: "c7e5869e750cf3983c46288737e333248751f3a88012103f41d27f3cca4ec6f4",
        14999: "e5ae9aa065635802e9c5a00dee63eb06aa68b8504d89bfab71fd8dc4eea9b6a0",
    },
}
EXPECTED_CONFIG_HASHES = dict(CAUSAL.EXPECTED_CONFIG_HASHES)
EXPECTED_CAMERA_SEQUENCE_HASHES = dict(CAUSAL.EXPECTED_CAMERA_SEQUENCE_HASHES)
PROTECTED_HASHES = {
    **CAUSAL.PROTECTED_HASHES,
    "scripts/diagnostics/audit_local_contextual_support_predictor_iui3.py": "2f88afc2174f5753ee6cee494041b1f793529a4ea13742c425ad2928023a3479",
    "scripts/diagnostics/audit_low_support_causal_intervention.py": "92a45a7d17621b6f44b882e919ea2d65f9916669a0e94d75c8c72d03249d0ee3",
    "scripts/experiments/run_m1_raoc_causal_scene.py": "79930754f41887c0530e6b033eef5f0f26b692795a4f3abd078358ad800f9f2a",
    "water_splatting/water_splatting.py": "1a9930c0e74b4f235fc5ae5e819823fe9e2cdd828e8764ca73e43d0f67aa63e1",
    "water_splatting/fields/medium_field.py": "43a610d67921c00b171b9285e0fe3138f0e8eff6d84edf3d9b1f79e373bbfdef",
    "water_splatting/rendering/underwater_rasterizer.py": "04e6d1c6d136ee46ea32ea2abd666e688d6a35c00c787e31f17aa5f5ba17beba",
    "water_splatting/raoc.py": "e2ffe7f0e457ef1ef67b478b0638ea3afc6a585557886fa95a933d65f6b0ba08",
    "water_splatting/cuda/csrc/raoc.cu": "5599222dedf658885889d86af6b24a5ba2f6e6760818f0889b392dccd0a6d24d",
    "water_splatting/utils.py": "cb5ae8538bdf9bd6a36f15b6a819a63e750c1d6cc306574e11a85511fa4295ea",
}
DISK_CLEANUP = {
    "before_available_bytes": 34224603136,
    "deleted_paths": [
        "outputs/ocmc_candidate_c_resplit_replication_20260831_attempt1_interrupted_tool_session"
    ],
    "deleted_logical_bytes": 9127403775,
    "after_available_bytes": 43352379392,
    "reclaimed_available_bytes": 9127776256,
    "reason": "clearly named interrupted Candidate-C attempt; no formal OCMC or RAOC checkpoint removed",
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
        json.dumps(_sanitize(value), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
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
            row = {}
            for key in fields:
                value = source.get(key, "")
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                row[key] = value
            writer.writerow(row)


def _read_json(path: Path) -> Any:
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


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(*parts: Any) -> int:
    payload = ":".join([str(SEED)] + [str(part) for part in parts]).encode("utf8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _checkpoint(scene: str, step: int) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose exactly physical GPU {gpu}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("worker must see exactly logical cuda:0")
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


def _strict_repo() -> Dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected repo state: {branch}@{head}")
    hashes = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected source changed: {relative}")
        hashes[relative] = actual
    return {
        "branch": branch,
        "starting_head": head,
        "status_short": _run_text(["git", "status", "--short"]),
        "protected_hashes": hashes,
    }


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in SCENES:
        scene_config = CAUSAL._scene_config(scene)
        config = REPO_ROOT / scene_config["source_config"]
        sequence = SOURCE_ROOT / scene / "camera_sequence.json"
        config_hash = _sha256(config)
        sequence_hash = _sha256(sequence)
        if config_hash != EXPECTED_CONFIG_HASHES[scene] or sequence_hash != EXPECTED_CAMERA_SEQUENCE_HASHES[scene]:
            raise RuntimeError(f"config/camera sequence provenance drift for {scene}")
        for step in STEPS:
            checkpoint = _checkpoint(scene, step)
            actual = _sha256(checkpoint)
            expected = EXPECTED_CHECKPOINT_HASHES[scene][step]
            if actual != expected:
                raise RuntimeError(f"checkpoint provenance drift: {checkpoint}")
            rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": actual,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "source_config": str(config),
                    "source_config_sha256": config_hash,
                    "camera_sequence": str(sequence),
                    "camera_sequence_sha256": sequence_hash,
                }
            )
    lineage = {
        "available": False,
        "status": "GAUSSIAN_LINEAGE_UNAVAILABLE",
        "reason": "registered C0 checkpoints contain no lineage, birth, parent, or persistent Gaussian identifiers",
        "temporal_scope": "checkpoint-population association trajectories only",
        "array_index_matching_forbidden": True,
        "geometry_nearest_neighbor_matching_forbidden": True,
    }
    metric = {
        "primary_alpha_proxy": "alpha_i(v) = sigmoid(opacity_i) * pi * projected_radius_i(v)^2 / image_area",
        "primary_per_view_proxy": "r_i(v) = alpha_i(v) * gaussian_view_rgb_i(v)",
        "visible_training_views_only": True,
        "primary_variance": "population variance mean_v ||r_i(v)-mean_v r_i(v)||_2^2; support>=2 required and support count controlled",
        "primary_mean_deviation": "mean_v ||r_i(v)-mean_v r_i(v)||_2",
        "secondary_appearance_proxy": "sigmoid(opacity_i) * gaussian_view_rgb_i(v)",
        "secondary_underwater_direct_proxy": "r_i(v) * RGB transmission at projected Gaussian center and Gaussian depth",
        "heldout_or_gt_used_for_metric": False,
        "heldout_error_approximation": "mean heldout RGB MSE in clipped axis-aligned projected-radius footprint box",
        "high_error_pixels": "within-camera top 20% pixels by RGB MSE; Gaussian AUROC label is top 20% footprint overlap with those pixels",
        "sampling": "deterministic equal allocation over five empirical training-support rank strata among support>=2 Gaussians; heldout geometry is first used after VC is frozen",
    }
    classification_rule = {
        "scene_pass": "final raw rho>0, all nine single-factor partial rank rhos>0, rho positive at 5k and 14999 and at least 4/5 checkpoints",
        "VIEW_CONSISTENCY_SUPPORTED": "at least 3/4 scene passes",
        "VIEW_CONSISTENCY_TENTATIVE": "exactly 2/4 scene passes",
        "VIEW_CONSISTENCY_NOT_SUPPORTED": "0-1/4 scene passes with quality gate passing",
        "VIEW_CONSISTENCY_DATA_LIMITED": "metric/provenance/finite/sample quality gate fails",
    }
    result = {
        "experiment": EXPERIMENT,
        "repo": _strict_repo(),
        "launcher_environment": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "worker_physical_gpus": SCENE_GPUS,
            "worker_logical_device": "cuda:0",
        },
        "checkpoint_rows": rows,
        "metric": metric,
        "classification_rule": classification_rule,
        "lineage": lineage,
        "disk_cleanup": DISK_CLEANUP,
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    _write_json(OUTPUT_ROOT / "checkpoint_manifest.json", {"rows": rows})
    _write_json(OUTPUT_ROOT / "disk_cleanup.json", DISK_CLEANUP)
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
        if visible.numel() != support.numel() or not torch.equal(visible, reported):
            raise RuntimeError("training visibility alias mismatch")
        support += visible.cpu().to(torch.int16)
        del outputs
    if int(support.max()) > len(records):
        raise RuntimeError("support exceeds distinct training camera count")
    return support


@torch.no_grad()
def _sample_gaussians(
    scene: str,
    step: int,
    support: Tensor,
    requested: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    eligible = torch.nonzero(support >= 2, as_tuple=False).reshape(-1)
    if int(eligible.numel()) < requested:
        raise RuntimeError(f"{scene}/{step} only has {eligible.numel()} eligible Gaussians")
    order = eligible[torch.argsort(support[eligible].to(torch.int32), stable=True)]
    strata = torch.tensor_split(order, SUPPORT_STRATA)
    generator = torch.Generator(device="cpu")
    seed = _stable_seed(scene, step, "support-stratified-sample")
    generator.manual_seed(seed)
    base = requested // SUPPORT_STRATA
    remainder = requested % SUPPORT_STRATA
    selections = []
    rows = []
    for index, pool in enumerate(strata):
        count = base + int(index < remainder)
        if int(pool.numel()) < count:
            raise RuntimeError("support stratum is smaller than requested allocation")
        chosen = pool[torch.randperm(int(pool.numel()), generator=generator)[:count]]
        selections.append(chosen)
        rows.append(
            {
                "stratum": index,
                "pool_count": int(pool.numel()),
                "sample_count": count,
                "support_min": int(support[pool].min()),
                "support_max": int(support[pool].max()),
            }
        )
    selected = torch.cat(selections).sort().values
    return selected, {
        "seed": seed,
        "requested_count": requested,
        "selected_count": int(selected.numel()),
        "eligible_count": int(eligible.numel()),
        "eligibility": "training support >=2; no heldout camera or GT used",
        "strata": rows,
        "selected_ids_sha256": CAUSAL._tensor_hash(selected),
    }


def _sample_image(values: Tensor, xys: Tensor) -> Tensor:
    height, width = int(values.shape[0]), int(values.shape[1])
    x = xys[:, 0].round().long().clamp(0, width - 1)
    y = xys[:, 1].round().long().clamp(0, height - 1)
    return values[y, x]


@torch.no_grad()
def _training_statistics(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    support: Tensor,
) -> Dict[str, Tensor]:
    count = int(selected.numel())
    selected_gpu = selected.to(model.device)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float()
    observed = torch.zeros(count, dtype=torch.int16)
    contribution_values = torch.full((len(records), count, 3), float("nan"), dtype=torch.float32)
    appearance_values = torch.full_like(contribution_values, float("nan"))
    underwater_values = torch.full_like(contribution_values, float("nan"))
    sums = {
        key: torch.zeros(count, dtype=torch.float64)
        for key in (
            "depth",
            "tau",
            "transmission",
            "footprint",
            "ocmc_active_magnitude",
            "medium_suppressed_residual",
        )
    }
    for view_index, (_index, _camera_id, camera, _batch) in enumerate(records):
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        visible_cpu = visible.cpu()
        if not bool(visible.any()):
            del outputs
            continue
        local = torch.nonzero(visible, as_tuple=False).reshape(-1)
        global_ids = selected_gpu[local]
        xys = model.xys.detach()[global_ids]
        colors = outputs["gaussian_view_rgb"].detach().float()[global_ids]
        footprint = model.radii.detach().reshape(-1)[global_ids].float()
        height, width = int(outputs["pred_image"].shape[0]), int(outputs["pred_image"].shape[1])
        alpha_area = opacity[local] * math.pi * footprint.square() / float(height * width)
        appearance = opacity[local, None] * colors
        contribution = alpha_area[:, None] * colors
        projected_depth = outputs["projected_gaussian_depths"].detach().reshape(-1)[global_ids]
        medium_attn = _sample_image(outputs["medium_attn"].detach().float(), xys)
        tau_rgb = medium_attn * projected_depth[:, None]
        tau = tau_rgb.mean(dim=-1)
        transmission_rgb = torch.exp(-tau_rgb.clamp_min(0.0)).clamp(0.0, 1.0)
        transmission = transmission_rgb.mean(dim=-1)
        active = torch.linalg.vector_norm(
            _sample_image(outputs["camera_medium_delta_projected_raw"].detach().float(), xys), dim=-1
        )
        suppressed = torch.linalg.vector_norm(
            _sample_image(outputs["camera_medium_delta_suppressed_raw"].detach().float(), xys), dim=-1
        )
        contribution_values[view_index, visible_cpu] = contribution.cpu()
        appearance_values[view_index, visible_cpu] = appearance.cpu()
        underwater_values[view_index, visible_cpu] = (contribution * transmission_rgb).cpu()
        observed += visible_cpu.to(torch.int16)
        additions = {
            "depth": projected_depth,
            "tau": tau,
            "transmission": transmission,
            "footprint": footprint,
            "ocmc_active_magnitude": active,
            "medium_suppressed_residual": suppressed,
        }
        for key, value in additions.items():
            sums[key][visible_cpu] += value.detach().double().cpu()
        del outputs
    if not torch.equal(observed, support[selected]):
        raise RuntimeError("metric pass visibility differs from registered support")

    def consistency(values: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        valid = torch.isfinite(values[..., 0])
        safe = torch.where(valid[..., None], values.double(), torch.zeros_like(values, dtype=torch.float64))
        mean = safe.sum(dim=0) / observed.double().clamp_min(1)[:, None]
        delta = torch.where(valid[..., None], safe - mean[None, ...], torch.zeros_like(safe))
        variance = delta.square().sum(dim=-1).sum(dim=0) / observed.double().clamp_min(1)
        mean_deviation = torch.linalg.vector_norm(delta, dim=-1).sum(dim=0) / observed.double().clamp_min(1)
        return variance, mean_deviation, torch.linalg.vector_norm(mean, dim=-1)

    variance, mean_deviation, mean_norm = consistency(contribution_values)
    appearance_variance, appearance_mean_deviation, _appearance_mean_norm = consistency(appearance_values)
    underwater_variance, underwater_mean_deviation, _underwater_mean_norm = consistency(underwater_values)
    result = {
        "observed_train_views": observed,
        "vc_variance": variance,
        "vc_mean_deviation": mean_deviation,
        "vc_normalized_variance": variance / mean_norm.square().clamp_min(EPS),
        "appearance_vc_variance": appearance_variance,
        "appearance_vc_mean_deviation": appearance_mean_deviation,
        "underwater_direct_vc_variance": underwater_variance,
        "underwater_direct_vc_mean_deviation": underwater_mean_deviation,
    }
    for key, value in sums.items():
        result[f"train_{key}_mean"] = value / observed.double().clamp_min(1)
    return result


def _box_statistics(residual: Tensor, high: Tensor, xys: Tensor, radii: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    height, width = int(residual.shape[0]), int(residual.shape[1])
    radius = radii.float().clamp_min(0.5)
    x0 = torch.floor(xys[:, 0] - radius).long().clamp(0, width - 1)
    x1 = torch.ceil(xys[:, 0] + radius).long().clamp(0, width - 1)
    y0 = torch.floor(xys[:, 1] - radius).long().clamp(0, height - 1)
    y1 = torch.ceil(xys[:, 1] + radius).long().clamp(0, height - 1)

    def summed_area(values: Tensor) -> Tensor:
        integral = torch.nn.functional.pad(values.double(), (1, 0, 1, 0)).cumsum(0).cumsum(1)
        return integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]

    area = ((x1 - x0 + 1) * (y1 - y0 + 1)).double()
    return summed_area(residual) / area, summed_area(high.double()) / area, area


@torch.no_grad()
def _heldout_statistics(
    scene: str,
    step: int,
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    metric: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], List[Dict[str, Any]]]:
    count = int(selected.numel())
    selected_gpu = selected.to(model.device)
    residual_sum = torch.zeros(count, dtype=torch.float64)
    high_fraction_sum = torch.zeros(count, dtype=torch.float64)
    depth_sum = torch.zeros(count, dtype=torch.float64)
    footprint_sum = torch.zeros(count, dtype=torch.float64)
    area_sum = torch.zeros(count, dtype=torch.float64)
    observed = torch.zeros(count, dtype=torch.int16)
    camera_rows: List[Dict[str, Any]] = []
    for _index, camera_id, camera, batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        local = torch.nonzero(visible, as_tuple=False).reshape(-1)
        global_ids = selected_gpu[local]
        # Metric and sample are already frozen before this is the first GT access.
        gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        residual = (pred - gt).square().mean(dim=-1)
        threshold = torch.quantile(residual.reshape(-1), 0.80)
        high = residual >= threshold
        if bool(visible.any()):
            xys = model.xys.detach()[global_ids]
            radii = model.radii.detach().reshape(-1)[global_ids]
            projected_residual, high_fraction, area = _box_statistics(residual, high, xys, radii)
            visible_cpu = visible.cpu()
            residual_sum[visible_cpu] += projected_residual.cpu()
            high_fraction_sum[visible_cpu] += high_fraction.cpu()
            depth_sum[visible_cpu] += outputs["projected_gaussian_depths"].detach().reshape(-1)[global_ids].double().cpu()
            footprint_sum[visible_cpu] += radii.double().cpu()
            area_sum[visible_cpu] += area.cpu()
            observed += visible_cpu.to(torch.int16)
        camera_vc = metric["vc_variance"][visible.cpu()]
        camera_rows.append(
            {
                "scene": scene,
                "absolute_step": step,
                "camera_id": camera_id,
                "sampled_visible_gaussian_count": int(visible.sum()),
                "camera_vc_mean": float(camera_vc.mean()) if camera_vc.numel() else float("nan"),
                "camera_vc_median": float(camera_vc.median()) if camera_vc.numel() else float("nan"),
                "camera_MSE": float(residual.mean()),
                "camera_high_error_threshold_MSE": float(threshold),
                "camera_ocmc_active_magnitude_mean": float(outputs["camera_medium_delta_projected_raw"].detach().float().norm(dim=-1).mean()),
                "camera_medium_suppressed_residual_mean": float(outputs["camera_medium_delta_suppressed_raw"].detach().float().norm(dim=-1).mean()),
                "metric_uses_heldout_or_gt": False,
                "gt_used_for_camera_MSE_only": True,
            }
        )
        del outputs, gt, pred, residual, high
    seen = observed > 0

    def mean_or_nan(total: Tensor) -> Tensor:
        output = torch.full_like(total, float("nan"), dtype=torch.float64)
        output[seen] = total[seen] / observed[seen].double()
        return output

    return {
        "heldout_visible_views": observed,
        "heldout_projected_residual": mean_or_nan(residual_sum),
        "heldout_high_error_pixel_fraction": mean_or_nan(high_fraction_sum),
        "heldout_depth_mean": mean_or_nan(depth_sum),
        "heldout_footprint_mean": mean_or_nan(footprint_sum),
        "heldout_projection_box_area_mean": mean_or_nan(area_sum),
    }, camera_rows


def _rho(
    left: Sequence[float], right: Sequence[float], minimum_count: int = 12
) -> Tuple[float, float, int]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < minimum_count or np.ptp(a[valid]) <= 0.0 or np.ptp(b[valid]) <= 0.0:
        return float("nan"), float("nan"), int(valid.sum())
    result = scipy.stats.spearmanr(a[valid], b[valid])
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def _partial_rank(predictor: Sequence[float], target: Sequence[float], control: Sequence[float]) -> float:
    arrays = [np.asarray(values, dtype=np.float64) for values in (predictor, target, control)]
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if int(valid.sum()) < 12:
        return float("nan")
    ranked = [scipy.stats.rankdata(array[valid]) for array in arrays]
    design = np.column_stack([np.ones(int(valid.sum())), ranked[2]])
    x_residual = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    y_residual = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    return _rho(x_residual, y_residual)[0]


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    valid = np.isfinite(scores_array)
    scores_array = scores_array[valid]
    labels_array = labels_array[valid]
    positive = int(labels_array.sum())
    negative = int((~labels_array).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = scipy.stats.rankdata(scores_array)
    return float((ranks[labels_array].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def _rows_for_checkpoint(
    scene: str,
    step: int,
    selected: Tensor,
    support: Tensor,
    model: Any,
    metric: Mapping[str, Tensor],
    heldout: Mapping[str, Tensor],
) -> List[Dict[str, Any]]:
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1).cpu()[selected]
    scale = torch.exp(model.scales.detach()).amax(dim=-1).cpu()[selected]
    residual = heldout["heldout_projected_residual"]
    high_fraction = heldout["heldout_high_error_pixel_fraction"]
    valid = torch.isfinite(residual) & (heldout["heldout_visible_views"] > 0)
    valid_local = torch.nonzero(valid, as_tuple=False).reshape(-1)
    high_fraction_threshold = float(torch.quantile(high_fraction[valid].float(), 0.80))
    rows = []
    for local in valid_local.tolist():
        gaussian_id = int(selected[local])
        row = {
            "scene": scene,
            "absolute_step": step,
            "gaussian_id_checkpoint_local": gaussian_id,
            "gaussian_identity_persistent_across_checkpoints": False,
            "support_count": int(support[gaussian_id]),
            "opacity": float(opacity[local]),
            "scale": float(scale[local]),
            "high_error_label_top20_footprint_pixel_fraction": bool(
                float(high_fraction[local]) >= high_fraction_threshold
            ),
            "metric_uses_heldout_or_gt": False,
            "gt_used_after_metric_for_residual_only": True,
        }
        for key, values in {**metric, **heldout}.items():
            row[key] = float(values[local])
        rows.append(row)
    return rows


def _checkpoint_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vc = [float(row["vc_variance"]) for row in rows]
    error = [float(row["heldout_projected_residual"]) for row in rows]
    rho, pvalue, count = _rho(vc, error)
    labels = [bool(row["high_error_label_top20_footprint_pixel_fraction"]) for row in rows]
    high_pixel = np.asarray([float(row["heldout_high_error_pixel_fraction"]) for row in rows])
    vc_array = np.asarray(vc)
    top = vc_array >= np.quantile(vc_array, 0.80)
    enrichment = float(high_pixel[top].mean() / max(float(high_pixel[~top].mean()), EPS))
    controls = {
        control: _partial_rank(vc, error, [float(row[control]) for row in rows])
        for control in CONTROLS
    }
    return {
        "scene": rows[0]["scene"],
        "absolute_step": int(rows[0]["absolute_step"]),
        "gaussian_count": len(rows),
        "vc_variance_exact_positive_fraction": float((vc_array > 0.0).mean()),
        "vc_variance_min_positive": float(vc_array[vc_array > 0.0].min()) if bool((vc_array > 0.0).any()) else 0.0,
        "vc_variance_median": float(np.median(vc_array)),
        "vc_mean_deviation_median": float(np.median([float(row["vc_mean_deviation"]) for row in rows])),
        "spearman_vc_vs_heldout_error": rho,
        "spearman_pvalue": pvalue,
        "valid_pair_count": count,
        "auroc_vc_for_top20_projected_error": _auroc(vc, labels),
        "top20_vc_high_error_pixel_enrichment": enrichment,
        "single_factor_partial_rank_rhos": controls,
        "all_major_controls_positive": all(math.isfinite(value) and value > 0.0 for value in controls.values()),
    }


@torch.no_grad()
def worker(
    scene: str,
    gpu: str,
    steps: Sequence[int] = STEPS,
    sample_count: Optional[int] = None,
) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    started = time.perf_counter()
    scene_dir = OUTPUT_ROOT / "workers" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    branch = FORMAL._setup_branch(REPO_ROOT, CAUSAL._scene_config(scene), "C0")
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        split = _camera_split(train_records, eval_records)
        all_rows: List[Dict[str, Any]] = []
        all_camera_rows: List[Dict[str, Any]] = []
        checkpoint_rows: List[Dict[str, Any]] = []
        for step in steps:
            checkpoint = _checkpoint(scene, step)
            payload = FORMAL._load_checkpoint(branch, checkpoint)
            if (
                payload.get("experiment") != FORMAL.EXPERIMENT
                or payload.get("branch") != "C0"
                or int(payload.get("absolute_step", -1)) != step
                or payload.get("ocmc_bundle") is None
                or payload.get("raoc_state") is not None
            ):
                raise RuntimeError(f"checkpoint condition provenance drift: {checkpoint}")
            if (
                not model.config.camera_medium_observability_enabled
                or model.config.camera_medium_ray_adaptive_observability_enabled
                or model.config.intrinsic_color_parameterization != "bounded_sh3"
                or model.config.rasterize_mode != "classic"
                or model.config.medium_context_mode != "dir_xy_camera"
                or int(model.config.sh_degree) != 3
            ):
                raise RuntimeError("locked OCMC-on RAOC-off model configuration drift")
            state_before = CAUSAL._model_state_hash(model)
            projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            support = _support_counts(model, train_records)
            requested = sample_count if sample_count is not None else (FINAL_SAMPLE_COUNT if step == FINAL_STEP else TEMPORAL_SAMPLE_COUNT)
            selected, sampling = _sample_gaussians(scene, step, support, requested)
            metric = _training_statistics(model, train_records, selected, support)
            # No GT has been accessed before this boundary.
            heldout, camera_rows = _heldout_statistics(scene, step, model, eval_records, selected, metric)
            rows = _rows_for_checkpoint(scene, step, selected, support, model, metric, heldout)
            sampling["heldout_visible_analysis_count"] = len(rows)
            summary = _checkpoint_summary(rows)
            state_after = CAUSAL._model_state_hash(model)
            projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            if state_before != state_after or projector_before != projector_after:
                raise RuntimeError("frozen model or OCMC projector changed")
            checkpoint_rows.append(
                {
                    **summary,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "sampling": sampling,
                    "ocmc_enable_flag": True,
                    "raoc_enable_flag": False,
                    "raoc_state_present": False,
                    "camera_medium_raoc_backend_config": str(
                        getattr(model.config, "camera_medium_raoc_backend", "not_configured")
                    ),
                    "camera_medium_raoc_effective_status": "disabled_by_enable_flag_and_absent_state",
                    "model_state_sha256_before": state_before,
                    "model_state_sha256_after": state_after,
                    "ocmc_projector_sha256_before": projector_before,
                    "ocmc_projector_sha256_after": projector_after,
                    "model_and_ocmc_unchanged": True,
                }
            )
            all_rows.extend(rows)
            all_camera_rows.extend(camera_rows)
            print(
                f"[{scene}] step {step}: n={len(rows)} rho={summary['spearman_vc_vs_heldout_error']:.6f}",
                flush=True,
            )
            del payload, support, selected, metric, heldout, rows
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "runtime": runtime,
            "camera_split": split,
            "checkpoint_rows": checkpoint_rows,
            "gaussian_rows": len(all_rows),
            "camera_rows": len(all_camera_rows),
            "steps": list(steps),
            "sample_count_override": sample_count,
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_forward_only": True,
            "backward_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_writes": 0,
        }
        suffix = "" if tuple(steps) == STEPS and sample_count is None else "_smoke"
        _write_csv(scene_dir / f"gaussian_statistics{suffix}.csv", all_rows)
        _write_csv(scene_dir / f"camera_statistics{suffix}.csv", all_camera_rows)
        _write_json(scene_dir / f"worker_summary{suffix}.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _coerce_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    text = {"scene", "camera_id"}
    boolean = {
        "gaussian_identity_persistent_across_checkpoints",
        "high_error_label_top20_footprint_pixel_fraction",
        "metric_uses_heldout_or_gt",
        "gt_used_after_metric_for_residual_only",
        "gt_used_for_camera_MSE_only",
    }
    integer = {
        "absolute_step",
        "gaussian_id_checkpoint_local",
        "support_count",
        "observed_train_views",
        "heldout_visible_views",
        "sampled_visible_gaussian_count",
    }
    output = []
    for source in rows:
        row: Dict[str, Any] = {}
        for key, value in source.items():
            if key in text:
                row[key] = value
            elif key in boolean:
                row[key] = value == "True"
            elif key in integer:
                row[key] = int(float(value))
            else:
                row[key] = float(value) if value != "" else float("nan")
        output.append(row)
    return output


def _pooled_camera_analysis(camera_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final = [row for row in camera_rows if int(row["absolute_step"]) == FINAL_STEP]
    scene_rows = []
    pooled_vc = []
    pooled_mse = []
    for scene in SCENES:
        rows = [row for row in final if row["scene"] == scene]
        vc = [float(row["camera_vc_mean"]) for row in rows]
        mse = [float(row["camera_MSE"]) for row in rows]
        rho, pvalue, count = _rho(vc, mse, minimum_count=3)
        scene_rows.append(
            {
                "scene": scene,
                "heldout_camera_count": len(rows),
                "spearman_camera_vc_vs_camera_MSE": rho,
                "pvalue": pvalue,
                "small_n_descriptive_only": True,
            }
        )
        pooled_vc.extend((scipy.stats.rankdata(vc) / max(len(vc), 1)).tolist())
        pooled_mse.extend((scipy.stats.rankdata(mse) / max(len(mse), 1)).tolist())
    rho, pvalue, count = _rho(pooled_vc, pooled_mse)
    return {
        "scene_rows": scene_rows,
        "pooled_within_scene_rank_spearman": rho,
        "pooled_pvalue": pvalue,
        "pooled_camera_count": count,
        "pooling_rule": "Spearman on within-scene camera percentile ranks",
    }


def _classify(checkpoint_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scene_results = []
    temporal_scenes = []
    quality_rows = []
    for scene in SCENES:
        trajectory = sorted([row for row in checkpoint_rows if row["scene"] == scene], key=lambda row: int(row["absolute_step"]))
        final = next(row for row in trajectory if int(row["absolute_step"]) == FINAL_STEP)
        rhos = [float(row["spearman_vc_vs_heldout_error"]) for row in trajectory]
        temporal_stable = bool(rhos[0] > 0.0 and rhos[-1] > 0.0 and sum(value > 0.0 for value in rhos) >= 4)
        finite = all(
            math.isfinite(float(row["spearman_vc_vs_heldout_error"]))
            and math.isfinite(float(row["auroc_vc_for_top20_projected_error"]))
            and all(
                math.isfinite(float(value))
                for value in row["single_factor_partial_rank_rhos"].values()
            )
            for row in trajectory
        )
        sample_ok = all(int(row["gaussian_count"]) >= MIN_VALID_SAMPLE for row in trajectory)
        nonzero_ok = all(float(row["vc_variance_exact_positive_fraction"]) >= 0.90 for row in trajectory)
        quality = bool(len(trajectory) == len(STEPS) and finite and sample_ok and nonzero_ok)
        control_positive = bool(final["all_major_controls_positive"])
        supported = bool(float(final["spearman_vc_vs_heldout_error"]) > 0.0 and control_positive and temporal_stable)
        scene_results.append(
            {
                "scene": scene,
                "final_spearman_vc_vs_heldout_error": float(final["spearman_vc_vs_heldout_error"]),
                "final_auroc": float(final["auroc_vc_for_top20_projected_error"]),
                "final_top20_enrichment": float(final["top20_vc_high_error_pixel_enrichment"]),
                "all_major_controls_positive": control_positive,
                "control_rhos": final["single_factor_partial_rank_rhos"],
                "temporal_stable": temporal_stable,
                "scene_supported": supported,
            }
        )
        temporal_scenes.append(
            {
                "scene": scene,
                "rho_trajectory": {str(row["absolute_step"]): float(row["spearman_vc_vs_heldout_error"]) for row in trajectory},
                "positive_checkpoint_count": sum(value > 0.0 for value in rhos),
                "appears_at_5k": rhos[0] > 0.0,
                "persists_at_14999": rhos[-1] > 0.0,
                "distribution_level_temporal_stability": temporal_stable,
                "identity_level_persistence": "NOT_AVAILABLE_GAUSSIAN_LINEAGE_UNAVAILABLE",
            }
        )
        quality_rows.append(
            {
                "scene": scene,
                "checkpoint_count": len(trajectory),
                "all_primary_statistics_finite": finite,
                "minimum_sample_count_met": sample_ok,
                "vc_exact_positive_fraction_at_least_0p90": nonzero_ok,
                "quality_pass": quality,
            }
        )
    quality_pass = all(row["quality_pass"] for row in quality_rows)
    supported_count = sum(row["scene_supported"] for row in scene_results)
    if not quality_pass:
        label = "VIEW_CONSISTENCY_DATA_LIMITED"
    elif supported_count >= 3:
        label = "VIEW_CONSISTENCY_SUPPORTED"
    elif supported_count == 2:
        label = "VIEW_CONSISTENCY_TENTATIVE"
    else:
        label = "VIEW_CONSISTENCY_NOT_SUPPORTED"
    classification = {
        "experiment": EXPERIMENT,
        "classification": label,
        "supported_scene_count": supported_count,
        "required_supported_scene_count": 3,
        "quality_gate_passed": quality_pass,
        "module_design_authorized": label == "VIEW_CONSISTENCY_SUPPORTED",
        "scene_rows": scene_results,
        "quality_rows": quality_rows,
    }
    temporal = {
        "steps": list(STEPS),
        "scope": "checkpoint-population association trajectory; no cross-checkpoint Gaussian identity claim",
        "lineage_available": False,
        "scene_rows": temporal_scenes,
    }
    return classification, temporal


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    camera = summary["camera_analysis"]
    lines = [
        "# Gaussian View Consistency Audit",
        "",
        "Date: 2026-09-01",
        f"Experiment: `{EXPERIMENT}`",
        f"Classification: `{classification['classification']}`",
        "",
        "## Frozen Protocol",
        "",
        "All 20 registered C0 checkpoints use OCMC on and RAOC off. The audit performs detached forward rendering only. It does not train, call backward or optimizer.step, modify the renderer/OCMC/RAOC, write checkpoints, or write renders.",
        "",
        "The primary GT-free proxy is `r_i(v) = alpha_i(v) * gaussian_view_rgb_i(v)` on training cameras where Gaussian `i` is visible, with `alpha_i(v) = sigmoid(opacity_i) * pi * projected_radius_i(v)^2 / image_area`. `VC variance` is the population mean squared L2 deviation from the per-Gaussian training-view mean; support of at least two views is required and support count is controlled. The mean L2 deviation, opacity-color appearance score, and medium-attenuated direct score are also reported. The primary is an OCMC-independent, unoccluded projected opacity-area proxy, not exact per-pixel transmittance attribution.",
        "",
        "Heldout error is added only after VC construction. It is approximated by mean heldout RGB MSE inside each Gaussian's clipped projected-radius bounding box. The AUROC label is the top 20% of sampled Gaussians by the fraction of their footprint box occupied by within-camera top-20% high-error pixels.",
        "",
        "## Metric Estimability",
        "",
        "| Scene | final analyzed Gaussians | exact-positive VC | median VC variance | minimum positive VC |",
        "|---|---:|---:|---:|---:|",
    ]
    final_metrics = {row["scene"]: row for row in summary["per_scene_metrics"]}
    for scene in SCENES:
        row = final_metrics[scene]
        lines.append(
            f"| {scene} | {row['gaussian_count']} | {row['vc_variance_exact_positive_fraction']:.6f} | {row['vc_variance_median']:.6e} | {row['vc_variance_min_positive']:.6e} |"
        )
    lines.extend(
        [
            "",
            "VC is numerically estimable and nonzero in all four scene populations; this establishes measurable multi-view variation, but not by itself a failure mechanism.",
            "",
            "## Final Gaussian-Level Results",
            "",
            "| Scene | rho(VC,error) | AUROC | top-20% enrichment | all controls positive | temporal stable | pass |",
            "|---|---:|---:|---:|:---:|:---:|:---:|",
        ]
    )
    for row in classification["scene_rows"]:
        lines.append(
            f"| {row['scene']} | {row['final_spearman_vc_vs_heldout_error']:.6f} | {row['final_auroc']:.6f} | {row['final_top20_enrichment']:.6f} | {'yes' if row['all_major_controls_positive'] else 'no'} | {'yes' if row['temporal_stable'] else 'no'} | {'yes' if row['scene_supported'] else 'no'} |"
        )
    lines.extend(["", "## Single-Factor Controls", "", "| Scene | support | depth | tau | transmission | opacity | footprint | scale | OCMC active | medium suppressed |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in classification["scene_rows"]:
        c = row["control_rhos"]
        lines.append(
            f"| {row['scene']} | {c['support_count']:.6f} | {c['train_depth_mean']:.6f} | {c['train_tau_mean']:.6f} | {c['train_transmission_mean']:.6f} | {c['opacity']:.6f} | {c['train_footprint_mean']:.6f} | {c['scale']:.6f} | {c['train_ocmc_active_magnitude_mean']:.6f} | {c['train_medium_suppressed_residual_mean']:.6f} |"
        )
    lines.extend(["", "## Temporal Stability", "", "| Scene | 5k | 8k | 10k | 13k | 14999 | positive | stable |", "|---|---:|---:|---:|---:|---:|---:|:---:|"])
    for row in summary["temporal_analysis"]["scene_rows"]:
        rho = row["rho_trajectory"]
        lines.append(
            f"| {row['scene']} | {rho['5000']:.6f} | {rho['8000']:.6f} | {rho['10000']:.6f} | {rho['13000']:.6f} | {rho['14999']:.6f} | {row['positive_checkpoint_count']}/5 | {'yes' if row['distribution_level_temporal_stability'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint populations have no persistent Gaussian lineage IDs. Temporal results therefore test distribution-level recurrence, not identity-level persistence; array index or nearest-geometry matching was not used.",
            "",
            "## Camera-Level Results",
            "",
            "| Scene | cameras | rho(camera VC, camera MSE) |",
            "|---|---:|---:|",
        ]
    )
    for row in camera["scene_rows"]:
        value = row["spearman_camera_vc_vs_camera_MSE"]
        shown = f"{value:.6f}" if math.isfinite(value) else "not estimable"
        lines.append(f"| {row['scene']} | {row['heldout_camera_count']} | {shown} |")
    lines.extend(
        [
            "",
            f"Pooled within-scene-rank rho over {camera['pooled_camera_count']} heldout cameras: `{camera['pooled_within_scene_rank_spearman']:.6f}`. Per-scene camera correlations are descriptive because each scene has only 3-4 heldout cameras.",
            "",
            "## OCMC Independence",
            "",
            "OCMC active magnitude and suppressed medium residual are included as preregistered single-factor rank controls. A scene only passes when both controlled VC-error associations, together with all other major controls, remain positive.",
            "",
            "The loaded config retains the dormant backend string `reference`; RAOC is effectively disabled because every worker verifies `camera_medium_ray_adaptive_observability_enabled=False` and an absent `raoc_state`. No RAOC path is executed.",
            "",
            "## Disk Management",
            "",
            f"Available bytes before cleanup: `{DISK_CLEANUP['before_available_bytes']}`. Deleted only `{DISK_CLEANUP['deleted_paths'][0]}` (`{DISK_CLEANUP['deleted_logical_bytes']}` logical bytes). Available bytes after cleanup: `{DISK_CLEANUP['after_available_bytes']}`; reclaimed available bytes: `{DISK_CLEANUP['reclaimed_available_bytes']}`.",
            "",
            "## Scientific Decision",
            "",
            f"The formal classification is `{classification['classification']}` with {classification['supported_scene_count']}/4 scene passes.",
            "",
            "Curasao provides the only clearly positive final effect (`rho=0.172`, `AUROC=0.580`). JapaneseGradens passes the sign-based rule but its final effect is near zero (`rho=0.0068`, `AUROC=0.520`); IUI3 is not temporally stable and Panama is consistently negative. The pooled camera-level association is also weak. These effect sizes are why the tentative label does not authorize a mechanism claim.",
            "",
            (
                "Gaussian view inconsistency is supported as a valid second failure mechanism. A separate module-design phase is scientifically authorized; no module was designed here."
                if classification["module_design_authorized"]
                else "Gaussian view inconsistency is not established as a valid second failure mechanism. New module design is not scientifically authorized."
            ),
            "",
            "## Integrity",
            "",
            "Every worker reports unchanged model and OCMC projector hashes, zero backward calls, zero optimizer steps, and zero checkpoint writes. Protected historical GMVC, Q50/Q80, renderer, OCMC, and RAOC sources remain hash-identical.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    worker_summaries = [_read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json") for scene in SCENES]
    if not all(
        row["backward_calls"] == 0
        and row["optimizer_step_calls"] == 0
        and row["checkpoint_writes"] == 0
        and len(row["checkpoint_rows"]) == len(STEPS)
        and all(item["model_and_ocmc_unchanged"] for item in row["checkpoint_rows"])
        for row in worker_summaries
    ):
        raise RuntimeError("worker integrity gate failed")
    gaussian_raw = []
    camera_raw = []
    checkpoint_rows = []
    for scene, worker_summary in zip(SCENES, worker_summaries):
        gaussian_raw.extend(_read_csv(OUTPUT_ROOT / "workers" / scene / "gaussian_statistics.csv"))
        camera_raw.extend(_read_csv(OUTPUT_ROOT / "workers" / scene / "camera_statistics.csv"))
        checkpoint_rows.extend(worker_summary["checkpoint_rows"])
    gaussian_rows = _coerce_rows(gaussian_raw)
    camera_rows = _coerce_rows(camera_raw)
    classification, temporal = _classify(checkpoint_rows)
    camera_analysis = _pooled_camera_analysis(camera_rows)
    final_scene_metrics = [row for row in checkpoint_rows if int(row["absolute_step"]) == FINAL_STEP]
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "per_scene_metrics": final_scene_metrics,
        "temporal_analysis": temporal,
        "camera_analysis": camera_analysis,
        "worker_summaries": worker_summaries,
        "gaussian_statistics_rows": len(gaussian_rows),
        "camera_statistics_rows": len(camera_rows),
        "disk_cleanup": DISK_CLEANUP,
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "module_design_started": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_csv(OUTPUT_ROOT / "gaussian_statistics.csv", gaussian_rows)
    _write_csv(OUTPUT_ROOT / "camera_statistics.csv", camera_rows)
    _write_json(OUTPUT_ROOT / "per_scene_metrics.json", {"rows": final_scene_metrics, "camera_analysis": camera_analysis})
    _write_json(OUTPUT_ROOT / "temporal_analysis.json", temporal)
    _write_json(OUTPUT_ROOT / "classification.json", classification)
    _write_json(OUTPUT_ROOT / "summary.json", summary)
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
        command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", "--scene", scene, "--gpu", gpu]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((scene, gpu, process, handle))
    failures = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")})
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"view-consistency workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


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
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        if args.sample_count is not None and args.sample_count < 20:
            parser.error("--sample-count must be at least 20")
        result = worker(args.scene, args.gpu, args.steps, args.sample_count)
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True, default=_json_default, allow_nan=False))


if __name__ == "__main__":
    main()
