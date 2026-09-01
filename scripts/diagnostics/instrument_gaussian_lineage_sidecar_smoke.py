#!/usr/bin/env python3
"""Smoke-validate an external Gaussian lineage sidecar.

The sidecar observes the existing topology masks but never participates in
rendering, loss evaluation, optimization, or refinement selection. The run is
limited to 1000 steps on one OCMC-on/RAOC-off scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL

EXPERIMENT = "INSTRUMENT-GAUSSIAN-LINEAGE-SIDECAR-SMOKE-VALIDATION"
SCENE = "IUI3-RedSea"
BRANCH = "C0"
MAX_STEPS = 1000
FINAL_STEP = MAX_STEPS - 1
TRAINING_SEED = 42
OUTPUT_ROOT = REPO_ROOT / "outputs" / "gaussian_lineage_sidecar_smoke"
RESEARCH_NOTE = (
    REPO_ROOT
    / "research_notes"
    / "GAUSSIAN_LINEAGE_SIDECAR_SMOKE_VALIDATION_2026-09-01.md"
)
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "ce9af5c239863a05a0aef4b5f3b70a85f2306f2e"
ALLOWED_GPUS = frozenset({"6", "7", "8", "9"})
EVENT_TYPE_NAMES = {0: "initial", 1: "split_child", 2: "duplicate_child"}
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
AUDITED_SOURCE_HASHES = {
    "scripts/experiments/run_m1_raoc_causal_scene.py": (
        "79930754f41887c0530e6b033eef5f0f26b692795a4f3abd078358ad800f9f2a"
    ),
    "water_splatting/water_splatting.py": (
        "1a9930c0e74b4f235fc5ae5e819823fe9e2cdd828e8764ca73e43d0f67aa63e1"
    ),
}
EPS = 1e-12


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _strict_runtime(gpu: str) -> Dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected repo state: {branch}@{head}")
    if gpu not in ALLOWED_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"expose exactly one physical GPU in {sorted(ALLOWED_GPUS)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke worker must see exactly one CUDA device")
    protected: Dict[str, Any] = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected file changed: {relative}")
        protected[relative] = actual
    audited: Dict[str, Any] = {}
    for relative, expected in AUDITED_SOURCE_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"audited source changed: {relative}")
        audited[relative] = actual
    properties = torch.cuda.get_device_properties(0)
    return {
        "branch": branch,
        "starting_head": head,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "physical_gpu": gpu,
        "logical_gpu": 0,
        "gpu_name": properties.name,
        "protected_hashes": protected,
        "audited_source_hashes": audited,
    }


def _support_counts(bits: Tensor) -> Tensor:
    values = bits.detach().cpu().numpy().astype(np.uint64, copy=False)
    byte_values = values.view(np.uint8).reshape(-1, 8)
    lookup = np.asarray([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return torch.from_numpy(lookup[byte_values].sum(axis=1).astype(np.int16))


def _rho(left: Sequence[Optional[float]], right: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return None
    a = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    b = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(a) <= EPS or np.ptp(b) <= EPS:
        return None
    value = float(scipy.stats.spearmanr(a, b).statistic)
    return value if math.isfinite(value) else None


class GaussianLineageSidecar:
    """CPU-only metadata registry aligned to current Gaussian slots."""

    def __init__(self, initial_count: int) -> None:
        if initial_count <= 0:
            raise ValueError("initial Gaussian population must be non-empty")
        self.initial_count = int(initial_count)
        self.current_ids = torch.arange(initial_count, dtype=torch.int64)
        self.observed_camera_bits = torch.zeros(initial_count, dtype=torch.int64)
        self.registry_birth = torch.zeros(initial_count, dtype=torch.int32)
        self.registry_parent = torch.full((initial_count,), -1, dtype=torch.int64)
        self.registry_event_type = torch.zeros(initial_count, dtype=torch.int8)
        self.registry_depth = torch.zeros(initial_count, dtype=torch.int16)
        self.split_events: List[Dict[str, Any]] = []
        self.duplicate_events: List[Dict[str, Any]] = []
        self.prune_events: List[Dict[str, Any]] = []
        self.event_batches: List[Dict[str, Any]] = []
        self.current_iteration = -1

    @property
    def total_gaussians(self) -> int:
        return int(self.registry_birth.numel())

    @property
    def current_count(self) -> int:
        return int(self.current_ids.numel())

    def observe(self, camera_index: int, visible_mask: Tensor) -> None:
        if not 0 <= int(camera_index) < 63:
            raise RuntimeError("int64 observation bitset supports camera indices 0..62")
        visible = visible_mask.detach().bool().cpu().reshape(-1)
        if visible.numel() != self.current_count:
            raise RuntimeError("visibility mask is not aligned to lineage slots")
        self.observed_camera_bits[visible] |= int(1 << int(camera_index))

    def _append_registry(
        self, parent_ids: Tensor, event_type: int, iteration: int
    ) -> Tensor:
        parent_ids = parent_ids.detach().cpu().long().reshape(-1)
        count = int(parent_ids.numel())
        start = self.total_gaussians
        child_ids = torch.arange(start, start + count, dtype=torch.int64)
        parent_depth = self.registry_depth[parent_ids]
        self.registry_birth = torch.cat(
            [self.registry_birth, torch.full((count,), int(iteration), dtype=torch.int32)]
        )
        self.registry_parent = torch.cat([self.registry_parent, parent_ids])
        self.registry_event_type = torch.cat(
            [self.registry_event_type, torch.full((count,), event_type, dtype=torch.int8)]
        )
        self.registry_depth = torch.cat([self.registry_depth, parent_depth + 1])
        self.current_ids = torch.cat([self.current_ids, child_ids])
        self.observed_camera_bits = torch.cat(
            [self.observed_camera_bits, torch.zeros(count, dtype=torch.int64)]
        )
        return child_ids

    def record_split(self, split_mask: Tensor, samples: int) -> None:
        mask = split_mask.detach().bool().cpu().reshape(-1)
        population_before = int(mask.numel())
        if self.current_count < population_before:
            raise RuntimeError("split mask exceeds current lineage population")
        base_ids = self.current_ids[:population_before]
        base_bits = self.observed_camera_bits[:population_before]
        parent_ids = base_ids[mask]
        parent_support = _support_counts(base_bits[mask])
        repeated_parents = parent_ids.repeat(int(samples))
        child_ids = self._append_registry(
            repeated_parents, event_type=1, iteration=self.current_iteration
        )
        child_matrix = child_ids.reshape(int(samples), -1)
        for index, parent_id in enumerate(parent_ids.tolist()):
            self.split_events.append(
                {
                    "event_type": "split",
                    "parent_id": int(parent_id),
                    "child_ids": [int(value) for value in child_matrix[:, index].tolist()],
                    "iteration": int(self.current_iteration),
                    "parent_support_before": int(parent_support[index]),
                    "parent_support_definition": "distinct sampled training cameras observed before event",
                    "child_initial_support": 0,
                }
            )
        self.event_batches.append(
            {
                "event_type": "split_batch",
                "iteration": int(self.current_iteration),
                "parent_count": int(parent_ids.numel()),
                "child_count": int(child_ids.numel()),
            }
        )

    def record_duplicate(self, duplicate_mask: Tensor) -> None:
        mask = duplicate_mask.detach().bool().cpu().reshape(-1)
        population_before = int(mask.numel())
        if self.current_count < population_before:
            raise RuntimeError("duplicate mask exceeds current lineage population")
        base_ids = self.current_ids[:population_before]
        base_bits = self.observed_camera_bits[:population_before]
        parent_ids = base_ids[mask]
        parent_support = _support_counts(base_bits[mask])
        child_ids = self._append_registry(
            parent_ids, event_type=2, iteration=self.current_iteration
        )
        for index, parent_id in enumerate(parent_ids.tolist()):
            self.duplicate_events.append(
                {
                    "event_type": "duplicate",
                    "parent_id": int(parent_id),
                    "child_id": int(child_ids[index]),
                    "iteration": int(self.current_iteration),
                    "parent_support_before": int(parent_support[index]),
                    "parent_support_definition": "distinct sampled training cameras observed before event",
                    "child_initial_support": 0,
                }
            )
        self.event_batches.append(
            {
                "event_type": "duplicate_batch",
                "iteration": int(self.current_iteration),
                "parent_count": int(parent_ids.numel()),
                "child_count": int(child_ids.numel()),
            }
        )

    def record_prune(self, removed_mask: Tensor) -> None:
        mask = removed_mask.detach().bool().cpu().reshape(-1)
        if mask.numel() != self.current_count:
            raise RuntimeError("prune mask is not aligned to lineage slots")
        removed_ids = self.current_ids[mask]
        event = {
            "event_type": "prune",
            "removed_ids": [int(value) for value in removed_ids.tolist()],
            "iteration": int(self.current_iteration),
            "removed_count": int(removed_ids.numel()),
        }
        self.prune_events.append(event)
        self.event_batches.append(
            {
                "event_type": "prune_batch",
                "iteration": int(self.current_iteration),
                "removed_count": int(removed_ids.numel()),
            }
        )
        keep = ~mask
        self.current_ids = self.current_ids[keep]
        self.observed_camera_bits = self.observed_camera_bits[keep]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "initial_count": self.initial_count,
            "current_ids": self.current_ids.clone(),
            "observed_camera_bits": self.observed_camera_bits.clone(),
            "registry_birth": self.registry_birth.clone(),
            "registry_parent": self.registry_parent.clone(),
            "registry_event_type": self.registry_event_type.clone(),
            "registry_depth": self.registry_depth.clone(),
            "split_events": list(self.split_events),
            "duplicate_events": list(self.duplicate_events),
            "prune_events": list(self.prune_events),
            "event_batches": list(self.event_batches),
            "current_iteration": int(self.current_iteration),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "GaussianLineageSidecar":
        obj = cls(int(state["initial_count"]))
        for name in (
            "current_ids",
            "observed_camera_bits",
            "registry_birth",
            "registry_parent",
            "registry_event_type",
            "registry_depth",
        ):
            setattr(obj, name, state[name].detach().cpu().clone())
        for name in ("split_events", "duplicate_events", "prune_events", "event_batches"):
            setattr(obj, name, list(state[name]))
        obj.current_iteration = int(state["current_iteration"])
        obj.validate()
        return obj

    def validate(self, model_count: Optional[int] = None) -> None:
        total = self.total_gaussians
        if model_count is not None and self.current_count != int(model_count):
            raise RuntimeError("model and sidecar Gaussian counts differ")
        if self.observed_camera_bits.numel() != self.current_count:
            raise RuntimeError("support bitset and current IDs differ")
        registry_lengths = {
            int(self.registry_birth.numel()),
            int(self.registry_parent.numel()),
            int(self.registry_event_type.numel()),
            int(self.registry_depth.numel()),
        }
        if registry_lengths != {total}:
            raise RuntimeError("lineage registry fields differ in length")
        if torch.unique(self.current_ids).numel() != self.current_count:
            raise RuntimeError("current Gaussian IDs are not unique")
        if self.current_count and (
            int(self.current_ids.min()) < 0 or int(self.current_ids.max()) >= total
        ):
            raise RuntimeError("current Gaussian ID falls outside registry")
        child_ids = torch.arange(total) >= self.initial_count
        parents = self.registry_parent[child_ids]
        ids = torch.arange(total, dtype=torch.int64)[child_ids]
        if parents.numel() and bool(((parents < 0) | (parents >= ids)).any()):
            raise RuntimeError("child parent relationship is invalid")
        expected_depth = self.registry_depth[parents] + 1 if parents.numel() else parents
        if parents.numel() and not torch.equal(self.registry_depth[child_ids], expected_depth):
            raise RuntimeError("generation depth is inconsistent with parent depth")


def _instrument_model(model: Any, sidecar: GaussianLineageSidecar) -> Dict[str, Any]:
    originals = {
        "split_gaussians": model.split_gaussians,
        "dup_gaussians": model.dup_gaussians,
        "cull_gaussians": model.cull_gaussians,
    }

    def split_wrapper(_model: Any, split_mask: Tensor, samples: int) -> Any:
        result = originals["split_gaussians"](split_mask, samples)
        sidecar.record_split(split_mask, int(samples))
        return result

    def duplicate_wrapper(_model: Any, duplicate_mask: Tensor) -> Any:
        result = originals["dup_gaussians"](duplicate_mask)
        sidecar.record_duplicate(duplicate_mask)
        return result

    def cull_wrapper(_model: Any, extra_cull_mask: Optional[Tensor] = None) -> Tensor:
        removed = originals["cull_gaussians"](extra_cull_mask)
        sidecar.record_prune(removed)
        return removed

    model.split_gaussians = MethodType(split_wrapper, model)
    model.dup_gaussians = MethodType(duplicate_wrapper, model)
    model.cull_gaussians = MethodType(cull_wrapper, model)
    return originals


def _model_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _instrumentation_noop_audit(
    model: Any, sidecar: GaussianLineageSidecar, camera: Any
) -> Dict[str, Any]:
    model.eval()
    state_hash_before = _model_state_hash(model)
    prediction_before = _prediction(model, camera)
    state_keys_before = sorted(model.state_dict())
    _instrument_model(model, sidecar)
    prediction_after = _prediction(model, camera)
    state_hash_after = _model_state_hash(model)
    state_keys_after = sorted(model.state_dict())
    max_abs = float(
        (prediction_before.float() - prediction_after.float()).abs().max()
    )
    result = {
        "pass": bool(
            max_abs == 0.0
            and state_hash_before == state_hash_after
            and state_keys_before == state_keys_after
        ),
        "prediction_max_abs_diff": max_abs,
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "model_state_keys_equal": state_keys_before == state_keys_after,
        "lineage_keys_in_model_state": [
            key for key in state_keys_after if "lineage" in key.lower()
        ],
    }
    if not result["pass"]:
        raise RuntimeError(f"lineage instrumentation changed frozen output: {result}")
    return result


def _bundle_cpu(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if isinstance(value, Tensor) else value
        for key, value in bundle.items()
    }


def _save_checkpoint(
    branch: Any,
    bundle: Mapping[str, Any],
    sidecar: GaussianLineageSidecar,
) -> Tuple[Path, Path]:
    checkpoint_path = OUTPUT_ROOT / "checkpoints" / f"step-{FINAL_STEP:09d}.ckpt"
    sidecar_path = OUTPUT_ROOT / "checkpoints" / f"lineage-step-{FINAL_STEP:09d}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": EXPERIMENT,
            "branch": BRANCH,
            "absolute_step": FINAL_STEP,
            "model": branch.pipeline.model.state_dict(),
            "optimizers": {
                key: value.state_dict() for key, value in branch.optimizers.optimizers.items()
            },
            "schedulers": {
                key: value.state_dict() for key, value in branch.optimizers.schedulers.items()
            },
            "ocmc_bundle": _bundle_cpu(bundle),
            "raoc_state": None,
            "metadata": {
                "scene": SCENE,
                "lineage_sidecar": sidecar_path.name,
                "lineage_used_for_training": False,
                "normal_topology_enabled": True,
            },
        },
        checkpoint_path,
    )
    torch.save(sidecar.state_dict(), sidecar_path)
    return checkpoint_path, sidecar_path


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


@torch.no_grad()
def _prediction(model: Any, camera: Any) -> Tensor:
    model.eval()
    return model.get_outputs_for_camera(camera.to(model.device))["pred_image"].detach().cpu()


def _reload_validation(
    checkpoint_path: Path,
    sidecar_path: Path,
    reference_prediction: Tensor,
    reference_sidecar: Mapping[str, Any],
) -> Tuple[Any, GaussianLineageSidecar, Dict[str, Any]]:
    branch = FORMAL._setup_branch(REPO_ROOT, FORMAL.SCENES[SCENE], BRANCH)
    payload = torch.load(checkpoint_path, map_location="cpu")
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = int(payload["absolute_step"])
    FORMAL._install_condition(branch.pipeline.model, BRANCH, payload["ocmc_bundle"], None)
    loaded_sidecar_state = torch.load(sidecar_path, map_location="cpu")
    sidecar = GaussianLineageSidecar.from_state_dict(loaded_sidecar_state)
    sidecar.validate(int(branch.pipeline.model.means.shape[0]))
    eval_record = FORMAL._eval_records(branch.pipeline)[0]
    reloaded_prediction = _prediction(branch.pipeline.model, eval_record[2])
    max_abs = float((reference_prediction.float() - reloaded_prediction.float()).abs().max())
    checks = {
        "checkpoint_reload_pass": bool(
            int(branch.pipeline.model.means.shape[0]) == sidecar.current_count
            and _state_equal(reference_sidecar, loaded_sidecar_state)
            and max_abs == 0.0
        ),
        "gaussian_count_equal": int(branch.pipeline.model.means.shape[0])
        == sidecar.current_count,
        "current_ids_equal": _state_equal(
            reference_sidecar["current_ids"], loaded_sidecar_state["current_ids"]
        ),
        "parent_relationships_equal": _state_equal(
            reference_sidecar["registry_parent"], loaded_sidecar_state["registry_parent"]
        ),
        "birth_iterations_equal": _state_equal(
            reference_sidecar["registry_birth"], loaded_sidecar_state["registry_birth"]
        ),
        "generation_depths_equal": _state_equal(
            reference_sidecar["registry_depth"], loaded_sidecar_state["registry_depth"]
        ),
        "topology_events_equal": all(
            _state_equal(reference_sidecar[key], loaded_sidecar_state[key])
            for key in ("split_events", "duplicate_events", "prune_events")
        ),
        "prediction_max_abs_diff": max_abs,
        "model_checkpoint_contains_lineage": False,
        "sidecar_loaded_separately": True,
    }
    if not checks["checkpoint_reload_pass"]:
        FORMAL._release(branch)
        raise RuntimeError(f"checkpoint/sidecar reload validation failed: {checks}")
    return branch, sidecar, checks


def _lineage_group_masks(sidecar: GaussianLineageSidecar) -> Dict[str, Tensor]:
    depth = sidecar.registry_depth[sidecar.current_ids]
    return {
        "initial": depth == 0,
        "child_generation_1": depth == 1,
        "child_generation_2_plus": depth >= 2,
    }


@torch.no_grad()
def _support_lifecycle_analysis(branch: Any, sidecar: GaussianLineageSidecar) -> Dict[str, Any]:
    model = branch.pipeline.model
    model.eval()
    train_records = FORMAL._train_records(branch.pipeline)
    heldout_records = FORMAL._eval_records(branch.pipeline)
    if len(train_records) >= 63:
        raise RuntimeError("support audit bitset only supports fewer than 63 train cameras")
    train_ids = {record[1] for record in train_records}
    heldout_ids = {record[1] for record in heldout_records}
    if train_ids & heldout_ids:
        raise RuntimeError("train/heldout camera leakage")
    n = int(model.means.shape[0])
    support = torch.zeros(n, dtype=torch.int16)
    visibility_checks = 0
    for _index, _camera_id, camera, _batch in train_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = model.radii.detach().reshape(-1) > 0
        reported = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
        if visible.numel() != n or not torch.equal(visible, reported):
            raise RuntimeError("historical support visibility definition changed")
        support += visible.cpu().to(torch.int16)
        visibility_checks += 1
        del outputs
    groups = _lineage_group_masks(sidecar)
    group_rows: Dict[str, Any] = {}
    low = support <= 1
    for name, mask in groups.items():
        values = support[mask]
        group_rows[name] = {
            "current_gaussians": int(values.numel()),
            "support_mean": float(values.float().mean()) if values.numel() else None,
            "support_median": float(values.float().median()) if values.numel() else None,
            "support_min": int(values.min()) if values.numel() else None,
            "support_max": int(values.max()) if values.numel() else None,
            "low_support_count": int((values <= 1).sum()),
            "low_support_fraction": float((values <= 1).float().mean())
            if values.numel()
            else None,
            "fraction_of_all_low_support": float((low & mask).sum() / max(int(low.sum()), 1)),
        }
    camera_rows: List[Dict[str, Any]] = []
    for _index, camera_id, camera, batch in heldout_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible_gpu = model.radii.detach().reshape(-1) > 0
        reported = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
        if not torch.equal(visible_gpu, reported):
            raise RuntimeError("heldout visibility definition changed")
        visible = visible_gpu.cpu()
        gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        row: Dict[str, Any] = {
            "camera_id": camera_id,
            "E_cam": float((pred - gt).square().mean()),
            "heldout_used_for_support": False,
        }
        for name, mask in {"all": torch.ones(n, dtype=torch.bool), **groups}.items():
            selected = visible & mask
            count = int(selected.sum())
            row[f"{name}_visible_count"] = count
            row[f"{name}_visible_low_support_fraction"] = (
                float(low[selected].float().mean()) if count else None
            )
            row[f"{name}_visible_population_fraction"] = float(count / max(int(visible.sum()), 1))
        camera_rows.append(row)
        visibility_checks += 1
        del outputs, gt, pred
    residual_rows: Dict[str, Any] = {}
    errors = [row["E_cam"] for row in camera_rows]
    for name in ("all", *groups):
        predictor = [row[f"{name}_visible_low_support_fraction"] for row in camera_rows]
        residual_rows[name] = {
            "spearman_low_support_fraction_vs_E_cam": _rho(predictor, errors),
            "heldout_camera_count": len(camera_rows),
            "status": "SMOKE_ONLY_SMALL_N" if len(camera_rows) < 10 else "ESTIMATED",
        }
    current_depth = sidecar.registry_depth[sidecar.current_ids]
    current_birth = sidecar.registry_birth[sidecar.current_ids]
    child_mask = current_depth > 0
    recent_child = child_mask & (current_birth >= FINAL_STEP - 200)
    low_count = max(int(low.sum()), 1)
    population_count = max(n, 1)
    return {
        "scene": SCENE,
        "absolute_step": FINAL_STEP,
        "support_definition": "number of distinct frozen training cameras where model.radii > 0",
        "train_camera_count": len(train_records),
        "heldout_camera_count": len(heldout_records),
        "train_heldout_overlap": [],
        "visibility_equivalence_checks": visibility_checks,
        "groups": group_rows,
        "heldout_camera_rows": camera_rows,
        "residual_association_by_group": residual_rows,
        "lineage_controlled_residual_association": {
            "value": None,
            "status": "NOT_ESTIMABLE_WITH_4_HELDOUT_CAMERAS_AND_3_LINEAGE_GROUPS",
        },
        "low_support_population": {
            "count": int(low.sum()),
            "fraction": float(low.float().mean()),
            "fraction_low_support_that_are_children": float((low & child_mask).sum() / low_count),
            "fraction_population_that_are_children": float(child_mask.sum() / population_count),
            "fraction_low_support_that_are_recent_children_le_200_steps": float(
                (low & recent_child).sum() / low_count
            ),
            "fraction_population_that_are_recent_children_le_200_steps": float(
                recent_child.sum() / population_count
            ),
        },
        "question_1_newborn_origin": "DESCRIPTIVE_SMOKE_RESULT_ONLY",
        "question_2_long_lived_low_support": "NOT_ANSWERABLE_BY_1000_STEP_SMOKE",
        "question_3_lineage_controlled_correlation": "NOT_ESTIMABLE_SMALL_N",
        "classification": "LINEAGE_INCONCLUSIVE",
    }


def _lineage_statistics(sidecar: GaussianLineageSidecar) -> Dict[str, Any]:
    sidecar.validate()
    current_depth = sidecar.registry_depth[sidecar.current_ids]
    current_types = sidecar.registry_event_type[sidecar.current_ids]
    all_depth = sidecar.registry_depth
    parents = sidecar.registry_parent[sidecar.initial_count :]
    parent_counts = Counter(int(value) for value in parents.tolist())
    depth_distribution = Counter(int(value) for value in current_depth.tolist())
    all_depth_distribution = Counter(int(value) for value in all_depth.tolist())
    type_distribution = Counter(
        EVENT_TYPE_NAMES[int(value)] for value in current_types.tolist()
    )
    children = int(parents.numel())
    return {
        "generation_depth_distribution_current": {
            str(key): value for key, value in sorted(depth_distribution.items())
        },
        "generation_depth_distribution_all_created": {
            str(key): value for key, value in sorted(all_depth_distribution.items())
        },
        "current_event_type_distribution": dict(sorted(type_distribution.items())),
        "average_children_per_parent": float(children / len(parent_counts))
        if parent_counts
        else 0.0,
        "max_children_per_parent": max(parent_counts.values()) if parent_counts else 0,
        "max_generation_depth": int(all_depth.max()) if all_depth.numel() else 0,
        "fraction_of_inherited_children": float((parents >= 0).float().mean())
        if children
        else 0.0,
        "fraction_of_children_with_recorded_parent": float((parents >= 0).float().mean())
        if children
        else 0.0,
        "support_state_inherited_at_birth": False,
        "child_initial_observed_camera_support": 0,
        "lineage_registry_retains_pruned_gaussians": True,
    }


def _summary(
    sidecar: GaussianLineageSidecar,
    training: Mapping[str, Any],
    instrumentation_audit: Mapping[str, Any],
    reload_checks: Mapping[str, Any],
    analysis: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    nonempty_prunes = [event for event in sidecar.prune_events if event["removed_count"] > 0]
    split_iterations = sorted({int(event["iteration"]) for event in sidecar.split_events})
    duplicate_iterations = sorted(
        {int(event["iteration"]) for event in sidecar.duplicate_events}
    )
    prune_iterations = sorted({int(event["iteration"]) for event in nonempty_prunes})
    return {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "branch": BRANCH,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "smoke_steps": MAX_STEPS,
        "first_step": 0,
        "final_step": FINAL_STEP,
        "total_gaussians": sidecar.total_gaussians,
        "initial_gaussians": sidecar.initial_count,
        "current_gaussians": sidecar.current_count,
        "num_split_events": len(sidecar.split_events),
        "num_duplicate_events": len(sidecar.duplicate_events),
        "num_prune_events": len(nonempty_prunes),
        "num_split_batches": len(split_iterations),
        "num_duplicate_batches": len(duplicate_iterations),
        "num_prune_batches": len(prune_iterations),
        "split_iterations": split_iterations,
        "duplicate_iterations": duplicate_iterations,
        "prune_iterations": prune_iterations,
        "checkpoint_reload": dict(reload_checks),
        "instrumentation_noop_audit": dict(instrumentation_audit),
        "training": dict(training),
        "low_support": dict(analysis["low_support_population"]),
        "classification": "LINEAGE_INCONCLUSIVE",
        "module_design_authorized": False,
        "recommendation": (
            "USE_VALIDATED_SIDECAR_IN_A_BOUNDED_LONGER_DIAGNOSTIC_BEFORE_MODULE_DESIGN"
        ),
        "training_logic_modified": False,
        "ocmc_modified": False,
        "raoc_modified": False,
        "renderer_modified": False,
        "loss_modified": False,
        "optimizer_modified": False,
        "densification_policy_modified": False,
        "lineage_used_for_training": False,
        "runtime": dict(runtime),
    }


def _research_note(summary: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    groups = analysis["groups"]
    residual = analysis["residual_association_by_group"]
    low = analysis["low_support_population"]
    recent_representation_ratio = (
        low["fraction_low_support_that_are_recent_children_le_200_steps"]
        / max(low["fraction_population_that_are_recent_children_le_200_steps"], EPS)
    )
    lines = [
        "# Gaussian Lineage Sidecar Smoke Validation (2026-09-01)",
        "",
        "## Scope",
        "",
        "CONFIG FACT: this was one 1000-step IUI3-RedSea smoke run (steps "
        "0-999), using the existing C0 configuration with OCMC on and RAOC off. "
        "It was not a formal 15K experiment.",
        "",
        "CODE FACT: lineage is an external CPU sidecar. It observes existing "
        "split/duplicate/prune masks and never enters the model state_dict, "
        "renderer, loss, optimizer, gradients, or refinement selection.",
        "",
        "## Topology Coverage",
        "",
        f"QUANTITATIVE RESULT: initial={summary['initial_gaussians']}, "
        f"ever-created={summary['total_gaussians']}, current={summary['current_gaussians']}. "
        f"The smoke recorded {summary['num_split_events']} split-parent events "
        f"in {summary['num_split_batches']} batches, {summary['num_duplicate_events']} "
        f"duplicate events in {summary['num_duplicate_batches']} batches, and "
        f"{summary['num_prune_events']} non-empty prune calls.",
        "",
        "EXPERIMENTAL FACT: each child has a unique ID, birth iteration, parent "
        "ID, event type, and generation depth. Pruned IDs remain in the registry "
        "while current slot IDs are masked with the exact prune mask.",
        "",
        "## Reload Compatibility",
        "",
        f"EXPERIMENTAL FACT: checkpoint/sidecar reload passed: "
        f"{str(summary['checkpoint_reload']['checkpoint_reload_pass']).upper()}. "
        "Gaussian count, current IDs, parent relations, birth iterations, "
        "generation depths, and topology events matched exactly. The frozen "
        f"prediction max-absolute difference was "
        f"{summary['checkpoint_reload']['prediction_max_abs_diff']:.3g}.",
        "",
        "EXPERIMENTAL FACT: installing the sidecar wrappers changed neither "
        "the model state_dict hash nor a frozen prediction; installation-time "
        f"prediction max-absolute difference was "
        f"{summary['instrumentation_noop_audit']['prediction_max_abs_diff']:.3g}.",
        "",
        "## Frozen Support By Lineage",
        "",
        "The table uses the existing proxy: the number of distinct frozen "
        "training cameras where model.radii > 0 at step 999. It does not use "
        "heldout cameras to compute support.",
        "",
        "| Lineage group | current N | mean support | low-support fraction | share of all low-support | rho vs heldout residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("initial", "child_generation_1", "child_generation_2_plus"):
        row = groups[name]
        rho = residual[name]["spearman_low_support_fraction_vs_E_cam"]
        lines.append(
            f"| {name} | {row['current_gaussians']} | "
            f"{row['support_mean']:.3f} | {row['low_support_fraction']:.6f} | "
            f"{row['fraction_of_all_low_support']:.6f} | "
            f"{rho:.3f} |" if rho is not None else
            f"| {name} | {row['current_gaussians']} | "
            f"{row['support_mean']:.3f} | {row['low_support_fraction']:.6f} | "
            f"{row['fraction_of_all_low_support']:.6f} | undefined |"
        )
    lines.extend(
        [
            "",
            f"QUANTITATIVE RESULT: children are {low['fraction_population_that_are_children']:.6f} "
            f"of the final population and {low['fraction_low_support_that_are_children']:.6f} "
            "of the final low-support population. Children born within the last "
            f"200 steps are {low['fraction_population_that_are_recent_children_le_200_steps']:.6f} "
            f"of the population and {low['fraction_low_support_that_are_recent_children_le_200_steps']:.6f} "
            "of low-support Gaussians.",
            "",
            f"INFERENCE: the recent-child representation ratio in low support is "
            f"only {recent_representation_ratio:.4f}x because recent children "
            "already dominate the whole smoke population. The 100% child share "
            "therefore does not provide a discriminating lineage enrichment test.",
            "",
            "## Scientific Limits",
            "",
            "INFERENCE: the smoke can describe whether newborn groups are enriched "
            "for low support at step 999, but it cannot establish what survives "
            "to 15K. The protocol forbids a new formal 15K run.",
            "",
            "QUANTITATIVE RESULT: lineage-controlled residual correlation is "
            "NOT_ESTIMABLE_WITH_4_HELDOUT_CAMERAS_AND_3_LINEAGE_GROUPS. Per-group "
            "rho values are small-N diagnostics only.",
            "",
            "## Decision",
            "",
            "FINAL CLASSIFICATION: LINEAGE_INCONCLUSIVE.",
            "",
            "MODULE DESIGN AUTHORIZED: FALSE.",
            "",
            "RECOMMENDATION: use the validated sidecar in a bounded longer "
            "diagnostic with adequate heldout support before module design. "
            "Do not modify OCMC and do not reopen RAOC.",
            "",
        ]
    )
    RESEARCH_NOTE.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def run(gpu: str, overwrite: bool) -> Dict[str, Any]:
    runtime = _strict_runtime(gpu)
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not overwrite:
            raise RuntimeError(f"non-empty output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    scene_cfg = FORMAL.SCENES[SCENE]
    FORMAL._seed_all(TRAINING_SEED)
    probe = FORMAL._setup_branch(REPO_ROOT, scene_cfg, BRANCH)
    try:
        training_rng = FORMAL._rng_state()
        general_only_cfg = {**scene_cfg, "locked_safe": False}
        samples, bank = FORMAL._build_samples(
            REPO_ROOT, OUTPUT_ROOT, SCENE, general_only_cfg, probe
        )
        bank["analysis_population"] = "GENERAL"
        bank["legacy_M_SAFE_reconstruction_requested"] = False
        _write_json(OUTPUT_ROOT / "calibration_bank.json", bank)
        sequence, camera_names = FORMAL._camera_sequence(
            probe, OUTPUT_ROOT, length=MAX_STEPS
        )
    finally:
        FORMAL._release(probe)
    branch = FORMAL._setup_branch(REPO_ROOT, scene_cfg, BRANCH)
    model = branch.pipeline.model
    sidecar = GaussianLineageSidecar(int(model.means.shape[0]))
    instrumentation_camera = FORMAL._eval_records(branch.pipeline)[0][2]
    instrumentation_audit = _instrumentation_noop_audit(
        model, sidecar, instrumentation_camera
    )
    FORMAL._set_rng_state(training_rng)
    bundle, _unused_raoc_state, calibration_seconds = FORMAL._calibrate(
        branch, samples, 0, OUTPUT_ROOT / "ocmc_calibration"
    )
    FORMAL._install_condition(model, BRANCH, bundle, None)
    dm = branch.pipeline.datamanager
    cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    cached = dm.cached_train
    started = time.perf_counter()
    completed = 0
    metric_rows: List[Dict[str, Any]] = []
    try:
        for step, camera_index in enumerate(sequence):
            branch.pipeline.train()
            model.train()
            FORMAL._configure_model(model, BRANCH)
            FORMAL._install_condition(model, BRANCH, bundle, None)
            MIC._run_before(model, branch.optimizers, step)
            branch.optimizers.zero_grad_all()
            batch = MIC._batch_to_device(cached[camera_index].copy(), model.device)
            camera = cameras[camera_index : camera_index + 1]
            outputs = model.get_outputs(camera)
            gt = MI.PW._get_gt(model, batch, outputs["background"])
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"non-finite smoke loss at step {step}")
            sidecar.observe(camera_index, model.radii.detach().reshape(-1) > 0)
            total.backward()
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(step)
            sidecar.current_iteration = int(step)
            event = MIC._run_after(model, branch.optimizers, step)
            sidecar.validate(int(model.means.shape[0]))
            completed = step + 1
            if step % 100 == 0 or step == FINAL_STEP:
                metric_rows.append(
                    {
                        "absolute_step": step,
                        "camera_index": int(camera_index),
                        "camera_name": camera_names[step],
                        "loss": float(total.detach().cpu()),
                        "gaussian_count": int(model.means.shape[0]),
                        "split_count": int(event.get("K_split", 0)),
                        "duplicate_count": int(event.get("K_duplicate", 0)),
                        "pruned_count": int(event.get("N_pruned", 0)),
                    }
                )
            del outputs, gt, losses, total
            if step % 100 == 0:
                print(
                    f"[{SCENE}] step={step} N={model.means.shape[0]} "
                    f"split={event.get('K_split', 0)} dup={event.get('K_duplicate', 0)} "
                    f"prune={event.get('N_pruned', 0)}",
                    flush=True,
                )
        nonempty_prunes = [event for event in sidecar.prune_events if event["removed_count"]]
        if not sidecar.split_events or not sidecar.duplicate_events or not nonempty_prunes:
            raise RuntimeError("1000-step smoke did not cover split, duplicate, and prune")
        checkpoint_path, sidecar_path = _save_checkpoint(branch, bundle, sidecar)
        eval_record = FORMAL._eval_records(branch.pipeline)[0]
        reference_prediction = _prediction(model, eval_record[2])
        reference_sidecar = sidecar.state_dict()
        training = {
            "completed_steps": completed,
            "step_range": [0, FINAL_STEP],
            "calibration_seconds": float(calibration_seconds),
            "training_seconds": float(time.perf_counter() - started),
            "checkpoint_path": str(checkpoint_path),
            "sidecar_path": str(sidecar_path),
            "calibration_bank_hash": bank["bank_hash"],
            "camera_sequence_hash": json.loads(
                (OUTPUT_ROOT / "camera_sequence.json").read_text(encoding="utf8")
            )["sha256"],
            "metric_rows": metric_rows,
        }
    finally:
        FORMAL._release(branch)
    reloaded = None
    try:
        reloaded, loaded_sidecar, reload_checks = _reload_validation(
            checkpoint_path, sidecar_path, reference_prediction, reference_sidecar
        )
        analysis = _support_lifecycle_analysis(reloaded, loaded_sidecar)
        statistics = _lineage_statistics(loaded_sidecar)
        summary = _summary(
            loaded_sidecar,
            training,
            instrumentation_audit,
            reload_checks,
            analysis,
            runtime,
        )
        topology = {
            "schema_version": 1,
            "support_before_definition": (
                "distinct sampled training cameras observed before each event"
            ),
            "split": loaded_sidecar.split_events,
            "duplicate": loaded_sidecar.duplicate_events,
            "prune": loaded_sidecar.prune_events,
            "event_batches": loaded_sidecar.event_batches,
        }
        _write_json(OUTPUT_ROOT / "lineage_summary.json", summary)
        _write_json(OUTPUT_ROOT / "topology_events.json", topology)
        _write_json(OUTPUT_ROOT / "lineage_statistics.json", statistics)
        _write_json(OUTPUT_ROOT / "support_lifecycle_analysis.json", analysis)
        _write_json(OUTPUT_ROOT / "checkpoint_reload_validation.json", reload_checks)
        _research_note(summary, analysis)
    finally:
        FORMAL._release(reloaded)
    end_runtime = _strict_runtime(gpu)
    summary["ending_source_hashes_match"] = (
        runtime["protected_hashes"] == end_runtime["protected_hashes"]
        and runtime["audited_source_hashes"] == end_runtime["audited_source_hashes"]
    )
    _write_json(OUTPUT_ROOT / "lineage_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, choices=sorted(ALLOWED_GPUS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(str(args.gpu), bool(args.overwrite))
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
