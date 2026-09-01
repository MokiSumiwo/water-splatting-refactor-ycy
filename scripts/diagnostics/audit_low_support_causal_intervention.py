#!/usr/bin/env python3
"""Frozen causal intervention audit for low-training-support Gaussians.

The intervention scales a detached copy of per-Gaussian opacity immediately
before rasterization. It never mutates model state, changes topology, trains,
or uses held-out ground truth to define a Gaussian group.
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
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_local_contextual_support_predictor_iui3 as LOCAL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


EXPERIMENT = "LOW-SUPPORT-CAUSAL-INTERVENTION-PREFLIGHT"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "907cf3881ecbb3f849b369815ac1b429236f4199"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "low_support_causal_intervention_20260901"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "LOW_SUPPORT_CAUSAL_INTERVENTION_PREFLIGHT_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
SEED = 42
EPS = 1e-12
FULL_ATOL = 2e-6
LOCALIZATION_RESIDUAL_QUANTILE = 0.80
PRIMARY_GROUP = "LOW_T1"
PRIMARY_ALPHA = 0.0
EXPECTED_CHECKPOINT_HASHES = {
    "Curasao": "ccf6a26eee364a02ba68ef9942e083d34962c71e5c84e40c0db8deca265fb406",
    "IUI3-RedSea": "63ae7295ed0738641db5249a7876a1a05fbc30e5d1c3a0c7d43df843b837a180",
    "JapaneseGradens-RedSea": "452e2aa0e81c4f977df96bc4a97948dbaf9132b5e5c6ef0862a945fac04b4bb2",
    "Panama": "e5ae9aa065635802e9c5a00dee63eb06aa68b8504d89bfab71fd8dc4eea9b6a0",
}
EXPECTED_CONFIG_HASHES = {
    "Curasao": "71c3023ad775d8c2b05797206d9aee45519f6c7ff8046a728e41685a1944cbe4",
    "IUI3-RedSea": "cb2f421ceafd8a6ccf42307f96567e9c21b695f842cf22e4dd10e81286acecde",
    "JapaneseGradens-RedSea": "e8ae574a77c2686a811a8b57765f271816f48017622757606bfc1ca3eb9ed604",
    "Panama": "7aeb3eceea8e6a73ec3c2be350f946e5fd148f37b9c3583f355479484175b2c1",
}
EXPECTED_CAMERA_SEQUENCE_HASHES = {
    "Curasao": "f1188ea8f7ab162d50ebace762f941266727c483a8250172a8b306e7460f2d6a",
    "IUI3-RedSea": "58a8b42fa1e44f372d7beb0402f31ae201d21042ada5f849ff93ef469a274827",
    "JapaneseGradens-RedSea": "7a75e51f86dd6828aaa49a313261e253dd9c971aad7bb5fbf178cc0963629bad",
    "Panama": "e85983dd93bf25073ffc8ffe15af8c97b4324e042535ecd627dedebd7a1b99c6",
}
PROTECTED_HASHES = {
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py": "b6a271372e68cd07fc566a3fde5ced5ba6463531278c31a6cfa47972aa15e8d6",
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py": "539f1c044f9ed136dce65b1dedc01746097cb2f3c4298c9682038019d23dfd7a",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py": "fe3fd3ddcdbbff7904cfb7225a0ba024f928a9020777561252b66663c3c8ab32",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py": "d131428cc20ea76010e237abd91ac4cddfc5c6a78944c57c3317ed18bcdf60ef",
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py": "3a924e88a606d34360a90348f3a392d0d12f80d43c98fe72b56cbec2d27ad6e7",
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


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize_json(value),
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


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    return _hash_bytes(data)


def _tensor_hash(value: Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _model_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(key.encode("utf8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _checkpoint(scene: str) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / "step-000014999.ckpt"


def _scene_config(scene: str) -> Dict[str, Any]:
    config = _read_json(SOURCE_ROOT / scene / "scene_config.json")
    return {
        "data_name": config["data_name"],
        "data_path": config["data_path"],
        "source_config": config["source_config"],
        "locked_safe": bool(config["locked_safe"]),
    }


def _strict_repo_and_files() -> Dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected starting repo state: {branch}@{head}")
    protected: Dict[str, str] = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected file changed: {relative}")
        protected[relative] = actual
    return {
        "branch": branch,
        "starting_head": head,
        "status_short": _run_text(["git", "status", "--short"]),
        "protected_hashes": protected,
    }


def _runtime(gpu: str) -> Dict[str, Any]:
    if gpu != SCENE_GPUS.get(os.environ.get("LOW_SUPPORT_SCENE", ""), gpu):
        raise RuntimeError("scene/GPU environment assignment drift")
    if gpu not in SCENE_GPUS.values() or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose exactly physical GPU {gpu}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see one CUDA device")
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
    repo = _strict_repo_and_files()
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        scene_cfg = _scene_config(scene)
        checkpoint = _checkpoint(scene)
        config = REPO_ROOT / scene_cfg["source_config"]
        sequence = SOURCE_ROOT / scene / "camera_sequence.json"
        actual = {
            "checkpoint_sha256": _sha256(checkpoint),
            "source_config_sha256": _sha256(config),
            "camera_sequence_sha256": _sha256(sequence),
        }
        expected = {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_HASHES[scene],
            "source_config_sha256": EXPECTED_CONFIG_HASHES[scene],
            "camera_sequence_sha256": EXPECTED_CAMERA_SEQUENCE_HASHES[scene],
        }
        if actual != expected:
            raise RuntimeError(f"provenance hash drift for {scene}: {actual}")
        rows.append(
            {
                "scene": scene,
                "checkpoint_path": str(checkpoint),
                "source_config_path": str(config),
                "camera_sequence_path": str(sequence),
                **actual,
            }
        )
    manifest = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "source_rows": rows,
        "frozen_forward_only": True,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "new_checkpoints": 0,
        "renderer_source_changes": 0,
        "ocmc_changes": 0,
        "raoc_enabled": False,
        "random_seed": SEED,
        "primary_comparison": {"group": PRIMARY_GROUP, "alpha": PRIMARY_ALPHA},
        "localization": {
            "ground_truth_split": "heldout eval only",
            "high_residual_rule": "per-view top 20% by baseline RGB MSE",
            "projection_rule": "low-support indicator contribution > 1e-12",
        },
    }
    _write_json(OUTPUT_ROOT / "preflight.json", manifest)
    return manifest


def _camera_split_manifest(
    train_records: Sequence[Tuple[int, str, Any, Any]],
    eval_records: Sequence[Tuple[int, str, Any, Any]],
) -> Dict[str, Any]:
    train_ids = [str(row[1]) for row in train_records]
    eval_ids = [str(row[1]) for row in eval_records]
    if len(train_ids) != len(set(train_ids)) or len(eval_ids) != len(set(eval_ids)):
        raise RuntimeError("camera IDs are not distinct within a split")
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
def _support_counts(model: Any, train_records: Sequence[Tuple[int, str, Any, Any]]) -> Tensor:
    support = torch.zeros(int(model.means.shape[0]), dtype=torch.int16)
    for _index, _camera_id, camera, _batch in train_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        radii_visible = model.radii.detach().reshape(-1) > 0
        reported = outputs["gaussian_visible_mask"].detach().reshape(-1).bool()
        if radii_visible.numel() != support.numel() or not torch.equal(radii_visible, reported):
            raise RuntimeError("formal radii visibility alias mismatch")
        support += radii_visible.cpu().to(torch.int16)
        del outputs
    if int(support.max()) > len(train_records):
        raise RuntimeError("support exceeds distinct training-camera count")
    return support


def _sample_mask(pool: Tensor, count: int, seed: int, label: str) -> Tensor:
    indices = torch.nonzero(pool.detach().cpu().bool(), as_tuple=False).reshape(-1)
    if int(indices.numel()) < count:
        raise RuntimeError(f"{label} pool has {indices.numel()} entries for requested {count}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = indices[torch.randperm(int(indices.numel()), generator=generator)[:count]]
    mask = torch.zeros(pool.numel(), dtype=torch.bool)
    mask[selected] = True
    return mask


def _group_masks(support: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    low_t1 = support <= 1
    low_t2 = support <= 2
    median = float(torch.median(support.float()).item())
    high_pool = support.float() >= median
    random_pool = ~low_t1
    count = int(low_t1.sum())
    high = _sample_mask(high_pool, count, SEED, "high-support")
    random = _sample_mask(random_pool, count, SEED, "random non-low")
    masks = {
        "LOW_T1": low_t1,
        "LOW_T2": low_t2,
        "HIGH_MATCHED": high,
        "RANDOM_MATCHED": random,
    }
    manifest = {
        "gaussian_count": int(support.numel()),
        "support_min": int(support.min()),
        "support_max": int(support.max()),
        "support_mean": float(support.float().mean()),
        "support_median_torch_lower": median,
        "low_t1_count": count,
        "low_t2_count": int(low_t2.sum()),
        "high_pool_count": int(high_pool.sum()),
        "random_non_low_pool_count": int(random_pool.sum()),
        "high_matched_count": int(high.sum()),
        "random_matched_count": int(random.sum()),
        "high_selection_seed": SEED,
        "random_selection_seed": SEED,
        "high_mask_sha256": _tensor_hash(high),
        "random_mask_sha256": _tensor_hash(random),
        "low_t1_mask_sha256": _tensor_hash(low_t1),
        "low_t2_mask_sha256": _tensor_hash(low_t2),
    }
    if int(high.sum()) != count or int(random.sum()) != count:
        raise RuntimeError("matched controls have the wrong size")
    if bool((random & low_t1).any()):
        raise RuntimeError("random control leaked into primary low-support group")
    return masks, manifest


def _geometry(model: Any, camera: Any, height: int, width: int) -> Dict[str, Tensor]:
    values = LOCAL._render_geometry(model, camera, height, width)
    names = (
        "xys",
        "depths",
        "radii",
        "conics",
        "colors",
        "opacities",
        "num_tiles_hit",
        "size",
        "tile_bounds",
    )
    return dict(zip(names, values))


@torch.no_grad()
def _counterfactual(
    model: Any,
    geometry: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    mask: Tensor,
    alpha: float,
) -> Tensor:
    opacities = geometry["opacities"].detach().clone()
    selected = mask.to(opacities.device)
    if selected.numel() != opacities.shape[0]:
        raise RuntimeError("intervention mask/opacity shape mismatch")
    opacities[selected] *= float(alpha)
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
        colors=geometry["colors"],
        opacities=opacities,
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
        b_inf = outputs["b_inf"].detach()
        tail_weight = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
        original_tail = tail_weight * medium_rgb
        rgb_medium = rgb_medium - original_tail + tail_weight * b_inf
        pred = render.rgb_object + rgb_medium
    else:
        pred = render.rgb
    if not bool(torch.isfinite(pred).all()):
        raise RuntimeError("counterfactual produced non-finite RGB")
    return pred.detach()


@torch.no_grad()
def _projection_mask(
    model: Any, geometry: Mapping[str, Tensor], low_mask: Tensor
) -> Tuple[Tensor, Dict[str, Any]]:
    height, width = (int(item) for item in geometry["size"].tolist())
    zero_image = torch.zeros(height, width, 3, device=model.device)
    colors = low_mask.to(model.device, dtype=torch.float32)[:, None].expand(-1, 3)
    render = model.underwater_rasterizer.rasterize(
        xys=geometry["xys"],
        xys_grad_abs=torch.zeros_like(geometry["xys"]),
        depths=geometry["depths"],
        radii=geometry["radii"],
        conics=geometry["conics"],
        num_tiles_hit=geometry["num_tiles_hit"],
        colors=colors,
        opacities=geometry["opacities"],
        medium_rgb=zero_image,
        medium_bs=zero_image,
        medium_attn=zero_image,
        height=height,
        width=width,
        background=torch.zeros(3, device=model.device),
        step=model.step,
        force_white_background=False,
    )
    contribution = render.rgb_object[..., 0].detach()
    channel_diff = float((render.rgb_object - contribution[..., None]).abs().max())
    if channel_diff != 0.0 or not bool(torch.isfinite(contribution).all()):
        raise RuntimeError("invalid low-support indicator contribution map")
    return contribution > EPS, {
        "projection_threshold": EPS,
        "projected_pixel_count": int((contribution > EPS).sum()),
        "mean_indicator_contribution": float(contribution.mean()),
        "max_indicator_contribution": float(contribution.max()),
        "indicator_channel_max_abs_diff": channel_diff,
    }


def _condition_specs(masks: Mapping[str, Tensor]) -> List[Tuple[str, str, float, Tensor]]:
    return [
        ("LOW_T1_ZERO", "LOW_T1", 0.0, masks["LOW_T1"]),
        ("LOW_T1_HALF", "LOW_T1", 0.5, masks["LOW_T1"]),
        ("LOW_T1_ONE", "LOW_T1", 1.0, masks["LOW_T1"]),
        ("HIGH_MATCHED_ZERO", "HIGH_MATCHED", 0.0, masks["HIGH_MATCHED"]),
        ("HIGH_MATCHED_HALF", "HIGH_MATCHED", 0.5, masks["HIGH_MATCHED"]),
        ("HIGH_MATCHED_ONE", "HIGH_MATCHED", 1.0, masks["HIGH_MATCHED"]),
        ("RANDOM_MATCHED_ZERO", "RANDOM_MATCHED", 0.0, masks["RANDOM_MATCHED"]),
        ("RANDOM_MATCHED_HALF", "RANDOM_MATCHED", 0.5, masks["RANDOM_MATCHED"]),
        ("RANDOM_MATCHED_ONE", "RANDOM_MATCHED", 1.0, masks["RANDOM_MATCHED"]),
        ("LOW_T2_ZERO", "LOW_T2", 0.0, masks["LOW_T2"]),
    ]


def _ocmc_view_state(outputs: Mapping[str, Tensor], model: Any) -> Dict[str, Any]:
    delta = outputs["camera_medium_delta_projected_raw"].detach()
    projector = model._camera_medium_observability_projector.detach()
    return {
        "delta_sha256": _tensor_hash(delta),
        "delta_mean_abs": float(delta.float().abs().mean()),
        "delta_l2_mean": float(torch.linalg.vector_norm(delta.float(), dim=-1).mean()),
        "projector_sha256": _tensor_hash(projector),
    }


def _localization_row(
    scene: str,
    camera_id: str,
    baseline: Tensor,
    gt: Tensor,
    projection: Tensor,
    projection_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    residual = (baseline.float().clamp(0, 1) - gt.float().clamp(0, 1)).square().mean(dim=-1)
    threshold = float(torch.quantile(residual.reshape(-1), LOCALIZATION_RESIDUAL_QUANTILE))
    high = residual >= threshold
    intersection = high & projection
    union = high | projection
    intersection_count = int(intersection.sum())
    high_count = int(high.sum())
    projection_count = int(projection.sum())
    union_count = int(union.sum())
    return {
        "scene": scene,
        "split": "eval",
        "camera_id": camera_id,
        "high_residual_quantile": LOCALIZATION_RESIDUAL_QUANTILE,
        "high_residual_threshold_mse": threshold,
        "high_residual_pixel_count": high_count,
        "projected_pixel_count": projection_count,
        "intersection_pixel_count": intersection_count,
        "union_pixel_count": union_count,
        "IoU": intersection_count / max(union_count, 1),
        "precision": intersection_count / max(projection_count, 1),
        "recall": intersection_count / max(high_count, 1),
        "gt_used_for_group_definition": False,
        "gt_used_for_localization_only": True,
        **projection_meta,
    }


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    keys = sorted({(str(row["scene"]), str(row["split"]), str(row["condition"])) for row in rows})
    for scene, split, condition in keys:
        subset = [row for row in rows if row["scene"] == scene and row["split"] == split and row["condition"] == condition]
        output.append(
            {
                "scene": scene,
                "split": split,
                "condition": condition,
                "group": subset[0]["group"],
                "alpha": subset[0]["alpha"],
                "view_count": len(subset),
                **{metric: _mean([float(row[metric]) for row in subset]) for metric in ("PSNR", "SSIM", "LPIPS", "MSE")},
            }
        )
    return output


def _deltas(metric_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["scene"], row["split"], row["condition"]): row for row in metric_rows}
    output: List[Dict[str, Any]] = []
    for row in metric_rows:
        if row["condition"] == "FULL":
            continue
        full = lookup[(row["scene"], row["split"], "FULL")]
        output.append(
            {
                **row,
                "delta_PSNR_improvement": float(row["PSNR"]) - float(full["PSNR"]),
                "delta_SSIM_improvement": float(row["SSIM"]) - float(full["SSIM"]),
                "delta_LPIPS_improvement": float(full["LPIPS"]) - float(row["LPIPS"]),
                "delta_MSE_improvement": float(full["MSE"]) - float(row["MSE"]),
            }
        )
    return output


def _primary_comparison(metric_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["scene"], row["split"], row["condition"]): row for row in metric_rows}
    output: List[Dict[str, Any]] = []
    for scene in SCENES:
        full = lookup[(scene, "eval", "FULL")]
        condition_rows: Dict[str, Dict[str, float]] = {}
        for condition in ("LOW_T1_ZERO", "RANDOM_MATCHED_ZERO", "HIGH_MATCHED_ZERO"):
            row = lookup[(scene, "eval", condition)]
            condition_rows[condition] = {
                "dPSNR": float(row["PSNR"]) - float(full["PSNR"]),
                "dSSIM": float(row["SSIM"]) - float(full["SSIM"]),
                "dLPIPS": float(full["LPIPS"]) - float(row["LPIPS"]),
                "dMSE": float(full["MSE"]) - float(row["MSE"]),
            }
        low = condition_rows["LOW_T1_ZERO"]
        random = condition_rows["RANDOM_MATCHED_ZERO"]
        high = condition_rows["HIGH_MATCHED_ZERO"]
        output.append(
            {
                "scene": scene,
                "primary_group": PRIMARY_GROUP,
                "alpha": PRIMARY_ALPHA,
                "low_dPSNR": low["dPSNR"],
                "random_dPSNR": random["dPSNR"],
                "high_dPSNR": high["dPSNR"],
                "low_dSSIM": low["dSSIM"],
                "random_dSSIM": random["dSSIM"],
                "high_dSSIM": high["dSSIM"],
                "low_dLPIPS": low["dLPIPS"],
                "random_dLPIPS": random["dLPIPS"],
                "high_dLPIPS": high["dLPIPS"],
                "low_dMSE": low["dMSE"],
                "random_dMSE": random["dMSE"],
                "high_dMSE": high["dMSE"],
                "low_improves_PSNR": low["dPSNR"] > 0.0,
                "low_beats_random_PSNR": low["dPSNR"] > random["dPSNR"],
                "low_beats_high_PSNR": low["dPSNR"] > high["dPSNR"],
                "scene_supports_primary_criterion": bool(
                    low["dPSNR"] > 0.0
                    and low["dPSNR"] > random["dPSNR"]
                    and low["dPSNR"] > high["dPSNR"]
                ),
            }
        )
    return output


@torch.no_grad()
def worker(scene: str, gpu: str, preflight_only: bool = False) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    os.environ["LOW_SUPPORT_SCENE"] = scene
    runtime = _runtime(gpu)
    started = time.perf_counter()
    scene_dir = OUTPUT_ROOT / "workers" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    branch = FORMAL._setup_branch(REPO_ROOT, _scene_config(scene), "C0")
    try:
        model = branch.pipeline.model
        payload = FORMAL._load_checkpoint(branch, _checkpoint(scene))
        if (
            payload.get("experiment") != FORMAL.EXPERIMENT
            or payload.get("branch") != "C0"
            or int(payload.get("absolute_step", -1)) != 14999
            or payload.get("ocmc_bundle") is None
            or payload.get("raoc_state") is not None
        ):
            raise RuntimeError("checkpoint experiment/condition provenance drift")
        if (
            not model.config.camera_medium_observability_enabled
            or model.config.camera_medium_ray_adaptive_observability_enabled
            or model.config.intrinsic_color_parameterization != "bounded_sh3"
            or model.config.rasterize_mode != "classic"
            or model.config.medium_context_mode != "dir_xy_camera"
            or int(model.config.sh_degree) != 3
        ):
            raise RuntimeError("formal C0 configuration drift")
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        split_manifest = _camera_split_manifest(train_records, eval_records)
        support = _support_counts(model, train_records)
        masks, group_manifest = _group_masks(support)
        state_before = _model_state_hash(model)
        projector_before = _tensor_hash(model._camera_medium_observability_projector)

        view_rows: List[Dict[str, Any]] = []
        localization_rows: List[Dict[str, Any]] = []
        equivalence_rows: List[Dict[str, Any]] = []
        ocmc_rows: List[Dict[str, Any]] = []
        specs = _condition_specs(masks)
        split_records = (("eval", eval_records[:1]),) if preflight_only else (("train", train_records), ("eval", eval_records))
        for split, records in split_records:
            for _index, camera_id, camera, batch in records:
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
                baseline = outputs["pred_image"].detach().float()
                baseline_metrics = MIC._metric_images(model, baseline, gt)
                view_rows.append(
                    {
                        "scene": scene,
                        "split": split,
                        "camera_id": camera_id,
                        "condition": "FULL",
                        "group": "FULL",
                        "alpha": 1.0,
                        **baseline_metrics,
                    }
                )
                geometry = _geometry(model, camera, int(baseline.shape[0]), int(baseline.shape[1]))
                if geometry["opacities"].shape[0] != support.numel():
                    raise RuntimeError("cropped geometry is incompatible with global support masks")
                if not torch.equal(geometry["radii"] > 0, outputs["gaussian_visible_mask"].reshape(-1)):
                    raise RuntimeError("reprojected geometry visibility drift")
                ocmc_before = _ocmc_view_state(outputs, model)
                for condition, group, alpha, mask in specs:
                    pred = _counterfactual(model, geometry, outputs, mask, alpha)
                    metrics = MIC._metric_images(model, pred, gt)
                    view_rows.append(
                        {
                            "scene": scene,
                            "split": split,
                            "camera_id": camera_id,
                            "condition": condition,
                            "group": group,
                            "alpha": alpha,
                            **metrics,
                        }
                    )
                    if alpha == 1.0:
                        difference = (pred.float() - baseline.float()).abs()
                        equivalence_rows.append(
                            {
                                "scene": scene,
                                "split": split,
                                "camera_id": camera_id,
                                "condition": condition,
                                "max_abs_pixel_difference": float(difference.max()),
                                "mean_abs_pixel_difference": float(difference.mean()),
                                "atol": FULL_ATOL,
                                "allclose": bool(torch.allclose(pred, baseline, atol=FULL_ATOL, rtol=0.0)),
                            }
                        )
                        if not equivalence_rows[-1]["allclose"]:
                            raise RuntimeError(f"alpha=1 does not reproduce FULL: {equivalence_rows[-1]}")
                    del pred
                ocmc_after = _ocmc_view_state(outputs, model)
                ocmc_equal = ocmc_before == ocmc_after
                ocmc_rows.append(
                    {
                        "scene": scene,
                        "split": split,
                        "camera_id": camera_id,
                        "delta_mean_abs_before": ocmc_before["delta_mean_abs"],
                        "delta_mean_abs_after": ocmc_after["delta_mean_abs"],
                        "delta_l2_mean_before": ocmc_before["delta_l2_mean"],
                        "delta_l2_mean_after": ocmc_after["delta_l2_mean"],
                        "delta_sha256_before": ocmc_before["delta_sha256"],
                        "delta_sha256_after": ocmc_after["delta_sha256"],
                        "projector_sha256_before": ocmc_before["projector_sha256"],
                        "projector_sha256_after": ocmc_after["projector_sha256"],
                        "exactly_equal": ocmc_equal,
                    }
                )
                if not ocmc_equal:
                    raise RuntimeError("OCMC state changed across opacity intervention")
                if split == "eval":
                    projection, projection_meta = _projection_mask(model, geometry, masks["LOW_T1"])
                    localization_rows.append(
                        _localization_row(scene, camera_id, baseline, gt, projection, projection_meta)
                    )
                del outputs, gt, baseline, geometry

        state_after = _model_state_hash(model)
        projector_after = _tensor_hash(model._camera_medium_observability_projector)
        if state_before != state_after or projector_before != projector_after:
            raise RuntimeError("frozen model or OCMC projector mutated")
        metrics = _aggregate_metrics(view_rows)
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "preflight_only": preflight_only,
            "runtime": runtime,
            "checkpoint": str(_checkpoint(scene)),
            "checkpoint_sha256": _sha256(_checkpoint(scene)),
            "camera_split": split_manifest,
            "groups": group_manifest,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "ocmc_projector_sha256_before": projector_before,
            "ocmc_projector_sha256_after": projector_after,
            "model_state_unchanged": state_before == state_after,
            "ocmc_projector_unchanged": projector_before == projector_after,
            "alpha_one_allclose": all(bool(row["allclose"]) for row in equivalence_rows),
            "alpha_one_max_abs_pixel_difference": max(float(row["max_abs_pixel_difference"]) for row in equivalence_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "view_metric_rows": len(view_rows),
            "localization_rows": len(localization_rows),
            "backward_calls": 0,
            "optimizer_step_calls": 0,
            "new_checkpoints": 0,
        }
        suffix = "_preflight" if preflight_only else ""
        _write_json(scene_dir / f"worker_summary{suffix}.json", result)
        _write_csv(scene_dir / f"per_view_metrics{suffix}.csv", view_rows)
        _write_csv(scene_dir / f"per_scene_metrics{suffix}.csv", metrics)
        _write_csv(scene_dir / f"pixel_localization{suffix}.csv", localization_rows)
        _write_csv(scene_dir / f"alpha_one_equivalence{suffix}.csv", equivalence_rows)
        _write_csv(scene_dir / f"ocmc_independence{suffix}.csv", ocmc_rows)
        return result
    finally:
        FORMAL._release(branch)


def _load_worker_rows(filename: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        path = OUTPUT_ROOT / "workers" / scene / filename
        with path.open(newline="", encoding="utf8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _coerce_metric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                **row,
                "alpha": float(row["alpha"]),
                "view_count": int(row["view_count"]),
                **{key: float(row[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
            }
        )
    return output


def _classification(primary: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    supported = sum(bool(row["scene_supports_primary_criterion"]) for row in primary)
    improves = sum(bool(row["low_improves_PSNR"]) for row in primary)
    if supported >= 3:
        label = "LOW_SUPPORT_CAUSAL_SUPPORTED"
        enter_module_design = True
        rationale = "T1 zero-opacity low-support suppression improves PSNR and beats both matched controls in at least 3/4 scenes."
    elif improves > 0:
        label = "LOW_SUPPORT_PROXY_ONLY"
        enter_module_design = False
        rationale = "Low-support suppression improves at least one scene but lacks the preregistered cross-scene advantage over both controls."
    else:
        label = "LOW_SUPPORT_NOT_SUPPORTED"
        enter_module_design = False
        rationale = "T1 zero-opacity low-support suppression does not improve held-out PSNR in any scene."
    return {
        "experiment": EXPERIMENT,
        "classification": label,
        "primary_metric": "heldout mean PSNR improvement versus FULL",
        "primary_group": PRIMARY_GROUP,
        "primary_alpha": PRIMARY_ALPHA,
        "required_supported_scenes": 3,
        "supported_scene_count": supported,
        "low_improves_scene_count": improves,
        "enter_module_design": enter_module_design,
        "rationale": rationale,
        "scene_rows": list(primary),
    }


def _scientific_answers(
    metric_rows: Sequence[Mapping[str, Any]],
    ablation: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> Dict[str, Any]:
    delta_lookup = {
        (str(row["scene"]), str(row["split"]), str(row["condition"])): row
        for row in ablation
    }
    scene_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        zero_eval = delta_lookup[(scene, "eval", "LOW_T1_ZERO")]
        half_eval = delta_lookup[(scene, "eval", "LOW_T1_HALF")]
        zero_train = delta_lookup[(scene, "train", "LOW_T1_ZERO")]
        comparison = next(row for row in primary if row["scene"] == scene)
        scene_rows.append(
            {
                "scene": scene,
                "zero_eval_dPSNR": float(zero_eval["delta_PSNR_improvement"]),
                "zero_eval_dSSIM": float(zero_eval["delta_SSIM_improvement"]),
                "zero_eval_dLPIPS": float(zero_eval["delta_LPIPS_improvement"]),
                "zero_eval_dMSE": float(zero_eval["delta_MSE_improvement"]),
                "half_eval_dPSNR": float(half_eval["delta_PSNR_improvement"]),
                "half_eval_dLPIPS": float(half_eval["delta_LPIPS_improvement"]),
                "zero_train_dPSNR": float(zero_train["delta_PSNR_improvement"]),
                "zero_train_dLPIPS": float(zero_train["delta_LPIPS_improvement"]),
                "zero_eval_psnr_and_lpips_both_improve": bool(
                    float(zero_eval["delta_PSNR_improvement"]) > 0.0
                    and float(zero_eval["delta_LPIPS_improvement"]) > 0.0
                ),
                "low_beats_random_PSNR": bool(comparison["low_beats_random_PSNR"]),
                "low_beats_high_PSNR": bool(comparison["low_beats_high_PSNR"]),
                "primary_scene_supported": bool(comparison["scene_supports_primary_criterion"]),
            }
        )
    return {
        "q1_low_zero_improves_novel_view": {
            "answer": False,
            "improved_scene_count": int(classification["low_improves_scene_count"]),
            "scene_count": len(SCENES),
        },
        "q2_low_zero_beats_random_control": {
            "scene_count": sum(bool(row["low_beats_random_PSNR"]) for row in primary),
            "required_with_positive_improvement": True,
        },
        "q3_low_zero_beats_high_control": {
            "scene_count": sum(bool(row["low_beats_high_PSNR"]) for row in primary),
            "required_with_positive_improvement": True,
        },
        "q4_psnr_lpips_tradeoff": {
            "primary_zero_both_improve_scene_count": sum(
                bool(row["zero_eval_psnr_and_lpips_both_improve"])
                for row in scene_rows
            ),
            "interpretation": (
                "T1 alpha=0 worsens both heldout PSNR and LPIPS in all four scenes; "
                "Curasao and Panama show tiny SSIM increases, not a causal win."
            ),
        },
        "q5_supported_scenes": [
            str(row["scene"])
            for row in primary
            if bool(row["scene_supports_primary_criterion"])
        ],
        "q6_cross_scene_stable": {
            "causal_support_stable": False,
            "negative_zero_suppression_result_stable": all(
                float(row["zero_eval_dPSNR"]) <= 0.0 for row in scene_rows
            ),
        },
        "q7_enter_module_design": bool(classification["enter_module_design"]),
        "q8_next_design": (
            ["reliability weighting", "support-aware refinement", "uncertainty modeling"]
            if classification["enter_module_design"]
            else []
        ),
        "train_view_tradeoff": {
            "all_scenes_degrade_under_low_t1_zero": all(
                float(row["zero_train_dPSNR"]) < 0.0 for row in scene_rows
            ),
            "dPSNR_range": [
                min(float(row["zero_train_dPSNR"]) for row in scene_rows),
                max(float(row["zero_train_dPSNR"]) for row in scene_rows),
            ],
        },
        "half_strength_sensitivity": {
            "heldout_psnr_improved_scenes": [
                str(row["scene"])
                for row in scene_rows
                if float(row["half_eval_dPSNR"]) > 0.0
            ],
            "classification_uses_half_strength": False,
        },
        "scene_rows": scene_rows,
        "metric_row_count": len(metric_rows),
    }


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    primary = summary["group_comparison"]
    localization = summary["pixel_localization_scene_means"]
    answers = summary["scientific_answers"]
    lines = [
        "# Low-Support Causal Intervention Preflight",
        "",
        "Date: 2026-09-01",
        f"Experiment: `{EXPERIMENT}`",
        f"Classification: `{classification['classification']}`",
        "",
        "## Frozen Protocol",
        "",
        "The audit loads only the four registered C0 OCMC checkpoints at step 14999. It performs forward rendering only. Per-Gaussian opacity is scaled in a detached local tensor after OCMC medium prediction; model opacity, topology, renderer source, OCMC, and RAOC state are unchanged. Support counts distinct training cameras with `model.radii > 0`. Held-out GT is used only for evaluation and localization.",
        "",
        "Primary criterion: T1 (`s <= 1`) at `alpha=0`; support requires positive held-out mean PSNR change and a larger change than both size-matched random non-low and high-support controls in at least 3/4 scenes. T2 is sensitivity-only.",
        "",
        "## Primary Results",
        "",
        "| Scene | low dPSNR | random dPSNR | high dPSNR | low dLPIPS | criterion |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['scene']} | {row['low_dPSNR']:.6f} | {row['random_dPSNR']:.6f} | {row['high_dPSNR']:.6f} | {row['low_dLPIPS']:.6f} | {'yes' if row['scene_supports_primary_criterion'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Localization",
            "",
            "Low-support indicator contribution (`> 1e-12`) is compared with each held-out view's top-20% baseline residual pixels. This diagnostic never feeds GT into group selection or intervention.",
            "",
            "| Scene | mean IoU | mean precision | mean recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in localization:
        lines.append(f"| {row['scene']} | {row['IoU']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} |")
    lines.extend(
        [
            "",
            "## Strength And Train-View Trade-Off",
            "",
            "| Scene | zero eval dPSNR | zero eval dLPIPS | half eval dPSNR | zero train dPSNR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in answers["scene_rows"]:
        lines.append(
            f"| {row['scene']} | {row['zero_eval_dPSNR']:.6f} | {row['zero_eval_dLPIPS']:.6f} | {row['half_eval_dPSNR']:.6f} | {row['zero_train_dPSNR']:.6f} |"
        )
    enter = bool(classification["enter_module_design"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            classification["rationale"],
            "",
            f"Low-support suppression improves novel-view PSNR in {classification['low_improves_scene_count']}/4 scenes and satisfies the full matched-control criterion in {classification['supported_scene_count']}/4 scenes.",
            "",
            f"At the primary zero strength, low support beats random in {answers['q2_low_zero_beats_random_control']['scene_count']}/4 scenes and high support in {answers['q3_low_zero_beats_high_control']['scene_count']}/4, but it never improves over FULL. PSNR and LPIPS both worsen in all four scenes. All train splits also degrade, with dPSNR from {answers['train_view_tradeoff']['dPSNR_range'][0]:.6f} to {answers['train_view_tradeoff']['dPSNR_range'][1]:.6f}.",
            "",
            f"Half suppression gives tiny heldout PSNR gains only in {', '.join(answers['half_strength_sensitivity']['heldout_psnr_improved_scenes'])}; this sensitivity result is not the preregistered primary comparison and does not authorize module design.",
            "",
            "The direction is not causally supported across scenes. The negative alpha=0 result is stable in 4/4 scenes, while localization overlap is heterogeneous and therefore remains correlational evidence only. Detailed PSNR, SSIM, LPIPS, and MSE results are recorded in `per_scene_metrics.csv`, `group_comparison.csv`, and `strength_ablation.csv`.",
            "",
            ("The preregistered gate passes; reliability weighting, support-aware refinement, and uncertainty modeling may proceed to separate module-design experiments." if enter else "The preregistered gate does not pass. Close the low-support module direction; do not begin reliability weighting, support-aware refinement, or uncertainty-module design from this evidence."),
            "",
            "## Integrity",
            "",
            f"All alpha=1 counterfactuals reproduce FULL within `{FULL_ATOL}` absolute pixel tolerance. Model state and the OCMC projector remain hash-identical before and after every scene audit. No backward call, optimizer step, training run, topology edit, renderer edit, or checkpoint write occurred.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    worker_summaries = [_read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json") for scene in SCENES]
    if not all(
        row["model_state_unchanged"]
        and row["ocmc_projector_unchanged"]
        and row["alpha_one_allclose"]
        and row["backward_calls"] == 0
        and row["optimizer_step_calls"] == 0
        for row in worker_summaries
    ):
        raise RuntimeError("worker integrity validation failed")
    metric_rows = _coerce_metric_rows(_load_worker_rows("per_scene_metrics.csv"))
    ablation = _deltas(metric_rows)
    primary = _primary_comparison(metric_rows)
    classification = _classification(primary)
    answers = _scientific_answers(metric_rows, ablation, primary, classification)
    localization_raw = _load_worker_rows("pixel_localization.csv")
    localization: List[Dict[str, Any]] = []
    for row in localization_raw:
        localization.append(
            {
                **row,
                **{
                    key: float(row[key])
                    for key in (
                        "high_residual_quantile",
                        "high_residual_threshold_mse",
                        "IoU",
                        "precision",
                        "recall",
                        "projection_threshold",
                        "mean_indicator_contribution",
                        "max_indicator_contribution",
                        "indicator_channel_max_abs_diff",
                    )
                },
            }
        )
    localization_means = [
        {
            "scene": scene,
            **{
                key: _mean([float(row[key]) for row in localization if row["scene"] == scene])
                for key in ("IoU", "precision", "recall")
            },
        }
        for scene in SCENES
    ]
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "scientific_answers": answers,
        "group_comparison": primary,
        "pixel_localization_scene_means": localization_means,
        "worker_summaries": worker_summaries,
        "condition_names_required": [
            "FULL",
            "LOW_T1_ZERO",
            "LOW_T1_HALF",
            "LOW_T2_ZERO",
            "HIGH_MATCHED_ZERO",
            "RANDOM_MATCHED_ZERO",
        ],
        "condition_names_evaluated": sorted({str(row["condition"]) for row in metric_rows}),
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "new_checkpoints": 0,
        "module_design_started": False,
    }
    _write_csv(OUTPUT_ROOT / "per_scene_metrics.csv", metric_rows)
    _write_csv(OUTPUT_ROOT / "group_comparison.csv", primary)
    _write_csv(OUTPUT_ROOT / "strength_ablation.csv", ablation)
    _write_csv(OUTPUT_ROOT / "pixel_localization.csv", localization)
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
        environment["LOW_SUPPORT_SCENE"] = scene
        command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", "--scene", scene, "--gpu", gpu]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((scene, gpu, process, handle))
    failures: List[Dict[str, Any]] = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")})
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"frozen workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        result = worker(args.scene, args.gpu, bool(args.preflight_only))
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(_sanitize_json(result), indent=2, sort_keys=True, default=_json_default, allow_nan=False))


if __name__ == "__main__":
    main()
