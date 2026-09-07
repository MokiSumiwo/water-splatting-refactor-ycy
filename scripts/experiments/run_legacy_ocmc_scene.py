#!/usr/bin/env python3
"""Train one Legacy-SH3 scene with the frozen OCMC projector.

This worker is the no-Bounded-SH3 source run for the three-improvement chain.
It keeps the registered OCMC calibration, camera sequence, optimizer, and
normal refinement callbacks, but never enables RAOC.  Only the compact
step-3000 and step-14999 checkpoints are retained because the former is the
common continuation state for the appearance and CICA workers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import yaml
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


SOURCE_SEQUENCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
FALLBACK_CONFIG = REPO_ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"
TRAINING_SEED = 42
FINAL_STEP = 14999
SNAPSHOT_STEPS = (3000, 14999)
REFRESH_STEPS = (0, 5000, 10000)
SCENES = tuple(FORMAL.SCENES)
PARAMETERIZATION = "legacy"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


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


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible != str(gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must expose exactly {gpu}, got {visible!r}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {
        "scene": scene,
        "physical_gpu": str(gpu),
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "python": str(sys.executable),
        "torch_version": torch.__version__,
        "cuda_visible_devices": visible,
    }


def _scene_cfg(scene: str) -> Mapping[str, Any]:
    if scene not in FORMAL.SCENES:
        raise ValueError(scene)
    config = dict(FORMAL.SCENES[scene])
    source = REPO_ROOT / str(config["source_config"])
    if not source.is_file():
        config["source_config"] = str(FALLBACK_CONFIG.relative_to(REPO_ROOT))
        config["config_fallback"] = str(FALLBACK_CONFIG)
    return config


def _load_sequence(scene: str, final_step: int = FINAL_STEP) -> List[Dict[str, Any]]:
    path = SOURCE_SEQUENCE_ROOT / scene / "camera_sequence.json"
    payload = json.loads(path.read_text(encoding="utf8"))
    rows = [dict(row) for row in payload["rows"]]
    if len(rows) != FINAL_STEP + 1 or [int(row["absolute_step"]) for row in rows] != list(range(FINAL_STEP + 1)):
        raise RuntimeError(f"invalid registered camera sequence: {path}")
    return rows[: int(final_step) + 1]


def _setup(repo: Path, scene_cfg: Mapping[str, Any]) -> Any:
    FORMAL.ACTIVE_PARAMETERIZATION = PARAMETERIZATION
    return FORMAL._setup_branch(repo, scene_cfg, "C0", PARAMETERIZATION)


def _training_rng(repo: Path, scene_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    FORMAL._seed_all(TRAINING_SEED)
    probe = _setup(repo, scene_cfg)
    try:
        return FORMAL._rng_state()
    finally:
        FORMAL._release(probe)


def _save_checkpoint(branch: Any, scene: str, step: int, output_dir: Path, bundle: Mapping[str, Any], rng: Mapping[str, Any]) -> Path:
    # Keep the source layout identical to the continuation workers: C0 is the
    # common OCMC-on state, while the source run itself has no RAOC arm.
    path = output_dir / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_cpu = {key: value.detach().cpu().clone() if isinstance(value, Tensor) else value for key, value in bundle.items()}
    torch.save(
        {
            "experiment": "LEGACY_OCMC_SOURCE_FOUR_SCENE",
            "branch": "C0",
            "absolute_step": int(step),
            "model": branch.pipeline.model.state_dict(),
            "optimizers": {key: value.state_dict() for key, value in branch.optimizers.optimizers.items()},
            "schedulers": {key: value.state_dict() for key, value in branch.optimizers.schedulers.items()},
            "scalers": dict(branch.scalers),
            "ocmc_bundle": bundle_cpu,
            "raoc_state": None,
            "rng_manifest": FORMAL._rng_manifest(rng),
            "metadata": {
                "scene": scene,
                "intrinsic_color_parameterization": PARAMETERIZATION,
                "ocmc_enabled": True,
                "raoc_enabled": False,
                "matched_camera_sequence": True,
            },
        },
        path,
    )
    return path


def _train(repo: Path, scene: str, gpu: str, output_dir: Path, final_step: int = FINAL_STEP) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    scene_cfg = _scene_cfg(scene)
    if final_step < 0 or final_step > FINAL_STEP:
        raise ValueError(f"final_step must be in [0, {FINAL_STEP}]")
    sequence = _load_sequence(scene, final_step)
    snapshot_steps = tuple(sorted({step for step in SNAPSHOT_STEPS if step <= final_step} | {int(final_step)}))
    refresh_steps = tuple(step for step in REFRESH_STEPS if step <= final_step)
    training_rng = _training_rng(repo, scene_cfg)
    branch = _setup(repo, scene_cfg)
    FORMAL._set_rng_state(training_rng)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    cached = dm.cached_train
    names = [str(row["camera_name"]) for row in sequence]
    output_dir.mkdir(parents=True, exist_ok=True)
    # No Bounded-SH3 checkpoint is part of this experiment.  In particular,
    # IUI3's historical locked-safe mask builder is not needed for OCMC
    # training and may reference archives that have intentionally been
    # removed.  Build the generic calibration bank directly.
    original_locked_safe = scene_cfg.get("locked_safe", False)
    if isinstance(scene_cfg, dict):
        scene_cfg["locked_safe"] = False
    samples, _bank = FORMAL._build_samples(repo, output_dir, scene, scene_cfg, branch)
    if isinstance(scene_cfg, dict):
        scene_cfg["locked_safe"] = original_locked_safe
    # Sample-bank construction is diagnostic-only and must not alter the
    # registered training RNG state.
    FORMAL._set_rng_state(training_rng)
    # The continuation experiments consume the registered full 15K sequence
    # from this file. Store the exact rows locally so the source directory is
    # self-contained and can be audited independently of the historical run.
    source_sequence = SOURCE_SEQUENCE_ROOT / scene / "camera_sequence.json"
    sequence_payload = json.loads(source_sequence.read_text(encoding="utf8"))
    sequence_rows = [dict(row) for row in sequence_payload["rows"]]
    encoded_sequence = json.dumps(sequence_rows, sort_keys=True, separators=(",", ":")).encode("utf8")
    _write_json(output_dir / "camera_sequence.json", {
        "scene": scene,
        "rows": sequence_rows,
        "length": len(sequence_rows),
        "absolute_step_start": 0,
        "absolute_step_end": FINAL_STEP,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "sha256": hashlib.sha256(encoded_sequence).hexdigest(),
        "source_path": str(source_sequence),
        "source_sha256": hashlib.sha256(source_sequence.read_bytes()).hexdigest(),
    })
    _write_csv(output_dir / "camera_sequence.csv", sequence_rows)
    _write_json(output_dir / "runtime.json", runtime)
    _write_json(output_dir / "experiment_manifest.json", {
        "experiment": "LEGACY_OCMC_SOURCE_FOUR_SCENE",
        "scene": scene,
        "parameterization": PARAMETERIZATION,
        "seed": TRAINING_SEED,
        "final_step": int(final_step),
        "snapshot_steps": list(snapshot_steps),
        "refresh_steps": list(refresh_steps),
        "camera_sequence_source": str(SOURCE_SEQUENCE_ROOT / scene / "camera_sequence.json"),
        "ocmc_enabled": True,
        "raoc_enabled": False,
    })
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    bundle = None
    started = time.perf_counter()
    try:
        for step, row in enumerate(sequence):
            camera_index = int(row["camera_index"])
            if step in refresh_steps:
                bundle, _state, _refresh_seconds = FORMAL._calibrate(branch, samples, step, output_dir)
                if not bundle:
                    raise RuntimeError("OCMC calibration returned no bundle")
            branch.pipeline.train()
            model.train()
            FORMAL._configure_model(model, "C0", PARAMETERIZATION)
            FORMAL._install_condition(model, "C0", bundle, None)
            MIC._run_before(model, branch.optimizers, step)
            branch.optimizers.zero_grad_all()
            batch = {key: value.to(model.device) if isinstance(value, Tensor) else value for key, value in cached[camera_index].copy().items()}
            camera = cameras[camera_index : camera_index + 1]
            iteration_started = time.perf_counter()
            outputs = model.get_outputs(camera)
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            if not bool(torch.isfinite(total).item()) or not FORMAL._finite_outputs(outputs):
                raise RuntimeError(f"non-finite state at {scene}/{step}")
            total.backward()
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(step)
            event = MIC._run_after(model, branch.optimizers, step)
            if event.get("refinement_called"):
                events.append({"absolute_step": step, "camera_name": names[step], **dict(event)})
            if step % 500 == 0 or step in SNAPSHOT_STEPS:
                with torch.no_grad():
                    pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
                    gt = FORMAL.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0.0, 1.0)
                rows.append({
                    "scene": scene,
                    "absolute_step": step,
                    "camera_index": camera_index,
                    "camera_name": names[step],
                    "L_total": float(total.detach().item()),
                    "gaussian_count": int(model.num_points),
                    "iteration_seconds": time.perf_counter() - iteration_started,
                    **MIC._metric_images(model, pred, gt),
                })
            if step in snapshot_steps:
                checkpoint = _save_checkpoint(branch, scene, step, output_dir, bundle, FORMAL._rng_state())
                checkpoints.append({"absolute_step": step, "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size})
                _write_csv(output_dir / "training_metrics.csv", rows)
                _write_csv(output_dir / "refinement_events.csv", events)
            del outputs, losses, total, batch
            if step % 250 == 0:
                torch.cuda.empty_cache()
        branch.pipeline.eval()
        model.eval()
        metrics = {}
        for split, records in (("train", FORMAL._train_records(branch.pipeline)), ("eval", FORMAL._eval_records(branch.pipeline))):
            values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
            for _idx, _view_id, camera, batch in records:
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    gt = FORMAL.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0.0, 1.0)
                    pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
                item = MIC._metric_images(model, pred, gt)
                for key in values:
                    values[key].append(item[key])
            metrics[split] = {key: sum(items) / len(items) for key, items in values.items()}
        payload = {
            "experiment": "LEGACY_OCMC_SOURCE_FOUR_SCENE",
            "scene": scene,
            "configuration": "legacy + dir_xy_camera + OCMC_ON + RAOC_OFF",
            "seed": TRAINING_SEED,
            "parameterization": PARAMETERIZATION,
            "final_step": int(final_step),
            "final_train": metrics["train"],
            "final_eval": metrics["eval"],
            "runtime": runtime,
            "training_wall_seconds": time.perf_counter() - started,
            "completed_updates": len(sequence),
            "checkpoint_rows": checkpoints,
            "refinement_event_count": len(events),
        }
        _write_json(output_dir / "final_summary.json", payload)
        _write_csv(output_dir / "final_metrics.csv", [{"scene": scene, "split": split, **values} for split, values in metrics.items()])
        return payload
    finally:
        FORMAL._release(branch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=FINAL_STEP)
    args = parser.parse_args()
    result = _train(args.repo.resolve(), args.scene, str(args.gpu), args.output_dir.resolve(), int(args.final_step))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
