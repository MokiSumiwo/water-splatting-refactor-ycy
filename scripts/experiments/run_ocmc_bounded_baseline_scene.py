#!/usr/bin/env python3
"""Train one bounded-SH3 scene with OCMC disabled.

This is the missing matched baseline for the existing four-scene OCMC run.
It deliberately reuses the registered WaterSplatting training path and only
turns off both observability paths.  The worker stores compact metrics only;
the existing OCMC checkpoints and camera sequences remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import yaml
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
FALLBACK_CONFIG = REPO_ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"
TRAINING_SEED = 42
LOG_STEPS = (0, 3000, 5000, 8000, 10000, 13000, 14999)
SCENES = FORMAL.SCENES
ALLOWED_GPUS = frozenset(str(i) for i in range(10))
MEDIUM_CONTEXT_MODE = "dir_only"


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
    if scene not in SCENES:
        raise ValueError(scene)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != gpu or gpu not in ALLOWED_GPUS:
        raise RuntimeError(f"expected one visible assigned GPU {gpu}, got {visible!r}")
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
        "python": sys.executable,
        "torch": torch.__version__,
        "total_memory_bytes": int(props.total_memory),
    }


def _configure_plain(model: Any, parameterization: str = "bounded_sh3") -> None:
    """Apply the original medium field and disable all observability paths."""

    FORMAL._configure_model(model, "C0")
    model.config.intrinsic_color_parameterization = parameterization
    model.config.medium_context_mode = MEDIUM_CONTEXT_MODE
    model.config.camera_medium_observability_enabled = False
    model.config.camera_medium_ray_adaptive_observability_enabled = False
    model.clear_camera_medium_ray_adaptive_observability_state()
    model.set_camera_medium_observability_projector(None)


def _scene_cfg(scene: str) -> Dict[str, Any]:
    """Use the retained canonical BND config when cleaned archives are absent."""

    config = dict(SCENES[scene])
    source = REPO_ROOT / str(config["source_config"])
    if source.is_file():
        return config
    if not FALLBACK_CONFIG.is_file():
        raise FileNotFoundError(f"no retained BND config for {scene}: {source}")
    config["source_config"] = str(FALLBACK_CONFIG.relative_to(REPO_ROOT))
    config["config_fallback"] = str(FALLBACK_CONFIG)
    return config


def _setup_plain(
    repo: Path,
    scene_cfg: Mapping[str, Any],
    parameterization: str = "bounded_sh3",
) -> Any:
    # FORMAL._setup_branch installs the registered bounded-SH3 config and
    # optimizer groups.  Reconfigure only the two observability flags after
    # construction so the model parameterization remains identical.
    if parameterization == "bounded_sh3":
        branch = FORMAL._setup_branch(repo, scene_cfg, "C0")
        _configure_plain(branch.pipeline.model, parameterization)
        return branch
    if parameterization != "legacy":
        raise ValueError(f"unsupported parameterization: {parameterization}")

    # Legacy must be selected before pipeline construction so the seed colors
    # are initialized in the legacy SH basis rather than as bounded logits.
    config_path = repo / str(scene_cfg["source_config"])
    config = yaml.load(config_path.read_text(encoding="utf8"), Loader=yaml.Loader)
    config.pipeline.datamanager._target = FORMAL.all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = False
    config.pipeline.datamanager.dataparser.data = repo / str(scene_cfg["data_path"])
    FORMAL._configure_model_config(config, "C0")
    config.pipeline.model.intrinsic_color_parameterization = "legacy"
    # This must be set before pipeline construction: the medium MLP input
    # dimension depends on the context mode.
    config.pipeline.model.medium_context_mode = MEDIUM_CONTEXT_MODE
    FORMAL._set_random_seed(int(config.machine.seed))
    pipeline = config.pipeline.setup(device=torch.device("cuda:0"), test_mode="test")
    _configure_plain(pipeline.model, parameterization)
    pipeline.model.clear_camera_medium_ray_adaptive_observability_state()
    pipeline.model.set_camera_medium_observability_projector(None)
    pipeline.model.step = 0
    optimizers = MIC._optimizer_groups(config, pipeline.model)
    optimizers = FORMAL.Optimizers(optimizers, pipeline.model.get_param_groups())
    pipeline.eval()
    return FORMAL.BranchState("C0", config_path, config, pipeline, optimizers, {})


def _load_sequence(scene: str) -> List[Dict[str, Any]]:
    path = SOURCE_ROOT / scene / "camera_sequence.json"
    payload = json.loads(path.read_text(encoding="utf8"))
    rows = payload["rows"]
    if len(rows) != 15000:
        raise RuntimeError(f"unexpected sequence length for {scene}: {len(rows)}")
    return rows


def _training_rng(
    repo: Path,
    scene_cfg: Mapping[str, Any],
    parameterization: str = "bounded_sh3",
) -> Mapping[str, Any]:
    # This mirrors the registered run: seed, construct the probe pipeline,
    # then capture RNG immediately before deterministic sampling/sequence use.
    FORMAL._seed_all(TRAINING_SEED)
    probe = _setup_plain(repo, scene_cfg, parameterization)
    try:
        return FORMAL._rng_state()
    finally:
        FORMAL._release(probe)


def _metric_row(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    return {key: float(value) for key, value in MIC._metric_images(model, pred, gt).items()}


def _evaluate_branch(branch: Any) -> Dict[str, Dict[str, float]]:
    """Evaluate the already-trained branch without reinitializing its model."""

    branch.pipeline.eval()
    model = branch.pipeline.model
    result: Dict[str, Dict[str, float]] = {}
    for split, records in (
        ("train", FORMAL._train_records(branch.pipeline)),
        ("eval", FORMAL._eval_records(branch.pipeline)),
    ):
        values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
        for _idx, _view_id, camera, batch in records:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = FORMAL.MI.PW._get_gt(
                    model, batch, outputs["background"]
                ).detach().float().clamp(0.0, 1.0)
                pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            metrics = _metric_row(model, pred, gt)
            for key in values:
                values[key].append(metrics[key])
        if not all(values.values()):
            raise RuntimeError(f"empty {split} evaluation split")
        result[split] = {
            key: sum(items) / len(items) for key, items in values.items()
        }
    return result


def _train(
    repo: Path,
    scene: str,
    scene_cfg: Mapping[str, Any],
    sequence: Sequence[Mapping[str, Any]],
    training_rng: Mapping[str, Any],
    output_dir: Path,
    parameterization: str = "bounded_sh3",
) -> Dict[str, Any]:
    branch = _setup_plain(repo, scene_cfg, parameterization)
    FORMAL._set_rng_state(training_rng)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    cached = dm.cached_train
    names = [str(row["camera_name"]) for row in sequence]
    train_rows: List[Dict[str, Any]] = []
    refinement_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        model.eval()
        for row in sequence:
            step = int(row["absolute_step"])
            camera_index = int(row["camera_index"])
            branch.pipeline.train()
            model.train()
            _configure_plain(model, parameterization)
            MIC._run_before(model, branch.optimizers, step)
            branch.optimizers.zero_grad_all()
            batch = {
                key: value.to(model.device) if isinstance(value, Tensor) else value
                for key, value in cached[camera_index].copy().items()
            }
            camera = cameras[camera_index : camera_index + 1]
            iteration_started = time.perf_counter()
            outputs = model.get_outputs(camera)
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(f"non-finite loss at {scene}/{step}")
            total.backward()
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(step)
            event = MIC._run_after(model, branch.optimizers, step)
            if event.get("refinement_called"):
                refinement_rows.append({"absolute_step": step, **dict(event)})
            if step in LOG_STEPS:
                with torch.no_grad():
                    pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
                    gt = FORMAL.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0.0, 1.0)
                metrics = _metric_row(model, pred, gt)
                train_rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "camera_index": camera_index,
                        "camera_name": names[step],
                        "L_total": float(total.detach().cpu().item()),
                        "gaussian_count": int(model.means.shape[0]),
                        "iteration_seconds": time.perf_counter() - iteration_started,
                        **metrics,
                    }
                )
            del outputs, losses, total, batch
            if step % 250 == 0:
                torch.cuda.empty_cache()
        runtime = {
            "training_wall_seconds": time.perf_counter() - started,
            "completed_updates": len(sequence),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        # Evaluate before releasing this branch. Recreating a branch here
        # would evaluate a fresh, untrained model rather than the checkpoint
        # produced by the 15,000 updates above.
        metrics = _evaluate_branch(branch)
        return {
            "train_rows": train_rows,
            "refinement_rows": refinement_rows,
            "runtime": runtime,
            "metrics": metrics,
        }
    finally:
        FORMAL._release(branch)


def run(
    repo: Path,
    scene: str,
    gpu: str,
    output_dir: Path,
    parameterization: str = "bounded_sh3",
) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = _scene_cfg(scene)
    sequence = _load_sequence(scene)
    training_rng = _training_rng(repo, scene_cfg, parameterization)
    training = _train(
        repo,
        scene,
        scene_cfg,
        sequence,
        training_rng,
        output_dir,
        parameterization,
    )
    metrics = training["metrics"]
    payload = {
        "experiment": "ORIGINAL_WATERSPLATTING_BASELINE_FOUR_SCENE",
        "scene": scene,
        "configuration": f"{parameterization} + {MEDIUM_CONTEXT_MODE} + OCMC_OFF + AUXILIARY_OFF + CICA_OFF + RAOC_OFF + MDRR_OFF",
        "medium_context_mode": MEDIUM_CONTEXT_MODE,
        "ocmc_enabled": False,
        "auxiliary_appearance_enabled": False,
        "cica_enabled": False,
        "raoc_enabled": False,
        "mdrr_enabled": False,
        "seed": TRAINING_SEED,
        "final_step": 14999,
        "camera_sequence_source": str(SOURCE_ROOT / scene / "camera_sequence.json"),
        "camera_sequence_match_required": True,
        "runtime": runtime,
        "training_runtime": training["runtime"],
        "completed_updates": int(training["runtime"]["completed_updates"]),
        "training_wall_seconds": float(training["runtime"]["training_wall_seconds"]),
        "final_train": metrics["train"],
        "final_eval": metrics["eval"],
        "refinement_event_count": len(training["refinement_rows"]),
    }
    _write_json(output_dir / "final_summary.json", payload)
    _write_csv(output_dir / "final_metrics.csv", [{"scene": scene, "split": split, **values} for split, values in metrics.items()])
    _write_csv(output_dir / "training_metrics.csv", training["train_rows"])
    _write_csv(output_dir / "refinement_events.csv", training["refinement_rows"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parameterization",
        choices=("bounded_sh3", "legacy"),
        default="bounded_sh3",
    )
    args = parser.parse_args()
    payload = run(
        args.repo.resolve(),
        args.scene,
        str(args.gpu),
        args.output_dir.resolve(),
        args.parameterization,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
