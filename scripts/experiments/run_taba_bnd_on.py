#!/usr/bin/env python3
"""Strictly paired chromatic-only TABA continuation on physical GPUs 6--9."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIRED = ROOT / "outputs/taba_paired_baseline_20260910_strict_v4"
DEFAULT_OUT = ROOT / "outputs/taba_chromatic_bnd_on_20260910"
SCENES = ("JapaneseGradens-RedSea", "Panama")
EXTENDED_SCENES = ("Curasao", "IUI3-RedSea")
GPUS = ("6", "7", "8", "9")
PRESETS = {"E0": (False, 0.50, 0.15, 0.20), "E1": (True, 0.50, 0.15, 0.20), "E2": (True, 0.40, 0.20, 0.30)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_environment():
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting" or Path(sys.prefix).name != "water_splatting":
        raise RuntimeError("Use the existing water_splatting conda environment")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or not set(visible) <= set(GPUS):
        raise RuntimeError("Only physical GPUs 6,7,8,9 are authorized")


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    index = {row[1].strip(): row[0].strip() for row in csv.reader(result.stdout.splitlines())}
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    active = {gpu: [] for gpu in GPUS}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) >= 3 and index.get(row[0].strip()) in active:
            active[index[row[0].strip()]].append({"pid": int(row[1]), "process": row[2].strip()})
    return active


def source_assets(scene, variant="bnd_on"):
    if scene in EXTENDED_SCENES:
        if variant != "bnd_on":
            raise ValueError("Extended scenes use the registered bounded C0 sources")
        root = ROOT / "outputs/m1_raoc_causal_four_scene_20260827"
        config = ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml"
        return root, config, {"source_provenance": "Registered BND C0@3000; worker verifies its original SHA256 before loading"}
    directory = PAIRED / scene / variant
    manifest = read_json(directory / "experiment_manifest.json")
    checkpoint = Path(manifest["source_checkpoint"])
    checkpoint.relative_to(ROOT)
    return checkpoint.parents[3], directory / "source_config.yml", manifest


def configure_worker(variant, log_path=None):
    from scripts.experiments import run_direct_mdrr_cica_scene as worker

    enabled, mid, temperature, floor = PRESETS[variant]
    worker.EXPERIMENT = "TABA_CHROMATIC_BND_ON_" + variant
    worker.ARMS = ("A2",)
    worker.AUXILIARY_APPEARANCE_ENABLED = False
    worker.AUXILIARY_APPEARANCE_STRENGTH = 0.0
    worker.OCMC_ENABLED = True
    worker.OCMC_STRENGTH = worker.OCMC_STRENGTH_START = 1.0
    worker.OCMC_STRENGTH_RAMP_STEPS = 0
    worker.CICA_ENABLED = True
    worker.CICA_RAMP_STEPS = 0
    worker.CICA_MIN_SUPPORT = 3
    worker.BRASH_ENABLED = worker.BVAM_ENABLED = False
    worker.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    worker.ALLOWED_GPUS = frozenset(GPUS)
    worker.ACTIVE_PARAMETERIZATION = "bounded_sh3"
    worker.ACTIVE_TRAINING_SEED = 42
    worker.ACTIVE_SNAPSHOT_STEPS = (5000, 10000, 14999)
    worker.ACTIVE_COMPACT_OUTPUT = False
    worker.TABA_EXPERIMENT = True
    worker.TABA_ENABLED = enabled
    worker.TABA_MODE = "chromatic" if enabled else "disabled"
    worker.TABA_UNCERTAINTY_LAMBDA = 0.50
    worker.TABA_WATER_AXIS_EPS = 1e-4
    worker.TABA_DIAGNOSTIC_ONLY = False
    worker.TABA_DIAGNOSTIC_ENABLED = True
    worker.TABA_TRANSMISSION_MID = mid
    worker.TABA_TRANSMISSION_TEMPERATURE = temperature
    worker.TABA_CHROMA_FLOOR = floor
    worker.TABA_LOG_PATH = str(log_path) if log_path is not None else None
    return worker


def source_audit(output):
    import torch
    import yaml

    worker = configure_worker("E0")
    rows = []
    for scene in SCENES:
        branches = {}
        for variant in ("bnd_on", "bnd_off"):
            root, config_path, manifest = source_assets(scene, variant)
            checkpoint = root / scene / "checkpoints/C0/step-000003000.ckpt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            expected_parameterization = "bounded_sh3" if variant == "bnd_on" else "legacy"
            assert payload["branch"] == "C0" and payload["absolute_step"] == 3000
            assert payload["metadata"]["intrinsic_color_parameterization"] == expected_parameterization
            assert payload["ocmc_bundle"] is not None and payload["raoc_state"] is None
            assert sha256(checkpoint) == manifest["source_checkpoint_sha256"]
            seed_file = "rng_manifest.json" if variant == "bnd_on" else "experiment_manifest.json"
            assert manifest["training_seed"] == read_json(root / scene / seed_file)["seed"] == 42
            sequence = read_json(root / scene / "camera_sequence.json")["rows"]
            recorded = read_json(PAIRED / scene / variant / "camera_sequence_hashes.json")["rows"]
            assert [row for row in sequence if int(row["absolute_step"]) > 3000] == recorded
            config = yaml.load(config_path.read_text(encoding="utf8"), Loader=yaml.Loader)
            config.pipeline.datamanager.dataparser.data = ROOT / "undistorted_data" / ("undistorted_" + scene)
            parser = config.pipeline.datamanager.dataparser.setup()
            split = {name: [str(path.relative_to(ROOT)) for path in parser.get_dataparser_outputs(split=name).image_filenames]
                     for name in ("train", "test")}
            assert set(split["train"]).isdisjoint(split["test"])
            branches[variant] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "source_root": str(root), "source_config": str(config_path),
                "source_parameterization": expected_parameterization, "seed": 42,
                "sequence_sha256": worker._hash_object(sequence), "continuation_sequence_sha256": worker._hash_object(recorded),
                "split": split, "model_keys": sorted(payload["model"]), "optimizer_groups": sorted(payload["optimizers"]),
                "finite_model": all(bool(torch.isfinite(value).all()) for value in payload["model"].values() if isinstance(value, torch.Tensor)),
            }
            del payload
        assert branches["bnd_on"]["sequence_sha256"] == branches["bnd_off"]["sequence_sha256"]
        assert branches["bnd_on"]["split"] == branches["bnd_off"]["split"]
        assert all(branch["finite_model"] for branch in branches.values())
        rows.append({"scene": scene, "paired": True, "branches": branches})
    result = {"passed": True, "scenes": rows, "runtime": {"python": sys.executable, "environment": os.environ["CONDA_DEFAULT_ENV"],
              "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}}
    write_json(output / "source_asset_audit.json", result)
    return result


def run_scene(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("Worker must expose only its assigned GPU")
    occupied = gpu_processes()[args.gpu]
    if occupied:
        raise RuntimeError(f"GPU {args.gpu} is occupied: {occupied}")
    subprocess.run(["nvidia-smi", "-i", args.gpu], check=True)
    target = args.output_root / args.scene / args.variant
    worker = configure_worker(args.variant, target / "taba_training.jsonl")
    root, config, _manifest = source_assets(args.scene)
    worker.ACTIVE_RENDER_ROOT = args.output_root / "renders" / args.variant
    if args.evaluate_only:
        worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG = root, config
        worker._runtime(args.scene, args.gpu)
        evaluation = worker._evaluate_arm(args.scene, "A2", target, worker.ACTIVE_RENDER_ROOT / args.scene)
        result = {"scene": args.scene, "variant": args.variant, "evaluation": evaluation,
                  "training": read_json(target / "training_summary.json"), "checkpoint_retained": True,
                  "evaluation_recovery": True}
        write_json(target / "taba_result.json", worker._sanitize(result))
        return
    result = worker.run(args.scene, "A2", args.gpu, target, args.max_step,
                        source_root=root, source_config=config, parameterization="bounded_sh3",
                        training_seed=42, compact_output=False)
    write_json(target / "taba_result.json", worker._sanitize(result))


def launch(args, *, jobs=None, worker_script=None, audit_sources=True):
    if audit_sources:
        source_audit(args.output_root)
    if jobs is None:
        jobs = [("JapaneseGradens-RedSea", "E0", "6"), ("JapaneseGradens-RedSea", "E1", "7"),
            ("Panama", "E0", "8"), ("Panama", "E1", "9"),
            ("JapaneseGradens-RedSea", "E2", "6"), ("Panama", "E2", "8")]
    pending, running, completed = list(jobs), {}, {}
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for gpu, (process, handle, key) in list(running.items()):
            status = process.poll()
            if status is not None:
                handle.close()
                completed[key] = status
                del running[gpu]
        occupied = gpu_processes()
        for job in list(pending):
            scene, variant, gpu = job
            available = [candidate for candidate in (gpu,) + tuple(value for value in GPUS if value != gpu)
                         if candidate not in running and not occupied[candidate]]
            if not available:
                continue
            gpu = available[0]
            key = scene + "_" + variant
            target = args.output_root / scene / variant
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f"Refusing to overwrite existing experiment: {target}")
            subprocess.run(["nvidia-smi", "-i", gpu], check=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log = (logs / (key + ".log")).open("x", encoding="utf8")
            command = [sys.executable, str(worker_script or Path(__file__).resolve()), "--scene", scene, "--variant", variant,
                       "--gpu", gpu, "--output-root", str(args.output_root), "--max-step", str(args.max_step)]
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (process, log, key)
            pending.remove(job)
            print(f"Started {key} on physical GPU {gpu}; PID {process.pid}", flush=True)
        write_json(args.output_root / "queue_status.json", {"pending": pending, "running": {gpu: value[2] for gpu, value in running.items()},
                   "completed": completed, "gpu_processes": occupied, "updated_unix": time.time()})
        if pending or running:
            print(json.dumps({"pending": len(pending), "running": [value[2] for value in running.values()], "completed": completed}), flush=True)
            time.sleep(30)
    if any(completed.values()):
        raise RuntimeError(f"Experiments failed; artifacts retained: {completed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES + EXTENDED_SCENES)
    parser.add_argument("--variant", choices=tuple(PRESETS), default="E0")
    parser.add_argument("--gpu", choices=GPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-step", type=int, default=14999)
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    check_environment()
    if args.audit_source:
        result = source_audit(args.output_root)
        print(json.dumps({"passed": result["passed"], "output": str(args.output_root / "source_asset_audit.json")}), flush=True)
    elif args.launch:
        launch(args)
    elif args.scene and args.gpu:
        run_scene(args)
    else:
        parser.error("Choose --audit-source, --launch, or --scene and --gpu")


if __name__ == "__main__":
    main()
