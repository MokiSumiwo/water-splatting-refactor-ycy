#!/usr/bin/env python3
"""Rerun the four-scene BND-ON M1+OCMC+CICA baseline from scratch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENES = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "research_runs/bnd_on_baseline_scratch15k_20261021_v2",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=ROOT / "research_runs/native_responsibility_raw_run_smoke_v3/source_config.yml",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    source_config = args.source_config.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    for scene in SCENES:
        data = ROOT / "undistorted_data" / f"undistorted_{scene}"
        if not (data / "sparse/0/points3D.bin").is_file():
            raise FileNotFoundError(data / "sparse/0/points3D.bin")

    output_root.mkdir(parents=True)
    (output_root / "logs").mkdir()
    manifest = {
        "experiment": "BND_ON_M1_OCMC_CICA_SCRATCH_15K",
        "scene_gpu": SCENES,
        "source_config": str(source_config),
        "python": str(PYTHON),
        "protocol": {
            "initialization": "raw COLMAP and seeded random medium network; no learned checkpoint",
            "updates": 15000,
            "seed": 42,
            "intrinsic_color_parameterization": "bounded_sh3",
            "medium_context_mode": "dir_xy_camera",
            "ocmc": "enabled, calibrated per scene at step 0, strength 1.0",
            "cica": "enabled, calibrated at step 10000; refresh at 12000 and 14000",
            "auxiliary_raoc_mdrr_bvam_brash": "disabled",
            "renderer_backward": "corrected_v1",
            "checkpoints": [15000],
            "evaluation": "project-local metrics.py, exact native-resolution eval views",
        },
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    processes = {}
    handles = {}
    for scene, gpu in SCENES.items():
        scene_dir = output_root / scene
        scene_dir.mkdir()
        log = (output_root / "logs" / f"{scene}.log").open("w")
        handles[scene] = log
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CONDA_DEFAULT_ENV="water_splatting")
        command = [
            str(PYTHON), str(ROOT / "scripts/experiments/run_bnd_on_baseline_worker_20261021.py"),
            "--scene", scene, "--gpu", gpu, "--output-dir", str(scene_dir),
            "--source-config", str(source_config),
        ]
        processes[scene] = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        print(f"launched {scene} on GPU {gpu}", flush=True)

    statuses = {scene: process.wait() for scene, process in processes.items()}
    for handle in handles.values():
        handle.close()
    (output_root / "worker_status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if any(statuses.values()):
        raise SystemExit(1)
    print(json.dumps({"output_root": str(output_root), "status": statuses}), flush=True)


if __name__ == "__main__":
    main()
