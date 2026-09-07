#!/usr/bin/env python3
"""Launch the compact four-scene bounded baseline on four free GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ocmc_bounded_baseline_four_scene_20260905"
WORKER = REPO_ROOT / "scripts" / "experiments" / "run_ocmc_bounded_baseline_scene.py"
SCENE_GPUS = {"Curasao": "0", "IUI3-RedSea": "1", "JapaneseGradens-RedSea": "2", "Panama": "3"}
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--parameterization",
        choices=("bounded_sh3", "legacy"),
        default="bounded_sh3",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"non-empty output root: {root}")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    _write(
        root / "launcher_manifest.json",
        {
            "experiment": "OCMC_BOUNDED_BASELINE_FOUR_SCENE",
            "scene_gpu": SCENE_GPUS,
            "python": str(PYTHON),
            "worker": str(WORKER),
            "protocol": f"{args.parameterization}, dir_xy_camera, OCMC off, RAOC off, seed 42, 15000 updates",
            "source_sequences": "outputs/m1_raoc_causal_four_scene_20260827/<scene>/camera_sequence.json",
        },
    )
    processes: Dict[str, subprocess.Popen[str]] = {}
    handles: Dict[str, Any] = {}
    for scene, gpu in SCENE_GPUS.items():
        scene_dir = root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        handles[scene] = handle
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["CONDA_DEFAULT_ENV"] = "water_splatting"
        command = [
            str(PYTHON),
            str(WORKER),
            "--repo",
            str(repo),
            "--scene",
            scene,
            "--gpu",
            gpu,
            "--output-dir",
            str(scene_dir),
            "--parameterization",
            args.parameterization,
        ]
        print(f"launching {scene} on physical GPU {gpu}", flush=True)
        processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    statuses = {scene: process.wait() for scene, process in processes.items()}
    for handle in handles.values():
        handle.close()
    _write(root / "worker_status.json", statuses)
    if any(code != 0 for code in statuses.values()):
        raise SystemExit(1)
    print(json.dumps({"output_root": str(root), "worker_status": statuses}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
