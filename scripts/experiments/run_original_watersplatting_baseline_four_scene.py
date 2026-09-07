#!/usr/bin/env python3
"""Launch the original dir-only WaterSplatting baseline on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "original_watersplatting_baseline_four_scene_20260905"
WORKER = REPO_ROOT / "scripts" / "experiments" / "run_ocmc_bounded_baseline_scene.py"
SCENE_GPUS = {"Curasao": "4", "IUI3-RedSea": "5", "JapaneseGradens-RedSea": "6", "Panama": "7"}
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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
            "experiment": "ORIGINAL_WATERSPLATTING_BASELINE_FOUR_SCENE",
            "scene_gpu": SCENE_GPUS,
            "python": str(PYTHON),
            "worker": str(WORKER),
            "protocol": "legacy SH3, dir_only, OCMC off, auxiliary off, CICA off, RAOC off, MDRR off, seed 42, 15000 updates",
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
            "legacy",
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
