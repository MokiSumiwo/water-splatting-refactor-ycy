#!/usr/bin/env python3
"""Launch the formal four-scene identifiability causal experiment on GPUs 6-9."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "identifiability_module_causal_iui3_20260902"
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--final-step", type=int, default=14999)
    parser.add_argument("--diagnostic-sample-count", type=int, default=256)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"non-empty output root: {root}")
    if len(set(SCENE_GPUS.values())) != len(SCENE_GPUS):
        raise RuntimeError("formal workers must use distinct GPUs")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = repo / "scripts" / "experiments" / "run_identifiability_module_causal_scene.py"
    analyzer = repo / "scripts" / "diagnostics" / "analyze_identifiability_module_causal_four_scene.py"
    manifest = {
        "experiment": "IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT",
        "scene_gpu": SCENE_GPUS,
        "exactly_one_visible_gpu_per_worker": True,
        "worker_script": str(worker),
        "analyzer_script": str(analyzer),
        "final_step": int(args.final_step),
        "diagnostic_sample_count": int(args.diagnostic_sample_count),
        "module_strength": 1.0,
        "no_sweep": True,
    }
    _write(root / "launcher_manifest.json", manifest)
    processes: Dict[str, subprocess.Popen[str]] = {}
    handles: Dict[str, Any] = {}
    try:
        for scene, gpu in SCENE_GPUS.items():
            scene_dir = root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            handle = (logs / f"{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["CONDA_DEFAULT_ENV"] = "water_splatting"
            command = [
                sys.executable,
                str(worker),
                "--scene",
                scene,
                "--gpu",
                gpu,
                "--output-dir",
                str(scene_dir),
                "--final-step",
                str(args.final_step),
                "--diagnostic-sample-count",
                str(args.diagnostic_sample_count),
            ]
            print(f"launching {scene} on physical GPU {gpu}", flush=True)
            processes[scene] = subprocess.Popen(
                command,
                cwd=str(repo),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        statuses = {scene: process.wait() for scene, process in processes.items()}
        _write(root / "worker_status.json", statuses)
        if any(code != 0 for code in statuses.values()):
            raise SystemExit(1)
        aggregate_handle = (logs / "aggregate.log").open("w", encoding="utf8")
        try:
            result = subprocess.run(
                [sys.executable, str(analyzer), "--output-root", str(root)],
                cwd=str(repo),
                env=os.environ.copy(),
                stdout=aggregate_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            aggregate_handle.close()
        _write(root / "aggregate_status.json", {"returncode": result.returncode})
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps({"output_root": str(root), "worker_status": statuses}, sort_keys=True))


if __name__ == "__main__":
    main()
