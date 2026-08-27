#!/usr/bin/env python3
"""Launch the registered four-scene RAOC causal experiment on GPUs 6-9."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
ALLOWED = frozenset(SCENE_GPUS.values())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--max-steps", type=int, default=15000, help="validation-only shortened worker run; default is the registered 15K protocol")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root.resolve()
    if set(SCENE_GPUS.values()) != ALLOWED or any(gpu in {"0", "1", "2", "3", "4", "5"} for gpu in ALLOWED):
        raise RuntimeError("formal GPU assignment policy is invalid")
    if root.exists() and any(root.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(f"non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "M1-RAOC-CAUSAL-FOUR-SCENE", "scene_gpu": SCENE_GPUS, "max_steps": int(args.max_steps), "exactly_one_visible_gpu_per_worker": True, "forbidden_physical_gpus": ["0", "1", "2", "3", "4", "5"], "worker_script": str(repo / "scripts/experiments/run_m1_raoc_causal_scene.py")}
    _write(root / "launcher_manifest.json", manifest)
    processes: Dict[str, subprocess.Popen[str]] = {}
    log_handles: Dict[str, Any] = {}
    worker = repo / "scripts/experiments" / "run_m1_raoc_causal_scene.py"
    try:
        for scene, gpu in SCENE_GPUS.items():
            scene_dir = root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            log = (root / "logs" / f"{scene}.log").open("w", encoding="utf8")
            log_handles[scene] = log
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [sys.executable, str(worker), "--repo", str(repo), "--scene", scene, "--gpu", gpu, "--output-dir", str(scene_dir), "--max-steps", str(args.max_steps)]
            print(f"launching {scene} on physical GPU {gpu}", flush=True)
            processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        statuses = {scene: process.wait() for scene, process in processes.items()}
        _write(root / "worker_status.json", statuses)
        if any(code != 0 for code in statuses.values()):
            raise SystemExit(1)
        analyzer = repo / "scripts" / "diagnostics" / "analyze_m1_raoc_causal_four_scene.py"
        command = [sys.executable, str(analyzer), "--output-root", str(root)]
        analysis_log = (root / "logs" / "aggregate.log").open("w", encoding="utf8")
        try:
            result = subprocess.run(command, cwd=str(repo), env=os.environ.copy(), stdout=analysis_log, stderr=subprocess.STDOUT, text=True)
        finally:
            analysis_log.close()
        _write(root / "aggregate_status.json", {"returncode": result.returncode, "command": command})
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        for handle in log_handles.values():
            handle.close()
    print(json.dumps({"output_root": str(root), "worker_status": statuses}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
