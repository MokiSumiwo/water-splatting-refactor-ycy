#!/usr/bin/env python3
"""Launch ten Curasao continuation runs with OS-generated random seeds."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = "Curasao"
GPU_IDS = tuple(str(index) for index in range(10))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "curasao_m1_ocmc_cica_no_aux_random_multiseed_20260907"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "outputs" / "legacy_no_bounded_three_improvements_20260905"
DEFAULT_SOURCE_CONFIG = REPO_ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"


def _random_seed(existing: set[int]) -> int:
    while True:
        candidate = secrets.randbelow(2**31 - 1) + 1
        if candidate not in existing:
            return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count != 10:
        raise RuntimeError("this formal experiment requires exactly ten random seeds")

    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    source_config = args.source_config.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"non-empty output root: {output_root}")
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds: List[int] = []
    for _ in range(args.count):
        seeds.append(_random_seed(set(seeds)))
    launch_manifest: Dict[str, Any] = {
        "experiment": "M1_OCMC_CICA_AUXILIARY_OFF_LEGACY_SH3_CURASAO_OS_RANDOM_SEED_SWEEP",
        "scene": SCENE,
        "seed_count": len(seeds),
        "seed_source": "Python secrets.randbelow backed by the operating-system cryptographic random source",
        "seed_generation": "ten unique seeds generated at launch; no sequential or user-specified seed schedule",
        "seeds": seeds,
        "gpus": list(GPU_IDS),
        "source_root": str(source_root),
        "source_config": str(source_config),
        "launched_at_unix": time.time(),
        "workers": [],
    }
    processes: List[tuple[int, int, subprocess.Popen[Any]]] = []
    worker_script = REPO_ROOT / "scripts" / "experiments" / "run_legacy_cica_no_aux_scene.py"
    for index, (seed, gpu) in enumerate(zip(seeds, GPU_IDS)):
        output_dir = output_root / f"seed_{seed}" / SCENE
        log_path = output_root / f"seed_{seed}_{SCENE}.log"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["CONDA_DEFAULT_ENV"] = "water_splatting"
        command = [
            sys.executable,
            str(worker_script),
            "--scene", SCENE,
            "--gpu", gpu,
            "--output-dir", str(output_dir),
            "--source-root", str(source_root),
            "--source-config", str(source_config),
            "--seed", str(seed),
            "--compact-output",
        ]
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((seed, int(process.pid), process))
        launch_manifest["workers"].append({"seed": seed, "gpu": gpu, "pid": int(process.pid), "output_dir": str(output_dir), "log": str(log_path)})
        log_handle.close()

    manifest_path = output_root / "random_launch_manifest.json"
    manifest_path.write_text(json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps({"event": "launched", "manifest": str(manifest_path), "seeds": seeds, "pids": [pid for _, pid, _ in processes]}, sort_keys=True), flush=True)

    failures: List[Dict[str, Any]] = []
    for seed, pid, process in processes:
        return_code = process.wait()
        print(json.dumps({"event": "completed", "seed": seed, "pid": pid, "return_code": return_code}, sort_keys=True), flush=True)
        if return_code != 0:
            failures.append({"seed": seed, "pid": pid, "return_code": return_code})
    launch_manifest["completed_at_unix"] = time.time()
    launch_manifest["failures"] = failures
    launch_manifest["success"] = not failures
    manifest_path.write_text(json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    if failures:
        raise SystemExit(f"random seed sweep failed: {failures}")


if __name__ == "__main__":
    main()
