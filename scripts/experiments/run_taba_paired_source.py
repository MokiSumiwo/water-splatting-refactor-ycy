#!/usr/bin/env python3
"""Build strictly paired step-3000 BND ON/OFF source checkpoints.

The two parameterizations use their registered source workers, but share the
same seed, scene camera sequence, OCMC protocol and 3001 updates.  This
launcher is intentionally limited to the two TABA diagnostic scenes.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
OUT = ROOT / "outputs" / "taba_paired_source_20260910_v2"
JOBS = {
    "JapaneseGradens-RedSea_bnd_on": ("JapaneseGradens-RedSea", "8", "on"),
    "Panama_bnd_on": ("Panama", "9", "on"),
    "JapaneseGradens-RedSea_bnd_off": ("JapaneseGradens-RedSea", "6", "off"),
    "Panama_bnd_off": ("Panama", "7", "off"),
}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"non-empty output root: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = {}
    handles = {}
    for key, (scene, gpu, variant) in JOBS.items():
        target = OUT / ("bnd_on" if variant == "on" else "bnd_off") / scene
        target.mkdir(parents=True, exist_ok=True)
        log = (logs / f"{key}.log").open("w", encoding="utf8")
        handles[key] = log
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["CONDA_DEFAULT_ENV"] = "water_splatting"
        if variant == "on":
            command = [
                str(PYTHON), str(ROOT / "scripts/experiments/run_m1_raoc_causal_scene.py"),
                "--repo", str(ROOT), "--scene", scene, "--gpu", gpu,
                "--output-dir", str(target), "--max-steps", "3001",
                "--parameterization", "bounded_sh3",
            ]
        else:
            command = [
                str(PYTHON), str(ROOT / "scripts/experiments/run_legacy_ocmc_scene.py"),
                "--repo", str(ROOT), "--scene", scene, "--gpu", gpu,
                "--output-dir", str(target), "--final-step", "3000",
            ]
        processes[key] = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        print(f"launched {key} on GPU {gpu}", flush=True)
    status = {key: int(proc.wait()) for key, proc in processes.items()}
    for handle in handles.values():
        handle.close()
    (OUT / "worker_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps({"output_root": str(OUT), "status": status}, sort_keys=True), flush=True)
    if any(code != 0 for code in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
