#!/usr/bin/env python3
"""Continue the paired step-3000 sources to the complete TABA baseline."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SOURCE = ROOT / "outputs" / "taba_paired_source_20260910_v2"
OUT = ROOT / "outputs" / "taba_paired_baseline_20260910_strict_v4"
CONFIG = ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"
JOBS = {
    "JapaneseGradens-RedSea_bnd_on": ("JapaneseGradens-RedSea", "8", "on"),
    "Panama_bnd_on": ("Panama", "9", "on"),
    "JapaneseGradens-RedSea_bnd_off": ("JapaneseGradens-RedSea", "6", "off"),
    "Panama_bnd_off": ("Panama", "7", "off"),
}


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"non-empty output root: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = {}
    handles = {}
    for key, (scene, gpu, variant) in JOBS.items():
        source_root = SOURCE / ("bnd_on" if variant == "on" else "bnd_off")
        source_ckpt = source_root / scene / "checkpoints" / "C0" / "step-000003000.ckpt"
        if not source_ckpt.is_file():
            raise FileNotFoundError(source_ckpt)
        target = OUT / scene / ("bnd_on" if variant == "on" else "bnd_off")
        target.mkdir(parents=True, exist_ok=True)
        log = (logs / f"{key}.log").open("w", encoding="utf8")
        handles[key] = log
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["CONDA_DEFAULT_ENV"] = "water_splatting"
        command = [
            str(PYTHON), str(ROOT / "scripts/experiments/run_taba_paired_baseline.py"),
            "--scene", scene, "--variant", ("bnd_on" if variant == "on" else "bnd_off"), "--gpu", gpu,
            "--output-dir", str(target), "--source-root", str(source_root),
            "--source-config", str(CONFIG), "--seed", "42",
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
