#!/usr/bin/env python3
"""Launch the four strict FSB renderer/CICA arms on physical GPUs 6--9.

One scene is pinned to each GPU; its four arms run sequentially so peak disk
and device usage remain bounded.  Every arm gets a fresh process and the same
step-3000 source checkpoint, seed, camera sequence and optimizer state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
GPUS = ("6", "7", "8", "9")
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-step", type=int, default=14999)
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    # Build the isolated fixed extension once before four workers are started.
    compile_env = dict(os.environ, CUDA_VISIBLE_DEVICES="6", CONDA_DEFAULT_ENV="water_splatting")
    compile_cmd = [str(PYTHON), "-c", (
        "from scripts.experiments import run_direct_mdrr_cica_scene as W; "
        "W._route_renderer_backend('fixed')"
    )]
    compile_log = (out / "logs" / "renderer_fixed_compile.log").open("w")
    compile_status = subprocess.run(compile_cmd, cwd=ROOT, env=compile_env,
                                    stdout=compile_log, stderr=subprocess.STDOUT).returncode
    compile_log.close()
    if compile_status:
        (out / "status.json").write_text(json.dumps({"compile": compile_status}, indent=2) + "\n")
        raise SystemExit(compile_status)

    processes = []
    for scene, gpu in zip(SCENES, GPUS):
        scene_dir = out / scene
        log_path = out / "logs" / f"{scene}.log"
        log = log_path.open("w")
        # This indirection keeps each scene's four arms sequential while all
        # four scene workers execute concurrently on GPUs 6--9.
        cmd = [str(PYTHON), str(ROOT / "scripts/experiments/run_four_way_fsb_renderer_cica_scene.py"),
               "--scene", scene, "--gpu", gpu, "--output-dir", str(scene_dir),
               "--max-step", str(args.max_step)]
        if args.keep_checkpoints:
            cmd.append("--keep-checkpoints")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   CONDA_DEFAULT_ENV="water_splatting", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   MAX_JOBS="1")
        processes.append((scene, subprocess.Popen(cmd, cwd=ROOT, env=env,
                                                  stdout=log, stderr=subprocess.STDOUT), log))
    statuses = {}
    for scene, proc, log in processes:
        statuses[scene] = proc.wait()
        log.close()
        (out / "status.json").write_text(json.dumps({**statuses, "compile": compile_status}, indent=2) + "\n")
    status = {**statuses, "compile": compile_status}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if any(value for value in status.values()) or len(statuses) != len(SCENES):
        raise SystemExit(1)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
