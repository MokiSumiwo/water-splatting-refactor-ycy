#!/usr/bin/env python3
"""Run all four arms sequentially for one scene on one pinned GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-step", type=int, default=14999)
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    statuses = {}
    for arm in ARMS:
        arm_dir = root / arm
        cmd = [str(PYTHON), str(ROOT / "scripts/experiments/run_four_way_fsb_renderer_cica_worker.py"),
               "--scene", args.scene, "--gpu", args.gpu, "--arm", arm,
               "--output-dir", str(arm_dir), "--max-step", str(args.max_step)]
        # Checkpoint cleanup is required by default.  It occurs only after
        # evaluation and never touches historical source assets.
        if not args.keep_checkpoints:
            cmd.append("--compact-output")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                   CONDA_DEFAULT_ENV="water_splatting", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   MAX_JOBS="1")
        log_path = root.parent / "logs" / f"{args.scene}_{arm}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            statuses[arm] = subprocess.run(cmd, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT).returncode
        if statuses[arm] != 0:
            break
    (root / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if any(statuses.values()) or len(statuses) != len(ARMS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
