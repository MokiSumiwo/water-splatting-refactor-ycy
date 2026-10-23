#!/usr/bin/env python3
"""Run strict FSB BND-ON continuation independently on all four scenes."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
GPUS = ("6", "7", "8", "9")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=4999)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degreewise", type=str, default="1,1,1")
    parser.add_argument("--schedule", choices=("constant", "smoothstep"), default="constant")
    parser.add_argument("--schedule-start", type=int, default=3000)
    parser.add_argument("--schedule-end", type=int, default=8000)
    parser.add_argument("--snapshots", default="5000,10000,14999")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    source_map = {
        "Curasao": ROOT / "outputs/bnd_on_cica_source_extra_20260914",
        "IUI3-RedSea": ROOT / "outputs/bnd_on_cica_source_extra_20260914",
        "JapaneseGradens-RedSea": ROOT / "outputs/taba_paired_source_20260910_v2/bnd_on",
        "Panama": ROOT / "outputs/taba_paired_source_20260910_v2/bnd_on",
    }
    config_map = {
        "Curasao": ROOT / "outputs/taba_chromatic_bnd_on_20260910/Curasao/E0/source_config.yml",
        "IUI3-RedSea": ROOT / "outputs/taba_chromatic_bnd_on_20260910/IUI3-RedSea/E0/source_config.yml",
        "JapaneseGradens-RedSea": ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml",
        "Panama": ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml",
    }
    processes = []
    logs = []
    for scene, gpu in zip(SCENES, GPUS):
        target = output / scene
        target.mkdir()
        log = (output / "logs" / f"{scene}.log").open("w")
        cmd = [
            str(PYTHON), str(ROOT / "scripts/experiments/run_fsb_worker.py"),
            "--scene", scene, "--gpu", gpu, "--output-dir", str(target),
            "--max-step", str(args.max_step), "--source-root", str(source_map[scene]),
            "--source-config", str(config_map[scene]),
            "--alpha", str(args.alpha),
            "--degreewise", args.degreewise,
            "--schedule", args.schedule, "--schedule-start", str(args.schedule_start),
            "--schedule-end", str(args.schedule_end), "--snapshots", args.snapshots,
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   CONDA_DEFAULT_ENV="water_splatting", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        processes.append((scene, subprocess.Popen(cmd, cwd=ROOT, env=env,
                                                  stdout=log, stderr=subprocess.STDOUT)))
        logs.append(log)
    status = {scene: proc.wait() for scene, proc in processes}
    for log in logs:
        log.close()
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if any(status.values()):
        raise SystemExit(1)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
