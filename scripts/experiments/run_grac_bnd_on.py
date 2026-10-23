#!/usr/bin/env python3
"""Registered GRAC screen and paired continuation on physical GPUs 6-9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_taba_bnd_on as original
from water_splatting.grac import GRAC_CONFIG

DEFAULT_OUT = ROOT / "outputs/grac_bnd_on_20260911"
PRESETS = {"E0": "disabled", "E1": "grac", "E2": "taba", "E3": "grac_taba"}
_original_configure = original.configure_worker


def configure_worker(variant, log_path=None):
    worker = _original_configure("E0", log_path)
    mode = PRESETS[variant]
    worker.EXPERIMENT = "GRAC_BND_ON_" + variant
    worker.GRAC_EXPERIMENT = True
    worker.GRAC_OPTIONS = dict(GRAC_CONFIG, enabled=variant != "E0", mode=mode)
    worker.GRAC_LOG_PATH = str(log_path.parent / "grac_training.jsonl") if log_path is not None else None
    worker.TABA_ENABLED = mode in ("taba", "grac_taba")
    worker.TABA_MODE = "directional" if worker.TABA_ENABLED else "disabled"
    worker.TABA_UNCERTAINTY_LAMBDA = 0.0
    return worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=original.SCENES + original.EXTENDED_SCENES)
    parser.add_argument("--variant", choices=tuple(PRESETS), default="E0")
    parser.add_argument("--gpu", choices=original.GPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-step", type=int, default=14999)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--launch-screen", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    original.check_environment()
    original.configure_worker = configure_worker
    if args.audit_source:
        original.source_audit(args.output_root)
    elif args.launch_screen:
        jobs = [(scene, arm, gpu) for arm in PRESETS
                for scene, gpu in (("JapaneseGradens-RedSea", "6"), ("Panama", "8"))]
        original.launch(args, jobs=jobs, worker_script=Path(__file__).resolve(), audit_sources=False)
    elif args.scene and args.gpu:
        original.run_scene(args)
        target = args.output_root / args.scene / args.variant
        original.write_json(target / "grac_result.json", original.read_json(target / "taba_result.json"))
    else:
        parser.error("choose --audit-source, --launch-screen, or --scene and --gpu")


if __name__ == "__main__":
    main()
