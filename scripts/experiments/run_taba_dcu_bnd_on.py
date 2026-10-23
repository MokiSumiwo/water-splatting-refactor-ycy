#!/usr/bin/env python3
"""Paired TABA-DCU continuation, preserving every registered checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_taba_bnd_on as original

DEFAULT_OUT = ROOT / "outputs/taba_dcu_bnd_on_20260910"
PRESETS = {
    "E0": ("disabled", .50, .15, .20, .50),
    "E1": ("chromatic", .50, .15, .20, .50),
    "E2": ("directional", .50, .15, .20, .0),
    "E3": ("directional_uncertainty", .50, .15, .20, .50),
    "E4": ("directional_uncertainty", .40, .20, .30, .50),
}
_original_configure = original.configure_worker


def configure_worker(variant, log_path=None):
    worker = _original_configure("E0", log_path)
    mode, mid, temperature, floor, uncertainty = PRESETS[variant]
    worker.EXPERIMENT = "TABA_DCU_BND_ON_" + variant
    worker.TABA_ENABLED = mode != "disabled"
    worker.TABA_MODE = mode
    worker.TABA_TRANSMISSION_MID = mid
    worker.TABA_TRANSMISSION_TEMPERATURE = temperature
    worker.TABA_CHROMA_FLOOR = floor
    worker.TABA_UNCERTAINTY_LAMBDA = uncertainty
    worker.TABA_WATER_AXIS_EPS = 1e-4
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
    parser.add_argument("--launch-remaining", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    original.check_environment()
    original.configure_worker = configure_worker
    if args.audit_source:
        original.source_audit(args.output_root)
    elif args.launch_remaining:
        jobs = [(scene, arm, gpu) for arm in ("E2", "E3", "E4")
                for scene, gpu in (("JapaneseGradens-RedSea", "6"), ("Panama", "8"))]
        original.launch(args, jobs=jobs, worker_script=Path(__file__).resolve(), audit_sources=False)
    elif args.scene and args.gpu:
        original.run_scene(args)
    else:
        parser.error("choose --audit-source or --scene and --gpu")


if __name__ == "__main__":
    main()
