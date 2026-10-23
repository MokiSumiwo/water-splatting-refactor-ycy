#!/usr/bin/env python3
"""Registered CRCC E0-E3 continuations on physical GPUs 6-9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_taba_bnd_on as original
from water_splatting.crcc import CRCC_CONFIG
from water_splatting.gcrc import GCRC_CONFIG
from water_splatting.grac import GRAC_CONFIG

DEFAULT_OUT = ROOT / "outputs/crcc_bnd_on_20260911"
PRESETS = {"E0": "disabled", "E1": "taba", "E2": "crcc", "E3": "crcc_taba"}
_original_configure = original.configure_worker


def configure_worker(variant, log_path=None):
    worker = _original_configure("E0", log_path)
    mode = PRESETS[variant]
    worker.EXPERIMENT = "CRCC_BND_ON_" + variant
    worker.CRCC_EXPERIMENT = True
    worker.CRCC_OPTIONS = dict(CRCC_CONFIG, enabled=variant in ("E2", "E3"), mode=mode)
    worker.CRCC_LOG_PATH = str(log_path.parent / "crcc_training.jsonl") if log_path is not None else None
    worker.GRAC_EXPERIMENT = worker.GCRC_EXPERIMENT = False
    worker.GRAC_OPTIONS, worker.GCRC_OPTIONS = dict(GRAC_CONFIG), dict(GCRC_CONFIG)
    worker.GRAC_LOG_PATH = worker.GCRC_LOG_PATH = None
    worker.TABA_ENABLED = mode in ("taba", "crcc_taba")
    worker.TABA_MODE = "directional" if worker.TABA_ENABLED else "disabled"
    worker.TABA_UNCERTAINTY_LAMBDA = 0.
    return worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=original.SCENES+original.EXTENDED_SCENES)
    p.add_argument("--variant", choices=tuple(PRESETS), default="E0")
    p.add_argument("--gpu", choices=original.GPUS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-step", type=int, default=14999)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--audit-source", action="store_true")
    args = p.parse_args()
    args.output_root = args.output_root.resolve()
    original.check_environment()
    original.configure_worker = configure_worker
    if args.audit_source:
        original.source_audit(args.output_root)
    elif args.scene and args.gpu:
        original.run_scene(args)
        target = args.output_root/args.scene/args.variant
        original.write_json(target/"crcc_result.json", original.read_json(target/"taba_result.json"))
    else:
        p.error("choose --audit-source or --scene and --gpu")


if __name__ == "__main__":
    main()
