#!/usr/bin/env python3
"""Run real-checkpoint and CICA/OCMC audits for all TABA-DCU presets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics import audit_taba_training as existing
from scripts.experiments.run_taba_dcu_bnd_on import DEFAULT_OUT, configure_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=existing.SCENES, required=True)
    parser.add_argument("--gpu", choices=existing.GPUS, required=True)
    args = parser.parse_args()
    existing.check_environment()
    if existing.gpu_processes()[args.gpu]:
        raise RuntimeError("assigned GPU is occupied")
    existing.configure_worker = configure_worker
    result = existing.audit(args.scene, args.gpu, DEFAULT_OUT / "engineering" / args.scene,
                            variants=("E1", "E2", "E3", "E4"))
    print(json.dumps({"scene": args.scene, "passed": result["passed"]}), flush=True)


if __name__ == "__main__":
    main()
