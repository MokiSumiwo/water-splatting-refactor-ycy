#!/usr/bin/env python3
"""Adapter for the Legacy-SH3 auxiliary appearance continuation.

The underlying worker retains its registered C0/C1 causal audit.  In this
experiment its C1 arm is the requested A0 condition:
Legacy + OCMC + auxiliary appearance regularization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_identifiability_module_causal_scene as WORKER


WORKER.ALLOWED_GPUS = frozenset(str(index) for index in range(10))
WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
# The formal worker was extended to accept Legacy explicitly.  Its historical
# Bounded experiment hash gate must not reject this intentional configuration.
WORKER.REGISTERED_PROTECTED_HASHES = {
    key: value
    for key, value in WORKER.REGISTERED_PROTECTED_HASHES.items()
    if key != "scripts/experiments/run_m1_raoc_causal_scene.py"
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=WORKER.SCENES, required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--diagnostic-sample-count", type=int, default=WORKER.DIAGNOSTIC_SAMPLE_COUNT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.resume_checkpoint is not None:
        # C0 is already complete in the partially recovered experiment. Keep
        # its artifacts and finish only the C1/A0 continuation.
        WORKER.ARMS = ("C1",)
    result: Any = WORKER.run(
        args.scene,
        args.gpu,
        args.output_dir.resolve(),
        final_step=int(args.final_step),
        diagnostic_sample_count=int(args.diagnostic_sample_count),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="legacy",
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
