#!/usr/bin/env python3
"""Run one no-Bounded-SH3 auxiliary/CICA continuation arm.

This adapter keeps the registered MDRR/CICA worker unchanged for its original
A1/A2/A3 experiment, while exposing the requested no-Bounded chain:

    A0 = Legacy + OCMC + auxiliary appearance regularization
    A2 = A0 + CICA

Both arms restore the same Legacy+OCMC C0 checkpoint at step 3000.
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

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


OUTPUT_ROOT = REPO_ROOT / "outputs" / "legacy_no_bounded_three_improvements_20260905"
RENDER_ROOT = REPO_ROOT / "renders" / "legacy_no_bounded_three_improvements_20260905"

# The historical worker uses A1/A2/A3 and fixed GPU-to-scene assignment.
# This adapter changes only the experiment matrix and scheduling policy.
WORKER.ARMS = ("A0", "A2")
WORKER.ALLOWED_GPUS = frozenset(str(index) for index in range(10))
WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
WORKER.ACTIVE_RENDER_ROOT = RENDER_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(WORKER.SCENE_GPUS), required=True)
    parser.add_argument("--arm", choices=WORKER.ARMS, required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    result: Any = WORKER.run(
        args.scene,
        args.arm,
        args.gpu,
        args.output_dir.resolve(),
        int(args.max_step),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="legacy",
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
