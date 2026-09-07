#!/usr/bin/env python3
"""Run Legacy-SH3 + OCMC + CICA with auxiliary appearance disabled.

This is an isolated A2 continuation experiment.  It starts from the common
Legacy + OCMC checkpoint at step 3000 and reuses the registered camera
sequence, while disabling the auxiliary appearance regularizer explicitly.
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


# Keep this matrix separate from the previous Auxiliary+CICA experiment.
WORKER.EXPERIMENT = "LEGACY_OCMC_CICA_NO_AUXILIARY_FOUR_SCENE"
WORKER.ARMS = ("A2",)
WORKER.AUXILIARY_APPEARANCE_ENABLED = False
WORKER.AUXILIARY_APPEARANCE_STRENGTH = 0.0
WORKER.ALLOWED_GPUS = frozenset(str(index) for index in range(10))
WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
WORKER.ACTIVE_RENDER_ROOT = REPO_ROOT / "renders" / "legacy_ocmc_cica_no_auxiliary_20260905"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(WORKER.SCENE_GPUS), required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compact-output", action="store_true")
    args = parser.parse_args()
    result: Any = WORKER.run(
        args.scene,
        "A2",
        args.gpu,
        args.output_dir.resolve(),
        int(args.max_step),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="legacy",
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
        training_seed=int(args.seed),
        compact_output=bool(args.compact_output),
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
