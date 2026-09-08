#!/usr/bin/env python3
"""Matched BND-OFF optimization arms for M1+OCMC and CICA.

The wrapper changes only schedules belonging to the two claimed innovation
lines.  It keeps the historical legacy-SH3, Auxiliary-OFF protocol and starts
each arm from the same OCMC C0@3000 checkpoint.
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


WORKER.EXPERIMENT = "LEGACY_OCMC_CICA_BND_OFF_SINGLE_FACTOR_FOLLOWUP"
WORKER.ARMS = ("A2",)
WORKER.AUXILIARY_APPEARANCE_ENABLED = False
WORKER.AUXILIARY_APPEARANCE_STRENGTH = 0.0
WORKER.ALLOWED_GPUS = frozenset(str(index) for index in range(10))
WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
WORKER.ACTIVE_SNAPSHOT_STEPS = (WORKER.FINAL_STEP,)


def _configure_variant(variant: str) -> None:
    # All variants are A2 continuations: MDRR is excluded from this matrix.
    if variant == "E0_fixed":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 1.0
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "E1_ocmc_off":
        WORKER.OCMC_ENABLED = False
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 1.0
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "E2_cica_off":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 1.0
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = False
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "E3_ocmc_constant_075":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 0.75
        WORKER.OCMC_STRENGTH_START = 0.75
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "E4_ocmc_constant_090":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 0.90
        WORKER.OCMC_STRENGTH_START = 0.90
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "E5_cica_support4":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 1.0
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 4
    elif variant == "ocmc_only_schedule":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 0.75
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 2000
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 0
        WORKER.CICA_MIN_SUPPORT = 3
    elif variant == "cica_only_schedule":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 1.0
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 2000
        WORKER.CICA_MIN_SUPPORT = 4
    elif variant == "combined_schedule":
        WORKER.OCMC_ENABLED = True
        WORKER.OCMC_STRENGTH = 1.0
        WORKER.OCMC_STRENGTH_START = 0.75
        WORKER.OCMC_STRENGTH_RAMP_STEPS = 2000
        WORKER.CICA_ENABLED = True
        WORKER.CICA_RAMP_STEPS = 2000
        WORKER.CICA_MIN_SUPPORT = 4
        WORKER.CICA_GRADIENT_FRACTION = 0.10
    else:
        raise ValueError(f"unknown variant: {variant}")


def _validate_variant(variant: str) -> None:
    if not 0.0 <= float(WORKER.OCMC_STRENGTH_START) <= 1.0:
        raise ValueError("OCMC start strength must be in [0, 1]")
    if not 0.0 <= float(WORKER.OCMC_STRENGTH) <= 1.0:
        raise ValueError("OCMC target strength must be in [0, 1]")
    if not isinstance(WORKER.OCMC_ENABLED, bool):
        raise ValueError("OCMC enabled flag must be boolean")
    if int(WORKER.OCMC_STRENGTH_RAMP_STEPS) < 0 or int(WORKER.CICA_RAMP_STEPS) < 0:
        raise ValueError("schedule lengths must be non-negative")
    if int(WORKER.CICA_MIN_SUPPORT) < 1 or int(WORKER.CICA_MIN_SUPPORT) > int(WORKER.CICA_BANK_MAX):
        raise ValueError("CICA minimum support must fit within the camera bank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(WORKER.SCENE_GPUS), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "E0_fixed",
            "E1_ocmc_off",
            "E2_cica_off",
            "E3_ocmc_constant_075",
            "E4_ocmc_constant_090",
            "E5_cica_support4",
            "ocmc_only_schedule",
            "cica_only_schedule",
            "combined_schedule",
        ),
        required=True,
    )
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compact-output", action="store_true")
    args = parser.parse_args()
    _configure_variant(args.variant)
    _validate_variant(args.variant)
    # Keep render assets for different arms disjoint.  The worker appends the
    # scene name below this root, so each process can run independently.
    WORKER.ACTIVE_RENDER_ROOT = (
        REPO_ROOT / "renders" / "m1_ocmc_cica_single_factor_20260908" / args.variant
    )
    result: Any = WORKER.run(
        args.scene,
        "A2",
        args.gpu,
        args.output_dir.resolve(),
        int(args.max_step),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="legacy",
        training_seed=int(args.seed),
        compact_output=bool(args.compact_output),
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
