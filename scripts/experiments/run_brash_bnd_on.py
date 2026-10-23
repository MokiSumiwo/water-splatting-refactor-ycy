#!/usr/bin/env python3
"""Run the isolated BRASH BND-ON E0-E3 continuation screening."""

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


WORKER.EXPERIMENT = "BRASH_BND_ON_CAPACITY_RECOVERY"
WORKER.ARMS = ("A2",)
WORKER.AUXILIARY_APPEARANCE_ENABLED = False
WORKER.AUXILIARY_APPEARANCE_STRENGTH = 0.0
WORKER.OCMC_ENABLED = True
WORKER.OCMC_STRENGTH = 1.0
WORKER.OCMC_STRENGTH_START = 1.0
WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
WORKER.CICA_ENABLED = True
WORKER.CICA_RAMP_STEPS = 0
WORKER.CICA_MIN_SUPPORT = 3
WORKER.ALLOWED_GPUS = frozenset(str(index) for index in range(10))
WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
WORKER.ACTIVE_SNAPSHOT_STEPS = (WORKER.FINAL_STEP,)


def _configure_variant(variant: str) -> None:
    WORKER.BRASH_ENABLED = variant != "E0_bnd_fixed"
    WORKER.BRASH_RESIDUAL_SCALE = 1.0
    WORKER.BRASH_GATE_INIT_LOGIT = 5.0
    WORKER.BRASH_GATE_MODE = "fixed_one" if variant == "E1_brash_fixed_gate" else "learned"
    WORKER.BRASH_GATE_PRIOR_ENABLED = variant == "E3_brash_reliability_prior"
    WORKER.BRASH_GATE_PRIOR_WEIGHT = 0.05 if WORKER.BRASH_GATE_PRIOR_ENABLED else 0.0
    WORKER.BRASH_MIN_SUPPORT = 3
    WORKER.BRASH_TRANSMISSION_FLOOR = 0.0
    WORKER.BRASH_DEPTH_TEMPERATURE = 1.0
    WORKER.BRASH_VIEW_SPREAD_SCALE = 1.0
    WORKER.BRASH_GATE_START_STEP = 10000
    WORKER.BRASH_GATE_RAMP_STEPS = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("E0_bnd_fixed", "E1_brash_fixed_gate", "E2_brash_learned_gate", "E3_brash_reliability_prior"),
        required=True,
    )
    parser.add_argument("--scene", choices=("JapaneseGradens-RedSea", "Panama"), required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compact-output", action="store_true")
    args = parser.parse_args()
    _configure_variant(args.variant)
    WORKER.ACTIVE_RENDER_ROOT = REPO_ROOT / "renders" / "brash_bnd_on_screening_20260909" / args.variant
    result: Any = WORKER.run(
        args.scene,
        "A2",
        args.gpu,
        args.output_dir.resolve(),
        int(args.max_step),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="bounded_sh3",
        training_seed=int(args.seed),
        compact_output=bool(args.compact_output),
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
