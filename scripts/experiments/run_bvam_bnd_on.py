#!/usr/bin/env python3
"""Run the isolated BVAM BND-ON E0/E2/E3/E4 screening."""

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


WORKER.EXPERIMENT = "BVAM_BND_ON_BOUNDED_APPEARANCE"
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
    WORKER.BRASH_ENABLED = variant == "E2_brash_fixed_gate"
    WORKER.BRASH_RESIDUAL_SCALE = 1.0
    WORKER.BRASH_GATE_INIT_LOGIT = 5.0
    WORKER.BRASH_GATE_MODE = "fixed_one"
    WORKER.BRASH_GATE_PRIOR_ENABLED = False
    WORKER.BRASH_GATE_PRIOR_WEIGHT = 0.0

    WORKER.BVAM_ENABLED = variant in ("E3_bvam_k2", "E4_bvam_pseudogt")
    WORKER.BVAM_NUM_EXPERTS = 2
    WORKER.BVAM_DELTA_SCALE = 1.0
    WORKER.BVAM_GATE_DEGREE = 1
    WORKER.BVAM_GATE_TEMPERATURE = 1.0
    WORKER.BVAM_MIN_SUPPORT = 3
    WORKER.BVAM_MIN_VIEW_SPREAD = 0.01
    WORKER.BVAM_PSEUDOGT_ENABLED = variant == "E4_bvam_pseudogt"
    WORKER.BVAM_PSEUDOGT_TEMPERATURE = 0.1
    WORKER.BVAM_PSEUDOGT_MIN_CONFIDENCE = 0.1
    WORKER.BVAM_GATE_LOSS_WEIGHT = 0.01 if WORKER.BVAM_PSEUDOGT_ENABLED else 0.0
    WORKER.BVAM_DIVERSITY_LOSS_WEIGHT = 1e-4 if WORKER.BVAM_ENABLED else 0.0
    WORKER.BVAM_GATE_ENTROPY_WEIGHT = 0.0
    WORKER.BVAM_DELTA_REGULARIZER_WEIGHT = 1e-4 if WORKER.BVAM_ENABLED else 0.0
    WORKER.BVAM_START_STEP = 10000
    WORKER.BVAM_RAMP_STEPS = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("E0_bnd_fixed", "E2_brash_fixed_gate", "E3_bvam_k2", "E4_bvam_pseudogt"),
        required=True,
    )
    parser.add_argument("--scene", choices=("JapaneseGradens-RedSea", "Panama"), required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=WORKER.FINAL_STEP)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _configure_variant(args.variant)
    WORKER.ACTIVE_RENDER_ROOT = REPO_ROOT / "renders" / "bvam_bnd_on_screening_20260909" / args.variant
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
        compact_output=False,
    )
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
