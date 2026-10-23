#!/usr/bin/env python3
"""Re-evaluate retained BVAM BND-ON checkpoints with restored support state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


def _configure(variant: str) -> None:
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
    WORKER.BRASH_ENABLED = False
    WORKER.BVAM_ENABLED = True
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
    WORKER.BVAM_DIVERSITY_LOSS_WEIGHT = 1e-4
    WORKER.BVAM_GATE_ENTROPY_WEIGHT = 0.0
    WORKER.BVAM_DELTA_REGULARIZER_WEIGHT = 1e-4
    WORKER.BVAM_START_STEP = 10000
    WORKER.BVAM_RAMP_STEPS = 2000
    WORKER.ACTIVE_SNAPSHOT_STEPS = (WORKER.FINAL_STEP,)
    WORKER.ACTIVE_COMPACT_OUTPUT = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("E3_bvam_k2", "E4_bvam_pseudogt"), required=True)
    parser.add_argument("--scene", choices=("JapaneseGradens-RedSea", "Panama"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    args = parser.parse_args()
    _configure(args.variant)
    WORKER.ACTIVE_SOURCE_ROOT = args.source_root.resolve()
    WORKER.ACTIVE_SOURCE_CONFIG = args.source_config.resolve()
    WORKER.ACTIVE_PARAMETERIZATION = "bounded_sh3"
    WORKER.ACTIVE_RENDER_ROOT = REPO_ROOT / "renders" / "bvam_bnd_on_screening_20260909"
    result = WORKER._evaluate_arm(args.scene, "A2", args.output_dir.resolve(), WORKER.ACTIVE_RENDER_ROOT / args.variant / args.scene)
    payload = {"scene": args.scene, "variant": args.variant, "evaluation": result}
    (args.output_dir / "evaluation_restore_audit.json").write_text(
        json.dumps(WORKER._sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    print(json.dumps(WORKER._sanitize(payload), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
