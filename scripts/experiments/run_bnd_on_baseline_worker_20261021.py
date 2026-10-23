#!/usr/bin/env python3
"""One scene of the BND-ON M1+OCMC+CICA scratch baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=sorted(worker.SCENE_GPUS))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    args = parser.parse_args()

    worker.EXPERIMENT = "BND_ON_M1_OCMC_CICA_SCRATCH_15K"
    worker.START_STEP = 0
    worker.FINAL_STEP = 15000
    worker.SNAPSHOT_STEPS = (15000,)
    worker.ACTIVE_SNAPSHOT_STEPS = (15000,)
    worker.ACTIVE_COMPACT_OUTPUT = False
    worker.ACTIVE_SAVE_COMPACT_RENDERS = False
    worker.ACTIVE_RENDER_ROOT = args.output_dir.resolve() / "renders"
    worker.OCMC_ENABLED = True
    worker.OCMC_STRENGTH = 1.0
    worker.OCMC_STRENGTH_START = 1.0
    worker.OCMC_STRENGTH_RAMP_STEPS = 0
    worker.CICA_ENABLED = True
    worker.CICA_START_STEP = 10000
    worker.CICA_RAMP_STEPS = 0
    worker.CICA_REFRESH_STEPS = (10000, 12000, 14000)
    worker.CICA_DERIVATIVE_MODE = "legacy"
    worker.RENDERER_BACKWARD_MODE = "corrected_v1"
    worker.RESPONSIBILITY_ENABLED = False
    worker.AUXILIARY_APPEARANCE_ENABLED = False
    worker.BRASH_ENABLED = False
    worker.BVAM_ENABLED = False
    worker.RAW_DATA_ONLY = True
    worker.ACTIVE_CICA_OVERRIDE = True
    worker.ACTIVE_MDRR_OVERRIDE = False

    result = worker.run(
        scene=args.scene,
        arm="B",
        gpu=args.gpu,
        output_dir=args.output_dir.resolve(),
        max_step=15000,
        source_config=args.source_config.resolve(),
        parameterization="bounded_sh3",
        training_seed=42,
        compact_output=False,
        raw_data_only=True,
    )
    print(json_result(result), flush=True)


def json_result(result: dict) -> str:
    import json
    return json.dumps({
        "scene": result["scene"],
        "final_step": result["final_step"],
        "completed_updates": result["completed_updates"],
        "checkpoint_count": result["checkpoint_count"],
        "ocmc_enabled": result["ocmc_enabled"],
        "cica_enabled": result["cica_enabled"],
    }, sort_keys=True)


if __name__ == "__main__":
    main()
