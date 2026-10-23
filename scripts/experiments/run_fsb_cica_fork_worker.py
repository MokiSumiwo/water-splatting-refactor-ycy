#!/usr/bin/env python3
"""One retained pre-CICA prefix or continuation from its complete state."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_four_way_fsb_renderer_cica_worker import W, SOURCE_MAP, CONFIG_MAP


def configure(scene, renderer, cica):
    W.EXPERIMENT = "FSB_CICA_SHARED_PREFIX_REPEAT_20260916"
    W.ARMS = ("PREFIX", "C0a", "C1a", "C1b", "C0b")
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.AUXILIARY_APPEARANCE_STRENGTH = 0.
    W.OCMC_ENABLED = True
    W.OCMC_STRENGTH = W.OCMC_STRENGTH_START = 1.
    W.OCMC_STRENGTH_RAMP_STEPS = 0
    W.CICA_ENABLED = True
    W.ACTIVE_CICA_OVERRIDE = True
    W.ACTIVE_MDRR_OVERRIDE = False
    W.CICA_DERIVATIVE_MODE = cica
    W.RENDERER_BACKWARD_MODE = renderer
    W.BRASH_ENABLED = W.BVAM_ENABLED = False
    W.START_STEP, W.FINAL_STEP = 3000, 14999
    W.ACTIVE_SNAPSHOT_STEPS = (14999,)
    W.ACTIVE_CHECKPOINT_DIRECTORY = None
    W.ACTIVE_COMPACT_OUTPUT = False
    W.HEADROOM_ALPHA = 1.
    W.DEGREEWISE_HEADROOM_ALPHAS = (1., 1., 1.)
    W.HEADROOM_SCHEDULE = "constant"
    W.ACTIVE_SOURCE_ROOT = SOURCE_MAP[scene]
    W.ACTIVE_SOURCE_CONFIG = CONFIG_MAP[scene]
    W.ACTIVE_PARAMETERIZATION = "bounded_headroom_sh3"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, choices=tuple(SOURCE_MAP))
    p.add_argument("--gpu", required=True, choices=("6", "7", "8", "9"))
    p.add_argument("--renderer", choices=("legacy", "fixed"), required=True)
    p.add_argument("--arm", choices=("PREFIX", "C0a", "C0b", "C1a", "C1b"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fork", type=Path)
    p.add_argument("--max-step", type=int)
    args = p.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting" or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("worker requires water_splatting and its one assigned physical GPU")
    prefix = args.arm == "PREFIX"
    if prefix == (args.fork is not None):
        raise ValueError("PREFIX has no fork; continuation requires one")
    mode = "exact_fsb" if args.arm.startswith("C1") else "legacy"
    configure(args.scene, args.renderer, mode)
    output = args.output.resolve()
    W.ACTIVE_RENDER_ROOT = output / "renders"
    W.run(args.scene, args.arm, args.gpu, output,
          max_step=args.max_step or (9999 if prefix else 14999),
          source_root=SOURCE_MAP[args.scene], source_config=CONFIG_MAP[args.scene],
          parameterization="bounded_headroom_sh3", training_seed=42,
          compact_output=False, fork_checkpoint=args.fork.resolve() if args.fork else None)


if __name__ == "__main__":
    main()
