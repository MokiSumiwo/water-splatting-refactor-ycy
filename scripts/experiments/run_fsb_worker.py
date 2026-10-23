#!/usr/bin/env python3
"""Strict FSB continuation worker.

FSB is the CICA+OCMC baseline using the bounded headroom SH3 appearance
parameterization.  This wrapper deliberately reuses the audited continuation
worker so that data ordering, optimizer state, and diagnostics stay identical
to the registered BND-ON protocol.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W

W.EXPERIMENT = "FSB_BND_ON_STRICT"
W.ARMS = ("A2",)
W.AUXILIARY_APPEARANCE_ENABLED = False
W.AUXILIARY_APPEARANCE_STRENGTH = 0.0
W.OCMC_ENABLED = True
W.OCMC_STRENGTH = 1.0
W.OCMC_STRENGTH_START = 1.0
W.OCMC_STRENGTH_RAMP_STEPS = 0
W.CICA_ENABLED = True
W.BRASH_ENABLED = False
W.BVAM_ENABLED = False
W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
W.START_STEP = 3000
W.FINAL_STEP = 14999
W.ACTIVE_SNAPSHOT_STEPS = (5000, 10000, 14999)
W.ACTIVE_COMPACT_OUTPUT = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=14999)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degreewise", type=str, default="1,1,1")
    parser.add_argument("--schedule", choices=("constant", "smoothstep"), default="constant")
    parser.add_argument("--schedule-start", type=int, default=3000)
    parser.add_argument("--schedule-end", type=int, default=8000)
    parser.add_argument("--snapshots", default="5000,10000,14999")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    W.HEADROOM_ALPHA = args.alpha
    W.HEADROOM_SCHEDULE = args.schedule
    W.HEADROOM_START_STEP = args.schedule_start
    W.HEADROOM_END_STEP = args.schedule_end
    from water_splatting.fields.gaussian_appearance import resolve_headroom_alpha
    resolve_headroom_alpha(3000, target=args.alpha, schedule=args.schedule,
                           start_step=args.schedule_start, end_step=args.schedule_end)
    W.ACTIVE_SNAPSHOT_STEPS = tuple(int(x) for x in args.snapshots.split(','))
    if args.schedule != "constant":
        W.EXPERIMENT = "FSB_CURRICULUM_SMOOTHSTEP_3000_8000"
    W.DEGREEWISE_HEADROOM_ALPHAS = tuple(float(x) for x in args.degreewise.split(','))
    if len(W.DEGREEWISE_HEADROOM_ALPHAS) != 3 or any(x < 0 or x > 1 for x in W.DEGREEWISE_HEADROOM_ALPHAS):
        raise ValueError("degreewise must contain three values in [0,1]")
    W.ACTIVE_RENDER_ROOT = ROOT / "renders" / args.output_dir.resolve().parent.name
    if args.max_step == W.FINAL_STEP and (W.ACTIVE_RENDER_ROOT / args.scene).exists():
        raise FileExistsError(f"Refusing to overwrite renders: {W.ACTIVE_RENDER_ROOT / args.scene}")
    W.run(
        args.scene,
        "A2",
        args.gpu,
        args.output_dir.resolve(),
        args.max_step,
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization="bounded_headroom_sh3",
        training_seed=42,
        compact_output=False,
    )


if __name__ == "__main__":
    main()
