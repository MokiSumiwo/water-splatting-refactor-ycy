#!/usr/bin/env python3
"""Train matched BND-OFF/BND-ON continuation baselines for TABA diagnostics."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W

W.EXPERIMENT = "TABA_PAIRED_BASELINE"
W.ARMS = ("A2",)
W.AUXILIARY_APPEARANCE_ENABLED = False
W.AUXILIARY_APPEARANCE_STRENGTH = 0.0
W.OCMC_ENABLED = True
W.OCMC_STRENGTH = 1.0
W.OCMC_STRENGTH_START = 1.0
W.OCMC_STRENGTH_RAMP_STEPS = 0
W.CICA_ENABLED = True
W.CICA_RAMP_STEPS = 0
W.CICA_MIN_SUPPORT = 3
W.BRASH_ENABLED = False
W.BVAM_ENABLED = False
W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
W.ACTIVE_SNAPSHOT_STEPS = (5000, 10000, 14999)
W.ACTIVE_COMPACT_OUTPUT = False
W.TABA_DIAGNOSTIC_ENABLED = True
W.START_STEP = 3000

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=("JapaneseGradens-RedSea", "Panama"), required=True)
    p.add_argument("--variant", choices=("bnd_off", "bnd_on"), required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-config", type=Path, required=True)
    args = p.parse_args()
    if args.variant == "bnd_on":
        source_root = args.source_root.resolve()
        parameterization = "bounded_sh3"
    else:
        source_root = args.source_root.resolve()
        parameterization = "legacy"
    source_config = args.source_config.resolve()
    W.ACTIVE_RENDER_ROOT = ROOT / "renders" / "taba_paired_baseline_20260910" / args.variant
    result = W.run(
        args.scene, "A2", args.gpu, args.output_dir.resolve(),
        W.FINAL_STEP, source_root=source_root, source_config=source_config,
        parameterization=parameterization, training_seed=args.seed,
        compact_output=False,
    )
    print(json.dumps(W._sanitize(result), indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
