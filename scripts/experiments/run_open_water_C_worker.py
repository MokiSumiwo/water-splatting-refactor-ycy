"""Paired C1/C2 runs from raw initialization with frozen train-only O masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as W
from water_splatting.open_water_C import OpenWaterC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=tuple(W.SCENE_GPUS), default="IUI3-RedSea")
    parser.add_argument("--arm", choices=("C1", "C2"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mask_dir is None:
        if args.scene != "IUI3-RedSea":
            parser.error("--mask-dir is required for a newly registered scene")
        args.mask_dir = ROOT / "research_runs/C_open_water_20261024/assets/IUI3-RedSea/masks"

    W.EXPERIMENT = "C_OPEN_WATER_NATIVE_TAIL_C1_C2"
    W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    W.RENDERER_BACKWARD_MODE = "corrected_v1"
    W.RESPONSIBILITY_ENABLED = False
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.ACTIVE_MDRR_OVERRIDE = False
    W.ACTIVE_CICA_OVERRIDE = True
    W.START_STEP = 0
    W.CICA_START_STEP = 3 if args.smoke else 10000
    W.CICA_REFRESH_STEPS = (3, 9) if args.smoke else (10000, 12000, 14000)
    W.ACTIVE_RENDER_ROOT = args.output / "renders"
    W._active_snapshots = lambda final: tuple(step for step in (10000, final) if step <= final)
    W.MIC._metric_images = lambda model, pred, gt: W._taba_rgb_metrics(pred, gt)
    original_route = W._route_renderer_backend

    def route(mode: str):
        result = original_route(mode)
        from scripts.experiments.geometry_projection_backend import install
        install()
        result["projection_backward"] = "clip_chain_rule_A_20261021"
        return result

    W._route_renderer_backend = route
    controllers = {}

    def hook(branch, outputs, batch, base_loss, step, output_dir, params, group_indices,
             camera_index, sequence_row):
        key = id(branch)
        if key not in controllers:
            controllers[key] = OpenWaterC(W, branch, output_dir, args.mask_dir.resolve(), args.arm, args.smoke, scene=args.scene)
        return controllers[key].apply(branch, outputs, batch, base_loss, step, output_dir,
                                      params, group_indices, camera_index, sequence_row)

    W.EXTRA_GRADIENT_HOOK = hook
    result = W.run(
        args.scene,
        "B",  # B is the native M1+OCMC+CICA arm with MDRR explicitly disabled.
        args.gpu,
        args.output.resolve(),
        max_step=20 if args.smoke else 15000,
        source_config=ROOT / "research_runs/bnd_on_baseline_scratch15k_20261021_v2" / args.scene / "source_config.yml",
        training_seed=42,
        raw_data_only=True,
    )
    for controller in controllers.values():
        controller.log.close()
    print(json.dumps({"scene": args.scene, "C_arm": args.arm, "smoke": args.smoke,
                      "final_step": result["final_step"], "training_wall_seconds": result["training_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
