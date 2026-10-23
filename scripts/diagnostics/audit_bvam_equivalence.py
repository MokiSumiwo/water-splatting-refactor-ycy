#!/usr/bin/env python3
"""Audit BND, BRASH-E1, and zero-initialized BVAM at one real camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


def _configure(*, brash: bool, bvam: bool) -> None:
    WORKER.BRASH_ENABLED = brash
    WORKER.BRASH_GATE_MODE = "fixed_one"
    WORKER.BVAM_ENABLED = bvam
    WORKER.BVAM_NUM_EXPERTS = 2
    WORKER.BVAM_DELTA_SCALE = 1.0
    WORKER.BVAM_GATE_DEGREE = 1
    WORKER.BVAM_GATE_TEMPERATURE = 1.0
    WORKER.BVAM_PSEUDOGT_ENABLED = False
    WORKER.BVAM_GATE_LOSS_WEIGHT = 0.0
    WORKER.BVAM_DIVERSITY_LOSS_WEIGHT = 0.0
    WORKER.BVAM_GATE_ENTROPY_WEIGHT = 0.0
    WORKER.BVAM_DELTA_REGULARIZER_WEIGHT = 0.0
    WORKER.OCMC_ENABLED = True
    WORKER.OCMC_STRENGTH = 1.0
    WORKER.OCMC_STRENGTH_START = 1.0
    WORKER.OCMC_STRENGTH_RAMP_STEPS = 0
    WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False


def _forward(scene: str, source_root: Path, source_config: Path, seed: int) -> Dict[str, Any]:
    WORKER.ACTIVE_SOURCE_ROOT = source_root.resolve()
    WORKER.ACTIVE_SOURCE_CONFIG = source_config.resolve()
    WORKER.ACTIVE_PARAMETERIZATION = "bounded_sh3"
    WORKER.ACTIVE_TRAINING_SEED = int(seed)
    WORKER._seed_all(int(seed))
    branch = None
    try:
        branch = WORKER._new_branch(scene)
        WORKER._load_source(branch, scene)
        records = WORKER._eval_records(branch)
        _index, view_id, camera, _batch = records[0]
        with torch.no_grad():
            outputs = branch.pipeline.model.get_outputs_for_camera(camera.to(branch.pipeline.model.device))
        names = ("pred_image", "rgb_object", "rgb_medium_finite", "rgb_tail", "gaussian_view_rgb", "rgb_clear", "clear_object_fullsh_raw")
        return {
            "view_id": str(view_id),
            "tensors": {name: outputs[name].detach().float().cpu() for name in names if isinstance(outputs.get(name), torch.Tensor)},
            "finite": bool(WORKER.FORMAL._finite_outputs(outputs)),
            "bvam_gate": outputs.get("bvam_gate_probs").detach().float().cpu().tolist() if isinstance(outputs.get("bvam_gate_probs"), torch.Tensor) else None,
        }
    finally:
        WORKER._release(branch)


def _compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    rows = {}
    for name in sorted(set(reference["tensors"]) & set(candidate["tensors"])):
        difference = (reference["tensors"][name] - candidate["tensors"][name]).abs()
        rows[name] = {"max_abs": float(difference.max().item()), "mean_abs": float(difference.mean().item())}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="JapaneseGradens-RedSea")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure(brash=False, bvam=False)
    bnd = _forward(args.scene, args.source_root, args.source_config, args.seed)
    _configure(brash=True, bvam=False)
    brash = _forward(args.scene, args.source_root, args.source_config, args.seed)
    _configure(brash=False, bvam=True)
    bvam = _forward(args.scene, args.source_root, args.source_config, args.seed)
    result = {
        "scene": args.scene,
        "seed": int(args.seed),
        "bnd_vs_brash_e1": _compare(bnd, brash),
        "bnd_vs_bvam_zero": _compare(bnd, bvam),
        "bnd_finite": bnd["finite"],
        "brash_finite": brash["finite"],
        "bvam_finite": bvam["finite"],
        "bvam_gate_uniform": bvam["bvam_gate"],
        "BRASH_E1_RENDER_EQUIVALENCE": all(row["max_abs"] <= 1e-6 for row in _compare(bnd, brash).values()),
        "BVAM_ZERO_RENDER_EQUIVALENCE": all(row["max_abs"] <= 1e-6 for row in _compare(bnd, bvam).values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True))
    if not result["BRASH_E1_RENDER_EQUIVALENCE"] or not result["BVAM_ZERO_RENDER_EQUIVALENCE"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
