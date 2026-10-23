#!/usr/bin/env python3
"""Check that the BRASH implementation is inert when disabled.

The audit loads the same registered BND-ON start checkpoint twice with the
same seed.  Both passes use the existing bounded-SH path; the second pass
also carries the BRASH configuration fields, but keeps ``brash_enabled``
false.  Exact tensor hashes are compared for the forward outputs and the
Gaussian colors.  This is an E0a/E0b implementation-equivalence check, not a
training experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


OUTPUT_KEYS = (
    "pred_image",
    "rgb_object",
    "rgb_medium_finite",
    "rgb_tail",
    "gaussian_view_rgb",
    "gaussian_view_logits",
    "gaussian_view_dc_rgb",
    "gaussian_view_dc_logits",
    "gaussian_sh_residual",
)


def _configure_disabled(*, explicit_brash_config: bool) -> None:
    WORKER.BRASH_ENABLED = False
    WORKER.BRASH_RESIDUAL_SCALE = 0.5 if explicit_brash_config else 1.0
    WORKER.BRASH_GATE_INIT_LOGIT = 2.0 if explicit_brash_config else 5.0
    WORKER.BRASH_GATE_MODE = "learned" if explicit_brash_config else "fixed_one"
    WORKER.BRASH_GATE_PRIOR_ENABLED = bool(explicit_brash_config)
    WORKER.BRASH_GATE_PRIOR_WEIGHT = 0.05 if explicit_brash_config else 0.0
    WORKER.BRASH_MIN_SUPPORT = 4 if explicit_brash_config else 3
    WORKER.BRASH_TRANSMISSION_FLOOR = 0.2 if explicit_brash_config else 0.0
    WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    WORKER.ACTIVE_PARAMETERIZATION = "bounded_sh3"
    WORKER.OCMC_ENABLED = True
    WORKER.OCMC_STRENGTH = 1.0
    WORKER.OCMC_STRENGTH_START = 1.0
    WORKER.OCMC_STRENGTH_RAMP_STEPS = 0


def _render_pass(scene: str, source_root: Path, source_config: Path, seed: int) -> dict[str, Any]:
    WORKER.ACTIVE_SOURCE_ROOT = source_root.resolve()
    WORKER.ACTIVE_SOURCE_CONFIG = source_config.resolve()
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
        hashes = {
            key: WORKER._hash_tensor(outputs[key])
            for key in OUTPUT_KEYS
            if isinstance(outputs.get(key), torch.Tensor)
        }
        return {
            "view_id": str(view_id),
            "hashes": hashes,
            "finite": bool(WORKER.FORMAL._finite_outputs(outputs)),
            "brash_enabled": bool(getattr(branch.pipeline.model.config, "brash_enabled", False)),
            "intrinsic_color_parameterization": str(branch.pipeline.model.config.intrinsic_color_parameterization),
        }
    finally:
        WORKER._release(branch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(sorted(WORKER.SCENE_GPUS)), required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(WORKER.ALLOWED_GPUS)), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_disabled(explicit_brash_config=False)
    WORKER._runtime(args.scene, args.gpu)
    first = _render_pass(args.scene, args.source_root, args.source_config, args.seed)
    _configure_disabled(explicit_brash_config=True)
    second = _render_pass(args.scene, args.source_root, args.source_config, args.seed)
    same = first["hashes"] == second["hashes"]
    result: Mapping[str, Any] = {
        "scene": args.scene,
        "seed": int(args.seed),
        "E0a_current_bnd_on": first,
        "E0b_brash_config_disabled": second,
        "exact_hash_match": same,
        "E0_DISABLED_EQUIVALENCE": bool(same and first["finite"] and second["finite"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(WORKER._sanitize(result), indent=2, sort_keys=True))
    if not result["E0_DISABLED_EQUIVALENCE"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
