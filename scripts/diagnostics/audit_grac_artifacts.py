#!/usr/bin/env python3
"""Verify every GRAC checkpoint, history tensor, native render and metric pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.diagnostics.audit_taba_artifacts import audit as audit_images, read_json, read_csv, require_finite
from scripts.experiments.run_grac_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, write_json
from water_splatting.grac import GRAC_FIELDS, GRAC_STATS


def audit(root):
    result = audit_images(root, variants=tuple(PRESETS))
    summary = read_json(root / "summary.json")
    states = []
    for run in summary["states"]:
        path = root / run["scene"] / run["variant"]
        assert (path / "grac_result.json").is_file()
        logs = [json.loads(line) for line in (path / "grac_training.jsonl").read_text().splitlines()]
        require_finite(logs)
        assert logs[-1]["step"] == 14999
        assert all(r["P_J_gt_1"] == r["P_intrinsic_rgb_gt_1"] == 0 for r in logs)
        for step in (5000,10000,14999):
            snapshots = read_csv(path / f"grac_statistics_step_{step:05d}.csv")
            assert all(key in row and row[key] != "" for row in snapshots if row["bucket"] == "all" for key in GRAC_STATS)
            payload = torch.load(path / "checkpoints" / f"step-{step:09d}.ckpt", map_location="cpu", weights_only=False)
            state = payload["model"]
            n = state["gauss_params.means"].shape[0]
            cfg = payload["metadata"]["grac_config"]
            assert cfg["mode"] == PRESETS[run["variant"]]
            tracked = cfg["mode"] in ("grac", "grac_taba")
            keys = [k for k in state if k.startswith("_grac_state.")]
            if tracked:
                assert len(keys) == len(GRAC_FIELDS)+1
                for key in GRAC_FIELDS:
                    value = state["_grac_state."+key]
                    assert value.shape == (n,3 if key == "direction_prototype" else 1)
                    assert not value.requires_grad and torch.isfinite(value).all()
                assert state["_grac_state.last_update_step"].item() == step
                count = state["_grac_state.visibility_count"]
                mcount = state["_grac_state.magnitude_count"]
                assert count.dtype == mcount.dtype == torch.int64
                assert (count >= 0).all() and (count <= step-3000).all()
                assert (mcount >= 0).all() and (mcount <= count).all()
                assert (state["_grac_state.magnitude_ema"] >= 0).all()
                assert (state["_grac_state.magnitude_sq_ema"] >= 0).all()
            else:
                assert not keys
            for group in ("means","features_dc","features_rest","opacities","scales","quats"):
                optimizer = payload["optimizers"][group]
                for value in optimizer["state"].values():
                    for name in ("exp_avg", "exp_avg_sq"):
                        if name in value:
                            assert value[name].shape == state["gauss_params."+group].shape
            states.append({"scene":run["scene"], "variant":run["variant"], "step":step,
                           "gaussians":n,"history_present":tracked,"passed":True})
            del payload, state
    result.update({"grac_state_and_optimizer_alignment_passed":True,"grac_checkpoint_states":states})
    write_json(root / "grac_artifact_audit.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    args = p.parse_args()
    check_environment()
    result = audit(args.output_root.resolve())
    print(json.dumps({k:v for k,v in result.items() if k != "grac_checkpoint_states"}),flush=True)


if __name__ == "__main__":
    main()
