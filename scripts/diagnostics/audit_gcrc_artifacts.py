#!/usr/bin/env python3
"""Verify GCRC run assets, state lineage shapes and optimizer alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch

from scripts.diagnostics.audit_taba_artifacts import audit as audit_images, read_json, read_csv, require_finite
from scripts.experiments.run_gcrc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, write_json
from water_splatting.gcrc import GCRC_FIELDS, GCRC_STATS


def audit(root):
    result=audit_images(root,variants=tuple(PRESETS),exact_fallback_variants=("E2","E3"))
    summary=read_json(root/"summary.json")
    states=[]
    for run in summary["states"]:
        path=root/run["scene"]/run["variant"]
        assert (path/"gcrc_result.json").is_file()
        logs=[json.loads(line) for line in (path/"gcrc_training.jsonl").read_text().splitlines()]
        require_finite(logs)
        assert logs[-1]["step"]==14999
        assert all(r["P_J_gt_1"]==r["P_intrinsic_rgb_gt_1"]==0 for r in logs)
        for step in (5000,10000,14999):
            snapshots=read_csv(path/f"gcrc_statistics_step_{step:05d}.csv")
            assert all(key in row and row[key]!="" for row in snapshots if row["bucket"]=="all" for key in GCRC_STATS)
            payload=torch.load(path/"checkpoints"/f"step-{step:09d}.ckpt",map_location="cpu",weights_only=False)
            state=payload["model"]
            n=state["gauss_params.means"].shape[0]
            cfg=payload["metadata"]["gcrc_config"]
            assert cfg["mode"]==PRESETS[run["variant"]]
            tracked=cfg["mode"] in ("gcrc","gcrc_taba")
            keys=[k for k in state if k.startswith("_gcrc_state.")]
            assert not any(k.startswith("_grac_state.") for k in state)
            if tracked:
                assert len(keys)==len(GCRC_FIELDS)+1
                for key in GCRC_FIELDS:
                    value=state["_gcrc_state."+key]
                    assert value.shape==(n,3 if key.endswith("prototype") else 1)
                    assert not value.requires_grad and torch.isfinite(value).all()
                assert state["_gcrc_state.last_update_step"].item()==step
                count=state["_gcrc_state.visibility_count"]
                assert count.dtype==torch.int64 and (count>=0).all() and (count<=step-3000).all()
                for prefix in ("water","residual"):
                    initialized=state[f"_gcrc_state.{prefix}_direction_initialized"]
                    assert initialized.dtype==torch.bool
                    prototype=state[f"_gcrc_state.{prefix}_direction_prototype"]
                    norm=prototype[initialized.flatten()].norm(dim=-1)
                    assert torch.allclose(norm,torch.ones_like(norm),atol=2e-5,rtol=0)
            else:
                assert not keys
            for group in ("means","features_dc","features_rest","opacities","scales","quats"):
                for value in payload["optimizers"][group]["state"].values():
                    for key in ("exp_avg","exp_avg_sq"):
                        if key in value:
                            assert value[key].shape==state["gauss_params."+group].shape
            states.append({"scene":run["scene"],"variant":run["variant"],"step":step,"gaussians":n,"history_present":tracked,"passed":True})
            del payload,state
    result.update({"gcrc_state_and_optimizer_alignment_passed":True,"gcrc_checkpoint_states":states})
    write_json(root/"gcrc_artifact_audit.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    check_environment()
    result=audit(args.output_root.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!="gcrc_checkpoint_states"}),flush=True)


if __name__=="__main__":
    main()
