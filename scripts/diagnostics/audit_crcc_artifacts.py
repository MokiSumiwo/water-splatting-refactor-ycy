#!/usr/bin/env python3
"""Verify all CRCC images, checkpoint history and Adam alignment."""

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
from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, write_json
from water_splatting.crcc import CRCC_FIELDS, CRCC_STATS


def audit(root):
    result=audit_images(root,variants=tuple(PRESETS))
    summary=read_json(root/'summary.json')
    states=[]
    for run in summary['states']:
        path=root/run['scene']/run['variant']
        assert (path/'crcc_result.json').is_file()
        logs=[json.loads(line) for line in (path/'crcc_training.jsonl').read_text().splitlines()]
        require_finite(logs)
        assert logs[-1]['step']==14999
        assert all(r['P_J_gt_1']==r['P_intrinsic_rgb_gt_1']==r['crcc_state_mismatch']==0 for r in logs)
        training=read_csv(path/'training_summary_A2.csv')
        assert all(float(row.get('crcc_'+group+'_grad_l2') or 0)==0 for row in training
                   for group in ('means','scales','quats','opacities','medium_mlp','direction_encoding'))
        for step in (5000,10000,14999):
            rows=read_csv(path/f'crcc_statistics_step_{step:05d}.csv')
            assert all(key in row and row[key]!='' for row in rows if row['bucket']=='all' for key in CRCC_STATS)
            payload=torch.load(path/'checkpoints'/f'step-{step:09d}.ckpt',map_location='cpu',weights_only=False)
            state=payload['model']
            n=state['gauss_params.means'].shape[0]
            cfg=payload['metadata']['crcc_config']
            assert cfg['mode']==PRESETS[run['variant']]
            tracked=run['variant'] in ('E2','E3')
            assert cfg['enabled']==tracked
            assert not any(k.startswith(('_grac_state.','_gcrc_state.')) for k in state)
            keys=[k for k in state if k.startswith('_crcc_state.')]
            if tracked:
                assert len(keys)==len(CRCC_FIELDS)+1
                for key in CRCC_FIELDS:
                    value=state['_crcc_state.'+key]
                    assert value.shape==(n,3 if key=='crcc_chroma_ema' else 1)
                    assert not value.requires_grad and torch.isfinite(value).all()
                assert state['_crcc_state.last_update_step'].item()==step
                count=state['_crcc_state.crcc_observation_count']
                initialized=state['_crcc_state.crcc_initialized']
                chroma=state['_crcc_state.crcc_chroma_ema']
                assert count.dtype==torch.int64 and (count>=0).all() and (count<=step-3000).all()
                assert initialized.dtype==torch.bool and torch.equal(initialized,count>0)
                assert (chroma.abs()<=1).all() and (chroma.sum(-1).abs()<2e-6).all()
            else:
                assert not keys
            for group in ('means','features_dc','features_rest','opacities','scales','quats'):
                for value in payload['optimizers'][group]['state'].values():
                    for key in ('exp_avg','exp_avg_sq'):
                        if key in value:
                            assert value[key].shape==state['gauss_params.'+group].shape
            states.append(dict(scene=run['scene'],variant=run['variant'],step=step,gaussians=n,
                               history_present=tracked,passed=True))
            del payload,state
    result.update(crcc_state_and_optimizer_alignment_passed=True,crcc_checkpoint_states=states)
    write_json(root/'crcc_artifact_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    check_environment()
    result=audit(args.output_root.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='crcc_checkpoint_states'}),flush=True)
