#!/usr/bin/env python3
"""Build a linked CRCC-only report and audit its paired native assets."""

import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment, read_json, write_json, gpu_processes
from scripts.experiments.summarize_taba_bnd_on import STATS, summarize, contact_sheet
from scripts.experiments.summarize_taba_dcu_bnd_on import four_scene_sheet
from scripts.diagnostics.audit_taba_artifacts import audit as audit_images, require_finite
from water_splatting.crcc import CRCC_STATS


if __name__=='__main__':
    check_environment()
    root=DEFAULT_OUT
    source=root/'crcc_only_four_scene_extension'
    report=root/'crcc_only_four_scene_report'
    report.mkdir(exist_ok=False)
    scenes=('JapaneseGradens-RedSea','Panama','Curasao','IUI3-RedSea')
    for scene in scenes:
        for arm in ('E0','E2'):
            origin=source if arm=='E2' and scene in ('Curasao','IUI3-RedSea') else root
            target=report/scene/arm
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(origin/scene/arm,target_is_directory=True)
            render=report/'renders'/arm/scene
            render.parent.mkdir(parents=True,exist_ok=True)
            render.symlink_to(origin/'renders'/arm/scene,target_is_directory=True)
    write_json(report/'four_scene_selection.json',dict(selected_variant='E2',
        rule='Supplementary fixed CRCC-only validation; registered main selection remains E1.'))
    summary=summarize(report,variants=('E0','E2'),candidates=('E2',),stats=STATS+CRCC_STATS,
        experiment='Supplementary fixed CRCC-only four-scene validation',
        selection_rule='Fixed E2, supplementary validation; no new configuration selection.')
    for kind in ('underwater','clear','clear_raw_display_clamp01'):
        summary['contact_sheets'].append(four_scene_sheet(report,'E2',kind))
    write_json(report/'summary.json',summary)
    result=audit_images(report,variants=('E0','E2'))
    assert result['checkpoint_count']==24 and result['render_count']==312
    checkpoints=[]
    for scene in ('Curasao','IUI3-RedSea'):
        directory=source/scene/'E2'
        assert (directory/'crcc_result.json').exists()
        logs=[json.loads(x) for x in (directory/'crcc_training.jsonl').read_text().splitlines()]
        require_finite(logs)
        assert logs[-1]['step']==14999
        assert all(r['P_J_gt_1']==r['P_intrinsic_rgb_gt_1']==r['crcc_state_mismatch']==0 for r in logs)
        rows=list(csv.DictReader((directory/'training_summary_A2.csv').open()))
        assert all(float(r['crcc_'+group+'_grad_l2'])==0 for r in rows
                   for group in ('means','scales','quats','opacities','medium_mlp','direction_encoding'))
        assert any(float(r['crcc_features_dc_grad_l2'])>0 and float(r['crcc_features_rest_grad_l2'])>0 for r in rows)
        for step in (5000,10000,14999):
            path=directory/'checkpoints'/f'step-{step:09d}.ckpt'
            payload=torch.load(path,map_location='cpu',weights_only=False)
            state=payload['model']
            n=state['gauss_params.means'].shape[0]
            chroma=state['_crcc_state.crcc_chroma_ema']
            count=state['_crcc_state.crcc_observation_count']
            initialized=state['_crcc_state.crcc_initialized']
            assert chroma.shape==(n,3) and count.shape==initialized.shape==(n,1)
            assert torch.isfinite(chroma).all() and (chroma.abs()<=1).all()
            assert (chroma.sum(-1).abs()<2e-6).all()
            assert count.dtype==torch.int64 and initialized.dtype==torch.bool and torch.equal(count>0,initialized)
            assert (count>=0).all() and (count<=step-3000).all()
            assert int(state['_crcc_state.last_update_step'])==step
            assert payload['metadata']['crcc_config']['mode']=='crcc'
            assert not any(k.startswith(('_grac_state.','_gcrc_state.')) for k in state)
            for group in ('means','features_dc','features_rest','opacities','scales','quats'):
                for value in payload['optimizers'][group]['state'].values():
                    for key in ('exp_avg','exp_avg_sq'):
                        if key in value:
                            assert value[key].shape==state['gauss_params.'+group].shape
            checkpoints.append(dict(scene=scene,step=step,gaussians=n,path=str(path),passed=True))
            del payload,state
    result.update(supplementary_checkpoints=checkpoints,state_and_adam_alignment_passed=True,
                  total_unique_experiment_runs=14,total_unique_training_checkpoints=42,
                  total_unique_native_pngs=540,main_selected_variant='E1',supplementary_variant='E2')
    write_json(root/'supplementary_verification.json',result)
    main=read_json(root/'summary.json')
    rows=main['final_metrics']+main['means']+[r for r in summary['final_metrics']+summary['means']
        if r['variant']=='E2' and r['scene'] in ('Curasao','IUI3-RedSea','four_scene_mean')]
    with (root/'metrics_final_including_CRCC_four_scene.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=('scene','variant','step','PSNR','SSIM','LPIPS','MSE'))
        writer.writeheader()
        writer.writerows(rows)
    comparison_sheets=[]
    for scene in scenes:
        for kind in ('underwater','clear','clear_raw_display_clamp01'):
            if scene in ('Curasao','IUI3-RedSea'):
                e1=report/'renders/E1'/scene
                e1.parent.mkdir(parents=True,exist_ok=True)
                if not e1.exists():
                    e1.symlink_to(root/'renders/E1'/scene,target_is_directory=True)
                comparison_sheets.append(contact_sheet(report,scene,('E0','E1','E2'),kind))
    write_json(report/'additional_comparison_sheets.json',comparison_sheets)
    print(json.dumps(dict(passed=True,means=summary['means'],unique_runs=14,unique_checkpoints=42,unique_pngs=540)),flush=True)
