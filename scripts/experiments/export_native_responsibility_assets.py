#!/usr/bin/env python3
"""Replay saved native models; export fixed-exposure comparisons and diagnostics."""
import argparse,json,sys,copy,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.native_responsibility_controller import load_event_asset

@torch.no_grad()
def export_arm(scene,arm,gpu,out,smoke=False):
    started=time.perf_counter()
    W.RAW_DATA_ONLY=True; W.RESPONSIBILITY_ENABLED=True; W.AUXILIARY_APPEARANCE_ENABLED=False
    W.ACTIVE_SOURCE_CONFIG=out/'source_config.yml'; W.ACTIVE_INITIALIZATION_DIR=out/'initialization'
    W.ACTIVE_TRAINING_SEED=0; W.RENDERER_BACKWARD_MODE='corrected_v1'
    # Read-only replay may use an idle GPU; training retains its fixed mapping.
    W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
    W._route_renderer_backend('corrected_v1'); W._runtime(scene,gpu); W._seed_all(0)
    branch=W._new_branch(scene); model=branch.pipeline.model
    train=W._train_records(branch); test=W._eval_records(branch)
    assets=out/'visual_assets'; assets.mkdir(exist_ok=True)
    boxes={}
    # History is rendered only, with no historical test-metric selection.
    for step in ([16] if smoke else [10000,12000,14000,15000]):
        W._load_snapshot(branch,scene,out/'checkpoints'/f'step-{step:09d}.ckpt')
        for _,view,camera,batch in test:
            outputs=model.get_outputs_for_camera(camera.to(model.device))
            folder=assets/'clear_history'/str(view); folder.mkdir(parents=True,exist_ok=True)
            W._save_native_rgb(folder/f'{step}.png',outputs['rgb_clear'].float().clamp(0,1))
            if step==10000:
                depth=outputs['depth'].detach().reshape(outputs['pred_image'].shape[:2]).float()
                gt=W._gt(model,W._batch_to_device(batch,model.device),outputs['background']).float().mean(-1)
                valid=torch.isfinite(depth)&(depth>0)
                edge=torch.zeros_like(gt)
                edge[1:]+=(gt[1:]-gt[:-1]).abs();edge[:,1:]+=(gt[:,1:]-gt[:,:-1]).abs()
                regions={'reliable_observed_edge':edge>torch.quantile(edge.reshape(-1),.85)}
                if valid.any():
                    regions.update(far=valid&(depth>=torch.quantile(depth[valid],.65)),
                                   near=valid&(depth<=torch.quantile(depth[valid],.35)))
                h,w=depth.shape;view_boxes={}
                for label,mask in regions.items():
                    yy,xx=torch.where(mask)
                    cy=int(yy.float().median()) if yy.numel() else h//2
                    cx=int(xx.float().median()) if xx.numel() else w//2
                    bw,bh=max(64,w//4),max(64,h//4)
                    x=max(0,min(w-bw,cx-bw//2));y=max(0,min(h-bh,cy-bh//2))
                    view_boxes[label]=[x,y,min(w,x+bw),min(h,y+bh)]
                boxes[str(view)]=view_boxes
    W._write_json(assets/'fixed_10k_crop_boxes.json',dict(rule='depth quantiles and observed edges at 10k; report uses B boxes for both arms',views=boxes))
    # A shared ordered train-camera sequence, explicitly not an interpolated path.
    for _,view,camera,batch in train:
        outputs=model.get_outputs_for_camera(camera.to(model.device))
        folder=assets/'camera_sequence'/str(view); folder.mkdir(parents=True,exist_ok=True)
        W._save_native_rgb(folder/'clear.png',outputs['rgb_clear'].float().clamp(0,1))
        W._save_native_rgb(folder/'underwater.png',outputs['pred_image'].float().clamp(0,1))
    events=[1] if smoke else [1,10,20]
    history=json.loads((out/'event_history.json').read_text())
    alarms=[r['event'] for r in history if r['alarms']]
    if alarms: events=sorted(set(events+[alarms[0]]))
    metrics=[]
    for event in events:
        old=load_event_asset(out/'events'/f'event_{event:02d}_S0.pt')
        capped=load_event_asset(out/'events'/f'event_{event:02d}_capped.pt')
        new=load_event_asset(out/'events'/f'event_{event:02d}_M1.pt')
        state=old['state']; steps=4 if smoke else 10000+200*(event-1)
        for split,ids in [('fit',state['fit_indices']),('monitor',state['monitor_indices'])]:
            for i in ids:
                _,view,camera,batch=train[i]; rendered={}; losses={}; native={}; clear={}
                for label,base,m1 in [('old_old',old,old),('old_new',old,new),('cap_old',capped,old),('cap_new',capped,new)]:
                    weights=dict(base['model'])
                    for key in weights:
                        if key.startswith(('medium_mlp.','direction_encoding.')): weights[key]=m1['model'][key]
                    model.load_state_dict(weights,strict=True); model.step=steps; model.eval()
                    outputs=model.get_outputs_for_camera(camera.to(model.device))
                    gt=W._gt(model,W._batch_to_device(batch,model.device),outputs['background']).float()
                    pred=outputs['pred_image'].float()
                    losses[label]=float((pred.clamp(0,1)-gt.clamp(0,1)).square().mean())
                    native[label]=float(sum(model.get_loss_dict(outputs,W._batch_to_device(batch,model.device),{}).values()))
                    rendered[label]=pred.clamp(0,1); clear[label]=outputs['rgb_clear'].float().clamp(0,1)
                D=losses['cap_old']-losses['old_old']
                K=(losses['cap_new']-losses['cap_old'])-(losses['old_new']-losses['old_old'])
                metrics.append(dict(scene=scene,arm=arm,event=event,split=split,view=str(view),loss='clamp01 RGB MSE',D=D,K=K,
                    cells=losses,native_loss_cells=native,
                    clear_m1_max_abs=float((clear['old_old']-clear['old_new']).abs().max())))
                # All fixed 6+2 cameras have four-grid assets, no monitored RGB is optimized here.
                folder=assets/'events'/f'{event:02d}'/split/str(view); folder.mkdir(parents=True,exist_ok=True)
                for kind,images in [('underwater',rendered),('clear',clear)]:
                    top=torch.cat([images['old_old'],images['old_new']],dim=1)
                    bottom=torch.cat([images['cap_old'],images['cap_new']],dim=1)
                    W._save_native_rgb(folder/f'{kind}_four_grid.png',torch.cat([top,bottom],dim=0))
    W._write_json(assets/'interaction.json',metrics)
    endpoint=torch.load(out/'checkpoints'/f"step-{16 if smoke else 15000:09d}.ckpt",map_location='cpu',weights_only=False)
    endpoint_state=endpoint['responsibility_state']
    W._write_json(assets/'endpoint_accounting.json',dict(
        optimizer_calls=endpoint_state['optimizer_calls'],
        rho_planned=endpoint_state['rho_planned'],rho_effective=endpoint_state['rho_effective'],
        paused=endpoint_state['paused'],event=endpoint_state['event_index'],
        phase=endpoint_state['phase']))
    W._write_json(assets/'display_protocol.json',dict(exposure='native, shared across arms',clamp=[0,1],resizing=False,
        four_grid='top old opacity/old M1 | old opacity/new M1; bottom capped opacity/old M1 | capped opacity/new M1',
        sequence='sorted existing training cameras, not interpolated novel path',history_metrics=False))
    W._write_json(assets/'export_costs.json',dict(wall_seconds=time.perf_counter()-started,
        optimizer_updates=0,includes='history, camera sequence, fixed event replay and asset I/O'))
    W._release(branch)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--arm',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    export_arm(a.scene,a.arm,a.gpu,a.output_dir.resolve())
if __name__=='__main__':main()
