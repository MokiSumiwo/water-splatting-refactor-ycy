#!/usr/bin/env python3
"""Bounded, zero-update native branch audits on existing smoke assets."""
import argparse, copy, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.native_responsibility_controller import ResponsibilityController as C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--gpu',required=True)
    a=p.parse_args(); root=a.run_dir.resolve()
    W.RAW_DATA_ONLY=True; W.RESPONSIBILITY_ENABLED=True; W.RESPONSIBILITY_SMOKE=True
    W.ACTIVE_SOURCE_CONFIG=root/'source_config.yml'; W.ACTIVE_INITIALIZATION_DIR=root/'initialization'
    W.ACTIVE_TRAINING_SEED=0; W.AUXILIARY_APPEARANCE_ENABLED=False
    W.RENDERER_BACKWARD_MODE='corrected_v1'; W._route_renderer_backend('corrected_v1'); W._runtime(a.scene,a.gpu)
    W._seed_all(0); branch=W._new_branch(a.scene); W._load_source(branch,a.scene)
    payload=torch.load(root/'checkpoints/step-000000006.ckpt',map_location='cpu',weights_only=False)
    W._restore_training_checkpoint(branch,payload,True)
    m=branch.pipeline.model; records=W._train_records(branch)
    c=C(W,branch,records,payload['arm'],root,True,payload['responsibility_state'])
    i=c.state['fit_indices'][0]; camera=records[i][2].to(m.device)
    assets={s:torch.load(root/'events'/('event_01_'+s+'.pt'),map_location='cpu',weights_only=False) for s in ['S0','capped','M1']}
    gaussian_keys=[k for k in assets['capped']['model'] if k.startswith('gauss_params.')]
    frozen=all(torch.equal(assets['capped']['model'][k],assets['M1']['model'][k]) for k in gaussian_keys)
    m.train(); m.step_cb(6)
    with torch.no_grad():
        renders={}
        for s in ['S0','capped','M1']:
            m.load_state_dict(assets[s]['model'],strict=True)
            out=m.get_outputs(camera)
            renders[s]={k:out[k].detach().float().clone() for k in ['pred_image','rgb_clear']}
        # Four-grid cross intervention: keep capped Gaussian, switch only M1.
        hybrid=dict(assets['capped']['model'])
        for k in hybrid:
            if k.startswith(('medium_mlp.','direction_encoding.')): hybrid[k]=assets['M1']['model'][k]
        m.load_state_dict(hybrid,strict=True); out=m.get_outputs(camera)
        m1_clear=float((out['rgb_clear'].float()-renders['capped']['rgb_clear']).abs().max())
        intervention={k:float((renders['S0'][k]-renders['capped'][k]).abs().max()) for k in renders['S0']}
    W._restore_training_checkpoint(branch,payload,True)
    m=branch.pipeline.model; c.model=m; c.state=C.to_device(c.state,m.device)
    m.train(); m.step_cb(6)
    batch=W._batch_to_device(records[i][3].copy(),m.device)
    baseline_outputs=m.get_outputs(camera)
    frozen_denominator=baseline_outputs['pred_image'].detach()+1e-3
    actual=sum(m.get_loss_dict(baseline_outputs,batch,{}).values())+c.inner_loss(baseline_outputs,i)[0]
    def loss():
        outputs=m.get_outputs(camera)
        terms=m.get_loss_dict(outputs,batch,{})
        # The native objective intentionally stop-gradients its relative
        # photometric normalizer. Finite differences must keep that node fixed.
        if m.config.photometric_normalization_mode=='relative_pred_detached':
            gt=m.composite_with_background(m.get_gt_img(batch['image']),outputs['background'])
            pred=outputs['pred_image']; den=frozen_denominator
            if 'mask' in batch: raise RuntimeError('audit needs explicit masked denominator handling')
            if m.config.main_loss=='reg_l1': recon=((gt-pred)/den).abs().mean()
            elif m.config.main_loss=='l1': recon=(gt-pred).abs().mean()
            else: recon=((pred-gt)/den).square().mean()
            x,y=(gt,pred) if m.config.ssim_loss=='ssim' else (gt/den,pred/den)
            sim=1-m.ssim(x.permute(2,0,1)[None],y.permute(2,0,1)[None])
            terms['main_loss']=(1-m.config.ssim_lambda)*recon+m.config.ssim_lambda*sim
        native=sum(terms.values())
        return native+c.inner_loss(outputs,i)[0]
    value=loss(); params=list(m.medium_mlp.parameters())
    grads=torch.autograd.grad(value,params,allow_unused=True)
    actual_grads=torch.autograd.grad(actual,params,allow_unused=True)
    surrogate_matches=all(torch.allclose(g,h,atol=1e-6,rtol=1e-5) for g,h in zip(grads,actual_grads) if g is not None)
    assert surrogate_matches and torch.allclose(value,actual,atol=1e-6,rtol=1e-5)
    choices=[(float(g.detach().abs().max()),j,int(g.detach().abs().argmax())) for j,g in enumerate(grads) if g is not None]
    _,j,index=max(choices); param=params[j]; derivative=float(grads[j].reshape(-1)[index])
    original=param.detach().clone(); finite=[]
    for epsilon in [.002,.001]:
        values=[]; positions=[]
        for sign in [1,-1]:
            with torch.no_grad():
                param.copy_(original); param.reshape(-1)[index]+=sign*epsilon
                positions.append(float(param.reshape(-1)[index]))
            values.append(float(loss().detach()))
        numeric=(values[0]-values[1])/(positions[0]-positions[1])
        finite.append(dict(epsilon=epsilon,analytic=derivative,numeric=numeric,
            relative_error=abs(numeric-derivative)/max(abs(numeric),abs(derivative),1e-8),
            wrong_sign_relative_error=abs(numeric+derivative)/max(abs(numeric),abs(derivative),1e-8)))
    with torch.no_grad(): param.copy_(original)
    gradient_pass=any(r['relative_error']<.15 and r['wrong_sign_relative_error']>1.5 for r in finite)
    # Exercise the real alarm branch on a disposable controller copy, no updates.
    saved=c.state; c.state=copy.deepcopy(saved)
    c.state.update(event_pre=dict(psnr=20.,roi_mse=.01,coverage=.9,edge_error=.1),
        post_inner=dict(psnr=19.,roi_mse=.02,coverage=.8,edge_error=.2),
        inner_alarms=['psnr','roi_mse','structure'],rho_effective=.0375,rho_last_good=0.)
    c.monitor=lambda:dict(psnr=19.,roi_mse=.02,coverage=.8,edge_error=.2)
    c.save_assets=lambda *args,**kwargs:None
    c.output=root/'disposable_alarm_audit'; c.output.mkdir(exist_ok=True)
    model_hash=W._model_state_hash(m); opt_hash=W._optimizer_hash(branch.optimizers)
    c.after(16)
    alarm_pass=c.state['paused'] and c.state['rho_effective']==0 and model_hash==W._model_state_hash(m) and opt_hash==W._optimizer_hash(branch.optimizers)
    c.state=saved
    # Controlled native cull on disposable loaded parameters, sync all cap IDs.
    medium={k:W._hash_object(v.state_dict()) for k,v in branch.optimizers.optimizers.items() if k in W._medium_optimizer_groups(branch.optimizers)}
    old_ids=c.state['cap']['gaussian_ids'].clone()
    n=m.num_points; deleted=torch.zeros(n,dtype=torch.bool,device=m.device); deleted[-1]=True
    for name,param in list(m.gauss_params.items()): m.gauss_params[name]=torch.nn.Parameter(param[~deleted])
    m.remove_from_all_optim(branch.optimizers,deleted)
    c.synchronize_cull(deleted)
    cull_pass=torch.equal(c.state['cap']['gaussian_ids'],old_ids[~deleted]) and all(W._hash_object(branch.optimizers.optimizers[k].state_dict())==v for k,v in medium.items())
    result=dict(optimizer_updates=0,gaussians_frozen_inner=frozen,intervention_max_abs=intervention,
        m1_only_clear_max_abs=m1_clear,gradient=finite,gradient_pass=gradient_pass,
        detached_normalizer_fixed=True,surrogate_matches_actual_gradient=surrogate_matches,
        alarm_pass=alarm_pass,cull_pass=bool(cull_pass),
        passed=frozen and m1_clear<=1e-6 and gradient_pass and alarm_pass and bool(cull_pass))
    if payload['arm']=='N' and c.state.get('coverage',0)>0:
        result['passed']=result['passed'] and all(v>1e-7 for v in intervention.values())
    W._write_json(root/'native_branch_audit.json',result); print(result,flush=True)
    assert result['passed'], 'native smoke audit failed; see native_branch_audit.json'

if __name__=='__main__': main()
