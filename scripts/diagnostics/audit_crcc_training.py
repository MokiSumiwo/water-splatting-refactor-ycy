#!/usr/bin/env python3
"""Real-source CRCC equivalence, state lifecycle and gradient audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT, configure_worker
from scripts.experiments.run_taba_bnd_on import GPUS, SCENES, PAIRED, check_environment, gpu_processes, source_assets, read_json, write_json


def audit(scene,gpu,output):
    worker = configure_worker("E0")
    root,config,_ = source_assets(scene)
    worker.ACTIVE_SOURCE_ROOT,worker.ACTIVE_SOURCE_CONFIG = root,config
    worker._runtime(scene,gpu)
    worker._seed_all(42)
    branch = worker._new_branch(scene)
    result = dict(scene=scene,gpu=gpu,checks={},variants={})
    checks = result['checks']
    try:
        payload = worker._load_source(branch,scene)
        model = branch.pipeline.model
        original_optimizer = worker._optimizer_hash(branch.optimizers)
        checks['old_checkpoint_disabled_exact'] = worker._model_state_hash(model)==worker._hash_object(payload['model'])
        historical = read_json(PAIRED/scene/'bnd_on/start_state_equivalence.json')['forward_probe']
        checks['historical_disabled_forward_hash'] = worker._forward_probe(model,branch.pipeline.datamanager)['tensor_sha256']==historical['tensor_sha256']
        records = worker._train_records(branch)
        _,_,camera,batch = records[0]
        camera = camera.to(model.device)
        batch = worker._batch_to_device(batch.copy(),model.device)
        def forward():
            model.train()
            model.step_cb(5000)
            model.zero_grad(set_to_none=True)
            out = model.get_outputs(camera)
            return out,model.get_loss_dict(out,batch,{})
        def snap(out,losses):
            keys = ('pred_image','rgb_object','rgb_medium','rgb_clear','depth','J_gaussian_raw','gaussian_view_rgb')
            return ({k:worker._hash_tensor(out[k]) for k in keys},{k:float(v.detach()) for k,v in losses.items()})
        baseline,losses = forward()
        baseline_snapshot = snap(baseline,losses)
        sum(losses.values()).backward()
        baseline_grads = {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
        branch.optimizers.optimizer_step_all()
        baseline_updated = (worker._model_state_hash(model),worker._optimizer_hash(branch.optimizers))
        del baseline,losses
        worker._load_source(branch,scene)
        model.config.crcc_enabled = True
        model.config.crcc_mode = 'disabled'
        with patch.object(model,'_ensure_crcc_state',side_effect=AssertionError('disabled CRCC accessed state')):
            out,losses = forward()
        checks['explicit_disabled_forward_loss_exact'] = snap(out,losses)==baseline_snapshot
        sum(losses.values()).backward()
        disabled_grads = {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
        branch.optimizers.optimizer_step_all()
        result['independent_CUDA_backward_optimizer_hash_exact'] = (worker._model_state_hash(model),worker._optimizer_hash(branch.optimizers))==baseline_updated
        del out,losses
        worker._load_source(branch,scene)
        model.zero_grad(set_to_none=True)
        for key,param in model.named_parameters():
            param.grad = baseline_grads[key].clone() if key in baseline_grads else None
        branch.optimizers.optimizer_step_all()
        checks['disabled_optimizer_same_gradient_update_exact'] = (worker._model_state_hash(model),worker._optimizer_hash(branch.optimizers))==baseline_updated
        worker._load_source(branch,scene)
        model.config.crcc_enabled = False
        out,losses = forward()
        sum(losses.values()).backward()
        repeated_grads = {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
        result['CUDA_backward_differences'] = {
            key:dict(disabled_max_abs=float((baseline_grads[key]-disabled_grads[key]).abs().max()),
                     repeated_baseline_max_abs=float((baseline_grads[key]-repeated_grads[key]).abs().max()))
            for key in baseline_grads if baseline_grads[key].numel()}
        checks['disabled_backward_within_atomic_tolerance'] = all(torch.allclose(baseline_grads[k],disabled_grads[k],atol=5e-5,rtol=1e-4) for k in baseline_grads)
        checks['repeated_baseline_within_same_tolerance'] = all(torch.allclose(baseline_grads[k],repeated_grads[k],atol=5e-5,rtol=1e-4) for k in baseline_grads)
        del out,losses,baseline_grads,disabled_grads,repeated_grads
        for arm in ('E1','E2','E3'):
            configure_worker(arm)
            worker._configure_taba(model.config)
            worker._load_source(branch,scene)
            vc = {}
            state = model._crcc_state
            vc['optimizer_exact'] = worker._optimizer_hash(branch.optimizers)==original_optimizer
            stripped = {k:v for k,v in model.state_dict().items() if not k.startswith('_crcc_state.')}
            vc['source_parameters_exact'] = worker._hash_object(stripped)==worker._hash_object(payload['model'])
            if state is not None:
                vc['cold_history_zero'] = not bool(state.crcc_observation_count.any()) and not bool(state.crcc_initialized.any())
                out,losses = forward()
                vc['cold_loss_zero'] = losses['crcc_loss'].item()==0
                del out,losses
                model.config.crcc_diagnostic_only = True
                out,losses = forward()
                vc['diagnostic_forward_loss_exact'] = snap(out,losses)==baseline_snapshot
                del out,losses
                model.config.crcc_diagnostic_only = False
                for step in range(3001,3009):
                    model.step_cb(step)
                    model.eval()
                    with torch.no_grad():
                        out = model.get_outputs(records[(step-3001)%len(records)][2].to(model.device))
                    state.update(out['crcc_chroma'],out['gaussian_visible_mask'],out['crcc_update_valid'],step=step)
                    del out
                path = output/arm/'state_roundtrip.ckpt'
                path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(model.state_dict(),path)
                saved = worker._hash_object(state.state_dict())
                model.load_state_dict(torch.load(path,map_location=model.device,weights_only=False),strict=True)
                state = model._crcc_state
                vc['history_roundtrip_exact'] = worker._hash_object(state.state_dict())==saved
                damaged = model.state_dict()
                damaged.pop('_crcc_state.crcc_initialized')
                try:
                    model.load_state_dict(damaged,strict=True)
                    vc['partial_history_rejected'] = False
                except RuntimeError:
                    vc['partial_history_rejected'] = True
                model.load_state_dict(torch.load(path,map_location=model.device,weights_only=False),strict=True)
                state = model._crcc_state
                before = worker._hash_object(state.state_dict())
                model.eval()
                with torch.no_grad():
                    model.get_outputs(camera)
                vc['eval_history_frozen'] = worker._hash_object(state.state_dict())==before
            before = worker._hash_object(state.state_dict()) if state is not None else None
            out,losses = forward()
            vc['strict_intrinsic_bounds'] = bool(((out['gaussian_view_rgb']>0)&(out['gaussian_view_rgb']<1)).all())
            vc['raw_J_bounded'] = bool(((out['J_gaussian_raw']>=0)&(out['J_gaussian_raw']<=1)).all())
            vc['finite_outputs'] = worker.FORMAL._finite_outputs(out)
            if state is not None:
                vc['forward_history_frozen'] = worker._hash_object(state.state_dict())==before
                vc['warmed_nonzero_loss'] = losses['crcc_loss'].item()>0 and bool(out['crcc_active'].any())
                vc['statistics_detached'] = all(not v.requires_grad for k,v in out.items() if k.startswith('crcc_') and k!='crcc_loss')
                params,indices = worker._parameter_entries(model)
                grads = torch.autograd.grad(losses['crcc_loss'],params,retain_graph=True,allow_unused=True)
                norms = worker._group_norms(grads,indices)
                allowed = set(indices.get('features_dc',[])+indices.get('features_rest',[]))
                vc['CRCC_appearance_only'] = worker._nonzero_group_norm(grads,allowed)==0
                vc['CRCC_updates_dc_rest'] = norms['features_dc']>0 and norms['features_rest']>0
                sum(losses.values()).backward()
                vc['backward_history_frozen'] = worker._hash_object(state.state_dict())==before
                count = state.crcc_observation_count.clone()
                valid = out['crcc_update_valid'] & out['gaussian_visible_mask'].reshape(-1,1)
                model.after_train(5000)
                vc['after_train_once'] = torch.equal(state.crcc_observation_count,count+valid.long())
                after = worker._hash_object(state.state_dict())
                model.after_train(5000)
                vc['repeat_callback_no_update'] = worker._hash_object(state.state_dict())==after
                result['variants'][arm] = dict(checks=vc,CRCC_gradient_norms=norms,raw_loss=float(out['crcc_loss']))
                del grads
            else:
                vc['TABA_directional_without_CRCC'] = model.config.taba_mode=='directional' and 'crcc_loss' not in losses
                result['variants'][arm] = dict(checks=vc)
            del out,losses
            worker._load_source(branch,scene)
            bank = worker._camera_bank(branch,output/arm)
            partner = worker._partner_mapping(branch,output/arm)
            routing = worker._gradient_routing_audit(branch,scene,'A2',output/arm,records,partner,bank)
            result['variants'][arm]['routing'] = routing
        result['passed'] = all(checks.values()) and all(all(v['checks'].values()) and v['routing']['GRADIENT_ROUTING_AUDIT'] for v in result['variants'].values())
        write_json(output/'audit.json',result)
        if not result['passed']:
            raise RuntimeError('CRCC engineering audit failed: '+str(output/'audit.json'))
        return result
    finally:
        worker._release(branch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--gpu',choices=GPUS,required=True)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUT/'engineering/verified')
    args = p.parse_args()
    check_environment()
    if gpu_processes()[args.gpu]:
        raise RuntimeError('assigned GPU is occupied')
    result = audit(args.scene,args.gpu,args.output_root/args.scene)
    print(json.dumps(dict(scene=args.scene,passed=result['passed'])),flush=True)


if __name__=='__main__':
    main()
