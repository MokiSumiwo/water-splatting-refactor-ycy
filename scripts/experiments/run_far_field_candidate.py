#!/usr/bin/env python3
"""Paired direct continuation for far-field candidates; no diagnostic admission gate."""
from __future__ import annotations
import argparse,csv,json,os,sys,time,gc
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.run_fsb_cica_fork_worker import configure,W
from water_splatting.far_field_recovery import CrossDistanceRadiance

SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
SOURCE=ROOT/'outputs/fsb_cica_fork_repeat_20260916_v1'
CDRC=ROOT/'outputs/cdrc_cross_distance_diagnostic_20260917_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--arm',choices=('BASE_A','CDRC','BASE_B'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args()
    assert os.environ.get('CONDA_DEFAULT_ENV')=='water_splatting'
    assert sys.executable=='/opt/anaconda3/envs/water_splatting/bin/python'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+SCENES.index(args.scene))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    configure(args.scene,'legacy','legacy');W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
    W.EXPERIMENT='FAR_FIELD_DIRECT_20260917';W._route_renderer_backend('legacy')
    W.ARMS=('BASE_A','CDRC','BASE_B')
    checkpoint=SOURCE/args.scene/'R0/C0a/checkpoints/step-000014999.ckpt'
    chash=W._sha256(checkpoint);branch=W._new_branch(args.scene)
    try:
        source=W._load_source(branch,args.scene)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        restored=W._restore_training_checkpoint(branch,payload,True)
        W._write_json(out/'restore_audit.json',restored)
        model=branch.pipeline.model;initial_hash=W._model_state_hash(model)
        records=W._train_records(branch);bank=W._camera_bank(branch,out)
        bank_records=[records[i] for i in bank['camera_indices']]
        module=CrossDistanceRadiance(CDRC/args.scene/'correspondences.csv',records,model.device) if args.arm=='CDRC' else None
        lambda_cica=float(payload['metadata']['resolved_lambda_CICA'])
        # Data order is common, independent of module execution and global RNG.
        rng=np.random.RandomState(20260917)
        order=[]
        while len(order)<args.steps:order.extend(rng.permutation(len(records)).tolist())
        order=order[:args.steps]
        sequence=[dict(update=i+1,absolute_step=15000+i,camera_index=k,view=str(records[k][1])) for i,k in enumerate(order)]
        W._write_csv(out/'camera_sequence.csv',sequence)
        W._write_json(out/'protocol.json',dict(scene=args.scene,arm=args.arm,steps=args.steps,
            source_checkpoint=str(checkpoint),source_sha256=chash,initial_model_sha256=initial_hash,
            camera_sequence_sha256=W._hash_object(sequence),new_loss_weight=.05,extra_every=4,extra_gradient_ratio_cap=.1,
            CICA_lambda=lambda_cica,CICA_refresh_every=500,normal_refinement=True,
            module_support=module.support if module else {},diagnostic_gate_required=False))
        # Restore exact continuation RNG after all deterministic setup.
        W._set_rng_state(payload['rng_state']);del payload
        controller=None;rows=[];events=[];started=time.perf_counter();max_mem=0
        auxiliary_calls=0;auxiliary_nonzero=0
        for i,ci in enumerate(order):
            step=15000+i
            model.step_cb(step)
            if controller is None or i%500==0 or int(controller['population'])!=model.num_points:
                model.eval()
                with torch.no_grad():controller=W._build_cica_controller(model,bank_records)
            branch.pipeline.train();model.train()
            model.aopt_before_train_iteration(branch.optimizers,step)
            model.medium_hold_before_train_iteration(branch.optimizers,step)
            branch.optimizers.zero_grad_all()
            params,groups=W._parameter_entries(model)
            allowed=sorted(set(groups.get('features_dc',[])+groups.get('features_rest',[])+groups.get('medium_mlp',[])+groups.get('direction_encoding',[])))
            extra_grads=None;extra_value=0.;info={}
            if module is not None and i%4==0:
                raw,info=module.loss(model,i//4,W._gt,W._batch_to_device)
                if not torch.isfinite(raw):raise RuntimeError('nonfinite extra loss')
                extra_value=float(raw.detach());auxiliary_calls+=1
                extra_grads=torch.autograd.grad(.05*raw,[params[k] for k in allowed],allow_unused=True)
                auxiliary_nonzero+=int(any(g is not None and bool(g.detach().abs().max()>0) for g in extra_grads))
                del raw
            _,view,camera,batch=records[ci]
            outputs=model.get_outputs(camera.to(model.device))
            losses=model.get_loss_dict(outputs,W._batch_to_device(batch.copy(),model.device),{})
            base=sum(losses.values());cica,cinfo=W._cica_loss(model,controller)
            if not bool(torch.isfinite(base)) or not W.FORMAL._finite_outputs(outputs):raise RuntimeError('nonfinite base outputs')
            bounded=outputs['gaussian_view_rgb']
            if bool((bounded>1.000001).any() | (bounded<-.000001).any()):raise RuntimeError('BND violated')
            cica_grads=torch.autograd.grad(lambda_cica*cica,[model.features_dc],retain_graph=True)
            base.backward()
            if cica_grads[0] is not None:
                model.features_dc.grad.add_(cica_grads[0])
            scale=0.;extra_norm=0.
            if extra_grads is not None:
                extra_norm=W._norm(extra_grads)
                base_norm=W._norm([params[k].grad for k in allowed])
                scale=min(1.,.1*base_norm/max(extra_norm,1e-12))
                for k,g in zip(allowed,extra_grads):
                    if g is not None:
                        if params[k].grad is None:params[k].grad=scale*g.detach()
                        else:params[k].grad.add_(g,alpha=scale)
            branch.optimizers.optimizer_step_all();branch.optimizers.scheduler_step_all(step)
            model.aopt_after_train_iteration(step);model.medium_hold_after_train_iteration(branch.optimizers,step)
            model.after_train(step=step)
            if step%int(model.config.refine_every)==0:
                model.refinement_after(branch.optimizers,step=step)
                events.append(dict(getattr(model,'_refinement_last_event',{}), step=step))
            max_mem=max(max_mem,torch.cuda.max_memory_allocated()/1024**3)
            if i==0 or (i+1)%100==0 or i+1==args.steps:
                row=dict(step=step,update=i+1,view=str(view),loss_rgb=float(base.detach()),loss_extra=extra_value,
                         extra_grad_norm=extra_norm,extra_grad_scale=scale,loss_cica=float(cica.detach()),
                         gaussians=model.num_points,seconds=time.perf_counter()-started,**info)
                rows.append(row);W._write_csv(out/'training.csv',rows)
                print(json.dumps(dict(scene=args.scene,arm=args.arm,**row)),flush=True)
                W._write_json(out/'progress.json',dict(completed=False,**row))
            del outputs,losses,base,cica,cica_grads,extra_grads,params
            if i%25==24:
                gc.collect();torch.cuda.empty_cache()
        final_step=14999+args.steps
        ckpt=W._save_checkpoint(branch,args.scene,args.arm,final_step,out,source,'',W._hash_tensor(controller['target_log_chroma']),'',lambda_cica,{'enabled':False})
        training_seconds=time.perf_counter()-started
        model.eval()
        aggregate,per_view,decomp,renders=W._evaluate_checkpoint(branch,args.arm,final_step,W._eval_records(branch),out/'renders'/args.scene)
        W._write_csv(out/'per_view_metrics.csv',per_view);W._write_csv(out/'evaluation_metrics.csv',[aggregate])
        W._write_csv(out/'render_manifest.csv',renders);W._write_csv(out/'refinement.csv',events)
        W._write_json(out/'decomposition.json',decomp)
        assert W._sha256(checkpoint)==chash
        result=dict(completed=True,scene=args.scene,arm=args.arm,updates=args.steps,checkpoint=str(ckpt),checkpoint_sha256=W._sha256(ckpt),
                    source_unchanged=True,training_seconds=training_seconds,peak_cuda_gb=max_mem,
                    auxiliary_calls=auxiliary_calls,auxiliary_nonzero_calls=auxiliary_nonzero,metrics=aggregate,
                    all_clear_bounded=float(decomp['P_J_gt_1'])==0)
        W._write_json(out/'result.json',result);print(json.dumps(result),flush=True)
    finally:W._release(branch)


if __name__=='__main__':main()
