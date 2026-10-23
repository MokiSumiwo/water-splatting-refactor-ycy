"""Real checkpoint no-update audit of B disabled identity and means-only routing."""
import argparse
import json
from pathlib import Path
import torch
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.geometry_projection_backend import install
from water_splatting.surface_support import SurfaceSupport
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_runs/surface_support_B_20261023'

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
    scene=a.scene;base=ROOT/'research_runs/geometry_gradient_A_20261021/formal'/scene/'B0'
    W._route_renderer_backend('corrected_v1');install()
    W.ACTIVE_SOURCE_CONFIG=base/'source_config.yml';W.ACTIVE_PARAMETERIZATION='bounded_sh3';W.ACTIVE_TRAINING_SEED=42
    branch=W._new_branch(scene)
    try:
        W._load_snapshot(branch,scene,base/'checkpoints/step-000010000.ckpt')
        m=branch.pipeline.model;m.eval()
        out=OUT/'routing_audit'/scene;out.mkdir(parents=True,exist_ok=False)
        (out/'camera_sequence.csv').symlink_to(base/'camera_sequence.csv')
        controller=SurfaceSupport(branch,out,OUT/'support'/scene,enabled=False)
        records=W._train_records(branch);results=[]
        for rec in records:
            if rec[1] not in controller.names:continue
            o=m.get_outputs(rec[2].to(m.device));loss=sum(m.get_loss_dict(o,W._batch_to_device(rec[3],m.device),{}).values())
            valid,_=controller.select(m,o,rec[1])
            if valid.sum()==0:continue
            params=[p for p in m.parameters() if p.requires_grad]
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            for p,g in zip(params,grads):p.grad=g
            before=[None if p.grad is None else p.grad.clone() for p in params]
            abs_before=m.xys_grad_abs.clone();rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
            controller.sequence[8000]=rec[1]
            controller.apply(m,o,loss,8000)
            disabled=all((g is None and p.grad is None) or (g is not None and torch.equal(g,p.grad)) for p,g in zip(params,before))
            controller.enabled=True;controller.apply(m,o,loss,8000)
            nonmeans=all((g is None and p.grad is None) or (g is not None and torch.equal(g,p.grad)) for p,g in zip(params,before) if p is not m.means)
            native=next(g for p,g in zip(params,before) if p is m.means)
            delta=m.means.grad-native
            cap=bool((delta.norm(dim=-1)<=.05*native.norm(dim=-1)+1e-8).all())
            no_rng=torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
            buffers=torch.equal(abs_before,m.xys_grad_abs)
            row=dict(camera=rec[1],supported=int(valid.sum()),disabled_exact=disabled,nonmeans_exact=nonmeans,
                     local_cap=cap,rng_exact=no_rng,densification_abs_exact=buffers,delta_norm=float(delta.norm()),optimizer_updates=0)
            row['passed']=disabled and nonmeans and cap and no_rng and buffers and row['delta_norm']>0
            results.append(row);break
        assert results and all(r['passed'] for r in results),results
        (out/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
    finally:W._release(branch)
if __name__=='__main__':main()
