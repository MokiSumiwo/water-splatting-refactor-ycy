"""Finite branch-VJP audit for the geometry-gradient experiment; no training."""
import sys, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.experiments import run_direct_mdrr_cica_scene as W

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 W._route_renderer_backend('corrected_v1')
 from scripts.experiments.geometry_projection_backend import install
 install()
 W.ACTIVE_SOURCE_CONFIG=ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene/'source_config.yml'
 W.ACTIVE_PARAMETERIZATION='bounded_sh3';W.ACTIVE_TRAINING_SEED=42;W.AUXILIARY_APPEARANCE_ENABLED=False
 b=W._new_branch(a.scene);m=b.pipeline.model;m.eval()
 results=[]
 try:
  records=W._train_records(b)
  for rec in [records[0],records[len(records)//2]]:
   cam=rec[2].to(m.device);batch=W._batch_to_device(rec[3],m.device)
   o=m.get_outputs(cam);loss=sum(m.get_loss_dict(o,batch,{}).values())
   r=torch.autograd.grad(loss,o['pred_image'],retain_graph=True)[0].detach()
   gd=torch.autograd.grad(o['rgb_object'],m.means,grad_outputs=r,retain_graph=True)[0]
   gm=torch.autograd.grad(o['rgb_medium'],m.means,grad_outputs=r,retain_graph=True)[0]
   gn=torch.autograd.grad(o['pred_image'],m.means,grad_outputs=r,retain_graph=True)[0]
   err=float((gd+gm-gn).norm()/gn.norm().clamp_min(1e-12))
   # A normalized direction away from sorting/visibility boundaries; three epsilons.
   direction=torch.zeros_like(gn); index=int(gn.norm(dim=-1).argmax()); direction[index]=gn[index].detach()/gn[index].norm().clamp_min(1e-12)
   ad=float((gn*direction).sum());fds=[];original=m.means.detach().clone()
   for h in [0.00002,0.00001,0.000005]:
    vals=[]
    for sign in [1,-1]:
     with torch.no_grad():
      m.means.copy_(original+sign*h*direction)
      z=m.get_outputs(cam);vals.append((z['pred_image'].double()*r.double()).sum().item())
      del z
    fd=(vals[0]-vals[1])/(2*h)
    fds.append(dict(h=h,fd=fd,error=abs(fd-ad),passed=abs(fd-ad)<=2e-5+.03*abs(ad)))
   with torch.no_grad():m.means.copy_(original)
   row=dict(view=rec[1],relative_branch_sum_error=err,direct_norm=float(gd.norm()),medium_norm=float(gm.norm()),native_norm=float(gn.norm()),directional_ad=ad,finite_difference=fds,
    passed=err<=1e-3 and sum(x['passed'] for x in fds)>=2)
   results.append(row);del o,loss,gd,gm,gn,r
  payload=dict(scene=a.scene,optimizer_updates=0,results=results,passed=all(x['passed'] for x in results))
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
 finally:W._release(b)
if __name__=='__main__':main()
