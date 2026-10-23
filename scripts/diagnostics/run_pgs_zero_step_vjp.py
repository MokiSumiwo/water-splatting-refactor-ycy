#!/usr/bin/env python3
"""Read-only PGS photometric-gradient shielding diagnostic."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
CK={'Curasao':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/Curasao/cica_on/checkpoints/step-000014999.ckpt','IUI3-RedSea':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/IUI3-RedSea/cica_on/checkpoints/step-000014999.ckpt','JapaneseGradens-RedSea':ROOT/'outputs/bnd_on_cica_ablation_20260914/JapaneseGradens-RedSea/cica_on/checkpoints/step-000014999.ckpt','Panama':ROOT/'outputs/bnd_on_cica_ablation_20260914/Panama/cica_on/checkpoints/step-000014999.ckpt'}
SRC=ROOT/'outputs/m1_raoc_causal_four_scene_20260827'; CFG=ROOT/'outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml'
def norm(a): return float(a.float().norm().item())
def dot(a,b): return float((a.float().reshape(-1)*b.float().reshape(-1)).sum().item())
def stats(g,n):
 ng,nn=norm(g),norm(n); c=dot(g,n)/(ng*nn+1e-20); q=max(0.,min(1.,c*c)); return {'full_grad_l2':ng,'nuisance_grad_l2':nn,'cosine_full_nuisance':c,'shieldable_energy_fraction':q,'projected_grad_l2':ng*(1-q)**.5}
def run(scene,out):
 W.ACTIVE_SOURCE_ROOT=SRC; W.ACTIVE_SOURCE_CONFIG=CFG; W.ACTIVE_PARAMETERIZATION='bounded_sh3'; W.CICA_ENABLED=True; W.OCMC_ENABLED=True; W.OCMC_STRENGTH=1.; W.OCMC_STRENGTH_START=1.; W.OCMC_STRENGTH_RAMP_STEPS=0; W.AUXILIARY_APPEARANCE_ENABLED=False; W.BRASH_ENABLED=False; W.BVAM_ENABLED=False; W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
 b=W._new_branch(scene)
 try:
  W._load_snapshot(b,scene,CK[scene]); model=b.pipeline.model; model.train(); rows=[]
  pmap={'features_dc':model.features_dc,'features_rest':model.features_rest,'means':model.means,'scales':model.scales,'quats':model.quats,'opacities':model.opacities}; names=list(pmap); params=list(pmap.values())
  for _,view,cam,batch in W._train_records(b)[:3]:
   model.zero_grad(set_to_none=True); o=model.get_outputs(cam.to(model.device)); gt=W._gt(model,W._batch_to_device(batch,model.device),o['background']).float().clamp(0,1); pred=o['pred_image']; obj=o['rgb_object']; lfull=(pred-gt).square().mean(); lnuis=(pred-obj).square().mean(); lobj=(obj-gt).square().mean(); gf=torch.autograd.grad(lfull,params,retain_graph=True,allow_unused=True); gn=torch.autograd.grad(lnuis,params,retain_graph=True,allow_unused=True); go=torch.autograd.grad(lobj,params,allow_unused=True)
   row={'scene':scene,'view_id':str(view),'full_loss':float(lfull.detach()),'nuisance_loss':float(lnuis.detach()),'object_loss':float(lobj.detach()),'transmission_mean':float(o['transmission'].mean().detach()),'accumulation_mean':float(o['accumulation'].mean().detach())}
   for name,a,n,x in zip(names,gf,gn,go):
    if a is not None: row[name]=stats(a,n if n is not None else torch.zeros_like(a)); row[name]['object_grad_l2']=norm(x) if x is not None else 0.
   rows.append(row)
  out.mkdir(parents=True,exist_ok=True); json.dump({'scene':scene,'checkpoint':str(CK[scene]),'step':14999,'read_only':True,'views':rows},open(out/'pgs_vjp.json','w'),indent=2)
 finally: W._release(b)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--scene',choices=SCENES,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); run(a.scene,a.output_dir.resolve())
if __name__=='__main__': main()
