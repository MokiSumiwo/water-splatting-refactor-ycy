#!/usr/bin/env python3
"""Read-only CVC front-surface stability audit on frozen BND checkpoints."""
import argparse, json, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
CK={'Curasao':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/Curasao/cica_on/checkpoints/step-000014999.ckpt','IUI3-RedSea':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/IUI3-RedSea/cica_on/checkpoints/step-000014999.ckpt','JapaneseGradens-RedSea':ROOT/'outputs/bnd_on_cica_ablation_20260914/JapaneseGradens-RedSea/cica_on/checkpoints/step-000014999.ckpt','Panama':ROOT/'outputs/bnd_on_cica_ablation_20260914/Panama/cica_on/checkpoints/step-000014999.ckpt'}
SRC=ROOT/'outputs/m1_raoc_causal_four_scene_20260827'; CFG=ROOT/'outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml'
def rho(x,y):
 if len(x)<3: return None
 rx=torch.argsort(torch.argsort(torch.tensor(x))).float(); ry=torch.argsort(torch.argsort(torch.tensor(y))).float(); a=rx-rx.mean(); b=ry-ry.mean(); return float((a*b).sum()/(a.norm()*b.norm()+1e-12))
def run(scene,out):
 W.ACTIVE_SOURCE_ROOT=SRC; W.ACTIVE_SOURCE_CONFIG=CFG; W.ACTIVE_PARAMETERIZATION='bounded_sh3'; W.CICA_ENABLED=True; W.OCMC_ENABLED=True; W.OCMC_STRENGTH=1.; W.OCMC_STRENGTH_START=1.; W.OCMC_STRENGTH_RAMP_STEPS=0; W.AUXILIARY_APPEARANCE_ENABLED=False; W.BRASH_ENABLED=False; W.BVAM_ENABLED=False; W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
 b=W._new_branch(scene)
 try:
  W._load_snapshot(b,scene,CK[scene]); m=b.pipeline.model; m.eval(); train=W._train_records(b); n=int(m.num_points); dev=m.device
  count=torch.zeros(n,device=dev); front=torch.zeros(n,device=dev); ds=torch.zeros(n,device=dev); d2=torch.zeros(n,device=dev); rs=torch.zeros(n,device=dev); r2=torch.zeros(n,device=dev); cs=torch.zeros(n,device=dev); c2=torch.zeros(n,device=dev)
  for _,_,cam,_ in train:
   with torch.no_grad():
    o=m.get_outputs_for_camera(cam.to(dev)); vis=o['gaussian_visible_mask'].reshape(-1).bool(); depth=o['projected_gaussian_depths'].reshape(-1).float(); rad=m.radii.reshape(-1).float(); col=o['gaussian_view_rgb'].float().mean(-1)
    valid=vis & torch.isfinite(depth) & torch.isfinite(rad)
    dv=depth[valid]; q=torch.quantile(dv,0.30) if dv.numel() else depth.new_tensor(0.)
    ids=torch.nonzero(valid,as_tuple=False).reshape(-1); count[ids]+=1; front[ids]+=(dv<=q).float(); ds[ids]+=dv; d2[ids]+=dv*dv; rs[ids]+=rad[ids]; r2[ids]+=rad[ids]*rad[ids]; cs[ids]+=col[ids]; c2[ids]+=col[ids]*col[ids]
  c=count.clamp_min(1.); fp=front/c; dm=ds/c; dv=(d2/c-dm*dm).clamp_min(0.).sqrt()/(dm.abs()+1e-6); rm=(rs/c); rv=(r2/c-rm*rm).clamp_min(0.).sqrt()/(rm.abs()+1e-6); cm=(cs/c); cv=(c2/c-cm*cm).clamp_min(0.).sqrt()
  stability=(fp*torch.exp(-dv)*torch.exp(-rv)*torch.exp(-2.*cv)).clamp(0,1); rows=[]
  for _,view,cam,batch in W._eval_records(b):
   with torch.no_grad():
    o=m.get_outputs_for_camera(cam.to(dev)); gt=W._gt(m,W._batch_to_device(batch,dev),o['background']).float().clamp(0,1); pred=o['pred_image'].float().clamp(0,1); vis=o['gaussian_visible_mask'].reshape(-1).bool(); err=float((pred-gt).square().mean().sqrt()); ids=vis.nonzero().reshape(-1); score=float(stability[ids].mean()) if ids.numel() else 0.; support=float(c[ids].mean()) if ids.numel() else 0.; rows.append({'scene':scene,'view_id':str(view),'front_surface_stability_mean':score,'mean_train_support_visible':support,'heldout_rmse':err,'heldout_psnr':-10*math.log10(max(err*err,1e-12)),'transmission_mean':float(o['transmission'].mean()),'accumulation_mean':float(o['accumulation'].mean()),'visible_gaussian_fraction':float(vis.float().mean())})
  payload={'scene':scene,'checkpoint':str(CK[scene]),'step':14999,'read_only':True,'train_views':len(train),'gaussian_count':n,'front_surface_stats':{'front_occurrence_mean':float(fp.mean()),'depth_cv_mean':float(dv.mean()),'footprint_cv_mean':float(rv.mean()),'color_std_mean':float(cv.mean()),'stability_mean':float(stability.mean()),'stability_p10':float(torch.quantile(stability,.1)),'stability_p50':float(torch.quantile(stability,.5)),'stability_p90':float(torch.quantile(stability,.9))},'heldout_rows':rows,'rho_stability_rmse':rho([r['front_surface_stability_mean'] for r in rows],[r['heldout_rmse'] for r in rows]),'rho_support_rmse':rho([r['mean_train_support_visible'] for r in rows],[r['heldout_rmse'] for r in rows])}
  out.mkdir(parents=True,exist_ok=True); json.dump(payload,open(out/'cvc_front_surface.json','w'),indent=2)
 finally: W._release(b)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--scene',choices=SCENES,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); run(a.scene,a.output_dir.resolve())
if __name__=='__main__': main()
