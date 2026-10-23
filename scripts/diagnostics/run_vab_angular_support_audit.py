#!/usr/bin/env python3
"""Frozen VAB angular-support audit; no training or parameter updates."""
import argparse,csv,json,math,sys,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
CK={'Curasao':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/Curasao/cica_on/checkpoints/step-000014999.ckpt','IUI3-RedSea':ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/IUI3-RedSea/cica_on/checkpoints/step-000014999.ckpt','JapaneseGradens-RedSea':ROOT/'outputs/bnd_on_cica_ablation_20260914/JapaneseGradens-RedSea/cica_on/checkpoints/step-000014999.ckpt','Panama':ROOT/'outputs/bnd_on_cica_ablation_20260914/Panama/cica_on/checkpoints/step-000014999.ckpt'}
SRC=ROOT/'outputs/m1_raoc_causal_four_scene_20260827'; CFG=ROOT/'outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml'
def direction(cam,center):
 p=cam.camera_to_worlds[0,:3,3].float(); return (center-p)/(center-p).norm().clamp_min(1e-8)
def spearman(x,y):
 rx=torch.argsort(torch.argsort(torch.tensor(x))).float(); ry=torch.argsort(torch.argsort(torch.tensor(y))).float(); return float(torch.nn.functional.cosine_similarity(rx-rx.mean(),ry-ry.mean(),dim=0).item())
def run(scene,out):
 W.ACTIVE_SOURCE_ROOT=SRC; W.ACTIVE_SOURCE_CONFIG=CFG; W.ACTIVE_PARAMETERIZATION='bounded_sh3'; W.CICA_ENABLED=True; W.OCMC_ENABLED=True; W.OCMC_STRENGTH=1.; W.OCMC_STRENGTH_START=1.; W.OCMC_STRENGTH_RAMP_STEPS=0; W.AUXILIARY_APPEARANCE_ENABLED=False; W.BRASH_ENABLED=False; W.BVAM_ENABLED=False; W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
 b=W._new_branch(scene)
 try:
  W._load_snapshot(b,scene,CK[scene]); model=b.pipeline.model; model.eval(); center=model.means.detach().float().mean(0); train=W._train_records(b); trdirs=torch.stack([direction(c.to(model.device),center) for _,_,c,_ in train]); rows=[]
  for _,view,cam,batch in W._eval_records(b):
   with torch.no_grad():
    o=model.get_outputs_for_camera(cam.to(model.device)); gt=W._gt(model,W._batch_to_device(batch,model.device),o['background']).float().clamp(0,1); pred=o['pred_image'].float().clamp(0,1); err=float((pred-gt).square().mean().sqrt());
   d=direction(cam.to(model.device),center); angles=torch.rad2deg(torch.acos((trdirs@d).clamp(-1,1))); rows.append({'scene':scene,'view_id':str(view),'nearest_angle_deg':float(angles.min()),'knn3_angle_deg':float(torch.topk(angles,3,largest=False).values.mean()),'train_views':len(train),'rmse':err,'psnr':-10*math.log10(max(err*err,1e-12)),'transmission_mean':float(o['transmission'].mean()),'accumulation_mean':float(o['accumulation'].mean()),'visible_gaussian_fraction':float(o['gaussian_visible_mask'].float().mean())})
  out.mkdir(parents=True,exist_ok=True); json.dump({'scene':scene,'checkpoint':str(CK[scene]),'step':14999,'read_only':True,'rows':rows,'nearest_angle_rmse_spearman':spearman([r['nearest_angle_deg'] for r in rows],[r['rmse'] for r in rows]),'knn3_angle_rmse_spearman':spearman([r['knn3_angle_deg'] for r in rows],[r['rmse'] for r in rows])},open(out/'vab_angular_support.json','w'),indent=2)
 finally: W._release(b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.scene,a.output_dir.resolve())
if __name__=='__main__':main()
