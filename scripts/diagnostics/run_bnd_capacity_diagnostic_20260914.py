#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,os,math
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
W.ENFORCE_SCENE_GPU_ASSIGNMENT=False; W.ACTIVE_PARAMETERIZATION='bounded_sh3'; W.OCMC_ENABLED=True; W.OCMC_STRENGTH=1.; W.OCMC_STRENGTH_START=1.; W.OCMC_STRENGTH_RAMP_STEPS=0; W.BRASH_ENABLED=False; W.BVAM_ENABLED=False; W.AUXILIARY_APPEARANCE_ENABLED=False; W.START_STEP=3000; W.FINAL_STEP=14999; W.CICA_ENABLED=True
SCENE_INFO={
 'Curasao':('outputs/bnd_on_cica_source_extra_20260914','outputs/taba_chromatic_bnd_on_20260910/Curasao/E0/source_config.yml','outputs/bnd_on_cica_ablation_extra3_20260914/Curasao/cica_on/checkpoints/step-000014999.ckpt'),
 'IUI3-RedSea':('outputs/bnd_on_cica_source_extra_20260914','outputs/taba_chromatic_bnd_on_20260910/IUI3-RedSea/E0/source_config.yml','outputs/bnd_on_cica_ablation_extra3_20260914/IUI3-RedSea/cica_on/checkpoints/step-000014999.ckpt'),
 'JapaneseGradens-RedSea':('outputs/taba_paired_source_20260910_v2/bnd_on','outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml','outputs/bnd_on_cica_ablation_20260914/JapaneseGradens-RedSea/cica_on/checkpoints/step-000014999.ckpt'),
 'Panama':('outputs/taba_paired_source_20260910_v2/bnd_on','outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml','outputs/bnd_on_cica_ablation_20260914/Panama/cica_on/checkpoints/step-000014999.ckpt')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args(); scene=a.scene; src,cfg,ck=SCENE_INFO[scene]; W.ACTIVE_SOURCE_ROOT=ROOT/src; W.ACTIVE_SOURCE_CONFIG=ROOT/cfg
 b=W._new_branch(scene)
 try:
  pay=torch.load(ROOT/ck,map_location='cpu',weights_only=False); b.pipeline.model.load_state_dict(pay['model'],strict=True); b.pipeline.model.step=14999; W._configure_model(b.pipeline.model); W.FORMAL._install_condition(b.pipeline.model,'C0',pay['ocmc_bundle'],None); W._configure_model(b.pipeline.model); b.pipeline.eval()
  records=W._eval_records(b); rows=[]; pop=int(b.pipeline.model.num_points)
  for _,vid,cam,batch in records:
   with torch.no_grad():
    o=b.pipeline.model.get_outputs_for_camera(cam.to(b.pipeline.model.device)); gt=W._gt(b.pipeline.model,W._batch_to_device(batch,b.pipeline.model.device),o['background']).float().clamp(0,1); pred=o['pred_image'].float().clamp(0,1)
    err=(pred-gt).abs().mean(-1); rgb=o['gaussian_view_rgb'].float(); dc=o['gaussian_view_dc_rgb'].float(); res=o['gaussian_color_residual'].float(); der=o['gaussian_sigmoid_derivative'].float(); vis=o['gaussian_visible_mask'].reshape(-1).bool(); logits=o['gaussian_view_logits'].float();
    sat=((rgb<.01)|(rgb>.99)).any(-1); near=((rgb<.05)|(rgb>.95)).any(-1); lowder=(der.mean(-1)<.05); highres=res.abs().mean(-1)>.15; eligible=vis & (~lowder)
    trans=o['transmission'].mean(-1); acc=o['accumulation'].squeeze(-1)
    for name,mask in [('all',torch.ones_like(err,dtype=torch.bool)),('high_error',err>torch.quantile(err.flatten(),.9)),('low_error',err<=torch.quantile(err.flatten(),.5)),('low_trans',trans<.35),('high_trans',trans>=.7),('low_acc',acc<.5),('high_acc',acc>=.5)]:
      rows.append({'scene':scene,'view_id':str(vid),'region':name,'population':pop,'pixel_fraction':float(mask.float().mean()),'pixel_mse':float((pred-gt).square().mean(-1)[mask].mean()) if mask.any() else None,'acc_mean':float(acc[mask].mean()) if mask.any() else None,'trans_mean':float(trans[mask].mean()) if mask.any() else None,'gaussian_visible_fraction':float(vis.float().mean()),'gaussian_sat_fraction':float(sat.float().mean()),'gaussian_near_bound_fraction':float(near.float().mean()),'gaussian_low_derivative_fraction':float(lowder.float().mean()),'gaussian_high_residual_fraction':float(highres.float().mean()),'gaussian_eligible_fraction':float(eligible.float().mean()),'logit_abs_p95':float(torch.quantile(logits.abs().flatten(),.95)),'dc_mean':float(dc.mean()),'rgb_mean':float(rgb.mean()),'residual_abs_mean':float(res.abs().mean()),'residual_abs_p95':float(torch.quantile(res.abs().flatten(),.95))})
  out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); json.dump({'scene':scene,'checkpoint':str(ROOT/ck),'step':14999,'population':pop,'views':len(records),'rows':rows},open(out,'w'),indent=2)
  print(json.dumps({'scene':scene,'views':len(records),'population':pop,'output':str(out)}))
 finally: W._release(b)
if __name__=='__main__': main()
