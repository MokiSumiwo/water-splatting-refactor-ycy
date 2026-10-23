"""Freeze training-camera review regions; candidates remain unknown until reviewed."""
import argparse,sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.experiments import run_direct_mdrr_cica_scene as W

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 W._route_renderer_backend('corrected_v1');W.ACTIVE_TRAINING_SEED=42;W.ACTIVE_PARAMETERIZATION='bounded_sh3'
 base=ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene
 W.ACTIVE_SOURCE_CONFIG=base/'source_config.yml';b=W._new_branch(a.scene)
 try:
  W._load_snapshot(b,a.scene,base/'checkpoints/step-000015000.ckpt');m=b.pipeline.model;m.eval();records=W._train_records(b);assets=[];panels=[]
  for k in [0,len(records)//2,len(records)-1]:
   rec=records[k]
   with torch.no_grad():
    o=m.get_outputs_for_camera(copy.deepcopy(rec[2]).to(m.device));gt=W._gt(m,W._batch_to_device(rec[3],m.device),o['background']).clamp(0,1)
   arrays=dict(gt=gt.cpu().numpy(),underwater=o['pred_image'].clamp(0,1).cpu().numpy(),clear=o['clear_object_fullsh_raw'].clamp(0,1).cpu().numpy())
   h,w=arrays['gt'].shape[:2];folder=a.output/str(rec[1]);folder.mkdir(exist_ok=True)
   for name,arr in arrays.items():Image.fromarray((arr*255).round().astype(np.uint8)).save(folder/(name+'.png'))
   # Freeze a uniform grid: selection labels do not assert water identity.
   boxes=[]
   for iy in range(3):
    for ix in range(3):boxes.append([ix*w//3,iy*h//3,(ix+1)*w//3,(iy+1)*h//3])
   assets.append(dict(camera=rec[1],camera_index=k,role='training_review',shape=[h,w],regions=[dict(id=f'{iy//3}{iy%3}',box=box,label='U',evidence='uniform pre-treatment review grid; no water label inferred') for iy,box in enumerate(boxes)]))
   tiles=[]
   for name,arr in arrays.items():
    im=Image.fromarray((arr*255).round().astype(np.uint8));im.thumbnail((600,450));tiles.append(im)
   row=Image.new('RGB',(1800,480));draw=ImageDraw.Draw(row)
   for j,im in enumerate(tiles):row.paste(im,(j*600,30));draw.text((j*600,5),str(rec[1])+' '+list(arrays)[j],fill='white')
   panels.append(row);del o
  sheet=Image.new('RGB',(1800,480*len(panels)))
  for i,row in enumerate(panels):sheet.paste(row,(0,480*i))
  sheet.save(a.output/'review.jpg')
  (a.output/'regions.json').write_text(json.dumps(dict(scene=a.scene,source=str(base),test_views_used=False,assets=assets,labels='All U pending cross-view evidence; blue appearance is not a water label.'),indent=2)+'\n')
 finally:W._release(b)
if __name__=='__main__':main()
