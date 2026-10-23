"""Endpoint float evaluation assets for paired geometry-gradient A."""
import argparse,sys,json,copy,subprocess,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.experiments import run_direct_mdrr_cica_scene as W

def contribution_assets(model,camera,outputs,asset,dest):
 """Actual alpha*T ROI contribution; ablations are diagnostics, not water labels."""
 geometry=W.FULL._project_for_camera(model,copy.deepcopy(camera).to(model.device))
 colors=outputs['gaussian_view_rgb'];n=len(colors);h,w=outputs['pred_image'].shape[:2]
 opacity=model.opacities.detach().sigmoid()
 if model.config.rasterize_mode=='antialiased':opacity=opacity*geometry['compensation'][:,None]
 zero=torch.zeros(h,w,device=model.device);rgbzero=torch.zeros(h,w,3,device=model.device)
 ids=torch.arange(n,device=model.device);saved=model.opacities.detach().clone();rows=[]
 for k,region in enumerate(asset['residual_review']):
  x0,y0,x1,y1=region['box'];mask=rgbzero.clone();mask[y0:y1,x0:x1]=1
  stats,_,_,_=W.FULL._responsibility_forward(W._responsibility_extension(),geometry,colors,opacity,rgbzero,ids,mask,zero,zero)
  # Full-SH RGB contribution may be signed; use nonnegative channel energy
  # only to rank coverage, while retaining the raw ROI alpha*T weights.
  weights=stats[:,1];mass=weights*colors.clamp_min(0).mean(-1)
  order=mass.argsort(descending=True);cumulative=mass[order].cumsum(0);total=mass.sum()
  count=int(torch.searchsorted(cumulative,.8*total).item())+1 if total>0 else 0
  selected=order[:count];tag=f"{asset['camera']}_roi{k}"
  np.savez_compressed(dest/(tag+'_sources.npz'),ids=selected.cpu().numpy(),roi_weight=weights[selected].cpu().numpy(),rgb=colors[selected].cpu().numpy(),depth=geometry['depths'][selected].cpu().numpy())
  try:
   with torch.no_grad():
    model.opacities[selected]=-100
    behind=model.get_outputs_for_camera(copy.deepcopy(camera).to(model.device))['clear_object_fullsh_raw'].clamp(0,1)
    Image.fromarray((behind.cpu().numpy()*255).round().astype('uint8')).save(dest/(tag+'_remaining.png'))
    model.opacities.copy_(saved);keep=torch.zeros(n,dtype=torch.bool,device=model.device);keep[selected]=True;model.opacities[~keep]=-100
    top=model.get_outputs_for_camera(copy.deepcopy(camera).to(model.device))['clear_object_fullsh_raw'].clamp(0,1)
    Image.fromarray((top.cpu().numpy()*255).round().astype('uint8')).save(dest/(tag+'_dominant.png'))
  finally:
   with torch.no_grad():model.opacities.copy_(saved)
  rows.append(dict(box=region['box'],label='U',selected=count,total_gaussians=n,ranking='ROI alpha*T times mean positive full-SH color',coverage=float(mass[selected].sum()/total) if total>0 else None,not_a_removal_label=True))
 (dest/(asset['camera']+'_contribution.json')).write_text(json.dumps(rows,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--arm',required=True);p.add_argument('--verify-smoke',action='store_true');p.add_argument('--campaign',type=Path,default=ROOT/'research_runs/geometry_gradient_A_20261021');a=p.parse_args()
 campaign=a.campaign.resolve();out=campaign/'formal'/a.scene/a.arm
 if a.verify_smoke:out=campaign/'smoke'/a.scene/'A'
 W._route_renderer_backend('corrected_v1')
 from scripts.experiments.geometry_projection_backend import install
 install();W.ACTIVE_SOURCE_CONFIG=out/'source_config.yml';W.ACTIVE_PARAMETERIZATION='bounded_sh3';W.ACTIVE_TRAINING_SEED=42
 b=W._new_branch(a.scene)
 try:
  W._load_snapshot(b,a.scene,out/('checkpoints/step-000000020.ckpt' if a.verify_smoke else 'checkpoints/step-000015000.ckpt'));m=b.pipeline.model;m.eval()
  dest=campaign/('export_smoke' if a.verify_smoke else 'assets')/a.scene/a.arm;dest.mkdir(parents=True,exist_ok=True)
  records=W._train_records(b);region=json.loads((ROOT/'research_runs/geometry_gradient_A_20261021/regions'/a.scene/'regions.json').read_text());roi_names={v['camera'] for v in region['assets']}
  review=[];vrows=[]
  for rec in records:
   with torch.no_grad():
    o=m.get_outputs_for_camera(copy.deepcopy(rec[2]).to(m.device));gt=W._gt(m,W._batch_to_device(rec[3],m.device),o['background']).clamp(0,1)
   pred=o['pred_image'].detach().float().clamp(0,1);clear=o['clear_object_fullsh_raw'].detach().float().clamp(0,1)
   if rec[1] in roi_names:
    np.savez_compressed(dest/(str(rec[1])+'.npz'),underwater=pred.cpu().numpy(),gt=gt.cpu().numpy(),clear=clear.cpu().numpy(),accumulation=o['accumulation'].cpu().numpy())
    h,w=clear.shape[:2];strip=Image.new('RGB',(3*w,h))
    for j,t in enumerate([gt,pred,clear]):strip.paste(Image.fromarray((t.cpu().numpy()*255).round().astype('uint8')),(j*w,0))
    strip.save(dest/(str(rec[1])+'.png'))
    with torch.no_grad():contribution_assets(m,rec[2],o,next(v for v in region['assets'] if v['camera']==rec[1]),dest)
   # Stream both native-resolution real-camera sequences to ffmpeg.
   if not vrows:
    h,w=clear.shape[:2];processes={}
    for name in ['clear','underwater']:
     processes[name]=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','3','-i','-','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(dest/(name+'.mp4'))],stdin=subprocess.PIPE)
   for name,t in [('clear',clear),('underwater',pred)]:processes[name].stdin.write((t.cpu().numpy()*255).round().astype('uint8').tobytes())
   vrows.append(str(rec[1]));del o
  for proc in processes.values():proc.stdin.close();assert proc.wait()==0
  # Preserve endpoint test floats; PNGs are display assets, never metric inputs.
  test_dir=dest/'eval_float';test_dir.mkdir(exist_ok=True);eval_rows=[]
  for _,name,camera,batch in W._eval_records(b):
   torch.cuda.synchronize();started=time.perf_counter()
   with torch.no_grad():
    o=m.get_outputs_for_camera(copy.deepcopy(camera).to(m.device))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    gt=W._gt(m,W._batch_to_device(batch,m.device),o['background']).detach().float().clamp(0,1)
   pred=o['pred_image'].detach().float().clamp(0,1)
   np.savez_compressed(test_dir/(str(name)+'.npz'),underwater=pred.cpu().numpy(),gt=gt.cpu().numpy(),clear_raw=o['clear_object_fullsh_raw'].detach().float().cpu().numpy())
   eval_rows.append(dict(scene=a.scene,arm=a.arm,view_id=str(name),render_seconds=seconds,**W._taba_rgb_metrics(pred,gt)))
  with (dest/'eval_metrics.csv').open('w') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(eval_rows[0]));writer.writeheader();writer.writerows(eval_rows)
  (dest/'camera_sequence.json').write_text(json.dumps(vrows,indent=2)+'\n')
  (dest/'complete.json').write_text(json.dumps(dict(complete=True,train_cameras=len(vrows),eval_cameras=len(eval_rows),fixed_float_regions=len(roi_names)))+'\n')
 finally:W._release(b)
if __name__=='__main__':main()
