"""Paired endpoint report; never choose a checkpoint from test performance."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_runs/geometry_gradient_A_20261021'
SCENES=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
KEYS=['PSNR','SSIM','LPIPS','MSE']
def main():
 rows=[];means={};protect=[]
 for scene in SCENES:
  means[scene]={}
  for arm in ['B0','A']:
   dest=OUT/'assets'/scene/arm
   assert json.loads((dest/'complete.json').read_text())['complete']
   views=list(csv.DictReader((dest/'eval_metrics.csv').open()))
   assert views
   rows.extend(views)
   means[scene][arm]={k:float(np.mean([float(r[k]) for r in views])) for k in KEYS}
  regions=json.loads((OUT/'regions'/scene/'regions.json').read_text())
  comparisons=OUT/'comparisons'/scene;comparisons.mkdir(parents=True,exist_ok=True)
  for asset in regions['assets']:
   name=asset['camera'];pairs={a:np.load(OUT/'assets'/scene/a/(name+'.npz')) for a in ['B0','A']}
   assert np.array_equal(pairs['B0']['gt'],pairs['A']['gt'])
   h,w=pairs['A']['clear'].shape[:2]
   # Original resolution clear, common display clamp, B0 above A.
   sheet=Image.new('RGB',(w,2*h))
   for j,a in enumerate(['B0','A']):sheet.paste(Image.fromarray((pairs[a]['clear']*255).round().astype('uint8')),(0,j*h))
   sheet.save(comparisons/(name+'_clear_B0_top_A_bottom.png'))
   for k,region in enumerate(asset['residual_review']+asset['structure_protection']):
    box=region['box'];x0,y0,x1,y1=box;crop=Image.new('RGB',(2*(x1-x0),y1-y0))
    for j,a in enumerate(['B0','A']):crop.paste(Image.fromarray((pairs[a]['clear'][y0:y1,x0:x1]*255).round().astype('uint8')),(j*(x1-x0),0))
    crop.save(comparisons/(name+f'_roi{k}_B0_left_A_right.png'))
   for k,r in enumerate(asset['structure_protection']):
    x0,y0,x1,y1=r['box']
    for a in ['B0','A']:
     pred=pairs[a]['underwater'][y0:y1,x0:x1];gt=pairs[a]['gt'][y0:y1,x0:x1]
     edge=lambda t:float(np.abs(np.diff(t,axis=0)).mean()+np.abs(np.diff(t,axis=1)).mean())
     protect.append(dict(scene=scene,camera=name,region=k,arm=a,MSE=float(np.square(pred-gt).mean()),clear_edge_proxy=edge(pairs[a]['clear'][y0:y1,x0:x1])))
   for pair in pairs.values():pair.close()
 macro={a:{k:float(np.mean([means[s][a][k] for s in SCENES])) for k in KEYS} for a in ['B0','A']}
 delta={k:macro['A'][k]-macro['B0'][k] for k in KEYS}
 guard=delta['PSNR']>=-.05 and all(means[s]['A']['PSNR']-means[s]['B0']['PSNR']>=-.10 for s in SCENES) and delta['SSIM']>=-.001 and delta['LPIPS']<=.002
 report=dict(per_scene=means,macro=macro,delta_A_minus_B0=delta,underwater_guard_pass=guard,residual_gate='N/A: frozen residual regions remain U; no independently verified O/F labels',promotion='NOT ESTABLISHED: requires visual structure review and reliable residual evidence',float_metrics='metrics.py, native resolution, equal scene macro mean')
 (OUT/'endpoint_report.json').write_text(json.dumps(report,indent=2)+'\n')
 for filename,data in [('endpoint_per_view.csv',rows),('protection_region_metrics.csv',protect)]:
  with (OUT/filename).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 lines=['# A 配对终点结果','', '| 场景 | B0 PSNR | A PSNR | 差值 |','|---|---:|---:|---:|']
 for s in SCENES:lines.append(f"| {s} | {means[s]['B0']['PSNR']:.6f} | {means[s]['A']['PSNR']:.6f} | {means[s]['A']['PSNR']-means[s]['B0']['PSNR']:+.6f} |")
 lines+=['',json.dumps(report,ensure_ascii=False,indent=2),'','残留区域为 U，不能据此宣称达到 20% 去残留门槛。投影梯度修复同时用于 B0/A，不能将其收益归于 A。']
 (OUT/'endpoint_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
