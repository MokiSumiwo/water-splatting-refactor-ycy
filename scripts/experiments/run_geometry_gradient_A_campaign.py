"""Launch the registered four-scene paired A campaign, preserving all artifacts."""
import os,sys,json,subprocess,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_runs/geometry_gradient_A_20261021'
PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES={'Curasao':'6','IUI3-RedSea':'7','JapaneseGradens-RedSea':'8','Panama':'9'}
def main():
 checks={}
 for scene in SCENES:
  f=OUT/'checks'/('IUI3_projection_fix.json' if scene=='IUI3-RedSea' else scene+'_fixed_vjp.json')
  checks[scene]=json.loads(f.read_text())['passed']
  smoke=json.loads((OUT/'smoke'/scene/'A/final_summary.json').read_text())
  assert smoke['completed_updates']==20 and smoke['smoke_reload_audit']['model_hash_match']
 assert all(checks.values())
 formal=OUT/'formal'
 if formal.exists():raise FileExistsError(formal)
 formal.mkdir();(OUT/'logs').mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
 files=['scripts/experiments/run_geometry_gradient_A_worker.py','scripts/experiments/run_direct_mdrr_cica_scene.py','scripts/experiments/geometry_projection_backend.py','water_splatting/geometry_gradient.py']
 hashes={}
 for f in files:
  src=ROOT/f;hashes[f]=hashlib.sha256(src.read_bytes()).hexdigest();shutil.copy2(src,OUT/'code'/src.name)
 manifest=dict(status='TRAINING',scenes=SCENES,arms=['B0','A'],seed=42,updates_per_job=15000,total_formal_updates=120000,smoke_updates=80,
  appearance='bounded_sh3',M1=True,OCMC=True,CICA=True,FSB=False,auxiliary=False,RAOC=False,renderer='corrected_v1 plus separately verified projection clamp VJP; native forward unchanged',
  schedule='eta .25; 500-2000 ramp; 2000-8000 hold; 8000-10000 decay; zero thereafter',
  structure_support='train-only COLMAP tracks >=3 views, error <=1px; center distance and 3sigma radius <= local 4-NN spacing',
  ROI_limitation='Review and protection regions frozen before treatment outputs. Residual regions remain U where independent F/O evidence is unavailable; cannot claim 20% water-removal gate passed from chroma alone.',
  checkpoint_steps=[0,10000,15000],metrics='metrics.py on native float tensors; exact same protocol for B0 and A',code_sha256=hashes,checks=checks)
 (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 def lane(item):
  scene,gpu=item;results=[]
  for arm in ['B0','A']:
   target=formal/scene/arm;log=OUT/'logs'/f'{scene}_{arm}.log'
   cmd=[PYTHON,str(ROOT/'scripts/experiments/run_geometry_gradient_A_worker.py'),'--scene',scene,'--arm',arm,'--gpu',gpu,'--output',str(target)]
   with log.open('w') as stream:
    p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CONDA_DEFAULT_ENV='water_splatting'),stdout=stream,stderr=subprocess.STDOUT)
    (OUT/'logs'/f'{scene}_{arm}.job.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2)+'\n')
    code=p.wait()
   results.append(dict(scene=scene,arm=arm,exit_code=code))
   if code:break
  return results
 with ThreadPoolExecutor(max_workers=4) as pool:results=sum(list(pool.map(lane,SCENES.items())),[])
 manifest.update(status='TRAINED' if len(results)==8 and all(r['exit_code']==0 for r in results) else 'FAILED',jobs=results)
 (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
