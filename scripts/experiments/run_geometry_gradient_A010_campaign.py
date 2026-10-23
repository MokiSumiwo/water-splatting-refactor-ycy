"""Four raw seed42 runs of the pre-registered weak A ablation."""
import os,sys,json,subprocess,hashlib,shutil,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_runs/geometry_gradient_A010_20261022_v2'
PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES={'Curasao':'6','IUI3-RedSea':'7','JapaneseGradens-RedSea':'8','Panama':'9'}
def main():
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();(OUT/'code').mkdir()
 files=['scripts/experiments/run_geometry_gradient_A010_worker.py','scripts/experiments/run_geometry_gradient_A010_campaign.py','scripts/experiments/run_geometry_gradient_A_worker.py','scripts/experiments/run_direct_mdrr_cica_scene.py','scripts/experiments/geometry_projection_backend.py','water_splatting/geometry_gradient.py']
 hashes={}
 for f in files:
  p=ROOT/f;hashes[f]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,OUT/'code'/p.name)
 manifest=dict(status='TRAINING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),eta_max=.10,seed=42,steps_per_scene=15000,total_updates=60000,initialization='raw COLMAP and random medium, no previous checkpoint',reference_campaign='geometry_gradient_A_20261021',same_B0_reused=True,code_sha256=hashes,decision='User-authorized pre-registered weak intensity ablation. Fixed training views show clearest residual reduction in Curasao with foreground retained; other scenes less clear. No verified O/F labels or quantitative removal success claimed. Previously inspected test images are disclosed exploratory evidence, not used to tune eta=.10.')
 (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 def lane(item):
  scene,gpu=item;cmd=[PYTHON,'-m','scripts.experiments.run_geometry_gradient_A010_worker','--scene',scene,'--arm','A','--gpu',gpu,'--output',str(OUT/'formal'/scene/'A')]
  with (OUT/'logs'/f'{scene}_A.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CONDA_DEFAULT_ENV='water_splatting'),stdout=log,stderr=subprocess.STDOUT)
   (OUT/'logs'/f'{scene}_A.job.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2)+'\n');code=p.wait()
  return dict(scene=scene,exit_code=code)
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lane,SCENES.items()))
 manifest.update(status='TRAINED' if all(r['exit_code']==0 for r in rows) else 'FAILED',jobs=rows)
 (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
