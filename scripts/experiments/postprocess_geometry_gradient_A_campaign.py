"""Wait for each endpoint, export on spare GPUs, then aggregate paired results."""
import os,sys,json,time,subprocess,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_runs/geometry_gradient_A_20261021'
PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
def lane(item):
 scene,gpu=item;rows=[]
 for arm in ['B0','A']:
  target=OUT/'formal'/scene/arm/'final_summary.json'
  while not target.exists():
   manifest=json.loads((OUT/'run_manifest.json').read_text())
   if manifest['status']=='FAILED':raise RuntimeError(f'Campaign failed while waiting for {scene}/{arm}')
   time.sleep(20)
  done=OUT/'assets'/scene/arm/'complete.json'
  if done.exists():rows.append(dict(scene=scene,arm=arm,already_exported=True));continue
  command=[PYTHON,str(ROOT/'scripts/experiments/export_geometry_gradient_A.py'),'--scene',scene,'--arm',arm]
  with (OUT/'logs'/f'{scene}_{arm}_export.log').open('w') as stream:
   result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=stream,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'Export failed {scene}/{arm}, exit={result.returncode}')
  rows.append(dict(scene=scene,arm=arm,exported=True))
 return rows
def main():
 snapshot=OUT/'code/postprocess';snapshot.mkdir(exist_ok=True)
 hashes={}
 for name in ['export_geometry_gradient_A.py','summarize_geometry_gradient_A.py','postprocess_geometry_gradient_A_campaign.py']:
  f=ROOT/'scripts/experiments'/name;shutil.copy2(f,snapshot/name);hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest()
 (snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 try:
  with ThreadPoolExecutor(max_workers=4) as pool:rows=sum(list(pool.map(lane,zip(SCENES,range(4)))),[])
  subprocess.run([PYTHON,str(ROOT/'scripts/experiments/summarize_geometry_gradient_A.py')],cwd=ROOT,check=True)
  (OUT/'postprocess_status.json').write_text(json.dumps(dict(status='COMPLETE',jobs=rows),indent=2)+'\n')
 except Exception as e:
  (OUT/'postprocess_status.json').write_text(json.dumps(dict(status='FAILED',error=repr(e)),indent=2)+'\n');raise
if __name__=='__main__':main()
