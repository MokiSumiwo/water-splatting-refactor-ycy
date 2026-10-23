"""Export weak A and compare to immutable previous paired B0 and A=.25."""
import os,json,time,subprocess,csv,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.experiments import summarize_geometry_gradient_A as summary
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_runs/geometry_gradient_A010_20261022_v2';OLD=ROOT/'research_runs/geometry_gradient_A_20261021';PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
def lane(item):
 scene,gpu=item
 while not (OUT/'formal'/scene/'A/final_summary.json').exists():
  if json.loads((OUT/'run_manifest.json').read_text())['status']=='FAILED':raise RuntimeError('Training campaign failed')
  time.sleep(20)
 with (OUT/'logs'/f'{scene}_export.log').open('w') as log:
  subprocess.run([PYTHON,str(ROOT/'scripts/experiments/export_geometry_gradient_A.py'),'--scene',scene,'--arm','A','--campaign',str(OUT)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
 dest=OUT/'assets'/scene/'B0';dest.symlink_to(OLD/'assets'/scene/'B0',target_is_directory=True)
 return scene
def main():
 try:
  (OUT/'regions').symlink_to(OLD/'regions',target_is_directory=True)
  with ThreadPoolExecutor(max_workers=4) as pool:scenes=list(pool.map(lane,zip(summary.SCENES,range(4))))
  summary.OUT=OUT;summary.main()
  comparison={}
  for scene in scenes:
   comparison[scene]={}
   for arm,folder in [('B0',OLD/'assets'/scene/'B0'),('A025',OLD/'assets'/scene/'A'),('A010',OUT/'assets'/scene/'A')]:
    rows=list(csv.DictReader((folder/'eval_metrics.csv').open()));comparison[scene][arm]={key:sum(float(r[key]) for r in rows)/len(rows) for key in summary.KEYS}
  (OUT/'three_arm_metrics.json').write_text(json.dumps(comparison,indent=2)+'\n')
  (OUT/'postprocess_status.json').write_text(json.dumps(dict(status='COMPLETE',scenes=scenes))+'\n')
 except Exception as e:
  (OUT/'postprocess_status.json').write_text(json.dumps(dict(status='FAILED',error=repr(e)))+'\n');raise
if __name__=='__main__':main()
