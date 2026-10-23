#!/usr/bin/env python3
import os,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; PY='/opt/anaconda3/envs/water_splatting/bin/python'; scenes=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
def main():
 out=ROOT/'outputs/cvc_front_surface_audit_20260915'; out.mkdir(parents=True,exist_ok=False); jobs=[]
 for s,g in zip(scenes,['6','7','8','9']):
  t=out/s; h=(out/(s+'.log')).open('w'); e=dict(os.environ,CUDA_VISIBLE_DEVICES=g,CONDA_DEFAULT_ENV='water_splatting'); jobs.append((s,subprocess.Popen([PY,str(ROOT/'scripts/diagnostics/run_cvc_front_surface_audit.py'),'--scene',s,'--output-dir',str(t)],cwd=ROOT,env=e,stdout=h,stderr=subprocess.STDOUT),h))
 status={s:p.wait() for s,p,_ in jobs}
 for _,_,h in jobs:h.close()
 (out/'status.json').write_text(json.dumps(status,indent=2)+'\n'); print(status)
if __name__=='__main__': main()
