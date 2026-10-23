#!/usr/bin/env python3
"""Strict four-scene SBR continuation from the registered BND-ON C0 state."""
import argparse, os, subprocess, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PY=Path('/opt/anaconda3/envs/water_splatting/bin/python')
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
GPUS=('6','7','8','9')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-step',type=int,default=4999); a=p.parse_args()
 out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False); (out/'logs').mkdir()
 source_map={
  'Curasao':ROOT/'outputs/bnd_on_cica_source_extra_20260914',
  'IUI3-RedSea':ROOT/'outputs/bnd_on_cica_source_extra_20260914',
  'JapaneseGradens-RedSea':ROOT/'outputs/taba_paired_source_20260910_v2/bnd_on',
  'Panama':ROOT/'outputs/taba_paired_source_20260910_v2/bnd_on'}
 cfg_map={s:(ROOT/'outputs/taba_chromatic_bnd_on_20260910'/s/'E0/source_config.yml' if s in ('Curasao','IUI3-RedSea') else ROOT/'outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml') for s in SCENES}
 jobs=[]
 for scene,gpu in zip(SCENES,GPUS):
  target=out/scene; target.mkdir(); log=(out/'logs'/f'{scene}.log').open('w')
  # The worker reads module switches from its imported module; use a tiny
  # protocol shim so SBR is exactly CICA-only, matching the FSB baseline.
  cmd=[str(PY),str(ROOT/'scripts/experiments/run_sbr_worker.py'),'--scene',scene,'--gpu',gpu,'--output-dir',str(target),'--max-step',str(a.max_step),'--source-root',str(source_map[scene]),'--source-config',str(cfg_map[scene])]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CONDA_DEFAULT_ENV='water_splatting',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  jobs.append((scene,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
 status={s:p.wait() for s,p,_ in jobs}
 for _,_,h in jobs:h.close()
 (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 if any(status.values()): raise SystemExit(1)
 print(json.dumps(status))
if __name__=='__main__':main()
