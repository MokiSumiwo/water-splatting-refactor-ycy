#!/usr/bin/env python3
"""Strict paired BND-ON CICA ablation for scenes with a common step-3000 C0."""
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON=Path('/opt/anaconda3/envs/water_splatting/bin/python')
SOURCE=ROOT/'outputs/taba_paired_source_20260910_v2/bnd_on'
CONFIG=ROOT/'outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml'
SCENES=('JapaneseGradens-RedSea','Panama')
JOBS=(('JapaneseGradens-RedSea','6',False),('JapaneseGradens-RedSea','7',True),('Panama','8',False),('Panama','9',True))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
 (out/'logs').mkdir()
 jobs=[]; handles={}
 for scene,gpu,cica in JOBS:
  tag='cica_on' if cica else 'cica_off'; target=out/scene/tag; target.mkdir(parents=True)
  source_ckpt=SOURCE/scene/'checkpoints/C0/step-000003000.ckpt'
  if not source_ckpt.is_file(): raise FileNotFoundError(source_ckpt)
  log=(out/'logs'/f'{scene}_{tag}.log').open('w'); handles[tag+scene]=log
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CONDA_DEFAULT_ENV='water_splatting')
  cmd=[str(PYTHON),str(ROOT/'scripts/experiments/run_bnd_on_cica_ablation_worker.py'),'--scene',scene,'--gpu',gpu,'--output-dir',str(target),'--source-root',str(SOURCE),'--source-config',str(CONFIG),'--seed','42']
  if not cica: cmd += ['--cica-off']
  jobs.append((scene,tag,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)))
 status={(s,t):p.wait() for s,t,p in jobs}; [h.close() for h in handles.values()]
 (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 if any(status.values()): raise SystemExit(1)
 print(json.dumps(status))
if __name__=='__main__': main()
