#!/usr/bin/env python3
"""Raw B/N worker with the shared native responsibility event controller."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,choices=list(W.SCENE_GPUS))
    p.add_argument('--arm',required=True,choices=['B','N'])
    p.add_argument('--gpu',required=True)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--source-config',type=Path,default=ROOT/'research_runs/native_responsibility_raw_run_smoke_v3/source_config.yml')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args()
    W.RESPONSIBILITY_ENABLED=True
    W.RESPONSIBILITY_SMOKE=a.smoke
    if a.smoke:
        # Only compress event timing; exercise actual CICA updates in smoke.
        W.CICA_START_STEP=3
        W.CICA_RAMP_STEPS=1
        W.CICA_REFRESH_STEPS=(3, 9)
    W.RENDERER_BACKWARD_MODE='corrected_v1'
    if a.evaluate_only:
        W.RAW_DATA_ONLY=True
        W.AUXILIARY_APPEARANCE_ENABLED=False
        W.ACTIVE_SOURCE_CONFIG=a.output_dir.resolve()/'source_config.yml'
        W.ACTIVE_INITIALIZATION_DIR=a.output_dir.resolve()/'initialization'
        W.ACTIVE_TRAINING_SEED=0
        W.FINAL_STEP=15000
        W._route_renderer_backend('corrected_v1')
        W._runtime(a.scene,a.gpu)
        W._seed_all(0)
        W._evaluate_arm(a.scene,a.arm,a.output_dir.resolve(),a.output_dir.resolve()/'renders'/a.scene)
        print('evaluated',a.scene,a.arm,15000,flush=True)
        return
    result=W.run(a.scene,a.arm,a.gpu,a.output_dir.resolve(),max_step=16 if a.smoke else 15000,
        source_config=a.source_config.resolve(),training_seed=0,raw_data_only=True,
        resume_checkpoint=a.resume_checkpoint)
    print('completed',result['scene'],result['arm'],result['final_step'],flush=True)

if __name__=='__main__': main()
