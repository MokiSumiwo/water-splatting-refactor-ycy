#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
W.EXPERIMENT='BND_ON_CICA_ABLATION'; W.ARMS=('A2',); W.AUXILIARY_APPEARANCE_ENABLED=False; W.AUXILIARY_APPEARANCE_STRENGTH=0.; W.OCMC_ENABLED=True; W.OCMC_STRENGTH=1.; W.OCMC_STRENGTH_START=1.; W.OCMC_STRENGTH_RAMP_STEPS=0; W.BRASH_ENABLED=False; W.BVAM_ENABLED=False; W.ENFORCE_SCENE_GPU_ASSIGNMENT=False; W.ACTIVE_SNAPSHOT_STEPS=(5000,10000,14999); W.ACTIVE_COMPACT_OUTPUT=False; W.TABA_DIAGNOSTIC_ENABLED=False; W.START_STEP=3000
p=argparse.ArgumentParser(); p.add_argument('--scene',required=True); p.add_argument('--gpu',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--source-config',type=Path,required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--cica-off',action='store_true'); a=p.parse_args()
W.CICA_ENABLED=not a.cica_off; W.ACTIVE_RENDER_ROOT=ROOT/'renders/bnd_on_cica_ablation_20260914'; W.run(a.scene,'A2',a.gpu,a.output_dir.resolve(),W.FINAL_STEP,source_root=a.source_root.resolve(),source_config=a.source_config.resolve(),parameterization='bounded_sh3',training_seed=a.seed,compact_output=False)
