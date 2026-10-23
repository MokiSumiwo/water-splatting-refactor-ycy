"""Paired B0/A from raw COLMAP, with identical baseline and corrected projection."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W
from water_splatting.geometry_gradient import GeometryGradient

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--arm',choices=['B0','A'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 W.EXPERIMENT='GEOMETRY_GRADIENT_A';W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
 W.RENDERER_BACKWARD_MODE='corrected_v1';W.RESPONSIBILITY_ENABLED=False
 W.AUXILIARY_APPEARANCE_ENABLED=False;W.ACTIVE_MDRR_OVERRIDE=False;W.ACTIVE_CICA_OVERRIDE=True
 W.START_STEP=0;W.ACTIVE_RENDER_ROOT=a.output/'renders'
 original_route=W._route_renderer_backend
 def route(mode):
  result=original_route(mode)
  from scripts.experiments.geometry_projection_backend import install
  install();result['projection_backward']='clip_chain_rule_A_20261021';return result
 W._route_renderer_backend=route
 # Raw runs normally save five intermediate checkpoints. Limit this campaign to
 # one rolling-resume point plus endpoint to fit the registered disk budget.
 W._active_snapshots=lambda max_step: tuple(x for x in [10000,max_step] if x<=max_step)
 W._evaluate_checkpoint_original=W._evaluate_checkpoint
 W.MIC._metric_images=lambda model,pred,gt: W._taba_rgb_metrics(pred,gt)
 controllers={}
 def hook(branch,outputs,loss,step,output):
  key=id(branch)
  if key not in controllers:controllers[key]=GeometryGradient(W,branch,output,a.arm=='A',a.smoke)
  controllers[key].apply(branch.pipeline.model,outputs,loss,step)
 W.GEOMETRY_GRADIENT_HOOK=hook
 if a.smoke:
  W.CICA_START_STEP=3;W.CICA_REFRESH_STEPS=(3,9)
 result=W.run(a.scene,'B',a.gpu,a.output.resolve(),max_step=20 if a.smoke else 15000,source_config=ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene/'source_config.yml',training_seed=42,raw_data_only=True)
 print(json.dumps(dict(scene=a.scene,arm=a.arm,final_step=result['final_step'])),flush=True)
if __name__=='__main__':main()
