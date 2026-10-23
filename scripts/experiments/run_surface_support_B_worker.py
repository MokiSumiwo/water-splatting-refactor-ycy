"""Independent B arm on frozen B0; no A geometry-gradient routing."""
import argparse
from pathlib import Path
from scripts.experiments import run_geometry_gradient_A_worker as base
from water_splatting.surface_support import SurfaceSupport


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--support',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--disabled',action='store_true');a=p.parse_args()
    W=base.W
    W.EXPERIMENT='SURFACE_SUPPORT_B_CENTER_ONLY';W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
    W.RENDERER_BACKWARD_MODE='corrected_v1';W.RESPONSIBILITY_ENABLED=False
    W.AUXILIARY_APPEARANCE_ENABLED=False;W.ACTIVE_MDRR_OVERRIDE=False;W.ACTIVE_CICA_OVERRIDE=True
    W.START_STEP=0;W.ACTIVE_RENDER_ROOT=a.output/'renders'
    original=W._route_renderer_backend
    def route(mode):
        result=original(mode)
        from scripts.experiments.geometry_projection_backend import install
        install();result['projection_backward']='clip_chain_rule_A_20261021';return result
    W._route_renderer_backend=route
    W._active_snapshots=lambda final:tuple(x for x in [10000,final] if x<=final)
    W.MIC._metric_images=lambda model,pred,gt:W._taba_rgb_metrics(pred,gt)
    controllers={}
    def hook(branch,outputs,loss,step,output):
        key=id(branch)
        if key not in controllers:controllers[key]=SurfaceSupport(branch,output,a.support,enabled=not a.disabled,smoke=a.smoke)
        controllers[key].apply(branch.pipeline.model,outputs,loss,step)
    W.GEOMETRY_GRADIENT_HOOK=hook
    if a.smoke:W.CICA_START_STEP=3;W.CICA_REFRESH_STEPS=(3,9)
    W.run(a.scene,'B',a.gpu,a.output.resolve(),max_step=20 if a.smoke else 15000,
          source_config=base.ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene/'source_config.yml',
          training_seed=42,raw_data_only=True)

if __name__=='__main__':main()
