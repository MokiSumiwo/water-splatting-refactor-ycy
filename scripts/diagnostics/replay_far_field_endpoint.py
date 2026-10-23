#!/usr/bin/env python3
"""Reload a retained candidate endpoint, including MST sidecar, and replay PNGs."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.run_fsb_cica_fork_worker import configure,W
from water_splatting.far_field_recovery_next import MediumScatteringTransport
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();result=json.loads((run/'result.json').read_text());scene=result['scene']
    assert os.environ.get('CONDA_DEFAULT_ENV')=='water_splatting'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+SCENES.index(scene))
    a.output.mkdir(parents=True,exist_ok=False)
    configure(scene,'legacy','legacy');W.ENFORCE_SCENE_GPU_ASSIGNMENT=False;W._route_renderer_backend('legacy')
    branch=W._new_branch(scene)
    try:
        W._load_source(branch,scene)
        endpoint=Path(result['checkpoint']);assert W._sha256(endpoint)==result['checkpoint_sha256']
        payload=torch.load(endpoint,map_location='cpu',weights_only=False)
        W._restore_training_checkpoint(branch,payload,True);model=branch.pipeline.model;model.eval()
        initial=W._model_state_hash(model);del payload
        original=model.get_outputs;transport=None
        if result.get('transport_checkpoint'):
            sidecar=torch.load(result['transport_checkpoint'],map_location='cpu',weights_only=False)
            transport=MediumScatteringTransport(sidecar['bounds'],model.device)
            transport.load_state_dict(sidecar['model'],strict=True);transport.eval()
            def transformed(camera,*args,**kwargs):
                saved={};call=model.underwater_rasterizer.rasterize
                def intercept(**kw):
                    r=call(**kw);saved.update(kwargs=kw,native=r);return r
                model.underwater_rasterizer.rasterize=intercept
                try:o=original(camera,*args,**kwargs)
                finally:model.underwater_rasterizer.rasterize=call
                return transport.transform(model,o,saved['kwargs'],saved['native'])
            model.get_outputs=transformed
        rows=[]
        with torch.no_grad():
            for _,view,cam,batch in W._eval_records(branch):
                o=model.get_outputs_for_camera(cam.to(model.device))
                directory=list((run/'renders'/scene).glob('step_*/'+str(view)+'/'+result['arm']))
                assert len(directory)==1
                for label,key in [('underwater','pred_image'),('clear_raw_display_clamp01','clear_object_fullsh_raw')]:
                    temp=a.output/(str(view)+'_'+label+'.png');W._save_native_rgb(temp,o[key])
                    old=np.asarray(Image.open(directory[0]/(label+'.png')),dtype=np.int16)
                    new=np.asarray(Image.open(temp),dtype=np.int16)
                    rows.append(dict(view=str(view),output=label,max_code_difference=int(np.abs(old-new).max()),
                                     different_channels=int((old!=new).sum())))
        assert W._model_state_hash(model)==initial and W._sha256(endpoint)==result['checkpoint_sha256']
        report=dict(completed=True,checkpoint_unchanged=True,model_unchanged=True,optimizer_steps=0,
                    transport_loaded=transport is not None,rows=rows,all_png_exact=all(r['max_code_difference']==0 for r in rows))
        W._write_json(a.output/'verification.json',report);print(json.dumps(report),flush=True)
    finally:W._release(branch)

if __name__=='__main__':main()
