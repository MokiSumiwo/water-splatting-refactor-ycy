#!/usr/bin/env python3
"""Matched two-parameter camera/water PSF training on frozen direct layers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import torch

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES,origin,save_rgb,write_csv
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment,gpu_processes,read_json,sha256,source_assets,write_json
from scripts.experiments.run_world_medium_screen import capture_view
from water_splatting.rendering.forward_scattering import SIGMAS,ScatteringPSF,blur_layers

OUT=ROOT/"outputs/transport_candidates_20260911/04_forward_scattering"


@torch.no_grad()
def depth_partition(worker,branch):
    depths=[]
    for _,_,camera,batch in worker._train_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        valid=k["radii"]>0
        depths.append(k["depths"][valid][::32].float())
    population=torch.cat(depths)
    edges=torch.quantile(population,population.new_tensor([1/3,2/3]))
    index=torch.bucketize(population,edges)
    centers=torch.stack([population[index==i].median() for i in range(3)])
    return edges,centers/centers[1]


@torch.no_grad()
def direct_layers(model,k,native,edges):
    index=torch.bucketize(k["depths"].contiguous(),edges)
    layers=[]
    for group in range(3):
        colors=k["colors"]*(index==group)[:,None]
        rendered=model.underwater_rasterizer.rasterize(**dict(k,colors=colors))
        layers.append(rendered.rgb_object.permute(2,0,1))
    layers=torch.stack(layers)
    error=float((layers.sum(0).permute(1,2,0)-native.rgb_object).abs().max())
    assert error<2e-6,error
    return layers,error


@torch.no_grad()
def training_cache(worker,branch,edges,directory):
    rows=[];chunks=[]
    for index,view,camera,batch in worker._train_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        layers,error=direct_layers(branch.pipeline.model,k,native,edges)
        generator=torch.Generator().manual_seed(43000+int(index))
        pixels=torch.randperm(k["width"]*k["height"],generator=generator)[:4096].cuda()
        bases=[]
        for sigma in SIGMAS:
            blurred=blur_layers(layers,sigma)
            bases.append(blurred.reshape(3,3,-1).permute(2,0,1)[pixels])
        basis=torch.stack(bases,2)
        chunks.append(dict(basis=basis,base=outputs["pred_image"].reshape(-1,3)[pixels],
                           direct=native.rgb_object.reshape(-1,3)[pixels],gt=gt.reshape(-1,3)[pixels]))
        rows.append(dict(view=view,pixels=pixels.cpu().tolist(),layer_equivalence_max=error))
        print(json.dumps(dict(event="cache",view=view)),flush=True)
    write_json(directory/"training_ray_manifest.json",dict(rows=rows))
    return {key:torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def train(arm,depths,data,directory):
    model=ScatteringPSF(depths,arm=="water_psf")
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    generator=torch.Generator().manual_seed(42)
    rows=[]
    for step in range(1,5001):
        index=torch.randint(data["gt"].shape[0],(512,),generator=generator).cuda()
        optimizer.zero_grad()
        prediction=data["base"][index]+model(data["basis"][index])-data["direct"][index]
        loss=(prediction-data["gt"][index]).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.raw.grad).all()
        optimizer.step()
        if step==1 or step%500==0:
            rows.append(dict(step=step,mse=float(loss.detach())))
    result=dict(steps=5000,training=rows,scattered_fraction=float(model.raw[0].sigmoid()),
                sigma_scale=float(8*model.raw[1].sigmoid()),weights=model.kernel_weights().detach().cpu().tolist(),parameter_count=2)
    torch.save(model.state_dict(),directory/(arm+".pt"))
    write_json(directory/(arm+"_training.json"),result)
    print(json.dumps(dict(event="trained",arm=arm,**result)),flush=True)
    return model.eval()


@torch.no_grad()
def evaluate(worker,branch,edges,models,directory):
    rows=[]
    for _,view,camera,batch in worker._eval_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        layers,error=direct_layers(branch.pipeline.model,k,native,edges)
        sums={arm:torch.zeros_like(layers[0]) for arm in models}
        started=time.perf_counter()
        weights={arm:model.kernel_weights() for arm,model in models.items()}
        for j,sigma in enumerate(SIGMAS):
            blurred=blur_layers(layers,sigma)
            for arm in models:
                sums[arm]+=(blurred*weights[arm][:,j,None,None,None]).sum(0)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        original=outputs["pred_image"].float().clamp(0,1)
        base=worker._taba_rgb_metrics(original,gt)
        rows.append(dict(scene=directory.name,view_id=view,arm="original",**base))
        target=directory/"renders"/view;target.mkdir(parents=True)
        source=origin(directory.name,"E0")/"renders/E0"/directory.name/"step_14999"/view/"A2"
        for name,source_name in (("underwater_original.png","underwater.png"),("clear_original.png","clear_raw_display_clamp01.png"),("gt.png","gt.png")):
            (target/name).symlink_to(source/source_name)
        for arm,value in sums.items():
            prediction=outputs["pred_image"]+value.permute(1,2,0)-native.rgb_object
            metrics=worker._taba_rgb_metrics(prediction.clamp(0,1),gt)
            row=dict(scene=directory.name,view_id=view,arm=arm,**metrics,delta_psnr=metrics["PSNR"]-base["PSNR"],
                     shared_basis_eval_seconds=seconds,layer_equivalence_max=error)
            rows.append(row)
            save_rgb(target/(arm+".png"),prediction)
            save_rgb(target/(arm+"_difference_x20.png"),20*(prediction-original).abs())
            print(json.dumps(dict(event="eval",**row)),flush=True)
    write_csv(directory/"per_view.csv",rows)
    return rows


def run_scene(args):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==args.gpu
    directory=args.output_root/args.scene;directory.mkdir(parents=True,exist_ok=False)
    worker=configure_worker("E0")
    worker.ACTIVE_SOURCE_ROOT,worker.ACTIVE_SOURCE_CONFIG,_=source_assets(args.scene)
    worker._runtime(args.scene,args.gpu)
    branch=worker._new_branch(args.scene)
    checkpoint=origin(args.scene,"E0")/args.scene/"E0/checkpoints/step-000014999.ckpt"
    digest=sha256(checkpoint)
    try:
        worker._load_snapshot(branch,args.scene,checkpoint)
        before=worker._model_state_hash(branch.pipeline.model)
        edges,depths=depth_partition(worker,branch)
        write_json(directory/"depth_partition.json",dict(edges=edges.tolist(),relative_centers=depths.tolist()))
        data=training_cache(worker,branch,edges,directory)
        models={arm:train(arm,depths,data,directory) for arm in ("camera_psf","water_psf")}
        del data
        rows=evaluate(worker,branch,edges,models,directory)
        assert before==worker._model_state_hash(branch.pipeline.model) and digest==sha256(checkpoint)
        write_json(directory/"result.json",dict(complete=True,rows=rows,original_checkpoint_unchanged=True,
            frozen_gaussians=True,clear_unchanged_by_construction=True,checkpoint=str(checkpoint),checkpoint_sha256=digest))
    finally:worker._release(branch)


def launch(args):
    args.output_root.mkdir(parents=True,exist_ok=False)
    assert not any(gpu_processes().values())
    write_json(args.output_root/"registration.json",dict(candidate=4,steps=5000,lr=.01,batch=512,train_pixels_per_camera=4096,
        psf_widths=SIGMAS,depth_bins=3,arms=["camera_psf","water_psf"],parameters_per_arm=2,
        approximation="Convolution of visible direct layers, not complete multiple scattering or hidden-surface transport",
        promotion="water-camera mean PSNR >=0.05, no scene below -0.1, mean LPIPS nonworse, absolute original safety"))
    jobs=[]
    for scene,gpu in zip(SCENES,("6","7","8","9")):
        handle=(args.output_root/(scene+".log")).open("x")
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--scene",scene,"--gpu",gpu,"--output-root",str(args.output_root)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="4"),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        jobs.append((scene,process,handle))
    while any(p.poll() is None for _,p,_ in jobs):
        write_json(args.output_root/"queue_status.json",{s:p.poll() for s,p,_ in jobs});time.sleep(5)
    status={s:p.returncode for s,p,_ in jobs}
    for _,_,handle in jobs:handle.close()
    write_json(args.output_root/"queue_status.json",status)
    assert not any(status.values()),status
    rows=[r for scene in SCENES for r in read_json(args.output_root/scene/"result.json")["rows"]]
    means=[]
    for scene in SCENES:
        for arm in ("original","camera_psf","water_psf"):
            selected=[r for r in rows if r["scene"]==scene and r["arm"]==arm]
            means.append(dict(scene=scene,arm=arm,**{key:statistics.mean(r[key] for r in selected) for key in ("PSNR","SSIM","LPIPS","MSE")}))
    deltas=[]
    for scene in SCENES:
        by_arm={r["arm"]:r for r in means if r["scene"]==scene}
        deltas.append(dict(scene=scene,psnr_water_minus_camera=by_arm["water_psf"]["PSNR"]-by_arm["camera_psf"]["PSNR"],
            lpips_water_minus_camera=by_arm["water_psf"]["LPIPS"]-by_arm["camera_psf"]["LPIPS"],
            psnr_water_minus_original=by_arm["water_psf"]["PSNR"]-by_arm["original"]["PSNR"]))
    passed=statistics.mean(r["psnr_water_minus_camera"] for r in deltas)>=.05 and min(r["psnr_water_minus_camera"] for r in deltas)>=-.1 and statistics.mean(r["lpips_water_minus_camera"] for r in deltas)<=0 and min(r["psnr_water_minus_original"] for r in deltas)>=-.1
    write_csv(args.output_root/"scene_means.csv",means)
    write_json(args.output_root/"summary.json",dict(complete=True,means=means,deltas=deltas,promotion_gate=passed,
        clear_accuracy_claim=False,sota_claim=False))
    print(json.dumps(dict(complete=True,deltas=deltas,promotion_gate=passed)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",choices=SCENES);parser.add_argument("--gpu",choices=("6","7","8","9"))
    parser.add_argument("--output-root",type=Path,default=OUT)
    args=parser.parse_args();args.output_root=args.output_root.resolve()
    check_environment()
    run_scene(args) if args.scene else launch(args)
