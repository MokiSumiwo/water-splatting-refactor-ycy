#!/usr/bin/env python3
"""Train three medium representations with identical frozen Gaussian rays."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.cpp_extension import load

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, origin, save_rgb, write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from water_splatting.rendering.segment_transport import ConstantMedium, RayMediumMLP, WorldMediumGrid, composite_fixed_rays

OUT = ROOT / "outputs/transport_candidates_20260911/02_world_medium"
ARMS = ("constant", "ray_mlp", "world_grid")


def extension():
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    return load(name="fixed_ray_extraction_v1", sources=[str(ROOT / "scripts/diagnostics/csrc/extract_fixed_rays.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"], verbose=False)


def extract(ext, kwargs, bins, pixels):
    assert pixels.numel() > 0 and pixels.min() >= 0 and pixels.max() < kwargs["height"]*kwargs["width"]
    capacity = 64
    while True:
        ids, weights, counts = ext.forward(pixels.contiguous(), bins[0], bins[1], kwargs["xys"], kwargs["conics"],
            kwargs["opacities"].reshape(-1), kwargs["height"], kwargs["width"], capacity)
        maximum = int(counts.max())
        if maximum <= capacity:
            size = max(1, maximum)
            ids, weights = ids[:, :size].long(), weights[:, :size]
            depths = kwargs["depths"][ids]
            return dict(depths=torch.where(weights > 0, depths, 0.), weights=weights, colors=kwargs["colors"][ids]), maximum
        capacity = 1 << (maximum-1).bit_length()
        if capacity > 8192:
            raise RuntimeError("Ray support exceeds 8192; no contributions were silently dropped")


def ray_coordinates(camera, pixels, width):
    rotation = camera.camera_to_worlds[0, :3, :3] @ torch.diag(pixels.new_tensor([1., -1., -1.], dtype=torch.float32))
    origin = camera.camera_to_worlds[0, :3, 3]
    xy = torch.stack(((pixels % width-camera.cx.item())/camera.fx.item(),
                      (pixels // width-camera.cy.item())/camera.fy.item()), -1)
    local = torch.cat((xy, torch.ones_like(xy[:, :1])), -1)
    world = local @ rotation.T
    normalized_xy = torch.stack((2*(pixels % width)/max(1,width-1)-1,
                                 2*(pixels // width)/max(1,int(camera.height.item())-1)-1), -1)
    return dict(origins=origin.expand(pixels.numel(), 3), directions=world,
                ray_norm=local.norm(dim=-1), image_xy=normalized_xy)


@torch.no_grad()
def capture_view(worker, model, camera, batch):
    captured = {}
    original = model.underwater_rasterizer.rasterize
    def intercept(**kwargs):
        result = original(**kwargs)
        captured.update(kwargs=kwargs, native=result)
        return result
    with patch.object(model.underwater_rasterizer, "rasterize", side_effect=intercept):
        outputs = model.get_outputs_for_camera(camera.to(model.device))
    gt = worker._gt(model, worker._batch_to_device(batch.copy(), model.device), outputs["background"]).float().clamp(0, 1)
    return captured["kwargs"], captured["native"], outputs, gt


def prepare_context(data, center, extent, fallback_far):
    center, extent = center.to(data["origins"]), extent.to(data["origins"])
    origin_normalized = (data["origins"]-center)/extent
    direction = data["directions"]/data["ray_norm"][:, None]
    xy = data["image_xy"]
    data["context"] = torch.cat((origin_normalized, direction, xy, xy.norm(dim=-1, keepdim=True)), -1)
    data["far"] = data["depths"].max(1).values.clamp_min(fallback_far)
    return data


def predict(medium, data, steps=64):
    fractions = (torch.arange(steps, device=data["depths"].device, dtype=torch.float32)+.5)/steps
    points = data["origins"][:, None]+data["directions"][:, None]*(data["far"][:, None]*fractions)[..., None]
    raw = medium(points, data["context"])
    return composite_fixed_rays(data["depths"], data["weights"], data["colors"], raw, data["far"], data["ray_norm"])


@torch.no_grad()
def cache_training(worker, branch, ext, directory, rays_per_view):
    model = branch.pipeline.model
    chunks, metadata, activations = [], [], []
    for camera_index, view, camera, batch in worker._train_records(branch):
        k, native, outputs, gt = capture_view(worker, model, camera, batch)
        total = k["height"]*k["width"]
        generator = torch.Generator(device="cpu").manual_seed(42100+int(camera_index))
        pixels = torch.randperm(total, generator=generator)[:min(rays_per_view,total)].to(model.device)
        data, maximum = extract(ext, k, _geometry_bins(k), pixels)
        data.update(ray_coordinates(camera.to(model.device), pixels, k["width"]))
        data["gt"] = gt.reshape(-1,3)[pixels]
        b = k["medium_attn"].reshape(-1,3)[pixels]
        bb = k["medium_bs"].reshape(-1,3)[pixels]
        ambient = k["medium_rgb"].reshape(-1,3)[pixels]
        original_direct = (data["weights"][...,None]*data["colors"]*(-b[:,None]*data["depths"][...,None]).exp()).sum(1)
        original_clear = (data["weights"][...,None]*data["colors"]).sum(1)
        direct_error = float((original_direct-native.rgb_object.reshape(-1,3)[pixels]).abs().max())
        clear_error = float((original_clear-native.j_raw.reshape(-1,3)[pixels]).abs().max())
        assert max(direct_error,clear_error) < 2e-6, (view,direct_error,clear_error)
        # Original beta uses z distance; divide by ray norm for a physical-length initialization.
        activations.append(torch.cat((ambient, bb/data["ray_norm"][:,None], b/data["ray_norm"][:,None]),-1).cpu())
        chunks.append({key:value.detach().float().cpu() for key,value in data.items()})
        metadata.append(dict(view=view,camera_index=camera_index,rays=pixels.numel(),max_contributors=maximum,
                             pixels=pixels.cpu().tolist(),native_direct_max_error=direct_error,native_clear_max_error=clear_error))
        print(json.dumps(dict(event="cache",view=view,rays=pixels.numel(),max_contributors=maximum)),flush=True)
    maximum = max(chunk["weights"].shape[1] for chunk in chunks)
    for chunk in chunks:
        padding = maximum-chunk["weights"].shape[1]
        for key in ("weights","depths"):
            chunk[key]=F.pad(chunk[key],(0,padding))
        chunk["colors"]=F.pad(chunk["colors"],(0,0,0,padding))
    data={key:torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    fallback_far=float(data["depths"].max(1).values.median().clamp_min(.1))
    endpoints=data["origins"]+data["directions"]*data["depths"].max(1).values.clamp_min(fallback_far)[:,None]
    locations=torch.cat((endpoints,data["origins"]))
    lower,upper=torch.quantile(locations,.01,dim=0),torch.quantile(locations,.99,dim=0)
    center=(lower+upper)*.5
    extent=((upper-lower)*.55).clamp_min(.1)
    data=prepare_context(data,center,extent,fallback_far)
    physical=torch.cat(activations).mean(0)
    initial=torch.cat((torch.logit(physical[:3].clamp(1e-5,1-1e-5)),torch.log(torch.expm1(physical[3:].clamp_min(1e-5)))))
    write_json(directory/"training_ray_manifest.json",dict(rows=metadata,total_rays=data["gt"].shape[0],
        capacity=maximum,center=center.tolist(),half_extent=extent.tolist(),fallback_far=fallback_far,initial_raw=initial.tolist()))
    return data,initial,center,extent,fallback_far


def train_arm(arm,data,initial,center,extent,steps,directory):
    torch.manual_seed(42)
    model={"constant":lambda:ConstantMedium(initial),"ray_mlp":lambda:RayMediumMLP(initial),
           "world_grid":lambda:WorldMediumGrid(initial,center,extent)}[arm]().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    generator=torch.Generator(device="cpu").manual_seed(42)
    size=data["gt"].shape[0]
    rows=[]
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        indices=torch.randint(size,(256,),generator=generator).cuda()
        batch={key:value[indices] for key,value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction=predict(model,batch)
        loss=(prediction-batch["gt"]).square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
        if step==1 or step%250==0 or step==steps:
            row=dict(arm=arm,step=step,mse=float(loss.detach()),elapsed_seconds=time.perf_counter()-started)
            rows.append(row)
            print(json.dumps(dict(event="train",**row)),flush=True)
    model.eval()
    mse_total=0.
    with torch.no_grad():
        for start in range(0,size,1024):
            batch={key:value[start:start+1024] for key,value in data.items()}
            mse_total+=float((predict(model,batch)-batch["gt"]).square().sum())
    target=directory/arm
    target.mkdir()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),steps=steps,seed=42,
                    initial_raw=initial,center=center,extent=extent),target/"medium_checkpoint.pt")
    write_json(target/"training.json",dict(complete=True,steps=steps,loss="RGB MSE",train_mse=mse_total/(size*3),
        rows=rows,wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        parameter_count=sum(p.numel() for p in model.parameters())))
    return model


@torch.no_grad()
def evaluate(worker,branch,ext,models,center,extent,fallback,directory):
    rows=[]
    for index,view,camera,batch in worker._eval_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        bins=_geometry_bins(k)
        total=k["width"]*k["height"]
        rendered={arm:torch.empty(total,3,device="cuda") for arm in ARMS}
        times={arm:0. for arm in ARMS}
        quadrature={arm:[] for arm in ARMS}
        native_error=0.
        started=time.perf_counter()
        for start in range(0,total,4096):
            pixels=torch.arange(start,min(start+4096,total),device="cuda",dtype=torch.int64)
            data,_=extract(ext,k,bins,pixels)
            data.update(ray_coordinates(camera.to("cuda"),pixels,k["width"]))
            data=prepare_context(data,center,extent,fallback)
            clear=(data["weights"][...,None]*data["colors"]).sum(1)
            native_error=max(native_error,float((clear-native.j_raw.reshape(-1,3)[pixels]).abs().max()))
            for arm,medium in models.items():
                torch.cuda.synchronize()
                before=time.perf_counter()
                prediction=predict(medium,data)
                torch.cuda.synchronize()
                times[arm]+=time.perf_counter()-before
                rendered[arm][pixels]=prediction
                if start==0:
                    q128=predict(medium,data,128)
                    quadrature[arm]=[float((q128-prediction).abs().mean()),float((q128-prediction).abs().max())]
        assert native_error<2e-6,(view,native_error)
        target=directory/"renders"/view
        target.mkdir(parents=True)
        source=origin(directory.name,"E0")/"renders/E0"/directory.name/"step_14999"/view/"A2"
        for name,source_name in (("underwater_original.png","underwater.png"),("clear_original.png","clear_raw_display_clamp01.png"),("gt.png","gt.png")):
            (target/name).symlink_to(source/source_name)
        original=outputs["pred_image"].float().clamp(0,1)
        base=worker._taba_rgb_metrics(original,gt)
        rows.append(dict(scene=directory.name,view_id=view,arm="original",**base))
        for arm,value in rendered.items():
            image=value.reshape(k["height"],k["width"],3)
            assert bool(torch.isfinite(image).all())
            metrics=worker._taba_rgb_metrics(image.clamp(0,1),gt)
            row=dict(scene=directory.name,view_id=view,arm=arm,**metrics,delta_psnr=metrics["PSNR"]-base["PSNR"],
                medium_eval_seconds=times[arm],quadrature_64_vs_128_mae=quadrature[arm][0],quadrature_max=quadrature[arm][1],
                native_clear_max_error=native_error)
            rows.append(row)
            save_rgb(target/(arm+".png"),image)
            save_rgb(target/(arm+"_difference_x10.png"),10*(image-original).abs())
            print(json.dumps(dict(event="eval",**row)),flush=True)
        write_json(target/"metrics.json",[r for r in rows if r["view_id"]==view])
        print(json.dumps(dict(event="view_complete",view=view,wall_seconds=time.perf_counter()-started)),flush=True)
    write_csv(directory/"per_view.csv",rows)
    return rows


def run_scene(args):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==args.gpu
    directory=args.output_root/args.scene
    directory.mkdir(parents=True,exist_ok=False)
    worker=configure_worker("E0")
    worker.ACTIVE_SOURCE_ROOT,worker.ACTIVE_SOURCE_CONFIG,_=source_assets(args.scene)
    runtime=worker._runtime(args.scene,args.gpu)
    branch=worker._new_branch(args.scene)
    checkpoint=origin(args.scene,"E0")/args.scene/"E0/checkpoints/step-000014999.ckpt"
    source_hash=sha256(checkpoint)
    try:
        worker._load_snapshot(branch,args.scene,checkpoint)
        before=worker._model_state_hash(branch.pipeline.model)
        ext=extension()
        data,initial,center,extent,fallback=cache_training(worker,branch,ext,directory,args.rays_per_view)
        data={key:value.cuda() for key,value in data.items()}
        models={arm:train_arm(arm,data,initial.cuda(),center.cuda(),extent.cuda(),args.steps,directory) for arm in ARMS}
        del data
        rows=evaluate(worker,branch,ext,models,center.cuda(),extent.cuda(),fallback,directory)
        assert before==worker._model_state_hash(branch.pipeline.model)
        assert source_hash==sha256(checkpoint)
        write_json(directory/"result.json",dict(complete=True,scene=args.scene,rows=rows,runtime=runtime,
            checkpoint=str(checkpoint),checkpoint_sha256=source_hash,frozen_gaussians_unchanged=True,
            medium_training_steps_per_arm=args.steps,clear_unchanged_by_construction=True))
    finally:
        worker._release(branch)


def launch(args):
    args.output_root.mkdir(parents=True,exist_ok=False)
    assert not any(gpu_processes().values())
    assert not read_json(ROOT/"outputs/transport_candidates_20260911/01_optical_path/verified_v2/summary.json")["thickness_screen_pass"]
    files=[Path(__file__),ROOT/"water_splatting/rendering/segment_transport.py",ROOT/"scripts/diagnostics/csrc/extract_fixed_rays.cu",ROOT/"water_splatting/csrc.so"]
    manifest=[dict(path=str(path.resolve()),sha256=sha256(path)) for path in files]
    write_json(args.output_root/"registration.json",dict(candidate=2,steps=args.steps,rays_per_training_view=args.rays_per_view,
        batch=256,lr=.003,seed=42,segments=64,arms=ARMS,manifest=manifest,gaussians_frozen=True))
    extension()
    jobs=[]
    for scene,gpu in zip(SCENES,("6","7","8","9")):
        handle=(args.output_root/(scene+".log")).open("x")
        command=[sys.executable,str(Path(__file__).resolve()),"--scene",scene,"--gpu",gpu,"--output-root",str(args.output_root),
                 "--steps",str(args.steps),"--rays-per-view",str(args.rays_per_view)]
        process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="4"),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        jobs.append((scene,process,handle))
    while any(process.poll() is None for _,process,_ in jobs):
        write_json(args.output_root/"queue_status.json",{scene:process.poll() for scene,process,_ in jobs})
        time.sleep(5)
    status={scene:process.returncode for scene,process,_ in jobs}
    for _,_,handle in jobs:handle.close()
    write_json(args.output_root/"queue_status.json",status)
    assert not any(status.values()),status
    rows=[row for scene in SCENES for row in read_json(args.output_root/scene/"result.json")["rows"]]
    means=[]
    for scene in SCENES:
        for arm in ("original",)+ARMS:
            selected=[row for row in rows if row["scene"]==scene and row["arm"]==arm]
            means.append(dict(scene=scene,arm=arm,views=len(selected),**{key:statistics.mean(r[key] for r in selected) for key in ("PSNR","SSIM","LPIPS","MSE")}))
    gains=[]
    for scene in SCENES:
        values={row["arm"]:row for row in means if row["scene"]==scene}
        gains.append(dict(scene=scene,grid_minus_mlp_psnr=values["world_grid"]["PSNR"]-values["ray_mlp"]["PSNR"],
                          grid_minus_mlp_lpips=values["world_grid"]["LPIPS"]-values["ray_mlp"]["LPIPS"],
                          grid_minus_original_psnr=values["world_grid"]["PSNR"]-values["original"]["PSNR"]))
    promoted=statistics.mean(r["grid_minus_mlp_psnr"] for r in gains)>=.05 and min(r["grid_minus_mlp_psnr"] for r in gains)>=-.1 and statistics.mean(r["grid_minus_mlp_lpips"] for r in gains)<=0
    assert all(sha256(Path(row["path"]))==row["sha256"] for row in manifest)
    write_csv(args.output_root/"scene_means.csv",means)
    write_json(args.output_root/"summary.json",dict(complete=True,means=means,comparisons=gains,promotion_gate=promoted,
        original_core_unchanged=True,scope="Fixed-Gaussian medium replacement, not full joint training or SOTA validation"))
    print(json.dumps(dict(complete=True,promotion_gate=promoted,comparisons=gains)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",choices=SCENES)
    parser.add_argument("--gpu",choices=("6","7","8","9"))
    parser.add_argument("--output-root",type=Path,default=OUT)
    parser.add_argument("--steps",type=int,default=5000)
    parser.add_argument("--rays-per-view",type=int,default=4096)
    args=parser.parse_args()
    args.output_root=args.output_root.resolve()
    check_environment()
    run_scene(args) if args.scene else launch(args)
