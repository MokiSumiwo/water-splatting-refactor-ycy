#!/usr/bin/env python3
"""Compare encoded/linear-sRGB transport with matched global calibration."""

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
import torch.nn.functional as F

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES,origin,save_rgb,write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment,gpu_processes,read_json,sha256,source_assets,write_json
from scripts.experiments.run_world_medium_screen import capture_view,extract,extension
from water_splatting.rendering.radiometric_transport import FrozenRadiometricTransport

OUT=ROOT/"outputs/transport_candidates_20260911/06_radiometric_domain"


def prepare(ext,k,bins,pixels,outputs=None,gt=None):
    data,_=extract(ext,k,bins,pixels)
    for source,target in (("medium_attn","beta"),("medium_bs","bs"),("medium_rgb","ambient")):
        data[target]=k[source].reshape(-1,3)[pixels]
    if outputs is not None:data["base"]=outputs["pred_image"].reshape(-1,3)[pixels]
    if gt is not None:data["gt"]=gt.reshape(-1,3)[pixels]
    return data


@torch.no_grad()
def cache(worker,branch,ext,directory):
    chunks=[];manifest=[]
    identity=FrozenRadiometricTransport().cuda()
    for index,view,camera,batch in worker._train_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        generator=torch.Generator().manual_seed(44000+int(index))
        pixels=torch.randperm(k["width"]*k["height"],generator=generator)[:4096].cuda()
        data=prepare(ext,k,_geometry_bins(k),pixels,outputs,gt)
        original,clear=identity(data)
        error=float((original-data["base"]).abs().max())
        assert error<2e-6,(view,error)
        data["clear_base"]=clear
        chunks.append({key:value.detach().cpu() for key,value in data.items()})
        manifest.append(dict(view=view,pixels=pixels.cpu().tolist(),native_max_error=error))
        print(json.dumps(dict(event="cache",view=view)),flush=True)
    maximum=max(chunk["weights"].shape[1] for chunk in chunks)
    for chunk in chunks:
        pad=maximum-chunk["weights"].shape[1]
        for key in ("weights","depths"):chunk[key]=F.pad(chunk[key],(0,pad))
        chunk["colors"]=F.pad(chunk["colors"],(0,0,0,pad))
    write_json(directory/"training_pixels.json",dict(rows=manifest))
    return {key:torch.cat([c[key] for c in chunks]).cuda() for key in chunks[0]}


def train(arm,data,directory):
    model=FrozenRadiometricTransport(arm=="linear_srgb").cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    generator=torch.Generator().manual_seed(42)
    rows=[]
    for step in range(1,5001):
        index=torch.randint(data["gt"].shape[0],(256,),generator=generator).cuda()
        batch={key:value[index] for key,value in data.items()}
        optimizer.zero_grad()
        pred,_=model(batch)
        loss=(pred-batch["gt"]).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.raw.grad).all()
        optimizer.step()
        if step==1 or step%500==0:rows.append(dict(step=step,mse=float(loss.detach())))
    torch.save(model.state_dict(),directory/(arm+".pt"))
    write_json(directory/(arm+"_training.json"),dict(steps=5000,rows=rows,raw=model.raw.detach().tolist(),
        parameter_count=9,global_only=True,per_test_view_fit=False))
    return model.eval()


@torch.no_grad()
def evaluate(worker,branch,ext,models,directory):
    rows=[]
    for index,view,camera,batch in worker._eval_records(branch):
        k,native,outputs,gt=capture_view(worker,branch.pipeline.model,camera,batch)
        bins=_geometry_bins(k);total=k["width"]*k["height"]
        pred={arm:torch.empty(total,3,device="cuda") for arm in models}
        clear={arm:torch.empty(total,3,device="cuda") for arm in models}
        started=time.perf_counter()
        for start in range(0,total,8192):
            pixels=torch.arange(start,min(start+8192,total),device="cuda",dtype=torch.int64)
            data=prepare(ext,k,bins,pixels)
            for arm,model in models.items():pred[arm][pixels],clear[arm][pixels]=model(data)
        original=outputs["pred_image"].clamp(0,1)
        base=worker._taba_rgb_metrics(original,gt)
        rows.append(dict(scene=directory.name,view_id=view,arm="original",**base))
        target=directory/"renders"/view;target.mkdir(parents=True)
        source=origin(directory.name,"E0")/"renders/E0"/directory.name/"step_14999"/view/"A2"
        for name,source_name in (("underwater_original.png","underwater.png"),("clear_original.png","clear_raw_display_clamp01.png"),("gt.png","gt.png")):
            (target/name).symlink_to(source/source_name)
        for arm in models:
            rgb=pred[arm].reshape_as(original);j=clear[arm].reshape_as(original)
            metrics=worker._taba_rgb_metrics(rgb.clamp(0,1),gt)
            row=dict(scene=directory.name,view_id=view,arm=arm,**metrics,delta_psnr=metrics["PSNR"]-base["PSNR"],
                clear_mae=float((j-native.j_raw).abs().mean()),near_black_fraction=float((j.max(-1).values<=3/255).float().mean()))
            rows.append(row)
            save_rgb(target/(arm+"_underwater.png"),rgb);save_rgb(target/(arm+"_clear.png"),j)
            save_rgb(target/(arm+"_clear_difference_x10.png"),10*(j-native.j_raw).abs())
            print(json.dumps(dict(event="eval",**row)),flush=True)
        print(json.dumps(dict(event="view_done",view=view,seconds=time.perf_counter()-started)),flush=True)
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
        ext=extension();data=cache(worker,branch,ext,directory)
        models={arm:train(arm,data,directory) for arm in ("encoded","linear_srgb")}
        del data
        rows=evaluate(worker,branch,ext,models,directory)
        assert before==worker._model_state_hash(branch.pipeline.model) and digest==sha256(checkpoint)
        write_json(directory/"result.json",dict(complete=True,rows=rows,checkpoint=str(checkpoint),checkpoint_sha256=digest,
            source_unchanged=True,gaussians_frozen=True,response_assumption="sRGB unverified for original PNG camera processing"))
    finally:worker._release(branch)


def launch(args):
    args.output_root.mkdir(parents=True,exist_ok=False)
    assert not any(gpu_processes().values())
    write_json(args.output_root/"registration.json",dict(candidate=6,steps=5000,lr=.01,seed=42,batch=256,train_pixels_per_camera=4096,
        arms=["encoded","linear_srgb"],parameters_per_arm=9,global_only=True,loss="RGB MSE",known_response=False,
        promotion="linear-encoded mean PSNR >=0.05; min scene >=-0.1; mean LPIPS nonworse; absolute original safety"))
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
        for arm in ("original","encoded","linear_srgb"):
            selected=[r for r in rows if r["scene"]==scene and r["arm"]==arm]
            means.append(dict(scene=scene,arm=arm,**{key:statistics.mean(r[key] for r in selected) for key in ("PSNR","SSIM","LPIPS","MSE")}))
    deltas=[]
    for scene in SCENES:
        values={r["arm"]:r for r in means if r["scene"]==scene}
        deltas.append(dict(scene=scene,psnr_linear_minus_encoded=values["linear_srgb"]["PSNR"]-values["encoded"]["PSNR"],
            lpips_linear_minus_encoded=values["linear_srgb"]["LPIPS"]-values["encoded"]["LPIPS"],
            psnr_linear_minus_original=values["linear_srgb"]["PSNR"]-values["original"]["PSNR"]))
    passed=statistics.mean(r["psnr_linear_minus_encoded"] for r in deltas)>=.05 and min(r["psnr_linear_minus_encoded"] for r in deltas)>=-.1 and statistics.mean(r["lpips_linear_minus_encoded"] for r in deltas)<=0 and min(r["psnr_linear_minus_original"] for r in deltas)>=-.1
    write_csv(args.output_root/"scene_means.csv",means)
    write_json(args.output_root/"summary.json",dict(complete=True,means=means,deltas=deltas,promotion_gate=passed,
        calibrated_camera_response=False,clear_accuracy_claim=False,sota_claim=False))
    print(json.dumps(dict(complete=True,deltas=deltas,promotion_gate=passed)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",choices=SCENES);parser.add_argument("--gpu",choices=("6","7","8","9"))
    parser.add_argument("--output-root",type=Path,default=OUT)
    args=parser.parse_args();args.output_root=args.output_root.resolve()
    check_environment()
    run_scene(args) if args.scene else launch(args)
