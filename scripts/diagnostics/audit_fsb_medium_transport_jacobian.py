#!/usr/bin/env python3
"""Frozen low-dimensional medium transport Jacobian diagnostic.

The script performs no optimizer or model update. It uses the production
forward rasterizer with frozen FSB+CICA step-14999 checkpoints, estimates two
medium-only corrections from training images, and evaluates them on every
test image. Existing assets are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_fsb_cica_fork_worker import W, SOURCE_MAP, CONFIG_MAP, configure

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
GPU = {scene: str(6 + i) for i, scene in enumerate(SCENES)}
MAP_ROOT = ROOT / "outputs/fsb_far_field_diagnostic_20260916_v1"
CHECKPOINT_ROOT = ROOT / "outputs/fsb_cica_fork_repeat_20260916_v1"
DEFAULT_OUT = ROOT / "outputs/fsb_medium_transport_jacobian_20260917_v2"
EPS = 0.01
RIDGE = 1e-3
Q_MAX = 0.25
MAX_FIT_PIXELS = 12000
STRIDE = 4
CONTROL_NAMES = tuple([f"beta_D_{c}" for c in "rgb"] + [f"beta_B_{c}" for c in "rgb"] + [f"B_inf_{c}" for c in "rgb"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in fields} for row in rows)


def load_frozen_maps(scene: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any], Dict[str, str]]:
    root = MAP_ROOT / scene
    audit = json.loads((root / "audit.json").read_text())
    assert audit["completed"] and audit["checkpoint_unchanged"] and audit["model_unchanged"]
    maps: Dict[str, Dict[str, np.ndarray]] = {}
    hashes = {str(root / "audit.json"): sha256(root / "audit.json")}
    for name in sorted(audit["camera_metadata"]):
        split, view = name.split("/")
        path = root / "maps" / f"{split}_{view}.npz"
        with np.load(path) as f:
            maps[name] = {k: f[k].copy() for k in ("A", "depth", "cv", "pred", "gt", "medium", "S", "t")}
        hashes[str(path)] = sha256(path)
    ckpt = Path(audit["checkpoint"])
    assert sha256(ckpt) == audit["checkpoint_sha256"]
    hashes[str(ckpt)] = audit["checkpoint_sha256"]
    return maps, audit, hashes


def capture(model: Any, camera: Any) -> Tuple[Mapping[str, Any], Any, Mapping[str, torch.Tensor]]:
    captured: Dict[str, Any] = {}
    original = model.underwater_rasterizer.rasterize

    def intercept(**kwargs: Any) -> Any:
        result = original(**kwargs)
        captured["kwargs"] = kwargs
        captured["render"] = result
        return result

    with patch.object(model.underwater_rasterizer, "rasterize", side_effect=intercept):
        outputs = model.get_outputs_for_camera(camera.to(model.device))
    if not captured:
        raise RuntimeError("rasterizer capture failed")
    return captured["kwargs"], captured["render"], outputs


def choose_indices(mask: np.ndarray, seed: int) -> np.ndarray:
    flat = np.flatnonzero(mask.reshape(-1))
    if len(flat) <= MAX_FIT_PIXELS:
        return flat
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(flat, size=MAX_FIT_PIXELS, replace=False))


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    arr = (tensor.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def contact_sheet(path: Path, panels: Sequence[Tuple[str, np.ndarray]]) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    width, height, cols = 400, 310, 4
    canvas = Image.new("RGB", (cols * width, math.ceil(len(panels) / cols) * height), "#17191d")
    draw = ImageDraw.Draw(canvas)
    for i, (label, arr) in enumerate(panels):
        image = Image.fromarray((np.nan_to_num(arr).clip(0, 1) * 255).round().astype(np.uint8))
        image.thumbnail((width - 10, height - 45), Image.Resampling.LANCZOS)
        x, y = (i % cols) * width, (i // cols) * height
        draw.text((x + 5, y + 6), label, fill="white", font=font)
        canvas.paste(image, (x + (width - image.width) // 2, y + 36))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def gt_for(model, batch, background):
    return W._gt(model, W._batch_to_device(batch, model.device), background).float().clamp(0,1)


def transform(kwargs, q):
    """An identity-exact odds shift for B_inf, and log scales for beta."""
    q = np.asarray(q, np.float64)
    out = dict(kwargs)
    for group, key in enumerate(('medium_attn', 'medium_bs', 'medium_rgb')):
        if not np.any(q[group*3:group*3+3]):
            continue
        shift = torch.tensor(q[group*3:group*3+3], dtype=kwargs[key].dtype, device=kwargs[key].device)
        e = shift.exp()
        p = kwargs[key]
        out[key] = (p*e if group < 2 else p*e/(1+p*(e-1))).contiguous()
    out['background'] = out['medium_rgb']
    return out


def fit_solution(xtx, xty, active=None):
    active = np.arange(9) if active is None else np.asarray(active)
    H = xtx[np.ix_(active, active)].astype(np.float64)
    g = xty[active].astype(np.float64)
    scale = np.sqrt(np.maximum(np.diag(H), 1e-24))
    normalized = H/scale[:,None]/scale[None,:]
    # z = scale*q, so mapping the normalized solution back MUST divide by scale.
    z = np.linalg.solve(normalized + RIDGE*np.eye(len(active)), g/scale)
    raw = np.zeros(9);raw[active] = z/scale
    clipped = np.clip(raw, -Q_MAX, Q_MAX)
    return raw, clipped, float(np.mean(np.abs(raw[active]-clipped[active])>1e-12))


def masks_for(m, bounds):
    d = m['depth'];stable = (m['A']>=.9)&(m['cv']<=.1)&np.isfinite(d)&(d>0)
    return dict(all=np.ones(d.shape,bool),near=stable&(d<bounds[0]),
                mid=stable&(d>=bounds[0])&(d<bounds[1]),far=stable&(d>=bounds[1]),
                loose_far=(m['A']>=.5)&np.isfinite(d)&(d>=bounds[1]),
                high_cv_far=(m['A']>=.5)&np.isfinite(d)&(d>=bounds[1])&(m['cv']>.1))


def moments(J, residual, indices):
    X = J.reshape(-1,3,9)[indices].reshape(-1,9).astype(np.float64)
    y = residual.reshape(-1,3)[indices].reshape(-1).astype(np.float64)
    return dict(H=X.T@X,g=X.T@y,y2=float(y@y),n=len(y))


def relative_error(mom, q):
    return float((mom['y2']-2*mom['g']@q+q@mom['H']@q)/max(mom['y2'],1e-24))


def add_moments(ms):
    return dict(H=sum((m['H'] for m in ms),np.zeros((9,9))),
                g=sum((m['g'] for m in ms),np.zeros(9)),y2=sum(m['y2'] for m in ms),n=sum(m['n'] for m in ms))


def support_check(base, result):
    # If the production binary changes contribution support, the claimed frozen-clear
    # experiment is invalid even though Gaussian parameters are unchanged.
    clear = float((base.j_raw-result.j_raw).abs().max())
    display = float((base.rgb_clear-result.rgb_clear).abs().max())
    acc = float((base.accumulation-result.accumulation).abs().max())
    depth = float((base.depth-result.depth).abs().max())
    if max(clear,display,acc,depth)>1e-6:
        raise RuntimeError(('medium change altered clear/support',clear,display,acc,depth))
    return dict(clear_max_abs=clear,clear_display_max_abs=display,accumulation_max_abs=acc,depth_max_abs=depth)


def fd_jacobian(model, kwargs, base, epsilon=EPS):
    call = model.underwater_rasterizer.rasterize
    cols=[]; checks=[]
    for i in range(9):
        q=np.zeros(9);q[i]=epsilon
        pos=call(**transform(kwargs,q));checks.append(support_check(base,pos))
        neg=call(**transform(kwargs,-q));checks.append(support_check(base,neg))
        cols.append(((pos.rgb.clamp(0,1)-neg.rgb.clamp(0,1))/(2*epsilon))[::STRIDE,::STRIDE].cpu().numpy())
        del pos,neg
    return np.stack(cols,-1), {k:max(x[k] for x in checks) for k in checks[0]}


def image_metrics(model, pred, gt):
    # Exact metric implementation used by the retained FSB branch, including LPIPS.
    return W.MIC._metric_images(model,pred,gt)


def run_scene(scene, out, smoke):
    started=time.perf_counter();scene_out=out/scene;scene_out.mkdir(parents=True,exist_ok=False)
    (scene_out/'visualizations').mkdir();(scene_out/'jacobians').mkdir()
    maps,audit,hashes=load_frozen_maps(scene);bounds=audit['depth_boundaries']
    source_paths=[Path(__file__),ROOT/'research_notes/FSB_MEDIUM_TRANSPORT_JACOBIAN_PROTOCOL_2026-09-17.md',
                  ROOT/'scripts/experiments/run_fsb_cica_fork_worker.py',ROOT/'scripts/experiments/run_direct_mdrr_cica_scene.py',
                  ROOT/'water_splatting/water_splatting.py',ROOT/'water_splatting/csrc.so']
    hashes.update({str(p):sha256(p) for p in source_paths})
    configure(scene,'legacy','legacy');W.ENFORCE_SCENE_GPU_ASSIGNMENT=False;W._route_renderer_backend('legacy')
    branch=W._new_branch(scene);model=branch.pipeline.model;checkpoint=Path(audit['checkpoint'])
    fit_rows=[];validation=[];replay_rows=[];view_moments={};train_fit=[];per_view=[];region_rows=[];support_rows=[]
    numerical=[];allnames=[];peak=0
    try:
        payload=W._load_snapshot(branch,scene,checkpoint);model.eval()
        initial=W._model_state_hash(model)
        assert initial==W._hash_object(payload['model'])==audit['model_state_sha256']
        del payload
        records={'train':sorted(W._train_records(branch),key=lambda r:str(r[1])),
                 'eval':sorted(W._eval_records(branch),key=lambda r:str(r[1]))}
        actual={s+'/'+str(r[1]) for s,rs in records.items() for r in rs}
        assert actual==set(maps)
        for split,filename in [('train','train_list.txt'),('eval','test_list.txt')]:
            p=ROOT/'undistorted_data'/('undistorted_'+scene)/filename
            expected={Path(x).stem for x in p.read_text().splitlines() if x.strip()}
            assert expected=={str(r[1]) for r in records[split]}
            hashes[str(p)]=sha256(p)
        if smoke: records={s:rs[:6 if s=='train' else 1] for s,rs in records.items()}
        assert model._effective_b_inf_mode()=='tied'
        torch.cuda.reset_peak_memory_stats()
        for vi,(idx,view,cam,batch) in enumerate(records['train']):
            name='train/'+str(view);m=maps[name]
            kw,render,o=capture(model,cam)
            base=o['pred_image'].clamp(0,1);gt=gt_for(model,batch,o['background'])
            assert torch.equal(o['b_inf'],kw['medium_rgb'])
            assert float((base-render.rgb.clamp(0,1)).abs().max())<2e-6
            assert float(kw['colors'].min())>=0 and float(kw['colors'].max())<=1
            b=base[::4,::4].cpu().numpy();y=gt[::4,::4].cpu().numpy()
            replay=float(np.max(np.abs(b-m['pred'])));assert replay==0,(name,replay)
            assert float(np.max(np.abs(y-m['gt'])))==0
            replay_rows.append(dict(view=name,pred_map_max_error=replay))
            J,sch=fd_jacobian(model,kw,render);support_rows.append(dict(view=name,phase='fd',**sch))
            if vi==0:
                Jhalf,_=fd_jacobian(model,kw,render,EPS/2)
                err=np.linalg.norm(J-Jhalf)/max(np.linalg.norm(Jhalf),1e-12)
                numerical.append(dict(view=name,test='epsilon_vs_half',relative_l2=float(err),max_abs=float(np.max(abs(J-Jhalf)))))
                # Directional nonlinear checks provide an independent linearization test.
                direction=np.arange(-4,5,dtype=np.float64);direction=direction/max(np.linalg.norm(direction),1)
                q=EPS*direction;rr=model.underwater_rasterizer.rasterize(**transform(kw,q))
                support_check(render,rr)
                actual=(rr.rgb.clamp(0,1)[::4,::4].cpu().numpy()-b)
                expected=J@q
                numerical.append(dict(view=name,test='directional_prediction',relative_l2=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(actual),1e-12)),max_abs=float(np.max(abs(actual-expected)))))
                del Jhalf,rr
            np.savez_compressed(scene_out/'jacobians'/(name.replace('/','_')+'.npz'),J=J,base=b,gt=y)
            masks=masks_for(m,bounds);view_moments[name]={}
            for arm,reg in [('all_fit','all'),('far_fit','far')]:
                indices=choose_indices(masks[reg],170917+int(idx)*37+(0 if arm=='all_fit' else 100000))
                mom=moments(J,y-b,indices)
                view_moments[name][arm]=mom
                fit_rows.append(dict(scene=scene,view=str(view),arm=arm,sampled_pixels=len(indices),replay_max_error=replay))
            print(json.dumps(dict(scene=scene,phase='jacobian',view=str(view))),flush=True)
            del J,kw,render,o,base,gt
        totals={arm:add_moments([v[arm] for v in view_moments.values()]) for arm in ('all_fit','far_fit')}
        raw_q={};q_values={};clip_fraction={};conditioning={};family_rows=[]
        for arm,mom in totals.items():
            raw,q,clip=fit_solution(mom['H'],mom['g']);raw_q[arm]=raw;q_values[arm]=q;clip_fraction[arm]=clip
            scale=np.sqrt(np.maximum(np.diag(mom['H']),1e-24));H=mom['H']/scale[:,None]/scale[None,:]
            eigen,V=np.linalg.eigh(H)
            conditioning[arm]=dict(normalized_eigenvalues=eigen.tolist(),normalized_condition=float(eigen[-1]/max(eigen[0],1e-24)),
                                   normalized_correlation=H.tolist(),clipped_tangent_error_ratio=relative_error(mom,q),
                                   raw_tangent_error_ratio=relative_error(mom,raw))
            for group in range(3):
                _,small,_=fit_solution(mom['H'],mom['g'],np.arange(group*3,group*3+3))
                family_rows.append(dict(arm=arm,family=('attn','bs','binf')[group],tangent_error_ratio=relative_error(mom,small),q=small.tolist()))
            for name,vs in view_moments.items():
                omit=vs[arm];others=dict(H=mom['H']-omit['H'],g=mom['g']-omit['g'],y2=mom['y2']-omit['y2'],n=mom['n']-omit['n'])
                _,lvo,_=fit_solution(others['H'],others['g'])
                validation.append(dict(arm=arm,heldout_train_view=name,pixels=omit['n']//3,
                                       baseline_sse=omit['y2'],linear_sse=omit['y2']*relative_error(omit,lvo),q=lvo.tolist()))
        fitout=dict(epsilon=EPS,ridge=RIDGE,q_max=Q_MAX,controls=CONTROL_NAMES,raw_q=raw_q,q=q_values,clip_fraction=clip_fraction,
                    normal_equations=totals,per_train_view_normal_equations=view_moments,conditioning=conditioning,
                    family_linear_controls=family_rows,coefficient_sign_agreement=int(np.sum(np.sign(q_values['all_fit'])==np.sign(q_values['far_fit']))),
                    fit_uses_eval=False)
        write_json(scene_out/'fit_coefficients.json',fitout);write_csv(scene_out/'leave_one_train_view.csv',validation)
        write_csv(scene_out/'fit_coefficients.csv',[dict(arm=arm,control=CONTROL_NAMES[i],q=float(q_values[arm][i]),raw_q=float(raw_q[arm][i])) for arm in q_values for i in range(9)])
        # Coefficients are saved before opening eval GT. No test-based selection.
        with (CHECKPOINT_ROOT/scene/'R0/C0a/per_view_metrics.csv').open() as f:old_metrics={r['view_id']:r for r in csv.DictReader(f)}
        for split,recs in records.items():
            for vi,(idx,view,cam,batch) in enumerate(recs):
                name=split+'/'+str(view);allnames.append(name);m=maps[name];mask=masks_for(m,bounds)
                kw,render,o=capture(model,cam);base=o['pred_image'].clamp(0,1);gt=gt_for(model,batch,o['background'])
                b=base[::4,::4].cpu().numpy();g=gt[::4,::4].cpu().numpy()
                assert np.array_equal(b,m['pred']) and np.array_equal(g,m['gt'])
                if split=='train':
                    with np.load(scene_out/'jacobians'/(name.replace('/','_')+'.npz')) as f:J=f['J']
                else:
                    J,sc=fd_jacobian(model,kw,render);support_rows.append(dict(view=name,phase='fd',**sc))
                    np.savez_compressed(scene_out/'jacobians'/(name.replace('/','_')+'.npz'),J=J,base=b,gt=g)
                    replay_rows.append(dict(view=name,pred_map_max_error=0.))
                    if vi==0:
                        half,_=fd_jacobian(model,kw,render,EPS/2)
                        numerical.append(dict(view=name,test='epsilon_vs_half',relative_l2=float(np.linalg.norm(J-half)/max(np.linalg.norm(half),1e-12)),max_abs=float(np.max(abs(J-half)))))
                        del half
                preds={'baseline':base};diagnostic_maps={};clear_checks={}
                for arm,q in q_values.items():
                    result=model.underwater_rasterizer.rasterize(**transform(kw,q))
                    sc=support_check(render,result);clear_checks[arm]=sc;support_rows.append(dict(view=name,phase=arm,**sc))
                    preds[arm]=result.rgb.clamp(0,1)
                    effect=preds[arm][::4,::4].cpu().numpy()-b;tangent=J@q
                    denominator=max(float(np.linalg.norm(effect)),1e-12)
                    numerical.append(dict(view=name,test=arm+'_tangent',relative_l2=float(np.linalg.norm(effect-tangent)/denominator),
                                           max_abs=float(np.max(abs(effect-tangent))),effect_rms=float(np.sqrt(np.mean(effect**2))),
                                           tangent_rms=float(np.sqrt(np.mean(tangent**2)))))
                    diagnostic_maps[arm+'_pred']=preds[arm][::4,::4].cpu().numpy()
                    diagnostic_maps[arm+'_linear_pred']=np.clip(b+tangent,0,1)
                    del result
                if split=='eval':
                    native_dir=scene_out/'renders'/str(view);native_dir.mkdir(parents=True)
                    old_dir=CHECKPOINT_ROOT/scene/'R0/C0a/renders'/scene/'step_14999'/str(view)/'C0a'
                    for filename,im in [('underwater.png',base),('clear.png',render.rgb_clear),('clear_raw_display_clamp01.png',render.j_raw),('gt.png',gt)]:
                        old=old_dir/filename;actual=(im.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                        assert np.array_equal(actual,np.asarray(Image.open(old))),(view,filename)
                        hashes[str(old)]=sha256(old)
                    save_rgb(native_dir/'gt.png',gt);save_rgb(native_dir/'baseline.png',base)
                    save_rgb(native_dir/'clear_raw.png',render.j_raw);save_rgb(native_dir/'clear_display.png',render.rgb_clear)
                    for arm in q_values:save_rgb(native_dir/(arm+'.png'),preds[arm])
                    # Identical clear: keep exact hashes/checks instead of duplicate assets.
                    np.savez_compressed(native_dir/'diagnostic_maps_stride4.npz',**diagnostic_maps)
                    contact_sheet(scene_out/'visualizations'/(str(view)+'_comparison.png'),[
                        ('Underwater GT',gt.cpu().numpy()),('FSB baseline',base.cpu().numpy()),
                        ('All-image medium fit',preds['all_fit'].cpu().numpy()),('Far-region medium fit',preds['far_fit'].cpu().numpy()),
                        ('All-fit minus base abs x10',(preds['all_fit']-base).abs().cpu().numpy()*10),
                        ('Far-fit minus base abs x10',(preds['far_fit']-base).abs().cpu().numpy()*10),
                        ('Clear raw: verified unchanged',render.j_raw.cpu().numpy()),('Clear display: verified unchanged',render.rgb_clear.cpu().numpy())])
                for arm,pred in preds.items():
                    mse=float((pred-gt).square().mean())
                    met=image_metrics(model,pred,gt) if split=='eval' else dict(MSE=mse,PSNR=-10*math.log10(max(mse,1e-20)))
                    if split=='eval' and arm=='baseline':
                        for key in ('PSNR','SSIM','LPIPS','MSE'):assert abs(met[key]-float(old_metrics[str(view)][key]))<(2e-5 if key=='PSNR' else 2e-6),(scene,view,key,met[key],old_metrics[str(view)][key])
                    per_view.append(dict(scene=scene,split=split,view=str(view),arm=arm,**met))
                    predgrid=pred[::4,::4].cpu().numpy();sq=(predgrid.astype(np.float64)-g)**2
                    linear=np.clip(b+J@q_values[arm],0,1) if arm!='baseline' else b
                    for region,regmask in mask.items():
                        n=int(regmask.sum());sse=float(sq[regmask].sum());lsse=float(((linear-g)[regmask]**2).sum())
                        region_rows.append(dict(scene=scene,split=split,view=str(view),arm=arm,region=region,pixels=n,sse=sse,
                                                mse=sse/(n*3) if n else None,linear_sse=lsse))
                if split=='train':
                    for arm,q in q_values.items():
                        reg='all' if arm=='all_fit' else 'far';rr=next(r for r in reversed(region_rows) if r['arm']==arm and r['region']==reg)
                        train_fit.append(dict(arm=arm,view=str(view),region=reg,nonlinear_mse=rr['mse']))
                print(json.dumps(dict(scene=scene,phase='nonlinear',split=split,view=str(view))),flush=True)
                peak=max(peak,torch.cuda.max_memory_allocated());del kw,render,o,base,gt,preds,J
        final=W._model_state_hash(model);assert initial==final
        assert sha256(checkpoint)==audit['checkpoint_sha256']
    finally:
        W._release(branch)
    for path,h in hashes.items():assert sha256(Path(path))==h,('input drift',path)
    write_json(scene_out/'input_hashes.json',dict(unchanged=True,hashes=hashes))
    write_csv(scene_out/'per_view_metrics.csv',per_view);write_csv(scene_out/'region_metrics.csv',region_rows)
    write_csv(scene_out/'fit_sampling.csv',fit_rows);write_csv(scene_out/'numerical_validation.csv',numerical)
    write_csv(scene_out/'support_checks.csv',support_rows);write_csv(scene_out/'replay_checks.csv',replay_rows)
    arms={}
    for arm in ('baseline','all_fit','far_fit'):
        rr=[r for r in per_view if r['split']=='eval' and r['arm']==arm]
        arms[arm]={key:float(np.mean([r[key] for r in rr])) for key in ('PSNR','SSIM','LPIPS','MSE')}
        for region in ('near','mid','far','loose_far','high_cv_far'):
            rs=[r for r in region_rows if r['split']=='eval' and r['arm']==arm and r['region']==region]
            n=sum(r['pixels'] for r in rs);arms[arm][region+'_rmse']=math.sqrt(sum(r['sse'] for r in rs)/(n*3)) if n else None
    s=dict(scene=scene,completed=True,implementation='v2',smoke=smoke,checkpoint=str(checkpoint),checkpoint_sha256=audit['checkpoint_sha256'],
           model_state_before=initial,model_state_after=final,model_unchanged=True,checkpoint_unchanged=True,input_unchanged=True,
           train_views=len(records['train']),eval_views=len(records['eval']),physical_gpu=GPU[scene],python=sys.executable,
           optimizer_steps=0,backward_calls=0,training_started=False,diagnostic_coefficients_fitted=True,
           metrics=arms,q={k:v.tolist() for k,v in q_values.items()},clip_fraction=clip_fraction,
           coefficient_sign_agreement=fitout['coefficient_sign_agreement'],
           max_clear_change=max(r['clear_max_abs'] for r in support_rows),max_support_change=max(r['accumulation_max_abs'] for r in support_rows),
           replay_max_error=max(r['pred_map_max_error'] for r in replay_rows),
           max_fd_halving_relative_error=max(r['relative_l2'] for r in numerical if r['test']=='epsilon_vs_half'),
           seconds=time.perf_counter()-started,peak_cuda_allocated_gb=peak/1024**3)
    write_json(scene_out/'summary.json',s);print(json.dumps(s,sort_keys=True),flush=True)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert os.environ.get('CONDA_DEFAULT_ENV')=='water_splatting'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU[args.scene]
    assert sys.executable=='/opt/anaconda3/envs/water_splatting/bin/python'
    torch.set_grad_enabled(False);torch.set_num_threads(2);np.random.seed(170917)
    args.output.resolve().mkdir(parents=True,exist_ok=True)
    run_scene(args.scene,args.output.resolve(),args.smoke)

if __name__=='__main__':main()
