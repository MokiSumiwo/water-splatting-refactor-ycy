#!/usr/bin/env python3
"""Frozen, native-compositing far-field observation/compensation/geometry audit."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_fsb_cica_fork_worker import configure, W

SCENES = ('Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama')
SOURCE = ROOT / 'outputs/fsb_cica_fork_repeat_20260916_v1'
STRIDE = 4
EPS = .01
MODES = ('attn', 'bs', 'rgb')
SH_C0 = .28209479177387814


def write_json(path, obj):
    W._write_json(path, obj)


def load_extension():
    name = 'cica_exact_dc_jacobian_cuda_v1'
    path = Path.home() / '.cache/torch_extensions/py38_cu118' / name / (name + '.so')
    if not path.is_file():
        raise FileNotFoundError('Require existing diagnostic binary; no compilation: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def sample(x):
    return x[::STRIDE, ::STRIDE].detach().float().cpu().numpy()


def avg(x, mask):
    v = x[mask]
    return float(np.mean(v)) if v.size else float('nan')


def quant(x, q):
    return float(np.quantile(x, q)) if x.size else float('nan')


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 32 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float('nan')
    return float(spearmanr(x[ok], y[ok]).correlation)


def grad(x):
    if x.ndim == 3:
        x = x.mean(-1)
    gy, gx = np.gradient(x)
    return np.sqrt(gx * gx + gy * gy)


def bands(data, boundaries, strict=True):
    a, cv, d = data['A'], data['cv'], data['depth']
    mask = (a >= (.9 if strict else .5)) & np.isfinite(d) & (d > 0)
    if strict:
        mask &= cv <= .1
    return {label: mask & (d >= lo) & (d < hi) for label, lo, hi in (
        ('near', 0, boundaries[0]), ('mid', boundaries[0], boundaries[1]),
        ('far', boundaries[1], np.inf))}


def summarize_view(data, scene, split, view, boundaries):
    rows = []
    for strict in (True, False):
        for region, mask in bands(data, boundaries, strict).items():
            row = dict(scene=scene, split=split, view=view, mask='surface_candidate' if strict else 'A_ge_0p5',
                       region=region, pixels=int(mask.sum()), fraction=float(mask.mean()),
                       depth=avg(data['depth'], mask), depth_cv=avg(data['cv'], mask),
                       depth_cv_gt_0p1=avg(data['cv'] > .1, mask), A=avg(data['A'], mask),
                       relative_depth_gradient=avg(grad(data['depth'])/np.maximum(data['depth'],1e-6),mask),
                       mse=avg((data['pred']-data['gt'])**2, mask),
                       clear_gradient=avg(grad(data['clear']), mask), gt_gradient=avg(grad(data['gt']), mask),
                       low_coverage_fraction=float((data['A'] < .5).mean()),
                       rho_cv_rgb_error=rho(data['cv'][mask], ((data['pred']-data['gt'])**2).mean(-1)[mask]))
            for c, name in enumerate('rgb'):
                t = data['t'][..., c]
                row.update({f't_{name}': avg(t, mask), f't_{name}_p10': quant(t[mask], .1),
                            f't_{name}_lt_0p1': avg(t < .1, mask), f't_{name}_lt_0p3': avg(t < .3, mask),
                            f'medium_{name}': avg(data['medium'][..., c], mask),
                            f'direct_{name}': avg(data['obj'][..., c], mask),
                            f'clear_{name}': avg(data['clear'][..., c], mask)})
            rows.append(row)
    return rows


def perturb(kwargs, mode, sign):
    key = 'medium_' + mode
    x = kwargs[key]
    if mode == 'rgb':
        y = x.clamp(1e-6, 1-1e-6)
        v = torch.sigmoid(torch.log(y/(1-y)) + sign*EPS)
    else:
        v = x*(1+sign*EPS)
    return dict(kwargs, **{key: v})


def camera_info(camera):
    return dict(c2w=camera.camera_to_worlds[0].detach().float().cpu().numpy(),
                fx=float(camera.fx.item()), fy=float(camera.fy.item()),
                cx=float(camera.cx.item()), cy=float(camera.cy.item()),
                height=int(camera.height.item()), width=int(camera.width.item()))


def color_map(x, lo, hi, name='viridis'):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import cm
    value = matplotlib.colormaps[name](np.clip((x-lo)/max(hi-lo, 1e-9), 0, 1))[..., :3]
    value[~np.isfinite(x)] = .25
    return value


def sheet(path, panels):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    width, height, columns = 400, 305, 4
    canvas = Image.new('RGB', (columns*width, math.ceil(len(panels)/columns)*height), '#16181a')
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        img = Image.fromarray((np.nan_to_num(arr).clip(0, 1)*255).round().astype(np.uint8))
        img.thumbnail((width, height-38), Image.Resampling.LANCZOS)
        x, y = idx%columns*width, idx//columns*height
        canvas.paste(img, (x+(width-img.width)//2, y+38))
        draw.text((x+5, y+4), label, font=font, fill='white')
    canvas.save(path)


def observation_sheet(path, data, bounds):
    masks = bands(data, bounds)
    groups = np.zeros((*data['A'].shape, 3), dtype=np.float32)
    for name, col in zip(('near', 'mid', 'far'), ((.15,.6,1),(.2,.85,.3),(1,.6,.1))):
        groups[masks[name]] = col
    sheet(path, [('Underwater GT', data['gt']), ('Underwater native', data['pred']),
                 ('Clear raw [0,1]; no GT', data['clear']), ('Near blue / mid green / far orange', groups),
                 ('Contribution transmission R [0,1]', color_map(data['t'][...,0],0,1)),
                 ('Contribution transmission G [0,1]', color_map(data['t'][...,1],0,1)),
                 ('Contribution transmission B [0,1]', color_map(data['t'][...,2],0,1)),
                 ('Additive medium RGB [0,1]', data['medium']),
                 ('Depth; common scene scale', color_map(data['depth'],0,bounds[1]*2)),
                 ('Relative depth std [0,0.2]', color_map(data['cv'],0,.2,'magma')),
                 ('Accumulation [0,1]', color_map(data['A'],0,1)),
                 ('Underwater abs residual x10', np.abs(data['pred']-data['gt'])*10)])


@torch.no_grad()
def render_capture(model, camera):
    saved = {}
    call = model.underwater_rasterizer.rasterize
    def intercept(**kwargs):
        result = call(**kwargs)
        saved.update(kwargs=kwargs, result=result)
        return result
    with patch.object(model.underwater_rasterizer, 'rasterize', side_effect=intercept):
        o = model.get_outputs_for_camera(camera.to(model.device))
    return o, saved['kwargs'], saved['result']


@torch.no_grad()
def information(extension, kwargs, unit):
    ids, bins, bounds = W.FULL._geometry_bins(kwargs)
    dev = kwargs['colors'].device
    tile = torch.tensor(bounds, dtype=torch.int32, device=dev)
    size = torch.tensor([kwargs['width'],kwargs['height'],1], dtype=torch.int32, device=dev)
    residual = torch.ones_like(kwargs['medium_attn'])
    def run(beta):
        return extension.cica_forward(tile, size, ids, bins, kwargs['xys'].float().contiguous(),
            kwargs['conics'].float().contiguous(), kwargs['colors'].float().contiguous(),
            kwargs['opacities'].reshape(-1).float().contiguous(), beta.float().contiguous(),
            kwargs['depths'].reshape(-1).float().contiguous(), residual.reshape(-1,3), 16, 1)
    n, info = run(kwargs['medium_attn'])
    n0, info0 = run(torch.zeros_like(kwargs['medium_attn']))
    rel = ((n.sum(0)/SH_C0 - unit.rgb_object.sum((0,1))).abs() / unit.rgb_object.sum((0,1)).clamp_min(1e-6)).max()
    rel0 = ((n0.sum(0)/SH_C0 - unit.j_raw.sum((0,1))).abs() / unit.j_raw.sum((0,1)).clamp_min(1e-6)).max()
    if max(float(rel), float(rel0)) > 3e-5:
        raise RuntimeError(f'Jacobian contribution equivalence failed: {rel}, {rel0}')
    return n/SH_C0, info/(SH_C0**2), n0[:,0]/SH_C0, info0[:,0]/(SH_C0**2), max(float(rel),float(rel0))


def grid_sample_numpy(arr, u, v, dev):
    h, w = arr.shape[:2]
    values = torch.from_numpy(arr.copy()).to(dev)
    if values.ndim == 2:
        values = values[..., None]
    grid = torch.stack([u/(STRIDE*(w-1))*2-1, v/(STRIDE*(h-1))*2-1],-1)[None,None]
    out = F.grid_sample(values.permute(2,0,1)[None],grid,align_corners=True,mode='bilinear')
    return out[0,:,0].T


@torch.no_grad()
def crossview(target, sources, bounds, dev):
    d, cam = target['depth'], target['camera']
    h,w = d.shape
    yy,xx = np.mgrid[:h,:w]
    z = torch.from_numpy(d.flatten()).to(dev)
    u = torch.from_numpy((xx*STRIDE).flatten()).to(dev)
    v = torch.from_numpy((yy*STRIDE).flatten()).to(dev)
    local = torch.stack([(u-cam['cx'])/cam['fx']*z, -(v-cam['cy'])/cam['fy']*z, -z],-1).float()
    c2w = torch.from_numpy(cam['c2w']).to(dev)
    world = local@c2w[:3,:3].T+c2w[:3,3]
    count = z.numel()
    inside_n = torch.zeros(count, device=dev)
    matches = torch.zeros_like(inside_n)
    matches2 = torch.zeros_like(inside_n)
    roundtrip_matches = torch.zeros_like(inside_n)
    bestz = torch.full_like(z, float('inf'))
    bestclear = torch.zeros(count,3,device=dev)
    bestgt = torch.zeros_like(bestclear)
    bestt = torch.zeros_like(bestclear)
    maxt = torch.zeros_like(bestclear)
    besterr = torch.full_like(z,float('inf'))
    for src in sources:
        sc = src['camera']; mat = torch.from_numpy(sc['c2w']).to(dev)
        p = (world-mat[:3,3])@mat[:3,:3]
        zs = -p[:,2]
        us = sc['fx']*p[:,0]/zs.clamp_min(1e-6)+sc['cx']
        vs = -sc['fy']*p[:,1]/zs.clamp_min(1e-6)+sc['cy']
        inside = (zs>0)&(us>=0)&(vs>=0)&(us<=STRIDE*(src['depth'].shape[1]-1))&(vs<=STRIDE*(src['depth'].shape[0]-1))
        sd = grid_sample_numpy(src['depth'],us,vs,dev)[:,0]
        sa = grid_sample_numpy(src['A'],us,vs,dev)[:,0]
        cv = grid_sample_numpy(src['cv'],us,vs,dev)[:,0]
        eligible = inside&(sa>=.9)&(cv<=.1)
        err = (sd-zs).abs()/zs.clamp_min(1e-6)
        inside_n += eligible.float()
        besterr = torch.minimum(besterr,torch.where(eligible,err,torch.full_like(err,float('inf'))))
        valid = eligible&(err<=.05)
        matches += valid.float(); matches2 += (eligible&(err<=.02)).float()
        back_local = torch.stack([(us-sc['cx'])/sc['fx']*sd, -(vs-sc['cy'])/sc['fy']*sd, -sd],-1)
        back_world = back_local@mat[:3,:3].T+mat[:3,3]
        back_target = (back_world-c2w[:3,3])@c2w[:3,:3]
        back_z = -back_target[:,2]
        back_u = cam['fx']*back_target[:,0]/back_z.clamp_min(1e-6)+cam['cx']
        back_v = -cam['fy']*back_target[:,1]/back_z.clamp_min(1e-6)+cam['cy']
        cycle = torch.sqrt((back_u-u)**2+(back_v-v)**2)
        valid &= (back_z>0)&(cycle<=2)
        roundtrip_matches += valid.float()
        ts = grid_sample_numpy(src['t'],us,vs,dev)
        maxt = torch.maximum(maxt,torch.where(valid[:,None],ts,torch.zeros_like(ts)))
        take = valid&(zs<bestz)
        bestz = torch.where(take,zs,bestz)
        bestt[take] = ts[take]
        bestclear[take] = grid_sample_numpy(src['clear'],us,vs,dev)[take]
        bestgt[take] = grid_sample_numpy(src['gt'],us,vs,dev)[take]
    arrays = dict(eligible_sources=inside_n, matches_5pct=matches, matches_2pct=matches2, roundtrip_matches=roundtrip_matches,
                  closest_depth=bestz, closest_clear=bestclear, closest_gt=bestgt,
                  closest_t=bestt, max_t=maxt, best_depth_error=besterr)
    arrays = {k: val.cpu().numpy().reshape((h,w)+(() if val.ndim==1 else (3,))) for k,val in arrays.items()}
    rows=[]
    for strict in (True,False):
        for region,mask in bands(target,bounds,strict).items():
            matched=mask&(arrays['roundtrip_matches']>0)
            eligible=mask&(arrays['eligible_sources']>0)
            closer=matched&(arrays['closest_depth']<.8*d)
            row=dict(mask='surface_candidate' if strict else 'A_ge_0p5',region=region,pixels=int(mask.sum()),
                     source_eligible_fraction=avg(arrays['eligible_sources']>0,mask),
                     any_match_5pct=avg(arrays['matches_5pct']>0,mask),
                     any_match_2pct=avg(arrays['matches_2pct']>0,mask),
                     any_roundtrip_2px=avg(arrays['roundtrip_matches']>0,mask),
                     unmatched_given_eligible=avg(arrays['matches_5pct']==0,eligible),
                     closer_source_fraction=avg(arrays['closest_depth']<.8*d,matched),
                     closest_source_depth_ratio=avg(arrays['closest_depth']/np.maximum(d,1e-6),matched),
                     clear_crossview_mae=avg(np.abs(arrays['closest_clear']-target['clear']),matched),
                     clear_crossview_mae_closer=avg(np.abs(arrays['closest_clear']-target['clear']),closer))
            for c,name in enumerate('rgb'):
                row[f'best_train_t_{name}']=avg(arrays['max_t'][...,c],matched)
                row[f'target_t_{name}_matched']=avg(target['t'][...,c],matched)
                row[f'closer_t_{name}']=avg(arrays['closest_t'][...,c],closer)
                row[f'target_t_{name}_closer']=avg(target['t'][...,c],closer)
            rows.append(row)
    return arrays, rows


@torch.no_grad()
def run(args):
    assert os.environ.get('CONDA_DEFAULT_ENV')=='water_splatting'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+SCENES.index(args.scene))
    assert sys.executable=='/opt/anaconda3/envs/water_splatting/bin/python'
    out=args.output.resolve()/args.scene
    out.mkdir(parents=True,exist_ok=False)
    (out/'visualizations').mkdir(); (out/'maps').mkdir()
    scene=args.scene
    configure(scene,'legacy','legacy')
    W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
    W._route_renderer_backend('legacy')
    extension, extpath=load_extension()
    runpath=SOURCE/scene/'R0/C0a'
    ckpt=runpath/'checkpoints/step-000014999.ckpt'
    ckhash=W._sha256(ckpt)
    start=time.perf_counter()
    branch=W._new_branch(scene)
    model=branch.pipeline.model
    try:
        payload=W._load_snapshot(branch,scene,ckpt)
        model.eval()
        initial=W._model_state_hash(model)
        assert initial==W._hash_object(payload['model'])
        del payload
        records={'train':W._train_records(branch),'eval':W._eval_records(branch)}
        if args.smoke:
            records={k:v[:1] for k,v in records.items()}
        with (runpath/'per_view_metrics.csv').open() as f:
            original_metrics={r['view_id']:r for r in csv.DictReader(f)}
        all_data={}; equivalence=[]; replay=[]; timings=[]
        n=int(model.num_points); dev=model.device
        sum_info=torch.zeros(n,3,device=dev); sum_info0=torch.zeros(n,device=dev)
        sum_weight=torch.zeros(n,device=dev); sum_depth=torch.zeros(n,device=dev)
        support=torch.zeros(n,device=dev); best_t=torch.zeros(n,3,device=dev)
        xtx=np.zeros((3,2,2),np.float64); xty={mode:np.zeros((3,2),np.float64) for mode in MODES}
        for split, recs in records.items():
            for idx,view,cam,batch in recs:
                tick=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                o,kw,render=render_capture(model,cam)
                call=model.underwater_rasterizer.rasterize
                gt=W._gt(model,W._batch_to_device(batch,dev),o['background']).float().clamp(0,1)
                unit=call(**dict(kw,colors=torch.ones_like(kw['colors'])))
                assert torch.equal(unit.accumulation,render.accumulation)
                max_A=float((unit.j_raw[...,0]-render.accumulation[...,0]).abs().max())
                if max_A>5e-6:
                    raise RuntimeError('Unit-color compositing/accumulation mismatch')
                t=unit.rgb_object/unit.j_raw.clamp_min(1e-8)
                data=dict(A=sample(render.accumulation)[...,0],depth=sample(render.depth)[...,0],
                          cv=sample(render.depth_std_relative)[...,0],clear=sample(render.j_raw),
                          pred=sample(o['pred_image'].clamp(0,1)),gt=sample(gt),obj=sample(render.rgb_object),
                          medium=sample(o['rgb_medium']),t=sample(t),S=sample(unit.rgb_object),
                          camera=camera_info(cam),split=split,view=str(view),probe={})
                if split=='train':
                    num,inf,w0,inf0,err=information(extension,kw,unit)
                    sum_info+=inf; sum_info0+=inf0; sum_weight+=w0
                    sum_depth+=w0*kw['depths'].reshape(-1)
                    valid=w0>=.1
                    support+=valid.float()
                    best_t=torch.maximum(best_t,torch.where(valid[:,None],num/w0[:,None].clamp_min(1e-8),torch.zeros_like(num)))
                    equivalence.append(dict(view=str(view),jacobian_sum_relative_error=err,unit_A_max_error=max_A))
                    del num,inf,w0,inf0
                for mode in MODES:
                    pos=call(**perturb(kw,mode,1)).rgb
                    neg=call(**perturb(kw,mode,-1)).rgb
                    data['probe'][mode]=sample((pos-neg)*.5)
                    del pos,neg
                if split=='train':
                    mask=(data['A']>=.9)&(data['cv']<=.1)&(data['depth']>0)
                    for c in range(3):
                        X=np.stack([data['obj'][...,c][mask],data['S'][...,c][mask]],-1).astype(np.float64)
                        xtx[c]+=X.T@X
                        for mode in MODES:
                            y=-data['probe'][mode][...,c][mask].astype(np.float64)
                            xty[mode][c]+=X.T@y
                if split=='eval':
                    target=out/'visualizations'/str(view); target.mkdir()
                    psnr=float(-10*torch.log10((o['pred_image'].clamp(0,1)-gt).square().mean()))
                    checks={}
                    for filename,img in [('underwater.png',o['pred_image']),('clear.png',render.rgb_clear),('clear_raw_display_clamp01.png',render.j_raw)]:
                        old=runpath/'renders'/scene/'step_14999'/str(view)/'C0a'/filename
                        actual=(img.float().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                        checks[filename]=bool(np.array_equal(actual,np.asarray(Image.open(old))))
                        W._save_native_rgb(target/filename,img)
                    W._save_native_rgb(target/'gt.png',gt)
                    assert all(checks.values()),checks
                    assert abs(psnr-float(original_metrics[str(view)]['PSNR']))<2e-5
                    replay.append(dict(view=str(view),png_exact=checks,psnr=psnr,original_psnr=original_metrics[str(view)]['PSNR']))
                all_data[(split,str(view))]=data
                timings.append(dict(split=split,view=str(view),seconds=time.perf_counter()-tick,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                print(json.dumps(dict(scene=scene,phase='observations',split=split,view=str(view),seconds=timings[-1]['seconds'])),flush=True)
                del o,kw,render,unit,t,gt
        train_data=[d for (s,v),d in all_data.items() if s=='train']
        pool=np.concatenate([d['depth'][(d['A']>=.9)&(d['cv']<=.1)&(d['depth']>0)] for d in train_data])
        boundaries=np.quantile(pool,[1/3,2/3]).tolist()
        coeff={mode:np.stack([np.linalg.lstsq(xtx[c],xty[mode][c],rcond=1e-10)[0] for c in range(3)]) for mode in MODES}
        write_json(out/'compensation_fit.json',dict(coefficients=coeff,xtx=xtx,bounds=boundaries,epsilon=EPS,
             definition='delta J=a*J+b shared across all Gaussian identities/views; training-only surface-candidate fit to negative symmetric medium probe; no GT'))
        rows=[]; tangent_rows=[]
        for (split,view),data in all_data.items():
            rows.extend(summarize_view(data,scene,split,view,boundaries))
            for region,mask in bands(data,boundaries).items():
                for mode in MODES:
                    delta=data['probe'][mode]
                    a,b=coeff[mode][:,0],coeff[mode][:,1]
                    compensated=delta+data['obj']*a+data['S']*b
                    for c,name in enumerate('rgb'):
                        e0=avg(delta[...,c]**2,mask); e1=avg(compensated[...,c]**2,mask)
                        tangent_rows.append(dict(scene=scene,split=split,view=view,region=region,mode=mode,channel=name,
                            pixels=int(mask.sum()),probe_rms=math.sqrt(e0),remaining_rms=math.sqrt(e1),
                            cancellable_energy=1-e1/max(e0,1e-20)))
            if split=='eval':
                observation_sheet(out/'visualizations'/(view+'_observation.png'),data,boundaries)
            np.savez_compressed(out/'maps'/(split+'_'+view+'.npz'),**{k:v for k,v in data.items() if isinstance(v,np.ndarray)},
                                **{'probe_'+m:v for m,v in data['probe'].items()})
        W._write_csv(out/'regions.csv',rows); W._write_csv(out/'compensation_tangent.csv',tangent_rows)
        gw=sum_weight.cpu().numpy(); gi=sum_info.cpu().numpy(); gi0=sum_info0.cpu().numpy()
        gd=(sum_depth/sum_weight.clamp_min(1e-8)).cpu().numpy(); gs=support.cpu().numpy(); bt=best_t.cpu().numpy()
        gauss_rows=[]
        for name,lo,hi in [('near',0,boundaries[0]),('mid',boundaries[0],boundaries[1]),('far',boundaries[1],np.inf)]:
            mask=(gw>=.1)&(gd>=lo)&(gd<hi)&(gi0>1e-10)
            row=dict(scene=scene,region=name,gaussians=int(mask.sum()),support_mean=avg(gs,mask),support_lt_3=avg(gs<3,mask))
            for c,ch in enumerate('rgb'):
                ratio=gi[:,c]/np.maximum(gi0,1e-20)
                row.update({f'info_retained_{ch}':float(gi[mask,c].sum()/max(gi0[mask].sum(),1e-20)),
                            f'info_ratio_{ch}_median':quant(ratio[mask],.5),
                            f'best_t_{ch}_lt_0p3':avg(bt[:,c]<.3,mask),f'best_t_{ch}_median':quant(bt[mask,c],.5)})
            gauss_rows.append(row)
        W._write_csv(out/'gaussian_observation.csv',gauss_rows)
        np.savez_compressed(out/'gaussian_observation.npz',weight=gw,information=gi,information_zero_medium=gi0,
                            mean_contribution_depth=gd,view_support=gs,best_transmission=bt)
        comp_rows=[]; geometry_rows=[]
        for idx,view,cam,batch in records['eval']:
            data=all_data[('eval',str(view))]
            o,kw,render=render_capture(model,cam); call=model.underwater_rasterizer.rasterize
            for mode in MODES:
                ab=torch.from_numpy(coeff[mode]).to(device=dev,dtype=kw['colors'].dtype)
                change=kw['colors']*ab[:,0]+ab[:,1]
                for sign in (-1,1):
                    inp=perturb(kw,mode,sign)
                    before=kw['colors']+sign*change
                    altered=before.clamp(0,1)
                    probe=call(**inp)
                    compensated=call(**dict(inp,colors=altered))
                    delta=sample(probe.rgb-render.rgb); cd=sample(compensated.rgb-render.rgb)
                    jc=sample(compensated.j_raw-render.j_raw)
                    for region,mask in bands(data,boundaries).items():
                        e0=avg(delta**2,mask); e1=avg(cd**2,mask)
                        comp_rows.append(dict(scene=scene,view=str(view),region=region,mode=mode,sign=sign,pixels=int(mask.sum()),
                            probe_rms=math.sqrt(e0),compensated_rms=math.sqrt(e1),cancellable_energy=1-e1/max(e0,1e-20),
                            clear_change_rms=math.sqrt(avg(jc**2,mask)),clipped_gaussian_channel_fraction=float(((before<0)|(before>1)).float().mean())))
                    del probe,compensated
            maps,geo=crossview(data,train_data,boundaries,dev)
            for row in geo:
                geometry_rows.append(dict(scene=scene,view=str(view),**row))
            np.savez_compressed(out/'maps'/('crossview_'+str(view)+'.npz'),**maps)
            valid=(maps['roundtrip_matches']>0)
            diff=np.abs(maps['closest_clear']-data['clear'])*5; diff[~valid]=0
            sheet(out/'visualizations'/(str(view)+'_geometry.png'),[
                ('Target clear raw',data['clear']),('Closest matched train clear',maps['closest_clear']),
                ('Clear cross-view abs diff x5',diff),('Roundtrip training matches [0,5]',color_map(maps['roundtrip_matches'],0,5)),
                ('Target underwater GT',data['gt']),('Closest matched train GT',maps['closest_gt']),
                ('Closest/target depth [0.5,1.5]',color_map(maps['closest_depth']/np.maximum(data['depth'],1e-6),.5,1.5)),
                ('Depth disagreement min [0,0.1]',color_map(maps['best_depth_error'],0,.1,'magma'))])
            print(json.dumps(dict(scene=scene,phase='compensation_geometry',view=str(view))),flush=True)
            del o,kw,render
        W._write_csv(out/'compensation_native.csv',comp_rows); W._write_csv(out/'geometry.csv',geometry_rows)
        W._write_csv(out/'timings.csv',timings)
        assert W._model_state_hash(model)==initial
        assert W._sha256(ckpt)==ckhash
        write_json(out/'audit.json',dict(scene=scene,completed=True,smoke=args.smoke,checkpoint=str(ckpt),checkpoint_sha256=ckhash,
            model_state_sha256=initial,model_unchanged=True,checkpoint_unchanged=True,optimizer_steps=0,backward_calls=0,
            train_views=len(records['train']),eval_views=len(records['eval']),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            python=sys.executable,extension=str(extpath),extension_sha256=W._sha256(extpath),replay=replay,
            information_equivalence=equivalence,depth_boundaries=boundaries,seconds=time.perf_counter()-start,
            camera_metadata={s+'/'+v:d['camera'] for (s,v),d in all_data.items()},
            limitations=['surface candidates are model-dependent, not semantic water masks','cross-view depth and roundtrip consistency are not GT geometry',
                         'compensation probes do not recover physical medium or train a new model','information proxy omits parameter cross terms and calibrated noise',
                         'no clear GT; not a restoration accuracy claim','regional pixel summaries use deterministic stride 4; Gaussian information uses native full resolution']))
    finally:
        W._release(branch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
