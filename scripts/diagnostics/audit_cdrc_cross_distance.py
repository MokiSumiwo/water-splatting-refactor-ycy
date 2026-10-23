#!/usr/bin/env python3
"""Read-only CDRC cross-distance correspondence and prediction audit.

The audit consumes frozen FSB far-field maps and original underwater images.
Formal prediction uses two no-grad native rasterizer calls per view, with model
and input hash checks and exact replay against the existing frozen maps.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
MAP_ROOT = ROOT / "outputs/fsb_far_field_diagnostic_20260916_v1"
DATA_ROOT = ROOT / "undistorted_data"
DEFAULT_OUT = ROOT / "outputs/cdrc_cross_distance_diagnostic_20260917_v1"
MAX_FEATURES = 1800
RATIO = 0.78
ZNCC_THRESHOLD = 0.50
DEPTH_RATIO_THRESHOLD = 1.25
DEPTH_REL_THRESHOLD = 0.05
ROUNDTRIP_PX = 2.0
FEATURE_REPROJECTION_PX = 3.0
STRIDE = 4.0


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in fields} for row in rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def mean_or_nan(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def median_or_nan(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def camera_center(cam: Mapping[str, Any]) -> np.ndarray:
    c2w = np.asarray(cam["c2w"], dtype=np.float64)
    return c2w[:3, 3]


def bilinear(arr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample a map whose pixels are spaced every STRIDE full-res pixels."""
    h, w = arr.shape[:2]
    x = np.asarray(uv[:, 0], dtype=np.float64) / STRIDE
    y = np.asarray(uv[:, 1], dtype=np.float64) / STRIDE
    inside = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x = np.clip(np.nan_to_num(x, nan=0, posinf=0, neginf=0), 0, w - 1)
    y = np.clip(np.nan_to_num(y, nan=0, posinf=0, neginf=0), 0, h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    if arr.ndim == 2:
        out = ((1 - dx) * (1 - dy) * arr[y0, x0] + dx * (1 - dy) * arr[y0, x1] +
               (1 - dx) * dy * arr[y1, x0] + dx * dy * arr[y1, x1])
        out[~inside] = np.nan
        return out.astype(np.float32)
    out = ((1 - dx)[:, None] * (1 - dy)[:, None] * arr[y0, x0] +
           dx[:, None] * (1 - dy)[:, None] * arr[y0, x1] +
           (1 - dx)[:, None] * dy[:, None] * arr[y1, x0] +
           dx[:, None] * dy[:, None] * arr[y1, x1])
    out[~inside] = np.nan
    return out.astype(np.float32)


def image_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0], 0, w - 1)
    y = np.clip(uv[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    dx = (x - x0).astype(np.float32); dy = (y - y0).astype(np.float32)
    return ((1 - dx)[:, None] * (1 - dy)[:, None] * image[y0, x0] +
            dx[:, None] * (1 - dy)[:, None] * image[y0, x1] +
            (1 - dx)[:, None] * dy[:, None] * image[y1, x0] +
            dx[:, None] * dy[:, None] * image[y1, x1]).astype(np.float32)


def project_from_depth(uv: np.ndarray, depth: np.ndarray, target_cam: Mapping[str, Any], source_cam: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project target pixels through target depth into source camera."""
    fx, fy, cx, cy = (float(target_cam[k]) for k in ("fx", "fy", "cx", "cy"))
    local = np.stack(((uv[:, 0] - cx) / fx * depth,
                      -(uv[:, 1] - cy) / fy * depth,
                      -depth), axis=1)
    c2w = np.asarray(target_cam["c2w"], dtype=np.float64)
    world = local @ c2w[:3, :3].T + c2w[:3, 3]
    s2w = np.asarray(source_cam["c2w"], dtype=np.float64)
    p = (world - s2w[:3, 3]) @ s2w[:3, :3]
    z = -p[:, 2]
    sfx, sfy, scx, scy = (float(source_cam[k]) for k in ("fx", "fy", "cx", "cy"))
    su = sfx * p[:, 0] / np.maximum(z, 1e-6) + scx
    sv = -sfy * p[:, 1] / np.maximum(z, 1e-6) + scy
    return np.stack((su, sv), axis=1), z.astype(np.float32), world.astype(np.float32)


def roundtrip(target_uv: np.ndarray, source_uv: np.ndarray, source_depth: np.ndarray,
              target_cam: Mapping[str, Any], source_cam: Mapping[str, Any]) -> np.ndarray:
    sfx, sfy, scx, scy = (float(source_cam[k]) for k in ("fx", "fy", "cx", "cy"))
    local = np.stack(((source_uv[:, 0] - scx) / sfx * source_depth,
                      -(source_uv[:, 1] - scy) / sfy * source_depth,
                      -source_depth), axis=1)
    s2w = np.asarray(source_cam["c2w"], dtype=np.float64)
    world = local @ s2w[:3, :3].T + s2w[:3, 3]
    t2w = np.asarray(target_cam["c2w"], dtype=np.float64)
    p = (world - t2w[:3, 3]) @ t2w[:3, :3]
    z = -p[:, 2]
    tfx, tfy, tcx, tcy = (float(target_cam[k]) for k in ("fx", "fy", "cx", "cy"))
    tu = tfx * p[:, 0] / np.maximum(z, 1e-6) + tcx
    tv = -tfy * p[:, 1] / np.maximum(z, 1e-6) + tcy
    return np.sqrt((tu - target_uv[:, 0]) ** 2 + (tv - target_uv[:, 1]) ** 2)


def patch_zncc(a: np.ndarray, b: np.ndarray, ua: np.ndarray, ub: np.ndarray, radius: int = 5) -> np.ndarray:
    gray_a = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = np.full(len(ua), np.nan, dtype=np.float32)
    for k, (pa, pb) in enumerate(zip(ua, ub)):
        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
        if (xa < radius or ya < radius or xb < radius or yb < radius or
                xa + radius >= gray_a.shape[1] or xb + radius >= gray_b.shape[1] or
                ya + radius >= gray_a.shape[0] or yb + radius >= gray_b.shape[0]):
            continue
        pa_ = gray_a[ya-radius:ya+radius+1, xa-radius:xa+radius+1].astype(np.float32)
        pb_ = gray_b[yb-radius:yb+radius+1, xb-radius:xb+radius+1].astype(np.float32)
        pa_ -= pa_.mean(); pb_ -= pb_.mean()
        den = np.linalg.norm(pa_) * np.linalg.norm(pb_)
        result[k] = float((pa_ * pb_).sum() / den) if den > 1e-6 else 0.0
    return result


def surface_candidate(map_data: Mapping[str, np.ndarray], uv: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fields = {k: bilinear(map_data[k], uv) for k in ("depth", "A", "cv", "t", "S", "medium", "pred", "gt")}
    valid = ((fields["A"] >= 0.9) & (fields["cv"] <= 0.1) &
             np.isfinite(fields["depth"]) & (fields["depth"] > 0))
    return valid, fields


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_scene(scene, smoke=False):
    audit_path = MAP_ROOT / scene / 'audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['completed'] and not audit['smoke'] and audit['checkpoint_unchanged']
    names = sorted(audit['camera_metadata'])
    if smoke:
        names = [k for k in names if k.startswith('train/')][:6] + [k for k in names if k.startswith('eval/')][:1]
    cams = {name: audit['camera_metadata'][name] for name in names}
    images, maps, hashes = {}, {}, {str(audit_path): sha256(audit_path)}
    dataset = DATA_ROOT / ('undistorted_' + scene)
    for split, prefix in [('train', 'train'), ('test', 'eval')]:
        p = dataset / (split + '_list.txt')
        expected = {Path(x).stem for x in p.read_text().splitlines() if x.strip()}
        actual = {k.split('/')[1] for k in audit['camera_metadata'] if k.startswith(prefix + '/')}
        assert actual == expected, (scene, split, actual ^ expected)
        hashes[str(p)] = sha256(p)
    checks = []
    for name in names:
        path = dataset / 'images/ColorImage' / (name.split('/')[1] + '.png')
        images[name] = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
        cam = cams[name]
        assert images[name].shape[:2] == (cam['height'], cam['width'])
        mp = MAP_ROOT / scene / 'maps' / (name.replace('/', '_') + '.npz')
        with np.load(mp) as loaded:
            maps[name] = {key: loaded[key] for key in ('A','depth','cv','pred','medium','t','S','gt','obj')}
        m = maps[name]
        gt_error = float(np.max(np.abs(m['gt'] - images[name][::4, ::4])))
        identity_error = float(np.max(np.abs(m['pred'] - np.clip(m['obj'] + m['medium'],0,1))))
        mass_error = float(np.max(np.abs(m['S'] - m['A'][...,None] * m['t'])))
        assert max(gt_error, identity_error, mass_error) < 2e-6
        checks.append(dict(view=name, image_map_gt_max_error=gt_error,
                           renderer_sum_max_error=identity_error, mass_max_error=mass_error))
        del m['obj']
        hashes[str(path)] = sha256(path)
        hashes[str(mp)] = sha256(mp)
    ckpt = Path(audit['checkpoint'])
    assert sha256(ckpt) == audit['checkpoint_sha256']
    hashes[str(ckpt)] = audit['checkpoint_sha256']
    return cams, images, maps, audit, hashes, checks


def sift_features(images):
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES, contrastThreshold=0.015)
    keypoints, descriptors = {}, {}
    for name, im in images.items():
        gray = cv2.cvtColor((im * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        # SIFT sometimes gives multiple orientations at exactly the same location.
        keep, seen = [], set()
        for i, k in enumerate(kp):
            xy = tuple(round(float(v), 3) for v in k.pt)
            if xy not in seen:
                keep.append(i); seen.add(xy)
        keypoints[name] = [kp[i] for i in keep]
        descriptors[name] = desc[keep] if keep else np.empty((0,128),np.float32)
    return keypoints, descriptors


def mutual_ratio_matches(desc_a, desc_b):
    counts = dict(raw_knn_ab=0, raw_knn_ba=0, ratio_ab=0, ratio_ba=0, ratio_mutual=0)
    if min(len(desc_a), len(desc_b)) < 2:
        return [], counts
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    ab = bf.knnMatch(desc_a,desc_b,k=2); ba = bf.knnMatch(desc_b,desc_a,k=2)
    rab = [p[0] for p in ab if len(p)==2 and p[0].distance < RATIO*p[1].distance]
    rba = {p[0].queryIdx:p[0].trainIdx for p in ba if len(p)==2 and p[0].distance < RATIO*p[1].distance}
    matches = [m for m in rab if rba.get(m.trainIdx)==m.queryIdx]
    counts.update(raw_knn_ab=sum(len(p)==2 for p in ab), raw_knn_ba=sum(len(p)==2 for p in ba),
                  ratio_ab=len(rab), ratio_ba=len(rba), ratio_mutual=len(matches))
    return matches, counts


def region(depth, bounds):
    return 'near' if depth < bounds[0] else ('mid' if depth < bounds[1] else 'far')


def directional_geometry(ua, ub, fa, fb, ca, cb):
    projected, z, _ = project_from_depth(ua, fa['depth'], ca, cb)
    reproj = np.linalg.norm(projected-ub, axis=-1)
    err = np.abs(fb['depth']-z)/np.maximum(z,1e-6)
    cyc = roundtrip(ua, projected, fb['depth'],ca,cb)
    ok = ((z>0)&(err<=DEPTH_REL_THRESHOLD)&(cyc<=ROUNDTRIP_PX)&(reproj<=FEATURE_REPROJECTION_PX))
    return ok, err, cyc, reproj


def pair_audit(a,b,cams,images,maps,keypoints,descriptors,bounds,native=None):
    matches, counts = mutual_ratio_matches(descriptors[a],descriptors[b])
    row = dict(view_a=a,view_b=b,split='eval' if a.startswith('eval/') else 'train',**counts,
               model_geometry=0,patch_zncc=0,cross_distance=0)
    if not matches:
        return row, [], []
    ua = np.asarray([keypoints[a][m.queryIdx].pt for m in matches],np.float32)
    ub = np.asarray([keypoints[b][m.trainIdx].pt for m in matches],np.float32)
    if native is None:
        va,fa = surface_candidate(maps[a],ua); vb,fb = surface_candidate(maps[b],ub)
    else:
        fa={k:v[[m.queryIdx for m in matches]] for k,v in native[a].items()}
        fb={k:v[[m.trainIdx for m in matches]] for k,v in native[b].items()}
        va=(fa['A']>=.9)&(fa['cv']<=.1)&np.isfinite(fa['depth'])&(fa['depth']>0)
        vb=(fb['A']>=.9)&(fb['cv']<=.1)&np.isfinite(fb['depth'])&(fb['depth']>0)
    ga,ea,ca,ra = directional_geometry(ua,ub,fa,fb,cams[a],cams[b])
    gb,eb,cb,rb = directional_geometry(ub,ua,fb,fa,cams[b],cams[a])
    geom = va & vb & ga & gb
    zncc = patch_zncc(images[a],images[b],ua,ub)
    patch_ok = geom & np.isfinite(zncc) & (zncc>=ZNCC_THRESHOLD)
    gap = np.maximum(fa['depth'],fb['depth'])/np.maximum(np.minimum(fa['depth'],fb['depth']),1e-6)
    cross = patch_ok & (gap>=DEPTH_RATIO_THRESHOLD)
    row.update(model_geometry=int(geom.sum()),patch_zncc=int(patch_ok.sum()),cross_distance=int(cross.sum()),
               median_cross_depth_ratio=median_or_nan(gap[cross]),median_zncc=median_or_nan(zncc[patch_ok]),
               median_reprojection_px=median_or_nan(np.maximum(ra,rb)[geom]))
    ia=image_sample(images[a],ua); ib=image_sample(images[b],ub)
    observations=[]
    directions=[(a,b,ua,ub,fa,fb,ia,ib,False)]
    if a.startswith('train/'):
        directions.append((b,a,ub,ua,fb,fa,ib,ia,True))
    for target,source,ut,us,ft,fs,it,is_,reverse in directions:
        for k in np.flatnonzero(cross):
            m=matches[k]
            rec=dict(target=target,source=source,split=target.split('/')[0],
                     keypoint=int(m.trainIdx if reverse else m.queryIdx),
                     source_keypoint=int(m.queryIdx if reverse else m.trainIdx),
                     target_region=region(float(ft['depth'][k]),bounds),depth_ratio=float(gap[k]),
                     depth_target=float(ft['depth'][k]),depth_source=float(fs['depth'][k]),
                     source_closer=bool(fs['depth'][k] < ft['depth'][k]),zncc=float(zncc[k]),
                     u=float(ut[k,0]),v=float(ut[k,1]),source_u=float(us[k,0]),source_v=float(us[k,1]))
            for c,ch in enumerate('rgb'):
                for label,val in [('target',it[k]),('source_rgb',is_[k]),('renderer',ft['pred'][k]),
                                  ('target_grid',ft['gt'][k]),('source_grid',fs['gt'][k]),
                                  ('renderer_grid',ft.get('grid_pred',ft['pred'])[k]),
                                  ('S_target_grid',ft.get('grid_S',ft['S'])[k]),
                                  ('S_source_grid',fs.get('grid_S',fs['S'])[k]),
                                  ('M_target_grid',ft.get('grid_medium',ft['medium'])[k]),
                                  ('M_source_grid',fs.get('grid_medium',fs['medium'])[k]),
                                  ('t_target',ft['t'][k]),('t_source',fs['t'][k]),
                                  ('S_target',ft['S'][k]),('S_source',fs['S'][k]),
                                  ('M_target',ft['medium'][k]),('M_source',fs['medium'][k])]:
                    rec[label+'_'+ch]=float(val[c])
            rec['t_advantage']=float(np.mean(fs['t'][k]-ft['t'][k]))
            observations.append(rec)
    debug=[(ua[k],ub[k],bool(cross[k])) for k in range(len(matches))]
    return row,observations,debug


def vector(rec,label):
    return np.asarray([rec[label+'_'+c] for c in 'rgb'],np.float64)


def prediction_record(group,scope):
    # Pick the nearest source by model depth, never by prediction error or test GT.
    group=sorted(group,key=lambda r:(r['depth_source'],r['source']))
    r=group[0]
    rec={k:r[k] for k in ('target','source','split','keypoint','target_region','depth_ratio','depth_target',
                          'depth_source','source_closer','u','v','source_u','source_v','t_advantage')}
    rec.update(scope=scope,source_count=len(group))
    ts,tt=vector(r,'t_source'),vector(r,'t_target')
    ss,st=vector(r,'S_source'),vector(r,'S_target')
    ms,mt=vector(r,'M_source'),vector(r,'M_target')
    source=vector(r,'source_rgb'); gt=vector(r,'target'); grid=vector(r,'target_grid')
    jt=(source-ms)/np.maximum(ts,.1)
    js=(source-ms)/np.maximum(ss,.1)
    masses=np.stack([vector(x,'S_source') for x in group])
    direct=np.stack([vector(x,'source_rgb')-vector(x,'M_source') for x in group])
    jm=(masses*direct).sum(0)/np.maximum((masses*masses).sum(0),.01)
    predictions={'renderer':vector(r,'renderer'),'source_copy':source,
                 'cdrc_t':tt*jt+mt,'cdrc_mass':st*js+mt,'cdrc_bounded':st*np.clip(js,0,1)+mt,
                 'cdrc_multi':st*np.clip(jm,0,1)+mt}
    for label,j in [('J_single',js),('J_multi',jm)]:
        rec[label+'_outside_fraction']=float(((j<0)|(j>1)).mean())
        rec[label+'_min']=float(j.min()); rec[label+'_max']=float(j.max())
    rec['source_mass_floor_fraction']=float((ss<.1).mean())
    rec['source_sampling_mse']=float(np.mean((source-vector(r,'source_grid'))**2))
    rec['target_sampling_mse']=float(np.mean((gt-grid)**2))
    for method,pred in predictions.items():
        rec[method+'_unclipped_mse']=float(np.mean((pred-gt)**2))
        rec[method+'_clip_fraction']=float(((pred<0)|(pred>1)).mean())
        pred=np.clip(pred,0,1)
        rec[method+'_mse']=float(np.mean((pred-gt)**2))
        rec[method+'_mae']=float(np.mean(np.abs(pred-gt)))
        rec[method+'_grid_mse']=float(np.mean((pred-grid)**2))
        for c,ch in enumerate('rgb'):
            rec[method+'_'+ch]=float(pred[c])
            rec['target_'+ch]=float(gt[c])
    # Independent same-sampling-lattice control: fit using interpolated source GT.
    jgrid=(vector(r,'source_grid')-vector(r,'M_source_grid'))/np.maximum(vector(r,'S_source_grid'),.1)
    rec['cdrc_grid_control_mse']=float(np.mean((np.clip(vector(r,'S_target_grid')*np.clip(jgrid,0,1)+vector(r,'M_target_grid'),0,1)-grid)**2))
    rec['renderer_grid_control_mse']=float(np.mean((vector(r,'renderer_grid')-grid)**2))
    return rec


METHODS=('renderer','source_copy','cdrc_t','cdrc_mass','cdrc_bounded','cdrc_multi')


def aggregate_prediction(rows):
    out={'n':len(rows),'target_views':len({r['target'] for r in rows})}
    for method in METHODS:
        out[method+'_rmse']=float(np.sqrt(np.mean([r[method+'_mse'] for r in rows]))) if rows else None
        views=sorted({r['target'] for r in rows})
        out[method+'_view_mean_rmse']=float(np.mean([np.sqrt(np.mean([r[method+'_mse'] for r in rows if r['target']==v])) for v in views])) if rows else None
        out[method+'_grid_rmse']=float(np.sqrt(np.mean([r[method+'_grid_mse'] for r in rows]))) if rows else None
    for key in ('J_single_outside_fraction','J_multi_outside_fraction','source_mass_floor_fraction','t_advantage'):
        out[key]=float(np.mean([r[key] for r in rows])) if rows else None
    out['t_advantage_median']=float(np.median([r['t_advantage'] for r in rows])) if rows else None
    out['target_sampling_rmse']=float(np.sqrt(np.mean([r['target_sampling_mse'] for r in rows]))) if rows else None
    out['cdrc_grid_control_rmse']=float(np.sqrt(np.mean([r['cdrc_grid_control_mse'] for r in rows]))) if rows else None
    out['renderer_grid_control_rmse']=float(np.sqrt(np.mean([r['renderer_grid_control_mse'] for r in rows]))) if rows else None
    # View-cluster bootstrap, descriptive only; few evaluation views are not independent trials.
    if rows:
        views=sorted({r['target'] for r in rows}); rng=np.random.RandomState(170917)
        for control in ('renderer','source_copy'):
            d=np.asarray([np.mean([r['cdrc_bounded_mse']-r[control+'_mse'] for r in rows if r['target']==v]) for v in views])
            draws=rng.choice(d,(2000,len(d)),replace=True).mean(1)
            out['bounded_minus_'+control+'_view_mse']=float(d.mean())
            out['bounded_minus_'+control+'_view_bootstrap_95']=np.quantile(draws,[.025,.975]).tolist()
    return out


def save_match_visualization(path,a,b,images,debug):
    ims=[Image.fromarray((images[x]*255).round().astype(np.uint8)) for x in (a,b)]
    scale=min(1,900/max(im.width for im in ims))
    ims=[im.resize((round(im.width*scale),round(im.height*scale))) for im in ims]
    canvas=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)+36),'#17191d')
    canvas.paste(ims[0],(0,36)); canvas.paste(ims[1],(ims[0].width,36)); draw=ImageDraw.Draw(canvas)
    draw.text((5,8),a+' | '+b+' | green: cross-distance; gray: rejected (max 100)',fill='white')
    # Restrict rejected matches for legibility; every accepted match remains visible.
    selected=[x for x in debug if not x[2]][:100]+[x for x in debug if x[2]]
    for ua,ub,ok in selected:
        x,y=ua*scale; xx,yy=ub*scale; xx+=ims[0].width; y+=36; yy+=36
        col=(40,240,90) if ok else (100,100,100)
        draw.line((x,y,xx,yy),fill=col,width=1)
        draw.ellipse((x-2,y-2,x+2,y+2),fill=col)
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def view_support(path,name,im,m,bounds,keypoints,supported):
    h,w=m['depth'].shape
    valid=(m['A']>=.9)&(m['cv']<=.1)&np.isfinite(m['depth'])&(m['depth']>0)
    regions=np.zeros((h,w,3),np.uint8)
    regions[valid&(m['depth']<bounds[0])]=(40,140,220)
    regions[valid&(m['depth']>=bounds[0])&(m['depth']<bounds[1])]=(60,170,80)
    regions[valid&(m['depth']>=bounds[1])]=(240,150,30)
    original=Image.fromarray((im*255).round().astype(np.uint8)); original.thumbnail((900,600))
    reg=Image.fromarray(regions).resize(original.size,Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(original.width*2,original.height+40),'#17191d')
    canvas.paste(original,(0,40)); canvas.paste(reg,(original.width,40));d=ImageDraw.Draw(canvas)
    d.text((4,5),name+' | cyan dots: validated closer source | near blue / mid green / far orange',fill='white')
    scale=original.width/im.shape[1]
    for idx in sorted(supported):
        u,v=keypoints[idx].pt; x,y=u*scale,v*scale+40
        for offset in (0,original.width):
            d.ellipse((x+offset-3,y-3,x+offset+3,y+3),outline=(0,255,255),width=2)
    canvas.save(path)


def native_features(scene,cams,keypoints,maps,out):
    """Two frozen native calls/view, then retain only values at measured features."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(ROOT))
    from scripts.diagnostics.audit_fsb_far_field import render_capture
    from scripts.experiments.run_fsb_cica_fork_worker import configure,W
    configure(scene,'legacy','legacy');W.ENFORCE_SCENE_GPU_ASSIGNMENT=False
    W._route_renderer_backend('legacy')
    audit=json.loads((MAP_ROOT/scene/'audit.json').read_text())
    ckpt=Path(audit['checkpoint'])
    branch=W._new_branch(scene)
    sampled={};checks=[];peak=0
    start=time.perf_counter()
    try:
        payload=W._load_snapshot(branch,scene,ckpt);model=branch.pipeline.model;model.eval()
        before=W._model_state_hash(model)
        assert before==W._hash_object(payload['model'])==audit['model_state_sha256']
        del payload
        records={'train':W._train_records(branch),'eval':W._eval_records(branch)}
        with torch.no_grad():
            for split,recs in records.items():
                for idx,view,cam,batch in recs:
                    name=split+'/'+str(view)
                    if name not in cams: continue
                    torch.cuda.reset_peak_memory_stats()
                    o,kw,render=render_capture(model,cam)
                    unit=model.underwater_rasterizer.rasterize(**dict(kw,colors=torch.ones_like(kw['colors'])))
                    assert torch.equal(unit.accumulation,render.accumulation)
                    fields={'A':render.accumulation,'depth':render.depth,'cv':render.depth_std_relative,
                            'pred':o['pred_image'].clamp(0,1),'medium':o['rgb_medium'],
                            'S':unit.rgb_object,'t':unit.rgb_object/unit.j_raw.clamp_min(1e-8)}
                    uv=torch.tensor([k.pt for k in keypoints[name]],device=model.device,dtype=torch.float32).reshape(-1,2)
                    h,w=fields['pred'].shape[:2]
                    grid=torch.stack((uv[:,0]/(w-1)*2-1,uv[:,1]/(h-1)*2-1),-1)[None,None]
                    sampled[name]={}
                    for field,tensor in fields.items():
                        if tensor.ndim==2: tensor=tensor[...,None]
                        val=F.grid_sample(tensor.permute(2,0,1)[None],grid,align_corners=True,mode='bilinear')[0,:,0].T
                        arr=val.float().cpu().numpy()
                        sampled[name][field]=arr[:,0] if field in ('A','depth','cv') else arr
                    # GT is only for interpolation control; source colors always come from input images.
                    sampled[name]['gt']=bilinear(maps[name]['gt'],uv.cpu().numpy())
                    for field in ('pred','S','medium'):
                        sampled[name]['grid_'+field]=bilinear(maps[name][field],uv.cpu().numpy())
                    replay={field:float((fields[field][::4,::4].squeeze(-1).cpu()-torch.from_numpy(maps[name][field])).abs().max()) for field in ('pred','A','depth','cv','S','t','medium')}
                    if max(replay.values())>2e-5: raise RuntimeError(('native replay mismatch',name,replay))
                    peak=max(peak,torch.cuda.max_memory_allocated()/1024**3)
                    checks.append(dict(view=name,replay_max_errors=replay))
                    print(json.dumps(dict(scene=scene,native_sampled=name,replay_max_error=max(replay.values()))),flush=True)
                    del o,kw,render,unit,fields,tensor,val
        after=W._model_state_hash(model)
        assert before==after and sha256(ckpt)==audit['checkpoint_sha256']
        arrays={name.replace('/','__')+'__'+field:arr for name,vals in sampled.items() for field,arr in vals.items()}
        arrays.update({name.replace('/','__')+'__uv':np.asarray([k.pt for k in keypoints[name]],np.float32).reshape(-1,2) for name in sampled})
        np.savez_compressed(out/'native_keypoint_samples.npz',**arrays)
        native=dict(completed=True,views=len(sampled),seconds=time.perf_counter()-start,peak_cuda_allocated_gb=peak,
                    model_state_before=before,model_state_after=after,model_unchanged=True,checkpoint_unchanged=True,
                    optimizer_steps=0,backward_calls=0,replay=checks)
        write_json(out/'native_audit.json',native)
        return sampled,native
    finally:
        W._release(branch)


def run_scene(scene,out,smoke,native_enabled=False):
    started=time.perf_counter(); scene_out=out/scene;scene_out.mkdir(parents=True,exist_ok=False)
    (scene_out/'visualizations').mkdir()
    cams,images,maps,audit,hashes,checks=load_scene(scene,smoke)
    keypoints,descriptors=sift_features(images);bounds=audit['depth_boundaries']
    native,native_audit=native_features(scene,cams,keypoints,maps,scene_out) if native_enabled else (None,{})
    train=[x for x in cams if x.startswith('train/')]; evaluation=[x for x in cams if x.startswith('eval/')]
    pairs=[(a,b) for i,a in enumerate(train) for b in train[i+1:]]+[(a,b) for a in evaluation for b in train]
    pair_rows=[]; observations=[];visuals=[]
    for i,(a,b) in enumerate(pairs):
        row,obs,debug=pair_audit(a,b,cams,images,maps,keypoints,descriptors,bounds,native)
        pair_rows.append(row);observations.extend(obs)
        visuals.append((row['cross_distance'],a,b,debug))
        visuals=sorted(visuals,key=lambda x:(-x[0],x[1],x[2]))[:4]
        if (i+1)%50==0 or i+1==len(pairs):
            print(json.dumps(dict(scene=scene,pairs_done=i+1,pairs_total=len(pairs),cross=sum(r['cross_distance'] for r in pair_rows))),flush=True)
    groups={}
    for r in observations:
        groups.setdefault((r['target'],r['keypoint']),[]).append(r)
    predictions=[]
    for group in groups.values():
        predictions.append(prediction_record(group,'all_cross'))
        closer=[r for r in group if r['source_closer']]
        if closer:
            predictions.append(prediction_record(closer,'closer_source'))
    for i,(_,a,b,debug) in enumerate(visuals):
        save_match_visualization(scene_out/'visualizations'/('pair_%02d.png'%i),a,b,images,debug)
    cover=[]
    for name in cams:
        m=maps[name];kp=keypoints[name]
        uv=np.asarray([k.pt for k in kp],np.float32).reshape(-1,2)
        if native is None:
            valid,fields=surface_candidate(m,uv)
        else:
            fields=native[name]
            valid=(fields['A']>=.9)&(fields['cv']<=.1)&np.isfinite(fields['depth'])&(fields['depth']>0)
        accepted={r['keypoint'] for r in predictions if r['target']==name and r['scope']=='closer_source'}
        view_support(scene_out/'visualizations'/(name.replace('/','_')+'_support.png'),name,images[name],m,bounds,kp,accepted)
        for reg in ('all','near','mid','far'):
            mask=(m['A']>=.9)&(m['cv']<=.1)&np.isfinite(m['depth'])&(m['depth']>0)
            kpmask=valid.copy()
            if reg!='all':
                lo,hi={'near':(0,bounds[0]),'mid':(bounds[0],bounds[1]),'far':(bounds[1],np.inf)}[reg]
                mask&=(m['depth']>=lo)&(m['depth']<hi)
                kpmask&=(fields['depth']>=lo)&(fields['depth']<hi)
            accepted_here={i for i in accepted if kpmask[i]}
            cells={(int(round(uv[i,1]/4)),int(round(uv[i,0]/4))) for i in accepted_here}
            cells={(y,x) for y,x in cells if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]}
            cover.append(dict(view=name,split=name.split('/')[0],region=reg,feature_count=len(kp),
                              candidate_keypoints=int(kpmask.sum()),supported_keypoints=len(accepted_here),
                              candidate_stride4_cells=int(mask.sum()),supported_stride4_cells=len(cells)))
    slices=[]
    for split in ('train','eval'):
        for scope in ('all_cross','closer_source'):
            for reg in ('all','near','mid','far'):
                for bucket in ('all','1.25_to_1.5','ge_1.5','t_advantage_ge_0.03'):
                    rows=[r for r in predictions if r['split']==split and r['scope']==scope and (reg=='all' or r['target_region']==reg)]
                    if bucket=='1.25_to_1.5': rows=[r for r in rows if r['depth_ratio']<1.5]
                    elif bucket=='ge_1.5': rows=[r for r in rows if r['depth_ratio']>=1.5]
                    elif bucket=='t_advantage_ge_0.03': rows=[r for r in rows if r['t_advantage']>=.03]
                    slices.append(dict(split=split,scope=scope,region=reg,bucket=bucket,**aggregate_prediction(rows)))
    primary=next(r for r in slices if (r['split'],r['scope'],r['region'],r['bucket'])==('train','closer_source','far','all'))
    far_cover=[r for r in cover if r['split']=='train' and r['region']=='far']
    far_fraction=sum(r['supported_keypoints'] for r in far_cover)/max(sum(r['candidate_keypoints'] for r in far_cover),1)
    all_closer=[r for r in predictions if r['split']=='train' and r['scope']=='closer_source']
    support_pass=(len(all_closer)>=100 and primary['n']>=50 and primary['target_views']>=5 and far_fraction>=.02)
    trans_pass=primary['n']>0 and primary['t_advantage_median']>=.03
    predictive_pass=primary['n']>0 and all(primary['cdrc_bounded_rmse']<primary[x+'_rmse'] and primary['cdrc_bounded_view_mean_rmse']<primary[x+'_view_mean_rmse'] for x in ('renderer','source_copy'))
    # Both gap buckets with >=20 observations must also improve.
    buckets=[r for r in slices if r['split']=='train' and r['scope']=='closer_source' and r['region']=='far' and r['bucket'] in ('1.25_to_1.5','ge_1.5') and r['n']>=20]
    predictive_pass=predictive_pass and all(all(r['cdrc_bounded_rmse']<r[x+'_rmse'] for x in ('renderer','source_copy')) for r in buckets)
    # No model is loaded. Verify every input byte again, including checkpoints.
    for p,h in hashes.items():
        assert sha256(Path(p))==h, 'input changed: '+p
    summary=dict(scene=scene,completed=True,smoke=smoke,train_views=len(train),eval_views=len(evaluation),
                 train_pairs=len(train)*(len(train)-1)//2,eval_train_pairs=len(evaluation)*len(train),
                 counts={split:{key:sum(r[key] for r in pair_rows if r['split']==split) for key in ('raw_knn_ab','ratio_ab','ratio_mutual','model_geometry','patch_zncc','cross_distance')} for split in ('train','eval')},
                 far_supported_keypoint_fraction=far_fraction,primary=primary,
                 admission=dict(support_pass=bool(support_pass),transmission_pass=bool(trans_pass),prediction_pass=bool(predictive_pass),scene_pass=bool(support_pass and trans_pass and predictive_pass)),
                 seconds=time.perf_counter()-started,peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 gpu_used=native_enabled,native_audit=native_audit,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),python=sys.executable,
                 training_started=False,backward_calls=0,optimizer_steps=0,clear_gt_available=False,
                 input_hashes_unchanged=True,checkpoint=audit['checkpoint'],checkpoint_sha256=audit['checkpoint_sha256'])
    write_csv(scene_out/'pair_matches.csv',pair_rows);write_csv(scene_out/'correspondences.csv',observations)
    write_csv(scene_out/'prediction_rows.csv',predictions);write_csv(scene_out/'prediction_slices.csv',slices)
    write_csv(scene_out/'coverage.csv',cover);write_json(scene_out/'prediction_slices.json',slices)
    write_json(scene_out/'input_verification.json',dict(hashes=hashes,checks=checks,unchanged=True))
    write_json(scene_out/'summary.json',summary)
    print(json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--native',action='store_true',help='Required for formal predictions; fixes stride-4 interpolation bias')
    args=p.parse_args()
    assert os.environ.get('CONDA_DEFAULT_ENV')=='water_splatting'
    assert sys.executable=='/opt/anaconda3/envs/water_splatting/bin/python'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+SCENES.index(args.scene))
    cv2.setNumThreads(2);cv2.setRNGSeed(170917)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert args.smoke or args.native, 'Formal audit requires native samples'
    run_scene(args.scene,out,args.smoke,args.native)


if __name__=='__main__':
    main()
