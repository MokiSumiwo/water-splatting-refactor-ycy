"""Experimental far-field objectives for bounded FSB continuation.

These helpers are opt-in from the experiment worker. They do not change the
production renderer or claim a physically identified clear solution.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F


def sample(image, uv):
    if image.ndim == 2:
        image = image[..., None]
    h, w = image.shape[:2]
    grid = torch.stack((uv[:,0] / (w-1)*2-1, uv[:,1] / (h-1)*2-1), -1)[None,None]
    return F.grid_sample(image.permute(2,0,1)[None], grid, align_corners=True)[0,:,0].T


def capture(model, camera):
    saved = {}
    rasterize = model.underwater_rasterizer.rasterize
    def wrapped(**kwargs):
        result = rasterize(**kwargs)
        saved.update(kwargs=kwargs, render=result)
        return result
    # Avoid unittest.mock call histories retaining entire training graphs.
    model.underwater_rasterizer.rasterize = wrapped
    try:
        outputs = model.get_outputs(camera)
    finally:
        model.underwater_rasterizer.rasterize = rasterize
    return outputs, saved['kwargs'], saved['render']


def transport_mass(model, kwargs):
    # Same native support, sorting, attenuation and termination as the main call.
    # Only the medium receives gradient through this unit-color call.
    frozen = {k: (v.detach() if isinstance(v,torch.Tensor) and not k.startswith('medium_') else v)
              for k,v in kwargs.items()}
    frozen['colors'] = torch.ones_like(kwargs['colors'])
    frozen['xys_grad_abs'] = torch.zeros_like(kwargs['xys_grad_abs'])
    return model.underwater_rasterizer.rasterize(**frozen).rgb_object


def robust(residual, delta=.05):
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction='mean') / delta


class CrossDistanceRadiance:
    """Train joint effective surface color/transport using observed correspondences.

    J_eff = sum(w*t*c)/sum(w*t), hence bounded under FSB. Reciprocal
    cross-rendering penalizes S_target*J_source+M_target against original GT.
    No inverse (GT-M)/t pseudo-color is created and no clear-buffer gradient is
    used (the legacy CUDA backward does not implement that gradient).
    """
    def __init__(self, path: Path, records, device):
        self.records = {str(r[1]):r for r in records}
        grouped = defaultdict(list)
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['split']!='train' or r['source_closer']!='True':
                    continue
                assert r['target'].startswith('train/') and r['source'].startswith('train/')
                grouped[(r['target'].split('/')[1],r['source'].split('/')[1])].append(r)
        self.pairs = []
        for (target,source), rows in sorted(grouped.items()):
            self.pairs.append((target,source,
                torch.tensor([[float(r['u']),float(r['v'])] for r in rows],device=device),
                torch.tensor([[float(r['source_u']),float(r['source_v'])] for r in rows],device=device)))
        self.support = dict(pairs=len(self.pairs),points=sum(len(p[2]) for p in self.pairs),
                            correspondence_path=str(path),test_rgb_used=False)

    def loss(self, model, ordinal, get_gt, batch_to_device):
        if not self.pairs:
            return model.features_dc.sum()*0, dict(points=0)
        target,source,ut,us = self.pairs[ordinal % len(self.pairs)]
        # All correspondences are used; count-normalized loss, no GT-based ranking.
        result=[]
        for name,uv in ((target,ut),(source,us)):
            _,_,camera,batch = self.records[name]
            o,kw,r = capture(model,camera.to(model.device))
            mass = transport_mass(model,kw)
            S = sample(mass,uv)
            J = sample(r.rgb_object,uv) / S.clamp_min(.05)
            M = sample(o['rgb_medium'],uv)
            gt = sample(get_gt(model,batch_to_device(batch,model.device),o['background']),uv).detach()
            valid = ((sample(r.accumulation,uv)[:,0].detach()>=.9) &
                     (sample(r.depth_std_relative,uv)[:,0].detach()<=.1))
            result.append((S,J,M,gt,valid))
        st,jt,mt,it,vt=result[0];ss,js,ms,is_,vs=result[1]
        valid=vt&vs
        if not bool(valid.any()):
            return (jt.sum()+js.sum())*0,dict(points=0,target=target,source=source)
        loss=.5*(robust((st*js+mt-it)[valid])+robust((ss*jt+ms-is_)[valid]))
        return loss,dict(points=int(valid.sum()),target=target,source=source,
                         effective_J_min=float(torch.minimum(jt.min(),js.min()).detach()),
                         effective_J_max=float(torch.maximum(jt.max(),js.max()).detach()))
