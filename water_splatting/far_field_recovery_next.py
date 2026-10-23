"""Explicit working definitions of LSPR and MST for 2026-09-17 experiments.

Neither expands the bounded Gaussian appearance range. LSPR is a geometric local
prior, not new supervision; MST is a visible-layer scattering approximation,
not full multiple scattering or a claim of algorithmic novelty.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from water_splatting.far_field_recovery import robust


class LocalSurfacePrior:
    def __init__(self,model,observations:Path):
        with np.load(observations) as f:
            ratio=f['information']/np.maximum(f['information_zero_medium'][:,None],1e-12)
            support=f['view_support'];weight=f['weight']
        pos=model.means.detach().float().cpu().numpy()
        assert len(pos)==len(ratio)
        scale=model.scales.detach().exp().max(-1).values.float().cpu().numpy()
        color=torch.sigmoid(.28209479177387814*model.features_dc.detach()).float().cpu().numpy()
        strong=(support>=3)&(weight>=.1)&(ratio.min(-1)>=.35)
        weak=(support>=2)&(weight>=.1)&(ratio.min(-1)<.2)
        anchor_ids=np.flatnonzero(strong);weak_ids=np.flatnonzero(weak)
        self.source_positions=pos
        self.source_scales=scale
        self.source_targets=np.zeros((len(pos),2),np.float32)
        self.source_weights=np.zeros(len(pos),np.float32)
        if len(anchor_ids)>=8 and len(weak_ids):
            tree=cKDTree(pos[anchor_ids]);distance,nnidx=tree.query(pos[weak_ids],k=8,workers=2)
            near_ids=anchor_ids[nnidx]
            local_scale=np.median(scale[near_ids],axis=1)
            # Radius and plane consistency guard against crossing disconnected surfaces.
            radius=6*np.maximum(scale[weak_ids],local_scale)
            valid=distance<=radius[:,None]
            points=pos[near_ids];center=points.mean(1);centered=points-center[:,None]
            cov=np.einsum('nki,nkj->nij',centered,centered)/8
            values,vectors=np.linalg.eigh(cov)
            normal=vectors[:,:,0]
            plane_distance=np.abs(((pos[weak_ids]-center)*normal).sum(-1))
            plane_ok=(values[:,0]/np.maximum(values.sum(-1),1e-12)<.1)&(plane_distance<=2*local_scale)
            ok=(valid.sum(-1)>=4)&plane_ok
            conf=ratio[near_ids].mean(-1)
            weights=valid*conf/np.maximum(distance,local_scale[:,None]*.25+1e-6)
            chroma=np.stack((np.log((color[:,0]+1e-6)/(color[:,1]+1e-6)),np.log((color[:,2]+1e-6)/(color[:,1]+1e-6))),-1)
            target=(weights[...,None]*chroma[near_ids]).sum(1)/np.maximum(weights.sum(1)[:,None],1e-9)
            # Preserve the local contrast distribution; no brightness pseudo-label.
            self.source_targets[weak_ids[ok]]=target[ok]
            self.source_weights[weak_ids[ok]]=np.clip(1-ratio[weak_ids[ok]].mean(-1)/.35,0,1)
        self.tree=cKDTree(pos)
        self.indices=None;self.population=-1
        self.support=dict(strong_anchors=int(strong.sum()),weak_candidates=int(weak.sum()),
                          accepted_priors=int((self.source_weights>0).sum()),test_rgb_used=False,
                          source=str(observations),prior='local frozen high-information DC log-chroma; no clear GT')

    def loss(self,model,ordinal,*unused):
        if self.population!=model.num_points:
            pos=model.means.detach().float().cpu().numpy()
            distance,indices=self.tree.query(pos,k=1,workers=2)
            weight=self.source_weights[indices].copy()
            weight[distance>2*self.source_scales[indices]]=0
            valid=np.flatnonzero(weight>0)
            self.indices=torch.as_tensor(valid,device=model.device,dtype=torch.long)
            self.targets=torch.as_tensor(self.source_targets[indices[valid]],device=model.device)
            self.weights=torch.as_tensor(weight[valid],device=model.device)
            self.population=model.num_points
        n=len(self.indices)
        if not n:return model.features_dc.sum()*0,dict(prior_points=0)
        size=min(n,32768);offset=(ordinal*size)%n
        ix=(torch.arange(size,device=model.device)+offset)%n
        dc=torch.sigmoid(.28209479177387814*model.features_dc[self.indices[ix]])
        chroma=torch.stack((torch.log((dc[:,0]+1e-6)/(dc[:,1]+1e-6)),torch.log((dc[:,2]+1e-6)/(dc[:,1]+1e-6))),-1)
        loss=F.huber_loss(chroma,self.targets[ix],delta=.05,reduction='none').mean(-1)/.05
        weight=self.weights[ix]
        return (loss*weight).sum()/weight.sum().clamp_min(1e-8),dict(prior_points=size,active_priors=n)


def blur_layer(layer,sigma):
    radius=int(math.ceil(3*sigma));x=torch.arange(-radius,radius+1,device=layer.device,dtype=layer.dtype)
    kernel=torch.exp(-.5*(x/sigma).square());kernel=kernel/kernel.sum()
    x=layer.permute(2,0,1)[None]
    x=F.conv2d(F.pad(x,(radius,radius,0,0),mode='reflect'),kernel[None,None,None,:].expand(3,1,1,-1),groups=3)
    x=F.conv2d(F.pad(x,(0,0,radius,radius),mode='reflect'),kernel[None,None,:,None].expand(3,1,-1,1),groups=3)
    return x[0].permute(1,2,0)


class MediumScatteringTransport(nn.Module):
    """Jointly train a positive depth-dependent PSF on exact visible direct layers."""
    def __init__(self,bounds,device):
        super().__init__()
        self.raw_tau=nn.Parameter(torch.full((3,),math.log(math.expm1(.01)),device=device))
        self.kernel_logits=nn.Parameter(torch.zeros(2,device=device))
        self.bounds=tuple(float(x) for x in bounds)
        self.layer_sum_max_error=0.;self.calls=0
        self.support=dict(parameters=5,sigmas=(1.,3.),max_scattered_fraction=.3,
                          initialization_tau=.01,depth_bounds=self.bounds,clear_forward_unchanged=True,
                          approximation='convolution of visible direct layers; no hidden surface transport')

    def transform(self,model,outputs,kwargs,native):
        depths=kwargs['depths'].detach().reshape(-1)
        index=torch.bucketize(depths,depths.new_tensor(self.bounds))
        rasterize=model.underwater_rasterizer.rasterize
        layers=[]
        for group in (0,1):
            result=rasterize(**dict(kwargs,colors=kwargs['colors']*(index==group)[:,None],
                                   xys_grad_abs=torch.zeros_like(kwargs['xys_grad_abs'])))
            layers.append(result.rgb_object)
        # Linearity in color gives the last layer without another full rasterization.
        layers.append(native.rgb_object-layers[0]-layers[1])
        centers=(self.bounds[0]*.5,(self.bounds[0]+self.bounds[1])*.5,self.bounds[1]*1.5)
        kernel_weights=self.kernel_logits.softmax(0)
        delta=torch.zeros_like(native.rgb_object)
        for layer,center in zip(layers,centers):
            fraction=.3*(1-torch.exp(-F.softplus(self.raw_tau)*center/self.bounds[1]))
            scattered=kernel_weights[0]*blur_layer(layer,1.)+kernel_weights[1]*blur_layer(layer,3.)
            delta=delta+fraction*(scattered-layer)
        if self.calls==0:
            self.layer_sum_max_error=float((sum(layers)-native.rgb_object).detach().abs().max())
            if self.layer_sum_max_error>2e-6:raise RuntimeError('MST layer decomposition mismatch')
        self.calls+=1
        out=dict(outputs)
        out['rgb_object']=outputs['rgb_object']+delta
        out['pred_image']=outputs['pred_image']+delta
        return out

    def statistics(self):
        return dict(tau=F.softplus(self.raw_tau).detach().cpu().tolist(),
                    kernel_weights=self.kernel_logits.softmax(0).detach().cpu().tolist(),
                    layer_sum_max_error=self.layer_sum_max_error)
