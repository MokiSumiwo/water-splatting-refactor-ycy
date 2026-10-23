"""Training-only B: conservative center-to-plane prior from frozen COLMAP tracks."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree


def schedule(step):
    if step<=500 or step>=10000:return 0.
    if step<=2000:return (step-500)/1500.
    if step<=8000:return 1.
    return (10000-step)/2000.


def huber_plane_loss(means, centers, normals, sigma):
    residual=((means-centers)*normals).sum(-1)/sigma
    absolute=residual.abs()
    return torch.where(absolute<=1,.5*residual.square(),absolute-.5).mean()


class SurfaceSupport:
    def __init__(self, branch, output, support, enabled=True, smoke=False):
        self.enabled=enabled;self.smoke=smoke;self.output=Path(output)
        self.model=branch.pipeline.model;self.branch=branch;self.lam=None;self.pending=None
        self.log=(self.output/'surface_support.jsonl').open('a',buffering=1)
        self.moves=(self.output/'surface_displacements.jsonl').open('a',buffering=1)
        data=np.load(Path(support)/'support.npz')
        raw=data['raw_xyz'];mapped=branch.pipeline.datamanager.train_dataparser_outputs.metadata['points3D_xyz'].cpu().numpy()
        if len(raw)!=len(mapped):raise ValueError('COLMAP point ordering mismatch')
        transform=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],mapped,rcond=None)[0]
        err=np.max(np.abs(np.c_[raw,np.ones(len(raw))]@transform-mapped))
        sv=np.linalg.svd(transform[:3],compute_uv=False);scale=sv.mean()
        if err>1e-4 or sv.max()/sv.min()>1.0001:raise ValueError('Expected a verified similarity transform')
        self.centers=np.c_[data['centers'],np.ones(len(data['centers']))]@transform
        self.normals=data['normals']@transform[:3]/scale
        self.radii=data['radii']*scale;self.sigma=data['sigma']*scale
        self.views=data['views'];self.names={str(n):i for i,n in enumerate(data['camera_names'])}
        self.tree=cKDTree(self.centers) if len(self.centers) else None
        self.sequence={int(r['absolute_step']):r['camera_name'] for r in csv.DictReader((self.output/'camera_sequence.csv').open())}
        self.device_centers=torch.as_tensor(self.centers,device=self.model.device,dtype=torch.float32)
        self.device_normals=torch.as_tensor(self.normals,device=self.model.device,dtype=torch.float32)
        self.device_sigma=torch.as_tensor(self.sigma,device=self.model.device,dtype=torch.float32)
        report=dict(enabled=enabled,patches=len(self.centers),transform_max_error=float(err),scale=float(scale),
                    rate=4,gradient_cap=.05,scope='means only; current visible camera must observe every patch vertex',
                    support_sha256=__import__('hashlib').sha256((Path(support)/'support.npz').read_bytes()).hexdigest())
        (self.output/'surface_support_config.json').write_text(json.dumps(report,indent=2)+'\n')
        self.handle=branch.optimizers.optimizers['means'].register_step_post_hook(self.after_step)
        torch.cuda.reset_peak_memory_stats(self.model.device)

    def select(self, model, outputs, camera):
        n=len(model.means)
        if self.tree is None or camera not in self.names:return np.zeros(n,dtype=bool),np.zeros(n,dtype=np.int64)
        xyz=model.means.detach().cpu().numpy()
        _,idx=self.tree.query(xyz,workers=1)
        diff=xyz-self.centers[idx];signed=(diff*self.normals[idx]).sum(-1)
        tangent=np.sqrt(np.maximum((diff*diff).sum(-1)-signed*signed,0))
        radius=3*model.scales.detach().exp().amax(-1).cpu().numpy()
        visible=outputs['gaussian_visible_mask'].cpu().numpy().reshape(-1)
        # Entire tangential 3-sigma footprint lies inside a conservative hull disk.
        valid=(tangent+radius<=self.radii[idx])&(np.abs(signed)<=3*self.sigma[idx])
        valid &= self.views[idx,self.names[camera]] & visible
        return valid,idx

    def apply(self, model, outputs, loss, step):
        weight=(1. if 4<=step<=16 else 0.) if self.smoke else schedule(step)
        if step%4 or weight==0:return
        start=time.perf_counter();valid,index=self.select(model,outputs,self.sequence[step])
        ids=torch.as_tensor(np.flatnonzero(valid),device=model.device,dtype=torch.long)
        row=dict(step=step,schedule=weight,supported=len(ids),population=len(valid),support_fraction=float(valid.mean()),enabled=self.enabled)
        if not len(ids):
            row['skip']='no eligible visible footprint';self.log.write(json.dumps(row)+'\n');return
        patches=torch.as_tensor(index[valid],device=model.device,dtype=torch.long)
        auxiliary=huber_plane_loss(model.means[ids],self.device_centers[patches],self.device_normals[patches],self.device_sigma[patches])
        grad=torch.autograd.grad(auxiliary,model.means)[0]
        with torch.no_grad():
            native=model.means.grad
            if native is None:raise RuntimeError('Native means gradient absent')
            if self.lam is None:self.lam=float(.05*native[ids].norm()/grad[ids].norm().clamp_min(1e-12))
            delta=grad*(self.lam*weight)
            # Local caps prevent a small supported subset receiving a large global budget.
            ratios=(.05*weight*native.norm(dim=-1)/delta.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
            delta*=ratios[:,None]
            ratio=float(delta.norm()/native.norm().clamp_min(1e-20))
            if not torch.isfinite(delta).all() or ratio>.050001:raise RuntimeError('B gradient cap violated')
            row.update(auxiliary=float(auxiliary),resolved_lambda=self.lam,gradient_ratio=ratio,
                       delta_norm=float(delta.norm()),native_norm=float(native.norm()))
            if self.enabled:model.means.grad.add_(delta)
            self.pending=(step,ids,model.means[ids].detach().clone(),model.scales[ids].detach().exp().amax(-1))
        torch.cuda.synchronize();row['hook_seconds']=time.perf_counter()-start
        row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(model.device)
        self.log.write(json.dumps(row)+'\n')

    def after_step(self, optimizer, args, kwargs):
        if self.pending is None:return
        step,ids,before,scales=self.pending
        with torch.no_grad():
            displacement=(self.model.means[ids]-before).norm(dim=-1)
            normalized=displacement/scales.clamp_min(1e-12)
            state=optimizer.state[self.model.means]
            row=dict(step=step,mean_displacement=float(displacement.mean()),max_displacement=float(displacement.max()),
                     normalized_p95=float(torch.quantile(normalized,.95)),normalized_max=float(normalized.max()),
                     adam_first_moment_norm=float(state['exp_avg'][ids].norm()),adam_second_moment_norm=float(state['exp_avg_sq'][ids].norm()))
            self.moves.write(json.dumps(row)+'\n')
        self.pending=None
