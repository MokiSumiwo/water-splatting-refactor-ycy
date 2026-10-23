"""Conservative training-only center-gradient preconditioner (experiment A)."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

class GeometryGradient:
    def __init__(self, worker, branch, output, enabled, smoke=False):
        self.worker=worker;self.enabled=enabled;self.smoke=smoke
        dm=branch.pipeline.datamanager
        meta=dm.train_dataparser_outputs.metadata
        from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary, read_images_binary
        data=Path(dm.config.dataparser.data)
        pts=read_points3D_binary(data/'sparse/0/points3D.bin')
        images=read_images_binary(data/'sparse/0/images.bin')
        names={Path(x).stem for x in (data/'train_list.txt').read_text().splitlines() if x.strip()}
        region_file=Path(worker.REPO_ROOT)/'research_runs/geometry_gradient_A_20261021/regions'/output.parent.name/'regions.json'
        # output.parent is the scene for formal and smoke jobs.
        held_out=set()
        if region_file.exists():
            held_out={v['camera'] for v in json.loads(region_file.read_text())['assets']}
        names-=held_out
        train_ids={i for i,v in images.items() if Path(v.name).stem in names}
        good=np.array([sum(int(i) in train_ids for i in p.image_ids)>=3 and p.error<=1.0 for p in pts.values()])
        xyz=meta['points3D_xyz'].detach().cpu().numpy()
        if len(xyz)!=len(good):raise RuntimeError('COLMAP metadata order mismatch')
        anchors=xyz[good];self.tree=cKDTree(anchors)
        if len(anchors)<4:raise RuntimeError('insufficient train-only triangulated anchors')
        # Local spacing is an uncertainty scale, not a measured surface normal.
        spacing=self.tree.query(anchors,k=4)[0][:,-1]
        self.spacing=np.maximum(spacing,1e-6)
        np.savez_compressed(output/'structure_support.npz',anchors=anchors,spacing=self.spacing)
        (output/'structure_support.json').write_text(json.dumps(dict(anchor_count=len(anchors),source='COLMAP tracks with >=3 real training cameras and reprojection error <=1px',support='center within local anchor spacing AND three-sigma maximum scale <= local spacing; no inferred water labels',smoke=smoke,excluded_validation_cameras=sorted(held_out)),indent=2)+'\n')
        self.log=(output/'geometry_gradient.jsonl').open('a',buffering=1)

    def eta(self,step):
        if not self.enabled:return 0.
        if self.smoke:return .25 if 2<=step<=15 else 0.
        if step<=500 or step>=10000:return 0.
        if step<=2000:return .25*(step-500)/1500
        if step<=8000:return .25
        return .25*(10000-step)/2000

    def apply(self,model,outputs,loss,step):
        eta=self.eta(step)
        if eta==0:return
        # Extra VJPs must not change the native densification accumulators.
        abs_before=model.xys_grad_abs.detach().clone()
        xy_before=None if model.xys.grad is None else model.xys.grad.detach().clone()
        r=torch.autograd.grad(loss,outputs['pred_image'],retain_graph=True)[0].detach()
        gd=torch.autograd.grad(outputs['rgb_object'],model.means,grad_outputs=r,retain_graph=True)[0]
        gm=torch.autograd.grad(outputs['rgb_medium'],model.means,grad_outputs=r,retain_graph=True)[0]
        with torch.no_grad():
            model.xys_grad_abs.copy_(abs_before)
            model.xys.grad=xy_before
            distance,nearest=self.tree.query(model.means.detach().cpu().numpy(),workers=1)
            local=self.spacing[nearest]
            radius=3*model.scales.detach().exp().amax(-1).cpu().numpy()
            supported=torch.as_tensor((distance<=local)&(radius<=local),device=model.means.device)
            q=gm.norm(dim=-1)/(gd.norm(dim=-1)+gm.norm(dim=-1)+1e-12)
            delta=eta*supported[:,None]*q[:,None]*gm
            if model.means.grad is None:raise RuntimeError('missing native center gradient')
            native=model.means.grad.clone()
            model.means.grad.sub_(delta)
            if not torch.isfinite(model.means.grad).all():raise RuntimeError('nonfinite preconditioned gradient')
            if step%100==0 or self.smoke:
                self.log.write(json.dumps(dict(step=step,eta=eta,support_fraction=float(supported.float().mean()),supported=int(supported.sum()),population=len(supported),direct_norm=float(gd.norm()),medium_norm=float(gm.norm()),native_norm=float(native.norm()),delta_norm=float(delta.norm()),q_mean=float(q.mean())))+'\n')
