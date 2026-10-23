"""Zero-update loss/VJP and real endpoint support-coverage audit."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from water_splatting.surface_support import huber_plane_loss, schedule
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_runs/surface_support_B_20261023'

def main():
    x=torch.tensor([[.2,.3,.02],[-.3,.4,.2]],dtype=torch.float64,requires_grad=True)
    centers=torch.zeros_like(x);normals=torch.tensor([[0.,0.,1.],[0.,0.,1.]],dtype=torch.float64);sigma=torch.tensor([.1,.1],dtype=torch.float64)
    passed=torch.autograd.gradcheck(lambda z:huber_plane_loss(z,centers,normals,sigma),(x,))
    g=torch.autograd.grad(huber_plane_loss(x,centers,normals,sigma),x)[0]
    assert (g[:,:2]==0).all() and (g[:,2]>0).all() and passed
    assert [schedule(t) for t in [0,500,2000,8000,10000,15000]]==[0,0,1,1,0,0]
    rows=[]
    for scene in ['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']:
        d=np.load(OUT/'support'/scene/'support.npz');raw=d['raw_xyz']
        base=ROOT/'research_runs/geometry_gradient_A_20261021/formal'/scene/'B0'
        init=torch.load(base/'initialization'/scene/'step-000000000.pt',map_location='cpu')['model']
        xyz=init['gauss_params.means'].numpy();transform=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],xyz,rcond=None)[0]
        scale=np.linalg.svd(transform[:3],compute_uv=False).mean()
        centers=np.c_[d['centers'],np.ones(len(d['centers']))]@transform;normals=d['normals']@transform[:3]/scale
        tree=cKDTree(centers)
        for step in [0,10000,15000]:
            state=init if step==0 else torch.load(base/'checkpoints'/f'step-{step:09d}.ckpt',map_location='cpu')['model']
            xyz=state['gauss_params.means'].numpy();r=3*state['gauss_params.scales'].exp().amax(-1).numpy()
            _,idx=tree.query(xyz);diff=xyz-centers[idx];signed=(diff*normals[idx]).sum(-1)
            tangent=np.sqrt(np.maximum((diff*diff).sum(-1)-signed*signed,0))
            ok=(tangent+r<=d['radii'][idx]*scale)&(np.abs(signed)<=3*d['sigma'][idx]*scale)
            rows.append(dict(scene=scene,step=step,population=len(xyz),eligible_footprints=int(ok.sum()),fraction=float(ok.mean()),
                             scope='upper bound before current-camera visibility and common-track gating'))
    report=dict(optimizer_updates=0,gradient_check=bool(passed),schedule_check=True,coverage=rows)
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
