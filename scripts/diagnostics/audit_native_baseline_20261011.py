"""Read-only native backend and legacy M1/OCMC/CICA checkpoint audit; no optimizer step."""
import sys,os,json,hashlib,copy,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from water_splatting.rendering.underwater_rasterizer import UnderwaterRasterizer
import water_splatting.cuda._backend as be
OUT=ROOT/'research_notes/native_baseline_audit_20261011'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def synthetic():
 r=UnderwaterRasterizer();H=W=16;device='cuda'
 kw=dict(xys=torch.tensor([[7.2,7.4],[8.1,7.7]],device=device),xys_grad_abs=torch.zeros(2,2,device=device),depths=torch.tensor([1.2,2.5],device=device),radii=torch.tensor([6,6],device=device,dtype=torch.int32),conics=torch.tensor([[.2,.01,.3],[.25,-.02,.22]],device=device),num_tiles_hit=torch.ones(2,device=device,dtype=torch.int32),colors=torch.tensor([[.4,.5,.6],[.7,.3,.2]],device=device),opacities=torch.tensor([[.6],[.4]],device=device),medium_rgb=torch.full((H,W,3),.3,device=device),medium_bs=torch.full((H,W,3),.2,device=device),medium_attn=torch.full((H,W,3),.15,device=device),height=H,width=W,background=None,step=14999)
 q=torch.zeros(H,W,3,device=device);q[7,7]=torch.tensor([.2,-.4,.7],device=device)
 names=['xys','depths','conics','colors','opacities','medium_rgb','medium_bs','medium_attn']
 leaves={k:kw[k].clone().requires_grad_() for k in names};o=r.rasterize(**dict(kw,**leaves));g=torch.autograd.grad((o.rgb*q).sum(),tuple(leaves.values()),allow_unused=True);rows=[]
 for name,grad in zip(names,g):
  disconnected=grad is None
  if disconnected:grad=torch.zeros_like(leaves[name])
  direction=torch.zeros_like(leaves[name]);
  if name.startswith('medium_'):direction[7,7,0]=1
  else:direction.reshape(-1)[0]=1
  ad=float((grad*direction).sum());fds=[]
  for h in [.002,.001,.0005]:
   with torch.no_grad():
    a=r.rasterize(**dict(kw,**{name:kw[name]+h*direction})).rgb;b=r.rasterize(**dict(kw,**{name:kw[name]-h*direction})).rgb
    fd=float(((a-b)*q).sum()/(2*h))
   fds.append({'h':h,'fd':fd,'absolute_error':abs(fd-ad),'passed':abs(fd-ad)<=2e-5+.03*abs(fd)})
  outside=None
  if name.startswith('medium_'):
   v=grad.clone();v[7,7]=0;outside=float(v.abs().max())
  rows.append({'parameter':name,'autograd':ad,'disconnected':disconnected,'finite_differences':fds,'outside_pixel_max':outside,'passed':sum(v['passed'] for v in fds)>=2 and (outside is None or outside<1e-9)})
 caps=[]
 for opacity in [.5,.99,.995,.999]:
  one=dict(kw);one.update(xys=torch.zeros(1,2,device=device),xys_grad_abs=torch.zeros(1,2,device=device),depths=torch.ones(1,device=device),radii=torch.tensor([3],device=device,dtype=torch.int32),conics=torch.tensor([[1.,0.,1.]],device=device),num_tiles_hit=torch.ones(1,device=device,dtype=torch.int32),colors=torch.full((1,3),.4,device=device,requires_grad=True),opacities=torch.tensor([[opacity]],device=device),medium_rgb=torch.zeros(1,1,3,device=device),medium_bs=torch.zeros(1,1,3,device=device),medium_attn=torch.zeros(1,1,3,device=device),height=1,width=1)
  oo=r.rasterize(**one);gg=torch.autograd.grad(oo.rgb_object.sum(),one['colors'])[0];forward=float(oo.rgb_object[0,0,0]/.4);back=float(gg[0,0]);caps.append({'opacity':opacity,'forward_derivative':forward,'backward_derivative':back,'passed':abs(forward-back)<2e-6})
 return {'directional_fd':rows,'alpha_cap_color':caps,'passed':all(v['passed'] for v in rows+caps),'scope':'native synthetic rasterizer; sparse physical-medium gradients must have zero cross-pixel leakage; no geometry visibility/sorting boundaries crossed'}
def real(scene):
 from scripts.experiments import run_direct_mdrr_cica_scene as W
 pth=ROOT/'outputs/legacy_ocmc_cica_no_auxiliary_20260905'/scene;manifest=json.loads((pth/'experiment_manifest.json').read_text());W.ACTIVE_SOURCE_CONFIG=pth/'source_config.yml';W.ACTIVE_SOURCE_ROOT=Path(manifest['source_root']);W.ACTIVE_PARAMETERIZATION='legacy';W.AUXILIARY_APPEARANCE_ENABLED=False;W.RENDERER_BACKWARD_MODE='legacy';W.ACTIVE_MDRR_OVERRIDE=False;W.ACTIVE_CICA_OVERRIDE=True
 W.FORMAL.ACTIVE_PARAMETERIZATION='legacy';W.AUX.ACTIVE_PARAMETERIZATION='legacy'
 b=W._new_branch(scene)
 try:
  cp=pth/'checkpoints/step-000014999.ckpt';p=W._load_snapshot(b,scene,cp);m=b.pipeline.model
  before=W._model_state_hash(m);records=W._train_records(b);outputs=[]
  for idx in [0,len(records)//2]:
   rec=records[idx]
   with torch.no_grad():o=m.get_outputs_for_camera(copy.deepcopy(rec[2]).to(m.device))
   outputs.append((rec,{k:o[k].detach().cpu() for k in ['pred_image','J_gaussian_raw','accumulation','medium_rgb','medium_bs','medium_attn']}))
   del o
  restore=W._restore_training_checkpoint(b,p,strict_runtime=False)
  W.FORMAL._install_condition(m,'C0',p['ocmc_bundle'],None);m.eval();comp=[]
  for rec,saved in outputs:
   with torch.no_grad():o=m.get_outputs_for_camera(copy.deepcopy(rec[2]).to(m.device))
   dif={k:float((o[k].cpu()-v).abs().max()) for k,v in saved.items()};comp.append({'view':rec[1],'max_abs':dif,'passed':all(x<=2e-6 for x in dif.values())});del o
  strict_error=None
  try:W._restore_training_checkpoint(b,p,strict_runtime=True)
  except RuntimeError as e:strict_error=str(e)
  return {'scene':scene,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'num_gaussians':m.num_points,'config':{k:getattr(m.config,k,None) for k in ['medium_context_mode','intrinsic_color_parameterization','camera_medium_observability_enabled','camera_medium_ray_adaptive_observability_enabled']},'checkpoint_metadata':p['metadata'],'available_state':list(p),'restore':restore,'forward_reload':comp,'model_unchanged':before==W._model_state_hash(m),'strict_continuation_rejected':strict_error,'exact_historical_resume_available':strict_error is None,'saved_cica_controller':any(k in p for k in ['cica_controller','controller']),'passed_render_and_optimizer_restore':restore['all_pass'] and all(x['passed'] for x in comp)}
 finally:W._release(b)
def main():
 identity={'backend':str(be._C.__file__),'backend_sha256':sha(be._C.__file__),'cuda_backward_source_sha256':sha(ROOT/'water_splatting/cuda/csrc/backward.cu'),'gpu':torch.cuda.get_device_name(),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'optimizer_steps':0,'cuda_backend_replaced':False}
 write('protocol.json',{'identity':identity,'forward_atol':2e-6,'fd_atol':2e-5,'fd_rtol':.03,'no_cross_pixel_leak_atol':1e-9,'scenes':['Panama','JapaneseGradens-RedSea','Curasao','IUI3-RedSea'],'no_training':True})
 result={'identity':identity,'synthetic':synthetic(),'scenes':[]};write('results.json',result);print('synthetic',result['synthetic']['passed'],flush=True)
 for scene in ['Panama','JapaneseGradens-RedSea','Curasao','IUI3-RedSea']:
  try:x=real(scene)
  except Exception:x={'scene':scene,'failure':traceback.format_exc()}
  result['scenes'].append(x);write('results.json',result);print(scene,x.get('passed_render_and_optimizer_restore'),x.get('failure',''),flush=True)
 result['backend_unchanged']=sha(be._C.__file__)==identity['backend_sha256'];result['admitted_for_new_training']=result['synthetic']['passed'] and all(x.get('passed_render_and_optimizer_restore',False) and x.get('exact_historical_resume_available',False) for x in result['scenes']);write('results.json',result)
if __name__=='__main__':main()
