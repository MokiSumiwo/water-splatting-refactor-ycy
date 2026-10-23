"""Experiment-local projection backward correction; native forward retained."""
from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parents[2]
def install():
 import water_splatting.cuda._backend as backend
 source=ROOT/'research_runs/geometry_gradient_A_20261021/projection_fix_source'
 ext=load(name='water_splatting_projection_clip_A_20261021',sources=sorted(str(p) for p in source.iterdir() if p.suffix in ('.cu','.cpp')),extra_include_paths=[str(ROOT/'water_splatting/cuda/csrc'),str(ROOT/'water_splatting/cuda/csrc/third_party/glm')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)
 original=backend._C
 class ProjectionBackend:
  gradient_abi_version=1
  def __getattr__(self,name):
   if name=='project_gaussians_backward':return ext.project_gaussians_backward
   return getattr(original,name)
 backend._C=ProjectionBackend()
 return ext
