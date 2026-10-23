"""Registered eta=.10 ablation; identical A worker and schedule scaled by .4."""
from scripts.experiments import run_geometry_gradient_A_worker as worker
from water_splatting.geometry_gradient import GeometryGradient
class GeometryGradient010(GeometryGradient):
 def eta(self,step):return .4*super().eta(step)
worker.GeometryGradient=GeometryGradient010
if __name__=='__main__':worker.main()
