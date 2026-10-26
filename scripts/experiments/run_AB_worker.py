"""A(0.1)+B: independent means interventions with one native reference gradient."""
import argparse
import functools
import json
from pathlib import Path

import torch

from scripts.experiments import run_geometry_gradient_A_worker as base
from scripts.experiments.run_AC_pair_worker import IsolatedGeometry010
from water_splatting.surface_support import SurfaceSupport

W = base.W


class ABHooks:
    def __init__(self, branch, output, support, use_a=True, use_b=True, smoke=False, telemetry=True):
        self.a = IsolatedGeometry010(W, branch, output, use_a, smoke)
        self.b = SurfaceSupport(branch, output, support, enabled=use_b, smoke=smoke)
        self.use_a, self.use_b, self.step = use_a, use_b, 0
        self.optimizer = branch.optimizers.optimizers['means']
        self.original_step = self.optimizer.step
        self.log = (output/'AB_optimizer_updates.jsonl').open('a', buffering=1)
        if telemetry:
            @functools.wraps(self.original_step)
            def measured_step(*args, **kwargs):
                model = branch.pipeline.model
                record = smoke or self.step % 500 == 0
                before = model.means.detach().clone() if record else None
                result = self.original_step(*args, **kwargs)
                # Fallback for torch versions where the registered post hook is
                # bypassed by the scheduler wrapper. pending=None prevents duplicates.
                self.b.after_step(self.optimizer, args, kwargs)
                if record:
                    with torch.no_grad():
                        displacement = (model.means-before).norm(dim=-1)
                        state = self.optimizer.state[model.means]
                        scale = model.scales.detach().exp().amax(-1).clamp_min(1e-12)
                        self.log.write(json.dumps(dict(step=self.step,
                            actual_update_l2=float(displacement.norm()),
                            normalized_displacement_p99=float(torch.quantile(displacement/scale, .99)),
                            normalized_displacement_max=float((displacement/scale).max()),
                            adam_first_moment_l2=float(state['exp_avg'].norm()),
                            adam_second_moment_l2=float(state['exp_avg_sq'].norm())), allow_nan=False)+'\n')
                return result
            # functools.wraps preserves this torch version's _with_counter flag;
            # original_step still performs the actual scheduler counter increment.
            self.optimizer.step = measured_step

    def apply(self, model, outputs, loss, step):
        self.step = step
        if self.a.eta(step) == 0 and (step % 4 or not (500 < step < 10000)) and not self.b.smoke:
            return
        native = model.means.grad.detach().clone()
        self.a.apply(model, outputs, loss, step)
        if self.use_b:
            # B calibration and per-Gaussian caps use the pre-A native gradient;
            # its additive term is then applied to the A gradient.
            self.b.apply(model, outputs, loss, step, native_reference=native)
        if not torch.isfinite(model.means.grad).all():
            raise RuntimeError('nonfinite A+B center gradient')

    def close(self):
        self.optimizer.step = self.original_step
        self.b.handle.remove()
        for stream in (self.a.log, self.b.log, self.b.moves, self.log):
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, choices=tuple(W.SCENE_GPUS))
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--arm', choices=['B0', 'A010', 'B', 'AB'], default='AB')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    W.EXPERIMENT = 'A010_B_' + a.arm
    W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    W.RENDERER_BACKWARD_MODE = 'corrected_v1'
    W.RESPONSIBILITY_ENABLED = False
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.ACTIVE_MDRR_OVERRIDE = False
    W.ACTIVE_CICA_OVERRIDE = True
    W.START_STEP = 0
    W.CICA_START_STEP = 3 if a.smoke else 10000
    W.CICA_REFRESH_STEPS = (3, 9) if a.smoke else (10000, 12000, 14000)
    W.ACTIVE_RENDER_ROOT = a.output/'renders'
    original_route = W._route_renderer_backend
    def route(mode):
        result = original_route(mode)
        from scripts.experiments.geometry_projection_backend import install
        install()
        result['projection_backward'] = 'clip_chain_rule_A_20261021'
        return result
    W._route_renderer_backend = route
    W._active_snapshots = lambda final: tuple(x for x in (10000, final) if x <= final)
    W.MIC._metric_images = lambda model, pred, gt: W._taba_rgb_metrics(pred, gt)
    controllers = []
    def hook(branch, outputs, loss, step, output):
        if not controllers:
            controllers.append(ABHooks(branch, output, a.support,
                a.arm in ('A010', 'AB'), a.arm in ('B', 'AB'), a.smoke))
            (output/'AB_config.json').write_text(json.dumps(dict(scene=a.scene, arm=a.arm,
                seed=42, smoke=a.smoke, updates=20 if a.smoke else 15000,
                A_eta_max=.1, B_native_gradient_cap=.05, B='center-to-supported-plane Huber',
                formula='g_native + delta_A(g_native) + delta_B(g_native)',
                auxiliary_densification_isolation=True,
                support=str(a.support.resolve())), indent=2)+'\n')
        controllers[0].apply(branch.pipeline.model, outputs, loss, step)
    W.GEOMETRY_GRADIENT_HOOK = hook
    try:
        result = W.run(a.scene, 'B', a.gpu, a.output.resolve(), max_step=20 if a.smoke else 15000,
            source_config=base.ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene/'source_config.yml',
            training_seed=42, raw_data_only=True)
        print(json.dumps(dict(scene=a.scene, arm=a.arm, final_step=result['final_step'])), flush=True)
    finally:
        for controller in controllers:
            controller.close()


if __name__ == '__main__':
    main()
