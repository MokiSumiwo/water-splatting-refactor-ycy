"""A010 + C2 on the paired native baseline, with independent off switches.

No change to the original single-module implementations or schedules. Formal
runs require the campaign's recorded decision; smoke uses disposable states.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from scripts.experiments import run_direct_mdrr_cica_scene as W
from water_splatting.geometry_gradient import GeometryGradient
from water_splatting.open_water_C import OpenWaterC
from water_splatting.auxiliary_gradient_isolation import isolate_densification

ROOT = Path(__file__).resolve().parents[2]


class Geometry010(GeometryGradient):
    def eta(self, step):
        return .4 * super().eta(step)


class IsolatedGeometry010(Geometry010):
    def apply(self, model, outputs, loss, step):
        if self.eta(step) == 0:
            return
        with isolate_densification(model):
            return super().apply(model, outputs, loss, step)


class IsolatedOpenWaterC(OpenWaterC):
    def apply(self, branch, outputs, batch, base_loss, step, *args):
        if self._schedule(step) <= 0 or step % 4:
            return super().apply(branch, outputs, batch, base_loss, step, *args)
        with isolate_densification(branch.pipeline.model):
            return super().apply(branch, outputs, batch, base_loss, step, *args)


class PairHooks:
    def __init__(self, branch, output, scene, masks, use_a, use_c, smoke):
        self.branch, self.output = branch, output
        self.a = IsolatedGeometry010(W, branch, output, True, smoke) if use_a else None
        self.c = IsolatedOpenWaterC(W, branch, output, masks, 'C2', smoke, scene=scene) if use_c else None
        self.smoke = smoke
        self.step = 0
        self.log = (output / 'AC_optimizer_updates.jsonl').open('a', buffering=1)
        self.original_step = {}
        self.hook_seconds = 0.0
        # Wrap the actual bound step method: the existing optimizer's built-in
        # post-hook was not invoked on this stack in the B experiment.
        for name in ['means', 'medium_mlp', 'direction_encoding', 'opacities']:
            opt = branch.optimizers.optimizers[name]
            original = opt.step
            self.original_step[name] = original
            def wrapped(*args, _name=name, _opt=opt, _step=original, **kwargs):
                record = self.smoke or self.step % 500 == 0 or self.step in (504, 9996, 10000, 15000)
                if not record:
                    return _step(*args, **kwargs)
                parameters = [p for pg in _opt.param_groups for p in pg['params']]
                before = [p.detach().clone() for p in parameters]
                result = _step(*args, **kwargs)
                with torch.no_grad():
                    delta_l2 = sum(float((p-v).float().square().sum()) for p, v in zip(parameters, before)) ** .5
                    moments = {}
                    for key in ['exp_avg', 'exp_avg_sq']:
                        moments[key + '_l2'] = sum(float(_opt.state[p][key].float().square().sum())
                            for p in parameters if key in _opt.state[p]) ** .5
                    row = dict(step=self.step, group=_name, actual_update_l2=delta_l2, **moments)
                    if _name == 'means':
                        displacement = (parameters[0]-before[0]).norm(dim=-1)
                        scale = self.branch.pipeline.model.scales.detach().exp().amax(-1).clamp_min(1e-12)
                        normalized = displacement / scale
                        row.update(normalized_displacement_p50=float(normalized.median()),
                                   normalized_displacement_p99=float(torch.quantile(normalized, .99)),
                                   normalized_displacement_max=float(normalized.max()))
                    self.log.write(json.dumps(row, allow_nan=False)+'\n')
                return result
            opt.step = wrapped

    def geometry(self, branch, outputs, loss, step, output):
        self.step = step
        started = time.perf_counter()
        if self.a is not None:
            self.a.apply(branch.pipeline.model, outputs, loss, step)
        self.hook_seconds = time.perf_counter() - started

    def extra(self, *args):
        info = self.c.apply(*args) if self.c is not None else {}
        return dict(info, AC_A_enabled=self.a is not None, AC_C2_enabled=self.c is not None,
                    AC_geometry_seconds=self.hook_seconds)

    def close(self):
        for name, step in self.original_step.items():
            self.branch.optimizers.optimizers[name].step = step
        self.log.close()
        if self.a is not None:
            self.a.log.close()
        if self.c is not None:
            self.c.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, choices=tuple(W.SCENE_GPUS))
    p.add_argument('--arm', choices=['B0', 'A010', 'C2', 'A010_C2'], default='A010_C2')
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--decision', type=Path)
    a = p.parse_args()
    if not a.smoke:
        if a.decision is None:
            p.error('Formal training requires a recorded scientific decision')
        decision = json.loads(a.decision.read_text())
        if (decision.get('launch_formal') is not True
                or a.arm not in decision.get('allowed_arms', [])
                or a.scene not in decision.get('scenes', [])):
            p.error('Formal run is outside the recorded decision')
    if a.output.exists():
        raise FileExistsError(a.output)
    W.EXPERIMENT = 'AC_ISOLATED_' + a.arm
    W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    W.RENDERER_BACKWARD_MODE = 'corrected_v1'
    W.RESPONSIBILITY_ENABLED = False
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.ACTIVE_MDRR_OVERRIDE = False
    W.ACTIVE_CICA_OVERRIDE = True
    W.START_STEP = 0
    W.CICA_START_STEP = 3 if a.smoke else 10000
    W.CICA_REFRESH_STEPS = (3, 9) if a.smoke else (10000, 12000, 14000)
    W.ACTIVE_RENDER_ROOT = a.output / 'renders'
    W._active_snapshots = lambda final: tuple(s for s in (10000, final) if s <= final)
    W.MIC._metric_images = lambda model, pred, gt: W._taba_rgb_metrics(pred, gt)
    route_original = W._route_renderer_backend
    def route(mode):
        result = route_original(mode)
        from scripts.experiments.geometry_projection_backend import install
        install()
        result['projection_backward'] = 'clip_chain_rule_A_20261021'
        return result
    W._route_renderer_backend = route
    controllers = {}
    def geometry(branch, outputs, loss, step, output):
        if id(branch) not in controllers:
            controllers[id(branch)] = PairHooks(branch, output, a.scene, a.mask_dir.resolve(),
                a.arm in ('A010', 'A010_C2'), a.arm in ('C2', 'A010_C2'), a.smoke)
            (output / 'AC_config.json').write_text(json.dumps(dict(scene=a.scene, arm=a.arm,
                seed=42, smoke=a.smoke, A_eta=.1, C_gradient_cap=.05,
                independent_hooks=True, auxiliary_densification_isolation=True,
                historical_C2_is_not_a_clean_control=True,
                gate_decision=str(a.decision),
                gate_decision_sha256=(__import__('hashlib').sha256(a.decision.read_bytes()).hexdigest()
                                      if a.decision is not None else None)), indent=2)+'\n')
        controllers[id(branch)].geometry(branch, outputs, loss, step, output)
    def extra(branch, *args):
        return controllers[id(branch)].extra(branch, *args)
    W.GEOMETRY_GRADIENT_HOOK = geometry
    W.EXTRA_GRADIENT_HOOK = extra
    try:
        result = W.run(a.scene, 'B', a.gpu, a.output.resolve(), max_step=20 if a.smoke else 15000,
            source_config=ROOT/'research_runs/bnd_on_baseline_scratch15k_20261021_v2'/a.scene/'source_config.yml',
            training_seed=42, raw_data_only=True)
        print(json.dumps(dict(scene=a.scene, arm=a.arm, final_step=result['final_step'])), flush=True)
    finally:
        for controller in controllers.values():
            controller.close()


if __name__ == '__main__':
    main()
