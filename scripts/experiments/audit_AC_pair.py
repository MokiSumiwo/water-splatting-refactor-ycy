"""Finite, zero-update audit of composing the actual A010 and C2 hooks."""
import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.run_AC_pair_worker import Geometry010, IsolatedGeometry010, IsolatedOpenWaterC
from water_splatting.open_water_C import OpenWaterC

ROOT = Path(__file__).resolve().parents[2]


def snapshot(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def equal(x, y):
    return all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)) for a, b in zip(x, y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corrected', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    W._route_renderer_backend('corrected_v1')
    from scripts.experiments.geometry_projection_backend import install
    install()
    run = ROOT/'research_runs/geometry_gradient_A_20261021/formal'/a.scene/'B0'
    W.ACTIVE_SOURCE_CONFIG = run/'source_config.yml'
    W.ACTIVE_PARAMETERIZATION = 'bounded_sh3'
    W.ACTIVE_TRAINING_SEED = 42
    b = W._new_branch(a.scene)
    try:
        W._load_snapshot(b, a.scene, run/'checkpoints/step-000010000.ckpt')
        m = b.pipeline.model
        m.train()
        m.step_cb(4000)
        before_hash = W._model_state_hash(m)
        # Controllers use output.parent.name as the scene, matching formal paths.
        ga = (IsolatedGeometry010 if a.corrected else Geometry010)(W, b, a.output, True)
        cc = (IsolatedOpenWaterC if a.corrected else OpenWaterC)(W, b, a.output, a.mask_dir, 'C2', scene=a.scene)
        rows = json.loads((a.mask_dir/'manifest.json').read_text())['rows']
        positive = sorted(r['camera'] for r in rows if r['pixels'])
        negative = sorted(r['camera'] for r in rows if not r['pixels'])
        names = [positive[0], positive[len(positive)//2]] + negative[:1]
        result = []
        for index, name, camera, raw_batch in W._train_records(b):
            if str(name) not in names:
                continue
            batch = W._batch_to_device(raw_batch, m.device)
            state = W._rng_state()
            variants = {}
            standalone = {}
            for variant in ['off', 'A010', 'C2', 'A010_C2']:
                W._set_rng_state(state)
                b.optimizers.zero_grad_all()
                o = m.get_outputs(copy.deepcopy(camera).to(m.device))
                loss = sum(m.get_loss_dict(o, batch, {}).values())
                params, groups = W._parameter_entries(m)
                native = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                W._assign_grads(params, native)
                base = snapshot(params)
                screen = m.xys_grad_abs.detach().clone()
                rng = W._hash_object(W._rng_state())
                ga.enabled = variant in ('A010', 'A010_C2')
                error = None
                info = {}
                try:
                    ga.apply(m, o, loss, 4000)
                    info = cc.apply(b, o, batch, loss, 4000 if variant in ('C2','A010_C2') else 10000,
                        a.output, params, groups, index, {'camera_name':str(name)})
                except RuntimeError as exc:
                    error = str(exc)
                grads = snapshot(params)
                combination_error = 0.0
                if variant in ('A010','C2'):
                    standalone[variant] = grads
                elif variant == 'A010_C2' and error is None:
                    expected = [standalone['A010'][j] if j in groups['means'] else standalone['C2'][j] for j in range(len(params))]
                    difference = sum(float((g-e).square().sum()) for g,e in zip(grads,expected) if g is not None)
                    denominator = sum(float(e.square().sum()) for e in expected if e is not None)
                    combination_error = (difference / max(denominator,1e-24))**.5
                c_ids = {j for group in ['medium_mlp','direction_encoding','opacities'] for j in groups[group]}
                allowed = set(groups['means']) if variant=='A010' else c_ids if variant=='C2' else set(groups['means'])|c_ids if variant=='A010_C2' else set()
                unchanged = equal([base[j] for j in range(len(params)) if j not in allowed],
                                  [grads[j] for j in range(len(params)) if j not in allowed])
                variants[variant] = dict(runtime_error=error,
                    combination_relative_l2=combination_error,
                    disabled_exact=equal(base, grads) if variant=='off' else None,
                    parameter_routing_exact=unchanged,
                    densification_buffer_exact=bool(torch.equal(screen,m.xys_grad_abs)),
                    densification_buffer_max_abs_delta=float((screen-m.xys_grad_abs).abs().max()),
                    densification_buffer_delta_l2=float((screen-m.xys_grad_abs).norm()),
                    densification_buffer_native_l2=float(screen.norm()),
                    rng_exact=rng==W._hash_object(W._rng_state()),
                    means_delta_l2=float((grads[groups['means'][0]]-base[groups['means'][0]]).norm()),
                    C_medium_delta_l2=info.get('C_medium_grad_added_l2',0),
                    C_opacity_delta_l2=info.get('C_opacity_grad_added_l2',0))
                del o, loss, params, native, base, grads
            row = dict(camera=str(name), empty_mask=not next(r['pixels'] for r in rows if r['camera']==str(name)),variants=variants)
            row['passed'] = all(v['runtime_error'] is None and v['parameter_routing_exact'] and v['densification_buffer_exact'] and v['rng_exact'] and v['combination_relative_l2'] < 1e-5 for v in variants.values())
            result.append(row)
        ga.log.close()
        cc.log.close()
        model_unchanged = before_hash == W._model_state_hash(m)
        report = dict(scene=a.scene, corrected=a.corrected, optimizer_updates=0, model_unchanged=model_unchanged, rows=result,
                      passed=model_unchanged and all(r['passed'] for r in result))
        (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        W._release(b)


if __name__ == '__main__':
    main()
