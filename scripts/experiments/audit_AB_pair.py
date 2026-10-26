"""Zero-update audit of actual ABHooks at two supported training cameras."""
import argparse
import json
from pathlib import Path

import torch
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.run_AB_worker import ABHooks
from scripts.experiments.geometry_projection_backend import install

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = ROOT/'research_runs/geometry_gradient_A_20261021/formal'/args.scene/'B0'
    (args.output/'camera_sequence.csv').symlink_to(run/'camera_sequence.csv')
    W._route_renderer_backend('corrected_v1'); install()
    W.ACTIVE_SOURCE_CONFIG = run/'source_config.yml'
    W.ACTIVE_PARAMETERIZATION = 'bounded_sh3'; W.ACTIVE_TRAINING_SEED = 42
    branch = W._new_branch(args.scene)
    hooks = None
    try:
        W._load_snapshot(branch, args.scene, run/'checkpoints/step-000010000.ckpt')
        model = branch.pipeline.model; model.train(); model.step_cb(4000)
        before_hash = W._model_state_hash(model)
        hooks = ABHooks(branch, args.output,
            ROOT/'research_runs/surface_support_B_20261023/support'/args.scene, telemetry=False)
        rows = []
        for _, name, camera, batch in W._train_records(branch):
            if name not in hooks.b.names: continue
            branch.optimizers.zero_grad_all()
            outputs = model.get_outputs(camera.to(model.device))
            valid, _ = hooks.b.select(model, outputs, name)
            if not valid.any(): continue
            loss = sum(model.get_loss_dict(outputs, W._batch_to_device(batch, model.device), {}).values())
            params, groups = W._parameter_entries(model)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            native = [None if g is None else g.detach().clone() for g in grads]
            buffer = model.xys_grad_abs.detach().clone()
            rng = W._hash_object(W._rng_state())
            hooks.b.sequence[4000] = name
            variants = {}; means = {}
            for label, use_a, use_b in [('off', False, False), ('A010', True, False), ('B', False, True), ('AB', True, True)]:
                W._assign_grads(params, [None if g is None else g.clone() for g in native])
                hooks.a.enabled = use_a; hooks.use_b = use_b; hooks.b.enabled = use_b
                hooks.apply(model, outputs, loss, 4000)
                means[label] = model.means.grad.detach().clone()
                other_exact = all((g is None and param.grad is None) or
                    (g is not None and torch.equal(g, param.grad)) for param, g in zip(params, native)
                    if param is not model.means)
                variants[label] = dict(nonmeans_exact=other_exact,
                    densification_exact=torch.equal(buffer, model.xys_grad_abs),
                    rng_exact=rng == W._hash_object(W._rng_state()),
                    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params))
            expected = means['A010'] + (means['B']-means['off'])
            relative = float((means['AB']-expected).norm()/expected.norm().clamp_min(1e-12))
            b_delta = means['B']-means['off']
            b_cap = bool((b_delta.norm(dim=-1) <= .05*means['off'].norm(dim=-1)+1e-8).all())
            native_means = native[groups['means'][0]]
            row = dict(camera=name, B_supported=int(valid.sum()), variants=variants,
                disabled_exact=torch.equal(means['off'], native_means),
                A_delta_l2=float((means['A010']-means['off']).norm()), B_delta_l2=float(b_delta.norm()),
                combination_relative_l2=relative, B_cap=b_cap)
            row['passed'] = (all(all(v.values()) for v in variants.values()) and row['disabled_exact']
                and b_cap and relative < 1e-5 and row['A_delta_l2'] > 0 and row['B_delta_l2'] > 0)
            rows.append(row)
            if len(rows) == 2: break
        model_unchanged = before_hash == W._model_state_hash(model)
        report = dict(scene=args.scene, optimizer_updates=0, model_unchanged=model_unchanged, rows=rows,
            passed=len(rows)==2 and model_unchanged and all(r['passed'] for r in rows))
        (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        if not report['passed']: raise RuntimeError('AB audit failed')
    finally:
        if hooks is not None: hooks.close()
        W._release(branch)


if __name__ == '__main__':
    main()
