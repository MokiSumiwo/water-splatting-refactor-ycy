#!/usr/bin/env python3
"""Frozen training-view curriculum audit and short-run gross-instability gate."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.experiments.run_fsb_worker import W
from water_splatting.fields.gaussian_appearance import resolve_headroom_alpha


def setup(scene, short_root=None):
    path = (short_root if short_root else ROOT / "outputs/fsb_bnd_on_four_scene_strict_20260915_v1") / scene
    m = json.loads((path / "experiment_manifest.json").read_text())
    W.ACTIVE_SOURCE_CONFIG = path / "source_config.yml"
    W.ACTIVE_SOURCE_ROOT = Path(m["source_root"])
    W.ACTIVE_PARAMETERIZATION = "bounded_headroom_sh3"
    W.FORMAL.ACTIVE_PARAMETERIZATION = W.ACTIVE_PARAMETERIZATION
    W.AUX.ACTIVE_PARAMETERIZATION = W.ACTIVE_PARAMETERIZATION
    W.HEADROOM_ALPHA = 1.0
    W.HEADROOM_SCHEDULE = "smoothstep" if short_root else "constant"
    W.HEADROOM_START_STEP, W.HEADROOM_END_STEP = 3000, 8000
    W.DEGREEWISE_HEADROOM_ALPHAS = (1., 1., 1.)
    W._seed_all(42)
    b = W._new_branch(scene)
    if short_root:
        W._load_snapshot(b, scene, path / "checkpoints/step-000005000.ckpt")
    else:
        W._load_source(b, scene)
    return b


def zero(scene, out):
    b = setup(scene)
    model = b.pipeline.model
    before_model, before_opt = W._model_state_hash(model), W._optimizer_hash(b.optimizers)
    rows, gradient_rows = [], []
    rec = W._train_records(b)
    indices = sorted(set((0, len(rec)//2, len(rec)-1)))
    # Endpoint forwards on the actual path, with the same absolute step for each pair.
    with torch.no_grad():
        for idx in indices:
            _, view, camera, _ = rec[idx]
            camera = camera.to(model.device)
            for step, reference in ((3000, "bounded_sh3"), (8000, "bounded_headroom_sh3")):
                model.step = step
                model.config.headroom_schedule = "smoothstep"
                model.config.intrinsic_color_parameterization = "bounded_headroom_sh3"
                curriculum = model.get_outputs_for_camera(camera)
                model.config.headroom_schedule = "constant"
                model.config.intrinsic_color_parameterization = reference
                endpoint = model.get_outputs_for_camera(camera)
                diffs = {k: float((curriculum[k]-endpoint[k]).abs().max()) for k in ("pred_image", "rgb_clear", "gaussian_view_rgb")}
                assert max(diffs.values()) < 1e-6, diffs
                rows.append(dict(scene=scene, view_id=view, absolute_step=step, kind="endpoint", **diffs))
                del curriculum, endpoint
    # Fix model step (and hence SH degree, resolution and renderer) for an alpha-only VJP sweep.
    model.step = 3000
    model.config.intrinsic_color_parameterization = "bounded_headroom_sh3"
    model.config.headroom_schedule = "constant"
    _, view, camera, batch = rec[0]
    batch = W._batch_to_device(batch, model.device)
    dc, rest = model.features_dc, model.features_rest
    previous_grad = None
    for alpha in (0., .352, .5, 1.):
        model.config.bounded_headroom_alpha = alpha
        outputs = model.get_outputs(camera.to(model.device))
        gt = W._gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        loss = (outputs["pred_image"]-gt).square().mean()
        grads = torch.autograd.grad(loss, (dc, rest))
        assert all(torch.isfinite(g).all() for g in grads)
        colors = outputs["gaussian_view_rgb"]
        assert 0 <= float(colors.min()) <= float(colors.max()) <= 1
        row = dict(scene=scene, view_id=view, alpha=alpha, mse=float(loss),
                   dc_grad_norm=float(grads[0].norm()), rest_grad_norm=float(grads[1].norm()),
                   gaussian_min=float(colors.min()), gaussian_max=float(colors.max()))
        for name, param, grad in zip(("features_dc", "features_rest"), (dc, rest), grads):
            optimizer = b.optimizers.optimizers[name]
            state = optimizer.state[param]
            momentum = state.get("exp_avg", torch.zeros_like(param))
            row[name+"_grad_momentum_cosine"] = float((grad*momentum).sum() / (grad.norm()*momentum.norm()).clamp_min(1e-30))
            # Read-only Adam update prediction; neither params nor moments are modified.
            cfg = optimizer.param_groups[0]
            beta1, beta2 = cfg["betas"]
            step = int(state.get("step", 0)) + 1
            avg = beta1*momentum + (1-beta1)*grad
            sq = beta2*state.get("exp_avg_sq", torch.zeros_like(param)) + (1-beta2)*grad.square()
            delta = cfg["lr"] * (avg/(1-beta1**step)) / ((sq/(1-beta2**step)).sqrt()+cfg["eps"])
            row[name+"_predicted_adam_delta_norm"] = float(delta.norm())
        if previous_grad is not None:
            row["rest_grad_cosine_to_previous_alpha"] = float((previous_grad*grads[1]).sum() / (previous_grad.norm()*grads[1].norm()).clamp_min(1e-30))
        previous_grad = grads[1].detach()
        gradient_rows.append(row)
        del outputs, loss, grads, gt
    after_model, after_opt = W._model_state_hash(model), W._optimizer_hash(b.optimizers)
    assert (before_model, before_opt) == (after_model, after_opt)
    steps = range(2999, 8002)
    alphas = [resolve_headroom_alpha(s, schedule="smoothstep") for s in steps]
    max_delta = max(y-x for x,y in zip(alphas, alphas[1:]))
    assert all(x <= y for x,y in zip(alphas, alphas[1:])) and max_delta <= .000300001
    W._write_csv(out / "endpoints.csv", rows)
    W._write_csv(out / "gradients.csv", gradient_rows)
    W._write_json(out / "audit.json", dict(pass_all=True, train_view_indices=indices,
                  test_views_used=False, optimizer_step_performed=False, model_hash=before_model,
                  optimizer_hash=before_opt, state_unchanged=True, max_alpha_change_per_step=max_delta,
                  alpha_at_5000=resolve_headroom_alpha(5000, schedule="smoothstep"),
                  universal_color_change_bound_per_step=max_delta))
    W._release(b)


def short(scene, root, out):
    results = []
    for method, short_root in (("FSB", None), ("CURR", root)):
        b = setup(scene, short_root)
        model = b.pipeline.model
        if method == "FSB":
            W._load_snapshot(b, scene, ROOT / "outputs/fsb_bnd_on_four_scene_strict_20260915_v1" / scene / "checkpoints/step-000005000.ckpt")
        rec = W._train_records(b)
        with torch.no_grad():
            for idx in sorted(set((0,len(rec)//2,len(rec)-1))):
                _,view,camera,batch = rec[idx]
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = W._gt(model, W._batch_to_device(batch,model.device), outputs["background"]).float().clamp(0,1)
                pred = outputs["pred_image"].float().clamp(0,1)
                results.append(dict(method=method, scene=scene, view_id=view, MSE=float((pred-gt).square().mean()),
                                    effective_alpha=float(outputs["headroom_alpha"])))
                del outputs,gt,pred
        W._release(b)
    avg = {m: sum(r["MSE"] for r in results if r["method"]==m)/3 for m in ("FSB","CURR")}
    ratio=avg["CURR"]/avg["FSB"]
    start = json.loads((root/scene/"start_state_equivalence.json").read_text())
    reload = json.loads((root/scene/"smoke_reload_audit.json").read_text())
    routing = json.loads((root/scene/"gradient_routing_audit.json").read_text())
    log = list(csv.DictReader((root/scene/"training_summary_A2.csv").open()))
    alpha_error=max(abs(float(r["headroom_alpha"])-resolve_headroom_alpha(int(r["absolute_step"]), schedule="smoothstep")) for r in log)
    passed = ratio <= 1.25 and start["START_STATE_EQUIVALENCE"] and reload["model_hash_match"] and reload["forward_finite"] and routing["GRADIENT_ROUTING_AUDIT"] and alpha_error<1e-7
    W._write_csv(out/"train_view_metrics.csv",results)
    W._write_json(out/"audit.json",dict(pass_all=passed,mean_train_mse=avg,curriculum_over_fsb_mse=ratio,
                  alpha_log_max_error=alpha_error, test_views_used=False, threshold=1.25,
                  note="Gross instability check only; not a validation-set generalization estimate"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",required=True, choices=tuple(W.SCENE_GPUS))
    p.add_argument("--gpu",required=True,choices=("6","7","8","9"))
    p.add_argument("--mode",choices=("zero","short"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--short-root",type=Path)
    a=p.parse_args()
    W._runtime(a.scene,a.gpu)
    out=a.output.resolve()/a.scene
    out.mkdir(parents=True,exist_ok=False)
    if a.mode=="zero":zero(a.scene,out)
    else:short(a.scene,a.short_root.resolve(),out)
    print((out/"audit.json").read_text(),flush=True)


if __name__=="__main__":main()
