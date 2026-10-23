#!/usr/bin/env python3
"""Read-only real-checkpoint audit of CICA's appearance DC Jacobian."""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.diagnostics.evaluate_fsb_degreewise_scene import configure, run_paths, W
from water_splatting.fields.gaussian_appearance import (
    compute_bounded_gaussian_colors,
    compute_bounded_headroom_gaussian_colors,
)
from water_splatting.rasterize import rasterize_gaussians

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
PROBE_ORDINALS = (0, 3, 5)


def relative_l2(value, reference):
    return float((value.double()-reference.double()).norm() /
                 reference.double().norm().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(a, b)/(a.norm()*b.norm()).clamp_min(1e-30))


def quantiles(values):
    values = values.detach().double().reshape(-1)
    if not values.numel():
        return dict(count=0, mean=None, p50=None, p90=None, p99=None, max=None)
    q = torch.quantile(values, values.new_tensor([.5, .9, .99]))
    return dict(count=values.numel(), mean=float(values.mean()),
                p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), max=float(values.max()))


def actual_dc_derivative(outputs, method):
    if method == "BND_CICA":
        c = outputs["gaussian_view_rgb"].detach().float()
        return c*(1-c)
    c0 = outputs["gaussian_view_dc_rgb"].detach().float()
    r = outputs["gaussian_sh_residual"].detach().float()
    h = torch.tanh(torch.where(r >= 0, c0, 1-c0)*r)
    factor = torch.where(r >= 0, 1-h+(1-c0)*r*(1-h*h),
                         1+h-c0*r*(1-h*h))
    return c0*(1-c0)*factor


@torch.no_grad()
def accumulate(model, camera, outputs, residual, linear):
    geometry = W.FULL._project_for_camera(model, camera)
    ids, bins, bounds = W.FULL._geometry_bins(geometry)
    return W._cica_extension().cica_forward(
        torch.tensor(bounds, dtype=torch.int32, device=model.device),
        torch.tensor([geometry["width"], geometry["height"], 1], dtype=torch.int32, device=model.device),
        ids.contiguous(), bins.contiguous(), geometry["xys"].float().contiguous(),
        geometry["conics"].float().contiguous(),
        outputs["gaussian_view_rgb"].detach().float().contiguous(),
        torch.sigmoid(model.opacities.detach()).reshape(-1).float().contiguous(),
        outputs["medium_attn"].detach().float().contiguous(),
        geometry["depths"].reshape(-1).float().contiguous(),
        residual.detach().reshape(-1, 3).float().contiguous(), 16, int(linear))


@torch.no_grad()
def evidence(numerator, information):
    info_sum = information.sum(-1)
    finite = torch.isfinite(numerator).all(-1) & torch.isfinite(information).all(-1)
    valid = finite & (info_sum > W.CICA_INFORMATION_FLOOR)
    delta = numerator/information.clamp_min(W.CICA_INFORMATION_FLOOR)
    return dict(delta_parameter=torch.where(valid[:, None], delta, torch.zeros_like(delta)),
                information=information, information_sum=info_sum, valid=valid)


def color_autograd_audit(model, camera, outputs, method, derivative):
    ids = torch.where(outputs["gaussian_visible_mask"].reshape(-1))[0]
    if len(ids) > 8192:
        ids = ids[torch.linspace(0, len(ids)-1, 8192, device=ids.device).long()]
    dc = model.features_dc[ids].detach().clone().requires_grad_(True)
    rest = model.features_rest[ids].detach()
    fn = compute_bounded_gaussian_colors if method == "BND_CICA" else compute_bounded_headroom_gaussian_colors
    result = fn(means=model.means[ids].detach(), features_dc=dc, features_rest=rest,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=model.config.sh_degree,
                active_sh_degree=min(model.step//model.config.sh_degree_interval, model.config.sh_degree))
    g, = torch.autograd.grad(result.rgb.sum(), (dc,))
    ref = derivative[ids]*W.SH_C0
    return dict(sampled_gaussians=len(ids),
                color_max_abs=float((result.rgb.detach()-outputs["gaussian_view_rgb"][ids]).abs().max()),
                derivative_max_abs=float((g-ref).abs().max()),
                derivative_relative_l2=relative_l2(ref, g))


@torch.no_grad()
def derivative_statistics(old, new, linear_info):
    q2 = linear_info.double()
    old, new = old.double(), new.double()
    mass = q2*new.square()
    diff = old-new
    relative = diff.abs()/new.abs().clamp_min(1e-8)
    ratio = old/new.clamp_min(1e-8)
    active = (q2 > 0) & (new > 1e-8)
    bins = torch.tensor([0, .25, .5, .75, .9, .99, 1.01, 1.1, 1.25, 1.5, 2, 4, float("inf")], device=old.device)
    indices = torch.bucketize(ratio.reshape(-1).contiguous(), bins[1:-1])
    hist = torch.zeros(len(bins)-1, device=old.device, dtype=torch.float64)
    hist.scatter_add_(0, indices, mass.reshape(-1))
    return dict(
        exact_information=float(mass.sum()), old_information=float((q2*old.square()).sum()),
        jacobian_error_squared=float((q2*diff.square()).sum()),
        weighted_jacobian_relative_l2=float(((q2*diff.square()).sum()/mass.sum().clamp_min(1e-30)).sqrt()),
        information_mass_error_gt_10=float(mass[relative > .1].sum()/mass.sum().clamp_min(1e-30)),
        information_mass_error_gt_25=float(mass[relative > .25].sum()/mass.sum().clamp_min(1e-30)),
        near_zero_derivative_information_fraction=float(mass[new <= 1e-8].sum()/mass.sum().clamp_min(1e-30)),
        active_ratio=quantiles(ratio[active]), ratio_histogram_information=hist.cpu().tolist(),
        ratio_histogram_edges=[0, .25, .5, .75, .9, .99, 1.01, 1.1, 1.25, 1.5, 2, 4, None])


def rebuild_controller(model, records, cache):
    iterator = iter(cache)
    calls = []

    def cached_evidence(m, camera, outputs, residual):
        calls.append(1)
        return next(iterator)

    # Reuse the exact target/weighted-median implementation without changing production code.
    with patch.object(W, "_cica_local_evidence", cached_evidence):
        result = W._build_cica_controller(model, records)
    assert len(calls) == len(records)
    return result


def controller_audit(model, old, new, lambda_cica):
    common = (old["gate"] > 0) & (new["gate"] > 0)
    union = (old["gate"] > 0) | (new["gate"] > 0)
    dt = new["target_log_chroma"]-old["target_log_chroma"]
    weights = new["weight"].double()*common
    loss_rows = {}
    grads = {}
    arms = {"old": old, "corrected": new,
            "target_only": {**old, "target_log_chroma": new["target_log_chroma"]},
            "weight_only": {**old, "weight": new["weight"]}}
    for name, controller in arms.items():
        loss, _ = W._cica_loss(model, controller)
        grad, = torch.autograd.grad(loss, (model.features_dc,))
        grads[name] = grad.detach()
        loss_rows[name] = dict(raw_loss=float(loss.detach()), raw_gradient_norm=float(grad.norm()),
                               saved_lambda_scaled_gradient_norm=float(grad.norm())*lambda_cica)
    for name in arms:
        loss_rows[name].update(cosine_to_old=cosine(grads[name], grads["old"]),
                              gradient_relative_change=relative_l2(grads[name], grads["old"]))
    dc = torch.sigmoid(model.features_dc.detach()*W.SH_C0)
    current = torch.stack((torch.log((dc[:, 0]+1e-6)/(dc[:, 1]+1e-6)),
                           torch.log((dc[:, 2]+1e-6)/(dc[:, 1]+1e-6))), -1).double()
    old_correction = old["target_log_chroma"]-current
    new_correction = new["target_log_chroma"]-current
    directional = common[:, None] & (old_correction.abs() > 1e-4) & (new_correction.abs() > 1e-4)
    flipped = (old_correction*new_correction < 0) & directional
    stats = dict(old_valid=int((old["gate"] > 0).sum()), corrected_valid=int((new["gate"] > 0).sum()),
                 common_valid=int(common.sum()), union_valid=int(union.sum()),
                 gate_changed=int((old["gate"] != new["gate"]).sum()),
                 support_changed=int((old["support"] != new["support"]).sum()),
                 target_abs_delta=quantiles(dt[common].abs()),
                 weighted_target_abs_delta=float((weights[:, None]*dt.abs()).sum()/(2*weights.sum()).clamp_min(1e-30)),
                 old_weight_sum=float(old["weight"].double().sum()), new_weight_sum=float(new["weight"].double().sum()),
                 weight_relative_l2=relative_l2(new["weight"], old["weight"]),
                 directional_channels=int(directional.sum()), correction_sign_flip_channels=int(flipped.sum()),
                 correction_sign_flip_fraction=float(flipped.sum()/directional.sum().clamp_min(1)),
                 gradients=loss_rows, saved_lambda_cica=lambda_cica)
    # Small deterministic sample only; no topology matching across different checkpoints.
    ids = torch.where(common)[0]
    if len(ids) > 1024:
        ids = ids[torch.linspace(0, len(ids)-1, 1024, device=ids.device).long()]
    data = torch.cat((ids[:, None].double(), old["target_log_chroma"][ids], new["target_log_chroma"][ids],
                      old["weight"][ids, None].double(), new["weight"][ids, None].double()), -1).cpu().tolist()
    names = ("gaussian_id", "old_target_rg", "old_target_bg", "new_target_rg", "new_target_bg", "old_weight", "new_weight")
    return stats, [dict(zip(names, row)) for row in data]


def output_hashes(outputs):
    return {k: W._hash_tensor(outputs[k]) for k in ("pred_image", "rgb_clear", "accumulation", "depth")}


@torch.no_grad()
def forward_adjoint_probe(model, camera, outputs, cotangent, dc_derivative, corrected_vjp, native_vjp):
    geometry = W.FULL._project_for_camera(model, camera)
    # Probe the observed disagreement direction; no model parameter is changed.
    direction = corrected_vjp-native_vjp
    direction = direction/direction.abs().max().clamp_min(1e-20)
    tangent_colors = dc_derivative*W.SH_C0*direction
    direct = rasterize_gaussians(
        geometry["xys"], torch.zeros_like(geometry["xys"]), geometry["depths"],
        geometry["radii"], geometry["conics"], geometry["num_tiles_hit"], tangent_colors,
        torch.sigmoid(model.opacities.detach()), outputs["medium_rgb"].detach(),
        outputs["medium_bs"].detach(), outputs["medium_attn"].detach(),
        geometry["height"], geometry["width"], 16, return_alpha=True,
        step=model.step, return_hit_stats=True)[0]
    lhs = (direct.double()*cotangent.double()).sum()
    rhs = (corrected_vjp.double()*direction.double()).sum()
    old_rhs = (native_vjp.double()*direction.double()).sum()
    return dict(direction="normalized_corrected_minus_native_vjp", forward_dot=float(lhs),
                corrected_dot=float(rhs), native_backward_dot=float(old_rhs),
                corrected_relative_error=float((rhs-lhs).abs()/lhs.abs().clamp_min(1e-20)),
                native_backward_relative_error=float((old_rhs-lhs).abs()/lhs.abs().clamp_min(1e-20)))


def synthetic_alpha_audit():
    rows = []
    for opacity in (.5, .99, .995, .999):
        color = torch.full((1, 3), .4, device="cuda", requires_grad=True)
        direct = rasterize_gaussians(
            torch.zeros(1, 2, device="cuda"), torch.zeros(1, 2, device="cuda"),
            torch.ones(1, device="cuda"), torch.tensor([3], device="cuda", dtype=torch.int32),
            torch.tensor([[1., 0., 1.]], device="cuda"), torch.ones(1, device="cuda", dtype=torch.int32),
            color, torch.tensor([[opacity]], device="cuda"), torch.zeros(1, 1, 3, device="cuda"),
            torch.zeros(1, 1, 3, device="cuda"), torch.zeros(1, 1, 3, device="cuda"),
            1, 1, 16, return_alpha=True, step=14999, return_hit_stats=True)[0]
        grad, = torch.autograd.grad(direct.sum(), (color,))
        rows.append(dict(opacity=opacity, forward_color_derivative=float(direct[0, 0, 0].detach()/.4),
                         native_backward_color_derivative=float(grad[0, 0])))
    return rows


def audit_method(scene, method, run, out):
    manifest = configure(run)
    assert manifest["locked_config"]["rasterize_mode"] == "classic"
    assert W.HEADROOM_ALPHA == 1 and W.DEGREEWISE_HEADROOM_ALPHAS == (1., 1., 1.)
    checkpoint = run/"checkpoints/step-000014999.ckpt"
    before_file = W._sha256(checkpoint)
    branch = None
    started = time.perf_counter()
    try:
        W._seed_all(42)
        branch = W._new_branch(scene)
        payload = W._load_snapshot(branch, scene, checkpoint)
        model = branch.pipeline.model
        before_model = W._model_state_hash(model)
        assert before_model == W._hash_object(payload["model"])
        lambda_cica = float(payload["metadata"]["resolved_lambda_CICA"])
        del payload
        before_config = W._locked_config(model)
        before_opt = W._optimizer_hash(branch.optimizers)
        bank = json.loads((run/"cica_camera_bank.json").read_text())
        train = W._train_records(branch)
        records = [train[i] for i in bank["camera_indices"]]
        assert [r[1] for r in records] == bank["camera_ids"] and len(records) == 6
        old_cache, new_cache, view_rows = [], [], []
        first_hashes = None
        W._cica_extension()
        torch.cuda.reset_peak_memory_stats()
        for ordinal, (_, view, camera, batch) in enumerate(records):
            camera = camera.to(model.device)
            native_probe = ordinal in PROBE_ORDINALS
            with torch.set_grad_enabled(native_probe):
                outputs = model.get_outputs(camera)
            if ordinal == 0:
                first_hashes = output_hashes(outputs)
            gt = W._gt(model, W._batch_to_device(batch, model.device), outputs["background"]).detach().float()
            residual = gt-outputs["pred_image"].detach().float()
            c = outputs["gaussian_view_rgb"].detach().float()
            old_d = c*(1-c)
            new_d = actual_dc_derivative(outputs, method)
            assert torch.isfinite(new_d).all() and float(new_d.min()) >= -1e-8
            local_check = color_autograd_audit(model, camera, outputs, method, new_d)
            linear_n, linear_i = accumulate(model, camera, outputs, residual, True)
            original_n, original_i = accumulate(model, camera, outputs, residual, False)
            factor_n, factor_i = linear_n*old_d, linear_i*old_d.square()
            new_n, new_i = linear_n*new_d, linear_i*new_d.square()
            old_ev, new_ev = evidence(original_n, original_i), evidence(new_n, new_i)
            old_cache.append(old_ev)
            new_cache.append(new_ev)
            row = dict(scene=scene, method=method, view_id=view, ordinal=ordinal,
                       local_autograd=local_check, statistics=derivative_statistics(old_d, new_d, linear_i),
                       old_factor_n_relative_l2=relative_l2(factor_n, original_n),
                       old_factor_i_relative_l2=relative_l2(factor_i, original_i),
                       local_delta_relative_l2=relative_l2(new_ev["delta_parameter"], old_ev["delta_parameter"]),
                       native_probes=[], forward_probes=[])
            if native_probe:
                native, = torch.autograd.grad((outputs["pred_image"]*residual).sum(), (model.features_dc,), retain_graph=True)
                row["native_probes"].append(dict(cotangent="gt_minus_pred", corrected_relative_l2=relative_l2(new_n, native),
                                                 old_relative_l2=relative_l2(original_n, native), corrected_cosine=cosine(new_n, native)))
                row["forward_probes"].append(dict(cotangent="gt_minus_pred", **forward_adjoint_probe(
                    model, camera, outputs, residual, new_d, new_n, native)))
                probe = (torch.arange(residual.numel(), device=model.device).reshape_as(residual)%17).float()/8-1
                pn, _ = accumulate(model, camera, outputs, probe, True)
                native2, = torch.autograd.grad((outputs["pred_image"]*probe).sum(), (model.features_dc,))
                row["native_probes"].append(dict(cotangent="fixed_signed", corrected_relative_l2=relative_l2(pn*new_d, native2),
                                                 old_relative_l2=relative_l2(pn*old_d, native2), corrected_cosine=cosine(pn*new_d, native2)))
                row["forward_probes"].append(dict(cotangent="fixed_signed", **forward_adjoint_probe(
                    model, camera, outputs, probe, new_d, pn*new_d, native2)))
                del native, native2, probe, pn
            row["forward_reference_pass"] = (local_check["derivative_max_abs"] <= 2e-6 and local_check["color_max_abs"] <= 2e-6
                           and row["old_factor_n_relative_l2"] <= 1e-3 and row["old_factor_i_relative_l2"] <= 1e-3
                           and all(p["corrected_relative_error"] <= 1e-3 for p in row["forward_probes"]))
            row["native_backward_pass"] = all(p["corrected_relative_l2"] <= 1e-3 for p in row["native_probes"])
            row["pass"] = row["forward_reference_pass"] and row["native_backward_pass"]
            view_rows.append(row)
            W._write_json(out/f"{method}_views.json", view_rows)
            print(scene, method, view, "J error", row["statistics"]["weighted_jacobian_relative_l2"], "pass", row["pass"], flush=True)
            del outputs, residual, gt, c, old_d, new_d, linear_n, linear_i, original_n, original_i, factor_n, factor_i, new_n, new_i
        old_controller = rebuild_controller(model, records, old_cache)
        new_controller = rebuild_controller(model, records, new_cache)
        target_stats, samples = controller_audit(model, old_controller, new_controller, lambda_cica)
        W._write_csv(out/f"{method}_target_samples.csv", samples)
        with torch.no_grad():
            final_outputs = model.get_outputs_for_camera(records[0][2].to(model.device))
        preservation = dict(model_hash_unchanged=before_model == W._model_state_hash(model),
                            config_unchanged=before_config == W._locked_config(model),
                            unused_branch_optimizer_unchanged=before_opt == W._optimizer_hash(branch.optimizers),
                            outputs_identical=first_hashes == output_hashes(final_outputs),
                            checkpoint_file_unchanged=before_file == W._sha256(checkpoint),
                            model_parameter_grads_unset=all(p.grad is None for p in model.parameters()))
        info = sum(r["statistics"]["exact_information"] for r in view_rows)
        error = sum(r["statistics"]["jacobian_error_squared"] for r in view_rows)
        summary = dict(scene=scene, method=method, checkpoint=str(checkpoint), checkpoint_sha256=before_file,
                       model_hash=before_model, bank=bank, test_views_used=False, optimizer_steps=0,
                       checkpoint_optimizer_loaded=False, native_reference_hashes=first_hashes,
                       weighted_jacobian_relative_l2=(error/max(info, 1e-30))**.5,
                       information_mass_error_gt_10=sum(r["statistics"]["exact_information"]*r["statistics"]["information_mass_error_gt_10"] for r in view_rows)/max(info, 1e-30),
                       information_mass_error_gt_25=sum(r["statistics"]["exact_information"]*r["statistics"]["information_mass_error_gt_25"] for r in view_rows)/max(info, 1e-30),
                       controllers=target_stats, preservation=preservation,
                       elapsed_seconds=time.perf_counter()-started,
                       peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                       forward_reference_pass=all(preservation.values()) and all(r["forward_reference_pass"] for r in view_rows),
                       native_backward_pass=all(r["native_backward_pass"] for r in view_rows),
                       pass_all=all(preservation.values()) and all(r["pass"] for r in view_rows))
        W._write_json(out/f"{method}_summary.json", summary)
        assert summary["forward_reference_pass"], "Forward-reference checks failed; inspect retained records"
        if method == "BND_CICA":
            assert summary["weighted_jacobian_relative_l2"] == 0
        return summary
    finally:
        W._release(branch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--gpu", choices=("6", "7", "8", "9"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    W._runtime(a.scene, a.gpu)
    out = a.output.resolve()/a.scene
    out.mkdir(parents=True, exist_ok=False)
    W._write_json(out/"synthetic_alpha_audit.json", synthetic_alpha_audit())
    paths = run_paths(a.scene, ROOT/"unused")
    summaries = [audit_method(a.scene, m, paths[m], out) for m in ("BND_CICA", "FSB")]
    W._write_json(out/"status.json", dict(complete=True, all_pass=all(s["pass_all"] for s in summaries),
                                        forward_reference_pass=all(s["forward_reference_pass"] for s in summaries)))


if __name__ == "__main__":
    main()
