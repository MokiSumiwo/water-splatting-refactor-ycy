#!/usr/bin/env python3
"""Reload all three registered arms; export native test renders and train-only SH diagnostics.

This is read-only evaluation. Configuration comes from each run's manifest,
because older checkpoints do not serialize fixed headroom coefficients.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.experiments.run_fsb_worker import W
from water_splatting.fields.gaussian_appearance import (
    compute_bounded_degreewise_headroom_gaussian_colors as degree_color,
    compute_bounded_headroom_gaussian_colors as head_color,
)
from water_splatting.sh import spherical_harmonics

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


def run_paths(scene, candidate, candidate_method="D1"):
    base = "bnd_on_cica_ablation_extra3_20260914" if scene in SCENES[:2] else "bnd_on_cica_ablation_20260914"
    return {
        "BND_CICA": ROOT / "outputs" / base / scene / "cica_on",
        "FSB": ROOT / "outputs/fsb_bnd_on_four_scene_strict_20260915_v1" / scene,
        candidate_method: candidate / scene,
    }


def configure(path):
    manifest = json.loads((path / "experiment_manifest.json").read_text())
    locked = manifest["locked_config"]
    W.ACTIVE_SOURCE_CONFIG = path / "source_config.yml"
    W.ACTIVE_SOURCE_ROOT = Path(manifest["source_root"])
    W.ACTIVE_PARAMETERIZATION = locked["intrinsic_color_parameterization"]
    W.FORMAL.ACTIVE_PARAMETERIZATION = W.ACTIVE_PARAMETERIZATION
    W.AUX.ACTIVE_PARAMETERIZATION = W.ACTIVE_PARAMETERIZATION
    W.HEADROOM_ALPHA = float(locked.get("bounded_headroom_alpha", 1.0))
    W.DEGREEWISE_HEADROOM_ALPHAS = tuple(locked.get("degreewise_headroom_alphas", (1., 1., 1.)))
    W.HEADROOM_SCHEDULE = locked.get("headroom_schedule", "constant")
    W.HEADROOM_START_STEP = locked.get("headroom_start_step", 3000)
    W.HEADROOM_END_STEP = locked.get("headroom_end_step", 8000)
    assert manifest["module_schedule"]["CICA"]["enabled"]
    assert not manifest["module_schedule"]["AUXILIARY_APPEARANCE"]["enabled"]
    assert manifest["training_seed"] == 42
    assert manifest["final_step"] == 14999
    return manifest


def protocol_audit(paths):
    starts = {k: json.loads((p / "start_state_equivalence.json").read_text()) for k, p in paths.items()}
    keys = ("source_checkpoint_sha256", "checkpoint_model_hash", "optimizer_hash", "scheduler_hash",
            "ocmc_projector_sha256", "training_rng_manifest")
    comparisons = {key: all(d[key] == starts["FSB"][key] for d in starts.values()) for key in keys}
    sequences = {k: list(csv.DictReader((p / "camera_sequence.csv").open())) for k, p in paths.items()}
    comparisons["camera_sequence_rows"] = all(v == sequences["FSB"] for v in sequences.values())
    comparisons["start_state_pass"] = all(d["START_STATE_EQUIVALENCE"] for d in starts.values())
    comparisons["gradient_audits_pass"] = all(json.loads((p / "gradient_routing_audit.json").read_text())["GRADIENT_ROUTING_AUDIT"] for p in paths.values())
    assert all(comparisons.values()), comparisons
    return comparisons


def math_audit(model):
    """Verify the nontrivial equivalence to scaling trainable SH coefficients."""
    ids = torch.linspace(0, model.means.shape[0]-1, min(4096, model.means.shape[0]), device=model.device).long()
    args = dict(means=model.means[ids].detach(), camera_position=model.means.new_tensor([[.2, -.1, .3]]),
                sh_degree=3, active_sh_degree=3)
    dc = model.features_dc[ids].detach().clone().requires_grad_(True)
    rest = model.features_rest[ids].detach().clone().requires_grad_(True)
    scale = rest.new_tensor([1.]*3 + [.75]*5 + [.5]*7)[None, :, None]
    deg = degree_color(**args, features_dc=dc, features_rest=rest, degree_alphas=(1., .75, .5))
    ref = head_color(**args, features_dc=dc, features_rest=rest*scale)
    v = torch.linspace(-1, 1, deg.rgb.numel(), device=model.device).reshape_as(deg.rgb)
    gd = torch.autograd.grad((deg.rgb*v).sum(), (dc, rest))
    gr = torch.autograd.grad((ref.rgb*v).sum(), (dc, rest))
    with torch.no_grad():
        unscaled = head_color(**args, features_dc=dc, features_rest=rest)
        compensated = degree_color(**args, features_dc=dc, features_rest=rest/scale, degree_alphas=(1., .75, .5))
        zero = degree_color(**args, features_dc=dc, features_rest=torch.zeros_like(rest), degree_alphas=(1., .75, .5))
    result = {
        "scaled_sh_forward_max_abs": float((deg.rgb-ref.rgb).abs().max()),
        "scaled_sh_vjp_dc_max_abs": float((gd[0]-gr[0]).abs().max()),
        "scaled_sh_vjp_rest_max_abs": float((gd[1]-gr[1]).abs().max()),
        "inverse_scaling_recovers_fsb_max_abs": float((compensated.rgb-unscaled.rgb).abs().max()),
        "zero_residual_dc_max_abs": float((zero.rgb-zero.dc_rgb).abs().max()),
        "rgb_min": float(deg.rgb.min()), "rgb_max": float(deg.rgb.max()),
    }
    assert max(result[k] for k in result if k.endswith("max_abs")) < 3e-6, result
    assert 0 <= result["rgb_min"] <= result["rgb_max"] <= 1
    return result


@torch.no_grad()
def train_degree_diagnostics(branch, method):
    model = branch.pipeline.model
    rows = []
    scales = W.DEGREEWISE_HEADROOM_ALPHAS
    for l, (lo, hi) in enumerate(((0, 3), (3, 8), (8, 15)), 1):
        coeff = model.features_rest[:, lo:hi].float()
        rows.append(dict(method=method, degree=l, kind="coefficient", gaussian_count=len(coeff),
                         raw_rms=float(coeff.square().mean().sqrt()),
                         effective_rms=float((coeff*scales[l-1]).square().mean().sqrt())))
    records = W._train_records(branch)
    for idx in sorted(set((0, len(records)//2, len(records)-1))):
        _, view, camera, _ = records[idx]
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = torch.where(outputs["gaussian_visible_mask"])[0]
        if len(visible) > 65536:
            visible = visible[torch.linspace(0, len(visible)-1, 65536, device=model.device).long()]
        means = model.means[visible]
        camera_pos = camera.camera_to_worlds[..., :3, 3].to(model.device)
        dirs = means - camera_pos
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        colors = torch.cat((model.features_dc[visible, None], model.features_rest[visible]), 1)
        prev = spherical_harmonics(0, dirs, colors)
        prev_rgb = torch.sigmoid(prev)
        for degree in (1, 2, 3):
            curr = spherical_harmonics(degree, dirs, colors)
            inc = curr-prev
            h = degree_color(means=means, features_dc=model.features_dc[visible], features_rest=model.features_rest[visible],
                             camera_position=camera_pos, sh_degree=3, active_sh_degree=degree, degree_alphas=scales)
            rows.append(dict(method=method, degree=degree, kind="train_visible", view_id=view,
                             sample_count=len(visible), raw_sh_rms=float(inc.square().mean().sqrt()),
                             effective_sh_rms=float((inc*scales[degree-1]).square().mean().sqrt()),
                             incremental_headroom_rgb_abs_mean=float((h.rgb-prev_rgb).abs().mean()),
                             cumulative_u_pos_mean=float(h.positive_utilization.mean()),
                             cumulative_u_neg_mean=float(h.negative_utilization.mean())))
            prev, prev_rgb = curr, h.rgb
        del outputs
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--gpu", choices=("6", "7", "8", "9"), required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-kind", choices=("degreewise", "curriculum"), default="degreewise")
    a = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == a.gpu
    assert os.environ.get("CONDA_DEFAULT_ENV") == "water_splatting"
    torch.set_num_threads(4)
    out = a.output.resolve() / a.scene
    out.mkdir(parents=True, exist_ok=False)
    candidate_method = "CURR" if a.candidate_kind == "curriculum" else "D1"
    paths = run_paths(a.scene, a.candidate.resolve(), candidate_method)
    W._write_json(out / "protocol_pairing.json", protocol_audit(paths))
    all_metrics, manifest_rows, audit_rows, degree_rows = [], [], [], []
    comparison = {}
    reference_views = None
    for method, path in paths.items():
        manifest = configure(path)
        branch = None
        try:
            W._seed_all(42)
            branch = W._new_branch(a.scene)
            ckpt = path / "checkpoints/step-000014999.ckpt"
            payload = W._load_snapshot(branch, a.scene, ckpt)
            assert payload["metadata"]["cica_enabled"]
            assert not payload["metadata"]["auxiliary_appearance_enabled"]
            model = branch.pipeline.model
            loaded_hash = W._model_state_hash(model)
            assert loaded_hash == W._hash_object(payload["model"])
            metadata_has_weights = "degreewise_headroom_alphas" in payload["metadata"]
            del payload
            records = sorted(W._eval_records(branch), key=lambda r: r[1])
            ids = [r[1] for r in records]
            if reference_views is None:
                reference_views = ids
            assert ids == reference_views
            orig = {r["view_id"]: r for r in csv.DictReader((path / "per_view_metrics.csv").open()) if int(r["absolute_step"]) == 14999}
            assert set(ids) == set(orig)
            method_rows = []
            for _, view, camera, batch in records:
                camera = camera.to(model.device)
                with torch.no_grad():
                    warm = model.get_outputs_for_camera(camera)
                    del warm
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    seconds = []
                    for repeat in range(3):
                        t = time.perf_counter()
                        outputs = model.get_outputs_for_camera(camera)
                        torch.cuda.synchronize()
                        seconds.append(time.perf_counter()-t)
                        if repeat < 2:
                            del outputs
                    peak_mb = torch.cuda.max_memory_allocated()/1024**2
                    gt = W._gt(model, W._batch_to_device(batch, model.device), outputs["background"]).detach().float().clamp(0, 1)
                raw = outputs["rgb_clear"].detach().float()
                pred = outputs["pred_image"].detach().float().clamp(0, 1)
                colors = outputs["gaussian_view_rgb"]
                accum = outputs["accumulation"].detach().float()
                assert torch.isfinite(pred).all() and torch.isfinite(raw).all() and torch.isfinite(colors).all()
                assert float(colors.min()) >= 0 and float(colors.max()) <= 1
                m = W.MIC._metric_images(model, pred, gt)
                errors = {key: abs(m[key]-float(orig[view][key])) for key in METRICS}
                assert errors["PSNR"] < 1e-4 and errors["SSIM"] < 1e-5 and errors["LPIPS"] < 1e-5 and errors["MSE"] < 1e-7, errors
                row = dict(scene=a.scene, method=method, view_id=view, height=pred.shape[0], width=pred.shape[1], **m,
                           clear_min=float(raw.min()), clear_max=float(raw.max()),
                           clear_black_fraction=float((raw.mean(-1)<.02).float().mean()),
                           accumulation_mean=float(accum.mean()), coverage_gt_001=float((accum>.01).float().mean()),
                           gaussian_color_min=float(colors.min()), gaussian_color_max=float(colors.max()),
                           render_seconds_median=float(np.median(seconds)), peak_render_allocated_mb=peak_mb)
                method_rows.append(row)
                images = {"underwater": pred, "clear": raw.clamp(0, 1), "gt": gt}
                if method == "FSB":
                    comparison[view] = {k: v.cpu().numpy() for k, v in images.items()}
                if method == candidate_method:
                    assert np.array_equal(gt.cpu().numpy(), comparison[view]["gt"])
                    for typ in ("underwater", "clear"):
                        diff = (images[typ].cpu().numpy()-comparison[view][typ])
                        row[typ+"_abs_diff_fsb_mean"] = float(np.abs(diff).mean())
                        images[typ+"_diff_fsb_x4"] = torch.from_numpy(np.clip(np.abs(diff)*4, 0, 1))
                for kind, tensor in images.items():
                    target = out / method / view / (kind+".png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    W._save_native_rgb(target, tensor)
                    manifest_rows.append(dict(scene=a.scene, method=method, view_id=view, kind=kind,
                                              path=str(target), sha256=W._sha256(target), checkpoint=str(ckpt)))
                del outputs, images, raw, pred, gt, accum, colors
            all_metrics.extend(method_rows)
            if method in ("FSB", candidate_method):
                degree_rows.extend(train_degree_diagnostics(branch, method))
            if method == "D1":
                W._write_json(out / "math_audit.json", math_audit(model))
            elif method == "CURR":
                _, _, camera, _ = W._train_records(branch)[0]
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    effective_alpha = float(outputs["headroom_alpha"])
                    model.config.headroom_schedule = "constant"
                    endpoint = model.get_outputs_for_camera(camera.to(model.device))
                    errors = {k: float((outputs[k]-endpoint[k]).abs().max()) for k in ("pred_image", "rgb_clear", "gaussian_view_rgb")}
                    model.config.headroom_schedule = W.HEADROOM_SCHEDULE
                assert effective_alpha == 1.0 and max(errors.values()) == 0.0
                W._write_json(out / "math_audit.json", dict(effective_alpha=effective_alpha,
                              final_fsb_endpoint_max_abs=errors, train_camera_only=True))
                del outputs, endpoint
            audit_rows.append(dict(method=method, checkpoint=str(ckpt), checkpoint_sha256=W._sha256(ckpt),
                                   manifest_sha256=W._sha256(path/"experiment_manifest.json"),
                                   model_hash=loaded_hash, reload_model_hash_match=True,
                                   parameterization=W.ACTIVE_PARAMETERIZATION, alpha=W.HEADROOM_ALPHA,
                                   degreewise=W.DEGREEWISE_HEADROOM_ALPHAS,
                                   headroom_schedule=W.HEADROOM_SCHEDULE,
                                   headroom_start_step=W.HEADROOM_START_STEP, headroom_end_step=W.HEADROOM_END_STEP,
                                   coefficient_source="experiment_manifest.locked_config",
                                   checkpoint_metadata_has_degree_weights=metadata_has_weights,
                                   original_metrics_match=True, evaluation_views=ids))
            print(a.scene, method, {m: float(np.mean([r[m] for r in method_rows])) for m in METRICS}, flush=True)
        finally:
            W._release(branch)
    W._write_csv(out / "per_view_metrics.csv", all_metrics)
    W._write_csv(out / "render_manifest.csv", manifest_rows)
    W._write_csv(out / "degree_diagnostics.csv", degree_rows)
    W._write_json(out / "reload_audit.json", audit_rows)
    W._write_json(out / "status.json", {"complete": True, "evaluation_views": reference_views, "methods": list(paths)})


if __name__ == "__main__":
    main()
