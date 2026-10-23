#!/usr/bin/env python3
"""Real-checkpoint BND equivalence, TABA bounds and gradient routing audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.experiments.run_taba_bnd_on import (
    DEFAULT_OUT, GPUS, PAIRED, SCENES, check_environment, configure_worker, gpu_processes,
    read_json, source_assets, write_json,
)
from water_splatting.fields.gaussian_appearance import compute_bounded_gaussian_colors


def audit(scene, gpu, output, variants=("E1", "E2")):
    worker = configure_worker("E0")
    root, config, _ = source_assets(scene)
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG = root, config
    worker._runtime(scene, gpu)
    worker._seed_all(42)
    branch = worker._new_branch(scene)
    result = {"scene": scene, "gpu": gpu}
    try:
        payload = worker._load_source(branch, scene)
        model = branch.pipeline.model
        result["old_checkpoint_strict_load"] = worker._model_state_hash(model) == worker._hash_object(payload["model"])
        old_probe = read_json(PAIRED / scene / "bnd_on/start_state_equivalence.json")["forward_probe"]
        probe = worker._forward_probe(model, branch.pipeline.datamanager)
        result["historical_bnd_forward_hash_match"] = probe["tensor_sha256"] == old_probe["tensor_sha256"]
        result["historical_probe"], result["current_probe"] = old_probe, probe
        records = worker._train_records(branch)
        _, _, camera, batch = records[0]
        camera = camera.to(model.device)
        batch = worker._batch_to_device(batch.copy(), model.device)
        model.train()
        model.step = 5000
        original_groups = tuple(model.get_param_groups())

        def forward():
            model.zero_grad(set_to_none=True)
            outputs = model.get_outputs(camera)
            losses = model.get_loss_dict(outputs, batch, {})
            return outputs, losses

        def snapshot(outputs, losses):
            names = ("pred_image", "rgb_object", "rgb_medium", "rgb_clear", "depth", "J_gaussian_raw", "gaussian_view_rgb")
            return {key: worker._hash_tensor(outputs[key]) for key in names}, {key: float(value.detach()) for key, value in losses.items()}

        first, first_losses = forward()
        first_hashes, first_loss_values = snapshot(first, first_losses)
        del first, first_losses
        model.config.taba_transmission_mid = 0.2
        model.config.taba_transmission_temperature = 0.3
        model.config.taba_chroma_floor = 0.01
        with patch("water_splatting.water_splatting.sample_taba_transmission", side_effect=AssertionError("disabled TABA sampled transmission")):
            disabled, disabled_losses = forward()
        second_hashes, second_loss_values = snapshot(disabled, disabled_losses)
        result["disabled_forward_exact"] = first_hashes == second_hashes
        result["disabled_main_loss_exact"] = first_loss_values == second_loss_values
        del disabled, disabled_losses
        model.config.taba_enabled = True
        model.config.taba_mode = "disabled"
        with patch("water_splatting.water_splatting.sample_taba_transmission", side_effect=AssertionError("disabled mode sampled transmission")):
            disabled_mode, disabled_mode_losses = forward()
        mode_hashes, mode_loss_values = snapshot(disabled_mode, disabled_mode_losses)
        result["disabled_mode_forward_exact"] = mode_hashes == first_hashes
        result["disabled_mode_loss_exact"] = mode_loss_values == first_loss_values
        del disabled_mode, disabled_mode_losses
        model.config.taba_enabled = False
        model.config.taba_diagnostic_only = True
        diagnostic, diagnostic_losses = forward()
        diagnostic_hashes, diagnostic_loss_values = snapshot(diagnostic, diagnostic_losses)
        result["diagnostic_only_forward_exact"] = diagnostic_hashes == first_hashes
        result["diagnostic_only_main_loss_exact"] = diagnostic_loss_values == first_loss_values
        del diagnostic, diagnostic_losses

        result["variants"] = {}
        for variant in variants:
            configure_worker(variant)
            worker._configure_taba(model.config)
            outputs, losses = forward()
            raw = outputs["J_gaussian_raw"]
            intrinsic = outputs["gaussian_view_rgb"]
            bounded = compute_bounded_gaussian_colors(
                means=model.means, features_dc=model.features_dc, features_rest=model.features_rest,
                camera_position=camera.camera_to_worlds[..., :3, 3], sh_degree=3, active_sh_degree=3,
            )
            rgb_loss = outputs["gaussian_view_logits"].sigmoid().square().mean()
            params, indices = worker._parameter_entries(model)
            gradients = torch.autograd.grad(rgb_loss, params, retain_graph=True, allow_unused=True)
            norms = worker._group_norms(gradients, indices)
            appearance_ids = set(indices.get("features_dc", []) + indices.get("features_rest", []))
            appearance_escape = worker._nonzero_group_norm(gradients, appearance_ids)
            del gradients, rgb_loss
            photometric_grads = torch.autograd.grad(sum(losses.values()), params, allow_unused=True)
            photometric_norms = worker._group_norms(photometric_grads, indices)
            luma_difference = (outputs["gaussian_sh_residual"].mean(-1) - bounded.sh_residual.mean(-1)).abs()
            rounding_bound = 4 * torch.finfo(bounded.logits.dtype).eps * (
                bounded.logits.abs() + bounded.dc_logits.abs() + bounded.sh_residual.abs()
            ).amax(dim=-1).clamp_min(1.0)
            checks = {
                "strict_intrinsic_bounds": bool(((intrinsic > 0) & (intrinsic < 1)).all()),
                "P_J_gt_1_zero": not bool((raw > 1).any()),
                "P_intrinsic_rgb_gt_1_zero": not bool((intrinsic > 1).any()),
                "all_taba_statistics_detached": all(not value.requires_grad for name, value in outputs.items() if name.startswith("taba_")),
                "appearance_only_gradients": appearance_escape == 0,
                "appearance_dc_gradient": norms["features_dc"] > 0,
                "appearance_rest_gradient": norms["features_rest"] > 0,
                "photometric_dc_gradient": photometric_norms["features_dc"] > 0,
                "photometric_rest_gradient": photometric_norms["features_rest"] > 0,
                "finite_forward_and_gradients": worker.FORMAL._finite_outputs(outputs) and all(g is None or bool(torch.isfinite(g).all()) for g in photometric_grads),
                "optimizer_groups_unchanged": tuple(model.get_param_groups()) == original_groups,
                "dc_outputs_unchanged": torch.equal(outputs["gaussian_view_dc_rgb"], bounded.dc_rgb),
            }
            if model.config.taba_mode == "chromatic":
                checks["luminance_logits_unchanged"] = bool((luma_difference <= rounding_bound).all())
            else:
                delta = bounded.logits - bounded.dc_logits
                chroma = delta - delta.mean(-1, keepdim=True)
                axis = outputs["taba_dcu_water_axis"]
                parallel = (chroma * axis).sum(-1, keepdim=True) * axis
                allocation = torch.where(outputs["taba_dcu_fallback_mask"], chroma, parallel)
                expected = bounded.logits + (outputs["taba_g_chroma"] - 1) * allocation
                checks["directional_formula_exact"] = torch.equal(outputs["gaussian_view_logits"], expected)
            from water_splatting.taba import sample_taba_transmission
            proxy = sample_taba_transmission(outputs["medium_attn"], outputs["depth"], outputs["accumulation"], model.xys, model.radii)
            checks["transmission_prepass_matches_final_render"] = torch.equal(proxy["transmission_rgb"], outputs["taba_transmission_rgb"])
            result["variants"][variant] = {"checks": checks, "appearance_gradient_norms": norms, "photometric_gradient_norms": photometric_norms,
                                           "appearance_escape_l2": appearance_escape, "intrinsic_min": float(intrinsic.min()),
                                           "intrinsic_max": float(intrinsic.max()), "J_max": float(raw.detach().max()),
                                           "luminance_max_abs_roundoff": float(luma_difference.max()),
                                           "luminance_roundoff_bound": "4 * dtype_eps * max(1, abs(full) + abs(dc) + abs(residual))"}
            del outputs, losses, bounded, photometric_grads
            bank = worker._camera_bank(branch, output / variant)
            partner = worker._partner_mapping(branch, output / variant)
            result["variants"][variant]["routing"] = worker._gradient_routing_audit(branch, scene, "A2", output / variant, records, partner, bank)

        result["passed"] = all(result[key] for key in ("old_checkpoint_strict_load", "historical_bnd_forward_hash_match",
            "disabled_forward_exact", "disabled_main_loss_exact", "disabled_mode_forward_exact", "disabled_mode_loss_exact",
            "diagnostic_only_forward_exact", "diagnostic_only_main_loss_exact")) and all(
            all(value["checks"].values()) and value["routing"]["GRADIENT_ROUTING_AUDIT"] for value in result["variants"].values())
        write_json(output / "audit.json", result)
        if not result["passed"]:
            raise RuntimeError(f"TABA engineering audit failed: {output / 'audit.json'}")
        return result
    finally:
        worker._release(branch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--gpu", choices=GPUS, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT / "engineering")
    args = parser.parse_args()
    check_environment()
    if gpu_processes()[args.gpu]:
        raise RuntimeError(f"GPU {args.gpu} is occupied")
    result = audit(args.scene, args.gpu, args.output_root / args.scene)
    print(json.dumps({"passed": result["passed"], "scene": args.scene}), flush=True)


if __name__ == "__main__":
    main()
