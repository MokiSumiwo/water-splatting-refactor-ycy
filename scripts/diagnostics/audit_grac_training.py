#!/usr/bin/env python3
"""Real-source GRAC equivalence, state lifecycle, bounds and gradient audit."""

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

from scripts.experiments.run_grac_bnd_on import DEFAULT_OUT, configure_worker
from scripts.experiments.run_taba_bnd_on import GPUS, PAIRED, SCENES, check_environment, gpu_processes, source_assets, read_json, write_json
from water_splatting.fields.gaussian_appearance import compute_bounded_gaussian_colors
from water_splatting.grac import apply_grac


def audit(scene, gpu, output):
    worker = configure_worker("E0")
    root, config, _ = source_assets(scene)
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG = root, config
    worker._runtime(scene, gpu)
    worker._seed_all(42)
    branch = worker._new_branch(scene)
    result = {"scene": scene, "gpu": gpu, "checks": {}, "variants": {}}
    checks = result["checks"]
    try:
        payload = worker._load_source(branch, scene)
        model = branch.pipeline.model
        checks["old_checkpoint_disabled_strict_load"] = worker._model_state_hash(model) == worker._hash_object(payload["model"])
        historical = read_json(PAIRED / scene / "bnd_on/start_state_equivalence.json")["forward_probe"]
        checks["historical_disabled_forward_hash"] = worker._forward_probe(model, branch.pipeline.datamanager)["tensor_sha256"] == historical["tensor_sha256"]
        records = worker._train_records(branch)
        _, _, camera, batch = records[0]
        camera = camera.to(model.device)
        batch = worker._batch_to_device(batch.copy(), model.device)
        groups = tuple(model.get_param_groups())
        original_optimizer = worker._optimizer_hash(branch.optimizers)

        def forward():
            model.train()
            model.step_cb(5000)
            model.zero_grad(set_to_none=True)
            out = model.get_outputs(camera)
            return out, model.get_loss_dict(out, batch, {})

        def snap(out, losses):
            keys = ("pred_image", "rgb_object", "rgb_medium", "rgb_clear", "depth", "J_gaussian_raw", "gaussian_view_rgb")
            return ({k:worker._hash_tensor(out[k]) for k in keys}, {k:float(v.detach()) for k,v in losses.items()})

        base, losses = forward()
        baseline = snap(base, losses)
        del base, losses
        model.config.grac_enabled = True
        model.config.grac_mode = "disabled"
        with patch.object(model, "_ensure_grac_state", side_effect=AssertionError("disabled path accessed GRAC history")):
            out, losses = forward()
        checks["explicit_disabled_forward_and_loss_exact"] = snap(out, losses) == baseline
        checks["disabled_optimizer_exact"] = worker._optimizer_hash(branch.optimizers) == original_optimizer
        del out, losses

        for arm in ("E1", "E2", "E3"):
            configure_worker(arm)
            worker._configure_taba(model.config)
            worker._load_source(branch, scene)
            state = model._grac_state
            vc = {}
            stripped = {k:v for k,v in model.state_dict().items() if not k.startswith("_grac_state.")}
            vc["old_checkpoint_parameters_exact"] = worker._hash_object(stripped) == worker._hash_object(payload["model"])
            vc["optimizer_exact"] = worker._optimizer_hash(branch.optimizers) == original_optimizer
            if state is not None:
                vc["old_checkpoint_history_zero"] = bool((state.visibility_count == 0).all() & ~state.reliability_initialized_mask.any())
                model.config.grac_diagnostic_only = True
                out, losses = forward()
                vc["diagnostic_forward_loss_exact"] = snap(out, losses) == baseline
                vc["diagnostic_forward_does_not_update_history"] = not bool(state.visibility_count.any())
                del out, losses
                model.config.grac_diagnostic_only = False
                model.eval()
                for step in range(3001, 3009):
                    _, _, view, _ = records[(step-3001) % len(records)]
                    with torch.no_grad():
                        out = model.get_outputs(view.to(model.device))
                        raw = out.get("taba_ungated_sh_residual")
                        if raw is None:
                            color = compute_bounded_gaussian_colors(means=model.means, features_dc=model.features_dc,
                                features_rest=model.features_rest, camera_position=view.camera_to_worlds[..., :3, 3].to(model.device),
                                sh_degree=3, active_sh_degree=3)
                            raw = color.sh_residual
                        chroma = raw - raw.mean(-1, keepdim=True)
                        state.update(chroma, out["gaussian_visible_mask"], step=step)
                    del out
                path = output / arm / "state_roundtrip.ckpt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
                saved = worker._hash_object(state.state_dict())
                model.load_state_dict(torch.load(path, map_location=model.device, weights_only=False), strict=True)
                state = model._grac_state
                vc["history_checkpoint_roundtrip_exact"] = worker._hash_object(state.state_dict()) == saved
                model.eval()
                with torch.no_grad():
                    eval_out = model.get_outputs(camera)
                vc["eval_history_unchanged"] = worker._hash_object(state.state_dict()) == saved
                del eval_out
            history_before_forward = worker._hash_object(state.state_dict()) if state is not None else None
            out, losses = forward()
            if state is not None:
                vc["forward_history_unchanged"] = worker._hash_object(state.state_dict()) == history_before_forward
            raw_j, rgb = out["J_gaussian_raw"], out["gaussian_view_rgb"]
            vc["raw_J_nonnegative_bounded"] = bool(((raw_j >= 0) & (raw_j <= 1)).all())
            vc["strict_intrinsic_bounds"] = bool(((rgb > 0) & (rgb < 1)).all())
            vc["all_statistics_detached"] = all(not v.requires_grad for k,v in out.items() if k.startswith(("grac_", "taba_")))
            vc["optimizer_groups_unchanged"] = tuple(model.get_param_groups()) == groups
            vc["finite_outputs"] = worker.FORMAL._finite_outputs(out)
            params, indices = worker._parameter_entries(model)
            grad = torch.autograd.grad(out["gaussian_view_logits"].sigmoid().square().mean(), params, retain_graph=True, allow_unused=True)
            norms = worker._group_norms(grad, indices)
            ids = set(indices.get("features_dc", []) + indices.get("features_rest", []))
            vc["appearance_only_gradients"] = worker._nonzero_group_norm(grad, ids) == 0
            vc["dc_and_rest_receive_gradient"] = norms["features_dc"] > 0 and norms["features_rest"] > 0
            photo_grad = torch.autograd.grad(sum(losses.values()), params, retain_graph=True, allow_unused=True)
            vc["finite_photometric_gradients"] = all(v is None or bool(torch.isfinite(v).all()) for v in photo_grad)
            del photo_grad, grad
            if state is not None:
                vc["warmed_gate_active"] = bool(out["grac_active"].any()) and bool((out["grac_gate"] < 1).any())
                before = state.visibility_count.clone()
                visible = out["gaussian_visible_mask"].reshape(-1,1)
                sum(losses.values()).backward()
                vc["backward_history_unchanged"] = worker._hash_object(state.state_dict()) == history_before_forward
                model.after_train(5000)
                vc["after_train_visibility_once"] = torch.equal(state.visibility_count, before + visible.long())
                count_after = state.visibility_count.clone()
                model.after_train(5000)
                vc["repeated_after_train_no_double_count"] = torch.equal(state.visibility_count, count_after)
                state.check_shape(model.num_points)
                if arm == "E1":
                    color = compute_bounded_gaussian_colors(means=model.means, features_dc=model.features_dc,
                        features_rest=model.features_rest, camera_position=camera.camera_to_worlds[..., :3, 3], sh_degree=3, active_sh_degree=3)
                    difference = (out["gaussian_sh_residual"].mean(-1)-color.sh_residual.mean(-1)).abs()
                    vc["luminance_preserved_roundoff"] = bool((difference <= 4e-6).all())
            else:
                vc["taba_mode_uses_directional"] = model.config.taba_mode == "directional" and "taba_dcu_water_axis" in out
            result["variants"][arm] = {"checks":vc, "gradient_norms":norms,
                "intrinsic_min":float(rgb.min()), "intrinsic_max":float(rgb.max()), "J_max":float(raw_j.max())}
            del out, losses, rgb, raw_j
            # Rebuild source optimizers after the standalone model roundtrip.
            worker._load_source(branch, scene)
            bank = worker._camera_bank(branch, output / arm)
            partner = worker._partner_mapping(branch, output / arm)
            routing = worker._gradient_routing_audit(branch, scene, "A2", output / arm, records, partner, bank)
            result["variants"][arm]["routing"] = routing
        result["passed"] = all(checks.values()) and all(all(v["checks"].values()) and v["routing"]["GRADIENT_ROUTING_AUDIT"] for v in result["variants"].values())
        write_json(output / "audit.json", result)
        if not result["passed"]:
            raise RuntimeError("GRAC engineering audit failed: " + str(output / "audit.json"))
        return result
    finally:
        worker._release(branch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--gpu", choices=GPUS, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT / "engineering/verified")
    args = parser.parse_args()
    check_environment()
    if gpu_processes()[args.gpu]:
        raise RuntimeError("assigned GPU is occupied")
    result = audit(args.scene, args.gpu, args.output_root / args.scene)
    print(json.dumps({"scene":args.scene,"passed":result["passed"]}), flush=True)


if __name__ == "__main__":
    main()
