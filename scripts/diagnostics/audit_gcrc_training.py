#!/usr/bin/env python3
"""Audit real-source GCRC disabled equality, history, bounds and gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.experiments.run_gcrc_bnd_on import DEFAULT_OUT, configure_worker
from scripts.experiments.run_taba_bnd_on import GPUS, PAIRED, SCENES, check_environment, gpu_processes, source_assets, read_json, write_json


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
        checks["old_checkpoint_disabled_exact"] = worker._model_state_hash(model) == worker._hash_object(payload["model"])
        historical = read_json(PAIRED / scene / "bnd_on/start_state_equivalence.json")["forward_probe"]
        checks["historical_disabled_forward_hash"] = worker._forward_probe(model, branch.pipeline.datamanager)["tensor_sha256"] == historical["tensor_sha256"]
        records = worker._train_records(branch)
        _, _, camera, batch = records[0]
        camera = camera.to(model.device)
        batch = worker._batch_to_device(batch.copy(), model.device)
        groups = tuple(model.get_param_groups())
        optimizer_hash = worker._optimizer_hash(branch.optimizers)

        def forward(camera=camera, step=5000):
            model.train()
            model.step_cb(step)
            model.zero_grad(set_to_none=True)
            out = model.get_outputs(camera)
            return out, model.get_loss_dict(out, batch, {})

        def snap(out, losses):
            keys = ("pred_image", "rgb_object", "rgb_medium", "rgb_clear", "depth", "J_gaussian_raw", "gaussian_view_rgb")
            return ({k:worker._hash_tensor(out[k]) for k in keys}, {k:float(v.detach()) for k,v in losses.items()})

        base, losses = forward()
        baseline = snap(base, losses)
        del base, losses
        model.config.gcrc_enabled = True
        with patch.object(model, "_ensure_gcrc_state", side_effect=AssertionError("disabled path accessed GCRC")):
            out, losses = forward()
        checks["explicit_disabled_forward_loss_exact"] = snap(out, losses) == baseline
        checks["disabled_optimizer_exact"] = worker._optimizer_hash(branch.optimizers) == optimizer_hash
        del out, losses

        for arm in ("E1", "E2", "E3"):
            configure_worker(arm)
            worker._configure_taba(model.config)
            worker._load_source(branch, scene)
            state = model._gcrc_state
            vc = {"optimizer_exact": worker._optimizer_hash(branch.optimizers) == optimizer_hash,
                  "old_checkpoint_parameters_exact": worker._hash_object({k:v for k,v in model.state_dict().items() if not k.startswith("_gcrc_state.")}) == worker._hash_object(payload["model"])}
            if state is not None:
                vc["old_history_zero"] = not bool(state.visibility_count.any() | state.water_direction_initialized.any() | state.residual_direction_initialized.any())
                model.config.gcrc_diagnostic_only = True
                out, losses = forward()
                vc["diagnostic_forward_loss_exact"] = snap(out, losses) == baseline
                vc["diagnostic_history_unchanged"] = not bool(state.visibility_count.any())
                del out, losses
                model.config.gcrc_diagnostic_only = False
                if arm == "E2":
                    out, losses = forward()
                    vc["cold_forward_loss_exact"] = snap(out, losses) == baseline
                    vc["cold_alpha_zero"] = not bool(out["gcrc_alpha"].any())
                    del out, losses
                # Exercise the same queued observation and after_train used by training.
                for step in range(3001, 3013):
                    _, _, view, _ = records[(step-3001) % len(records)]
                    out, losses = forward(view.to(model.device), step=step)
                    del out, losses
                    model.xys_grad_abs.zero_()
                    model.after_train(step)
                path = output / arm / "state_roundtrip.ckpt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
                saved = worker._hash_object(state.state_dict())
                model.load_state_dict(torch.load(path, map_location=model.device, weights_only=False), strict=True)
                state = model._gcrc_state
                vc["history_roundtrip_exact"] = worker._hash_object(state.state_dict()) == saved
                corrupted = model.state_dict()
                corrupted.pop("_gcrc_state.visibility_count")
                try:
                    model.load_state_dict(corrupted, strict=True)
                    vc["partial_state_rejected"] = False
                except RuntimeError as error:
                    vc["partial_state_rejected"] = "incomplete GCRC" in str(error)
                model.load_state_dict(torch.load(path, map_location=model.device, weights_only=False), strict=True)
                state = model._gcrc_state
                model.eval()
                with torch.no_grad():
                    out = model.get_outputs(camera)
                vc["eval_history_frozen"] = worker._hash_object(state.state_dict()) == saved
                del out
                saved = worker._hash_object(state.state_dict())
            out, losses = forward()
            rgb, j = out["gaussian_view_rgb"], out["J_gaussian_raw"]
            vc["intrinsic_closed_bounds"] = bool(((rgb >= 0) & (rgb <= 1)).all())
            vc["raw_J_bounds"] = bool(((j >= 0) & (j <= 1)).all())
            vc["statistics_detached"] = all(not v.requires_grad for k,v in out.items() if k.startswith(("gcrc_", "taba_")))
            vc["optimizer_groups_unchanged"] = tuple(model.get_param_groups()) == groups
            params, indices = worker._parameter_entries(model)
            gradients = torch.autograd.grad(out["gaussian_view_logits"].sigmoid().square().mean(), params, retain_graph=True, allow_unused=True)
            norms = worker._group_norms(gradients, indices)
            ids = set(indices.get("features_dc", []) + indices.get("features_rest", []))
            vc["appearance_only_gradients"] = worker._nonzero_group_norm(gradients, ids) == 0
            vc["dc_rest_gradients"] = norms["features_dc"] > 0 and norms["features_rest"] > 0
            photo = torch.autograd.grad(sum(losses.values()), params, retain_graph=True, allow_unused=True)
            vc["finite_photometric_gradients"] = all(v is None or bool(torch.isfinite(v).all()) for v in photo)
            del photo, gradients
            if state is not None:
                vc["forward_history_frozen"] = worker._hash_object(state.state_dict()) == saved
                vc["warm_correction_active"] = bool(out["gcrc_corrected"].any())
                corrected_rgb = rgb[out["gcrc_corrected"].reshape(-1)]
                vc["corrected_intrinsic_strict_bounds"] = bool(((corrected_rgb > 0) & (corrected_rgb < 1)).all())
                vc["alpha_bounds"] = bool(((out["gcrc_alpha"] >= 0) & (out["gcrc_alpha"] <= .5)).all())
                before = state.visibility_count.clone()
                visible = out["gaussian_visible_mask"].reshape(-1, 1)
                sum(losses.values()).backward()
                vc["backward_history_frozen"] = worker._hash_object(state.state_dict()) == saved
                model.after_train(5000)
                vc["callback_count_once"] = torch.equal(state.visibility_count, before+visible.long())
                saved_after = worker._hash_object(state.state_dict())
                model.after_train(5000)
                vc["callback_no_double_update"] = worker._hash_object(state.state_dict()) == saved_after
            else:
                vc["taba_directional"] = model.config.taba_mode == "directional" and "taba_dcu_water_axis" in out
            result["variants"][arm] = {"checks":vc, "gradient_norms":norms,
                "intrinsic_min":float(rgb.min()), "intrinsic_max":float(rgb.max()), "J_max":float(j.max())}
            del out, losses, rgb, j
            worker._load_source(branch, scene)
            bank = worker._camera_bank(branch, output / arm)
            partner = worker._partner_mapping(branch, output / arm)
            result["variants"][arm]["routing"] = worker._gradient_routing_audit(branch, scene, "A2", output / arm, records, partner, bank)
        # The new mode selector must reproduce the retained GRAC implementation.
        configure_worker("E0")
        worker._configure_taba(model.config)
        model.config.grac_enabled, model.config.grac_mode = True, "grac"
        worker._load_source(branch, scene)
        out, losses = forward()
        legacy = snap(out, losses)
        del out, losses
        model.config.grac_enabled, model.config.grac_mode = False, "disabled"
        model.config.gcrc_mode = "grac"
        worker._load_source(branch, scene)
        out, losses = forward()
        checks["retained_grac_mode_exact"] = snap(out, losses) == legacy and model._grac_state is not None and model._gcrc_state is None
        del out, losses
        result["passed"] = all(checks.values()) and all(all(v["checks"].values()) and v["routing"]["GRADIENT_ROUTING_AUDIT"] for v in result["variants"].values())
        write_json(output / "audit.json", result)
        if not result["passed"]:
            raise RuntimeError("GCRC engineering audit failed: " + str(output / "audit.json"))
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
    print({"scene":args.scene, "passed":result["passed"]}, flush=True)


if __name__ == "__main__":
    main()
