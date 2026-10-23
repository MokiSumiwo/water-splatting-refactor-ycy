#!/usr/bin/env python3
"""Strict paired TABA diagnosis for the two-scene BND ON/OFF checkpoints.

This is read-only: it loads checkpoints, renders training views, and writes
detached statistics.  ON and OFF use the same view IDs and deterministic pixel
indices; transmission buckets are defined from their per-pixel average.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_direct_mdrr_cica_scene as W
from water_splatting.rendering.medium_jacobian import analytic_medium_jacobian_actions

SCENES = ("JapaneseGradens-RedSea", "Panama")
STEPS = (5000, 10000, 14999)
BUCKETS = ("low", "middle", "high")
PIXEL_SAMPLE = 2048
SENSITIVITY_CHUNK = 2048


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")

_write_json = _write


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(x: Tensor) -> float:
    x = x.detach().float().reshape(-1)
    x = x[torch.isfinite(x)]
    return float(x.mean().item()) if x.numel() else float("nan")


def _std(x: Sequence[float]) -> float:
    vals = torch.tensor([v for v in x if math.isfinite(float(v))], dtype=torch.float64)
    return float(vals.std(unbiased=False).item()) if vals.numel() else float("nan")


def _corr(x: Tensor, y: Tensor) -> float:
    x, y = x.detach().float().reshape(-1), y.detach().float().reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if x.numel() < 3 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _pixels(height: int, width: int, scene: str, step: int, view_id: str) -> Tensor:
    total = int(height) * int(width)
    seed = int.from_bytes(hashlib.sha256(f"{scene}:{step}:{view_id}".encode()).digest()[:8], "little") % (2**63 - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if total <= PIXEL_SAMPLE:
        return torch.arange(total, dtype=torch.long)
    return torch.randperm(total, generator=g)[:PIXEL_SAMPLE].sort().values


def _flat(value: Tensor) -> Tensor:
    return value.detach().float().reshape(-1, *value.shape[2:]) if value.ndim >= 3 else value.detach().float().reshape(-1)


def _checkpoint(root: Path, scene: str, step: int, variant: str) -> Path:
    base = root if not scene else root / scene
    if variant == "on":
        candidates = [base / "checkpoints" / f"step-{step:09d}.ckpt", base / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"]
    else:
        candidates = [base / "checkpoints" / f"step-{step:09d}.ckpt", base / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {variant} checkpoint for {scene}@{step}: {candidates}")


def _load_branch(scene: str, gpu: str, source_root: Path, config: Path, variant: str) -> Any:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    W.ACTIVE_SOURCE_ROOT = source_root.resolve()
    W.ACTIVE_SOURCE_CONFIG = config.resolve()
    W.ACTIVE_PARAMETERIZATION = "bounded_sh3" if variant == "on" else "legacy"
    W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    W.OCMC_ENABLED = True
    W.CICA_ENABLED = True
    W.BRASH_ENABLED = False
    W.BVAM_ENABLED = False
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.TABA_DIAGNOSTIC_ENABLED = False
    return W._new_branch(scene)


def _load_checkpoint(branch: Any, path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = int(payload.get("absolute_step", 0))
    W._configure_model(branch.pipeline.model)
    from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
    FORMAL._install_condition(branch.pipeline.model, "C0", payload.get("ocmc_bundle"), None)
    W._configure_model(branch.pipeline.model)
    branch.pipeline.eval()
    return payload


def _gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return W._gt(model, W._batch_to_device(batch, model.device), background).detach().float().clamp(0.0, 1.0)


def _render(branch: Any, view: Tuple[int, str, Any, Mapping[str, Any]], scene: str, step: int) -> Dict[str, Any]:
    index, view_id, camera, batch = view
    model = branch.pipeline.model
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = _gt(model, batch, outputs["background"])
    pred = outputs["pred_image"].detach().float()
    obj = outputs["rgb_object"].detach().float()
    med = outputs["rgb_medium"].detach().float()
    clear = outputs["rgb_clear"].detach().float()
    t = outputs["transmission"].detach().float().mean(dim=-1)
    depth = outputs["depth"].detach().float().squeeze(-1)
    attn = outputs["medium_attn"].detach().float().mean(dim=-1)
    residual = pred - gt
    obj_res = (obj - gt).abs().mean(dim=-1)
    med_res = (med - (gt - obj)).abs().mean(dim=-1)
    clear_res = (clear - obj).abs().mean(dim=-1)
    luma = residual.mean(dim=-1)
    chroma = (residual - luma[..., None]).norm(dim=-1)
    j = outputs.get("J_gaussian")
    j_flat = j.detach().float().reshape(-1) if isinstance(j, Tensor) else torch.empty(0)
    clear_flat = clear.detach().float().reshape(-1, 3)
    tau_flat = outputs["tau_D"].detach().float().reshape(-1)
    h, w = int(t.shape[0]), int(t.shape[1])
    pixels = _pixels(h, w, scene, step, str(view_id))
    return {
        "scene": scene, "step": int(step), "camera_index": int(index), "view_id": str(view_id),
        "gt": gt.cpu(), "pred": pred.cpu(), "obj": obj.cpu(), "med": med.cpu(), "clear": clear.cpu(),
        "t": t.cpu(), "depth": depth.cpu(), "attn": attn.cpu(), "obj_res": obj_res.cpu(), "med_res": med_res.cpu(),
        "clear_res": clear_res.cpu(), "luma_res": luma.abs().cpu(), "chroma_res": chroma.cpu(), "photometric": residual.abs().mean(dim=-1).cpu(),
        "p_j_gt_1": float((j_flat > 1.0).float().mean().item()) if j_flat.numel() else float("nan"),
        "j_p99": float(torch.quantile(j_flat, 0.99).item()) if j_flat.numel() else float("nan"),
        "clear_min": float(clear_flat.min().item()) if clear_flat.numel() else float("nan"),
        "clear_max": float(clear_flat.max().item()) if clear_flat.numel() else float("nan"),
        "tau_p90": float(torch.quantile(tau_flat, 0.90).item()) if tau_flat.numel() else float("nan"),
        "pixels": pixels,
    }


def _project_geometry(model: Any, camera: Any, colors: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    camera = camera.to(model.device)
    down = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / down)
    try:
        R = camera.camera_to_worlds[0, :3, :3]
        T = camera.camera_to_worlds[0, :3, 3:4]
        R = R @ torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
        Ri = R.T
        viewmat = torch.eye(4, device=model.device, dtype=R.dtype)
        viewmat[:3, :3] = Ri
        viewmat[:3, 3:4] = -Ri @ T
        H, Wd = int(camera.height.item()), int(camera.width.item())
        return model.underwater_rasterizer.project(
            means=model.means.detach(), scales=model.scales.detach(), quats=model.quats.detach(), viewmat=viewmat,
            fx=camera.fx.item(), fy=camera.fy.item(), cx=camera.cx.item(), cy=camera.cy.item(), height=H, width=Wd,
            clip_thresh=model.config.clip_thresh,
        )[:6]
    finally:
        camera.rescale_output_resolution(down)


def _sensitivity(branch: Any, view: Tuple[int, str, Any, Mapping[str, Any]], pixels: Tensor) -> Tuple[Tensor, str]:
    model = branch.pipeline.model
    camera = view[2]
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera.to(model.device))
    t = outputs["transmission"].detach().float()
    H, Wd = int(t.shape[0]), int(t.shape[1])
    # OCMC exposes the effective raw decomposition on every C0 forward.
    if "camera_medium_raw_base" in outputs and "camera_medium_delta_projected_raw" in outputs:
        raw = (outputs["camera_medium_raw_base"] + outputs["camera_medium_delta_projected_raw"]).detach().float().reshape(-1, 9)
        source = "ocmc_effective_raw"
    elif "camera_medium_raw_unprojected" in outputs:
        raw = outputs["camera_medium_raw_unprojected"].detach().float().reshape(-1, 9)
        source = "medium_raw_unprojected_fallback"
    else:
        raise RuntimeError("checkpoint forward did not expose medium raw decomposition")
    colors = outputs["gaussian_view_rgb"].detach().float()
    opacities = torch.sigmoid(model.opacities.detach())
    xys, depths, radii, conics, _comp, num_tiles_hit = _project_geometry(model, camera, colors)
    directions = torch.eye(9, device=model.device, dtype=torch.float32)
    actions = analytic_medium_jacobian_actions(
        xys=xys, depths=depths, radii=radii, conics=conics, colors=colors, opacities=opacities,
        num_tiles_hit=num_tiles_hit, height=H, width=Wd, block_width=model.underwater_rasterizer.block_width,
        raw_medium=raw, raw_directions=directions, density_bias=float(model.medium_density_bias),
        chunk_size=SENSITIVITY_CHUNK, pixel_indices=pixels.to(model.device),
    )
    return actions.float().norm(dim=(1, 2)).detach().cpu(), source


def _bucket(t: Tensor) -> Dict[str, Tensor]:
    return {"low": t < 0.33, "middle": (t >= 0.33) & (t < 0.66), "high": t >= 0.66}


def _scene_summary(rows: Sequence[Mapping[str, Any]], scene: str, step: int) -> Dict[str, Any]:
    selected = [r for r in rows if r["scene"] == scene and int(r["step"]) == int(step)]
    out: Dict[str, Any] = {"scene": scene, "step": int(step), "view_count": len({r["view_id"] for r in selected})}
    for bucket in BUCKETS:
        part = [r for r in selected if r["bucket"] == bucket]
        out[f"{bucket}_view_count"] = len(part)
        out[f"{bucket}_pixel_count"] = int(sum(int(r["pixel_count"]) for r in part))
        for key in ("rgb_object_residual_on", "rgb_medium_residual_on", "clear_residual_on", "luminance_residual_on", "chromatic_residual_on", "chromatic_residual_delta_on_minus_off", "S_med", "mean_transmission"):
            vals = [float(r[key]) for r in part]
            finite = [v for v in vals if math.isfinite(v)]
            out[f"{bucket}_{key}_scene_mean"] = float(sum(finite) / len(finite)) if finite else float("nan")
            out[f"{bucket}_{key}_view_std"] = _std(vals)
    return out


def _support_mean(branch: Any, records: Mapping[str, Tuple[int, str, Any, Mapping[str, Any]]]) -> float:
    values: List[float] = []
    model = branch.pipeline.model
    for _view in records.values():
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(_view[2].to(model.device))
        visible = outputs.get("gaussian_visible_mask")
        if isinstance(visible, Tensor):
            values.append(float(visible.detach().float().sum().item()))
    return float(sum(values) / len(values)) if values else float("nan")


def run_scene(scene: str, gpu: str, on_root: Path, off_root: Path, config: Path, out_dir: Path) -> None:
    on_branch = off_branch = None
    all_rows: List[Dict[str, Any]] = []
    try:
        on_branch = _load_branch(scene, gpu, on_root, config, "on")
        off_branch = _load_branch(scene, gpu, off_root, config, "off")
        records_on = {str(v): (i, str(v), c, b) for i, v, c, b in W._train_records(on_branch)}
        records_off = {str(v): (i, str(v), c, b) for i, v, c, b in W._train_records(off_branch)}
        if set(records_on) != set(records_off):
            raise RuntimeError(f"train view mismatch for {scene}")
        available_steps = [step for step in STEPS if (on_root / "checkpoints" / f"step-{step:09d}.ckpt").is_file() and (off_root / "checkpoints" / f"step-{step:09d}.ckpt").is_file()]
        for step in available_steps:
            on_path, off_path = _checkpoint(on_root, "", step, "on"), _checkpoint(off_root, "", step, "off")
            _load_checkpoint(on_branch, on_path)
            _load_checkpoint(off_branch, off_path)
            support_on = _support_mean(on_branch, records_on)
            support_off = _support_mean(off_branch, records_off)
            for view_id in sorted(records_on):
                on = _render(on_branch, records_on[view_id], scene, step)
                off = _render(off_branch, records_off[view_id], scene, step)
                if on["pixels"].tolist() != off["pixels"].tolist():
                    raise RuntimeError("pixel pairing mismatch")
                sens, source = _sensitivity(on_branch, records_on[view_id], on["pixels"])
                common_t = (on["t"] + off["t"]) * 0.5
                for name, mask in _bucket(common_t).items():
                    if not bool(mask.any()):
                        continue
                    sample = on["pixels"]
                    sample_t = common_t.reshape(-1)[sample]
                    sm = sample_t < 0.33 if name == "low" else ((sample_t >= 0.33) & (sample_t < 0.66) if name == "middle" else sample_t >= 0.66)
                    sens_value = _mean(sens[sm]) if bool(sm.any()) else float("nan")
                    m = mask
                    all_rows.append({
                        "scene": scene, "step": int(step), "view_id": view_id, "camera_index": int(on["camera_index"]), "bucket": name,
                        "pixel_count": int(m.sum()), "sampled_pixel_count": int(sm.sum()), "mean_transmission": _mean(common_t[m]),
                        "rgb_object_residual_on": _mean(on["obj_res"][m]), "rgb_object_residual_off": _mean(off["obj_res"][m]), "rgb_object_residual_delta_on_minus_off": _mean((on["obj_res"] - off["obj_res"])[m]),
                        "rgb_medium_residual_on": _mean(on["med_res"][m]), "rgb_medium_residual_off": _mean(off["med_res"][m]), "rgb_medium_residual_delta_on_minus_off": _mean((on["med_res"] - off["med_res"])[m]),
                        "clear_residual_on": _mean(on["clear_res"][m]), "clear_residual_off": _mean(off["clear_res"][m]), "clear_residual_delta_on_minus_off": _mean((on["clear_res"] - off["clear_res"])[m]),
                        "luminance_residual_on": _mean(on["luma_res"][m]), "luminance_residual_off": _mean(off["luma_res"][m]), "luminance_residual_delta_on_minus_off": _mean((on["luma_res"] - off["luma_res"])[m]),
                        "chromatic_residual_on": _mean(on["chroma_res"][m]), "chromatic_residual_off": _mean(off["chroma_res"][m]), "chromatic_residual_delta_on_minus_off": _mean((on["chroma_res"] - off["chroma_res"])[m]),
                        "photometric_residual_on": _mean(on["photometric"][m]), "photometric_residual_off": _mean(off["photometric"][m]),
                        # This is the detached mean number of visible Gaussians
                        # per training view.  Pixel-level assignment to a
                        # Gaussian is renderer-dependent, so it is reported
                        # separately from per-Gaussian multiview support.
                        "visible_gaussian_count_mean_on": support_on, "visible_gaussian_count_mean_off": support_off,
                        "depth_variance_on": float(on["depth"][m].var().item()), "depth_variance_off": float(off["depth"][m].var().item()),
                        "medium_attenuation_on": _mean(on["attn"][m]), "medium_attenuation_off": _mean(off["attn"][m]),
                        "S_med": sens_value, "sensitivity_source": source,
                        "P_J_gt_1_on": on["p_j_gt_1"], "J_p99_on": on["j_p99"], "tau_D_p90_on": on["tau_p90"],
                        "clear_rgb_min_on": on["clear_min"], "clear_rgb_max_on": on["clear_max"],
                        "corr_t_chromatic_on": _corr(common_t[m], on["chroma_res"][m]), "corr_t_chromatic_delta_on_minus_off": _corr(common_t[m], (on["chroma_res"] - off["chroma_res"])[m]),
                        "corr_t_medium_on": _corr(common_t[m], on["med_res"][m]), "corr_t_object_on": _corr(common_t[m], on["obj_res"][m]),
                    })
        summaries = [_scene_summary(all_rows, scene, step) for step in available_steps]
        _write_json(out_dir / f"{scene}_missing_steps.json", {"requested_steps": list(STEPS), "available_steps": available_steps, "missing_steps": [s for s in STEPS if s not in available_steps]})
        _write_csv(out_dir / f"{scene}_paired_bucket_per_view.csv", all_rows)
        _write_json(out_dir / f"{scene}_paired_bucket_per_view.json", {"rows": all_rows})
        _write_json(out_dir / f"{scene}_paired_bucket_scene_summary.json", {"rows": summaries})
    finally:
        W._release(on_branch)
        W._release(off_branch)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--on-root", type=Path, required=True)
    p.add_argument("--off-root", type=Path, required=True)
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_scene(args.scene, args.gpu, args.on_root.resolve(), args.off_root.resolve(), args.source_config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
