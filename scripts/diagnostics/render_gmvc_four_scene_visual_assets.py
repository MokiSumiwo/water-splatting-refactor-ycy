#!/usr/bin/env python
"""Export four-scene GMVC visual-audit assets without changing training state."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


RUNS = ("A0", "P30-MHOLD")
SCENE_ORDER = ("Curasao", "JapaneseGradens", "IUI3", "Panama")
RGB_VALUE_RANGE = "[0,1]"
PNG_NORMALIZATION = "clamp_to_[0,1]_then_uint8"
NO_TONE_MAPPING = "none; linear tensor values are clamped to [0,1] for PNG save"
COLOR_SPACE = "renderer/dataset RGB tensor space; saved as PNG RGB without color conversion"
RESIDUAL_DIFF_SCALE = 4.0


@dataclass(frozen=True)
class RunSpec:
    label: str
    config: Path
    step: int = 15000


@dataclass(frozen=True)
class SceneSpec:
    name: str
    a0_config: Path
    p30_config: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scene_specs(repo: Path) -> Dict[str, SceneSpec]:
    return {
        "Curasao": SceneSpec(
            name="Curasao",
            a0_config=repo
            / "outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml",
            p30_config=repo
            / "outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml",
        ),
        "JapaneseGradens": SceneSpec(
            name="JapaneseGradens",
            a0_config=repo
            / "outputs/gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_a0/config.yml",
            p30_config=repo
            / "outputs/gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_mhold/config.yml",
        ),
        "IUI3": SceneSpec(
            name="IUI3",
            a0_config=repo
            / "outputs/gmvc_v3_four_scene_a0_iui3_redsea_seed42_step10000_to_15000/water-splatting/"
            "gmvc_v3_four_scene_a0_iui3_redsea_seed42_step10000_to_15000_20260806_gmvc_four_scene_p30_mhold_15k_a0/config.yml",
            p30_config=repo
            / "outputs/gmvc_v3_four_scene_p30_mhold_iui3_redsea_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_four_scene_p30_mhold_iui3_redsea_seed42_step13000_to_15000_20260806_gmvc_four_scene_p30_mhold_15k_mhold/config.yml",
        ),
        "Panama": SceneSpec(
            name="Panama",
            a0_config=repo
            / "outputs/gmvc_v3_four_scene_a0_panama_seed42_step10000_to_15000/water-splatting/"
            "gmvc_v3_four_scene_a0_panama_seed42_step10000_to_15000_20260806_gmvc_four_scene_p30_mhold_15k_a0/config.yml",
            p30_config=repo
            / "outputs/gmvc_v3_four_scene_p30_mhold_panama_seed42_step13000_to_15000/water-splatting/"
            "gmvc_v3_four_scene_p30_mhold_panama_seed42_step13000_to_15000_20260806_gmvc_four_scene_p30_mhold_15k_mhold/config.yml",
        ),
    }


COMPONENT_DEFINITIONS: Dict[str, str] = {
    "transmission": "exp(-(medium_attn * depth).clamp_min(0)); computed from renderer outputs at the eval camera.",
    "attenuation": "outputs['medium_attn']; per-pixel direct-signal attenuation coefficient from the medium field.",
    "backscatter": "outputs['b_inf'] * (1 - exp(-(medium_bs * depth).clamp_min(0))); diagnostic endpoint backscatter term.",
    "B_inf": "outputs['b_inf']; renderer asymptotic medium color output. These configs use b_inf_mode='tied'.",
    "medium_rgb": "outputs['medium_rgb']; per-pixel medium/background color predicted by the medium field.",
    "direct_object_signal": "outputs['rgb_object']; rasterizer direct object contribution in the underwater render.",
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_branch(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    except Exception:
        return "unknown"


def _setup_pipeline(config_path: Path, step: int, test_mode: str) -> Tuple[Any, Any, Path, int]:
    def _update_config(config: Any) -> Any:
        config.load_step = int(step)
        return config

    return eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode=test_mode,
        update_config_callback=_update_config,
    )


def _to_hwc(value: Tensor, key: str) -> Tensor:
    out = value.detach().float()
    if out.ndim == 2:
        out = out[..., None]
    if out.ndim != 3:
        raise ValueError(f"{key} must be HxW or HxWxC, got {tuple(out.shape)}")
    if out.shape[-1] in (1, 3):
        return out
    return out.mean(dim=-1, keepdim=True)


def _to_rgb(value: Tensor, *, clamp: bool = True) -> Tensor:
    image = _to_hwc(value, "image")
    if clamp:
        image = image.clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    return image


def _uint8(image: Tensor) -> np.ndarray:
    arr = _to_rgb(image).detach().cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, image: Tensor) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _uint8(image)
    Image.fromarray(arr).save(path)
    height, width = int(arr.shape[0]), int(arr.shape[1])
    return width, height


def _stats(value: Tensor) -> Dict[str, Any]:
    data = value.detach().float().cpu()
    finite = torch.isfinite(data)
    finite_values = data[finite]
    if finite_values.numel() == 0:
        return {
            "shape": list(data.shape),
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "shape": list(data.shape),
        "finite": bool(finite.all().item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
    }


def _range_string(value: Tensor) -> str:
    stats = _stats(value)
    if stats["min"] is None:
        return "unavailable"
    return f"[{stats['min']:.8g},{stats['max']:.8g}]"


def _image_name(pipeline: Any, image_idx: int) -> str:
    dataset = pipeline.datamanager.eval_dataset
    try:
        filenames = dataset._dataparser_outputs.image_filenames
        return Path(filenames[int(image_idx)]).name
    except Exception:
        return f"eval_{int(image_idx):04d}"


def _camera_id(camera: Any, image_idx: int) -> int:
    if camera.metadata is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(image_idx)


def _camera_signature(camera: Any) -> Dict[str, Any]:
    return {
        "height": int(camera.height.item()),
        "width": int(camera.width.item()),
        "fx": float(camera.fx.item()),
        "fy": float(camera.fy.item()),
        "cx": float(camera.cx.item()),
        "cy": float(camera.cy.item()),
        "camera_to_worlds": camera.camera_to_worlds.detach().cpu().reshape(-1).tolist(),
    }


def _same_camera(lhs: Mapping[str, Any], rhs: Mapping[str, Any], atol: float = 1e-6) -> bool:
    scalar_keys = ("height", "width", "fx", "fy", "cx", "cy")
    for key in scalar_keys:
        if abs(float(lhs[key]) - float(rhs[key])) > atol:
            return False
    left = torch.tensor(lhs["camera_to_worlds"], dtype=torch.float64)
    right = torch.tensor(rhs["camera_to_worlds"], dtype=torch.float64)
    return bool(torch.allclose(left, right, atol=atol, rtol=0.0))


def _force_dc_proxy_context(model: Any) -> Dict[str, Any]:
    attrs = [
        "gmvc_enabled",
        "gmvc_v3_enabled",
        "lambda_gmvc_object",
        "gmvc_start_step",
        "gmvc_stop_step",
        "gmvc_v3_object_source",
        "gmvc_intrinsic_source",
        "gmvc_intrinsic_use_dc_proxy",
    ]
    saved = {name: getattr(model.config, name, None) for name in attrs}
    saved["_training"] = bool(model.training)
    saved["_step"] = int(getattr(model, "step", 0))
    model.train()
    model.step = max(10001, int(getattr(model, "step", 10001)))
    model.config.gmvc_enabled = True
    model.config.gmvc_v3_enabled = True
    model.config.lambda_gmvc_object = 1.0
    model.config.gmvc_start_step = 0
    model.config.gmvc_stop_step = 10**9
    model.config.gmvc_v3_object_source = "J_proxy_raw"
    model.config.gmvc_intrinsic_source = "J_proxy_raw"
    model.config.gmvc_intrinsic_use_dc_proxy = True
    return saved


def _restore_context(model: Any, saved: Dict[str, Any]) -> None:
    was_training = bool(saved.pop("_training"))
    old_step = int(saved.pop("_step"))
    for name, value in saved.items():
        setattr(model.config, name, value)
    model.step = old_step
    model.train(was_training)


def _component_payload(outputs: Mapping[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
    unavailable: Dict[str, str] = {}
    components: Dict[str, Tensor] = {}
    required = ("depth", "medium_attn", "medium_bs", "medium_rgb", "rgb_object")
    missing = [key for key in required if key not in outputs]
    if missing:
        for key in COMPONENT_DEFINITIONS:
            unavailable[key] = f"missing renderer keys: {','.join(missing)}"
        return components, unavailable

    depth = _to_hwc(outputs["depth"], "depth")
    if depth.shape[-1] != 1:
        depth = depth.mean(dim=-1, keepdim=True)
    medium_attn = _to_hwc(outputs["medium_attn"], "medium_attn")
    medium_bs = _to_hwc(outputs["medium_bs"], "medium_bs")
    medium_rgb = _to_hwc(outputs["medium_rgb"], "medium_rgb")
    rgb_object = _to_hwc(outputs["rgb_object"], "rgb_object")
    if "b_inf" not in outputs:
        components.update(
            {
                "transmission": torch.exp(-(medium_attn * depth).clamp_min(0.0)),
                "attenuation": medium_attn,
                "medium_rgb": medium_rgb,
                "direct_object_signal": rgb_object,
            }
        )
        unavailable["B_inf"] = "outputs['b_inf'] is unavailable"
        unavailable["backscatter"] = "requires outputs['b_inf']"
        return {key: value.detach().float().cpu() for key, value in components.items()}, unavailable

    b_inf = _to_hwc(outputs["b_inf"], "b_inf")
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    components = {
        "transmission": transmission,
        "attenuation": medium_attn,
        "backscatter": backscatter,
        "B_inf": b_inf,
        "medium_rgb": medium_rgb,
        "direct_object_signal": rgb_object,
    }
    return {key: value.detach().float().cpu() for key, value in components.items()}, unavailable


def _dewatered_proxy(original_outputs: Mapping[str, Tensor], proxy_outputs: Mapping[str, Tensor]) -> Tuple[Tensor, str]:
    for source, outputs in (("J_proxy_raw", proxy_outputs), ("J_gaussian_raw", original_outputs), ("J_raw", original_outputs)):
        if source in outputs:
            return _to_hwc(outputs[source], source).detach().float().cpu(), source
    raise KeyError("No dewatered proxy source found: J_proxy_raw, J_gaussian_raw, and J_raw are absent")


def _render_checkpoint(run: RunSpec, test_mode: str, max_images: int) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(run.config, run.step, test_mode)
    model = pipeline.model
    model.eval()
    active_sh_degree = int(model._get_active_sh_degree()) if hasattr(model, "_get_active_sh_degree") else None
    configured_sh_degree = int(getattr(model.config, "sh_degree", -1))
    rendered: List[Dict[str, Any]] = []
    data_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    if max_images > 0:
        data_loader = data_loader[: int(max_images)]

    with torch.no_grad():
        for view_id, (camera, batch) in enumerate(data_loader):
            outputs = model.get_outputs_for_camera(camera=camera)
            _assert_finite_outputs(outputs, ("pred_image", "depth", "medium_attn", "medium_bs", "medium_rgb", "rgb_object"))
            _, images = model.get_image_metrics_and_images(outputs, batch)
            pred = _to_hwc(outputs["pred_image"], "pred_image").detach().float().clamp(0.0, 1.0).cpu()
            gt = _to_hwc(images["gt"], "gt").detach().float().clamp(0.0, 1.0).cpu()
            image_idx_raw = batch.get("image_idx", view_id)
            image_idx = int(image_idx_raw.item() if torch.is_tensor(image_idx_raw) else image_idx_raw)
            camera_sig = _camera_signature(camera)
            camera_id = _camera_id(camera, image_idx)

            saved = _force_dc_proxy_context(model)
            try:
                proxy_outputs = model.get_outputs_for_camera(camera=camera)
            finally:
                _restore_context(model, saved)
            components, unavailable = _component_payload(outputs)
            dewatered, dewatered_source = _dewatered_proxy(outputs, proxy_outputs)
            rendered.append(
                {
                    "view_id": int(view_id),
                    "image_idx": int(image_idx),
                    "image_name": _image_name(pipeline, image_idx),
                    "camera_id": int(camera_id),
                    "camera_signature": camera_sig,
                    "width": int(pred.shape[1]),
                    "height": int(pred.shape[0]),
                    "gt": gt,
                    "underwater": pred,
                    "dewatered": dewatered,
                    "dewatered_source": dewatered_source,
                    "components": components,
                    "unavailable_components": unavailable,
                }
            )

    result = {
        "label": run.label,
        "config": str(run.config),
        "requested_step": int(run.step),
        "loaded_step": int(loaded_step),
        "checkpoint": str(checkpoint_path),
        "experiment_name": getattr(config, "experiment_name", ""),
        "active_sh_degree": active_sh_degree,
        "configured_sh_degree": configured_sh_degree,
        "rendered": rendered,
    }
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _assert_finite_outputs(outputs: Mapping[str, Tensor], keys: Sequence[str]) -> None:
    for key in keys:
        if key not in outputs:
            raise KeyError(f"renderer output missing required key: {key}")
        value = outputs[key]
        if not torch.isfinite(value.detach().float()).all():
            raise ValueError(f"renderer output contains non-finite values: {key}")


def _view_selection(view_ids: Sequence[int]) -> List[int]:
    ids = list(view_ids)
    if len(ids) <= 3:
        return ids
    return [ids[0], ids[len(ids) // 2], ids[-1]]


def _label_tile(path: Path, label: str, tile_width: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if tile_width > 0 and image.width > tile_width:
        height = max(1, int(round(image.height * tile_width / image.width)))
        image = image.resize((tile_width, height), Image.Resampling.BILINEAR)
    pad = 22
    canvas = Image.new("RGB", (image.width, image.height + pad), (255, 255, 255))
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 4), label, fill=(0, 0, 0))
    return canvas


def _write_sheet(path: Path, rows: List[List[Tuple[str, Path]]], tile_width: int) -> Tuple[int, int]:
    if not rows:
        raise ValueError(f"No rows for contact sheet {path}")
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_label_tile(tile_path, label, tile_width) for label, tile_path in row]
        width = sum(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        row_img = Image.new("RGB", (width, height), (255, 255, 255))
        x = 0
        for tile in tiles:
            row_img.paste(tile, (x, 0))
            x += tile.width
        rendered_rows.append(row_img)
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return sheet_width, sheet_height


def _manifest_row(
    *,
    scene: str,
    view_id: str,
    camera_id: str,
    run: str,
    checkpoint_step: Any,
    image_type: str,
    component: str,
    path: Path,
    width: int,
    height: int,
    value_range: str,
    normalization: str,
    source_checkpoint: str,
    source_config: str,
    tone_mapping: str = NO_TONE_MAPPING,
    color_space: str = COLOR_SPACE,
    source_tensor: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "scene": scene,
        "view_id": view_id,
        "camera_id": camera_id,
        "run": run,
        "checkpoint_step": checkpoint_step,
        "image_type": image_type,
        "component": component,
        "file_path": str(path),
        "width": int(width),
        "height": int(height),
        "value_range": value_range,
        "normalization": normalization,
        "tone_mapping": tone_mapping,
        "color_space": color_space,
        "source_checkpoint": source_checkpoint,
        "source_config": source_config,
        "source_tensor": source_tensor,
        "notes": notes,
    }


def _write_scene_outputs(
    scene: str,
    scene_output: Path,
    evaluated: Mapping[str, Dict[str, Any]],
    tile_width: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    a0 = evaluated["A0"]
    p30 = evaluated["P30-MHOLD"]
    view_ids = [int(item["view_id"]) for item in a0["rendered"]]
    selected = _view_selection(view_ids)
    per_view_paths: Dict[int, Dict[str, Path]] = {}
    unavailable: Dict[str, Dict[str, str]] = {}
    validation = {
        "camera_view_match": True,
        "resolution_match": True,
        "finite_tensors": True,
        "same_configured_sh_degree": a0["configured_sh_degree"] == p30["configured_sh_degree"],
        "same_active_sh_degree": a0["active_sh_degree"] == p30["active_sh_degree"],
        "same_requested_step": a0["requested_step"] == p30["requested_step"],
        "same_loaded_step": a0["loaded_step"] == p30["loaded_step"],
        "same_tone_mapping": True,
        "same_color_space": True,
        "same_crop": True,
    }

    for a0_view, p30_view in zip(a0["rendered"], p30["rendered"]):
        view_id = int(a0_view["view_id"])
        if view_id != int(p30_view["view_id"]):
            raise ValueError(f"{scene}: view id mismatch {view_id} vs {p30_view['view_id']}")
        if int(a0_view["image_idx"]) != int(p30_view["image_idx"]) or a0_view["image_name"] != p30_view["image_name"]:
            raise ValueError(f"{scene} view {view_id}: image index/name mismatch between A0 and P30-MHOLD")
        if not _same_camera(a0_view["camera_signature"], p30_view["camera_signature"]):
            raise ValueError(f"{scene} view {view_id}: camera signature mismatch")
        if tuple(a0_view["underwater"].shape) != tuple(p30_view["underwater"].shape):
            raise ValueError(f"{scene} view {view_id}: underwater render shape mismatch")
        if tuple(a0_view["dewatered"].shape) != tuple(p30_view["dewatered"].shape):
            raise ValueError(f"{scene} view {view_id}: dewatered render shape mismatch")

        camera_id = str(a0_view["camera_id"])
        width = int(a0_view["width"])
        height = int(a0_view["height"])
        uv_dir = scene_output / "underwater" / "per_view"
        dv_dir = scene_output / "dewatered" / "per_view"
        mv_dir = scene_output / "medium" / "per_view"
        view_tag = f"view_{view_id:04d}"
        paths: Dict[str, Path] = {}
        gt = a0_view["gt"]
        a0_under = a0_view["underwater"]
        p30_under = p30_view["underwater"]
        a0_residual = (a0_under - gt).abs()
        p30_residual = (p30_under - gt).abs()
        residual_delta = (
            0.5
            + RESIDUAL_DIFF_SCALE
            * (p30_residual.mean(dim=-1, keepdim=True) - a0_residual.mean(dim=-1, keepdim=True))
        ).clamp(0.0, 1.0)

        underwater_images = [
            ("gt_underwater", "GT", "gt_underwater", gt, "images['gt']"),
            ("a0_underwater", "A0", "underwater", a0_under, "outputs['pred_image']"),
            ("p30_mhold_underwater", "P30-MHOLD", "underwater", p30_under, "outputs['pred_image']"),
            ("a0_abs_rgb_residual", "A0", "underwater_residual", a0_residual, "abs(A0 pred_image - GT)"),
            (
                "p30_mhold_abs_rgb_residual",
                "P30-MHOLD",
                "underwater_residual",
                p30_residual,
                "abs(P30-MHOLD pred_image - GT)",
            ),
            (
                "p30_mhold_minus_a0_abs_rgb_residual_diff",
                "P30-MHOLD_minus_A0",
                "underwater_residual",
                residual_delta,
                "0.5 + 4.0 * (mean_abs_residual_P30 - mean_abs_residual_A0)",
            ),
        ]
        for stem, run_label, image_type, tensor, source_tensor in underwater_images:
            path = uv_dir / f"{view_tag}_{stem}.png"
            saved_width, saved_height = _save_png(path, tensor)
            paths[stem] = path
            source_run = run_label if run_label in evaluated else "A0"
            checkpoint = evaluated[source_run]["checkpoint"] if source_run in evaluated else ""
            config = evaluated[source_run]["config"] if source_run in evaluated else ""
            step = evaluated[source_run]["loaded_step"] if source_run in evaluated else ""
            normalization = PNG_NORMALIZATION
            value_range = _range_string(tensor)
            if stem.endswith("residual_diff"):
                normalization = (
                    "signed_mean_abs_residual_delta_to_gray; "
                    f"neutral=0.5 scale={RESIDUAL_DIFF_SCALE} clamp_to_[0,1]_then_uint8"
                )
                value_range = "visualized_[0,1]; raw_signed_delta_recorded_in_source_tensor"
            rows.append(
                _manifest_row(
                    scene=scene,
                    view_id=str(view_id),
                    camera_id=camera_id,
                    run=run_label,
                    checkpoint_step=step,
                    image_type=image_type,
                    component="rgb" if "residual" not in image_type else stem.replace(f"{view_tag}_", ""),
                    path=path,
                    width=saved_width,
                    height=saved_height,
                    value_range=value_range,
                    normalization=normalization,
                    source_checkpoint=checkpoint,
                    source_config=config,
                    source_tensor=source_tensor,
                )
            )

        dewatered_diff = (p30_view["dewatered"].clamp(0.0, 1.0) - a0_view["dewatered"].clamp(0.0, 1.0)).abs()
        dewatered_images = [
            ("a0_dewatered", "A0", "dewatered", a0_view["dewatered"], a0_view["dewatered_source"]),
            ("p30_mhold_dewatered", "P30-MHOLD", "dewatered", p30_view["dewatered"], p30_view["dewatered_source"]),
            (
                "p30_mhold_minus_a0_dewatered_abs_difference",
                "P30-MHOLD_minus_A0",
                "dewatered_difference",
                dewatered_diff,
                "abs(P30-MHOLD dewatered - A0 dewatered)",
            ),
        ]
        for stem, run_label, image_type, tensor, source_tensor in dewatered_images:
            path = dv_dir / f"{view_tag}_{stem}.png"
            saved_width, saved_height = _save_png(path, tensor.clamp(0.0, 1.0))
            paths[stem] = path
            source_run = run_label if run_label in evaluated else ""
            rows.append(
                _manifest_row(
                    scene=scene,
                    view_id=str(view_id),
                    camera_id=camera_id,
                    run=run_label,
                    checkpoint_step=evaluated[source_run]["loaded_step"] if source_run else "",
                    image_type=image_type,
                    component="dewatered_proxy" if image_type == "dewatered" else "absolute_difference",
                    path=path,
                    width=saved_width,
                    height=saved_height,
                    value_range=_range_string(tensor),
                    normalization=PNG_NORMALIZATION,
                    source_checkpoint=evaluated[source_run]["checkpoint"] if source_run else "",
                    source_config=evaluated[source_run]["config"] if source_run else "",
                    source_tensor=source_tensor,
                    notes="model intrinsic proxy; no clear-image GT is used",
                )
            )

        for run_label, view in (("A0", a0_view), ("P30-MHOLD", p30_view)):
            unavailable.setdefault(run_label, {}).update(view["unavailable_components"])
            for component, tensor in view["components"].items():
                filename_run = "a0" if run_label == "A0" else "p30_mhold"
                path = mv_dir / f"{view_tag}_{filename_run}_{component}.png"
                saved_width, saved_height = _save_png(path, tensor)
                paths[f"{filename_run}_{component}"] = path
                rows.append(
                    _manifest_row(
                        scene=scene,
                        view_id=str(view_id),
                        camera_id=camera_id,
                        run=run_label,
                        checkpoint_step=evaluated[run_label]["loaded_step"],
                        image_type="medium_component",
                        component=component,
                        path=path,
                        width=saved_width,
                        height=saved_height,
                        value_range=_range_string(tensor),
                        normalization=PNG_NORMALIZATION,
                        source_checkpoint=evaluated[run_label]["checkpoint"],
                        source_config=evaluated[run_label]["config"],
                        source_tensor=COMPONENT_DEFINITIONS.get(component, component),
                    )
                )
        per_view_paths[view_id] = paths

    sheet_rows = _write_contact_sheets(scene, scene_output, per_view_paths, selected, view_ids, tile_width)
    rows.extend(sheet_rows)
    scene_summary = {
        "scene": scene,
        "view_count": len(view_ids),
        "view_ids": view_ids,
        "representative_view_ids": selected,
        "runs": {
            label: {
                "config": data["config"],
                "requested_step": data["requested_step"],
                "loaded_step": data["loaded_step"],
                "checkpoint": data["checkpoint"],
                "experiment_name": data["experiment_name"],
                "configured_sh_degree": data["configured_sh_degree"],
                "active_sh_degree": data["active_sh_degree"],
            }
            for label, data in evaluated.items()
        },
        "contact_sheets": {
            key: row["file_path"]
            for row in sheet_rows
            for key in [f"{row['image_type']}:{row['component']}"]
        },
        "unavailable_components": unavailable,
        "validation": validation,
    }
    return rows, scene_summary


def _write_contact_sheets(
    scene: str,
    scene_output: Path,
    per_view_paths: Mapping[int, Mapping[str, Path]],
    selected: Sequence[int],
    all_views: Sequence[int],
    tile_width: int,
) -> List[Dict[str, Any]]:
    manifest_rows: List[Dict[str, Any]] = []

    def add_sheet(kind: str, subset: Sequence[int], group: str, component: str, rows: List[List[Tuple[str, Path]]]) -> None:
        path = scene_output / group / "contact_sheets" / f"contact_sheet_{kind}_{component}.png"
        width, height = _write_sheet(path, rows, tile_width)
        manifest_rows.append(
            _manifest_row(
                scene=scene,
                view_id=kind,
                camera_id="multiple",
                run="comparison",
                checkpoint_step=15000,
                image_type="contact_sheet",
                component=component,
                path=path,
                width=width,
                height=height,
                value_range=RGB_VALUE_RANGE,
                normalization="tiles reuse corresponding per-view PNG normalization",
                source_checkpoint="A0 and P30-MHOLD",
                source_config="A0 and P30-MHOLD",
                source_tensor=f"views={list(subset)}",
                notes="contact sheet labels are identifiers only; no visual-quality annotation",
            )
        )

    for kind, subset in (("all", all_views), ("representative", selected)):
        underwater_rows = [
            [
                (f"{scene} view {view_id} GT", per_view_paths[view_id]["gt_underwater"]),
                (f"view {view_id} A0", per_view_paths[view_id]["a0_underwater"]),
                (f"view {view_id} P30-MHOLD", per_view_paths[view_id]["p30_mhold_underwater"]),
                (f"view {view_id} A0 residual", per_view_paths[view_id]["a0_abs_rgb_residual"]),
                (f"view {view_id} P30 residual", per_view_paths[view_id]["p30_mhold_abs_rgb_residual"]),
                (f"view {view_id} P30-A0 residual", per_view_paths[view_id]["p30_mhold_minus_a0_abs_rgb_residual_diff"]),
            ]
            for view_id in subset
        ]
        add_sheet(kind, subset, "underwater", "underwater_comparison", underwater_rows)

        dewatered_rows = [
            [
                (f"{scene} view {view_id} A0", per_view_paths[view_id]["a0_dewatered"]),
                (f"view {view_id} P30-MHOLD", per_view_paths[view_id]["p30_mhold_dewatered"]),
                (
                    f"view {view_id} abs difference",
                    per_view_paths[view_id]["p30_mhold_minus_a0_dewatered_abs_difference"],
                ),
            ]
            for view_id in subset
        ]
        add_sheet(kind, subset, "dewatered", "dewatered_comparison", dewatered_rows)

        for component in COMPONENT_DEFINITIONS:
            component_rows: List[List[Tuple[str, Path]]] = []
            present = True
            for view_id in subset:
                a0_key = f"a0_{component}"
                p30_key = f"p30_mhold_{component}"
                if a0_key not in per_view_paths[view_id] or p30_key not in per_view_paths[view_id]:
                    present = False
                    break
                component_rows.append(
                    [
                        (f"{scene} view {view_id} A0 {component}", per_view_paths[view_id][a0_key]),
                        (f"view {view_id} P30-MHOLD {component}", per_view_paths[view_id][p30_key]),
                    ]
                )
            if present:
                add_sheet(kind, subset, "medium", component, component_rows)
    return manifest_rows


def _write_manifest_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def _write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "view_id",
        "camera_id",
        "run",
        "checkpoint_step",
        "image_type",
        "component",
        "file_path",
        "width",
        "height",
        "value_range",
        "normalization",
        "tone_mapping",
        "color_space",
        "source_checkpoint",
        "source_config",
        "source_tensor",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_review_index(path: Path, scene_summaries: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    lines = [
        "# GMVC Four-Scene Visual Review Index",
        "",
        "This index lists generated image locations only. It does not include visual-quality conclusions.",
        "",
        f"Manifest JSON: {root / 'manifest.json'}",
        f"Manifest CSV: {root / 'manifest.csv'}",
        "",
    ]
    for scene in SCENE_ORDER:
        if scene not in scene_summaries:
            continue
        scene_dir = root / scene
        lines.extend(
            [
                f"# {scene}",
                "",
                "## Underwater",
                f"Representative contact sheet: {scene_dir / 'underwater/contact_sheets/contact_sheet_representative_underwater_comparison.png'}",
                f"All-view contact sheet: {scene_dir / 'underwater/contact_sheets/contact_sheet_all_underwater_comparison.png'}",
                f"All views: {scene_dir / 'underwater/per_view'}",
                "",
                "## Dewatered",
                f"Representative contact sheet: {scene_dir / 'dewatered/contact_sheets/contact_sheet_representative_dewatered_comparison.png'}",
                f"All-view contact sheet: {scene_dir / 'dewatered/contact_sheets/contact_sheet_all_dewatered_comparison.png'}",
                f"All views: {scene_dir / 'dewatered/per_view'}",
                "",
                "## Medium",
            ]
        )
        for component in COMPONENT_DEFINITIONS:
            lines.append(
                f"{component}: {scene_dir / 'medium/contact_sheets' / ('contact_sheet_representative_' + component + '.png')}"
            )
        lines.append(f"All medium per-view images: {scene_dir / 'medium/per_view'}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _verify_saved_images(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing: List[str] = []
    bad_size: List[str] = []
    empty: List[str] = []
    for row in rows:
        path = Path(str(row["file_path"]))
        if not path.exists():
            missing.append(str(path))
            continue
        if path.stat().st_size <= 0:
            empty.append(str(path))
        with Image.open(path) as image:
            width, height = image.size
        if int(row["width"]) != width or int(row["height"]) != height:
            bad_size.append(str(path))
    return {
        "missing_files": missing,
        "empty_files": empty,
        "size_mismatches": bad_size,
        "pass": not missing and not empty and not bad_size,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = _repo_root()
    specs = _scene_specs(repo)
    selected_scenes = args.scene or list(SCENE_ORDER)
    output_dir = args.output_dir
    manifest_rows: List[Dict[str, Any]] = []
    scene_summaries: Dict[str, Dict[str, Any]] = {}

    for scene in selected_scenes:
        if scene not in specs:
            raise ValueError(f"Unknown scene {scene}. Valid scenes: {', '.join(SCENE_ORDER)}")
        spec = specs[scene]
        for config_path in (spec.a0_config, spec.p30_config):
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            ckpt_path = config_path.parent / "nerfstudio_models/step-000015000.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
        evaluated = {
            "A0": _render_checkpoint(RunSpec("A0", spec.a0_config), args.test_mode, args.max_images),
            "P30-MHOLD": _render_checkpoint(RunSpec("P30-MHOLD", spec.p30_config), args.test_mode, args.max_images),
        }
        scene_rows, scene_summary = _write_scene_outputs(scene, output_dir / scene, evaluated, args.contact_tile_width)
        manifest_rows.extend(scene_rows)
        scene_summaries[scene] = scene_summary

    verification = _verify_saved_images(manifest_rows)
    summary_counts = _count_by_scene(manifest_rows)
    manifest_payload = {
        "diagnostic": "gmvc_four_scene_visual_audit",
        "created_by_script": str(Path(__file__).resolve().relative_to(repo)),
        "repo": str(repo),
        "git_branch": _git_branch(repo),
        "git_commit": _git_commit(repo),
        "output_root": str(output_dir),
        "test_mode": args.test_mode,
        "view_scope": "all eval views from fixed_indices_eval_dataloader",
        "definitions": {
            "gt_underwater": "images['gt'] from model.get_image_metrics_and_images; dataset underwater RGB composited with outputs['background'].",
            "underwater": "outputs['pred_image'] from model.get_outputs_for_camera at the eval camera/checkpoint; clamped only for PNG save.",
            "underwater_residual": "absolute RGB residual abs(render - GT), plus a signed mean-residual delta image for P30-MHOLD minus A0.",
            "dewatered": (
                "diagnostic model intrinsic proxy: J_proxy_raw rendered under a forced GMVC DC-proxy context when available; "
                "fallback is J_gaussian_raw then J_raw. This is not clear-image GT."
            ),
            "dewatered_difference": "absolute RGB difference abs(P30-MHOLD dewatered proxy - A0 dewatered proxy).",
            "medium_component": COMPONENT_DEFINITIONS,
        },
        "comparability": {
            "same_camera": "checked per scene/view by image_idx, image name, intrinsics, resolution, and camera_to_worlds.",
            "same_resolution": "checked per scene/view for A0 and P30-MHOLD tensor shapes.",
            "same_crop": "no diagnostic crop is applied.",
            "same_tone_mapping": NO_TONE_MAPPING,
            "same_color_space": COLOR_SPACE,
            "same_sh_degree": "checked per scene between configured and active SH degree for A0 and P30-MHOLD.",
            "same_save_format": "PNG RGB uint8 for every per-view image and contact sheet.",
            "residual_delta_visualization": (
                "mean-channel signed residual delta mapped as clamp(0.5 + "
                f"{RESIDUAL_DIFF_SCALE} * (P30_abs_residual - A0_abs_residual), 0, 1)."
            ),
            "component_display_range": "all listed medium components are saved with clamp_to_[0,1]_then_uint8; raw per-image min/max is recorded.",
        },
        "scenes": scene_summaries,
        "counts": summary_counts,
        "verification": verification,
        "manifest_rows": manifest_rows,
    }
    _write_manifest_json(output_dir / "manifest.json", manifest_payload)
    _write_manifest_csv(output_dir / "manifest.csv", manifest_rows)
    _write_review_index(output_dir / "VISUAL_REVIEW_INDEX.md", scene_summaries, output_dir)
    return manifest_payload


def _count_by_scene(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        scene = str(row["scene"])
        image_type = str(row["image_type"])
        counts.setdefault(
            scene,
            {
                "underwater": 0,
                "dewatered": 0,
                "medium_component": 0,
                "contact_sheet": 0,
                "gt_underwater": 0,
                "underwater_residual": 0,
                "dewatered_difference": 0,
            },
        )
        if image_type in counts[scene]:
            counts[scene][image_type] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", default=[], help="Scene name; defaults to all four scenes.")
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--contact-tile-width", type=int, default=260)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/gmvc_four_scene_visual_audit"),
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output_root": str(args.output_dir),
                "manifest_json": str(args.output_dir / "manifest.json"),
                "manifest_csv": str(args.output_dir / "manifest.csv"),
                "review_index": str(args.output_dir / "VISUAL_REVIEW_INDEX.md"),
                "counts": result["counts"],
                "verification_pass": result["verification"]["pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
