#!/usr/bin/env python
"""Read-only diagnostics for BND RGB trade-offs.

This script intentionally does not train, step optimizers, edit checkpoints, or
change renderer/model definitions. Historical BND configs used the old
``sigmoid_sh`` label; the diagnostic maps that label to the clean branch
``bounded_sh3`` value in memory after loading the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.water_splatting import SH2RGB, SHLogits2RGB


SCENE_CONFIGS: Dict[str, Dict[str, str]] = {
    "Curasao": {
        "M1": "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml",
        "BND": "outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml",
    },
    "JapaneseGradens": {
        "M1": "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml",
        "BND": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/config.yml",
    },
    "IUI3": {
        "M1": "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml",
        "BND": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/config.yml",
    },
    "Panama": {
        "M1": "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml",
        "BND": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml",
    },
}

TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
FINAL_NOMINAL_STEP = 15000
CHANNELS = ("r", "g", "b")


@dataclass
class LoadedRun:
    scene: str
    run: str
    nominal_step: int
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any

    @property
    def model(self) -> Any:
        return self.pipeline.model


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _quantile(flat: torch.Tensor, q: float) -> float:
    flat = flat.detach().float().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    if q <= 0.0:
        return float(flat.min().item())
    if q >= 1.0:
        return float(flat.max().item())
    k = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, k).values.item())


def _stats(flat: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    flat = flat.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in ("mean", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _quantile(flat, 0.01),
        f"{prefix}p05": _quantile(flat, 0.05),
        f"{prefix}p10": _quantile(flat, 0.10),
        f"{prefix}p50": _quantile(flat, 0.50),
        f"{prefix}p90": _quantile(flat, 0.90),
        f"{prefix}p95": _quantile(flat, 0.95),
        f"{prefix}p99": _quantile(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_stats(tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
    tensor = tensor.detach().float()
    out: Dict[str, float] = {}
    if tensor.ndim == 1:
        return _stats(tensor, prefix=f"{prefix}_")
    if tensor.shape[-1] == 3:
        for idx, channel in enumerate(CHANNELS):
            out.update(_stats(tensor[..., idx], prefix=f"{prefix}_{channel}_"))
        out.update(_stats(tensor.reshape(-1), prefix=f"{prefix}_all_"))
    else:
        out.update(_stats(tensor.reshape(-1), prefix=f"{prefix}_"))
    return out


def _thresholds(tensor: torch.Tensor, prefix: str, thresholds: Sequence[float], op: str = "lt") -> Dict[str, float]:
    flat = tensor.detach().float().reshape(-1)
    out: Dict[str, float] = {}
    if flat.numel() == 0:
        for threshold in thresholds:
            out[f"{prefix}_{op}_{threshold:g}"] = float("nan")
        return out
    for threshold in thresholds:
        if op == "lt":
            value = (flat < threshold).float().mean()
        elif op == "gt":
            value = (flat > threshold).float().mean()
        elif op == "abs_gt":
            value = (flat.abs() > threshold).float().mean()
        else:
            raise ValueError(op)
        out[f"{prefix}_{op}_{threshold:g}"] = float(value.item())
    return out


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    steps: Dict[int, Path] = {}
    if not ckpt_dir.exists():
        return steps
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            step = int(path.stem.split("-")[1])
        except Exception:
            continue
        steps[step] = path
    return steps


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    return None


def _load_run(repo: Path, scene: str, run: str, nominal_step: int) -> LoadedRun:
    config_path = repo / SCENE_CONFIGS[scene][run]
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing checkpoint for {scene} {run} nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    if run == "BND":
        pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    elif run == "M1":
        pipeline.model.config.intrinsic_color_parameterization = "legacy"
    pipeline.eval()
    return LoadedRun(
        scene=scene,
        run=run,
        nominal_step=nominal_step,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        loaded_step=loaded_step,
        config=config,
        pipeline=pipeline,
    )


def _release_loaded(item: Optional[LoadedRun]) -> None:
    if item is None:
        return
    try:
        del item.pipeline
    except Exception:
        pass
    torch.cuda.empty_cache()


def _view_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    rows = []
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        view_id = Path(filename).stem
        rows.append((eval_index, view_id, camera, batch))
    return rows


def _metric_images(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
    }


def _eval_view_no_grad(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        metrics = _metric_images(model, outputs["pred_image"], gt)
    return outputs, gt, metrics


def _safe_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _norm_rgb(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(tensor.float(), dim=-1)


def _mse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(((pred.float() - gt.float()) ** 2).mean().item())


def _rgb_to_uint8(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _scalar_to_uint8(values: torch.Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), 1e-8)
    arr = (values.detach().float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _overlay_mask(base_scalar: torch.Tensor, mask: torch.Tensor, scale: float, color: Tuple[int, int, int] = (255, 40, 40)) -> Image.Image:
    image = _scalar_to_uint8(base_scalar, scale).convert("RGB")
    pix = image.load()
    mask_cpu = mask.detach().bool().cpu()
    for y in range(image.height):
        for x in range(image.width):
            if bool(mask_cpu[y, x]):
                old = pix[x, y]
                pix[x, y] = tuple(int(0.45 * old[i] + 0.55 * color[i]) for i in range(3))
    return image


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 28
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((6, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        width = sum(tile.width for tile in tiles) + 6 * (len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows) + 6 * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _save_image(path: Path, image: Image.Image, manifest: List[Dict[str, Any]], **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    row = {"file_path": str(path), "width": image.width, "height": image.height}
    row.update(meta)
    manifest.append(row)


def _append_component_stats(row: Dict[str, Any], outputs: Mapping[str, torch.Tensor], run: str) -> None:
    medium_attn = outputs["medium_attn"]
    tau = outputs["tau_D"]
    transmission = outputs["transmission"]
    clear = outputs["clear_object_fullsh_raw"]
    row.update(_channel_stats(medium_attn, "beta_D"))
    row.update(_channel_stats(tau, "tau_D"))
    row.update(_channel_stats(transmission, "T_D"))
    row.update(_thresholds(transmission, "T_D", (0.30, 0.20, 0.10, 0.05), "lt"))
    row.update(_channel_stats(clear, "J"))
    row.update(_thresholds(clear, "J", (0.95, 0.99, 1.0, 1.5, 2.0), "gt"))
    if run == "BND":
        if "gaussian_view_rgb" in outputs:
            c = outputs["gaussian_view_rgb"]
            visible = outputs.get("gaussian_visible_mask")
            if visible is not None and visible.numel() == c.shape[0]:
                c = c[visible.bool()]
            row.update(_channel_stats(c, "c"))
            row.update(_thresholds(c, "c", (0.95, 0.99), "gt"))
        if "gaussian_view_logits" in outputs:
            s = outputs["gaussian_view_logits"].detach()
            visible = outputs.get("gaussian_visible_mask")
            if visible is not None and visible.numel() == s.shape[0]:
                s = s[visible.bool()]
            row.update(_channel_stats(s, "s"))
            row.update(_thresholds(s, "s_abs", (5.0, 8.0, 10.0), "abs_gt"))
        if "gaussian_sigmoid_derivative" in outputs:
            g = outputs["gaussian_sigmoid_derivative"].detach()
            visible = outputs.get("gaussian_visible_mask")
            if visible is not None and visible.numel() == g.shape[0]:
                g = g[visible.bool()]
            row.update(_channel_stats(g, "sigmoid_derivative"))


def _features_stats(model: Any, scene: str, run: str, nominal_step: int) -> Dict[str, Any]:
    with torch.no_grad():
        rest = model.features_rest.detach().float().reshape(model.features_rest.shape[0], -1)
        dc = model.features_dc.detach().float()
        rest_norm = torch.linalg.norm(rest, dim=-1)
        dc_norm = torch.linalg.norm(dc, dim=-1)
        ratio = rest_norm / dc_norm.clamp_min(1e-8)
    row: Dict[str, Any] = {
        "scene": scene,
        "run": run,
        "step": nominal_step,
        "gaussian_count": int(model.num_points),
    }
    row.update(_stats(rest_norm, "features_rest_norm_"))
    row.update(_stats(dc_norm, "features_dc_norm_"))
    row.update(_stats(ratio, "features_rest_dc_ratio_"))
    return row


def _sh_capacity_stats(outputs: Mapping[str, torch.Tensor], model: Any, scene: str, run: str, view_id: str) -> Dict[str, Any]:
    full = outputs["gaussian_view_rgb"].detach().float()
    visible = outputs.get("gaussian_visible_mask")
    if run == "BND":
        if "gaussian_view_dc_rgb" in outputs:
            dc_rgb = outputs["gaussian_view_dc_rgb"].detach().float().to(full.device)
        else:
            dc_rgb = SHLogits2RGB(model.features_dc.detach().float()).to(full.device)
    else:
        dc_rgb = torch.clamp(SH2RGB(model.features_dc.detach().float()), min=0.0).to(full.device)
    residual = torch.linalg.norm(full - dc_rgb, dim=-1)
    row: Dict[str, Any] = {"scene": scene, "run": run, "view_id": view_id}
    row.update(_stats(residual, "R_SH_all_"))
    if visible is not None and visible.numel() == residual.numel():
        visible_mask = visible.detach().bool().to(residual.device)
        row.update(_stats(residual[visible_mask], "R_SH_visible_"))
        row["visible_fraction"] = float(visible.float().mean().item())
    else:
        row["visible_fraction"] = float("nan")
    return row


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and row[key] == row[key]]
        out[key] = _mean(vals)
    return out


def trajectory_audit(repo: Path, scenes: Sequence[str], output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    for scene in scenes:
        for run in ("M1", "BND"):
            config_path = repo / SCENE_CONFIGS[scene][run]
            available = _available_steps(config_path)
            for nominal_step in TRAJECTORY_STEPS:
                actual = _actual_step(config_path, nominal_step)
                if actual is None:
                    missing.append(
                        {
                            "scene": scene,
                            "run": run,
                            "nominal_step": nominal_step,
                            "config_path": str(config_path),
                            "available_steps": ";".join(str(s) for s in sorted(available)),
                            "status": "MISSING_CHECKPOINT",
                        }
                    )
                    continue
                loaded: Optional[LoadedRun] = None
                try:
                    loaded = _load_run(repo, scene, run, nominal_step)
                    model = loaded.model
                    metric_rows: List[Dict[str, Any]] = []
                    component_rows: List[Dict[str, Any]] = []
                    sh_rows: List[Dict[str, Any]] = []
                    for _, view_id, camera, batch in _view_records(loaded):
                        outputs, _, metrics = _eval_view_no_grad(model, camera, batch)
                        mrow = {"view_id": view_id, **metrics}
                        metric_rows.append(mrow)
                        crow: Dict[str, Any] = {}
                        _append_component_stats(crow, outputs, run)
                        component_rows.append(crow)
                        if nominal_step == FINAL_NOMINAL_STEP:
                            sh_rows.append(_sh_capacity_stats({k: _safe_cpu(v) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}, model, scene, run, view_id))
                    row: Dict[str, Any] = {
                        "scene": scene,
                        "run": run,
                        "nominal_step": nominal_step,
                        "loaded_step": loaded.loaded_step,
                        "checkpoint_path": str(loaded.checkpoint_path),
                        "config_path": str(loaded.config_path),
                        "intrinsic_color_parameterization": model.config.intrinsic_color_parameterization,
                        "num_eval_views": len(metric_rows),
                        "gaussian_count": int(model.num_points),
                    }
                    row.update(_aggregate_metric_rows(metric_rows, ("psnr", "ssim", "lpips")))
                    for key in sorted({key for item in component_rows for key in item}):
                        vals = [float(item[key]) for item in component_rows if key in item and item[key] == item[key]]
                        row[key] = _mean(vals)
                    rows.append(row)
                    if nominal_step == FINAL_NOMINAL_STEP:
                        feature_rows.append(_features_stats(model, scene, run, nominal_step))
                        _write_csv(output_dir / "intermediate" / f"{scene}_{run}_sh_capacity_views.csv", sh_rows)
                finally:
                    _release_loaded(loaded)
    return rows, missing, feature_rows


def _delta_rows(trajectory_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["scene"], row["run"], int(row["nominal_step"])): row for row in trajectory_rows}
    deltas = []
    for scene in sorted({row["scene"] for row in trajectory_rows}):
        for step in TRAJECTORY_STEPS:
            m1 = by_key.get((scene, "M1", step))
            bnd = by_key.get((scene, "BND", step))
            if not m1 or not bnd:
                continue
            row: Dict[str, Any] = {"scene": scene, "nominal_step": step}
            for key in ("psnr", "ssim", "lpips", "tau_D_all_p90", "J_all_p99", "T_D_lt_0.1", "gaussian_count"):
                if key in m1 and key in bnd:
                    row[f"delta_{key}"] = float(bnd[key]) - float(m1[key])
            deltas.append(row)
    return deltas


def _late_recovery_flags(trajectory_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["scene"], row["run"], int(row["nominal_step"])): row for row in trajectory_rows}
    flags = []
    for scene in sorted({row["scene"] for row in trajectory_rows}):
        b13 = by_key.get((scene, "BND", 13000))
        b15 = by_key.get((scene, "BND", 15000))
        m13 = by_key.get((scene, "M1", 13000))
        m15 = by_key.get((scene, "M1", 15000))
        bnd_psnr_gain = float("nan")
        gap_shrink = float("nan")
        flag = False
        if b13 and b15:
            bnd_psnr_gain = float(b15["psnr"]) - float(b13["psnr"])
            flag = flag or bnd_psnr_gain >= 0.10
        if b13 and b15 and m13 and m15:
            gap13 = float(b13["psnr"]) - float(m13["psnr"])
            gap15 = float(b15["psnr"]) - float(m15["psnr"])
            gap_shrink = gap15 - gap13
            flag = flag or gap_shrink >= 0.10
        flags.append(
            {
                "scene": scene,
                "BND_PSNR_13k_to_15k": bnd_psnr_gain,
                "BND_minus_M1_gap_shrink_13k_to_15k": gap_shrink,
                "BND_LATE_RGB_RECOVERY": bool(flag),
                "POSSIBLE_UNDERCONVERGENCE": bool(flag),
                "note": "M1 13k missing prevents gap-shrink test" if not m13 else "",
            }
        )
    return flags


def _cache_final_outputs(loaded: LoadedRun) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = loaded.model
    model_config = model.config
    machine_config = getattr(loaded.config, "machine", None)
    rows: List[Dict[str, Any]] = []
    metadata = {
        "scene": loaded.scene,
        "run": loaded.run,
        "nominal_step": loaded.nominal_step,
        "loaded_step": loaded.loaded_step,
        "config_path": str(loaded.config_path),
        "checkpoint_path": str(loaded.checkpoint_path),
        "intrinsic_color_parameterization": model.config.intrinsic_color_parameterization,
        "gaussian_count": int(model.num_points),
        "seed": getattr(machine_config, "seed", ""),
        "max_num_iterations": getattr(loaded.config, "max_num_iterations", ""),
        "steps_per_save": getattr(loaded.config, "steps_per_save", ""),
        "sh_degree": getattr(model_config, "sh_degree", ""),
        "medium_context_mode": getattr(model_config, "medium_context_mode", ""),
        "b_inf_mode": getattr(model_config, "b_inf_mode", ""),
        "infinite_water_enabled": getattr(model_config, "infinite_water_enabled", ""),
        "mlp_type": getattr(model_config, "mlp_type", ""),
    }
    for eval_index, view_id, camera, batch in _view_records(loaded):
        outputs, gt, metrics = _eval_view_no_grad(model, camera, batch)
        item: Dict[str, Any] = {
            "eval_index": eval_index,
            "view_id": view_id,
            "metrics": metrics,
            "gt": _safe_cpu(gt),
            "sh_capacity": _sh_capacity_stats(outputs, model, loaded.scene, loaded.run, view_id),
        }
        for key in (
            "pred_image",
            "direct_object_signal",
            "rgb_medium",
            "clear_object_fullsh_raw",
            "tau_D",
            "transmission",
            "depth",
            "accumulation",
            "medium_attn",
            "medium_bs",
            "medium_rgb",
            "gaussian_view_rgb",
            "gaussian_view_logits",
            "gaussian_sigmoid_derivative",
            "gaussian_visible_mask",
            "gaussian_view_dc_rgb",
        ):
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                item[key] = _safe_cpu(outputs[key])
        rows.append(item)
    return rows, metadata


def _comp_masks(m1: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    j_scalar = m1["clear_object_fullsh_raw"].float().amax(dim=-1)
    tau_scalar = m1["tau_D"].float().mean(dim=-1)
    t_scalar = m1["transmission"].float().amin(dim=-1)
    tau_thr = torch.quantile(tau_scalar.reshape(-1), 0.90)
    j95_thr = torch.quantile(j_scalar.reshape(-1), 0.95)
    masks = {
        "J1": j_scalar > 1.0,
        "J95": j_scalar >= j95_thr,
        "TAU90": tau_scalar >= tau_thr,
        "TLOW": t_scalar < 0.1,
    }
    masks["COMP"] = masks["J1"] | masks["TAU90"] | masks["TLOW"]
    return masks


def final_pair_audit(
    repo: Path,
    scenes: Sequence[str],
    output_dir: Path,
    render_dir: Path,
    tile_width: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    per_view_rows: List[Dict[str, Any]] = []
    enrichment_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    hybrid_rows: List[Dict[str, Any]] = []
    jacobian_rows: List[Dict[str, Any]] = []
    sh_rows: List[Dict[str, Any]] = []
    strat_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    summary_underwater_rows = []
    summary_decomp_rows = []

    for scene in scenes:
        m1_loaded = bnd_loaded = None
        try:
            m1_loaded = _load_run(repo, scene, "M1", FINAL_NOMINAL_STEP)
            m1_rows, m1_meta = _cache_final_outputs(m1_loaded)
        finally:
            _release_loaded(m1_loaded)
        try:
            bnd_loaded = _load_run(repo, scene, "BND", FINAL_NOMINAL_STEP)
            bnd_rows, bnd_meta = _cache_final_outputs(bnd_loaded)
            metric_model = bnd_loaded.model
        except Exception:
            _release_loaded(bnd_loaded)
            raise

        checkpoint_rows.extend([m1_meta, bnd_meta])
        m1_by_view = {row["view_id"]: row for row in m1_rows}
        bnd_by_view = {row["view_id"]: row for row in bnd_rows}
        if list(m1_by_view) != list(bnd_by_view):
            raise RuntimeError(f"{scene} view mismatch: {list(m1_by_view)} vs {list(bnd_by_view)}")

        scene_delta_e_max = 0.0
        scene_resid_max = 0.0
        scene_tau_max = 3.0
        for view_id in m1_by_view:
            m1 = m1_by_view[view_id]
            bnd = bnd_by_view[view_id]
            gt = m1["gt"]
            e_m1 = _norm_rgb(m1["pred_image"] - gt)
            e_bnd = _norm_rgb(bnd["pred_image"] - gt)
            excess = (e_bnd - e_m1).clamp_min(0.0)
            scene_delta_e_max = max(scene_delta_e_max, float(excess.max().item()))
            scene_resid_max = max(scene_resid_max, float(e_m1.max().item()), float(e_bnd.max().item()))
            scene_tau_max = max(scene_tau_max, float(m1["tau_D"].max().item()), float(bnd["tau_D"].max().item()))

        closure_abs: List[float] = []
        component_acc = {"delta_mse": [], "C_direct": [], "C_medium": [], "C_cross": [], "closure": []}
        hybrid_metric_acc: Dict[str, List[Dict[str, float]]] = {"M1": [], "BND": [], "Hybrid-D": [], "Hybrid-M": []}
        scene_enrichment: Dict[str, Dict[str, float]] = {}
        worst_order: List[Tuple[float, str]] = []
        best_order: List[Tuple[float, str]] = []

        for view_id in m1_by_view:
            m1 = m1_by_view[view_id]
            bnd = bnd_by_view[view_id]
            gt = m1["gt"]
            psnr_delta = float(bnd["metrics"]["psnr"]) - float(m1["metrics"]["psnr"])
            row = {
                "scene": scene,
                "view_id": view_id,
                "M1_PSNR": m1["metrics"]["psnr"],
                "BND_PSNR": bnd["metrics"]["psnr"],
                "Delta_PSNR": psnr_delta,
                "M1_SSIM": m1["metrics"]["ssim"],
                "BND_SSIM": bnd["metrics"]["ssim"],
                "Delta_SSIM": float(bnd["metrics"]["ssim"]) - float(m1["metrics"]["ssim"]),
                "M1_LPIPS": m1["metrics"]["lpips"],
                "BND_LPIPS": bnd["metrics"]["lpips"],
                "Delta_LPIPS": float(bnd["metrics"]["lpips"]) - float(m1["metrics"]["lpips"]),
            }
            per_view_rows.append(row)
            worst_order.append((psnr_delta, view_id))
            best_order.append((psnr_delta, view_id))

            e_m1 = _norm_rgb(m1["pred_image"] - gt)
            e_bnd = _norm_rgb(bnd["pred_image"] - gt)
            excess = (e_bnd - e_m1).clamp_min(0.0)
            masks = _comp_masks(m1)
            denom = float(excess.sum().item())
            for mask_name, mask in masks.items():
                area = float(mask.float().mean().item())
                efrac = float(excess[mask].sum().item() / denom) if denom > 1e-12 else 0.0
                enrichment = efrac / area if area > 1e-12 else float("inf")
                enrichment_rows.append(
                    {
                        "scene": scene,
                        "view_id": view_id,
                        "mask": mask_name,
                        "mask_area": area,
                        "excess_error_fraction": efrac,
                        "residual_enrichment": enrichment,
                    }
                )
                scene_item = scene_enrichment.setdefault(mask_name, {"area_sum": 0.0, "efrac_sum": 0.0, "count": 0.0})
                scene_item["area_sum"] += area
                scene_item["efrac_sum"] += efrac
                scene_item["count"] += 1.0

            pred0 = m1["pred_image"]
            pred1 = bnd["pred_image"]
            d0 = m1["direct_object_signal"]
            d1 = bnd["direct_object_signal"]
            med0 = m1["rgb_medium"]
            med1 = bnd["rgb_medium"]
            closure0 = (pred0 - (d0 + med0)).abs()
            closure1 = (pred1 - (d1 + med1)).abs()
            closure_abs.extend([float(closure0.mean().item()), float(closure1.mean().item())])
            e0 = pred0 - gt
            delta_d = d1 - d0
            delta_m = med1 - med0
            delta_mse = _mse(pred1, gt) - _mse(pred0, gt)
            c_direct = float((2.0 * (e0 * delta_d).sum(dim=-1) + delta_d.square().sum(dim=-1)).mean().item() / 3.0)
            c_medium = float((2.0 * (e0 * delta_m).sum(dim=-1) + delta_m.square().sum(dim=-1)).mean().item() / 3.0)
            c_cross = float((2.0 * (delta_d * delta_m).sum(dim=-1)).mean().item() / 3.0)
            closure = delta_mse - (c_direct + c_medium + c_cross)
            component_acc["delta_mse"].append(delta_mse)
            component_acc["C_direct"].append(c_direct)
            component_acc["C_medium"].append(c_medium)
            component_acc["C_cross"].append(c_cross)
            component_acc["closure"].append(closure)
            component_rows.append(
                {
                    "scene": scene,
                    "view_id": view_id,
                    "Delta_MSE": delta_mse,
                    "C_direct": c_direct,
                    "C_medium": c_medium,
                    "C_cross": c_cross,
                    "component_closure_error": closure,
                    "additive_closure_M1_mean_abs": float(closure0.mean().item()),
                    "additive_closure_M1_max_abs": float(closure0.max().item()),
                    "additive_closure_BND_mean_abs": float(closure1.mean().item()),
                    "additive_closure_BND_max_abs": float(closure1.max().item()),
                }
            )

            hybrid_d = (d1 + med0).clamp(0.0, 1.0)
            hybrid_m = (d0 + med1).clamp(0.0, 1.0)
            for name, pred in (("M1", pred0), ("BND", pred1), ("Hybrid-D", hybrid_d), ("Hybrid-M", hybrid_m)):
                metrics = _metric_images(metric_model, pred.to(metric_model.device), gt.to(metric_model.device))
                hybrid_metric_acc[name].append(metrics)
                hybrid_rows.append({"scene": scene, "view_id": view_id, "image": name, **metrics})

            if "gaussian_sigmoid_derivative" in bnd:
                deriv = bnd["gaussian_sigmoid_derivative"]
                visible = bnd.get("gaussian_visible_mask")
                if isinstance(visible, torch.Tensor) and visible.numel() == deriv.shape[0]:
                    deriv = deriv[visible.bool()]
                jrow = {"scene": scene, "view_id": view_id}
                jrow.update(_channel_stats(deriv, "sigmoid_derivative"))
                p50 = jrow.get("sigmoid_derivative_all_p50", float("nan"))
                jrow["approx_jacobian_compensation_factor"] = 1.0 / p50 if p50 and p50 == p50 else float("nan")
                jacobian_rows.append(jrow)

            sh_rows.append(dict(m1["sh_capacity"]))
            sh_rows.append(dict(bnd["sh_capacity"]))

            if scene in {"JapaneseGradens", "Panama"}:
                strat_rows.extend(_stratification_rows(scene, view_id, m1, bnd, gt, e_m1, e_bnd, excess))

            raw_dir = output_dir / "raw_maps" / scene / str(view_id)
            raw_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "M1_residual": e_m1,
                    "BND_residual": e_bnd,
                    "BND_excess_residual": excess,
                    "masks": masks,
                    "mask_definition": {
                        "J_scalar": "max RGB channel of M1 clear_object_fullsh_raw",
                        "J95": "top 5 percent of J_scalar within this view",
                        "TAU90": "top 10 percent of mean RGB M1 tau_D within this view",
                        "TLOW": "min RGB M1 transmission < 0.1",
                        "COMP": "J1 OR TAU90 OR TLOW",
                    },
                },
                raw_dir / "residual_and_masks.pt",
            )

        # Aggregate component and enrichment rows.
        scene_component = {
            "scene": scene,
            "view_id": "AGGREGATE",
            "Delta_MSE": _mean(component_acc["delta_mse"]),
            "C_direct": _mean(component_acc["C_direct"]),
            "C_medium": _mean(component_acc["C_medium"]),
            "C_cross": _mean(component_acc["C_cross"]),
            "component_closure_error": _mean(component_acc["closure"]),
            "additive_closure_mean_abs": _mean(closure_abs),
        }
        total_pos = sum(max(scene_component[key], 0.0) for key in ("C_direct", "C_medium", "C_cross"))
        for key in ("C_direct", "C_medium", "C_cross"):
            scene_component[f"{key}_positive_fraction"] = max(float(scene_component[key]), 0.0) / total_pos if total_pos > 1e-12 else 0.0
        component_rows.append(scene_component)
        for mask_name, vals in scene_enrichment.items():
            area = vals["area_sum"] / vals["count"]
            efrac = vals["efrac_sum"] / vals["count"]
            enrichment_rows.append(
                {
                    "scene": scene,
                    "view_id": "AGGREGATE",
                    "mask": mask_name,
                    "mask_area": area,
                    "excess_error_fraction": efrac,
                    "residual_enrichment": efrac / area if area > 1e-12 else float("inf"),
                }
            )

        # Scene visual sheets.
        worst_views = [view for _, view in sorted(worst_order)[:5]]
        if len(worst_views) == 0:
            worst_views = list(m1_by_view)[:1]
        sheet_a: List[List[Tuple[str, Image.Image]]] = []
        sheet_b: List[List[Tuple[str, Image.Image]]] = []
        sheet_c: List[List[Tuple[str, Image.Image]]] = []
        sheet_d: List[List[Tuple[str, Image.Image]]] = []
        sheet_e: List[List[Tuple[str, Image.Image]]] = []
        for view_id in worst_views:
            m1 = m1_by_view[view_id]
            bnd = bnd_by_view[view_id]
            gt = m1["gt"]
            e_m1 = _norm_rgb(m1["pred_image"] - gt)
            e_bnd = _norm_rgb(bnd["pred_image"] - gt)
            excess = (e_bnd - e_m1).clamp_min(0.0)
            masks = _comp_masks(m1)
            sheet_a.append(
                [
                    (f"{scene} {view_id} GT", _rgb_to_uint8(gt)),
                    ("M1 underwater", _rgb_to_uint8(m1["pred_image"])),
                    ("BND underwater", _rgb_to_uint8(bnd["pred_image"])),
                    ("M1 abs residual", _scalar_to_uint8(e_m1, scene_resid_max)),
                    ("BND abs residual", _scalar_to_uint8(e_bnd, scene_resid_max)),
                    ("BND excess", _scalar_to_uint8(excess, scene_delta_e_max)),
                ]
            )
            sheet_b.append(
                [
                    (f"{scene} {view_id} M1 clear", _rgb_to_uint8(m1["clear_object_fullsh_raw"])),
                    ("BND clear", _rgb_to_uint8(bnd["clear_object_fullsh_raw"])),
                    ("M1 tau", _scalar_to_uint8(m1["tau_D"].mean(dim=-1), scene_tau_max)),
                    ("BND tau", _scalar_to_uint8(bnd["tau_D"].mean(dim=-1), scene_tau_max)),
                    ("M1 T", _rgb_to_uint8(m1["transmission"])),
                    ("BND T", _rgb_to_uint8(bnd["transmission"])),
                ]
            )
            sheet_c.append(
                [
                    (f"{scene} {view_id} J1 overlay", _overlay_mask(excess, masks["J1"], scene_delta_e_max)),
                    ("TAU90 overlay", _overlay_mask(excess, masks["TAU90"], scene_delta_e_max, (255, 180, 0))),
                    ("COMP overlay", _overlay_mask(excess, masks["COMP"], scene_delta_e_max, (255, 40, 180))),
                ]
            )
            sheet_d.append(
                [
                    (f"{scene} {view_id} M1 direct", _rgb_to_uint8(m1["direct_object_signal"])),
                    ("BND direct", _rgb_to_uint8(bnd["direct_object_signal"])),
                    ("abs Delta direct", _rgb_to_uint8((bnd["direct_object_signal"] - m1["direct_object_signal"]).abs())),
                    ("M1 medium", _rgb_to_uint8(m1["rgb_medium"])),
                    ("BND medium", _rgb_to_uint8(bnd["rgb_medium"])),
                    ("abs Delta medium", _rgb_to_uint8((bnd["rgb_medium"] - m1["rgb_medium"]).abs())),
                ]
            )
            hybrid_d = (bnd["direct_object_signal"] + m1["rgb_medium"]).clamp(0.0, 1.0)
            hybrid_m = (m1["direct_object_signal"] + bnd["rgb_medium"]).clamp(0.0, 1.0)
            sheet_e.append(
                [
                    (f"{scene} {view_id} GT", _rgb_to_uint8(gt)),
                    ("M1", _rgb_to_uint8(m1["pred_image"])),
                    ("BND", _rgb_to_uint8(bnd["pred_image"])),
                    ("Hybrid-D", _rgb_to_uint8(hybrid_d)),
                    ("Hybrid-M", _rgb_to_uint8(hybrid_m)),
                ]
            )
        scene_render_dir = render_dir / scene
        for name, rows in (
            ("worst_views_underwater_residual.png", sheet_a),
            ("worst_views_clear_tau_transmission.png", sheet_b),
            ("worst_views_compensation_mask_overlays.png", sheet_c),
            ("worst_views_direct_medium_components.png", sheet_d),
            ("worst_views_hybrid_counterfactuals.png", sheet_e),
        ):
            out_path = scene_render_dir / name
            _save_sheet(out_path, rows, tile_width=tile_width)
            manifest.append({"file_path": str(out_path), "scene": scene, "output_type": name, "view_ids": ";".join(worst_views)})

        # Representative rows for four-scene sheets: median Delta PSNR.
        median_view = sorted(best_order)[len(best_order) // 2][1]
        m1 = m1_by_view[median_view]
        bnd = bnd_by_view[median_view]
        gt = m1["gt"]
        e_m1 = _norm_rgb(m1["pred_image"] - gt)
        e_bnd = _norm_rgb(bnd["pred_image"] - gt)
        summary_underwater_rows.append(
            [
                (f"{scene} {median_view} GT", _rgb_to_uint8(gt)),
                ("M1", _rgb_to_uint8(m1["pred_image"])),
                ("BND", _rgb_to_uint8(bnd["pred_image"])),
                ("M1 residual", _scalar_to_uint8(e_m1, scene_resid_max)),
                ("BND residual", _scalar_to_uint8(e_bnd, scene_resid_max)),
            ]
        )
        summary_decomp_rows.append(
            [
                (f"{scene} {median_view} M1 clear", _rgb_to_uint8(m1["clear_object_fullsh_raw"])),
                ("BND clear", _rgb_to_uint8(bnd["clear_object_fullsh_raw"])),
                ("M1 tau", _scalar_to_uint8(m1["tau_D"].mean(dim=-1), scene_tau_max)),
                ("BND tau", _scalar_to_uint8(bnd["tau_D"].mean(dim=-1), scene_tau_max)),
            ]
        )
        _release_loaded(bnd_loaded)

    _save_sheet(render_dir / "four_scene_rgb_tradeoff_summary.png", summary_underwater_rows, tile_width=tile_width)
    _save_sheet(render_dir / "four_scene_decomposition_summary.png", summary_decomp_rows, tile_width=tile_width)
    manifest.extend(
        [
            {"file_path": str(render_dir / "four_scene_rgb_tradeoff_summary.png"), "scene": "ALL", "output_type": "four_scene_rgb_tradeoff_summary"},
            {"file_path": str(render_dir / "four_scene_decomposition_summary.png"), "scene": "ALL", "output_type": "four_scene_decomposition_summary"},
        ]
    )
    _write_json(render_dir / "manifest.json", manifest)
    _write_csv(render_dir / "manifest.csv", manifest)
    index_lines = ["# BND RGB Trade-off Visual Compare Index", ""]
    for item in manifest:
        index_lines.append(f"- {item.get('scene')}: {item.get('output_type')} - `{item.get('file_path')}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf8")
    return per_view_rows, enrichment_rows, component_rows, hybrid_rows, jacobian_rows, sh_rows, strat_rows, checkpoint_rows, manifest


def _stratification_rows(
    scene: str,
    view_id: str,
    m1: Mapping[str, torch.Tensor],
    bnd: Mapping[str, torch.Tensor],
    gt: torch.Tensor,
    e_m1: torch.Tensor,
    e_bnd: torch.Tensor,
    excess: torch.Tensor,
) -> List[Dict[str, Any]]:
    lum_w = torch.tensor([0.2126, 0.7152, 0.0722])
    gt_lum = (gt.float() * lum_w).sum(dim=-1)
    m1_lum = (m1["pred_image"].float() * lum_w).sum(dim=-1)
    depth = m1["depth"].float().squeeze(-1)
    accum = m1["accumulation"].float().squeeze(-1)
    h, w = gt_lum.shape
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
    radial = torch.sqrt(xx.square() + yy.square())
    delta_e = e_bnd - e_m1
    delta_sq = (bnd["pred_image"] - gt).square().mean(dim=-1) - (m1["pred_image"] - gt).square().mean(dim=-1)
    fields = {
        "GT_luminance": gt_lum,
        "M1_luminance": m1_lum,
        "depth": depth,
        "accumulation": accum,
        "radial_position": radial,
    }
    rows: List[Dict[str, Any]] = []
    total_excess = float(excess.sum().item())
    for name, values in fields.items():
        flat = values.reshape(-1)
        qs = torch.quantile(flat.float(), torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        for idx in range(5):
            lo = qs[idx]
            hi = qs[idx + 1]
            if idx == 4:
                mask = (values >= lo) & (values <= hi)
            else:
                mask = (values >= lo) & (values < hi)
            if not bool(mask.any()):
                continue
            rows.append(
                {
                    "scene": scene,
                    "view_id": view_id,
                    "stratification": name,
                    "bin": idx,
                    "bin_min": float(lo.item()),
                    "bin_max": float(hi.item()),
                    "pixel_fraction": float(mask.float().mean().item()),
                    "mean_delta_e": float(delta_e[mask].mean().item()),
                    "mean_delta_mse": float(delta_sq[mask].mean().item()),
                    "positive_excess_fraction": float(excess[mask].sum().item() / total_excess) if total_excess > 1e-12 else 0.0,
                }
            )
    return rows


def _append_grad_tensor_stats(row: Dict[str, Any], label: str, grads: Sequence[torch.Tensor]) -> None:
    valid = [grad.detach().float().reshape(-1) for grad in grads if grad is not None and grad.numel() > 0]
    if not valid:
        row[f"{label}_grad_param_count"] = 0
        row[f"{label}_grad_l2"] = 0.0
        row[f"{label}_grad_mean_abs"] = 0.0
        row[f"{label}_grad_p95_abs"] = 0.0
        row[f"{label}_grad_l2_per_param"] = 0.0
        return
    l2_sq = sum(float(grad.square().sum().item()) for grad in valid)
    count = sum(grad.numel() for grad in valid)
    abs_flat = torch.cat([grad.abs() for grad in valid], dim=0)
    row[f"{label}_grad_param_count"] = int(count)
    row[f"{label}_grad_l2"] = math.sqrt(l2_sq)
    row[f"{label}_grad_mean_abs"] = float(abs_flat.mean().item())
    row[f"{label}_grad_p95_abs"] = _quantile(abs_flat, 0.95)
    row[f"{label}_grad_l2_per_param"] = row[f"{label}_grad_l2"] / math.sqrt(max(1, count))


def _append_medium_param_grad_stats(row: Dict[str, Any], model: Any) -> None:
    medium_grads: List[torch.Tensor] = []
    direction_grads: List[torch.Tensor] = []
    medium_names: List[str] = []
    direction_names: List[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.startswith("medium_mlp"):
            medium_grads.append(param.grad)
            medium_names.append(name)
        elif name.startswith("direction_encoding"):
            direction_grads.append(param.grad)
            direction_names.append(name)
    _append_grad_tensor_stats(row, "medium_mlp_params", medium_grads)
    _append_grad_tensor_stats(row, "direction_encoding_params", direction_grads)
    row["medium_mlp_grad_parameter_names"] = ";".join(medium_names)
    row["direction_encoding_grad_parameter_names"] = ";".join(direction_names)

    # Only a plain Linear medium head exposes branch-specific output rows.
    weight = getattr(getattr(model, "medium_mlp", None), "weight", None)
    if isinstance(weight, torch.Tensor) and weight.grad is not None and weight.grad.ndim >= 2 and weight.grad.shape[0] == 9:
        bias = getattr(model.medium_mlp, "bias", None)
        bias_grad = bias.grad if isinstance(bias, torch.Tensor) else None
        for label, slc in (
            ("medium_rgb_head_params", slice(0, 3)),
            ("medium_bs_head_params", slice(3, 6)),
            ("medium_attn_head_params", slice(6, 9)),
        ):
            grads = [weight.grad[slc]]
            if bias_grad is not None:
                grads.append(bias_grad[slc])
            _append_grad_tensor_stats(row, label, grads)


def gradient_audit(repo: Path, scenes: Sequence[str], per_view_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    worst_by_scene: Dict[str, List[str]] = {}
    for scene in scenes:
        scene_rows = [row for row in per_view_rows if row["scene"] == scene]
        worst = [str(row["view_id"]) for row in sorted(scene_rows, key=lambda x: float(x["Delta_PSNR"]))[:3]]
        reps = [str(row["view_id"]) for row in sorted(scene_rows, key=lambda x: abs(float(x["Delta_PSNR"])))[:2]]
        selected = []
        for item in worst + reps:
            if item not in selected:
                selected.append(item)
        worst_by_scene[scene] = selected
    rows: List[Dict[str, Any]] = []
    for scene in scenes:
        for run in ("M1", "BND"):
            loaded: Optional[LoadedRun] = None
            try:
                loaded = _load_run(repo, scene, run, FINAL_NOMINAL_STEP)
                model = loaded.model
                model.eval()
                views = {view_id: (camera, batch) for _, view_id, camera, batch in _view_records(loaded)}
                for view_id in worst_by_scene[scene]:
                    camera, batch = views[view_id]
                    row: Dict[str, Any] = {"scene": scene, "run": run, "view_id": view_id, "status": "OK"}
                    try:
                        model.zero_grad(set_to_none=True)
                        outputs = model.get_outputs(camera.to(model.device))
                        for key in ("medium_rgb", "medium_bs", "medium_attn", "direct_object_signal", "rgb_medium"):
                            if key in outputs and isinstance(outputs[key], torch.Tensor) and outputs[key].requires_grad:
                                outputs[key].retain_grad()
                        if run == "BND" and "gaussian_view_logits" in outputs:
                            outputs["gaussian_view_logits"].retain_grad()
                        loss_dict = model.get_loss_dict(outputs, batch)
                        loss = loss_dict["main_loss"]
                        loss.backward()
                        row["loss"] = float(loss.detach().item())
                        for name, param in (
                            ("features_dc", model.features_dc),
                            ("features_rest", model.features_rest),
                        ):
                            grad = param.grad
                            if grad is None:
                                row[f"{name}_grad_l2"] = 0.0
                                row[f"{name}_grad_mean_abs"] = 0.0
                                row[f"{name}_grad_p95_abs"] = 0.0
                                row[f"{name}_grad_l2_per_param"] = 0.0
                            else:
                                flat = grad.detach().float().abs().reshape(-1)
                                row[f"{name}_grad_l2"] = float(torch.linalg.norm(grad.detach().float()).item())
                                row[f"{name}_grad_mean_abs"] = float(flat.mean().item())
                                row[f"{name}_grad_p95_abs"] = _quantile(flat, 0.95)
                                row[f"{name}_grad_l2_per_param"] = row[f"{name}_grad_l2"] / math.sqrt(max(1, flat.numel()))
                        _append_medium_param_grad_stats(row, model)
                        for key, label in (
                            ("medium_rgb", "medium_rgb_output"),
                            ("medium_bs", "medium_bs_output"),
                            ("medium_attn", "medium_attn_output"),
                            ("direct_object_signal", "direct_output"),
                            ("rgb_medium", "medium_render_output"),
                        ):
                            tensor = outputs.get(key)
                            grad = getattr(tensor, "grad", None) if isinstance(tensor, torch.Tensor) else None
                            if grad is not None:
                                flat = grad.detach().float().abs().reshape(-1)
                                row[f"{label}_grad_l2"] = float(torch.linalg.norm(grad.detach().float()).item())
                                row[f"{label}_grad_mean_abs"] = float(flat.mean().item())
                                row[f"{label}_grad_p95_abs"] = _quantile(flat, 0.95)
                        if run == "BND" and "gaussian_view_logits" in outputs:
                            logits = outputs["gaussian_view_logits"]
                            grad_s = logits.grad
                            if grad_s is not None:
                                deriv = outputs["gaussian_sigmoid_derivative"].detach().float().clamp_min(1e-6)
                                grad_c_est = grad_s.detach().float() / deriv
                                row["dL_ds_l2"] = float(torch.linalg.norm(grad_s.detach().float()).item())
                                row["dL_dc_est_l2"] = float(torch.linalg.norm(grad_c_est).item())
                                row["dL_ds_over_dL_dc_est"] = row["dL_ds_l2"] / max(row["dL_dc_est_l2"], 1e-12)
                                row["sigmoid_derivative_median_for_grad"] = _quantile(deriv, 0.50)
                    except RuntimeError as exc:
                        row["status"] = "FAILED"
                        row["error"] = str(exc)
                        row["traceback"] = traceback.format_exc(limit=3)
                        torch.cuda.empty_cache()
                    finally:
                        model.zero_grad(set_to_none=True)
                    rows.append(row)
            finally:
                _release_loaded(loaded)
    _write_json(output_dir / "bnd_no_step_gradient_audit.json", rows)
    _write_csv(output_dir / "bnd_no_step_gradient_audit.csv", rows)
    return rows


def _scene_aggregate(rows: Sequence[Mapping[str, Any]], scene: str, key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get("scene") == scene and row.get("view_id") != "AGGREGATE" and key in row]
    return _mean(vals)


def final_summary(
    scenes: Sequence[str],
    trajectory_rows: Sequence[Mapping[str, Any]],
    late_flags: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    enrichment_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    jacobian_rows: Sequence[Mapping[str, Any]],
    sh_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    traj = {(row["scene"], row["run"], int(row["nominal_step"])): row for row in trajectory_rows}
    late = {row["scene"]: row for row in late_flags}
    comp_agg = {row["scene"]: row for row in component_rows if row.get("view_id") == "AGGREGATE"}
    enrich_agg = {(row["scene"], row["mask"]): row for row in enrichment_rows if row.get("view_id") == "AGGREGATE"}
    rows: List[Dict[str, Any]] = []
    for scene in scenes:
        m1 = traj.get((scene, "M1", 15000), {})
        bnd = traj.get((scene, "BND", 15000), {})
        comp = comp_agg.get(scene, {})
        comp_mask = enrich_agg.get((scene, "COMP"), {})
        delta_psnr = float(bnd.get("psnr", float("nan"))) - float(m1.get("psnr", float("nan")))
        tau_drop = 1.0 - float(bnd.get("tau_D_all_p90", float("nan"))) / max(float(m1.get("tau_D_all_p90", 1.0)), 1e-12)
        j_drop = 1.0 - float(bnd.get("J_all_p99", float("nan"))) / max(float(m1.get("J_all_p99", 1.0)), 1e-12)
        loss_conc = bool(
            float(comp_mask.get("residual_enrichment", 0.0)) >= 1.5
            and float(comp_mask.get("excess_error_fraction", 0.0)) >= 0.35
        )
        partial_conc = bool(
            not loss_conc
            and float(comp_mask.get("residual_enrichment", 0.0)) >= 1.2
            and float(comp_mask.get("excess_error_fraction", 0.0)) >= 0.25
        )
        c_direct = float(comp.get("C_direct", 0.0))
        c_medium = float(comp.get("C_medium", 0.0))
        c_cross = float(comp.get("C_cross", 0.0))
        direct_dom = c_direct > max(c_medium, c_cross, 0.0) * 1.25 and c_direct > 0
        medium_dom = c_medium > max(c_direct, c_cross, 0.0) * 1.25 and c_medium > 0
        cross_dom = c_cross > max(c_direct, c_medium, 0.0) * 1.25 and c_cross > 0
        deriv_med = _scene_aggregate(jacobian_rows, scene, "sigmoid_derivative_all_p50")
        jac_factor = 1.0 / deriv_med if deriv_med and deriv_med == deriv_med else float("nan")
        r_m1 = _mean(float(row["R_SH_visible_p50"]) for row in sh_rows if row.get("scene") == scene and row.get("run") == "M1" and "R_SH_visible_p50" in row)
        r_bnd = _mean(float(row["R_SH_visible_p50"]) for row in sh_rows if row.get("scene") == scene and row.get("run") == "BND" and "R_SH_visible_p50" in row)
        sh_ratio = r_bnd / max(r_m1, 1e-12)
        grad_bnd_rest = _mean(float(row.get("features_rest_grad_l2_per_param", 0.0)) for row in gradient_rows if row.get("scene") == scene and row.get("run") == "BND" and row.get("status") == "OK")
        sigmoid_limit = bool(deriv_med == deriv_med and deriv_med < 0.23 and sh_ratio < 0.75)
        if delta_psnr < -0.15:
            if bool(late.get(scene, {}).get("POSSIBLE_UNDERCONVERGENCE", False)):
                root = "UNDERCONVERGENCE"
            elif loss_conc and direct_dom:
                root = "LEGACY_COMPENSATION_REMOVAL"
            elif sigmoid_limit and direct_dom:
                root = "SH_OPTIMIZATION_LIMIT"
            elif medium_dom:
                root = "MEDIUM_LIMIT"
            elif partial_conc or sigmoid_limit:
                root = "MIXED"
            else:
                root = "DISTRIBUTED_TRADEOFF"
        elif scene == "JapaneseGradens" and float(bnd.get("ssim", 0.0)) - float(m1.get("ssim", 0.0)) < -0.0015:
            root = "MIXED" if (partial_conc or sigmoid_limit or direct_dom or medium_dom) else "DISTRIBUTED_TRADEOFF"
        else:
            root = "MIXED" if sigmoid_limit else "DISTRIBUTED_TRADEOFF"
        rows.append(
            {
                "scene": scene,
                "Delta_PSNR": delta_psnr,
                "Delta_SSIM": float(bnd.get("ssim", float("nan"))) - float(m1.get("ssim", float("nan"))),
                "Delta_LPIPS": float(bnd.get("lpips", float("nan"))) - float(m1.get("lpips", float("nan"))),
                "tau_p90_reduction": tau_drop,
                "J_p99_reduction": j_drop,
                "MECHANISM_STILL_VALID": tau_drop > 0.15 and j_drop > 0.15,
                "POSSIBLE_UNDERCONVERGENCE": bool(late.get(scene, {}).get("POSSIBLE_UNDERCONVERGENCE", False)),
                "LOSS_CONCENTRATED_IN_LEGACY_COMPENSATION_REGIONS": loss_conc,
                "PARTIAL_CONCENTRATION": partial_conc,
                "DIRECT_DOMINATED_RGB_LOSS": direct_dom,
                "MEDIUM_DOMINATED_RGB_LOSS": medium_dom,
                "CROSS_TERM_DOMINATED": cross_dom,
                "SIGMOID_JACOBIAN_OPTIMIZATION_LIMIT": sigmoid_limit,
                "DISTRIBUTED_RGB_TRADEOFF": root == "DISTRIBUTED_TRADEOFF",
                "MIXED_ROOT_CAUSE": root == "MIXED",
                "ROOT_CAUSE_PRIMARY": root,
                "COMP_mask_area": comp_mask.get("mask_area", float("nan")),
                "COMP_excess_error_fraction": comp_mask.get("excess_error_fraction", float("nan")),
                "COMP_residual_enrichment": comp_mask.get("residual_enrichment", float("nan")),
                "C_direct": c_direct,
                "C_medium": c_medium,
                "C_cross": c_cross,
                "sigmoid_derivative_median": deriv_med,
                "approx_jacobian_compensation_factor": jac_factor,
                "R_SH_visible_p50_M1": r_m1,
                "R_SH_visible_p50_BND": r_bnd,
                "R_SH_visible_p50_BND_over_M1": sh_ratio,
                "BND_features_rest_grad_l2_per_param_mean": grad_bnd_rest,
            }
        )
    return rows


def _write_visual_index(render_dir: Path) -> None:
    lines = ["# BND RGB Trade-off Visual Assets", ""]
    for path in sorted(render_dir.rglob("*.png")):
        lines.append(f"- `{path}`")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_rgb_tradeoff_diagnosis_20260809"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_rgb_tradeoff_diagnosis_20260809"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/bnd_rgb_tradeoff_diagnosis_20260809"))
    parser.add_argument("--scenes", nargs="+", default=["Curasao", "JapaneseGradens", "IUI3", "Panama"])
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--skip-gradients", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    logs_dir = (repo / args.logs_dir).resolve() if not args.logs_dir.is_absolute() else args.logs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows, missing_rows, feature_rows = trajectory_audit(repo, args.scenes, output_dir)
    trajectory_delta = _delta_rows(trajectory_rows)
    late_flags = _late_recovery_flags(trajectory_rows)

    (
        per_view_rows,
        enrichment_rows,
        component_rows,
        hybrid_rows,
        jacobian_rows,
        sh_rows,
        strat_rows,
        checkpoint_rows,
        manifest_rows,
    ) = final_pair_audit(repo, args.scenes, output_dir, render_dir, args.tile_width)

    gradient_rows: List[Dict[str, Any]] = []
    if not args.skip_gradients:
        gradient_rows = gradient_audit(repo, args.scenes, per_view_rows, output_dir)

    summary_rows = final_summary(
        args.scenes,
        trajectory_rows,
        late_flags,
        per_view_rows,
        enrichment_rows,
        component_rows,
        jacobian_rows,
        sh_rows,
        gradient_rows,
    )

    _write_json(output_dir / "bnd_trajectory_audit.json", {"rows": trajectory_rows, "missing": missing_rows, "delta": trajectory_delta, "late_flags": late_flags})
    _write_csv(output_dir / "bnd_trajectory_audit.csv", trajectory_rows)
    _write_csv(output_dir / "bnd_trajectory_delta.csv", trajectory_delta)
    _write_csv(output_dir / "bnd_missing_checkpoints.csv", missing_rows)
    _write_json(output_dir / "bnd_per_view_rgb_delta.json", per_view_rows)
    _write_csv(output_dir / "bnd_per_view_rgb_delta.csv", per_view_rows)
    _write_json(output_dir / "bnd_residual_enrichment.json", enrichment_rows)
    _write_csv(output_dir / "bnd_residual_enrichment.csv", enrichment_rows)
    _write_json(output_dir / "bnd_component_mse_attribution.json", component_rows)
    _write_csv(output_dir / "bnd_component_mse_attribution.csv", component_rows)
    _write_json(output_dir / "bnd_hybrid_metrics.json", hybrid_rows)
    _write_csv(output_dir / "bnd_hybrid_metrics.csv", hybrid_rows)
    _write_json(output_dir / "bnd_sigmoid_jacobian_audit.json", jacobian_rows)
    _write_csv(output_dir / "bnd_sigmoid_jacobian_audit.csv", jacobian_rows)
    _write_json(output_dir / "bnd_sh_capacity_audit.json", sh_rows)
    _write_csv(output_dir / "bnd_sh_capacity_audit.csv", sh_rows)
    _write_json(output_dir / "bnd_features_parameter_audit.json", feature_rows)
    _write_csv(output_dir / "bnd_features_parameter_audit.csv", feature_rows)
    _write_json(output_dir / "bnd_stratified_residual_audit.json", strat_rows)
    _write_csv(output_dir / "bnd_stratified_residual_audit.csv", strat_rows)
    _write_json(output_dir / "bnd_checkpoint_audit.json", checkpoint_rows)
    _write_csv(output_dir / "bnd_checkpoint_audit.csv", checkpoint_rows)
    _write_json(output_dir / "bnd_rgb_tradeoff_final_summary.json", summary_rows)
    _write_csv(output_dir / "bnd_rgb_tradeoff_final_summary.csv", summary_rows)
    _write_json(output_dir / "manifest.json", {"outputs": sorted(str(p) for p in output_dir.rglob("*") if p.is_file()), "renders": manifest_rows})
    _write_visual_index(render_dir)
    print(json.dumps({"summary": summary_rows, "output_dir": str(output_dir), "render_dir": str(render_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
