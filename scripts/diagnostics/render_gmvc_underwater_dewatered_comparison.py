#!/usr/bin/env python
"""Render underwater and dewatered GMVC comparisons on eval cameras."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


METRIC_KEYS = ("psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1")
COMPONENT_KEYS = (
    "J_proxy_raw",
    "J_gaussian_raw",
    "J_raw",
    "transmission",
    "backscatter_endpoint",
    "rgb_medium",
    "b_inf",
    "medium_attn",
    "medium_bs",
    "depth",
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
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


def _luma(rgb: Tensor) -> Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _extra_metrics(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0)
    gt = gt.detach().float().clamp(0.0, 1.0)
    residual = pred - gt
    pred_luma = _luma(pred)
    gt_luma = _luma(gt)
    chroma_residual = (pred - pred_luma) - (gt - gt_luma)
    return {
        "rgb_l1": float(residual.abs().mean().item()),
        "luminance_l1": float((pred_luma - gt_luma).abs().mean().item()),
        "chroma_l1": float(chroma_residual.abs().mean().item()),
    }


def _image_name(pipeline: Any, image_idx: int) -> str:
    dataset = pipeline.datamanager.eval_dataset
    try:
        filenames = dataset._dataparser_outputs.image_filenames
        return Path(filenames[int(image_idx)]).name
    except Exception:
        return f"eval_{int(image_idx):04d}"


def _as_hwc(value: Tensor, key: str) -> Tensor:
    out = value.detach().float()
    if out.ndim == 2:
        out = out[..., None]
    if out.ndim != 3:
        raise ValueError(f"{key} must be HxW or HxWxC, got {tuple(out.shape)}")
    if out.shape[-1] == 1:
        return out
    if out.shape[-1] == 3:
        return out
    return out.mean(dim=-1, keepdim=True)


def _finite_flat(value: Tensor) -> Tensor:
    flat = value.detach().float().reshape(-1).cpu()
    return flat[torch.isfinite(flat)]


def _nearest_rank(flat: Tensor, q: float) -> float:
    flat = _finite_flat(flat)
    if flat.numel() == 0:
        return 0.0
    rank = max(1, min(int(flat.numel()), math.ceil(float(q) * int(flat.numel()))))
    return float(flat.kthvalue(rank).values.item())


def _range(values: Iterable[Tensor]) -> Dict[str, float]:
    finite = [_finite_flat(value) for value in values]
    finite = [value for value in finite if value.numel() > 0]
    if not finite:
        return {"p01": 0.0, "p50": 0.0, "p99": 1.0}
    flat = torch.cat(finite)
    p01 = _nearest_rank(flat, 0.01)
    p50 = _nearest_rank(flat, 0.50)
    p99 = _nearest_rank(flat, 0.99)
    if p99 <= p01:
        p99 = p01 + 1e-6
    return {"p01": p01, "p50": p50, "p99": p99}


def _normalize(value: Tensor, vis_range: Dict[str, float]) -> Tensor:
    lo = float(vis_range["p01"])
    hi = float(vis_range["p99"])
    return ((value.detach().float() - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)


def _to_rgb_vis(value: Tensor, vis_range: Dict[str, float] | None = None) -> Tensor:
    image = _as_hwc(value, "vis")
    if vis_range is not None:
        image = _normalize(image, vis_range)
    else:
        image = image.clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    return image


def _to_uint8(image: Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, image: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(_to_rgb_vis(image))).save(path)


def _save_vis_png(path: Path, image: Tensor, vis_range: Dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(_to_rgb_vis(image, vis_range))).save(path)


def _resize_for_sheet(image: Tensor, tile_width: int) -> Image.Image:
    pil = Image.fromarray(_to_uint8(_to_rgb_vis(image)))
    if tile_width > 0 and pil.width > tile_width:
        height = max(1, int(round(pil.height * (tile_width / pil.width))))
        pil = pil.resize((tile_width, height), Image.Resampling.BILINEAR)
    return pil


def _label_tile(tile: Image.Image, label: str) -> Image.Image:
    pad = 22
    canvas = Image.new("RGB", (tile.width, tile.height + pad), (255, 255, 255))
    canvas.paste(tile.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return canvas


def _stack_grid(rows: List[List[Tuple[str, Tensor]]], tile_width: int) -> Image.Image:
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_label_tile(_resize_for_sheet(image, tile_width), label) for label, image in row]
        height = max(tile.height for tile in tiles)
        width = sum(tile.width for tile in tiles)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width
        rendered_rows.append(canvas)
    total_width = max(row.width for row in rendered_rows)
    total_height = sum(row.height for row in rendered_rows)
    sheet = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


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


def _component_payload(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    depth = _as_hwc(outputs["depth"], "depth")
    if depth.shape[-1] != 1:
        depth = depth.mean(dim=-1, keepdim=True)
    medium_attn = _as_hwc(outputs["medium_attn"], "medium_attn")
    medium_bs = _as_hwc(outputs["medium_bs"], "medium_bs")
    b_inf = _as_hwc(outputs.get("b_inf", outputs["medium_rgb"]), "b_inf")
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter_endpoint = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    payload = {
        "J_gaussian_raw": _as_hwc(outputs["J_gaussian_raw"], "J_gaussian_raw"),
        "J_raw": _as_hwc(outputs["J_raw"], "J_raw"),
        "transmission": transmission.detach(),
        "backscatter_endpoint": backscatter_endpoint.detach(),
        "rgb_medium": _as_hwc(outputs["rgb_medium"], "rgb_medium"),
        "b_inf": b_inf,
        "medium_attn": medium_attn,
        "medium_bs": medium_bs,
        "depth": depth,
    }
    if "J_proxy_raw" in outputs:
        payload["J_proxy_raw"] = _as_hwc(outputs["J_proxy_raw"], "J_proxy_raw")
    return {key: value.detach().float().cpu() for key, value in payload.items()}


def _eval_checkpoint(label: str, config_path: Path, step: int, args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, step, args.test_mode)
    model = pipeline.model
    model.eval()
    rows: List[Dict[str, Any]] = []
    rendered: Dict[int, Dict[str, Any]] = {}
    data_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    if args.max_images > 0:
        data_loader = data_loader[: int(args.max_images)]
    with torch.no_grad():
        for view_index, (camera, batch) in enumerate(data_loader):
            outputs = model.get_outputs_for_camera(camera=camera)
            metrics, images = model.get_image_metrics_and_images(outputs, batch)
            pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0).cpu()
            gt = images["gt"].detach().float().clamp(0.0, 1.0).cpu()
            image_idx_raw = batch.get("image_idx", view_index)
            image_idx = int(image_idx_raw.item() if torch.is_tensor(image_idx_raw) else image_idx_raw)
            extra = _extra_metrics(pred, gt)
            row = {
                "view_index": int(view_index),
                "image_idx": image_idx,
                "image_name": _image_name(pipeline, image_idx),
                "psnr": float(metrics.get("psnr", 0.0)),
                "ssim": float(metrics.get("ssim", 0.0)),
                "lpips": float(metrics.get("lpips", 0.0)),
                **extra,
            }
            rows.append(row)

            saved = _force_dc_proxy_context(model) if args.force_dc_proxy else None
            try:
                proxy_outputs = model.get_outputs_for_camera(camera=camera)
                components = _component_payload(proxy_outputs)
            finally:
                if saved is not None:
                    _restore_context(model, saved)
            rendered[int(view_index)] = {
                "gt": gt,
                "pred": pred,
                "components": components,
                "row": row,
            }
    result = {
        "label": label,
        "config": str(config_path),
        "requested_step": int(step),
        "step": int(loaded_step),
        "checkpoint": str(checkpoint_path),
        "metrics": rows,
        "rendered": rendered,
        "experiment_name": getattr(config, "experiment_name", ""),
    }
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _mean_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {key: float(sum(float(row[key]) for row in rows) / max(len(rows), 1)) for key in METRIC_KEYS}


def _delta(value: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value[key] - base[key]) for key in METRIC_KEYS}


def _select_views(view_indices: List[int]) -> List[int]:
    if len(view_indices) <= 3:
        return view_indices
    return [view_indices[0], view_indices[len(view_indices) // 2], view_indices[-1]]


def _component_ranges(evaluated: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[Tensor]] = {key: [] for key in COMPONENT_KEYS}
    for result in evaluated.values():
        for payload in result["rendered"].values():
            for key, value in payload["components"].items():
                if key in values:
                    values[key].append(value)
    return {key: _range(group) for key, group in values.items() if group}


def _write_view_outputs(
    output_dir: Path,
    view_index: int,
    evaluated: Dict[str, Dict[str, Any]],
    ranges: Dict[str, Dict[str, float]],
    reference_run: str,
) -> Dict[str, Any]:
    view_dir = output_dir / f"view_{view_index:04d}"
    underwater_dir = view_dir / "underwater"
    dewatered_dir = view_dir / "dewatered"
    components_dir = view_dir / "components"
    reference = evaluated[reference_run]["rendered"][view_index]
    gt = reference["gt"]
    _save_png(underwater_dir / "gt.png", gt)
    outputs: Dict[str, Any] = {"gt": str(underwater_dir / "gt.png")}
    residuals: Dict[str, Tensor] = {}
    preds: Dict[str, Tensor] = {}
    dewatered: Dict[str, Tensor] = {}

    for label, result in evaluated.items():
        slug = label.lower()
        payload = result["rendered"][view_index]
        pred = payload["pred"]
        preds[label] = pred
        residual = (pred - gt).abs()
        residuals[label] = residual
        _save_png(underwater_dir / f"{slug}_underwater.png", pred)
        _save_png(underwater_dir / f"{slug}_abs_residual.png", residual)
        outputs[f"{slug}_underwater"] = str(underwater_dir / f"{slug}_underwater.png")
        outputs[f"{slug}_abs_residual"] = str(underwater_dir / f"{slug}_abs_residual.png")

        components = payload["components"]
        clean = components.get("J_proxy_raw", components.get("J_gaussian_raw", components["J_raw"]))
        dewatered[label] = clean.clamp(0.0, 1.0)
        _save_png(dewatered_dir / f"{slug}_j_proxy_or_clear.png", clean.clamp(0.0, 1.0))
        outputs[f"{slug}_j_proxy_or_clear"] = str(dewatered_dir / f"{slug}_j_proxy_or_clear.png")
        for comp_key, comp_value in components.items():
            vis_range = ranges.get(comp_key)
            _save_vis_png(components_dir / f"{slug}_{comp_key}.png", comp_value, vis_range)
            outputs[f"{slug}_{comp_key}"] = str(components_dir / f"{slug}_{comp_key}.png")

    if reference_run in residuals:
        ref_residual = residuals[reference_run].mean(dim=-1, keepdim=True)
        for label, residual in residuals.items():
            if label == reference_run:
                continue
            diff = (0.5 + 4.0 * (residual.mean(dim=-1, keepdim=True) - ref_residual)).clamp(0.0, 1.0)
            diff = diff.expand_as(gt)
            path = underwater_dir / f"{label.lower()}_minus_{reference_run.lower()}_abs_residual_diff.png"
            _save_png(path, diff)
            outputs[f"{label.lower()}_minus_{reference_run.lower()}_abs_residual_diff"] = str(path)

    if reference_run in dewatered:
        ref_clean = dewatered[reference_run]
        for label, clean in dewatered.items():
            if label == reference_run:
                continue
            diff = (clean - ref_clean).abs().clamp(0.0, 1.0)
            path = dewatered_dir / f"{label.lower()}_minus_{reference_run.lower()}_dewatered_abs_diff.png"
            _save_png(path, diff)
            outputs[f"{label.lower()}_minus_{reference_run.lower()}_dewatered_abs_diff"] = str(path)

    return outputs


def _write_contact_sheets(
    output_dir: Path,
    selected_views: List[int],
    evaluated: Dict[str, Dict[str, Any]],
    ranges: Dict[str, Dict[str, float]],
    reference_run: str,
    tile_width: int,
) -> Dict[str, str]:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    underwater_rows: List[List[Tuple[str, Tensor]]] = []
    dewatered_rows: List[List[Tuple[str, Tensor]]] = []
    component_rows: List[List[Tuple[str, Tensor]]] = []

    for view_index in selected_views:
        gt = evaluated[reference_run]["rendered"][view_index]["gt"]
        underwater_row: List[Tuple[str, Tensor]] = [(f"view {view_index} GT", gt)]
        dewatered_row: List[Tuple[str, Tensor]] = []
        component_row: List[Tuple[str, Tensor]] = []
        reference_residual = None
        reference_clean = None
        if reference_run in evaluated:
            reference_pred = evaluated[reference_run]["rendered"][view_index]["pred"]
            reference_residual = (reference_pred - gt).abs().mean(dim=-1, keepdim=True)
            ref_components = evaluated[reference_run]["rendered"][view_index]["components"]
            reference_clean = ref_components.get("J_proxy_raw", ref_components.get("J_gaussian_raw", ref_components["J_raw"])).clamp(0.0, 1.0)
        for label, result in evaluated.items():
            pred = result["rendered"][view_index]["pred"]
            residual = (pred - gt).abs()
            underwater_row.append((f"{label} underwater", pred))
            underwater_row.append((f"{label} residual", residual))
            components = result["rendered"][view_index]["components"]
            clean = components.get("J_proxy_raw", components.get("J_gaussian_raw", components["J_raw"])).clamp(0.0, 1.0)
            dewatered_row.append((f"view {view_index} {label} clean", clean))
            if label != reference_run and reference_residual is not None:
                diff = (0.5 + 4.0 * (residual.mean(dim=-1, keepdim=True) - reference_residual)).expand_as(gt).clamp(0.0, 1.0)
                underwater_row.append((f"{label}-{reference_run} residual", diff))
            if label != reference_run and reference_clean is not None:
                dewatered_row.append((f"{label}-{reference_run} clean abs", (clean - reference_clean).abs()))

        p30_label = "P30-MHOLD" if "P30-MHOLD" in evaluated else next(reversed(evaluated))
        p30_components = evaluated[p30_label]["rendered"][view_index]["components"]
        component_row.extend(
            [
                (f"view {view_index} depth", _to_rgb_vis(p30_components["depth"], ranges["depth"])),
                (f"{p30_label} transmission", _to_rgb_vis(p30_components["transmission"], ranges["transmission"])),
                (f"{p30_label} backscatter", _to_rgb_vis(p30_components["backscatter_endpoint"], ranges["backscatter_endpoint"])),
                (f"{p30_label} B_inf", _to_rgb_vis(p30_components["b_inf"], ranges["b_inf"])),
                (f"{p30_label} medium_attn", _to_rgb_vis(p30_components["medium_attn"], ranges["medium_attn"])),
                (f"{p30_label} medium_bs", _to_rgb_vis(p30_components["medium_bs"], ranges["medium_bs"])),
            ]
        )
        underwater_rows.append(underwater_row)
        dewatered_rows.append(dewatered_row)
        component_rows.append(component_row)

    underwater_path = contact_dir / "underwater_contact_sheet.png"
    dewatered_path = contact_dir / "dewatered_contact_sheet.png"
    components_path = contact_dir / "medium_components_contact_sheet.png"
    _stack_grid(underwater_rows, tile_width).save(underwater_path)
    _stack_grid(dewatered_rows, tile_width).save(dewatered_path)
    _stack_grid(component_rows, tile_width).save(components_path)
    return {
        "underwater": str(underwater_path),
        "dewatered": str(dewatered_path),
        "medium_components": str(components_path),
    }


def _parse_run_spec(value: str) -> Tuple[str, Path, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    label, rest = value.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    config_text, step_text = rest.rsplit(":", 1)
    return label.strip(), Path(config_text), int(step_text)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_specs = [("A0", args.a0_config, args.a0_step), ("P30-MHOLD", args.p30_config, args.p30_step)]
    run_specs.extend(_parse_run_spec(item) for item in args.run)
    evaluated: Dict[str, Dict[str, Any]] = {}
    for label, config_path, step in run_specs:
        evaluated[label] = _eval_checkpoint(label, config_path, step, args)

    ranges = _component_ranges(evaluated)
    a0_views = sorted(evaluated["A0"]["rendered"].keys())
    selected_views = _select_views(a0_views)
    per_view: List[Dict[str, Any]] = []
    for view_index in a0_views:
        outputs = _write_view_outputs(args.output_dir, view_index, evaluated, ranges, args.reference_run)
        row = {
            "view_index": int(view_index),
            "image_idx": int(evaluated["A0"]["rendered"][view_index]["row"]["image_idx"]),
            "image_name": evaluated["A0"]["rendered"][view_index]["row"]["image_name"],
            "runs": {
                label: {key: evaluated[label]["rendered"][view_index]["row"][key] for key in METRIC_KEYS}
                for label in evaluated
            },
            "outputs": outputs,
        }
        row["delta_vs_a0"] = {
            label: _delta(metrics, row["runs"]["A0"])
            for label, metrics in row["runs"].items()
            if label != "A0"
        }
        per_view.append(row)
        view_dir = args.output_dir / f"view_{view_index:04d}"
        (view_dir / "per_view_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf8")

    means = {label: _mean_metrics(result["metrics"]) for label, result in evaluated.items()}
    mean_delta_vs_a0 = {
        label: _delta(mean, means["A0"])
        for label, mean in means.items()
        if label != "A0"
    }
    contact_sheets = _write_contact_sheets(
        args.output_dir,
        selected_views,
        evaluated,
        ranges,
        args.reference_run,
        args.contact_tile_width,
    )
    summary = {
        "diagnostic": "gmvc_underwater_dewatered_comparison",
        "scene_name": args.scene_name,
        "test_mode": args.test_mode,
        "reference_run": args.reference_run,
        "dewatered_definition": (
            "J_proxy_raw rendered with a diagnostic DC-proxy context when available; "
            "fallback is J_gaussian_raw/J_raw. This is a model intrinsic proxy, not clear-image GT."
        ),
        "runs": {
            label: {
                "config": evaluated[label]["config"],
                "requested_step": int(evaluated[label]["requested_step"]),
                "step": int(evaluated[label]["step"]),
                "checkpoint": evaluated[label]["checkpoint"],
                "experiment_name": evaluated[label]["experiment_name"],
                "mean": means[label],
            }
            for label in evaluated
        },
        "mean_delta_vs_a0": mean_delta_vs_a0,
        "per_view": per_view,
        "selected_contact_views": selected_views,
        "contact_sheets": contact_sheets,
        "component_visualization_ranges": ranges,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", default="")
    parser.add_argument("--a0-config", type=Path, required=True)
    parser.add_argument("--a0-step", type=int, default=15000)
    parser.add_argument("--p30-config", type=Path, required=True)
    parser.add_argument("--p30-step", type=int, default=15000)
    parser.add_argument("--run", action="append", default=[], help="Extra run as LABEL=CONFIG:STEP")
    parser.add_argument("--reference-run", default="A0")
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--contact-tile-width", type=int, default=320)
    parser.add_argument("--force-dc-proxy", dest="force_dc_proxy", action="store_true")
    parser.add_argument("--no-force-dc-proxy", dest="force_dc_proxy", action="store_false")
    parser.set_defaults(force_dc_proxy=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "visualization_summary.json"),
                "mean_delta_vs_a0": result["mean_delta_vs_a0"],
                "views": len(result["per_view"]),
                "contact_sheets": result["contact_sheets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
