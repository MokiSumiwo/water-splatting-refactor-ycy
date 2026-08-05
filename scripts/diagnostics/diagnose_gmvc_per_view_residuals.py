#!/usr/bin/env python
"""Compare two GMVC checkpoints on each eval view."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchvision.utils as vutils
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


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


def _eval_checkpoint(config_path: Path, step: int, test_mode: str, max_images: int) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, step, test_mode)
    model = pipeline.model
    model.eval()
    rows: List[Dict[str, Any]] = []
    rendered: List[Dict[str, Any]] = []
    data_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    if max_images > 0:
        data_loader = data_loader[: int(max_images)]
    with torch.no_grad():
        for view_idx, (camera, batch) in enumerate(data_loader):
            outputs = model.get_outputs_for_camera(camera=camera)
            metrics, images = model.get_image_metrics_and_images(outputs, batch)
            pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0).cpu()
            gt = images["gt"].detach().float().clamp(0.0, 1.0).cpu()
            image_idx_raw = batch.get("image_idx", view_idx)
            image_idx = int(image_idx_raw.item() if torch.is_tensor(image_idx_raw) else image_idx_raw)
            extra = _extra_metrics(pred, gt)
            rows.append(
                {
                    "view_index": int(view_idx),
                    "image_idx": image_idx,
                    "image_name": _image_name(pipeline, image_idx),
                    "psnr": float(metrics.get("psnr", 0.0)),
                    "ssim": float(metrics.get("ssim", 0.0)),
                    "lpips": float(metrics.get("lpips", 0.0)),
                    **extra,
                }
            )
            rendered.append({"view_index": int(view_idx), "image_idx": image_idx, "gt": gt, "pred": pred})
    return {
        "config": config,
        "checkpoint": str(checkpoint_path),
        "step": int(loaded_step),
        "metrics": rows,
        "rendered": rendered,
    }


def _save_hwc(path: Path, image: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(image.permute(2, 0, 1).clamp(0.0, 1.0), path)


def _make_contact(gt: Tensor, a0: Tensor, p30: Tensor) -> Tensor:
    a0_abs = (a0 - gt).abs().mean(dim=-1, keepdim=True).expand_as(gt).clamp(0.0, 1.0)
    p30_abs = (p30 - gt).abs().mean(dim=-1, keepdim=True).expand_as(gt).clamp(0.0, 1.0)
    diff = (p30_abs[..., :1] - a0_abs[..., :1]).expand_as(gt)
    diff_vis = (0.5 + 4.0 * diff).clamp(0.0, 1.0)
    tiles = [gt, a0, p30, a0_abs, p30_abs, diff_vis]
    separator = torch.ones((gt.shape[0], max(gt.shape[1] // 200, 4), 3), dtype=gt.dtype)
    row: List[Tensor] = []
    for idx, tile in enumerate(tiles):
        if idx:
            row.append(separator)
        row.append(tile.clamp(0.0, 1.0))
    return torch.cat(row, dim=1)


def _mean_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]
    return {key: float(sum(float(row[key]) for row in rows) / max(len(rows), 1)) for key in keys}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    a0 = _eval_checkpoint(args.a0_config, args.a0_step, args.test_mode, args.max_images)
    p30 = _eval_checkpoint(args.p30_config, args.p30_step, args.test_mode, args.max_images)

    a0_by_view = {row["view_index"]: row for row in a0["metrics"]}
    p30_by_view = {row["view_index"]: row for row in p30["metrics"]}
    a0_rendered = {row["view_index"]: row for row in a0["rendered"]}
    p30_rendered = {row["view_index"]: row for row in p30["rendered"]}

    per_view: List[Dict[str, Any]] = []
    for view_index in sorted(a0_by_view):
        a0_row = a0_by_view[view_index]
        p30_row = p30_by_view[view_index]
        gt = a0_rendered[view_index]["gt"]
        a0_pred = a0_rendered[view_index]["pred"]
        p30_pred = p30_rendered[view_index]["pred"]
        view_dir = args.output_dir / f"view_{view_index:04d}"
        _save_hwc(view_dir / "gt.png", gt)
        _save_hwc(view_dir / "a0_rgb.png", a0_pred)
        _save_hwc(view_dir / "p30_rgb.png", p30_pred)
        _save_hwc(view_dir / "a0_abs_residual.png", (a0_pred - gt).abs())
        _save_hwc(view_dir / "p30_abs_residual.png", (p30_pred - gt).abs())
        a0_abs = (a0_pred - gt).abs().mean(dim=-1, keepdim=True)
        p30_abs = (p30_pred - gt).abs().mean(dim=-1, keepdim=True)
        diff_vis = (0.5 + 4.0 * (p30_abs - a0_abs)).expand_as(gt).clamp(0.0, 1.0)
        _save_hwc(view_dir / "p30_minus_a0_abs_residual_diff.png", diff_vis)
        _save_hwc(view_dir / "contact_sheet.png", _make_contact(gt, a0_pred, p30_pred))
        delta = {
            key: float(p30_row[key] - a0_row[key])
            for key in ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]
        }
        row = {
            "view_index": int(view_index),
            "image_idx": int(a0_row["image_idx"]),
            "image_name": a0_row["image_name"],
            "a0": {key: a0_row[key] for key in ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]},
            "p30": {key: p30_row[key] for key in ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]},
            "delta_p30_minus_a0": delta,
            "outputs": {
                "gt": str(view_dir / "gt.png"),
                "a0_rgb": str(view_dir / "a0_rgb.png"),
                "p30_rgb": str(view_dir / "p30_rgb.png"),
                "contact_sheet": str(view_dir / "contact_sheet.png"),
            },
        }
        per_view.append(row)
        (view_dir / "per_view_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf8")

    summary = {
        "diagnostic": "gmvc_per_view_residuals",
        "a0": {
            "config": str(args.a0_config),
            "step": int(a0["step"]),
            "checkpoint": a0["checkpoint"],
            "mean": _mean_metrics(a0["metrics"]),
        },
        "p30": {
            "config": str(args.p30_config),
            "step": int(p30["step"]),
            "checkpoint": p30["checkpoint"],
            "mean": _mean_metrics(p30["metrics"]),
        },
        "mean_delta_p30_minus_a0": {
            key: float(_mean_metrics(p30["metrics"])[key] - _mean_metrics(a0["metrics"])[key])
            for key in ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]
        },
        "per_view": per_view,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_view_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    del a0, p30
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-config", type=Path, required=True)
    parser.add_argument("--a0-step", type=int, default=13000)
    parser.add_argument("--p30-config", type=Path, required=True)
    parser.add_argument("--p30-step", type=int, default=13000)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "per_view_residual_summary.json"),
                "mean_delta_p30_minus_a0": result["mean_delta_p30_minus_a0"],
                "views": len(result["per_view"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
