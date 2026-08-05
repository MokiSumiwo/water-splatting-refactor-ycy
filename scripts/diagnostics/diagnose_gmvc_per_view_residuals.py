#!/usr/bin/env python
"""Compare GMVC checkpoints on each eval view."""

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


def _make_contact(gt: Tensor, predictions: Dict[str, Tensor], reference_run: str) -> Tensor:
    residuals = {
        name: (pred - gt).abs().mean(dim=-1, keepdim=True).expand_as(gt).clamp(0.0, 1.0)
        for name, pred in predictions.items()
    }
    tiles = [gt]
    tiles.extend(predictions[name] for name in predictions)
    tiles.extend(residuals[name] for name in predictions)
    if reference_run in residuals:
        reference = residuals[reference_run][..., :1]
        for name, residual in residuals.items():
            if name == reference_run:
                continue
            diff_vis = (0.5 + 4.0 * (residual[..., :1] - reference)).expand_as(gt).clamp(0.0, 1.0)
            tiles.append(diff_vis)
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


def _parse_run_spec(value: str) -> Tuple[str, Path, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    label, rest = value.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    config_text, step_text = rest.rsplit(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    try:
        step = int(step_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid run step in {value}") from exc
    return label, Path(config_text), step


def _metric_keys() -> List[str]:
    return ["psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"]


def _delta_metrics(value: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(value[key] - base[key]) for key in _metric_keys()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_specs: List[Tuple[str, Path, int]] = [("A0", args.a0_config, args.a0_step)]
    if args.run:
        run_specs.extend(_parse_run_spec(item) for item in args.run)
    elif args.p30_config is not None:
        run_specs.append(("P30", args.p30_config, args.p30_step))
    else:
        raise ValueError("Provide either --p30-config or at least one --run LABEL=CONFIG:STEP")

    evaluated: Dict[str, Dict[str, Any]] = {}
    for label, config_path, step in run_specs:
        evaluated[label] = _eval_checkpoint(config_path, step, args.test_mode, args.max_images)

    by_view = {
        label: {row["view_index"]: row for row in result["metrics"]}
        for label, result in evaluated.items()
    }
    rendered = {
        label: {row["view_index"]: row for row in result["rendered"]}
        for label, result in evaluated.items()
    }
    a0_by_view = by_view["A0"]

    per_view: List[Dict[str, Any]] = []
    for view_index in sorted(a0_by_view):
        a0_row = a0_by_view[view_index]
        gt = rendered["A0"][view_index]["gt"]
        view_dir = args.output_dir / f"view_{view_index:04d}"
        _save_hwc(view_dir / "gt.png", gt)
        predictions: Dict[str, Tensor] = {}
        outputs: Dict[str, str] = {"gt": str(view_dir / "gt.png")}
        for label in evaluated:
            pred = rendered[label][view_index]["pred"]
            predictions[label] = pred
            slug = label.lower()
            _save_hwc(view_dir / f"{slug}_rgb.png", pred)
            _save_hwc(view_dir / f"{slug}_abs_residual.png", (pred - gt).abs())
            outputs[f"{slug}_rgb"] = str(view_dir / f"{slug}_rgb.png")
            outputs[f"{slug}_abs_residual"] = str(view_dir / f"{slug}_abs_residual.png")
        reference = args.reference_run
        if reference in predictions:
            ref_abs = (predictions[reference] - gt).abs().mean(dim=-1, keepdim=True)
            for label, pred in predictions.items():
                if label == reference:
                    continue
                diff_vis = (0.5 + 4.0 * ((pred - gt).abs().mean(dim=-1, keepdim=True) - ref_abs)).expand_as(gt)
                diff_path = view_dir / f"{label.lower()}_minus_{reference.lower()}_abs_residual_diff.png"
                _save_hwc(diff_path, diff_vis.clamp(0.0, 1.0))
                outputs[f"{label.lower()}_minus_{reference.lower()}_abs_residual_diff"] = str(diff_path)
        contact_sheet = view_dir / "contact_sheet.png"
        _save_hwc(contact_sheet, _make_contact(gt, predictions, reference))
        outputs["contact_sheet"] = str(contact_sheet)

        run_metrics = {
            label: {key: by_view[label][view_index][key] for key in _metric_keys()}
            for label in evaluated
        }
        delta_vs_a0 = {
            label: _delta_metrics(row, run_metrics["A0"])
            for label, row in run_metrics.items()
            if label != "A0"
        }
        delta_vs_reference = {}
        if reference in run_metrics:
            delta_vs_reference = {
                label: _delta_metrics(row, run_metrics[reference])
                for label, row in run_metrics.items()
                if label != reference
            }
        row = {
            "view_index": int(view_index),
            "image_idx": int(a0_row["image_idx"]),
            "image_name": a0_row["image_name"],
            "runs": run_metrics,
            "delta_vs_a0": delta_vs_a0,
            "delta_vs_reference": delta_vs_reference,
            "outputs": outputs,
        }
        if "P30" in run_metrics:
            row["a0"] = run_metrics["A0"]
            row["p30"] = run_metrics["P30"]
            row["delta_p30_minus_a0"] = delta_vs_a0["P30"]
        per_view.append(row)
        (view_dir / "per_view_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf8")

    means = {label: _mean_metrics(result["metrics"]) for label, result in evaluated.items()}
    mean_delta_vs_a0 = {
        label: _delta_metrics(mean, means["A0"])
        for label, mean in means.items()
        if label != "A0"
    }
    mean_delta_vs_reference = {}
    if args.reference_run in means:
        mean_delta_vs_reference = {
            label: _delta_metrics(mean, means[args.reference_run])
            for label, mean in means.items()
            if label != args.reference_run
        }
    summary = {
        "diagnostic": "gmvc_per_view_residuals",
        "reference_run": args.reference_run,
        "runs": {
            label: {
                "config": str(config_path),
                "requested_step": int(requested_step),
                "step": int(evaluated[label]["step"]),
                "checkpoint": evaluated[label]["checkpoint"],
                "mean": means[label],
            }
            for label, config_path, requested_step in run_specs
        },
        "mean_delta_vs_a0": mean_delta_vs_a0,
        "mean_delta_vs_reference": mean_delta_vs_reference,
        "per_view": per_view,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if "P30" in evaluated:
        summary["a0"] = summary["runs"]["A0"]
        summary["p30"] = summary["runs"]["P30"]
        summary["mean_delta_p30_minus_a0"] = mean_delta_vs_a0["P30"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_view_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    del evaluated
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-config", type=Path, required=True)
    parser.add_argument("--a0-step", type=int, default=13000)
    parser.add_argument("--p30-config", type=Path, default=None)
    parser.add_argument("--p30-step", type=int, default=13000)
    parser.add_argument("--run", action="append", default=[], help="Extra run as LABEL=CONFIG:STEP")
    parser.add_argument("--reference-run", default="A0")
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "per_view_residual_summary.json"),
                "mean_delta_vs_a0": result["mean_delta_vs_a0"],
                "mean_delta_vs_reference": result["mean_delta_vs_reference"],
                "views": len(result["per_view"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
