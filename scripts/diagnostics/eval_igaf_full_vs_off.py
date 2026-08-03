"""Evaluate one IGAF checkpoint with IGAF enabled and disabled at inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _metric_means(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["psnr", "ssim", "lpips"]
    return {key: _stats([row[key] for row in rows])["mean"] for key in keys}


def _load_baseline(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf8"))
    results = payload.get("results", payload)
    return {key: float(results[key]) for key in ["psnr", "ssim", "lpips"] if key in results}


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((a * b).sum().div(denom).item())


def _set_igaf_mode(model: Any, mode: str) -> None:
    if mode == "full":
        model.config.igaf_inference_enabled = True
    elif mode == "off":
        model.config.igaf_inference_enabled = False
    else:
        raise ValueError(f"unknown IGAF mode: {mode}")


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=_update_config)
    pipeline.eval()
    model = pipeline.model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.step = int(step)

    rows: List[Dict[str, Any]] = []
    full_metric_rows: List[Dict[str, float]] = []
    off_metric_rows: List[Dict[str, float]] = []
    delta_abs: List[float] = []
    delta_p95: List[float] = []
    delta_residual_corr: List[float] = []

    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if args.max_images is not None and image_idx >= int(args.max_images):
                break

            _set_igaf_mode(model, "full")
            full_outputs = model.get_outputs_for_camera(camera=camera)
            full_metrics, _full_images = model.get_image_metrics_and_images(full_outputs, batch)

            _set_igaf_mode(model, "off")
            off_outputs = model.get_outputs_for_camera(camera=camera)
            off_metrics, _off_images = model.get_image_metrics_and_images(off_outputs, batch)

            gt = model.composite_with_background(model.get_gt_img(batch["image"]), off_outputs["background"]).detach()
            full_pred = full_outputs["pred_image"].detach()
            off_pred = off_outputs["pred_image"].detach()
            diff = (full_pred - off_pred).abs().mean(dim=-1)
            residual = (off_pred - gt).abs().mean(dim=-1)

            full_row = {key: _as_float(full_metrics[key]) for key in ["psnr", "ssim", "lpips"]}
            off_row = {key: _as_float(off_metrics[key]) for key in ["psnr", "ssim", "lpips"]}
            full_metric_rows.append(full_row)
            off_metric_rows.append(off_row)

            image_delta_abs = float(diff.mean().item())
            image_delta_p95 = float(torch.quantile(diff.reshape(-1), 0.95).item())
            image_corr = _corr(diff, residual)
            delta_abs.append(image_delta_abs)
            delta_p95.append(image_delta_p95)
            delta_residual_corr.append(image_corr)
            rows.append(
                {
                    "image_index": int(image_idx),
                    "full": full_row,
                    "off": off_row,
                    "delta_full_minus_off_abs_mean": image_delta_abs,
                    "delta_full_minus_off_abs_p95": image_delta_p95,
                    "delta_vs_off_residual_corr": image_corr,
                }
            )

    full = _metric_means(full_metric_rows)
    off = _metric_means(off_metric_rows)
    baseline = _load_baseline(args.baseline_json)
    delta_vs_baseline: Dict[str, Any] = {}
    retention: Dict[str, float] = {}
    if baseline is not None:
        delta_vs_baseline = {
            "full": {key: full[key] - baseline[key] for key in baseline if key in full},
            "off": {key: off[key] - baseline[key] for key in baseline if key in off},
        }
        psnr_den = full["psnr"] - baseline.get("psnr", full["psnr"])
        lpips_den = baseline.get("lpips", full["lpips"]) - full["lpips"]
        retention = {
            "rho_psnr": float((off["psnr"] - baseline["psnr"]) / psnr_den) if abs(psnr_den) > 1e-8 else 0.0,
            "rho_lpips": float((baseline["lpips"] - off["lpips"]) / lpips_den) if abs(lpips_den) > 1e-8 else 0.0,
        }

    repo = Path(__file__).resolve().parents[2]
    return {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(step),
        "git_commit": _git_commit(repo),
        "max_images": args.max_images,
        "baseline_json": str(args.baseline_json) if args.baseline_json is not None else None,
        "metrics": {
            "full": full,
            "off": off,
            "full_std": {key: _stats([row[key] for row in full_metric_rows])["std"] for key in ["psnr", "ssim", "lpips"]},
            "off_std": {key: _stats([row[key] for row in off_metric_rows])["std"] for key in ["psnr", "ssim", "lpips"]},
        },
        "delta_vs_baseline": delta_vs_baseline,
        "retention": retention,
        "delta_image": {
            "abs_mean": _stats(delta_abs),
            "abs_p95": _stats(delta_p95),
            "residual_corr": _stats(delta_residual_corr),
        },
        "images": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: result[k] for k in ["experiment_name", "checkpoint_step", "metrics", "retention", "delta_image"]}, indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
