#!/usr/bin/env python
"""Decompose learned attenuation spectra into scene/view/pixel components.

This diagnostic is read-only. It loads a checkpoint, evaluates fixed views, and
separates centered log attenuation spectra:

    z(p, v) = log beta_D(p, v) - mean_c log beta_D(p, v)
    s_v     = mean_p z(p, v)
    r(p, v) = z(p, v) - s_v

It reports whether attenuation spectral violations mainly live in the view
spectrum or in the per-pixel residual, and whether either correlates with far
clear-color bias.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
    denom = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-20))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((a * b).sum().div(denom).clamp(-1.0, 1.0).item())


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[..., None]
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    return value[mask]


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]


def _chroma(rgb: torch.Tensor) -> torch.Tensor:
    return rgb - rgb.mean(dim=-1, keepdim=True)


def _medium_axis_projection(j: torch.Tensor, medium_rgb: torch.Tensor) -> torch.Tensor:
    jc = _chroma(j.detach().float())
    mc = _chroma(medium_rgb.detach().float())
    unit = mc / mc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (jc * unit).sum(dim=-1, keepdim=True)


def _blue_green_minus_red(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.detach().float()
    return (0.5 * (rgb[..., 1:2] + rgb[..., 2:3]) - rgb[..., 0:1])


def _valid_mask(outputs: Dict[str, torch.Tensor], threshold: float) -> torch.Tensor:
    depth = outputs["depth"].detach().float()
    accumulation = outputs["accumulation"].detach().float()
    if depth.ndim == 2:
        depth = depth[..., None]
    if accumulation.ndim == 2:
        accumulation = accumulation[..., None]
    return torch.isfinite(depth) & (depth > 0.0) & (accumulation > float(threshold))


def _far_mask(outputs: Dict[str, torch.Tensor], valid: torch.Tensor, fraction: float) -> torch.Tensor:
    depth = outputs["depth"].detach().float()
    if depth.ndim == 2:
        depth = depth[..., None]
    selector = valid.squeeze(-1)
    if not bool(selector.any().item()):
        return valid & False
    d = depth.squeeze(-1)
    values = d[selector]
    q = torch.quantile(values, max(0.0, min(1.0, 1.0 - float(fraction))))
    return valid & (depth >= q)


def _order_bad_centered(z: torch.Tensor) -> torch.Tensor:
    return ~((z[..., 0] >= z[..., 1]) & (z[..., 1] >= z[..., 2]))


def _scene_variance(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    z_all = torch.cat([row["_z_cpu"] for row in rows], dim=0)
    s_all = torch.cat([row["_s_cpu"] for row in rows], dim=0)
    r_all = torch.cat([row["_r_cpu"] for row in rows], dim=0)
    global_mean = z_all.mean(dim=0, keepdim=True)
    total = (z_all - global_mean).square().mean().clamp_min(1e-20)
    view = (s_all - global_mean).square().mean()
    residual = r_all.square().mean()
    return {
        "total_spectral_variance": float(total.item()),
        "view_spectrum_variance": float(view.item()),
        "pixel_residual_variance": float(residual.item()),
        "view_spectrum_variance_fraction": float((view / total).item()),
        "pixel_residual_variance_fraction": float((residual / total).item()),
    }


def _view_result(
    *,
    image_idx: int,
    outputs: Dict[str, torch.Tensor],
    valid_threshold: float,
    far_fraction: float,
) -> Dict[str, Any]:
    attn = outputs["medium_attn"].detach().float().clamp_min(1e-8)
    log_beta = attn.log()
    z = log_beta - log_beta.mean(dim=-1, keepdim=True)
    valid = _valid_mask(outputs, valid_threshold)
    far = _far_mask(outputs, valid, far_fraction)
    selector = valid.squeeze(-1)
    if not bool(selector.any().item()):
        selector = torch.ones(*attn.shape[:2], device=attn.device, dtype=torch.bool)
    z_vals = z[selector]
    s_v = z_vals.mean(dim=0)
    s_v = s_v - s_v.mean()
    r = z - s_v.view(1, 1, 3)
    r = r - r.mean(dim=-1, keepdim=True)

    full_bad = _order_bad_centered(z)
    view_bad_scalar = bool(_order_bad_centered(s_v.view(1, 1, 3)).reshape(-1)[0].item())
    view_bad = torch.full_like(full_bad, view_bad_scalar, dtype=torch.bool)
    residual_induced = (~view_bad) & full_bad
    view_unresolved = view_bad & full_bad
    residual_corrected = view_bad & (~full_bad)

    j = outputs["J_gaussian_raw"].detach().float().clamp(0.0, 1.0)
    medium_rgb = outputs.get("b_inf", outputs["medium_rgb"]).detach().float()
    j_medium_axis = _medium_axis_projection(j, medium_rgb)
    j_bg_minus_red = _blue_green_minus_red(j)
    r_norm = r.norm(dim=-1, keepdim=True)
    r_abs = r.abs().mean(dim=-1, keepdim=True)
    s_bg = float((s_v[1] + s_v[2] - 2.0 * s_v[0]).item())

    far_selector = far.squeeze(-1)
    valid_selector = valid.squeeze(-1)
    if not bool(far_selector.any().item()):
        far_selector = valid_selector

    row: Dict[str, Any] = {
        "image_index": int(image_idx),
        "valid_coverage": float(valid.float().mean().item()),
        "far_coverage": float(far.float().mean().item()),
        "view_spectrum_r": float(s_v[0].item()),
        "view_spectrum_g": float(s_v[1].item()),
        "view_spectrum_b": float(s_v[2].item()),
        "view_spectrum_bg_minus_red": s_bg,
        "view_spectrum_violation": float(view_bad_scalar),
        "full_violation_rate_valid": float(full_bad[valid_selector].float().mean().item()),
        "view_unresolved_violation_rate_valid": float(view_unresolved[valid_selector].float().mean().item()),
        "residual_induced_violation_rate_valid": float(residual_induced[valid_selector].float().mean().item()),
        "residual_corrected_view_violation_rate_valid": float(residual_corrected[valid_selector].float().mean().item()),
        "pixel_residual_abs_valid": _stats(r_abs[valid_selector]),
        "pixel_residual_norm_valid": _stats(r_norm[valid_selector]),
        "pixel_residual_abs_far": _stats(r_abs[far_selector]),
        "pixel_residual_norm_far": _stats(r_norm[far_selector]),
        "far_j_medium_axis_mean": float(j_medium_axis[far_selector].mean().item()),
        "far_j_bg_minus_red_mean": float(j_bg_minus_red[far_selector].mean().item()),
        "far_j_luma_mean": float(_luma(j)[far_selector].mean().item()),
        "corr_pixel_residual_abs_vs_j_medium_axis_valid": _corr(r_abs[valid_selector], j_medium_axis[valid_selector]),
        "corr_pixel_residual_abs_vs_j_bg_minus_red_valid": _corr(r_abs[valid_selector], j_bg_minus_red[valid_selector]),
        "corr_pixel_residual_abs_vs_j_medium_axis_far": _corr(r_abs[far_selector], j_medium_axis[far_selector]),
        "corr_pixel_residual_abs_vs_j_bg_minus_red_far": _corr(r_abs[far_selector], j_bg_minus_red[far_selector]),
        "_z_cpu": z[selector].detach().cpu(),
        "_s_cpu": s_v.view(1, 3).expand(z_vals.shape[0], 3).detach().cpu(),
        "_r_cpu": r[selector].detach().cpu(),
    }
    return row


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.asarray(values, dtype=np.float64).mean()) if values else 0.0


def _corr_rows(rows: List[Dict[str, Any]], key_a: str, key_b: str) -> float:
    a = torch.tensor([float(row[key_a]) for row in rows], dtype=torch.float32)
    b = torch.tensor([float(row[key_b]) for row in rows], dtype=torch.float32)
    return _corr(a, b)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    return {
        "valid_coverage_mean": _mean(public_rows, "valid_coverage"),
        "far_coverage_mean": _mean(public_rows, "far_coverage"),
        "view_spectrum_violation_rate": _mean(public_rows, "view_spectrum_violation"),
        "full_violation_rate_valid": _mean(public_rows, "full_violation_rate_valid"),
        "view_unresolved_violation_rate_valid": _mean(public_rows, "view_unresolved_violation_rate_valid"),
        "residual_induced_violation_rate_valid": _mean(public_rows, "residual_induced_violation_rate_valid"),
        "residual_corrected_view_violation_rate_valid": _mean(public_rows, "residual_corrected_view_violation_rate_valid"),
        "pixel_residual_abs_valid_mean": float(
            np.asarray([row["pixel_residual_abs_valid"]["mean"] for row in public_rows], dtype=np.float64).mean()
        ),
        "pixel_residual_abs_far_mean": float(
            np.asarray([row["pixel_residual_abs_far"]["mean"] for row in public_rows], dtype=np.float64).mean()
        ),
        "far_j_medium_axis_mean": _mean(public_rows, "far_j_medium_axis_mean"),
        "far_j_bg_minus_red_mean": _mean(public_rows, "far_j_bg_minus_red_mean"),
        "corr_view_spectrum_bg_vs_far_j_medium_axis": _corr_rows(
            public_rows, "view_spectrum_bg_minus_red", "far_j_medium_axis_mean"
        ),
        "corr_view_spectrum_bg_vs_far_j_bg_minus_red": _corr_rows(
            public_rows, "view_spectrum_bg_minus_red", "far_j_bg_minus_red_mean"
        ),
        "corr_pixel_residual_abs_vs_j_medium_axis_valid_mean": _mean(
            public_rows, "corr_pixel_residual_abs_vs_j_medium_axis_valid"
        ),
        "corr_pixel_residual_abs_vs_j_bg_minus_red_valid_mean": _mean(
            public_rows, "corr_pixel_residual_abs_vs_j_bg_minus_red_valid"
        ),
        "corr_pixel_residual_abs_vs_j_medium_axis_far_mean": _mean(
            public_rows, "corr_pixel_residual_abs_vs_j_medium_axis_far"
        ),
        "corr_pixel_residual_abs_vs_j_bg_minus_red_far_mean": _mean(
            public_rows, "corr_pixel_residual_abs_vs_j_bg_minus_red_far"
        ),
        **_scene_variance(rows),
    }


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model

    rows: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_images:
                break
            outputs = model.get_outputs_for_camera(camera=camera)
            rows.append(
                _view_result(
                    image_idx=image_idx,
                    outputs=outputs,
                    valid_threshold=args.valid_accumulation_threshold,
                    far_fraction=args.far_fraction,
                )
            )

    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result = {
        "experiment": "attn_spectrum_decomposition",
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "test_mode": args.test_mode,
        "max_images": args.max_images,
        "valid_accumulation_threshold": args.valid_accumulation_threshold,
        "far_fraction": args.far_fraction,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": _aggregate(rows),
        "views": public_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "attn_spectrum_decomposition.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, indent=2))
    print(f"saved={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="inference")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-name", type=str, default="unknown")
    parser.add_argument("--valid-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--far-fraction", type=float, default=0.25)
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
