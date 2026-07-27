#!/usr/bin/env python
"""Contribution-aware per-Gaussian sensitivity diagnostics.

The diagnostic uses masked scalar probes and parameter gradients instead of
classifying Gaussians by projected center. It is still a proxy, but it accounts
for the differentiable footprint, transmittance, overlap, and visibility encoded
by the renderer backward path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.water_splatting import SH2RGB


MASK_KEYS = ("water", "object", "boundary")
PROBE_NAMES = (
    "water_accumulation",
    "object_accumulation",
    "boundary_accumulation",
    "water_proxy_luma",
    "water_proxy_bluegreen",
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_mask(mask_dir: Path, view_index: int, key: str, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    path = mask_dir / f"view_{view_index:04d}_{key}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32)).to(device=device)[..., None]


def _zero_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def _safe_grad_abs(param: torch.nn.Parameter, length: int) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros(length, device=param.device)
    grad = param.grad.detach().reshape(length, -1).float()
    return grad.norm(dim=-1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[..., None]
    return (value * mask).sum() / mask.sum().clamp_min(1e-6)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]


def _bluegreen(rgb: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.maximum(rgb[..., 1:2], rgb[..., 2:3]) - rgb[..., 0:1])


def _probe_scalar(outputs: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], probe: str) -> torch.Tensor | None:
    if probe == "water_accumulation":
        return _masked_mean(outputs["accumulation"], masks["water"])
    if probe == "object_accumulation":
        return _masked_mean(outputs["accumulation"], masks["object"])
    if probe == "boundary_accumulation":
        return _masked_mean(outputs["accumulation"], masks["boundary"])
    if probe == "water_proxy_luma":
        if "J_proxy_raw" not in outputs:
            return None
        return _masked_mean(_luma(outputs["J_proxy_raw"]), masks["water"])
    if probe == "water_proxy_bluegreen":
        if "J_proxy_raw" not in outputs:
            return None
        return _masked_mean(_bluegreen(outputs["J_proxy_raw"]), masks["water"])
    raise KeyError(probe)


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


def _top_fraction_summary(
    *,
    score: torch.Tensor,
    object_score: torch.Tensor,
    boundary_score: torch.Tensor,
    fractions: List[float],
) -> Dict[str, Dict[str, float]]:
    total = score.sum().clamp_min(1e-12)
    object_total = object_score.sum().clamp_min(1e-12)
    boundary_total = boundary_score.sum().clamp_min(1e-12)
    n = score.numel()
    result: Dict[str, Dict[str, float]] = {}
    for frac in fractions:
        k = max(int(round(n * frac)), 1)
        top_vals, top_idx = torch.topk(score, k=min(k, n), largest=True)
        result[f"top_{frac:.0%}"] = {
            "count": int(top_idx.numel()),
            "water_score_share": float(top_vals.sum().item() / total.item()),
            "object_score_share": float(object_score[top_idx].sum().item() / object_total.item()),
            "boundary_score_share": float(boundary_score[top_idx].sum().item() / boundary_total.item()),
            "mean_water_score": float(top_vals.mean().item()) if top_vals.numel() else 0.0,
            "mean_object_score": float(object_score[top_idx].mean().item()) if top_idx.numel() else 0.0,
            "mean_boundary_score": float(boundary_score[top_idx].mean().item()) if top_idx.numel() else 0.0,
        }
    return result


def _top_candidates(
    *,
    model: Any,
    water_score: torch.Tensor,
    object_score: torch.Tensor,
    boundary_score: torch.Tensor,
    view_count: torch.Tensor,
    top_k: int,
) -> List[Dict[str, Any]]:
    denom = object_score + boundary_score + 1e-12
    candidate_score = water_score * (water_score / denom)
    top_vals, top_idx = torch.topk(candidate_score, k=min(top_k, candidate_score.numel()), largest=True)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)
    scale = model.scales.detach().exp().max(dim=-1).values.reshape(-1)
    dc_rgb = SH2RGB(model.features_dc.detach()).clamp(0.0, 1.0)
    out: List[Dict[str, Any]] = []
    for value, idx in zip(top_vals.detach().cpu(), top_idx.detach().cpu()):
        i = int(idx.item())
        rgb = dc_rgb[i].detach().cpu().tolist()
        out.append(
            {
                "index": i,
                "candidate_score": float(value.item()),
                "water_score": float(water_score[i].item()),
                "object_score": float(object_score[i].item()),
                "boundary_score": float(boundary_score[i].item()),
                "view_count": int(view_count[i].item()),
                "opacity": float(opacity[i].item()),
                "max_scale": float(scale[i].item()),
                "dc_rgb": [float(x) for x in rgb],
                "bluegreen_minus_red": float(max(rgb[1], rgb[2]) - rgb[0]),
            }
        )
    return out


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    model = pipeline.model
    model.config.clear_proxy_enabled = bool(args.enable_clear_proxy)

    n = int(model.num_points)
    accumulators: Dict[str, Dict[str, torch.Tensor]] = {
        probe: {
            "opacity": torch.zeros(n, device=model.device),
            "scale": torch.zeros(n, device=model.device),
            "features_dc": torch.zeros(n, device=model.device),
            "means": torch.zeros(n, device=model.device),
            "view_count": torch.zeros(n, device=model.device),
        }
        for probe in PROBE_NAMES
    }
    image_summaries: List[Dict[str, Any]] = []

    for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if image_idx >= args.max_images:
            break
        with torch.no_grad():
            shape_outputs = model.get_outputs(camera)
        h, w = shape_outputs["rgb"].shape[:2]
        masks = {key: _load_mask(args.mask_dir, image_idx, key, (h, w), model.device) for key in MASK_KEYS}
        image_entry: Dict[str, Any] = {
            "image_index": image_idx,
            "mask_coverage": {key: float(mask.mean().item()) for key, mask in masks.items()},
            "probes": {},
        }

        for probe in PROBE_NAMES:
            _zero_grads(model)
            outputs = model.get_outputs(camera)
            scalar = _probe_scalar(outputs, masks, probe)
            if scalar is None or not scalar.requires_grad:
                image_entry["probes"][probe] = {
                    "available": scalar is not None,
                    "requires_grad": bool(scalar.requires_grad) if scalar is not None else False,
                    "scalar": float(scalar.detach().item()) if scalar is not None else 0.0,
                    "opacity_grad_norm": 0.0,
                }
                continue
            scalar.backward()
            opacity_grad = _safe_grad_abs(model.opacities, n)
            scale_grad = _safe_grad_abs(model.scales, n)
            dc_grad = _safe_grad_abs(model.features_dc, n)
            means_grad = _safe_grad_abs(model.means, n)
            accumulators[probe]["opacity"] += opacity_grad
            accumulators[probe]["scale"] += scale_grad
            accumulators[probe]["features_dc"] += dc_grad
            accumulators[probe]["means"] += means_grad
            accumulators[probe]["view_count"] += (opacity_grad > args.nonzero_threshold).float()
            image_entry["probes"][probe] = {
                "available": True,
                "requires_grad": True,
                "scalar": float(scalar.detach().item()),
                "opacity_grad_norm": float(torch.linalg.vector_norm(opacity_grad).item()),
                "scale_grad_norm": float(torch.linalg.vector_norm(scale_grad).item()),
                "features_dc_grad_norm": float(torch.linalg.vector_norm(dc_grad).item()),
                "means_grad_norm": float(torch.linalg.vector_norm(means_grad).item()),
            }
            _zero_grads(model)
        image_summaries.append(image_entry)

    water = accumulators["water_accumulation"]["opacity"]
    obj = accumulators["object_accumulation"]["opacity"]
    boundary = accumulators["boundary_accumulation"]["opacity"]
    view_count = accumulators["water_accumulation"]["view_count"]
    aggregate: Dict[str, Any] = {
        "num_points": n,
        "probes": {},
        "water_accumulation_top_fractions": _top_fraction_summary(
            score=water,
            object_score=obj,
            boundary_score=boundary,
            fractions=[0.01, 0.05, 0.10],
        ),
        "top_candidates_by_water_vs_object_boundary": _top_candidates(
            model=model,
            water_score=water,
            object_score=obj,
            boundary_score=boundary,
            view_count=view_count,
            top_k=args.top_k,
        ),
    }
    for probe, groups in accumulators.items():
        aggregate["probes"][probe] = {
            "opacity": _stats(groups["opacity"]),
            "scale": _stats(groups["scale"]),
            "features_dc": _stats(groups["features_dc"]),
            "means": _stats(groups["means"]),
            "view_count": _stats(groups["view_count"]),
            "total_opacity_sensitivity": float(groups["opacity"].sum().item()),
        }

    repo = Path(__file__).resolve().parents[2]
    return {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(repo),
        "mask_dir": str(args.mask_dir),
        "max_images": int(args.max_images),
        "enable_clear_proxy": bool(args.enable_clear_proxy),
        "aggregate": aggregate,
        "images": image_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--nonzero-threshold", type=float, default=1e-12)
    parser.add_argument("--enable-clear-proxy", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"]["water_accumulation_top_fractions"], indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
