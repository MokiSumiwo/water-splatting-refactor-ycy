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
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any]]:
    max_count = max_images if max_images > 0 else 10**9
    if split == "eval":
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            yield image_idx, camera.to(device) if hasattr(camera, "to") else camera
        return

    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), max_count)
    for image_idx in range(count):
        camera = dataset.cameras[image_idx : image_idx + 1]
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera


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
    base_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    total = score.sum().clamp_min(1e-12)
    object_total = object_score.sum().clamp_min(1e-12)
    boundary_total = boundary_score.sum().clamp_min(1e-12)
    if base_mask is None:
        pool_idx = torch.arange(score.numel(), device=score.device)
    else:
        pool_idx = torch.where(base_mask.reshape(-1))[0]
    n = int(pool_idx.numel())
    result: Dict[str, Dict[str, float]] = {}
    if n == 0:
        return {
            f"top_{frac:.0%}": {
                "count": 0,
                "water_score_share": 0.0,
                "object_score_share": 0.0,
                "boundary_score_share": 0.0,
                "mean_water_score": 0.0,
                "mean_object_score": 0.0,
                "mean_boundary_score": 0.0,
            }
            for frac in fractions
        }
    for frac in fractions:
        k = max(int(round(n * frac)), 1)
        local_vals, local_idx = torch.topk(score[pool_idx], k=min(k, n), largest=True)
        top_idx = pool_idx[local_idx]
        result[f"top_{frac:.0%}"] = {
            "count": int(top_idx.numel()),
            "water_score_share": float(local_vals.sum().item() / total.item()),
            "object_score_share": float(object_score[top_idx].sum().item() / object_total.item()),
            "boundary_score_share": float(boundary_score[top_idx].sum().item() / boundary_total.item()),
            "mean_water_score": float(local_vals.mean().item()) if local_vals.numel() else 0.0,
            "mean_object_score": float(object_score[top_idx].mean().item()) if top_idx.numel() else 0.0,
            "mean_boundary_score": float(boundary_score[top_idx].mean().item()) if top_idx.numel() else 0.0,
        }
    return result


def _top_k_summary(
    *,
    score: torch.Tensor,
    object_score: torch.Tensor,
    boundary_score: torch.Tensor,
    counts: List[int],
) -> Dict[str, Dict[str, float]]:
    total = score.sum().clamp_min(1e-12)
    object_total = object_score.sum().clamp_min(1e-12)
    boundary_total = boundary_score.sum().clamp_min(1e-12)
    result: Dict[str, Dict[str, float]] = {}
    for count in counts:
        k = min(max(int(count), 1), score.numel())
        top_vals, top_idx = torch.topk(score, k=k, largest=True)
        result[f"top_{count}"] = {
            "count": int(top_idx.numel()),
            "water_score_share": float(top_vals.sum().item() / total.item()),
            "object_score_share": float(object_score[top_idx].sum().item() / object_total.item()),
            "boundary_score_share": float(boundary_score[top_idx].sum().item() / boundary_total.item()),
            "mean_water_score": float(top_vals.mean().item()) if top_vals.numel() else 0.0,
        }
    return result


def _cumulative_counts(score: torch.Tensor, thresholds: List[float]) -> Dict[str, int]:
    total = score.sum().clamp_min(1e-12)
    sorted_score = torch.sort(score.reshape(-1), descending=True).values
    cumsum = torch.cumsum(sorted_score, dim=0)
    result: Dict[str, int] = {}
    for threshold in thresholds:
        idx = torch.searchsorted(cumsum, total * float(threshold), right=False)
        result[f"count_for_{int(round(threshold * 100))}%"] = int(min(int(idx.item()) + 1, sorted_score.numel()))
    return result


def _positive_quantile(values: torch.Tensor, quantile: float) -> float:
    positive = values.detach().float().reshape(-1)
    positive = positive[torch.isfinite(positive) & (positive > 0)]
    if positive.numel() == 0:
        return float("inf")
    return float(torch.quantile(positive, float(quantile)).item())


def _candidate_selection(
    *,
    water_score: torch.Tensor,
    object_score: torch.Tensor,
    boundary_score: torch.Tensor,
    proxy_score: torch.Tensor,
    view_count: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    eps = 1e-12
    water_threshold = _positive_quantile(water_score, args.candidate_water_quantile)
    proxy_threshold = _positive_quantile(proxy_score, args.candidate_proxy_quantile)
    object_ratio = object_score / water_score.clamp_min(eps)
    boundary_ratio = boundary_score / water_score.clamp_min(eps)
    candidate_mask = (
        (water_score >= water_threshold)
        & (view_count >= int(args.candidate_min_view_count))
        & (object_ratio <= float(args.candidate_object_ratio_max))
        & (boundary_ratio <= float(args.candidate_boundary_ratio_max))
    )
    if bool(args.candidate_require_proxy):
        candidate_mask &= proxy_score >= proxy_threshold

    proxy_norm = proxy_score / proxy_score.max().clamp_min(eps)
    candidate_score = water_score * torch.exp(
        -float(args.candidate_object_penalty) * object_ratio
        -float(args.candidate_boundary_penalty) * boundary_ratio
    )
    candidate_score = candidate_score * proxy_norm.clamp_min(0.0)
    summary = {
        "candidate_count": int(candidate_mask.sum().item()),
        "candidate_fraction": float(candidate_mask.float().mean().item()),
        "water_threshold": water_threshold,
        "proxy_threshold": proxy_threshold,
        "min_view_count": int(args.candidate_min_view_count),
        "object_ratio_max": float(args.candidate_object_ratio_max),
        "boundary_ratio_max": float(args.candidate_boundary_ratio_max),
        "require_proxy": bool(args.candidate_require_proxy),
        "selected_water_score_sum": float(water_score[candidate_mask].sum().item()) if candidate_mask.any() else 0.0,
        "selected_proxy_score_sum": float(proxy_score[candidate_mask].sum().item()) if candidate_mask.any() else 0.0,
    }
    return candidate_mask, candidate_score, summary


def _top_candidates(
    *,
    model: Any,
    water_score: torch.Tensor,
    object_score: torch.Tensor,
    boundary_score: torch.Tensor,
    proxy_score: torch.Tensor,
    features_rest_score: torch.Tensor,
    view_count: torch.Tensor,
    candidate_score: torch.Tensor,
    candidate_mask: Optional[torch.Tensor],
    top_k: int,
) -> List[Dict[str, Any]]:
    ranking_score = candidate_score
    if candidate_mask is not None and candidate_mask.any():
        ranking_score = torch.where(candidate_mask, candidate_score, torch.full_like(candidate_score, -torch.inf))
    top_vals, top_idx = torch.topk(ranking_score, k=min(top_k, ranking_score.numel()), largest=True)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)
    scale = model.scales.detach().exp().max(dim=-1).values.reshape(-1)
    dc_rgb = SH2RGB(model.features_dc.detach()).clamp(0.0, 1.0)
    sh_rest_norm = model.features_rest.detach().reshape(model.features_rest.shape[0], -1).norm(dim=-1)
    out: List[Dict[str, Any]] = []
    for value, idx in zip(top_vals.detach().cpu(), top_idx.detach().cpu()):
        if not torch.isfinite(value):
            continue
        i = int(idx.item())
        rgb = dc_rgb[i].detach().cpu().tolist()
        out.append(
            {
                "index": i,
                "candidate_score": float(value.item()),
                "selected_candidate": bool(candidate_mask[i].item()) if candidate_mask is not None else False,
                "water_score": float(water_score[i].item()),
                "water_proxy_bluegreen_score": float(proxy_score[i].item()),
                "object_score": float(object_score[i].item()),
                "boundary_score": float(boundary_score[i].item()),
                "features_rest_score": float(features_rest_score[i].item()),
                "view_count": int(view_count[i].item()),
                "opacity": float(opacity[i].item()),
                "max_scale": float(scale[i].item()),
                "sh_rest_norm": float(sh_rest_norm[i].item()),
                "dc_rgb": [float(x) for x in rgb],
                "bluegreen_minus_red": float(max(rgb[1], rgb[2]) - rgb[0]),
            }
        )
    return out


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model
    model.config.clear_proxy_enabled = bool(args.enable_clear_proxy)

    n = int(model.num_points)
    accumulators: Dict[str, Dict[str, torch.Tensor]] = {
        probe: {
            "opacity": torch.zeros(n, device=model.device),
            "scale": torch.zeros(n, device=model.device),
            "features_dc": torch.zeros(n, device=model.device),
            "features_rest": torch.zeros(n, device=model.device),
            "means": torch.zeros(n, device=model.device),
            "view_count": torch.zeros(n, device=model.device),
        }
        for probe in PROBE_NAMES
    }
    image_summaries: List[Dict[str, Any]] = []

    for image_idx, camera in _camera_items(pipeline, args.split, args.max_images, model.device):
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
            rest_grad = _safe_grad_abs(model.features_rest, n)
            means_grad = _safe_grad_abs(model.means, n)
            accumulators[probe]["opacity"] += opacity_grad
            accumulators[probe]["scale"] += scale_grad
            accumulators[probe]["features_dc"] += dc_grad
            accumulators[probe]["features_rest"] += rest_grad
            accumulators[probe]["means"] += means_grad
            accumulators[probe]["view_count"] += (opacity_grad > args.nonzero_threshold).float()
            image_entry["probes"][probe] = {
                "available": True,
                "requires_grad": True,
                "scalar": float(scalar.detach().item()),
                "opacity_grad_norm": float(torch.linalg.vector_norm(opacity_grad).item()),
                "scale_grad_norm": float(torch.linalg.vector_norm(scale_grad).item()),
                "features_dc_grad_norm": float(torch.linalg.vector_norm(dc_grad).item()),
                "features_rest_grad_norm": float(torch.linalg.vector_norm(rest_grad).item()),
                "means_grad_norm": float(torch.linalg.vector_norm(means_grad).item()),
            }
            _zero_grads(model)
        image_summaries.append(image_entry)

    water = accumulators["water_accumulation"]["opacity"]
    obj = accumulators["object_accumulation"]["opacity"]
    boundary = accumulators["boundary_accumulation"]["opacity"]
    proxy = accumulators["water_proxy_bluegreen"]["opacity"]
    features_rest = accumulators["water_proxy_bluegreen"]["features_rest"]
    view_count = accumulators["water_accumulation"]["view_count"]
    active_mask = (
        (water > args.nonzero_threshold)
        | (obj > args.nonzero_threshold)
        | (boundary > args.nonzero_threshold)
    )
    candidate_mask, candidate_score, candidate_summary = _candidate_selection(
        water_score=water,
        object_score=obj,
        boundary_score=boundary,
        proxy_score=proxy,
        view_count=view_count,
        args=args,
    )
    aggregate: Dict[str, Any] = {
        "num_points": n,
        "split": args.split,
        "active_gaussian_count": int(active_mask.sum().item()),
        "nonzero_water_sensitivity_count": int((water > args.nonzero_threshold).sum().item()),
        "visible_union_count": int(active_mask.sum().item()),
        "probes": {},
        "candidate_selection": candidate_summary,
        "water_accumulation_top_fractions_all": _top_fraction_summary(
            score=water,
            object_score=obj,
            boundary_score=boundary,
            fractions=[0.01, 0.05, 0.10],
        ),
        "water_accumulation_top_fractions_active": _top_fraction_summary(
            score=water,
            object_score=obj,
            boundary_score=boundary,
            fractions=[0.01, 0.05, 0.10],
            base_mask=active_mask,
        ),
        "water_accumulation_top_counts": _top_k_summary(
            score=water,
            object_score=obj,
            boundary_score=boundary,
            counts=[50, 100, 500],
        ),
        "water_accumulation_cumulative_counts": _cumulative_counts(
            water,
            thresholds=[0.50, 0.80, 0.90],
        ),
        "top_candidates_by_train_view_gate": _top_candidates(
            model=model,
            water_score=water,
            object_score=obj,
            boundary_score=boundary,
            proxy_score=proxy,
            features_rest_score=features_rest,
            view_count=view_count,
            candidate_score=candidate_score,
            candidate_mask=candidate_mask,
            top_k=args.top_k,
        ),
    }
    for probe, groups in accumulators.items():
        aggregate["probes"][probe] = {
            "opacity": _stats(groups["opacity"]),
            "scale": _stats(groups["scale"]),
            "features_dc": _stats(groups["features_dc"]),
            "features_rest": _stats(groups["features_rest"]),
            "means": _stats(groups["means"]),
            "view_count": _stats(groups["view_count"]),
            "total_opacity_sensitivity": float(groups["opacity"].sum().item()),
        }

    candidate_output: Optional[Dict[str, str]] = None
    if args.candidate_output_prefix is not None:
        pt_path = args.candidate_output_prefix.with_suffix(".pt")
        json_path = args.candidate_output_prefix.with_suffix(".json")
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_payload = {
            "candidate_mask": candidate_mask.detach().cpu(),
            "candidate_score": candidate_score.detach().cpu(),
            "water_score": water.detach().cpu(),
            "object_score": obj.detach().cpu(),
            "boundary_score": boundary.detach().cpu(),
            "water_proxy_bluegreen_score": proxy.detach().cpu(),
            "features_rest_score": features_rest.detach().cpu(),
            "view_count": view_count.detach().cpu(),
            "summary": candidate_summary,
            "split": args.split,
            "load_config": str(args.load_config),
            "checkpoint": str(checkpoint_path),
            "step": int(step),
        }
        torch.save(candidate_payload, pt_path)
        json_path.write_text(
            json.dumps(
                {
                    "pt_path": str(pt_path),
                    "summary": candidate_summary,
                    "top_candidates": aggregate["top_candidates_by_train_view_gate"][: min(args.top_k, 20)],
                },
                indent=2,
            ),
            encoding="utf8",
        )
        candidate_output = {"pt": str(pt_path), "json": str(json_path)}

    repo = Path(__file__).resolve().parents[2]
    return {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(repo),
        "mask_dir": str(args.mask_dir),
        "split": args.split,
        "max_images": int(args.max_images),
        "enable_clear_proxy": bool(args.enable_clear_proxy),
        "candidate_output": candidate_output,
        "aggregate": aggregate,
        "images": image_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["train", "eval"], default="eval")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--nonzero-threshold", type=float, default=1e-12)
    parser.add_argument("--enable-clear-proxy", action="store_true")
    parser.add_argument("--candidate-output-prefix", type=Path, default=None)
    parser.add_argument("--candidate-min-view-count", type=int, default=5)
    parser.add_argument("--candidate-water-quantile", type=float, default=0.995)
    parser.add_argument("--candidate-proxy-quantile", type=float, default=0.95)
    parser.add_argument("--candidate-object-ratio-max", type=float, default=0.10)
    parser.add_argument("--candidate-boundary-ratio-max", type=float, default=0.10)
    parser.add_argument("--candidate-object-penalty", type=float, default=4.0)
    parser.add_argument("--candidate-boundary-penalty", type=float, default=4.0)
    parser.add_argument("--candidate-require-proxy", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "split": result["aggregate"]["split"],
                "active_gaussian_count": result["aggregate"]["active_gaussian_count"],
                "candidate_selection": result["aggregate"]["candidate_selection"],
                "water_top_active": result["aggregate"]["water_accumulation_top_fractions_active"],
                "candidate_output": result["candidate_output"],
            },
            indent=2,
        )
    )
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
