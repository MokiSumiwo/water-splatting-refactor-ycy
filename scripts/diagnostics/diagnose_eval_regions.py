#!/usr/bin/env python
"""Evaluate Phase 2.5 water/object/boundary masks for a candidate checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _output_j(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs.get("J_object", outputs.get("J_gaussian", outputs["J"])).detach().float().clamp(0.0, 1.0)


def _gradient_magnitude(luma: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(luma)
    dy = torch.zeros_like(luma)
    dx[:, 1:, :] = (luma[:, 1:, :] - luma[:, :-1, :]).abs()
    dy[1:, :, :] = (luma[1:, :, :] - luma[:-1, :, :]).abs()
    return (dx + dy) * 0.5


def _values(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image[mask.squeeze(-1)]


def _stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _fraction_gt(values: torch.Tensor, threshold: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float((values > threshold).float().mean().item())


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> float:
    den_value = float(den.item()) if den.numel() else 0.0
    if den_value <= 1e-8:
        return 0.0
    return float(num.item()) / den_value


def _load_mask(mask_dir: Path, image_idx: int) -> Dict[str, torch.Tensor]:
    path = mask_dir / f"view_{image_idx:04d}_regions.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval-region mask: {path}")
    payload = torch.load(path, map_location="cpu")
    return {
        "water": payload["water"].bool(),
        "object": payload["object"].bool(),
        "boundary": payload["boundary"].bool(),
    }


def _collect_reference(args: argparse.Namespace) -> Dict[int, Dict[str, torch.Tensor]]:
    _config, pipeline, _checkpoint_path, _step = eval_setup(args.reference_config)
    pipeline.eval()
    refs: Dict[int, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            j_luma = _luma(_output_j(outputs)).detach().cpu()
            refs[image_idx] = {
                "accumulation": outputs["accumulation"].detach().float().clamp(0.0, 1.0).cpu(),
                "j_luma": j_luma,
                "j_grad": _gradient_magnitude(j_luma),
            }
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return refs


def _maybe_save_heatmaps(
    output_dir: Path,
    image_idx: int,
    masks: Dict[str, torch.Tensor],
    outputs_cpu: Dict[str, torch.Tensor],
) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    j_luma = _luma(_output_j(outputs_cpu)).cpu()
    accumulation = outputs_cpu["accumulation"].detach().float().clamp(0.0, 1.0).cpu()
    hit_confidence = outputs_cpu.get("hit_confidence")
    m_capacity = outputs_cpu.get("m_capacity")

    for name, mask in masks.items():
        save_image(mask.float().permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_{name}_mask.png")
    save_image(accumulation.permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_accumulation.png")
    save_image((masks["water"].float() * accumulation).permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_water_accumulation_overlay.png")
    save_image((masks["water"].float() * j_luma).permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_water_J_leakage_overlay.png")
    if hit_confidence is not None:
        hit = hit_confidence.detach().float().clamp(0.0, 1.0).cpu()
        save_image(hit.permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_q_hit.png")
        save_image((masks["object"].float() * hit).permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_object_q_hit_overlay.png")
    if m_capacity is not None:
        cap = m_capacity.detach().float().clamp(0.0, 1.0).cpu()
        save_image(cap.permute(2, 0, 1), output_dir / f"view_{image_idx:04d}_m_capacity.png")


def _threshold_table(q_hit_water: torch.Tensor, q_hit_object: torch.Tensor, thresholds: Iterable[float]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        water_fp = (q_hit_water > threshold).float().sum()
        object_tp = (q_hit_object > threshold).float().sum()
        precision = _safe_ratio(object_tp, object_tp + water_fp)
        recall = _safe_ratio(object_tp, torch.tensor(float(q_hit_object.numel())))
        water_fpr = _safe_ratio(water_fp, torch.tensor(float(q_hit_water.numel())))
        rows.append(
            {
                "threshold": float(threshold),
                "object_precision_vs_water": precision,
                "object_recall": recall,
                "water_false_positive_rate": water_fpr,
            }
        )
    return rows


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    reference = _collect_reference(args)
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_summaries: List[Dict[str, Any]] = []
    all_water_accum: List[torch.Tensor] = []
    all_water_j: List[torch.Tensor] = []
    all_object_accum: List[torch.Tensor] = []
    all_object_j: List[torch.Tensor] = []
    all_boundary_grad: List[torch.Tensor] = []
    all_ref_object_accum: List[torch.Tensor] = []
    all_ref_object_j: List[torch.Tensor] = []
    all_ref_boundary_grad: List[torch.Tensor] = []
    q_water: Dict[str, List[torch.Tensor]] = {"q_alpha": [], "q_conc": [], "q_hit": [], "depth_std_relative": []}
    q_object: Dict[str, List[torch.Tensor]] = {"q_alpha": [], "q_conc": [], "q_hit": [], "depth_std_relative": []}

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break
            masks = _load_mask(args.mask_dir, image_idx)
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            outputs_cpu = {
                key: value.detach().float().cpu()
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor) and value.ndim >= 2
            }
            accumulation = outputs_cpu["accumulation"].clamp(0.0, 1.0)
            j_luma = _luma(_output_j(outputs_cpu))
            j_grad = _gradient_magnitude(j_luma)
            ref = reference[image_idx]

            water_accum = _values(accumulation, masks["water"])
            water_j = _values(j_luma, masks["water"])
            object_accum = _values(accumulation, masks["object"])
            object_j = _values(j_luma, masks["object"])
            boundary_grad = _values(j_grad, masks["boundary"])
            ref_object_accum = _values(ref["accumulation"], masks["object"])
            ref_object_j = _values(ref["j_luma"], masks["object"])
            ref_boundary_grad = _values(ref["j_grad"], masks["boundary"])

            all_water_accum.append(water_accum)
            all_water_j.append(water_j)
            all_object_accum.append(object_accum)
            all_object_j.append(object_j)
            all_boundary_grad.append(boundary_grad)
            all_ref_object_accum.append(ref_object_accum)
            all_ref_object_j.append(ref_object_j)
            all_ref_boundary_grad.append(ref_boundary_grad)

            for output_key, bucket_key in (
                ("hit_q_alpha", "q_alpha"),
                ("hit_q_conc", "q_conc"),
                ("hit_confidence", "q_hit"),
                ("depth_std_relative", "depth_std_relative"),
            ):
                if output_key in outputs_cpu:
                    q_water[bucket_key].append(_values(outputs_cpu[output_key].clamp(0.0, 1.0), masks["water"]))
                    q_object[bucket_key].append(_values(outputs_cpu[output_key].clamp(0.0, 1.0), masks["object"]))

            image_summaries.append(
                {
                    "image_index": image_idx,
                    "water_pixels": int(masks["water"].sum().item()),
                    "object_pixels": int(masks["object"].sum().item()),
                    "boundary_pixels": int(masks["boundary"].sum().item()),
                    "water_accumulation": _stats(water_accum),
                    "water_J_luma": _stats(water_j),
                    "object_accumulation": _stats(object_accum),
                    "object_J_luma": _stats(object_j),
                    "object_accumulation_retention_vs_reference": _safe_ratio(object_accum.mean(), ref_object_accum.mean()),
                    "object_J_luma_retention_vs_reference": _safe_ratio(object_j.mean(), ref_object_j.mean()),
                    "boundary_J_gradient_retention_vs_reference": _safe_ratio(boundary_grad.mean(), ref_boundary_grad.mean()),
                }
            )

            if args.save_heatmaps:
                _maybe_save_heatmaps(args.output_dir / "heatmaps", image_idx, masks, outputs_cpu)

    water_accum_all = torch.cat(all_water_accum) if all_water_accum else torch.empty(0)
    water_j_all = torch.cat(all_water_j) if all_water_j else torch.empty(0)
    object_accum_all = torch.cat(all_object_accum) if all_object_accum else torch.empty(0)
    object_j_all = torch.cat(all_object_j) if all_object_j else torch.empty(0)
    boundary_grad_all = torch.cat(all_boundary_grad) if all_boundary_grad else torch.empty(0)
    ref_object_accum_all = torch.cat(all_ref_object_accum) if all_ref_object_accum else torch.empty(0)
    ref_object_j_all = torch.cat(all_ref_object_j) if all_ref_object_j else torch.empty(0)
    ref_boundary_grad_all = torch.cat(all_ref_boundary_grad) if all_ref_boundary_grad else torch.empty(0)

    q_water_all = {key: torch.cat(vals) if vals else torch.empty(0) for key, vals in q_water.items()}
    q_object_all = {key: torch.cat(vals) if vals else torch.empty(0) for key, vals in q_object.items()}
    thresholds = [float(item) for item in args.hit_thresholds.split(",") if item]

    aggregate = {
        "water_accumulation": _stats(water_accum_all),
        "water_J_luma": _stats(water_j_all),
        "object_accumulation": _stats(object_accum_all),
        "object_J_luma": _stats(object_j_all),
        "boundary_J_gradient": _stats(boundary_grad_all),
        "object_accumulation_retention_vs_reference": _safe_ratio(object_accum_all.mean(), ref_object_accum_all.mean()),
        "object_J_luma_retention_vs_reference": _safe_ratio(object_j_all.mean(), ref_object_j_all.mean()),
        "boundary_J_gradient_retention_vs_reference": _safe_ratio(boundary_grad_all.mean(), ref_boundary_grad_all.mean()),
        "q_alpha_water": _stats(q_water_all["q_alpha"]),
        "q_alpha_object": _stats(q_object_all["q_alpha"]),
        "q_conc_water": _stats(q_water_all["q_conc"]),
        "q_conc_object": _stats(q_object_all["q_conc"]),
        "q_hit_water": _stats(q_water_all["q_hit"]),
        "q_hit_object": _stats(q_object_all["q_hit"]),
        "depth_std_relative_water": _stats(q_water_all["depth_std_relative"]),
        "depth_std_relative_object": _stats(q_object_all["depth_std_relative"]),
        "water_q_alpha_gt_0p5_fraction": _fraction_gt(q_water_all["q_alpha"], 0.5),
        "water_low_depth_std_lt_0p20_fraction": float((q_water_all["depth_std_relative"] < 0.20).float().mean().item())
        if q_water_all["depth_std_relative"].numel() > 0
        else 0.0,
        "hit_threshold_table": _threshold_table(q_water_all["q_hit"], q_object_all["q_hit"], thresholds),
    }

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "load_config": str(args.load_config),
        "reference_config": str(args.reference_config),
        "mask_dir": str(args.mask_dir),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "aggregate": aggregate,
        "images": image_summaries,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--hit-thresholds", type=str, default="0.60,0.70,0.80")
    parser.add_argument("--save-heatmaps", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_dir / "eval_region_diagnostic.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
