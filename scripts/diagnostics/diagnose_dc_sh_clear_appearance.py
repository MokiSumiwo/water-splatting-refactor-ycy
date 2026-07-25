#!/usr/bin/env python
"""Post-hoc DC/SH clear-appearance diagnostics for the DualColor phase."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


VARIANTS: Dict[str, Tuple[bool, float, float, str]] = {
    "A0": (False, 1.0, 1.0, "full_sh"),
    "A1": (True, 0.0, 0.0, "dc_only"),
    "A2": (True, 1.0, 0.0, "dc_luma"),
    "A3": (True, 1.0, 0.05, "dc_luma_chroma005"),
    "A4": (True, 1.0, 0.10, "dc_luma_chroma010"),
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _values(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image[mask.squeeze(-1)]


def _stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _dominance(j: torch.Tensor) -> Dict[str, float]:
    red = j[..., 0] - torch.maximum(j[..., 1], j[..., 2])
    green = j[..., 1] - torch.maximum(j[..., 0], j[..., 2])
    blue = j[..., 2] - torch.maximum(j[..., 0], j[..., 1])
    return {
        "red_dominance_ratio": float((red > 0.05).float().mean().item()),
        "green_dominance_ratio": float((green > 0.05).float().mean().item()),
        "blue_dominance_ratio": float((blue > 0.05).float().mean().item()),
        "red_minus_max_gb_mean": float(red.mean().item()),
        "green_minus_max_rb_mean": float(green.mean().item()),
        "blue_minus_max_rg_mean": float(blue.mean().item()),
    }


def _load_far_mask(mask_dir: Path | None, image_idx: int) -> torch.Tensor | None:
    if mask_dir is None:
        return None
    path = mask_dir / f"view_{image_idx:04d}_far.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    return payload["mask"].bool()


def _load_region_masks(mask_dir: Path | None, image_idx: int) -> Dict[str, torch.Tensor]:
    if mask_dir is None:
        return {}
    path = mask_dir / f"view_{image_idx:04d}_regions.pt"
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    return {key: payload[key].bool() for key in ("water", "object", "boundary")}


def _save_image(path: Path, image: torch.Tensor) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image.detach().float().clamp(0.0, 1.0).cpu().permute(2, 0, 1), path)


def _variant_names(requested: str) -> List[str]:
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; valid={sorted(VARIANTS)}")
    return names


def _append_region_stats(
    bucket: Dict[str, List[torch.Tensor]],
    prefix: str,
    image: torch.Tensor,
    masks: Dict[str, torch.Tensor],
) -> None:
    luma = _luma(image)
    for name, mask in masks.items():
        bucket.setdefault(f"{prefix}_{name}_luma", []).append(_values(luma.cpu(), mask))
        bucket.setdefault(f"{prefix}_{name}_blue_minus_green", []).append(_values((image[..., 2:3] - image[..., 1:2]).cpu(), mask))
        bucket.setdefault(f"{prefix}_{name}_green_minus_red", []).append(_values((image[..., 1:2] - image[..., 0:1]).cpu(), mask))


def _aggregate(buckets: Dict[str, List[torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, chunks in buckets.items():
        if key == "J_rgb":
            continue
        out[key] = _stats(torch.cat(chunks) if chunks else torch.empty(0))
    return out


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    model = pipeline.model
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_names = _variant_names(args.variants)
    reference_j: Dict[int, torch.Tensor] = {}
    variant_summaries: Dict[str, Any] = {}

    with torch.no_grad():
        for variant in variant_names:
            enabled, eta_l, eta_c, label = VARIANTS[variant]
            model.config.dual_color_enabled = enabled
            model.config.clear_sh_luminance_scale = eta_l
            model.config.clear_sh_chroma_scale = eta_c

            buckets: Dict[str, List[torch.Tensor]] = {
                "J_rgb": [],
                "J_luma": [],
                "J_blue_minus_green": [],
                "J_green_minus_red": [],
                "accumulation": [],
            }
            image_rows: List[Dict[str, Any]] = []
            variant_dir = args.output_dir / f"{variant}_{label}"

            for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
                if image_idx >= args.max_images:
                    break
                outputs = model.get_outputs_for_camera(camera=camera)
                metrics, _images = model.get_image_metrics_and_images(outputs, batch)
                j = outputs["J"].detach().float().clamp(0.0, 1.0).cpu()
                rgb = outputs["rgb"].detach().float().clamp(0.0, 1.0).cpu()
                accum = outputs["accumulation"].detach().float().clamp(0.0, 1.0).cpu()
                buckets["J_rgb"].append(j.reshape(-1, 3))
                buckets["J_luma"].append(_luma(j))
                buckets["J_blue_minus_green"].append(j[..., 2:3] - j[..., 1:2])
                buckets["J_green_minus_red"].append(j[..., 1:2] - j[..., 0:1])
                buckets["accumulation"].append(accum)

                masks = _load_region_masks(args.region_mask_dir, image_idx)
                if masks:
                    _append_region_stats(buckets, "J", j, masks)
                    for name, mask in masks.items():
                        buckets.setdefault(f"accumulation_{name}", []).append(_values(accum, mask))

                far_mask = _load_far_mask(args.far_mask_dir, image_idx)
                if far_mask is not None:
                    _append_region_stats(buckets, "J_far", j, {"mask": far_mask})
                    buckets.setdefault("accumulation_far", []).append(_values(accum, far_mask))

                if args.save_images:
                    _save_image(variant_dir / f"view_{image_idx:04d}_rgb.png", rgb)
                    _save_image(variant_dir / f"view_{image_idx:04d}_J.png", j)
                    _save_image(variant_dir / f"view_{image_idx:04d}_accumulation.png", accum.expand_as(j))
                    if variant == "A0":
                        reference_j[image_idx] = j
                    elif image_idx in reference_j:
                        diff = (reference_j[image_idx] - j).abs()
                        _save_image(variant_dir / f"view_{image_idx:04d}_absdiff_vs_A0.png", diff)

                image_rows.append(
                    {
                        "image_index": image_idx,
                        "underwater_metrics": metrics,
                        "J_dominance": _dominance(j),
                        "J_luma": _stats(_luma(j)),
                    }
                )

            aggregate = _aggregate(buckets)
            aggregate["J_dominance"] = _dominance(
                torch.cat(buckets["J_rgb"], dim=0) if buckets["J_rgb"] else torch.empty(0, 3)
            )
            variant_summaries[variant] = {
                "label": label,
                "dual_color_enabled": enabled,
                "clear_sh_luminance_scale": eta_l,
                "clear_sh_chroma_scale": eta_c,
                "aggregate": aggregate,
                "images": image_rows,
            }

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment": "dc_sh_clear_appearance",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(repo),
        "max_images": args.max_images,
        "far_mask_dir": str(args.far_mask_dir) if args.far_mask_dir else None,
        "region_mask_dir": str(args.region_mask_dir) if args.region_mask_dir else None,
        "variants": variant_summaries,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--variants", type=str, default="A0,A1,A2,A3,A4")
    parser.add_argument("--far-mask-dir", type=Path, default=None)
    parser.add_argument("--region-mask-dir", type=Path, default=None)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_dir / "dc_sh_clear_appearance_summary.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v["aggregate"] for k, v in result["variants"].items()}, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
