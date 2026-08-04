#!/usr/bin/env python
"""Check GMVC medium point-query consistency against full medium maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from nerfstudio.utils.eval_utils import eval_setup


R_EDIT = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))


def _camera_iter(pipeline: Any, split: str, max_images: int):
    max_count = max_images if max_images > 0 else 10**9
    if split == "eval":
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            yield image_idx, camera, batch
        return
    dataset = pipeline.datamanager.train_dataset
    for image_idx in range(min(len(dataset.cameras), max_count)):
        yield image_idx, dataset.cameras[image_idx : image_idx + 1], {"image": dataset[image_idx]["image"]}


def _sample_hwc(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    h, w = image.shape[:2]
    grid_x = 2.0 * xy[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].T.contiguous()


def _ray_directions(camera: Any, xy: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    width = float(camera.width.item())
    height = float(camera.height.item())
    x_full = xy[:, 0] * (width / max(width - 1.0, 1.0))
    y_full = xy[:, 1] * (height / max(height - 1.0, 1.0))
    p_view = torch.stack(
        [
            (x_full - float(camera.cx.item())) / float(camera.fx.item()),
            (y_full - float(camera.cy.item())) / float(camera.fy.item()),
            torch.ones_like(xy[:, 0]),
        ],
        dim=-1,
    ).to(device=device, dtype=dtype)
    p_view = p_view / torch.linalg.norm(p_view, dim=-1, keepdim=True).clamp_min(1e-8)
    rotation = camera.camera_to_worlds[0, :3, :3].to(device=device, dtype=dtype) @ R_EDIT.to(device=device, dtype=dtype)
    directions = p_view @ rotation.T
    return directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)


def _image_xy_norm(height: int, width: int, xy: torch.Tensor) -> torch.Tensor:
    x = 2.0 * xy[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * xy[:, 1] / max(height - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--samples-per-image", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    _, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model
    device = model.device
    rows = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    with torch.no_grad():
        for image_idx, camera, _ in _camera_iter(pipeline, args.split, args.max_images):
            camera = camera.to(device) if hasattr(camera, "to") else camera
            outputs = model.get_outputs(camera)
            h, w = outputs["medium_attn"].shape[:2]
            count = min(int(args.samples_per_image), h * w)
            flat = torch.randperm(h * w, generator=generator)[:count]
            y = torch.div(flat, w, rounding_mode="floor").float()
            x = (flat % w).float()
            xy = torch.stack([x, y], dim=-1).to(device=device)
            image_xy = _image_xy_norm(h, w, xy).to(device=device, dtype=outputs["medium_attn"].dtype)
            directions = _ray_directions(camera, xy, device=device, dtype=outputs["medium_attn"].dtype)
            camera_centers = camera.camera_to_worlds[0, :3, 3].to(device=device, dtype=outputs["medium_attn"].dtype)
            camera_centers = camera_centers[None, :].expand(count, 3)
            point = model._query_gmvc_medium_points(directions, image_xy, camera_centers, None)
            for key, point_key in [
                ("medium_attn", "medium_attn"),
                ("medium_bs", "medium_bs"),
                ("medium_rgb", "medium_rgb"),
                ("b_inf", "b_inf"),
            ]:
                full = outputs.get(key, outputs.get("medium_rgb"))
                sampled = _sample_hwc(full, xy).to(device=device, dtype=outputs["medium_attn"].dtype)
                delta = (sampled - point[point_key]).abs()
                rows.append(
                    {
                        "image_index": int(image_idx),
                        "key": key,
                        "samples": int(count),
                        "max_abs": float(delta.max().item()),
                        "mean_abs": float(delta.mean().item()),
                    }
                )

    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_point_query_consistency",
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "rows": rows,
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "mean_abs": sum(row["mean_abs"] for row in rows) / max(len(rows), 1),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
