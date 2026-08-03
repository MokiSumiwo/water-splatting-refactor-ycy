#!/usr/bin/env python
"""Frozen checkpoint small-pose oracle audit for CPNA."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _stats(values: List[float]) -> Dict[str, float]:
    vals = torch.tensor([float(v) for v in values if math.isfinite(float(v))], dtype=torch.float32)
    if vals.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(vals.mean().item()),
        "p50": float(torch.quantile(vals, 0.50).item()),
        "p90": float(torch.quantile(vals, 0.90).item()),
        "p95": float(torch.quantile(vals, 0.95).item()),
        "max": float(vals.max().item()),
    }


def _skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack(
        [
            torch.stack([zero, -v[2], v[1]]),
            torch.stack([v[2], zero, -v[0]]),
            torch.stack([-v[1], v[0], zero]),
        ]
    )


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega).clamp_min(1e-12)
    K = _skew(omega / theta)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _metrics(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0)
    gt = gt.detach().float().clamp(0.0, 1.0)
    pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
    gt_nchw = gt.permute(2, 0, 1).unsqueeze(0)
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw)),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
    }


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any, Dict[str, Any]]]:
    limit = int(max_images) if int(max_images) > 0 else 10**9
    if split == "eval":
        for idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if idx >= limit:
                break
            yield idx, camera.to(device) if hasattr(camera, "to") else camera, batch
        return
    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), limit)
    for idx in range(count):
        camera = dataset.cameras[idx : idx + 1]
        yield idx, camera.to(device) if hasattr(camera, "to") else camera, {"image": dataset[idx]["image"]}


def _make_pose_camera(camera: Any, raw_rot: torch.Tensor, raw_trans: torch.Tensor, *, mode: str, rot_bound: float, trans_bound: float) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    c2w = camera.camera_to_worlds[0]
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    omega = torch.tanh(raw_rot) * float(rot_bound) if mode in {"P1", "P3"} else torch.zeros_like(raw_rot)
    delta_t = torch.tanh(raw_trans) * float(trans_bound) if mode in {"P2", "P3"} else torch.zeros_like(raw_trans)
    R_new = _so3_exp(omega) @ R
    c2w_new = torch.cat([R_new, (t + delta_t).reshape(3, 1)], dim=1)
    pose_camera = copy.deepcopy(camera)
    pose_camera.camera_to_worlds = c2w_new.unsqueeze(0)
    return pose_camera, omega, delta_t


def _fit_pose(
    model: Any,
    camera: Any,
    gt: torch.Tensor,
    *,
    mode: str,
    steps: int,
    lr: float,
    rot_bound_rad: float,
    trans_bound: float,
) -> Dict[str, Any]:
    if mode == "P0":
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera=camera)
        metrics = _metrics(model, outputs["pred_image"], gt)
        return {"mode": mode, **metrics, "dpsnr": 0.0, "dssim": 0.0, "dlpips": 0.0, "status": "identity"}

    raw_rot = torch.zeros(3, device=model.device, dtype=gt.dtype, requires_grad=True)
    raw_trans = torch.zeros(3, device=model.device, dtype=gt.dtype, requires_grad=True)
    params = []
    if mode in {"P1", "P3"}:
        params.append(raw_rot)
    if mode in {"P2", "P3"}:
        params.append(raw_trans)
    optim = torch.optim.Adam(params, lr=float(lr))
    status = "ok"
    last_loss = 0.0
    base_outputs = model.get_outputs(camera)
    base_metrics = _metrics(model, base_outputs["pred_image"], gt)
    best_loss = float((base_outputs["pred_image"].detach().clamp(0.0, 1.0) - gt.detach()).abs().mean().item())
    best_raw_rot = raw_rot.detach().clone()
    best_raw_trans = raw_trans.detach().clone()

    for _ in range(max(int(steps), 1)):
        optim.zero_grad(set_to_none=True)
        pose_camera, _omega, _delta_t = _make_pose_camera(
            camera,
            raw_rot,
            raw_trans,
            mode=mode,
            rot_bound=float(rot_bound_rad),
            trans_bound=float(trans_bound),
        )
        outputs = model.get_outputs(pose_camera)
        pred = outputs["pred_image"].clamp(0.0, 1.0)
        loss = (pred - gt.detach()).abs().mean()
        if not loss.requires_grad:
            status = "no_gradient_path"
            break
        try:
            loss.backward()
        except RuntimeError as exc:
            status = f"backward_failed: {exc}"
            break
        optim.step()
        last_loss = float(loss.detach().item())
        if last_loss < best_loss:
            best_loss = last_loss
            best_raw_rot = raw_rot.detach().clone()
            best_raw_trans = raw_trans.detach().clone()

    with torch.no_grad():
        pose_camera, omega, delta_t = _make_pose_camera(
            camera,
            best_raw_rot,
            best_raw_trans,
            mode=mode,
            rot_bound=float(rot_bound_rad),
            trans_bound=float(trans_bound),
        )
        outputs = model.get_outputs_for_camera(camera=pose_camera)
        metrics = _metrics(model, outputs["pred_image"], gt)
    return {
        "mode": mode,
        **metrics,
        "dpsnr": metrics["psnr"] - base_metrics["psnr"],
        "dssim": metrics["ssim"] - base_metrics["ssim"],
        "dlpips": metrics["lpips"] - base_metrics["lpips"],
        "rotation_deg": float(torch.linalg.norm(omega).detach().item() * 180.0 / math.pi),
        "translation_norm": float(torch.linalg.norm(delta_t).detach().item()),
        "fit_l1": best_loss,
        "last_l1": last_loss,
        "status": status,
    }


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=_update_config)
    pipeline.eval()
    model = pipeline.model
    device = model.device
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _scene_center, scene_scale_tensor = model._get_scene_normalization(dtype=torch.float32, device=device)
    scene_scale = float(scene_scale_tensor.detach().item()) if torch.is_tensor(scene_scale_tensor) else float(scene_scale_tensor)
    trans_bound = float(args.translation_bound_fraction) * max(scene_scale, 1e-6)
    rot_bound_rad = float(args.rotation_bound_degrees) * math.pi / 180.0
    modes = ["P0", "P1", "P2", "P3"]

    rows = []
    for image_idx, camera, batch in _camera_items(pipeline, args.split, int(args.max_images), device):
        with torch.no_grad():
            base_outputs = model.get_outputs_for_camera(camera=camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), base_outputs["background"]).detach()
        image_row = {"image_index": int(image_idx), "modes": {}}
        for mode in modes:
            image_row["modes"][mode] = _fit_pose(
                model,
                camera,
                gt,
                mode=mode,
                steps=int(args.steps),
                lr=float(args.lr),
                rot_bound_rad=rot_bound_rad,
                trans_bound=trans_bound,
            )
        rows.append(image_row)

    aggregate: Dict[str, Any] = {}
    for mode in modes:
        mode_rows = [row["modes"][mode] for row in rows]
        aggregate[mode] = {
            "dpsnr": _stats([row["dpsnr"] for row in mode_rows]),
            "dssim": _stats([row["dssim"] for row in mode_rows]),
            "dlpips": _stats([row["dlpips"] for row in mode_rows]),
            "rotation_deg": _stats([row.get("rotation_deg", 0.0) for row in mode_rows]),
            "translation_norm": _stats([row.get("translation_norm", 0.0) for row in mode_rows]),
            "ok_count": int(sum(row.get("status") in {"ok", "identity"} for row in mode_rows)),
        }
    best_pose = max(["P1", "P2", "P3"], key=lambda key: aggregate[key]["dpsnr"]["mean"])
    gate = {
        "best_pose": best_pose,
        "best_pose_dpsnr_ge_0p10": aggregate[best_pose]["dpsnr"]["mean"] >= 0.10,
        "best_pose_dlpips_le_neg_0p001": aggregate[best_pose]["dlpips"]["mean"] <= -0.001,
        "all_optimizations_ok": all(aggregate[mode]["ok_count"] == len(rows) for mode in ["P1", "P2", "P3"]),
    }
    gate["pose_oracle_passes"] = bool(
        gate["best_pose_dpsnr_ge_0p10"] and gate["best_pose_dlpips_le_neg_0p001"] and gate["all_optimizations_ok"]
    )
    return {
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "max_images": int(args.max_images),
        "rotation_bound_degrees": float(args.rotation_bound_degrees),
        "translation_bound_fraction": float(args.translation_bound_fraction),
        "translation_bound_world": trans_bound,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregate,
        "gate": gate,
        "images": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--split", choices=["eval", "train"], default="eval")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--rotation-bound-degrees", type=float, default=0.5)
    parser.add_argument("--translation-bound-fraction", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cpna_pose_oracle"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    result = diagnose(args)
    output_json = args.output_json
    if output_json is None:
        scene = args.scene_name or "scene"
        output_json = args.output_dir / f"{scene}_cpna_pose_oracle.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"scene_name": result["scene_name"], "aggregate": result["aggregate"], "gate": result["gate"]}, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
