#!/usr/bin/env python
"""Frozen checkpoint exposure / white-balance oracle audit for CPNA."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


@dataclass
class PhotometricTheta:
    exposure: torch.Tensor
    white_balance: torch.Tensor

    def vector(self) -> torch.Tensor:
        return torch.cat([self.exposure.reshape(1), self.white_balance.reshape(3)])


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _stats(values: Iterable[float]) -> Dict[str, float]:
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


def _srgb_to_linear(value: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return value.clamp(0.0, 1.0).pow(float(gamma))


def _linear_to_srgb(value: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return value.clamp(0.0, 1.0).pow(1.0 / float(gamma))


def _apply_transform(pred: torch.Tensor, exposure: torch.Tensor, wb_raw: torch.Tensor, *, mode: str) -> torch.Tensor:
    linear = _srgb_to_linear(pred)
    if mode == "C0":
        return pred
    e = exposure.reshape(1, 1, 1) if mode in {"CE", "CEW"} else torch.zeros(1, 1, 1, device=pred.device, dtype=pred.dtype)
    wb = wb_raw - wb_raw.mean()
    wb = wb.reshape(1, 1, 3) if mode in {"CW", "CEW"} else torch.zeros(1, 1, 3, device=pred.device, dtype=pred.dtype)
    corrected = torch.exp(e) * torch.exp(wb) * linear
    return _linear_to_srgb(corrected)


def _valid_mask(pred: torch.Tensor, gt: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return (
        (pred.detach() >= low)
        & (pred.detach() <= high)
        & (gt.detach() >= low)
        & (gt.detach() <= high)
    ).all(dim=-1, keepdim=True)


def _region_masks(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    accumulation = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
    return {
        "whole": torch.ones_like(accumulation, dtype=torch.bool),
        "object": accumulation >= 0.55,
        "water": accumulation <= 0.20,
    }


def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    keep = mask.detach().bool().expand_as(pred)
    if int(keep.sum().item()) == 0:
        return 0.0
    return float((pred.detach().float() - gt.detach().float()).abs()[keep].mean().item())


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


def _fit_oracle(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    mode: str,
    valid: torch.Tensor,
    steps: int,
    lr: float,
) -> Tuple[torch.Tensor, PhotometricTheta, float]:
    exposure = torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)
    wb = torch.zeros(3, device=pred.device, dtype=pred.dtype, requires_grad=True)
    if mode == "C0":
        return pred.detach(), PhotometricTheta(exposure.detach(), (wb - wb.mean()).detach()), 0.0
    params = []
    if mode in {"CE", "CEW"}:
        params.append(exposure)
    if mode in {"CW", "CEW"}:
        params.append(wb)
    optim = torch.optim.Adam(params, lr=float(lr))
    valid_float = valid.detach().float()
    denom = valid_float.sum().clamp_min(1.0)
    last_loss = pred.new_tensor(0.0)
    for _ in range(max(int(steps), 1)):
        optim.zero_grad(set_to_none=True)
        corrected = _apply_transform(pred.detach(), exposure, wb, mode=mode)
        residual = (corrected - gt.detach()).abs().mean(dim=-1, keepdim=True)
        last_loss = (residual * valid_float).sum() / denom
        last_loss.backward()
        optim.step()
    corrected = _apply_transform(pred.detach(), exposure.detach(), wb.detach(), mode=mode).detach()
    return corrected, PhotometricTheta(exposure.detach(), (wb.detach() - wb.detach().mean()).detach()), float(last_loss.detach().item())


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


def _collect_predictions(model: Any, items: Iterator[Tuple[int, Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, camera, batch in items:
            outputs = model.get_outputs_for_camera(camera=camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"]).detach()
            pred = outputs["pred_image"].detach()
            c2w = camera.camera_to_worlds[0].detach()
            records.append(
                {
                    "image_index": int(idx),
                    "camera": camera,
                    "pred": pred,
                    "gt": gt,
                    "outputs": {k: v.detach() for k, v in outputs.items() if torch.is_tensor(v)},
                    "center": c2w[:3, 3].detach(),
                    "view_direction": c2w[:3, 2].detach() / c2w[:3, 2].detach().norm().clamp_min(1e-6),
                }
            )
    return records


def _evaluate_record(
    model: Any,
    record: Dict[str, Any],
    *,
    modes: List[str],
    fit_steps: int,
    fit_lr: float,
    saturation_low: float,
    saturation_high: float,
) -> Dict[str, Any]:
    pred = record["pred"]
    gt = record["gt"]
    valid = _valid_mask(pred, gt, saturation_low, saturation_high)
    regions = _region_masks(record["outputs"])
    base_metrics = _metrics(model, pred, gt)
    result: Dict[str, Any] = {
        "image_index": int(record["image_index"]),
        "valid_fraction": float(valid.float().mean().item()),
        "base": {
            **base_metrics,
            "region_l1": {name: _masked_l1(pred, gt, mask) for name, mask in regions.items()},
        },
        "oracles": {},
    }
    for mode in modes:
        corrected, theta, fit_loss = _fit_oracle(
            pred,
            gt,
            mode=mode,
            valid=valid,
            steps=fit_steps,
            lr=fit_lr,
        )
        metrics = _metrics(model, corrected, gt)
        result["oracles"][mode] = {
            **metrics,
            "dpsnr": metrics["psnr"] - base_metrics["psnr"],
            "dssim": metrics["ssim"] - base_metrics["ssim"],
            "dlpips": metrics["lpips"] - base_metrics["lpips"],
            "fit_l1": fit_loss,
            "exposure": float(theta.exposure.item()),
            "exposure_gain": float(torch.exp(theta.exposure).item()),
            "white_balance": [float(v) for v in theta.white_balance.detach().cpu().tolist()],
            "white_balance_gain": [float(v) for v in torch.exp(theta.white_balance).detach().cpu().tolist()],
            "region_l1": {name: _masked_l1(corrected, gt, mask) for name, mask in regions.items()},
            "region_l1_delta": {
                name: _masked_l1(corrected, gt, mask) - result["base"]["region_l1"][name]
                for name, mask in regions.items()
            },
        }
    return result


def _theta_knn_predict(
    target: Dict[str, Any],
    sources: List[Dict[str, Any]],
    *,
    k: int,
    beta: float,
    tau: float,
    scene_scale: float,
) -> torch.Tensor:
    distances = []
    for src in sources:
        center_dist = torch.linalg.norm(target["center"] - src["center"]) / max(float(scene_scale), 1e-6)
        dot = torch.clamp((target["view_direction"] * src["view_direction"]).sum(), -1.0, 1.0)
        angle = torch.arccos(dot) / math.pi
        distances.append(float((center_dist + float(beta) * angle).item()))
    order = sorted(range(len(sources)), key=lambda idx: distances[idx])[: max(int(k), 1)]
    weights = torch.tensor([math.exp(-(distances[idx] ** 2) / max(float(tau) ** 2, 1e-8)) for idx in order], device=target["center"].device)
    thetas = torch.stack([sources[idx]["theta"] for idx in order], dim=0)
    return (weights[:, None] * thetas).sum(dim=0) / weights.sum().clamp_min(1e-8)


def _apply_theta_vector(pred: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return _apply_transform(pred, theta[0], theta[1:4], mode="CEW")


def _aggregate(rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for mode in modes:
        mode_rows = [row["oracles"][mode] for row in rows]
        aggregate[mode] = {
            "psnr": _stats(row["psnr"] for row in mode_rows),
            "ssim": _stats(row["ssim"] for row in mode_rows),
            "lpips": _stats(row["lpips"] for row in mode_rows),
            "dpsnr": _stats(row["dpsnr"] for row in mode_rows),
            "dssim": _stats(row["dssim"] for row in mode_rows),
            "dlpips": _stats(row["dlpips"] for row in mode_rows),
            "exposure_gain": _stats(row["exposure_gain"] for row in mode_rows),
            "wb_gain_min": _stats(min(row["white_balance_gain"]) for row in mode_rows),
            "wb_gain_max": _stats(max(row["white_balance_gain"]) for row in mode_rows),
            "object_l1_delta": _stats(row["region_l1_delta"]["object"] for row in mode_rows),
            "water_l1_delta": _stats(row["region_l1_delta"]["water"] for row in mode_rows),
        }
    return aggregate


def _loo_predictability(model: Any, records: List[Dict[str, Any]], rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    if len(records) < 3:
        return {"enabled": False, "reason": "not enough train views"}
    for record, row in zip(records, rows):
        theta = row["oracles"]["CEW"]
        record["theta"] = torch.tensor([theta["exposure"], *theta["white_balance"]], device=record["pred"].device, dtype=record["pred"].dtype)
    centers = torch.stack([r["center"] for r in records], dim=0)
    scene_scale = float(torch.linalg.norm(centers - centers.mean(dim=0), dim=-1).quantile(0.95).item())
    loo_rows = []
    for idx, record in enumerate(records):
        sources = [r for j, r in enumerate(records) if j != idx]
        theta_pred = _theta_knn_predict(
            record,
            sources,
            k=int(args.knn_neighbors),
            beta=float(args.knn_angle_weight),
            tau=float(args.knn_tau),
            scene_scale=scene_scale,
        )
        corrected = _apply_theta_vector(record["pred"], theta_pred)
        base = _metrics(model, record["pred"], record["gt"])
        metrics = _metrics(model, corrected, record["gt"])
        oracle = rows[idx]["oracles"]["CEW"]
        oracle_psnr_gain = oracle["dpsnr"]
        oracle_lpips_gain = -oracle["dlpips"]
        loo_psnr_gain = metrics["psnr"] - base["psnr"]
        loo_lpips_gain = -(metrics["lpips"] - base["lpips"])
        loo_rows.append(
            {
                "image_index": int(record["image_index"]),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "dpsnr": loo_psnr_gain,
                "dssim": metrics["ssim"] - base["ssim"],
                "dlpips": metrics["lpips"] - base["lpips"],
                "oracle_dpsnr": oracle_psnr_gain,
                "oracle_dlpips": oracle["dlpips"],
                "psnr_gain_recovery": float(loo_psnr_gain / oracle_psnr_gain) if abs(oracle_psnr_gain) > 1e-8 else 0.0,
                "lpips_gain_recovery": float(loo_lpips_gain / oracle_lpips_gain) if abs(oracle_lpips_gain) > 1e-8 else 0.0,
                "predicted_theta": [float(v) for v in theta_pred.detach().cpu().tolist()],
                "oracle_theta": [float(v) for v in record["theta"].detach().cpu().tolist()],
            }
        )
    return {
        "enabled": True,
        "scene_scale": scene_scale,
        "aggregate": {
            "dpsnr": _stats(row["dpsnr"] for row in loo_rows),
            "dlpips": _stats(row["dlpips"] for row in loo_rows),
            "psnr_gain_recovery": _stats(row["psnr_gain_recovery"] for row in loo_rows),
            "lpips_gain_recovery": _stats(row["lpips_gain_recovery"] for row in loo_rows),
        },
        "rows": loo_rows,
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
    modes = ["CE", "CW", "CEW"]

    eval_records = _collect_predictions(model, _camera_items(pipeline, "eval", int(args.max_eval_images), device))
    eval_rows = [
        _evaluate_record(
            model,
            record,
            modes=modes,
            fit_steps=int(args.fit_steps),
            fit_lr=float(args.fit_lr),
            saturation_low=float(args.saturation_low),
            saturation_high=float(args.saturation_high),
        )
        for record in eval_records
    ]

    train_records = _collect_predictions(model, _camera_items(pipeline, "train", int(args.max_train_images), device))
    train_rows = [
        _evaluate_record(
            model,
            record,
            modes=["CEW"],
            fit_steps=int(args.fit_steps),
            fit_lr=float(args.fit_lr),
            saturation_low=float(args.saturation_low),
            saturation_high=float(args.saturation_high),
        )
        for record in train_records
    ]
    loo = _loo_predictability(model, train_records, train_rows, args)

    eval_agg = _aggregate(eval_rows, modes)
    gate = {
        "eval_cew_dpsnr_ge_0p15": eval_agg["CEW"]["dpsnr"]["mean"] >= 0.15,
        "eval_cew_dlpips_le_neg_0p002": eval_agg["CEW"]["dlpips"]["mean"] <= -0.002,
        "eval_object_l1_improves": eval_agg["CEW"]["object_l1_delta"]["mean"] < 0.0,
        "eval_water_l1_improves": eval_agg["CEW"]["water_l1_delta"]["mean"] < 0.0,
    }
    if loo.get("enabled"):
        gate["loo_recovers_psnr_gain_ge_50pct"] = loo["aggregate"]["psnr_gain_recovery"]["mean"] >= 0.50
        gate["loo_recovers_lpips_gain_ge_40pct"] = loo["aggregate"]["lpips_gain_recovery"]["mean"] >= 0.40
    gate["phase0c0d_passes"] = bool(all(gate.values())) if gate else False

    return {
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "eval_aggregate": eval_agg,
        "train_cew_aggregate": _aggregate(train_rows, ["CEW"]),
        "loo_predictability": loo,
        "gate": gate,
        "eval_images": eval_rows,
        "train_images": train_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--fit-lr", type=float, default=0.03)
    parser.add_argument("--saturation-low", type=float, default=0.03)
    parser.add_argument("--saturation-high", type=float, default=0.97)
    parser.add_argument("--knn-neighbors", type=int, default=4)
    parser.add_argument("--knn-angle-weight", type=float, default=0.25)
    parser.add_argument("--knn-tau", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cpna_photometric_oracle"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_json
    if output_json is None:
        scene = args.scene_name or "scene"
        output_json = args.output_dir / f"{scene}_cpna_photometric_oracle.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    summary = {
        "scene_name": result["scene_name"],
        "eval_CE": result["eval_aggregate"]["CE"],
        "eval_CW": result["eval_aggregate"]["CW"],
        "eval_CEW": result["eval_aggregate"]["CEW"],
        "loo_predictability": result["loo_predictability"].get("aggregate", result["loo_predictability"]),
        "gate": result["gate"],
    }
    print(json.dumps(summary, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
