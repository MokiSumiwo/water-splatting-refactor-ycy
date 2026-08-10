#!/usr/bin/env python
"""Post-training responsibility audit for Panama BND-UNORM.

This diagnostic is read-only. It uses fixed M1 final high-J masks on the
Panama eval views and measures the formal image-gradient responsibility of the
K1 relative objective and the UNORM absolute objective at matched checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


EPS = 1e-12
FINAL_STEP = 15000
TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
REGIONS = (
    "M1_HIGH_J",
    "M1_LOW_J",
    "BRIGHT_Q5",
    "DARK_BOTTOM_QUINTILE",
    "BOTTOM20",
    "LOW_TRANSMISSION",
)


def _load_stage_module() -> Any:
    stage_path = Path(__file__).resolve().with_name("summarize_bnd_stage_panama.py")
    spec = importlib.util.spec_from_file_location("summarize_bnd_stage_panama_for_unorm_resp", stage_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {stage_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_unorm_runs(stage: Any) -> None:
    stage.RUNS = {
        "M1": stage.RunSpec(
            name="M1",
            config_relpath=(
                "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
                "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
            ),
            parameterization="legacy",
            role="reference_m1",
            reused=True,
        ),
        "K1-RST": stage.RunSpec(
            name="K1-RST",
            config_relpath=(
                "outputs/dewater_bounded_sh3_cross_scene_20260808/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
                "config.yml"
            ),
            parameterization="bounded_sh3",
            role="matched_restart_control",
            reused=True,
        ),
        "STAGE": stage.RunSpec(
            name="STAGE",
            config_relpath=(
                "outputs/bnd_unorm_panama_20260810/"
                "panama_bnd_unorm_seed42_step0_to_15000/water-splatting/"
                "20260810_bnd_unorm/config.yml"
            ),
            parameterization="bounded_sh3",
            role="absolute_photometric_normalization_candidate",
            reused=False,
        ),
    }
    stage._spec_and_config_for_step = lambda repo, run, nominal_step: (stage.RUNS[run], repo / stage.RUNS[run].config_relpath)  # type: ignore[attr-defined]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _release(loaded: Optional[Any]) -> None:
    if loaded is not None:
        try:
            del loaded.pipeline
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _formal_gt_pred(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Tuple[Tensor, Tensor]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    return gt_img, pred_img


def _loss_map_for_mode(gt: Tensor, pred: Tensor, mode: str) -> Tensor:
    if mode == "absolute":
        return torch.abs(gt - pred).mean(dim=-1)
    if mode == "relative_pred_detached":
        return torch.abs((gt - pred) / (pred.detach() + 1e-3)).mean(dim=-1)
    raise ValueError(mode)


def _objective_mode(model: Any) -> str:
    return str(getattr(model.config, "photometric_normalization_mode", "relative_pred_detached"))


def _m1_masks(repo: Path, stage: Any) -> Dict[str, Dict[str, Tensor]]:
    loaded = None
    try:
        loaded = stage._load_run(repo, "M1", FINAL_STEP)
        model = loaded.model
        cached: Dict[str, Dict[str, Tensor]] = {}
        lumas: List[Tensor] = []
        for _, view_id, camera, _batch in stage._view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(_batch["image"]), outputs["background"])
            luma = stage._luma(gt.detach().float().cpu())
            lumas.append(luma.reshape(-1))
            support = outputs["accumulation"].detach().float().cpu()[..., 0] > 0.01
            clear = outputs["clear_object_fullsh_raw"].detach().float().cpu()
            transmission = outputs["transmission"].detach().float().cpu()
            cached[view_id] = {
                "support": support,
                "clear": clear,
                "transmission": transmission,
                "luma": luma,
            }
            del outputs, gt
        bright_threshold = stage._safe_quantile(torch.cat(lumas), 0.80)
        dark_threshold = stage._safe_quantile(torch.cat(lumas), 0.20)
        out: Dict[str, Dict[str, Tensor]] = {}
        for view_id, item in cached.items():
            support = item["support"]
            clear = item["clear"]
            transmission = item["transmission"]
            luma = item["luma"]
            high = support & (clear.amax(dim=-1) > 1.0)
            height, width = high.shape
            yy = torch.linspace(0.0, 1.0, height).reshape(height, 1).expand(height, width)
            out[view_id] = {
                "M1_HIGH_J": high,
                "M1_LOW_J": support & ~high,
                "BRIGHT_Q5": luma > bright_threshold,
                "DARK_BOTTOM_QUINTILE": luma < dark_threshold,
                "BOTTOM20": yy >= 0.8,
                "LOW_TRANSMISSION": support & (transmission.amin(dim=-1) < 0.1),
            }
        return out
    finally:
        _release(loaded)


def _empty_accum() -> Dict[str, Dict[str, float]]:
    return {
        region: {
            "pixels": 0.0,
            "total_pixels": 0.0,
            "mse": 0.0,
            "mse_total": 0.0,
            "loss": 0.0,
            "loss_total": 0.0,
            "grad": 0.0,
            "grad_total": 0.0,
        }
        for region in REGIONS
    }


def _accumulate(
    accum: MutableMapping[str, Dict[str, float]],
    masks: Mapping[str, Tensor],
    mse_map: Tensor,
    loss_map: Tensor,
    grad_map: Tensor,
) -> None:
    mse_cpu = mse_map.detach().float().cpu()
    loss_cpu = loss_map.detach().float().cpu()
    grad_cpu = grad_map.detach().float().cpu()
    total_pixels = float(mse_cpu.numel())
    mse_total = float(mse_cpu.sum().item())
    loss_total = float(loss_cpu.sum().item())
    grad_total = float(grad_cpu.sum().item())
    for region in REGIONS:
        mask = masks[region].detach().bool().cpu()
        pix = float(mask.sum().item())
        accum[region]["pixels"] += pix
        accum[region]["total_pixels"] += total_pixels
        accum[region]["mse"] += float(mse_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["mse_total"] += mse_total
        accum[region]["loss"] += float(loss_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["loss_total"] += loss_total
        accum[region]["grad"] += float(grad_cpu[mask].sum().item()) if pix else 0.0
        accum[region]["grad_total"] += grad_total


def _row_from_accum(base: Mapping[str, Any], region: str, value: Mapping[str, float]) -> Dict[str, Any]:
    pixel_fraction = value["pixels"] / max(value["total_pixels"], EPS)
    mse_share = value["mse"] / max(value["mse_total"], EPS)
    loss_share = value["loss"] / max(value["loss_total"], EPS)
    grad_share = value["grad"] / max(value["grad_total"], EPS)
    return {
        **base,
        "view_id": "AGGREGATE",
        "region": region,
        "pixel_count": value["pixels"],
        "pixel_fraction": pixel_fraction,
        "mse": value["mse"] / max(value["pixels"], EPS),
        "mse_error_share": mse_share,
        "mse_enrichment": mse_share / max(pixel_fraction, EPS),
        "formal_l1_loss_share": loss_share,
        "formal_l1_loss_enrichment": loss_share / max(pixel_fraction, EPS),
        "total_image_gradient_share": grad_share,
        "total_image_gradient_enrichment": grad_share / max(pixel_fraction, EPS),
        "responsibility_ratio": grad_share / max(mse_share, EPS),
    }


def _audit_run_step(repo: Path, stage: Any, masks_by_view: Mapping[str, Mapping[str, Tensor]], run: str, step: int) -> List[Dict[str, Any]]:
    loaded = None
    rows: List[Dict[str, Any]] = []
    accum = _empty_accum()
    try:
        loaded = stage._load_run(repo, run, step)
        model = loaded.model
        mode = _objective_mode(model)
        base = {
            "scene": "Panama",
            "run": run,
            "nominal_step": step,
            "loaded_step": int(loaded.loaded_step),
            "photometric_normalization_mode": mode,
        }
        for _, view_id, camera, batch in stage._view_records(loaded):
            model.zero_grad(set_to_none=True)
            outputs = model.get_outputs(camera.to(model.device))
            gt, pred = _formal_gt_pred(model, outputs, batch)
            main_loss = model.get_loss_dict(outputs, batch)["main_loss"]
            grad = torch.autograd.grad(main_loss, outputs["pred_image"], retain_graph=False, allow_unused=False)[0]
            grad_map = torch.linalg.norm(grad.detach().float(), dim=-1)
            mse_map = (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)
            loss_map = _loss_map_for_mode(gt, pred, mode)
            view_accum = _empty_accum()
            _accumulate(view_accum, masks_by_view[view_id], mse_map, loss_map, grad_map)
            _accumulate(accum, masks_by_view[view_id], mse_map, loss_map, grad_map)
            for region, value in view_accum.items():
                rows.append({**_row_from_accum(base, region, value), "view_id": view_id})
            del outputs, gt, pred, main_loss, grad, grad_map, mse_map, loss_map
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for region, value in accum.items():
            rows.append(_row_from_accum(base, region, value))
        return rows
    finally:
        _release(loaded)


def _append_manifest(output_dir: Path, file_paths: Sequence[Path]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    rows = [
        {
            "file_path": str(path),
            "scene": "Panama",
            "run": "K1-RST;UNORM",
            "step": "all" if "trajectory" in path.stem else str(FINAL_STEP),
            "output_type": path.stem,
            "view_ids": "MTN_1539;MTN_1529;MTN_1547",
        }
        for path in file_paths
    ]
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    existing = list(manifest.get("outputs", []))
    seen = {row.get("file_path") for row in existing if isinstance(row, dict)}
    for row in rows:
        if row["file_path"] not in seen:
            existing.append(row)
    manifest["outputs"] = existing
    _write_json(manifest_path, manifest)

    csv_rows: List[Dict[str, Any]] = []
    if manifest_csv.exists() and manifest_csv.stat().st_size > 0:
        with manifest_csv.open("r", encoding="utf8") as handle:
            csv_rows.extend(csv.DictReader(handle))
    seen_csv = {row.get("file_path") for row in csv_rows}
    csv_rows.extend(row for row in rows if row["file_path"] not in seen_csv)
    _write_csv(manifest_csv, csv_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_unorm_panama_20260810"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = _load_stage_module()
    _patch_unorm_runs(stage)
    masks = _m1_masks(repo, stage)

    trajectory_rows: List[Dict[str, Any]] = []
    for run in ("K1-RST", "STAGE"):
        for step in TRAJECTORY_STEPS:
            trajectory_rows.extend(_audit_run_step(repo, stage, masks, run, step))

    final_rows = [
        row
        for row in trajectory_rows
        if int(row["nominal_step"]) == FINAL_STEP and row["run"] in {"K1-RST", "STAGE"}
    ]
    trajectory_json = output_dir / "responsibility_trajectory.json"
    trajectory_csv = output_dir / "responsibility_trajectory.csv"
    final_json = output_dir / "final_responsibility_audit.json"
    final_csv = output_dir / "final_responsibility_audit.csv"
    _write_json(trajectory_json, trajectory_rows)
    _write_csv(trajectory_csv, trajectory_rows)
    _write_json(final_json, final_rows)
    _write_csv(final_csv, final_rows)
    _append_manifest(output_dir, [trajectory_json, trajectory_csv, final_json, final_csv])
    print(json.dumps({"rows": len(trajectory_rows), "final_rows": len(final_rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
