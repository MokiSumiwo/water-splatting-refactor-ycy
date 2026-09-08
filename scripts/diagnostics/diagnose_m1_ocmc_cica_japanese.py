#!/usr/bin/env python3
"""Read-only OCMC/CICA diagnostics for the legacy BND-OFF mainline.

The script deliberately reuses the production continuation worker's model
setup and CICA evidence code.  It loads frozen checkpoints only; no optimizer
step, topology update, or checkpoint write is performed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER


STEPS = (5000, 10000, 14999)
SCENES = ("JapaneseGradens-RedSea", "Panama")
SOURCE_ROOT = REPO_ROOT / "outputs" / "legacy_ocmc_cica_no_auxiliary_20260905"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "diagnostics_m1_ocmc_cica_20260908"
EPS = 1e-12
QUANTILE_MAX_N = 1_000_000


def _scalar(value: Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _stats(value: Tensor, prefix: str) -> Dict[str, float]:
    # Quantile is a reporting-only operation. Move before subsampling so a
    # very large CUDA tensor cannot trigger an out-of-range device index.
    flat = value.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "std", "p50", "p90", "p99", "max")}
    if flat.numel() > QUANTILE_MAX_N:
        positions = torch.linspace(0, flat.numel() - 1, QUANTILE_MAX_N).long().clamp_max(flat.numel() - 1)
        flat = flat[positions]
    quantiles = torch.quantile(flat.cpu(), torch.tensor([0.5, 0.9, 0.99]))
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}_p50": float(quantiles[0].item()),
        f"{prefix}_p90": float(quantiles[1].item()),
        f"{prefix}_p99": float(quantiles[2].item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def _corr(a: Tensor, b: Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y / denom.clamp_min(EPS)).item()) if denom.item() > EPS else 0.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _projector_rows(model: Any, scene: str, step: int) -> Dict[str, Any]:
    projector = model._camera_medium_observability_projector
    spectrum = model._camera_medium_observability_spectrum
    scale = model._camera_medium_observability_scale
    if projector is None:
        return {"scene": scene, "absolute_step": step, "projector_present": False}
    singular = torch.linalg.svdvals(projector.detach().float().cpu())
    row: Dict[str, Any] = {
        "scene": scene,
        "absolute_step": step,
        "projector_present": True,
        "projector_trace": float(torch.trace(projector.detach().float()).item()),
        "projector_fro_norm": float(torch.linalg.matrix_norm(projector.detach().float()).item()),
        "projector_rank_numeric": int((singular > 1e-6).sum().item()),
        "projector_spectrum_min": float(singular.min().item()),
        "projector_spectrum_max": float(singular.max().item()),
        "projector_spectrum_mean": float(singular.mean().item()),
        "projector_scale_min": float(scale.detach().float().min().item()) if scale is not None else None,
        "projector_scale_max": float(scale.detach().float().max().item()) if scale is not None else None,
        "projector_scale_mean": float(scale.detach().float().mean().item()) if scale is not None else None,
        "projector_stored_spectrum_min": float(spectrum.detach().float().min().item()) if spectrum is not None else None,
        "projector_stored_spectrum_max": float(spectrum.detach().float().max().item()) if spectrum is not None else None,
    }
    return row


def _ocmc_audit(branch: Any, scene: str, step: int, records: Sequence[Any]) -> List[Dict[str, Any]]:
    model = branch.pipeline.model
    rows: List[Dict[str, Any]] = []
    for index, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = WORKER._gt(model, WORKER._batch_to_device(batch, model.device), outputs["background"]).detach().float()
        raw = outputs.get("camera_medium_raw_unprojected")
        base = outputs.get("camera_medium_raw_base")
        delta = outputs.get("camera_medium_delta_raw")
        projected = outputs.get("camera_medium_delta_projected_raw")
        suppressed = outputs.get("camera_medium_delta_suppressed_raw")
        if not all(isinstance(value, Tensor) for value in (raw, base, delta, projected, suppressed)):
            raise RuntimeError(f"OCMC diagnostic outputs missing for {scene} view={view_id}")
        delta = delta.detach().float()
        projected = projected.detach().float()
        suppressed = suppressed.detach().float()
        raw = raw.detach().float()
        base = base.detach().float()
        pred = outputs["pred_image"].detach().float()
        medium_rgb = outputs["rgb_medium"].detach().float()
        medium_finite = outputs.get("rgb_medium_finite", medium_rgb).detach().float()
        rgb_tail = outputs.get("rgb_tail", torch.zeros_like(medium_rgb)).detach().float()
        rgb_residual = pred - gt
        delta_flat = delta.reshape(-1, 9)
        projected_flat = projected.reshape(-1, 9)
        suppressed_flat = suppressed.reshape(-1, 9)
        raw_flat = raw.reshape(-1, 9)
        weak_fraction = suppressed_flat.square().sum(dim=-1) / delta_flat.square().sum(dim=-1).clamp_min(EPS)
        row: Dict[str, Any] = {
            "scene": scene,
            "absolute_step": step,
            "view_index": int(index),
            "view_id": str(view_id),
            "height": int(delta.shape[0]),
            "width": int(delta.shape[1]),
            "ocmc_enabled": bool(WORKER.OCMC_ENABLED),
            "projector_refresh": "none",
            "projector_source": "C0@3000 frozen bundle",
            "delta_raw_l2_mean": float(torch.linalg.vector_norm(delta_flat, dim=-1).mean().item()),
            "projected_raw_l2_mean": float(torch.linalg.vector_norm(projected_flat, dim=-1).mean().item()),
            "suppressed_raw_l2_mean": float(torch.linalg.vector_norm(suppressed_flat, dim=-1).mean().item()),
            "suppressed_energy_fraction_mean": float(weak_fraction.mean().item()),
            "suppressed_energy_fraction_p90": float(torch.quantile(weak_fraction.cpu(), 0.9).item()),
            "medium_rgb_delta_corr_with_rgb_residual": _corr(medium_rgb, rgb_residual),
            "medium_finite_delta_corr_with_rgb_residual": _corr(medium_finite, rgb_residual),
            "raw_delta_energy_corr_with_rgb_residual_energy": _corr(
                torch.linalg.vector_norm(delta_flat, dim=-1),
                torch.linalg.vector_norm(rgb_residual.reshape(-1, 3), dim=-1),
            ),
        }
        row.update(_stats(delta, "delta_raw"))
        row.update(_stats(projected, "delta_projected_raw"))
        row.update(_stats(suppressed, "delta_suppressed_raw"))
        row.update(_stats(medium_rgb, "medium_rgb"))
        row.update(_stats(medium_finite, "medium_rgb_finite"))
        row.update(_stats(rgb_tail, "rgb_tail"))
        row.update(_stats(outputs["medium_bs"].detach().float(), "backscatter_density"))
        row.update(_stats(outputs["medium_attn"].detach().float(), "attenuation_density"))
        rows.append(row)
        del outputs, gt, raw, base, delta, projected, suppressed
    return rows


@torch.no_grad()
def _cica_dispersion(model: Any, bank_records: Sequence[Any]) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
    count = int(model.num_points)
    chroma_views: List[Tensor] = []
    information_views: List[Tensor] = []
    visible_views: List[Tensor] = []
    support = torch.zeros(count, device=model.device, dtype=torch.int16)
    for _index, _view_id, camera, batch in bank_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = WORKER._gt(model, WORKER._batch_to_device(batch, model.device), outputs["background"]).detach().float()
        residual = gt - outputs["pred_image"].detach().float()
        evidence = WORKER._cica_local_evidence(model, camera, outputs, residual)
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1).bool() & evidence["valid"]
        if bool(visible.any()):
            dc = torch.clamp(WORKER.SH_C0 * model.features_dc.detach().float() + 0.5, min=1e-6)
            corrected = torch.clamp(dc + WORKER.SH_C0 * evidence["delta_parameter"], min=1e-6)
            chroma = torch.stack(
                [torch.log((corrected[:, 0] + 1e-6) / (corrected[:, 1] + 1e-6)),
                 torch.log((corrected[:, 2] + 1e-6) / (corrected[:, 1] + 1e-6))],
                dim=-1,
            ).double()
            chroma = torch.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
            chroma_views.append(chroma)
            information_views.append(evidence["information_sum"].double())
            support += visible.to(torch.int16)
            visible_views.append(visible)
        del outputs, gt, residual, evidence
    if not chroma_views:
        controller = {
            "target_log_chroma": torch.zeros(count, 2, device=model.device),
            "support": support,
            "valid_count": support.sum() * 0,
            "population": torch.tensor(count, device=model.device),
            "gate": torch.zeros(count, device=model.device),
            "weight": torch.zeros(count, device=model.device),
        }
        return {"population": count, "valid_count": 0, "valid_fraction": 0.0}, controller
    values = torch.stack(chroma_views, dim=0)
    weights = torch.stack(information_views, dim=0)
    visible = torch.stack(visible_views, dim=0)
    valid = support >= int(WORKER.CICA_MIN_SUPPORT)
    visible_f = visible.to(values.dtype)
    support_f = visible_f.sum(dim=0).clamp_min(1.0)
    mean = (values * visible_f[..., None]).sum(dim=0) / support_f[..., None]
    view_var = (((values - mean.unsqueeze(0)).square() * visible_f[..., None]).sum(dim=0) / support_f[..., None]).mean(dim=-1)
    masked_values = values.masked_fill(~visible[..., None], float("nan"))
    median = torch.nanmedian(masked_values, dim=0).values
    mad = torch.nanmean((values - median.unsqueeze(0)).abs().masked_fill(~visible[..., None], float("nan")), dim=0).mean(dim=-1)
    order = torch.argsort(values, dim=0, stable=True)
    sorted_values = torch.gather(values, 0, order)
    sorted_weights = torch.gather(weights[..., None].expand(-1, -1, 2), 0, order)
    cumulative = sorted_weights.cumsum(dim=0)
    half = sorted_weights.sum(dim=0) * 0.5
    median_index = (cumulative >= half[None]).to(torch.int64).argmax(dim=0)
    target = torch.gather(sorted_values, 0, median_index[None].expand(1, -1, -1)).squeeze(0)
    target_mad = torch.nanmean((values - target.unsqueeze(0)).abs().masked_fill(~visible[..., None], float("nan")), dim=0).mean(dim=-1)
    stats = {
        "population": count,
        "valid_count": int(valid.sum().item()),
        "valid_fraction": float(valid.float().mean().item()),
        "support_mean": float(support.float().mean().item()),
        "support_p50": float(torch.quantile(support.float().cpu(), 0.5).item()),
        "support_p90": float(torch.quantile(support.float().cpu(), 0.9).item()),
        "view_chromaticity_variance_mean": float(view_var[valid].float().mean().item()) if bool(valid.any()) else 0.0,
        "view_chromaticity_variance_p90": float(torch.quantile(view_var[valid].float().cpu(), 0.9).item()) if bool(valid.any()) else 0.0,
        "unweighted_median_abs_dispersion_mean": float(mad[valid].float().mean().item()) if bool(valid.any()) else 0.0,
        "weighted_target_abs_dispersion_mean": float(target_mad[valid].float().mean().item()) if bool(valid.any()) else 0.0,
        "target_log_chroma_std_rg": float(target[valid, 0].float().std(unbiased=False).item()) if bool(valid.any()) else 0.0,
        "target_log_chroma_std_bg": float(target[valid, 1].float().std(unbiased=False).item()) if bool(valid.any()) else 0.0,
        "target_mean_abs_from_unweighted_mean": float((target[valid] - mean[valid]).abs().mean().item()) if bool(valid.any()) else 0.0,
    }
    controller = {
        "target_log_chroma": target.detach().float(),
        "support": support.detach(),
        "valid_count": valid.sum().detach(),
        "valid": valid.detach(),
        "population": torch.tensor(count, device=model.device),
        "gate": valid.float().detach(),
        "weight": weights.sum(dim=0).float().detach(),
    }
    return stats, controller


def _gradient_probe(model: Any, records: Sequence[Any], controller: Mapping[str, Tensor]) -> Dict[str, float]:
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    cica_raw, _ = WORKER._cica_loss(model, controller)
    cica_grad = torch.autograd.grad(cica_raw, model.features_dc, retain_graph=True, allow_unused=True)[0]
    _index, _view_id, camera, batch = records[0]
    with torch.enable_grad():
        outputs = model.get_outputs(camera.to(model.device))
    losses = model.get_loss_dict(outputs, WORKER._batch_to_device(batch, model.device), {})
    base = sum(losses.values())
    base_grad = torch.autograd.grad(base, model.features_dc, allow_unused=True)[0]
    result = {
        "cica_loss": float(cica_raw.detach().item()),
        "cica_features_dc_grad_l2": float(torch.linalg.vector_norm(cica_grad).item()) if cica_grad is not None else 0.0,
        "photometric_loss": float(base.detach().item()),
        "photometric_features_dc_grad_l2": float(torch.linalg.vector_norm(base_grad).item()) if base_grad is not None else 0.0,
    }
    model.zero_grad(set_to_none=True)
    model.train(mode=was_training)
    del outputs, losses, base, cica_raw, cica_grad, base_grad
    return result


def _run_scene(scene: str, source_config: Path, output_dir: Path) -> None:
    WORKER.ACTIVE_SOURCE_ROOT = SOURCE_ROOT.resolve()
    WORKER.ACTIVE_SOURCE_CONFIG = source_config.resolve()
    WORKER.ACTIVE_PARAMETERIZATION = "legacy"
    WORKER.OCMC_ENABLED = True
    WORKER.CICA_ENABLED = True
    WORKER.CICA_MIN_SUPPORT = 3
    WORKER.AUXILIARY_APPEARANCE_ENABLED = False
    branch = None
    try:
        branch = WORKER._new_branch(scene)
        records = WORKER._train_records(branch)
        temp = output_dir / scene
        temp.mkdir(parents=True, exist_ok=True)
        bank_path = SOURCE_ROOT / scene / "cica_camera_bank.json"
        bank_payload = json.loads(bank_path.read_text(encoding="utf8"))
        bank_records = [records[int(index)] for index in bank_payload["camera_indices"]]
        ocmc_rows: List[Dict[str, Any]] = []
        cica_rows: List[Dict[str, Any]] = []
        projector_rows: List[Dict[str, Any]] = []
        for step in STEPS:
            checkpoint = SOURCE_ROOT / scene / "checkpoints" / f"step-{step:09d}.ckpt"
            payload = WORKER._load_snapshot(branch, scene, checkpoint)
            branch.pipeline.model.eval()
            print(f"[diagnostic] {scene} step={step} OCMC", flush=True)
            ocmc_rows.extend(_ocmc_audit(branch, scene, step, records))
            projector_rows.append(_projector_rows(branch.pipeline.model, scene, step))
            print(f"[diagnostic] {scene} step={step} CICA", flush=True)
            cica_stats, controller = _cica_dispersion(branch.pipeline.model, bank_records)
            cica_stats.update({"scene": scene, "absolute_step": step})
            cica_stats.update(_gradient_probe(branch.pipeline.model, records, controller))
            cica_rows.append(cica_stats)
            del payload, controller
            gc.collect()
            torch.cuda.empty_cache()
        _write_csv(temp / "ocmc_per_view.csv", ocmc_rows)
        _write_csv(temp / "cica_diagnostics.csv", cica_rows)
        _write_csv(temp / "ocmc_projector.csv", projector_rows)
        (temp / "diagnostic_manifest.json").write_text(
            json.dumps(_json_safe({"scene": scene, "steps": STEPS, "source_root": str(SOURCE_ROOT), "projector_refresh": "none", "projector_source": "C0@3000 frozen bundle"}), indent=2) + "\n",
            encoding="utf8",
        )
    finally:
        if branch is not None:
            del branch
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, action="append", dest="scenes")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    visible = str(torch.cuda.current_device())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES set to one physical GPU")
    if str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES")) != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    WORKER.ALLOWED_GPUS = frozenset({str(args.gpu)})
    selected = tuple(args.scenes or SCENES)
    for scene in selected:
        _run_scene(scene, SOURCE_ROOT / scene / "source_config.yml", args.output_root.resolve())
    print(json.dumps({"output_root": str(args.output_root.resolve()), "scenes": selected, "steps": STEPS}, indent=2), flush=True)


if __name__ == "__main__":
    main()
