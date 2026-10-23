#!/usr/bin/env python3
"""Transmission-bucket diagnostic for the existing BND checkpoints.

This module is analysis-only: it never calls backward or an optimizer and it
does not alter the model configuration.  It compares retained BND checkpoints
against the matched legacy (BND-OFF) checkpoints using training views only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_direct_mdrr_cica_scene as WORKER
from water_splatting.fields.gaussian_appearance import compute_bounded_gaussian_colors

SCENES = {
    "JapaneseGradens-RedSea": "6",
    "Panama": "7",
}
STEPS = (14999,)
ON_ROOT = ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OFF_ROOT = ROOT / "outputs" / "legacy_no_bounded_three_improvements_20260905"
CONFIG = ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"


def _finite(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().isfinite()


def _mean(x: torch.Tensor) -> float:
    x = x.detach().float()
    x = x[_finite(x)]
    return float(x.mean().item()) if x.numel() else float("nan")


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = _finite(x) & _finite(y)
    x, y = x[mask].float(), y[mask].float()
    if x.numel() < 3 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _path(root: Path, scene: str, step: int, bounded: bool) -> Path:
    if bounded:
        return root / scene / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"
    return root / scene / "A2" / "checkpoints" / f"step-{step:09d}.ckpt"


def _load(branch: Any, path: Path, bounded: bool) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = branch.pipeline.model
    model.config.intrinsic_color_parameterization = "bounded_sh3" if bounded else "legacy"
    model.load_state_dict(payload["model"], strict=True)
    model.step = int(payload.get("absolute_step", 0))
    if payload.get("ocmc_bundle") is not None:
        from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
        FORMAL._install_condition(model, "C0", payload["ocmc_bundle"], None)
    model.eval()
    return payload


def _appearance_stats(outputs: Mapping[str, torch.Tensor], bounded: bool) -> Dict[str, float]:
    if bounded and "gaussian_sh_residual" in outputs:
        delta = outputs["gaussian_sh_residual"].detach().float()
        luma = delta.mean(dim=-1)
        chroma = delta - luma[:, None]
        visible = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
        rgb = outputs.get("gaussian_view_rgb", torch.empty(0, device=delta.device)).detach().float()
        return {"luma_residual": _mean(luma[visible].abs()), "chroma_residual": _mean(chroma[visible].norm(dim=-1)),
                "clear_rgb_min": float(rgb[visible].min().item()) if bool(visible.any()) else float("nan"),
                "clear_rgb_max": float(rgb[visible].max().item()) if bool(visible.any()) else float("nan"),
                "p_j_gt_1": 0.0, "j_p99": 0.0}
    return {"luma_residual": float("nan"), "chroma_residual": float("nan"), "clear_rgb_min": float("nan"), "clear_rgb_max": float("nan"), "p_j_gt_1": float("nan"), "j_p99": float("nan")}


def _render_variant(scene: str, bounded: bool, step: int, gpu: str, out_dir: Path) -> List[Dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    WORKER.ACTIVE_SOURCE_CONFIG = CONFIG
    WORKER.ACTIVE_SOURCE_ROOT = ON_ROOT if bounded else OFF_ROOT
    WORKER.ACTIVE_PARAMETERIZATION = "bounded_sh3" if bounded else "legacy"
    WORKER.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    branch = WORKER._new_branch(scene)
    try:
        path = _path(WORKER.ACTIVE_SOURCE_ROOT, scene, step, bounded)
        if not path.is_file():
            raise FileNotFoundError(path)
        _load(branch, path, bounded)
        records = WORKER._train_records(branch)
        # First pass establishes training-view support for each Gaussian.
        support = torch.zeros(branch.pipeline.model.num_points, device=branch.pipeline.model.device)
        for _, _, camera, _ in records:
            with torch.no_grad():
                branch.pipeline.model.get_outputs_for_camera(camera.to(branch.pipeline.model.device))
            support += (branch.pipeline.model.radii.reshape(-1) > 0).float()
        rows: List[Dict[str, Any]] = []
        model = branch.pipeline.model
        for _, view_id, camera, batch in records:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = WORKER._gt(model, WORKER._batch_to_device(batch, model.device), outputs["background"]).detach().float().clamp(0, 1)
            t = outputs["transmission"].detach().float().mean(dim=-1)
            depth = outputs["depth"].detach().float().squeeze(-1)
            obj = outputs["rgb_object"].detach().float()
            med = outputs["rgb_medium"].detach().float()
            clear = outputs["rgb_clear"].detach().float()
            pred = outputs["pred_image"].detach().float()
            bins = ((t < 0.33), ((t >= 0.33) & (t < 0.66)), (t >= 0.66))
            # Gaussian-view gates use detached renderer transmission sampled at
            # projected Gaussian centers and detached multi-view support.
            vis = model.radii.reshape(-1) > 0
            xys = model.xys.detach()
            xi = xys[:, 0].round().long().clamp(0, t.shape[1] - 1)
            yi = xys[:, 1].round().long().clamp(0, t.shape[0] - 1)
            tg = t[yi, xi].clamp(0, 1)
            sg = ((support - 3.0) / 3.0).clamp(0, 1)
            sg = sg.detach()
            gl = (0.60 + 0.40 * torch.sigmoid((tg - 0.35) / 0.20) * sg).detach()
            gc = (0.20 + 0.80 * torch.sigmoid((tg - 0.50) / 0.15) * sg).detach()
            app = _appearance_stats(outputs, bounded)
            for name, mask in zip(("low", "middle", "high"), bins):
                if not bool(mask.any()):
                    continue
                pix_t = t[mask]
                pix_gt = gt[mask]
                pix_obj = obj[mask]
                pix_med = med[mask]
                pix_clear = clear[mask]
                pix_pred = pred[mask]
                obj_res = (pix_obj - pix_gt).abs().mean(dim=-1)
                med_res = (pix_med - (pix_gt - pix_obj)).abs().mean(dim=-1)
                clear_res = (pix_clear - pix_obj).abs().mean(dim=-1)
                chroma_res = (pix_clear - pix_clear.mean(dim=-1, keepdim=True)).norm(dim=-1)
                rows.append({"scene": scene, "variant": "BND_ON" if bounded else "BND_OFF", "step": step, "view_id": str(view_id), "bucket": name,
                    "pixel_count": int(mask.sum()), "mean_transmission": _mean(pix_t), "rgb_object_residual": _mean(obj_res),
                    "rgb_medium_residual": _mean(med_res), "clear_intrinsic_residual": _mean(clear_res), "luminance_residual": app["luma_residual"],
                    "chromatic_residual": app["chroma_residual"], "bnd_on_off_chromatic_residual": float("nan"),
                    "support_mean": _mean(support[vis]), "depth_variance": float(depth[mask].var().item()), "medium_attenuation": _mean(outputs["medium_attn"][mask]),
                    "g_luma_mean": _mean(gl[vis]), "g_luma_p10": float(torch.quantile(gl[vis], 0.10).item()) if bool(vis.any()) else float("nan"),
                    "g_chroma_mean": _mean(gc[vis]), "g_chroma_p10": float(torch.quantile(gc[vis], 0.10).item()) if bool(vis.any()) else float("nan"),
                    "clear_rgb_min": app["clear_rgb_min"], "clear_rgb_max": app["clear_rgb_max"], "P_J_gt_1": app["p_j_gt_1"], "J_p99": app["j_p99"],
                    "tau_D_p90": float(torch.quantile(outputs["tau_D"].detach().float().reshape(-1), 0.90).item()),
                    "corr_t_chroma": _corr(pix_t, chroma_res), "corr_t_medium": _corr(pix_t, med_res), "corr_t_object": _corr(pix_t, obj_res)})
        return rows
    finally:
        WORKER._release(branch)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=sorted(SCENES), required=True)
    p.add_argument("--variant", choices=("on", "off"), required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows: List[Dict[str, Any]] = []
    for step in STEPS:
        path = _path(ON_ROOT if args.variant == "on" else OFF_ROOT, args.scene, step, args.variant == "on")
        if not path.is_file():
            (args.output_dir / "missing_steps.txt").parent.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "missing_steps.txt").open("a", encoding="utf8") as f:
                f.write(f"{args.scene},{args.variant},{step},{path}\n")
            continue
        rows.extend(_render_variant(args.scene, args.variant == "on", step, args.gpu, args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (args.output_dir / f"{args.scene}_{args.variant}.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (args.output_dir / f"{args.scene}_{args.variant}.json").write_text(json.dumps({"scene": args.scene, "variant": args.variant, "rows": rows}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
