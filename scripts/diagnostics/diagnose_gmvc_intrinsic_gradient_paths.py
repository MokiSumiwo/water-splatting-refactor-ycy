#!/usr/bin/env python
"""Verify GMVC intrinsic-loss gradient paths for Gaussian DC calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _grad_norm(grads: Iterable[Optional[torch.Tensor]]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().square().sum().item())
    return float(total**0.5)


def _safe_grad(loss: torch.Tensor, params: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
    if not bool(getattr(loss, "requires_grad", False)):
        return tuple(None for _ in params)
    return torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )


def _train_camera_items(pipeline: Any, max_images: int):
    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), max_images if max_images > 0 else len(dataset.cameras))
    device = pipeline.model.device
    for image_idx in range(count):
        camera = dataset.cameras[image_idx : image_idx + 1]
        if getattr(camera, "metadata", None) is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = torch.tensor([image_idx])
        batch = {"image": dataset[image_idx]["image"]}
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, batch


def _run_source_probe(model: Any, camera: Any, batch: Dict[str, torch.Tensor], source: str, args: argparse.Namespace) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    model.config.gmvc_enabled = True
    model.config.gmvc_diagnostic_only = False
    model.config.gmvc_track_bank_path = str(args.gmvc_track_bank)
    model.config.gmvc_start_step = int(args.probe_step) - 1
    model.config.gmvc_stop_step = int(args.probe_step) + 1000
    model.config.gmvc_ramp_steps = 0
    model.config.lambda_gmvc_j = 0.0
    model.config.lambda_gmvc_range = 0.0
    model.config.lambda_gmvc_binf = 0.0
    model.config.lambda_gmvc_intrinsic = float(args.lambda_gmvc_intrinsic)
    model.config.gmvc_intrinsic_source = source
    model.config.gmvc_intrinsic_use_dc_proxy = bool(args.use_dc_proxy)
    model.config.gmvc_max_tracks_per_step = int(args.max_tracks)
    model.config.gmvc_grad_log_path = None
    model.step = int(args.probe_step)

    outputs = model.get_outputs(camera)
    metrics: Dict[str, torch.Tensor] = {}
    loss_dict = model.get_loss_dict(outputs, batch, metrics)
    loss = loss_dict.get("gmvc_intrinsic_loss")
    if loss is None:
        return {
            "source": source,
            "loss": 0.0,
            "requires_grad": False,
            "gmvc_intrinsic_raw": float(metrics.get("gmvc_intrinsic_raw", torch.zeros((), device=model.device)).detach().float().item()),
            "gmvc_intrinsic_source_available": float(
                metrics.get("gmvc_intrinsic_source_available", torch.zeros((), device=model.device)).detach().float().item()
            ),
            "features_dc_grad_norm": 0.0,
            "rgb_features_dc_grad_norm": 0.0,
            "intrinsic_to_rgb_dc_grad_ratio": 0.0,
            "geometry_grad_norm": 0.0,
            "opacity_grad_norm": 0.0,
            "medium_grad_norm": 0.0,
            "has_J_proxy_raw": "J_proxy_raw" in outputs,
            "J_proxy_uses_dc_colors": float(
                outputs.get("J_proxy_uses_dc_colors", torch.zeros((), device=model.device)).detach().float().item()
            ),
            "camera_index": int(outputs.get("camera_index", torch.tensor(-1, device=model.device)).detach().cpu().reshape(-1)[0].item()),
            "skipped": "gmvc_intrinsic_loss_missing",
        }

    dc_params = [model.gauss_params["features_dc"]]
    geom_params = [
        model.gauss_params["means"],
        model.gauss_params["scales"],
        model.gauss_params["quats"],
    ]
    opacity_params = [model.gauss_params["opacities"]]
    medium_params = list(model.medium_mlp.parameters()) + list(model.direction_encoding.parameters())

    dc_grad = _safe_grad(loss, dc_params)
    geom_grad = _safe_grad(loss, geom_params)
    opacity_grad = _safe_grad(loss, opacity_params)
    medium_grad = _safe_grad(loss, medium_params)

    main_loss = loss_dict.get("main_loss")
    rgb_dc_grad = _safe_grad(main_loss, dc_params) if main_loss is not None else tuple(None for _ in dc_params)

    result = {
        "source": source,
        "loss": float(loss.detach().float().item()),
        "requires_grad": bool(getattr(loss, "requires_grad", False)),
        "gmvc_intrinsic_raw": float(metrics.get("gmvc_intrinsic_raw", loss.detach()).detach().float().item()),
        "gmvc_intrinsic_source_available": float(
            metrics.get("gmvc_intrinsic_source_available", torch.zeros((), device=model.device)).detach().float().item()
        ),
        "features_dc_grad_norm": _grad_norm(dc_grad),
        "rgb_features_dc_grad_norm": _grad_norm(rgb_dc_grad),
        "intrinsic_to_rgb_dc_grad_ratio": _grad_norm(dc_grad) / (_grad_norm(rgb_dc_grad) + 1e-12),
        "geometry_grad_norm": _grad_norm(geom_grad),
        "opacity_grad_norm": _grad_norm(opacity_grad),
        "medium_grad_norm": _grad_norm(medium_grad),
        "has_J_proxy_raw": "J_proxy_raw" in outputs,
        "J_proxy_uses_dc_colors": float(
            outputs.get("J_proxy_uses_dc_colors", torch.zeros((), device=model.device)).detach().float().item()
        ),
        "camera_index": int(outputs.get("camera_index", torch.tensor(-1, device=model.device)).detach().cpu().reshape(-1)[0].item()),
    }
    model.zero_grad(set_to_none=True)
    return result


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
    model.train()
    model.step = int(args.probe_step)

    selected = None
    probes = None
    for image_idx, camera, batch in _train_camera_items(pipeline, args.max_images):
        candidate_probes = [
            _run_source_probe(model, camera, batch, "J_gaussian_raw", args),
            _run_source_probe(model, camera, batch, "J_proxy_raw", args),
        ]
        if any(probe.get("gmvc_intrinsic_source_available", 0.0) > 0.0 for probe in candidate_probes):
            selected = (image_idx, camera, batch)
            probes = candidate_probes
            break
    if selected is None or probes is None:
        raise RuntimeError("No train camera produced valid GMVC intrinsic samples.")
    image_idx, _camera, _batch = selected
    result = {
        "diagnostic": "gmvc_intrinsic_gradient_paths",
        "experiment_name": getattr(config, "experiment_name", ""),
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "load_step": int(step),
        "probe_step": int(args.probe_step),
        "train_image_index": int(image_idx),
        "gmvc_track_bank": str(args.gmvc_track_bank),
        "lambda_gmvc_intrinsic": float(args.lambda_gmvc_intrinsic),
        "use_dc_proxy": bool(args.use_dc_proxy),
        "probes": probes,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--probe-step", type=int, default=10001)
    parser.add_argument("--gmvc-track-bank", type=Path, required=True)
    parser.add_argument("--lambda-gmvc-intrinsic", type=float, default=1.0)
    parser.add_argument("--max-tracks", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--use-dc-proxy", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
