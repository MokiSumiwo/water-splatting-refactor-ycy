#!/usr/bin/env python
"""Audit which rendered outputs send gradients to model parameters.

This script is intentionally diagnostic-only. It loads a checkpoint, runs a
small number of eval cameras, backpropagates one scalar output probe at a time,
and records gradient statistics for Gaussian and medium parameters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


ProbeFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _tensor_stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"norm": 0.0, "abs_mean": 0.0, "abs_max": 0.0, "nonzero_ratio": 0.0}
    abs_flat = flat.abs()
    return {
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "abs_mean": float(abs_flat.mean().item()),
        "abs_max": float(abs_flat.max().item()),
        "nonzero_ratio": float((abs_flat > 1e-12).float().mean().item()),
    }


def _zero_like_stats() -> Dict[str, float]:
    return {"norm": 0.0, "abs_mean": 0.0, "abs_max": 0.0, "nonzero_ratio": 0.0}


def _named_parameters(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    if hasattr(model, "gauss_params"):
        for name, param in model.gauss_params.items():
            yield f"gaussian.{name}", param
    if hasattr(model, "medium_mlp"):
        for name, param in model.medium_mlp.named_parameters():
            yield f"medium_mlp.{name}", param
    if hasattr(model, "direction_encoding"):
        for name, param in model.direction_encoding.named_parameters():
            yield f"direction_encoding.{name}", param


def _clear_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    if hasattr(model, "xys") and getattr(model, "xys") is not None and getattr(model.xys, "retains_grad", False):
        if model.xys.grad is not None:
            model.xys.grad = None


def _collect_grad_stats(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name, param in _named_parameters(model):
        stats[name] = _tensor_stats(param.grad) if param.grad is not None else _zero_like_stats()
    if hasattr(model, "xys") and getattr(model, "xys") is not None:
        xys = model.xys
        if getattr(xys, "retains_grad", False) and xys.grad is not None:
            stats["screen.xys"] = _tensor_stats(xys.grad)
        else:
            stats["screen.xys"] = _zero_like_stats()
    if hasattr(model, "xys_grad_abs") and getattr(model, "xys_grad_abs") is not None:
        stats["screen.xys_grad_abs"] = _tensor_stats(model.xys_grad_abs)
    if hasattr(model, "xys_grad_abs_proxy") and getattr(model, "xys_grad_abs_proxy") is not None:
        stats["screen.xys_grad_abs_proxy"] = _tensor_stats(model.xys_grad_abs_proxy)
    return stats


def _sum_norm(stats: Dict[str, Dict[str, float]], prefix: str) -> float:
    return float(sum(v["norm"] for k, v in stats.items() if k.startswith(prefix)))


def _probe_scalar(outputs: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in outputs:
        raise KeyError(key)
    value = outputs[key]
    return value.float().mean()


def _build_probes() -> Dict[str, ProbeFn]:
    keys = (
        "rgb",
        "rgb_object",
        "rgb_medium",
        "rgb_medium_total",
        "J_gaussian_raw",
        "accumulation",
        "final_transmittance",
        "rgb_tail",
        "J_proxy_raw",
        "J_proxy",
        "J_proxy_abs_diff_from_renderer_clear",
    )
    return {key: (lambda outputs, probe_key=key: _probe_scalar(outputs, probe_key)) for key in keys}


def _run_probe(model: torch.nn.Module, camera: Any, probe_name: str, probe_fn: ProbeFn) -> Dict[str, Any]:
    _clear_grads(model)
    # Do not use Nerfstudio's get_outputs_for_camera here: the base method is
    # decorated with torch.no_grad(), which would hide the paths being audited.
    outputs = model.get_outputs(camera)
    result: Dict[str, Any] = {
        "probe": probe_name,
        "available": probe_name in outputs,
        "requires_grad": False,
        "scalar": 0.0,
        "parameter_stats": {},
        "aggregate": {},
        "error": None,
    }
    try:
        scalar = probe_fn(outputs)
    except Exception as exc:
        result["available"] = False
        result["error"] = repr(exc)
        return result

    result["requires_grad"] = bool(scalar.requires_grad)
    result["scalar"] = float(scalar.detach().item())
    if not scalar.requires_grad:
        return result

    try:
        scalar.backward()
        stats = _collect_grad_stats(model)
        result["parameter_stats"] = stats
        result["aggregate"] = {
            "gaussian_norm": _sum_norm(stats, "gaussian."),
            "medium_norm": _sum_norm(stats, "medium_mlp.") + _sum_norm(stats, "direction_encoding."),
            "screen_xys_norm": stats.get("screen.xys", _zero_like_stats())["norm"],
            "screen_abs_grad_norm": stats.get("screen.xys_grad_abs", _zero_like_stats())["norm"],
            "screen_proxy_abs_grad_norm": stats.get("screen.xys_grad_abs_proxy", _zero_like_stats())["norm"],
            "opacities_norm": stats.get("gaussian.opacities", _zero_like_stats())["norm"],
            "scales_norm": stats.get("gaussian.scales", _zero_like_stats())["norm"],
            "features_dc_norm": stats.get("gaussian.features_dc", _zero_like_stats())["norm"],
            "means_norm": stats.get("gaussian.means", _zero_like_stats())["norm"],
        }
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        _clear_grads(model)
    return result


def _aggregate_probe_results(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_probe: Dict[str, List[Dict[str, Any]]] = {}
    for image in images:
        for probe in image["probes"]:
            by_probe.setdefault(probe["probe"], []).append(probe)

    aggregate: Dict[str, Any] = {}
    for probe_name, rows in sorted(by_probe.items()):
        keys = (
            "gaussian_norm",
            "medium_norm",
            "screen_xys_norm",
            "screen_abs_grad_norm",
            "screen_proxy_abs_grad_norm",
            "opacities_norm",
            "scales_norm",
            "features_dc_norm",
            "means_norm",
        )
        entry: Dict[str, Any] = {
            "count": len(rows),
            "available_count": sum(bool(row.get("available")) for row in rows),
            "requires_grad_count": sum(bool(row.get("requires_grad")) for row in rows),
            "error_count": sum(row.get("error") is not None for row in rows),
        }
        for key in keys:
            vals = torch.tensor(
                [float(row.get("aggregate", {}).get(key, 0.0)) for row in rows],
                dtype=torch.float32,
            )
            entry[f"{key}_mean"] = float(vals.mean().item()) if vals.numel() else 0.0
            entry[f"{key}_max"] = float(vals.max().item()) if vals.numel() else 0.0
        aggregate[probe_name] = entry
    return aggregate


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
    if args.enable_clear_proxy:
        model.config.clear_proxy_enabled = True
    probes = _build_probes()
    images: List[Dict[str, Any]] = []

    for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if image_idx >= args.max_images:
            break
        image_entry: Dict[str, Any] = {"image_index": image_idx, "probes": []}
        for probe_name, probe_fn in probes.items():
            image_entry["probes"].append(_run_probe(model, camera, probe_name, probe_fn))
        images.append(image_entry)

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "max_images": int(args.max_images),
        "enable_clear_proxy": bool(args.enable_clear_proxy),
        "aggregate": _aggregate_probe_results(images),
        "images": images,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--enable-clear-proxy", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
