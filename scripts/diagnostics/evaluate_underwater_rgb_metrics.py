#!/usr/bin/env python3
"""Evaluate WaterSplatting RGB metrics with the project's ``metrics.py`` semantics.

This evaluator is intentionally separate from ``ns-eval``.  It uses the same
image loader and metric implementations as ``metrics.py``:

* images are converted to RGB and represented as float tensors in [0, 1];
* prediction and ground truth files are matched by the exact filename;
* no resize, crop, or color-space conversion is performed;
* PSNR, SSIM, and local lpipsPyTorch LPIPS-VGG v0.1 are computed per view;
* the reported aggregate is the arithmetic mean over valid views.

The per-view CSV makes it possible to audit missing files, shape mismatches,
and failed metric computations instead of silently changing the evaluation
set.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the project module so the loader and metric implementations remain
# tied to metrics.py rather than to an ns-eval or torchvision replacement.
import metrics as project_metrics  # noqa: E402
from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp"}
def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _iter_image_names(directory: Path) -> Iterable[str]:
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path.name


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().reshape(-1).mean().cpu().item())


def _load(path: Path) -> torch.Tensor:
    # _load_rgb_tensor contains the exact PIL/imageio and [0, 1] handling used
    # by metrics.py.  Its module-level device is set by main() below.
    return project_metrics._load_rgb_tensor(path)


def _metric_row(
    name: str,
    pred_dir: Path,
    gt_dir: Path,
    lpips_model: LPIPS,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "prediction_file": name,
        "ground_truth_file": name,
        "status": "pending",
        "pred_height": None,
        "pred_width": None,
        "gt_height": None,
        "gt_width": None,
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "MSE": None,
        "error": "",
    }

    pred_path = pred_dir / name
    gt_path = gt_dir / name
    if not gt_path.is_file():
        row["status"] = "missing_gt"
        row["error"] = f"ground-truth file not found: {gt_path}"
        return row

    try:
        pred = _load(pred_path)
    except Exception as exc:
        row["status"] = "pred_read_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        gt = _load(gt_path)
    except Exception as exc:
        row["status"] = "gt_read_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["pred_height"] = int(pred.shape[-2])
        row["pred_width"] = int(pred.shape[-1])
        return row

    row["pred_height"] = int(pred.shape[-2])
    row["pred_width"] = int(pred.shape[-1])
    row["gt_height"] = int(gt.shape[-2])
    row["gt_width"] = int(gt.shape[-1])

    if pred.shape != gt.shape:
        row["status"] = "size_mismatch"
        row["error"] = (
            f"prediction shape {tuple(pred.shape)} != "
            f"ground-truth shape {tuple(gt.shape)}; no resize is applied"
        )
        return row

    try:
        with torch.no_grad():
            # These are deliberately the same reference calls used in
            # metrics._eval_rgb_set, plus MSE from the same loaded tensors.
            row["SSIM"] = _scalar(project_metrics.ssim(pred, gt))
            row["PSNR"] = _scalar(project_metrics.psnr(pred, gt))
            row["LPIPS"] = _scalar(lpips_model(pred, gt))
            row["MSE"] = _scalar((pred - gt).square().mean(dim=(1, 2, 3)))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "metric_error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def _mean(rows: List[Dict[str, object]], metric: str) -> Optional[float]:
    values = [row[metric] for row in rows if row["status"] == "ok" and row[metric] is not None]
    if not values:
        return None
    # metrics.py stacks float32 per-view tensors before taking the mean. Keep
    # that detail so the aggregate is numerically reproducible as well.
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prediction_file",
        "ground_truth_file",
        "status",
        "pred_height",
        "pred_width",
        "gt_height",
        "gt_width",
        "PSNR",
        "SSIM",
        "LPIPS",
        "MSE",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception:
        versions["torchvision"] = "unavailable"
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except Exception:
        versions["pillow"] = "unavailable"
    return versions


def evaluate(
    pred_dir: Path,
    gt_dir: Path,
    output_csv: Path,
    output_json: Path,
    device: torch.device,
    strict: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ground-truth directory does not exist: {gt_dir}")

    # metrics.py's loader refers to its module-level DEVICE at call time.
    project_metrics.DEVICE = device

    names = list(_iter_image_names(pred_dir))
    print(f"[RGB metrics] pred_dir={pred_dir}")
    print(f"[RGB metrics] gt_dir={gt_dir}")
    print(f"[RGB metrics] prediction images={len(names)} device={device}")
    print("[RGB metrics] loading local lpipsPyTorch VGG v0.1 weights...")
    lpips_model = LPIPS(net_type="vgg", version="0.1").to(device).eval()

    rows = []
    for index, name in enumerate(names, start=1):
        row = _metric_row(name, pred_dir, gt_dir, lpips_model)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"[{index:04d}/{len(names):04d}] {name}: "
                f"PSNR={float(row['PSNR']):.6f} "
                f"SSIM={float(row['SSIM']):.6f} "
                f"LPIPS={float(row['LPIPS']):.6f}"
            )
        else:
            print(f"[{index:04d}/{len(names):04d}] {name}: {row['status']} ({row['error']})")

    valid_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    aggregate = {
        "num_prediction_images": len(names),
        "num_valid_pairs": len(valid_rows),
        "num_failed_pairs": len(failed_rows),
        "num_missing_gt": sum(row["status"] == "missing_gt" for row in rows),
        "num_size_mismatch": sum(row["status"] == "size_mismatch" for row in rows),
        "num_read_errors": sum("read_error" in str(row["status"]) for row in rows),
        "PSNR": _mean(rows, "PSNR"),
        "SSIM": _mean(rows, "SSIM"),
        "LPIPS": _mean(rows, "LPIPS"),
        "MSE": _mean(rows, "MSE"),
    }

    payload = {
        "evaluation": "WaterSplatting underwater RGB metrics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "prediction_directory": str(pred_dir.resolve()),
        "ground_truth_directory": str(gt_dir.resolve()),
        "device": str(device),
        "metric_source": {
            "loader": "metrics.py::_load_rgb_tensor",
            "psnr": "utils.image_utils.psnr",
            "ssim": "utils.loss_utils.ssim",
            "lpips": "lpipsPyTorch.modules.lpips.LPIPS",
            "lpips_net": "vgg",
            "lpips_version": "0.1",
            "mse": "mean squared RGB error over the same [0, 1] tensors",
            "input_range": "[0, 1]",
            "pairing": "exact prediction filename == ground-truth filename",
            "spatial_processing": "none: no resize, crop, or padding",
            "aggregation": "arithmetic mean of per-view values over status=ok rows",
        },
        "versions": _versions(),
        "aggregate": aggregate,
        "rows": rows,
    }
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n[RGB metrics] aggregate over valid views")
    print(
        f"pairs={aggregate['num_valid_pairs']}/{aggregate['num_prediction_images']} "
        f"PSNR={aggregate['PSNR']} SSIM={aggregate['SSIM']} LPIPS={aggregate['LPIPS']}"
    )
    print(f"[RGB metrics] per-view CSV: {output_csv}")
    print(f"[RGB metrics] audit JSON:    {output_json}")

    if strict and failed_rows:
        raise RuntimeError(f"strict mode: {len(failed_rows)} image pairs failed")
    if not valid_rows:
        raise RuntimeError("no valid prediction/ground-truth pairs were evaluated")
    return aggregate, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR/SSIM/LPIPS using the WaterSplatting metrics.py RGB protocol."
    )
    parser.add_argument("--pred-dir", required=True, type=Path, help="directory containing rendered RGB images")
    parser.add_argument("--gt-dir", required=True, type=Path, help="directory containing matching RGB ground truth")
    parser.add_argument("--output-csv", type=Path, default=Path("underwater_rgb_metrics.csv"), help="per-view CSV path")
    parser.add_argument("--output-json", type=Path, default=Path("underwater_rgb_metrics.json"), help="audit JSON path")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="metric device")
    parser.add_argument("--strict", action="store_true", help="fail if any prediction image is not a valid pair")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        device = _resolve_device(args.device)
        evaluate(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            output_csv=args.output_csv,
            output_json=args.output_json,
            device=device,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"[RGB metrics] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
