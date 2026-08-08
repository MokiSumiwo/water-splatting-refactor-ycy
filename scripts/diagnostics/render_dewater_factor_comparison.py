#!/usr/bin/env python
"""Render generic dewatering-factor comparison contact sheets from diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw


ALPHAS = ("0p00", "0p25", "0p50", "0p75", "1p00")


def _load_summary(root: Path) -> Dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf8"))


def _image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "run"


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 30
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((6, 8), label, fill="black")
    return canvas


def _sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Path]]], tile_width: int) -> Tuple[int, int]:
    if not rows:
        raise ValueError(f"No rows for {path}")
    rendered = []
    for row in rows:
        tiles = [_tile(_image(tile_path), label, tile_width) for label, tile_path in row]
        width = sum(tile.width for tile in tiles) + 8 * (len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        row_img = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            row_img.paste(tile, (x, 0))
            x += tile.width + 8
        rendered.append(row_img)
    sheet_width = max(row.width for row in rendered)
    sheet_height = sum(row.height for row in rendered) + 8 * (len(rendered) - 1)
    out = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered:
        out.paste(row, (0, y))
        y += row.height + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return sheet_width, sheet_height


def _per_view(root: Path, scene: str, view_id: int, component: str) -> Path:
    return root / scene / "per_view" / f"view_{view_id:04d}" / f"{component}.png"


def _alpha(root: Path, scene: str, view_id: int, alpha: str) -> Path:
    return root / scene / "alpha_sweep" / f"view_{view_id:04d}" / f"partial_deattenuation_alpha_{alpha}.png"


def _parse_run(item: str) -> Tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"--run must be NAME=ROOT, got {item}")
    name, root = item.split("=", 1)
    return name, Path(root)


def _assert_matching_views(run_summaries: Mapping[str, Mapping[str, Any]]) -> Tuple[List[int], List[int], List[str]]:
    first_name = next(iter(run_summaries))
    first_rows = run_summaries[first_name].get("per_view", [])
    view_ids = [int(row["view_id"]) for row in first_rows]
    camera_ids = [int(row["camera_id"]) for row in first_rows]
    image_names = [str(row.get("image_name", "")) for row in first_rows]
    for name, summary in run_summaries.items():
        rows = summary.get("per_view", [])
        current_views = [int(row["view_id"]) for row in rows]
        current_cameras = [int(row["camera_id"]) for row in rows]
        if current_views != view_ids or current_cameras != camera_ids:
            raise ValueError(
                f"View/camera mismatch for {name}: views={current_views}, cameras={current_cameras}; "
                f"expected views={view_ids}, cameras={camera_ids}"
            )
    return view_ids, camera_ids, image_names


def render(args: argparse.Namespace) -> Dict[str, Any]:
    runs = [_parse_run(item) for item in args.run]
    roots = {name: root for name, root in runs}
    summaries = {name: _load_summary(root) for name, root in runs}
    view_ids, camera_ids, image_names = _assert_matching_views(summaries)
    loaded_steps = {name: int(summary.get("loaded_step", args.step)) for name, summary in summaries.items()}
    manifest: List[Dict[str, Any]] = []
    run_names = [name for name, _ in runs]

    def add(filename: str, output_type: str, rows: Sequence[Sequence[Tuple[str, Path]]]) -> None:
        path = args.output_dir / filename
        width, height = _sheet(path, rows, int(args.tile_width))
        manifest.append(
            {
                "file_path": str(path),
                "scene": args.scene,
                "step": int(args.step),
                "loaded_steps": ";".join(f"{run}:{loaded_steps[run]}" for run in run_names),
                "output_type": output_type,
                "runs": ";".join(run_names),
                "view_ids": ";".join(str(v) for v in view_ids),
                "camera_ids": ";".join(str(v) for v in camera_ids),
                "width": width,
                "height": height,
                "display_logic": "reuse_existing_diagnostic_png_mapping; no auto exposure, WB, gamma, or per-image normalization",
            }
        )

    components = [
        ("underwater", "contact_sheet_underwater_factor_candidates.png", "underwater_rgb", True),
        ("direct_object_signal", "contact_sheet_direct_object_signal_factor_candidates.png", "direct_object_signal", False),
        ("clear_clamp01", "contact_sheet_clear_clamp01_factor_candidates.png", "clear_object_fullsh_clamp01", False),
        ("clear_ws_tonemap", "contact_sheet_clear_ws_tonemap_factor_candidates.png", "clear_object_fullsh_ws_tonemap", False),
        ("transmission", "contact_sheet_transmission_factor_candidates.png", "transmission", False),
        ("tau_D", "contact_sheet_tau_d_factor_candidates.png", "tau_D_visualization", False),
    ]
    for output_type, filename, component, include_gt in components:
        rows = []
        for view_id, camera_id, image_name in zip(view_ids, camera_ids, image_names):
            row: List[Tuple[str, Path]] = []
            if include_gt:
                row.append((f"view {view_id} cam {camera_id} {image_name} | GT", _per_view(roots[run_names[0]], args.scene, view_id, "gt_underwater")))
            for run in run_names:
                row.append((f"view {view_id} cam {camera_id} | {run} {output_type}", _per_view(roots[run], args.scene, view_id, component)))
            rows.append(row)
        add(filename, output_type, rows)

    for run in run_names:
        rows = []
        for view_id, camera_id, image_name in zip(view_ids, camera_ids, image_names):
            rows.append(
                [
                    (
                        f"view {view_id} cam {camera_id} {image_name} | {run} alpha {alpha.replace('p', '.')}",
                        _alpha(roots[run], args.scene, view_id, alpha),
                    )
                    for alpha in ALPHAS
                ]
            )
        add(f"contact_sheet_alpha_sweep_{_slug(run)}.png", f"alpha_sweep_{run}", rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = args.output_dir / "manifest.json"
    manifest_csv = args.output_dir / "manifest.csv"
    index = args.output_dir / "VISUAL_COMPARE_INDEX.md"
    payload = {
        "scene": args.scene,
        "step": int(args.step),
        "loaded_steps": loaded_steps,
        "runs": run_names,
        "view_ids": view_ids,
        "camera_ids": camera_ids,
        "image_names": image_names,
        "roots": {run: str(root) for run, root in roots.items()},
        "files": manifest,
    }
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    with manifest_csv.open("w", newline="", encoding="utf8") as handle:
        fieldnames = [
            "file_path",
            "scene",
            "step",
            "loaded_steps",
            "output_type",
            "runs",
            "view_ids",
            "camera_ids",
            "width",
            "height",
            "display_logic",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Dewatering Factor Step-15000 Visual Comparison",
        "",
        f"Scene: {args.scene}",
        f"Runs: {', '.join(run_names)}",
        f"View IDs: {', '.join(str(v) for v in view_ids)}",
        "",
        "## Contact Sheets",
        "",
    ]
    lines.extend(f"- {row['output_type']}: `{row['file_path']}`" for row in manifest)
    lines.extend(["", "## Manifests", "", f"- JSON: `{manifest_json}`", f"- CSV: `{manifest_csv}`"])
    index.write_text("\n".join(lines) + "\n", encoding="utf8")
    return {"output_dir": str(args.output_dir), "manifest": str(manifest_json), "files": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--run", action="append", required=True, help="Run root as NAME=path/to/diagnostic/root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=420)
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
