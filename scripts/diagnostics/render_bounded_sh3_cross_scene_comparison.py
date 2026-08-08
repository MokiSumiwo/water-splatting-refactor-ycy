#!/usr/bin/env python
"""Render M1-vs-BND cross-scene comparison sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw


SHEETS = (
    ("underwater", "contact_sheet_underwater_m1_vs_bnd.png", "underwater_rgb", True),
    ("clear_raw", "contact_sheet_clear_raw_m1_vs_bnd.png", "clear_object_fullsh_raw_display", False),
    ("clear_clamp01", "contact_sheet_clear_clamp01_m1_vs_bnd.png", "clear_object_fullsh_clamp01", False),
    ("direct_object_signal", "contact_sheet_direct_object_signal_m1_vs_bnd.png", "direct_object_signal", False),
    ("transmission", "contact_sheet_transmission_m1_vs_bnd.png", "transmission", False),
    ("tau_D", "contact_sheet_tau_d_m1_vs_bnd.png", "tau_D_visualization", False),
)


def _load_summary(root: Path) -> Dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf8"))


def _parse_pair(text: str) -> Tuple[str, Path, Path]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--scene-pair must be SCENE:M1_ROOT:BND_ROOT, got {text}")
    return parts[0], Path(parts[1]), Path(parts[2])


def _image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _diff(path_a: Path, path_b: Path) -> Image.Image:
    a = _image(path_a)
    b = _image(path_b)
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.BILINEAR)
    return ImageChops.difference(a, b)


def _mask_from_j(path: Path) -> Image.Image:
    src = _image(path)
    mask = Image.new("RGB", src.size, "black")
    src_pixels = src.load()
    dst_pixels = mask.load()
    for y in range(src.height):
        for x in range(src.width):
            dst_pixels[x, y] = (255, 255, 255) if max(src_pixels[x, y]) >= 253 else (0, 0, 0)
    return mask


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 30
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((6, 8), label, fill="black")
    return canvas


def _sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int) -> Tuple[int, int]:
    rendered = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
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


def _assert_match(scene: str, m1: Mapping[str, Any], bnd: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rows_m1 = list(m1.get("per_view", []))
    rows_bnd = list(bnd.get("per_view", []))
    views_m1 = [int(row["view_id"]) for row in rows_m1]
    views_bnd = [int(row["view_id"]) for row in rows_bnd]
    cams_m1 = [int(row["camera_id"]) for row in rows_m1]
    cams_bnd = [int(row["camera_id"]) for row in rows_bnd]
    if views_m1 != views_bnd or cams_m1 != cams_bnd:
        raise ValueError(f"{scene} view/camera mismatch: M1 {views_m1}/{cams_m1}; BND {views_bnd}/{cams_bnd}")
    return rows_m1, rows_bnd


def _file(row: Mapping[str, Any], component: str) -> Path:
    return Path(row.get("files", {}).get(component, ""))


def render(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = [_parse_pair(item) for item in args.scene_pair]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    scene_records: Dict[str, Dict[str, Any]] = {}

    def record(scene: str, path: Path, output_type: str, view_ids: Sequence[int], camera_ids: Sequence[int], width: int, height: int, note: str = "") -> None:
        manifest.append(
            {
                "file_path": str(path),
                "scene": scene,
                "output_type": output_type,
                "runs": "M1;BND",
                "step": 15000,
                "view_ids": ";".join(str(v) for v in view_ids),
                "camera_ids": ";".join(str(v) for v in camera_ids),
                "width": width,
                "height": height,
                "display_logic": "diagnostic PNGs reused directly; no auto exposure, WB, gamma, histogram equalization, or per-image normalization",
                "notes": note,
            }
        )

    for scene, m1_root, bnd_root in pairs:
        m1 = _load_summary(m1_root)
        bnd = _load_summary(bnd_root)
        rows_m1, rows_bnd = _assert_match(scene, m1, bnd)
        view_ids = [int(row["view_id"]) for row in rows_m1]
        camera_ids = [int(row["camera_id"]) for row in rows_m1]
        scene_records[scene] = {"m1": rows_m1, "bnd": rows_bnd, "view_ids": view_ids, "camera_ids": camera_ids}
        scene_dir = args.output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        for output_type, filename, component, include_gt in SHEETS:
            rows = []
            for row_m1, row_bnd in zip(rows_m1, rows_bnd):
                view_id = int(row_m1["view_id"])
                camera_id = int(row_m1["camera_id"])
                m1_path = _file(row_m1, component)
                bnd_path = _file(row_bnd, component)
                sheet_row: List[Tuple[str, Image.Image]] = []
                if include_gt:
                    sheet_row.append((f"{scene} view {view_id} cam {camera_id} | GT", _image(_file(row_m1, "gt_underwater"))))
                sheet_row.extend(
                    [
                        (f"{scene} view {view_id} cam {camera_id} | M1 {output_type}", _image(m1_path)),
                        (f"{scene} view {view_id} cam {camera_id} | BND {output_type}", _image(bnd_path)),
                        (f"{scene} view {view_id} cam {camera_id} | abs diff", _diff(m1_path, bnd_path)),
                    ]
                )
                rows.append(sheet_row)
            out_path = scene_dir / filename
            width, height = _sheet(out_path, rows, args.tile_width)
            record(scene, out_path, output_type, view_ids, camera_ids, width, height)

        mask_rows = []
        for row_bnd in rows_bnd:
            view_id = int(row_bnd["view_id"])
            camera_id = int(row_bnd["camera_id"])
            clear_path = _file(row_bnd, "clear_object_fullsh_raw_display")
            mask_rows.append(
                [
                    (f"{scene} view {view_id} cam {camera_id} | BND clear raw", _image(clear_path)),
                    (f"{scene} view {view_id} cam {camera_id} | display J>=0.99 mask", _mask_from_j(clear_path)),
                ]
            )
        mask_path = scene_dir / "boundary_saturation_mask_bnd.png"
        width, height = _sheet(mask_path, mask_rows, args.tile_width)
        record(
            scene,
            mask_path,
            "boundary_saturation_mask_bnd",
            view_ids,
            camera_ids,
            width,
            height,
            "mask is derived from BND clear raw display PNG with any RGB channel >=253/255; diagnostic-only",
        )

    summary_dir = args.output_dir / "four_scene_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    clear_rows = []
    underwater_rows = []
    for scene in [item[0] for item in pairs]:
        row_m1 = scene_records[scene]["m1"][0]
        row_bnd = scene_records[scene]["bnd"][0]
        view_id = int(row_m1["view_id"])
        camera_id = int(row_m1["camera_id"])
        clear_rows.append(
            [
                (f"{scene} view {view_id} cam {camera_id} | M1 clear raw", _image(_file(row_m1, "clear_object_fullsh_raw_display"))),
                (f"{scene} view {view_id} cam {camera_id} | BND clear raw", _image(_file(row_bnd, "clear_object_fullsh_raw_display"))),
            ]
        )
        underwater_rows.append(
            [
                (f"{scene} view {view_id} cam {camera_id} | GT", _image(_file(row_m1, "gt_underwater"))),
                (f"{scene} view {view_id} cam {camera_id} | M1", _image(_file(row_m1, "underwater_rgb"))),
                (f"{scene} view {view_id} cam {camera_id} | BND", _image(_file(row_bnd, "underwater_rgb"))),
            ]
        )
    clear_path = summary_dir / "four_scene_clear_raw_summary.png"
    width, height = _sheet(clear_path, clear_rows, args.tile_width)
    record("four_scene", clear_path, "four_scene_clear_raw_summary", [0], [0], width, height, "first diagnostic eval view per scene")
    underwater_path = summary_dir / "four_scene_underwater_summary.png"
    width, height = _sheet(underwater_path, underwater_rows, args.tile_width)
    record("four_scene", underwater_path, "four_scene_underwater_summary", [0], [0], width, height, "first diagnostic eval view per scene")

    manifest_json = args.output_dir / "manifest.json"
    manifest_csv = args.output_dir / "manifest.csv"
    index = args.output_dir / "VISUAL_COMPARE_INDEX.md"
    payload = {"scene_pairs": [item[0] for item in pairs], "files": manifest}
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    with manifest_csv.open("w", newline="", encoding="utf8") as handle:
        fieldnames = sorted({key for row in manifest for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = ["# Bounded SH3 Cross-Scene Visual Compare", ""]
    for scene, _, _ in pairs:
        lines.extend([f"## {scene}", ""])
        for row in manifest:
            if row["scene"] == scene:
                lines.append(f"- {row['output_type']}: `{row['file_path']}`")
        lines.append("")
    lines.extend(
        [
            "## Four Scene Summary",
            "",
            f"- clear raw: `{clear_path}`",
            f"- underwater: `{underwater_path}`",
            "",
            "## Manifest",
            "",
            f"- JSON: `{manifest_json}`",
            f"- CSV: `{manifest_csv}`",
        ]
    )
    index.write_text("\n".join(lines) + "\n", encoding="utf8")
    return {"output_dir": str(args.output_dir), "manifest": str(manifest_json), "files": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-pair", action="append", required=True, help="SCENE:M1_DIAG_ROOT:BND_DIAG_ROOT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=340)
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
