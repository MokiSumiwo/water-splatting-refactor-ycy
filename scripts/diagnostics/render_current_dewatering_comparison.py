#!/usr/bin/env python3
"""Build a compact four-scene dewatering comparison from retained renders.

The figure is visualization-only. It does not alter RGB values beyond the
existing PNG files, aspect-preserving resize, labels, and layout composition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
VIEWS = {
    "Curasao": "MTN_1296",
    "IUI3-RedSea": "MTN_5911",
    "JapaneseGradens-RedSea": "MTN_1098",
    "Panama": "MTN_1539",
}
SOURCE_ROOT = REPO_ROOT / "renders" / "legacy_ocmc_cica_no_auxiliary_20260905"
INPUT_ROOT = REPO_ROOT / "undistorted_data"
DEFAULT_OUTPUT = REPO_ROOT / "renders" / "current_dewatering_cica_no_auxiliary_20260905"


def _font(size: int, bold: bool = False) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_path(scene: str, view: str) -> Path:
    return INPUT_ROOT / f"undistorted_{scene}" / "images" / "ColorImage" / f"{view}.png"


def _render_path(scene: str, view: str, name: str) -> Path:
    return SOURCE_ROOT / scene / "step_14999" / view / "A2" / name


def _fit(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 24, 28))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def _make_sheet(output: Path, columns: Sequence[Tuple[str, str]], tile: Tuple[int, int]) -> List[Dict[str, str]]:
    margin = 28
    label_h = 44
    row_label_w = 340
    title_h = 66
    gap = 12
    width = margin * 2 + row_label_w + gap + len(columns) * tile[0] + (len(columns) - 1) * gap
    height = margin * 2 + title_h + len(SCENES) * (label_h + tile[1]) + (len(SCENES) - 1) * gap
    sheet = Image.new("RGB", (width, height), (245, 246, 247))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(28, bold=True)
    header_font = _font(19, bold=True)
    scene_font = _font(20, bold=True)
    note_font = _font(15)
    draw.text((margin, margin), "Current Dewatering Comparison", fill=(22, 27, 31), font=title_font)
    draw.text((margin, margin + 36), "Legacy SH3 + M1 + OCMC + CICA, Auxiliary OFF, step 14999", fill=(75, 84, 91), font=note_font)

    manifest_rows: List[Dict[str, str]] = []
    x0 = margin + row_label_w + gap
    for col_idx, (label, _) in enumerate(columns):
        x = x0 + col_idx * (tile[0] + gap)
        draw.rectangle((x, margin + title_h - 2, x + tile[0], margin + title_h + label_h - 4), fill=(224, 229, 232))
        bbox = draw.textbbox((0, 0), label, font=header_font)
        draw.text((x + (tile[0] - (bbox[2] - bbox[0])) // 2, margin + title_h + 7), label, fill=(26, 33, 38), font=header_font)

    y = margin + title_h + label_h
    for scene in SCENES:
        view = VIEWS[scene]
        draw.rectangle((margin, y, margin + row_label_w, y + tile[1]), fill=(35, 45, 51))
        draw.text((margin + 15, y + 16), scene, fill=(255, 255, 255), font=scene_font)
        for col_idx, (_, name) in enumerate(columns):
            path = _input_path(scene, view) if name == "input" else _render_path(scene, view, name)
            if not path.is_file():
                raise FileNotFoundError(path)
            x = x0 + col_idx * (tile[0] + gap)
            sheet.paste(_fit(Image.open(path), tile), (x, y))
            manifest_rows.append({
                "scene": scene,
                "view_id": view,
                "column": name,
                "path": str(path),
                "sha256": _sha256(path),
            })
        y += label_h + tile[1] + gap
    sheet.save(output, format="PNG", optimize=True)
    return manifest_rows


def run(output_dir: Path) -> Mapping[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tile = (440, 292)
    main_columns = (
        ("Input underwater", "input"),
        ("Model underwater", "underwater.png"),
        ("Native clear", "clear.png"),
    )
    raw_columns = (
        ("Input underwater", "input"),
        ("Model underwater", "underwater.png"),
        ("Raw clear clamp01", "clear_raw_display_clamp01.png"),
    )
    main_path = output_dir / "current_dewatering_comparison.png"
    raw_path = output_dir / "current_dewatering_raw_clamp01_comparison.png"
    main_rows = _make_sheet(main_path, main_columns, tile)
    raw_rows = _make_sheet(raw_path, raw_columns, tile)
    manifest = {
        "configuration": "Legacy SH3 + dir_xy_camera + OCMC ON + CICA ON + Auxiliary OFF + RAOC OFF + MDRR OFF",
        "absolute_step": 14999,
        "paired_real_clear_gt_found": False,
        "postprocessing": "aspect-preserving resize, neutral letterbox, labels, borders, and PNG encoding only",
        "selected_views": VIEWS,
        "main_png": str(main_path),
        "raw_clamp01_png": str(raw_path),
        "main_rows": main_rows,
        "raw_rows": raw_rows,
    }
    (output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    with (output_dir / "comparison_manifest.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ["sheet", "scene", "view_id", "column", "path", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sheet_name, rows in (("native", main_rows), ("raw_clamp01", raw_rows)):
            for row in rows:
                writer.writerow({"sheet": sheet_name, **row})
    return {"main": str(main_path), "raw_clamp01": str(raw_path), "manifest": str(output_dir / "comparison_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
