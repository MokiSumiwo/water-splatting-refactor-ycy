#!/usr/bin/env python
"""Build one low-resolution Curasao underwater/dewatered visual audit sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


COLUMNS = (
    ("GT underwater", "underwater", "gt.png"),
    ("A0 underwater", "underwater", "a0_underwater.png"),
    ("P30-MHOLD underwater", "underwater", "p30-mhold_underwater.png"),
    ("A0 dewatered proxy", "dewatered", "a0_j_proxy_or_clear.png"),
    ("P30-MHOLD dewatered proxy", "dewatered", "p30-mhold_j_proxy_or_clear.png"),
    ("A0 residual", "underwater", "a0_abs_residual.png"),
    ("P30-MHOLD residual", "underwater", "p30-mhold_abs_residual.png"),
)


def _tile(path: Path, label: str, width: int, label_height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height + label_height), "white")
    tile.paste(image, (0, label_height))
    ImageDraw.Draw(tile).text((4, 4), label, fill="black")
    return tile


def build_sheet(root: Path, output: Path, tile_width: int) -> tuple[int, int]:
    summary = json.loads((root / "visualization_summary.json").read_text(encoding="utf8"))
    label_height = 18
    rows: list[Image.Image] = []
    for view in summary["per_view"]:
        view_dir = root / f"view_{int(view['view_index']):04d}"
        row_tiles = [
            _tile(view_dir / group / filename, label, tile_width, label_height)
            for label, group, filename in COLUMNS
        ]
        row_height = max(tile.height for tile in row_tiles)
        header = Image.new("RGB", (tile_width, row_height), "white")
        ImageDraw.Draw(header).text((4, 4), f"view_{int(view['view_index']):04d}", fill="black")
        row = Image.new("RGB", (header.width + sum(tile.width for tile in row_tiles), row_height), "white")
        x = 0
        row.paste(header, (x, 0))
        x += header.width
        for tile in row_tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        rows.append(row)

    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return canvas.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("renders/gmvc_four_scene_p30_mhold_15k/Curasao/visualization"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "renders/gmvc_four_scene_p30_mhold_15k/Curasao/visualization/contact_sheets/"
            "curasao_underwater_dewatered_audit.png"
        ),
    )
    parser.add_argument("--tile-width", type=int, default=190)
    args = parser.parse_args()
    size = build_sheet(args.root, args.output, args.tile_width)
    print(json.dumps({"output": str(args.output), "size": size}))


if __name__ == "__main__":
    main()
