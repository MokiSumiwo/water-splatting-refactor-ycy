"""Freeze the manually reviewed IUI3 open-water masks for experiment C.

The mask is deliberately a small, fixed image-space box from the v2 manual
review.  It is applied only to real training cameras; test/evaluation cameras
are never used to construct the auxiliary target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCENE = "IUI3-RedSea"
DATA = ROOT / "undistorted_data" / "undistorted_IUI3-RedSea"
EVIDENCE = ROOT / "research_runs" / "C_preflight_20261024" / "audit" / "iui3_O_candidate" / "candidate_manifest_v2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(EVIDENCE.read_text(encoding="utf8"))
    if manifest.get("scene") != SCENE or manifest.get("status") != "CANDIDATE_PENDING_MANUAL_REVIEW":
        raise RuntimeError("unexpected candidate evidence manifest")
    x0, y0, x1, y1 = manifest["normalized_xyxy"]
    train_names = [Path(name).stem for name in (DATA / "train_list.txt").read_text().splitlines() if name.strip()]
    rows = []
    for name in train_names:
        image_path = DATA / "images" / "ColorImage" / f"{name}.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        left = int(round(x0 * width))
        top = int(round(y0 * height))
        right = int(round(x1 * width))
        bottom = int(round(y1 * height))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        mask_path = output / f"{name}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        rows.append({
            "camera": name,
            "mask": str(mask_path),
            "width": width,
            "height": height,
            "box_xyxy_exclusive": [left, top, right, bottom],
            "pixels": int(mask.astype(bool).sum()),
            "fraction": float(mask.astype(bool).mean()),
            "mask_sha256": sha256(mask_path),
            "source_image_sha256": sha256(image_path),
        })
    payload = {
        "experiment": "C_OPEN_WATER_NATIVE_TAIL",
        "scene": SCENE,
        "state": "FROZEN_IUI3_ONLY",
        "label": "O",
        "source_evidence": str(EVIDENCE),
        "source_evidence_sha256": sha256(EVIDENCE),
        "normalized_xyxy": [x0, y0, x1, y1],
        "train_only": True,
        "test_masks_created": False,
        "unknown_outside_mask": True,
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(output), "scene": SCENE, "views": len(rows), "pixels_per_view": rows[0]["pixels"]}, indent=2))


if __name__ == "__main__":
    main()
