"""Recreate the manually reviewed, sparse training masks for the C extension.

These small image-space annotations are an explicit open-water prior, not a
segmentation model or a proof that no unseen surface exists. All other pixels
and all unlisted training views stay unknown and receive no C supervision.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

ROOT = Path(__file__).resolve().parents[2]
SELECTION = {
    "Curasao": {"cameras": [1293, 1294, 1298, 1299, 1302, 1306, 1307, 1308],
                "box": [.90, .02, .97, .07]},
    "JapaneseGradens-RedSea": {
        "cameras": [1093, 1094, 1095, 1096, 1097, 1099, 1100, 1101, 1102, 1103, 1104],
        "box": [.74, .02, .97, .10]},
    "Panama": {"cameras": [1534, 1535, 1543], "box": [.82, .015, .94, .045]},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(scene, output):
    data = ROOT / "undistorted_data" / ("undistorted_" + scene)
    names = [Path(n).stem for n in (data / "train_list.txt").read_text().splitlines() if n.strip()]
    tests = {Path(n).stem for n in (data / "test_list.txt").read_text().splitlines() if n.strip()}
    selected = {f"MTN_{n}" for n in SELECTION[scene]["cameras"]}
    assert selected <= set(names) and not set(names) & tests
    images_path = data / "sparse/0/images.bin"
    images = {Path(v.name).stem: v for v in read_images_binary(images_path).values()}
    output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (1440, ((len(names) + 3) // 4) * 260), "white")
    rows = []
    for i, name in enumerate(names):
        src = data / "images/ColorImage" / f"{name}.png"
        with Image.open(src) as image:
            image = image.convert("RGB")
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        box = None
        tracks = tracks_dil12 = 0
        if name in selected:
            box = [round(v * (w if j % 2 == 0 else h)) for j, v in enumerate(SELECTION[scene]["box"])]
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = 255
            xy = images[name].xys
            valid = images[name].point3D_ids >= 0
            def count(pad):
                return int(np.sum(valid & (xy[:, 0] >= x0-pad) & (xy[:, 0] < x1+pad)
                                  & (xy[:, 1] >= y0-pad) & (xy[:, 1] < y1+pad)))
            tracks, tracks_dil12 = count(0), count(12)
            # This veto complements visual review; absence of tracks alone is not an O label.
            if tracks_dil12:
                raise RuntimeError(f"registered-track conflict in frozen candidate: {scene}/{name}")
            ImageDraw.Draw(image).rectangle(box, outline="yellow", width=3)
        path = output / f"{name}.png"
        Image.fromarray(mask).save(path)
        rows.append(dict(camera=name, width=w, height=h, pixels=int((mask>0).sum()),
                         fraction=float((mask>0).mean()), box_xyxy_exclusive=box,
                         status="REVIEWED_O_PRIOR" if box else "UNKNOWN_NO_SUPERVISION",
                         mask_sha256=sha256(path), source_image_sha256=sha256(src),
                         registered_tracks=tracks, registered_tracks_dil12=tracks_dil12))
        image.thumbnail((360, 235))
        x, y = (i % 4)*360, (i // 4)*260
        sheet.paste(image, (x, y+22))
        ImageDraw.Draw(sheet).text((x+4, y+4), name + (" O" if box else " U / skip"), fill="black")
    sheet.save(output.parent / "review_overlay.jpg", quality=95)
    manifest = dict(scene=scene, state="FROZEN_SPARSE_TRAIN_ONLY", label="O",
                    train_only=True, test_masks_created=False, unknown_outside_mask=True,
                    normalized_xyxy=SELECTION[scene]["box"], rows=rows,
                    train_list_sha256=sha256(data/"train_list.txt"),
                    test_list_sha256=sha256(data/"test_list.txt"),
                    colmap_images_sha256=sha256(images_path),
                    review="Full training RGB, upper-half crops, enlarged original candidate crops, adjacent real training views; registered-track veto including 12px margin.",
                    limitation="Conservative manual O prior and approximate infinite water assumption; not dense geometry truth. Zero COLMAP tracks alone did not establish the annotation. Sparse view coverage is intentional.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(scene=scene, train_views=len(names), O_views=len(selected),
                         selected_pixel_fraction=next(r['fraction'] for r in rows if r['pixels']),
                         all_train_pixel_fraction=sum(r['fraction'] for r in rows)/len(rows))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for scene in SELECTION:
        prepare(scene, a.output / scene / "masks")
