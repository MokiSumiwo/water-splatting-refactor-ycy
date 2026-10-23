#!/usr/bin/env python3
"""Build documentation figures from retained native renders; no model execution."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_notes/assets/project_overview_20260917'
SOURCE = ROOT / 'outputs/fsb_cica_fork_repeat_20260916_v1'
LATEST = ROOT / 'outputs/far_field_three_candidates_20260917_v1/summary'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SCENES = ('Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama')
VIEWS = ('MTN_1296', 'MTN_5894', 'MTN_1098', 'MTN_1529')
FILENAMES = ('gt.png', 'underwater.png', 'clear_raw_display_clamp01.png')
BG, INK, MUTED = '#f3f5f7', '#172638', '#526175'
YELLOW, RED = '#ffd64c', '#ff577a'
manifest = {'source_configuration': 'FSB+CICA, BND ON, R0/C0a, step 14999',
            'selection': 'Manual explanatory examples, not quantitative ROI selection',
            'display': 'Existing raw clear clamp01 PNG; no exposure, color, or sharpening changes',
            'source_images': [], 'figures': [], 'metrics': {}}


def font(size):
    return ImageFont.truetype(FONT, size, index=0)


def label(canvas, xy, text, size=23, fill=INK):
    ImageDraw.Draw(canvas).text(xy, text, font=font(size), fill=fill)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(scene, view, filename):
    p = SOURCE / scene / 'R0/C0a/renders' / scene / 'step_14999' / view / 'C0a' / filename
    with Image.open(p) as image:
        im = image.convert('RGB')
    record = dict(path=str(p.relative_to(ROOT)), sha256=sha(p), size=list(im.size))
    if record not in manifest['source_images']:
        manifest['source_images'].append(record)
    return im


def fit(canvas, im, x, y, width, height):
    scale = min(width / im.width, height / im.height)
    size = (round(im.width * scale), round(im.height * scale))
    method = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST
    rendered = im.resize(size, method)
    canvas.paste(rendered, (x + (width-size[0])//2, y + (height-size[1])//2))


def save(canvas, name, **details):
    canvas.save(OUT / name)
    manifest['figures'].append(dict(file=name, size=list(canvas.size), **details))


def overview():
    canvas = Image.new('RGB', (1836, 1970), BG)
    label(canvas, (26, 16), '当前保留方案：带水新视角重建与去水体渲染', 34)
    label(canvas, (26, 69), 'BND ON · M1 + OCMC + CICA + FSB · step 14999 · 四场景真实测试视图', 24, MUTED)
    for row, (scene, view) in enumerate(zip(SCENES, VIEWS)):
        y = 118 + row*454
        label(canvas, (26, y), f'{scene} / {view}', 25)
        for j, (name, title) in enumerate(zip(FILENAMES, ('真实带水照片（GT）', '模型带水渲染', '模型去水渲染（没有 clear GT）'))):
            x = 24 + j*600
            label(canvas, (x, y+37), title, 22, MUTED)
            fit(canvas, native(scene, view, name), x, y+78, 588, 363)
    label(canvas, (26, 1934), '图像仅等比缩放排版；去水图为 raw clear 的 [0,1] 显示，不做调色、提亮或锐化。', 20, MUTED)
    save(canvas, '01_four_scene_overview.png', views=dict(zip(SCENES, VIEWS)))


def problem_sheet(scene, view, rois, name):
    canvas = Image.new('RGB', (1836, 980), BG)
    label(canvas, (24, 14), f'{scene} / {view}：两类远景问题', 32)
    label(canvas, (24, 62), '左：去水全图及位置标记。右：同一位置的带水／去水局部；未做颜色调整。', 23, MUTED)
    clear = native(scene, view, FILENAMES[2])
    under = native(scene, view, FILENAMES[1])
    marked = clear.copy()
    draw = ImageDraw.Draw(marked)
    for box, key, color, _ in rois:
        draw.rectangle(box, outline=color, width=7)
        draw.rectangle((box[0], box[1], box[0]+54, box[1]+58), fill=color)
        draw.text((box[0]+10, box[1]+1), key, font=font(38), fill=INK)
    fit(canvas, marked, 24, 125, 900, 694)
    label(canvas, (24, 830), 'A：远景表面偏色、对比度和细节不足。', 24)
    label(canvas, (24, 873), 'B：远处高斯球残留，表现为椭圆色块／条纹。', 24)
    label(canvas, (24, 916), 'B 是外观定位，并非逐个 Gaussian 的因果鉴定。', 21, MUTED)
    for i, (box, key, color, description) in enumerate(rois):
        y = 125 + i*402
        label(canvas, (956, y), f'{key}  {description}', 25)
        ImageDraw.Draw(canvas).rectangle((946, y+2, 950, y+38), fill=color)
        for j, (im, title) in enumerate(((under, '带水渲染'), (clear, '去水渲染'))):
            x = 956+j*438
            label(canvas, (x, y+44), title, 21, MUTED)
            fit(canvas, im.crop(box), x, y+83, 416, 278)
    save(canvas, name, scene=scene, view=view,
         rois=[dict(key=k, pixel_box=list(b), meaning=d) for b,k,_,d in rois])


def metrics():
    means = {}
    for scene in SCENES:
        path = SOURCE/scene/'R0/C0a/per_view_metrics.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        means[scene] = {k: statistics.mean(float(r[k]) for r in rows) for k in ('PSNR','SSIM','LPIPS')}
        means[scene]['views'] = len(rows)
    means['macro'] = {k: statistics.mean(means[s][k] for s in SCENES) for k in ('PSNR','SSIM','LPIPS')}
    manifest['metrics']['illustrated_checkpoint'] = means
    p = ROOT/'outputs/bnd_on_cica_ablation_extra3_20260914/four_scene_summary.json'
    raw = json.loads(p.read_text())
    recomputed = {k: statistics.mean(x['delta_on_minus_off'][k] for x in raw['scenes'].values())
                  for k in ('PSNR','SSIM','LPIPS','MSE')}
    manifest['metrics']['BND_ON_CICA_recomputed_delta'] = recomputed
    for k,v in recomputed.items():
        assert abs(v-raw['macro']['delta_on_minus_off'][k]) < 1e-12


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    overview()
    problem_sheet('JapaneseGradens-RedSea', 'MTN_1098', [
        ((65, 230, 465, 495), 'A', YELLOW, '远景表面：灰蓝偏色、纹理模糊'),
        ((320, 0, 750, 220), 'B', RED, '水域上方：大椭圆色块与边缘拖影'),
    ], '02_japanese_far_problems.png')
    problem_sheet('Panama', 'MTN_1529', [
        ((400, 130, 860, 435), 'A', YELLOW, '远景表面：仍有青蓝色覆盖'),
        ((1260, 0, 1740, 318), 'B', RED, '水域上方：拉伸高斯斑块与条纹'),
    ], '03_panama_far_problems.png')
    copies = [
        (ROOT/'outputs/fsb_far_field_diagnostic_20260916_v1/summary/transmission_by_depth.png', '04_transmission_by_depth.png'),
        (LATEST/'psnr_gain_vs_repeat.png', '05_latest_candidate_metrics.png'),
        (LATEST/'visualizations/four_scene_far_crops.png', '06_latest_candidate_far_crops.png'),
    ]
    for source,name in copies:
        shutil.copyfile(source, OUT/name)
        manifest['figures'].append(dict(file=name, source=str(source.relative_to(ROOT)), sha256=sha(source)))
    metrics()
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT), figures=len(manifest['figures']), metrics=manifest['metrics']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
