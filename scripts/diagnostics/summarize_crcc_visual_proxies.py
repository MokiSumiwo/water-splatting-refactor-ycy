#!/usr/bin/env python3
"""Describe native clear-display pixels; these are not clear reconstruction metrics."""

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment, write_json


def collect(root):
    rows=[]
    paths=list((root/'renders').glob('*/*/step_14999/*/A2/clear_raw_display_clamp01.png'))
    paths+=list((root/'crcc_only_four_scene_extension/renders').glob('*/*/step_14999/*/A2/clear_raw_display_clamp01.png'))
    for path in sorted(paths):
        with Image.open(path) as image:
            rgb=np.asarray(image,dtype=np.float32)/255.
        row=dict(variant=path.parents[4].name,scene=path.parents[3].name,view=path.parents[1].name,
                 width=rgb.shape[1],height=rgb.shape[0],path=str(path),
                 near_black_fraction=float((rgb.max(-1)<=3/255.).mean()),
                 dark_fraction=float((rgb.mean(-1)<.03).mean()),
                 cyan_like_fraction=float((((rgb[...,1]+rgb[...,2])*.5-rgb[...,0]>.1)
                                           &(rgb[...,1]>.1)&(rgb[...,2]>.1)).mean()))
        rows.append(row)
    assert rows
    keys=('near_black_fraction','dark_fraction','cyan_like_fraction')
    means=[]
    for scene,variant in sorted({(row['scene'],row['variant']) for row in rows}):
        selected=[r for r in rows if r['scene']==scene and r['variant']==variant]
        means.append(dict(scene=scene,variant=variant,**{k:statistics.mean(r[k] for r in selected) for k in keys}))
    with (root/'visual_proxies_per_view.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root/'visual_proxies.json',dict(rows=rows,scene_means=means,
        definitions=dict(near_black='max_RGB <= 3/255',dark='mean_RGB < .03',
                         cyan_like='(G+B)/2-R>.1 and G>.1 and B>.1'),
        limitations='Native 8-bit raw-clear display pixels; no resize/crop. These global color fractions '
                     'mix valid shadows, real colors and artifacts. They are descriptive proxies only, '
                     'not physical accuracy, hole segmentation, stripe detection or clear-ground-truth metrics.'))
    return means


if __name__=='__main__':
    check_environment()
    print(json.dumps(collect(DEFAULT_OUT)),flush=True)
