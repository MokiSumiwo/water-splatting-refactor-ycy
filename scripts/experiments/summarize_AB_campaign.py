"""Summarize A+B underwater endpoint metrics against the frozen paired B0."""
import csv
import json
from pathlib import Path

import numpy as np

SCENES = ['Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama']
KEYS = ['PSNR', 'SSIM', 'LPIPS', 'MSE']


def read_metrics(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f'no endpoint metrics in {path}')
    return rows


def summarize(out: Path):
    out = Path(out)
    rows = []
    means = {}
    view_rows = []
    for scene in SCENES:
        means[scene] = {}
        for arm in ['B0', 'AB']:
            metric_rows = read_metrics(out/'assets'/scene/arm/'eval_metrics.csv')
            means[scene][arm] = {key: float(np.mean([float(row[key]) for row in metric_rows])) for key in KEYS}
            rows.extend(dict(row, scene=scene, arm=arm) for row in metric_rows)
        a_rows = {row['view_id']: row for row in read_metrics(out/'assets'/scene/'AB'/'eval_metrics.csv')}
        b_rows = {row['view_id']: row for row in read_metrics(out/'assets'/scene/'B0'/'eval_metrics.csv')}
        if set(a_rows) != set(b_rows):
            raise RuntimeError(f'test view mismatch in {scene}: {set(a_rows) ^ set(b_rows)}')
        for name in sorted(a_rows):
            a_float = np.load(out/'assets'/scene/'AB'/'eval_float'/f'{name}.npz')
            b_float = np.load(out/'assets'/scene/'B0'/'eval_float'/f'{name}.npz')
            if not np.array_equal(a_float['gt'], b_float['gt']):
                raise RuntimeError(f'test target mismatch for {scene}/{name}')
            a_float.close(); b_float.close()
            view_rows.append(dict(scene=scene, view_id=name,
                **{f'B0_{key}': float(b_rows[name][key]) for key in KEYS},
                **{f'AB_{key}': float(a_rows[name][key]) for key in KEYS},
                **{f'delta_{key}': float(a_rows[name][key])-float(b_rows[name][key]) for key in KEYS}))
    macro = {arm: {key: float(np.mean([means[scene][arm][key] for scene in SCENES])) for key in KEYS}
             for arm in ['B0', 'AB']}
    delta_macro = {key: macro['AB'][key]-macro['B0'][key] for key in KEYS}
    delta_scene = {scene: {key: means[scene]['AB'][key]-means[scene]['B0'][key] for key in KEYS} for scene in SCENES}
    guard = (delta_macro['PSNR'] >= -.05
        and all(delta_scene[scene]['PSNR'] >= -.10 for scene in SCENES)
        and delta_macro['SSIM'] >= -.001 and delta_macro['LPIPS'] <= .002)
    report = dict(experiment='A(0.1)+B center-support combination', scenes=SCENES,
        per_scene=means, delta_per_scene=delta_scene, macro_scene_equal=macro,
        delta_macro=delta_macro, underwater_guard_pass=guard,
        updates_per_scene=15000, seed=42, metric_source='native-resolution float endpoint; metrics.py-compatible worker path',
        residual_gate='N/A: fixed residual regions remain U; no independent O/F truth was used for A+B',
        interpretation='single-seed combined-arm result; exploratory, not an independent estimate of A or B effects')
    (out/'endpoint_report.json').write_text(json.dumps(report, indent=2)+'\n')
    fields = ['scene', 'view_id'] + [f'{arm}_{key}' for arm in ['B0', 'AB'] for key in KEYS] + [f'delta_{key}' for key in KEYS]
    with (out/'endpoint_per_view.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(view_rows)
    lines = ['# A(0.1)+B 四场景带水终点评估', '',
        'seed 42；每场景从原始初始化训练 15,000 步。场景内对测试视角取均值，再对四场景等权平均。', '',
        '| 场景 | PSNR | SSIM | LPIPS | MSE | ΔPSNR vs B0 |',
        '|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        val = means[scene]['AB']; d = delta_scene[scene]
        lines.append(f"| {scene} | {val['PSNR']:.6f} | {val['SSIM']:.6f} | {val['LPIPS']:.6f} | {val['MSE']:.9f} | {d['PSNR']:+.6f} |")
    val = macro['AB']; d = delta_macro
    lines += ['', f"四场景等权宏平均：PSNR **{val['PSNR']:.6f} dB**，SSIM **{val['SSIM']:.6f}**，LPIPS **{val['LPIPS']:.6f}**，MSE **{val['MSE']:.9f}**。",
        f"相对配对 B0 宏平均变化：PSNR {d['PSNR']:+.6f} dB，SSIM {d['SSIM']:+.6f}，LPIPS {d['LPIPS']:+.6f}，MSE {d['MSE']:+.9f}。",
        '', f"带水质量保护门槛：{'通过' if guard else '未通过'}。单 seed，组合探索结果；残留 O/F 门槛仍为 N/A。", '']
    (out/'endpoint_report.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
