#!/usr/bin/env python3
"""Report paired metric differences and CRCC training scales without significance claims."""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment, read_json, write_json

METRICS=('PSNR','SSIM','LPIPS','MSE')


if __name__=='__main__':
    check_environment()
    root=DEFAULT_OUT
    summary=read_json(root/'summary.json')
    extra=read_json(root/'crcc_only_four_scene_report/summary.json')
    summary['final_metrics'] += [r for r in extra['final_metrics'] if r['variant']=='E2' and r['scene'] in ('Curasao','IUI3-RedSea')]
    summary['means'] += [r for r in extra['means'] if r['variant']=='E2' and r['scene']=='four_scene_mean']
    summary['states'] += [r for r in extra['states'] if r['variant']=='E2' and r['scene'] in ('Curasao','IUI3-RedSea')]
    records={(row['scene'],row['variant']):row for row in summary['final_metrics']+summary['means']}
    differences=[]
    for (scene,variant),row in records.items():
        for reference in ('E0','E1','E2'):
            if reference==variant or (scene,reference) not in records or variant=='E0':
                continue
            baseline=records[scene,reference]
            differences.append(dict(scene=scene,variant=variant,reference=reference,
                                    **{key:row[key]-baseline[key] for key in METRICS}))
    selected=summary['selected_variant']
    four=records.get(('four_scene_mean',selected))
    references={
        'historical_BND_OFF':(30.031764,.927403,.137838,.002015746),
        'historical_BND_ON':(29.899295,.927596,.140541,.002031043),
        'historical_TABA':(29.895461,.927672,.140587,.002050006),
        'historical_GRAC':(29.883284,.927647,.140731,.002083455),
    }
    gaps={name:{key:four[key]-value for key,value in zip(METRICS,values)} for name,values in references.items()} if four else {}
    scales=[]
    for run in summary['states']:
        directory=Path(run['path'])
        rows=list(csv.DictReader((directory/'training_summary_A2.csv').open()))
        logs=[json.loads(line) for line in (directory/'crcc_training.jsonl').read_text().splitlines()]
        log_lookup={row['step']:row for row in logs[1:]}
        ratios=[]
        for row in rows:
            step=int(row['absolute_step'])
            if step in log_lookup and float(row['L_RGB'])>0:
                ratios.append(log_lookup[step]['crcc_loss_weighted']/float(row['L_RGB']))
        training=read_json(directory/'training_summary.json')
        scales.append(dict(scene=run['scene'],variant=run['variant'],
            training_wall_seconds=training['training_wall_seconds'],final_gaussians=training['final_population'],
            median_logged_CRCC_to_existing_loss=statistics.median(ratios) if ratios else 0,
            max_logged_CRCC_to_existing_loss=max(ratios) if ratios else 0,
            max_logged_invalid_sample_fraction=max(row['crcc_invalid_transmission_fraction'] for row in logs),
            max_CRCC_dc_gradient=max(float(row.get('crcc_features_dc_grad_l2') or 0) for row in rows),
            max_CRCC_rest_gradient=max(float(row.get('crcc_features_rest_grad_l2') or 0) for row in rows),
            max_CICA_gradient_escape=max(float(row['cica_escape_grad_l2']) for row in rows)))
    result=dict(paired_differences=differences,selected_minus_historical_references=gaps,
                CRCC_only_minus_historical_references={name:{key:records['four_scene_mean','E2'][key]-value
                    for key,value in zip(METRICS,values)} for name,values in references.items()},
                training_scales=scales,notes='Differences are variant minus reference. Higher PSNR/SSIM and '
                'lower LPIPS/MSE are better. One seed; CUDA atomic nondeterminism and screening selection '
                'preclude statistical significance or robust superiority claims. Loss ratios use logged '
                'training views, not held-out tests; Charbonnier has a nonzero epsilon floor.')
    write_json(root/'comparisons.json',result)
    print(json.dumps(result),flush=True)
