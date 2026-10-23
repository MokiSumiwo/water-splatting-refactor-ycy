#!/usr/bin/env python3
"""Collect CRCC metrics with registered multi-metric and visual selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, read_json, write_json
from scripts.experiments.summarize_taba_bnd_on import STATS, summarize
from scripts.experiments.summarize_taba_dcu_bnd_on import four_scene_sheet
from water_splatting.crcc import CRCC_STATS

RULE = ("Equal mean rank of two-scene mean PSNR/SSIM (higher better) and LPIPS/MSE "
        "(lower better), among stable E1/E2/E3; ties use PSNR. Inspect all screen sheets "
        "and veto serious visual regressions before the fixed four-scene extension. "
        "Exploratory selection, one seed, no clear-image ground truth.")


def rank_candidates(rows):
    ranked = []
    for row in rows:
        ranks = {key: 1 + sum(sign*other[key] > sign*row[key] for other in rows)
                 for key, sign in (("PSNR", 1), ("SSIM", 1), ("LPIPS", -1), ("MSE", -1))}
        ranked.append(dict(variant=row['variant'], ranks=ranks, mean_rank=sum(ranks.values())/4,
                           PSNR=row['PSNR']))
    return sorted(ranked, key=lambda row: (row['mean_rank'], -row['PSNR'], row['variant']))


def collect(root):
    rankings = []
    def select(rows):
        rankings.extend(rank_candidates(rows))
        decision = root/'four_scene_selection.json'
        if decision.exists():
            chosen = read_json(decision)
            assert chosen['visual_review_complete']
            assert chosen['selected_variant'] in {r['variant'] for r in rankings}
            return chosen['selected_variant']
        return rankings[0]['variant']
    result = summarize(root, variants=tuple(PRESETS), candidates=('E1','E2','E3'),
                       stats=STATS+CRCC_STATS, experiment='CRCC, BND ON',
                       selector=select, selection_rule=RULE)
    result['multi_metric_ranking'] = rankings
    result['visual_selection_confirmed'] = (root/'four_scene_selection.json').exists()
    lines = ['', '## CRCC diagnostics', '',
        'Current intrinsic clear RGB uses the existing safe sigmoid of appearance logits.',
        'E3 uses post-TABA appearance. EMA targets come from previous completed training iterations.',
        'Transmission is sampled at rounded Gaussian centers from the existing final renderer.',
        'Visibility is radii > 0; no stable per-Gaussian contribution is exposed, so alpha weight is 1.',
        'The medium floor is applied once. Invalid transmission has zero effective sample weight.',
        'Two prior valid observations are required; counts include repeated cameras and inherited lineage.',
        'There is no luma penalty, no trainable CRCC parameter, and no geometry/medium gradient.',
        'E0/E1 CRCC statistics are neutral placeholders. Quantiles are means of per-view quantiles.',
        'The first training JSONL step5000 entry is a cold-source routing audit; formal rows start3500.',
        'Clear display uses J/(1+J); the separate raw display uses clamp01 only for visualization.',
        '', 'Metric ranking: '+json.dumps(rankings),
        'Visual review confirmed: '+str(result['visual_selection_confirmed'])]
    for keys in (CRCC_STATS[:12], CRCC_STATS[12:]):
        lines += ['', '| Scene | Arm | '+' | '.join(keys)+' |', '|---|---|'+'---:|'*len(keys)]
        for row in result['diagnostics']:
            if row['step']==14999:
                lines.append('| '+row['scene']+' | '+row['variant']+' | '+' | '.join(f'{row[k]:.9g}' for k in keys)+' |')
    if result['four_scene_complete']:
        for kind in ('underwater','clear','clear_raw_display_clamp01'):
            result['contact_sheets'].append(four_scene_sheet(root,result['selected_variant'],kind))
    write_json(root/'summary.json',result)
    report = root/'report.md'
    report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    check_environment()
    result=collect(args.output_root.resolve())
    print(json.dumps(dict(runs=len(result['final_metrics']), selected=result['selected_variant'],
                          four_scene_complete=result['four_scene_complete'])),flush=True)
