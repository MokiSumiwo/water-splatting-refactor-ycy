#!/usr/bin/env python3
"""Run the registered CRCC screen, then the visually reviewed fixed extension."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments import run_taba_bnd_on as original
from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.summarize_crcc_bnd_on import collect
from scripts.diagnostics.audit_crcc_artifacts import audit

FILES = [
    'water_splatting/crcc.py','water_splatting/gcrc.py','water_splatting/grac.py','water_splatting/taba.py',
    'water_splatting/water_splatting.py','water_splatting/fields/gaussian_appearance.py',
    'scripts/experiments/run_direct_mdrr_cica_scene.py','scripts/experiments/run_m1_raoc_causal_scene.py',
    'scripts/experiments/run_crcc_bnd_on.py','scripts/experiments/run_taba_bnd_on.py',
    'scripts/experiments/run_crcc_registered_experiment.py','scripts/experiments/summarize_crcc_bnd_on.py',
    'scripts/experiments/summarize_taba_bnd_on.py','scripts/diagnostics/audit_crcc_training.py',
    'scripts/diagnostics/audit_crcc_artifacts.py','tests/test_crcc.py',
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--stage',choices=('screen','extend'),required=True)
    args=parser.parse_args()
    root=args.output_root.resolve()
    original.check_environment()
    evidence=original.read_json(root/'engineering/test_results.json')
    assert evidence['pytest_passed'] and evidence['py_compile_passed'] and evidence['diff_check_passed']
    for scene in original.SCENES:
        assert original.read_json(root/'engineering/verified_v3'/scene/'audit.json')['passed']
    assert original.read_json(root/'engineering/cuda_nonregression.json')['no_regression']
    assert original.read_json(root/'source_asset_audit.json')['passed']
    sources=original.read_json(ROOT/'outputs/taba_dcu_bnd_on_20260910/four_scene_source_audit.json')
    for row in sources['rows']:
        assert original.sha256(Path(row['checkpoint']))==row['sha256']
    snapshot=root/'code_snapshot'
    if args.stage=='screen':
        sources['reverified_for']='CRCC BND ON'
        original.write_json(root/'four_scene_source_audit.json',sources)
        snapshot.mkdir(exist_ok=False)
        manifest=[]
        for name in FILES:
            target=snapshot/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
            manifest.append(dict(path=name,sha256=original.sha256(target)))
        original.write_json(snapshot/'manifest.json',dict(files=manifest,
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()))
        jobs=[(scene,arm,gpu) for arm in PRESETS for scene,gpu in (('JapaneseGradens-RedSea','6'),('Panama','8'))]
    else:
        for row in original.read_json(snapshot/'manifest.json')['files']:
            assert original.sha256(ROOT/row['path'])==row['sha256'], row['path']
        decision=original.read_json(root/'four_scene_selection.json')
        assert decision['visual_review_complete'] and decision['selected_variant'] in ('E1','E2','E3')
        selected=decision['selected_variant']
        jobs=[('Curasao','E0','6'),('Curasao',selected,'7'),('IUI3-RedSea','E0','8'),('IUI3-RedSea',selected,'9')]
    original.launch(SimpleNamespace(output_root=root,max_step=14999),jobs=jobs,
                    worker_script=ROOT/'scripts/experiments/run_crcc_bnd_on.py',audit_sources=False)
    summary=collect(root)
    assert all(row['integrity_passed'] for row in summary['states'])
    assert all(row['passed'] for row in summary['pairings'])
    if args.stage=='screen':
        assert len(summary['final_metrics'])==8
        for source,target in (('summary.json','two_scene_summary.json'),('report.md','two_scene_report.md')):
            shutil.copy2(root/source,root/target)
        print(json.dumps(dict(stage='screen_complete',provisional=summary['selected_variant'],
                              visual_review_required=True)),flush=True)
    else:
        result=audit(root)
        print(json.dumps(dict(complete_runs=len(summary['final_metrics']),audit_passed=result['passed'])),flush=True)


if __name__=='__main__':
    main()
