#!/usr/bin/env python3
"""Complete fixed CRCC-only four-scene evidence when selection favors TABA."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments import run_taba_bnd_on as original
from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT


if __name__=='__main__':
    original.check_environment()
    decision=original.read_json(DEFAULT_OUT/'four_scene_selection.json')
    assert decision['selected_variant']=='E1' and decision['visual_review_complete']
    supplement=decision['supplementary_CRCC_four_scene_validation']
    assert supplement['variant']=='E2' and supplement['registered_before_any_extension_training']
    for row in original.read_json(DEFAULT_OUT/'code_snapshot/manifest.json')['files']:
        assert original.sha256(ROOT/row['path'])==row['sha256']
    output=ROOT/supplement['output_root']
    original.launch(SimpleNamespace(output_root=output,max_step=14999),
        jobs=[('Curasao','E2','6'),('IUI3-RedSea','E2','8')],
        worker_script=ROOT/'scripts/experiments/run_crcc_bnd_on.py',audit_sources=False)
    print(json.dumps(dict(supplementary_runs_complete=True,root=str(output))),flush=True)
