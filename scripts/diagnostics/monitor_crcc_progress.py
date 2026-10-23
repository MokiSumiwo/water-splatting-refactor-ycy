#!/usr/bin/env python3
"""Record bounded runtime observations without touching training state."""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment


if __name__=='__main__':
    check_environment()
    runs=[]
    paths=list(DEFAULT_OUT.glob('*/*/crcc_training.jsonl'))
    paths+=list((DEFAULT_OUT/'crcc_only_four_scene_extension').glob('*/*/crcc_training.jsonl'))
    for path in sorted(paths):
        lines=path.read_text().splitlines()
        if not lines:
            continue
        row=json.loads(lines[-1])
        item=dict(scene=path.parent.parent.name,arm=path.parent.name,
                  step=row['step'],cold_audit_only=len(lines)==1,
                  complete=(path.parent/'crcc_result.json').exists(),
                  checkpoint_count=len(list((path.parent/'checkpoints').glob('*.ckpt'))))
        for key in ('crcc_loss_raw','crcc_loss_weighted','crcc_effective_count',
                    'crcc_initialized_ratio','crcc_state_mismatch','P_J_gt_1','P_intrinsic_rgb_gt_1'):
            item[key]=row[key]
        runs.append(item)
    raw=subprocess.check_output(['nvidia-smi','-i','6,7,8,9',
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[dict(gpu=int(r[0]),memory_mib=int(r[1]),utilization=int(r[2])) for r in csv.reader(raw.splitlines())]
    result=dict(unix=time.time(),runs=runs,gpus=gpus)
    with (DEFAULT_OUT/'runtime_monitor.jsonl').open('a') as handle:
        handle.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False),flush=True)
