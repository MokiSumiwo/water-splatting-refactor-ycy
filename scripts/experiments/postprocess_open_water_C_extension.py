"""Run read-only B0/C1/C2 diagnostics after successful training completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args()
    out = a.campaign.resolve()
    def lane(item):
        scene, gpu = item
        jobs = []
        status_path = out/(scene+'_diagnostic_status.json')
        for arm in ['B0', 'C1', 'C2']:
            dest = out/'diagnostics'/scene/arm
            if (dest/'complete.json').exists():
                jobs.append(dict(scene=scene,arm=arm,status='COMPLETE',reused=True))
                continue
            if arm != 'B0':
                while True:
                    manifest = json.loads((out/'run_manifest.json').read_text())
                    job = next(r for r in manifest['jobs'] if r['scene']==scene and r['arm']==arm)
                    if job['status'] == 'FAILED':
                        raise RuntimeError(f'Training failed: {scene}/{arm}')
                    if job['status'] == 'COMPLETE':
                        break
                    status_path.write_text(json.dumps(dict(status='WAITING_FOR_TRAINING',arm=arm,jobs=jobs),indent=2)+'\n')
                    time.sleep(15)
            run = ROOT/'research_runs/geometry_gradient_A_20261021/formal'/scene/'B0' if arm=='B0' else out/'formal'/scene/arm
            cmd = [sys.executable,'-m','scripts.experiments.evaluate_open_water_C_endpoint',
                '--scene',scene,'--arm',arm,'--run-dir',str(run),
                '--mask-dir',str(out/'assets'/scene/'masks'),'--output',str(dest)]
            row = dict(scene=scene,arm=arm,command=cmd,status='RUNNING',started_unix=time.time())
            jobs.append(row)
            status_path.write_text(json.dumps(dict(status='RUNNING',jobs=jobs),indent=2)+'\n')
            with (out/'logs'/f'{scene}_{arm}_diagnostics.log').open('w') as log:
                result = subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode,finished_unix=time.time(),status='COMPLETE' if result.returncode==0 else 'FAILED')
            status_path.write_text(json.dumps(dict(status=row['status'],jobs=jobs),indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'Diagnostics failed: {scene}/{arm}; see log')
        return jobs
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = list(pool.map(lane,zip(['Curasao','JapaneseGradens-RedSea','Panama'],['7','8','9'])))
    (out/'postprocess_status.json').write_text(json.dumps(dict(status='COMPLETE',jobs=jobs),indent=2)+'\n')


if __name__ == '__main__':
    main()
