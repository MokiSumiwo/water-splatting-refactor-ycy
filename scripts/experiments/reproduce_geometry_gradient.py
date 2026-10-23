"""Fresh-output launcher for the frozen B0/A025/A010 protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCENES = ['Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['B0', 'A025', 'A010'], default='A010')
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    p.add_argument('--gpu', default='0', help='Run scenes sequentially on this GPU')
    p.add_argument('--export', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    module = 'scripts.experiments.run_geometry_gradient_A010_worker' if a.arm == 'A010' else 'scripts.experiments.run_geometry_gradient_A_worker'
    arm = 'B0' if a.arm == 'B0' else 'A'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CONDA_DEFAULT_ENV='water_splatting')
    manifest = {'arm': a.arm, 'seed': 42, 'steps': 15000, 'scenes': a.scenes, 'status': 'RUNNING'}
    def save():
        (output/'reproduction_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        for scene in a.scenes:
            subprocess.run([sys.executable, '-m', module, '--scene', scene, '--arm', arm, '--gpu', a.gpu,
                            '--output', str(output/'formal'/scene/arm)], cwd=ROOT, env=env, check=True)
            if a.export:
                subprocess.run([sys.executable, '-m', 'scripts.experiments.export_geometry_gradient_A',
                                '--scene', scene, '--arm', arm, '--campaign', str(output)], cwd=ROOT, env=env, check=True)
        manifest['status'] = 'COMPLETE'
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save()

if __name__ == '__main__':
    main()
