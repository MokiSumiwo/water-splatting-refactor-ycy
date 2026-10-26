"""Launch the four-scene seed-42 A(0.1)+B 15k campaign and endpoint export."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_runs/AB_20261026'
SCENES = ['Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama']
GPUS = ['6', '7', '8', '9']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f'campaign path already exists: {OUT}')
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError('less than 15 GiB free for campaign checkpoints and endpoint assets')
    support_root = ROOT / 'research_runs/surface_support_B_20261023/support'
    for scene in SCENES:
        if not (support_root / scene / 'support.npz').is_file():
            raise FileNotFoundError(support_root / scene / 'support.npz')
    checks = {}
    for scene in SCENES:
        smoke = ROOT/'research_runs/AB_validation_20261026/smoke_final'/scene/'AB'
        result = json.loads((smoke/'final_summary.json').read_text())
        assert result['completed_updates'] == 20 and result['smoke_reload_audit']['model_hash_match'] and result['smoke_reload_audit']['forward_finite']
        audit = ROOT/'research_runs/AB_validation_20261026/audit'/scene/'AB/audit.json'
        assert json.loads(audit.read_text())['passed']
        old = ROOT/'research_runs/geometry_gradient_A_20261021/formal'/scene/'B0'
        current = json.loads((smoke/'start_state_equivalence.json').read_text())
        previous = json.loads((old/'start_state_equivalence.json').read_text())
        assert current['model_hash'] == previous['model_hash']
        assert current['training_rng_manifest'] == previous['training_rng_manifest']
        assert (smoke/'camera_sequence.csv').read_bytes() == (old/'camera_sequence.csv').read_bytes()
        checks[scene] = dict(updates=20, reload_hash_match=True, forward_finite=True,
            initialization_rng_sequence_match_B0=True, audit_sha256=sha(audit))
    OUT.mkdir(parents=True)
    (OUT / 'formal').mkdir()
    (OUT / 'logs').mkdir()
    (OUT / 'code').mkdir()
    baseline_root = ROOT / 'research_runs/geometry_gradient_A_20261021/assets'
    for scene in SCENES:
        target = OUT / 'assets' / scene
        target.mkdir(parents=True)
        (target / 'B0').symlink_to(baseline_root / scene / 'B0', target_is_directory=True)
    code_paths = [
        'scripts/experiments/run_AB_worker.py',
        'scripts/experiments/run_AB_campaign.py',
        'scripts/experiments/run_direct_mdrr_cica_scene.py',
        'scripts/experiments/run_geometry_gradient_A_worker.py',
        'scripts/experiments/run_AC_pair_worker.py',
        'scripts/experiments/geometry_projection_backend.py',
        'scripts/experiments/export_geometry_gradient_A.py',
        'scripts/experiments/audit_AB_pair.py',
        'scripts/experiments/summarize_AB_campaign.py',
        'water_splatting/auxiliary_gradient_isolation.py',
        'water_splatting/geometry_gradient.py',
        'water_splatting/surface_support.py',
    ]
    hashes = {}
    for rel in code_paths:
        source = ROOT / rel
        hashes[rel] = sha(source)
        shutil.copy2(source, OUT / 'code' / source.name)
    manifest = dict(status='TRAINING', experiment='A(0.1)+B center-to-supported-plane',
        seed=42, updates_per_scene=15000, total_formal_updates=60000,
        renderer='corrected_v1', source='fresh raw COLMAP plus random medium initialization',
        schedule='A eta max .10; B every fourth update; both active 501-9999 with registered schedules',
        B_support_scope='frozen support assets from the registered B run; no O/F residual labels',
        code_sha256=hashes, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        smoke_checks=checks, actual_smoke_updates=180,
        smoke_budget_deviation='80 preliminary + 20 Panama telemetry repeat + 80 final; disposable states only', jobs=[])
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def train(item):
        scene, gpu = item
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CONDA_DEFAULT_ENV='water_splatting')
        dest = OUT / 'formal' / scene / 'AB'
        support = support_root / scene
        command = [sys.executable, '-m', 'scripts.experiments.run_AB_worker', '--scene', scene,
                   '--gpu', gpu, '--output', str(dest), '--support', str(support)]
        started = time.time()
        with (OUT / 'logs' / f'{scene}_train.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (OUT / 'logs' / f'{scene}_job.json').write_text(json.dumps(dict(pid=process.pid, command=command)) + '\n')
            code = process.wait()
        if code:
            return dict(scene=scene, exit_code=code, phase='training', seconds=time.time()-started)
        command = [sys.executable, '-m', 'scripts.experiments.export_geometry_gradient_A',
                   '--scene', scene, '--arm', 'AB', '--campaign', str(OUT)]
        with (OUT / 'logs' / f'{scene}_export.log').open('w') as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        return dict(scene=scene, exit_code=code, phase='endpoint_export', seconds=time.time()-started)

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest['jobs'] = list(pool.map(train, zip(SCENES, GPUS)))
    if all(row['exit_code'] == 0 for row in manifest['jobs']):
        from scripts.experiments.summarize_AB_campaign import summarize
        summarize(OUT)
        manifest['status'] = 'COMPLETE'
    else:
        manifest['status'] = 'FAILED'
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
