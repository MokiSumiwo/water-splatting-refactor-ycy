"""Read-only reload, float metric recomputation, and pairing checks for A+B."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scripts.experiments import run_direct_mdrr_cica_scene as W
from scripts.experiments.geometry_projection_backend import install
from scripts.experiments.run_AB_campaign import sha

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['PSNR', 'SSIM', 'LPIPS', 'MSE']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    out = args.campaign.resolve(); scene = args.scene
    formal = out/'formal'/scene/'AB'
    dest = out/'verification'/scene; dest.mkdir(parents=True, exist_ok=False)
    W._route_renderer_backend('corrected_v1'); install()
    W.ACTIVE_SOURCE_CONFIG = formal/'source_config.yml'
    W.ACTIVE_PARAMETERIZATION = 'bounded_sh3'; W.ACTIVE_TRAINING_SEED = 42
    checkpoint = formal/'checkpoints/step-000015000.ckpt'
    reload = W._smoke_reload_audit(scene, 'AB', dest, checkpoint)
    roots = {
        'B0': ROOT/'research_runs/geometry_gradient_A_20261021/assets'/scene/'B0',
        'A010': ROOT/'research_runs/geometry_gradient_A010_20261022_v2/assets'/scene/'A',
        'B': ROOT/'research_runs/surface_support_B_20261023/assets'/scene/'B',
        'AB': out/'assets'/scene/'AB',
    }
    test_names = None; gt_reference = {}; rows = []; comparison = []
    for arm, root in roots.items():
        original = {r['view_id']: r for r in csv.DictReader((root/'eval_metrics.csv').open())}
        if test_names is None: test_names = set(original)
        assert set(original) == test_names
        assert {p.stem for p in (root/'eval_float').glob('*.npz')} == test_names
        for name in sorted(test_names):
            with np.load(root/'eval_float'/f'{name}.npz') as data:
                gt = data['gt'].copy(); pred = data['underwater'].copy()
            assert gt.shape == pred.shape and np.isfinite(pred).all() and np.isfinite(gt).all()
            if arm == 'B0': gt_reference[name] = gt
            assert np.array_equal(gt, gt_reference[name])
            values = W._taba_rgb_metrics(torch.from_numpy(pred).cuda(), torch.from_numpy(gt).cuda())
            diff = {k: abs(values[k]-float(original[name][k])) for k in KEYS}
            # Historic LPIPS exports have ~6e-6 implementation/numerical drift;
            # use one exact worker path for all arms in this new report.
            tolerance = {'PSNR':1e-6, 'SSIM':1e-7, 'MSE':1e-10, 'LPIPS':1e-6 if arm=='AB' else 2e-5}
            assert all(diff[k] <= tolerance[k] for k in KEYS), (arm, name, diff)
            comparison.append(dict(arm=arm, view_id=name, absolute_difference=diff))
            rows.append(dict(scene=scene, arm=arm, view_id=name, **values))
    summary = json.loads((formal/'final_summary.json').read_text())
    assert summary['completed_updates']==15000 and summary['final_step']==15000
    training_endpoint = {r['view_id']: r for r in csv.DictReader((formal/'per_view_metrics.csv').open())
                         if int(r['absolute_step']) == 15000 and r['split'] == 'eval'}
    assert set(training_endpoint) == test_names
    endpoint_comparison = []
    for row in rows:
        if row['arm'] != 'AB':
            continue
        difference = {k: abs(row[k]-float(training_endpoint[row['view_id']][k])) for k in KEYS}
        assert all(difference[k] <= {'PSNR':1e-6, 'SSIM':1e-7, 'MSE':1e-10, 'LPIPS':1e-6}[k] for k in KEYS)
        endpoint_comparison.append(dict(view_id=row['view_id'], absolute_difference=difference))
    old = ROOT/'research_runs/geometry_gradient_A_20261021/formal'/scene/'B0'
    init = json.loads((formal/'start_state_equivalence.json').read_text())
    native = json.loads((old/'start_state_equivalence.json').read_text())
    assert init['model_hash']==native['model_hash'] and init['training_rng_manifest']==native['training_rng_manifest']
    assert (formal/'camera_sequence.csv').read_bytes()==(old/'camera_sequence.csv').read_bytes()
    result = dict(scene=scene, complete=True, optimizer_updates=0, raw_initialization_rng_sequence_match=True,
        endpoint_reload=reload, checkpoint_sha256=sha(checkpoint),
        per_scene={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in KEYS} for arm in roots},
        recomputed_views=rows, metric_comparison=comparison, training_endpoint_comparison=endpoint_comparison,
        training_seconds=summary['training_wall_seconds'], final_population=summary['final_population'])
    (dest/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(scene=scene, complete=True, per_scene=result['per_scene'])), flush=True)


if __name__ == '__main__':
    main()
