"""Read-only paired rendering of the frozen structure-protection crops."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.experiments import run_direct_mdrr_cica_scene as W

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    regions = json.loads((ROOT / 'research_runs/geometry_gradient_A_20261021/regions' /
                          args.scene / 'regions.json').read_text())['assets']
    by_camera = {row['camera']: row['structure_protection'] for row in regions}
    W._route_renderer_backend('corrected_v1')
    from scripts.experiments.geometry_projection_backend import install
    install()
    results = []
    for name, run in [('B0', args.baseline), ('C2_corrected', args.candidate)]:
        W.ACTIVE_SOURCE_CONFIG = run / 'source_config.yml'
        W.ACTIVE_PARAMETERIZATION = 'bounded_sh3'
        W.ACTIVE_TRAINING_SEED = 42
        branch = W._new_branch(args.scene)
        try:
            payload = W._load_snapshot(branch, args.scene, run/'checkpoints/step-000015000.ckpt')
            model = branch.pipeline.model
            model.eval()
            assert W._model_state_hash(model) == W._hash_object(payload['model'])
            del payload
            for _, camera_name, camera, batch in W._train_records(branch):
                camera_name = str(camera_name)
                if camera_name not in by_camera:
                    continue
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(copy.deepcopy(camera).to(model.device))
                    gt = W._gt(model, W._batch_to_device(batch, model.device),
                                outputs['background']).detach().float().clamp(0, 1)
                pred = outputs['pred_image'].detach().float().clamp(0, 1)
                for index, region in enumerate(by_camera[camera_name]):
                    x0, y0, x1, y1 = map(int, region['box'])
                    pred_crop = pred[y0:y1, x0:x1].cpu().numpy()
                    gt_crop = gt[y0:y1, x0:x1].cpu().numpy()
                    np.savez_compressed(args.output/f'{camera_name}_P{index}_{name}.npz',
                                        underwater=pred_crop, gt=gt_crop)
                    preview = np.concatenate([gt_crop, pred_crop], axis=1)
                    Image.fromarray((preview*255).round().astype(np.uint8)).save(
                        args.output/f'{camera_name}_P{index}_{name}.png')
                    results.append(dict(scene=args.scene, camera=camera_name,
                                        region=index, arm=name,
                                        box=[x0, y0, x1, y1]))
                del outputs, gt, pred
        finally:
            W._release(branch)
    (args.output/'manifest.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(scene=args.scene, crops=len(results))), flush=True)


if __name__ == '__main__':
    main()
