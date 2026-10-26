"""Read-only, fixed-region diagnosis before the A010+C2 combination.

Uses existing masks, not newly selected labels; all C-mask measurements are
in-sample. Source ablations are renderer diagnostics, never deletion labels.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.experiments import run_direct_mdrr_cica_scene as W

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_image(path, value):
    Image.fromarray((value.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype('uint8')).save(path)


def sources(model, camera, outputs, box, dest, tag):
    geometry = W.FULL._project_for_camera(model, copy.deepcopy(camera).to(model.device))
    colors = outputs['gaussian_view_rgb']
    n = len(colors)
    h, w = outputs['pred_image'].shape[:2]
    opacity = model.opacities.detach().sigmoid()
    if model.config.rasterize_mode == 'antialiased':
        opacity = opacity * geometry['compensation'][:, None]
    zero = torch.zeros(h, w, device=model.device)
    rgbzero = torch.zeros(h, w, 3, device=model.device)
    x0, y0, x1, y1 = box
    mask = rgbzero.clone()
    mask[y0:y1, x0:x1] = 1
    stats, _, _, _ = W.FULL._responsibility_forward(
        W._responsibility_extension(), geometry, colors, opacity, rgbzero,
        torch.arange(n, device=model.device), mask, zero, zero)
    mass = stats[:, 1] * colors.clamp_min(0).mean(-1)
    total = mass.sum()
    order = mass.argsort(descending=True)
    count = int(torch.searchsorted(mass[order].cumsum(0), .8 * total)) + 1 if total > 0 else 0
    selected = order[:count]
    saved = model.opacities.detach().clone()
    try:
        model.opacities[selected] = -100
        behind = model.get_outputs_for_camera(copy.deepcopy(camera).to(model.device))
        behind_occ = float(behind['accumulation'][y0:y1, x0:x1].mean())
        save_image(dest / (tag + '_remaining.png'), behind['clear_object_fullsh_raw'][y0:y1, x0:x1])
        model.opacities.copy_(saved)
        keep = torch.zeros(n, dtype=torch.bool, device=model.device)
        keep[selected] = True
        model.opacities[~keep] = -100
        top = model.get_outputs_for_camera(copy.deepcopy(camera).to(model.device))
        save_image(dest / (tag + '_dominant.png'), top['clear_object_fullsh_raw'][y0:y1, x0:x1])
    finally:
        model.opacities.copy_(saved)
    np.savez_compressed(dest / (tag + '_sources.npz'), ids=selected.cpu().numpy(),
                        weight=stats[selected, 1].cpu().numpy(), depth=geometry['depths'][selected].cpu().numpy())
    return dict(selected=count, population=n, contribution_mass=float(total),
                coverage=float(mass[selected].sum() / total) if total > 0 else None,
                remaining_occupancy=behind_occ, not_a_deletion_label=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--step', type=int, default=15000)
    p.add_argument('--sources', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    W._route_renderer_backend('corrected_v1')
    from scripts.experiments.geometry_projection_backend import install
    install()
    W.ACTIVE_SOURCE_CONFIG = a.run_dir / 'source_config.yml'
    W.ACTIVE_PARAMETERIZATION = 'bounded_sh3'
    W.ACTIVE_TRAINING_SEED = 42
    b = W._new_branch(a.scene)
    try:
        checkpoint = a.run_dir / 'checkpoints' / ('step-%09d.ckpt' % a.step)
        payload = W._load_snapshot(b, a.scene, checkpoint)
        m = b.pipeline.model
        m.eval()
        before_hash = W._model_state_hash(m)
        assert before_hash == W._hash_object(payload['model']) and payload['absolute_step'] == a.step
        del payload
        manifest = json.loads((a.mask_dir / 'manifest.json').read_text())
        masks = {r['camera']: r for r in manifest['rows']}
        nonempty = sorted(n for n, r in masks.items() if r['pixels'])
        source_names = {nonempty[0], nonempty[len(nonempty)//2], nonempty[-1]}
        regions = json.loads((ROOT / 'research_runs/geometry_gradient_A_20261021/regions' / a.scene / 'regions.json').read_text())
        regions = {r['camera']: r for r in regions['assets']}
        rows, contribution_rows = [], []
        for _, name, camera, batch in W._train_records(b):
            name = str(name)
            if name not in nonempty and name not in regions:
                continue
            with torch.no_grad():
                o = m.get_outputs_for_camera(copy.deepcopy(camera).to(m.device))
                gt = W._gt(m, W._batch_to_device(batch, m.device), o['background']).float().clamp(0, 1)
                pred = o['pred_image'].float().clamp(0, 1)
                clear = o['clear_object_fullsh_raw'].float().clamp(0, 1)
                assert torch.isfinite(pred).all() and torch.isfinite(clear).all()
                boxes = []
                if name in nonempty:
                    row = masks[name]
                    box = row.get('box_xyxy_exclusive', row.get('box'))
                    if box is None:
                        mask = np.asarray(Image.open(a.mask_dir / (name + '.png'))) > 0
                        ys, xs = np.where(mask)
                        box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                    boxes.append(('O', 0, box))
                if name in regions:
                    for label, key in [('U', 'residual_review'), ('P', 'structure_protection')]:
                        boxes.extend((label, i, r['box']) for i, r in enumerate(regions[name][key]))
                    save_image(a.output / (name + '_clear.png'), clear)
                for label, idx, box in boxes:
                    x0, y0, x1, y1 = box
                    c = clear[y0:y1, x0:x1]
                    occ = o['accumulation'][y0:y1, x0:x1].float()
                    edge = .5 * ((c[1:]-c[:-1]).abs().mean()+(c[:, 1:]-c[:, :-1]).abs().mean())
                    tag = '%s_%s%d' % (name, label, idx)
                    row = dict(scene=a.scene, arm=a.arm, step=a.step, camera=name, label=label, region=idx,
                               pixels=(x1-x0)*(y1-y0), occupancy_mean=float(occ.mean()),
                               occupancy_gt_0p01=float((occ > .01).float().mean()),
                               clear_mean=float(c.mean()), clear_edge_proxy=float(edge),
                               underwater_MSE=float((pred[y0:y1, x0:x1]-gt[y0:y1, x0:x1]).square().mean()),
                               medium_only_MSE=float((o['b_inf'][y0:y1, x0:x1]-gt[y0:y1, x0:x1]).square().mean()))
                    rows.append(row)
                    save_image(a.output / (tag + '_clear.png'), c)
                    save_image(a.output / (tag + '_gt.png'), gt[y0:y1, x0:x1])
                    if a.sources and ((label == 'O' and name in source_names) or label == 'U'):
                        contribution_rows.append(dict(camera=name, label=label, region=idx, box=box,
                            **sources(m, camera, o, box, a.output, tag)))
                del o, gt, pred, clear
        assert W._model_state_hash(m) == before_hash
        with (a.output / 'regions.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = dict(complete=True, scene=a.scene, arm=a.arm, step=a.step,
                      checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      model_unchanged=True, mask_manifest_sha256=digest(a.mask_dir / 'manifest.json'),
                      source_cameras=sorted(source_names), contributions=contribution_rows,
                      scope='Frozen training masks; O in-sample for C. U remains unknown. No updates or test-based selection.')
        (a.output / 'complete.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'contributions'}), flush=True)
    finally:
        W._release(b)


if __name__ == '__main__':
    main()
