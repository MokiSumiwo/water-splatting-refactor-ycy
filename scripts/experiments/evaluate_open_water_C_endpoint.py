"""Read-only endpoint verification and frozen training-region C diagnostics."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from scripts.experiments import run_direct_mdrr_cica_scene as W

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    if rows:
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--arm", choices=["B0", "C1", "C2"], required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    W._route_renderer_backend("corrected_v1")
    from scripts.experiments.geometry_projection_backend import install
    install()
    W.ACTIVE_SOURCE_CONFIG = a.run_dir/"source_config.yml"
    W.ACTIVE_PARAMETERIZATION = "bounded_sh3"
    W.ACTIVE_TRAINING_SEED = 42
    b = W._new_branch(a.scene)
    video = None
    try:
        checkpoint = a.run_dir/"checkpoints/step-000015000.ckpt"
        payload = W._load_snapshot(b, a.scene, checkpoint)
        m = b.pipeline.model
        m.eval()
        checkpoint_model_hash = W._hash_object(payload["model"])
        reloaded_hash = W._model_state_hash(m)
        assert checkpoint_model_hash == reloaded_hash and payload["absolute_step"] == 15000
        del payload
        manifest = json.loads((a.mask_dir/"manifest.json").read_text())
        masks = {r["camera"]: np.asarray(Image.open(a.mask_dir/(r["camera"]+".png"))).copy()>0 for r in manifest["rows"]}
        region_path = ROOT/"research_runs/geometry_gradient_A_20261021/regions"/a.scene/"regions.json"
        regions = {v["camera"]:v for v in json.loads(region_path.read_text())["assets"]}
        open_rows, protection, sequence = [], [], []
        started = time.perf_counter()
        for _, name, camera, batch in W._train_records(b):
            with torch.no_grad():
                outputs = m.get_outputs_for_camera(copy.deepcopy(camera).to(m.device))
                gt = W._gt(m, W._batch_to_device(batch, m.device), outputs["background"]).detach().float().clamp(0,1)
            pred = outputs["pred_image"].detach().float().clamp(0,1)
            clear = outputs["clear_object_fullsh_raw"].detach().float().clamp(0,1)
            assert torch.isfinite(pred).all() and torch.isfinite(clear).all()
            mask = torch.as_tensor(masks[str(name)], device=m.device)
            assert mask.shape == pred.shape[:2]
            if mask.any():
                occ = outputs["accumulation"].detach().float().squeeze(-1)[mask]
                b_inf = outputs["b_inf"].detach().float()
                assert torch.equal(outputs['b_inf'], outputs['medium_rgb'])
                open_rows.append(dict(scene=a.scene, arm=a.arm, camera=str(name), pixels=int(mask.sum()),
                    occupancy_mean=float(occ.mean()), occupancy_p95=float(torch.quantile(occ,.95)),
                    fraction_occupancy_gt_0p01=float((occ>.01).float().mean()),
                    clear_mean=float(clear[mask].mean()), underwater_MSE=float((pred[mask]-gt[mask]).square().mean()),
                    medium_only_MSE=float((b_inf[mask]-gt[mask]).square().mean())))
            if str(name) in regions:
                for i, region in enumerate(regions[str(name)]["structure_protection"]):
                    x0,y0,x1,y1 = region["box"]
                    c = clear[y0:y1,x0:x1]
                    edge = .5*((c[1:]-c[:-1]).abs().mean()+(c[:,1:]-c[:,:-1]).abs().mean())
                    protection.append(dict(scene=a.scene, arm=a.arm, camera=str(name), region=i,
                        underwater_MSE=float((pred[y0:y1,x0:x1]-gt[y0:y1,x0:x1]).square().mean()), clear_edge_proxy=float(edge)))
                np.savez_compressed(a.output/(str(name)+"_review.npz"),
                    underwater=pred.cpu().numpy(), gt=gt.cpu().numpy(), clear=clear.cpu().numpy())
                strip = Image.new('RGB',(pred.shape[1]*3,pred.shape[0]))
                for j,t in enumerate([gt,pred,clear]):
                    strip.paste(Image.fromarray((t.cpu().numpy()*255).round().astype('uint8')),(j*pred.shape[1],0))
                strip.save(a.output/(str(name)+"_review.jpg"),quality=95)
            if video is None:
                h,w = clear.shape[:2]
                video = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-threads','1','-f','rawvideo',
                    '-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','3','-i','-',
                    '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-threads','1',
                    '-crf','18','-pix_fmt','yuv420p',str(a.output/'clear_train_sequence.mp4')],stdin=subprocess.PIPE)
            video.stdin.write((clear.cpu().numpy()*255).round().astype('uint8').tobytes())
            sequence.append(str(name))
            del outputs, gt, pred, clear
        video.stdin.close()
        assert video.wait() == 0
        video = None
        (a.output/'camera_sequence.json').write_text(json.dumps(sequence,indent=2)+'\n')
        write_csv(a.output/'open_water_regions.csv',open_rows)
        write_csv(a.output/'protection_regions.csv',protection)
        official = {r['view_id']:r for r in csv.DictReader((a.run_dir/'per_view_metrics.csv').open()) if int(r['absolute_step'])==15000}
        tests, errors = [], []
        floats = a.output/'eval_float'
        floats.mkdir(exist_ok=True)
        for _, name, camera, batch in W._eval_records(b):
            with torch.no_grad():
                o = m.get_outputs_for_camera(copy.deepcopy(camera).to(m.device))
                gt = W._gt(m,W._batch_to_device(batch,m.device),o['background']).detach().float().clamp(0,1)
            pred = o['pred_image'].detach().float().clamp(0,1)
            values = W._taba_rgb_metrics(pred,gt)
            tests.append(dict(scene=a.scene,arm=a.arm,view_id=str(name),**values))
            delta = {k:abs(v-float(official[str(name)][k])) for k,v in values.items()}
            # LPIPS VGG has small run-to-run kernel variation on this stack;
            # report the measured discrepancy and reject only material drift.
            assert delta['PSNR']<1e-4 and delta['SSIM']<1e-6 and delta['LPIPS']<2e-5 and delta['MSE']<1e-8, delta
            errors.append(dict(view_id=str(name),**delta))
            np.savez_compressed(floats/(str(name)+'.npz'), underwater=pred.cpu().numpy(),gt=gt.cpu().numpy(),
                clear_raw=o['clear_object_fullsh_raw'].detach().float().cpu().numpy())
        assert set(official) == {r['view_id'] for r in tests}
        write_csv(a.output/'eval_metrics.csv',tests)
        with checkpoint.open('rb') as stream:
            digest=hashlib.sha256()
            for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
        audit=dict(complete=True,scene=a.scene,arm=a.arm,checkpoint_sha256=digest.hexdigest(),
            model_hash_match=True,model_hash=reloaded_hash,metric_max_abs_errors=errors,
            train_views=len(sequence),open_water_views=len(open_rows),eval_views=len(tests),
            diagnostic_seconds=time.perf_counter()-started,
            scope='Training O masks and existing foreground protection regions; no updates; O metrics are in-sample diagnostics, not test segmentation truth.')
        (a.output/'complete.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(audit),flush=True)
    finally:
        if video is not None:
            video.stdin.close()
            video.wait()
        W._release(b)


if __name__ == '__main__':
    main()
