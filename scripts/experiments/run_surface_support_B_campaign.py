"""Run the registered B stage and its endpoint exports, with durable status."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_runs/surface_support_B_20261023'
OLD=ROOT/'research_runs/geometry_gradient_A_20261021'
SCENES=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']

def summary():
    # Reuse exact A comparison algorithm on B aliases, then label all output B.
    from scripts.experiments import summarize_geometry_gradient_A as S
    for scene in SCENES:
        assets=OUT/'assets'/scene
        (assets/'A').symlink_to('B',target_is_directory=True)
        (assets/'B0').symlink_to(OLD/'assets'/scene/'B0',target_is_directory=True)
    (OUT/'regions').symlink_to(OLD/'regions',target_is_directory=True)
    S.OUT=OUT;S.main()
    p=OUT/'endpoint_report.json'
    text=p.read_text().replace('"A":','"B":').replace('delta_A_minus_B0','delta_B_minus_B0')
    p.write_text(text)
    p=OUT/'endpoint_report.md';text=p.read_text().replace('A 配对','B 配对').replace('A PSNR','B PSNR').replace('"A":','"B":').replace('delta_A_minus_B0','delta_B_minus_B0').replace('B0/A','B0/B').replace('归于 A','归于 B');p.write_text(text)
    for p in (OUT/'comparisons').rglob('*.png'):
        p.rename(p.with_name(p.name.replace('_A_','_B_')))
    for name in ['endpoint_per_view.csv','protection_region_metrics.csv']:
        p=OUT/name;p.write_text(p.read_text().replace(',A,',',B,'))
    for scene in SCENES:(OUT/'assets'/scene/'A').unlink()


def main():
    if (OUT/'formal').exists():raise FileExistsError('Formal campaign already exists')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Insufficient free disk for four endpoints')
    for scene in SCENES:
        d=json.loads((OUT/'smoke'/scene/'B/final_summary.json').read_text())
        assert d['completed_updates']==20 and d['smoke_reload_audit']['model_hash_match']
    (OUT/'formal').mkdir();(OUT/'code').mkdir()
    paths=['water_splatting/surface_support.py','scripts/experiments/prepare_surface_support_B.py',
           'scripts/experiments/run_surface_support_B_worker.py','scripts/experiments/run_surface_support_B_campaign.py',
           'scripts/experiments/run_direct_mdrr_cica_scene.py','scripts/experiments/geometry_projection_backend.py']
    hashes={}
    for f in paths:
        src=ROOT/f;hashes[f]=hashlib.sha256(src.read_bytes()).hexdigest();shutil.copy2(src,OUT/'code'/src.name)
    manifest=dict(status='TRAINING',seed=42,steps_per_scene=15000,total_formal_updates=60000,smoke_updates=80,
                  arm='B center-only',A_enabled=False,code_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  support_scope='Sparse COLMAP patches; low coverage; no O/F labels; no residual gate claim',jobs=[])
    def save(): (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    def lane(item):
        scene,gpu=item;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CONDA_DEFAULT_ENV='water_splatting')
        cmd=[sys.executable,'-m','scripts.experiments.run_surface_support_B_worker','--scene',scene,'--gpu',str(gpu),
             '--output',str(OUT/'formal'/scene/'B'),'--support',str(OUT/'support'/scene)]
        start=time.time()
        with (OUT/'logs'/f'{scene}_formal.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'logs'/f'{scene}_job.json').write_text(json.dumps(dict(pid=p.pid,command=cmd))+'\n')
            code=p.wait()
        if code:return dict(scene=scene,exit_code=code,phase='training')
        with (OUT/'logs'/f'{scene}_export.log').open('w') as log:
            code=subprocess.run([sys.executable,'-m','scripts.experiments.export_geometry_gradient_A','--scene',scene,
                                 '--arm','B','--campaign',str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        return dict(scene=scene,exit_code=code,phase='export',wall_seconds=time.time()-start)
    with ThreadPoolExecutor(max_workers=4) as pool:manifest['jobs']=list(pool.map(lane,zip(SCENES,[6,7,8,9])))
    manifest['status']='EXPORTED' if all(j['exit_code']==0 for j in manifest['jobs']) else 'FAILED';save()
    if manifest['status']=='EXPORTED':
        try:summary();manifest['status']='COMPLETE'
        except BaseException as e:manifest.update(status='POSTPROCESS_FAILED',error=repr(e));raise
        finally:save()

if __name__=='__main__':main()
