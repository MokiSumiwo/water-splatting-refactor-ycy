#!/usr/bin/env python3
"""Prepare and audit the native responsibility-transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENE_MAP = {
    "Panama": "undistorted_Panama",
    "IUI3-RedSea": "undistorted_IUI3-RedSea",
    "Curasao": "undistorted_Curasao",
    "JapaneseGradens-RedSea": "undistorted_JapaneseGradens-RedSea",
}
REQUIRED = (
    "step-10000 candidate/reference opacity bank",
    "real-opacity projection with Adam momentum handling",
    "20-event in-budget schedule and M1-only subproblem",
    "native counterfactual renders and event state serialization",
    "alarm lifecycle, reload audit, float evaluation, and assets",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(config_path: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    config = yaml.safe_load(config_path.read_text())
    resolved = {"source_config": str(config_path.resolve()), "repo": str(ROOT), **config}
    scene_data = {}
    for scene, dirname in SCENE_MAP.items():
        data = ROOT / "undistorted_data" / dirname
        sparse = data / "sparse" / "0"
        if not data.is_dir() or not sparse.is_dir():
            raise FileNotFoundError(f"raw COLMAP data is unavailable for {scene}: {sparse}")
        scene_data[scene] = {"data": str(data), "sparse": str(sparse)}
    source_config = ROOT / "research_runs/native_responsibility_raw_run_smoke_v3/source_config.yml"
    if not source_config.is_file(): raise FileNotFoundError(source_config)
    payload = dict(status='PREPARED',resolved=resolved,scene_data=scene_data,
        reference_config=str(source_config),reference_config_sha256=digest(source_config),
        python=sys.executable,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        formal_updates=0,smoke_updates=0,blockers=[],
        lr_schedule='retained accepted baseline scheduler max_steps=15000, unchanged',
        topology='native before 10000; frozen after reference, no formal cull',
        pixel_empty_strata='renormalize nonempty stratum weights; unused pixel slots remain unused')
    output.mkdir(parents=True,exist_ok=True)
    (output/'checks').mkdir()
    (output/'code').mkdir()
    files=['scripts/experiments/'+name+'.py' for name in [
        'run_native_responsibility_15k','run_native_responsibility_worker',
        'native_responsibility_controller','run_direct_mdrr_cica_scene','audit_native_responsibility_smoke']]
    payload['code_identity']={f:digest(ROOT/f) for f in files}
    for f in files: shutil.copy2(ROOT/f,output/'code'/Path(f).name)
    shutil.copy2(source_config,output/'native_source_config.yml')
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(payload,sort_keys=False))
    (output/'run_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps(dict(complete=False,status='PREPARED'),indent=2)+'\n')


def _run_manifest(run_dir, status='RUNNING'):
    path=run_dir/'run_manifest.json'; payload=json.loads(path.read_text())
    payload['status']=status; path.write_text(json.dumps(payload,indent=2)+'\n'); return payload

def _worker_cmd(scene, arm, gpu, out, iterations, resume=None):
    cmd=[sys.executable,str(ROOT/'scripts/experiments/run_native_responsibility_worker.py'),
         '--scene',scene,'--arm',arm,'--gpu',str(gpu),'--output-dir',str(out)]
    if iterations < 15000: cmd += ['--smoke']
    if resume is not None: cmd += ['--resume-checkpoint',str(resume)]
    return cmd

def train_all(run_dir, iterations, max_parallel, resume_policy):
    import os,time,threading
    from concurrent.futures import ThreadPoolExecutor
    if max_parallel not in (1,2,3,4): raise ValueError('max-parallel must be 1..4')
    if resume_policy!='same-run-only' or iterations!=15000: raise ValueError('requires same-run-only and 15000 iterations')
    if not json.loads((run_dir/'checks/preflight.json').read_text())['passed']: raise RuntimeError('preflight failed')
    gpu={'Panama':'9','IUI3-RedSea':'7','Curasao':'6','JapaneseGradens-RedSea':'8'}
    (run_dir/'jobs').mkdir(exist_ok=True);(run_dir/'logs').mkdir(exist_ok=True)
    prior={}
    if (run_dir/'active_job.json').is_file():
        d=json.loads((run_dir/'active_job.json').read_text()); prior[(d['scene'],d['arm'])]=d
    def alive(pid):
        f=Path('/proc')/str(pid)/'stat'
        return f.exists() and f.read_text().split()[2]!='Z'
    def job(scene,arm):
        out=run_dir/scene/arm;out.mkdir(parents=True,exist_ok=True)
        record=run_dir/'jobs'/f'{scene}_{arm}.json'
        running=prior.get((scene,arm))
        if record.is_file(): running=json.loads(record.read_text())
        if running and alive(running['pid']):
            record.write_text(json.dumps(dict(running,adopted=True),indent=2)+'\n')
            while alive(running['pid']):time.sleep(15)
        final=out/'final_summary.json'
        if final.is_file() and json.loads(final.read_text()).get('final_step')==15000:return dict(scene=scene,arm=arm,status='complete')
        resume=None
        if any(out.iterdir()):
            ckpts=sorted((out/'checkpoints').glob('step-*.ckpt'))
            if not ckpts:raise RuntimeError(f'nonempty job without checkpoint: {out}')
            resume=ckpts[-1]
            if resume.name=='step-000015000.ckpt':
                cmd=_worker_cmd(scene,arm,gpu[scene],out,iterations)+['--evaluate-only']
                with (run_dir/'logs'/f'{scene}_{arm}_evaluation_repair.log').open('a') as stream:
                    subprocess.run(cmd,cwd=ROOT,env={**dict(os.environ),'CUDA_VISIBLE_DEVICES':gpu[scene],'CONDA_DEFAULT_ENV':'water_splatting'},stdout=stream,stderr=subprocess.STDOUT,check=True)
                summary=json.loads((out/'training_summary.json').read_text())
                (out/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                return dict(scene=scene,arm=arm,status='evaluation_recovered')
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={row.split(',')[0].strip():int(row.split(',')[1]) for row in usage.strip().splitlines()}
        if memory[gpu[scene]]>500:raise RuntimeError(f'GPU {gpu[scene]} occupied; no process terminated')
        cmd=_worker_cmd(scene,arm,gpu[scene],out,iterations,resume)+['--source-config',str(run_dir/'native_source_config.yml')]
        log=run_dir/'logs'/f'{scene}_{arm}.log'
        with log.open('a') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,env={**dict(os.environ),'CUDA_VISIBLE_DEVICES':gpu[scene],'CONDA_DEFAULT_ENV':'water_splatting'},stdout=stream,stderr=subprocess.STDOUT)
            record.write_text(json.dumps(dict(pid=child.pid,scene=scene,arm=arm,log=str(log),command=cmd),indent=2)+'\n')
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,cmd)
        return dict(scene=scene,arm=arm,status='complete')
    def scene_jobs(scene):return [job(scene,arm) for arm in ['B','N']]
    _run_manifest(run_dir,'TRAINING')
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        results=list(pool.map(scene_jobs,SCENE_MAP))
    payload=_run_manifest(run_dir,'TRAINED');payload.update(training_jobs=sum(results,[]),formal_updates=120000)
    (run_dir/'run_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')

def evaluate_all(run_dir):
    import csv
    gpu={'Panama':'9','IUI3-RedSea':'7','Curasao':'6','JapaneseGradens-RedSea':'8'}
    rows=[]; _run_manifest(run_dir,'EVALUATING')
    for scene in SCENE_MAP:
        for arm in ('B','N'):
            out=run_dir/scene/arm
            cmd=[sys.executable,str(ROOT/'scripts/experiments/run_native_responsibility_worker.py'),'--scene',scene,'--arm',arm,'--gpu',gpu[scene],'--output-dir',str(out),'--evaluate-only']
            metric_path=out/'evaluation_metrics.csv'
            existing=[]
            if metric_path.is_file():
                with metric_path.open() as stream:existing=list(csv.DictReader(stream))
            native=list((out/'renders'/scene/'step_15000').glob(f'*/{arm}/native_float.npz'))
            endpoint_ready=(len(existing)==1 and existing[0]['absolute_step']=='15000'
                and existing[0]['arm']==arm and len(native)==int(existing[0]['view_count'])
                and (out/'per_view_metrics.csv').is_file())
            if not endpoint_ready:
                subprocess.run(cmd,cwd=ROOT,env={**dict(__import__('os').environ),'CUDA_VISIBLE_DEVICES':gpu[scene],'CONDA_DEFAULT_ENV':'water_splatting'},check=True)
            with (out/'evaluation_metrics.csv').open() as stream:
                rows.extend(dict(scene=scene,**row) for row in csv.DictReader(stream))
            export=[sys.executable,str(ROOT/'scripts/experiments/export_native_responsibility_assets.py'),
                '--scene',scene,'--arm',arm,'--gpu',gpu[scene],'--output-dir',str(out)]
            assets=out/'visual_assets'
            if not all((assets/name).is_file() for name in ['interaction.json','endpoint_accounting.json','export_costs.json','display_protocol.json']):
                subprocess.run(export,cwd=ROOT,env={**dict(__import__('os').environ),'CUDA_VISIBLE_DEVICES':gpu[scene],'CONDA_DEFAULT_ENV':'water_splatting'},check=True)
    (run_dir/'evaluation_summary.json').write_text(json.dumps(rows,indent=2)+'\n'); _run_manifest(run_dir,'EVALUATED')

def report(run_dir):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiments/report_native_responsibility.py'),
                    '--run-dir',str(run_dir)],cwd=ROOT,check=True)
    reports=run_dir/'reports'; reports.mkdir(exist_ok=True)
    manifest=json.loads((run_dir/'run_manifest.json').read_text())
    jobs=[]
    for path in run_dir.glob('*/*/final_summary.json'):
        jobs.append(json.loads(path.read_text()))
    (reports/'report.json').write_text(json.dumps({'manifest':manifest,'jobs':jobs},indent=2)+'\n')
    complete=json.loads((run_dir/'completion.json').read_text())['complete']
    _run_manifest(run_dir,'REPORTED' if complete else 'PARTIAL_REPORT_TRAINING')

def command(args):
    if args.action=='prepare': prepare(args.config.resolve(),args.output.resolve()); print(f'prepared {args.output.resolve()}'); return 0
    run_dir=args.run_dir.resolve()
    if args.action=='check':
        if not (run_dir/'run_manifest.json').is_file(): raise FileNotFoundError(run_dir/'run_manifest.json')
        smoke_paths={
            'Panama/B':'native_responsibility_event_smoke_Panama_B_v1',
            'Panama/N':'native_responsibility_event_smoke_v1',
            'IUI3-RedSea/B':'native_responsibility_event_smoke_IUI3_B_v2',
            'IUI3-RedSea/N':'native_responsibility_event_smoke_IUI3_N_v1',
            'Curasao/B':'native_responsibility_event_smoke_Curasao_B_v1',
            'Curasao/N':'native_responsibility_event_smoke_Curasao_N_v1',
            'JapaneseGradens-RedSea/B':'native_responsibility_event_smoke_Japanese_B_v1',
            'JapaneseGradens-RedSea/N':'native_responsibility_event_smoke_Japanese_N_v1'}
        evidence={}; total=0
        for key,dirname in smoke_paths.items():
            root=ROOT/'research_runs'/dirname
            replay=json.loads((root/'resume_next_update_audit.json').read_text())
            reload=json.loads((root/'smoke_reload_audit.json').read_text())
            if not replay['passed'] or not reload['forward_finite'] or not reload['model_hash_match']:
                raise RuntimeError(f'smoke failed: {key}')
            if replay['actual_updates']>20: raise RuntimeError(f'smoke budget exceeded: {key}')
            total+=replay['actual_updates']; evidence[key]=dict(path=str(root),replay=replay)
        branches=json.loads((ROOT/'research_runs/native_responsibility_event_smoke_v1/native_branch_audit.json').read_text())
        if not branches['passed']: raise RuntimeError('native branch checks failed')
        passed=total<=160
        result=dict(passed=passed,smoke_actual_updates=total,runs=evidence,representative_branches=branches,
            projection='reuse existing FP32-tolerance validation; no repeated projection updates')
        (run_dir/'checks/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest=_run_manifest(run_dir,'CHECKED'); manifest['smoke_updates']=total
        (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(passed=passed,actual_smoke_updates=total),indent=2)); return 0 if passed else 1
    if args.action=='train-all': train_all(run_dir,args.iterations,args.max_parallel,args.resume_policy); return 0
    if args.action=='evaluate': evaluate_all(run_dir); return 0
    if args.action=='report': report(run_dir); return 0
    if args.action=='run-all':
        train_all(run_dir,args.iterations,args.max_parallel,args.resume_policy); evaluate_all(run_dir); report(run_dir); return 0
    raise ValueError(args.action)

def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("check")
    p.add_argument("--run-dir", type=Path, required=True)
    for action in ("train-all", "evaluate", "report", "run-all"):
        p = subs.add_parser(action)
        p.add_argument("--run-dir", type=Path)
        p.add_argument("--config", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--iterations", type=int, default=15000)
        p.add_argument("--max-parallel", type=int, default=1)
        p.add_argument("--resume-policy", default="same-run-only")
        if action=='evaluate':
            p.add_argument('--step',type=int,choices=[15000],default=15000)
            p.add_argument('--native-float',action='store_true')
            p.add_argument('--make-clear-montages',action='store_true')
            p.add_argument('--make-videos',action='store_true')
    return command(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
