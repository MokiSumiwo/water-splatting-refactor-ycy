#!/usr/bin/env python3
"""Run requested candidates sequentially, with four physical scene GPU lanes."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',default='LSPR,MST')
    p.add_argument('--wait-for',type=Path)
    p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir();(out/'provenance').mkdir()
    sources=['scripts/experiments/run_far_field_next_candidate.py','scripts/experiments/run_far_field_next_lanes.py',
             'water_splatting/far_field_recovery.py','water_splatting/far_field_recovery_next.py','scripts/experiments/run_direct_mdrr_cica_scene.py',
             'scripts/experiments/run_fsb_cica_fork_worker.py','water_splatting/water_splatting.py',
             'water_splatting/fields/gaussian_appearance.py','research_notes/LSPR_MST_DIRECT_TRAINING_PROTOCOL_2026-09-17.md']
    hashes={}
    for name in sources:
        src=ROOT/name;dst=out/'provenance'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);hashes[name]=sha(src)
    (out/'provenance/sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    status={'completed':{},'active':{},'done':False}
    failed=False
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    queues={scene:a.arms.split(',') for scene in SCENES}
    running={}
    while any(queues.values()) or running:
        for i,scene in enumerate(SCENES):
            if scene in running:
                arm,proc,log=running[scene];code=proc.poll()
                if code is None:continue
                log.close();status['completed'][scene+'/'+arm]=code
                del running[scene];del status['active'][scene]
                if code:queues[scene].clear();failed=True
            if queues[scene]:
                if a.wait_for:
                    if not a.wait_for.is_file():continue
                    try:previous=json.loads(a.wait_for.read_text())
                    except json.JSONDecodeError:continue
                    code=previous.get('completed',{}).get(scene+'/BASE_B')
                    if code is None:continue
                    if code:raise RuntimeError('upstream BASE_B failed')
                if any(sha(ROOT/name)!=h for name,h in hashes.items()):raise RuntimeError('source changed mid-suite')
                arm=queues[scene].pop(0)
                log=(out/'logs'/(scene+'_'+arm+'.log')).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+i),CONDA_DEFAULT_ENV='water_splatting',
                         OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='4',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1')
                cmd=[PYTHON,str(ROOT/'scripts/experiments/run_far_field_next_candidate.py'),'--scene',scene,'--arm',arm,
                     '--steps',str(a.steps),'--output',str(out/scene/arm)]
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[scene]=(arm,proc,log);status['active'][scene]={'arm':arm,'pid':proc.pid,'gpu':6+i}
        save()
        if running or any(queues.values()):time.sleep(5)
    status.update(done=True,failed=failed);save()
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
