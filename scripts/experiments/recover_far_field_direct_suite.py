#!/usr/bin/env python3
"""Preserve completed CDRC scenes; retry Panama with activation recomputation."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
OUT=ROOT/'outputs/far_field_direct_training_20260917_v3'
PREV=ROOT/'outputs/far_field_direct_training_20260917_v2'

def main():
    OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();(OUT/'provenance').mkdir()
    names=['scripts/experiments/run_far_field_candidate.py','scripts/experiments/run_far_field_candidate_lowmem.py',
           'scripts/experiments/recover_far_field_direct_suite.py','water_splatting/far_field_recovery.py','water_splatting/far_field_recovery_lowmem.py']
    hashes={}
    for name in names:
        dst=OUT/'provenance'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (OUT/'provenance/sha256.json').write_text(json.dumps(hashes,indent=2))
    queues={s:([('CDRC','run_far_field_candidate_lowmem.py')] if s=='Panama' else [])+[('BASE_B','run_far_field_candidate.py')] for s in SCENES}
    active={};completed={};failed=False
    for s in SCENES:
        (OUT/s).mkdir();(OUT/s/'BASE_A').symlink_to(PREV/s/'BASE_A',target_is_directory=True)
        if s!='Panama':(OUT/s/'CDRC').symlink_to(PREV/s/'CDRC',target_is_directory=True)
    while any(queues.values()) or active:
        for i,s in enumerate(SCENES):
            if s in active:
                arm,proc,log=active[s];code=proc.poll()
                if code is None:continue
                log.close();completed[s+'/'+arm]=code;del active[s]
                if code:queues[s].clear();failed=True
            if queues[s]:
                if s!='Panama' and not (PREV/s/'CDRC/result.json').exists():continue
                arm,worker=queues[s].pop(0)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+i),CONDA_DEFAULT_ENV='water_splatting',OMP_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='4',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1')
                log=(OUT/'logs'/(s+'_'+arm+'.log')).open('x')
                proc=subprocess.Popen([PYTHON,str(ROOT/'scripts/experiments'/worker),'--scene',s,'--arm',arm,'--steps','2000','--output',str(OUT/s/arm)],
                                      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[s]=(arm,proc,log)
        (OUT/'status.json').write_text(json.dumps(dict(completed=completed,active={s:dict(arm=x[0],pid=x[1].pid) for s,x in active.items()},done=not(any(queues.values()) or active),failed=failed),indent=2)+'\n')
        if active or any(queues.values()):time.sleep(5)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
