#!/usr/bin/env python3
"""Wait for all requested runs, index retained assets, summarize and replay MST."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PYTHON='/opt/anaconda3/envs/water_splatting/bin/python'
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
DIRECT=ROOT/'outputs/far_field_direct_training_20260917_v3'
NEXT=ROOT/'outputs/far_field_next_training_20260917_v1'
OUT=ROOT/'outputs/far_field_three_candidates_20260917_v1'

def main():
    OUT.mkdir(exist_ok=False)
    for scene in SCENES:
        (OUT/scene).mkdir()
        for arm in ('BASE_A','CDRC','LSPR','MST','BASE_B'):
            target=(DIRECT if arm in ('BASE_A','BASE_B','CDRC') else NEXT)/scene/arm
            (OUT/scene/arm).symlink_to(target,target_is_directory=True)
    (OUT/'sources.json').write_text(json.dumps(dict(direct=str(DIRECT),next=str(NEXT),arms=['BASE_A','CDRC','LSPR','MST','BASE_B']),indent=2))
    while True:
        states=[]
        for p in (DIRECT/'status.json',NEXT/'status.json'):
            try:states.append(json.loads(p.read_text()))
            except (FileNotFoundError,json.JSONDecodeError):states.append({})
        if any(s.get('done') and s.get('failed') for s in states):raise RuntimeError('upstream failure; do not summarize incomplete runs')
        if all(s.get('done') for s in states):break
        time.sleep(5)
    env=dict(os.environ,CONDA_DEFAULT_ENV='water_splatting',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1')
    with (OUT/'summary.log').open('x') as log:
        subprocess.run([PYTHON,str(ROOT/'scripts/diagnostics/summarize_far_field_direct.py'),'--input',str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[]
    for i,scene in enumerate(SCENES):
        log=(OUT/(scene+'_MST_replay.log')).open('x')
        proc=subprocess.Popen([PYTHON,str(ROOT/'scripts/diagnostics/replay_far_field_endpoint.py'),'--run',str(OUT/scene/'MST'),'--output',str(OUT/'endpoint_replay'/scene)],
            cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(6+i)),stdout=log,stderr=subprocess.STDOUT)
        jobs.append((scene,proc,log))
    codes={}
    while jobs:
        pending=[]
        for scene,proc,log in jobs:
            code=proc.poll()
            if code is None:pending.append((scene,proc,log));continue
            codes[scene]=code;log.close()
        jobs=pending
        if jobs:time.sleep(5)
    (OUT/'completion.json').write_text(json.dumps(dict(training_complete=True,summary_complete=True,replay_exit_codes=codes),indent=2)+'\n')
    if any(codes.values()):raise SystemExit(1)

if __name__=='__main__':main()
