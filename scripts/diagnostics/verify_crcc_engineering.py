#!/usr/bin/env python3
"""Persist reproducible test, compile and whitespace-check evidence."""

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments.run_crcc_registered_experiment import FILES
from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment, write_json


if __name__=='__main__':
    check_environment()
    output=DEFAULT_OUT/'engineering'
    commands={
        'pytest':[sys.executable,'-m','pytest','tests','-q'],
        'py_compile':[sys.executable,'-m','py_compile',*FILES,
                      'scripts/diagnostics/verify_crcc_engineering.py'],
        'diff_check':['git','diff','--check'],
    }
    result={'python':sys.executable}
    for name,command in commands.items():
        completed=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/(name+'.log')).write_text(completed.stdout)
        result[name+'_passed']=completed.returncode==0
        result[name+'_command']=command
        print(completed.stdout,flush=True)
    write_json(output/'test_results.json',result)
    print(json.dumps(result),flush=True)
    if not all(result[name+'_passed'] for name in commands):
        raise SystemExit(1)
