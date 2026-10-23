#!/usr/bin/env python3
"""Run the registered GCRC screen and one fixed four-scene extension."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments import run_taba_bnd_on as original
from scripts.experiments.run_gcrc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.summarize_gcrc_bnd_on import collect
from scripts.diagnostics.audit_gcrc_artifacts import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    root=args.output_root.resolve()
    original.check_environment()
    for scene in original.SCENES:
        assert original.read_json(root/"engineering/verified"/scene/"audit.json")["passed"]
    assert original.read_json(root/"engineering/cuda_nonregression.json")["no_regression"]
    assert original.read_json(root/"source_asset_audit.json")["passed"]
    assert original.read_json(root/"engineering/test_results.json")["pytest_passed"]
    sources=original.read_json(ROOT/"outputs/taba_dcu_bnd_on_20260910/four_scene_source_audit.json")
    for row in sources["rows"]:
        assert original.sha256(Path(row["checkpoint"]))==row["sha256"]
    sources["reverified_for"]="GCRC BND ON"
    original.write_json(root/"four_scene_source_audit.json",sources)
    files=["water_splatting/gcrc.py","water_splatting/grac.py","water_splatting/taba.py",
           "water_splatting/water_splatting.py","water_splatting/fields/gaussian_appearance.py",
           "scripts/experiments/run_direct_mdrr_cica_scene.py","scripts/experiments/run_gcrc_bnd_on.py",
           "scripts/experiments/run_taba_bnd_on.py"]
    snapshot=root/"code_snapshot"
    snapshot.mkdir(exist_ok=False)
    manifest=[]
    for name in files:
        target=snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        manifest.append({"path":name,"sha256":original.sha256(target)})
    original.write_json(snapshot/"manifest.json",{"files":manifest,
        "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()})
    launch_args=SimpleNamespace(output_root=root,max_step=14999)
    worker=ROOT/"scripts/experiments/run_gcrc_bnd_on.py"
    jobs=[(scene,arm,gpu) for arm in PRESETS for scene,gpu in (("JapaneseGradens-RedSea","6"),("Panama","8"))]
    original.launch(launch_args,jobs=jobs,worker_script=worker,audit_sources=False)
    screen=collect(root)
    assert len(screen["final_metrics"])==8 and all(s["integrity_passed"] for s in screen["states"])
    assert all(p["passed"] for p in screen["pairings"])
    selected=screen["selected_variant"]
    assert selected in ("E1","E2","E3")
    original.write_json(root/"four_scene_selection.json",{"selected_variant":selected,"mode":PRESETS[selected],
        "rule":screen["selection_rule"],"two_scene_means":screen["means"]})
    for source,target in (("summary.json","two_scene_summary.json"),("report.md","two_scene_report.md")):
        shutil.copy2(root/source,root/target)
    print(json.dumps({"stage":"four_scene_extension","selected":selected}),flush=True)
    jobs=[("Curasao","E0","6"),("Curasao",selected,"7"),("IUI3-RedSea","E0","8"),("IUI3-RedSea",selected,"9")]
    original.launch(launch_args,jobs=jobs,worker_script=worker,audit_sources=False)
    summary=collect(root)
    result=audit(root)
    print(json.dumps({"complete_runs":len(summary["final_metrics"]),"audit_passed":result["passed"]}),flush=True)


if __name__=="__main__":
    main()
