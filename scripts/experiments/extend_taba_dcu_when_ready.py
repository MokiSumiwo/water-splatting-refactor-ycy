#!/usr/bin/env python3
"""Freeze the full two-scene screen, then run one fixed four-scene comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.audit_taba_artifacts import audit
from scripts.experiments import run_taba_bnd_on as original
from scripts.experiments.run_taba_dcu_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.summarize_taba_dcu_bnd_on import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    original.check_environment()
    targets = [root / scene / arm / "taba_result.json" for scene in original.SCENES for arm in PRESETS]
    while not all(path.is_file() for path in targets):
        print(json.dumps({"stage": "waiting_for_screen", "complete": sum(path.is_file() for path in targets),
                          "required": len(targets)}), flush=True)
        time.sleep(30)
    summary = collect(root)
    assert all(state["integrity_passed"] for state in summary["states"])
    assert all(pair["passed"] for pair in summary["pairings"])
    selected = summary["selected_variant"]
    assert selected in PRESETS and selected != "E0"
    selection_path = root / "four_scene_selection.json"
    if selection_path.exists():
        raise RuntimeError("selection already frozen; inspect retained state before launching again")
    original.write_json(selection_path, {"selected_variant": selected, "preset": PRESETS[selected],
        "rule": summary["selection_rule"], "stable_variants": summary["stable_variants"],
        "two_scene_means": summary["means"], "selected_unix": time.time()})
    for source, target in (("summary.json", "two_scene_summary.json"), ("report.md", "two_scene_report.md")):
        if (root / target).exists():
            raise RuntimeError("refusing to overwrite frozen two-scene evidence")
        shutil.copy2(root / source, root / target)
    print(json.dumps({"stage": "extension", "selected": selected}), flush=True)
    jobs = [("Curasao", "E0", "6"), ("Curasao", selected, "7"),
            ("IUI3-RedSea", "E0", "8"), ("IUI3-RedSea", selected, "9")]
    original.launch(SimpleNamespace(output_root=root, max_step=14999), jobs=jobs,
                    worker_script=ROOT / "scripts/experiments/run_taba_dcu_bnd_on.py", audit_sources=False)
    summary = collect(root)
    integrity = audit(root, variants=tuple(PRESETS))
    print(json.dumps({"complete_runs": len(summary["final_metrics"]), "artifact_audit": integrity}), flush=True)


if __name__ == "__main__":
    main()
