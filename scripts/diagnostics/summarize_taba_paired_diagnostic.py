#!/usr/bin/env python3
"""Aggregate strict paired TABA bucket rows and apply the conservative gate."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BUCKETS = ("low", "middle", "high")

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
    rows: List[Dict[str, Any]] = []; scene_reports = []
    for path in sorted(a.root.glob("*/ *_paired_bucket_per_view.csv".replace(" ", ""))):
        df = pd.read_csv(path)
        if df.empty: continue
        scene = str(df.scene.iloc[0]); rows.extend(df.to_dict("records"))
        for step in sorted(df.step.unique()):
            d = df[df.step == step]
            report = {"scene": scene, "step": int(step), "view_count": int(d.view_id.nunique())}
            for b in BUCKETS:
                q = d[d.bucket == b]
                report[f"{b}_views"] = int(len(q))
                for col in ("rgb_object_residual_delta_on_minus_off", "rgb_medium_residual_delta_on_minus_off", "chromatic_residual_delta_on_minus_off", "luminance_residual_delta_on_minus_off", "S_med"):
                    report[f"{b}_{col}_mean"] = float(q[col].mean()) if len(q) else float("nan")
                corr = q["corr_t_chromatic_delta_on_minus_off"].dropna()
                report[f"{b}_positive_direction_fraction"] = float((corr > 0).mean()) if len(corr) else float("nan")
            scene_reports.append(report)
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / "paired_bucket_all_views.csv").open("w", newline="", encoding="utf8") as f:
        if rows:
            fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (a.root / "paired_bucket_scene_summary.json").write_text(json.dumps(scene_reports, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf8")
    # D1 is deliberately conservative: require a complete requested checkpoint
    # set and >=60% directional agreement in at least one scene.
    complete = all(any(r["scene"] == s and r["step"] == 14999 for r in scene_reports) for s in ("JapaneseGradens-RedSea", "Panama"))
    eligible = []
    for scene in ("JapaneseGradens-RedSea", "Panama"):
        q = [r for r in scene_reports if r["scene"] == scene]
        for r in q:
            if r.get("low_positive_direction_fraction", 0.0) >= 0.6 and r.get("low_chromatic_residual_delta_on_minus_off_mean", 0.0) > 0 and r.get("high_chromatic_residual_delta_on_minus_off_mean", 0.0) <= r.get("low_chromatic_residual_delta_on_minus_off_mean", 0.0):
                eligible.append((scene, r["step"]))
    decision = {"D1_formally_passes": bool(complete and eligible), "complete_requested_steps": complete, "eligible_scene_steps": eligible, "status": "diagnostic_only_unverified" if not (complete and eligible) else "candidate_requires_review"}
    (a.root / "D1_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(decision, sort_keys=True))

if __name__ == "__main__": main()
