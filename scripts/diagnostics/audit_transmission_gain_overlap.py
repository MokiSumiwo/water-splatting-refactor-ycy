#!/usr/bin/env python3
"""Audit overlap between transmission/support regions and gain error changes."""
from pathlib import Path
import csv, json, math, statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/bnd_transmission_gated_gain_20260914_v2"
SCENES = ("JapaneseGradens-RedSea", "Panama", "Curasao", "IUI3-RedSea")

def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def main():
    rows=[]; train=[]
    for scene in SCENES:
        manifest=json.loads((OUT/scene/"camera_pixels.json").read_text())
        for r in manifest:
            train.append(dict(scene=scene, view=r["view"], validation=r["validation"],
                              eligible_fraction=r.get("eligible_fraction", 0.0), eligible_pixels=r.get("eligible_pixels", 0)))
        for viewdir in sorted((OUT/scene/"renders").iterdir()):
            if not (viewdir/"regions.npz").exists(): continue
            with np.load(viewdir/"original_float.npz") as a:
                base=np.clip(a["underwater"],0,1); gt=a["gt"]
            with np.load(viewdir/"regions.npz") as m: masks={k:m[k].astype(bool) for k in m.files}
            baseerr=np.mean((base-gt)**2,axis=-1)
            for seed in (42,123,3407):
                path=viewdir/f"seed_{seed}_spatial_float.npz"
                if not path.exists(): continue
                pred=np.clip(np.load(path)["underwater"],0,1)
                newerr=np.mean((pred-gt)**2,axis=-1)
                improvement=baseerr-newerr
                for region,mask in masks.items():
                    if mask.shape != improvement.shape: continue
                    count=int(mask.sum())
                    rows.append(dict(scene=scene,view=viewdir.name,seed=seed,region=region,pixels=count,
                                     improvement_mse=float(improvement[mask].mean()) if count else None,
                                     improved_fraction=float((improvement[mask]>0).mean()) if count else None,
                                     error_share=float(baseerr[mask].sum()/max(baseerr.sum(),1e-30)) if count else None,
                                     global_improvement_mse=float(improvement.mean()),
                                     global_improved_fraction=float((improvement>0).mean())))
    (OUT/"gain_overlap_audit.json").write_text(json.dumps(dict(complete=True, rows=rows, training=train),indent=2)+"\n")
    with (OUT/"gain_overlap_audit.csv").open("w",newline="") as f:
        fields=list(rows[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary=[]
    for scene in SCENES:
        x=[r for r in rows if r["scene"]==scene and r["region"]=="all"]
        summary.append(dict(scene=scene, views=len(x), mean_improvement_mse=statistics.mean(r["improvement_mse"] for r in x),
                            mean_improved_fraction=statistics.mean(r["improved_fraction"] for r in x),
                            positive_views=sum(r["improvement_mse"]>0 for r in x)))
    (OUT/"gain_overlap_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(summary=summary, training=train)))

if __name__ == "__main__": main()
