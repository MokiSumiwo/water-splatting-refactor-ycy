#!/usr/bin/env python3
"""Final prespecified curriculum comparison; no schedule/checkpoint selection."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.summarize_fsb_degreewise import SCENES, METRICS, csv_rows, write_csv, mean, image_row, sheet

METHODS = ("BND_CICA", "FSB", "CURR")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    a=p.parse_args()
    candidate=a.candidate.resolve()
    comp=candidate/"comparison"
    out=comp/"summary"
    out.mkdir(exist_ok=False)
    rows, means, render_inputs, per_view_deltas=[], [], [], []
    all_images={False:[],True:[]}
    for scene in SCENES:
        status=json.loads((comp/scene/"status.json").read_text())
        assert status["complete"]
        assert all(json.loads((comp/scene/"protocol_pairing.json").read_text()).values())
        math=json.loads((comp/scene/"math_audit.json").read_text())
        assert math["effective_alpha"] == 1 and max(math["final_fsb_endpoint_max_abs"].values())==0
        scene_rows=csv_rows(comp/scene/"per_view_metrics.csv")
        renders=csv_rows(comp/scene/"render_manifest.csv")
        for row in renders:
            assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"]
        for view in status["evaluation_views"]:
            gt=[r for r in renders if r["view_id"]==view and r["kind"]=="gt"]
            assert len(gt)==3 and len({r["sha256"] for r in gt})==1
            c=next(r for r in scene_rows if r["view_id"]==view and r["method"]=="CURR")
            for method in METHODS[:2]:
                ref=next(r for r in scene_rows if r["view_id"]==view and r["method"]==method)
                per_view_deltas.append(dict(scene=scene,view_id=view,reference=method,
                    **{m:float(c[m])-float(ref[m]) for m in METRICS}))
        rows.extend(scene_rows)
        render_inputs.extend(renders)
        for method in METHODS:
            group=[r for r in scene_rows if r["method"]==method]
            keys=METRICS+("clear_black_fraction","accumulation_mean","coverage_gt_001","render_seconds_median","peak_render_allocated_mb")
            means.append(dict(scene=scene,method=method,view_count=len(group),**{k:mean(group,k) for k in keys}))
        for clear in (False,True):
            kind="clear" if clear else "underwater"
            images=[]
            crops=[]
            for view in status["evaluation_views"]:
                images.append(image_row(comp,scene,view,clear=clear,candidate_method="CURR",candidate_label="FSB curriculum"))
                crops.append(image_row(comp,scene,view,clear=clear,crop=True,candidate_method="CURR",candidate_label="FSB curriculum"))
            sheet(images,out/f"{scene}_{kind}.jpg")
            sheet(crops,out/f"{scene}_{kind}_center_crops.jpg")
            all_images[clear].extend(images)
    for clear,images in all_images.items():
        sheet(images,out/("clear_all_views.jpg" if clear else "underwater_all_views.jpg"))
    for method in METHODS:
        group=[r for r in means if r["method"]==method]
        means.append(dict(scene="MACRO",method=method,view_count=sum(r["view_count"] for r in group),**{k:mean(group,k) for k in keys}))
    deltas=[]
    for scene in (*SCENES,"MACRO"):
        cur=next(r for r in means if r["scene"]==scene and r["method"]=="CURR")
        for method in METHODS[:2]:
            ref=next(r for r in means if r["scene"]==scene and r["method"]==method)
            deltas.append(dict(scene=scene,reference=method,**{m:cur[m]-ref[m] for m in METRICS}))
    macro={r["method"]:r for r in means if r["scene"]=="MACRO"}
    gates=dict(macro_psnr_ge_fsb=macro["CURR"]["PSNR"]>=macro["FSB"]["PSNR"],
               macro_ssim_ge_bnd=macro["CURR"]["SSIM"]>=macro["BND_CICA"]["SSIM"],
               macro_lpips_le_fsb=macro["CURR"]["LPIPS"]<=macro["FSB"]["LPIPS"],
               scene_psnr_drop_le_010=all(r["PSNR"]>=-.1 for r in deltas if r["reference"]=="FSB" and r["scene"]!="MACRO"))
    decision=dict(metric_gates=gates,metric_gates_all_pass=all(gates.values()),
                  visual_review="see separate research note; proxy changes cannot certify clear accuracy")
    write_csv(out/"scene_metrics.csv",means)
    write_csv(out/"metric_deltas.csv",deltas)
    write_csv(out/"per_view_metrics.csv",rows)
    write_csv(out/"per_view_deltas.csv",per_view_deltas)
    write_csv(out/"input_render_manifest.csv",render_inputs)
    (out/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    lines=["# FSB Curriculum: Four-Scene Final Evaluation","",
           "One prespecified schedule: smoothstep alpha 0->1 over steps 3000->8000, then unchanged FSB. OCMC+CICA ON, Auxiliary OFF, seed 42, identical scene-specific source at step 3000.",
           "All metrics use the fixed 14999 checkpoint. Means are per scene and then equally across scenes; 13 existing test views total.","",
           "| Scene | BND+CICA PSNR | FSB PSNR | Curriculum PSNR | Delta vs FSB | Delta SSIM vs FSB | Delta LPIPS vs FSB |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for s in (*SCENES,"MACRO"):
        vals={r["method"]:r for r in means if r["scene"]==s}
        d=next(r for r in deltas if r["scene"]==s and r["reference"]=="FSB")
        lines.append(f"| {s} | {vals['BND_CICA']['PSNR']:.6f} | {vals['FSB']['PSNR']:.6f} | {vals['CURR']['PSNR']:.6f} | {d['PSNR']:+.6f} | {d['SSIM']:+.6f} | {d['LPIPS']:+.6f} |")
    lines += ["","| Method | Macro PSNR | SSIM | LPIPS | MSE |","|---|---:|---:|---:|---:|"]
    for m,r in macro.items():lines.append(f"| {m} | {r['PSNR']:.6f} | {r['SSIM']:.9f} | {r['LPIPS']:.9f} | {r['MSE']:.9f} |")
    lines += ["","## Gate Decision","",f"Prespecified metric gates: `{json.dumps(gates)}`.","",
              "Metric pass is necessary, not sufficient: clear visual review and repeat-seed evidence are required before claiming a robust gain.","",
              "## Visualizations","","[All underwater views](underwater_all_views.jpg): GT, BND+CICA, FSB, curriculum, absolute curriculum-FSB difference x4.","",
              "[All clear views](clear_all_views.jpg): BND+CICA, FSB, curriculum, absolute difference x4. No clear GT, no exposure adjustment.",""]
    for s in SCENES:lines.append(f"- {s}: [underwater]({s}_underwater.jpg), [clear]({s}_clear.jpg), [underwater fixed center crop]({s}_underwater_center_crops.jpg), [clear fixed center crop]({s}_clear_center_crops.jpg)")
    lines += ["","## Audit And Limits","",
              "All source hashes, initial model/optimizer/scheduler/projector hashes, continuation RNG and camera sequences match. Independent reload reproduces all stored final metrics, and native GT hashes match across all methods. Current checkpoint schedule metadata is validated against the manifest. Final curriculum rendering equals FSB exactly for the same checkpoint.","",
              "Zero-step gradients and endpoints were evaluated only on training views; no optimizer update was applied. The short training gate uses three predetermined training cameras per scene and only excludes >25% MSE instability. It is not validation for unseen views. Smoke results did not select a schedule.","",
              "The test set has been viewed during previous research; these remain exploratory single-seed comparisons, not a sealed SOTA benchmark or proof of significance. Clear proxy shifts are not accuracy metrics. Normal joint training changes geometry/medium indirectly even though no new geometry or medium loss was added.","",
              "Checkpoints are retained at short-run step 5000, formal step 8000, and formal step 14999; no checkpoint selection occurred. Short and formal runs restart from the same original source rather than resuming partial RNG hashes.","",
              "## Render Diagnostics","",
              "Warm get_outputs_for_camera median of three calls per view; table averages views per scene. Peak allocated CUDA memory includes resident model, cached tensors and rendered outputs, not a controlled performance benchmark.","",
              "| Scene | Method | Render ms | Peak allocated MB | Clear black fraction | Coverage A>0.01 |","|---|---|---:|---:|---:|---:|"]
    for r in means:lines.append(f"| {r['scene']} | {r['method']} | {1000*r['render_seconds_median']:.3f} | {r['peak_render_allocated_mb']:.1f} | {r['clear_black_fraction']:.6f} | {r['coverage_gt_001']:.6f} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(macro=macro,decision=decision),indent=2))
    print(out/"REPORT.md")


if __name__=="__main__":main()
