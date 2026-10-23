#!/usr/bin/env python3
"""Build native-resolution side-by-side assets and an evidence-based report."""
import argparse,csv,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
SCENES=['Panama','IUI3-RedSea','Curasao','JapaneseGradens-RedSea']
def read(path):return json.loads(path.read_text())
def concat(paths,labels,out):
    images=[Image.open(p).convert('RGB') for p in paths]
    if len({im.size for im in images})!=1: raise RuntimeError(f'native dimensions differ: {paths}')
    w,h=images[0].size; canvas=Image.new('RGB',(w*len(images),h+24),'black'); d=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)): canvas.paste(im,(i*w,24));d.text((i*w+8,5),label,fill='white')
    out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out)
    return images

def report(run):
    rows=[]; costs=[]; diagnoses=[]; interactions=[]; assets=run/'comparisons'; all_complete=True
    for scene in SCENES:
        B=run/scene/'B';N=run/scene/'N'
        for arm,out in [('B',B),('N',N)]:
            metrics=out/'evaluation_metrics.csv'
            summary=out/'final_summary.json'
            if not metrics.is_file() or not summary.is_file():all_complete=False;continue
            with metrics.open() as f:
                for r in csv.DictReader(f): rows.append(dict(scene=scene,**r))
            s=read(summary)
            if s.get('final_step')!=15000 or s.get('completed_updates')!=15000:
                all_complete=False
            accounting=out/'visual_assets/endpoint_accounting.json'
            costs.append(dict(scene=scene,arm=arm,summary=s,
                endpoint_accounting=read(accounting) if accounting.is_file() else None))
            export_cost=out/'visual_assets/export_costs.json'
            if export_cost.is_file():costs[-1]['asset_export']=read(export_cost)
            per_view=out/'per_view_metrics.csv'
            if per_view.is_file():
                with per_view.open() as stream:
                    costs[-1]['endpoint_render_seconds']=sum(float(r['render_seconds']) for r in csv.DictReader(stream))
            if (out/'event_history.json').is_file():diagnoses.append(dict(scene=scene,arm=arm,events=read(out/'event_history.json')))
            if (out/'visual_assets/interaction.json').is_file():
                interactions.extend(read(out/'visual_assets/interaction.json'))
            else:
                all_complete=False
            if not (out/'visual_assets/export_costs.json').is_file():
                all_complete=False
            for step in [0,5000,10000,12000,14000,15000]:
                if not (out/'checkpoints'/f'step-{step:09d}.ckpt').is_file():
                    all_complete=False
        renderB=B/'renders'/scene/'step_15000';renderN=N/'renders'/scene/'step_15000'
        if renderB.is_dir() and renderN.is_dir():
            for view in sorted(p.name for p in renderB.iterdir() if p.is_dir()):
                bp=renderB/view/'B';np=renderN/view/'N';dest=assets/scene/view
                for kind in ['underwater','clear']:
                    ims=concat([bp/'gt.png',bp/f'{kind}.png',np/f'{kind}.png'],['Observation (reference)',f'B 15000 {kind}',f'N 15000 {kind}'],dest/f'{kind}.png')
                    w,h=ims[0].size
                    boxfile=B/'visual_assets/fixed_10k_crop_boxes.json'
                    boxes=read(boxfile)['views'].get(view,{}) if boxfile.is_file() else {}
                    for label,box in boxes.items():
                        cropped=[im.crop(box) for im in ims]; panel=Image.new('RGB',(sum(im.width for im in cropped),cropped[0].height))
                        x=0
                        for im in cropped:panel.paste(im,(x,0));x+=im.width
                        panel.save(dest/f'{kind}_{label}_native_crop.png')
                for arm,out in [('B',B),('N',N)]:
                    history=out/'visual_assets/clear_history'/view
                    if history.is_dir():concat([history/f'{step}.png' for step in [10000,12000,14000,15000]],
                        [f'{arm} clear {step}' for step in [10000,12000,14000,15000]],dest/f'{arm}_clear_history.png')
        seqB=B/'visual_assets/camera_sequence';seqN=N/'visual_assets/camera_sequence'
        if seqB.is_dir() and seqN.is_dir():
            views=sorted(set(p.name for p in seqB.iterdir())&set(p.name for p in seqN.iterdir()))
            for kind in ['clear','underwater']:
                frames=assets/scene/f'{kind}_camera_sequence_frames';frames.mkdir(parents=True,exist_ok=True)
                for i,view in enumerate(views):concat([seqB/view/f'{kind}.png',seqN/view/f'{kind}.png'],[f'B {view}',f'N {view}'],frames/f'{i:05d}.png')
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','2','-i',str(frames/'%05d.png'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',str(assets/scene/f'{kind}_existing_camera_sequence.mp4')],check=True)
    reports=run/'reports';reports.mkdir(exist_ok=True)
    (reports/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (reports/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
    (reports/'mechanism.json').write_text(json.dumps(diagnoses,indent=2)+'\n')
    (reports/'interactions.json').write_text(json.dumps(interactions,indent=2)+'\n')
    table=['| Scene | Arm | PSNR | SSIM | LPIPS | MSE |','|---|---|---:|---:|---:|---:|']
    for r in rows:table.append('| '+ ' | '.join([r['scene'],r['arm']]+[f"{float(r[k]):.6g}" for k in ['PSNR','SSIM','LPIPS','MSE']])+' |')
    if all_complete and len(rows)==8:
        for arm in ['B','N']:
            group=[r for r in rows if r['arm']==arm]
            table.append('| '+' | '.join(['Four-scene equal mean',arm]+[f"{sum(float(r[k]) for r in group)/len(group):.6g}" for k in ['PSNR','SSIM','LPIPS','MSE']])+' |')
    text='# Native responsibility 15k\n\n'+('Eight endpoint evaluations complete.\n\n' if all_complete and len(rows)==8 else 'Partial run; no complete four-scene mean is reported.\n\n')+'\n'.join(table)+'\n\n'
    text+='Native-resolution comparisons are in ../comparisons. All panels use native exposure and clamp [0,1], with no per-arm color adjustment. Crops use shared B-10k depth-quantile and observed-edge coordinates, without resampling; they are visual context, not water truth. Videos show existing sorted training cameras, not an interpolated path. There is no clear ground truth and no clear PSNR.\n\n'
    text+='Event interaction uses clamp [0,1] RGB MSE: D=L(cap,old)-L(old,old); K=[L(cap,new)-L(cap,old)]-[L(old,new)-L(old,old)]. K<0 indicates specific compensation within this model, not physical water recovery. Low candidate coverage remains a scientific limitation.\n'
    accounting=run/'checks/update_accounting.json'
    if accounting.is_file():
        ledger=read(accounting)
        text+=f"\nSmoke actual updates: {ledger['total_actual_updates']} (event smoke {ledger['event_smoke_updates']}, prior raw worker {ledger['prior_raw_worker_updates']}). Panama/B accumulated 21 updates versus the per-arm limit of 20; the one-update overrun is retained as a protocol deviation. Smoke and replay updates are excluded from formal training.\n"
    text+='\n| Scene | Arm | Events | Candidate coverage | Final event rho | Persistent alarm events | Candidate hits at event projection |\n|---|---|---:|---:|---:|---:|---:|\n'
    for diagnosis in diagnoses:
        events=diagnosis['events']
        if not events:continue
        text+='| '+ ' | '.join([diagnosis['scene'],diagnosis['arm'],str(len(events)),f"{events[0]['coverage']:.4f}",f"{events[-1]['rho_applied']:.4f}",str(sum(bool(e['alarms']) for e in events)),str(sum(e.get('projection',{}).get('actual_candidate_hits',0) for e in events))])+' |\n'
    text+='\nRho above is the dose applied at the final event, before any alarm relaxation at its end; event history retains each applied dose and alarm. Candidate hits count event-start projections, not all per-update projections. Coverage below 0.50 misses the selection target; coverage below 0.20 limits the hypothesis test. Planned dose alone is not evidence of capacity reduction.\n'
    text+='\n| Scene | Arm | Last pre-projection capacity / 10k reference | Last post-projection capacity / reference | Max event opacity violation |\n|---|---|---:|---:|---:|\n'
    for diagnosis in diagnoses:
        events=diagnosis['events']
        if not events:continue
        p=events[-1].get('projection',{});den=p.get('weighted_tau_reference',0)
        ratios=[f"{p[k]/den:.6g}" if den>0 else 'N/A' for k in ['weighted_tau_before','weighted_tau_after']]
        violation=max(e.get('projection',{}).get('max_opacity_violation',0) for e in events)
        text+='| '+' | '.join([diagnosis['scene'],diagnosis['arm']]+ratios+[f'{violation:.6g}'])+' |\n'
    text+='\nCapacity uses fixed 10k candidate footprints. It does not alone measure visible anomaly removal or identify how much later Gaussian layers and SH replace the removed contribution. Four-grid replay and clear history are necessary visual evidence; complete attribution of replacement is not established by these scalar diagnostics.\n'
    text+='\n| Scene | Arm | Fit mean D | Fit mean K | Monitor mean D | Monitor mean K |\n|---|---|---:|---:|---:|---:|\n'
    for scene in SCENES:
        for arm in ['B','N']:
            group=[r for r in interactions if r['scene']==scene and r['arm']==arm]
            if not group:continue
            vals=[]
            for split in ['fit','monitor']:
                g=[r for r in group if r['split']==split]
                vals.extend([f"{sum(r[k] for r in g)/len(g):.7g}" if g else 'N/A' for k in ['D','K']])
            text+='| '+' | '.join([scene,arm]+vals)+' |\n'
    text+='\nInteractions average the exported fixed events and views; monitor cameras are held out of the M1-only window, but may occur in normal training. No recovery percentage is inferred for small D. A single seed does not establish statistical significance.\n'
    total=sum(int(c['summary'].get('final_step',0))-int(c['summary'].get('first_step',1))+1 for c in costs)
    text+=f'\nCompleted-job formal iterations: {total}; expected full budget: 120000. Training wall time summed across completed jobs: '+f"{sum(c['summary'].get('training_wall_seconds',0) for c in costs):.1f} seconds (parallel job times, not elapsed campaign time). Detailed costs are in costs.json; native float renders accompany endpoint PNGs.\n"
    (reports/'README.md').write_text(text)
    (run/'completion.json').write_text(json.dumps(dict(complete=all_complete and len(rows)==8,endpoint_rows=len(rows)),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();report(a.run_dir.resolve())
if __name__=='__main__':main()
