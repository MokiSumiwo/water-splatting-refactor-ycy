#!/usr/bin/env python3
"""Paired native metrics, fixed-mask diagnostics and all-view contact sheets."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
SOURCE=ROOT/'outputs/fsb_cica_fork_repeat_20260916_v1'
MAPS=ROOT/'outputs/fsb_far_field_diagnostic_20260916_v1'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255.
def panel(canvas,arr,x,y,w,h,label):
    im=Image.fromarray((np.clip(arr,0,1)*255).round().astype(np.uint8));im.thumbnail((w-8,h-32))
    canvas.paste(im,(x+(w-im.width)//2,y+30+(h-32-im.height)//2))
    ImageDraw.Draw(canvas).text((x+6,y+7),label,font=ImageFont.truetype(FONT,13),fill='white')
def edge(im):
    gray=im.mean(-1);dy,dx=np.gradient(gray);return np.sqrt(dx*dx+dy*dy)

def gain_plot(out,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    arms=list(dict.fromkeys(r['arm'] for r in contrasts))
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    x=np.arange(len(SCENES));width=.72/max(len(arms),1)
    for i,scene in enumerate(SCENES):
        gap=next(r['repeat_psnr_gap'] for r in contrasts if r['scene']==scene)
        ax.add_patch(Rectangle((i-.42,-gap),.84,2*gap,color='#999999',alpha=.2,
                              label='Observed baseline repeat gap (+/-); not CI' if i==0 else None))
    for i,arm in enumerate(arms):
        y=[next(r['delta_PSNR'] for r in contrasts if r['scene']==scene and r['arm']==arm) for scene in SCENES]
        ax.bar(x+(i-(len(arms)-1)/2)*width,y,width,label=arm)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,['Curasao','IUI3','JapaneseGradens','Panama'])
    ax.set_ylabel('PSNR change vs repeated baseline mean (dB)')
    ax.set_title('Same initial state, 2000 updates, all original test views')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'psnr_gain_vs_repeat.png',dpi=180);fig.savefig(out/'psnr_gain_vs_repeat.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.input.resolve();out=a.output.resolve() if a.output else root/'summary'
    out.mkdir(parents=True,exist_ok=False);(out/'visualizations').mkdir()
    metrics=[];perview=[];regionrows=[];clearrows=[];contrasts=[];audit=[];gallery=[];overview=[]
    available=[]
    for arm in ('BASE_A','CDRC','LSPR','MST','BASE_B'):
        if all((root/s/arm/'result.json').is_file() for s in SCENES):available.append(arm)
    assert 'BASE_A' in available and 'BASE_B' in available
    for scene in SCENES:
        originals=SOURCE/scene/'R0/C0a'
        initial=readcsv(originals/'per_view_metrics.csv')
        views=[r['view_id'] for r in initial];scene_metrics={}
        for arm in ['INITIAL']+available:
            rows=initial if arm=='INITIAL' else readcsv(root/scene/arm/'per_view_metrics.csv')
            assert {r['view_id'] for r in rows}==set(views)
            avg={key:float(np.mean([float(r[key]) for r in rows])) for key in ('PSNR','SSIM','LPIPS','MSE')}
            record=dict(scene=scene,arm=arm,views=len(rows),**avg);metrics.append(record);scene_metrics[arm]=record
            perview.extend(dict(r,scene=scene,arm=arm) for r in rows)
            if arm!='INITIAL':
                res=json.loads((root/scene/arm/'result.json').read_text());protocol=json.loads((root/scene/arm/'protocol.json').read_text())
                restore=json.loads((root/scene/arm/'restore_audit.json').read_text())
                assert res['completed'] and res['source_unchanged'] and restore['all_pass']
                assert sha(Path(res['checkpoint']))==res['checkpoint_sha256']
                assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']
                if res.get('transport_checkpoint'):
                    assert Path(res['transport_checkpoint']).is_file()
                audit.append(dict(scene=scene,arm=arm,restore_hashes=restore['hashes'],sequence=protocol['camera_sequence_sha256'],
                                  updates=res['updates'],checkpoint_hash_pass=True,source_hash_pass=True,transport_sha256=sha(Path(res['transport_checkpoint'])) if res.get('transport_checkpoint') else None,bounded=res['all_clear_bounded'],
                                  auxiliary_nonzero=res['auxiliary_nonzero_calls'],seconds=res['training_seconds'],peak_cuda_gb=res['peak_cuda_gb']))
        for arm in [x for x in available if not x.startswith('BASE')]:
            contrasts.append(dict(scene=scene,arm=arm,**{
                'delta_'+key:scene_metrics[arm][key]-.5*(scene_metrics['BASE_A'][key]+scene_metrics['BASE_B'][key]) for key in ('PSNR','SSIM','LPIPS')},
                repeat_psnr_gap=abs(scene_metrics['BASE_A']['PSNR']-scene_metrics['BASE_B']['PSNR'])))
        bounds=json.loads((MAPS/scene/'audit.json').read_text())['depth_boundaries']
        for view in views:
            orig=originals/'renders'/scene/'step_14999'/view/'C0a'
            gt=rgb(orig/'gt.png');under={'INITIAL':rgb(orig/'underwater.png')};clear={'INITIAL':rgb(orig/'clear_raw_display_clamp01.png')}
            for arm in available:
                paths=list((root/scene/arm/'renders'/scene).glob('step_*/'+view+'/'+arm))
                assert len(paths)==1
                under[arm]=rgb(paths[0]/'underwater.png');clear[arm]=rgb(paths[0]/'clear_raw_display_clamp01.png')
                assert np.array_equal(rgb(paths[0]/'gt.png'),gt)
            with np.load(MAPS/scene/'maps'/('eval_'+view+'.npz')) as f:
                d=f['depth'];valid=(f['A']>=.9)&(f['cv']<=.1)&(d>0)&np.isfinite(d)
            masks={'all':np.ones_like(valid),'near':valid&(d<bounds[0]),'mid':valid&(d>=bounds[0])&(d<bounds[1]),'far':valid&(d>=bounds[1])}
            for reg,mask in masks.items():
                if not mask.any():continue
                for arm in under:
                    # All region comparisons use identical native stride-4 cells,
                    # and PNG quantization; these do not replace tensor PSNR above.
                    x=under[arm][::4,::4];y=gt[::4,::4];c=clear[arm][::4,::4];b=clear['BASE_A'][::4,::4]
                    mse=float(np.mean((x[mask]-y[mask])**2))
                    regionrows.append(dict(scene=scene,view=view,region=reg,arm=arm,n=int(mask.sum()),mse=mse,psnr=-10*math.log10(max(mse,1e-12))))
                    clearrows.append(dict(scene=scene,view=view,region=reg,arm=arm,n=int(mask.sum()),
                        mae_vs_base=float(np.abs(c[mask]-b[mask]).mean()),black_fraction=float((c[mask].max(-1)<.05).mean()),
                        edge_magnitude=float(edge(c)[mask].mean()),blue_minus_red=float((c[...,2]-c[...,0])[mask].mean())))
            cols=['GT','INITIAL']+available;w=350;h=270
            canvas=Image.new('RGB',(w*len(cols),h*3),'#17191d')
            for j,arm in enumerate(cols):
                if arm=='GT':
                    panel(canvas,gt,j*w,0,w,h,scene+'/'+view+' underwater GT')
                    panel(canvas,np.zeros_like(gt),j*w,h,w,h,'Clear: no ground truth')
                    panel(canvas,np.zeros_like(gt),j*w,2*h,w,h,'Clear abs delta x10 vs BASE_A')
                else:
                    panel(canvas,under[arm],j*w,0,w,h,arm+' underwater')
                    panel(canvas,clear[arm],j*w,h,w,h,arm+' raw clear [0,1]')
                    panel(canvas,10*np.abs(clear[arm]-clear['BASE_A']),j*w,2*h,w,h,arm+' clear abs delta x10')
            name=scene+'_'+view+'.png';canvas.save(out/'visualizations'/name);gallery.append(name)
            # Crop location depends only on fixed pretraining far mask.
            patch=48
            integral=np.pad(masks['far'].astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
            scores=integral[patch:,patch:]-integral[:-patch,patch:]-integral[patch:,:-patch]+integral[:-patch,:-patch]
            yy,xx=np.unravel_index(np.argmax(scores),scores.shape);y0,x0=int(yy*4),int(xx*4);size=patch*4
            crop=Image.new('RGB',(w*len(cols),h*2),'#17191d')
            for j,arm in enumerate(cols):
                arr=gt if arm=='GT' else under[arm];c=np.zeros_like(gt) if arm=='GT' else clear[arm]
                panel(crop,arr[y0:y0+size,x0:x0+size],j*w,0,w,h,arm+' fixed far crop')
                panel(crop,c[y0:y0+size,x0:x0+size],j*w,h,w,h,('No clear GT' if arm=='GT' else arm+' clear'))
            crop_name=scene+'_'+view+'_far_crop.png';crop.save(out/'visualizations'/crop_name);gallery.append(crop_name)
            if view==sorted(views)[0]:overview.append((scene,crop))
    for scene in SCENES:
        rs=[r for r in audit if r['scene']==scene]
        assert all(r['restore_hashes']==rs[0]['restore_hashes'] and r['sequence']==rs[0]['sequence'] for r in rs)
    avg=[]
    for arm in ['INITIAL']+available:
        rs=[r for r in metrics if r['arm']==arm]
        avg.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rs])) for k in ('PSNR','SSIM','LPIPS','MSE')}))
    avgdiff=[]
    for arm in [x for x in available if not x.startswith('BASE')]:
        rs=[r for r in contrasts if r['arm']==arm]
        avgdiff.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rs])) for k in ('delta_PSNR','delta_SSIM','delta_LPIPS','repeat_psnr_gap')}))
    gain_plot(out,contrasts)
    writecsv(out/'scene_metrics.csv',metrics);writecsv(out/'per_view_metrics.csv',perview);writecsv(out/'contrasts.csv',contrasts)
    writecsv(out/'four_scene_means.csv',avg);writecsv(out/'four_scene_contrasts.csv',avgdiff)
    writecsv(out/'region_metrics_png_stride4.csv',regionrows);writecsv(out/'clear_proxies.csv',clearrows)
    (out/'verification.json').write_text(json.dumps(dict(complete=True,paired_restores=True,camera_order_equal=True,source_unchanged=True,rows=audit),indent=2)+'\n')
    mosaic=Image.new('RGB',(overview[0][1].width,sum(im.height+30 for _,im in overview)),'#17191d');yy=0
    for scene,im in overview:
        ImageDraw.Draw(mosaic).text((5,yy+5),scene+' first test view / fixed far crop',fill='white');mosaic.paste(im,(0,yy+30));yy+=im.height+30
    mosaic.save(out/'visualizations/four_scene_far_crops.png')
    page=['<!doctype html><meta charset="utf-8"><title>Far-field direct training</title><style>body{background:#17191d;color:#ddd;font:16px sans-serif}img{width:100%;height:auto}a{color:#8cf}</style>',
          '<h1>Paired far-field training</h1><p>Native underwater / bounded raw clear. No clear GT; no exposure or color correction.</p>']
    page+=['<p>'+html.escape(n)+'</p><a href="visualizations/'+n+'"><img loading="lazy" src="visualizations/'+n+'"></a>' for n in gallery]
    (out/'index.html').write_text('\n'.join(page))
    report=['# 远景候选直接训练结果','',f'已完成：{", ".join(available)}；每路 {audit[0]["updates"]} 次续训，四场景全部 13 张测试图。','',
            '所有分支从同一场景的同一 FSB+CICA checkpoint 恢复。收益对照为 BASE_A/BASE_B 均值，不以多训练 2000 步相对旧 checkpoint 的变化冒充模块收益。','',
            '| 方法 | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|']
    report += [f'| {r["arm"]} | {r["PSNR"]:.6f} | {r["SSIM"]:.6f} | {r["LPIPS"]:.6f} |' for r in avg]
    report += ['', '| 场景 | 方法 | ΔPSNR | ΔSSIM | ΔLPIPS | baseline 重复 PSNR 差 |','|---|---|---:|---:|---:|---:|']
    report += [f'| {r["scene"]} | {r["arm"]} | {r["delta_PSNR"]:+.6f} | {r["delta_SSIM"]:+.6f} | {r["delta_LPIPS"]:+.6f} | {r["repeat_psnr_gap"]:.6f} |' for r in contrasts]
    report += ['', '## 四场景等权收益', '', '| 方法 | ΔPSNR | ΔSSIM | ΔLPIPS | 对照重复差 |', '|---|---:|---:|---:|---:|']
    report += [f'| {r["arm"]} | {r["delta_PSNR"]:+.6f} | {r["delta_SSIM"]:+.6f} | {r["delta_LPIPS"]:+.6f} | {r["repeat_psnr_gap"]:.6f} |' for r in avgdiff]
    report += ['', '## 固定远景区域（PNG stride-4，非全图指标）', '', '| 场景 | 方法 | 远景 PSNR 变化 | Clear MAE vs BASE_A |', '|---|---|---:|---:|']
    for scene in SCENES:
        for arm in [x for x in available if not x.startswith('BASE')]:
            far=[r for r in regionrows if r['scene']==scene and r['region']=='far']
            gain=np.mean([r['psnr'] for r in far if r['arm']==arm])-np.mean([r['psnr'] for r in far if r['arm'] in ('BASE_A','BASE_B')])
            change=np.mean([r['mae_vs_base'] for r in clearrows if r['scene']==scene and r['region']=='far' and r['arm']==arm])
            report.append(f'| {scene} | {arm} | {gain:+.5f} | {change:.6f} |')
    report += ['', '## 实际更新与资源', '', '| 场景 | 方法 | 非零辅助更新 / 总更新 | 训练秒数 | PyTorch 峰值显存 GB |', '|---|---|---:|---:|---:|']
    report += [f'| {r["scene"]} | {r["arm"]} | {r["auxiliary_nonzero"]} / {r["updates"]} | {r["seconds"]:.1f} | {r["peak_cuda_gb"]:.2f} |' for r in audit]
    report += ['', '全图指标来自原生浮点输出；区域表和 clear 差异使用原生 PNG 的固定 stride-4 采样，含 8-bit 量化。区域由初始模型确定，不随方法重新挑选。重复差不是统计置信区间。',
               'LSPR/MST 是本轮明确声明的工作定义：低观测表面的局部颜色先验 / 可见层的散射传输联合训练。LSPR 没有新增监督；MST 并非完整多重散射，与历史冻结 PSF 拟合有重叠，不声称已证明方法新颖性。',
               'CDRC 使用真实训练图像对应进行联合前向拟合；有效颜色为透射率加权 J_eff，并非已识别真实 clear。额外梯度只影响颜色和介质；base 几何训练保留。',
               '本轮无 clear GT。更亮、黑色减少、边缘变化仅是视觉诊断；不能作为去水准确率。未宣称 SOTA。',
               '', '[收益与重复差图](psnr_gain_vs_repeat.png) · [全部 26 张拼接](index.html) · [四场景远景拼接](visualizations/four_scene_far_crops.png) · [状态与哈希](verification.json)']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(completed=True,means=avg,contrasts=avgdiff,images=len(gallery)+1),indent=2))

if __name__=='__main__':main()
