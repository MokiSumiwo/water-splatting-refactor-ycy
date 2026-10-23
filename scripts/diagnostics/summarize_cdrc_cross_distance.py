#!/usr/bin/env python3
"""Summarize frozen CDRC evidence; never train or produce a claimed improved clear image."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
LABELS=('Curasao','IUI3','JapaneseGradens','Panama')
METHODS=('renderer','source_copy','cdrc_bounded','cdrc_multi')
TITLES=('Frozen renderer','Source pixel copy','CDRC bounded','CDRC multi-source')
COLORS=('#4c78a8','#a7a7a7','#f58518','#54a24b')
MAP_ROOT=ROOT/'outputs/fsb_far_field_diagnostic_20260916_v1'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,data):
    if not data:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def select(slices,split,scope='closer_source',region='far',bucket='all'):
    return next(r for r in slices if (r['split'],r['scope'],r['region'],r['bucket'])==(split,scope,region,bucket))


def rgb(rec,key):return np.array([float(rec[key+'_'+c]) for c in 'rgb'])


def rmse(recs,key):return float(np.sqrt(np.mean([float(r[key+'_mse']) for r in recs]))) if recs else None


def fmt(x,digits=5):return '—' if x is None or not np.isfinite(x) else f'{x:.{digits}f}'


def image_panel(canvas,img,box,label):
    x,y,w,h=box;img=img.copy();img.thumbnail((w-8,h-36),Image.Resampling.LANCZOS)
    canvas.paste(img,(x+(w-img.width)//2,y+34+(h-36-img.height)//2))
    ImageDraw.Draw(canvas).text((x+5,y+7),label,font=ImageFont.truetype(FONT,15),fill='white')


def original_paths(scene,view):
    return ROOT/'outputs/fsb_cica_fork_repeat_20260916_v1'/scene/'R0/C0a/renders'/scene/'step_14999'/view/'C0a'


def visualizations(out,summaries,predictions,coverage):
    gallery=[];overview=[];manifest=[]
    out.mkdir()
    for scene in SCENES:
        audit=json.loads((MAP_ROOT/scene/'audit.json').read_text())
        views=sorted(k.split('/')[1] for k in audit['camera_metadata'] if k.startswith('eval/'))
        scene_images=[]
        for view in views:
            base=original_paths(scene,view)
            recs=[r for r in predictions[scene] if r['target']=='eval/'+view and r['scope']=='closer_source']
            far=[r for r in recs if r['target_region']=='far']
            imgs=[Image.open(base/x).convert('RGB') for x in ('gt.png','underwater.png','clear_raw_display_clamp01.png')]
            for file in ('gt.png','underwater.png','clear_raw_display_clamp01.png'):
                manifest.append({'scene':scene,'view':view,'path':str(base/file),'sha256':sha(base/file)})
            overlay=imgs[0].copy();d=ImageDraw.Draw(overlay)
            for r in recs:
                x,y=float(r['u']),float(r['v']);col=(0,255,255) if r['target_region']=='far' else (255,210,40)
                d.ellipse((x-5,y-5,x+5,y+5),outline=col,width=3)
            canvas=Image.new('RGB',(1680,370),'#17191d')
            for j,(img,label) in enumerate(zip(imgs+[overlay],('Underwater GT','Original FSB underwater','Original FSB clear; no clear GT',f'Sparse support: {len(far)} far / {len(recs)} all'))):
                image_panel(canvas,img,(j*420,30,420,338),label)
            ImageDraw.Draw(canvas).text((6,5),scene+' / '+view+' | cyan: far target; yellow: nearer bands. All original renders unchanged.',font=ImageFont.truetype(FONT,15),fill='white')
            file=out/(scene+'_'+view+'_overview.png');canvas.save(file);gallery.append((scene+' / '+view+' original outputs + correspondence support',file))
            scene_images.append((len(far),view,canvas.copy()))
            # Fixed selection by depth then feature index, with no selection by prediction error.
            chosen=sorted(far or recs,key=lambda r:(-float(r['depth_target']),int(r['keypoint'])))[:6]
            if not chosen:continue
            chips=Image.new('RGB',(1680,50+len(chosen)*212),'#17191d');draw=ImageDraw.Draw(chips)
            draw.text((8,8),scene+' / '+view+' | Point predictions only; columns 1-4 are unmodified image crops.',font=ImageFont.truetype(FONT,16),fill='white')
            for i,r in enumerate(chosen):
                yy=50+i*212;u,v=float(r['u']),float(r['v']);su,sv=float(r['source_u']),float(r['source_v'])
                source=Image.open(ROOT/'undistorted_data'/('undistorted_'+scene)/'images/ColorImage'/(r['source'].split('/')[1]+'.png')).convert('RGB')
                panels=[(source,su,sv,'Source underwater'),(imgs[0],u,v,'Target underwater GT'),(imgs[1],u,v,'Frozen underwater'),(imgs[2],u,v,'Frozen clear; no GT')]
                for j,(im,x,y,title) in enumerate(panels):
                    crop=im.crop((round(x)-64,round(y)-64,round(x)+64,round(y)+64)).resize((164,164),Image.Resampling.NEAREST)
                    dc=ImageDraw.Draw(crop);dc.line((76,82,88,82),fill=(255,0,255));dc.line((82,76,82,88),fill=(255,0,255))
                    image_panel(chips,crop,(j*212,yy,212,210),title)
                for j,(method,title) in enumerate(zip(('target',)+METHODS,('GT at point',)+TITLES)):
                    x=858+j*161;color=tuple(np.clip(rgb(r,method)*255,0,255).round().astype(int))
                    draw.text((x,yy+10),title,font=ImageFont.truetype(FONT,12),fill='white')
                    draw.rectangle((x+15,yy+37,x+132,yy+132),fill=color)
                    err='' if method=='target' else f'RMSE {math.sqrt(float(r[method+"_mse"])):.4f}'
                    draw.text((x,yy+143),err,font=ImageFont.truetype(FONT,13),fill='white')
                draw.text((856,yy+172),f"z ratio {float(r['depth_ratio']):.2f} | source {r['source']} | {r['source_count']} sources | target keypoint {r['keypoint']}",font=ImageFont.truetype(FONT,13),fill='white')
            file=out/(scene+'_'+view+'_point_predictions.png');chips.save(file);gallery.append((scene+' / '+view+' matched crops and point predictions',file))
        selected=sorted(scene_images,key=lambda x:(-x[0],x[1]))[0]
        overview.append(selected[2]);
        scene_canvas=Image.new('RGB',(1680,370*len(scene_images)),'#17191d')
        for i,(_,view,im) in enumerate(scene_images):scene_canvas.paste(im,(0,370*i))
        file=out/(scene+'_all_eval_views.png');scene_canvas.save(file)
        gallery.append((scene+' all evaluation views',file))
    joint=Image.new('RGB',(1680,370*4),'#17191d')
    for i,im in enumerate(overview):joint.paste(im,(0,370*i))
    file=out/'four_scene_overview.png';joint.save(file)
    gallery.insert(0,('Four scenes; per-scene view chosen by most far support, not prediction quality',file))
    return gallery,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/cdrc_cross_distance_diagnostic_20260917_v1');args=p.parse_args()
    root=args.input.resolve();out=root/'summary';out.mkdir(exist_ok=False)
    summaries={s:json.loads((root/s/'summary.json').read_text()) for s in SCENES}
    slices={s:json.loads((root/s/'prediction_slices.json').read_text()) for s in SCENES}
    pred={s:rows(root/s/'prediction_rows.csv') for s in SCENES};cov={s:rows(root/s/'coverage.csv') for s in SCENES}
    verification=[]
    for scene in SCENES:
        s=summaries[scene];assert s['completed'] and not s['smoke'] and s['gpu_used'] and s['input_hashes_unchanged']
        native=s['native_audit'];assert native['model_unchanged'] and native['views']==s['train_views']+s['eval_views']
        assert native['model_state_before']==native['model_state_after']
        assert max(v for x in native['replay'] for v in x['replay_max_errors'].values())==0
        assert s['optimizer_steps']==s['backward_calls']==0
        assert sha(Path(s['checkpoint']))==s['checkpoint_sha256']
        for split in ('train','eval'):
            for scope in ('all_cross','closer_source'):
                for reg in ('all','near','mid','far'):
                    rr=[r for r in pred[scene] if r['split']==split and r['scope']==scope and (reg=='all' or r['target_region']==reg)]
                    assert len(rr)==len({(r['target'],r['keypoint']) for r in rr})
                    target=select(slices[scene],split,scope,reg)
                    assert len(rr)==target['n']
                    for key in METHODS:
                        expected=target[key+'_rmse']; actual=rmse(rr,key)
                        assert expected is actual or (expected is not None and abs(expected-actual)<1e-12)
        verification.append({'scene':scene,'completed':True,'csv_recomputed':True,'native_replay_max_error':0.,'checkpoint_unchanged':True})
    protocol_hash=json.loads((root/'provenance/sha256.json').read_text())
    for file,h in protocol_hash.items():assert sha(ROOT/file)==h
    (out/'verification.json').write_text(json.dumps(dict(passed=True,scenes=verification,formal_source_unchanged=True,training_started=False),indent=2)+'\n')
    table=[]
    for scene in SCENES:
        for split in ('train','eval'):
            r=select(slices[scene],split)
            c=[x for x in cov[scene] if x['split']==split and x['region']=='far']
            n=sum(int(x['candidate_keypoints']) for x in c);k=sum(int(x['supported_keypoints']) for x in c)
            spatialn=sum(int(x['candidate_stride4_cells']) for x in c);spatialk=sum(int(x['supported_stride4_cells']) for x in c)
            table.append(dict(scene=scene,split=split,far_targets=r['n'],target_views=r['target_views'],far_candidate_features=n,far_supported_fraction=k/max(n,1),
                              far_candidate_cells=spatialn,far_supported_cells=spatialk,far_supported_cell_fraction=spatialk/max(spatialn,1),
                              **{key+'_rmse':r[key+'_rmse'] for key in METHODS},t_advantage=r['t_advantage_median'],
                              inverse_J_outside_fraction=r['J_single_outside_fraction'],multi_inverse_J_outside_fraction=r['J_multi_outside_fraction'],
                              train_scene_pass=summaries[scene]['admission']['scene_pass']))
    write_csv(out/'far_prediction_summary.csv',table)
    # Export figures with the same scales in train/eval panels.
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    xx=np.arange(4);width=.19
    for ax,split in zip(axes,('train','eval')):
        for i,(key,title,color) in enumerate(zip(METHODS,TITLES,COLORS)):
            vals=[select(slices[s],split)[key+'_rmse'] for s in SCENES]
            ax.bar(xx+(i-1.5)*width,[v if v is not None else np.nan for v in vals],width,label=title,color=color)
        ns=[select(slices[s],split)['n'] for s in SCENES]
        ax.set_xticks(xx);ax.set_xticklabels([name+'\nn='+str(n) for name,n in zip(LABELS,ns)])
        ax.set_title(('Training' if split=='train' else 'Evaluation')+' far targets from closer sources')
        ax.set_ylabel('Matched-point RGB RMSE [0,1]; lower is better');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    ymax=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:ax.set_ylim(0,ymax)
    axes[0].legend(fontsize=9);fig.suptitle('Frozen CDRC diagnostic: sparse matched pixels, not whole-image quality')
    for ext in ('png','pdf'):fig.savefig(out/('prediction_comparison.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for i,split in enumerate(('train','eval')):
        subset=[r for r in table if r['split']==split]
        axes[0].bar(xx+(i-.5)*.35,[r['far_targets'] for r in subset],.35,label=split)
        axes[1].bar(xx+(i-.5)*.35,[100*r['far_supported_fraction'] for r in subset],.35,label=split)
        axes[2].bar(xx+(i-.5)*.35,[100*r['inverse_J_outside_fraction'] if r['inverse_J_outside_fraction'] is not None else np.nan for r in subset],.35,label=split)
    for ax,title in zip(axes,('Unique far target keypoints','Far candidate keypoints supported (%)','Inverse J channels outside [0,1] (%)')):
        ax.set_title(title,fontsize=10);ax.set_xticks(xx);ax.set_xticklabels(LABELS,rotation=15,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('Support is sparse and model-filtered; these are not dense surface coverage estimates')
    for ext in ('png','pdf'):fig.savefig(out/('support_and_color_range.'+ext),dpi=180)
    plt.close(fig)
    gallery,manifest=visualizations(out/'visualizations',summaries,pred,cov)
    write_csv(out/'original_visual_assets.csv',manifest)
    links=[('Prediction errors',out/'prediction_comparison.png'),('Support and color range',out/'support_and_color_range.png')]+gallery
    body=['<!doctype html><meta charset="utf-8"><title>CDRC frozen diagnostic</title><style>body{background:#17191d;color:#eee;font:16px sans-serif;max-width:1700px;margin:24px auto}img{max-width:100%;height:auto}a{color:#83c5ff}section{margin:28px 0}</style>',
          '<h1>CDRC frozen cross-distance diagnostic</h1><p>All original FSB outputs remain unchanged. Point predictions are diagnostic swatches only. No training or new clear rendering; no clear ground truth. Cyan: far support; yellow: other support.</p>']
    for title,path in links:body.append('<section><h2>'+html.escape(title)+'</h2><a href="'+str(path.relative_to(out))+'"><img loading="lazy" src="'+str(path.relative_to(out))+'"></a></section>')
    for scene in SCENES:
        body.append('<h2>'+scene+' all train/eval support maps and matching examples</h2>')
        for path in sorted((root/scene/'visualizations').glob('*.png')):
            body.append('<a href="../'+scene+'/visualizations/'+path.name+'">'+path.stem+'</a> &nbsp; ')
    (out/'index.html').write_text('\n'.join(body))
    admitted=sum(summaries[s]['admission']['scene_pass'] for s in SCENES)
    report=['# CDRC 跨距离共享辐射冻结诊断结果','',f'四场景完整诊断完成：75 张训练图、13 张测试图。达到训练准入规则的场景为 **{admitted}/4**（要求至少 3/4）。',
            '','模型、checkpoint、相机和原始渲染保持冻结。使用 conda water_splatting、物理 GPU 6/7/8/9；0 次反向传播、0 次优化更新。',
            '','## 原生分辨率远景点预测','',
            '以下 RMSE 在 RGB [0,1] 域计算，仅包含通过真实图像匹配与双向几何检查、且拥有较近训练来源的远景特征点。每个目标特征点只计一次。**不是全图 PSNR，也不是去水恢复准确率。**',
            '','| 场景 | 划分 | 唯一远景点 / 视图数 | 原渲染 RMSE | 来源复制 | CDRC 有界 | CDRC 多来源 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in table:report.append(f"| {r['scene']} | {r['split']} | {r['far_targets']} / {r['target_views']} | "+' | '.join(fmt(r[k+'_rmse']) for k in METHODS)+' |')
    report+=['','## 训练侧支持与准入','',
             '| 场景 | 有效远景特征比例 | 稀疏覆盖格比例 | 来源透射率优势中位数 | 反演 J 越界通道比例 | 支持 / 透射率 / 预测准入 |',
             '|---|---:|---:|---:|---:|---|']
    for r in table:
        if r['split']!='train':continue
        a=summaries[r['scene']]['admission']
        report.append(f"| {r['scene']} | {r['far_supported_fraction']:.2%} | {r['far_supported_cell_fraction']:.4%} | {fmt(r['t_advantage'],3)} | {fmt(None if r['inverse_J_outside_fraction'] is None else 100*r['inverse_J_outside_fraction'],2)}% | "+' / '.join('通过' if a[k] else '未通过' for k in ('support_pass','transmission_pass','prediction_pass'))+' |')
    report+=['','## 测量与证据边界','',
             '1. 单共享表面颜色使用 S=sum(w_i*t_i)=A*t；不把归一化透射率 t 当成实际合成质量。对照同时保留 t-only、无界反演、有界反演、多来源有界最小二乘。',
             '2. 每个训练图像对均检查；测试目标只使用训练图像来源。SIFT 双向 ratio + mutual、双向深度/重投影、局部 ZNCC 检查后，要求深度比 >=1.25。所有阈值在正式结果前固定。',
             '3. 主统计按较近来源预测远景。源由深度选择，不按 RGB 误差选最佳。train/eval、远中近、深度比桶及透射率优势桶完整保存在 prediction_slices.json/csv。',
             '4. 特征点和插值渲染图的采样偏差能改变方法排序。smoke 发现该问题后，正式实验改为原生分辨率重渲染，同位置双线性采样。88 张视图的原生输出在既有地图采样格上均精确复现，最大差异为 0。',
             '5. 训练目标虽未参与该点的来源颜色反演，却已被 checkpoint 训练使用；这不是完整留一视图重训。测试特征选择检查了测试图像纹理，因此结果是有选择的对应点诊断，不是全图无偏泛化评估。',
             '6. 深度是相机 z，不是有物理标定的光程。透射率优势独立报告。稳定几何筛选排除了大量混合层/水域；低纹理远景也可能缺少 SIFT，不能由此断言那里没有任何可用信息。',
             '7. 单共享辐射假设忽略角度外观、多层颜色混合和局部光照变化；冻结 medium 也可能错误。诊断失败否定当前可操作的伪标签/迁移机制，不能证明所有跨距离联合介质估计都不可行。',
             '8. 多来源控制只纳入与同一目标分别匹配的来源，不是已验证全闭环三维轨迹。J 越界是反演与有界颜色模型冲突，不足以单独归咎于介质或证明清晰颜色错误。',
             '9. 没有生成新的整张 CDRC clear 图像。图库显示原始 GT、underwater、clear、稀疏支持及对应局部、点预测色块。无 clear GT，不以亮度或黑区减少宣称改善。',
             '','## 资源与可复核性','',
             '| 场景 | GPU | 诊断总耗时(s) | 原生采样阶段(s) | 峰值 CUDA 分配(GB) |',
             '|---|---:|---:|---:|---:|']
    for s in SCENES:
        r=summaries[s];n=r['native_audit'];report.append(f"| {s} | {r['cuda_visible_devices']} | {r['seconds']:.1f} | {n['seconds']:.1f} | {n['peak_cuda_allocated_gb']:.2f} |")
    report+=['','耗时包含初始化、特征、匹配、可视化、哈希等，属于诊断开销，不能解释为新增训练模块成本。正式输入前后哈希一致；汇总再次独立重算 CSV 指标、核验 checkpoint 和源码。',
             '','## 决策','',
             ('准入规则通过，下一步才可设计小规模严格配对训练。' if admitted>=3 else '当前 CDRC 未达到训练准入规则，不启动该机制训练，不通过放宽支持条件、改曝光或新增 Gaussian 补救。'),
             '','[全部可视化](index.html) · [四场景拼接](visualizations/four_scene_overview.png) · [指标 CSV](far_prediction_summary.csv) · [独立验证](verification.json)',
             '','逐场景原始 pair_matches.csv、correspondences.csv、prediction_rows.csv、coverage.csv、native_keypoint_samples.npz 和输入哈希均保留在上一级场景目录。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    allhash={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_sha256.json'}
    (out/'artifact_sha256.json').write_text(json.dumps(allhash,indent=2)+'\n')
    print(json.dumps(dict(completed=True,admitted_scenes=admitted,report=str(out/'REPORT.md'),gallery=str(out/'index.html'),figures=len(gallery)),indent=2))

if __name__=='__main__':main()
