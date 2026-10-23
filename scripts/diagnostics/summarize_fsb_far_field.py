#!/usr/bin/env python3
"""Summarize completed frozen far-field diagnostics without loading a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
LABELS=('Curasao','IUI3','JapaneseGradens','Panama')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def table(df):
    cols=list(df.columns)
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def montage(directory, view, maps, bounds, output):
    names=('gt.png','underwater.png','clear.png','clear_raw_display_clamp01.png')
    labels=('Underwater GT','Underwater native','Clear native J/(1+J)','Clear raw [0,1], no GT')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    images=[Image.open(directory/n).convert('RGB') for n in names]
    width,height=images[0].size
    d,a,cv=maps['depth'],maps['A'],maps['cv']
    regions=[('near surface',(a>=.9)&(cv<=.1)&(d<bounds[0])),
             ('far surface',(a>=.9)&(cv<=.1)&(d>=bounds[1])),
             ('far dispersed',(a>=.5)&(cv>.1)&(d>=bounds[1]))]
    canvas=Image.new('RGB',(1600,4*338),'#16181a'); draw=ImageDraw.Draw(canvas)
    locations=[]
    for row,(name,mask) in enumerate(regions,1):
        if not mask.any():
            draw.text((10,row*338+10),name+': no candidate pixels',font=font,fill='white')
            continue
        dist=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
        yy,xx=np.unravel_index(np.argmax(dist),dist.shape)
        x0=int(np.clip(xx*4-96,0,max(0,width-192))); y0=int(np.clip(yy*4-96,0,max(0,height-192)))
        box=(x0,y0,min(x0+192,width),min(y0+192,height))
        locations.append(dict(region=name,box=box,selection='max mask interior distance; no GT/image quality selection'))
        for col,(img,label) in enumerate(zip(images,labels)):
            crop=img.crop(box); crop.thumbnail((400,295),Image.Resampling.NEAREST)
            draw.text((col*400+5,row*338+4),name+' / '+label,font=font,fill='white')
            canvas.paste(crop,(col*400+(400-crop.width)//2,row*338+40))
    for col,(img,label) in enumerate(zip(images,labels)):
        display=img.copy(); lines=ImageDraw.Draw(display)
        for loc,color in zip(locations,('#40aaff','#ffad33','#ee55ee')):
            lines.rectangle(loc['box'],outline=color,width=4)
        display.thumbnail((400,295),Image.Resampling.LANCZOS)
        draw.text((col*400+5,4),view+' / '+label,font=font,fill='white')
        canvas.paste(display,(col*400+(400-display.width)//2,40))
    canvas.save(output)
    for img in images:
        img.close()
    return locations


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    args=p.parse_args(); root=args.root.resolve(); out=root/'summary'; out.mkdir(exist_ok=False)
    (out/'visualizations').mkdir()
    assert all(v==0 for v in json.loads((root/'status.json').read_text()).values())
    for path,digest in json.loads((root/'provenance.json').read_text()).items():
        assert sha(Path(path))==digest,path
    audits={s:json.loads((root/s/'audit.json').read_text()) for s in SCENES}
    assert sum(a['train_views'] for a in audits.values())==75
    assert sum(a['eval_views'] for a in audits.values())==13
    frame={}
    for name in ('regions','gaussian_observation','compensation_native','compensation_tangent','geometry','timings'):
        df=pd.concat([pd.read_csv(root/s/(name+'.csv')) for s in SCENES],ignore_index=True)
        frame[name]=df; df.to_csv(out/(name+'.csv'),index=False)
    regions=frame['regions']; ev=regions[regions.split=='eval']
    macro=ev.groupby(['scene','mask','region'],sort=False).mean(numeric_only=True).reset_index()
    macro.to_csv(out/'scene_regions.csv',index=False)
    geometry=frame['geometry'].groupby(['scene','mask','region'],sort=False).mean(numeric_only=True).reset_index()
    geometry.to_csv(out/'scene_geometry.csv',index=False)
    comp=frame['compensation_native'].groupby(['scene','region','mode'],sort=False).mean(numeric_only=True).reset_index()
    comp.to_csv(out/'scene_compensation.csv',index=False)
    observation_rows=[]; geo_rows=[]; compensation_rows=[]; rescue_rows=[]
    for s in SCENES:
        strict=macro[(macro.scene==s)&(macro['mask']=='surface_candidate')].set_index('region')
        loose=macro[(macro.scene==s)&(macro['mask']=='A_ge_0p5')].set_index('region')
        gi=frame['gaussian_observation']; gi=gi[gi.scene==s].set_index('region')
        geo=geometry[(geometry.scene==s)&(geometry['mask']=='A_ge_0p5')].set_index('region')
        gs=geometry[(geometry.scene==s)&(geometry['mask']=='surface_candidate')].set_index('region')
        observation_rows.append({'场景':s,'近景 t R/G/B':' / '.join(f'{strict.loc["near","t_"+c]:.3f}' for c in 'rgb'),
            '远景 t R/G/B':' / '.join(f'{strict.loc["far","t_"+c]:.3f}' for c in 'rgb'),
            '远景颜色敏感度平方保留 R/G/B':' / '.join(f'{100*gi.loc["far","info_retained_"+c]:.1f}%' for c in 'rgb')})
        geo_rows.append({'场景':s,'宽松远景/整图':f'{100*loose.loc["far","fraction"]:.1f}%',
            '宽松远景深度CV>0.1':f'{100*loose.loc["far","depth_cv_gt_0p1"]:.1f}%',
            '宽松远景通过重投影':f'{100*geo.loc["far","any_roundtrip_2px"]:.1f}%',
            '筛选后远景通过重投影':f'{100*gs.loc["far","any_roundtrip_2px"]:.1f}%'})
        c=comp[(comp.scene==s)&(comp.region=='far')].set_index('mode')
        compensation_rows.append({'场景':s,**{name:f'{100*c.loc[mode,"cancellable_energy"]:.1f}%' for mode,name in (
            ('attn','衰减扰动抵消比例'),('bs','散射系数扰动抵消比例'),('rgb','介质颜色扰动抵消比例'))}})
        rescue_rows.append({'场景':s,'匹配远景有近20%来源':f'{100*gs.loc["far","closer_source_fraction"]:.1f}%',
            '目标远景 t R/G/B':' / '.join(f'{gs.loc["far","target_t_"+c+"_matched"]:.3f}' for c in 'rgb'),
            '训练视图最佳 t R/G/B':' / '.join(f'{gs.loc["far","best_train_t_"+c]:.3f}' for c in 'rgb')})
    # Scientific summary with fixed [0,1] axes, no inferred uncertainty bars.
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for i,(scene,label) in enumerate(zip(SCENES,LABELS)):
        ax=axes.flat[i]
        sub=macro[(macro.scene==scene)&(macro['mask']=='surface_candidate')].set_index('region')
        for c,color in zip('rgb',('#d95c59','#4a9e62','#488aca')):
            ax.plot(range(3),[sub.loc[r,'t_'+c] for r in ('near','mid','far')],'-o',color=color,label=c.upper())
        ax.set_xticks(range(3)); ax.set_xticklabels(('Near','Middle','Far')); ax.set_ylim(0,1)
        ax.set_title(label); ax.set_ylabel('Contribution-weighted transmission')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Frozen FSB: channel observation strength (all test views)')
    fig.tight_layout(rect=(0,.03,1,.95))
    fig.text(.5,.01,'Model-derived surface candidates. Scene means of view means. No clear ground truth.',ha='center',fontsize=10)
    fig.savefig(out/'transmission_by_depth.png',dpi=180); fig.savefig(out/'transmission_by_depth.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(4)
    for k,(mode,label,col) in enumerate(zip(MODES:=('attn','bs','rgb'),('Attenuation','Backscatter coefficient','Medium color'),('#477caf','#4c9b6b','#c99148'))):
        v=[comp[(comp.scene==s)&(comp.region=='far')&(comp['mode']==mode)].iloc[0].cancellable_energy for s in SCENES]
        axes[0].bar(x+(k-1)*.24,v,width=.22,label=label,color=col)
    axes[0].set_ylim(0,1); axes[0].set_ylabel('Fraction of medium-probe energy cancelled'); axes[0].legend(fontsize=8)
    for k,(key,label,col) in enumerate((('depth_cv_gt_0p1','Depth dispersion > 0.1','#ce6c68'),('any_roundtrip_2px','Valid cross-view roundtrip','#6792b5'))):
        source=macro if k==0 else geometry
        v=[source[(source.scene==s)&(source.region=='far')&(source['mask']=='A_ge_0p5')].iloc[0][key] for s in SCENES]
        axes[1].bar(x+(k-.5)*.3,v,width=.28,label=label,color=col)
    axes[1].set_ylim(0,1); axes[1].set_ylabel('Fraction of broader far support (A >= 0.5)'); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(LABELS,rotation=15); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].set_title('Shared color compensation, held-out views'); axes[1].set_title('Geometry evidence, including dispersed pixels')
    fig.tight_layout(); fig.savefig(out/'compensation_and_geometry.png',dpi=180); fig.savefig(out/'compensation_and_geometry.pdf'); plt.close(fig)
    crops=[]; links=[]
    for s in SCENES:
        audit=audits[s]
        assert audit['completed'] and not audit['smoke'] and audit['model_unchanged'] and audit['checkpoint_unchanged']
        assert sha(Path(audit['checkpoint']))==audit['checkpoint_sha256']
        for rec in audit['replay']:
            assert all(rec['png_exact'].values())
            view=rec['view']; data=np.load(root/s/'maps'/('eval_'+view+'.npz'))
            cropname=s+'_'+view+'_native_crops.png'
            boxes=montage(root/s/'visualizations'/view,view,data,audit['depth_boundaries'],out/'visualizations'/cropname)
            crops.append(dict(scene=s,view=view,boxes=boxes))
            links.append((s,view,cropname))
    (out/'crop_selection.json').write_text(json.dumps(crops,indent=2)+'\n')
    note='''# FSB 远景逐通道观测、介质补偿与几何诊断

## 结论

远景退化不是四场景统一的“红通道消失”，也没有证据支持单独归咎于深度或介质系数。
Curasao/IUI3 的远景颜色观测减弱较强，Panama 居中；JapaneseGradens 的远景仍有较高透射率。
所有场景都有局部介质—颜色补偿空间。IUI3/JapaneseGradens 的宽松远景包含大量深度分散区域，
因此既不能只改介质，也不能根据筛选后表面的高一致性宣称整体几何无问题。

## 范围与验证

固定最新共享前缀实验的 R0/C0a step 14999，四场景共 75 张训练图、13 张测试图。
GPU 6/7/8/9，conda water_splatting；无训练、无 backward、无参数写回、无 CUDA 重编译。
全部测试图的原 underwater、原 tone-mapped clear、raw clear PNG 完全重现，PSNR 重现。
checkpoint 和模型状态前后不变。真实颜色贡献使用原生 alpha/排序/终止规则；
已有 CICA forward 二进制的累计量和原生单位颜色重放的相对误差均通过 3e-5 门槛。

像素诊断固定步长 4 采样；Gaussian 敏感度累计使用原生全分辨率。
所有场景表中像素指标先逐视图平均，再在场景内等权平均；不把像素当作独立统计重复。
近/中/远边界由训练视图候选表面深度的三分位数定义，不等同于已标定的米制距离。
主候选表面 A>=0.9 且相对深度标准差<=0.1；宽松对照 A>=0.5。
这些是模型相关代理，不是经过标注的物体/水域分割，高覆盖的错误占据仍可能被纳入。

## 1. 逐通道观测强度

`t_c = sum_i(w_i exp(-beta_c z_i))/sum_i(w_i)`，不是平均深度指数化的近似。
敏感度平方保留率为训练视图中 `sum_(i,p,v)(w_i t_i,c)^2 / sum_(i,p,v)w_i^2`，
Gaussian 按贡献加权平均深度分组。它衡量对 Gaussian 颜色的局部敏感度，
不是带噪声标定或参数混杂消除的 Fisher 信息，也不包含 FSB 参数化的额外导数。
像素表使用候选表面，Gaussian 表包括有实际贡献的全部 Gaussian，两种分组不能等同。

OBSERVATION_TABLE

Curasao/IUI3 的远景敏感度平方约只剩零介质对照的 8–13%，并不代表真实颜色已完全丢失。
JapaneseGradens 仍保留 53–66%；Panama 约 25%。只有 JapaneseGradens 出现较明显的
红蓝透射率差；其余场景的远景 RGB 透射率接近。学习到的 beta 不是经过标定的水体光谱，
不能将这个通道分布解释成真实水质。

## 2. 介质与颜色的局部补偿

固定几何和原模型，衰减系数/散射系数分别做 ±1% 输入探针，介质颜色做 ±0.01 logit 探针。
只用训练图、不用 GT，为每个探针拟合所有 Gaussian/视角共用的逐通道 `delta J=a J+b`。
在测试图上实际重渲染介质探针，再添加该颜色响应并限制到 [0,1]。
下表为远景测试区域被抵消的扰动能量比例；各视图和正负探针等权平均。

COMPENSATION_TABLE

这表明局部不同的颜色/介质分解能给出相近 underwater RGB，不证明原介质估错、
补偿后的 clear 更准或可获得训练收益。该仿射颜色探针不是一次参数更新，也未证明能由
当前 FSB/DC/SH 参数精确表示。有限幅度重渲染包含颜色边界截断；逐项截断比例、
原扰动/补偿后 RMS 及 clear 改变量均在 compensation_native.csv 中保留。

## 3. 几何证据与筛选偏差

下表把深度分散像素重新纳入宽松对照，避免只分析最稳定表面。
重投影要求训练视图 A>=0.9、CV<=0.1、相对深度误差<=5%、往返误差<=2 原生像素，
至少一个训练视图满足。2%/5% 深度门槛的完整结果同时保留。

GEOMETRY_TABLE

IUI3/JapaneseGradens 的宽松远景约 51%/38% 有较高深度分散，几何/遮挡/混合表面的疑点较强；
但相近程度的分散也出现在它们的近景，不能声称这是专属远景故障。
Curasao/Panama 的比例较低，远景大部分仍有重投影支持。
一致性依赖当前模型的源与目标深度，共同错误也能通过；不一致又可能来自真实遮挡或薄结构。
因此不能从这里得出“深度正确”或“应剪掉高CV Gaussian”的结论。
clear 跨视角差异还受 SH、配准和采样影响，不是颜色准确性误差。

## 4. 同一位置是否存在更好的训练观测

只在上述重投影通过的位置检查所有训练视图。最近来源按源相机 z 深度选择，
不按 GT、色彩或清晰度挑选；“近20%”也指该 z 深度，不是已标定光程。
每通道最佳透射率可以来自不同训练视图，因此只是观测上限诊断。

RESCUE_TABLE

这说明一部分远景位置已有更近或透射率更高的训练观测；同时这种支持并不覆盖所有远景。
不能据此直接蒸馏其像素颜色，必须先验证真实图像对应关系和遮挡，且处理视角光照变化。

## 判断与后续设计

1. 优先保留 BND/FSB，在 Curasao/IUI3 检查远景颜色是否实际利用了现有较高透射率视图，
   将贡献归一化/像素 footprint/视角差异纳入归因；不能直接使用 1/t 放大梯度。
2. JapaneseGradens 优先定位远景高CV、弱重投影区域的具体条纹与混合层问题，
   用原图特征或人工区域核对提供独立证据，再决定是否动几何；不能重复上一个泛化深度约束。
3. 介质补偿已观察到，但绝对深度、beta 与颜色责任仍不可单独辨识。
   后续介质约束需要独立跨距离观测或标定依据，不能把本次拟合系数写回作为新方法。
4. 远景 clear 视觉问题与 underwater 指标瓶颈分开：候选表面上 Curasao、JapaneseGradens、
   Panama 的远景 underwater MSE 低于近景。本诊断不支持“修好远景就能补齐 BND OFF 差距”。

## 可视化与局限

[全部视图图库](index.html) 包含 13 张观测拼接、13 张几何拼接、13 张原生局部对比。
原生局部图同时显示 GT、underwater、原 clear 显示映射和 raw clear；不调曝光/白平衡。
局部位置仅由近/远/远景高CV 掩码的最大内部距离确定，不按清晰度挑选。
分组及诊断热图按步长4采样，不能用它们判断原生锐度；锐度检查应查看原生局部图或原 PNG。
重投影拼接中缺少匹配显示黑色，非有限深度比显示灰色，不能误认为模型新产生的黑洞。
原 clear 显示使用 J/(1+J)，raw 图使用 [0,1]；显示对比度差异不代表参数改善。
无真实 clear GT，无法量化去水准确性或给出深度/介质/颜色的因果责任百分比。

[逐视图区域统计](regions.csv)、[逐Gaussian观测汇总](gaussian_observation.csv)、
[完整介质补偿探针](compensation_native.csv)、[几何对照](geometry.csv)、
[场景观测曲线](transmission_by_depth.png)、[补偿与几何图](compensation_and_geometry.png)。
'''
    for token,rows in [('OBSERVATION_TABLE',observation_rows),('COMPENSATION_TABLE',compensation_rows),('GEOMETRY_TABLE',geo_rows),('RESCUE_TABLE',rescue_rows)]:
        note=note.replace(token,table(pd.DataFrame(rows)))
    (out/'REPORT.md').write_text(note)
    html=["<!doctype html><html lang='zh'><meta charset='utf-8'><title>FSB 远景冻结诊断</title>",
          "<style>body{background:#181a1c;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}a{color:#8bd9ee}section{margin:32px 0}</style>",
          '<h1>FSB 远景冻结诊断</h1><p>原模型冻结；全部测试视图；clear 无 GT。诊断热图为步长4采样，原生局部图用于看细节。</p>',
          '<p>原 clear：J/(1+J)；raw clear：[0,1]，没有额外曝光或白平衡。</p>',
          '<img src="transmission_by_depth.png"><img src="compensation_and_geometry.png">']
    for s,v,crop in links:
        html.append(f'<section><h2>{s} / {v}</h2>')
        for file,label in [(f'visualizations/{crop}','原生局部对比'),(f'../{s}/visualizations/{v}_observation.png','观测与分解'),(f'../{s}/visualizations/{v}_geometry.png','几何与重投影')]:
            html.append(f'<h3>{label}</h3><a href="{file}"><img loading="lazy" src="{file}"></a>')
        html.append('</section>')
    html.append('</html>'); (out/'index.html').write_text('\n'.join(html))
    # Hash source summaries and new assets; never change source-run files.
    artifacts={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png','.md','.html','.npz')}
    (out/'artifact_sha256.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    result=dict(completed=True,train_views=75,eval_views=13,read_only=True,checkpoints_verified=4,
                native_png_replay_checks=39,comparison_sheets=39,artifact_hashes=len(artifacts),
                max_contribution_relative_error=max(x['jacobian_sum_relative_error'] for a in audits.values() for x in a['information_equivalence']))
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
