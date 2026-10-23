"""Display paired endpoint test floats at common exposure and native resolution."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research_runs/geometry_gradient_A_20261021'
SCENES=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
def rgb(a):
 if not np.isfinite(a).all():raise ValueError('nonfinite image')
 return Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8))
def main():
 dest=BASE/'comparisons_test';dest.mkdir(exist_ok=True);manifest={}
 for scene in SCENES:
  folders={a:BASE/'assets'/scene/a for a in ['B0','A']}
  names={a:[r['view_id'] for r in csv.DictReader((folders[a]/'eval_metrics.csv').open())] for a in folders}
  assert names['B0']==names['A'] and len(names['A'])==len(set(names['A']))
  sheets={};sizes=[];row_y=0
  for i,name in enumerate(names['A']):
   with np.load(folders['B0']/'eval_float'/(name+'.npz')) as b,np.load(folders['A']/'eval_float'/(name+'.npz')) as a:
    assert np.array_equal(b['gt'],a['gt'])
    assert b['clear_raw'].shape==a['clear_raw'].shape==b['underwater'].shape==a['underwater'].shape==b['gt'].shape
    h,w=b['gt'].shape[:2];sizes.append([h,w])
    if i==0:
     sheets['clear']=Image.new('RGB',(2*w,(h+44)*len(names['A'])),(24,24,24))
     sheets['underwater']=Image.new('RGB',(3*w,(h+44)*len(names['A'])),(24,24,24))
    assert [h,w]==sizes[0], 'mixed resolution: use individual view assets'
    for key,panels in [('clear',[('B0 clear',b['clear_raw']),('A clear',a['clear_raw'])]),('underwater',[('GT underwater',b['gt']),('B0 underwater',b['underwater']),('A underwater',a['underwater'])])]:
     draw=ImageDraw.Draw(sheets[key])
     for j,(label,array) in enumerate(panels):
      draw.text((j*w+12,row_y+7),f'{scene} | TEST {name} | {label}',font=FONT,fill='white')
      sheets[key].paste(rgb(array),(j*w,row_y+44))
    row_y+=h+44
  for key,sheet in sheets.items():
   sheet.save(dest/f'{scene}_{key}_test_native.png')
   preview=sheet.copy();preview.thumbnail((2400,2400));preview.save(dest/f'{scene}_{key}_test_preview.jpg',quality=94,subsampling=0)
  manifest[scene]=dict(test_cameras=names['A'],native_view_shapes=sizes,clear_columns=['B0 clear','A clear'],underwater_columns=['GT underwater','B0 underwater','A underwater'],display='common clamp [0,1], round to uint8; no exposure/color adjustment; native PNG pixels unscaled')
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (dest/'README.md').write_text('# Test-view comparisons\n\nEach row is one test camera. Clear: B0 left, A right. Underwater: GT left, B0 middle, A right. Native PNGs preserve source resolution; preview JPGs are resized for browsing. Both arms use identical clamp [0,1], with no independent color correction.\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
