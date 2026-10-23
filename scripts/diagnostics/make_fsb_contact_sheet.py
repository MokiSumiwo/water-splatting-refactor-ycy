from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
scenes=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
def first(root,scene):
 cand=list((root/scene/'step_14999').glob('*/A2/underwater.png')) or list((root/scene/'step_14999').glob('*/*/underwater.png'))
 p=cand[0]
 return p
def main():
 out=ROOT/'outputs/fsb_bnd_on_four_scene_20260914_v2/visualizations'; out.mkdir(parents=True,exist_ok=True)
 cols=['BND ON+CICA','FSB headroom','Difference']
 W,H=360,250; canvas=Image.new('RGB',(len(cols)*W,len(scenes)*H),'#111'); d=ImageDraw.Draw(canvas)
 for i,s in enumerate(scenes):
  base_root=ROOT/'renders/bnd_on_cica_ablation_extra3_20260914'
  if not (base_root/s).exists(): base_root=ROOT/'renders/bnd_on_cica_ablation_20260914'
  base=first(base_root,s)
  fsb=first(ROOT/'renders/fsb_bnd_on_four_scene_20260914',s)
  a=Image.open(base).convert('RGB').resize((W,H-28)); b=Image.open(fsb).convert('RGB').resize((W,H-28))
  import PIL.ImageChops as C
  diff=C.difference(a,b)
  for j,(im,label) in enumerate([(a,cols[0]),(b,cols[1]),(diff,cols[2])]):
   canvas.paste(im,(j*W,i*H+28)); d.text((j*W+6,i*H+7),f'{s} | {label}',fill='white')
 canvas.save(out/'fsb_vs_bnd_on_cica_underwater.jpg',quality=95)
if __name__=='__main__': main()
