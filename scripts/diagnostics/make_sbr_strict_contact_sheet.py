from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
SCENES=['Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama']
BASE={'Curasao':ROOT/'renders/bnd_on_cica_ablation_extra3_20260914/Curasao/step_14999',
      'IUI3-RedSea':ROOT/'renders/bnd_on_cica_ablation_extra3_20260914/IUI3-RedSea/step_14999',
      'JapaneseGradens-RedSea':ROOT/'renders/taba_paired_baseline_20260910/bnd_on/JapaneseGradens-RedSea/step_14999',
      'Panama':ROOT/'renders/taba_paired_baseline_20260910/bnd_on/Panama/step_14999'}
SBR=ROOT/'renders/direct_mdrr_cica_four_scene_20260903'
def pick(root,scene):
 target=root/scene if (root/scene).exists() else root
 p=next(target.glob('*/A2/underwater.png'),None)
 if p is None: p=next(target.glob('*/*/A2/underwater.png'))
 return p
def main():
 out=ROOT/'outputs/sbr_bnd_on_four_scene_strict_20260915/visualizations'; out.mkdir(parents=True,exist_ok=True)
 w,h=360,250; canvas=Image.new('RGB',(3*w,4*h),'#111'); d=ImageDraw.Draw(canvas)
 for i,s in enumerate(SCENES):
  a=Image.open(pick(BASE[s],s)).convert('RGB'); b=Image.open(pick(SBR,s)).convert('RGB').resize(a.size)
  a=a.resize((w,h-28)); b=b.resize((w,h-28)); diff=ImageEnhance.Brightness(ImageChops.difference(a,b)).enhance(4)
  for j,(im,label) in enumerate([(a,'BND ON+CICA'),(b,'SBR strict'),(diff,'4x abs diff')]):
   x,y=j*w,i*h; d.text((x+6,y+7),f'{s} | {label}',fill='white'); canvas.paste(im,(x,y+28))
 canvas.save(out/'sbr_strict_vs_bnd_on_cica_underwater.jpg',quality=95)
if __name__=='__main__': main()
