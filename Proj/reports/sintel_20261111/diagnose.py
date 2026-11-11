"""Read-only evaluation audit: existing receipts + five deterministic 20-frame clips."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from explore_a.evaluation.session import entries, bind, cohorts, samples, forward_frame
from explore_a.evaluation.part2_protocol import restore_canvas
from explore_a.online.factory import make_runtime
from explore_a.data.reading import canvas_transform
from explore_a.evaluation.sintel_ac import image_scores
from explore_a.evaluation.alignment import moments, fit, depth

ROOT=Path(__file__).resolve().parent
PROJ=next(p for p in ROOT.parents if (p/'src/explore_a').is_dir())
EVAL=PROJ/'outputs/explore_a/shared_models_evaluation_v2'
names=['farneback_unet_guided_lr3e-4','raft_unet_guided_lr3e-4']
torch.set_num_threads(1)
bindings=[bind(e, EVAL/'models') for e in entries(PROJ/'maintenance/evaluate_models.json') if e['name'] in names]
c=copy.deepcopy(bindings[0]['config'])
c['execution']['backend']='online'
c['remote_online']['enabled']=False
runtime=make_runtime(c,['farneback','raft'])
cases=list(cohorts('benchmark',c,Path.home()/'Documents/ResourcesLow/Datasets',Path.home()/'Documents/Resources/Datasets/TUM-RGBD'))
targets=['ambush_2','market_6','alley_1','rgbd_bonn_static','2011_09_26_drive_0079_sync']
records=[]

def color(value,low,high,cmap=cv2.COLORMAP_TURBO,mask=None):
    a=np.nan_to_num((value-low)/max(high-low,1e-9)).clip(0,1)
    result=cv2.cvtColor(cv2.applyColorMap((a*255).astype(np.uint8),cmap),cv2.COLOR_BGR2RGB)
    if mask is not None: result[~mask]=0
    return result

def tile(array,label):
    im=Image.fromarray(array).convert('RGB'); im.thumbnail((320,180))
    panel=Image.new('RGB',(320,205),'white'); panel.paste(im,((320-im.width)//2,23))
    ImageDraw.Draw(panel).text((5,5),label,fill='black');return panel

try:
 for seq in targets:
    case=next(x for x in cases if x['sequence']==seq)
    n=len(case['frames']); start=max(1,(n-20)//2); stop=min(n,start+20)
    panels=[];movie=[]; items=[]
    for i,(raw,gt,valid,history) in enumerate(samples(case,c)):
      if i<start: continue
      if i>=stop: break
      raw['frozen_ticket']=runtime.prepare(raw['row'],i,{'farneback':False,'raft':False},phase='benchmark',native=True)
      with torch.inference_mode(): values=forward_frame(bindings,raw,runtime,True,history,details=True)
      rgb=raw['images'][False]; shape=gt.shape
      base=restore_canvas(values['raw_canvas'],shape,c['data']['canvas_kitti' if case['dataset']=='kitti' else 'canvas_indoor'])
      canvas=raw['canvas']
      mask=(np.isfinite(gt)&(gt>0)) if case['dataset']=='sintel' else valid
      target=np.minimum(gt,72) if case['dataset']=='sintel' else gt
      limits=(1e-5,80) if case['dataset']=='sintel' else (1e-3,case['protocol']['max_depth'])
      mapping=fit(moments(base[mask],1/target[mask].astype(np.float64)))
      db=depth(base,mapping,limits)
      eb=np.abs(db-target)/np.maximum(target,1e-6)
      sample={'sequence':seq,'dataset':case['dataset'],'index':i,'rgb':raw['row']['current_rgb'],
              'gt_valid_fraction':float(mask.mean()),'gt_ge72_fraction':float((gt[mask]>=72).mean()),
              'raw_shared_gauge_absrel':float(eb[mask].mean()),'models':{}}
      fields={}
      for name in names:
       prefix=name+'/specific_out/'
       out=restore_canvas(values[name],shape,canvas)
       h=restore_canvas(values[prefix+'historical'].squeeze(),shape,canvas)
       v=restore_canvas(values[prefix+'visibility'].squeeze(),shape,canvas)>.999
       w=restore_canvas(values[prefix+'historical_weight'].squeeze(),shape,canvas)
       ds=depth(out,mapping,limits);eh=np.abs(depth(h,mapping,limits)-target)/np.maximum(target,1e-6)
       es=np.abs(ds-target)/np.maximum(target,1e-6)
       available=mask&v
       a,b=mapping
       if a>0:
        desired=(1/np.maximum(target,1e-6)-b)/a
        oracle=np.where(v,np.clip(desired,np.minimum(base,h),np.maximum(base,h)),base)
        eo=np.abs(depth(oracle,mapping,limits)-target)/np.maximum(target,1e-6)
       else: eo=eb
       scale=float(values[prefix+'field_scale'].reshape(-1)[0])
       err0=a*base+b-1/np.maximum(target,1e-6)
       errh=a*h+b-1/np.maximum(target,1e-6)
       corr=float(np.corrcoef(err0[available],errh[available])[0,1]) if available.sum()>10 else None
       sample['models'][name]={'availability':float(v[mask].mean()),'weight_all':float(w[mask].mean()),
        'weight_available':float(w[available].mean()) if available.any() else 0,
        'normalized_change':float((np.abs(out-base)/max(scale,1e-6))[mask].mean()),
        'history_better_fraction_available':float((eh[available]<eb[available]).mean()) if available.any() else 0,
        'current_history_inverse_error_corr':corr,
        'shared_gauge_absrel':float(es[mask].mean()),'oracle_shared_gauge_absrel':float(eo[mask].mean()),
        'official_image_score':image_scores(out,gt) if case['dataset']=='sintel' else None}
       fields[name]=(ds,es,w)
      items.append(sample);records.append(sample)
      frame=Image.fromarray(rgb);frame.thumbnail((640,360));movie.append(frame)
      if i in [start,start+9,stop-1]:
       hi=float(np.percentile(target[mask],95));lo=float(np.percentile(target[mask],5))
       fb=fields[names[0]];ra=fields[names[1]]
       panels.append([tile(rgb,f'RGB frame {i}'),tile(color(target,lo,hi,mask=mask),'GT depth (P5-P95)'),
        tile(color(db,lo,hi,mask=mask),'Raw depth, baseline GT gauge'),tile(color(fb[0],lo,hi,mask=mask),'FB depth, SAME gauge'),
        tile(color(ra[0],lo,hi,mask=mask),'RAFT depth, SAME gauge'),
        tile(color(fb[1]-eb,-.02,.02,cv2.COLORMAP_COOLWARM if hasattr(cv2,'COLORMAP_COOLWARM') else cv2.COLORMAP_JET,mask),'FB error delta [-.02,+.02]'),
        tile(color(fb[2],0,1,mask=mask),'FB historical weight [0,1]')])
    sheet=Image.new('RGB',(2240,205*len(panels)),'white')
    for y,row in enumerate(panels):
      for x,t in enumerate(row):sheet.paste(t,(320*x,205*y))
    sheet.save(ROOT/f'{seq}.png')
    movie[0].save(ROOT/f'{seq}.gif',save_all=True,append_images=movie[1:],duration=100,loop=0)
    (ROOT/'sample_diagnostics.json').write_text(json.dumps(records,indent=2))
    print('DONE',seq,start,stop,flush=True)
finally: runtime.close()
summary=[]
for seq in targets:
 rows=[x for x in records if x['sequence']==seq]
 item={'sequence':seq,'dataset':rows[0]['dataset'],'start':rows[0]['index'],'end_inclusive':rows[-1]['index'],
       'frames':len(rows),'gt_ge72_fraction':float(np.mean([r['gt_ge72_fraction'] for r in rows])),
       'raw_shared_gauge_absrel':float(np.mean([r['raw_shared_gauge_absrel'] for r in rows])),'models':{}}
 for name in names:
  item['models'][name]={k:float(np.mean([r['models'][name][k] for r in rows])) for k in rows[0]['models'][name] if k!='official_image_score'}
 summary.append(item)
(ROOT/'sample_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
