import re,glob,numpy as np,pandas as pd
from shapely.geometry import Polygon,box
svg=open('scripts/grade_campaign.py').read().split('svg = "')[1].split('"')[0]
GTW=Polygon([[-float(p.split(',')[0]),float(p.split(',')[1])] for p in svg.split()])
strip=GTW.intersection(box(-8.6,1.2,-8.18,7.1))  # behind the worktop fronts
for D in sorted(glob.glob('run_*'),key=lambda d:int(d[4:])):
    try:
        g=pd.read_csv(f'{D}/gt_error.csv',on_bad_lines='skip');A=np.c_[g.gt_x_world,g.gt_y_world];E=np.c_[g.est_x,g.est_y]
        am,em=A.mean(0),E.mean(0);U,_,Vt=np.linalg.svd((A-am).T@(E-em));R=Vt.T@U.T
        Q=np.array([[float(c) for c in v.split(',')] for v in re.search(r'points="([^"]+)"',open(f'{D}/learnt.svg').read()).group(1).split()]);Q[:,0]*=-1
        Qw=Polygon((Q-em)@R+am).buffer(0)
        print(D, f'worktop strip covered {Qw.intersection(strip).area/strip.area:.2f}', f'missing GT total {GTW.difference(Qw).area:.2f} m2')
    except Exception as e: print(D,'skip',e)
D='run_3';g=pd.read_csv(f'{D}/gt_error.csv',on_bad_lines='skip');A=np.c_[g.gt_x_world,g.gt_y_world];E=np.c_[g.est_x,g.est_y]
am,em=A.mean(0),E.mean(0);U,_,Vt=np.linalg.svd((A-am).T@(E-em));R=Vt.T@U.T
Q=np.array([[float(c) for c in v.split(',')] for v in re.search(r'points="([^"]+)"',open(f'{D}/learnt.svg').read()).group(1).split()]);Q[:,0]*=-1
miss=GTW.difference(Polygon((Q-em)@R+am).buffer(0))
for p in sorted(getattr(miss,'geoms',[miss]),key=lambda p:-p.area)[:5]: print(f'{p.area:.2f} m2 bounds',np.round(p.bounds,2))
print('visited x range', A[:,0].min().round(2),A[:,0].max().round(2),'y', A[:,1].min().round(2),A[:,1].max().round(2))
