import sys,re,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from shapely.geometry import Polygon
exec(open('scripts/grade_campaign.py').read().split('def score')[0].split('OUT =')[0]+"\n")
svg=open('scripts/grade_campaign.py').read().split('svg = "')[1].split('"')[0]
GTW=np.array([[-float(p.split(',')[0]),float(p.split(',')[1])] for p in svg.split()])
D='run_3'; g=pd.read_csv(f'{D}/gt_error.csv',on_bad_lines='skip')
A=np.c_[g.gt_x_world,g.gt_y_world];E=np.c_[g.est_x,g.est_y];am,em=A.mean(0),E.mean(0)
U,_,Vt=np.linalg.svd((A-am).T@(E-em));R=Vt.T@U.T
gtp=(GTW-am)@R.T+em
pr=pd.read_csv(f'{D}/proto_room.csv',on_bad_lines='skip');tp=pr[pr.promoted==1].ts_ms.iloc[0]
ml=[l.strip().split(';') for l in open(f'{D}/map_layout.csv') if not l.startswith('#')]
def poly(s): return np.array([[float(c) for c in v.split(',')] for v in s.split()])
pre=[poly(m[2]) for m in ml if int(m[0])<=tp]; P=pre[-1] if pre else poly(ml[0][2])
fig,ax=plt.subplots(figsize=(10,10))
ax.fill(*gtp.T,color='#4a9',alpha=.25,label='GT')
ax.plot(*np.r_[P,P[:1]].T,'k-',lw=1.2,label='layout at promotion')
m=g.ts_ms<=tp; ax.plot(*E[m.values].T,color='#e80',lw=1,label='path to promotion'); ax.plot(*E[~m.values].T,color='#e80',lw=.5,ls=':',label='path after')
t=pr.dropna(subset=['plan_tx']); ax.plot(t.plan_tx,t.plan_ty,'bx',ms=8,label='plan targets')
ax.plot(0,0,'bo');ax.set_aspect('equal');ax.grid(alpha=.3);ax.legend()
plt.savefig('run3_diag.png',dpi=70)
inv=lambda q:(np.atleast_2d(q)-em)@R+am
print('layout bottom-left/right corners world:'); print(inv(P[np.argsort(P[:,1])[:4]]).round(2))
print('start world', inv([0,0]).round(2))
