import numpy as np,pandas as pd
D='run_3';g=pd.read_csv(f'{D}/gt_error.csv',on_bad_lines='skip');A=np.c_[g.gt_x_world,g.gt_y_world];E=np.c_[g.est_x,g.est_y]
am,em=A.mean(0),E.mean(0);U,_,Vt=np.linalg.svd((A-am).T@(E-em));R=Vt.T@U.T
ml=[l.strip().split(';') for l in open(f'{D}/map_layout.csv') if not l.startswith('#')]
t0=int(ml[0][0])
for m in ml[::3]:
    P=np.array([[float(c) for c in v.split(',')] for v in m[2].split()]);W=(P-em)@R+am
    sel=(W[:,1]>3.0)&(W[:,1]<7.5); w=W[sel,0].min() if sel.any() else np.nan
    s=(W[:,0]<-5.5)&(W[:,0]>-8.6)&(W[:,1]<2.5); sy=W[s,1].min() if s.any() else np.nan
    print(f'{(int(m[0])-t0)/1000:6.1f}s  westmost x(y3..7.5)={w:6.2f}  south y(SW room)={sy:5.2f}  verts={len(P)}')
