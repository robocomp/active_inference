import re,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
svg=open('scripts/grade_campaign.py').read().split('svg = "')[1].split('"')[0]
GTW=np.array([[-float(p.split(',')[0]),float(p.split(',')[1])] for p in svg.split()])
fig,axs=plt.subplots(1,3,figsize=(18,7))
for ax,r in zip(axs,[3,4,9]):
    D=f'run_{r}';g=pd.read_csv(f'{D}/gt_error.csv',on_bad_lines='skip');A=np.c_[g.gt_x_world,g.gt_y_world];E=np.c_[g.est_x,g.est_y]
    am,em=A.mean(0),E.mean(0);U,_,Vt=np.linalg.svd((A-am).T@(E-em));R=Vt.T@U.T
    Q=np.array([[float(c) for c in v.split(',')] for v in re.search(r'points="([^"]+)"',open(f'{D}/learnt.svg').read()).group(1).split()]);Q[:,0]*=-1
    Qw=(Q-em)@R+am
    ax.fill(*GTW.T,color='#4a9',alpha=.25);ax.plot(*np.r_[Qw,Qw[:1]].T,'r-');ax.plot(*A.T,color='#e80',lw=.7);ax.plot(*A[0],'bo')
    ax.set_title(D);ax.set_aspect('equal');ax.grid(alpha=.3)
plt.tight_layout();plt.savefig('run3_vs_4_9_world.png',dpi=60)
