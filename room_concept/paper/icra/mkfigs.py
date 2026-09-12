"""Figures for the paper. Every number is recomputed from the raw artefacts, never retyped."""
import csv, glob, math, os, sys, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,
                     'xtick.labelsize':7,'ytick.labelsize':7,'figure.dpi':200,
                     'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.4,
                     'font.family':'serif','mathtext.fontset':'cm'})
SD=sys.argv[1]; ROOT=sys.argv[2]

# ── data: matched corners (declared sigma vs realised error) ────────────────────────────────
def matched(polydir, corpus):
    truth=[]
    for line in open(corpus):
        if ';' not in line: continue
        _,v=line.split(';',1)
        truth.append([tuple(float(c) for c in p.split(',')) for p in v.split() if ',' in p])
    out=[]
    for path in sorted(glob.glob(os.path.join(polydir,'*.csv'))):
        idx=int(os.path.basename(path).rsplit('_',1)[1].split('.')[0])
        if idx>=len(truth): continue
        hdr=open(path).readline().split(); ox,oy=float(hdr[2]),float(hdr[3])
        last=None
        for ln in open(path):
            if not ln.startswith('#') and ln.strip(): last=ln
        if not last: continue
        f=last.split(';')
        try:
            verts=[(float(a.split(',')[0])+ox,float(a.split(',')[1])+oy) for a in f[3].split() if ',' in a]
            sig=[float(x) for x in f[4].split()]
        except (IndexError,ValueError): continue
        if not verts or len(sig)!=len(verts): continue
        T=truth[idx]
        C=np.array([[math.hypot(px-tx,py-ty) for tx,ty in T] for px,py in verts])
        ri,ci=linear_sum_assignment(C)
        for a,b in zip(ri,ci): out.append((sig[a],float(C[a,b])))
    return np.array(out)

A=matched(f"{SD}/calib/poly594",      f"{ROOT}/datasets/matterport_layout/corpus594.txt")  # no floor
B=matched(f"{SD}/calib/poly594_floor",f"{ROOT}/datasets/matterport_layout/corpus594.txt")  # with floor
print(f"fig1: {len(A)} corners without the floor, {len(B)} with")

# ── Fig 1: the claim, in one image ─────────────────────────────────────────────────────────
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5),sharex=True,sharey=True)
for k,(D,tag) in enumerate(((A,'propagated statistics only'),(B,r'with the irreducible term $\sigma_m$'))):
    s,e=D[:,0],D[:,1]
    ax[k].loglog(s,e,'.',ms=1.1,alpha=0.18,color='#1f4e79',rasterized=True)
    lim=np.array([2e-3,3.0])
    ax[k].loglog(lim,lim,'k-',lw=0.9,label=r'$\mathrm{error}=\sigma$')
    ax[k].loglog(lim,2*lim,'k--',lw=0.7,label=r'$\mathrm{error}=2\sigma$')
    # binned median of the realised error
    bins=np.logspace(np.log10(3e-3),np.log10(1.0),14)
    idx=np.digitize(s,bins); xs=[];ys=[]
    for i in range(1,len(bins)):
        m=idx==i
        if m.sum()>=25: xs.append(np.median(s[m])); ys.append(np.median(e[m]))
    ax[k].loglog(xs,ys,'o-',color='#c00000',ms=3,lw=1.3,label='median realised error')
    ax[k].axvline(0.06,color='#444',lw=0.8,ls=':')
    ax[k].set_xlabel(r'declared corner $\sigma$  [m]')
    ax[k].set_title(tag,fontsize=8)
    ax[k].set_xlim(3e-3,2.0); ax[k].set_ylim(1e-3,3.0)
ax[0].set_ylabel('true corner error  [m]')
ax[0].text(0.062,1.6,'publish bar',fontsize=6,color='#444',rotation=90,va='top')
ax[1].legend(loc='lower right',framealpha=0.9)
fig.tight_layout(pad=0.4); fig.savefig('fig/calibration.pdf',bbox_inches='tight'); plt.close(fig)

# ── Fig 2: NIS by motion state ─────────────────────────────────────────────────────────────
R=[r for r in csv.DictReader(open(f"{ROOT}/tmp/corner_probe.csv")) if r.get('c11_1')]
F=lambda r,k: float(r[k])
sg=lambda r: math.hypot(F(r,'sdet_xx'),0)*0+math.sqrt(max(0,(F(r,'sdet_xx')+F(r,'sdet_yy'))/2))
spd=lambda r: math.sqrt(max(0,(F(r,'sprd_xx')+F(r,'sprd_yy'))/2))
OLD_BASE, NEW_BASE = 0.04**2, 0.06**2
def d2_shipped(r):
    dd = NEW_BASE - OLD_BASE
    Sxx=F(r,'sdet_xx')+dd+F(r,'sprd_xx'); Syy=F(r,'sdet_yy')+dd+F(r,'sprd_yy')
    Sxy=F(r,'sdet_xy')+F(r,'sprd_xy'); det=Sxx*Syy-Sxy*Sxy
    if det<=1e-12: return float('nan')
    nx,ny=F(r,'nu_x'),F(r,'nu_y')
    return (Syy*nx*nx-2*Sxy*nx*ny+Sxx*ny*ny)/det
from collections import defaultdict
byf=defaultdict(dict); per=defaultdict(list)
for r in R:
    f=int(F(r,'frame')); byf[f][int(F(r,'model_index'))]=(F(r,'pred_x'),F(r,'pred_y'))
    if sg(r)<0.08 and spd(r)<0.05: per[f].append(r)
frames=sorted(byf); rows=[]
for a,b in zip(frames,frames[1:]):
    if b-a!=1 or b not in per or not per[b]: continue
    c=set(byf[a])&set(byf[b])
    if len(c)<3: continue
    P=np.array([byf[a][k] for k in c]); Q=np.array([byf[b][k] for k in c])
    H=(P-P.mean(0)).T@(Q-Q.mean(0))
    U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)); Rm=Vt.T@np.diag([1,d])@U.T
    # ⚠ The logged d2 was produced by a build carrying base_sigma = 0.04, which we measured to be
    # overconfident and replaced with the 0.06 the paper describes. Every term of S is logged
    # separately, so d2 is recomputed here for the SHIPPED model rather than quoting a superseded one.
    rows.append((abs(math.degrees(math.atan2(Rm[1,0],Rm[0,0]))),
                 st.mean(d2_shipped(x)/2 for x in per[b]),
                 st.mean(spd(x) for x in per[b]),
                 st.median([math.hypot(F(x,'nu_x'),F(x,'nu_y')) for x in per[b]])))
rows=np.array(rows)
bands=[(0,0.25),(0.25,0.5),(0.5,1.0),(1.0,2.0),(2.0,4.0)]
lab=['<0.25\n(at rest)','0.25-0.5','0.5-1','1-2','2-4\n(fast)']
fig,ax=plt.subplots(figsize=(3.4,2.2))
x=np.arange(len(bands))
nis=[];prd=[];nu=[];n=[]
for lo,hi in bands:
    m=(rows[:,0]>=lo)&(rows[:,0]<hi)
    nis.append(rows[m,1].mean()); prd.append(rows[m,2].mean()); nu.append(rows[m,3].mean()); n.append(m.sum())
ax.bar(x,nis,0.6,color='#1f4e79',label=r'$\mathrm{NIS}/\mathrm{dof}$')
ax.axhline(1.0,color='#c00000',lw=1.0,ls='--')
ax.text(len(bands)-0.4,1.08,'calibrated',fontsize=6,color='#c00000',ha='right')
for i,(v,c) in enumerate(zip(nis,n)): ax.text(i,v+0.06,f'n={c}',ha='center',fontsize=5.5)
ax2=ax.twinx(); ax2.plot(x,prd,'o-',color='#e07b00',ms=3,lw=1.2,label=r'$\sigma_{\mathrm{pred}}$ [m]')
ax2.set_ylabel(r'$\sigma_{\mathrm{pred}}$  [m]',color='#e07b00'); ax2.tick_params(axis='y',colors='#e07b00')
ax2.grid(False); ax2.set_ylim(0,max(prd)*1.35)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel(r'rotation rate  $|\Delta\theta|$  [deg/frame]')
ax.set_ylabel(r'$\mathrm{NIS}/\mathrm{dof}$'); ax.set_ylim(0,max(nis)*1.25)
fig.tight_layout(pad=0.3); fig.savefig('fig/motion_nis.pdf',bbox_inches='tight'); plt.close(fig)
print(f"fig2: {len(rows)} frame pairs; NIS by band {[round(v,2) for v in nis]}")
