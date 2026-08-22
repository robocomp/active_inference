#!/usr/bin/env python3
"""A/B the command-velocity prior. usage: ab.py A.csv B.csv"""
import csv, sys, numpy as np

def load(p):
    rows=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            try: rows.append({k:float(v) for k,v in r.items()})
            except: pass
    K=['ts_ms','gt_x','gt_y','gt_theta','est_x','est_y','est_theta','sdf_mse','iters','imu_dtheta','wheel_dtheta']
    d=np.array([[r[k] for k in K] for r in rows])
    return d.T

def analyse(p,label):
    ts,gx,gy,gth,ex,ey,eth,mse,it,imu,wh=load(p)
    t=(ts-ts[0])/1000.0
    chg=np.r_[True,(np.diff(gx)!=0)|(np.diff(gy)!=0)|(np.diff(gth)!=0)]
    GX=np.interp(t,t[chg],gx[chg]); GY=np.interp(t,t[chg],gy[chg])
    GT=np.interp(t,t[chg],np.unwrap(gth)[chg]); ET=np.unwrap(eth)
    z=it==0
    runs=[];i=0
    while i<len(z):
        if z[i]:
            j=i
            while j<len(z) and z[j]: j+=1
            if j-i>=5: runs.append((i,j))
            i=j
        else: i+=1
    g=np.array([GT[b-1]-GT[a] for a,b in runs])
    o=np.array([imu[a:b].sum()+wh[a:b].sum() for a,b in runs])
    e=np.array([ET[b-1]-ET[a] for a,b in runs])
    L=np.array([np.hypot(np.diff(GX[a:b]),np.diff(GY[a:b])).sum() for a,b in runs])
    Le=np.array([np.hypot(np.diff(ex[a:b]),np.diff(ey[a:b])).sum() for a,b in runs])
    dur=np.array([t[b-1]-t[a] for a,b in runs])
    gain=lambda x,y: abs((x@y)/(x@x))
    m=np.abs(g)>0.15; mt=L>0.10
    print(f"\n===== {label} =====")
    print(f" {len(t)} rows, {t[-1]:.0f} s   early-exit {z.mean()*100:.1f}%   optimizer fired {(~z).sum()} times"
          f" ({(~z).sum()/t[-1]*60:.1f}/min)")
    print(f" {m.sum()} runs, {np.degrees(np.abs(g[m]).sum()):.0f} deg open-loop rotation, {L[mt].sum():.1f} m path")
    if m.sum()>2:
        print(f"   odometry prior vs GT      {(gain(g[m],o[m])-1)*100:+.2f} %")
        print(f"   PUBLISHED pose vs GT      {(gain(g[m],e[m])-1)*100:+.2f} %   <-- the sawtooth driver")
        print(f"   published vs odometry     {(gain(o[m],e[m])-1)*100:+.2f} %   <-- 0 means the flag took")
    if mt.sum()>2:
        print(f"   published path vs GT path {(gain(L[mt],Le[mt])-1)*100:+.2f} %")
    print(f"   ramp duration  p50 {np.median(dur):.2f} s   p95 {np.percentile(dur,95):.2f} s")
    print(f"   pred error     p50 {np.median(mse[z]):.4f}  p95 {np.percentile(mse[z],95):.4f}  max {mse[z].max():.4f}")

for p,l in zip(sys.argv[1::2], sys.argv[2::2]): analyse(p,l)
