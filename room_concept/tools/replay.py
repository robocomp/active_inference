import csv, numpy as np, sys
class P:
    def __init__(s,p0,q): s.x=0.0; s.p=p0; s.q=q; s.n=0
    def upd(s,h,resid,r):
        s.p+=s.q; S=h*s.p*h+max(r,1e-12)
        if S<=0: return
        k=s.p*h/S; s.x+=k*resid; s.p=max((1-k*h)*s.p,0.0); s.n+=1
for path in sys.argv[1:]:
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rows.append({k:float(v) for k,v in r.items()})
            except: pass
    if 'dx_local' not in rows[0]: print(f"{path}: no dx_local, skip"); continue
    K=['ts_ms','est_x','est_y','est_theta','iters','pred_x','pred_y','pred_theta',
       'dx_local','dy_local','imu_dtheta','wheel_dtheta','cov_tt','sdf_mse']
    d=np.array([[r[k] for k in K] for r in rows])
    ts,ex,ey,eth,it,px,py,pth,dxl,dyl,imu,wh,covtt,mse=d.T
    wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
    c=np.cos(eth); sn=np.sin(eth)
    RX=ex-px; RY=ey-py
    r_fwd=-RX*sn+RY*c          # forward axis is theta+90 deg
    r_lat= RX*c +RY*sn
    r_th =wrap(eth-pth)
    dth=imu+wh
    yaw=P(1e-4,1e-9); kv=P(4e-4,1e-9); kw=P(4e-4,1e-9)
    corr=it>0
    A=dict(f=0.,l=0.,t=0.,rf=0.,rl=0.,rt=0.,pv=0.,tv=0.)
    prev=False; eps=[]
    for i in range(len(ts)):
        A['f']+=dyl[i]; A['l']+=dxl[i]; A['t']+=dth[i]
        if corr[i]:
            A['rf']+=r_fwd[i]; A['rl']+=r_lat[i]; A['rt']+=r_th[i]
            A['tv']=max(A['tv'],covtt[i] if covtt[i]>0 else 0.)
        if prev and not corr[i]:
            rpos=max(A['tv']*0+1e-4,1e-6)     # no pos-cov column; use a fixed floor for the replay
            rth =max(A['tv'],1e-6)
            kv.upd(A['f'], A['rf'], rpos)
            yaw.upd(-A['f'], A['rl'], rpos)
            kw.upd(A['t'], A['rt'], rth)
            eps.append((A['f'],A['t'],A['rf'],A['rl'],A['rt'],kv.x,kw.x,yaw.x))
            A=dict(f=0.,l=0.,t=0.,rf=0.,rl=0.,rt=0.,pv=0.,tv=0.)
        prev=corr[i]
    print(f"\n===== {path}: {len(eps)} episodes =====")
    if not eps: continue
    print(f"{'d_fwd m':>9} {'d_th deg':>9} {'r_fwd mm':>9} {'r_lat mm':>9} {'r_th deg':>9} "
          f"{'k_v':>9} {'k_w':>9} {'yaw deg':>9}")
    for e in eps[-14:]:
        print(f"{e[0]:9.2f} {np.degrees(e[1]):9.1f} {e[2]*1000:9.1f} {e[3]*1000:9.1f} {np.degrees(e[4]):9.2f} "
              f"{1+e[5]:9.5f} {1+e[6]:9.5f} {np.degrees(e[7]):9.3f}")
    print(f"  FINAL: forward scale {1+kv.x:.5f} ({kv.x*100:+.2f}%)  "
          f"omega scale {1+kw.x:.5f} ({kw.x*100:+.2f}%)  yaw {np.degrees(yaw.x):+.3f} deg")
    print(f"  updates: k_v {kv.n}, k_w {kw.n}, yaw {yaw.n}")
