import csv, numpy as np, sys
p=sys.argv[1]
rows=[]
with open(p) as f:
    for r in csv.DictReader(f):
        try: rows.append({k:float(v) for k,v in r.items()})
        except: pass
K=['ts_ms','gt_x','gt_y','gt_theta','est_x','est_y','est_theta','sdf_mse','iters',
   'imu_dtheta','wheel_dtheta','pred_x','pred_y','pred_theta']
d=np.array([[r[k] for k in K] for r in rows])
ts,gx,gy,gth,ex,ey,eth,mse,it,imu,wh,px,py,pth=d.T
t=(ts-ts[0])/1000.0
chg=np.r_[True,(np.diff(gx)!=0)|(np.diff(gy)!=0)|(np.diff(gth)!=0)]
GX=np.interp(t,t[chg],gx[chg]); GY=np.interp(t,t[chg],gy[chg]); GT=np.interp(t,t[chg],np.unwrap(gth)[chg])
v=np.hypot(np.gradient(GX,t),np.gradient(GY,t)); w=np.gradient(GT,t)
print(f"{len(t)} rows, {t[-1]:.0f} s   moving {(v>0.15).mean()*100:.0f}%   early-exit {(it==0).mean()*100:.1f}%")

wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
z=it==0
# --- SELF-CHECK: on early-exit cycles the published pose IS the prediction
dz=np.hypot(ex[z]-px[z], ey[z]-py[z])
print(f"\nSELF-CHECK (iters==0 => est must equal pred): max |est-pred| = {dz.max()*1000:.3f} mm"
      f"  {'OK' if dz.max()<1e-4 else '*** PLUMBING WRONG ***'}")

# --- correction vector, in the ROBOT frame, summed over each optimized stretch
cx=ex-px; cy=ey-py; cth=wrap(eth-pth)
h=eth
along = cx*np.cos(h)+cy*np.sin(h)
cross =-cx*np.sin(h)+cy*np.cos(h)

teeth=[];i=0
while i<len(z):
    if z[i]:
        a=i
        while i<len(z) and z[i]: i+=1
        b=i                       # ramp = [a,b)
        c=i
        while c<len(z) and not z[c]: c+=1   # optimized stretch = [b,c)
        if b-a>=4 and c>b:
            T=t[b-1]-t[a]
            L=np.hypot(np.diff(GX[a:b]),np.diff(GY[a:b])).sum()
            rot=abs(GT[b-1]-GT[a])
            teeth.append(dict(dur=T, dist=L, rot=np.degrees(rot),
                              vmean=L/max(T,1e-3), wmean=np.degrees(rot)/max(T,1e-3),
                              rise=mse[a:b].max()-mse[a:b].min(), peak=mse[a:b].max(),
                              c_along=along[b:c].sum(), c_cross=cross[b:c].sum(),
                              c_th=np.degrees(cth[b:c].sum()), nopt=c-b))
        i=c
    else: i+=1
if not teeth: print("\nno complete teeth yet — keep driving"); sys.exit()
T=teeth
print(f"\n=== {len(T)} complete teeth (ramp -> correction) ===")
g=lambda k: np.array([x[k] for x in T])
print(f"{'v m/s':>6} {'w d/s':>6} {'dur s':>6} {'dist m':>7} {'rot d':>6} {'rise':>7} {'peak':>7}"
      f" {'ALONG mm':>9} {'CROSS mm':>9} {'dTH deg':>8}")
for x in sorted(T,key=lambda q:-q['peak'])[:18]:
    print(f"{x['vmean']:6.2f} {x['wmean']:6.1f} {x['dur']:6.2f} {x['dist']:7.2f} {x['rot']:6.1f}"
          f" {x['rise']:7.4f} {x['peak']:7.4f} {x['c_along']*1000:+9.1f} {x['c_cross']*1000:+9.1f} {x['c_th']:+8.2f}")
print(f"\n medians: |along| {np.median(np.abs(g('c_along')))*1000:.1f} mm"
      f"   |cross| {np.median(np.abs(g('c_cross')))*1000:.1f} mm"
      f"   |dtheta| {np.median(np.abs(g('c_th'))):.2f} deg")
st=(g('rot')<8)&(g('dist')>0.4)
print(f"\n STRAIGHT teeth (rot<8 deg, dist>0.4 m): n={st.sum()}")
if st.sum()>2:
    print(f"   |along| {np.median(np.abs(g('c_along')[st]))*1000:.1f} mm"
          f"   |cross| {np.median(np.abs(g('c_cross')[st]))*1000:.1f} mm"
          f"   |dtheta| {np.median(np.abs(g('c_th')[st])):.2f} deg"
          f"   rise {np.median(g('rise')[st]):.4f}")
rt=g('rot')>=15
print(f" ROTATING teeth (rot>=15 deg): n={rt.sum()}")
if rt.sum()>2:
    print(f"   |along| {np.median(np.abs(g('c_along')[rt]))*1000:.1f} mm"
          f"   |cross| {np.median(np.abs(g('c_cross')[rt]))*1000:.1f} mm"
          f"   |dtheta| {np.median(np.abs(g('c_th')[rt])):.2f} deg")
if len(T)>6:
    print("\n correlation of the SDF rise with each correction component:")
    for k in ['c_along','c_cross','c_th']:
        print(f"   {k:8s} {np.corrcoef(np.abs(g(k)),g('rise'))[0,1]:+.3f}")
    print(" correlation of |cross| with: "
          f"dist {np.corrcoef(g('dist'),np.abs(g('c_cross')))[0,1]:+.3f}  "
          f"speed {np.corrcoef(g('vmean'),np.abs(g('c_cross')))[0,1]:+.3f}  "
          f"rot {np.corrcoef(g('rot'),np.abs(g('c_cross')))[0,1]:+.3f}")
