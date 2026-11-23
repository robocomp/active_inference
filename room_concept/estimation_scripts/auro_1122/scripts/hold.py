import csv,sys,math,numpy as np
r=sys.argv[1]
R=list(csv.DictReader(open(f'{r}/gt_error.csv')))
t=np.array([float(x['ts_ms']) for x in R])/1000; t0=t[0]
gx=np.array([float(x['gt_x']) for x in R]); gy=np.array([float(x['gt_y']) for x in R]); gth=np.array([float(x['gt_theta']) for x in R])
ex=np.array([float(x['est_x']) for x in R]); ey=np.array([float(x['est_y']) for x in R]); eth=np.array([float(x['est_theta']) for x in R])
wh=np.array([float(x['wheel_dtheta']) for x in R]); im=np.array([float(x['imu_dtheta']) for x in R])
dt=np.diff(t)
print(r,'rows',len(t),'dur %.0f s'%(t[-1]-t0),'dt median %.3f p99 %.3f max %.2f'%(np.median(dt),np.percentile(dt,99),dt.max()))
W=float(sys.argv[2]) if len(sys.argv)>2 else 20
for a in np.arange(0,t[-1]-t0,W):
    m=(t-t0>=a)&(t-t0<a+W)
    if m.sum()<3: print('%4.0f no rows'%a); continue
    i=np.where(m)[0]
    gp=np.sum(np.hypot(np.diff(gx[i]),np.diff(gy[i]))); ep=np.sum(np.hypot(np.diff(ex[i]),np.diff(ey[i])))
    gr=np.sum(np.abs(np.angle(np.exp(1j*np.diff(gth[i]))))); er=np.sum(np.abs(np.angle(np.exp(1j*np.diff(eth[i])))))
    print('%4.0f n%3d gt path %.2f m rot %.2f | est path %.2f m rot %.2f | wheel %.2f imu %.2f | gt (%.2f,%.2f)'%(a,m.sum(),gp,gr,ep,er,np.abs(wh[i]).sum(),np.abs(im[i]).sum(),gx[i[-1]],gy[i[-1]]))
