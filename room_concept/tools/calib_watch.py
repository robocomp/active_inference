#!/usr/bin/env python3
"""Watch the online motion-calibration parameters converge.

usage: calib_watch.py [gt_error.csv]

Prints one line per closed episode: the motion that drove it, the correction the optimizer
applied, and where each parameter moved to. A parameter that never moves means H was ~0 (no
informative motion) or the feature is off -- both look identical from outside, so `episodes`
is printed to tell them apart.
"""
import csv, sys, numpy as np
p = sys.argv[1] if len(sys.argv) > 1 else "tmp/sdf_localizer/gt_error.csv"
rows = []
with open(p) as f:
    for r in csv.DictReader(f):
        try: rows.append({k: float(v) for k, v in r.items()})
        except ValueError: pass
need = ['ts_ms','iters','dy_local','dx_local','imu_dtheta','wheel_dtheta',
        'calib_k_v','calib_k_w','calib_yaw','calib_eps','sdf_mse','est_x','est_y','est_theta',
        'pred_x','pred_y','pred_theta']
missing = [k for k in need if k not in rows[0]]
if missing:
    sys.exit(f"missing columns {missing} — this CSV predates the calibrator")
d = np.array([[r[k] for k in need] for r in rows])
ts,it,dyl,dxl,imu,wh,kv,kw,yaw,eps,mse,ex,ey,eth,px,py,pth = d.T
t = (ts - ts[0]) / 1000.0
print(f"{len(t)} rows, {t[-1]:.0f} s   optimizer cycles {(it>0).sum()}   episodes closed {int(eps.max())}")
if eps.max() == 0:
    print("no episodes yet: drive further, or the feature is off (k_v/k_w exactly 1.0 and yaw 0.0)")
    print(f"  current: k_v={kv[-1]:.6f}  k_w={kw[-1]:.6f}  yaw={np.degrees(yaw[-1]):+.4f} deg")
    sys.exit()
wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
c, sn = np.cos(eth), np.sin(eth)
RX, RY = ex - px, ey - py
r_fwd = -RX*sn + RY*c          # forward axis is theta+90 deg on this robot
r_lat =  RX*c  + RY*sn
r_th  = wrap(eth - pth)
print(f"\n{'t s':>7} {'d_fwd m':>8} {'d_th deg':>9} {'r_fwd mm':>9} {'r_lat mm':>9} {'r_th deg':>9}"
      f" {'k_v':>9} {'k_w':>9} {'yaw deg':>8}")
acc = np.zeros(5); prev = False; last = 0
for i in range(len(t)):
    acc[0] += dyl[i]; acc[1] += imu[i] + wh[i]
    if it[i] > 0:
        acc[2] += r_fwd[i]; acc[3] += r_lat[i]; acc[4] += r_th[i]
    if prev and it[i] == 0:
        if int(eps[i]) != last:
            print(f"{t[i]:7.1f} {acc[0]:8.2f} {np.degrees(acc[1]):9.1f} {acc[2]*1000:9.1f}"
                  f" {acc[3]*1000:9.1f} {np.degrees(acc[4]):9.2f}"
                  f" {kv[i]:9.5f} {kw[i]:9.5f} {np.degrees(yaw[i]):+8.3f}")
            last = int(eps[i])
        acc = np.zeros(5)
    prev = it[i] > 0
print(f"\nFINAL  k_v {kv[-1]:.5f} ({(kv[-1]-1)*100:+.2f}%)   k_w {kw[-1]:.5f} ({(kw[-1]-1)*100:+.2f}%)"
      f"   yaw {np.degrees(yaw[-1]):+.3f} deg   over {int(eps[-1])} episodes")
# ── The metric ───────────────────────────────────────────────────────────────────────────────
# Two figures of merit, both normalised by MOTION so they cannot be improved by standing still:
#   * integral of the prediction error over distance driven -- the area under the sawtooth, which is
#     what "the predictor is wrong between corrections" actually costs.
#   * optimizations per metre -- how often the gate had to be rescued.
# A p50 of the error is the wrong statistic: it is dominated by parked cycles, where the predictor
# predicts nothing and therefore always looks accurate.
dt = np.diff(t, prepend=t[0])
dist = np.abs(dyl)
tot_d = dist.sum()
if tot_d > 1.0:
    # PER WINDOW, never halves. A single pathological burst (the optimizer pinned near 100% for a
    # few minutes, seen twice now) smeared across a half reads as a trend and is not one. Windows
    # make a burst look like what it is: two bad rows among good ones.
    print(f"\n{'window':>13} {'m driven':>9} {'opt/m':>8} {'err/m':>8} {'k_v':>9} {'k_w':>9} {'yaw deg':>8} {'eps':>5}")
    W = 60.0
    edges = np.arange(0, t[-1] + W, W)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (t >= a_) & (t < b_)
        if m.sum() < 30: continue
        dd = dist[m].sum()
        if dd < 0.3:
            print(f"{a_:6.0f}-{b_:5.0f}s {dd:9.2f}   (parked)                 "
                  f"{kv[m][-1]:9.5f} {kw[m][-1]:9.5f} {np.degrees(yaw[m][-1]):+8.3f} {int(eps[m][-1]):5d}")
            continue
        print(f"{a_:6.0f}-{b_:5.0f}s {dd:9.2f} {(it[m]>0).sum()/dd:8.2f} "
              f"{(mse[m]*dist[m]).sum()/dd:8.4f} {kv[m][-1]:9.5f} {kw[m][-1]:9.5f} "
              f"{np.degrees(yaw[m][-1]):+8.3f} {int(eps[m][-1]):5d}")
    print(f"  OVERALL {tot_d:8.2f} m   opt/m {(it>0).sum()/tot_d:.2f}"
          f"   err/m {(mse*dist).sum()/tot_d:.4f}")

