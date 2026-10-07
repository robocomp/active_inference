#!/usr/bin/env python3
"""Is the slow pose-field bias identifiable WITHOUT ground truth? (Fable window memo §3, prototype)

The scan-only pose z_n = x_n + b(x_n) + e_n. Over a chain of consecutive cycles,
    D(n, m) = z_n - z_m - sum(odom_{m+1..n})  =  odometry error over the stretch + b(x_n) - b(x_m) + e_n - e_m
so, binned by PATH LENGTH d between n and m,
    E[D^2] = (odometry variance over d)  +  2 sigma_b^2 (1 - exp(-d / l))  +  2 s r.
GT-free route: odometry variance from the learner's per-cycle coefficients (short lags, where b cancels),
the excess over it = the bias structure function -> sigma_b, l.
GT route (sim only, the referee): the structure function of (z - gt) over the same lags.
usage: bias_field_probe.py innov.csv heading.csv
"""
import sys
import numpy as np, pandas as pd

ip, hp = sys.argv[1], sys.argv[2]
# optional coefficient overrides, e.g. k_long=4.77e-4 k_th_turn=3.4e-4 mv_th=1.08e-4 (from noise_innov_replay)
over = dict(a.split("=") for a in sys.argv[3:])
d = pd.read_csv(ip).dropna(subset=["z_x"]).reset_index(drop=True)
h = pd.read_csv(hp, comment="#")
d = d.merge(h[["ts_ms", "gt_x", "gt_y", "gt_th"]], on="ts_ms", how="left")
print(f"{len(d)} innovation rows, {d.gt_x.notna().mean():.0%} joined to GT")

# chains: consecutive cycles (gap <= 80 ms); odom in world from body (fwd = (-sin, cos), lat = (cos, sin))
gap = np.r_[np.inf, np.diff(d.ts_ms.values)]
chain = np.cumsum(gap > 80)
th = d.z_th.values - 0.5 * d.odom_th.values
ox = -np.sin(th) * d.odom_fwd + np.cos(th) * d.odom_lat
oy = np.cos(th) * d.odom_fwd + np.sin(th) * d.odom_lat
d["ox"], d["oy"] = ox, oy
d["step"] = np.hypot(ox, oy)
# predicted per-cycle ODOMETRY variance (translation x+y, heading) from the final learnt coefficients:
k = d.iloc[-1].copy()
for name in ["mv_fwd", "mv_lat", "mv_th"]:
    if name not in k: k[name] = 0.0
for kk, vv in over.items(): k[kk] = float(vv)
pm = d.p_move if "p_move" in d else pd.Series(1.0, index=d.index)
var_tr = (k.k_long + k.k_lat) * d.step + k.k_lat_turn * np.abs(d.odom_th) + 2 * k.k_t_trans * 0.05 \
         + k.q_fwd * d.odom_fwd ** 2 + (k.mv_fwd + k.mv_lat) * pm * 0.05
var_th = k.k_th_turn * np.abs(d.odom_th) + k.k_t_rot * 0.05 + k.q_th * d.odom_th ** 2 + k.mv_th * pm * 0.05
d["vtr"], d["vth"] = var_tr, var_th

lags = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
rows = []
for cid, c in d.groupby(chain):
    if len(c) < 5: continue
    s = np.r_[0, np.cumsum(c.step.values)]            # path length at each row boundary
    zx, zy, zt = c.z_x.values, c.z_y.values, c.z_th.values
    cx, cy, ct = np.r_[0, np.cumsum(c.ox)], np.r_[0, np.cumsum(c.oy)], np.r_[0, np.cumsum(c.odom_th)]
    cvtr, cvth = np.r_[0, np.cumsum(c.vtr)], np.r_[0, np.cumsum(c.vth)]
    gx, gy, gt = c.gt_x.values, c.gt_y.values, c.gt_th.values
    n = len(c)
    for i in range(0, n, 3):                           # thin the start points
        for L in lags:
            j = np.searchsorted(s, s[i] + L)
            if j >= n: break
            Dx = zx[j] - zx[i] - (cx[j] - cx[i]); Dy = zy[j] - zy[i] - (cy[j] - cy[i])
            Dt = np.remainder(zt[j] - zt[i] - (ct[j] - ct[i]) + np.pi, 2 * np.pi) - np.pi
            r = dict(L=L, D2=Dx * Dx + Dy * Dy, Dt2=Dt * Dt, Ptr=cvtr[j] - cvtr[i], Pth=cvth[j] - cvth[i])
            if np.isfinite(gx[i]) and np.isfinite(gx[j]):   # referee: structure of (z - gt)
                ex = (zx[j] - gx[j]) - (zx[i] - gx[i]); ey = (zy[j] - gy[j]) - (zy[i] - gy[i])
                et = np.remainder((zt[j] - gt[j]) - (zt[i] - gt[i]) + np.pi, 2 * np.pi) - np.pi
                r.update(G2=ex * ex + ey * ey, Gt2=et * et)
            rows.append(r)
R = pd.DataFrame(rows)
if R.empty: sys.exit("no chains long enough yet")
print("\n lag(m)   n    E[D^2]-odom (mm^2)   GT bias SF (mm^2) | heading: E[Dt^2]-odom   GT  (mdeg^2 -> printed as deg rms)")
for L, g in R.groupby("L"):
    ex_tr = (g.D2 - g.Ptr).mean() * 1e6
    ex_th = (g.Dt2 - g.Pth).mean()
    gt_tr = g.G2.mean() * 1e6 if "G2" in g and g.G2.notna().any() else np.nan
    gt_th = g.Gt2.mean() if "Gt2" in g and g.Gt2.notna().any() else np.nan
    print(f" {L:5.2f} {len(g):6d}   {ex_tr:10.1f}            {gt_tr:10.1f}       |  "
          f"{np.degrees(np.sqrt(max(ex_th, 0))):.3f}  {np.degrees(np.sqrt(gt_th)) if np.isfinite(gt_th) else float('nan'):.3f} deg"
          f"   (odom part: {np.sqrt(g.Ptr.mean())*1e3:.1f} mm, {np.degrees(np.sqrt(g.Pth.mean())):.3f} deg)")
print("\n Saturation of the GT column = 2 sigma_b^2 (two independent bias samples); its rise length = l.")
print(" If the GT-free column tracks the GT one, the bias is learnable without ground truth.")
