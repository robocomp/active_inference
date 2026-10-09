#!/usr/bin/env python3
"""Grade a scan-to-scan innovation log (tmp/noise_innov/innov_<ts>.csv, motion_noise_innov.h).

Checks, per motion class (parked / driving / pivot) and body axis:
  * calibration of the model: y/m = innovation^2 / model variance should have MEAN 1 and MEDIAN ~0.455
    (chi-square with 1 dof). A class far off means the model misses something there.
  * stability: the learnt coefficients over the run (no collapse to 0, no runaway).
usage: noise_innov_report.py [csv]   (default: newest)
"""
import glob, os, sys
import numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("tmp/noise_innov/innov_*.csv"), key=os.path.getmtime)
d = pd.read_csv(path).dropna()
print(f"{os.path.basename(path)}: {len(d)} cycles")
mov = np.abs(d.odom_fwd) > 0.002
turn = np.abs(d.odom_th) > 0.005
cls = np.where(~mov & ~turn, "parked", np.where(turn & ~mov, "pivot", "driving"))
d["cls"] = cls
print("\n calibration y/m (mean should be 1.0, median ~0.455):")
for c in ["parked", "driving", "pivot"]:
    s = d[d.cls == c]
    if len(s) < 20:
        print(f"  {c:8s} n={len(s)} (too few)"); continue
    row = []
    for a in ["fwd", "lat", "th"]:
        r = s["y_" + a] / s["m_" + a]
        row.append(f"{a} mean {r.mean():5.2f} med {r.median():5.3f}")
    print(f"  {c:8s} n={len(s):5d}  " + " | ".join(row))
names = ["k_long", "k_lat", "k_lat_turn", "k_th_turn", "k_t_trans", "k_t_rot", "s_fwd", "s_lat", "s_th", "q_fwd", "q_lat", "q_th"]
print("\n coefficients: first / 25% / 50% / 75% / last of the run, min after the first 10%")
n = len(d)
for k in names:
    v = d[k].to_numpy()
    q = [v[0], v[n // 4], v[n // 2], v[3 * n // 4], v[-1]]
    print(f"  {k:10s} " + "  ".join(f"{x:.3e}" for x in q) + f"   min {v[n // 10:].min():.3e}")
print("\n motion seen: parked %d, driving %d, pivot %d cycles; path %.1f m, turned %.1f rad" %
      ((cls == "parked").sum(), (cls == "driving").sum(), (cls == "pivot").sum(),
       np.abs(d.odom_fwd).sum(), np.abs(d.odom_th).sum()))
