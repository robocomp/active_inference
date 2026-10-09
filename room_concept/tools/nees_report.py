#!/usr/bin/env python3
"""Posterior consistency report for a tmp/heading/heading_*.csv run.

Grades the pose posterior against supervisor ground truth, using ONLY rows where
the GT actually changed (the supervisor pose arrives at ~10 Hz against a 20 Hz
LiDAR cycle, so every other row carries a stale GT with a ~55 ms heading lead)
and only while the robot is moving (est speed > --vmin).

  solve cycles (sur_scored==1): error in the BODY frame vs sur_pq_{fwd,lat,th}
  every moving row:             heading error vs cov_tt (the published variance)

Prints rms(error/sigma) per axis -- 1.0 is consistent, >1 overconfident -- plus
NEES means and the 95% coverage (|e| < 1.96 sigma, should be ~95%).

usage: nees_report.py [csv ...]   (default: newest tmp/heading/heading_*.csv)
"""
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def axis_line(name, e, var):
    ok = np.isfinite(e) & np.isfinite(var) & (var > 0)
    e, var = e[ok], var[ok]
    if len(e) < 5:
        return f"  {name:5s}  n={len(e):4d}  (too few rows)"
    z = e / np.sqrt(var)
    unit = 1e3 if name != "th" else np.degrees(1.0)
    u = "mm" if name != "th" else "deg"
    return (f"  {name:5s}  n={len(e):4d}  rms err={np.sqrt(np.mean(e**2))*unit:7.2f} {u}"
            f"  rms sigma={np.sqrt(np.mean(var))*unit:7.2f} {u}"
            f"  err/sigma={np.sqrt(np.mean(z**2)):5.2f}x"
            f"  cover95={100*np.mean(np.abs(z) < 1.96):5.1f}%")


def report(path, vmin):
    d = pd.read_csv(path, comment="#")
    d = d[np.isfinite(d.gt_x) & np.isfinite(d.est_x)].reset_index(drop=True)
    print(f"\n== {os.path.basename(path)}  rows={len(d)}")
    if len(d) < 3:
        print("  no GT rows"); return
    t = d.ts_ms.to_numpy() / 1e3
    fresh = np.r_[True, (np.diff(d.gt_x) != 0) | (np.diff(d.gt_y) != 0) | (np.diff(d.gt_th) != 0)]
    dt = np.r_[np.nan, np.diff(t)]
    speed = np.hypot(np.r_[np.nan, np.diff(d.est_x)], np.r_[np.nan, np.diff(d.est_y)]) / dt
    moving = speed > vmin
    sel = fresh & moving
    print(f"  fresh GT rows={fresh.sum()}  moving={moving.sum()}  fresh&moving={sel.sum()}"
          f"  driving time~{np.nansum(dt[moving]):.0f} s")

    ex, ey = (d.est_x - d.gt_x).to_numpy(), (d.est_y - d.gt_y).to_numpy()
    # The agent's theta is forward-at-+90deg; gt_th is the supervisor's standard heading,
    # so gt_th = est_th + pi/2 exactly. Subtract the CONSTANT (not the mean): a real bias must show.
    eth = wrap((d.est_th - d.gt_th).to_numpy() + np.pi / 2)
    # body axes from the GT heading (standard convention: forward = (cos, sin))
    c, s = np.cos(d.gt_th.to_numpy()), np.sin(d.gt_th.to_numpy())
    e_fwd = c * ex + s * ey
    e_lat = -s * ex + c * ey

    solve = sel & (d.sur_scored.to_numpy() == 1)
    print(f" SOLVE cycles (sur_pq_*), n={solve.sum()}")
    print(axis_line("fwd", e_fwd[solve], d.sur_pq_fwd.to_numpy()[solve]))
    print(axis_line("lat", e_lat[solve], d.sur_pq_lat.to_numpy()[solve]))
    print(axis_line("th", eth[solve], d.sur_pq_th.to_numpy()[solve]))
    print(f" PUBLISHED heading (cov_tt), all fresh&moving rows")
    print(axis_line("th", eth[sel], d.cov_tt.to_numpy()[sel]))
    if "pf_tt" in d.columns:
        # the pose-field bias term (pose_field_bias.h) as it WOULD be published (PoseFieldPublish adds it to cov_tt;
        # if it is already on, cov_tt contains it and this double-counts — the header line says which)
        pf_tt, pf_xx = d.pf_tt.to_numpy(), d.pf_xx.to_numpy()
        print(f" + POSE-FIELD term (sigma_b now {d.pf_sigma_b.iloc[-1]*1e3:.1f} mm, l {d.pf_len.iloc[-1]:.2f} m; "
              f"add it ONLY if PoseFieldPublish was off)")
        print(axis_line("fwd", e_fwd[solve], d.sur_pq_fwd.to_numpy()[solve] + pf_xx[solve]))
        print(axis_line("lat", e_lat[solve], d.sur_pq_lat.to_numpy()[solve] + pf_xx[solve]))
        print(axis_line("th", eth[solve], d.sur_pq_th.to_numpy()[solve] + pf_tt[solve]))
        print(axis_line("th", eth[sel], d.cov_tt.to_numpy()[sel] + pf_tt[sel]) + "   (published)")
    print(f" mean bias (fresh&moving): fwd {np.mean(e_fwd[sel])*1e3:+.1f} mm  "
          f"lat {np.mean(e_lat[sel])*1e3:+.1f} mm  th {np.degrees(np.mean(eth[sel])):+.3f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--vmin", type=float, default=0.05, help="m/s; rows slower than this are parked")
    a = ap.parse_args()
    paths = a.csv or sorted(glob.glob("tmp/heading/heading_*.csv"), key=os.path.getmtime)[-1:]
    if not paths:
        sys.exit("no csv")
    for p in paths:
        report(p, a.vmin)


if __name__ == "__main__":
    main()
