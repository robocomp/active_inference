#!/usr/bin/env python3
"""Grade a joint-calibration monitor log. usage: joint_calib_report.py [csv] [--inject x y yaw_deg]
Prints the last solve's joint vs chained camera yaw, the eps<->camera-yaw correlation, and (with
--inject, sim only) whether the joint helios lever/yaw recovered the planted mount error within 2 sigma.
Plan: docs/superpowers/plans/2026-10-05-joint-calibration.md (Tasks 4 and 6)."""
import argparse, glob, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--inject", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
a = ap.parse_args()
path = a.csv or max(glob.glob("tmp/joint_calib/joint_*.csv"), key=os.path.getmtime, default=None)
if not path: sys.exit("no joint_calib csv (is JointCalibMonitor on?)")
d = pd.read_csv(path)
need = {"ts_ms", "episodes", "eps_yaw", "eps_yaw_sd", "lever_x", "lever_x_sd", "lever_y", "lever_y_sd"}
missing = need - set(d.columns)
if missing: sys.exit(f"malformed: missing {sorted(missing)}")
if d.empty: sys.exit(f"{os.path.basename(path)}: header only, no solve yet")
last = d.iloc[-1]
print(f"{os.path.basename(path)}: {len(d)} solves, last at {last.episodes:.0f} episodes")
for c in [c for c in d.columns if c.endswith(".yaw") and not c.startswith("chain_")]:
    cam = c[:-4]
    print(f"  {cam}: joint yaw {np.degrees(last[c]):+.3f} +- {np.degrees(last[c + '_sd']):.3f} deg | "
          f"chained {np.degrees(last['chain_' + c]):+.3f} deg | corr(eps, yaw) {last['corr_eps_' + cam + 'yaw']:+.2f}")
print(f"  helios: eps_yaw {np.degrees(last.eps_yaw):+.3f} +- {np.degrees(last.eps_yaw_sd):.3f} deg, "
      f"lever ({last.lever_x*1e3:+.1f} +- {last.lever_x_sd*1e3:.1f}, {last.lever_y*1e3:+.1f} +- {last.lever_y_sd*1e3:.1f}) mm")
if a.inject:
    # Signed, on the median of the last N solves; PASS needs detection AND agreement (the prior alone
    # would pass "within 2 sigma of 0.5 deg"). K_EPS mirrors rc::joint::kEpsPerLidarYaw.
    K_EPS = -1.0
    tail = d.tail(min(20, len(d))).median(numeric_only=True)
    x, y, yaw = a.inject[0], a.inject[1], np.radians(a.inject[2])
    est = {"lever_x": (tail.lever_x, tail.lever_x_sd, x), "lever_y": (tail.lever_y, tail.lever_y_sd, y),
           "yaw": (tail.eps_yaw / K_EPS, tail.eps_yaw_sd, yaw)}
    ok = True
    for k, (v, sd, t) in est.items():
        if t == 0.0:
            continue                                  # not planted in this leg
        detected, agrees = abs(v) > 2 * sd, abs(v - t) < 2 * sd
        print(f"  {k}: est {v:+.5f} +- {sd:.5f} vs planted {t:+.5f} | detected {detected} agrees {agrees}")
        ok = ok and detected and agrees
    print(f"  implied kEps from the yaw leg: {np.sign(tail.eps_yaw) * np.sign(yaw) if yaw else float('nan'):+.0f}")
    sys.exit(0 if ok else 1)
