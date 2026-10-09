#!/usr/bin/env python3
"""Grade a joint-calibration monitor log. usage: joint_calib_report.py [csv] [--inject x y yaw_deg]
Prints the last solve's joint vs chained camera yaw, the eps<->camera-yaw correlation, and (with
--inject, sim only) whether the joint helios lever/yaw recovered the planted mount error within 2 sigma.
Plan: docs/superpowers/plans/2026-10-05-joint-calibration.md (Tasks 4 and 6).

r2 MOUNTS (plan 2026-10-08): joint_calib_report.py --mounts [mounts_csv] [--inject-helios X Y Z ROLL PITCH YAW]
[--inject-bpearl X Y Z ROLL PITCH YAW]  (m, deg; Pose6 about each sensor origin, as LidarMountInject*/BpearlMountInject*).
Prints helios 6 + bpearl 6 (+ cameras) with sigma and the per-factor information shares, helios.dz as "prior"
(no factor informs it), the acting yaw_offset (must be 0: single owner), and grades each PLANTED parameter
with r1's signed three-part rule (informed, |est| > 2 sigma, |est - plant| < 2 sigma) plus, for every
parameter, the ABSOLUTE error in mm / deg (the floor factor's sigma is tiny, so 2 sigma tests systematics)."""
import argparse, glob, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--inject", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
ap.add_argument("--mounts", action="store_true", help="grade a tmp/joint_calib/mounts_*.csv (r2)")
ap.add_argument("--inject-helios", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
ap.add_argument("--inject-bpearl", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
a = ap.parse_args()


def report_mounts(a):
    path = a.csv or max(glob.glob("tmp/joint_calib/mounts_*.csv"), key=os.path.getmtime, default=None)
    if not path: sys.exit("no mounts csv (is MountFactors on?)")
    d = pd.read_csv(path)
    if d.empty: sys.exit(f"{os.path.basename(path)}: header only")
    tail = d.tail(min(20, len(d))).median(numeric_only=True)
    last = d.iloc[-1]
    print(f"{os.path.basename(path)}: {len(d)} rows | blocks n: kin {last.n_kin:.0f} floor {last.n_floor:.0f} "
          f"vert_h {last.n_vert_h:.0f} vert_b {last.n_vert_b:.0f}")
    yo = d.yaw_offset.abs().max()
    print(f"  yaw_offset (eps ACTING on the odometry): max |.| over the run = {np.degrees(yo):.4f} deg "
          f"-> {'OK (single owner)' if yo == 0 else 'NOT ZERO: eps acted, the yaw grade is invalid'}")
    shares = [c[len("helios.dx_sh_"):] for c in d.columns if c.startswith("helios.dx_sh_")]
    plants = {"helios": a.inject_helios, "bpearl": a.inject_bpearl}
    ok = True
    for dev in ("helios", "bpearl"):
        for k, p in enumerate(("dx", "dy", "dz", "droll", "dpitch", "dyaw")):
            n = f"{dev}.{p}"
            if n not in d.columns: continue
            ang = k >= 3
            u, f = ("deg", np.degrees) if ang else ("mm", lambda x: 1e3 * x)
            v, sd = tail[n], tail[n + "_sd"]
            sh = " ".join(f"{s}:{tail[f'{n}_sh_{s}']:.2f}" for s in shares)
            prior = tail.get(f"{n}_sh_prior", 0.0) > 0.999
            line = f"  {n:14s} " + ("prior (no factor informs it)" if prior else f"{f(v):+8.3f} +- {f(sd):.3f} {u}") + f" | {sh}"
            plant = plants[dev]
            if plant is not None:
                t = plant[k] if not ang else np.radians(plant[k])
                line += f" | abs err {f(v - t):+.3f} {u}"
                if t != 0.0 and not prior:
                    informed, detected, agrees = tail[f"{n}_sh_prior"] < 0.9, abs(v) > 2 * sd and np.sign(v) == np.sign(t), abs(v - t) < 2 * sd
                    line += f" | informed {informed} detected {detected} agrees {agrees}"
                    ok = ok and informed and detected and agrees
            print(line)
    for c in [c for c in d.columns if c.endswith(".yaw") and not c.startswith(("helios", "bpearl"))]:
        print(f"  {c:14s} {np.degrees(tail[c]):+8.3f} +- {np.degrees(tail[c + '_sd']):.3f} deg (body frame)")
    sys.exit(0 if ok else 1)


if a.mounts:
    report_mounts(a)
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
