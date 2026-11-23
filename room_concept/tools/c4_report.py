#!/usr/bin/env python3
"""AuRo C4 control (2026-11-23): gyro bias +0.002 rad/s injected at the bridge, calibrator ON vs OFF.

    tools/c4_report.py <run_dir_or_heading_csv> ...      (a run dir: its heading_*.csv and c4_conditions.txt)

Per leg, from the heading CSV (one row per localiser result; arm7_report.py's loader):
  * HEADING DRIFT while moving, deg/s, by thirds of the DISTANCE driven: the signed sum of
    (predicted heading increment - true heading increment) over rows where the robot moved, divided by the
    time spent moving. This is the quantity of EXPERIMENT.md §13's arm-7 verdict table ("predicted heading
    drift while moving, deg/s, thirds"). The predicted increment is pred_th[i] - est_th[i-1], the motion
    model's own guess before the localiser corrects it, so it carries the bias the calibrator has (not) removed.
  * GT heading error per rad turned, 5-s windows (arm7_report.grade).
  * b_omega ± sigma at the end of each third and at the end; whether b_omega was APPLIED (calib_applied bit).
The injected bias is +0.002 rad/s = +0.115 deg/s at the gyro.
"""
import math, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arm7_report import load, thirds_by_path, grade, W, PARAMS   # noqa: E402


def leg(path):
    d = load(path)
    ex, ey, eth, pth = d["est_x"], d["est_y"], d["est_th"], d["pred_th"]
    t = d["ts_ms"] / 1000.0
    gth = d["gt_th"]
    dpth = np.r_[0.0, W(pth[1:] - eth[:-1])]
    dgth = np.r_[0.0, W(np.diff(gth))]
    dt = np.r_[0.0, np.diff(t)]
    step = np.r_[0.0, np.hypot(np.diff(ex), np.diff(ey))]
    moving = (step > 0.002) & np.isfinite(dgth) & (dt < 1.0)
    out = {"path": step.sum(), "T": t[-1] - t[0], "drift": [], "b": []}
    for s in thirds_by_path(ex, ey):
        idx = np.arange(len(ex))[s]
        m = moving[idx]
        e = (dpth[idx] - dgth[idx])[m]
        out["drift"].append(math.degrees(e.sum()) / max(dt[idx][m].sum(), 1e-9))
        j = idx[-1]
        out["b"].append((d["v_b_omega"][j], d["s_b_omega"][j]) if "v_b_omega" in d else (float("nan"), float("nan")))
    e_all = (dpth - dgth)[moving]
    out["drift_all"] = math.degrees(e_all.sum()) / max(dt[moving].sum(), 1e-9)
    g = grade(d)
    out["gt"] = g["gt"]
    out["calib"] = g["calib"]
    i_b = PARAMS.index("b_omega")
    out["b_applied"] = bool(int(d["calib_applied"][-1]) >> i_b & 1) if "calib_applied" in d else None
    return out


def main():
    rows = []
    for a in sys.argv[1:]:
        cond = ""
        if os.path.isdir(a):
            cf = os.path.join(a, "c4_conditions.txt")
            cond = open(cf).readline().strip() if os.path.exists(cf) else ""
            hs = sorted(glob.glob(os.path.join(a, "heading_*.csv")))
            if not hs:
                print(f"{a}: no heading csv"); continue
            path = hs[-1]
        else:
            path = a
        r = leg(path)
        name = os.path.basename(a.rstrip("/"))
        b = r["calib"].get("b_omega", (float("nan"), float("nan")))
        print(f"\n== {name}  {cond}")
        print(f"   {r['path']:.0f} m in {r['T'] / 60:.1f} min; b_omega applied at end: {r['b_applied']}")
        print("   heading drift while moving (pred - truth), deg/s, thirds: " + " / ".join(f"{x:+.3f}" for x in r["drift"])
              + f"   all {r['drift_all']:+.3f}   (injected bias at the gyro: +0.115 deg/s)")
        print("   b_omega at end of thirds: " + " / ".join(f"{v:+.5f}±{s:.5f}" for v, s in r["b"])
              + f"   final {b[0]:+.5f}±{b[1]:.5f}  (injected +0.00200)")
        if r["gt"]:
            print(f"   GT heading error {r['gt']['deg_per_rad']:.3f} deg per rad turned ({r['gt']['turned']:.0f} rad)")
        rows.append((name, r))
    return rows


if __name__ == "__main__":
    main()
