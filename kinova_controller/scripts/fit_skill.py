#!/usr/bin/env python3
"""
fit_skill.py — offline step (c) of the skill-learning loop.

Reads the per-rep dataset produced by the probe rounds (Controller.dataset_path)
and, optionally, the controller log with the [tiplog] stream, and fits the three
things that license moving from closed-loop to model-based open-loop reaching:

  1. GRASP-POSE BIAS — the command offset that maximises grasp quality. The probe
     perturbs the command relative to the *perceived* bottle; if the best grasps
     cluster away from zero offset, that cluster is a systematic calibration bias
     (base_tf_/mount/bottle-frame). The correction to apply is −bias.

  2. TOLERANCE BUDGET / SENSITIVITY — how fast grasp quality falls off as the
     command leaves the optimum. From a quadratic fit of bottle_rise on the
     perturbation, plus a logistic fit of success vs offset magnitude. This is the
     envelope within which an open-loop command still grasps.

  3. ACTUATION MODEL (G, tau) — per joint from the q̇_cmd / q̇_meas log: a static
     gain G = q̇_meas/q̇_cmd and a first-order (ARX) model giving gain + lag, the
     feedforward needed for open-loop tracking.

Usage:
    fit_skill.py DATASET.csv [--joints JOINT_ACTUATION.csv] [--dt 0.02] [--rise-thresh 0.06]

Dependencies: numpy, pandas.
"""
import argparse
import sys

import numpy as np
import pandas as pd


# ── 1 & 2: grasp-pose bias + tolerance from the per-rep CSV ───────────────────
def fit_grasp(df: pd.DataFrame, rise_thresh: float) -> None:
    n = len(df)
    succ = df["success"].astype(int)
    print(f"\n=== per-rep dataset: {n} reps, {succ.sum()} success "
          f"({100.0 * succ.mean():.0f}%) ===")
    if n < 8:
        print("  (few reps — fits will be noisy; run a longer probe round)")

    p = df[["dx_perp", "dz_axis", "dazi"]].to_numpy(float)
    rise = df["bottle_rise"].to_numpy(float)

    # Quality model: bottle_rise ~ b0 + g·p + ½ pᵀ H p  (separable quadratic per
    # axis is enough and robust at small n). Fit y = X·c with X = [1, p, p²].
    X = np.column_stack([np.ones(n), p, p ** 2])
    c, *_ = np.linalg.lstsq(X, rise, rcond=None)
    b0 = c[0]
    g = c[1:4]          # linear terms
    h = c[4:7]          # quadratic (curvature) terms
    # Per-axis optimum of b0 + g·x + h·x²  is  x* = −g/(2h)  (where h<0 = a peak).
    names = ["dx_perp (tangent)", "dz_axis (axial)", "dazi (azimuth)"]
    print("\n--- grasp-pose bias (offset that maximises grasp quality) ---")
    bias = np.zeros(3)
    for i, nm in enumerate(names):
        if h[i] < -1e-6:
            xstar = -g[i] / (2.0 * h[i])
            bias[i] = xstar
            unit = "rad" if i == 2 else "m"
            print(f"  {nm:18s}: optimum {xstar:+.4f} {unit}  "
                  f"(curvature {h[i]:+.3f}, well-defined peak)")
        else:
            print(f"  {nm:18s}: no clear peak (curvature {h[i]:+.3f}); "
                  f"linear slope {g[i]:+.3f} — widen probe range or add reps")
    print("  → apply −bias as a calibration correction to the nominal grasp command:")
    print(f"      dx_perp {-bias[0]:+.4f} m, dz_axis {-bias[1]:+.4f} m, "
          f"dazi {-bias[2]:+.4f} rad")

    # Tolerance: logistic-ish — bin by offset magnitude, report success rate, and
    # the largest magnitude that still grasped.
    print("\n--- tolerance budget (success vs command-offset magnitude) ---")
    mag = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)   # positional offset only [m]
    if n >= 6:
        edges = np.quantile(mag, [0, 0.34, 0.67, 1.0])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (mag >= lo) & (mag <= hi + 1e-12)
            if m.sum():
                print(f"  |pos offset| {lo*1000:4.0f}–{hi*1000:4.0f} mm : "
                      f"{succ[m].mean()*100:3.0f}% success ({m.sum()} reps)")
    if succ.sum():
        print(f"  largest positional offset that still grasped: "
              f"{mag[succ.astype(bool)].max()*1000:.0f} mm")
    if (~succ.astype(bool)).any():
        print(f"  smallest positional offset that MISSED:        "
              f"{mag[~succ.astype(bool)].min()*1000:.0f} mm")

    # Terminal convergence error vs perturbation (does probing hurt the standoff?).
    if "commit_epos" in df:
        ok = df["commit_epos"] > 0
        if ok.any():
            print(f"\n  standoff commit error: mean {df.loc[ok,'commit_epos'].mean()*1000:.1f} mm, "
                  f"settle time mean {df.loc[ok,'track_ticks'].mean():.0f} cycles")


# ── 3: actuation model from the joint-velocity log ────────────────────────────
def fit_actuation(joints_path: str, dt: float) -> None:
    """Per-joint commanded-vs-measured q̇. Rows already pair q̇_cmd[k-1] with the
    q̇_meas it produced (the controller logs the previous command against the current
    measurement), so a direct regression is correctly command→response aligned."""
    dj = pd.read_csv(joints_path)
    njoints = sum(c.startswith("qd_cmd") for c in dj.columns)
    print(f"\n=== actuation model: {len(dj)} cycles, {njoints} joints "
          f"(joint q̇_cmd vs q̇_meas) ===")
    if len(dj) < 50:
        print("  (few cycles — run a longer round)")

    Gs, taus = [], []
    for j in range(njoints):
        c = dj[f"qd_cmd{j}"].to_numpy(float)
        m = dj[f"qd_meas{j}"].to_numpy(float)
        act = np.abs(c) > 0.02                         # only where actually commanded
        if act.sum() < 20:
            print(f"  joint {j}: too few active samples")
            continue
        cc, mm = c[act], m[act]
        # Static gain through the origin.
        G = float((cc @ mm) / (cc @ cc))
        # First-order ARX over the full (time-ordered) stream: m[k]=a·m[k-1]+b·c[k].
        a = b = np.nan
        y = m[1:]; X = np.column_stack([m[:-1], c[1:]])
        keep = np.abs(c[1:]) > 0.02
        if keep.sum() > 20:
            (a, b), *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        tau = (-dt / np.log(a)) if (0 < a < 1) else np.nan
        Gss = (b / (1 - a)) if (0 < a < 1) else G
        Gs.append(G)
        if np.isfinite(tau): taus.append(tau)
        tau_s = f"{tau*1000:5.0f} ms" if np.isfinite(tau) else "  n/a "
        print(f"  joint {j}: static G={G:5.3f}  ARX Gss={Gss:5.3f}  lag τ={tau_s} "
              f"({int(act.sum())} samples)")
    if Gs:
        print(f"  → mean static gain {np.mean(Gs):.3f}"
              + (f", mean lag {np.mean(taus)*1000:.0f} ms" if taus else "")
              + ".  Feedforward for open-loop: pre-scale q̇_cmd by 1/G"
              + (" and lead by τ." if taus else "."))


# ── 4: learning-improvement metric (precision re-weighting) ───────────────────
def summarize_metrics(metrics_path: str) -> None:
    dm = pd.read_csv(metrics_path)
    ok = dm[dm["success"] == 1]
    print(f"\n=== learning improvement: {len(dm)} reps "
          f"({ok['success'].count()} successful) ===")
    if len(ok) < 6:
        print("  (too few successful reps to compare early vs late)")
        return
    k = max(1, len(ok) // 3)
    early, late = ok.iloc[:k], ok.iloc[-k:]
    def row(tag, d):
        print(f"  {tag:10s} conf {d['confidence'].mean():.2f} | "
              f"cycle {d['cycle_time_s'].mean():5.1f} s | "
              f"{d['observations'].mean():5.1f} observations/rep")
    row("early", early); row("late", late)
    dct = 100.0 * (1 - late['cycle_time_s'].mean() / early['cycle_time_s'].mean())
    dob = 100.0 * (1 - late['observations'].mean() / max(1e-9, early['observations'].mean()))
    print(f"  → after learning: pick&place time {dct:+.0f}%, "
          f"observations {dob:+.0f}%  (negative = reduction = improvement)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="per-rep CSV (Controller.dataset_path)")
    ap.add_argument("--joints", help="joint-velocity CSV (Controller.joint_log_path) for G, tau")
    ap.add_argument("--metrics", help="per-rep metrics CSV (Controller.metrics_path) for the learning summary")
    ap.add_argument("--dt", type=float, default=0.02, help="control period [s] (Period.Compute)")
    ap.add_argument("--rise-thresh", type=float, default=0.06,
                    help="grasp-success rise threshold [m] (LIFT_CONFIRM_RISE_M)")
    a = ap.parse_args()

    df = pd.read_csv(a.dataset)
    if df.empty:
        print("dataset is empty — run a probe round first.")
        return 1
    fit_grasp(df, a.rise_thresh)
    if a.joints:
        fit_actuation(a.joints, a.dt)
    if a.metrics:
        summarize_metrics(a.metrics)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
