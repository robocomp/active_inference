#!/usr/bin/env python3
"""
pose_error_stats.py — the pose-uncertainty inputs of the local-motion policy, fitted from room_concept's
sdf_localizer gt_error.csv logs (Task 7; ruling T8-2).

Two numbers the policy needs and nothing in the controller provides (local_policy.h, PolicyInput):

  (a) pose_drift_sigma_m / pose_drift_time_s — the ABSOLUTE pose error e(t) = est(t) - gt(t) as an
      Ornstein-Uhlenbeck process. The ProgressMonitor needs the std of its CHANGE over a plan's elapsed
      time tau, sigma*sqrt(2(1-exp(-tau/T))). So we fit exactly that: the empirical variogram
          g(tau) = E[(e(t+tau) - e(t))^2]  (per axis)   ~=   2 sigma^2 (1 - exp(-tau/T))
      over tau in [0.1, 60] s, least squares on log g (every lag decade weighs the same). A white
      "nugget" (estimator jitter) is fitted too, g = 2 s_w^2 + 2 sigma^2 (1 - exp(-tau/T)), and reported, but the
      policy's model has no nugget, so the 2-parameter fit is the one to configure.
      Yaw is fitted the same way (rad), for the record.

  (b) sigma_rel_xy / sigma_rel_yaw — the RELATIVE body-vs-returns error over a rollout (~1 s): how wrong
      the robot's own account of its motion over W seconds is. For each W-second window the relative motion
      (end pose expressed in the start pose's frame: x right, y forward) is computed from
        GT        the Webots supervisor pose,
        ODOM      the localiser's motion prior, dead-reckoned: increment_k = pred_k (-) est_{k-1}
                  (pred = the prediction the localiser made for row k from row k-1's estimate — wheels/IMU/
                  command prior, before the scan correction), composed over the window,
        EST       the localiser's own estimate,
      and the error vs GT is taken per axis. Reported for W = 0.5, 1, 2 s, over all windows and over MOVING
      windows (GT moved >= 0.05 m or turned >= 0.05 rad in the window). The policy value is the per-axis RMS
      of ODOM over MOVING 1 s windows (the ruling: "sigma_rel (odometry)").

Conventions (verified on the logs): est_theta is the controller's theta (forward at +90 deg), gt_theta is the
heading, so gt_yaw_in_est_convention = gt_theta - pi/2. GT updates at ~10 Hz while rows come at ~20 Hz; it is
used only at rows where it CHANGED (fresh) and linearly interpolated between them (no measurable lag:
along-track error / speed = -0.08..+0.01 s on try105/try16/try41/try93).

⚑ Data cleaning, flagged (this is a fit, not a control gate): samples with |e_xy| > 1.0 m or |e_yaw| > 20 deg
are LOST/relocalising/room-frame-switch episodes, not drift; they split a segment and are counted, not fitted.
A one-step jump of either pose > 0.5 m also splits (a room-frame switch at a door). Gaps > 0.5 s split.
A run is HEALTHY when >= 95 % of its time is clean; pooled values use healthy runs only.

Parsing: pandas' C parser (locale-independent; the CLAUDE.md locale hazard is a C/C++ strtof one).

Usage:
  pose_error_stats.py [--csv out.csv] FILE_OR_GLOB...
  e.g. pose_error_stats.py '../room_concept/estimation_scripts/crossing/*/gt_error.csv' \
                           ../room_concept/tmp/sdf_localizer/gt_error.csv
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

COLS = ['ts_ms', 'gt_x', 'gt_y', 'gt_theta', 'est_x', 'est_y', 'est_theta', 'pred_x', 'pred_y', 'pred_theta']
GRID_DT = 0.1            # s, resampling grid (GT is ~10 Hz)
MAX_GAP_S = 0.5
GROSS_XY_M = 1.0
GROSS_YAW_RAD = math.radians(20.0)
JUMP_M = 0.5
HEALTHY_FRAC = 0.95
WINDOWS_S = (0.5, 1.0, 2.0)
MOVING_M, MOVING_RAD = 0.05, 0.05
VARIO_LAGS_S = np.unique(np.round(np.geomspace(0.1, 300.0, 50) / GRID_DT).astype(int)) * GRID_DT
T_MAX_S = 1e4           # fit bound; a fit AT the bound is a random walk over the observed lags (see 'q')


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load(path):
    d = pd.read_csv(path, usecols=lambda c: c in COLS, engine='c', comment=None)
    d = d.dropna()
    d = d[(d.ts_ms > 0)].sort_values('ts_ms').drop_duplicates('ts_ms')
    return d


def segments(t, mask_ok):
    """Index ranges [a, b) of consecutive clean rows with no time gap > MAX_GAP_S."""
    out, a = [], None
    for i in range(len(t)):
        ok = mask_ok[i] and (a is None or t[i] - t[i - 1] <= MAX_GAP_S)
        if ok and a is None:
            a = i
        elif not ok and a is not None:
            out.append((a, i)); a = i if mask_ok[i] else None
    if a is not None:
        out.append((a, len(t)))
    return [(a, b) for a, b in out if t[b - 1] - t[a] >= 5.0]


def odom_track(d, a, b):
    """Dead-reckoned pose from the localiser's motion prior over rows a..b-1 (start = est at row a)."""
    ex, ey, eth = d['est_x'].values, d['est_y'].values, d['est_theta'].values
    px, py, pth = d['pred_x'].values, d['pred_y'].values, d['pred_theta'].values
    n = b - a
    ox, oy, oth = np.empty(n), np.empty(n), np.empty(n)
    ox[0], oy[0], oth[0] = ex[a], ey[a], eth[a]
    for k in range(1, n):
        i = a + k
        c, s = math.cos(eth[i - 1]), math.sin(eth[i - 1])
        dx, dy = px[i] - ex[i - 1], py[i] - ey[i - 1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy           # increment in the previous estimate's frame
        dth = wrap(pth[i] - eth[i - 1])
        c2, s2 = math.cos(oth[k - 1]), math.sin(oth[k - 1])
        ox[k] = ox[k - 1] + c2 * lx - s2 * ly
        oy[k] = oy[k - 1] + s2 * lx + c2 * ly
        oth[k] = oth[k - 1] + dth
    return ox, oy, oth


def relative(x, y, th, i, j):
    c, s = np.cos(th[i]), np.sin(th[i])
    dx, dy = x[j] - x[i], y[j] - y[i]
    return c * dx + s * dy, -s * dx + c * dy, wrap(th[j] - th[i])


def ou_model(p, tau):
    s, T = np.exp(p[0]), np.exp(p[1])
    return 2.0 * s * s * (1.0 - np.exp(-tau / T))


def ou_nugget_model(p, tau):
    return 2.0 * np.exp(p[2]) ** 2 + ou_model(p[:2], tau)


def fit_ou(tau, g, nugget=False):
    m = np.isfinite(g) & (g > 0)
    if m.sum() < 4:
        return None
    tau, g = tau[m], g[m]
    p0 = [0.5 * math.log(max(g.max(), 1e-12) / 2.0), math.log(2.0)]
    if nugget:
        p0.append(0.5 * math.log(max(g.min(), 1e-12) / 2.0))
        f = lambda p: np.log(ou_nugget_model(p, tau)) - np.log(g)
    else:
        f = lambda p: np.log(ou_model(p, tau)) - np.log(g)
    r = least_squares(f, p0, bounds=([-15, math.log(0.01)] + ([-20] if nugget else []),
                                     [3, math.log(T_MAX_S)] + ([3] if nugget else [])))
    out = {'sigma': float(np.exp(r.x[0])), 'T': float(np.exp(r.x[1])), 'rms_log': float(np.sqrt(np.mean(r.fun ** 2)))}
    # small-tau diffusion rate: g ~= q^2 tau for tau << T (well defined even when T runs to its bound)
    out['q'] = out['sigma'] * math.sqrt(2.0 / out['T'])
    out['at_bound'] = out['T'] > 0.99 * T_MAX_S
    if nugget:
        out['nugget'] = float(np.exp(r.x[2]))
    return out


def analyse(path):
    d = load(path)
    if len(d) < 200:
        return None
    t = (d.ts_ms.values - d.ts_ms.values[0]) / 1000.0
    gx, gy = d.gt_x.values, d.gt_y.values
    gth = d.gt_theta.values - np.pi / 2.0
    ex, ey, eth = d.est_x.values, d.est_y.values, d.est_theta.values
    exy = np.hypot(ex - gx, ey - gy)
    eyaw = np.abs(wrap(eth - gth))
    jump = np.r_[False, (np.hypot(np.diff(gx), np.diff(gy)) > JUMP_M) | (np.hypot(np.diff(ex), np.diff(ey)) > JUMP_M)]
    clean = (exy <= GROSS_XY_M) & (eyaw <= GROSS_YAW_RAD) & ~jump
    dt = np.r_[np.diff(t), 0.0]
    clean_frac = float(np.sum(dt * clean) / max(np.sum(dt), 1e-9))
    fresh = np.r_[True, (np.diff(gx) != 0) | (np.diff(gy) != 0) | (np.diff(d.gt_theta.values) != 0)]

    vario_xy = {k: [0.0, 0] for k in range(len(VARIO_LAGS_S))}
    vario_yaw = {k: [0.0, 0] for k in range(len(VARIO_LAGS_S))}
    rel = {(w, src): [] for w in WINDOWS_S for src in ('odom', 'est')}
    rel_moving = {(w, src): [] for w in WINDOWS_S for src in ('odom', 'est')}
    err_all = []
    path_m = 0.0
    for a, b in segments(t, clean):
        idx = np.arange(a, b)
        fi = idx[fresh[a:b]]
        if len(fi) < 10:
            continue
        tg = np.arange(t[fi[0]], t[fi[-1]], GRID_DT)
        if len(tg) < 20:
            continue
        Gx, Gy = np.interp(tg, t[fi], gx[fi]), np.interp(tg, t[fi], gy[fi])
        Gth = np.interp(tg, t[fi], np.unwrap(gth[fi]))
        Ex, Ey = np.interp(tg, t[idx], ex[idx]), np.interp(tg, t[idx], ey[idx])
        Eth = np.interp(tg, t[idx], np.unwrap(eth[idx]))
        ox, oy, oth = odom_track(d, a, b)
        Ox, Oy = np.interp(tg, t[idx], ox), np.interp(tg, t[idx], oy)
        Oth = np.interp(tg, t[idx], oth)
        path_m += float(np.sum(np.hypot(np.diff(Gx), np.diff(Gy))))
        e = np.stack([Ex - Gx, Ey - Gy])
        ey_ = wrap(Eth - Gth)
        err_all.append(np.c_[e.T, ey_])
        for k, lag in enumerate(VARIO_LAGS_S):
            L = int(round(lag / GRID_DT))
            if L >= len(tg):
                continue
            de = e[:, L:] - e[:, :-L]
            vario_xy[k][0] += float(np.sum(de ** 2)); vario_xy[k][1] += de.size
            dy_ = wrap(ey_[L:] - ey_[:-L])
            vario_yaw[k][0] += float(np.sum(dy_ ** 2)); vario_yaw[k][1] += dy_.size
        for w in WINDOWS_S:
            L = int(round(w / GRID_DT))
            if L >= len(tg):
                continue
            i, j = np.arange(len(tg) - L), np.arange(L, len(tg))
            gxr, gyr, gthr = relative(Gx, Gy, Gth, i, j)
            moving = (np.hypot(gxr, gyr) >= MOVING_M) | (np.abs(gthr) >= MOVING_RAD)
            for src, (X, Y, TH) in (('odom', (Ox, Oy, Oth)), ('est', (Ex, Ey, Eth))):
                xr, yr, thr = relative(X, Y, TH, i, j)
                er = np.c_[xr - gxr, yr - gyr, wrap(thr - gthr)]
                rel[(w, src)].append(er)
                rel_moving[(w, src)].append(er[moving])

    res = {'run': os.path.basename(os.path.dirname(path)) or path, 'path': path, 'rows': len(d), 'ts0': int(d.ts_ms.values[0]), 'dur_s': float(t[-1]),
           'clean_frac': clean_frac, 'healthy': clean_frac >= HEALTHY_FRAC, 'path_m': path_m}
    if not err_all:
        return res
    E = np.vstack(err_all)
    res['bias_x'], res['bias_y'] = float(E[:, 0].mean()), float(E[:, 1].mean())
    res['std_xy'] = float(np.sqrt(0.5 * (E[:, 0].var() + E[:, 1].var())))
    res['std_yaw'] = float(E[:, 2].std())
    gxy = np.array([vario_xy[k][0] / (2 * max(vario_xy[k][1] // 2, 1)) if vario_xy[k][1] else np.nan
                    for k in range(len(VARIO_LAGS_S))])            # per-axis mean of (de)^2
    gyaw = np.array([vario_yaw[k][0] / vario_yaw[k][1] if vario_yaw[k][1] else np.nan for k in range(len(VARIO_LAGS_S))])
    res['vario_xy'], res['vario_yaw'] = gxy, gyaw
    res['vario_n'] = np.array([vario_xy[k][1] for k in range(len(VARIO_LAGS_S))])
    for name, g in (('xy', gxy), ('yaw', gyaw)):
        f2, f3 = fit_ou(VARIO_LAGS_S, g), fit_ou(VARIO_LAGS_S, g, nugget=True)
        if f2:
            res[f'ou_{name}_sigma'], res[f'ou_{name}_T'], res[f'ou_{name}_fit'] = f2['sigma'], f2['T'], f2['rms_log']
            res[f'ou_{name}_q'] = f2['q']
            for tau in (1.0, 5.0):
                k = int(np.argmin(np.abs(VARIO_LAGS_S - tau)))
                res[f'drift_{name}_{tau:g}s'] = float(np.sqrt(g[k])) if np.isfinite(g[k]) else float('nan')
        if f3:
            res[f'ou3_{name}_sigma'], res[f'ou3_{name}_T'], res[f'ou3_{name}_nugget'] = f3['sigma'], f3['T'], f3['nugget']
    for (w, src), lst in rel.items():
        A = np.vstack(lst) if lst else np.zeros((0, 3))
        M = np.vstack(rel_moving[(w, src)]) if rel_moving[(w, src)] else np.zeros((0, 3))
        for tag, X in (('all', A), ('mov', M)):
            key = f'{src}_{tag}_{w:g}s'
            res[key + '_n'] = len(X)
            if len(X) >= 20:
                res[key + '_x'] = float(np.sqrt(np.mean(X[:, 0] ** 2)))      # lateral (x right)
                res[key + '_y'] = float(np.sqrt(np.mean(X[:, 1] ** 2)))      # forward
                res[key + '_xy'] = float(np.sqrt(0.5 * np.mean(X[:, 0] ** 2 + X[:, 1] ** 2)))   # per-axis RMS
                res[key + '_yaw'] = float(np.sqrt(np.mean(X[:, 2] ** 2)))
                res[key + '_xy_p95'] = float(np.percentile(np.hypot(X[:, 0], X[:, 1]), 95))
    res['_rel_mov'] = {k: (np.vstack(v) if v else np.zeros((0, 3))) for k, v in rel_moving.items()}
    return res


def q(vals, p):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    return float(np.percentile(v, p)) if len(v) else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--csv', help='per-run table to write')
    a = ap.parse_args()
    paths = []
    for f in a.files:
        paths += sorted(glob.glob(f)) if any(ch in f for ch in '*?[') else [f]
    runs, seen = [], {}
    for p in paths:
        try:
            r = analyse(p)
        except Exception as e:      # a malformed log is reported, not fatal
            print(f'# {p}: {e}', file=sys.stderr)
            continue
        if r is not None:
            # the SAME log under two paths (room_concept/tmp/sdf_localizer/ is the live copy a crossing dir archives)
            # counts once: identified by its first stamp (the archive may have been copied while the live file grew)
            key = r['ts0']
            if key in seen:
                print(f"# {p}: duplicate of {seen[key]} — skipped", file=sys.stderr)
                continue
            seen[key] = p
            runs.append(r)
            print(f"# {r['run']}: clean {r['clean_frac']:.3f} path {r['path_m']:.1f} m", file=sys.stderr)

    hdr = (f"{'run':28s} {'dur_s':>7s} {'path_m':>6s} {'clean':>5s} | {'OU s_xy':>7s} {'T_xy':>6s} {'fit':>4s} "
           f"{'s_yaw':>6s} {'T_yaw':>6s} | {'odom1s_xy':>9s} {'odom1s_yaw':>10s} {'est1s_xy':>8s} {'n_mov':>6s}")
    print(hdr)
    for r in runs:
        g = lambda k, fmt: (format(r[k], fmt) if k in r and np.isfinite(r[k]) else '-')
        print(f"{r['run'][:28]:28s} {r['dur_s']:7.0f} {r['path_m']:6.1f} {r['clean_frac']:5.2f}{'*' if not r['healthy'] else ' '}| "
              f"{g('ou_xy_sigma', '7.4f')} {g('ou_xy_T', '6.1f')} {g('ou_xy_fit', '4.2f')} "
              f"{g('ou_yaw_sigma', '6.4f')} {g('ou_yaw_T', '6.1f')} | {g('odom_mov_1s_xy', '9.4f')} "
              f"{g('odom_mov_1s_yaw', '10.4f')} {g('est_mov_1s_xy', '8.4f')} {r.get('odom_mov_1s_n', 0):6d}")
    print("(* = not healthy: < 95 % clean time; excluded from the pooled values)")

    H = [r for r in runs if r['healthy'] and 'ou_xy_sigma' in r]
    print(f"\n== PER-RUN SPREAD over {len(H)} healthy runs (median [p10, p90]) ==")
    for k, unit in (('ou_xy_sigma', 'm'), ('ou_xy_T', 's'), ('ou_xy_q', 'm/sqrt(s)'), ('drift_xy_1s', 'm'), ('drift_xy_5s', 'm'),
                    ('ou3_xy_nugget', 'm'), ('ou_yaw_sigma', 'rad'), ('ou_yaw_T', 's'), ('drift_yaw_1s', 'rad'),
                    ('std_xy', 'm'),
                    ('odom_mov_0.5s_xy', 'm'), ('odom_mov_1s_xy', 'm'), ('odom_mov_2s_xy', 'm'),
                    ('odom_mov_1s_x', 'm'), ('odom_mov_1s_y', 'm'),
                    ('odom_mov_0.5s_yaw', 'rad'), ('odom_mov_1s_yaw', 'rad'), ('odom_mov_2s_yaw', 'rad'),
                    ('est_mov_1s_xy', 'm'), ('est_mov_1s_yaw', 'rad'), ('odom_all_1s_xy', 'm')):
        vals = [r.get(k) for r in H]
        print(f"  {k:20s} {q(vals, 50):9.4f} [{q(vals, 10):.4f}, {q(vals, 90):.4f}] {unit}")

    # Pooled: variogram summed over healthy runs (weighted by pair count), and all moving windows together.
    if H:
        # EQUAL WEIGHT PER RUN: an 11-hour parked run (tour3_1) would otherwise outvote every driven one.
        gxy = np.nanmean(np.vstack([r['vario_xy'] for r in H]), axis=0)
        gyaw = np.nanmean(np.vstack([r['vario_yaw'] for r in H]), axis=0)
        print("\n== POOLED (healthy runs, each run weighted equally) ==")
        f2, f3 = fit_ou(VARIO_LAGS_S, gxy), fit_ou(VARIO_LAGS_S, gxy, nugget=True)
        print(f"  OU xy   2-param: sigma {f2['sigma']:.4f} m  T {f2['T']:.1f} s  (rms log-resid {f2['rms_log']:.2f}; "
              f"q {f2['q']:.5f} m/sqrt(s){'; T AT BOUND = random walk' if f2['at_bound'] else ''})")
        print(f"  OU xy   +nugget: sigma {f3['sigma']:.4f} m  T {f3['T']:.1f} s  nugget {f3['nugget']:.4f} m")
        fy = fit_ou(VARIO_LAGS_S, gyaw)
        print(f"  OU yaw  2-param: sigma {fy['sigma']:.4f} rad  T {fy['T']:.1f} s")
        print("  variogram (tau s : sqrt(E[de^2]) per axis m | model): " + ', '.join(
            f"{tau:g}:{math.sqrt(g):.4f}|{math.sqrt(ou_model([math.log(f2['sigma']), math.log(f2['T'])], tau)):.4f}"
            for tau, g in zip(VARIO_LAGS_S, gxy) if any(abs(tau - x) < 0.051 * x for x in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0))))
        for w in WINDOWS_S:
            for src in ('odom', 'est'):
                X = np.vstack([r['_rel_mov'][(w, src)] for r in H])
                if len(X):
                    print(f"  rel {src:4s} {w:g} s moving (n {len(X):6d}): per-axis RMS {math.sqrt(0.5 * np.mean(X[:, 0]**2 + X[:, 1]**2)):.4f} m "
                          f"(x {math.sqrt(np.mean(X[:, 0]**2)):.4f}, y {math.sqrt(np.mean(X[:, 1]**2)):.4f}; |e| p95 "
                          f"{np.percentile(np.hypot(X[:, 0], X[:, 1]), 95):.4f}) yaw RMS {math.sqrt(np.mean(X[:, 2]**2)):.4f} rad")

    if a.csv:
        keys = sorted({k for r in runs for k in r if not k.startswith('_') and not k.startswith('vario')})
        with open(a.csv, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in runs:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    main()
