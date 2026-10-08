#!/usr/bin/env python3
"""Offline diagnosis of the kinematic factor's helios-lever Dx SHORTFALL (plan 2026-10-08 r2.2 item 8):
19-27 mm recovered for a 30 mm planted lever while Dy and psi are fine.

usage (from room_concept/):  tools/kinematic_shortfall_diag.py [STAMP ...]
      STAMP = a run stamp, e.g. 2026-10-08_12-18-41 (default: the two planted runs + the 10-07 null)

Reads tmp/noise_innov/innov_<stamp>.csv (the delta stream), tmp/sdf_localizer/log_<stamp>.csv (early-exit flag and the
PREDICTED pose the scan-only z is linearised at) and tmp/heading/heading_<stamp>.csv (ground truth, v_eps_yaw,
calib_applied). The sdf/heading logs may carry a stamp a second off: the nearest file is used.

The questions, each answered with numbers:
  1. the per-cycle weighted fit, as KinematicMount does it (rows of r2.2 item 1, weight (1-r)/m)
  2. CW vs CCW turns fitted separately: a lever is EVEN in the turn direction, a timing/deskew or a forward-scale
     confound is ODD; and pivots vs arcs
  3. the weighting: without the while-moving term mv*p_move*dt, unweighted, without outlier down-weighting
  4. a forward-scale (k_v) column and its correlation with Dx
  5. did a residual eps_yaw ACT (calib_applied bit 1, v_eps_yaw)?
  6. the scan-only pose's Gauss-Newton GAIN: on early-exit cycles z is ONE GN step from the dead-reckoned
     prediction; against ground truth, z - pred = g (gt - pred). g < 1 attenuates every delta signal by g while
     the estimate dead-reckons (delta_n = g D_n), and the deficit arrives as a burst at the next solve.
  7. the refit with z un-attenuated (z* = pred + (z - pred)/g), and telescoped 1 s blocks (honest sigma under the
     MA(1) of consecutive deltas)
"""
import glob, os, sys
import numpy as np, pandas as pd

R = "tmp"
stamps = sys.argv[1:] or ["2026-10-07_11-14-36", "2026-10-08_10-56-32", "2026-10-08_12-18-41"]


def nearest(pattern, stamp):
    """sdf_localizer / heading logs may be stamped a second earlier than the innov log."""
    fs = sorted(glob.glob(pattern))
    if not fs: return None
    key = lambda f: abs(pd.Timestamp(stamp.replace("_", " ").replace("-", ":", 5).replace(":", "-", 2))
                        - pd.Timestamp(os.path.basename(f).split("_", 1)[1][:19].replace("_", " ").replace("-", ":", 5).replace(":", "-", 2)))
    return min(fs, key=key)


def wls(X, y, w):
    ok = np.isfinite(w) & (w > 0) & np.isfinite(y) & np.isfinite(X).all(1)
    X, y, w = X[ok], y[ok], w[ok]
    H = (X * w[:, None]).T @ X
    C = np.linalg.inv(H)
    return C @ ((X * w[:, None]).T @ y), np.sqrt(np.diag(C)), C


def fit(d, wf=None, wl=None, kv=False, dfwd=None, dlat=None):
    dth = d.odom_th.values; df = d.odom_fwd.values; s = 2 * np.sin(dth / 2); n = len(d)
    k = 4 if kv else 3
    Xf = np.zeros((n, k)); Xl = np.zeros((n, k)); Xf[:, 0] = s; Xl[:, 1] = -s; Xl[:, 2] = df
    if kv: Xf[:, 3] = df
    wf = (1 - d.r_out_fwd.values) / d.m_fwd.values if wf is None else wf
    wl = (1 - d.r_out_lat.values) / d.m_lat.values if wl is None else wl
    y = np.r_[d.d_fwd.values if dfwd is None else dfwd, d.d_lat.values if dlat is None else dlat]
    return wls(np.vstack([Xf, Xl]), y, np.r_[wf, wl])


def fmt(p, sd, C=None):
    o = f"dx {p[0]*1e3:+6.1f}+-{sd[0]*1e3:4.1f}  dy {p[1]*1e3:+6.1f}+-{sd[1]*1e3:4.1f} mm  psi {np.degrees(p[2]):+.2f}+-{np.degrees(sd[2]):.2f} deg"
    if len(p) > 3: o += f"  k_v {100*p[3]:+.2f}+-{100*sd[3]:.2f}%  corr(dx,k_v) {C[0,3]/sd[0]/sd[3]:+.2f}"
    return o


for st in stamps:
    fi = f"{R}/noise_innov/innov_{st}.csv"
    if not os.path.exists(fi): print(f"== {st}: no {fi}"); continue
    d = pd.read_csv(fi)
    d = d[np.isfinite(d.odom_th) & np.isfinite(d.d_fwd) & np.isfinite(d.z_x)].reset_index(drop=True)
    print(f"\n== {st}: {len(d)} fed cycles")
    print("  1 per-cycle fit (KinematicMount rows)  ", fmt(*fit(d)[:2]))
    dth, df = d.odom_th.values, d.odom_fwd.values
    turning = np.abs(dth) > 1e-3
    rad = np.abs(df) / np.maximum(np.abs(dth), 1e-9)
    piv, arc = turning & (rad < 0.15), turning & (rad >= 0.15)
    for lab, m in [("CCW turns + straights", (dth > 0) | ~turning), ("CW turns + straights", (dth < 0) | ~turning),
                   ("pivots + straights", piv | ~turning), ("arcs + straights", arc | ~turning),
                   ("CCW arcs + straights", (arc & (dth > 0)) | ~turning), ("CW arcs + straights", (arc & (dth < 0)) | ~turning)]:
        print(f"  2 {lab:36s}", fmt(*fit(d[m].reset_index(drop=True))[:2]))
    mvf = d.x12_fwd.values * d.mv_fwd.values; mvl = d.x13_lat.values * d.mv_lat.values
    print("  3 without the mv*p_move*dt term in m   ", fmt(*fit(d, (1 - d.r_out_fwd) / (d.m_fwd - mvf), (1 - d.r_out_lat) / (d.m_lat - mvl))[:2]))
    one = np.ones(len(d))
    print("  3 unweighted (OLS)                     ", fmt(*fit(d, one / np.median(d.m_fwd), one / np.median(d.m_lat))[:2]))
    print("  3 no outlier down-weighting            ", fmt(*fit(d, 1 / d.m_fwd.values, 1 / d.m_lat.values)[:2]))
    p, sd, C = fit(d, kv=True)
    print("  4 + forward-scale column               ", fmt(p, sd, C))
    fh = nearest(f"{R}/heading/heading_*.csv", st)
    fs = nearest(f"{R}/sdf_localizer/log_*.csv", st)
    h = pd.read_csv(fh, comment="#") if fh else None
    if h is not None:
        m = d.merge(h[["ts_ms", "v_eps_yaw", "calib_applied"]], on="ts_ms", how="left")
        ca = m.calib_applied.fillna(0).astype(int).values
        print(f"  5 eps_yaw ACTING on the odometry in {100*np.mean((ca >> 1) & 1):.0f}% of cycles, "
              f"estimate median {np.degrees(np.nanmedian(m.v_eps_yaw)):+.3f} deg ({os.path.basename(fh)})")
    if h is None or fs is None: continue
    s = pd.read_csv(fs, usecols=["ts_ms", "early_exit", "pred_x", "pred_y", "res_x", "res_y"])
    m = d.merge(h[["ts_ms", "gt_x", "gt_y", "gt_th"]], on="ts_ms").merge(s, on="ts_ms")
    m = m[np.isfinite(m.gt_x) & np.isfinite(m.pred_x)].reset_index(drop=True)
    t = m.ts_ms.values.astype(float); tau = 30.0     # GT is logged ~30 ms off the scan stamp (fitted on the heading)
    gx, gy = np.interp(t + tau, t, m.gt_x.values), np.interp(t + tau, t, m.gt_y.values)
    off = np.angle(np.mean(np.exp(1j * (m.gt_th - m.z_th))))
    th = np.interp(t + tau, t, np.unwrap(m.gt_th.values - off))
    ee = m.early_exit.values == 1
    pol = (np.abs(m.res_x - m.pred_x) + np.abs(m.res_y - m.pred_y)) > 1e-6
    n = ee.sum(); c, sn = np.cos(th[ee]), np.sin(th[ee])
    X = np.zeros((2 * n, 5)); y = np.r_[(m.z_x - m.pred_x).values[ee], (m.z_y - m.pred_y).values[ee]]
    X[:n, 0] = gx[ee] - m.pred_x.values[ee]; X[n:, 0] = gy[ee] - m.pred_y.values[ee]
    X[:n, 1] = c; X[:n, 2] = -sn; X[n:, 1] = sn; X[n:, 2] = c; X[:n, 3] = 1; X[n:, 4] = 1
    g = np.linalg.lstsq(X, y, rcond=None)[0][0]
    print(f"  6 early-exit cycles {100*ee.mean():.1f}% (polished {100*pol[ee].mean():.1f}%): z - pred = g (gt - pred), g = {g:.3f}")
    # 7 un-attenuate and refit (same rows, same weights)
    th_m = m.z_th.values - 0.5 * m.odom_th.values; snm, csm = np.sin(th_m), np.cos(th_m)
    fwd, lat = np.c_[-snm, csm], np.c_[csm, snm]
    ow = m.odom_fwd.values[:, None] * fwd + m.odom_lat.values[:, None] * lat
    dw = m.d_fwd.values[:, None] * fwd + m.d_lat.values[:, None] * lat
    z = np.c_[m.z_x, m.z_y]; pred = np.c_[m.pred_x, m.pred_y]
    zs = z.copy(); zs[ee] = pred[ee] + (z[ee] - pred[ee]) / g
    cont = np.r_[False, np.linalg.norm((z - ow - dw)[1:] - z[:-1], axis=1) < 1e-4]
    dws = zs - np.r_[[[np.nan, np.nan]], zs[:-1]] - ow
    ok = cont & np.isfinite(dws).all(1)
    mm = m[ok].reset_index(drop=True)
    print("  7 same rows, z as logged               ", fmt(*fit(mm)[:2]))
    print("  7 same rows, z un-attenuated by g      ", fmt(*fit(mm, dfwd=np.sum(dws * fwd, 1)[ok], dlat=np.sum(dws * lat, 1)[ok])[:2]))
    # telescoped 1 s blocks, world frame: sum delta = [R(th_e) - R(th_s)] t - psi J O + k O
    J = np.array([[0, -1], [1, 0]]); Rm = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    zprev_th = m.z_th.values - m.odom_th.values - m.d_th.values
    rows, ys = [], []
    K = 20
    for s0 in range(1, len(m) - K, K):
        e = s0 + K - 1
        if not cont[s0:e + 1].all(): continue
        A = Rm(m.z_th.values[e]) - Rm(zprev_th[s0]); O = ow[s0:e + 1].sum(0)
        rows.append(np.c_[A, -(J @ O), O]); ys.append(dw[s0:e + 1].sum(0))
    Xb, yb = np.vstack(rows), np.concatenate(ys)
    pb = np.linalg.lstsq(Xb, yb, rcond=None)[0]; rb = yb - Xb @ pb
    Cb = np.linalg.inv(Xb.T @ Xb) * (rb @ rb) / (len(yb) - 4); sb = np.sqrt(np.diag(Cb))
    print(f"  7 telescoped {K}-cycle blocks (OLS)       ", fmt(pb[:3], sb[:3]))
