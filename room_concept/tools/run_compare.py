#!/usr/bin/env python3
"""Compare localiser runs on motion-normalised metrics.

usage: run_compare.py A.csv "label A" B.csv "label B" ...

Every figure is per unit of MOTION, because a run that spent longer parked would otherwise
look better for free. Reports:
  opt/m   optimizer firings per metre driven   -- how often the gate had to be rescued
  err/m   distance-weighted mean prediction error -- the area under the sawtooth
  mm/m, mm/rad   RAMP STEEPNESS: how fast the error grows per unit of motion inside one ramp.
                 This separates "the tooth is steeper" from "the tooth starts higher", which
                 opt/m alone conflates and which turned out to be the whole question.
  floor   median error immediately after a correction -- what the optimizer cannot remove.

Windows where the optimizer was pinned (>50% of cycles) are excluded and reported separately:
that pathological burst has recurred several times and smears any average it lands in.
"""
import csv, sys, numpy as np

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rows.append({k: float(v) for k, v in r.items()})
            except ValueError: pass
    return rows

def analyse(path, lab):
    rows = load(path)
    ts = np.array([r['ts_ms'] for r in rows]); it = np.array([r['iters'] for r in rows])
    mse = np.array([r['sdf_mse'] for r in rows])
    gx = np.array([r['gt_x'] for r in rows]); gy = np.array([r['gt_y'] for r in rows])
    gth = np.array([r['gt_theta'] for r in rows])
    t = (ts - ts[0]) / 1000.0
    chg = np.r_[True, (np.diff(gx) != 0) | (np.diff(gy) != 0) | (np.diff(gth) != 0)]
    GX = np.interp(t, t[chg], gx[chg]); GY = np.interp(t, t[chg], gy[chg])
    GT = np.interp(t, t[chg], np.unwrap(gth)[chg])
    step = np.r_[0, np.hypot(np.diff(GX), np.diff(GY))]
    keep = np.zeros(len(t), bool); burst_m = 0.0
    for a, b in zip(np.arange(0, t[-1] + 60, 60)[:-1], np.arange(0, t[-1] + 60, 60)[1:]):
        m = (t >= a) & (t < b)
        if m.sum() < 30: continue
        if (it[m] > 0).mean() > 0.5: burst_m += step[m].sum(); continue
        keep |= m
    d = step[keep].sum()
    if d < 5: return print(f"{lab:34s}  too little motion ({d:.1f} m)")
    optm = (it[keep] > 0).sum() / d
    errm = (mse[keep] * step[keep]).sum() / d
    fired = np.where((it > 0) & np.r_[False, it[:-1] == 0])[0]
    floor = np.median(mse[np.minimum(fired + 1, len(mse) - 1)]) if len(fired) > 3 else float('nan')
    z = it == 0; A = []; B = []; i = 0
    while i < len(z):
        if z[i]:
            a = i
            while i < len(z) and z[i]: i += 1
            b = i
            if b - a >= 8:
                dd = np.r_[0, np.cumsum(np.hypot(np.diff(GX[a:b]), np.diff(GY[a:b])))]
                rr = np.r_[0, np.cumsum(np.abs(np.diff(GT[a:b])))]
                if dd[-1] > 0.3 or rr[-1] > 0.3:
                    sol, *_ = np.linalg.lstsq(np.c_[dd, rr], mse[a:b] - mse[a], rcond=None)
                    if dd[-1] > 0.3: A.append(sol[0])
                    if rr[-1] > 0.3: B.append(sol[1])
        else: i += 1
    sa = np.median(A) * 1000 if len(A) > 3 else float('nan')
    sb = np.median(B) * 1000 if len(B) > 3 else float('nan')
    print(f"{lab:34s} {d:7.1f} {optm:7.2f} {errm:8.4f} {sa:8.2f} {sb:9.2f} {floor:8.4f}"
          f"  {('burst %.0f m' % burst_m) if burst_m > 1 else ''}")

print(f"{'run':34s} {'m':>7} {'opt/m':>7} {'err/m':>8} {'mm/m':>8} {'mm/rad':>9} {'floor':>8}")
for p, l in zip(sys.argv[1::2], sys.argv[2::2]):
    analyse(p, l)
