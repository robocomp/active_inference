#!/usr/bin/env python3
"""heading_fusion_report.py — read tmp/heading/heading_*.csv and judge the wheel x gyro fusion.

    python3 tools/heading_fusion_report.py                 # newest run
    python3 tools/heading_fusion_report.py a.csv b.csv     # several runs (e.g. fusion on vs off)

Per run it reports:
  1. what the fusion did: gyro weight while turning fast / turning slowly / stopped;
  2. PARKED heading walk: how fast the PREDICTED heading turns while the wheels read stationary
     (the risk flagged in config.toml), and how fast truth turns there (sim only);
  3. channel scales against truth (sim only): regress d(gt_th) on raw wheel and raw gyro rotation
     over 2 s windows -- the wheel's and the gyro's own scale errors, measured independently;
  4. prediction quality: heading correction the optimiser had to apply per radian turned and per
     second, i.e. how good the motion prior was;
  5. the calibrator at the end: value, sigma, informed, for every parameter.
Pure numpy; no pandas.
"""
import glob, os, sys
import numpy as np


def load(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    head = lines[0].strip().split(",")
    rows = [l.strip().split(",") for l in lines[1:] if l.strip()]
    rows = [r for r in rows if len(r) == len(head)]
    a = np.array([[float(x) if x not in ("", "nan", "-nan") else np.nan for x in r] for r in rows])
    return {h: a[:, i] for i, h in enumerate(head)}, len(rows)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def report(path):
    d, n = load(path)
    print(f"\n=== {os.path.basename(path)}  ({n} cycles, "
          f"{(d['ts_ms'][-1] - d['ts_ms'][0]) / 1000:.0f} s, fusion={int(np.nanmedian(d['fusion']))}) ===")
    if n < 50:
        print("  too few cycles"); return
    dt = d["total_dt"]
    ok = dt > 0
    rate = np.where(ok, np.abs(d["raw_wheel"]) / np.maximum(dt, 1e-6), np.nan)
    parked = ok & (d["zupt_dt"] >= 0.9 * dt)
    slow = ok & ~parked & (rate < 0.3)
    fast = ok & (rate >= 0.3)

    # 1. weights
    print("1. gyro weight (time-weighted mean per class; 1 = all gyro, 0 = all wheels)")
    for name, m in (("fast turn  >=0.3 rad/s", fast), ("slow / straight", slow), ("parked (ZUPT)", parked)):
        w = d["gyro_weight"][m & (d["gyro_weight"] >= 0)]
        tt = dt[m].sum()
        print(f"   {name:24s} {tt:7.1f} s   weight {np.mean(w) if len(w) else float('nan'):.3f}"
              f"   dens_w {np.nanmean(np.where(d['dens_w'][m] >= 0, d['dens_w'][m], np.nan)) if m.any() else float('nan'):.4f}"
              f"   dens_g {np.nanmean(np.where(d['dens_g'][m] >= 0, d['dens_g'][m], np.nan)) if m.any() else float('nan'):.4f}")

    # 2. parked walk: predicted heading change while parked
    pk_t = dt[parked].sum()
    if pk_t > 1:
        walk = d["dth_total"][parked].sum() / pk_t
        print(f"2. PARKED {pk_t:.0f} s: predicted heading walks {np.degrees(walk) * 60:+.2f} deg/min"
              f"  (|cycle| rms {np.degrees(np.sqrt(np.mean(d['dth_total'][parked] ** 2))):.4f} deg)")
        g = d["gt_th"]
        if np.isfinite(g).sum() > 10:
            dg = wrap(np.diff(g))
            pm = parked[1:] & np.isfinite(dg)
            if pm.any():
                print(f"   truth over the same cycles walks {np.degrees(dg[pm].sum() / dt[1:][pm].sum()) * 60:+.2f} deg/min")
    else:
        print("2. no parked time in this run")

    # 3. channel scales vs truth (sim)
    g = d["gt_th"]
    if np.isfinite(g).sum() > 100:
        t = (d["ts_ms"] - d["ts_ms"][0]) / 1000.0
        edges = np.arange(0, t[-1], 2.0)
        idx = np.digitize(t, edges)
        Y, Xw, Xg, T = [], [], [], []
        for k in np.unique(idx):
            m = np.where(idx == k)[0]
            if len(m) < 5: continue
            m = m[1:]  # increments need a previous row
            if not np.isfinite(g[m - 1]).all() or not np.isfinite(g[m]).all(): continue
            full = d["gyro_dt"][m].sum() >= 0.95 * dt[m].sum()   # gyro bracketed the whole window
            if not full: continue
            Y.append(wrap(g[m] - g[m - 1]).sum()); Xw.append(d["raw_wheel"][m].sum())
            Xg.append(d["raw_gyro"][m].sum()); T.append(dt[m].sum())
        Y, Xw, Xg, T = map(np.array, (Y, Xw, Xg, T))
        if len(Y) > 20 and np.abs(Xw).sum() > 1:
            kw = np.linalg.lstsq(Xw[:, None], Y, rcond=None)[0][0]
            A = np.stack([Xg, -T], 1)
            kg, bg = np.linalg.lstsq(A, Y, rcond=None)[0]
            print(f"3. vs TRUTH over {len(Y)} 2-s windows:  wheel scale {kw:.4f}  (k_omega_w = {kw - 1:+.4f})"
                  f"   gyro scale {kg:.4f} (k_omega = {kg - 1:+.4f})  gyro bias {bg:+.2e} rad/s")
    else:
        print("3. no ground truth (real robot): channel scales from truth skipped")

    # 4. prediction quality. Two traps measured on the first live log (2026-10-01): rows with no
    # prediction yet carry pred = 0, and during start-up the room frame is RE-ANCHORED (est-gt jumped
    # 60 -> -90 deg), so absolute headings are not comparable across a run. Both are handled by
    # skipping pred-less rows and grading INCREMENTS over windows, which a re-anchor cannot touch
    # except in the one window that contains it -- hence medians, not sums.
    has_pred = ~((d["pred_x"] == 0) & (d["pred_y"] == 0) & (d["pred_th"] == 0))
    corr = np.abs(wrap(d["est_th"] - d["pred_th"]))[has_pred]
    print(f"4. optimiser heading correction per cycle (deg): median {np.degrees(np.median(corr)):.3f}"
          f"  p90 {np.degrees(np.percentile(corr, 90)):.3f}  p99 {np.degrees(np.percentile(corr, 99)):.3f}"
          f"  ({(~has_pred).sum()} rows without a prediction skipped)")
    if np.isfinite(g).sum() > 100:
        t = (d["ts_ms"] - d["ts_ms"][0]) / 1000.0
        for win in (2.0, 10.0):
            k = np.searchsorted(t, np.arange(t[0], t[-1] - win, win))
            k2 = np.searchsorted(t, t[k] + win)
            m = (k2 < len(t)) & np.isfinite(g[k]) & np.isfinite(g[np.minimum(k2, len(t) - 1)])
            k, k2 = k[m], k2[m]
            err = np.degrees(np.abs(wrap((d["est_th"][k2] - d["est_th"][k]) - (g[k2] - g[k]))))
            perr = np.degrees(np.abs(wrap(np.array([d["dth_total"][i + 1:j + 1].sum() for i, j in zip(k, k2)])
                                          - (g[k2] - g[k]))))
            print(f"   |heading-increment error| over {win:.0f} s windows (deg, median / p90):"
                  f"  published {np.median(err):.3f} / {np.percentile(err, 90):.3f}"
                  f"   motion prior alone {np.median(perr):.3f} / {np.percentile(perr, 90):.3f}")

    # 5. calibrator at the end
    names = [h[2:] for h in d if h.startswith("v_")]
    mask = int(d["calib_informed_mask"][-1])
    print(f"5. calibrator at end ({int(d['calib_episodes'][-1])} episodes, cond {d['calib_cond'][-1]:.1f}):")
    for i, nme in enumerate(names):
        print(f"   {nme:10s} {d['v_' + nme][-1]:+.5f} ± {d['s_' + nme][-1]:.5f}  {'informed' if mask >> i & 1 else '-'}")


def rest_report(d):
    """6. Rest on the prediction (PreintZuptOnPrediction): wander while TRULY still, and the moving
    prediction error that a rest hypothesis must not raise. Needs ground truth."""
    g = d.get("gt_th")
    if g is None or np.isfinite(g).sum() < 100:
        print("6. no ground truth: rest/wander section skipped"); return
    on = int(np.nanmax(d["rest_on"])) if "rest_on" in d else 0
    t = (d["ts_ms"] - d["ts_ms"][0]) / 1000.0
    gx, gy, gth = d["gt_x"], d["gt_y"], np.unwrap(d["gt_th"])
    ex, ey, eth = d["est_x"], d["est_y"], np.unwrap(d["est_th"])
    # truly still: truth moves < 1 mm and < 0.01 deg within each 5-s block; stretches of >= 30 s
    blk = (t // 5).astype(int); still_blk = []
    for b in np.unique(blk):
        m = blk == b
        if m.sum() < 20: continue
        if np.hypot(np.ptp(gx[m]), np.ptp(gy[m])) < 1e-3 and np.degrees(np.ptp(gth[m])) < 0.01: still_blk.append(b)
    runs, cur = [], []
    for b in still_blk:
        if cur and b != cur[-1] + 1: runs.append(cur); cur = []
        cur.append(b)
    if cur: runs.append(cur)
    runs = [r for r in runs if len(r) >= 6]
    print(f"6. REST ON PREDICTION = {on}.  truly-still stretches >= 30 s: {len(runs)}")
    tot_t = 0; wx = []; wth = []
    for r in runs:
        m = np.isin(blk, r); T = t[m][-1] - t[m][0]; tot_t += T
        dxy = 1e3 * np.hypot(ex[m] - ex[m][0], ey[m] - ey[m][0]).max()
        dth = np.degrees(np.abs(eth[m] - eth[m][0]).max())
        wx.append(dxy / (T / 60)); wth.append(dth / (T / 60))
        print(f"   {t[m][0]/60:5.1f}-{t[m][-1]/60:5.1f} min: estimate wandered up to {dxy:6.1f} mm, {dth:.3f} deg"
              f"   (truth {1e3*np.hypot(np.ptp(gx[m]),np.ptp(gy[m])):.1f} mm, {np.degrees(np.ptp(gth[m])):.3f} deg)")
    if runs:
        print(f"   still time {tot_t/60:.1f} min; wander rate median {np.median(wx):.1f} mm/min, {np.median(wth):.3f} deg/min")
    if on and "rest_gain_tr" in d:
        gt_, gr_ = d["rest_gain_tr"], d["rest_gain_ro"]
        sm = np.isin(blk, sum(runs, [])) if runs else np.zeros(len(t), bool)
        print(f"   gain P(moving) while still: tr p50 {np.nanmedian(gt_[sm]) if sm.any() else float('nan'):.3f}  ro p50 {np.nanmedian(gr_[sm]) if sm.any() else float('nan'):.3f}"
              f" | while moving: tr p10 {np.nanpercentile(gt_[~sm],10):.3f}  ro p10 {np.nanpercentile(gr_[~sm],10):.3f}")
    # moving prediction error: predicted increment (pred - previous est) against truth, per 2-s window
    pdx = d["pred_x"][1:] - ex[:-1]; pdy = d["pred_y"][1:] - ey[:-1]
    has = ~((d["pred_x"][1:] == 0) & (d["pred_y"][1:] == 0))
    gdx = np.diff(gx); gdy = np.diff(gy)
    w2 = (t[1:] // 2).astype(int); errs = []
    for b in np.unique(w2):
        m = (w2 == b) & has
        if m.sum() < 10: continue
        path = np.hypot(gdx[m], gdy[m]).sum()
        if path < 0.05: continue          # moving windows only (>= 2.5 cm/s)
        errs.append(abs(np.hypot(pdx[m], pdy[m]).sum() - path) / path)
    if errs:
        print(f"   moving windows: |predicted path - true path| / true path  median {100*np.median(errs):.2f} %  p90 {100*np.percentile(errs,90):.2f} %  (n={len(errs)})")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "..", "tmp", "heading", "heading_*.csv")))[-1:]
    if not paths:
        sys.exit("no tmp/heading/heading_*.csv yet — run room_concept first")
    for p in paths:
        report(p)
        rest_report(load(p)[0])
