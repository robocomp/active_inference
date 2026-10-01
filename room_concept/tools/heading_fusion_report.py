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

    # 4. prediction quality: optimiser's heading correction
    corr = wrap(d["est_th"] - d["pred_th"])
    turned = np.abs(d["dth_total"]).sum()
    print(f"4. heading correction applied by the optimiser: total |corr| {np.degrees(np.abs(corr).sum()):.1f} deg"
          f" over {np.degrees(turned):.0f} deg turned  -> {np.abs(corr).sum() / max(turned, 1e-6) * 100:.2f} % per rad,"
          f"  {np.degrees(np.abs(corr).sum()) / max(dt.sum(), 1e-6) * 60:.2f} deg/min")
    if np.isfinite(g).sum() > 100:
        e = wrap(d["est_th"] - g); e = e[np.isfinite(e)]
        e = wrap(e - np.angle(np.mean(np.exp(1j * e))))
        print(f"   estimated heading vs truth (offset removed): rms {np.degrees(np.sqrt(np.mean(e ** 2))):.3f} deg")

    # 5. calibrator at the end
    names = [h[2:] for h in d if h.startswith("v_")]
    mask = int(d["calib_informed_mask"][-1])
    print(f"5. calibrator at end ({int(d['calib_episodes'][-1])} episodes, cond {d['calib_cond'][-1]:.1f}):")
    for i, nme in enumerate(names):
        print(f"   {nme:10s} {d['v_' + nme][-1]:+.5f} ± {d['s_' + nme][-1]:.5f}  {'informed' if mask >> i & 1 else '-'}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "..", "tmp", "heading", "heading_*.csv")))[-1:]
    if not paths:
        sys.exit("no tmp/heading/heading_*.csv yet — run room_concept first")
    for p in paths:
        report(p)
