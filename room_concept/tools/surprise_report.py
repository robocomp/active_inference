#!/usr/bin/env python3
"""Per-run surprise report: how much the room evidence had to tell the motion model (surprise.h).

    tools/surprise_report.py [heading_<start>.csv ...]     (default: the newest in tmp/heading/)

For each run:
  * KL total, per second and per metre driven (est path), and per scored correction (median);
  * the calibration ratio  sum(mismatch) / sum(expected): ~1 calibrated, >1 the prediction is biased or
    over-confident, <1 under-confident; with a bootstrap 90% interval over corrections;
  * the same split into thirds of the run, which is where online self-calibration should show (later
    thirds cheaper than the first, ratio drifting toward 1).
Several files are printed side by side, so an A/B (calibration on/off) reads in one table.
"""
import csv, glob, math, os, random, sys

def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    if not rows or 'sur_kl' not in rows[0]:
        return None
    out = []
    for r in rows:
        try:
            out.append(dict(t=float(r['ts_ms']) / 1000.0, x=float(r['est_x']), y=float(r['est_y']),
                            sc=r['sur_scored'] == '1', kl=float(r['sur_kl']), mis=float(r['sur_mismatch']),
                            exp=float(r['sur_expected']), info=float(r['sur_info'])))
        except (ValueError, KeyError):
            pass   # a truncated last line
    return out

def path_len(rows):
    return sum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(rows, rows[1:]))

def ratio_ci(ev, n=2000, seed=1):
    if len(ev) < 2:
        return float('nan'), float('nan')
    rng = random.Random(seed)
    rs = []
    for _ in range(n):
        s = [ev[rng.randrange(len(ev))] for _ in ev]
        e = sum(x['exp'] for x in s)
        if e > 0:
            rs.append(sum(x['mis'] for x in s) / e)
    rs.sort()
    return rs[int(0.05 * len(rs))], rs[int(0.95 * len(rs)) - 1]

def summary(rows):
    T = rows[-1]['t'] - rows[0]['t'] if len(rows) > 1 else 0.0
    L = path_len(rows)
    ev = [r for r in rows if r['sc'] and math.isfinite(r['kl'])]
    kl = sum(r['kl'] for r in ev); mis = sum(r['mis'] for r in ev); exp = sum(r['exp'] for r in ev)
    med = sorted(r['kl'] for r in ev)[len(ev) // 2] if ev else float('nan')
    lo, hi = ratio_ci(ev)
    nan = float('nan')
    return dict(T=T, L=L, n=len(rows), nsc=len(ev), kl=kl, kl_s=kl / T if T > 0 else nan,
                kl_m=kl / L if L > 0.05 else nan, med=med, mis=mis, info=kl - mis,
                ratio=mis / exp if exp > 0 else nan, lo=lo, hi=hi)

def fmt(v, f='{:.3g}'):
    return '--' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v)

def main():
    files = sys.argv[1:] or sorted(glob.glob('tmp/heading/heading_*.csv'), key=os.path.getmtime)[-1:]
    if not files:
        sys.exit('no tmp/heading/heading_*.csv')
    lines = [('run', 'dur s', 'path m', 'corrections', 'KL nats', 'nats/s', 'nats/m',
              'median/corr', 'mismatch', 'info', 'mis/expected [90%]')]
    thirds = []
    for p in files:
        rows = load(p)
        name = os.path.basename(p).removeprefix('heading_').removesuffix('.csv')
        if rows is None:
            print(f'{name}: no sur_* columns (logged before the surprise measure existed)'); continue
        if len(rows) < 2:
            print(f'{name}: too few rows'); continue
        s = summary(rows)
        lines.append((name, fmt(s['T'], '{:.0f}'), fmt(s['L'], '{:.1f}'), str(s['nsc']), fmt(s['kl']),
                      fmt(s['kl_s']), fmt(s['kl_m']), fmt(s['med']), fmt(s['mis']), fmt(s['info']),
                      f"{fmt(s['ratio'], '{:.2f}')} [{fmt(s['lo'], '{:.2f}')}, {fmt(s['hi'], '{:.2f}')}]"))
        # thirds of the DISTANCE driven, not of time: a leg that parks after its route would put all
        # of its driving in the first third
        cum = [0.0]
        for a, b in zip(rows, rows[1:]):
            cum.append(cum[-1] + math.hypot(b['x'] - a['x'], b['y'] - a['y']))
        cut = [0] + [next(i for i, c in enumerate(cum) if c >= cum[-1] * k / 3) for k in (1, 2)] + [len(rows)]
        parts = [rows[cut[k]:cut[k + 1]] for k in range(3)]
        thirds.append((name, [summary(q) if len(q) > 1 else None for q in parts]))
    if len(lines) > 1:
        w = [max(len(l[i]) for l in lines) for i in range(len(lines[0]))]
        for l in lines:
            print('  '.join(c.rjust(w[i]) for i, c in enumerate(l)))
    for name, parts in thirds:
        print(f'\n{name} by thirds of the distance driven   (nats/s | nats/m | mis/expected)')
        for k, s in enumerate(parts):
            if s is None: continue
            print(f'  {k + 1}/3  {fmt(s["kl_s"]):>8} | {fmt(s["kl_m"]):>8} | {fmt(s["ratio"], "{:.2f}"):>5}'
                  f'   ({s["nsc"]} corrections, {s["L"]:.1f} m)')

def per_axis(path):
    """Which AXIS is mis-scaled? For each body axis, sum(c^2) / sum(P_pred - P_post): ~1 honest, <1 the motion
    covariance is too wide on that axis, >1 too narrow. Split by open-loop length, because a per-cycle floor
    that adds up linearly shows as a ratio that FALLS with the number of cycles accumulated."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    if not rows or 'sur_c_fwd' not in rows[0]:
        return
    ev = []
    for r in rows:
        if r.get('sur_scored') != '1': continue
        try:
            ev.append({k: float(r[k]) for k in ('sur_open', 'sur_c_fwd', 'sur_c_lat', 'sur_c_th', 'sur_pp_fwd',
                                                 'sur_pp_lat', 'sur_pp_th', 'sur_pq_fwd', 'sur_pq_lat', 'sur_pq_th')})
        except ValueError:
            pass
    if not ev: return
    print(f'\n{os.path.basename(path)} per axis: sum(c^2)/sum(P_pred-P_post)   (~1 honest, <1 too wide, >1 too narrow)')
    print('  open-loop cycles     n    forward   lateral   heading')
    for lo, hi in ((1, 2), (2, 5), (5, 40), (40, 400), (400, 10**9)):
        e = [x for x in ev if lo <= x['sur_open'] < hi]
        if not e: continue
        def ratio(a):
            den = sum(max(x[f'sur_pp_{a}'] - x[f'sur_pq_{a}'], 0.0) for x in e)
            return sum(x[f'sur_c_{a}'] ** 2 for x in e) / den if den > 0 else float('nan')
        print(f'  {lo:>5}-{hi if hi < 10**9 else "inf":<6}  {len(e):6d}   {ratio("fwd"):7.3f}   {ratio("lat"):7.3f}   {ratio("th"):7.3f}')
    # the learnt scale (motion_cov_scale.h): where kappa stood at each third of the run, and the flags in force
    if 'k_fwd' in rows[0]:
        ks = []
        for r in rows:
            try: ks.append((float(r['k_fwd']), float(r['k_lat']), float(r['k_th'])))
            except ValueError: pass
        if ks:
            print(f"  MotionPreintegration={rows[0].get('preint')}  MotionCovLearn={rows[0].get('cov_learn')}"
                  f"   kappa (fwd, lat, th) at 1/3 2/3 end:")
            for k in (len(ks) // 3, 2 * len(ks) // 3, len(ks) - 1):
                print(f'    row {k:6d}   {ks[k][0]:7.3f}   {ks[k][1]:7.3f}   {ks[k][2]:7.3f}')


if __name__ == '__main__':
    main()
    for p in (sys.argv[1:] or sorted(glob.glob('tmp/heading/heading_*.csv'), key=os.path.getmtime)[-1:]):
        per_axis(p)
