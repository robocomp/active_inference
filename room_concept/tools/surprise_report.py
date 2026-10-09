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
  * drift(): the SYSTEMATIC part of the prediction error per metre and per radian (GT-free self-calibration
    measure; calibration should drive it to 0) separated from the random densities.
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
    # the learnt noise components (motion_noise_vc.h): where they stood at each third of the run
    if 'vc_k_long' in rows[0]:
        cols = ('vc_k_long', 'vc_k_lat', 'vc_k_lat_turn', 'vc_k_th_turn', 'vc_k_t_trans', 'vc_k_t_rot',
                'vc_rho_fwd', 'vc_rho_lat', 'vc_rho_th')
        ks = []
        for r in rows:
            try: ks.append([float(r[c]) for c in cols])
            except (ValueError, KeyError): pass
        if ks:
            tr = sum(1 for r in rows if r.get('vc_trained') == '1' and r.get('sur_scored') == '1')
            print(f"  MotionNoiseProportional={rows[0].get('noise_prop')}  MotionNoiseLearn={rows[0].get('noise_learn')}"
                  f"  ({tr} training corrections)")
            print('    row     ' + ''.join(f'{c[3:]:>12}' for c in cols))
            for k in (0, len(ks) // 3, 2 * len(ks) // 3, len(ks) - 1):
                print(f'    {k:6d}  ' + ''.join(f'{v:12.3e}' for v in ks[k]))

def drift(path, min_open=5, n_boot=1000):
    """SELF-CALIBRATION, GT-free: how much of the odometry's prediction error is SYSTEMATIC.

    Per open-loop stretch (between two scored corrections) take the correction c (scan solve minus
    prediction, per body axis), the signed forward travel d (body frame, m) and the signed turn phi (rad)
    accumulated over it. Calibration errors (wheel scale, track, gyro scale, mount yaw) grow LINEARLY and
    with a sign; slip and sensor noise grow like a square root and average out. So, per axis:
        E[c]                 = b_d * d + b_phi * phi + b_t * T (systematic: what calibration must remove -> 0)
        E[(c - E[c])^2]      = r + a_d * |d| + a_phi * |phi|   (random densities: calibration leaves them)
    T is the stretch duration: a GYRO BIAS turns the heading per SECOND, even parked, and without this term
    parked stretches load their heading corrections onto a near-zero turn and report nonsense slopes.
    A raw "error per metre" mixes the two. Stretches shorter than min_open cycles are left out: those end
    because the scan disagreed (early-exit selection), not because a drift budget ran out. b is reported
    in mm/m, mm/rad (translation) and mrad/m, mrad/rad (heading), with a bootstrap 90 % interval."""
    import numpy as np
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    if not rows or 'sur_c_fwd' not in rows[0]:
        return
    th_key = 'est_th' if 'est_th' in rows[0] else 'est_theta'
    pth_key = 'pred_th' if 'pred_th' in rows[0] else 'pred_theta'
    S = []   # (t, d, phi, T, c_fwd, c_lat, c_th)
    d = phi = 0.0
    prev = None
    t0 = None
    for r in rows:
        try:
            x, y, th = float(r['est_x']), float(r['est_y']), float(r[th_key])
            px, py, pth = float(r['pred_x']), float(r['pred_y']), float(r[pth_key])
        except (ValueError, KeyError):
            continue
        # The covariates are the ODOMETRY's motion: this row's prediction minus the previous row's estimate.
        # Taking the estimated path instead leaks the correction itself into d and phi (c/phi -> 1 rad/rad).
        if prev is not None:
            dx, dy = px - prev[0], py - prev[1]
            d += -math.sin(prev[2]) * dx + math.cos(prev[2]) * dy          # body forward = (-sin th, cos th)
            phi += math.remainder(pth - prev[2], 2 * math.pi)
        prev = (x, y, th)
        t_now = float(r['ts_ms']) / 1000.0
        if t0 is None:
            t0 = t_now
        if r.get('sur_scored') == '1':
            try:
                op = float(r['sur_open'])
                c = (float(r['sur_c_fwd']), float(r['sur_c_lat']), float(r['sur_c_th']))
                if op >= min_open and all(math.isfinite(v) for v in c):
                    S.append((t_now, d, phi, t_now - t0) + c)
            except (ValueError, KeyError):
                pass
            d = phi = 0.0      # the correction resets the open-loop stretch
            t0 = t_now
    if len(S) < 8:
        print(f'\n{os.path.basename(path)} drift: only {len(S)} stretches >= {min_open} cycles -- not enough')
        return
    A = np.array(S)
    X = A[:, 1:4]

    def fit(Xs, cs):
        # pass 1: OLS for the mean; pass 2: variance model from the residuals; pass 3: WLS with it
        b = np.linalg.lstsq(Xs, cs, rcond=None)[0]
        res2 = (cs - Xs @ b) ** 2
        V = np.column_stack([np.ones(len(cs)), np.abs(Xs[:, :2])])
        a = np.clip(np.linalg.lstsq(V, res2, rcond=None)[0], 0.0, None)
        var = np.maximum(V @ a, 1e-12)
        w = 1.0 / var
        b = np.linalg.lstsq(Xs * np.sqrt(w)[:, None], cs * np.sqrt(w), rcond=None)[0]
        # standard errors from the WLS normal matrix: an UNEXCITED part (no travel / no turn) shows a huge
        # +-, instead of a confident nonsense slope
        se = np.sqrt(np.diag(np.linalg.pinv((Xs * w[:, None]).T @ Xs)))
        return b, a, se

    rng = np.random.default_rng(1)
    names = (('forward', 1e3, 'mm'), ('lateral', 1e3, 'mm'), ('heading', 1e3, 'mrad'))
    print(f'\n{os.path.basename(path)} SYSTEMATIC DRIFT (self-calibration residual), {len(S)} stretches >= {min_open} cycles,'
          f' {np.abs(X[:, 0]).sum():.1f} m / {math.degrees(np.abs(X[:, 1]).sum()):.0f} deg')
    print('  axis       b per metre [90%]            b per radian [90%]           b per second [90%]           random: per m      per rad    floor')
    for k, (nm, sc, u) in enumerate(names):
        cs = A[:, 4 + k]
        b, a, _ = fit(X, cs)
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(cs), len(cs))
            try: boot.append(fit(X[i], cs[i])[0])
            except np.linalg.LinAlgError: pass
        boot = np.array(boot)
        lo, hi = np.percentile(boot, 5, axis=0), np.percentile(boot, 95, axis=0)
        print(f'  {nm:8s} {sc*b[0]:+8.2f} {u}/m [{sc*lo[0]:+.2f},{sc*hi[0]:+.2f}]   '
              f'{sc*b[1]:+8.2f} {u}/rad [{sc*lo[1]:+.2f},{sc*hi[1]:+.2f}]   '
              f'{sc*b[2]:+8.3f} {u}/s [{sc*lo[2]:+.3f},{sc*hi[2]:+.3f}]   '
              f'{sc*math.sqrt(a[1]):6.2f} {u}/sqrt(m) {sc*math.sqrt(a[2]):6.2f} {u}/sqrt(rad) {sc*math.sqrt(a[0]):5.2f} {u}')
    # convergence: the same systematic fit by thirds of the run (in time)
    if len(S) >= 24:
        print('  by thirds (b +- se; fwd/lat per metre in mm/m, heading per radian in mrad/rad):')
        for j, part in enumerate(np.array_split(A, 3)):
            fs = [fit(part[:, 1:4], part[:, 4 + k]) for k in range(3)]
            print(f'    {j + 1}/3  fwd {1e3*fs[0][0][0]:+7.2f} +-{1e3*fs[0][2][0]:<7.2f} lat {1e3*fs[1][0][0]:+7.2f} +-{1e3*fs[1][2][0]:<7.2f}'
                  f' heading {1e3*fs[2][0][1]:+7.2f} +-{1e3*fs[2][2][1]:<7.2f} gyro-bias {1e3*fs[2][0][2]:+6.3f} mrad/s'
                  f' ({len(part)} stretches, {np.abs(part[:, 1]).sum():.1f} m, {math.degrees(np.abs(part[:, 2]).sum()):.0f} deg)')


if __name__ == '__main__':
    main()
    for p in (sys.argv[1:] or sorted(glob.glob('tmp/heading/heading_*.csv'), key=os.path.getmtime)[-1:]):
        per_axis(p)
        drift(p)
