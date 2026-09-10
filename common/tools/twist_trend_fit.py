#!/usr/bin/env python3
"""twist_trend_fit — would a higher-order forward guess be worth building?

The pose handed to a consumer is often the last measured one walked forward along the robot's
velocity, assuming that velocity is CONSTANT over the gap. The obvious upgrade is a second-order
model: estimate acceleration and add half-a-t-squared. This tool decides whether that is worth
doing, on live data, before anyone writes it.

★DIFFERENCING TWO READINGS IS NOT THE SAME AS FITTING THROUGH N, and conflating them gives the
wrong answer. Differencing two adjacent twists amplifies noise (measured 2026-09-10: lag-1
correlation -0.41, 61% sign flips -- the signature of differentiated white noise). Fitting a trend
through N averages noise DOWN by about sqrt(N). So a "no" from the differencing test says nothing
about the fitting one; this tool tests the fit.

It reports three things per window length, and all three are needed:
  · R^2  -- how much of the speed variation a straight line explains. High means there IS a trend
            to recover; low means no window length will help.
  · residual scatter -- the noise floor left over, in m/s or rad/s.
  · median |a| -- the fitted acceleration, and the correction it implies at real gap lengths. THIS
            is the number that decides it: compare it against the measured error of the guess
            (controller overlay CSV, extrap_check_err_m) and see what fraction it could remove.

Measured 2026-09-10 on Shadow while driving: linear R^2 ~0.5-0.7 at N=10 (real trend), fitted
a ~0.35 m/s^2, implying corrections of 0.28 mm at a 40 ms gap and 4.0 mm at 150 ms -- against
measured guess errors of 1.9 mm and 17.3 mm. So the correction could remove at most a fifth of the
error, and with R^2 ~0.5 about half of what it applied would be noise. Verdict: real but not worth
building. The error at long gaps is NOT dominated by acceleration.

★A run where the robot is still proves nothing -- at rest every model fits equally well -- so
still snapshots are excluded and a fully-parked run reports no rows rather than a false pass.

  usage:  python3 twist_trend_fit.py
"""
import sys, math
from PySide6.QtCore import QCoreApplication, QTimer
import pydsr

app = QCoreApplication(sys.argv)
G = pydsr.DSRGraph(0, "twist_fit_test", 956)
samples = []

def attr(o, n):
    try: return list(o.attrs[n].value) if n in o.attrs else None
    except Exception: return None

def fit(ts, ys):
    n = len(ts)
    if n < 3: return None
    t0 = ts[0]
    x = [(t - t0) / 1000.0 for t in ts]
    mx = sum(x)/n; my = sum(ys)/n
    sxx = sum((a-mx)**2 for a in x); sxy = sum((a-mx)*(b-my) for a, b in zip(x, ys))
    if sxx <= 0: return None
    slope = sxy/sxx; inter = my - slope*mx
    ss_res = sum((b - (inter+slope*a))**2 for a, b in zip(x, ys))
    ss_tot = sum((b-my)**2 for b in ys)
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return slope, r2, math.sqrt(ss_res/n)

def grab():
    r = G.get_nodes_by_type("robot"); m = G.get_nodes_by_type("room")
    if not r or not m: return
    e = G.get_edge(r[0].id, m[0].id, "RT")
    if e is None: return
    ts = attr(e, "rt_timestamps"); twl = attr(e, "rt_twist_linear"); twa = attr(e, "rt_twist_angular")
    if not ts or not twl or not twa: return
    blocks = sorted((t, i) for i, t in enumerate(ts) if t != 0)
    if len(blocks) < 10: return
    T = [b[0] for b in blocks]
    VY = [twl[b[1]*3+1] for b in blocks]      # the dominant component
    WZ = [twa[b[1]*3+2] for b in blocks]
    speed = max(math.hypot(twl[b[1]*3], twl[b[1]*3+1]) for b in blocks)
    for lab, series in (("linear", VY), ("angular", WZ)):
        for N in (5, 10, 25):
            f = fit(T[-N:], series[-N:])
            if f: samples.append((lab, N, f[0], f[1], f[2], speed))

def report():
    if not samples: print("no samples"); return
    print("Straight-line fit through the last N twists in the ring:")
    print("  %-8s %3s  %5s  %10s  %13s  %10s" % ("channel","N","runs","median R^2","resid scatter","median |a|"))
    for lab in ("linear", "angular"):
        for N in (5, 10, 25):
            rows = [s for s in samples if s[0] == lab and s[1] == N and s[5] > 0.05]
            if not rows: continue
            r2 = sorted(x[3] for x in rows); sc = sorted(x[4] for x in rows)
            sl = sorted(abs(x[2]) for x in rows)
            a = sl[len(sl)//2]
            print("  %-8s %3d  %5d  %10.3f  %10.4f  %10.3f" % (lab, N, len(rows), r2[len(r2)//2], sc[len(sc)//2], a))
            if N == 10:
                unit = "mm" if lab == "linear" else "deg"
                k = 1000.0 if lab == "linear" else 57.2958
                print("     -> correction 0.5*a*dt^2 at 40/100/150 ms: %.2f / %.2f / %.2f %s"
                      % (0.5*a*0.04**2*k, 0.5*a*0.10**2*k, 0.5*a*0.15**2*k, unit))
    moving = [s for s in samples if s[5] > 0.05]
    print("\n  (%d of %d ring snapshots had the robot moving; still snapshots are excluded --" %
          (len(moving)//6 if moving else 0, len(samples)//6))
    print("   at rest every model fits equally well and the comparison means nothing)")
    app.quit()

QTimer.singleShot(1000, lambda: [QTimer.singleShot(300*i, grab) for i in range(60)])
QTimer.singleShot(20000, report)
sys.exit(app.exec())
