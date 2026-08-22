#!/usr/bin/env python3
"""room_concept's published pose vs the Webots supervisor pose (tmp/sdf_localizer/gt_error.csv).

★ CIRCULAR statistics throughout. A linear regression on headings is wrong the moment the run
covers more than ~180 deg: it wraps at +-pi and reports a fictitious gain and a huge residual
(measured 2026-08-22: gain 0.82, "residual" 265 deg, all of it an artefact of the fit).

★ The two poses are in DIFFERENT FRAMES -- GT is world, the estimate is the room frame, whose
orientation room_concept picks for itself. A CONSTANT offset is therefore expected and benign.
What matters is whether the offset is constant. It has been observed to sit in two stable MODES
36 deg apart, so the mean offset alone is meaningless -- the mode breakdown is the real output.
"""
import csv, math, os, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'tmp/sdf_localizer/gt_error.csv'
if not os.path.exists(path):
    sys.exit("no %s -- is robot_concept publishing robot_gt_* (simulation only)?" % path)

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi
def num(r, k):
    try: return float(r[k])
    except (TypeError, ValueError, KeyError): return None

R = []
for r in csv.DictReader(open(path)):
    t, g, e, m = (num(r, k) for k in ('ts_ms', 'gt_theta', 'est_theta', 'sdf_mse'))
    it, cv = num(r, 'iters'), num(r, 'cov_tt')
    if None in (t, g, e, m): continue
    R.append((t, math.degrees(wrap(wrap(e) - wrap(g))), m, it, cv))
R.sort()

# Converged frames only: a badly fitted pose is not evidence about the localiser's accuracy.
MSE_MAX = float(os.environ.get('GT_MSE_MAX', '0.06'))

# converged rows only, as (ts, gt_theta, est_theta) — used by the rotation-tracking section below.
rot = []
for _r in csv.DictReader(open(path)):
    _t, _g, _e, _m = (num(_r, k) for k in ('ts_ms', 'gt_theta', 'est_theta', 'sdf_mse'))
    if None in (_t, _g, _e, _m) or _m >= MSE_MAX:
        continue
    rot.append((_t, wrap(_g), wrap(_e)))
rot.sort()
if len(R) < 30: sys.exit("only %d rows -- drive around first" % len(R))
t0 = R[0][0]
print("rows %d over %.0f s" % (len(R), (R[-1][0]-t0)/1000))

# --- modes: histogram peaks, then assign within +-10 deg -------------------------------------
BIN = 5
h = {}
for _, d, *_ in R: h[int((d+180)//BIN)] = h.get(int((d+180)//BIN), 0) + 1
peaks = sorted(h.items(), key=lambda kv: -kv[1])
centres, used = [], []
for b, n in peaks:
    c = b*BIN - 180 + BIN/2
    if n < 0.04*len(R): break
    if all(abs(c-u) > 20 for u in used): centres.append(c); used.append(c)
centres.sort()

print("\nmode    n     occ     sdf_mse p50   sdf p95   early-exit   fails 0.06 gate")
for c in centres:
    sel = [x for x in R if abs(x[1]-c) <= 10]
    mse = sorted(x[2] for x in sel)
    its = [x[3] for x in sel if x[3] is not None]
    ee  = 100*sum(1 for i in its if i <= 0)/len(its) if its else -1
    bad = 100*sum(1 for x in sel if x[2] >= 0.06)/len(sel)
    print(" %+6.1f %4d  %5.1f%%    %8.4f  %8.4f      %4.0f%%        %5.1f%%"
          % (c, len(sel), 100*len(sel)/len(R), mse[len(mse)//2], mse[int(0.95*len(mse))], ee, bad))
tr = [x for x in R if all(abs(x[1]-c) > 10 for c in centres)]
if tr:
    mse = sorted(x[2] for x in tr)
    print(" %-6s %4d  %5.1f%%    %8.4f  %8.4f   (transitions)"
          % ("trans", len(tr), 100*len(tr)/len(R), mse[len(mse)//2], mse[int(0.95*len(mse))]))

if len(centres) > 1:
    print("\n  ⚠ %d MODES, %.0f deg apart. The fit is landing in more than one minimum, so the mean"
          % (len(centres), max(centres)-min(centres)))
    print("    offset is meaningless and any yaw calibration below that spread is unmeasurable.")
    best = min(centres, key=lambda c: st.median([x[2] for x in R if abs(x[1]-c) <= 10]))
    print("    Lowest-residual mode is %+.1f deg -- if the robot settles elsewhere, it is choosing"
          % best)
    print("    the WORSE minimum, which points at the early-exit gate, not at any mounting angle.")
else:
    c = centres[0]
    sel = [x[1] for x in R if abs(x[1]-c) <= 10]
    S = sum(math.sin(math.radians(x)) for x in sel); C = sum(math.cos(math.radians(x)) for x in sel)
    mu = math.degrees(math.atan2(S, C)); Rbar = math.hypot(S, C)/len(sel)
    res = sorted(abs(math.degrees(wrap(math.radians(x-mu)))) for x in sel)
    print("\n  single stable mode: offset %+.2f deg, concentration %.4f" % (mu, Rbar))
    print("  residual about it: p50 %.2f  p95 %.2f  max %.2f deg  <- the localiser's real yaw error"
          % (res[len(res)//2], res[int(0.95*len(res))], res[-1]))

# --- rotation tracking, with the FRAME CONVENTION divided out ----------------------------------
# ★ The room frame is built through the SVG loader's mirror_x (room = -x_svg, y_svg), so it is
# LEFT-handed relative to the world and est_theta runs BACKWARDS against gt_theta BY CONVENTION.
# A negative gain here is EXPECTED and is not a defect. What must hold is that the two TRACK:
# cumulative rotations proportional, |gain| = 1.
# ★ Do NOT judge this on per-frame increments. The kinematic clamp spreads each correction over
# several frames, so the instantaneous slope sits well below 1 (measured 0.43) while the integral is
# fine. Only the CUMULATIVE comparison means anything.
cum_gt, cum_est = [0.0], [0.0]
for i in range(1, len(rot)):
    cum_gt.append(cum_gt[-1] + wrap(rot[i][1] - rot[i-1][1]))
    cum_est.append(cum_est[-1] + wrap(rot[i][2] - rot[i-1][2]))

print("\n--- rotation tracking (frame convention divided out) ---")
print("  %d converged rows; net rotation: GT %+.1f deg, est %+.1f deg"
      % (len(rot), math.degrees(cum_gt[-1]), math.degrees(cum_est[-1])))
if abs(cum_gt[-1]) < math.radians(30):
    print("  ⚠ only %.0f deg of net rotation — TURN THE ROBOT and re-run."
          % math.degrees(abs(cum_gt[-1])))
    print("    Below ~30 deg the gain is dominated by noise (a stationary run once read 4.17).")
else:
    n = len(cum_gt); mx = sum(cum_gt)/n; my = sum(cum_est)/n
    sxx = sum((x-mx)**2 for x in cum_gt)
    gain = sum((x-mx)*(y-my) for x, y in zip(cum_gt, cum_est))/sxx
    res = [y - (gain*x + (my - gain*mx)) for x, y in zip(cum_gt, cum_est)]
    rms = math.sqrt(sum(r*r for r in res)/n)
    print("  handedness   %-5s   (mirror_x convention => LEFT expected, not a fault)"
          % ("LEFT" if gain < 0 else "right"))
    print("  |gain|       %.4f   (1.0 = the estimate turns exactly as much as truth)" % abs(gain))
    print("  residual     rms %.2f deg, max %.2f deg   <- the real rotation error"
          % (math.degrees(rms), math.degrees(max(abs(r) for r in res))))
