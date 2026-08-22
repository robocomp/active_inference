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

# --- static vs moving: separates a bias from lag/clamp transients -----------------------------
mv = []
for i in range(1, len(R)):
    dt = (R[i][0]-R[i-1][0])/1000.0
    if 0.005 < dt < 1.0:
        mv.append((abs(wrap(math.radians(R[i][1]-R[i-1][1])))/dt, R[i][1]))
still = [d for w, d in mv if w < 0.05]
if still and centres:
    c = min(centres, key=lambda c: abs(c - st.median(still)))
    res = sorted(abs(d-c) for d in still if abs(d-c) <= 20)
    if res:
        print("\n  at rest (|w|<0.05 rad/s, n=%d): residual p50 %.2f  p95 %.2f deg"
              % (len(res), res[len(res)//2], res[int(0.95*len(res))]))
