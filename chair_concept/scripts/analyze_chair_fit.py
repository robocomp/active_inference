#!/usr/bin/env python3
"""Analyze a chair_concept AI2 run — repeatable fit metric (mirrors table_concept/scripts/analyze_table_fit.py).

Chair state θ = [cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h].
GT: EDIT the dims below to your Webots chair (defaults ≈ WoodenChair). seat_w/seat_d/seat_h/back_h are
SIZES (frame-independent). cz is the floor height in the ROOM frame (≈ −ROBOT_BASE_Z, the robot-base
datum, ~−0.03 m). cx,cy have no absolute GT offline → report POSITION STABILITY. A chair has a defined
front (the backrest), so yaw is FULLY DETERMINED — no symmetry fold; yaw drift is a plain circular std.

Usage:  python3 analyze_chair_fit.py [etc/ai2_log.csv]
"""
import csv, sys, math, statistics as st
from collections import Counter, defaultdict

ROBOT_BASE_Z = 0.0296                       # robot-base z above the world floor (room z=0 datum)
GT = dict(seat_w=0.45, seat_d=0.45, seat_h=0.45, back_h=0.45)   # ← EDIT to the real chair
GT_CZ = -ROBOT_BASE_Z                        # chair floor in the room frame
CONV_THRESH_M = 0.06                         # "converged" = all size errors < this, sustained

path = sys.argv[1] if len(sys.argv) > 1 else "etc/ai2_log.csv"
all_rows = list(csv.DictReader(open(path)))
if not all_rows:
    print("empty CSV"); sys.exit(1)
def F(r, k): return float(r[k])
def pct(c, p): c = sorted(c); return c[min(len(c)-1, int(p*len(c)))] if c else 0.0
def mean(xs): return sum(xs)/len(xs) if xs else 0.0

by_node = defaultdict(list)
for r in all_rows:
    by_node[r["node"]].append(r)
nodes = sorted(by_node, key=lambda k: -len(by_node[k]))
print(f"file={path}  instances={len(nodes)}: " + ", ".join(f"{k}({len(by_node[k])})" for k in nodes))

def analyze(node, rows):
    n = len(rows)
    has = lambda k: k in rows[0]
    col = lambda k: [F(r, k) for r in rows if has(k)]
    print(f"\n{'='*60}\nINSTANCE {node}  cycles={n}")
    tail = rows[int(0.8*n):] or rows[-1:]

    # size DOFs vs GT + consistency
    print(f"\nFINAL error (last {len(tail)} cycles, mean±std, cm):")
    def err(k, gt): return [abs(F(r, k) - gt) for r in tail]
    for k, gt in (("seat_w", GT["seat_w"]), ("seat_d", GT["seat_d"]),
                  ("seat_h", GT["seat_h"]), ("back_h", GT["back_h"]), ("cz", GT_CZ)):
        e = err(k, gt)
        print(f"  {k:7s} (GT {gt:+.3f}): {mean(e)*100:5.1f} ± {st.pstdev(e)*100:4.1f}")
    cxs = [F(r, "cx") for r in tail]; cys = [F(r, "cy") for r in tail]
    print(f"  centre STABILITY (room-frame XY std, no absolute GT): "
          f"{st.pstdev(cxs)*100 if len(cxs)>1 else 0:.1f} / {st.pstdev(cys)*100 if len(cys)>1 else 0:.1f} cm")

    f = tail[-1]
    print(f"\nFINAL estimate: cx={F(f,'cx'):.2f} cy={F(f,'cy'):.2f} cz={F(f,'cz'):.3f} yaw={F(f,'yaw'):.3f} "
          f"sw={F(f,'seat_w'):.3f} sd={F(f,'seat_d'):.3f} sh={F(f,'seat_h'):.3f} bh={F(f,'back_h'):.3f}")
    print("FINAL posterior std (cm): " + " ".join(
        f"{k}={mean([F(r,'std_'+k) for r in tail])*100:.1f}"
        for k in ("cx","cy","cz","sw","sd","sh","bh")) +
        f" yaw={math.degrees(mean([F(r,'std_yaw') for r in tail])):.1f}°")

    # consistency: size |err|/σ (<2 good)
    def z(e, s): return e/s if s > 1e-6 else float("inf")
    con = []
    for k, sk in (("seat_w","std_sw"),("seat_d","std_sd"),("seat_h","std_sh"),("back_h","std_bh")):
        con.append(f"{k}={z(abs(F(f,k)-GT[k]), F(f,sk)):.1f}")
    print("CONSISTENCY (final |err|/σ, <2 good): " + " ".join(con))

    # convergence
    conv = None
    for i in range(n):
        if all(max(abs(F(r,k)-GT[k]) for k in GT) < CONV_THRESH_M for r in rows[i:]):
            conv = i; break
    print(f"\nCONVERGENCE (<{CONV_THRESH_M*100:.0f} cm all size DOFs, sustained): "
          + (f"cycle {conv}/{n} (~{100*conv/n:.0f}% in)" if conv is not None else "NOT reached"))

    # gates + load
    g = col("gated"); mv = col("motion_var"); npts = col("npts"); rng = col("range") if has("range") else []
    en = col("energy")
    print(f"\nGATES/LOAD: gated={100*sum(g)/n:.0f}%  npts med={pct(npts,.5):.0f}  motion_var p90={pct(mv,.9):.4f}"
          + (f"  range med={pct(rng,.5):.1f}m p90={pct(rng,.9):.1f}m" if rng else ""))
    print(f"energy: med={pct(en,.5):.3f} p90={pct(en,.9):.3f}  (fit residual; want low+stable)")

    # CONTAMINATION: clutter_frac = the mixture's own estimate of the fraction of the mask it can't explain
    # (off-model points, e.g. the table bleeding into a chair mask). The decisive read is the correlation
    # with position JUMPS: a >15 cm centroid jump with HIGH clutter ⇒ contamination the clutter rejected but
    # that still dragged the centroid; with LOW clutter ⇒ the model absorbed the table points (worse).
    if has("clutter_frac"):
        cl = col("clutter_frac")
        jumps = [(i, math.hypot(F(rows[i],"cx")-F(rows[i-1],"cx"), F(rows[i],"cy")-F(rows[i-1],"cy")))
                 for i in range(1, n)]
        big = [i for i, d in jumps if d > 0.15]
        cl_at_jump = mean([F(rows[i],"clutter_frac") for i in big]) if big else 0.0
        print(f"clutter_frac: med={100*pct(cl,.5):.0f}% p90={100*pct(cl,.9):.0f}%  | "
              f"pos-jumps>15cm: {len(big)}/{n} (clutter@jump={100*cl_at_jump:.0f}% vs baseline {100*pct(cl,.5):.0f}%)")

    # RANGE vs YAW STABILITY — a chair's yaw is fully determined (backrest), so NO fold: drift = plain
    # circular std. Measured on POST-LOCK rows only (skip everything up to the last >20° yaw jump — the
    # one-time orientation-lock transient), else the drift mixes the pre-lock seed with the settled value
    # and reports a huge, meaningless number. Checks: claimed σyaw GROWS with range (the yaw cap); actual
    # post-lock drift LOW & flat across range (a far view widens σ but must not rotate the chair).
    if has("range") and has("yaw"):
        def dyaw(a, b): return math.degrees(abs(math.atan2(math.sin(a-b), math.cos(a-b))))
        ys = [F(r, "yaw") for r in rows]
        settle = 0
        for i in range(1, len(ys)):
            if dyaw(ys[i], ys[i-1]) > 20.0: settle = i   # last big orientation jump
        crows = rows[settle:]
        def circ_std_deg(a):
            if len(a) < 2: return 0.0
            cs = mean([math.cos(y) for y in a]); sn = mean([math.sin(y) for y in a])
            Rr = math.hypot(cs, sn)
            return 180.0 if Rr <= 1e-9 else math.degrees(math.sqrt(max(0.0, -2*math.log(Rr))))
        print(f"\nRANGE vs YAW STABILITY (post-lock rows {settle}..{n}; drift = circular std):")
        for lo, hi, lab in [(0,2,"<2m"),(2,4,"2-4m"),(4,6,"4-6m"),(6,1e9,">6m")]:
            b = [r for r in crows if lo <= F(r,"range") < hi]
            if not b: continue
            drift = circ_std_deg([F(r,"yaw") for r in b])
            sy = math.degrees(mean([F(r,"std_yaw") for r in b]))
            print(f"  {lab:6s} {len(b):4d}   drift={drift:6.1f}°   claimed-σyaw={sy:5.1f}°")

for nd in nodes:
    analyze(nd, by_node[nd])

# ── Cross-instance tracker health (row order = global clock; 'cycle' is per-instance) ─────────
if len(nodes) >= 1:
    print(f"\n{'='*60}\nTRACKER HEALTH")
    idx = {id(r): i for i, r in enumerate(all_rows)}
    rng = {nd: (min(idx[id(r)] for r in rs), max(idx[id(r)] for r in rs)) for nd, rs in by_node.items()}
    last = len(all_rows) - 1
    live = [sum(1 for nd in nodes if rng[nd][0] <= i <= rng[nd][1]) for i in range(len(all_rows))]
    print(f"  instances seen: {len(nodes)}  | live at once: med={int(st.median(live))} max={max(live)}"
          + ("   ⚠ OVER-SEGMENTED?" if len(nodes) > 1 else ""))
    ctr = {nd: (mean([F(r,"cx") for r in rs]), mean([F(r,"cy") for r in rs])) for nd, rs in by_node.items()}
    for nd in nodes:
        b, e = rng[nd]; cx, cy = ctr[nd]
        stale = "" if e >= last-1 else f"  last obs @row {e}/{last} (stale — looked away, not deleted)"
        print(f"    {nd:10s}: rows {b}..{e} ({len(by_node[nd])})  centroid=({cx:+.2f},{cy:+.2f})  "
              + ("from start" if b <= 1 else f"born @row {b}") + stale)
    # cluster by centroid: two instances closer than a seat-diagonal are the SAME physical chair split.
    OVERLAP_M = 0.60   # chairs ~0.5 m; centres closer than this ⇒ fragmentation (want 1 instance/chair)
    clusters = []
    for nd in nodes:
        cx, cy = ctr[nd]
        for cl in clusters:
            if math.hypot(cx-cl[0][0], cy-cl[0][1]) < OVERLAP_M:
                cl[1].append(nd); break
        else:
            clusters.append([(cx, cy), [nd]])
    print(f"  centroid clusters (physical chairs): {len(clusters)}")
    for (cx, cy), members in clusters:
        tag = "✓ single instance" if len(members) == 1 else f"⚠ FRAGMENTED into {len(members)}: {members}"
        print(f"    ({cx:+.2f},{cy:+.2f}): {tag}")
