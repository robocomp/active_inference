#!/usr/bin/env python3
"""Analyze a table_concept AI2 run vs ground truth — repeatable experiment metric.

Ground truth: table(11) in hallway_and_beta_empty.wbt = size 1.5 x 1.4 x 0.74 m (world frame).
IMPORTANT z datum: the belief works in the ROOM frame, whose z=0 is anchored at the robot
BASE, which sits ROBOT_BASE_Z above the world floor (Shadow translation z in the .wbt). So the
room-frame tabletop height is the world height MINUS the base offset — compare H against that,
not the raw 0.74, or you bake in a spurious ~3 cm "bias". The table's room-frame XY position is
NOT the room origin (table(11) is at world (-3.9,-11)); without the live room transform we can't
set an absolute centre GT, so we report POSITION STABILITY (does cx,cy stay put) instead.

Two-table test: two IDENTICAL 1.5x1.4 tables, well separated. The fit per table is the easy
part; the test is the TRACKER — two distinct stable instances (no spurious duplicate spawned
on one table, no wrong MERGE fusing the two). Same size GT applies to every table_* node. The
cross-instance section reports instance count over time, each instance's birth cycle, and the
pairwise centroid separation (a SMALL separation between two live instances ⇒ a duplicate of one
table; an instance vanishing mid-run ⇒ a wrong merge/death).

Goal: the estimated size (w,h,H) and pose converge to GT with shrinking, calibrated
uncertainty, robustly under the controller's active-perception motion; the degradation
gates (motion downweight via R, truncation gate) help rather than hurt.

Usage:  python3 analyze_table_fit.py [etc/ai2_log.csv]
"""
import csv, sys, math, statistics as st
from collections import Counter, defaultdict

ROBOT_BASE_Z = 0.0296                              # Shadow base z above world floor (room z=0 datum)
GT_LONG, GT_SHORT = 1.5, 1.4                       # the two footprint dims (unordered)
GT_H = 0.74 - ROBOT_BASE_Z                          # room-frame tabletop height (world 0.74 − base)
CONV_THRESH_M = 0.05                                # "converged" = all size errors < 5 cm and sustained
DUP_SEP_M = 0.50                                    # two live instances closer than this ⇒ likely duplicate

path = sys.argv[1] if len(sys.argv) > 1 else "etc/ai2_log.csv"
all_rows = list(csv.DictReader(open(path)))
if not all_rows:
    print("empty CSV"); sys.exit(1)
def F(r, k): return float(r[k])
def pct(c, p): c = sorted(c); return c[min(len(c)-1, int(p*len(c)))] if c else 0.0
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def size_errs(r):
    w, h, H = F(r,"w"), F(r,"h"), F(r,"H")
    return (abs(max(w,h)-GT_LONG), abs(min(w,h)-GT_SHORT), abs(H-GT_H))

by_node = defaultdict(list)
for r in all_rows:
    by_node[r["node"]].append(r)
# analyze every table instance, biggest (most cycles) first
nodes = sorted(by_node, key=lambda k: -len(by_node[k]))
print(f"file={path}  instances={len(nodes)}: " + ", ".join(f"{k}({len(by_node[k])})" for k in nodes))

def analyze(node, rows):
    n = len(rows)
    has = lambda k: k in rows[0]
    col = lambda k: [F(r, k) for r in rows if has(k)]
    print(f"\n{'='*60}\nINSTANCE {node}  cycles={n}")
    tail = rows[int(0.8*n):] or rows[-1:]
    el = [size_errs(r)[0] for r in tail]; es = [size_errs(r)[1] for r in tail]
    eH = [size_errs(r)[2] for r in tail]
    cxs = [F(r,"cx") for r in tail]; cys = [F(r,"cy") for r in tail]
    print(f"\nFINAL error (last {len(tail)} cycles, mean±std, cm):")
    print(f"  long  (GT 1.50): {mean(el)*100:5.1f} ± {st.pstdev(el)*100:4.1f}")
    print(f"  short (GT 1.40): {mean(es)*100:5.1f} ± {st.pstdev(es)*100:4.1f}")
    print(f"  H  (GT {GT_H:.3f} room): {mean(eH)*100:5.1f} ± {st.pstdev(eH)*100:4.1f}   [world 0.74 − base {ROBOT_BASE_Z:.3f}]")
    print(f"  centre STABILITY (room-frame XY std, no absolute GT): "
          f"{st.pstdev(cxs)*100 if len(cxs)>1 else 0:.1f} / {st.pstdev(cys)*100 if len(cys)>1 else 0:.1f} cm")

    f = tail[-1]
    print(f"\nFINAL estimate: w={F(f,'w'):.3f} h={F(f,'h'):.3f} H={F(f,'H'):.3f} "
          f"cx={F(f,'cx'):.3f} cy={F(f,'cy'):.3f} yaw={F(f,'yaw'):.3f}")
    print(f"FINAL posterior std (cm): w={mean([F(r,'std_w') for r in tail])*100:.1f} "
          f"h={mean([F(r,'std_h') for r in tail])*100:.1f} "
          f"H={mean([F(r,'std_H') for r in tail])*100:.1f} "
          f"cx={mean([F(r,'std_cx') for r in tail])*100:.1f} cy={mean([F(r,'std_cy') for r in tail])*100:.1f}")

    # convergence: first cycle from which ALL size errors stay < threshold
    conv = None
    for i in range(n):
        if all(max(size_errs(r)[:3]) < CONV_THRESH_M for r in rows[i:]):
            conv = i; break
    print(f"\nCONVERGENCE (<{CONV_THRESH_M*100:.0f} cm all dims, sustained): "
          + (f"cycle {conv}/{n} (~{100*conv/n:.0f}% in)" if conv is not None else "NOT reached"))

    # consistency: GT inside ~2 sigma? (long/short matched to whichever of w,h is bigger)
    wbig = F(f,"w") >= F(f,"h")
    sigma_long  = F(f,"std_w") if wbig else F(f,"std_h")
    sigma_short = F(f,"std_h") if wbig else F(f,"std_w")
    def z(e, s): return e/s if s > 1e-6 else float("inf")
    print(f"CONSISTENCY (final |err|/σ, <2 good): long={z(size_errs(f)[0],sigma_long):.1f} "
          f"short={z(size_errs(f)[1],sigma_short):.1f} H={z(size_errs(f)[2],F(f,'std_H')):.1f}")

    # gate activity + load
    g = col("gated"); mv = col("motion_var"); tf = col("trunc_frac"); npts = col("npts")
    dd = col("motion_dotd") if has("motion_dotd") else []
    print(f"\nGATES/LOAD: gated={100*sum(g)/n:.0f}%  npts med={pct(npts,.5):.0f}  "
          f"motion_var p90={pct(mv,.9):.4f}  trunc_frac p90={pct(tf,.9):.2f}"
          + (f"  dotd p90={pct(dd,.9):.2f}" if dd else ""))
    en = col("energy")
    print(f"energy: med={pct(en,.5):.3f} p90={pct(en,.9):.3f}  (fit residual; want low+stable)")

    # Degraded (moving, e.g. corner runs) vs good-view (still) split — the core test of the gates:
    # while moving, the estimate should NOT drift and σ should NOT keep collapsing (honest uncertainty);
    # the close/still passes are where it should sharpen. Split on motion_dotd (camera ego-motion speed).
    if has("motion_dotd"):
        MOVE = 0.2  # m/s, camera ego-motion (analysis split only — not a control knob)
        mov = [r for r in rows if F(r, "motion_dotd") > MOVE]
        sta = [r for r in rows if F(r, "motion_dotd") <= MOVE]
        def seg(name, gg):
            if not gg:
                print(f"  {name:6s} n=0"); return
            sw = st.pstdev([F(r,"w") for r in gg]) if len(gg) > 1 else 0.0
            sh = st.pstdev([F(r,"h") for r in gg]) if len(gg) > 1 else 0.0
            sH = st.pstdev([F(r,"H") for r in gg]) if len(gg) > 1 else 0.0
            print(f"  {name:6s} n={len(gg):4d}  est-drift(w,h,H)cm=({sw*100:.1f},{sh*100:.1f},{sH*100:.1f})  "
                  f"claimed σw={mean([F(r,'std_w') for r in gg])*100:.1f}cm  gated={100*mean([F(r,'gated') for r in gg]):.0f}%  "
                  f"var_p90={pct([F(r,'motion_var') for r in gg],.9):.4f}")
        print(f"\nDEGRADED vs GOOD-VIEW (split at dotd>{MOVE} m/s):")
        print("  (est-drift = the estimate's own std within the segment — want SMALL while moving = fit protected)")
        seg("MOVING", mov)
        seg("STILL",  sta)

for nd in nodes:
    analyze(nd, by_node[nd])

# ── Cross-instance / tracker health (the point of the two-table test) ───────────────────────
# NOTE: the 'cycle' column is PER-INSTANCE (processed_cycles), not a global clock, so we use the
# CSV row order as the global timeline. An instance is "live" between its first and last row.
if len(nodes) >= 1:
    print(f"\n{'='*60}\nTRACKER HEALTH (two-table association test)")
    idx = {id(r): i for i, r in enumerate(all_rows)}      # global row index = time
    rng = {nd: (min(idx[id(r)] for r in rs), max(idx[id(r)] for r in rs)) for nd, rs in by_node.items()}
    last = len(all_rows) - 1
    # live count over the timeline: how many instances' [first,last] ranges cover each row
    live = [sum(1 for nd in nodes if rng[nd][0] <= i <= rng[nd][1]) for i in range(len(all_rows))]
    print(f"  instances seen: {len(nodes)} for 2 physical tables  | live at once: "
          f"med={int(st.median(live))} max={max(live)} (want a steady 2)"
          + ("   ⚠ OVER-SEGMENTED" if len(nodes) > 2 else ""))
    # mean centroid per instance (cx,cy are meaningful room-frame coords)
    ctr = {nd: (mean([F(r,"cx") for r in rs]), mean([F(r,"cy") for r in rs])) for nd, rs in by_node.items()}
    for nd in nodes:
        b, e = rng[nd]
        born = "from start" if b <= 1 else f"born @row {b}"
        # death is disabled in config, so a gap just means "no fresh observations" (robot looked away)
        stale = "" if e >= last - 1 else f"  last obs @row {e}/{last} (stale after — robot looked away, not deleted)"
        cx, cy = ctr[nd]
        print(f"    {nd:10s}: rows {b}..{e} ({len(by_node[nd])})  centroid=({cx:+.2f},{cy:+.2f})  {born}{stale}")
    # cluster instances by centroid: two instances whose tables would physically OVERLAP
    # (centres closer than ~a table diagonal) are the SAME physical table fragmented.
    OVERLAP_M = 0.5 * math.hypot(GT_LONG, GT_SHORT)       # ~1.0 m: closer than this ⇒ same table
    clusters = []
    for nd in nodes:
        cx, cy = ctr[nd]
        for cl in clusters:
            if math.hypot(cx-cl[0][0], cy-cl[0][1]) < OVERLAP_M:
                cl[1].append(nd); break
        else:
            clusters.append([(cx, cy), [nd]])
    print(f"  centroid clusters (physical tables): {len(clusters)}")
    for (cx, cy), members in clusters:
        tag = "✓ single instance" if len(members) == 1 else f"⚠ FRAGMENTED into {len(members)}: {members}"
        print(f"    ({cx:+.2f},{cy:+.2f}): {tag}")
