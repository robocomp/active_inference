#!/usr/bin/env python3
"""Room 2 is a KNOWN RECTANGLE, so it can be graded without aligning anything.

Edge lengths and corner angles do not depend on where the estimator put its map frame, so the
comparison below needs no rigid fit -- which matters, because a rigid fit absorbs the global pose
error and would hide exactly what we are trying to measure.

Truth: 6.000 m x 4.000 m, from the WALL SOLIDS of tworooms-piso.wbt (west inner face x = -7.770,
east -1.770, south y = 9.480, north 13.480; thickness 0.110).

⚠ This file used to say 6.22 x 4.39 "with a 3 cm ambiguity". That was the FLOOR SLAB, which extends
under the walls, and it is retracted in datasets/webots_piso/ROOM2.md. The slab figure made a 18 cm
width error look like 4 cm, i.e. read a badly over-confident estimate as a comfortable one. A
variable named for a room is not a statement about which surface it describes -- check the walls.
"""
import math, sys, statistics as st

W_TRUE, H_TRUE, H_AMBIG = 6.000, 4.000, 0.0   # wall solids: exact, no ambiguity
path = sys.argv[1] if len(sys.argv) > 1 else 'tmp/layout_trace.csv'
rows = [l.strip().split(';') for l in open(path) if l.strip() and not l.startswith('#')]
if not rows:
    print("empty trace"); sys.exit(0)
ts0 = int(rows[0][1])

def parse(r):
    verts = [tuple(float(c) for c in v.split(',')) for v in r[3].split() if ',' in v]
    sig   = [float(x) for x in r[4].split()] if r[4].strip() else []
    return verts, sig, r[6]

print(f"{len(rows)} frames over {(int(rows[-1][1])-ts0)/1000:.0f} s\n")
print(f"  {'frame':>6s} {'t(s)':>6s} {'n':>3s} {'edges (m, sorted)':>34s} {'worst sig':>9s} {'med sig':>8s} {'pub':>4s}")
step = max(1, len(rows)//14)
for i in list(range(0, len(rows), step)) + [len(rows)-1]:
    v, s, cp = parse(rows[i])
    if not v: continue
    e = sorted(math.dist(v[k], v[(k+1) % len(v)]) for k in range(len(v)))
    txt = " ".join(f"{x:5.2f}" for x in e[:6]) + (" ..." if len(e) > 6 else "")
    print(f"  {rows[i][0]:>6s} {(int(rows[i][1])-ts0)/1000:6.0f} {len(v):3d} {txt:>34s} "
          f"{max(s) if s else 0:9.3f} {st.median(s) if s else 0:8.3f} {cp:>4s}")

# ── the measurement, on frames that actually look like a 4-corner room ──────────────────────
good = [r for r in rows if len(parse(r)[0]) == 4]
print(f"\n  frames with a 4-corner polygon: {len(good)}/{len(rows)}")
if good:
    print(f"\n  {'t(s)':>6s} {'width':>7s} {'height':>7s} {'w err':>7s} {'h err':>8s} {'sigma':>7s} {'err/sigma':>9s}")
    sel = good[::max(1, len(good)//12)] + [good[-1]]
    for r in sel:
        v, s, _ = parse(r)
        e = [math.dist(v[k], v[(k+1) % 4]) for k in range(4)]
        w = (e[0] + e[2]) / 2; h = (e[1] + e[3]) / 2
        if w < h: w, h = h, w
        we = w - W_TRUE
        he = min(abs(h - H_TRUE), abs(h - (H_TRUE - H_AMBIG))) * (1 if h > H_TRUE else -1)
        sg = st.median(s) if s else float('nan')
        print(f"  {(int(r[1])-ts0)/1000:6.0f} {w:7.3f} {h:7.3f} {we:+7.3f} {he:+8.3f} {sg:7.4f} "
              f"{abs(we)/sg if sg > 0 else float('nan'):9.2f}")
    v, s, _ = parse(good[-1])
    e = [math.dist(v[k], v[(k+1) % 4]) for k in range(4)]
    w = max((e[0]+e[2])/2, (e[1]+e[3])/2); h = min((e[0]+e[2])/2, (e[1]+e[3])/2)
    ang = []
    for k in range(4):
        a = (v[(k-1) % 4][0]-v[k][0], v[(k-1) % 4][1]-v[k][1])
        b = (v[(k+1) % 4][0]-v[k][0], v[(k+1) % 4][1]-v[k][1])
        ang.append(math.degrees(math.acos(max(-1, min(1, (a[0]*b[0]+a[1]*b[1]) /
                    (math.hypot(*a)*math.hypot(*b)))))))
    print(f"\n  FINAL: {w:.3f} x {h:.3f} m   (truth 6.000 x 4.000)")
    print(f"         width error {w-W_TRUE:+.3f} m, height error {h-H_TRUE:+.3f} m")
    print(f"         corner angles {' '.join(f'{a:.1f}' for a in ang)} deg")
    print(f"         declared sigma: median {st.median(s):.4f}  worst {max(s):.4f} m")
