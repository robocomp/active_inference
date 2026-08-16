#!/usr/bin/env python3
"""Map the coverage of the kitchen corner region from a live kitchen_cells.csv.

★WHY A MAP AND NOT A PROBE. The first version of this check sampled ONE point, 10 cm in from both
walls, and asked whether exactly one run covered it. It reported "OK (exactly once)" on a run whose
corner had a 0.39 x 0.69 m hole in it — the single sample happened to land in the one strip that was
covered. A one-point probe cannot distinguish "the corner is filled" from "the corner is filled
where I happened to look", and it reads as a pass either way, which is worse than no check at all.

Prints a coverage grid (. = uncovered, N = covered by N runs) plus the largest uncovered rectangle,
so a hole shows up as a shape rather than as a number that happens to be 1.

    python3 corner_map.py [path/to/kitchen_cells.csv]
"""
import csv
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "../etc/kitchen_cells.csv"
rows = [r for r in csv.DictReader(open(path)) if r["active"] == "1"]
if not rows:
    print("no active cells")
    raise SystemExit(1)
last = max(int(r["cycle"]) for r in rows)
cells = [r for r in rows if int(r["cycle"]) == last and r["tier"] == "0"]
print(f"cycle {last}, {len(cells)} active base cells")

def geom(r):
    return tuple(float(r[k]) for k in ("cx", "cy", "yaw", "L", "d"))

def inside(g, px, py):
    cx, cy, yaw, L, d = g
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - cx, py - cy
    return abs(c * dx + s * dy) <= L / 2 and abs(-s * dx + c * dy) <= d / 2

for r in cells:
    cx, cy, yaw, L, d = geom(r)
    print(f"  {r['cell']:22s} cx={cx:7.3f} cy={cy:7.3f} yaw={yaw:6.3f} L={L:5.3f} d={d:5.3f}")

# Bound the map to the runs themselves, padded by a depth so the corner void is inside the window.
xs_all, ys_all = [], []
for r in cells:
    cx, cy, yaw, L, d = geom(r)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            c, s = math.cos(yaw), math.sin(yaw)
            xs_all.append(cx + c * sx * L - s * sy * d)
            ys_all.append(cy + s * sx * L + c * sy * d)
pad = max(float(r["d"]) for r in cells)
x0, x1 = min(xs_all) - pad, max(xs_all) + pad
y0, y1 = min(ys_all) - pad, max(ys_all) + pad
step = 0.05
nx = int((x1 - x0) / step) + 1
ny = int((y1 - y0) / step) + 1
grid = [[sum(1 for r in cells if inside(geom(r), x0 + i * step, y0 + j * step))
         for i in range(nx)] for j in range(ny)]

print(f"\ncoverage, x {x0:.2f}..{x1:.2f}  y {y0:.2f}..{y1:.2f}  step {step} m   (. = 0)")
for j in range(ny - 1, -1, -1):
    print(f"y={y0 + j * step:6.2f} " + "".join("." if n == 0 else str(min(n, 9)) for n in grid[j]))

# ★THE QUESTION IS ABOUT THE CORNER SQUARE, NOT THE WINDOW. Scanning the whole map for the largest
# empty rectangle finds the open floor (6.8 m^2 of room), which is not a defect and drowns the thing
# we care about. So ask the model's own question: for each perpendicular pair, the corner square is
# (one run's depth band) x (the other's), and it must be covered exactly once.
def chart_of(r):
    cx, cy, yaw, L, d = geom(r)
    t0, t1 = float(r["t0"]), float(r["t1"])
    u = (math.cos(yaw), math.sin(yaw))
    n = (-math.sin(yaw), math.cos(yaw))
    low = (cx - L / 2 * u[0], cy - L / 2 * u[1])
    A = (low[0] - t0 * u[0] - d / 2 * n[0], low[1] - t0 * u[1] - d / 2 * n[1])
    return A, u, n, d

print()
for a in range(len(cells)):
    for b in range(a + 1, len(cells)):
        Aa, ua, na, da = chart_of(cells[a])
        Ab, ub, nb, db = chart_of(cells[b])
        if abs(ua[0] * ub[0] + ua[1] * ub[1]) > 0.5:
            continue                                   # not perpendicular
        den = ua[0] * (-ub[1]) - ua[1] * (-ub[0])
        if abs(den) < 1e-9:
            continue
        rhs = (Ab[0] - Aa[0], Ab[1] - Aa[1])
        s_ = (rhs[0] * (-ub[1]) - rhs[1] * (-ub[0])) / den
        V = (Aa[0] + s_ * ua[0], Aa[1] + s_ * ua[1])   # the wall LINES cross here
        # the square spans V by each run's depth, along its own inward normal
        tot = cov1 = cov0 = cov2 = 0
        m = 12
        for i in range(m + 1):
            for j in range(m + 1):
                px = V[0] + na[0] * da * i / m + nb[0] * db * j / m
                py = V[1] + na[1] * da * i / m + nb[1] * db * j / m
                n_ = sum(1 for r in cells if inside(geom(r), px, py))
                tot += 1
                cov0 += n_ == 0
                cov1 += n_ == 1
                cov2 += n_ > 1
        verdict = ("OK" if cov1 == tot else
                   f"HOLE {100*cov0//tot}% uncovered" + (f", {100*cov2//tot}% doubled" if cov2 else ""))
        print(f"corner {cells[a]['cell']} x {cells[b]['cell']}  V=({V[0]:.3f},{V[1]:.3f})  "
              f"square {da:.2f}x{db:.2f} m -> {verdict}")
overl = sum(1 for j in range(ny) for i in range(nx) if grid[j][i] > 1)
print(f"\ndoubly-covered samples in the window: {overl} of {nx * ny}")
