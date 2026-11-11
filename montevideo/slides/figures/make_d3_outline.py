"""Day 3, Step 3: from the pass/hit grid to an outline, computed. Run from slides/ with ../.venv/bin/python.

A toy of layout_acquisition.tex l.526-560 on a simulated room: an L room turned 12 degrees from the grid,
three short cabinet fronts 30 cm off one wall, and a tall wardrobe standing free. Four scans fill a 5 cm
pass/hit grid; then the five steps: (1) the free region that contains the robot, holes filled; (2) the room's
axis from the quadrupled gradient angles; (3) trace the border in that frame and simplify it (18 cm);
(4) straight horizontal/vertical edges, corners where they meet; (5) each edge to the weighted median of the
hits near it (40 cm reach). Writes d3_outline_{a..d}.pdf (d: the result; the 5 cm snap is in the code, too small to see).
"""
from pathlib import Path

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label, map_coordinates

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})
COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"
rng = np.random.default_rng(11)

TH = np.deg2rad(12.0); OFF = np.array([0.8, 0.3])
Rw = np.array([[np.cos(TH), -np.sin(TH)], [np.sin(TH), np.cos(TH)]])
to_w = lambda p: np.asarray(p) @ Rw.T + OFF                     # room frame -> world (grid) frame
ROOM = np.array([(0, 0), (6, 0), (6, 2.5), (3.5, 2.5), (3.5, 4.5), (0, 4.5)], float)
segs = [(ROOM[i], ROOM[(i + 1) % 6]) for i in range(6)]
segs += [(np.array([x, 0.3]), np.array([x + 0.4, 0.3])) for x in (0.5, 1.6, 2.7)]        # cabinet fronts
W0 = np.array([4.4, 1.0]); segs += [(W0 + a, W0 + b) for a, b in [((0, 0), (.5, 0)), ((.5, 0), (.5, .5)), ((.5, .5), (0, .5)), ((0, .5), (0, 0))]]
SEGS = [(to_w(p), to_w(q)) for p, q in segs]
POSES = [to_w(p) for p in [(1.4, 1.3), (4.9, 0.6), (1.6, 3.6), (5.2, 2.0)]]

CELL, X0, Y0, NX, NY = 0.05, -0.6, -0.6, 170, 150
n = np.zeros((NY, NX)); h = np.zeros((NY, NX))
cell = lambda p: (int((p[1] - Y0) / CELL), int((p[0] - X0) / CELL))

for o in POSES:
    for a in np.linspace(-np.pi, np.pi, 1440, endpoint=False):
        d = np.array([np.cos(a), np.sin(a)]); best = np.inf
        for p, q in SEGS:
            e = q - p; M = np.array([d, -e]).T
            if abs(np.linalg.det(M)) < 1e-9: continue
            t, u = np.linalg.solve(M, p - o)
            if t > 0.05 and 0 <= u <= 1: best = min(best, t)
        if not np.isfinite(best) or best > 8: continue
        r = best + rng.normal(0, 0.01)
        seen = set()
        for t in np.arange(0, r - 0.03, CELL / 2):
            seen.add(cell(o + t * d))
        end = cell(o + r * d)
        seen.discard(end)
        for c in seen: n[c] += 1
        h[end] += 1

matter = h > n; free = (n > 0) & ~matter
# (1) the free region that contains the robot, closed by one cell, holes filled
lab, _ = label(binary_closing(free, iterations=1))
region0 = lab == lab[cell(POSES[0])]
region = binary_fill_holes(region0)
# (2) the axis: gradient directions at surface cells, angles x4, circular mean
# as free_space_polygon.h l.495-511: smooth the surface mask (>= 2 hits), take its gradient at EVERY cell,
# weight by its magnitude (both flanks of a wall line point opposite ways; x4 makes them agree)
sm = gaussian_filter((h >= 2).astype(float), 1.0); gy, gx = np.gradient(sm)
m = np.hypot(gx, gy) > 0
phi = np.arctan2(gy[m], gx[m]); w = np.hypot(gx[m], gy[m])
th0 = 0.25 * np.arctan2((w * np.sin(4 * phi)).sum(), (w * np.cos(4 * phi)).sum())
print(f"true axis {np.rad2deg(TH):.1f} deg, estimated {np.rad2deg(th0):.1f} deg")
# (3) resample the region into the axis frame, trace its border, simplify (Douglas-Peucker, 18 cm)
R0 = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
U0, V0, NU, NV = -1.0, -1.0, 170, 140
uu, vv = np.meshgrid(U0 + (np.arange(NU) + .5) * CELL, V0 + (np.arange(NV) + .5) * CELL)
wp = np.c_[uu.ravel(), vv.ravel()] @ R0.T
rot = map_coordinates(region.astype(float), [(wp[:, 1] - Y0) / CELL - .5, (wp[:, 0] - X0) / CELL - .5], order=0).reshape(NV, NU)
lines = contourpy.contour_generator(uu, vv, rot).lines(0.5)
trace = max(lines, key=len)


def dp(P, eps):
    if len(P) < 3: return P
    a, b = P[0], P[-1]; ab = b - a; L = np.hypot(*ab) or 1e-9
    d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    k = int(np.argmax(d))
    if d[k] > eps: return np.vstack([dp(P[:k + 1], eps)[:-1], dp(P[k:], eps)])
    return np.array([a, b])


k0 = int(np.argmin(trace[:, 0] + trace[:, 1]))
ring = np.vstack([trace[k0:], trace[1:k0 + 1]])
far = int(np.argmax(np.hypot(*(ring - ring[0]).T)))          # a closed ring: split it at its far point
simp = np.vstack([dp(ring[:far + 1], 0.18)[:-1], dp(ring[far:], 0.18)])
# (4) class each edge H or V, place it at its mean coordinate, merge neighbours, intersect
edges = []
for p, q in zip(simp[:-1], simp[1:]):
    dvec = q - p; L = np.hypot(*dvec)
    if L < 1e-6: continue
    if abs(dvec[0]) >= abs(dvec[1]): edges.append(["H", (p[1] + q[1]) / 2, L])
    else: edges.append(["V", (p[0] + q[0]) / 2, L])
merged = []
for e in edges:
    if merged and merged[-1][0] == e[0]:
        a = merged[-1]; tot = a[2] + e[2]; a[1] = (a[1] * a[2] + e[1] * e[2]) / tot; a[2] = tot
    else: merged.append(list(e))
if len(merged) > 1 and merged[0][0] == merged[-1][0]:
    a, e = merged[0], merged.pop(); tot = a[2] + e[2]; a[1] = (a[1] * a[2] + e[1] * e[2]) / tot; a[2] = tot


def corners(E):
    V = []
    for i in range(len(E)):
        e, f = E[i - 1], E[i]
        V.append((f[1], e[1]) if e[0] == "H" else (e[1], f[1]))
    return np.array(V)


V1 = corners(merged)
# (5) snap each edge to the hit-weighted median of the hits within 40 cm that project onto it
hy, hx = np.nonzero(h > 0); hw = h[hy, hx]
hp = (np.c_[X0 + (hx + .5) * CELL, Y0 + (hy + .5) * CELL]) @ R0          # world -> axis frame
snapped = [list(e) for e in merged]
for i, e in enumerate(merged):
    a, b = V1[i], V1[(i + 1) % len(V1)]
    if e[0] == "H":
        lo, hi = sorted([a[0], b[0]]); sel = (hp[:, 0] > lo) & (hp[:, 0] < hi) & (np.abs(hp[:, 1] - e[1]) < 0.4); off = hp[sel, 1]
    else:
        lo, hi = sorted([a[1], b[1]]); sel = (hp[:, 1] > lo) & (hp[:, 1] < hi) & (np.abs(hp[:, 0] - e[1]) < 0.4); off = hp[sel, 0]
    if sel.sum() > 3:
        ws = hw[sel]; o = np.argsort(off); cw = np.cumsum(ws[o])
        snapped[i][1] = off[o][np.searchsorted(cw, cw[-1] / 2)]
V2 = corners(snapped)
true_axis = (to_w(ROOM)) @ R0
print("corners", len(V1), "->", len(V2))


def fig(name, draw, size=(3.0, 2.6)):
    f, ax = plt.subplots(figsize=size); ax.set_aspect("equal"); ax.axis("off"); draw(ax)
    f.subplots_adjust(0, 0, 1, 1); f.savefig(HERE / name); plt.close(f)


EXT = (X0, X0 + NX * CELL, Y0, Y0 + NY * CELL)


def pa(ax):
    img = np.ones((NY, NX, 3)); img[free] = (0.80, 0.88, 0.96); img[matter] = (0.1, 0.1, 0.1)
    ax.imshow(img, origin="lower", extent=EXT, interpolation="nearest")
    for p in POSES: ax.plot(*p, "o", ms=4, color=ACCENT)


def pb(ax):
    img = np.ones((NY, NX, 3)); img[region] = (0.80, 0.88, 0.96); img[region & ~region0] = (0.98, 0.75, 0.70)
    ax.imshow(img, origin="lower", extent=EXT, interpolation="nearest")
    ax.plot(*POSES[0], "o", ms=4, color=ACCENT)


def pc(ax_unused=None):
    f = plt.figure(figsize=(3.0, 2.0))
    for k, (ang, title) in enumerate([(phi, "wall directions"), (4 * phi, "each angle $\\times 4$")]):
        ax = f.add_subplot(1, 2, k + 1, projection="polar")
        hist, edges_ = np.histogram(np.mod(ang, 2 * np.pi), bins=72, range=(0, 2 * np.pi), weights=w)
        ax.bar(edges_[:-1], hist / hist.max(), width=2 * np.pi / 72, color=COOL, align="edge")
        ax.set_yticks([]); ax.set_xticks([]); ax.set_title(title, fontsize=10, pad=4)
        if k: ax.plot([4 * th0, 4 * th0], [0, 1.1], color=ACCENT, lw=2)
    f.subplots_adjust(0.02, 0.0, 0.98, 0.86, wspace=0.25); f.savefig(HERE / "d3_outline_c.pdf"); plt.close(f)


def pd(ax):
    # the result over the whole room: the traced border (jagged, black), the outline (red, its corners),
    # and the hits (grey): the cabinets and the wardrobe stay INSIDE the outline
    ax.imshow(rot, origin="lower", extent=(U0, U0 + NU * CELL, V0, V0 + NV * CELL), cmap="Blues", vmin=0, vmax=4.0, interpolation="nearest")
    ax.plot(hp[:, 0], hp[:, 1], "s", ms=0.9, color="0.35")
    ax.plot(ring[:, 0], ring[:, 1], color="0.1", lw=0.7)
    Q = np.vstack([V2, V2[:1]]); ax.plot(Q[:, 0], Q[:, 1], color=ACCENT, lw=1.8)
    ax.plot(V2[:, 0], V2[:, 1], "o", ms=4.5, color=ACCENT)
    ax.set_xlim(U0 + 0.5, U0 + NU * CELL - 0.15); ax.set_ylim(V0 + 0.5, V0 + NV * CELL - 0.6)


fig("d3_outline_a.pdf", pa); fig("d3_outline_b.pdf", pb); pc()
fig("d3_outline_d.pdf", pd)
