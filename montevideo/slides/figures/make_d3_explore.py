"""Day 3, Step 4 (the explorer) on the real room R1, computed. Run from slides/figures.

The room is layouts/apartamento_layout.svg (the main room of the simulated apartment, metres). From a
start pose, rays up to 8 m mark the floor the robot has seen. Every free cell is a candidate place to
look from; its gain is the unseen floor its rays would uncover (plan_coverage's rule; 720 rays here so the\nseen area fills cleanly). Writes
d3_explore_a.pdf (seen / unseen from the start) and d3_explore_b.pdf (the best next place and the floor it uncovers), d3_explore_c.pdf (after it).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
COOL, ACCENT, MUTED = "#2E6DB4", "#C0392B", "#7A7A7A"

PTS = ("0.2150,0.0000 0.1960,0.5940 0.5110,0.5940 0.5110,1.1940 0.1780,1.1940 0.0000,6.8920 0.2220,7.1660 "
       "4.1544,7.1639 4.1944,5.0099 4.3136,5.0241 4.2136,9.2331 5.8710,9.2560 5.8840,8.0070 6.3300,8.0260 "
       "6.3360,7.1690 5.9200,7.1630 5.9320,7.1020 8.3920,7.0900 8.4940,1.2100 8.1100,1.2100 8.1100,0.6100 "
       "8.5050,0.6100 8.5070,0.4610 8.2820,0.1790 5.1840,0.0680 5.2070,0.8220 4.3184,0.8148 4.3164,2.4978 "
       "4.1966,2.5052 4.2346,0.8192 3.2320,0.7920 3.2360,0.0780")
P = np.array([[float(v) for v in p.split(",")] for p in PTS.split()])
SEGS = [(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
inside = MPath(P)

CELL = 0.1
xs = np.arange(0, 8.6, CELL) + CELL / 2; ys = np.arange(0, 9.3, CELL) + CELL / 2
GX, GY = np.meshgrid(xs, ys)
FREE = inside.contains_points(np.c_[GX.ravel(), GY.ravel()], radius=-0.05).reshape(GX.shape)


A0 = P; E0 = np.roll(P, -1, axis=0) - P                      # wall segments: start + edge vector


def ray_lens(o, ang):
    """Distance along each ray from o to the first wall, capped at 8 m (vectorised)."""
    d = np.c_[np.cos(ang), np.sin(ang)][:, None, :]            # (n,1,2)
    e = E0[None]; w = (A0 - o)[None]                            # (1,m,2)
    den = d[..., 0] * (-e[..., 1]) - d[..., 1] * (-e[..., 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * (-e[..., 1]) - w[..., 1] * (-e[..., 0])) / den
        u = (d[..., 0] * w[..., 1] - d[..., 1] * w[..., 0]) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    return np.minimum(np.where(ok, t, np.inf).min(1), 8.0)


ANG = np.linspace(-np.pi, np.pi, 720, endpoint=False)
CENTRES = np.c_[GX.ravel(), GY.ravel()]


def view(o):
    r = ray_lens(o, ANG)
    return o + r[:, None] * np.c_[np.cos(ANG), np.sin(ANG)]


def seen_from(o):
    return MPath(view(o)).contains_points(CENTRES).reshape(GX.shape) & FREE


START = np.array([1.6, 3.0])
seen = seen_from(START)
best, best_gain = None, -1
for i in range(2, FREE.shape[0] - 2, 4):
    for j in range(2, FREE.shape[1] - 2, 4):
        if not FREE[i, j] or not seen[i, j]: continue        # reachable = already seen free floor
        o = np.array([xs[j], ys[i]]); g = (seen_from(o) & ~seen).sum()
        if g > best_gain: best, best_gain = o, g
print("start unseen m2", round((FREE & ~seen).sum() * CELL**2, 1), "best", best, "gain m2", round(best_gain * CELL**2, 1))
seen2 = seen | seen_from(best)
print("after, unseen m2", round((FREE & ~seen2).sum() * CELL**2, 1))


def base(ax, s):
    img = np.ones(FREE.shape + (3,))
    img[FREE] = (0.78, 0.78, 0.78); img[s] = (0.80, 0.88, 0.96)
    ax.imshow(img, origin="lower", extent=(0, xs[-1] + CELL / 2, 0, ys[-1] + CELL / 2), interpolation="nearest")
    Q = np.vstack([P, P[:1]]); ax.plot(Q[:, 0], Q[:, 1], color="0.15", lw=1.2)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.2, 8.7); ax.set_ylim(-0.2, 9.45)


def robot(ax, o, col, mk="o", ms=6):
    ax.plot(*o, mk, ms=ms, color=col, mec="white", mew=0.9, zorder=6)


def pa(ax):
    base(ax, seen); robot(ax, START, "k")


def pb(ax):
    base(ax, seen)
    new = seen_from(best) & ~seen
    img = np.zeros(FREE.shape + (4,)); img[new] = (0.75, 0.22, 0.17, 0.45)
    ax.imshow(img, origin="lower", extent=(0, xs[-1] + CELL / 2, 0, ys[-1] + CELL / 2), interpolation="nearest")
    V = view(best)
    for q in V[::24]: ax.plot([best[0], q[0]], [best[1], q[1]], color=ACCENT, lw=0.5, alpha=0.7)
    robot(ax, START, "k"); robot(ax, best, ACCENT, "*", 13)


def pc(ax):
    base(ax, seen2); robot(ax, START, "k"); robot(ax, best, ACCENT, "*", 13)


for name, draw in [("d3_explore_a.pdf", pa), ("d3_explore_b.pdf", pb), ("d3_explore_c.pdf", pc)]:
    f, ax = plt.subplots(figsize=(2.9, 3.1)); draw(ax)
    f.subplots_adjust(0, 0, 1, 1); f.savefig(HERE / name); plt.close(f)
