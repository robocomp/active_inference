"""Day 3, localisation: the room's distance field, computed. Run from slides/figures.

The L room of the other Day 3 figures. d(p) = distance from p to the nearest wall segment (unsigned, as
room_concept's polygon mode; the code calls it sdf). A scan taken at the true pose is placed at a wrong
pose; each return's d is its error and -grad d points it to its wall.
Writes d3_sdf_a.pdf (the field, its level lines) and d3_sdf_b.pdf (a misplaced scan, pulled by -grad d).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
COOL, ACCENT = "#2E6DB4", "#C0392B"
rng = np.random.default_rng(3)

ROOM = np.array([(0, 0), (6, 0), (6, 2.5), (3.5, 2.5), (3.5, 4.5), (0, 4.5)], float)
A = ROOM; E = np.roll(ROOM, -1, axis=0) - ROOM


def dist(p):
    """Distance from each point of p (n,2) to the nearest wall segment, and the gradient (unit, away from it)."""
    w = p[:, None, :] - A[None]
    t = np.clip((w * E[None]).sum(-1) / (E ** 2).sum(-1)[None], 0, 1)
    c = A[None] + t[..., None] * E[None]
    v = p[:, None, :] - c; d = np.hypot(v[..., 0], v[..., 1])
    k = d.argmin(1); i = np.arange(len(p))
    return d[i, k], v[i, k] / np.maximum(d[i, k], 1e-9)[:, None]


def cast(o, th, n=72):
    out = []
    for a in np.linspace(-np.pi, np.pi, n, endpoint=False):
        dv = np.array([np.cos(th + a), np.sin(th + a)]); best = np.inf
        for p, e in zip(A, E):
            M = np.array([dv, -e]).T
            if abs(np.linalg.det(M)) < 1e-9: continue
            t, u = np.linalg.solve(M, p - o)
            if t > 0.05 and 0 <= u <= 1: best = min(best, t)
        r = best + rng.normal(0, 0.01)
        out.append((r * np.cos(a), r * np.sin(a)))
    return np.array(out)


def place(pts, o, th):
    c, s = np.cos(th), np.sin(th)
    return pts @ np.array([[c, s], [-s, c]]) + o


xs = np.linspace(-0.3, 6.3, 331); ys = np.linspace(-0.3, 4.8, 256)
GX, GY = np.meshgrid(xs, ys)
D = dist(np.c_[GX.ravel(), GY.ravel()])[0].reshape(GX.shape)
Q = np.vstack([ROOM, ROOM[:1]])


def base(ax):
    ax.imshow(np.minimum(D, 1.2), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="Blues_r", vmin=-0.2, vmax=1.3)
    ax.contour(GX, GY, D, levels=[0.25, 0.5, 0.75, 1.0], colors="white", linewidths=0.7)
    ax.plot(Q[:, 0], Q[:, 1], color="0.1", lw=1.6)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])


def robot(ax, o, th, col):
    c, s = np.cos(th), np.sin(th)
    tri = np.array([[0.25, 0], [-0.14, 0.13], [-0.14, -0.13]]) @ np.array([[c, s], [-s, c]]) + o
    ax.fill(tri[:, 0], tri[:, 1], color=col, zorder=5)


def pa(ax):
    base(ax)


TRUE_O, TRUE_TH = np.array([1.8, 1.6]), 0.3
GUESS_O, GUESS_TH = TRUE_O + np.array([0.28, -0.2]), TRUE_TH + 0.12
scan = cast(TRUE_O, TRUE_TH)


def pb(ax):
    base(ax)
    w = place(scan, GUESS_O, GUESS_TH); d, g = dist(w)
    for q, dq, gq in zip(w, d, g):
        ax.annotate("", xy=q - dq * gq, xytext=q, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.9, mutation_scale=7))
    ax.plot(w[:, 0], w[:, 1], ".", ms=4, color=ACCENT, zorder=4)
    robot(ax, GUESS_O, GUESS_TH, ACCENT)


for name, draw in [("d3_sdf_a.pdf", pa), ("d3_sdf_b.pdf", pb)]:
    f, ax = plt.subplots(figsize=(3.3, 2.6)); draw(ax)
    f.subplots_adjust(0, 0, 1, 1); f.savefig(HERE / name); plt.close(f)
