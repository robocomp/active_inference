"""Day 3, Step 1 (hidden SLAM): one scan registered against the map, computed, not drawn. Run from slides/.

A toy of layout_acquisition.tex l.344-416: a hits-only grid (5 cm), its distance field D, a new scan placed
at the wheels' guess, then robust Gauss-Newton on (x, y, heading) until the returns sit on the hits.
Writes d3_slam_a.pdf (the wheels' guess), d3_slam_b.pdf (the distance field), d3_slam_c.pdf (aligned).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})
COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"
rng = np.random.default_rng(7)

ROOM = np.array([(0, 0), (6, 0), (6, 2.5), (3.5, 2.5), (3.5, 4.5), (0, 4.5)], float)
SEGS = [(ROOM[i], ROOM[(i + 1) % len(ROOM)]) for i in range(len(ROOM))]
CELL, X0, Y0, NX, NY = 0.05, -0.5, -0.5, 140, 112


def cast(pose, n=360, noise=0.01):
    """Returns (robot frame) of a 360-degree scan from pose, against the room's walls."""
    x, y, th = pose
    out = []
    for a in np.linspace(-np.pi, np.pi, n, endpoint=False):
        d = np.array([np.cos(th + a), np.sin(th + a)])
        best = np.inf
        for p, q in SEGS:
            e = q - p
            M = np.array([d, -e]).T
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            t, u = np.linalg.solve(M, p - np.array([x, y]))
            if t > 0.05 and 0 <= u <= 1:
                best = min(best, t)
        if np.isfinite(best):
            r = best + rng.normal(0, noise)
            out.append((r * np.cos(a), r * np.sin(a)))
    return np.array(out)


def place(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


# the map: hits from three earlier, already registered scans (cells hit at least twice)
hits = np.zeros((NY, NX))
START = np.array([0.7, 0.7, 0.0])
for pose in [tuple(START), (1.5, 3.4, 1.2), (4.6, 1.2, 2.8)]:
    for _ in range(2):
        w = place(cast(np.array(pose)), np.array(pose))
        ix = ((w[:, 0] - X0) / CELL).astype(int); iy = ((w[:, 1] - Y0) / CELL).astype(int)
        np.add.at(hits, (iy, ix), 1)
surface = hits >= 2
D = distance_transform_edt(~surface) * CELL
gy, gx = np.gradient(D, CELL)


def field(arr, w):
    return map_coordinates(arr, [(w[:, 1] - Y0) / CELL - 0.5, (w[:, 0] - X0) / CELL - 0.5], order=1, mode="nearest")


# the new scan, and the wheels' guess of where it was taken
truth = np.array([2.1, 1.6, 0.35])
scan = cast(truth)[::2]
guess = truth + np.array([0.30, -0.22, 0.17])

pose, path = guess.copy(), [guess.copy()]
for _ in range(40):
    w = place(scan, pose)
    r = field(D, w)
    g = np.c_[field(gx, w), field(gy, w)]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dw_dth = scan @ np.array([[-s, c], [-c, -s]])
    J = np.c_[g, (g * dw_dth).sum(1)]
    wt = 1 / (1 + (r / 0.10) ** 2)                      # Cauchy, c = 10 cm
    H = J.T @ (wt[:, None] * J) + 1e-6 * np.eye(3)
    step = -np.linalg.solve(H, J.T @ (wt * r))
    pose = pose + step; path.append(pose.copy())
    if np.abs(step).max() < 1e-4:
        break
print("guess error", np.round(guess - truth, 3), "final error", np.round(pose - truth, 4), "steps", len(path) - 1)

sy, sx = np.nonzero(surface)
MX, MY = X0 + (sx + 0.5) * CELL, Y0 + (sy + 0.5) * CELL


def robot(ax, p, col):
    c, s = np.cos(p[2]), np.sin(p[2])
    tri = np.array([[0.22, 0], [-0.12, 0.11], [-0.12, -0.11]]) @ np.array([[c, s], [-s, c]]) + p[:2]
    ax.fill(tri[:, 0], tri[:, 1], color=col, zorder=5)


def base(ax):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.4, 6.4); ax.set_ylim(-0.4, 4.9)


def panel(name, draw):
    fig, ax = plt.subplots(figsize=(3.2, 2.7)); base(ax); draw(ax)
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(HERE / name); plt.close(fig)


def a(ax):
    ax.plot(MX, MY, "s", ms=1.6, color="0.55")
    w = place(scan, guess)
    for k in range(0, len(w), 6):
        q = w[k]; t = q - field(D, q[None])[0] * np.r_[field(gx, q[None]), field(gy, q[None])]
        ax.plot([q[0], t[0]], [q[1], t[1]], color=ACCENT, lw=0.6, alpha=0.7)
    ax.plot(w[:, 0], w[:, 1], ".", ms=3.2, color=ACCENT)
    robot(ax, guess, ACCENT)
    # where the run started: the map's origin is fixed there and never moves
    sx0, sy0 = START[:2]
    ax.annotate("", xy=(sx0 + 0.7, sy0), xytext=(sx0, sy0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.annotate("", xy=(sx0, sy0 + 0.7), xytext=(sx0, sy0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.plot(sx0, sy0, "o", ms=5, mfc="white", mec="k", mew=1.4, zorder=6)
    ax.text(sx0 + 0.08, sy0 - 0.1, "start = origin", fontsize=10, va="top")


def b(ax):
    ax.imshow(np.minimum(D, 1.0), origin="lower", extent=(X0, X0 + NX * CELL, Y0, Y0 + NY * CELL), cmap="Blues_r", vmin=-0.15, vmax=1.0)
    ax.contour(np.linspace(X0, X0 + NX * CELL, NX), np.linspace(Y0, Y0 + NY * CELL, NY), D, levels=[0.15, 0.3, 0.5], colors="white", linewidths=0.6)
    ax.plot(MX, MY, "s", ms=1.2, color="0.1")


def c(ax):
    ax.plot(MX, MY, "s", ms=1.6, color="0.55")
    for p in path[1:-1]:
        w = place(scan, p); ax.plot(w[:, 0], w[:, 1], ".", ms=1.0, color=MUTED, alpha=0.35)
    w = place(scan, pose)
    ax.plot(w[:, 0], w[:, 1], ".", ms=3.2, color=COOL)
    robot(ax, pose, COOL)


panel("d3_slam_a.pdf", a)
panel("d3_slam_b.pdf", b)
panel("d3_slam_c.pdf", c)
