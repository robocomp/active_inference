"""Day 4: the localiser's window grows and slides as the robot drives.

An L-shaped room with fixed corners. The robot drives a loop. Every time it has moved enough, a new
pose joins the window: a wheel factor links it to the previous pose, and a corner factor links it to
each room corner it can see. Only the newest pose carries the wall term (its scan, drawn as hits on
the walls). With five poses in the window, the next one pushes the oldest out: it fades, and what it
knew becomes the prior on the new oldest pose.

Writes media/pose_window.mp4 and figures/pose_window_poster.png (run from slides/).
Usage: python3 figures/make_window_video.py [--preview]
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle, RegularPolygon

HERE = Path(__file__).resolve().parent.parent          # slides/
PREVIEW = "--preview" in sys.argv
FPS = 30
DURATION = 40.0
W_IN, H_IN, DPI = (12.8, 7.2, 50) if PREVIEW else (12.8, 7.2, 150)

ACCENT = "#C0392B"; COOL = "#2E6DB4"; GOOD = "#2E8B57"; MUTED = "#7A7A7A"; INK = "#222222"

# ---- the room: an L, corners fixed ------------------------------------------------------------------
ROOM = np.array([[0, 0], [8, 0], [8, 3.2], [4.6, 3.2], [4.6, 5.6], [0, 5.6]], float)
CORNERS = ROOM.copy()
WALLS = [(ROOM[i], ROOM[(i + 1) % len(ROOM)]) for i in range(len(ROOM))]
WINDOW, STRIDE, RANGE = 5, 1.1, 4.2                   # drawn stride (the robot uses 10 cm / 9 deg)

# ---- the path: a smooth loop through the L ----------------------------------------------------------
ctrl = np.array([[1.2, 1.0], [4.0, 0.9], [6.9, 1.2], [6.9, 2.3], [4.2, 2.2], [3.4, 3.4], [3.3, 4.8],
                 [1.4, 4.7], [1.0, 3.0], [1.2, 1.0]])
t_c = np.linspace(0, 1, len(ctrl))
tt = np.linspace(0, 1, 2000)
from scipy.interpolate import CubicSpline
cs = CubicSpline(t_c, ctrl, bc_type="periodic")
PATH = cs(tt)
dP = cs(tt, 1)
HEAD = np.arctan2(dP[:, 1], dP[:, 0])
seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))]
LENGTH = seg[-1]


def pose_at(s):
    i = np.searchsorted(seg, s % LENGTH)
    i = min(max(i, 0), len(PATH) - 1)
    return PATH[i], HEAD[i]


def ray_hit(p, ang):
    d = np.array([np.cos(ang), np.sin(ang)])
    best = np.inf
    for a, b in WALLS:
        e = b - a
        M = np.array([d, -e]).T
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        s, u = np.linalg.solve(M, a - p)
        if s > 1e-6 and 0 <= u <= 1:
            best = min(best, s)
    return p + best * d if np.isfinite(best) else None


def visible(p, c):
    if np.linalg.norm(c - p) > RANGE:
        return False
    d = c - p; dist = np.linalg.norm(d); d /= dist
    for a, b in WALLS:
        e = b - a
        M = np.array([d, -e]).T
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        s, u = np.linalg.solve(M, a - p)
        if 1e-6 < s < dist - 1e-3 and 1e-6 < u < 1 - 1e-6:
            return False
    return True


# ---- simulate the window over time ------------------------------------------------------------------
n_frames = int(DURATION * FPS)
speed = LENGTH * 1.0 / (DURATION - 1.5)               # one loop
events = []                                           # (frame, kind)
slots, dropped = [], []                               # slot: dict(p, h, corners, born)
last_s = -1e9
frame_state = []
for f in range(n_frames):
    s = speed * f / FPS
    p, h = pose_at(s)
    if s - last_s >= STRIDE:
        last_s = s
        cs_vis = [j for j, c in enumerate(CORNERS) if visible(p, c)]
        slots.append(dict(p=p.copy(), h=h, corners=cs_vis, born=f))
        if len(slots) > WINDOW:
            old = slots.pop(0)
            dropped.append(dict(slot=old, at=f))
    frame_state.append((p.copy(), h, [dict(x) for x in slots], [d for d in dropped if f - d["at"] < FPS * 1.5], len(dropped) > 0))

# ---- draw ---------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_axes([0.03, 0.12, 0.62, 0.82])
lg = fig.add_axes([0.67, 0.12, 0.31, 0.82]); lg.axis("off")
cap = fig.add_axes([0.03, 0.0, 0.94, 0.1]); cap.axis("off")

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{int(W_IN * DPI)}x{int(H_IN * DPI)}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       str(HERE / "media" / ("pose_window_preview.mp4" if PREVIEW else "pose_window.mp4"))],
                      stdin=subprocess.PIPE)


def draw_legend():
    lg.clear(); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    y = 0.92
    items = [("pose", "a pose (5 at most)"),
             ("wheel", "wheels: pose to next pose"),
             ("corner", "corner seen from a pose"),
             ("wall", "walls: newest scan only"),
             ("prior", "prior: what dropped\nposes left")]
    for kind, text in items:
        if kind == "pose":
            lg.add_patch(Circle((0.06, y), 0.025, fc="#DCE6F3", ec=COOL, lw=2))
        elif kind == "wheel":
            lg.add_patch(plt.Rectangle((0.045, y - 0.012), 0.03, 0.024, fc=INK))
        elif kind == "corner":
            lg.plot([0.0, 0.12], [y, y], color=GOOD, lw=1.5)
            lg.add_patch(plt.Rectangle((0.048, y - 0.01), 0.022, 0.02, fc=GOOD))
        elif kind == "wall":
            lg.plot([0.03, 0.06, 0.09], [y, y, y], "o", color=ACCENT, ms=4)
        else:
            lg.add_patch(plt.Rectangle((0.045, y - 0.012), 0.03, 0.024, fc=MUTED))
        lg.text(0.16, y, text, va="center", fontsize=14, color=INK)
        y -= 0.12
    lg.text(0.0, y - 0.02, "F = walls + corners\n     + wheels + prior", fontsize=16, color=INK, weight="bold", va="top")
    lg.text(0.0, y - 0.2, "a new pose every 10 cm or 9°\n(drawn sparser here)",
            fontsize=13, color=MUTED, va="top")


for f, (p, h, sl, drop, any_dropped) in enumerate(frame_state):
    ax.clear(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.6, 8.6); ax.set_ylim(-0.6, 6.2)
    ax.add_patch(Polygon(ROOM, closed=True, fc="#F4F4F4", ec=INK, lw=3))
    for c in CORNERS:
        ax.add_patch(RegularPolygon(c, 4, radius=0.17, orientation=np.pi / 4, fc="#CFE8D8", ec=GOOD, lw=2, zorder=5))
    # dropped pose fading out, and the prior square on the new oldest
    for d in drop:
        a = max(0.0, 1 - (f - d["at"]) / (FPS * 1.5))
        q = d["slot"]["p"]
        ax.add_patch(Circle(q, 0.2, fc="none", ec=MUTED, lw=2, alpha=a, ls="--", zorder=6))
    if sl:
        oldest = sl[0]["p"]
        if any_dropped:
            pr = oldest + np.array([-0.45, -0.3])
            ax.plot([pr[0], oldest[0]], [pr[1], oldest[1]], color=MUTED, lw=1.5, zorder=4)
            ax.add_patch(plt.Rectangle(pr - 0.08, 0.16, 0.16, fc=MUTED, zorder=7))
    # corner factors
    for k, s_ in enumerate(sl):
        for j in s_["corners"]:
            c = CORNERS[j]; m = 0.55 * s_["p"] + 0.45 * c
            ax.plot([s_["p"][0], c[0]], [s_["p"][1], c[1]], color=GOOD, lw=1.3, alpha=0.75, zorder=3)
            ax.add_patch(plt.Rectangle(m - 0.06, 0.12, 0.12, fc=GOOD, zorder=4))
    # wheel factors
    for a_, b_ in zip(sl[:-1], sl[1:]):
        ax.plot([a_["p"][0], b_["p"][0]], [a_["p"][1], b_["p"][1]], color=INK, lw=1.6, zorder=4)
        m = 0.5 * (a_["p"] + b_["p"])
        ax.add_patch(plt.Rectangle(m - 0.07, 0.14, 0.14, fc=INK, zorder=5))
    # newest scan
    if sl:
        nw = sl[-1]
        for ang in np.linspace(-np.pi, np.pi, 48, endpoint=False):
            hit = ray_hit(nw["p"], nw["h"] + ang)
            if hit is not None and np.linalg.norm(hit - nw["p"]) < 6:
                ax.plot(*hit, "o", color=ACCENT, ms=3.5, zorder=6)
    # poses
    for k, s_ in enumerate(sl):
        ax.add_patch(Circle(s_["p"], 0.2, fc="#DCE6F3", ec=COOL, lw=2.5, zorder=8))
        ax.text(*s_["p"], f"$X_{k}$", ha="center", va="center", fontsize=11, zorder=9)
    # robot
    tri = np.array([[0.32, 0], [-0.2, 0.17], [-0.2, -0.17]])
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    ax.add_patch(Polygon(tri @ R.T + p, closed=True, fc=INK, zorder=10))
    draw_legend()
    # caption: what just happened
    cap.clear(); cap.axis("off")
    msg = "the robot drives; only its newest pose sees the walls"
    if sl:
        age = f - sl[-1]["born"]
        if drop and age < FPS * 1.5:
            msg = "a sixth pose: the oldest leaves the window, and what it knew becomes the prior"
        elif age < FPS * 1.2:
            n = len(sl[-1]['corners'])
            msg = f"new pose: linked to the previous one by the wheels, and to the {n} corner{'s' if n != 1 else ''} it sees"
    cap.text(0.5, 0.5, msg, ha="center", va="center", fontsize=18, color=INK)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    ff.stdin.write(buf.tobytes())
    if f == int(n_frames * 0.55) and not PREVIEW:
        fig.savefig(HERE / "figures" / "pose_window_poster.png", dpi=DPI)

ff.stdin.close(); ff.wait()
print("frames:", n_frames)
