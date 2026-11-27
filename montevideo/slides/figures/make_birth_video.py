"""Day 4: a chair is born when what it explains outweighs what it costs.

A 3-D room. The robot drives towards a chair; its camera's field of view is a cone on the floor.
The chair's surface points it sees pile up as red, unexplained points. A panel on the right weighs
what one more object would explain (it grows with every point) against what it costs (fixed), both in
nats. When the first passes the second the chair's model, a blue box at its fitted pose, is born; as
the robot keeps looking, the box sharpens. Numbers are illustrative.

Writes media/chair_birth.mp4 and figures/chair_birth_poster.png (run from slides/).
Usage: python3 figures/make_birth_video.py [--preview]
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

HERE = Path(__file__).resolve().parent.parent
PREVIEW = "--preview" in sys.argv
FPS = 30
DURATION = 30.0
W_IN, H_IN, DPI = (12.8, 7.2, 50) if PREVIEW else (12.8, 7.2, 150)
rng = np.random.default_rng(3)

ACCENT = "#C0392B"; COOL = "#2E6DB4"; GOOD = "#2E8B57"; MUTED = "#7A7A7A"; INK = "#222222"

ROOM = (6.0, 4.0, 1.2)                         # x, y, wall height drawn (low walls, to see inside)
CHAIR = np.array([4.4, 2.4])                   # chair centre on the floor
CH_W, CH_SEAT, CH_BACK = 0.48, 0.46, 0.95      # width/depth, seat height, back top
COST = 14.0                                    # nats: what one more object costs (complexity)
PER_POINT = 0.15                               # nats per explained point (illustrative)
FOV, RANGE = np.deg2rad(42), 4.2

# ---- the chair as boxes (seat, back, four legs) -------------------------------------------------------
def box(c, s):
    (x, y, z), (dx, dy, dz) = c, s
    v = np.array([[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
                  [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return [v[i] for i in f]

cx, cy = CHAIR - CH_W / 2
CHAIR_FACES = (box((cx, cy, CH_SEAT - 0.05), (CH_W, CH_W, 0.05)) +
               box((cx + CH_W - 0.05, cy, CH_SEAT), (0.05, CH_W, CH_BACK - CH_SEAT)) +
               sum([box((cx + a, cy + b, 0), (0.04, 0.04, CH_SEAT - 0.05)) for a in (0.02, CH_W - 0.06)
                    for b in (0.02, CH_W - 0.06)], []))

def sample_chair_points(n):
    """Points on the chair's surfaces (seat top, back front, legs)."""
    pts = []
    for _ in range(n):
        u = rng.random()
        if u < 0.45:
            pts.append([cx + rng.random() * CH_W, cy + rng.random() * CH_W, CH_SEAT])
        elif u < 0.8:
            pts.append([cx + CH_W - 0.05, cy + rng.random() * CH_W, CH_SEAT + rng.random() * (CH_BACK - CH_SEAT)])
        else:
            a = rng.choice([0.04, CH_W - 0.04]); b = rng.choice([0.04, CH_W - 0.04])
            pts.append([cx + a, cy + b, rng.random() * (CH_SEAT - 0.05)])
    return np.array(pts) + rng.normal(0, 0.015, (n, 3))

# ---- the robot's path: drive in, slow down near the chair ------------------------------------------------
n_frames = int(DURATION * FPS)
t = np.linspace(0, 1, n_frames)
s = np.clip(t / 0.8, 0, 1); s = s * s * (3 - 2 * s)            # ease in/out, then stand still
P0, P1 = np.array([0.8, 0.7]), CHAIR + np.array([-1.5, -0.35])
mid = np.array([2.2, 2.6])
PATH = ((1 - s) ** 2)[:, None] * P0 + (2 * s * (1 - s))[:, None] * mid + (s ** 2)[:, None] * P1
HEAD = np.unwrap(np.arctan2(CHAIR[1] - PATH[:, 1], CHAIR[0] - PATH[:, 0]))

# ---- evidence over time ---------------------------------------------------------------------------------
seen = np.zeros((0, 3)); explained = []; born_at = None; frame_pts = []
for f in range(n_frames):
    p, h = PATH[f], HEAD[f]
    d = CHAIR - p; dist = np.linalg.norm(d)
    ang = np.arctan2(d[1], d[0]) - h
    in_view = abs((ang + np.pi) % (2 * np.pi) - np.pi) < FOV and dist < RANGE
    if in_view and f % 6 == 0:
        k = int(np.clip(4 * (RANGE - dist) / RANGE, 1, 4))
        seen = np.vstack([seen, sample_chair_points(k)])
    explained.append(PER_POINT * len(seen))
    if born_at is None and explained[-1] > COST:
        born_at = f
    frame_pts.append(len(seen))
if "--dry" in sys.argv:
    print("born at", None if born_at is None else round(born_at / FPS, 1), "s; points", len(seen)); sys.exit()

# ---- draw ---------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_axes([0.0, 0.1, 0.66, 0.9], projection="3d", computed_zorder=False)
bar = fig.add_axes([0.70, 0.30, 0.27, 0.55])
cap = fig.add_axes([0.03, 0.0, 0.94, 0.1]); cap.axis("off")

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{int(W_IN * DPI)}x{int(H_IN * DPI)}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       str(HERE / "media" / ("chair_birth_preview.mp4" if PREVIEW else "chair_birth.mp4"))],
                      stdin=subprocess.PIPE)

LX, LY, LZ = ROOM
floor = [[[0, 0, 0], [LX, 0, 0], [LX, LY, 0], [0, LY, 0]]]
walls = [[[0, LY, 0], [LX, LY, 0], [LX, LY, LZ], [0, LY, LZ]], [[LX, 0, 0], [LX, LY, 0], [LX, LY, LZ], [LX, 0, LZ]]]

for f in range(n_frames):
    p, h = PATH[f], HEAD[f]
    ax.clear(); ax.set_axis_off()
    ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.set_zlim(0, 2.0)
    ax.set_box_aspect((LX, LY, 2.0), zoom=1.45)
    ax.view_init(elev=32, azim=-62 + 18 * f / n_frames)
    ax.add_collection3d(Poly3DCollection(floor, fc="#EFEFEF", ec="#BBBBBB", zorder=0))
    ax.add_collection3d(Poly3DCollection(walls, fc="#E2E2E2", ec="#AAAAAA", alpha=0.9, zorder=1))
    ax.add_collection3d(Poly3DCollection(CHAIR_FACES, fc="#B9A58C", ec="#7D6A58", lw=0.5, zorder=3))
    # field of view on the floor
    a0, a1 = h - FOV, h + FOV
    arc = np.linspace(a0, a1, 20)
    cone = [[ [p[0], p[1], 0.01] ] + [[p[0] + RANGE * np.cos(a), p[1] + RANGE * np.sin(a), 0.01] for a in arc]]
    cone[0] = [[min(max(x, 0), LX), min(max(y, 0), LY), z] for x, y, z in cone[0]]
    ax.add_collection3d(Poly3DCollection(cone, fc=COOL, alpha=0.12, ec="none", zorder=2))
    # robot: a short cylinder-ish box with a heading tick
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    base = (np.array([[-0.25, -0.2], [0.25, -0.2], [0.25, 0.2], [-0.25, 0.2]]) @ R.T) + p
    top = [[*q, 0.35] for q in base]; bot = [[*q, 0.0] for q in base]
    rob = [bot, top] + [[bot[i], bot[(i + 1) % 4], top[(i + 1) % 4], top[i]] for i in range(4)]
    ax.add_collection3d(Poly3DCollection(rob, fc="#333333", ec="#111111", zorder=4))
    tip = p + R @ np.array([0.45, 0])
    ax.plot([p[0], tip[0]], [p[1], tip[1]], [0.36, 0.36], color=ACCENT, lw=3, zorder=5)
    # points seen so far
    n = frame_pts[f]
    if n:
        pts = seen[:n]
        col = COOL if (born_at is not None and f >= born_at) else ACCENT
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=9, c=col, depthshade=False, zorder=6)
    # the chair's model once born: a box that sharpens
    if born_at is not None and f >= born_at:
        age = (f - born_at) / FPS
        grow = 0.12 * np.exp(-age / 3.0)                      # shrinking extra margin = sharpening
        m = box((cx - grow, cy - grow, 0), (CH_W + 2 * grow, CH_W + 2 * grow, CH_BACK + grow))
        edges = []
        for face in m:
            for i in range(4):
                edges.append([face[i], face[(i + 1) % 4]])
        ax.add_collection3d(Line3DCollection(edges, colors=COOL, linewidths=2.2, zorder=7))
    # evidence panel
    bar.clear()
    e = explained[f]
    bar.barh([1], [min(e, COST * 2.5)], color=COOL, height=0.55)
    bar.barh([0], [COST], color=ACCENT, height=0.55, alpha=0.75)
    bar.axvline(COST, color=ACCENT, ls="--", lw=1.5)
    bar.set_yticks([0, 1], ["its cost", "what a chair\nwould explain"], fontsize=14)
    bar.set_xlim(0, COST * 2.5); bar.set_xlabel("nats", fontsize=13)
    for sp in ("top", "right"):
        bar.spines[sp].set_visible(False)
    bar.set_title("one more object?", fontsize=16, color=INK)
    born = born_at is not None and f >= born_at
    bar.text(0.98, -0.42, "born" if born else "not yet", transform=bar.transAxes, ha="right",
             fontsize=18, color=COOL if born else MUTED, weight="bold")
    # caption
    cap.clear(); cap.axis("off")
    if born:
        msg = "it explains more than it costs: chair_1 is born, and sharpens as the robot looks"
    elif n:
        msg = f"{n} points no object explains: each one adds to what a new chair would explain"
    else:
        msg = "the robot drives; nothing in its view yet"
    cap.text(0.5, 0.5, msg, ha="center", va="center", fontsize=17, color=INK)
    fig.canvas.draw()
    ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if not PREVIEW and born_at is not None and f == born_at + FPS * 2:
        fig.savefig(HERE / "figures" / "chair_birth_poster.png", dpi=DPI)

ff.stdin.close(); ff.wait()
print("frames:", n_frames, " born at", None if born_at is None else round(born_at / FPS, 1), "s")
