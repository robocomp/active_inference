"""Day 4: the robot picks, again and again, the future with the lowest expected free energy G.

A room seen from above, a door on the right (the goal) and a chair whose position the robot knows only
roughly (its 2-sigma ellipse). Every half metre the robot imagines a fan of short future paths. For each
one it predicts what it would see, and scores
    G = -I + lambda * d
I: the information the readings along the path would carry about the chair (epistemic);
d: how far the path ends from the door (pragmatic, lambda nats per metre).
The fan is drawn coloured by G; the robot drives the start of the best path, really looks, and plans
again. While the chair is uncertain, paths that look at it win; once it is known, I is small for every
path and only the door counts. Numbers are illustrative.

Writes media/efe_paths.mp4 and figures/efe_paths_poster.png (run from slides/).
Usage: python3 figures/make_efe_video.py [--preview|--dry]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
PREVIEW = "--preview" in sys.argv
DRY = "--dry" in sys.argv
FPS = 30
W_IN, H_IN, DPI = (12.8, 7.2, 50) if PREVIEW else (12.8, 7.2, 150)

ACCENT = "#C0392B"; COOL = "#2E6DB4"; GOOD = "#2E8B57"; MUTED = "#7A7A7A"; INK = "#222222"

# ---- the world -------------------------------------------------------------------------------------------
LX, LY = 8.0, 5.0
DOOR = np.array([LX, 1.2]); DOOR_W = 0.9
CHAIR_TRUE = np.array([3.9, 4.2])
CHAIR_MU0 = np.array([3.5, 3.7])            # the robot's first guess
SIG0 = 1.0                                  # prior sd of the chair position (m), per axis
FOV, RANGE = np.deg2rad(45), 5.5
def sigma_o(r): return 0.3 * r              # sd of one look at the chair from range r (m)
LAMBDA = 0.9                                # nats per metre from the door
V = 0.5                                     # m/s
HORIZON, STEP = 2.2, 0.25                   # imagined path length, spacing of imagined looks (m)
CURV = np.linspace(-1.3, 1.3, 11)           # the fan: constant-curvature arcs
REPLAN = 0.5                                # metres driven between plans

def arc(p, h, k, L=HORIZON, ds=STEP):
    s = np.arange(ds, L + 1e-9, ds)
    if abs(k) < 1e-6:
        x = p[0] + s * np.cos(h); y = p[1] + s * np.sin(h); th = np.full_like(s, h)
    else:
        th = h + k * s
        x = p[0] + (np.sin(th) - np.sin(h)) / k; y = p[1] - (np.cos(th) - np.cos(h)) / k
    return np.c_[x, y], th

def sees(p, h, c):
    d = c - p; r = np.linalg.norm(d)
    a = (np.arctan2(d[1], d[0]) - h + np.pi) % (2 * np.pi) - np.pi
    return r < RANGE and abs(a) < FOV, r

def feasible(P):
    m = 0.3
    inside = (P[:, 0] > m) & (P[:, 0] < LX - m) & (P[:, 1] > m) & (P[:, 1] < LY - m)
    near_door = (np.abs(P[:, 1] - DOOR[1]) < DOOR_W / 2) & (P[:, 0] > LX - 1.0)
    clear = np.linalg.norm(P - CHAIR_TRUE, axis=1) > 0.7
    return bool(np.all((inside | near_door) & clear))

def score(p, h, k, prec):
    P, TH = arc(p, h, k)
    if not feasible(P):
        return None
    gain = 0.0
    for q, t in zip(P, TH):
        ok, r = sees(q, t, MU)
        if ok:
            gain += 1.0 / sigma_o(r) ** 2
    I = 2 * 0.5 * np.log(1 + gain / prec)            # two axes, same precision
    d = np.linalg.norm(P[-1] - DOOR)
    return dict(P=P, k=k, I=I, prag=LAMBDA * d, G=-I + LAMBDA * d)

# ---- simulate ----------------------------------------------------------------------------------------------
rng = np.random.default_rng(1)
p, h = np.array([0.9, 0.9]), np.deg2rad(20)
MU = CHAIR_MU0.copy(); prec = 1 / SIG0 ** 2
frames = []                                   # per frame: pose, MU, sd, fan, best, trail, phase
trail = [p.copy()]
done = False
while not done and len(frames) < 60 * FPS:
    fan = [c for c in (score(p, h, k, prec) for k in CURV) if c is not None]
    best = min(fan, key=lambda c: c["G"])
    hold = int(0.6 * FPS)                     # show the fan before moving
    for _ in range(hold):
        frames.append(dict(p=p.copy(), h=h, mu=MU.copy(), sd=prec ** -0.5, fan=fan, best=best,
                           trail=list(trail), moving=False))
    # drive REPLAN metres along the best arc, really looking
    P, TH = arc(p, h, best["k"], L=REPLAN, ds=V / FPS)
    seen_since = 0.0
    for q, t in zip(P, TH):
        p, h = q, t; trail.append(p.copy()); seen_since += V / FPS
        if seen_since >= STEP:
            seen_since = 0.0
            ok, r = sees(p, h, CHAIR_TRUE)
            if ok:
                z = CHAIR_TRUE + rng.normal(0, sigma_o(r), 2)
                w = 1 / sigma_o(r) ** 2
                MU = (prec * MU + w * z) / (prec + w); prec += w
        frames.append(dict(p=p.copy(), h=h, mu=MU.copy(), sd=prec ** -0.5, fan=fan, best=best,
                           trail=list(trail), moving=True))
        if np.linalg.norm(p - DOOR) < 0.45:
            done = True; break
for _ in range(int(1.5 * FPS)):
    frames.append(dict(frames[-1]))
T_=np.array(trail); print("max y", T_[:,1].max().round(2), "min dist chair", np.linalg.norm(T_-CHAIR_TRUE,axis=1).min().round(2))
print("frames", len(frames), "seconds", round(len(frames) / FPS, 1), "final sd", round(prec ** -0.5, 3))
if DRY:
    sys.exit()

# ---- draw ----------------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse, Rectangle, Wedge
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list("g", [GOOD, "#B9D8C6", "#E2E2E2"])
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_axes([0.01, 0.12, 0.64, 0.86])
bar = fig.add_axes([0.78, 0.50, 0.20, 0.36])
side = fig.add_axes([0.68, 0.12, 0.31, 0.27]); side.axis("off")
cap = fig.add_axes([0.03, 0.0, 0.94, 0.1]); cap.axis("off")
out = HERE / "media" / ("efe_paths_preview.mp4" if PREVIEW else "efe_paths.mp4")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{int(W_IN * DPI)}x{int(H_IN * DPI)}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
poster_at = None
for f, st in enumerate(frames):
    ax.clear(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.2, LX + 1.3); ax.set_ylim(-0.3, LY + 0.3)
    floor = ax.add_patch(Rectangle((0, 0), LX, LY, fc="#F4F4F4", ec="none"))
    for seg in ([(0, 0), (LX, 0)], [(0, 0), (0, LY)], [(0, LY), (LX, LY)],
                [(LX, 0), (LX, DOOR[1] - DOOR_W / 2)], [(LX, DOOR[1] + DOOR_W / 2), (LX, LY)]):
        ax.plot(*zip(*seg), color=INK, lw=3)
    ax.plot([LX, LX], [DOOR[1] - DOOR_W / 2, DOOR[1] + DOOR_W / 2], color=GOOD, lw=6, alpha=0.5)
    ax.text(LX + 0.15, DOOR[1], "goal:\nthe door", color=GOOD, fontsize=13, va="center", ha="left", weight="bold")
    # the chair: truth (small, dark) and belief (ellipse, 2 sd)
    ax.add_patch(Rectangle(CHAIR_TRUE - 0.22, 0.44, 0.44, fc="#C8A57A", ec="#7A5A30", lw=1.5, zorder=3))
    el = Ellipse(st["mu"], 4 * st["sd"], 4 * st["sd"], fc=COOL, alpha=0.15, ec=COOL, lw=2, zorder=4)
    ax.add_patch(el); el.set_clip_path(floor)
    ax.plot(*st["mu"], "+", color=COOL, ms=12, mew=2, zorder=5)
    if st["sd"] > 0.6:
        ax.text(st["mu"][0], st["mu"][1] - 0.7, "where the chair\nmay be", color=COOL, fontsize=12, ha="center", va="center")
    else:
        ax.text(st["mu"][0] - 2 * st["sd"] - 0.1, st["mu"][1], "where the chair\nmay be", color=COOL, fontsize=12,
                ha="right", va="center")
    # the fan, coloured by G
    Gs = np.array([c["G"] for c in st["fan"]]); lo, hi = Gs.min(), Gs.max() + 1e-9
    for c in sorted(st["fan"], key=lambda c: -c["G"]):
        is_best = c is st["best"]
        col = CMAP((c["G"] - lo) / (hi - lo))
        P = np.vstack([st["p"] if not st["moving"] else c["P"][0], c["P"]]) if not st["moving"] else c["P"]
        ax.plot(P[:, 0], P[:, 1], color=col, lw=4 if is_best else 2, alpha=1 if is_best else 0.9,
                zorder=6 if is_best else 5, solid_capstyle="round")
    # trail and robot with its field of view
    T = np.array(st["trail"]); ax.plot(T[:, 0], T[:, 1], color=MUTED, lw=1.5, ls=":", zorder=4)
    ax.add_patch(Wedge(st["p"], 1.6, np.rad2deg(st["h"] - FOV), np.rad2deg(st["h"] + FOV), fc=COOL, alpha=0.08, zorder=2))
    tri = np.array([[0.3, 0], [-0.18, 0.16], [-0.18, -0.16]])
    R = np.array([[np.cos(st["h"]), -np.sin(st["h"])], [np.sin(st["h"]), np.cos(st["h"])]])
    ax.add_patch(Polygon(tri @ R.T + st["p"], closed=True, fc=INK, zorder=9))
    # the chosen path's G, as signed bars
    b = st["best"]
    bar.clear()
    vals = [-b["I"], b["prag"], b["G"]]
    bar.barh([2, 1, 0], vals, color=[COOL, ACCENT, GOOD], height=0.6)
    bar.axvline(0, color=INK, lw=1)
    bar.set_xlim(-3.5, 7.5); bar.set_ylim(-0.6, 2.6)
    bar.set_yticks([2, 1, 0], ["epistemic: $-I$", "pragmatic: $\\lambda d$", "$G$"], fontsize=14)
    bar.set_xlabel("nats", fontsize=12)
    for sp in ("top", "right", "left"):
        bar.spines[sp].set_visible(False)
    bar.set_title("the chosen path", fontsize=15, color=INK)
    side.clear(); side.axis("off")
    side.text(0.02, 0.80, r"$G = -I + \lambda\, d$", fontsize=18, color=INK)
    side.text(0.02, 0.55, "I: what its looks would teach about the chair\n"
              "d: how far it ends from the door\n"
              "darker path = lower G", fontsize=11.5, color=MUTED, va="top")
    side.text(0.02, -0.05, "illustrative numbers", fontsize=11, color=MUTED)
    cap.clear(); cap.axis("off")
    if not st["moving"]:
        msg = "it imagines futures and scores each one: what it would learn, and how near the goal it ends"
    elif b["I"] > 0.25 * b["prag"]:
        msg = "the chair is uncertain: paths that look at it pay, so it turns to look"
    else:
        msg = "the chair is known: looking teaches little, so the goal decides"
    cap.text(0.5, 0.5, msg, ha="center", va="center", fontsize=17, color=INK)
    fig.canvas.draw()
    ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if poster_at is None and f > 3 * FPS and not st["moving"] and not PREVIEW:
        fig.savefig(HERE / "figures" / "efe_paths_poster.png", dpi=DPI); poster_at = f
ff.stdin.close(); ff.wait()
print("written", out)
