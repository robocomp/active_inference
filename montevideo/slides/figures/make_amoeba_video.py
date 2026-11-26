"""Day 2 opener: an amoeba drifts in a pond and a model of the pond grows inside it.

The amoeba meets the pond only through its membrane: sensors on it light up when something is
within reach, and only what they touch enters the model drawn inside the cell. The model is
coarser than the pond, sharp where the amoeba has been, and fades back to grey where it has not
been for a while. In the last part the amoeba heads for the places its model knows least.

Writes media/amoeba_pond.mp4 and figures/amoeba_pond_poster.png (run from slides/).
Usage: python3 figures/make_amoeba_video.py [--preview]   (--preview: low resolution)
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent.parent          # slides/
PREVIEW = "--preview" in sys.argv
FPS = 30
DURATION = 42.0
W_IN, H_IN, DPI = (12.8, 7.2, 50) if PREVIEW else (12.8, 7.2, 150)   # 1920x1080 in full
LX, LY = 16.0, 9.0                                     # pond size, in world units
rng = np.random.default_rng(7)

# ---- colours: the deck's palette ------------------------------------------------------------------
ACCENT = np.array([0xC0, 0x39, 0x2B]) / 255
COOL = np.array([0x2E, 0x6D, 0xB4]) / 255
GOOD = np.array([0x2E, 0x8B, 0x57]) / 255
ROCK = np.array([0x7D, 0x6A, 0x58]) / 255
WARM = np.array([0xE5, 0x98, 0x66]) / 255
WATER = np.array([0xDD, 0xEB, 0xF4]) / 255
UNKNOWN = np.array([0.80, 0.80, 0.80])

# ---- the pond (the hidden world) -------------------------------------------------------------------
NX, NY = 640, 360                                      # world raster
xs = (np.arange(NX) + 0.5) / NX * LX
ys = (np.arange(NY) + 0.5) / NY * LY
X, Y = np.meshgrid(xs, ys)


def blob(cx, cy, r, wob=0.25, seed=0):
    """A soft irregular blob: 1 inside, 0 outside."""
    g = np.random.default_rng(seed)
    th = np.arctan2(Y - cy, X - cx)
    rr = r * (1 + wob * sum(g.normal() * np.sin(k * th + g.uniform(0, 6.28)) / k for k in range(2, 6)))
    d = np.hypot(X - cx, Y - cy)
    return np.clip((rr - d) / 0.08, 0, 1)


rocks = [(3.2, 6.9, 0.75), (12.6, 2.1, 0.95), (9.0, 6.6, 0.55), (5.6, 1.7, 0.6), (14.3, 7.0, 0.5)]
rock_mask = np.zeros_like(X)
for i, (cx, cy, r) in enumerate(rocks):
    rock_mask = np.maximum(rock_mask, blob(cx, cy, r, seed=i))
weed_mask = np.zeros_like(X)
for i, (cx, cy) in enumerate([(7.4, 3.6), (1.6, 2.6), (13.0, 5.2), (10.6, 0.9)]):
    for j in range(14):                                # a cluster of small fronds
        a = rng.uniform(0, 2 * np.pi); rd = rng.uniform(0, 0.75)
        weed_mask = np.maximum(weed_mask, blob(cx + rd * np.cos(a), cy + rd * np.sin(a), rng.uniform(0.08, 0.16), seed=100 + 20 * i + j))
warm_mask = np.exp(-((X - 4.8) ** 2 + (Y - 4.6) ** 2) / (2 * 0.9 ** 2)) * 0.9

world_rgb = np.ones((NY, NX, 3)) * WATER
for mask, col in [(warm_mask, WARM), (weed_mask, GOOD), (rock_mask, ROCK)]:
    world_rgb = world_rgb * (1 - mask[..., None]) + col * mask[..., None]
feature = np.clip(rock_mask + weed_mask + warm_mask, 0, 1)   # where there is something to sense

# soft moving water texture
tex_a = gaussian_filter(rng.normal(size=(NY, NX)), 18); tex_a /= np.abs(tex_a).max()
tex_b = gaussian_filter(rng.normal(size=(NY, NX)), 25); tex_b /= np.abs(tex_b).max()

# ---- the model inside the amoeba: coarser than the pond ---------------------------------------------
MX, MY = 96, 54
model_rgb = world_rgb[NY // MY // 2::NY // MY, NX // MX // 2::NX // MX][:MY, :MX]   # what a look reveals
mxs = (np.arange(MX) + 0.5) / MX * LX
mys = (np.arange(MY) + 0.5) / MY * LY
MXg, MYg = np.meshgrid(mxs, mys)
conf = np.zeros((MY, MX))                              # how sure the model is of each cell

# ---- particles drifting in a slow current ----------------------------------------------------------
NP = 160
pp = np.column_stack([rng.uniform(0, LX, NP), rng.uniform(0, LY, NP)])


def current(p, t):
    return np.column_stack([0.25 + 0.15 * np.sin(0.6 * p[:, 1] + 0.3 * t), 0.12 * np.cos(0.5 * p[:, 0] - 0.2 * t)])


# ---- the amoeba ------------------------------------------------------------------------------------
R = 1.75                                               # base radius
SENSE = 1.9                                            # sensing reach from the centre
N_SENSORS = 7
c = np.array([2.6, 4.4]); v = np.array([0.6, 0.0]); heading = 0.1
wander_phase = rng.uniform(0, 100, 2)
T_EXPLORE = 0.58 * DURATION                            # after this, head where the model knows least
T_REVEAL = DURATION - 5.0                              # the model is shown large at the end
target = None


def membrane(c, t, n=160):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R * (1 + 0.07 * np.sin(3 * th + 1.3 * t) + 0.05 * np.sin(5 * th - 2.1 * t) + 0.04 * np.sin(2 * th + 0.7 * t))
    return np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)]), th, r


def at(field, p):
    i = np.clip((p[:, 1] / LY * NY).astype(int), 0, NY - 1); j = np.clip((p[:, 0] / LX * NX).astype(int), 0, NX - 1)
    return field[i, j]


# ---- figure ----------------------------------------------------------------------------------------
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.axis("off")
bg = ax.imshow(world_rgb, extent=(0, LX, 0, LY), origin="lower", interpolation="bilinear", zorder=0)
parts = ax.scatter(pp[:, 0], pp[:, 1], s=4, c=[COOL * 0.6 + 0.4], alpha=0.5, zorder=1, linewidths=0)
reach = Polygon(np.zeros((3, 2)), closed=True, facecolor=(1, 1, 1, 0.0), edgecolor=GOOD, lw=0.0, zorder=2)
ax.add_patch(reach)
cell = Polygon(np.zeros((3, 2)), closed=True, facecolor=ACCENT * 0.08 + 0.92, edgecolor=ACCENT, lw=4.5, zorder=3)
ax.add_patch(cell)
inner = ax.imshow(np.zeros((MY, MX, 4)), origin="lower", interpolation="nearest", zorder=4)
inner_frame = Polygon(np.zeros((3, 2)), closed=True, fill=False, edgecolor=ACCENT * 0.3 + 0.7, lw=0.8, ls=(0, (2, 2)), zorder=5)
ax.add_patch(inner_frame)
self_dot = ax.scatter([0], [0], s=14, c=[ACCENT], zorder=6, linewidths=0)
sensors = ax.scatter(np.zeros(N_SENSORS), np.zeros(N_SENSORS), s=60, c=[GOOD], edgecolors="white", linewidths=0.8, zorder=7)
glows = ax.scatter(np.zeros(N_SENSORS), np.zeros(N_SENSORS), s=300, c=[GOOD], alpha=0.0, linewidths=0, zorder=6)
veil = ax.imshow(np.ones((2, 2, 4)), extent=(0, LX, 0, LY), zorder=8, alpha=0.0)
panel = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.12,rounding_size=0.25", facecolor="white",
                       edgecolor=ACCENT, lw=3, zorder=9, alpha=0.0)
ax.add_patch(panel)
big = ax.imshow(np.ones((MY, MX, 3)), origin="lower", interpolation="nearest", zorder=10, alpha=0.0, extent=(0, 1, 0, 1))


def model_image(opaque=False):
    if opaque:                                         # the large panel: unknown is grey
        k = conf[..., None]
        return UNKNOWN * (1 - k) + model_rgb * k
    return np.dstack([model_rgb * 0.92, np.clip(conf * 1.15, 0, 1)])  # inside the cell: it grows


ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{int(W_IN * DPI)}x{int(H_IN * DPI)}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       str(HERE / "media" / ("amoeba_pond_preview.mp4" if PREVIEW else "amoeba_pond.mp4"))],
                      stdin=subprocess.PIPE)

dt = 1.0 / FPS
n_frames = int(DURATION * FPS)
for f in range(n_frames):
    t = f * dt
    # -- motion: wander first, then go where the model knows least
    if t < T_EXPLORE:
        heading += (0.55 * np.sin(0.31 * t + wander_phase[0]) + 0.35 * np.sin(0.73 * t + wander_phase[1])) * dt
        desired = 0.95 * np.array([np.cos(heading), np.sin(heading)])
    else:
        blur = gaussian_filter(conf, 6)
        if target is None or blur[int(target[1] / LY * MY), int(target[0] / LX * MX)] > 0.55:
            margin = (MXg > R + 0.4) & (MXg < LX - R - 0.4) & (MYg > R + 0.4) & (MYg < LY - R - 0.4)
            margin &= gaussian_filter(rock_mask, 25)[::NY // MY, ::NX // MX][:MY, :MX] < 0.05
            score = np.where(margin, blur + 0.03 * np.hypot(MXg - c[0], MYg - c[1]), np.inf)
            k = np.unravel_index(np.argmin(score), score.shape)
            target = np.array([MXg[k], MYg[k]])
        to = target - c
        desired = 1.1 * to / max(np.linalg.norm(to), 1e-6)
    # keep off the banks and slide round the rocks
    push = np.zeros(2)
    for lo, hi, axis in [(R + 0.3, LX - R - 0.3, 0), (R + 0.3, LY - R - 0.3, 1)]:
        if c[axis] < lo: push[axis] += (lo - c[axis]) * 3
        if c[axis] > hi: push[axis] -= (c[axis] - hi) * 3
    for rx, ry, rr in rocks:
        d = c - np.array([rx, ry]); dn = np.linalg.norm(d)
        if dn < R + rr + 0.2:
            n_ = d / dn; tg = np.array([-n_[1], n_[0]])
            mid = np.array([LX / 2, LY / 2]) - c                       # break ties towards open water
            tang = tg * np.sign(np.dot(desired, tg) + 0.5 * np.dot(mid / np.linalg.norm(mid), tg) + 1e-9)
            push += n_ * (R + rr + 0.2 - dn) * 2.5 + tang * 1.2
    if t < T_EXPLORE and np.linalg.norm(push) > 0.3:
        heading = np.arctan2(*(desired + push)[::-1])                # turn away from what it bumped
    v += (desired + push - v) * 0.04
    c = c + v * dt
    # -- sensing: the model learns what is within reach; elsewhere it slowly forgets
    d = np.hypot(MXg - c[0], MYg - c[1])
    gain = 0.10 * np.exp(-(d / SENSE) ** 4)
    conf = conf + gain * (1 - conf)
    conf *= 0.9993
    # -- water, particles
    shimmer = 0.025 * (np.sin(0.7 * t) * tex_a + np.cos(0.45 * t) * tex_b)
    bg.set_data(np.clip(world_rgb + shimmer[..., None], 0, 1))
    pp = pp + current(pp, t) * dt; pp[:, 0] %= LX; pp[:, 1] %= LY
    parts.set_offsets(pp)
    # -- membrane, the reach of the senses, sensors
    poly, th, r = membrane(c, t)
    cell.set_xy(poly)
    th_r = np.linspace(0, 2 * np.pi, 80)
    reach.set_xy(np.column_stack([c[0] + SENSE * 1.12 * np.cos(th_r), c[1] + SENSE * 1.12 * np.sin(th_r)]))
    reach.set_edgecolor((*GOOD, 0.25)); reach.set_linewidth(1.0); reach.set_linestyle((0, (3, 3)))
    sang = np.linspace(0, 2 * np.pi, N_SENSORS, endpoint=False) + 0.15 * t
    rs = R * (1 + 0.07 * np.sin(3 * sang + 1.3 * t) + 0.05 * np.sin(5 * sang - 2.1 * t) + 0.04 * np.sin(2 * sang + 0.7 * t))
    sp = np.column_stack([c[0] + rs * np.cos(sang), c[1] + rs * np.sin(sang)])
    probe = sp + 0.45 * np.column_stack([np.cos(sang), np.sin(sang)])
    touch = np.maximum(at(feature, sp), at(feature, probe))
    sensors.set_offsets(sp)
    glows.set_offsets(sp); glows.set_alpha(None)
    glows.set_facecolors([(*GOOD, 0.45 * tk) for tk in touch])
    glows.set_sizes(250 + 150 * touch * (1 + 0.3 * np.sin(8 * t)))
    # -- the model inside the cell: the whole pond, shrunk, clipped by the membrane
    w_in = 2.0 * R; h_in = w_in * LY / LX
    ext = (c[0] - w_in / 2, c[0] + w_in / 2, c[1] - h_in / 2, c[1] + h_in / 2)
    inner.set_data(model_image()); inner.set_extent(ext); inner.set_clip_path(cell)
    inner_frame.set_clip_path(cell); inner_frame.set_xy([(ext[0], ext[2]), (ext[1], ext[2]), (ext[1], ext[3]), (ext[0], ext[3])])
    self_dot.set_offsets([[ext[0] + c[0] / LX * w_in, ext[2] + c[1] / LY * h_in]])
    # -- the end: the model, large, next to the pond
    if t > T_REVEAL:
        a = min(1.0, (t - T_REVEAL) / 1.5)
        veil.set_data(np.ones((2, 2, 4)) * np.array([1, 1, 1, 0.45 * a])); veil.set_alpha(None)
        pw = 6.4; ph = pw * LY / LX
        if 'px' not in globals():                      # place the panel once, away from the amoeba
            px = 0.5 if c[0] > LX / 2 else LX - pw - 0.5
            py = LY - ph - 0.5 if c[1] < LY / 2 else 0.5
        panel.set_bounds(px, py, pw, ph); panel.set_alpha(a)
        big.set_data(model_image(opaque=True)); big.set_extent((px, px + pw, py, py + ph)); big.set_alpha(a)
    if f % 60 == 0 and '--debug' in sys.argv: print(f'{t:5.1f} c={c.round(2)} v={v.round(2)} target={target}')
    fig.canvas.draw()
    ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if f == n_frames - 1 and not PREVIEW:
        fig.savefig(HERE / "figures" / "amoeba_pond_poster.png", dpi=100)

ff.stdin.close(); ff.wait()
print("done:", n_frames, "frames")
