"""Day 3, 'What an object agent sees: a mask'. Run from slides/ with ../.venv/bin/python.

Panels a and b are REAL: a crop of a simulated 360-degree frame (retina/etc/depth_frames/1786014449567.jpg,
x 960..1920, y 300..780) and the masks retina's own segmenter (yoloe-26x-seg-household-v3.onnx, conf 0.25,
CPU) returned for it, cached in masks_src/ so this script needs neither the model nor the robot.
Panel c is a SKETCH: points sampled from the chair template, not this frame's depth.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
SRC = HERE / "masks_src"
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})
COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"
X0, X1, Y0, Y1 = 60, 620, 50, 345          # the part of the crop shown

img = plt.imread(SRC / "frame_crop.png")
dets = json.load(open(SRC / "polys.json"))


def frame(ax):
    ax.imshow(img)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")


# a: what the camera delivers
fig, ax = plt.subplots(figsize=(4.2, 2.2)); frame(ax)
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(HERE / "d3_mask_a.pdf"); plt.close(fig)

# b: the masks, each with a label and a score; a runner-up label shares the same pixels
SHOW = [("radiator", GOOD, (64, 192), None),
        ("coffee table", COOL, (100, 318), "desk"),
        ("refrigerator", ACCENT, (360, 70), None),
        ("tv", "#8E44AD", (440, 282), "microwave"),
        ("cube shelf", "#D68910", (150, 70), None)]
fig, ax = plt.subplots(figsize=(4.2, 2.2)); frame(ax)
ax.imshow(np.full_like(img[..., :3], 1.0), alpha=0.45)        # wash out the frame so masks read
for label, col, (tx, ty), second in SHOW:
    d = next(d for d in dets if d["label"] == label)
    ax.add_patch(Polygon(d["poly"], closed=True, fc=col, ec=col, alpha=0.55, lw=1.2))
    txt = f"{label} {d['score']:.2f}"
    if second:
        d2 = next(d for d in dets if d["label"] == second)
        txt += f"\nor {second} {d2['score']:.2f}"
    ax.text(tx, ty, txt, fontsize=8.5, color="white", fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(HERE / "d3_mask_b.pdf"); plt.close(fig)

# c: a chair mask, deprojected through depth and the pose (sketch from the chair template)
rng = np.random.default_rng(3)
cam = np.array([-2.6, 0.0, 1.25])


def noisy(p):
    rngs = np.linalg.norm(p - cam, axis=1)
    sig = 0.004 + 0.0025 * rngs**2                                   # depth noise grows with range
    return p + rng.normal(size=p.shape) * sig[:, None] * (p - cam) / rngs[:, None]


# the chair template (chair.concept.toml): seat 0.60 x 0.52, top at 0.595, backrest 0.655 on the far edge, 4 legs
W, D, SH, BH, LG = 0.60, 0.52, 0.595, 0.655, 0.0375
seat = np.c_[rng.uniform(-D/2, D/2, 380), rng.uniform(-W/2, W/2, 380), np.full(380, SH)]
back = np.c_[np.full(300, D/2), rng.uniform(-W/2, W/2, 300), rng.uniform(SH, SH + BH, 300)]
legs = np.vstack([np.c_[np.full(45, lx - LG), np.full(45, ly) + rng.uniform(-LG, LG, 45), rng.uniform(0.02, SH - 0.075, 45)]
                  for lx in (-D/2 + LG, D/2 - LG) for ly in (-W/2 + LG, W/2 - LG)])
inl = noisy(np.r_[seat, back, legs])
# what a loose mask edge lets in: floor around the base, the wall far behind the top's rim
floor = np.c_[rng.uniform(-0.55, 0.2, 30), rng.uniform(-0.4, 0.4, 30), np.zeros(30)]
wall = np.c_[np.full(18, 1.6), rng.uniform(-0.4, 0.4, 18), rng.uniform(0.9, 1.25, 18)]
out = noisy(np.r_[floor, wall])

fig = plt.figure(figsize=(3.4, 2.6)); ax = fig.add_subplot(projection="3d")
ax.scatter(*inl.T, s=1.6, c=COOL, alpha=0.8, depthshade=False)
ax.scatter(*out.T, s=7, c=ACCENT, marker="x", lw=0.9, depthshade=False)
ax.set_box_aspect((1.4, 1.2, 1.3)); ax.view_init(elev=18, azim=-125)
ax.set_xlim(-0.55, 1.65); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.3)
ax.set_axis_off()
fig.subplots_adjust(-0.3, -0.3, 1.25, 1.2); fig.savefig(HERE / "d3_mask_c.pdf"); plt.close(fig)
