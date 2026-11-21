#!/usr/bin/env python3
"""mask_figure.py — paper figure: a ZED RGB frame next to the masks retina published for it.

Reads a retina [ZedDump] directory recorded with `dump_masks = true` (see retina/src/mask_dump_csv.h
for the file layout). Needs no running fleet.

    python3 -I retina/tools/mask_figure.py DUMP_DIR --pick            # list good candidate stamps
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP              # (a) RGB + (b) masks
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP --topview    # + (c) top view of one mask
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP --topview --mask 2 --out fig/masks

Panels
  (a) the RGB frame (<stamp>.jpg)
  (b) translucent masks with "class score" labels; solid outline = YOLO-seg, dashed = ADE20K semantic
  (c) optional: one mask's camera-frame support points, placed in the room frame with the camera pose
      row for the same stamp in poses.csv (cam_x, cam_y, cam_yaw). The pose file only carries yaw, so
      the camera is taken as level (roll = pitch = 0) — exact for the Shadow ZED mount up to its tilt.

Output: <out>.pdf and <out>.png, 17 cm wide, 300 dpi (default <out> = <DUMP_DIR>/fig_<stamp>).
--pick lists stamps with >= 2 distinct classes and at least one semantic-only mask.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CM = 1.0 / 2.54
FIG_W_IN = 17.0 * CM
DPI = 300
PALETTE = plt.get_cmap("tab10").colors


# ── loading ──────────────────────────────────────────────────────────────────────────────────────────
def masks_path(d, stamp, suffix=""):
    return os.path.join(d, f"{stamp}.masks{suffix}.csv")


def load_index(d, stamp):
    with open(masks_path(d, stamp), newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(dict(
            mask=int(r["mask"]), label=r["label"], label_id=int(r["label_id"]),
            confidence=float(r["confidence"]), segmenter=r["segmenter"],
            n_pixels=int(r["n_pixels"]), n_points=int(r["n_points"]),
            stride=int(r["pixel_stride"]), img_w=int(r["img_w"]), img_h=int(r["img_h"])))
    return out


def load_long(path, width):
    """mask -> (N, width) float array."""
    acc = defaultdict(list)
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if row:
                acc[int(row[0])].append([float(v) for v in row[1:1 + width]])
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def load_pose(d, stamp):
    p = os.path.join(d, "poses.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["stamp_ms"]) == stamp:
                return dict(x=float(r["cam_x"]), y=float(r["cam_y"]), z=float(r["cam_z"]),
                            yaw=float(r["cam_yaw"]))
    return None


def list_stamps(d):
    return sorted(int(n.split(".")[0]) for n in os.listdir(d)
                  if n.endswith(".masks.csv") and n.split(".")[0].isdigit())


# ── --pick ───────────────────────────────────────────────────────────────────────────────────────────
def pick(d):
    found = 0
    for s in list_stamps(d):
        if not os.path.exists(os.path.join(d, f"{s}.jpg")):
            continue
        idx = load_index(d, s)
        labels = {m["label"] for m in idx}
        sem_only = sorted({m["label"] for m in idx if m["segmenter"] == "semantic"}
                          - {m["label"] for m in idx if m["segmenter"] == "yolo"})
        if len(labels) >= 2 and sem_only:
            found += 1
            desc = ", ".join(f"{m['label']}({m['segmenter'][0]}) {m['confidence']:.2f}" for m in idx)
            print(f"{s}  masks={len(idx)}  classes={len(labels)}  semantic-only={sem_only}  [{desc}]")
    if not found:
        print("no stamp with >= 2 classes and a semantic-only mask", file=sys.stderr)
    return 0 if found else 1


# ── drawing ──────────────────────────────────────────────────────────────────────────────────────────
def silhouette(px, stride, h, w):
    """Decimated (col,row) pixels -> full-resolution binary mask (each sample covers stride x stride)."""
    m = np.zeros((h, w), np.uint8)
    if px is None or len(px) == 0:
        return m
    cols = np.clip(px[:, 0].astype(int), 0, w - 1)
    rows = np.clip(px[:, 1].astype(int), 0, h - 1)
    for dr in range(stride):
        for dc in range(stride):
            m[np.clip(rows + dr, 0, h - 1), np.clip(cols + dc, 0, w - 1)] = 1
    if stride > 1:   # close the sampling grid's gaps so the overlay reads as a region
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stride + 1, 2 * stride + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def class_colours(idx):
    labels = []
    for m in idx:
        if m["label"] not in labels:
            labels.append(m["label"])
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def draw_masks(ax, rgb, idx, pix, colours, min_conf):
    h, w = rgb.shape[:2]
    over = rgb.astype(float) / 255.0
    shown = [m for m in idx if m["confidence"] >= min_conf]
    sils = {}
    for m in shown:
        sil = silhouette(pix.get(m["mask"]), m["stride"], h, w)
        sils[m["mask"]] = sil
        c = np.asarray(colours[m["label"]])
        a = 0.45
        sel = sil.astype(bool)
        over[sel] = (1 - a) * over[sel] + a * c
    ax.imshow(np.clip(over, 0, 1), interpolation="nearest")
    for m in shown:
        sil = sils[m["mask"]]
        if not sil.any():
            continue
        cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ls = "--" if m["segmenter"] == "semantic" else "-"
        for c in cnts:
            c = c[:, 0, :]
            c = np.vstack([c, c[:1]])
            ax.plot(c[:, 0], c[:, 1], ls, color=colours[m["label"]], lw=0.9)
        ys, xs = np.nonzero(sil)
        # Label at the silhouette's top-centre, nudged inside the frame.
        tx = float(np.clip(xs.mean(), 0.08 * w, 0.92 * w))
        ty = float(np.clip(ys.min() + 0.04 * h, 0.05 * h, 0.95 * h))
        ax.text(tx, ty, f"{m['label']} {m['confidence']:.2f}", fontsize=6, ha="center", va="top",
                color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colours[m["label"]],
                                         lw=0.6, alpha=0.85))


def draw_topview(ax, idx, pts, pose, chosen, colours):
    c, s = np.cos(pose["yaw"]), np.sin(pose["yaw"])
    R = np.array([[c, -s], [s, c]])
    t = np.array([pose["x"], pose["y"]])

    def to_room(p):
        return p[:, :2] @ R.T + t   # camera x right, y forward (depth); level camera assumed

    rng = np.random.default_rng(0)
    for m in idx:
        p = pts.get(m["mask"])
        if p is None or len(p) == 0 or m["mask"] == chosen["mask"]:
            continue
        q = to_room(p)
        if len(q) > 3000:
            q = q[rng.choice(len(q), 3000, replace=False)]
        ax.scatter(q[:, 0], q[:, 1], s=0.3, color="0.75", linewidths=0, rasterized=True)
    q = to_room(pts[chosen["mask"]])
    if len(q) > 8000:
        q = q[rng.choice(len(q), 8000, replace=False)]
    ax.scatter(q[:, 0], q[:, 1], s=0.6, color=colours[chosen["label"]], linewidths=0, rasterized=True)
    # Camera: position + optical axis (camera +y).
    fwd = R @ np.array([0.0, 1.0])
    ax.plot(*t, marker="o", ms=3, color="black")
    span = max(0.3, 0.25 * float(np.max(np.linalg.norm(q - t, axis=1))))
    ax.annotate("", xy=t + span * fwd, xytext=t,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("room x (m)", fontsize=7)
    ax.set_ylabel("room y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, lw=0.3, color="0.85")


def choose_mask(idx, pts, want):
    if want is not None:
        for m in idx:
            if m["mask"] == want:
                return m if len(pts.get(want, [])) else None
        return None
    cands = [m for m in idx if len(pts.get(m["mask"], []))]
    if not cands:
        return None
    # Prefer a semantic-only class (the point of the figure), then the most support points.
    yolo_labels = {m["label"] for m in idx if m["segmenter"] == "yolo"}
    return max(cands, key=lambda m: (m["segmenter"] == "semantic" and m["label"] not in yolo_labels,
                                     m["n_points"]))


# ── main ─────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("stamp", nargs="?", type=int)
    ap.add_argument("--pick", action="store_true", help="list candidate stamps and exit")
    ap.add_argument("--topview", action="store_true", help="add panel (c): one mask's points, room frame")
    ap.add_argument("--mask", type=int, default=None, help="mask index for panel (c) (default: auto)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="hide masks below this confidence")
    ap.add_argument("--out", default=None, help="output path without extension")
    a = ap.parse_args()

    if a.pick:
        return pick(a.dump_dir)
    if a.stamp is None:
        ap.error("give a STAMP or --pick")

    jpg = os.path.join(a.dump_dir, f"{a.stamp}.jpg")
    bgr = cv2.imread(jpg, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read {jpg}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not os.path.exists(masks_path(a.dump_dir, a.stamp)):
        sys.exit(f"no {masks_path(a.dump_dir, a.stamp)} — was the dump recorded with dump_masks = true?")
    idx = load_index(a.dump_dir, a.stamp)
    pix = load_long(masks_path(a.dump_dir, a.stamp, ".pix"), 2)
    if idx and (idx[0]["img_w"], idx[0]["img_h"]) != (rgb.shape[1], rgb.shape[0]):
        sys.exit(f"mask pixel space {idx[0]['img_w']}x{idx[0]['img_h']} != image {rgb.shape[1]}x{rgb.shape[0]}")
    colours = class_colours(idx)

    chosen = pose = pts = None
    if a.topview:
        pts = load_long(masks_path(a.dump_dir, a.stamp, ".pts"), 3)
        pose = load_pose(a.dump_dir, a.stamp)
        chosen = choose_mask(idx, pts, a.mask)
        if pose is None:
            print(f"no poses.csv row for {a.stamp}; skipping panel (c)", file=sys.stderr)
        elif chosen is None:
            print("no mask with support points for panel (c); skipping it", file=sys.stderr)
    three = chosen is not None and pose is not None

    h, w = rgb.shape[:2]
    n = 3 if three else 2
    widths = [1.0, 1.0, 0.8] if three else [1.0, 1.0]
    img_w_in = FIG_W_IN * 1.0 / sum(widths)
    fig_h = img_w_in * h / w + 0.35
    fig, axes = plt.subplots(1, n, figsize=(FIG_W_IN, fig_h), gridspec_kw=dict(width_ratios=widths))

    axes[0].imshow(rgb, interpolation="nearest")
    draw_masks(axes[1], rgb, idx, pix, colours, a.min_conf)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    if three:
        draw_topview(axes[2], idx, pts, pose, chosen, colours)
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", fontsize=8, loc="left")
    if three:
        axes[2].set_title(f"{chosen['label']}, {chosen['n_points']} pts", fontsize=7, loc="right")
    # Outline-style key for (b).
    axes[1].plot([], [], "-", color="0.3", lw=0.9, label="YOLO-seg")
    axes[1].plot([], [], "--", color="0.3", lw=0.9, label="ADE20K semantic")
    axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                   handlelength=2.2, borderaxespad=0.2)

    fig.tight_layout(pad=0.3, w_pad=0.6)
    out = a.out or os.path.join(a.dump_dir, f"fig_{a.stamp}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf", dpi=DPI)
    fig.savefig(out + ".png", dpi=DPI)
    print(f"wrote {out}.pdf and {out}.png ({len(idx)} masks{', top view of mask %d' % chosen['mask'] if three else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
