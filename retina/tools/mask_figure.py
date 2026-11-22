#!/usr/bin/env python3
"""mask_figure.py — paper figure: a ZED RGB frame next to the masks retina published for it.

Reads a retina [ZedDump] directory recorded with `dump_masks = true` (see retina/src/mask_dump_csv.h
for the file layout). Needs no running fleet.

    python3 -I retina/tools/mask_figure.py DUMP_DIR --pick            # list good candidate stamps
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP              # (a) RGB + (b) masks
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP --topview    # + (c) top view of one mask
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP --topview --mask 2 --out fig/masks
    python3 -I retina/tools/mask_figure.py DUMP_DIR STAMP --masks 0,5,8 --topview --mask 0 \
        --instance door_1 --ai2-log RUN/ai2_log.csv          # chosen masks only + instance box in (c)

Options added for paper use: --masks i,j,k shows only those mask indices (print the omitted ones in
the caption); --instance NAME [--ai2-log CSV] draws the instance's final box (cx,cy,w,s,yaw from the
last ai2_log row of that node) in the room frame in (c), next to the points of --mask.
Labels in (b) are placed with collision avoidance and leader lines, always inside the frame.

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


def draw_masks(ax, rgb, idx, pix, colours, min_conf, keep=None):
    h, w = rgb.shape[:2]
    over = rgb.astype(float) / 255.0
    shown = [m for m in idx if m["confidence"] >= min_conf and (keep is None or m["mask"] in keep)]
    sils = {}
    anchors = []
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
            ax.plot(c[:, 0], c[:, 1], ls, color=colours[m["label"]], lw=1.0)
        ys, xs = np.nonzero(sil)
        # anchor = the silhouette pixel closest to its centroid (always on the mask)
        d2 = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
        k = int(np.argmin(d2))
        anchors.append(dict(x=float(xs[k]), y=float(ys[k]), area=int(sil.sum()), colour=colours[m["label"]],
                            text=f"{m['label']} {m['confidence']:.2f}"))
    return anchors


def place_labels(ax, anchors, fs=6.5):
    """Greedy collision-free placement of the "class score" boxes, with leader lines. Call after the
    figure layout is final (it measures text in display pixels)."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    clip = ax.get_window_extent(rend)
    placed = []      # display-pixel boxes (x0, y0, x1, y1) already taken
    lead = []        # leader segments already drawn (display px), to keep them from crossing labels

    def overlap(a, b, pad=2.0):
        return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])

    for an in sorted(anchors, key=lambda a: a["area"]):
        t = ax.text(0, 0, an["text"], fontsize=fs, ha="center", va="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=an["colour"], lw=0.7, alpha=0.92),
                    zorder=5)
        bb = t.get_window_extent(rend)
        bw, bh = bb.width, bb.height
        ax_px = ax.transData.transform((an["x"], an["y"]))
        best = None
        for r in [0] + list(np.arange(0.6, 14, 0.5) * bh):
            for ang in (np.linspace(0, 2 * np.pi, 16, endpoint=False) if r else [0.0]):
                cx = ax_px[0] + r * np.cos(ang)
                cy = ax_px[1] + r * np.sin(ang)
                box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                if box[0] < clip.x0 + 1 or box[2] > clip.x1 - 1 or box[1] < clip.y0 + 1 or box[3] > clip.y1 - 1:
                    continue
                if any(overlap(box, q) for q in placed):
                    continue
                if r and any(overlap((min(ax_px[0], cx), min(ax_px[1], cy), max(ax_px[0], cx), max(ax_px[1], cy)),
                                     q, pad=-3) for q in placed):
                    continue      # the leader would run across another label
                best = (cx, cy, box, r)
                break
            if best:
                break
        if best is None:   # nowhere free: keep it centred on the anchor, clipped inside the frame
            cx = float(np.clip(ax_px[0], clip.x0 + bw / 2 + 1, clip.x1 - bw / 2 - 1))
            cy = float(np.clip(ax_px[1], clip.y0 + bh / 2 + 1, clip.y1 - bh / 2 - 1))
            best = (cx, cy, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 0)
        cx, cy, box, r = best
        placed.append(box)
        px, py = inv.transform((cx, cy))
        t.set_position((px, py))
        ax.plot(an["x"], an["y"], "o", ms=2.2, mfc=an["colour"], mec="white", mew=0.4, zorder=4)
        if r:
            # leader from the anchor to the nearest point on the label box
            qx = float(np.clip(ax_px[0], box[0], box[2]))
            qy = float(np.clip(ax_px[1], box[1], box[3]))
            lx, ly = inv.transform((qx, qy))
            ax.plot([an["x"], lx], [an["y"], ly], "-", color=an["colour"], lw=0.7, zorder=4)


def load_instance(path, name):
    last = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["node"] == name:
                last = r
    if last is None:
        sys.exit(f"no rows for node {name} in {path}")
    return dict(name=name, cx=float(last["cx"]), cy=float(last["cy"]), w=float(last["w"]),
                s=float(last["s"]), yaw=float(last["yaw"]), cycle=int(last["cycle"]))


def draw_topview(ax, idx, pts, pose, chosen, colours, inst=None, keep=None):
    c, s = np.cos(pose["yaw"]), np.sin(pose["yaw"])
    R = np.array([[c, -s], [s, c]])
    t = np.array([pose["x"], pose["y"]])

    def to_room(p):
        return p[:, :2] @ R.T + t   # camera x right, y forward (depth); level camera assumed

    rng = np.random.default_rng(0)
    for m in idx:
        p = pts.get(m["mask"])
        if p is None or len(p) == 0 or m["mask"] == chosen["mask"] or (keep is not None and m["mask"] not in keep):
            continue
        q = to_room(p)
        if len(q) > 3000:
            q = q[rng.choice(len(q), 3000, replace=False)]
        ax.scatter(q[:, 0], q[:, 1], s=0.3, color="0.75", linewidths=0, rasterized=True)
    q = to_room(pts[chosen["mask"]])
    if len(q) > 8000:
        q = q[rng.choice(len(q), 8000, replace=False)]
    ax.scatter(q[:, 0], q[:, 1], s=1.5, color=colours[chosen["label"]], linewidths=0, rasterized=True, zorder=4)
    extent = [q, t[None, :]]
    if inst is not None:
        ci, si = np.cos(inst["yaw"]), np.sin(inst["yaw"])
        Ri = np.array([[ci, -si], [si, ci]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * 0.5 * np.array([inst["w"], inst["s"]])
        box = corners @ Ri.T + np.array([inst["cx"], inst["cy"]])
        ax.fill(box[:, 0], box[:, 1], facecolor="0.9", edgecolor="black", lw=0.8, zorder=1)
        ax.plot([], [], "-", color="black", lw=1.0, label=f"instance {inst['name']}")
        extent.append(box)
    # Camera: position + optical axis (camera +y).
    fwd = R @ np.array([0.0, 1.0])
    ax.plot(*t, marker="o", ms=3, color="black")
    span = max(0.3, 0.25 * float(np.max(np.linalg.norm(q - t, axis=1))))
    ax.annotate("", xy=t + span * fwd, xytext=t,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    allp = np.vstack(extent)
    lo, hi = allp.min(0), allp.max(0)
    ctr, half = (lo + hi) / 2, max((hi - lo).max() / 2 * 1.15, 0.5)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("room x (m)", fontsize=7)
    ax.set_ylabel("room y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, lw=0.3, color="0.85")
    ax.plot([], [], "o", ms=3, color="black", label="camera (arrow = optical axis)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, handlelength=1.5)


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
    ap.add_argument("--masks", default=None, help="comma-separated mask indices to show (default: all)")
    ap.add_argument("--instance", default=None, help="instance node name whose final box is drawn in (c)")
    ap.add_argument("--ai2-log", default=None, help="ai2_log.csv for --instance (default: DUMP_DIR/../ai2_log.csv)")
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
    keep = None if a.masks is None else {int(v) for v in a.masks.split(",")}
    if keep is not None:
        omitted = [f"{m['mask']}:{m['label']} {m['confidence']:.2f}({m['segmenter']})" for m in idx if m["mask"] not in keep]
        print("omitted masks:", ", ".join(omitted) or "none")
    inst = None
    if a.instance:
        inst = load_instance(a.ai2_log or os.path.join(a.dump_dir, "..", "ai2_log.csv"), a.instance)

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
    if three:
        fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN * 0.645))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], wspace=0.10, hspace=0.12)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_W_IN / 2 * h / w + 0.35))
        axes = list(axes)

    axes[0].imshow(rgb, interpolation="nearest")
    anchors = draw_masks(axes[1], rgb, idx, pix, colours, a.min_conf, keep)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    if three:
        draw_topview(axes[2], idx, pts, pose, chosen, colours, inst, keep)
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", fontsize=8, loc="left", pad=2)
    if three:
        axes[2].set_title(f"{chosen['label']} {chosen['confidence']:.2f}: {chosen['n_points']} points",
                          fontsize=6.5, loc="right", pad=2)
    axes[1].plot([], [], "-", color="0.3", lw=1.0, label="YOLO-seg")
    axes[1].plot([], [], "--", color="0.3", lw=1.0, label="ADE20K semantic")
    axes[1].legend(fontsize=6, loc="lower right", frameon=True, framealpha=0.85, handlelength=2.0,
                   borderpad=0.3).set_zorder(6)

    if not three:
        fig.tight_layout(pad=0.3, w_pad=0.6)
    else:
        fig.subplots_adjust(left=0.01, right=0.985, top=0.96, bottom=0.05)
    fig.canvas.draw()
    place_labels(axes[1], anchors)
    out = a.out or os.path.join(a.dump_dir, f"fig_{a.stamp}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf", dpi=DPI)
    fig.savefig(out + ".png", dpi=DPI)
    print(f"wrote {out}.pdf and {out}.png ({len(idx)} masks{', top view of mask %d' % chosen['mask'] if three else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
