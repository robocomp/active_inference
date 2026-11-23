#!/usr/bin/env python3
"""Figure for E2 (paper §5.3): six ScanNet rooms in one strip (the frame the image baseline saw, the annotated
objects), and below them a grid of the top function under each source (✓, or ✗ and the wrong answer). Reads tools/e2_out/e2_per_room_scannet.csv (tools/e2_per_room.py) and the frames
cached in tools/e2_out/image_answers.jsonl; writes paper/figures/e2_rooms.pdf.

    python3 tools/mosaic_e2.py
"""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAMES = Path("/media/datadisk/scannet/tasks/scannet_frames_25k")
ROOMS = ["scene0003_00", "scene0217_00", "scene0061_00", "scene0026_00", "scene0093_00", "scene0497_00"]
SOURCES = [("vlm", "VLM elicited"), ("vlm_cal", "VLM calibrated"), ("handwritten", "Hand-written"),
           ("image", "One image"), ("objects", "Object list")]
GOOD, BAD, INK, MUTED = "#008300", "#c0392b", "#1f1f1e", "#6b6a63"


def main() -> None:
    per = {r["room"]: r for r in csv.DictReader((HERE / "e2_out/e2_per_room_scannet.csv").open())}
    for rec in map(json.loads, (HERE / "e2_out/objects_answers.jsonl").open()):
        per[rec["room"]]["objects_answer"] = rec["answer"]
    frame = {r["scan"]: r["frames"][0] for r in map(json.loads, (HERE / "e2_out/image_answers.jsonl").open())
             if r["k"] == 1 and r.get("num_ctx") is None}
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    # one strip of six rooms (fits a two-column page width), then a source x room grid of the top functions
    fig = plt.figure(figsize=(6.9, 2.2))
    gs = fig.add_gridspec(2, 7, width_ratios=[0.5] + [1] * 6, height_ratios=[1, 0.78], hspace=0.42, wspace=0.06)
    for c, room in enumerate(ROOMS, start=1):
        r = per[room]
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(plt.imread(FRAMES / room / "color" / frame[room]))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(r["truth"], fontsize=7.5, fontweight="bold", color=INK, loc="left", pad=2)
        objs = textwrap.fill(r["objects"].replace(";", ", "), 21)
        ax.text(0, -0.04, objs, transform=ax.transAxes, va="top", fontsize=4.9, color=MUTED, style="italic",
                linespacing=1.05)
    g = fig.add_subplot(gs[1, :])
    g.set_xlim(0, 6.5), g.set_ylim(len(SOURCES), 0)
    g.axis("off")
    for i, (k, name) in enumerate(SOURCES):
        y = i + 0.5
        g.text(0.48, y, name, ha="right", va="center", fontsize=6.3, color=INK)
        for c, room in enumerate(ROOMS):
            r = per[room]
            kk = "objects_answer" if k == "objects" else k
            ok = r[kk] == r["truth"]
            x = 0.5 + c * 1.0 + 0.03
            g.text(x, y, "✓" if ok else "✗", ha="left", va="center", fontsize=7, color=GOOD if ok else BAD,
                   fontweight="bold")
            if not ok:
                g.text(x + 0.13, y, r[kk], ha="left", va="center", fontsize=6, color=BAD)
    out = HERE.parent / "paper/figures/e2_rooms.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
