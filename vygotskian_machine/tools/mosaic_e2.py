#!/usr/bin/env python3
"""Figure for E2 (paper §5.3): six ScanNet rooms, the frame the image baseline saw, the annotated objects, and the
top function under each source. Reads tools/e2_out/e2_per_room_scannet.csv (tools/e2_per_room.py) and the frames
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
           ("image", "One image")]
GOOD, BAD, INK, MUTED = "#008300", "#c0392b", "#1f1f1e", "#6b6a63"


def main() -> None:
    per = {r["room"]: r for r in csv.DictReader((HERE / "e2_out/e2_per_room_scannet.csv").open())}
    frame = {r["scan"]: r["frames"][0] for r in map(json.loads, (HERE / "e2_out/image_answers.jsonl").open())
             if r["k"] == 1 and r.get("num_ctx") is None}
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 3, figsize=(6.9, 4.9), gridspec_kw={"hspace": 1.05, "wspace": 0.06})
    for ax, room in zip(axs.flat, ROOMS):
        r = per[room]
        ax.imshow(plt.imread(FRAMES / room / "color" / frame[room]))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{r['truth']}", fontsize=8, fontweight="bold", color=INK, loc="left", pad=3)
        objs = textwrap.fill(r["objects"].replace(";", ", "), 36)
        ax.text(0, -0.03, objs, transform=ax.transAxes, va="top", fontsize=6.3, color=MUTED, style="italic",
                linespacing=1.1)
        for i, (k, name) in enumerate(SOURCES):
            ok = r[k] == r["truth"]
            y = -0.21 - 0.1 * i
            ax.text(0, y, "✓" if ok else "✗", transform=ax.transAxes, va="top", fontsize=7.5,
                    color=GOOD if ok else BAD, fontweight="bold")
            ax.text(0.07, y, f"{name}: {r[k]}", transform=ax.transAxes, va="top", fontsize=6.8, color=INK)
    out = HERE.parent / "paper/figures/e2_rooms.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
