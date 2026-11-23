#!/usr/bin/env python3
"""Figure for E2 (paper §5.3): function evidence on ScanNet, held out.

(a) macro F1 and top-1 accuracy per source of π_cf, plus the one-image and object-list baselines;
(b) F1 per function and source, as a heatmap with the values printed.
Reads tools/e2_out/e2_scannet_rooms.csv and tools/e2_out/image_answers.jsonl (k=1); writes paper/figures/e2_results.pdf.

    python3 tools/plot_e2.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from function_evidence_e2 import scores  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dd"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

ORDER = [("objects", "VLM asked, object list"), ("cooc", "Co-occurrence"), ("vlm_cal", "VLM, calibrated"), ("handwritten", "Hand-written"),
         ("image", "One image"), ("vlm", "VLM, as elicited"), ("uniform", "Uniform")]
FUNCS = [("bathroom", "bath"), ("kitchen", "kitchen"), ("bedroom", "bed"), ("office", "office"),
         ("living room", "living"), ("corridor", "corr.")]


def load() -> dict[str, tuple[float, float, dict[str, float]]]:
    out: dict[str, tuple[float, float, dict[str, float]]] = {}
    for r in csv.DictReader((HERE / "e2_out/e2_scannet_rooms.csv").open()):
        m = out.setdefault(r["source"], (float(r["macro_f1"]), float(r["top1"]), {}))
        m[2][r["function"]] = float(r["f1"])
    rooms = list(csv.DictReader((HERE / "corpora/scannet_rooms.csv").open()))
    truth = [r["function"] for r in rooms]
    funcs = sorted(set(truth))
    ans = {r["scan"]: r["answer"] for r in map(json.loads, (HERE / "e2_out/image_answers.jsonl").open())
           if r["k"] == 1 and r.get("num_ctx") is None}
    obj = {r["room"]: r["answer"] for r in map(json.loads, (HERE / "e2_out/objects_answers.jsonl").open())}
    for name, src in (("image", ans), ("objects", obj)):
        a = [src[r["room"]] for r in rooms]
        mf1, per = scores([{x} for x in a], truth, funcs, list(range(len(rooms))))
        out[name] = (mf1, sum(x == t for x, t in zip(a, truth)) / len(rooms), {f: v[2] for f, v in per.items()})
    return out


def main() -> None:
    d = load()
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.32})

    ys = list(range(len(ORDER)))[::-1]
    for y, (k, _) in zip(ys, ORDER):
        f1, t1, _ = d[k]
        a.plot([f1, t1], [y + 0.13, y - 0.13], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        a.scatter(f1, y + 0.13, s=34, color=BLUE, marker="o", zorder=3, edgecolors="white", linewidths=1)
        a.scatter(t1, y - 0.13, s=34, color=ORANGE, marker="D", zorder=3, edgecolors="white", linewidths=1)
        if k != "uniform":
            a.text(max(f1, t1) + 0.025, y, f"{f1:.2f} / {t1:.2f}", va="center", fontsize=7, color=MUTED)
    a.set_yticks(ys, [n for _, n in ORDER])
    a.set_xlim(0, 1.12)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.grid(axis="x", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.set_title("(a) ScanNet, 457 rooms: macro F1 / top-1", fontsize=8, loc="left", color=INK)
    a.scatter([], [], s=34, color=BLUE, marker="o", label="macro F1")
    a.scatter([], [], s=34, color=ORANGE, marker="D", label="top-1")
    a.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.2, borderaxespad=0.1)

    rows = [k for k, _ in ORDER if k != "uniform"]
    M = [[d[k][2][f] for f, _ in FUNCS] for k in rows]
    b.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            b.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    b.set_xticks(range(len(FUNCS)), [n for _, n in FUNCS])
    SHORT = {"objects": "Obj. list", "cooc": "Co-occ.", "vlm_cal": "VLM cal.", "handwritten": "Hand-wr.", "image": "Image",
             "vlm": "VLM elic."}
    b.set_yticks(range(len(rows)), [SHORT[k] for k in rows])
    b.tick_params(length=0)
    b.xaxis.tick_top()
    for s in b.spines.values():
        s.set_visible(False)
    b.set_title("(b) F1 per function", fontsize=8, loc="left", color=INK, pad=14)

    out = HERE.parent / "paper/figures/e2_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
