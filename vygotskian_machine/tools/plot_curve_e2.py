#!/usr/bin/env python3
"""Figure for E2 (paper §5.3): macro F1 and top-1 on the held-out half of ScanNet against the number of
person-confirmed rooms, for the shared level map, Eq. (lh_beta_prior) per cell, and counts alone.
Reads tools/e2_out/calib_curve_scannet.csv (tools/calib_curve_e2.py); writes paper/figures/e2_curve.pdf.

    python3 tools/plot_curve_e2.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a63", "#e4e3dd"
LINES = [("level_map", "VLM order + level map", BLUE, "o"),
         ("eq5", "VLM + Eq. 5 per cell", ORANGE, "D"),
         ("cooc", "counts alone", AQUA, "s")]
REFS = [("handwritten", "hand-written"), ("vlm", "VLM elicited")]


def main() -> None:
    rows = list(csv.DictReader((HERE / "e2_out/calib_curve_scannet.csv").open()))
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.3), sharey=True, gridspec_kw={"wspace": 0.1})
    ns = sorted({int(r["n_confirmed"]) for r in rows if int(r["n_confirmed"]) > 0})
    pos = {n: i for i, n in enumerate(ns)}
    for ax, col, title in zip(axs, ("macro_f1", "top1"), ("(a) macro F1", "(b) top-1")):
        for name, label in REFS:
            v = [float(r[col]) for r in rows if r["source"] == name]
            y = sum(v) / len(v)
            ax.axhline(y, color=MUTED, lw=0.8, ls="--" if name == "handwritten" else ":", label=label)
        for name, label, color, m in LINES:
            by = defaultdict(list)
            for r in rows:
                if r["source"] == name and int(r["n_confirmed"]) > 0:
                    by[int(r["n_confirmed"])].append(float(r[col]))
            xs = [pos[n] for n in ns]
            mean = [sum(by[n]) / len(by[n]) for n in ns]
            lo = [sorted(by[n])[int(0.025 * len(by[n]))] for n in ns]
            hi = [sorted(by[n])[min(len(by[n]) - 1, int(0.975 * len(by[n])))] for n in ns]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(xs, mean, color=color, lw=1.4, marker=m, ms=2.8, label=label)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns], fontsize=6)
        ax.set_xlabel("confirmed rooms", color=INK, fontsize=6.5, labelpad=1)
        ax.set_ylim(0.25, 0.95)
        ax.set_yticks([0.3, 0.5, 0.7, 0.9])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=7, loc="left", color=INK, pad=2)
    h, l = axs[0].get_legend_handles_labels()
    h, l = h[2:] + h[:2], l[2:] + l[:2]  # curves first, reference lines second
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=5.8,
               handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
    out = HERE.parent / "paper/figures/e2_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
