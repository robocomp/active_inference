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
LINES = [("level_map", "VLM order + level map from rooms", BLUE, "o"),
         ("eq5", "VLM + Eq. 5 per cell", ORANGE, "D"),
         ("cooc", "counts alone", AQUA, "s")]
REFS = [("handwritten", "hand-written"), ("vlm", "VLM as elicited")]


def main() -> None:
    rows = list(csv.DictReader((HERE / "e2_out/calib_curve_scannet.csv").open()))
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    ns = sorted({int(r["n_confirmed"]) for r in rows if int(r["n_confirmed"]) > 0})
    pos = {n: i for i, n in enumerate(ns)}
    for ax, col, title in zip(axs, ("macro_f1", "top1"), ("(a) macro F1", "(b) top-1")):
        for name, label in REFS:
            v = [float(r[col]) for r in rows if r["source"] == name]
            y = sum(v) / len(v)
            ax.axhline(y, color=MUTED, lw=0.8, ls="--" if name == "handwritten" else ":")
            ax.text(len(ns) - 1, y + 0.012, label, ha="right", va="bottom", fontsize=6.5, color=MUTED)
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
            ax.plot(xs, mean, color=color, lw=2, marker=m, ms=4, label=label)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_xlabel("person-confirmed rooms", color=INK)
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=8, loc="left", color=INK)
    axs[0].legend(loc="lower right", frameon=False, fontsize=6.8)
    out = HERE.parent / "paper/figures/e2_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
