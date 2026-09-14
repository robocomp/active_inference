import re, matplotlib
matplotlib.use("Agg")
# ⚠ TYPE 42 (TrueType), NOT matplotlib's DEFAULT TYPE 3. IEEE/PaperCept forbid Type 3 fonts outright
# and the automated checker rejects on them; every figure in this paper was Type 3 until 2026-09-14.
# It is one line here and a resubmission if it is missed.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly

html = open("datasets/matterport_layout/mosaic_cells_120.html", encoding="utf-8").read()
cells = re.findall(
    r'<figure class="cell"><svg viewBox="([^"]+)"[^>]*>'
    r'<polygon class="tru-poly" points="([^"]+)"/>'
    r'<polygon class="est-poly" points="([^"]+)"/></svg>'
    r'<figcaption class="cap[^"]*">(\d+) corners &middot; IoU ([0-9.]+)</figcaption>', html)
print("parsed", len(cells), "cells")

def pts(s):
    return [tuple(map(float, p.split(","))) for p in s.split()]

# Sorted by IoU and sampled evenly, so the panel spans the population instead of flattering it:
# the worst room in the corpus is the first cell shown.
cells = sorted(cells, key=lambda c: float(c[4]))
N_COLS, N_ROWS = 8, 4
n = N_COLS * N_ROWS
idx = [round(i * (len(cells) - 1) / (n - 1)) for i in range(n)]
sel = [cells[i] for i in idx]

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(7.16, 3.9))
for ax, (vb, tru, est, ncor, iou) in zip(axes.ravel(), sel):
    T, E = pts(tru), pts(est)
    ax.add_patch(MPoly(T, closed=True, facecolor="#dfe7ef", edgecolor="#31465c", lw=0.9, zorder=1))
    ax.add_patch(MPoly(E, closed=True, facecolor="none", edgecolor="#c2410c", lw=0.9, zorder=2))
    xs = [p[0] for p in T + E]; ys = [p[1] for p in T + E]
    mx, my = (max(xs) - min(xs)) * 0.06 + 0.1, (max(ys) - min(ys)) * 0.06 + 0.1
    ax.set_xlim(min(xs) - mx, max(xs) + mx); ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, -0.02, iou, transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            color="#31465c")
fig.subplots_adjust(left=0.004, right=0.996, top=0.965, bottom=0.035, wspace=0.06, hspace=0.30)
fig.text(0.004, 0.988, "truth", color="#31465c", fontsize=6.4, va="top", fontweight="bold")
fig.text(0.055, 0.988, "estimate", color="#c2410c", fontsize=6.4, va="top", fontweight="bold")
fig.text(0.996, 0.988, "IoU, worst of 594 at left", color="#31465c", fontsize=6.0, va="top", ha="right")
fig.savefig("paper/icra/fig/mosaic.pdf")
print("wrote paper/icra/fig/mosaic.pdf  (IoU range %s .. %s)" % (sel[0][4], sel[-1][4]))
