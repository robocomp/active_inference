import matplotlib
matplotlib.use("Agg")
# ⚠ TYPE 42 (TrueType), NOT matplotlib's DEFAULT TYPE 3. IEEE/PaperCept forbid Type 3 fonts outright
# and the automated checker rejects on them; every figure in this paper was Type 3 until 2026-09-14.
# It is one line here and a resubmission if it is missed.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly

# ── SOURCE: THE WHOLE CORPUS, NOT A SUBSET ──────────────────────────────────────────────────────
# This figure used to be built from `mosaic_cells_120.html`, a 120-room render, while its caption
# claimed the panels were sampled across the 594 and that "the population's worst room is at the left
# of the top row". That was FALSE: the worst of the 120 scored IoU 0.821, and three rooms in the full
# corpus are worse than that (0.654, 0.754, 0.756). A figure whose selling point is that it does not
# flatter the method must not quietly exclude the three rooms that flatter it least.
# `mosaic_cells_594.tsv` is extracted from the 594-room run and carries every room that produced a
# closed polygon, so the claim is now true by construction.
SRC = "datasets/matterport_layout/mosaic_cells_594.tsv"

cells = []
for line in open(SRC, encoding="utf-8"):
    if line.startswith("#"):
        continue
    room, name, ncor, iou, tru, est = line.rstrip("\n").split("\t")
    cells.append((int(ncor), float(iou),
                  [tuple(map(float, p.split(","))) for p in tru.split()],
                  [tuple(map(float, p.split(","))) for p in est.split()]))
print(f"parsed {len(cells)} rooms from the full corpus")

# Sorted by IoU and sampled evenly, so the panel spans the population instead of flattering it:
# the worst room in the corpus is the first cell shown.
cells.sort(key=lambda c: c[1])
N_COLS, N_ROWS = 8, 4
n = N_COLS * N_ROWS
sel = [cells[round(i * (len(cells) - 1) / (n - 1))] for i in range(n)]

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(7.16, 3.9))
for ax, (ncor, iou, T, E) in zip(axes.ravel(), sel):
    ax.add_patch(MPoly(T, closed=True, facecolor="#dfe7ef", edgecolor="#31465c", lw=0.9, zorder=1))
    ax.add_patch(MPoly(E, closed=True, facecolor="none", edgecolor="#c2410c", lw=0.9, zorder=2))
    xs = [p[0] for p in T + E]; ys = [p[1] for p in T + E]
    mx, my = (max(xs) - min(xs)) * 0.06 + 0.1, (max(ys) - min(ys)) * 0.06 + 0.1
    ax.set_xlim(min(xs) - mx, max(xs) + mx); ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, -0.02, f"{iou:.3f}", transform=ax.transAxes, ha="center", va="top",
            fontsize=5.2, color="#31465c")

fig.text(0.012, 0.985, "truth", ha="left", va="top", fontsize=6.0, color="#31465c", weight="bold")
fig.text(0.075, 0.985, "estimate", ha="left", va="top", fontsize=6.0, color="#c2410c", weight="bold")
fig.text(0.988, 0.985, "IoU, worst of 594 at left", ha="right", va="top", fontsize=6.0,
         color="#31465c")
fig.subplots_adjust(left=0.004, right=0.996, top=0.955, bottom=0.035, wspace=0.05, hspace=0.30)
fig.savefig("paper/icra/fig/mosaic.pdf")
print("wrote paper/icra/fig/mosaic.pdf")
sn = [c[0] for c in sel]
print(f"  drawn panels: IoU {sel[0][1]:.3f} .. {sel[-1][1]:.3f} | truth corners {min(sn)} .. {max(sn)}")
