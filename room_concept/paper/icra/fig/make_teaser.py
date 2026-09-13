import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Circle

SRC = "datasets/webots_piso/runs/room2_parked_then_rotated_1357.csv"
BAR = 0.06          # publish gate, metres — the same constant the agent gates on
TRUTH_W, TRUTH_H = 6.000, 4.000

rows = {}
with open(SRC, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.rstrip("\n").split(";")
        if len(p) < 7: continue
        rows[int(p[0])] = p

def parse(p):
    verts = [tuple(map(float, v.split(","))) for v in p[3].split()]
    csig  = [float(x) for x in p[4].split()]
    esig  = [float(x) for x in p[5].split()]
    return verts, csig, esig

WANT = [(12999, "parked, 681 s"), (13478, "after one rotation in place")]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
for ax, (fr, title) in zip(axes, WANT):
    V, C, E = parse(rows[fr])
    ax.add_patch(MPoly(V, closed=True, facecolor="#f6f2ea", edgecolor="none", zorder=0))
    # per-edge band: half-width is that wall's own offset sigma, drawn to scale
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        sd = E[i] if i < len(E) else 0.0
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#0e7490",
                lw=max(0.8, 2.0 * sd * 6.0), alpha=0.30, solid_capstyle="butt", zorder=1)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#1f2328", lw=1.3, zorder=3)
    # per-corner sigma disc, radius = the sigma itself; green under the gate, orange over it
    for (x, y), s in zip(V, C):
        ok = s <= BAR
        ax.add_patch(Circle((x, y), s, facecolor="#177245" if ok else "#c2410c",
                            alpha=0.22 if ok else 0.16, edgecolor="none", zorder=2))
        ax.add_patch(Circle((x, y), 0.05, facecolor="#177245" if ok else "#c2410c",
                            edgecolor="none", zorder=4))
    worst = max(C)
    ax.set_title(f"{title}\nworst corner $\\sigma$ = {worst:.3f} m"
                 + ("  (publishable)" if worst <= BAR else "  (withheld)"),
                 fontsize=7.4, color="#1f2328", pad=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.0, 3.0)

# the truth rectangle, dashed, on both panels — it is the same room throughout
for ax in axes:
    ax.add_patch(MPoly([(-TRUTH_W/2, -TRUTH_H/2), (TRUTH_W/2, -TRUTH_H/2),
                        (TRUTH_W/2, TRUTH_H/2), (-TRUTH_W/2, TRUTH_H/2)], closed=True,
                       facecolor="none", edgecolor="#31465c", lw=0.8, ls=(0, (4, 3)), zorder=5))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.02)
fig.text(0.5, 0.015, "dashed: ground truth 6.000 $\\times$ 4.000 m   ·   discs: per-corner $\\sigma$, "
                     "to scale   ·   bands: per-edge offset $\\sigma_d$",
         ha="center", fontsize=6.3, color="#31465c")
fig.savefig("paper/icra/fig/teaser.pdf")
print("wrote paper/icra/fig/teaser.pdf")
for fr, t in WANT:
    V, C, E = parse(rows[fr]); print(f"  {t}: worst sigma {max(C):.4f}  worst edge {max(E):.4f}")
