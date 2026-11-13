"""Figures for day2.tex. Runs the Day 2 lab solution source, then plots its own variables. Run from slides/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABS = HERE.parents[1] / "labs"
sys.path.insert(0, str(LABS))
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"

src = (LABS / "src" / "day2.py").read_text()
g = {"__name__": "day2"}
exec(compile(src, "day2.py", "exec"), g)
plt.close("all")

# 1. the price of the boundary
fig, ax = plt.subplots(figsize=(4.6, 3.0))
styles = {2: (COOL, ":"), 5: (MUTED, "--"), 10: (ACCENT, "-")}
for n, p in g["PRICES"].items():
    c, ls = styles[n]
    r, p = g["RATIOS"], np.asarray(p)
    m = r <= 10.0
    ax.semilogx(r[m], p[m], color=c, ls=ls, lw=2.2)
    ax.text(r[m][-1] * 1.15, p[m][-1], f"{n} objects", ha="left", va="center", fontsize=10)
ax.set_xlabel(r"shared error / independent error, $s^2/r$"); ax.set_ylabel("price of the boundary (nats)")
ax.set_ylim(bottom=0); ax.set_xlim(right=130)
fig.tight_layout(); fig.savefig(HERE / "d2_price.pdf"); plt.close(fig)

# 2. gate against precision, on two passes
fig, ax = plt.subplots(figsize=(4.6, 3.0))
G = g["GATES"]
for lateral, c, ls in ((1.0, COOL, "-"), (6.0, ACCENT, "-")):
    r = g["GATE_RMSE"][lateral] * 100
    ax.semilogy(G, r, color=c, lw=2.2, ls=ls)
    ax.axhline(g["WEIGHTED_RMSE"][lateral] * 100, color=c, ls="--", lw=1.2)
    i = int(np.argmin(r))
    ax.plot(G[i], r[i], "o", color=c, ms=8, mec="white", mew=1.5, zorder=5)
ax.text(11.8, g["GATE_RMSE"][1.0][-1] * 100 * 0.62, "near pass", ha="right", fontsize=10)
ax.text(1.6, 110, "far: none admitted", ha="left", fontsize=9)
ax.text(11.8, g["WEIGHTED_RMSE"][1.0] * 100 * 0.68, "dashed: precision, no knob", ha="right", fontsize=9, color="0.25"); ax.set_ylim(0.4, 500)
ax.text(9.0, g["GATE_RMSE"][6.0][-1] * 100 * 1.35, "far pass", ha="left", fontsize=10)
ax.set_xlabel("gate $k$ (m)"); ax.set_ylabel("rmse (cm)")
fig.tight_layout(); fig.savefig(HERE / "d2_gate.pdf"); plt.close(fig)

# 3. a bias shared across frames
fig, ax = plt.subplots(figsize=(4.6, 3.0))
n = g["N_FRAMES"][1:]
for a, c, ls, lab in ((1.0, ACCENT, "-", r"$\alpha=1$"), (0.3, COOL, "--", r"$\alpha=0.3$"), (0.05, MUTED, ":", r"$\alpha=0.05$"), (0.0, GOOD, "-", r"$\alpha=0$: a gate")):
    ax.semilogx(n, g["displacement"](n, a) * 100, color=c, ls=ls, lw=2.2, label=lab)
ax.axhline(g["BIAS"] * 100, color="0.3", lw=0.8)
ax.text(1.2, g["BIAS"] * 100 + 0.07, "the whole bias $b$", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.04))
ax.set_ylim(0, 3.5)
ax.set_xlabel("biased frames $n$"); ax.set_ylabel("mean displaced (cm)")
fig.tight_layout(); fig.savefig(HERE / "d2_bias.pdf"); plt.close(fig)
print("day 2 figures written")
