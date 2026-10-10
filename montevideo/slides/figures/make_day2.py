"""Figures for day2.tex. Runs the Day 2 solution source, then plots its own variables. Run from slides/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABS = HERE.parents[1] / "labs"
sys.path.insert(0, str(LABS))
src = (LABS / "src" / "day2.py").read_text()
src = "\n".join(l for l in src.splitlines() if l.strip() not in ("# BEGIN SOLUTION", "# END SOLUTION"))
g = {"__name__": "day2"}
exec(compile(src, "day2.py", "exec"), g)
plt.close("all")
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
draw, se2 = g["draw"], g["se2"]

# 1. one scan and the room fitted to it
fig, ax = plt.subplots(figsize=(5.2, 4.0))
Q0, room = g["Q0"], g["room"]
ax.fill(*np.vstack([room, room[:1]]).T, fc="royalblue", alpha=0.12, ec="royalblue", lw=1.5)
ax.plot(Q0[:, 0], Q0[:, 1], ".", ms=2.5, color="crimson")
draw.robot([0, 0, 0], ax)
ax.set_aspect("equal"); ax.set_xlabel("map x (m)"); ax.set_ylabel("map y (m)")
ax.set_title(f"IoU {g['polygon_iou'](se2.transform_points(g['TRUE0'], room), g['truth_room']):.3f}", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d2_room.pdf"); plt.close(fig)

# 2. the derived band against wall length
fig, ax = plt.subplots(figsize=(4.4, 2.9))
Ls = np.linspace(0.5, 20, 100)
ax.plot(Ls, g["band_halfwidth"](g["A_scan"], Ls) / g["SIGMA"])
ax.axhline(3, color="0.6", lw=0.8, ls="--"); ax.text(15, 3.05, "3σ", color="0.4")
ax.set_xlabel("wall length L (m)"); ax.set_ylabel("band half-width (σ)")
fig.tight_layout(); fig.savefig(HERE / "d2_band.pdf"); plt.close(fig)

# 3. the localisation track
fig, ax = plt.subplots(figsize=(5.4, 3.9))
tr = g["true_room"]
ax.fill(*np.vstack([tr, tr[:1]]).T, fc="0.95", ec="0.2")
truths, beliefs = g["truths"], g["beliefs"]
ax.plot(truths[:, 0], truths[:, 1], "k-", lw=1, label="truth")
mus = np.array([b.mu for b in beliefs])
ax.plot(mus[:, 0], mus[:, 1], ".", color="crimson", ms=3, label="belief")
for b in beliefs[::5]:
    draw.covariance(b.mu, b.Sigma * 400, ax)
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("2σ ellipses, ×20", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d2_track.pdf"); plt.close(fig)

# 4. NEES for three maps (recomputed: the notebook only prints it)
scores = []
for poly in (g["true_room"], g["room"], g["bad_room"]):
    scores.append(np.mean([g["nees"](*g["drive"](poly, seed)).mean() for seed in range(4)]))
fig, ax = plt.subplots(figsize=(4.4, 2.9))
ax.bar(["true room", "estimated\n(one scan)", "east wall\n+1.5 cm"], scores, color=["0.6", "royalblue", "crimson"])
ax.axhline(3, color="k", lw=0.8, ls="--"); ax.text(2.45, 3.5, "honest = 3", ha="right", fontsize=9)
ax.set_yscale("log"); ax.set_ylabel("mean NEES")
for i, s in enumerate(scores):
    ax.text(i, s * 1.15, f"{s:.1f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "d2_nees.pdf"); plt.close(fig)

# 5. symmetry: the last scan placed at the true pose and at its half-turn
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.fill(*np.vstack([tr, tr[:1]]).T, fc="0.95", ec="0.2")
for X, col in ((g["X_true"], "crimson"), (g["X_flip"], "royalblue")):
    pts = se2.transform_points(X, g["P_last"])
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color=col, alpha=0.6)
    draw.robot(X, ax, color=col)
ax.set_aspect("equal"); ax.set_title("same scan, two poses: ΔF ≈ 0", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d2_symmetry.pdf"); plt.close(fig)

# 6. the common-mode ceiling
Ns, rows = g["Ns"], g["rows"]
rms = [np.sqrt(np.mean([e ** 2 for e, _, _ in rows[N]])) * 1000 for N in Ns]
ind = [np.mean([n for _, n, _ in rows[N]]) * 1000 for N in Ns]
wood = [np.mean([w for _, _, w in rows[N]]) * 1000 for N in Ns]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
ax.loglog(Ns, rms, "o-", color="k", label="actual error")
ax.loglog(Ns, ind, "--", color="crimson", label="independent points")
ax.loglog(Ns, wood, "-", color="royalblue", label="Woodbury (shared pose)")
ax.set_xlabel("points N"); ax.set_ylabel("error in face offset (mm)"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "d2_ceiling.pdf"); plt.close(fig)
print("day 2 figures written; NEES", np.round(scores, 2))
