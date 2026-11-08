"""Figures for day3.tex. Runs the Day 3 solution source, then plots its own variables. Run from slides/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABS = HERE.parents[1] / "labs"
sys.path.insert(0, str(LABS))
src = (LABS / "src" / "day3.py").read_text()
src = "\n".join(l for l in src.splitlines() if l.strip() not in ("# BEGIN SOLUTION", "# END SOLUTION"))
g = {"__name__": "day3"}
exec(compile(src, "day3.py", "exec"), g)
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
fig.tight_layout(); fig.savefig(HERE / "d3_room.pdf"); plt.close(fig)

# 2. the derived band against wall length
fig, ax = plt.subplots(figsize=(4.4, 2.9))
Ls = np.linspace(0.5, 20, 100)
ax.plot(Ls, g["band_halfwidth"](g["A_scan"], Ls) / g["SIGMA"])
ax.axhline(3, color="0.6", lw=0.8, ls="--"); ax.text(15, 3.05, "3σ", color="0.4")
ax.set_xlabel("wall length L (m)"); ax.set_ylabel("band half-width (σ)")
fig.tight_layout(); fig.savefig(HERE / "d3_band.pdf"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(HERE / "d3_track.pdf"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(HERE / "d3_nees.pdf"); plt.close(fig)

# 5. symmetry: the last scan placed at the true pose and at its half-turn
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.fill(*np.vstack([tr, tr[:1]]).T, fc="0.95", ec="0.2")
for X, col in ((g["X_true"], "crimson"), (g["X_flip"], "royalblue")):
    pts = se2.transform_points(X, g["P_last"])
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color=col, alpha=0.6)
    draw.robot(X, ax, color=col)
ax.set_aspect("equal"); ax.set_title("same scan, two poses: ΔF ≈ 0", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d3_symmetry.pdf"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(HERE / "d3_ceiling.pdf"); plt.close(fig)

COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"

# 7. birth: dF as detections arrive
traces = g["traces"]
fig, ax = plt.subplots(figsize=(4.4, 2.8))
k = np.arange(1, 11)
for name, col, ls in (("sharp views", COOL, "-"), ("distant views", GOOD, "--"), ("reflections", ACCENT, ":")):
    ax.plot(k, traces[name], color=col, ls=ls, lw=2)
ax.axhline(0, color="k", lw=0.8)
ax.text(3.2, -37, "sharp views", fontsize=9); ax.text(7.2, 5, "distant views", fontsize=9); ax.text(3.6, 28, "reflections", fontsize=9)
ax.set_xlabel("detections so far"); ax.set_ylabel(r"$\Delta F$ (nats)")
fig.tight_layout(); fig.savefig(HERE / "d3_birth.pdf"); plt.close(fig)

# 8. existence log-odds along the tour, the evidence rule
tr_ev, _ = g["tour_results"]["evidence"]
fig, ax = plt.subplots(figsize=(4.6, 2.7))
styles = {"plant": (GOOD, "--"), "chair": (MUTED, "-."), "vase": (ACCENT, "-")}
for name, (col, ls) in styles.items():
    ax.plot([s[name] for s in tr_ev], color=col, ls=ls, lw=2)
ax.axhline(g["L_DEAD"], color="k", lw=0.8, ls="--"); ax.text(238, g["L_DEAD"] + 0.15, "removed", ha="right", fontsize=8)
ax.annotate("vase", (60, tr_ev[60]["vase"]), xytext=(84, -2.2), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.annotate("chair: held", (150, tr_ev[150]["chair"]), xytext=(160, 0.4), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.annotate("plant: confirmed, then held", (30, tr_ev[30]["plant"]), xytext=(40, 3.5), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.set_xlabel("frame"); ax.set_ylabel("existence log-odds"); ax.set_ylim(-4.2, 4.4)
fig.tight_layout(); fig.savefig(HERE / "d3_existence.pdf"); plt.close(fig)

# 9. a constant Q forgets on the agent's clock (thesis numbers, not the lab's)
n = np.linspace(0, 1320, 200)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(n, np.sqrt(9.0e-4 + 2.5e-5 * n), color=ACCENT, lw=2)
ax.plot(1320, 0.170, "o", color=COOL, ms=8, mec="white", mew=1.5, zorder=5)
ax.annotate("logged 0.170 m", (1320, 0.170), xytext=(760, 0.125), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.text(330, 0.135, r"$\sqrt{\Sigma_0+Qn}$", fontsize=11)
ax.set_xlabel("frames with no points"); ax.set_ylabel(r"$\sigma_w$ (m)"); ax.set_ylim(0, 0.2)
fig.tight_layout(); fig.savefig(HERE / "d3_constq.pdf"); plt.close(fig)

# 10. constant Q against inferred volatility
mu_q, S_q, mu_v, S_v, tc = g["mu_q"], g["S_q"], g["mu_v"], g["S_v"], g["true_c"]
t_s = np.arange(len(tc)) / 10.0
fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.1), sharex=True, gridspec_kw={"height_ratios": (1, 1.1)})
axes[0].plot(t_s, tc, color="k", lw=0.8, ls=":"); axes[0].plot(t_s, mu_q, color=ACCENT, lw=1.3); axes[0].plot(t_s, mu_v, color=COOL, lw=1.3, ls="--")
axes[0].set_ylabel("position (m)", fontsize=9); axes[0].set_ylim(-0.12, 0.6)
axes[0].text(400, 0.12, "spurious\nmask", fontsize=8, ha="center"); axes[0].text(530, 0.12, "60 s\nunseen", fontsize=8, ha="center"); axes[0].text(610, 0.32, "moved", fontsize=8, ha="left")
axes[1].semilogy(t_s, np.sqrt(S_q), color=ACCENT, lw=1.5); axes[1].semilogy(t_s, np.sqrt(S_v), color=COOL, lw=1.5, ls="--")
axes[1].text(160, 0.0135, "constant $Q$", fontsize=9, color="0.1"); axes[1].text(160, 0.0022, "inferred $Q$", fontsize=9, color="0.1"); axes[1].set_ylim(0.0016, 0.2)
axes[1].set_ylabel("sd (m)", fontsize=9); axes[1].set_xlabel("time (s)", fontsize=9)
fig.tight_layout(h_pad=0.4); fig.savefig(HERE / "d3_volatility.pdf"); plt.close(fig)
print("day 3 figures written; NEES", np.round(scores, 2), "table face nats", round(float(g["evidence"]), 1), "L_edge", round(float(g["L_edge"]), 2))

