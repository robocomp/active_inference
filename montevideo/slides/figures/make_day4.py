"""Figures for day4.tex. Runs the Day 4 solution source, then plots its own variables. Run from slides/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABS = HERE.parents[1] / "labs"
sys.path.insert(0, str(LABS))
src = (LABS / "src" / "day4.py").read_text()
src = "\n".join(l for l in src.splitlines() if l.strip() not in ("# BEGIN SOLUTION", "# END SOLUTION"))
g = {"__name__": "day4"}
exec(compile(src, "day4.py", "exec"), g)
plt.close("all")
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def mi_two_states(m0, m1, s, x):
    """I(state; o) in nats for two equally likely states with Gaussian outcomes."""
    p0, p1, mix = npdf(x, m0, s), npdf(x, m1, s), 0.5 * (npdf(x, m0, s) + npdf(x, m1, s))
    dx = x[1] - x[0]
    return 0.5 * dx * (np.sum(p0 * np.log((p0 + 1e-300) / (mix + 1e-300))) + np.sum(p1 * np.log((p1 + 1e-300) / (mix + 1e-300))))


# 1. the two terms of G, pictured.
x = np.linspace(-4, 6, 1000)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 1.95), sharey=True)
mis = []
for ax, (m1, name) in zip(axs[:2], ((3.0, "a look that discriminates"), (0.3, "a look that cannot"))):
    ax.plot(x, npdf(x, 0, 0.8), color=COOL, lw=2)
    ax.plot(x, npdf(x, m1, 0.8), color=ACCENT, lw=2, ls="--")
    ax.set_title(name, fontsize=9.5, loc="left")
    mis.append(mi_two_states(0, m1, 0.8, x))
    ax.text(0.98, 0.9, f"$I(s;o)={mis[-1]:.2f}$ nats", transform=ax.transAxes, ha="right", fontsize=9)
axs[0].text(-3.7, 0.2, "if $s_1$", fontsize=9); axs[0].text(4.5, 0.2, "if $s_2$", fontsize=9); axs[0].set_ylim(0, 0.72)
pref, pred = npdf(x, 3.5, 0.9), npdf(x, 1.0, 1.0)
axs[2].fill_between(x, pref, color=GOOD, alpha=0.3); axs[2].plot(x, pred, color="0.25", lw=2, ls=":")
axs[2].text(3.2, 0.47, r"preferred $p(\tilde o)$", fontsize=9); axs[2].text(-3.7, 0.3, "predicted\n" r"$q(\tilde o\mid\pi)$", fontsize=9)
risk = -np.sum(pred * np.log(pref + 1e-300)) * (x[1] - x[0])
axs[2].set_title("pragmatic: far from preferred", fontsize=9.5, loc="left")
for ax in axs:
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel(r"predicted observation $\tilde o$", fontsize=9)
fig.tight_layout(w_pad=0.6); fig.savefig(HERE / "d4_terms.pdf"); plt.close(fig)
print(f"two looks: I = {mis[0]:.3f}, {mis[1]:.3f} nats")

# 2. what sets the price of a view: information relative to what is held, and whether the mask arrives.
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.15))
ratio = np.logspace(-2, 2.3, 200)
axs[0].semilogx(ratio, 0.5 * np.log(1 + ratio), color=COOL, lw=2.5)
axs[0].set_xlabel(r"new information / held precision, $I_j/Y_j$", fontsize=10); axs[0].set_ylabel(r"$\Delta H$ (nats)", fontsize=10)
axs[0].text(5, 0.25, "barely seen:\nworth a lot", fontsize=9); axs[0].text(0.012, 0.75, "well known:\nworth little", fontsize=9)
f = np.linspace(0, 1, 400)
axs[1].plot(f, g["sigmoid"]((f - 0.10) / 0.06) * g["sigmoid"]((0.60 - f) / 0.06), color=ACCENT, lw=2.5)
axs[1].set_xlabel("image fill $f$", fontsize=10); axs[1].set_ylabel(r"$P(\mathrm{detect})$", fontsize=10)
axs[1].text(0.0, 0.72, "too few\npixels", fontsize=9); axs[1].text(0.74, 0.5, "truncated", fontsize=9); axs[1].text(0.2, 0.5, "the mask arrives", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "d4_currency.pdf"); plt.close(fig)

# 3. the hand-off from the lab.
trace = g["trace"]
t = [tr["t"] for tr in trace]
fig, ax = plt.subplots(figsize=(6.4, 2.3))
ax.plot(t, [tr["table"] for tr in trace], color=COOL, lw=2.5)
ax.plot(t, [tr["bottle"] for tr in trace], color=ACCENT, lw=2.5, ls="--")
ax.text(0.2, 4.45, "table's best offer", fontsize=9); ax.text(0.2, 7.0, "bottle's best offer", fontsize=9)
marks = {"aff_table": ("T", COOL), "aff_bottle": ("B", ACCENT), "aff_door": ("D", "0.25")}
for tt, tr in zip(t, trace):
    m, c = marks[tr["holder"]]
    ax.text(tt, -1.0, m, ha="center", va="center", fontsize=9, fontweight="bold", color=c)
ax.text(t[-1] + 0.6, -1.0, "held by", fontsize=8, va="center", color="0.3")
ax.axhline(0, color="0.8", lw=0.8)
ax.set_ylim(-1.5, 7.8); ax.set_xlim(-0.5, t[-1] + 3); ax.set_xlabel("cycle", fontsize=10); ax.set_ylabel(r"$\mathbb{E}[\Delta H]$ (nats)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d4_handoff.pdf"); plt.close(fig)

# 4. convergence in metres: three speeds, one curve.
L = g["L"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
xs = np.linspace(0, 6, 200)
styles = {0.15: ("-", COOL), 0.35: ("--", ACCENT), 0.7: (":", GOOD)}
for v, (a, b) in g["decay_runs"].items():
    ls, c = styles[v]
    axs[0].plot(a[:, 0] / L, a[:, 1] / 0.1, ls, color=c, lw=2, label=f"{v} m/s")
    axs[1].plot(b[:, 0] / L, b[:, 1] * 100, ls, color=c, lw=2)
axs[0].plot(xs, (1 + xs) * np.exp(-xs), color="k", lw=0.8)
axs[0].legend(frameon=False, fontsize=8.5, loc="upper right"); axs[0].set_xlim(0, 6); axs[1].set_xlim(0, 6)
axs[0].set_title("released to one side", fontsize=9.5, loc="left"); axs[1].set_title("released on it, mis-aimed", fontsize=9.5, loc="left")
axs[1].plot([1, 1], [0, 0.2 * L / np.e * 100], color="0.4", lw=0.8)
axs[1].text(1.15, 3.75, r"peak $\phi_0L/e$ at $s=L$", fontsize=9, va="bottom")
axs[0].set_xlabel("distance driven, $s/L$", fontsize=10); axs[1].set_xlabel("distance driven, $s/L$", fontsize=10)
axs[0].set_ylabel("$e/e_0$", fontsize=10); axs[1].set_ylabel("$e$ (cm)", fontsize=10); axs[1].set_ylim(0, 4.6)
fig.tight_layout(); fig.savefig(HERE / "d4_decay.pdf"); plt.close(fig)

# 5. a route deformed, followed by an unchanged law.
fig, ax = plt.subplots(figsize=(6.0, 2.0))
ax.plot(*g["route0"].xy.T, ":", color=MUTED, lw=2)
ax.plot(*g["path1"].T, "-", color=ACCENT, lw=2.5)
ax.add_patch(plt.Circle(g["OBST"], g["OBST_R"], color="0.3"))
ax.text(4.1, 1.72, "route, deformed", fontsize=9); ax.text(0.55, 0.62, "route as planned", fontsize=9, color="0.3")
ax.text(3.12, 0.72, "a box left\non the floor", fontsize=8.5, va="top")
ax.set_aspect("equal"); ax.set_xlim(0.4, 5.6); ax.set_ylim(0.35, 2.1); ax.set_xlabel("x (m)", fontsize=10); ax.set_ylabel("y (m)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d4_route.pdf"); plt.close(fig)

# 6. the grid after exploring, and the trap in one bar chart.
looks, grid, room = g["looks"], g["grid"], g["room"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": (1.25, 1)})
axs[0].imshow(grid.p.T, origin="lower", extent=(0, 6, 0, 4), cmap="gray_r", vmin=0, vmax=1)
axs[0].plot(*np.array([c for c, *_ in looks]).T, "o-", color=ACCENT, ms=4, lw=1.2)
axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].set_title("honest explorer, then stopped", fontsize=9.5, loc="left")
for s in axs[0].spines.values():
    s.set_visible(False)
res = {}
for tm in (False, True):
    _, lk = g["explore"](0, through_matter=tm)
    res[tm] = (sum(l[1] for l in lk), sum(l[2] for l in lk), len(lk))
xb = np.arange(2)
axs[1].bar(xb - 0.19, [res[False][0], res[True][0]], 0.36, color=COOL, label="promised")
axs[1].bar(xb + 0.19, [res[False][1], res[True][1]], 0.36, color="#B8B8B8", hatch="//", edgecolor="white", label="collected")
axs[1].set_xticks(xb, [f"honest\n({res[False][2]} looks)", f"through matter\n({res[True][2]}, never stops)"], fontsize=9)
axs[1].set_ylabel("nats", fontsize=10); axs[1].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2); axs[1].set_ylim(0, 7400)
fig.tight_layout(); fig.savefig(HERE / "d4_explorer.pdf"); plt.close(fig)
print("explorer honest/through-matter (promised, collected, looks):", res)

# 7. the sawtooth's cause: corrections with one sign.
corr, cov = g["corr"], g["cov"]
d = np.cumsum(cov[:, 0])
fig, ax = plt.subplots(figsize=(4.6, 2.2))
ax.plot(d, corr[:, 0] * 1000, ".", ms=3.5, color=COOL)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("distance driven (m)", fontsize=10); ax.set_ylabel("forward correction (mm)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "d4_sawtooth.pdf"); plt.close(fig)

# 8. which motion teaches which parameter: self_calibration.tex, table tab:sc-obs-motion.
rows = ["parked", "straight", "spin, one rate", "spin, two rates", "tour"]
cols = [r"$k_r$", r"$\delta$", r"$k_b$", r"$k_\omega$", r"$b_\omega$", r"$x_h$", r"$y_h$", r"$\varepsilon_{yaw}$", r"$\tau_o$", r"$\tau_g$"]
T = np.array([[1.00, 1.00, 1.00, 1.00, 0.59, 1.00, 1.00, 1.00, 1.00, 1.00],
              [0.23, 0.05, 1.00, 1.00, 0.59, 1.00, 1.00, 0.26, 0.12, 1.00],
              [0.71, 1.00, 0.71, 0.05, 1.00, 0.08, 0.08, 1.00, 1.00, 1.00],
              [0.71, 1.00, 0.71, 0.04, 0.86, 0.08, 0.08, 1.00, 0.09, 0.06],
              [0.34, 0.08, 0.35, 0.05, 0.59, 0.11, 0.11, 0.41, 0.04, 0.02]])
fig, ax = plt.subplots(figsize=(7.0, 2.3))
ax.imshow(T, cmap="Blues_r", vmin=0, vmax=1.15, aspect="auto")
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        ax.text(j, i, f"{T[i, j]:.2f}".replace("1.00", "–"), ha="center", va="center", fontsize=9, color="white" if T[i, j] < 0.45 else "0.2")
ax.set_xticks(range(len(cols)), cols, fontsize=11); ax.set_yticks(range(len(rows)), rows, fontsize=10)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "d4_obs.pdf"); plt.close(fig)

# 9. why a constant spin confuses scale and bias: the lab's conditional and marginal sigmas.
ob = g["observability"]
names = ["spin", "spin, two rates", "spin, then park"]
fig, ax = plt.subplots(figsize=(4.8, 2.4))
xb = np.arange(3)
ax.bar(xb - 0.19, [ob[n][1][1] for n in names], 0.36, color=COOL, label="other unknowns known")
ax.bar(xb + 0.19, [ob[n][0][1] for n in names], 0.36, color="#B8B8B8", hatch="//", edgecolor="white", label="estimated too")
ax.set_xticks(xb, names, fontsize=9.5); ax.set_ylabel(r"$\sigma/\sigma_{prior}$ of $s_\omega$", fontsize=10); ax.legend(frameon=False, fontsize=8.5); ax.set_ylim(0, 0.62)
fig.tight_layout(); fig.savefig(HERE / "d4_schur.pdf"); plt.close(fig)

# 10. drift and the prior that must not move.
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
n = np.logspace(0, 5, 100)
axs[0].loglog(n, [g["steady_ratio"](v) for v in n], color=COOL, lw=2.5)
axs[0].loglog(n, 1.1 * n ** -0.25, color="0.4", lw=1, ls=":")
axs[0].text(20, 0.62, "slope −1/4", fontsize=9.5)
axs[0].set_xlabel(r"information in one drift time, $i\,\sigma_p^2T_d$", fontsize=10); axs[0].set_ylabel(r"steady $\sigma/\sigma_p$", fontsize=10)
hist = g["history"]
k = np.arange(1, len(hist) + 1)
axs[1].plot(k, [h.sd[0] * 1000 for h in hist], "o-", color=ACCENT, lw=2, ms=8, mec="white", mew=1.5)
axs[1].set_ylim(0, 4.6); axs[1].set_xticks(k); axs[1].set_xlabel("solves over the SAME data", fontsize=10); axs[1].set_ylabel(r"reported sd ($\times10^{-3}$)", fontsize=10)
axs[1].text(2.6, 3.3, "confidence from\nno new evidence", fontsize=9.5)
fig.tight_layout(); fig.savefig(HERE / "d4_drift.pdf"); plt.close(fig)
print("day 4 figures written")
