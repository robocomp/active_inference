"""Figures for day1.tex, computed with the same toyworld code the lab uses. Run from slides/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs"))
from toyworld import Gaussian, Lidar, draw, info_gain, laplace, se2, two_room_apartment  # noqa: E402
from toyworld.geometry import point_segment_distance  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(0)
world = two_room_apartment()
lidar = Lidar(n_beams=90, fov=np.deg2rad(90), sigma=0.02)
oracle = Lidar(n_beams=90, fov=np.deg2rad(90), sigma=0.0)
NORTH = np.array([[[0.0, 4.0], [6.0, 4.0]]])
pose0 = np.array([4.5, 1.0, np.deg2rad(110)])


def wall_points(scan):
    pts = se2.transform_points(scan.pose, scan.points())
    bearings = scan.pose[2] + scan.angles[scan.hit]
    m = (scan.labels[scan.hit] == "wall") & (point_segment_distance(pts, NORTH)[:, 0] < 0.15)
    return pts[m], bearings[m]


def res(p):
    return lambda t: np.cos(t[0]) * p[:, 0] + np.sin(t[0]) * p[:, 1] - t[1]


def jac(p):
    return lambda t: np.stack([-np.sin(t[0]) * p[:, 0] + np.cos(t[0]) * p[:, 1], -np.ones(len(p))], axis=1)


def isig(b, alpha=np.pi / 2):
    return 0.02 * np.abs(np.cos(b) * np.cos(alpha) + np.sin(b) * np.sin(alpha))


scan0 = lidar.scan(world, pose0, rng)
p0, b0 = wall_points(scan0)
wall0 = laplace(res(p0), [np.pi / 2, 4.0], isig(b0), jacobian=jac(p0)).belief

# 1. the scan and the belief fan
fig, ax = plt.subplots(figsize=(6.0, 4.2))
draw.world(world, ax)
draw.scan(scan0, ax=ax, rays=True)
draw.robot(pose0, ax)
for a, d in Gaussian(wall0.mu, wall0.Sigma * 400).sample(25, rng):
    x = np.array([-0.3, 6.3])
    ax.plot(x, (d - np.cos(a) * x) / np.sin(a), color="royalblue", lw=0.6, alpha=0.5)
ax.set_xlim(-0.4, 6.6); ax.set_ylim(-0.3, 4.5); ax.set_aspect("equal")
ax.set_title("one scan, one wall belief (samples, sd ×20)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "d1_wall_belief.pdf"); plt.close(fig)

# 2. honesty test: predicted vs empirical sd, uniform vs incidence sigma
rows = {}
for name, sf in (("uniform σ", lambda b: 0.02), ("incidence σ", isig)):
    fits, sds = [], []
    for _ in range(400):
        p, b = wall_points(lidar.scan(world, pose0, rng))
        r = laplace(res(p), [np.pi / 2, 4.0], sf(b), jacobian=jac(p))
        fits.append(r.belief.mu); sds.append(r.belief.sd)
    rows[name] = np.mean(sds, axis=0) / np.std(fits, axis=0)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
x = np.arange(2)
for i, (name, ratio) in enumerate(rows.items()):
    ax.bar(x + 0.38 * i, ratio, 0.36, label=name)
ax.axhline(1, color="k", lw=0.8)
ax.set_xticks(x + 0.19, [r"$\alpha$", r"$d$"]); ax.set_ylim(0.9, 1.2)
ax.set_ylabel("predicted sd / empirical sd"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "d1_honesty.pdf"); plt.close(fig)

# 3. Occam: F_free - F_Manhattan vs prior width
manh = laplace(lambda t: res(p0)(np.array([np.pi / 2, t[0]])), [3.0], isig(b0), prior=Gaussian([3.0], [[4.0]]))
widths = np.deg2rad(np.logspace(-1, np.log10(30), 30))
dF = [laplace(res(p0), [np.pi / 2, 3.0], isig(b0), jacobian=jac(p0),
              prior=Gaussian([np.pi / 2, 3.0], np.diag([s ** 2, 4.0]))).free_energy - manh.free_energy for s in widths]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogx(np.degrees(widths), dF, "o-", ms=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("prior sd on the wall angle (deg)"); ax.set_ylabel(r"$F_{free}-F_{Manh}$ (nats)")
fig.tight_layout(); fig.savefig(OUT / "d1_occam.pdf"); plt.close(fig)

# 4. candidate views with their gains
cands = {"stay": [4.5, 1.0, np.deg2rad(110)], "under wall": [3.0, 3.3, np.pi / 2],
         "far west": [0.8, 1.0, np.deg2rad(60)], "behind table": [2.2, 1.4, np.pi / 2],
         "east corner": [5.5, 0.5, np.deg2rad(135)]}
fig, ax = plt.subplots(figsize=(6.0, 4.2))
draw.world(world, ax)
for name, v in cands.items():
    p, b = wall_points(oracle.scan(world, np.asarray(v), rng))
    g = info_gain(wall0.precision, jac(p)(wall0.mu), np.diag(isig(b, wall0.mu[0]) ** 2))
    draw.robot(v, ax, color="crimson" if name == "stay" else "0.35")
    ax.annotate(f"{name}\n{g:.2f} nats", np.asarray(v[:2]), fontsize=8, xytext=(6, -16), textcoords="offset points")
ax.plot(p0[:, 0], p0[:, 1], ".", color="crimson", ms=3)
ax.set_xlim(-0.4, 6.6); ax.set_ylim(-0.3, 4.4); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "d1_candidates.pdf"); plt.close(fig)

# 5. diminishing returns
p, b = wall_points(oracle.scan(world, pose0, rng))
J1, R1 = jac(p)(wall0.mu), np.diag(isig(b) ** 2)
L1 = J1.T @ np.linalg.solve(R1, J1)
ks = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(ks, [info_gain(k * L1, J1, R1) for k in ks], "o", label="info_gain")
ax.plot(ks, np.log((ks + 1) / ks), "-", label=r"$\ln\frac{k+1}{k}$")
ax.set_xlabel("look number k"); ax.set_ylabel(r"$\Delta H$ (nats)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "d1_diminishing.pdf"); plt.close(fig)
# 6. probability basics: a 2-D Gaussian joint p(s, o), its marginals, and one conditional slice.
#    s = distance to the wall, prior N(2, 0.5^2); o = range reading, o | s ~ N(s, 0.2^2).
mu_s, sd_s, sd_o, o_star = 2.0, 0.5, 0.2, 2.6
mu = np.array([mu_s, mu_s])
S = np.array([[sd_s ** 2, sd_s ** 2], [sd_s ** 2, sd_s ** 2 + sd_o ** 2]])
g = np.linspace(0.4, 3.6, 300)
X, Y = np.meshgrid(g, g)
D = np.stack([X - mu[0], Y - mu[1]], -1)
Z = np.exp(-0.5 * np.einsum("...i,ij,...j", D, np.linalg.inv(S), D)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
npdf = lambda x, m, v: np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
post_v = S[0, 0] - S[0, 1] ** 2 / S[1, 1]
post_m = mu[0] + S[0, 1] / S[1, 1] * (o_star - mu[1])
fig = plt.figure(figsize=(6.4, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax); axr = fig.add_subplot(gs[1, 1], sharey=ax)
ax.contour(X, Y, Z, levels=6, colors="0.35", linewidths=0.8)
ax.axhline(o_star, color="crimson", lw=1.2, ls="--")
ax.text(0.5, o_star + 0.06, r"observe $o=2.6$", color="crimson", fontsize=9)
ax.set_xlabel(r"$s$: distance to the wall (m)"); ax.set_ylabel(r"$o$: range reading (m)")
axt.fill_between(g, npdf(g, mu_s, sd_s ** 2), color="#2E6DB4", alpha=0.25, label=r"marginal $p(s)$ (prior)")
axt.plot(g, npdf(g, post_m, post_v), color="crimson", lw=1.5, label=r"conditional $p(s\mid o=2.6)$")
axt.legend(frameon=False, fontsize=8, loc="upper left"); axt.axis("off")
axr.fill_betweenx(g, npdf(g, mu_s, S[1, 1]), color="#2E8B57", alpha=0.25)
axr.text(0.05, 0.55, r"marginal" "\n" r"$p(o)$", fontsize=8, transform=axr.transAxes); axr.axis("off")
fig.savefig(OUT / "d1_gaussian_ops.pdf", bbox_inches="tight"); plt.close(fig)
print(f"gaussian ops: posterior s | o=2.6 = {post_m:.3f} +- {np.sqrt(post_v):.3f}")

# 6b. marginalisation only: the same joint and its two shadows, no observation, no conditional.
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax); axr = fig.add_subplot(gs[1, 1], sharey=ax)
ax.contourf(X, Y, Z, levels=7, cmap="Blues", alpha=0.85)
ax.set_xlabel(r"$s$: distance to the wall (m)"); ax.set_ylabel(r"$o$: range reading (m)")
ax.text(0.6, 3.25, r"joint $p(s,o)$", fontsize=14, color="#2E6DB4")
ax.annotate("", xy=(3.55, 2.0), xytext=(2.75, 2.0), arrowprops=dict(arrowstyle="-|>", color="#2E8B57", lw=2))
ax.text(2.78, 2.12, r"add along $s$", fontsize=13, color="#2E8B57")
axt.fill_between(g, npdf(g, mu_s, sd_s ** 2), color="#2E6DB4", alpha=0.3)
axt.text(0.5, 0.45, r"$p(s)$", fontsize=14, color="#2E6DB4"); axt.axis("off")
axr.fill_betweenx(g, npdf(g, mu_s, S[1, 1]), color="#2E8B57", alpha=0.3)
axr.text(0.0, 3.5, r"$p(o)=\int p(s,o)\,ds$", fontsize=13, color="#2E8B57"); axr.axis("off")
fig.savefig(OUT / "d1_gaussian_marg.pdf", bbox_inches="tight"); plt.close(fig)

# 6c. conditioning only: the slice of the joint at o = 2.6 has area p(o = 2.6) (that is marginalisation);
#     dividing by that area raises every height by the same factor and makes the area 1.
p_o = npdf(o_star, mu_s, S[1, 1])
Dsl = np.stack([g - mu[0], np.full_like(g, o_star) - mu[1]], -1)
slice_ = np.exp(-0.5 * np.einsum("...i,ij,...j", Dsl, np.linalg.inv(S), Dsl)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
cond = slice_ / p_o
pk = g[np.argmax(cond)]
fig = plt.figure(figsize=(7.2, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(2.2, 4), wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax); axr = fig.add_subplot(gs[1, 1], sharey=ax)
ax.contourf(X, Y, Z, levels=7, cmap="Blues", alpha=0.85)
ax.axhline(o_star, color="crimson", lw=1.6, ls="--")
ax.text(0.5, o_star + 0.07, r"observe $o=2.6$", color="crimson", fontsize=14)
ax.set_xlabel(r"$s$: distance to the wall (m)"); ax.set_ylabel(r"$o$: range reading (m)")
axt.fill_between(g, cond, color="crimson", alpha=0.25); axt.plot(g, cond, color="crimson", lw=1.5)
axt.fill_between(g, slice_, color="0.55", alpha=0.6)
axt.annotate("", xy=(pk, cond.max() * 0.97), xytext=(pk, slice_.max() * 1.03),
             arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.6))
axt.text(pk + 0.07, 0.5 * (slice_.max() + cond.max()), rf"$\div\,{p_o:.2f}$", color="crimson", fontsize=15, va="center")
axt.text(0.45, cond.max() * 0.12, rf"grey: the slice, area ${p_o:.2f}$", fontsize=14, color="0.3")
axt.text(0.45, cond.max() * 0.62, r"red: $p(s\mid o{=}2.6)$, area $1$", fontsize=14, color="crimson")
axt.set_ylim(0, cond.max() * 1.08); axt.axis("off")
axr.fill_betweenx(g, npdf(g, mu_s, S[1, 1]), color="#2E8B57", alpha=0.3)
axr.plot([0, p_o], [o_star, o_star], color="crimson", lw=1.6)
axr.text(0.12, 3.3, r"$p(o)$", fontsize=14, color="#2E8B57")
axr.text(0.02, o_star + 0.12, f"{p_o:.2f} = area\nof the slice", fontsize=13, color="crimson", va="bottom"); axr.axis("off")
fig.savefig(OUT / "d1_gaussian_cond.pdf", bbox_inches="tight"); plt.close(fig)
print(f"conditioning: p(o=2.6) = {p_o:.3f}")

# 6d. factorisation, continuous: the same joint built two ways, as 2-D images over (s, o).
#     Row 1: p(s) (vertical stripes) x p(o|s) (the sensor band o = s +- 0.2) = p(s,o).
#     Row 2: p(o) (horizontal stripes) x p(s|o) (a band of another tilt)   = the same p(s,o).
k_so = S[0, 1] / S[1, 1]; v_so = S[0, 0] - S[0, 1] ** 2 / S[1, 1]
P_s = npdf(X, mu_s, sd_s ** 2)
P_o_s = npdf(Y, X, sd_o ** 2)
P_o = npdf(Y, mu_s, S[1, 1])
P_s_o = npdf(X, mu_s + k_so * (Y - mu_s), v_so)
rows = [(P_s, r"$p(s)$", "Blues", P_o_s, r"$p(o\mid s)$"), (P_o, r"$p(o)$", "Greens", P_s_o, r"$p(s\mid o)$")]
fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4), sharex=True, sharey=True)
for r, (A, ta, ca, B, tb) in enumerate(rows):
    for c, (M, t, cmap) in enumerate([(A, ta, ca), (B, tb, "Reds"), (A * B, r"$p(s,o)$", "Blues")]):
        a_ = axes[r, c]
        a_.imshow(M, origin="lower", extent=(g[0], g[-1], g[0], g[-1]), cmap=cmap, aspect="equal")
        a_.set_title(t, fontsize=16)
        a_.set_xticks([1, 2, 3]); a_.set_yticks([1, 2, 3]); a_.tick_params(labelsize=12)
    fig.text(0.355, 0.73 - 0.455 * r, r"$\times$", fontsize=24, ha="center", va="center")
    fig.text(0.655, 0.73 - 0.455 * r, r"$=$", fontsize=22, ha="center", va="center")
for a_ in axes[1]: a_.set_xlabel(r"$s$ (m)", fontsize=13)
for a_ in axes[:, 0]: a_.set_ylabel(r"$o$ (m)", fontsize=13)
axes[0, 1].annotate("", xy=(1.0, 3.4), xytext=(1.0, 0.55), arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.5))
axes[0, 1].text(1.12, 2.6, "each\ncolumn\nadds to 1", fontsize=11, color="0.2")
axes[1, 1].annotate("", xy=(3.5, 1.0), xytext=(0.5, 1.0), arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.5))
axes[1, 1].text(1.9, 0.6, "each row adds to 1", fontsize=11, color="0.2")
fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig.savefig(OUT / "d1_gaussian_fact.pdf", bbox_inches="tight"); plt.close(fig)
print("max |row1 - row2| =", float(np.abs(P_s * P_o_s - P_o * P_s_o).max()))

COOL, ACCENT, GOOD, MUTED = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"

# 7. Bayes as the three operations, same numbers as block 6. One shared y-scale, so the last step
#    visibly IS a division by the evidence.
s = np.linspace(0.8, 3.4, 600)
prior, like = npdf(s, mu_s, sd_s ** 2), npdf(o_star, s, sd_o ** 2)
product = prior * like
evidence = product.sum() * (s[1] - s[0])
assert np.isclose(evidence, npdf(o_star, mu_s, S[1, 1]), atol=1e-3)
posterior = product / evidence
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharex=True, sharey=True)
titles = [r"1. factorise: $p(o\mid s)\,p(s)$", r"2. marginalise: $p(o)$ = area", r"3. condition: $p(s\mid o)$"]
for a, t in zip(axes, titles):
    a.set_title(t, fontsize=15, loc="left"); a.set_xlabel(r"$s$ (m)", fontsize=13); a.set_yticks([])
    a.spines["left"].set_visible(False); a.tick_params(labelsize=12); a.set_xticks([1, 2, 3])
axes[0].fill_between(s, prior, color=COOL, alpha=0.3); axes[0].plot(s, prior, color=COOL, lw=1.5)
axes[0].plot(s, like, color=GOOD, lw=2.5, ls="--")
axes[0].text(0.85, 0.9, r"$p(s)$", fontsize=14, color=COOL); axes[0].text(2.75, 1.7, r"$p(o\mid s)$", fontsize=14, color=GOOD)
axes[1].fill_between(s, product, color=MUTED, alpha=0.35); axes[1].plot(s, product, color=MUTED, lw=1.5)
axes[1].text(0.85, 0.9, f"area\n$= {evidence:.2f}$", fontsize=14, color="0.3")
axes[2].fill_between(s, posterior, color=ACCENT, alpha=0.35); axes[2].plot(s, posterior, color=ACCENT, lw=2)
axes[2].text(0.85, 1.25, f"$\\div\\,{evidence:.2f}$", fontsize=16, color=ACCENT)
axes[2].text(0.85, 0.75, f"${post_m:.2f}\\pm{np.sqrt(post_v):.2f}$ m", fontsize=13, color=ACCENT)
fig.tight_layout(w_pad=0.6); fig.savefig(OUT / "d1_bayes_ops.pdf"); plt.close(fig)
print(f"bayes ops: evidence p(o=2.6) = {evidence:.3f}")

# 8. complexity against inaccuracy along the belief's mean (the Kalman example, r = 0.2^2).
m, p_, y, r = mu_s, sd_s ** 2, o_star, sd_o ** 2
Sig = 1 / (1 / p_ + 1 / r)
mus = np.linspace(1.9, 2.85, 400)
complexity = 0.5 * ((Sig + (mus - m) ** 2) / p_ - 1 + np.log(p_ / Sig))
inaccuracy = ((y - mus) ** 2 + Sig) / (2 * r)                     # -E_q ln p(o|s), constant dropped
F = complexity + inaccuracy
mu_star = (m / p_ + y / r) / (1 / p_ + 1 / r)
assert abs(mus[np.argmin(F)] - mu_star) < 0.01
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.plot(mus, complexity, color=COOL, lw=2, ls="--"); ax.plot(mus, inaccuracy, color=GOOD, lw=2, ls=":")
ax.plot(mus, F, color=ACCENT, lw=2.5)
ax.axvline(mu_star, color=MUTED, lw=0.8)
ax.plot(mu_star, F.min(), "o", color=ACCENT, ms=8, mec="white", mew=1.5, zorder=5)
ax.text(mu_star + 0.02, 5.4, f"belief {mu_star:.2f}", ha="left", fontsize=13, color="0.25")
ax.text(1.94, 0.12, "complexity", fontsize=13, color=COOL)
ax.text(2.64, 0.12, "inaccuracy", fontsize=13, color=GOOD)
ax.text(2.24, 3.6, r"$F$", fontsize=16, color=ACCENT)
ax.set_xlim(1.9, 2.85); ax.set_ylim(0, 6)
ax.set_xlabel(r"belief mean $\mu$ (m): prior 2.0, reading 2.6", fontsize=13); ax.set_ylabel("nats", fontsize=13); ax.tick_params(labelsize=11)
fig.tight_layout(); fig.savefig(OUT / "d1_complexity.pdf"); plt.close(fig)

# 9. Laplace: the bottom of the bowl is the mean, its curvature the precision.
x = np.linspace(-1.6, 1.6, 500)
fig, axes = plt.subplots(2, 2, figsize=(9.0, 3.9), sharex=True, gridspec_kw={"height_ratios": (1.15, 1)})
for j, (k, name) in enumerate(((14.0, "sharp bowl: sure belief"), (2.0, "flat bowl: unsure belief"))):
    U = k * (x ** 2 / 2 - 0.12 * x ** 3 + 0.05 * x ** 4)
    exact = np.exp(-U); exact /= exact.sum() * (x[1] - x[0])
    top, bot = axes[0, j], axes[1, j]
    top.plot(x, U, color="0.3", lw=2.5, label=r"$-\ln p(o,s)$")
    top.plot(x, k * x ** 2 / 2, color=ACCENT, lw=2, ls="--", label="parabola at the bottom")
    top.plot(0, 0, "o", color=ACCENT, ms=8, zorder=5)
    top.set_ylim(-0.9, 4); top.set_title(f"{name}, curvature $\\mathcal{{H}}={k:.0f}$", fontsize=13); top.set_yticks([])
    top.text(0, -0.3, r"$\mu$", fontsize=14, color=ACCENT, ha="center", va="top")
    bot.fill_between(x, npdf(x, 0, 1 / k), color=ACCENT, alpha=0.35, label=r"belief $q(s)$")
    bot.plot(x, exact, color="0.25", lw=1.2, label=r"exact $p(s\mid o)$")
    bot.set_ylim(0, 1.7); bot.set_yticks([]); bot.set_xlabel(r"$s-\mu$", fontsize=12)
    bot.text(0.98, 0.7, f"$\\Sigma = 1/{k:.0f} = {1 / k:.2f}$", transform=bot.transAxes, ha="right", fontsize=13, color=ACCENT)
axes[0, 0].legend(loc="upper left", fontsize=10, frameon=False); axes[1, 0].legend(loc="upper left", fontsize=10, frameon=False)
fig.tight_layout(h_pad=0.4); fig.savefig(OUT / "d1_laplace.pdf"); plt.close(fig)

# 10. the continuous joint as a surface: the same p(s, o) as block 6, drawn so that VOLUME is
#     probability. One patch of (s, o) is lifted out; its volume is printed and shown on the slide.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
gc = np.linspace(0.4, 3.6, 70)
Xc, Yc = np.meshgrid(gc, gc)
Dc = np.stack([Xc - mu[0], Yc - mu[1]], -1)
dens = lambda D_: np.exp(-0.5 * np.einsum("...i,ij,...j", D_, np.linalg.inv(S), D_)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
Zc = dens(Dc)
s0, s1, o0, o1 = 2.0, 2.5, 2.0, 2.5                      # the patch: a square on the ridge (0.23; avoids numbers used elsewhere)
gp_s, gp_o = np.linspace(s0, s1, 40), np.linspace(o0, o1, 40)
Xp, Yp = np.meshgrid(gp_s, gp_o)
Zp = dens(np.stack([Xp - mu[0], Yp - mu[1]], -1))
p_patch = Zp.mean() * (s1 - s0) * (o1 - o0)
fig = plt.figure(figsize=(6.0, 4.6))
ax = fig.add_subplot(projection="3d")
# ONE surface, the patch coloured in place: matplotlib does not depth-sort two surfaces.
cen_s, cen_o = 0.5 * (Xc[:-1, :-1] + Xc[1:, 1:]), 0.5 * (Yc[:-1, :-1] + Yc[1:, 1:])
in_patch = (cen_s >= s0) & (cen_s <= s1) & (cen_o >= o0) & (cen_o <= o1)
fc = plt.cm.Blues(0.25 + 0.7 * Zc[:-1, :-1] / Zc.max())
fc[in_patch] = matplotlib.colors.to_rgba("#E08E0B")
ax.plot_surface(Xc, Yc, Zc, facecolors=fc, linewidth=0, antialiased=False, rstride=1, cstride=1, shade=False)
for xa, ya, xb, yb in [(s0, o0, s1, o0), (s1, o0, s1, o1), (s1, o1, s0, o1), (s0, o1, s0, o0)]:
    t = np.linspace(0, 1, 30)
    ax.plot(xa + t * (xb - xa), ya + t * (yb - ya), np.zeros_like(t), color="#E08E0B", lw=1.4)
for xe, ye in [(s0, o0), (s1, o0), (s1, o1)]:
    ax.plot([xe, xe], [ye, ye], [0, dens(np.array([xe - mu[0], ye - mu[1]]))], color="#E08E0B", lw=1.0, ls=":")
ax.set_xlabel(r"$s$: distance to the wall (m)", labelpad=6)
ax.set_ylabel(r"$o$: range reading (m)", labelpad=6)
ax.set_zlabel(r"$p(s,o)$", labelpad=2)
ax.contour(Xc, Yc, Zc, levels=4, zdir="z", offset=0, colors=COOL, linewidths=0.8)
ax.set_zticks([]); ax.view_init(elev=30, azim=-40)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
fig.savefig(OUT / "d1_joint_continuous.pdf", bbox_inches="tight"); plt.close(fig)
print(f"continuous joint: P(s in [{s0},{s1}], o in [{o0},{o1}]) = {p_patch:.3f}")

# 11. approximate the posterior: a skewed p(s|o) and three Gaussian beliefs q, each closer in KL.
xs = np.linspace(0.0, 6.0, 3000); dx = xs[1] - xs[0]
post = xs ** 2 * np.exp(-xs / 0.7); post /= post.sum() * dx           # a gamma(3, 0.7): not Gaussian


def kl_qp(q, p):
    m = q > 1e-12
    return float((q[m] * np.log(q[m] / np.maximum(p[m], 1e-300))).sum() * dx)


def gpdf(x, m, sd):
    return np.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


fig, ax = plt.subplots(figsize=(8.0, 2.8))
ax.fill_between(xs, post, color=ACCENT, alpha=0.25); ax.plot(xs, post, color=ACCENT, lw=2.5)
ax.text(0.05, 0.5, r"$p(s\mid o)$: unknown to the robot", color=ACCENT, fontsize=13)
for (m_, sd_), al in zip([(4.5, 0.9), (3.3, 0.95), (2.1, 1.0)], [0.35, 0.65, 1.0]):
    q = gpdf(xs, m_, sd_)
    ax.plot(xs, q, color=COOL, lw=2, ls="--", alpha=al)
    ax.text(m_, q.max() + 0.015, f"KL = {kl_qp(q, post):.2f}", color=COOL, alpha=al, fontsize=12, ha="center")
ax.annotate("", xy=(2.5, 0.56), xytext=(4.3, 0.56), arrowprops=dict(arrowstyle="-|>", color=COOL, lw=2))
ax.text(3.4, 0.59, r"move the belief $q(s)$", color=COOL, fontsize=13, ha="center")
ax.set_xlim(0, 6); ax.set_ylim(0, 0.68); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(r"$s$", fontsize=13); ax.tick_params(labelsize=11)
fig.tight_layout(); fig.savefig(OUT / "d1_approx.pdf"); plt.close(fig)

# 12. which KL is a choice: a two-mode posterior, fitted by KL[q||p] (ours) and by KL[p||q].
two = 0.55 * gpdf(xs, 1.6, 0.35) + 0.45 * gpdf(xs, 4.4, 0.4)
mf = (xs * two).sum() * dx; sf = np.sqrt(((xs - mf) ** 2 * two).sum() * dx)   # KL[p||q]: match moments
best = min(((kl_qp(gpdf(xs, m_, s_), two), m_, s_) for m_ in np.linspace(1, 5, 161) for s_ in np.linspace(0.2, 2, 91)))
_, mr, sr = best
fig, ax = plt.subplots(figsize=(8.0, 2.8))
ax.fill_between(xs, two, color=ACCENT, alpha=0.25); ax.plot(xs, two, color=ACCENT, lw=2.5)
ax.plot(xs, gpdf(xs, mr, sr), color=COOL, lw=2.5)
ax.plot(xs, gpdf(xs, mf, sf), color=MUTED, lw=2.5, ls="--")
ax.text(2.05, 1.05, r"$\mathrm{KL}[q\,\|\,p]$ (ours): one mode, too sure", color=COOL, fontsize=13)
ax.text(2.55, 0.45, r"$\mathrm{KL}[p\,\|\,q]$: covers both," + "\ntoo broad", color=MUTED, fontsize=13)
ax.text(4.75, 0.55, r"$p(s\mid o)$", color=ACCENT, fontsize=13)
ax.set_xlim(0, 6); ax.set_ylim(0, 1.25); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(r"$s$", fontsize=13); ax.tick_params(labelsize=11)
fig.tight_layout(); fig.savefig(OUT / "d1_kl_choice.pdf"); plt.close(fig)
print(f"kl choice: reverse fit {mr:.2f} +- {sr:.2f}, forward fit {mf:.2f} +- {sf:.2f}")

# 13. what KL measures: q against p at three distances; below, ln q and ln p, whose gap (weighted by q) is the KL.
xk = np.linspace(-4.0, 6.0, 4000); dk = xk[1] - xk[0]
fig, axs = plt.subplots(2, 3, figsize=(10.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [0.85, 1.15]})
for c, (d, sq) in enumerate([(0.0, 1.0), (1.0, 1.0), (0.0, 0.4)]):
    pk, qk = gpdf(xk, 0.0, 1.0), gpdf(xk, d, sq)
    kl = float((qk * np.log(qk / pk)).sum() * dk)
    a = axs[0, c]
    a.fill_between(xk, pk, color=ACCENT, alpha=0.2); a.plot(xk, pk, color=ACCENT, lw=2.2, label=r"$p$")
    a.plot(xk, qk, color=COOL, lw=2.2, ls="--", label=r"$q$")
    a.set_title(["same belief", "one width apart", "same mean, narrower"][c], fontsize=12)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_ylim(0, 1.1 * max(pk.max(), qk.max())); a.set_xticks([])
    b = axs[1, c]
    lp, lq = np.log(pk), np.log(qk)
    for xs_ in np.arange(d - 2.4 * sq, d + 2.4 * sq + 1e-9, 0.2 * sq):
        w = gpdf(xs_, d, sq) / gpdf(d, d, sq)
        y0, y1 = np.log(gpdf(xs_, 0, 1)), np.log(gpdf(xs_, d, sq))
        b.plot([xs_, xs_], [y0, y1], color="#E08E0B" if y1 >= y0 else MUTED, lw=3, alpha=0.15 + 0.85 * w, solid_capstyle="butt")
    b.plot(xk, lp, color=ACCENT, lw=2.2, label=r"$\ln p$"); b.plot(xk, lq, color=COOL, lw=2.2, ls="--", label=r"$\ln q$")
    b.text(0.02, 0.97, f"KL = {kl:.1f}", transform=b.transAxes, ha="left", va="top", fontsize=14, color="#B06E00", weight="bold")
    b.set_yticks([]); b.spines["left"].set_visible(False); b.set_ylim(-6.5, 1.2)
    b.set_xlim(-3.5, 4.0); b.set_xticks([])
axs[0, 0].legend(loc="upper left", fontsize=12, frameon=False)
axs[1, 0].legend(loc="upper right", fontsize=12, frameon=False)
fig.tight_layout(h_pad=0.3); fig.savefig(OUT / "d1_kl_explain.pdf"); plt.close(fig)

# 14. Active inference on the wall: Laplace F, perception and action both descend it.
rg = np.random.default_rng(3)
m_pref, P_p, P_o, k_mu, k_a, dt_ = 1.0, 4.0, 25.0, 0.04, 2.0, 0.1
s_, mu_, a_ = 2.6, 2.6, 0.0
hist = []
for _ in range(50):
    o_ = s_ + rg.normal(0, 0.03)
    e_o, e_p = o_ - mu_, mu_ - m_pref
    mu_ += k_mu * (P_o * e_o - P_p * e_p)
    a_ += -k_a * P_o * e_o * dt_
    s_ += a_ * dt_
    hist.append((s_, mu_, o_, a_))
hist = np.array(hist); tt = np.arange(len(hist)) * dt_
fig, (ax, axb) = plt.subplots(2, 1, figsize=(4.6, 3.5), sharex=True, gridspec_kw={"height_ratios": (2.2, 1)})
ax.axhline(m_pref, color=COOL, ls="--", lw=1.5); ax.text(4.9, m_pref - 0.1, "expected: 1.0 m", color=COOL, ha="right", va="top", fontsize=11)
ax.plot(tt, hist[:, 2], ".", color=GOOD, ms=4, label=r"reading $o$")
ax.plot(tt, hist[:, 0], color="0.3", lw=2, label=r"true distance $s$")
ax.plot(tt, hist[:, 1], color=ACCENT, lw=2, label=r"belief $\mu$")
ax.set_ylabel("m", fontsize=11); ax.legend(loc="upper right", fontsize=10, frameon=False); ax.set_ylim(0.6, 2.8)
axb.plot(tt, hist[:, 3], color="0.45", lw=2); axb.axhline(0, color="black", lw=0.5)
axb.set_ylabel(r"action $a$", fontsize=11); axb.set_xlabel("time (s)", fontsize=11)
fig.tight_layout(h_pad=0.3); fig.savefig(OUT / "d1_ai_loop.pdf"); plt.close(fig)
print(f"ai loop: s(5 s) = {hist[-1,0]:.2f}, mu = {hist[-1,1]:.2f}")

# 15. Gauss-Newton in 1-D: range to a landmark 0.8 m aside, h(s) = sqrt(s^2 + 0.8^2).
o_gn, m_gn, Po_gn, Pp_gn, d_gn = 2.0, 1.0, 25.0, 4.0, 0.8
h_gn = lambda s: np.sqrt(s * s + d_gn * d_gn)
J_gn = lambda s: s / h_gn(s)
E_gn = lambda s: 0.5 * Po_gn * (o_gn - h_gn(s)) ** 2 + 0.5 * Pp_gn * (s - m_gn) ** 2
sg = np.linspace(0.0, 2.6, 600)
fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.plot(sg, E_gn(sg), color="0.25", lw=2.5, label=r"bowl $-\ln p(o,s)$")
s_k = 0.6
for k, al in enumerate([0.45, 0.7, 1.0]):
    r_k = o_gn - h_gn(s_k); H_k = Po_gn * J_gn(s_k) ** 2 + Pp_gn
    g_k = -Po_gn * J_gn(s_k) * r_k + Pp_gn * (s_k - m_gn)
    ds_k = -g_k / H_k
    par = E_gn(s_k) + g_k * (sg - s_k) + 0.5 * H_k * (sg - s_k) ** 2
    if k < 2:
        ax.plot(sg, par, color=ACCENT, ls="--", lw=1.6, alpha=al, label="parabola at $s_k$" if k == 1 else None)
    ax.plot(s_k, E_gn(s_k), "o", color=ACCENT, ms=7, alpha=al, zorder=5)
    dx_, dy_ = [(0.12, 0.6), (0.12, 0.5), (-0.12, 0.6)][k]
    ax.text(s_k + dx_, E_gn(s_k) + dy_, f"$s_{k}$", color=ACCENT, ha="center", fontsize=14, alpha=al)
    if k < 2:
        ax.annotate("", xy=(s_k + ds_k, E_gn(s_k + ds_k)), xytext=(s_k, E_gn(s_k)),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.5, alpha=al, connectionstyle="arc3,rad=0"))
    s_k += ds_k
ax.text(s_k, -1.2, rf"$\mu = {s_k:.2f}$,  $\Sigma = 1/{H_k:.1f}$", color=ACCENT, fontsize=12, ha="center", va="top")
ax.set_xlim(0, 2.6); ax.set_ylim(-2.6, 16); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(r"$s$ (m)", fontsize=12); ax.tick_params(labelsize=11)
ax.legend(loc="upper right", fontsize=11, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "d1_gauss_newton.pdf"); plt.close(fig)
print(f"gauss-newton: mu = {s_k:.3f}, H = {H_k:.2f}, sigma = {1/np.sqrt(H_k):.3f}")

print("day 1 figures written")
