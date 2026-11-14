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
fig, ax = plt.subplots(figsize=(3.7, 1.9))
ax.plot(mus, complexity, color=COOL, lw=2, ls="--"); ax.plot(mus, inaccuracy, color=GOOD, lw=2, ls=":")
ax.plot(mus, F, color=ACCENT, lw=2.5)
ax.axvline(mu_star, color=MUTED, lw=0.8)
ax.plot(mu_star, F.min(), "o", color=ACCENT, ms=8, mec="white", mew=1.5, zorder=5)
ax.text(mu_star - 0.02, 5.4, f"belief {mu_star:.2f}", ha="right", fontsize=9, color="0.25")
ax.text(1.93, 1.3, "complexity", fontsize=10); ax.annotate("inaccuracy", xy=(2.11, ((y - 2.11) ** 2 + Sig) / (2 * r)), xytext=(1.93, 2.3), fontsize=10, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8)); ax.text(2.29, 3.4, r"$F$", fontsize=12)
ax.set_xlim(1.9, 2.85); ax.set_ylim(0, 6)
ax.set_xlabel(r"belief mean $\mu$ (prior 2.0, reading 2.6)", fontsize=10); ax.set_ylabel("nats + const", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "d1_complexity.pdf"); plt.close(fig)

# 9. Laplace: curvature of the bowl at the mode sets the width of the belief.
x = np.linspace(-1.6, 1.6, 500)
fig, axes = plt.subplots(2, 2, figsize=(4.0, 2.2), sharex=True, gridspec_kw={"height_ratios": (1.2, 1)})
for j, (k, name) in enumerate(((14.0, "sharp bowl"), (2.0, "flat bowl"))):
    U = k * (x ** 2 / 2 - 0.12 * x ** 3 + 0.05 * x ** 4)
    exact = np.exp(-U); exact /= exact.sum() * (x[1] - x[0])
    top, bot = axes[0, j], axes[1, j]
    top.plot(x, U, color=MUTED, lw=2); top.plot(x, k * x ** 2 / 2, color=ACCENT, lw=1.5, ls="--")
    top.set_ylim(0, 4); top.set_title(f"{name}, $\\mathcal{{H}}={k:.0f}$", fontsize=10, loc="left"); top.set_yticks([])
    bot.fill_between(x, npdf(x, 0, 1 / k), color=ACCENT, alpha=0.35); bot.plot(x, exact, color="0.25", lw=1)
    bot.set_ylim(0, 1.7); bot.set_yticks([]); bot.set_xlabel(r"$s-\mu$")
    bot.text(0.98, 0.72, f"$\\Sigma = {1 / k:.2f}$", transform=bot.transAxes, ha="right", fontsize=9)
axes[0, 0].set_ylabel(r"$-\ln p(o,s)$", fontsize=9); axes[1, 0].set_ylabel(r"$q(s)$", fontsize=9)
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

print("day 1 figures written")
