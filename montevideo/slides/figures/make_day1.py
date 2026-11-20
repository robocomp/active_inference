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

# 1. one wall as a belief, in the ROBOT's frame (the room frame comes later in the course).
#    Separate scan of the room without the table; the global rng is advanced exactly as before so the
#    later figures that reuse scan0/wall0 and rng do not change.
_ = Gaussian(wall0.mu, wall0.Sigma * 400).sample(25, rng); _ = Gaussian(wall0.mu, wall0.Sigma).sample(300, rng)
from toyworld.world import World  # noqa: E402
COOL_, ACCENT_, GOOD_, MUTED_ = "#2E6DB4", "#C0392B", "#2E8B57", "#7A7A7A"
rng1 = np.random.default_rng(11)
world_nb = World(world.rooms, world.doors, [t_ for t_ in world.things if t_.name != "table"])
scan_r = lidar.scan(world_nb, pose0, rng1)
pts_r = scan_r.points(); ang_r = scan_r.angles[scan_r.hit]
on_wall = (scan_r.labels[scan_r.hit] == "wall") & (point_segment_distance(se2.transform_points(pose0, pts_r), NORTH)[:, 0] < 0.15)
pw, bw = pts_r[on_wall], ang_r[on_wall]
alpha_true = np.pi / 2 - pose0[2]; d_true = 4.0 - pose0[1]
wall_r = laplace(res(pw), [alpha_true, d_true], isig(bw, alpha_true), jacobian=jac(pw)).belief
mu_r, S_r = wall_r.mu, wall_r.Sigma
nrm = np.array([np.cos(mu_r[0]), np.sin(mu_r[0])]); tng = np.array([-nrm[1], nrm[0]])
pivot = pw.mean(axis=0); L_piv = abs(pivot @ tng)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
inv = lambda q_: se2.transform_points(se2.inverse(pose0) if hasattr(se2, "inverse") else None, q_)
c_, s_ = np.cos(-pose0[2]), np.sin(-pose0[2])
to_r = lambda q_: (np.asarray(q_) - pose0[:2]) @ np.array([[c_, s_], [-s_, c_]])
for a_w, b_w in world_nb.room_walls("living"):
    seg = to_r(np.array([a_w, b_w]))
    ax.plot(seg[:, 0], seg[:, 1], color="0.75", lw=2)
for p_ in pts_r:
    ax.plot([0, p_[0]], [0, p_[1]], color=GOOD_, lw=0.3, alpha=0.25)
ax.plot(pts_r[:, 0], pts_r[:, 1], ".", color=GOOD_, ms=3, zorder=5)
samp25 = Gaussian(mu_r, S_r).sample(25, rng1)
for a_s, d_s in mu_r + (samp25 - mu_r) * 20:
    n_ = np.array([np.cos(a_s), np.sin(a_s)]); t_ = np.array([-n_[1], n_[0]])
    u_ = np.array([-3.5, 5.5])
    seg = d_s * n_ + u_[:, None] * t_
    ax.plot(seg[:, 0], seg[:, 1], color=ACCENT_, lw=0.6, alpha=0.3)
draw.robot(np.zeros(3), ax, color="#555555")
ax.annotate("", d_true * np.array([np.cos(alpha_true), np.sin(alpha_true)]), (0, 0),
            arrowprops=dict(arrowstyle="-|>", color=COOL_, lw=1.6))
ax.text(*(0.5 * d_true * np.array([np.cos(alpha_true), np.sin(alpha_true)]) + [0.05, -0.45]), "$d$", color=COOL_, fontsize=13)
ax.text(0.15, -0.45, "robot", fontsize=9, color="#555555")
ax.set_aspect("equal"); ax.set_xlim(-0.8, 5.0); ax.set_ylim(-3.0, 3.8)
ax.set_xlabel("forward (m)"); ax.set_ylabel("left (m)")
ax.set_title("robot's frame: 25 walls drawn from the belief (spread ×20)", fontsize=10)
T = np.diag([np.rad2deg(1.0), 1000.0])
m2, S2 = T @ (mu_r - [alpha_true, d_true]), T @ S_r @ T
w, V = np.linalg.eigh(S2); th = np.linspace(0, 2 * np.pi, 200)
pts2 = (T @ (Gaussian(mu_r, S_r).sample(300, rng1) - [alpha_true, d_true]).T).T
bx.scatter(pts2[:, 0], pts2[:, 1], s=6, color=ACCENT_, alpha=0.3, lw=0, label="possible walls under $q$")
p25 = (T @ (samp25 - [alpha_true, d_true]).T).T
bx.scatter(p25[:, 0], p25[:, 1], s=22, color=ACCENT_, edgecolor="black", lw=0.6, zorder=5, label="the 25 drawn on the left")
for k, a_, lab in ((1.515, 0.9, "68%"), (2.448, 0.5, "95%")):
    e = m2[:, None] + V @ (k * np.sqrt(w)[:, None] * np.stack([np.cos(th), np.sin(th)]))
    bx.plot(e[0], e[1], color=ACCENT_, lw=1.3, alpha=a_)
    j_ = np.argmax(e[0]); bx.text(e[0, j_] + 0.006, e[1, j_], lab, color=ACCENT_, fontsize=9, ha="left", va="center")
bx.plot(*m2, "D", color="white", mec=ACCENT_, mew=2, ms=8, zorder=7)
bx.annotate("$\\mu$: bottom of $F$,\nthe estimate", m2, (0.66, 0.86), textcoords="axes fraction", fontsize=9, color=ACCENT_,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=ACCENT_))
bx.legend(loc="lower right", fontsize=8, frameon=False)
bx.plot([0], [0], "x", color="black", ms=9, mew=2)
bx.annotate("true wall\n(unknown to the robot)", (0, 0), (0.04, 0.05), textcoords="axes fraction", fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
rho = S_r[0, 1] / np.sqrt(S_r[0, 0] * S_r[1, 1])
bx.text(0.03, 0.97, f"correlation $\\rho = {rho:.2f}$", transform=bx.transAxes, ha="left", va="top", fontsize=11, color=ACCENT_)
bx.set_xlim(-0.17, 0.3); bx.set_ylim(-10, 14)
bx.set_xlabel("angle $\\alpha$ − true (degrees)"); bx.set_ylabel("distance $d$ − true (mm)")
bx.set_title(f"belief $q(\\alpha, d)$: each dot is one possible wall\n$\\sigma_\\alpha = {np.rad2deg(np.sqrt(S_r[0,0])):.2f}°$, $\\sigma_d = {1000*np.sqrt(S_r[1,1]):.0f}$ mm", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "d1_wall_belief.pdf"); plt.close(fig)
print(f"wall (robot frame): n={len(pw)} alpha={np.rad2deg(mu_r[0]):.2f} deg d={mu_r[1]:.3f} rho={rho:.2f} L={L_piv:.2f}")

# 2. honesty test: predicted vs empirical sd, uniform vs incidence sigma
rows = {}
for name, sf in (("same 2 cm for every point", lambda b: 0.02), ("2 cm along the beam, projected", isig)):
    fits, sds = [], []
    for _ in range(400):
        p, b = wall_points(lidar.scan(world, pose0, rng))
        r = laplace(res(p), [np.pi / 2, 4.0], sf(b), jacobian=jac(p))
        fits.append(r.belief.mu); sds.append(r.belief.sd)
    rows[name] = np.mean(sds, axis=0) / np.std(fits, axis=0)
fig, ax = plt.subplots(figsize=(4.8, 3.6))
x = np.arange(2)
for i, ((name, ratio), col) in enumerate(zip(rows.items(), (MUTED_, GOOD_))):
    ax.bar(x + 0.38 * i, ratio, 0.36, label=name, color=col)
    for xi, v in zip(x + 0.38 * i, ratio):
        ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(1, color="k", lw=1.2)
ax.text(0.69, 1.003, "1 = right", fontsize=10, va="bottom", ha="center")
ax.axhspan(1.0, 1.25, color=ACCENT_, alpha=0.06)
ax.text(0.69, 1.22, "belief too wide", color=ACCENT_, fontsize=10, va="top", ha="center")
ax.set_xticks(x + 0.19, [r"angle $\alpha$", r"distance $d$"], fontsize=11); ax.set_ylim(0.9, 1.25)
ax.set_ylabel("predicted spread / actual scatter", fontsize=10)
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=9, loc="upper center", ncol=1)
fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(OUT / "d1_honesty.pdf"); plt.close(fig)

# 3. Occam: F_free - F_Manhattan vs prior width
manh = laplace(lambda t: res(p0)(np.array([np.pi / 2, t[0]])), [3.0], isig(b0), prior=Gaussian([3.0], [[4.0]]))
widths = np.deg2rad(np.logspace(-1, np.log10(30), 30))
dF = [laplace(res(p0), [np.pi / 2, 3.0], isig(b0), jacobian=jac(p0),
              prior=Gaussian([np.pi / 2, 3.0], np.diag([s ** 2, 4.0]))).free_energy - manh.free_energy for s in widths]
# the same comparison with the wall points tilted by 1.5 deg about their middle (deterministic, no rng)
_c = p0.mean(axis=0); _th = np.deg2rad(1.5)
p_tilt = (p0 - _c) @ np.array([[np.cos(_th), np.sin(_th)], [-np.sin(_th), np.cos(_th)]]) + _c
def _dF(pp, sd):
    m_ = laplace(lambda t: res(pp)(np.array([np.pi / 2, t[0]])), [3.0], isig(b0), prior=Gaussian([3.0], [[4.0]]))
    f_ = laplace(res(pp), [np.pi / 2, 3.0], isig(b0), jacobian=jac(pp), prior=Gaussian([np.pi / 2, 3.0], np.diag([sd ** 2, 4.0])))
    return f_.free_energy - m_.free_energy, f_.belief.sd[0]
d_flat, sa_flat = _dF(p0, np.deg2rad(5.0)); d_tilt, _ = _dF(p_tilt, np.deg2rad(5.0))
print(f"occam: s=5deg flat {d_flat:.1f} nats, tilted {d_tilt:.1f} nats, sigma_alpha {np.rad2deg(sa_flat):.3f} deg")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
ax.semilogx(np.degrees(widths), dF, "o-", ms=3, color="0.2", label="computed")
ax.semilogx(np.degrees(widths), np.log(widths / sa_flat), "--", color=ACCENT_, label=r"$\ln(s/\sigma_\alpha)$")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"$s$: prior spread of the angle (degrees)"); ax.set_ylabel("$F$(free) $-$ $F$(fixed)  (nats)")
ax.set_title("wall at 90°: the extra angle only costs", fontsize=11); ax.legend(frameon=False, fontsize=10)
bars = bx.bar([0, 1], [d_flat, d_tilt], color=[MUTED_, GOOD_], width=0.55)
bx.axhline(0, color="k", lw=0.8)
bx.text(0, d_flat + 3, f"+{d_flat:.1f}\nfixed wins", ha="center", va="bottom", fontsize=10)
bx.text(1, d_tilt - 3, f"{d_tilt:.0f}\nfree wins", ha="center", va="top", fontsize=10)
bx.set_xticks([0, 1], ["wall at 90°", "wall tilted 1.5°"]); bx.set_ylim(-100, 25)
bx.set_ylabel("$F$(free) $-$ $F$(fixed)  (nats)"); bx.set_title("the data decide ($s = 5°$)", fontsize=11)
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

# 8. complexity against inaccuracy along the belief's mean (wall example); constants dropped so each bowl is 0 at its anchor.
m, p_, y, r = mu_s, sd_s ** 2, o_star, sd_o ** 2
mus = np.linspace(1.8, 2.85, 400)
complexity = 0.5 * (mus - m) ** 2 / p_            # 0 at the prior mean
inaccuracy = 0.5 * (y - mus) ** 2 / r             # 0 at the reading
F = complexity + inaccuracy
mu_star = (m / p_ + y / r) / (1 / p_ + 1 / r)
assert abs(mus[np.argmin(F)] - mu_star) < 0.01
def spring(ax_, x0, x1, y0, n, amp, **kw):
    xs_ = np.linspace(x0, x1, 2 * n + 1); ys_ = y0 + amp * np.array([0] + [(-1) ** k for k in range(2 * n - 1)] + [0])
    ax_.plot(xs_, ys_, **kw)


fig, (ax0, ax) = plt.subplots(2, 1, figsize=(7.0, 4.3), sharex=True, gridspec_kw={"height_ratios": (1.0, 2.3)})
# top: the belief as a block held by two springs, a soft one to the odometry estimate and a stiff one to the LiDAR reading
ax0.axis("off"); ax0.set_ylim(-1, 1.6)
spring(ax0, m, mu_star - 0.02, 0, 7, 0.35, color=COOL, lw=1.4)
spring(ax0, mu_star + 0.02, y, 0, 4, 0.35, color=GOOD, lw=3.2)
for x_, c_ in ((m, COOL), (y, GOOD)):
    ax0.plot([x_, x_], [-0.6, 0.6], color=c_, lw=4)
ax0.add_patch(plt.Rectangle((mu_star - 0.02, -0.35), 0.04, 0.7, color=ACCENT, zorder=5))
ax0.text(m, 0.85, "odometry says 2.0 m\n(±0.5: soft spring)", color=COOL, ha="center", va="bottom", fontsize=11)
ax0.text(y + 0.02, 0.85, "LiDAR says 2.6 m\n(±0.2: stiff spring)", color=GOOD, ha="center", va="bottom", fontsize=11)
ax0.text(mu_star - 0.03, -0.75, "belief", color=ACCENT, ha="right", va="center", fontsize=11)
# bottom: the energy in each spring and their sum
ax.plot(mus, complexity, color=COOL, lw=2.5, ls="--"); ax.plot(mus, inaccuracy, color=GOOD, lw=2.5, ls=":")
ax.plot(mus, F, color=ACCENT, lw=3)
ax.axvline(mu_star, color=ACCENT, lw=0.8, ls=":")
ax.plot(mu_star, F.min(), "o", color=ACCENT, ms=10, mec="white", mew=1.5, zorder=7)
ax.text(mu_star - 0.02, 3.9, f"belief {mu_star:.2f}:\nbottom of $F$", fontsize=12, color=ACCENT, va="top", ha="right")
ax.text(1.83, 0.75, "complexity:\nsoft-spring energy", fontsize=12, color=COOL, va="bottom")
ax.text(2.84, 3.9, "inaccuracy:\nstiff-spring energy", fontsize=12, color=GOOD, ha="right", va="top")
ax.text(2.2, 2.55, r"$F$ = total", fontsize=14, color=ACCENT)
ax.set_xlim(1.8, 2.85); ax.set_ylim(0, 4.0); ax.set_yticks([])
ax.set_xticks([2.0, 2.2, 2.4, 2.6, 2.8]); ax.tick_params(labelsize=11)
ax.set_xlabel(r"distance to the wall $\mu$ (m)", fontsize=12); ax.set_ylabel("energy (nats)", fontsize=12)
fig.tight_layout(h_pad=0.2); fig.savefig(OUT / "d1_complexity.pdf"); plt.close(fig)

# 9a. a generic one-peak curve g(s) = s^3 e^{-s} and its -ln: a bowl; Laplace parabola and Gaussian at mu = 3.
sa = np.linspace(0.3, 9.0, 800)
ga = sa ** 3 * np.exp(-sa); ga /= ga.sum() * (sa[1] - sa[0])
mua, Ha = 3.0, 1 / 3.0                    # -ln g = s - 3 ln s: slope 0 at s = 3, curvature 3/s^2 = 1/3
fig, (l_, r_) = plt.subplots(1, 2, figsize=(9.0, 2.6))
l_.plot(sa, ga, color="0.15", lw=2.4, label="$g(s)$: one peak")
l_.plot(sa, npdf(sa, mua, 1 / Ha), color=ACCENT, lw=1.8, ls="--", label=r"Gaussian $\mathcal{N}(\mu, 1/\mathcal{H})$")
l_.axvline(mua, color="0.5", lw=0.8, ls=":")
l_.set_ylim(0, 0.3); l_.set_yticks([]); l_.set_xlabel("$s$"); l_.set_title("$g(s)$", fontsize=12)
l_.legend(loc="upper right", fontsize=9, frameon=False)
U = -np.log(ga); U0 = U.min(); U -= U0
r_.plot(sa, U, color="0.15", lw=2.4, label=r"$-\ln g(s)$: a bowl")
r_.plot(sa, 0.5 * Ha * (sa - mua) ** 2, color=ACCENT, lw=1.8, ls="--", label=r"parabola $\frac{1}{2}\mathcal{H}(s-\mu)^2$")
r_.plot(mua, 0, "o", color=ACCENT, ms=7, zorder=5)
r_.text(mua, -0.35, r"$\mu$", color=ACCENT, fontsize=13, ha="center", va="top")
r_.axvline(mua, color="0.5", lw=0.8, ls=":")
r_.set_ylim(-0.6, 5.5); r_.set_yticks([]); r_.set_xlabel("$s$"); r_.set_title(r"$-\ln g(s)$", fontsize=12)
r_.legend(loc="upper center", fontsize=9, frameon=False)
fig.tight_layout(w_pad=5)
xm_ = 0.5 * (l_.get_position().x1 + r_.get_position().x0)
fig.patches.append(matplotlib.patches.FancyArrowPatch((xm_ - 0.035, 0.55), (xm_ + 0.035, 0.55), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=18, color="0.2", lw=1.6))
fig.text(xm_, 0.62, r"$-\ln$", ha="center", fontsize=13)
fig.savefig(OUT / "d1_laplace_generic.pdf"); plt.close(fig)

# 9. Laplace on the frame-43 example: apparent size o = 1/s, prior N(3, 1^2), reading o = 0.4.
#    Bowl U(s) = -ln p(o, s); its bottom is the mean, its curvature the precision. Two sensor noises.
xs9 = np.linspace(1.2, 6.0, 3000); dx9 = xs9[1] - xs9[0]
fig, axes = plt.subplots(2, 2, figsize=(9.0, 3.9), sharex=True, gridspec_kw={"height_ratios": (1.15, 1)})
for j, (so9, name) in enumerate(((0.025, "precise sensor (noise $\\pm 0.025$)"), (0.1, "the sensor of frame 43 (noise $\\pm 0.1$)"))):
    U = 0.5 * ((xs9 - 3.0) / 1.0) ** 2 + 0.5 * ((0.4 - 1 / xs9) / so9) ** 2
    i0 = np.argmin(U); mu9 = xs9[i0]; U -= U[i0]
    H9 = (U[i0 + 5] - 2 * U[i0] + U[i0 - 5]) / (5 * dx9) ** 2
    Hs = f"{H9:.0f}" if H9 >= 10 else f"{H9:.1f}"
    exact = np.exp(-U); exact /= exact.sum() * dx9
    top, bot = axes[0, j], axes[1, j]
    top.plot(xs9, U, color="0.3", lw=2.5, label=r"bowl $-\ln p(o,s)$")
    top.plot(xs9, 0.5 * H9 * (xs9 - mu9) ** 2, color=ACCENT, lw=2, ls="--", label="parabola at the bottom")
    top.plot(mu9, 0, "o", color=ACCENT, ms=8, zorder=5)
    top.set_ylim(-0.9, 5); top.set_yticks([])
    top.set_title(f"{name}\ncurvature $\\mathcal{{H}} = {Hs}$", fontsize=11)
    top.text(mu9, -0.3, r"$\mu$", fontsize=14, color=ACCENT, ha="center", va="top")
    bot.fill_between(xs9, npdf(xs9, mu9, 1 / H9), color=ACCENT, alpha=0.35, label=r"belief $q(s)$")
    bot.plot(xs9, exact, color="0.15", lw=1.4, label=r"exact $p(s\mid o)$")
    bot.set_ylim(0, 1.25 * max(exact.max(), npdf(mu9, mu9, 1 / H9))); bot.set_yticks([])
    bot.set_xlabel(r"$s$: distance (m)", fontsize=11)
    bot.text(0.98, 0.6, f"$\\mu = {mu9:.2f}$ m\n$\\Sigma = 1/{Hs}$, $\\sigma = {1 / np.sqrt(H9):.2f}$ m",
             transform=bot.transAxes, ha="right", fontsize=11, color=ACCENT)
    print(f"laplace {so9}: mu={mu9:.3f} H={H9:.2f} sigma={1/np.sqrt(H9):.3f}")
axes[0, 0].legend(loc="upper right", fontsize=10, frameon=False); axes[1, 0].legend(loc="lower right", fontsize=10, frameon=False)
fig.tight_layout(h_pad=0.4); fig.savefig(OUT / "d1_laplace.pdf"); plt.close(fig)

# 9c. why sigma^2 = 1/H: F as a function of the belief's spread sigma, for the two bowls of block 9.
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=False)
for ax_, H_, name in ((axs[0], 40.76, "precise sensor, $\\mathcal{H} = 41$"), (axs[1], 2.74, "sensor of frame 43, $\\mathcal{H} = 2.7$")):
    sgm = np.linspace(0.03, 1.3, 600)
    cost = 0.5 * H_ * sgm ** 2
    pay = -0.5 * np.log(sgm ** 2)
    Ftot = cost + pay
    off = Ftot.min()
    ax_.plot(sgm, cost, color=ACCENT, lw=1.8, ls="--", label=r"$\frac{1}{2}\mathcal{H}\sigma^2$: grows with $\sigma$")
    ax_.plot(sgm, pay, color=COOL, lw=1.8, ls="--", label=r"$-\frac{1}{2}\ln\sigma^2$: falls with $\sigma$")
    ax_.plot(sgm, Ftot, color="0.1", lw=2.6, label=r"$F$: their sum (+ terms without $\sigma$)")
    sb = 1 / np.sqrt(H_)
    ax_.plot(sb, 0.5 + np.log(np.sqrt(H_)), "o", color="0.1", ms=8, zorder=5)
    ax_.axvline(sb, color="0.1", lw=0.8, ls=":")
    ax_.annotate(f"smallest $F$ at $\\sigma^2 = 1/\\mathcal{{H}}$,\nso $\\sigma = \\sqrt{{1/\\mathcal{{H}}}} = {sb:.2f}$ m", (sb, 0.5 + np.log(np.sqrt(H_))), (0.97, 0.95), textcoords="axes fraction", ha="right", va="top", fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"))
    ax_.set_ylim(-1.5, 5.5); ax_.set_xlim(0, 1.3); ax_.set_yticks([])
    ax_.set_xlabel(r"$\sigma$: spread of the belief $q$ (m)"); ax_.set_title(name, fontsize=11)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / "d1_sigma_balance.pdf"); plt.close(fig)

# 9b. why the posterior is not Gaussian: start from a nonlinear likelihood.
#     Sensor: apparent size o = 1/s (twice as far, half as tall), noise on o. Read p(o|s) as a function of s:
#     a lopsided bump. Times a Gaussian prior: a lopsided posterior.
sg = np.linspace(0.8, 7.0, 2000); dsg = sg[1] - sg[0]
m_pr, sd_pr, o_r, so = 3.0, 1.0, 0.4, 0.1
lik = np.exp(-0.5 * ((o_r - 1 / sg) / so) ** 2)
prior = npdf(sg, m_pr, sd_pr ** 2)
post = prior * lik; post /= post.sum() * dsg
mp = (sg * post).sum() * dsg; vp = ((sg - mp) ** 2 * post).sum() * dsg
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
a1.fill_between(sg, lik, color=GOOD, alpha=0.2); a1.plot(sg, lik, color=GOOD, lw=2.4)
a1.set_title(r"1. likelihood $p(o\mid s)$, as a function of $s$", fontsize=11)
a1.text(4.3, 0.35, "long tail\nto the far side", color=GOOD, fontsize=10)
ins = a1.inset_axes([0.56, 0.56, 0.42, 0.3])
ins.plot(sg, 1 / sg, color="0.2", lw=1.6)
ins.axhspan(o_r - so, o_r + so, color=GOOD, alpha=0.25)
ins.set_xlim(0.8, 7); ins.set_ylim(0, 1.0); ins.set_xticks([]); ins.set_yticks([])
ins.set_title("sensor: $o = 1/s$", fontsize=9, pad=2)
a1.set_ylim(0, 1.75)
a2.fill_between(sg, prior, color=COOL, alpha=0.2); a2.plot(sg, prior, color=COOL, lw=2.4)
a2.set_title(r"2. prior $p(s)$: a Gaussian", fontsize=11)
a2.set_ylim(0, 1.45 * prior.max())
a3.plot(sg, prior / prior.max() * post.max() * 0.6, color=COOL, lw=1, alpha=0.6)
a3.fill_between(sg, post, color="0.3", alpha=0.25); a3.plot(sg, post, color="0.1", lw=2.4, label=r"posterior $p(s\mid o)$")
a3.plot(sg, npdf(sg, mp, vp), color=ACCENT, lw=1.6, ls="--", label="a Gaussian, for comparison")
a3.set_title(r"3. prior $\times$ likelihood: lopsided", fontsize=11)
a3.set_ylim(0, 1.45 * post.max()); a3.legend(loc="upper right", fontsize=9, frameon=False)
for ax_ in (a1, a2, a3):
    ax_.set_yticks([]); ax_.set_xlabel(r"$s$: distance (m)"); ax_.set_xlim(0.8, 7.0)
fig.tight_layout(w_pad=3.5)
for (l_, r_), sym in (((a1, a2), r"$\times$"), ((a2, a3), "=")):
    fig.text(0.5 * (l_.get_position().x1 + r_.get_position().x0), 0.55, sym, fontsize=22, ha="center", va="center")
fig.savefig(OUT / "d1_why_skew.pdf"); plt.close(fig)

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
fig, (ax, axb) = plt.subplots(2, 1, figsize=(5.2, 3.9), sharex=True, gridspec_kw={"height_ratios": (2.4, 1)})
ax.axhline(m_pref, color=COOL, ls="--", lw=1.5); ax.text(4.95, m_pref - 0.16, "goal = prior mean: 1.0 m", color=COOL, ha="right", va="top", fontsize=10)
ax.plot(tt, hist[:, 2], ".", color=GOOD, ms=4, label=r"reading $o$")
ax.plot(tt, hist[:, 0], color="0.3", lw=2, label=r"true distance $s$")
ax.plot(tt, hist[:, 1], color=ACCENT, lw=2, label=r"belief $\mu$")
ax.annotate("starts 2.6 m\nfrom the wall", (0.1, 2.55), (0.9, 2.45), fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("drives in until the reading\nagrees with the belief", (1.1, 1.05), (1.6, 1.6), fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_ylabel("distance (m)", fontsize=10); ax.legend(loc="upper right", fontsize=9, frameon=False); ax.set_ylim(0.6, 2.85)
axb.plot(tt, hist[:, 3], color="0.45", lw=2); axb.axhline(0, color="black", lw=0.5)
axb.set_ylabel("speed $a$\n(m/s)", fontsize=10); axb.set_xlabel("time (s)", fontsize=10)
axb.text(4.95, -1.5, "negative: towards the wall", ha="right", fontsize=9, color="0.4")
fig.tight_layout(h_pad=0.3); fig.savefig(OUT / "d1_ai_loop.pdf"); plt.close(fig)
print(f"ai loop: s(5 s) = {hist[-1,0]:.2f}, mu = {hist[-1,1]:.2f}")

# 15. Gauss-Newton in 1-D: range to a landmark 0.8 m aside, h(s) = sqrt(s^2 + 0.8^2).
o_gn, m_gn, Po_gn, Pp_gn, d_gn = 2.0, 1.0, 25.0, 4.0, 0.8
h_gn = lambda s: np.sqrt(s * s + d_gn * d_gn)
J_gn = lambda s: s / h_gn(s)
E_gn = lambda s: 0.5 * Po_gn * (o_gn - h_gn(s)) ** 2 + 0.5 * Pp_gn * (s - m_gn) ** 2
sg = np.linspace(0.0, 2.6, 600)
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(sg, E_gn(sg), color="0.2", lw=2.6, label=r"bowl $-\ln p(o,s)$")
s_k = 0.6
shades = [ACCENT, "#E08E0B", COOL]
for k in range(3):
    r_k = o_gn - h_gn(s_k); H_k = Po_gn * J_gn(s_k) ** 2 + Pp_gn
    g_k = -Po_gn * J_gn(s_k) * r_k + Pp_gn * (s_k - m_gn)
    ds_k = -g_k / H_k
    par = E_gn(s_k) + g_k * (sg - s_k) + 0.5 * H_k * (sg - s_k) ** 2
    ax.plot(sg, par, color=shades[k], ls="--", lw=1.7)
    ax.plot(s_k, E_gn(s_k), "o", color=shades[k], ms=7, zorder=5)
    lab_xy = [(0.25, 14.0), (2.35, 9.5), (1.25, 6.5)][k]
    ax.text(*lab_xy, f"$P_{k}$", color=shades[k], fontsize=14, ha="center", fontweight="bold")
    ax.text(s_k, E_gn(s_k) - 1.1, f"$s_{k}$", color=shades[k], fontsize=12, ha="center", va="top")
    if k < 2:
        ax.annotate("", xy=(s_k + ds_k, 0.5 * H_k * ds_k ** 2 + g_k * ds_k + E_gn(s_k)), xytext=(s_k, E_gn(s_k)),
                    arrowprops=dict(arrowstyle="-|>", color=shades[k], lw=1.5))
    print(f"GN step {k}: s={s_k:.3f} bowl={E_gn(s_k):.2f} slope={g_k:.2f} H={H_k:.2f} -> {s_k + ds_k:.3f}")
    s_k += ds_k
ax.set_xlim(0, 2.6); ax.set_ylim(-2.6, 16); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel(r"$s$ (m)", fontsize=12); ax.tick_params(labelsize=11)
ax.legend(loc="upper right", fontsize=11, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "d1_gauss_newton.pdf"); plt.close(fig)
print(f"gauss-newton: mu = {s_k:.3f}, H = {H_k:.2f}, sigma = {1/np.sqrt(H_k):.3f}")

print("day 1 figures written")
