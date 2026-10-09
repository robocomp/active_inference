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
print("day 1 figures written")
