# %% [markdown]
# # Day 2 — A wall as a belief
#
# You will
#
# 1. fit a wall to a LiDAR scan as a **Laplace belief**, a Gaussian whose covariance is the inverse
#    curvature of the free energy (frames 18--23);
# 4. lower the same free energy by **perception** and by **action**, and see where each one stops
#    (frames 24--27).
#
# Do parts 1 and 4 first. Parts 2 and 3 go with the two backup slides: do them if you have time.
#
# 2. check whether that covariance is right, and fix it when it is not;
# 3. use the **free energy** to choose between two models of the same wall.

# %%
import sys
from pathlib import Path

LABS = Path.cwd() if (Path.cwd() / "toyworld").exists() else Path.cwd().parent
sys.path.insert(0, str(LABS))

import matplotlib.pyplot as plt
import numpy as np

from toyworld import Gaussian, Lidar, draw, info_gain, laplace, se2, two_room_apartment
from toyworld.geometry import point_segment_distance

rng = np.random.default_rng(0)
world = two_room_apartment()

# %% [markdown]
# # A wall as a belief
#
# Choosing *where to look next* is Day 5, once action has its own functional. Today the robot knows
# its own pose exactly, and we cheat on *which* LiDAR points belong to the wall. Day 4 removes both
# cheats.

# %% [markdown]
# ## 0. Meet the world
#
# Two rooms, a door, a table, a bottle and a chair. The sensor today is a forward-looking LiDAR
# with a 90° field of view: 90 beams, 2 cm of independent range noise per beam.

# %%
lidar = Lidar(n_beams=90, fov=np.deg2rad(90), sigma=0.02)
pose0 = np.array([4.5, 1.0, np.deg2rad(110)])   # the robot's (known) pose today
scan0 = lidar.scan(world, pose0, rng)

fig, ax = plt.subplots(figsize=(9, 4.5))
draw.world(world, ax)
draw.scan(scan0, ax=ax, rays=True)
draw.robot(pose0, ax)
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 5.5)
print("what the beams hit:", {l: int((scan0.labels == l).sum()) for l in set(scan0.labels)})

# %% [markdown]
# The robot receives only `scan.points()`: hit points in its own sensor frame. Everything with
# `true_` in its name, and `scan.labels`, is ground truth, kept for grading. The cell below is
# today's cheat: it uses the true labels and the true wall geometry to pick out the points on the
# living room's north wall.

# %%
NORTH = np.array([[[0.0, 4.0], [6.0, 4.0]]])   # the living room's north wall, as a segment


def wall_points(scan, wall=NORTH, tol=0.15):
    """World-frame hit points on `wall`, and the world bearing of each beam. (Uses ground truth!)"""
    pts = se2.transform_points(scan.pose, scan.points())
    bearings = scan.pose[2] + scan.angles[scan.hit]
    on_wall = (scan.labels[scan.hit] == "wall") & (point_segment_distance(pts, wall)[:, 0] < tol)
    return pts[on_wall], bearings[on_wall]


pts0, bear0 = wall_points(scan0)
print(f"{len(pts0)} points on the north wall, spanning x = {pts0[:, 0].min():.2f} .. {pts0[:, 0].max():.2f} m")

# %% [markdown]
# ## 1. A wall as a Laplace belief
#
# Model the wall as an infinite line with unit normal $n(\alpha) = (\cos\alpha, \sin\alpha)$ at
# distance $d$ from the origin. A point $p$ lies on it when $n(\alpha)^\top p = d$, so the
# residual of point $i$ is
#
# $$ r_i(\alpha, d) = \cos\alpha\, x_i + \sin\alpha\, y_i - d, \qquad
#    r_i \sim \mathcal N(0, \sigma^2). $$
#
# **Exercise 1.** Write the residual and its Jacobian $\partial r_i / \partial(\alpha, d)$.
# (Check yourself: the row for point $i$ should be $[\,-\sin\alpha\, x_i + \cos\alpha\, y_i,\; -1\,]$.)

# %%
def residual(pts):
    def r(theta):
        alpha, d = theta
        # BEGIN SOLUTION
        return np.cos(alpha) * pts[:, 0] + np.sin(alpha) * pts[:, 1] - d
        # END SOLUTION
    return r


def jacobian(pts):
    def J(theta):
        alpha, d = theta
        # BEGIN SOLUTION
        return np.stack([-np.sin(alpha) * pts[:, 0] + np.cos(alpha) * pts[:, 1], -np.ones(len(pts))], axis=1)
        # END SOLUTION
    return J

# %% [markdown]
# **Exercise 2.** Fit the wall with `laplace(residual, x0, sigma, jacobian=...)`. It runs
# Gauss–Newton to the mode, and returns the belief $\mathcal N(\mu, H^{-1})$ together with the
# Laplace free energy. Start from a deliberately wrong guess, $\alpha_0 = 100°$ and
# $d_0 = 3.5$ m. Then draw 30 lines sampled from the belief.

# %%
# BEGIN SOLUTION
fit0 = laplace(residual(pts0), [np.deg2rad(100), 3.5], 0.02, jacobian=jacobian(pts0))
# END SOLUTION
wall0 = fit0.belief
print(f"alpha = {np.degrees(wall0.mu[0]):.3f}° ± {np.degrees(wall0.sd[0]):.3f}°,  d = {wall0.mu[1]:.4f} ± {wall0.sd[1]:.4f} m")
print(f"correlation(alpha, d) = {wall0.Sigma[0, 1] / np.prod(wall0.sd):+.2f};  Gauss-Newton iterations: {fit0.iterations}")


def draw_wall_belief(belief, ax, n=30, color="royalblue", x=np.array([-0.5, 6.5])):
    for a, d in belief.sample(n, rng):
        ax.plot(x, (d - np.cos(a) * x) / np.sin(a), color=color, lw=0.6, alpha=0.4)


fig, ax = plt.subplots(figsize=(9, 3))
draw.world(world, ax, things=False)
draw_wall_belief(Gaussian(wall0.mu, wall0.Sigma * 100), ax)   # ×10 in sd, so you can see it
ax.plot(pts0[:, 0], pts0[:, 1], ".", color="crimson", ms=3)
ax.set_xlim(-0.5, 6.5); ax.set_ylim(3.0, 4.6); ax.set_title("belief samples (sd ×10)")

# %% [markdown]
# **Question.** Why are $\alpha$ and $d$ correlated, and what sets the sign? (Hint: where is the
# origin relative to the points? Move the origin to the centre of the points and the correlation
# should disappear.)

# %% [markdown] solution
# *Answer.* $d$ is a distance from the **origin**, and the points sit at $x \approx 2$–$6$ m.
# Rotating the line about the centroid of the points changes its distance to the origin, so the
# data cannot pin $\alpha$ without moving $d$ with it. The Jacobian column for $\alpha$ is the lever
# arm $-\sin\alpha\,x_i + \cos\alpha\,y_i \approx -x_i$. Centre the parameterisation on the
# centroid and the information matrix becomes diagonal.

# %% [markdown]
# ## 2. Is the covariance right? (if you have time; backup slide)
#
# $\Sigma = H^{-1}$ is a **claim** about how far the fit would wander if you scanned again. Test the
# claim: scan 500 times from the same pose, fit each scan, and compare the spread of the fits with
# the predicted $\sigma$.
#
# **Exercise 3.** Fill in the loop.

# %%
def monte_carlo(sigma_fn, n=500):
    fits, sigmas = [], []
    for _ in range(n):
        s = lidar.scan(world, pose0, rng)
        p, b = wall_points(s)
        # BEGIN SOLUTION
        res = laplace(residual(p), [np.pi / 2, 4.0], sigma_fn(p, b), jacobian=jacobian(p))
        fits.append(res.belief.mu)
        sigmas.append(res.belief.sd)
        # END SOLUTION
    return np.std(fits, axis=0), np.mean(sigmas, axis=0)


emp, pred = monte_carlo(lambda p, b: 0.02)
print("uniform sigma:   empirical sd", emp.round(5), "  predicted sd", pred.round(5), "  ratio", (pred / emp).round(3))

# %% [markdown]
# The prediction is about 10% **too pessimistic**. That is not Monte Carlo noise; re-run it and the
# gap stays. Something in the likelihood is wrong.
#
# **Exercise 4.** The LiDAR's noise is along the **beam**, but the residual measures distance along
# the wall's **normal**. A range error $\epsilon$ on a beam with direction $u$ moves the point by
# $\epsilon\,(u^\top n)$ off the wall. So the residual's standard deviation is
# $\sigma_i = \sigma\,|u_i^\top n|$, not $\sigma$. Write that per-point sigma and re-run the test.

# %%
def incidence_sigma(p, bearings, alpha=np.pi / 2):
    # BEGIN SOLUTION
    u = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)
    n = np.array([np.cos(alpha), np.sin(alpha)])
    return 0.02 * np.abs(u @ n)
    # END SOLUTION


emp, pred = monte_carlo(incidence_sigma)
print("incidence sigma: empirical sd", emp.round(5), "  predicted sd", pred.round(5), "  ratio", (pred / emp).round(3))

# %% [markdown]
# **Lesson.** A Laplace covariance is exactly as right as the likelihood you wrote. Here the wrong
# likelihood made the belief *under*-confident. That costs little. Day 4 has a wrong likelihood
# that makes it badly *over*-confident, which costs a great deal. The real robot's localiser
# weights each LiDAR return by the incidence $|\hat r^\top \hat n|$ for exactly this reason.
#
# From here on, fit walls with `incidence_sigma`.

# %%
fit0 = laplace(residual(pts0), [np.pi / 2, 4.0], incidence_sigma(pts0, bear0), jacobian=jacobian(pts0))
wall0 = fit0.belief

# %% [markdown]
# ## 3. Free energy as the price of a model (if you have time; backup slide)
#
# Rooms are mostly rectilinear. A **Manhattan** model of this wall fixes $\alpha = 90°$ and has a
# single parameter, $d$. The **free** model keeps both parameters, with a prior
# $\alpha \sim \mathcal N(90°, s^2)$. Both models fit the data equally well, so which does the free
# energy prefer, and by how much?
#
# The Laplace free energy approximates the surprise $-\ln p(o \mid \text{model})$. Lower is better,
# and the difference is a log Bayes factor in nats.
#
# **Exercise 5.** Compute $\Delta F = F_\text{free} - F_\text{Manhattan}$ for prior widths
# $s$ from 0.1° to 30°. Use the prior $d \sim \mathcal N(3, 2^2)$ in both models.

# %%
sig0 = incidence_sigma(pts0, bear0)
manhattan = laplace(lambda th: residual(pts0)(np.array([np.pi / 2, th[0]])), [3.0], sig0,
                    prior=Gaussian([3.0], [[4.0]]))
widths = np.deg2rad(np.logspace(-1, np.log10(30), 25))
dF = []
for s in widths:
    # BEGIN SOLUTION
    free = laplace(residual(pts0), [np.pi / 2, 3.0], sig0, jacobian=jacobian(pts0),
                   prior=Gaussian([np.pi / 2, 3.0], np.diag([s ** 2, 4.0])))
    dF.append(free.free_energy - manhattan.free_energy)
    # END SOLUTION

plt.figure(figsize=(6, 3.2))
plt.semilogx(np.degrees(widths), dF, "o-")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("prior sd on the wall angle, s (deg)"); plt.ylabel(r"$F_{free} - F_{Manhattan}$ (nats)")
plt.title("the complexity price of a free parameter")

# %% [markdown]
# Above zero, the Manhattan model wins. The penalty grows like $\ln(s/\sigma_\alpha)$: it is the
# complexity term, $\mathrm{KL}[q\,\|\,p]$, charging for how much the prior had to be squeezed to
# reach the posterior.
#
# **Exercise 6.** Is the Manhattan model therefore always right? Rotate the points by 1.5° about
# their centroid, which is what a slightly skew wall would look like, and compare again at
# $s = 5°$.

# %%
c = pts0.mean(axis=0)
tilted = (pts0 - c) @ se2.rot(np.deg2rad(1.5)).T + c
# BEGIN SOLUTION
m_t = laplace(lambda th: residual(tilted)(np.array([np.pi / 2, th[0]])), [3.0], sig0, prior=Gaussian([3.0], [[4.0]]))
f_t = laplace(residual(tilted), [np.pi / 2, 3.0], sig0, jacobian=jacobian(tilted),
              prior=Gaussian([np.pi / 2, 3.0], np.diag([np.deg2rad(5) ** 2, 4.0])))
print(f"tilted wall: F_free - F_Manhattan = {f_t.free_energy - m_t.free_energy:+.1f} nats")
# END SOLUTION

# %% [markdown]
# **Lesson.** Nobody wrote "if the angle error exceeds k degrees, abandon Manhattan". The
# accuracy term grew until it outweighed the complexity term. That is the course's first instance
# of **no thresholds**: the decision falls out of comparing two free energies.

# %% [markdown]
# ## 4. Perception and action lower the same $F$
#
# The robot of frames 25--27. $s$: its distance to the wall. We write the likelihood
# $p(o \mid s) = \mathcal N(o;\, s,\, 0.2^2)$ and, as the prior, the **goal**
# $p(s) = \mathcal N(1.0,\, 0.5^2)$: where we want the robot to be. With the Laplace belief
# $q = \mathcal N(\mu, \sigma^2)$ and $\sigma$ at its best value, up to constants,
#
# $$F(\mu, o) = \tfrac12\,25\,(o-\mu)^2 + \tfrac12\,4\,(\mu-1)^2, \qquad 25 = 1/0.2^2,\; 4 = 1/0.5^2.$$
#
# **Exercise 7 (perception).** Hold the reading fixed at $o = 1.5$. Start at $\mu = 1.0$ and step
# $\mu \leftarrow \mu - \eta\,\partial F/\partial\mu$ with $\eta = 0.01$ until it stops moving.
# Where does $\mu$ stop, and what is $F$ there? Repeat for $o = 1.0,\ 1.25,\ 2.0$ and compare
# each final $F$ with $\tfrac12\,(o-1)^2/0.29$, which is the surprise $-\ln p(o)$ without its
# constant: $p(o) = \mathcal N(o;\, 1,\, 0.2^2 + 0.5^2)$.

# %%
def F_wall(mu, o):
    return 0.5 * 25 * (o - mu) ** 2 + 0.5 * 4 * (mu - 1) ** 2


for o in (1.0, 1.25, 1.5, 2.0):
    mu = 1.0
    # BEGIN SOLUTION
    for _ in range(2000):
        mu -= 0.01 * (-25 * (o - mu) + 4 * (mu - 1))
    # END SOLUTION
    print(f"o = {o:.2f}: mu stops at {mu:.3f}, F = {F_wall(mu, o):.3f}, "
          f"surprise without constant = {0.5 * (o - 1) ** 2 / 0.29:.3f}")

# %% [markdown] solution
# $\mu$ stops at $(25\,o + 4\cdot1)/29$, and the final $F$ equals the surprise every time
# (0.431 for $o = 1.5$). Perception can lower $F$ down to the surprise and no further: the surprise
# depends only on $o$, and no belief changes $o$.

# %% [markdown]
# **Exercise 8 (action).** Now the robot drives. It starts at $s = 2.6$ m with $\mu = 2.6$ and speed
# $a = 0$. Every 0.1 s: read $o = s + $ noise (use 0.03 m, the real LiDAR); take both slopes of $F$ at the current
# $\mu$, then step $\mu$ as above with $\eta = 0.04$ and step the speed $a \leftarrow a - 2.0\,\dfrac{\partial F}{\partial o}\,
# \dfrac{\partial o}{\partial a}$, where driving at $a$ for 0.1 s moves the robot by $a \cdot 0.1$,
# so $\partial o/\partial a = 0.1$ s; then move, $s \leftarrow s + 0.1\,a$. Run 5 s. Plot $s$,
# $\mu$ and $F$ against time, and the surprise $\tfrac12\,(o-1)^2/0.29$ next to $F$.

# %%
rng_a = np.random.default_rng(3)
s_true, mu, a = 2.6, 2.6, 0.0
log = []
for _ in range(50):
    o = s_true + rng_a.normal(0, 0.03)
    # BEGIN SOLUTION
    dF_dmu = -25 * (o - mu) + 4 * (mu - 1)      # both slopes at the current mu
    dF_do = 25 * (o - mu)
    mu -= 0.04 * dF_dmu
    a -= 2.0 * dF_do * 0.1
    s_true += 0.1 * a
    # END SOLUTION
    log.append((s_true, mu, F_wall(mu, o), 0.5 * (o - 1) ** 2 / 0.29))
log = np.array(log); t = 0.1 * np.arange(len(log))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.2), sharex=True)
ax1.plot(t, log[:, 0], label="true distance $s$"); ax1.plot(t, log[:, 1], label=r"belief $\mu$")
ax1.axhline(1.0, ls="--", color="k", lw=0.8); ax1.set_ylabel("m"); ax1.legend()
ax2.plot(t, log[:, 2], label="$F$"); ax2.plot(t, log[:, 3], ls="--", label="surprise (no constant)")
ax2.set_ylabel("nats"); ax2.set_xlabel("time (s)"); ax2.legend()

# %% [markdown] solution
# $F$ sits just above the surprise the whole time: perception keeps it there. The surprise itself
# falls only because the robot drives, which changes $o$. Both reach about 0 when the robot is at
# 1 m, the goal. Nobody wrote "drive to 1 m": the goal is a prior, and lowering $F$ makes it true.

# %% [markdown]
# ## Take-aways
# - A belief is a mean **and** a covariance. The covariance is the curvature of the free energy,
#   and it is only as right as the likelihood.
# - Perception lowers $F$ down to the surprise $-\ln p(o)$; only action, by changing $o$, lowers
#   the surprise itself. Put the goal in the prior and the robot drives until it comes true.
# - Model choice is free-energy comparison: complexity against accuracy, in nats, with no
#   threshold.
# - On Day 4 the robot loses its known pose and its labelled points. The wall becomes four walls,
#   the pose becomes one more belief, and the walls and the pose must be estimated *together*.
