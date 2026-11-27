# %% [markdown]
# # Day 2 — A wall as a belief
#
# You will
#
# 1. derive, by hand, the free energy, its Laplace form and the split of $G$, and check each
#    numerically (section 1; in class run the checks, the derivations of Exercises 1 and 3 are homework);
# 2. walk the belief to the bottom of $F$ by gradient descent: perception, step by step (section 2; 10 min);
# 3. meet the world (section 3; 5 min);
# 4. fit a wall's distance to a LiDAR scan as a **Laplace belief**, a Gaussian whose variance is the
#    inverse curvature of the free energy (section 4; 25 min);
# 5. check whether that spread is right, and fix it when it is not (section 5; homework);
# 6. use the **free energy** to choose between two models of the same wall (section 6; homework).
#
# In class: Exercises 2, 4, 5 and 6, and the checks of 1 and 3, about 60 minutes.
# Lowering $F$ by action is Day 4's lab.

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
# ## 1. By hand: three derivations (20 min in class: the checks)
#
# The slides show the results; here you derive them yourself, with pen and paper, then check each
# one numerically. Use the wall of the springs frame: prior $p(s) = \mathcal N(m, \sigma_p^2)$ with
# $m = 2.0$ m and $\sigma_p = 0.5$ m, and a reading $o = 2.6$ m with noise $\sigma_o = 0.2$ m, so
# $p(o \mid s) = \mathcal N(o; s, \sigma_o^2)$.

# %%
m, sigma_p = 2.0, 0.5          # prior
o, sigma_o = 2.6, 0.2          # reading and its noise
s = np.linspace(-1.0, 5.0, 6001); ds = s[1] - s[0]


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)


joint = gauss(o, s, sigma_o) * gauss(s, m, sigma_p)      # p(o | s) p(s), on the grid
p_o = joint.sum() * ds                                   # the evidence, by brute force
posterior = joint / p_o
print(f"p(o) = {p_o:.3f}   surprise -ln p(o) = {-np.log(p_o):.3f} nats")

# %% [markdown]
# **Exercise 1 — the free energy.** Start from the definition of the KL (frame "Kullback--Leibler")
# and Bayes in logs, $\ln p(s\mid o) = \ln[p(o\mid s)\,p(s)] - \ln p(o)$. Show that
# $$\mathrm{KL}[q \,\|\, p(s\mid o)] = F[q,o] + \ln p(o), \qquad F[q,o] = \int q(s)\ln\frac{q(s)}{p(o\mid s)\,p(s)}\,ds .$$
# Then explain in one line why $F \ge -\ln p(o)$, and when the two are equal.
#
# *Hints*: substitute Bayes inside the log; split the integral in three; $\ln p(o)$ has no $s$;
# $\int q = 1$.

# %% [markdown] solution
# **Solution 1.**
# $\mathrm{KL} = \int q\,[\ln q - \ln p(s\mid o)]\,ds
# = \int q\,[\ln q - \ln p(o\mid s)p(s) + \ln p(o)]\,ds
# = \int q \ln\frac{q}{p(o\mid s)p(s)}\,ds + \ln p(o)\int q\,ds = F + \ln p(o)$.
# Since $\mathrm{KL} \ge 0$, $F \ge -\ln p(o)$, with equality exactly when $q$ is the posterior.

# %% [markdown]
# Check it: write `free_energy(mu, sd)` for $q = \mathcal N(\mu, \text{sd}^2)$ on the grid, and the KL to
# the grid posterior. For any $q$, $F - \mathrm{KL}$ should be the surprise.

# %%
def free_energy(mu, sd):
    q = gauss(s, mu, sd)
    # BEGIN SOLUTION
    ok = q > 1e-300
    return np.sum(q[ok] * (np.log(q[ok]) - np.log(joint[ok]))) * ds
    # END SOLUTION


def kl_to_posterior(mu, sd):
    q = gauss(s, mu, sd)
    # BEGIN SOLUTION
    ok = q > 1e-300
    return np.sum(q[ok] * (np.log(q[ok]) - np.log(posterior[ok]))) * ds
    # END SOLUTION


for mu, sd in [(2.0, 0.5), (2.6, 0.1), (2.52, 0.19)]:
    F, K = free_energy(mu, sd), kl_to_posterior(mu, sd)
    print(f"q = N({mu}, {sd}^2):  F = {F:6.3f}   KL = {K:6.3f}   F - KL = {F - K:.3f}")

# %% [markdown]
# **Exercise 2 — $F$ under Laplace, and $\sigma^2 = 1/H$.** Take $q = \mathcal N(\mu, \sigma^2)$ and the
# parabola of the bowl around $\mu$,
# $-\ln p(o,s) \approx -\ln p(o,\mu) + b\,(s-\mu) + \tfrac12 H (s-\mu)^2$. Show that
# $$F(\mu,\sigma) \approx -\ln p(o,\mu) + \tfrac12 H\sigma^2 - \tfrac12\ln(2\pi e\,\sigma^2),$$
# then set $\partial F/\partial\sigma = 0$ and find $\sigma^2$. Finally, for this wall compute $H$ by
# differentiating $-\ln p(o,s)$ twice in $s$ (it is two parabolas, the two springs).
#
# *Hints*: $\int q\ln q = -\tfrac12\ln(2\pi e\sigma^2)$ for a Gaussian;
# $\mathbb E_q[s-\mu] = 0$; $\mathbb E_q[(s-\mu)^2] = \sigma^2$.

# %% [markdown] solution
# **Solution 2.** $F = \int q\ln q + \mathbb E_q[-\ln p(o,s)]
# \approx -\tfrac12\ln(2\pi e\sigma^2) - \ln p(o,\mu) + b\cdot 0 + \tfrac12 H\sigma^2$.
# $\partial F/\partial\sigma = H\sigma - 1/\sigma = 0 \Rightarrow \sigma^2 = 1/H$.
# Here $-\ln p(o,s) = \frac{(o-s)^2}{2\sigma_o^2} + \frac{(s-m)^2}{2\sigma_p^2} + \text{const}$, so
# $H = 1/\sigma_o^2 + 1/\sigma_p^2 = 25 + 4 = 29$ and $\sigma = 1/\sqrt{29} \approx 0.186$ m: precisions add.

# %% [markdown]
# Check it: scan $F$ over a grid of $(\mu, \sigma)$ with your `free_energy`, and compare the minimum
# with $\sigma = 1/\sqrt H$. Is $F$ at the minimum equal to the surprise? Why, for this wall?

# %%
H = 1 / sigma_o**2 + 1 / sigma_p**2
mus, sds = np.linspace(2.3, 2.7, 81), np.linspace(0.10, 0.35, 51)
# BEGIN SOLUTION
Fgrid = np.array([[free_energy(mu, sd) for sd in sds] for mu in mus])
i, j = np.unravel_index(Fgrid.argmin(), Fgrid.shape)
# END SOLUTION
print(f"minimum at mu = {mus[i]:.3f}, sigma = {sds[j]:.3f};  1/sqrt(H) = {1/np.sqrt(H):.3f}")
print(f"F at the minimum = {Fgrid[i, j]:.3f}   surprise = {-np.log(p_o):.3f}")

# %% [markdown] solution
# The sensor is linear and both factors are Gaussian, so the posterior is itself Gaussian: the
# best $q$ is exactly the posterior, the KL is zero, and $F$ touches the surprise (0.92 nats).

# %% [markdown]
# **Exercise 3 — splitting the expected free energy.** Before acting, the robot scores an action
# $a$ by
# $$G(a) = \mathbb E_{q(o,s\mid a)}\big[\ln q(s\mid a) - \ln \tilde p(o,s)\big],
# \qquad q(o,s\mid a) = q(s\mid a)\,p(o\mid s),$$
# where $q(s\mid a)$ is the belief after the action and before any reading, and $\tilde p(o,s)$ is the
# model with the goal $\tilde p(o)$ as the prior over readings. Using
# $\tilde p(o,s) \approx q(s\mid o,a)\,\tilde p(o)$, with $q(s\mid o,a)$ the belief after reading $o$,
# show that
# $$G(a) = -\,\mathbb E_{q(o\mid a)}\Big[\mathrm{KL}\big[q(s\mid o,a)\,\|\,q(s\mid a)\big]\Big] - \mathbb E_{q(o\mid a)}\big[\ln\tilde p(o)\big].$$
#
# *Hints*: the log of a product is a sum; regroup the two beliefs; an expectation over
# $q(o,s\mid a)$ is an expectation over $o$ of an expectation over $s$ given $o$.

# %% [markdown] solution
# **Solution 3.** $\ln q(s\mid a) - \ln\tilde p(o,s) \approx \ln q(s\mid a) - \ln q(s\mid o,a) - \ln\tilde p(o)
# = -[\ln q(s\mid o,a) - \ln q(s\mid a)] - \ln\tilde p(o)$. Averaging the bracket over $s$ given $o$
# (that is, under $q(s\mid o,a)$) gives the KL; averaging over $o$ gives the outer expectation.
# The minus sign of the learning term comes from using the belief *before* the reading,
# $q(s\mid a)$, in the definition of $G$: a choice (Millidge, Tschantz & Buckley, 2021).

# %% [markdown]
# Check the learning term the way the robot computes it (slide "How the robot computes $G$"):
# imagine many readings, update as if each were real (a Kalman step), and average how much the
# belief would learn. For a Gaussian belief the answer is $\tfrac12\ln(\sigma^2_\text{before}/\sigma^2_\text{after})$.
# Compare a good sensor ($\sigma_o = 0.1$) with a poor one ($\sigma_o = 1.0$).

# %%
def kl_gauss(mu1, sd1, mu0, sd0):
    """KL[N(mu1, sd1^2) || N(mu0, sd0^2)]"""
    return np.log(sd0 / sd1) + (sd1**2 + (mu1 - mu0) ** 2) / (2 * sd0**2) - 0.5


def learning_term(mu_b, sd_b, sigma_obs, n=20000):
    # BEGIN SOLUTION
    s_img = rng.normal(mu_b, sd_b, n)                    # 1 imagine the state
    o_img = rng.normal(s_img, sigma_obs)                  # 2 imagine a reading
    k = sd_b**2 / (sd_b**2 + sigma_obs**2)                # 3 update as if it were real
    mu_a, sd_a = mu_b + k * (o_img - mu_b), np.sqrt((1 - k) * sd_b**2)
    return kl_gauss(mu_a, sd_a, mu_b, sd_b).mean()        # 4 score, and average
    # END SOLUTION


for sigma_obs in (0.1, 1.0):
    sd_after = np.sqrt(1 / (1 / sigma_p**2 + 1 / sigma_obs**2))
    print(f"sensor {sigma_obs}:  imagined average = {learning_term(m, sigma_p, sigma_obs):.3f}"
          f"   formula = {0.5 * np.log(sigma_p**2 / sd_after**2):.3f} nats")

# %% [markdown]
# ## 2. Perception, one step at a time (10 min)
#
# Same wall. Without the terms in $\sigma$ (slide "Perception: the belief follows the
# precision-weighted prediction error"), with $h(s) = s$:
# $$F(\mu) = \tfrac12\,\pi_o\,(o - \mu)^2 + \tfrac12\,\pi_p\,(\mu - m)^2, \qquad
# \frac{\partial F}{\partial\mu} = -\pi_o\,(o - \mu) + \pi_p\,(\mu - m),$$
# with $\pi_o = 1/\sigma_o^2 = 25$ and $\pi_p = 1/\sigma_p^2 = 4$.
#
# **Exercise 4 — gradient descent on the belief.** Start at the prior, $\mu = m$, and repeat
# $\mu \leftarrow \mu - \eta\,\partial F/\partial\mu$ with $\eta = 0.01$ for 100 steps. Plot $\mu$
# against the step. Where does it stop? Compare with the Kalman update
# $(\pi_p m + \pi_o o)/(\pi_p + \pi_o)$, and with the minimum you found in Exercise 2.
# Then try $\eta = 0.06$ and $\eta = 0.08$. What happens, and why? ($\partial^2F/\partial\mu^2 = \pi_p + \pi_o$.)

# %%
pi_o, pi_p = 1 / sigma_o**2, 1 / sigma_p**2


def descend(eta, steps=100):
    mu, path = m, [m]
    # BEGIN SOLUTION
    for _ in range(steps):
        mu -= eta * (-pi_o * (o - mu) + pi_p * (mu - m))
        path.append(mu)
    # END SOLUTION
    return np.array(path)


fig, ax = plt.subplots(figsize=(7, 3))
for eta in (0.01, 0.06, 0.08):
    path = descend(eta)
    ax.plot(path[:30], marker=".", label=f"η = {eta}")
    print(f"eta = {eta}: mu after 100 steps = {path[-1]:.4g}")
kalman_mu = (pi_p * m + pi_o * o) / (pi_p + pi_o)
ax.axhline(kalman_mu, ls="--", color="k", lw=0.8, label="Kalman")
ax.set_ylim(1.5, 3.5); ax.set_xlabel("step"); ax.set_ylabel(r"$\mu$ (m)"); ax.legend()
print(f"Kalman: {kalman_mu:.4f}")

# %% [markdown] solution
# With $\eta = 0.01$, $\mu$ walks from 2.0 to 2.517, the Kalman mean and the minimum of Exercise 2: the
# Kalman update reaches the bottom of $F$ in one step, gradient descent in many. With $\eta = 0.06$ it
# overshoots and oscillates but still settles. With $\eta = 0.08$ it diverges: each step
# multiplies the error by $1 - \eta(\pi_p + \pi_o)$, which is $-1.32$, larger than 1 in size. The step
# size must stay below $2/(\pi_p + \pi_o) \approx 0.069$: the sharper the bowl, the smaller the step.

# %% [markdown]
# # A wall as a belief
#
# Action and choosing *where to look next* come on Day 4.
# Today the robot knows its own pose exactly, and we cheat on *which* LiDAR points belong to the
# wall. Day 4 removes both cheats.

# %% [markdown]
# ## 3. Meet the world (5 min)
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
# ## 4. A wall as a Laplace belief (25 min)
#
# The north wall runs along $x$, so one number describes it: its distance $d$ from the origin,
# the line $y = d$. (Rooms whose walls meet at right angles: the angle is known. Section 6 frees it.)
# Each point $i$ on the wall gives a reading $y_i = d + \text{noise}$, with noise $\sigma = 0.02$ m.
# The prior is $d \sim \mathcal N(m, \sigma_p^2)$ with $m = 3$ m, $\sigma_p = 2$ m: "somewhere around 3 m".
#
# The free energy, without the terms in $\sigma$, is the same two-spring sum as on the slides,
# with one spring per point:
# $$F(d) = \sum_i \frac{(y_i - d)^2}{2\sigma^2} + \frac{(d - m)^2}{2\sigma_p^2}.$$
#
# **Exercise 5 — the Laplace belief, numerically.** Write `F_wall(d)`. Evaluate it on a grid of
# $d$, take the bottom as the mean $\mu$, measure the curvature $H$ there with a finite
# difference, $H \approx [F(\mu+h) - 2F(\mu) + F(\mu-h)]/h^2$, and set $\sigma_d = 1/\sqrt{H}$.

# %%
y0 = pts0[:, 1]
m_d, sigma_pd, sigma = 3.0, 2.0, 0.02


def F_wall(d, y=y0, sig=sigma):
    # BEGIN SOLUTION
    return np.sum((y - d) ** 2 / (2 * sig**2)) + (d - m_d) ** 2 / (2 * sigma_pd**2)
    # END SOLUTION


grid = np.linspace(3.9, 4.1, 20001)
# BEGIN SOLUTION
Fs = np.array([F_wall(d) for d in grid])
mu_d = grid[Fs.argmin()]
h = 1e-3
H_d = (F_wall(mu_d + h) - 2 * F_wall(mu_d) + F_wall(mu_d - h)) / h**2
# END SOLUTION
print(f"{len(y0)} points:  mu = {mu_d:.4f} m,  H = {H_d:.0f},  sigma_d = 1/sqrt(H) = {1 / np.sqrt(H_d) * 1000:.2f} mm")

# %% [markdown]
# **Exercise 6 — the same belief, by formula.** $F$ is a sum of parabolas, so its bottom and its
# curvature have a closed form: the precisions add, and the mean is the precision-weighted average
# (the Kalman update of the slides, with one reading per point):
# $$H = \frac{1}{\sigma_p^2} + \sum_i \frac{1}{\sigma_i^2}, \qquad
#   \mu = \frac{1}{H}\Big(\frac{m}{\sigma_p^2} + \sum_i \frac{y_i}{\sigma_i^2}\Big).$$
# Write `fit_d(y, sig)` (with `sig` a number or one value per point) and check it against
# Exercise 5. Then draw 30 walls sampled from the belief. Finally: fit only the first 16 points.
# By how much does $\sigma_d$ grow, and why?

# %%
def fit_d(y, sig):
    """The Laplace belief over d: returns (mu, sigma_d)."""
    pi = 1 / np.broadcast_to(np.asarray(sig, float), y.shape) ** 2
    # BEGIN SOLUTION
    H = 1 / sigma_pd**2 + pi.sum()
    mu = (m_d / sigma_pd**2 + np.sum(pi * y)) / H
    # END SOLUTION
    return mu, 1 / np.sqrt(H)


mu0, sd0 = fit_d(y0, sigma)
mu16, sd16 = fit_d(y0[:16], sigma)
print(f"formula, all points: mu = {mu0:.4f} m, sigma_d = {sd0 * 1000:.2f} mm")
print(f"first 16 points:     mu = {mu16:.4f} m, sigma_d = {sd16 * 1000:.2f} mm  (x{sd16 / sd0:.2f})")

fig, ax = plt.subplots(figsize=(9, 3))
draw.world(world, ax, things=False)
for d in rng.normal(mu0, 10 * sd0, 30):                 # x10 in sd, so you can see it
    ax.axhline(d, color="royalblue", lw=0.6, alpha=0.4)
ax.plot(pts0[:, 0], pts0[:, 1], ".", color="crimson", ms=3)
ax.set_xlim(-0.5, 6.5); ax.set_ylim(3.0, 4.6); ax.set_title("belief samples (sd ×10)")

# %% [markdown] solution
# Both ways give the same $\mu$ and $\sigma_d \approx 2.5$ mm: about $0.02/\sqrt{N}$, because every
# point adds the same precision $1/\sigma^2$ and the prior adds almost nothing. With 16 points of 65,
# $\sigma_d$ grows by about $\sqrt{65/16} \approx 2$: four times fewer readings give a quarter of the
# precision, so twice the spread. The belief is a mean **and** a spread, and the spread is the
# curvature of $F$.

# %% [markdown]
# ## 5. Is the spread right? (homework; backup slide)
#
# $\sigma_d$ is a **claim** about how far the fit would wander if you scanned again. Test the
# claim: scan 400 times from the same pose, fit each scan, and compare the spread of the fits with
# the predicted $\sigma_d$.
#
# **Exercise 7.** Fill in the loop.

# %%
def monte_carlo(sigma_fn, n=400):
    fits, sds = [], []
    for _ in range(n):
        s = lidar.scan(world, pose0, rng)
        p, b = wall_points(s)
        # BEGIN SOLUTION
        mu, sd = fit_d(p[:, 1], sigma_fn(p, b))
        fits.append(mu)
        sds.append(sd)
        # END SOLUTION
    return np.std(fits), np.mean(sds)


emp, pred = monte_carlo(lambda p, b: 0.02)
print(f"uniform sigma:   empirical sd {emp * 1000:.2f} mm   predicted {pred * 1000:.2f} mm   ratio {pred / emp:.3f}")

# %% [markdown]
# The prediction is a few per cent **too pessimistic**. That is not Monte Carlo noise; re-run it
# and the gap stays. Something in the likelihood is wrong.
#
# **Exercise 8.** The LiDAR's noise is along the **beam**, but the residual measures distance along
# the wall's **normal**. A range error $\epsilon$ on a beam with direction $u$ moves the point by
# $\epsilon\,(u^\top n)$ off the wall. So the reading's standard deviation is
# $\sigma_i = \sigma\,|u_i^\top n|$, not $\sigma$. Write that per-point sigma and re-run the test.

# %%
def incidence_sigma(p, bearings, alpha=np.pi / 2):
    # BEGIN SOLUTION
    u = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)
    n = np.array([np.cos(alpha), np.sin(alpha)])
    return 0.02 * np.abs(u @ n)
    # END SOLUTION


emp, pred = monte_carlo(incidence_sigma)
print(f"incidence sigma: empirical sd {emp * 1000:.2f} mm   predicted {pred * 1000:.2f} mm   ratio {pred / emp:.3f}")

# %% [markdown]
# **Lesson.** A Laplace covariance is exactly as right as the likelihood you wrote. Here the wrong
# likelihood made the belief *under*-confident. That costs little. Day 4 has a wrong likelihood
# that makes it badly *over*-confident, which costs a great deal. The real robot's localiser
# weights each LiDAR return by the incidence $|\hat r^\top \hat n|$ for exactly this reason.
#
# From here on, fit walls with `incidence_sigma`.

# %% [markdown]
# ## 6. Free energy as the price of a model (homework; backup slide)
#
# Rooms are mostly rectilinear. A **Manhattan** model of this wall fixes $\alpha = 90°$ and has a
# single parameter, $d$. The **free** model keeps both parameters, with a prior
# $\alpha \sim \mathcal N(90°, s^2)$. Both models fit the data equally well, so which does the free
# energy prefer, and by how much?
#
# The Laplace free energy approximates the surprise $-\ln p(o \mid \text{model})$. Lower is better,
# and the difference is a log Bayes factor in nats.
#
# With the angle free, the wall is a line with unit normal $n(\alpha) = (\cos\alpha, \sin\alpha)$ at
# distance $d$: the residual of point $i$ is $r_i(\alpha, d) = \cos\alpha\,x_i + \sin\alpha\,y_i - d$.
# With two numbers, the bowl of $F$ has a curvature in each direction and between them (a $2\times2$
# matrix $H$; Day 4). `laplace` finds its bottom and its curvature for you; the cell below gives it
# the residual and its slopes.

# %%
def residual(pts):
    return lambda th: np.cos(th[0]) * pts[:, 0] + np.sin(th[0]) * pts[:, 1] - th[1]


def jacobian(pts):
    return lambda th: np.stack([-np.sin(th[0]) * pts[:, 0] + np.cos(th[0]) * pts[:, 1], -np.ones(len(pts))], axis=1)

# %% [markdown]
# **Exercise 9.** Compute $\Delta F = F_\text{free} - F_\text{Manhattan}$ for prior widths
# $w$ from 0.1° to 30°. Use the prior $d \sim \mathcal N(3, 2^2)$ in both models.

# %%
sig0 = incidence_sigma(pts0, bear0)
manhattan = laplace(lambda th: residual(pts0)(np.array([np.pi / 2, th[0]])), [3.0], sig0,
                    prior=Gaussian([3.0], [[4.0]]))
widths = np.deg2rad(np.logspace(-1, np.log10(30), 25))
dF = []
for w in widths:
    # BEGIN SOLUTION
    free = laplace(residual(pts0), [np.pi / 2, 3.0], sig0, jacobian=jacobian(pts0),
                   prior=Gaussian([np.pi / 2, 3.0], np.diag([w ** 2, 4.0])))
    dF.append(free.free_energy - manhattan.free_energy)
    # END SOLUTION

plt.figure(figsize=(6, 3.2))
plt.semilogx(np.degrees(widths), dF, "o-")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("prior sd on the wall angle, w (deg)"); plt.ylabel(r"$F_{free} - F_{Manhattan}$ (nats)")
plt.title("the complexity price of a free parameter")

# %% [markdown]
# Above zero, the Manhattan model wins. The penalty grows like $\ln(w/\sigma_\alpha)$, with $\sigma_\alpha$ the spread of the fitted angle: it is the
# complexity term, $\mathrm{KL}[q\,\|\,p]$, charging for how much the prior had to be squeezed to
# reach the posterior.
#
# **Exercise 10.** Is the Manhattan model therefore always right? Rotate the points by 1.5° about
# their centroid, which is what a slightly skew wall would look like, and compare again at
# $w = 5°$.

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
# ## Take-aways
# - A belief is a mean **and** a covariance. The covariance is the curvature of the free energy,
#   and it is only as right as the likelihood.
# - Model choice is free-energy comparison: complexity against accuracy, in nats, with no
#   threshold.
# - On Day 4 the robot loses its known pose and its labelled points. The wall becomes four walls,
#   the pose becomes one more belief, and the walls and the pose must be estimated *together*.
