# %% [markdown]
# # Day 1 — The grammar of uncertainty, and a wall as a belief
#
# **Part A (before the break): the three operations by hand.** Everything in this course is built
# from three operations on a joint distribution: **factorise**, **marginalise**, **condition**. You
# will run them on the door table from the slides, then on the Gaussian wall, then use them to see
# why we write the joint in causal order, and why the robot's action depends on the world only
# through what it has seen.
#
# **Part B (after the break): a wall as a belief.** You will
#
# 1. fit a wall to a LiDAR scan as a **Laplace belief**, a Gaussian whose covariance is the inverse
#    curvature of the free energy;
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
# # Part A — the three operations, by hand
#
# ## A1. The door table
#
# $D$: is the door open? $O$: does the LiDAR see a gap where the door is? We store a joint over
# discrete variables as a numpy array with one axis per variable.

# %%
p_D = np.array([0.3, 0.7])                    # [open, closed]
p_O_given_D = np.array([[0.9, 0.1],           # row d: p(O = [gap, wall] | D = d)
                        [0.2, 0.8]])

# %% [markdown]
# **Exercise A1 — factorise, marginalise, condition.** Build the joint `joint[d, o]` with the
# product rule $p(d, o) = p(o \mid d)\,p(d)$. Then compute the marginal $p(O)$ by summing out $D$,
# and the posterior $p(D \mid O = \text{gap})$ by taking the gap column and dividing it by its sum.
# (Check yourself: the joint is $0.27, 0.03, 0.14, 0.56$; $p(O=\text{gap}) = 0.41$;
# $p(\text{open} \mid \text{gap}) \approx 0.66$.)

# %%
# BEGIN SOLUTION
joint = p_O_given_D * p_D[:, None]
p_O = joint.sum(axis=0)
p_D_given_gap = joint[:, 0] / p_O[0]
# END SOLUTION
print("joint:\n", joint, "\nsums to", joint.sum())
print("p(O) =", p_O, "  p(D | gap) =", p_D_given_gap.round(3))

# %% [markdown]
# **Exercise A2 — the other factorisation.** Compute $p(D \mid O)$ for both values of $O$ and
# check that $p(D \mid O)\,p(O)$ rebuilds the same joint. Bayes' rule is nothing but this identity:
# two factorisations of one joint.

# %%
# BEGIN SOLUTION
p_D_given_O = joint / p_O[None, :]            # column o is p(D | O = o)
# END SOLUTION
print("each column of p(D|O) adds to 1:", p_D_given_O.sum(axis=0))
print("rebuilt from p(D|O) p(O):", np.allclose(p_D_given_O * p_O[None, :], joint))

# %% [markdown]
# ## A2. The Gaussian wall
#
# $s$ is the distance to a wall, prior $\mathcal N(2, 0.5^2)$; $o$ is a range reading,
# $o \mid s \sim \mathcal N(s, 0.2^2)$. The joint over $(s, o)$ is Gaussian, with the mean and
# covariance below (built on the slides from the model).
#
# **Exercise A3 — marginalise and condition a Gaussian.**
# 1. Marginal $p(o)$: cross out the row and column of $s$. (Check: $\mathcal N(2, 0.29)$.)
# 2. Conditional $p(s \mid o = 2.6)$ in gain form: $k = \Sigma_{so}/\Sigma_{oo}$,
#    mean $\mu_s + k\,(o - \mu_o)$, variance $\Sigma_{ss} - k\,\Sigma_{os}$. (Check: $k = 0.86$,
#    $2.52 \pm 0.19$ m.)
# 3. Check it without any formula: draw a million samples of $(s, o)$, keep those with $o$ within
#    1 cm of 2.6, and compute the mean and standard deviation of their $s$.

# %%
mu_so = np.array([2.0, 2.0])
Sigma_so = np.array([[0.25, 0.25], [0.25, 0.29]])
o_star = 2.6
# BEGIN SOLUTION
m_o, v_o = mu_so[1], Sigma_so[1, 1]
k = Sigma_so[0, 1] / Sigma_so[1, 1]
m_post = mu_so[0] + k * (o_star - mu_so[1])
v_post = Sigma_so[0, 0] - k * Sigma_so[1, 0]
samples = rng.multivariate_normal(mu_so, Sigma_so, size=1_000_000)
kept = samples[np.abs(samples[:, 1] - o_star) < 0.01, 0]
# END SOLUTION
print(f"p(o) = N({m_o:.2f}, {v_o:.2f})    gain k = {k:.2f}")
print(f"formula: {m_post:.3f} ± {np.sqrt(v_post):.3f}    samples ({len(kept)} kept): {kept.mean():.3f} ± {kept.std():.3f}")

# %% [markdown]
# **Exercise A4 — Bayes on a grid: the three operations in a row.** Without the Gaussian formulas:
# on a grid of $s$ values, (1) **factorise**: multiply the prior $p(s)$ by the sensor model
# $p(o = 2.6 \mid s)$; (2) **marginalise**: the area under that product is $p(o = 2.6)$;
# (3) **condition**: divide the product by the area. Plot all three, and check the area is 0.40
# and the posterior is $2.52 \pm 0.19$ m again.

# %%
def npdf(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


s_grid = np.linspace(0.0, 4.0, 2001)
ds = s_grid[1] - s_grid[0]
# BEGIN SOLUTION
product = npdf(s_grid, 2.0, 0.5 ** 2) * npdf(o_star, s_grid, 0.2 ** 2)
area = product.sum() * ds
posterior = product / area
# END SOLUTION
m_g = (s_grid * posterior).sum() * ds
sd_g = np.sqrt(((s_grid - m_g) ** 2 * posterior).sum() * ds)
print(f"area p(o=2.6) = {area:.3f}    posterior {m_g:.3f} ± {sd_g:.3f}")
fig, axs = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for ax_, y, t in zip(axs, [product, product, posterior], ["1. factorise", "2. marginalise: area", "3. condition: divide"]):
    ax_.fill_between(s_grid, y, alpha=0.4); ax_.set_title(t); ax_.set_xlabel("s (m)")

# %% [markdown]
# **Question.** For one wall this took a grid of 2001 points. How many points would a grid need for
# the robot's pose $(x, y, \theta)$ and four walls of two parameters each, at the same resolution
# per dimension? This is why step 2 is where exact inference dies.

# %% [markdown] solution
# *Answer.* 11 dimensions at 2001 points each: $2001^{11} \approx 2 \times 10^{36}$ grid points.
# The area $p(o)$ cannot be computed by brute force; the afternoon is about not computing it.

# %% [markdown]
# ## A3. Three variables, six orderings, one causal order
#
# Now the names used all week: $S$ the state (is the door open?), $O$ the observation (does the
# LiDAR see a gap?), $A$ the action (does the robot point its LiDAR towards the door, or away?).
# At the first step the robot has seen nothing, so its choice of $A$ cannot depend on $S$; the
# observation depends on both. The causal model is $p(S)\,p(A)\,p(O \mid S, A)$.

# %%
p_S = np.array([0.3, 0.7])                    # [open, closed]
p_A = np.array([0.5, 0.5])                    # [towards the door, away]
p_O_given_SA = np.array([[[0.9, 0.1],         # [s, a, o]: open,   towards -> p(O = [gap, wall])
                          [0.5, 0.5]],        #            open,   away
                         [[0.2, 0.8],         #            closed, towards
                          [0.5, 0.5]]])       #            closed, away


def causal_joint(p_S, p_A, p_O_given_SA):
    j = p_S[:, None, None] * p_A[None, :, None] * p_O_given_SA          # [s, a, o]
    return j.transpose(0, 2, 1)                                          # -> [s, o, a]


joint3 = causal_joint(p_S, p_A, p_O_given_SA)

# %% [markdown]
# **Exercise A5 — count the numbers.** A factor $p(X \mid \text{parents})$ over binary variables
# needs one free number per combination of its parents' values (the other value of $X$ is
# 1 minus it). Write `n_numbers(n_parents)` and count each ordering. (Check: causal
# $1 + 1 + 4 = 6$; anti-causal $p(O)\,p(A \mid O)\,p(S \mid O, A)$: $1 + 2 + 4 = 7$.) Then count
# $N = 20$ binary variables, with every factor keeping all earlier variables, against at most
# $k = 3$ parents each.

# %%
def n_numbers(n_parents):
    # BEGIN SOLUTION
    return 2 ** n_parents
    # END SOLUTION


print("causal     :", n_numbers(0) + n_numbers(0) + n_numbers(2))
print("anti-causal:", n_numbers(0) + n_numbers(1) + n_numbers(2))
print("N = 20, nothing dropped:", sum(n_numbers(i) for i in range(20)), "   at most 3 parents:", 20 * n_numbers(3))

# %% [markdown]
# **Exercise A6 — read the anti-causal factors off the joint.** From `joint3` alone (axes
# `[s, o, a]`), compute the three factors of $p(O)\,p(A \mid O)\,p(S \mid O, A)$, and check that
# their product is `joint3` again. Then:
#
# 1. Does $p(A \mid S)$ depend on $S$? Does $p(S \mid O, A)$ depend on $A$?
# 2. Replace the LiDAR by a worse sensor, which sees a gap with probability 0.6 / 0.4 when it looks
#    at an open / closed door. Which causal factors change? Which anti-causal factors change?

# %%
def anticausal_factors(j):
    # BEGIN SOLUTION
    p_O = j.sum(axis=(0, 2))
    p_OA = j.sum(axis=0)                       # [o, a]
    p_A_given_O = p_OA / p_O[:, None]
    p_S_given_OA = j / p_OA[None, :, :]
    return p_O, p_A_given_O, p_S_given_OA
    # END SOLUTION


p_O3, p_A_given_O, p_S_given_OA = anticausal_factors(joint3)
print("anti-causal product == joint:", np.allclose(p_O3[None, :, None] * p_A_given_O[None] * p_S_given_OA, joint3))
p_SA = joint3.sum(axis=1)
print("p(A=towards | S) for S = open, closed:", (p_SA[:, 0] / p_SA.sum(axis=1)).round(3), " <- S drops out")
print("p(S=open | O=gap, A) for A = towards, away:", p_S_given_OA[0, 0, :].round(3), " <- A does not")

worse = p_O_given_SA.copy()
worse[0, 0], worse[1, 0] = [0.6, 0.4], [0.4, 0.6]
new = anticausal_factors(causal_joint(p_S, p_A, worse))
for name, x, y in zip(["p(O)", "p(A|O)", "p(S|O,A)"], (p_O3, p_A_given_O, p_S_given_OA), new):
    print(f"new sensor changes anti-causal {name:9s}: {not np.allclose(x, y)}")

# %% [markdown] solution
# In the causal order only $p(O \mid S, A)$ changes with the sensor: that is the modularity
# argument. In the anti-causal order all three factors change, and none of them is something
# anyone could write down from physics. Note also *why* $p(S \mid O, A)$ needs $A$: a gap is strong
# evidence for an open door when the LiDAR looked at it (0.66) and none at all when it looked away
# (0.30, the prior).

# %% [markdown]
# ## A4. The robot acts on what it sees, not on the world
#
# Two steps. At step 1 the robot looks at the door and gets $o_1$ (gap or wall). At step 2 it
# chooses $a_2$: it looks at the door again with a probability equal to how unsure it is,
# $1 - |2\,p(\text{open} \mid o_1) - 1|$. The sampler below runs the world forwards in causal
# order.

# %%
def run_two_steps(n):
    s = rng.random(n) < p_S[0]                                     # True = open
    gap = rng.random(n) < np.where(s, 0.9, 0.2)                    # o1, LiDAR towards the door
    belief = np.where(gap, joint[0, 0] / p_O[0], joint[0, 1] / p_O[1])   # p(open | o1), from A1
    look_again = rng.random(n) < 1 - np.abs(2 * belief - 1)        # a2
    return s, gap, look_again


s_, gap_, a2_ = run_two_steps(1_000_000)

# %% [markdown]
# **Exercise A7 — the umbrella test.** From the samples, estimate
# 1. $p(a_2 = \text{look} \mid S)$ for $S$ = open and closed. Is $a_2$ independent of $S$?
# 2. $p(a_2 = \text{look} \mid S, o_1 = \text{gap})$ for $S$ = open and closed. And now?
#
# (You take an umbrella because of the forecast, not because of tomorrow's rain.)

# %%
# BEGIN SOLUTION
by_S = [a2_[s_ == v].mean() for v in (True, False)]
by_S_gap = [a2_[(s_ == v) & gap_].mean() for v in (True, False)]
# END SOLUTION
print("p(look | S = open, closed)            :", np.round(by_S, 3))
print("p(look | S = open, closed, o1 = gap)  :", np.round(by_S_gap, 3))

# %% [markdown] solution
# Without $o_1$, the action depends on the door: 0.62 against 0.22, because an open door tends
# to produce a gap, a gap makes the robot fairly sure, and a sure robot does not look again. Given
# $o_1$, both are 0.68: the door reaches the action only through what the robot has seen.

# %% [markdown]
# # Part B — a wall as a belief (after the break)
#
# Choosing *where to look next* is Day 4, once action has its own functional. Today the robot knows
# its own pose exactly, and we cheat on *which* LiDAR points belong to the wall. Day 3 removes both
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
# ## 2. Is the covariance right?
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
# likelihood made the belief *under*-confident. That costs little. Day 3 has a wrong likelihood
# that makes it badly *over*-confident, which costs a great deal. The real robot's localiser
# weights each LiDAR return by the incidence $|\hat r^\top \hat n|$ for exactly this reason.
#
# From here on, fit walls with `incidence_sigma`.

# %%
fit0 = laplace(residual(pts0), [np.pi / 2, 4.0], incidence_sigma(pts0, bear0), jacobian=jacobian(pts0))
wall0 = fit0.belief

# %% [markdown]
# ## 3. Free energy as the price of a model
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
# ## Take-aways
# - A belief is a mean **and** a covariance. The covariance is the curvature of the free energy,
#   and it is only as right as the likelihood.
# - Model choice is free-energy comparison: complexity against accuracy, in nats, with no
#   threshold.
# - Tomorrow the robot loses its known pose and its labelled points. The wall becomes four walls,
#   the pose becomes one more belief, and the walls and the pose must be estimated *together*.
