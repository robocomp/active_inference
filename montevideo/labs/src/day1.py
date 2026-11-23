# %% [markdown]
# # Day 1 — The grammar of uncertainty
#
# Everything in this course is built from three operations on a joint distribution: **factorise**,
# **marginalise**, **condition**. You will run them on the door table from the slides, then on the
# Gaussian wall, then use them to see why we write the joint in causal order, and why the robot's
# action depends on the world only through what it has seen.

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
# # The three operations, by hand
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
