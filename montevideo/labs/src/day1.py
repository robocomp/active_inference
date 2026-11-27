# %% [markdown]
# # Day 1 — The grammar of uncertainty
#
# Everything in this course is built from three operations on a joint distribution: **factorise**,
# **marginalise**, **condition**. You will run them on the door table from the slides, then on the
# Gaussian wall, then use them to see why we write the joint in causal order, and why the robot's
# action depends on the world only through what it has seen. Last, you run them over time: the
# Kalman filter, and what happens when the model leaves something out.

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
# ## 1. The door table
#
# $S$: is the door open (the state)? $O$: does the LiDAR see a gap where the door is? We store a joint over
# discrete variables as a numpy array with one axis per variable.

# %%
p_S = np.array([0.3, 0.7])                    # [open, closed]
p_O_given_S = np.array([[0.9, 0.1],           # row s: p(O = [gap, wall] | S = s)
                        [0.2, 0.8]])

# %% [markdown]
# **Exercise 1 — factorise, marginalise, condition.** Build the joint `joint[s, o]` with the
# product rule $p(s, o) = p(o \mid s)\,p(s)$. Then compute the marginal $p(O)$ by summing out $S$,
# and the posterior $p(S \mid O = \text{gap})$ by taking the gap column and dividing it by its sum.
# (Check yourself: the joint is $0.27, 0.03, 0.14, 0.56$; $p(O=\text{gap}) = 0.41$;
# $p(\text{open} \mid \text{gap}) \approx 0.66$.)

# %%
# BEGIN SOLUTION
joint = p_O_given_S * p_S[:, None]
p_O = joint.sum(axis=0)
p_S_given_gap = joint[:, 0] / p_O[0]
# END SOLUTION
print("joint:\n", joint, "\nsums to", joint.sum())
print("p(O) =", p_O, "  p(S | gap) =", p_S_given_gap.round(3))

# %% [markdown]
# **Exercise 2 — the other factorisation (homework).** Compute $p(S \mid O)$ for both values of $O$ and
# check that $p(S \mid O)\,p(O)$ rebuilds the same joint. Bayes' rule is nothing but this identity:
# two factorisations of one joint.

# %%
# BEGIN SOLUTION
p_S_given_O = joint / p_O[None, :]            # column o is p(S | O = o)
# END SOLUTION
print("each column of p(S|O) adds to 1:", p_S_given_O.sum(axis=0))
print("rebuilt from p(S|O) p(O):", np.allclose(p_S_given_O * p_O[None, :], joint))

# %% [markdown]
# ## 2. The Gaussian wall
#
# $s$ is the distance to a wall, prior $\mathcal N(2, 0.5^2)$; $o$ is a range reading,
# $o \mid s \sim \mathcal N(s, 0.2^2)$. The joint over $(s, o)$ is Gaussian, with the mean and
# covariance below (built on the slides from the model).
#
# **Exercise 3 — marginalise and condition a Gaussian.**
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
# **Exercise 4 — Bayes on a grid: the three operations in a row.** Without the Gaussian formulas:
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
# The area $p(o)$ cannot be computed by brute force; Day 2 is about not computing it.

# %% [markdown]
# ## 3. Three variables, six orderings, one causal order
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
# **Exercise 5 — count the numbers.** A factor $p(X \mid \text{parents})$ over binary variables
# needs one free number per combination of its parents' values (the other value of $X$ is
# 1 minus it). Write `n_numbers(n_parents)` and count each ordering. (Check: causal
# $1 + 1 + 4 = 6$; anti-causal $p(O)\,p(A \mid O)\,p(S \mid O, A)$: $1 + 2 + 4 = 7$.) At home, count
# $N = 20$ binary variables, with every factor keeping all earlier variables, against at most
# $k = 3$ parents each.

# %%
def n_numbers(n_parents):
    # BEGIN SOLUTION
    return 2 ** n_parents
    # END SOLUTION


print("causal     :", n_numbers(0) + n_numbers(0) + n_numbers(2))
print("anti-causal:", n_numbers(0) + n_numbers(1) + n_numbers(2))
print("N = 20, nothing dropped:", sum(n_numbers(i) for i in range(20)), "   at most 3 parents (an upper bound):", 20 * n_numbers(3))

# %% [markdown]
# **Exercise 6 — read the anti-causal factors off the joint.** From `joint3` alone (axes
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
# ## 4. The robot acts on what it sees, not on the world
#
# Two steps. At step 1 the robot looks at the door and gets $o_1$ (gap or wall). At step 2 it
# chooses $a_2$: it looks at the door again with a probability equal to how unsure it is,
# $1 - |2\,p(\text{open} \mid o_1) - 1|$. The sampler below runs the world forwards in causal
# order.

# %%
def run_two_steps(n):
    s = rng.random(n) < p_S[0]                                     # True = open
    gap = rng.random(n) < np.where(s, 0.9, 0.2)                    # o1, LiDAR towards the door
    belief = np.where(gap, joint[0, 0] / p_O[0], joint[0, 1] / p_O[1])   # p(open | o1), from Exercise 1
    look_again = rng.random(n) < 1 - np.abs(2 * belief - 1)        # a2
    return s, gap, look_again


s_, gap_, a2_ = run_two_steps(1_000_000)

# %% [markdown]
# **Exercise 7 — the action depends on what was seen.** From the samples, estimate
# 1. $p(a_2 = \text{look} \mid S)$ for $S$ = open and closed. Is $a_2$ independent of $S$?
# 2. $p(a_2 = \text{look} \mid S, o_1 = \text{gap})$ for $S$ = open and closed. And now?

# %%
# BEGIN SOLUTION
by_S = [a2_[s_ == v].mean() for v in (True, False)]
by_S_gap = [a2_[(s_ == v) & gap_].mean() for v in (True, False)]
# END SOLUTION
print("p(look | S = open, closed)            :", np.round(by_S, 3))
print("p(look | S = open, closed, o1 = gap)  :", np.round(by_S_gap, 3))

# %% [markdown] solution
# Without $o_1$, the action depends on the door: 0.62 against 0.22, because an open door tends
# to produce a gap, a gap leaves the robot unsure ($p(\text{open} \mid \text{gap}) = 0.66$), and an unsure
# robot looks again; a closed door tends to produce a wall, which leaves it sure (0.05). Given
# $o_1$, both are 0.68: the door reaches the action only through what the robot has seen.

# %% [markdown]
# ## 5. Time: the Kalman filter
#
# The wall of the slides: $s$ is the distance to the wall. Each step the robot drives $a$ metres
# towards it, then reads the LiDAR. The model (frame "Gaussian and linear"):
# $s_t = s_{t-1} - a_t + w$, $w \sim \mathcal N(0, \sigma_w^2)$, and $o_t = s_t + v$, $v \sim \mathcal N(0, \sigma_v^2)$.
#
# **Exercise 8 — one step.** Write `kalman_step`: predict with the motion model, then update with
# the reading. Reproduce the slide "The effect on the belief": from $2.00 \pm 0.50$, drive 0.5 m
# with $\sigma_w = 0.2$, read $1.3$ with $\sigma_v = 0.2$. You should get $1.50 \pm 0.54$, then $1.32 \pm 0.19$.

# %%
def kalman_step(mu, var, a, o, var_w, var_v):
    """One predict + update. Returns (predicted mean, predicted var, mean, var)."""
    # BEGIN SOLUTION
    mu_p, var_p = mu - a, var + var_w              # predict: move, and widen
    k = var_p / (var_p + var_v)                    # the gain
    mu_u, var_u = mu_p + k * (o - mu_p), (1 - k) * var_p   # update: pull to the reading, narrow
    # END SOLUTION
    return mu_p, var_p, mu_u, var_u


mu_p, var_p, mu_u, var_u = kalman_step(2.0, 0.5**2, 0.5, 1.3, 0.2**2, 0.2**2)
print(f"predicted {mu_p:.2f} ± {np.sqrt(var_p):.2f}   after the reading {mu_u:.2f} ± {np.sqrt(var_u):.2f}")

# %% [markdown]
# **Exercise 9 — a loop, and a wheel bias (homework).** The robot starts 5 m from the wall, believing
# $4.8 \pm 0.5$, and drives 0.3 m per step for 12 steps. Its wheels err by $\sigma_w = 0.02$ m; the
# LiDAR by $\sigma_v = 0.2$ m. Run the filter and plot the true distance, the mean and a $\pm 2\sigma$ band.
# Then make the wheels drive 30% further than commanded (`bias = 1.3`), **without telling the
# model**. Is the belief still inside its band? How sure is it?

# %%
def run(bias, steps=12, a=0.3, var_w=0.02**2, var_v=0.2**2, seed=1):
    g = np.random.default_rng(seed)
    s, mu, var = 5.0, 4.8, 0.5**2
    S, M, V = [s], [mu], [var]
    for _ in range(steps):
        # BEGIN SOLUTION
        s = s - bias * a + g.normal(0, np.sqrt(var_w))   # the world: the wheels may be biased
        o = s + g.normal(0, np.sqrt(var_v))              # the reading
        _, _, mu, var = kalman_step(mu, var, a, o, var_w, var_v)   # the model: no bias
        # END SOLUTION
        S.append(s); M.append(mu); V.append(var)
    return np.array(S), np.array(M), np.array(V)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, bias in zip(axes, (1.0, 1.3)):
    S_, M_, V_ = run(bias)
    t = np.arange(len(S_))
    ax.plot(t, S_, "k-", label="true distance")
    ax.plot(t, M_, color="C3", label="belief mean")
    ax.fill_between(t, M_ - 2 * np.sqrt(V_), M_ + 2 * np.sqrt(V_), color="C3", alpha=0.2, label="±2σ")
    ax.set_title(f"bias = {bias}"); ax.set_xlabel("step")
    print(f"bias {bias}: final error {M_[-1] - S_[-1]:+.2f} m, belief σ {np.sqrt(V_[-1]):.2f} m, "
          f"error / σ = {(M_[-1] - S_[-1]) / np.sqrt(V_[-1]):+.1f}")
axes[0].set_ylabel("distance to the wall (m)"); axes[0].legend()

# %% [markdown] solution
# Without the bias, the truth stays inside the band, and the belief's σ narrows to about 0.07 m: the
# precise wheels carry the belief between noisy readings. With the bias, the belief stays just as
# narrow but lags the truth by several σ: the robot is **sure and wrong**. The readings pull it
# back each step, but the model trusts its wheels (σ_w = 0.02) far more than the LiDAR (σ_v = 0.2),
# so the pull is weak. The cure is not a threshold: put the bias in the state $s$, as a variable to
# estimate (Markov's warning on the slides; Day 4).
