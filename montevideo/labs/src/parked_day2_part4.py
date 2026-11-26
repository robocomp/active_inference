# Day 2 lab part 4 (perception and action), taken out of Day 2 on 2026-11-26 for the Day 4 lab.
# Not built by build_notebooks.py (the name does not start with 'day'). Needs F_wall's setup only.

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 4. Perception and action lower the same $F$ (25 min)
#
# The robot of frames 27--30. $s$: its distance to the wall. We write the likelihood
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

