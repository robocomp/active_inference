# %% [markdown]
# # Day 3 — Acting to know: the body and the room
#
# **The challenge, stage 2 (second half) and the end of stage 1.** The robot can now find itself in
# a room. Two things are still missing, and today they turn out to be the same problem.
#
# - **A. The body is a belief too.** The toy robot's wheels and gyro are not what its designers
#   wrote down. Learn the error from the corrections the room supplies, and find out which motions
#   teach which part of the body.
# - **B. Where to look next.** The robot's map of the room still has holes. Choose viewpoints by the
#   information a look is expected to buy, stop when nothing worth buying is left, and meet the
#   trap the thesis fell into.
#
# The thread through both parts: **what the robot can learn depends on what it does.** It is a
# property of the parameter and the action together.

# %%
import sys
from pathlib import Path

LABS = Path.cwd() if (Path.cwd() / "toyworld").exists() else Path.cwd().parent
sys.path.insert(0, str(LABS))

import matplotlib.pyplot as plt
import numpy as np

from toyworld import Gaussian, Grid, Lidar, Robot, draw, laplace, se2, two_room_apartment
from toyworld.beliefs import binary_entropy, linear_gaussian_update, numeric_jacobian, sigmoid
from toyworld.geometry import distance_to_polygon, polygon_area, segments_of
from toyworld.grid import expected_gain
from toyworld.robot import Body, hidden_body, integrate

world = two_room_apartment()
lidar = Lidar(n_beams=360, sigma=0.02, max_range=12.0)
SIGMA = 0.02
room = world.room("living").polygon     # today the room is given: the operating regime, world frame

# %% [markdown]
# ## A. The body is a belief too
#
# ### A1. The sawtooth
#
# The robot predicts each step from its odometry, then corrects the prediction against the room.
# If the odometry has a **systematic** error, every correction points the same way, and the next
# step makes the same error again. The robot's trajectory is a sawtooth: a ramp of accumulated
# error, a drop when the room corrects it, the same ramp again.
#
# Below is Day 2's localiser, run with the robot's **nominal** body on a robot whose real body is
# hidden. `tour` drives two laps of a loop through four waypoints. The controller reads the belief,
# never the truth.

# %%
NOMINAL = np.array([1.0, 1.0, 0.0])          # s_v, s_w, b: how the robot reads its own sensors
PRIOR_SD = np.array([0.05, 0.05, 0.02])      # what it is prepared to believe about each


def localise(prior, P_robot, polygon, iters=3):
    """Day 2's localiser. The state may carry more than the pose; only s[:3] moves the scan."""
    A, x = polygon_area(polygon), prior.mu.copy()
    res = lambda P: (lambda s: distance_to_polygon(se2.transform_points(s[:3], P), polygon))
    for _ in range(iters):
        r = res(P_robot)(x)
        rho = A / (20.0 * np.sqrt(2 * np.pi) * SIGMA) * np.exp(-r ** 2 / (2 * SIGMA ** 2))
        w = rho / (1 + rho)
        use = w > 1e-3
        fit = laplace(res(P_robot[use]), x, SIGMA / np.sqrt(w[use]), prior=prior)
        x = fit.belief.mu
    return fit.belief


def goto(mu, waypoint, dt=0.5):
    """Turn on the spot towards the waypoint, then drive at it. Reads only the belief."""
    d = waypoint - mu[:2]
    err = se2.wrap(np.arctan2(d[1], d[0]) - mu[2])
    if abs(err) > 0.05:
        return 0.0, np.clip(err / dt, -np.pi / 4, np.pi / 4)
    return min(0.4, np.hypot(*d) / dt), np.clip(err / dt, -0.3, 0.3)


LOOP = np.array([[5, 0.8], [5, 3.4], [1, 3.4], [1, 0.8]])
START = np.array([1.0, 0.8, 0.0])


def tour(body, seed, predict_fn, learn, laps=2):
    """Drive the loop. Returns (true pose, prediction, posterior, odometry) per step."""
    r = np.random.default_rng(seed)
    robot = Robot(START.copy(), body, r)
    body_var = PRIOR_SD ** 2 if learn else np.full(3, 1e-14)   # not learning = certain of the nominal body
    belief = Gaussian([*START, *NOMINAL], np.diag([1e-6] * 3 + list(body_var)))
    waypoints, k, log = [w for _ in range(laps) for w in LOOP], 0, []
    while k < len(waypoints) and len(log) < 400:
        if np.hypot(*(waypoints[k] - belief.mu[:2])) < 0.05:
            k += 1
            continue
        odo = robot.step(*goto(belief.mu, waypoints[k]), 0.5)
        pred = predict_fn(belief, odo)
        assert world.room_of(robot.pose[:2]) == "living", "the route left the room!"
        post = localise(pred, lidar.scan(world, robot.pose, r).points(), room)
        log.append((robot.pose.copy(), pred, post, odo))
        belief = post
    return log


def nees(log):
    out = []
    for pose, _, post, _ in log:
        e = pose - post.mu[:3]
        e[2] = se2.wrap(e[2])
        out.append(e @ np.linalg.solve(post.Sigma[:3, :3], e))
    return np.mean(out)

# %% [markdown]
# The state is $s = (x, y, \theta, s_v, s_\omega, b)$: the pose, and three body parameters. The
# robot's true forward speed is $s_v$ times what the encoders report. Its true turn rate is
# $s_\omega(\omega_\text{gyro} - b)$. The nominal body is $(1, 1, 0)$.
#
# **Exercise 1.** Write `body_motion(s, v_meas, w_meas, dt)`: integrate the pose with the speeds
# *as the body parameters interpret them*, and carry the parameters unchanged.

# %%
def body_motion(s, v_meas, w_meas, dt):
    # BEGIN SOLUTION
    s_v, s_w, b = s[3:]
    return np.concatenate([integrate(s[:3], s_v * v_meas, s_w * (w_meas - b), dt), s[3:]])
    # END SOLUTION


def predict_body(belief, odo, Td=np.inf, enc=0.005, gyro=0.003, slip_v=0.02, slip_w=0.02):
    """Push the belief through body_motion. Noise is the sensors' and the wheels' own, and grows
    with speed. No constant process noise. With a finite drift time Td, each body parameter relaxes
    towards its nominal value and regains prior variance at the rate the drift prior says."""
    F = numeric_jacobian(lambda s: body_motion(s, odo.v, odo.w, odo.dt), belief.mu)
    G = numeric_jacobian(lambda u: body_motion(belief.mu, u[0], u[1], odo.dt), [odo.v, odo.w])
    sd_v, sd_w = np.hypot(enc, slip_v * abs(odo.v)), np.hypot(gyro, slip_w * abs(odo.w))
    mu = body_motion(belief.mu, odo.v, odo.w, odo.dt)
    Sigma = F @ belief.Sigma @ F.T + G @ np.diag([sd_v ** 2, sd_w ** 2]) @ G.T
    if np.isfinite(Td):
        a = np.exp(-odo.dt / Td)
        A = np.diag([1, 1, 1, a, a, a])
        mu[3:] = NOMINAL + a * (mu[3:] - NOMINAL)
        Sigma = A @ Sigma @ A.T + np.diag([0, 0, 0, *(PRIOR_SD ** 2 * (1 - a * a))])
    return Gaussian(mu, Sigma)


TRUE_BODY = hidden_body()
TRUTH = np.array([1 / TRUE_BODY.wheel_scale, 1 / TRUE_BODY.gyro_scale, TRUE_BODY.gyro_bias])
nominal_log = tour(TRUE_BODY, 1, predict_body, learn=False)
print(f"{len(nominal_log)} steps;  NEES with the nominal body: {nees(nominal_log):.2f}   (3 = honest)")

# %% [markdown]
# **Exercise 2.** The correction of each step is $r = \mu_\text{post} - \mu_\text{pred}$. Express it
# in the robot's own frame (forward, sideways, heading). Then fit the forward correction against
# the distance driven, $r_\text{fwd} \approx \beta_d\, d$, and the heading correction against the
# gyro's rotation and the elapsed time, $r_\phi \approx \beta_\phi\,\phi + \beta_t\,T$.
#
# What should the coefficients be, if the hidden body is the only cause?

# %%
corr, cov = [], []
for _, pred, post, odo in nominal_log:
    d, th = post.mu[:3] - pred.mu[:3], pred.mu[2]
    # BEGIN SOLUTION
    corr.append([np.cos(th) * d[0] + np.sin(th) * d[1], -np.sin(th) * d[0] + np.cos(th) * d[1], se2.wrap(d[2])])
    cov.append([odo.v * odo.dt, odo.w * odo.dt, odo.dt])
    # END SOLUTION
corr, cov = np.array(corr), np.array(cov)
# BEGIN SOLUTION
beta_d = corr[:, 0] @ cov[:, 0] / (cov[:, 0] @ cov[:, 0])
beta_phi, beta_t = np.linalg.lstsq(cov[:, 1:], corr[:, 2], rcond=None)[0]
# END SOLUTION
print(f"forward: {beta_d:+.4f} per metre      (s_v − 1 = {TRUTH[0] - 1:+.4f})")
print(f"heading: {beta_phi:+.4f} per radian,  {beta_t:+.4f} rad/s   "
      f"(s_w − 1 = {TRUTH[1] - 1:+.4f};  −s_w·b = {-TRUTH[1] * TRUTH[2]:+.4f})")

fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(np.cumsum(cov[:, 0]), corr[:, 0] * 1000, ".-", ms=3)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("distance driven (m)"); ax.set_ylabel("forward correction (mm)")
ax.set_title("every correction points the same way: a mean error, not noise")

# %% [markdown]
# **Question.** The nominal filter is over-confident (NEES well above 3). The tempting cure is to
# widen the prediction's covariance. What would that do to the coefficients you just fitted?

# %% [markdown] solution
# *Answer.* Nothing. A covariance says how much the prediction may be wrong, not in which
# direction. A wider prediction lets the room pull harder, so each correction is cheaper, but the
# next step makes the same mean error again, and the corrections keep their sign and their slope.
# The thesis draws this as the sawtooth: widening the prior lengthens the tooth and leaves the
# slope. A mean error can only be removed by changing the mean of the prediction, that is, the
# parameters it is computed from.

# %% [markdown]
# ### A2. Put the body in the state
#
# The cure is to stop treating $(s_v, s_\omega, b)$ as constants and estimate them with the pose.
# Nothing new is needed. They are three more numbers in the belief, with a prior, and the room's
# corrections reach them through the cross-covariance the prediction builds.
#
# **Exercise 3.** Run the tour again with `learn=True`. Report each parameter's estimate, its
# posterior sd, and its error in units of that sd ($z$). Compare the NEES with the nominal run.

# %%
learn_log = tour(TRUE_BODY, 1, predict_body, learn=True)
final = learn_log[-1][2]
# BEGIN SOLUTION
z = (final.mu[3:] - TRUTH) / final.sd[3:]
# END SOLUTION
for name, m, s, t, zz in zip(("s_v", "s_w", "b"), final.mu[3:], final.sd[3:], TRUTH, z):
    print(f"{name:4s} {m:8.4f} ± {s:.4f}   truth {t:.4f}   z = {zz:+.2f}")
print(f"NEES: nominal body {nees(nominal_log):.2f}  →  learnt body {nees(learn_log):.2f}")

fig, axs = plt.subplots(1, 3, figsize=(10, 2.6))
traj = np.array([p.mu[3:] for _, _, p, _ in learn_log])
sds = np.array([p.sd[3:] for _, _, p, _ in learn_log])
for i, (ax, name) in enumerate(zip(axs, ("s_v", "s_ω", "b (rad/s)"))):
    ax.fill_between(range(len(traj)), traj[:, i] - 2 * sds[:, i], traj[:, i] + 2 * sds[:, i], alpha=0.25)
    ax.plot(traj[:, i]); ax.axhline(TRUTH[i], color="k", ls="--", lw=0.8); ax.set_title(name)
    ax.set_xlabel("step")
fig.tight_layout()

# %% [markdown]
# ### A3. What each motion makes observable
#
# A parameter is learnt only if the motion moves it into the data. Run five short routes from rest,
# each 8 s, and look at what each one taught.
#
# There are two ways to ask "how well is $s_\omega$ known?":
# - **conditional**: as if every other unknown were known, $1/\sqrt{\Lambda_{ii}}$ with
#   $\Lambda = \Sigma^{-1}$;
# - **marginal**: with the other unknowns estimated too, $\sqrt{\Sigma_{ii}}$. The difference is the
#   Schur complement: the part of a parameter's effect that another unknown could equally have
#   produced does not count.
#
# **Exercise 4.** For each route, compute both, divided by the prior sd.

# %%
def run(body, commands, seed, start, prior_sd=PRIOR_SD):
    r = np.random.default_rng(seed)
    robot = Robot(np.array(start, float), body, r)
    b = Gaussian([*start, *NOMINAL], np.diag([1e-6] * 3 + list(np.asarray(prior_sd) ** 2)))
    for v, w in commands:
        b = localise(predict_body(b, robot.step(v, w, 0.5)), lidar.scan(world, robot.pose, r).points(), room)
    return b


ROUTES = {
    "parked":          ([(0, 0)] * 16, (3.0, 2.0, 0.0)),
    "straight":        ([(0.4, 0)] * 16, (1.0, 0.8, 0.0)),
    "spin":            ([(0, np.pi / 4)] * 16, (3.0, 2.0, 0.0)),
    "spin, two rates": ([(0, np.pi / 4)] * 8 + [(0, np.pi / 8)] * 8, (3.0, 2.0, 0.0)),
    "spin, then park": ([(0, np.pi / 4)] * 8 + [(0, 0)] * 8, (3.0, 2.0, 0.0)),
}
print(f"{'':16s} {'marginal σ/σ_prior':>26s}   {'conditional σ/σ_prior':>26s}")
print(f"{'':16s} {'s_v':>8s}{'s_w':>9s}{'b':>9s}   {'s_v':>8s}{'s_w':>9s}{'b':>9s}")
observability = {}
for name, (cmds, start) in ROUTES.items():
    b = run(TRUE_BODY, cmds, 2, start)
    # BEGIN SOLUTION
    marginal = b.sd[3:] / PRIOR_SD
    conditional = 1 / np.sqrt(np.diag(np.linalg.inv(b.Sigma))[3:]) / PRIOR_SD
    # END SOLUTION
    observability[name] = (marginal, conditional)
    print(f"{name:16s} {marginal[0]:8.3f}{marginal[1]:9.3f}{marginal[2]:9.3f}   "
          f"{conditional[0]:8.3f}{conditional[1]:9.3f}{conditional[2]:9.3f}")

# %% [markdown]
# **Questions.**
# - Which route teaches the gyro bias best, and why does it cost nothing?
# - On "spin", $s_\omega$ looks well determined conditionally and poorly marginally. Why? Why do
#   two rates, or a spin followed by a pause, fix it?

# %% [markdown] solution
# *Answers.*
# - **Parked.** While the room shows the heading is not changing, whatever the gyro reports is its
#   bias. The bias's covariate is elapsed time and nothing else, so the action that gathers the
#   most information about it is the one that spends time and does nothing else. Navigation
#   engineers call this a zero-velocity update; here it falls out of the Jacobian. The same pause
#   says nothing about $s_v$: the wheels report no distance, and $s_v \times 0 = 0$.
# - At a constant turn rate, the heading error is $(s_\omega - 1)\,\omega T$ from the scale and
#   $-s_\omega b\, T$ from the bias. Both grow in proportion to $T$, so the data see one
#   combination of the two and cannot split it. Conditionally, with the bias known, the scale is
#   sharp. Marginally, the bias can explain the same data, and the Schur complement removes that
#   share. A second rate, or a pause (rate zero), makes the two covariates differ, and the
#   confusion goes. This is why the thesis computes observability rather than arguing it.

# %% [markdown]
# ### A4. What a known error should do to the estimate
#
# How do you test an estimator whose truth you do not know? Inject an error of known size where the
# estimator cannot see it, and check the estimate moves by that amount. The thesis did this with a
# 3% wheel-scale error: the estimate moved by 104% of it.
#
# A Bayesian estimator does **not** return the whole injection. It returns the fraction its data can
# see against its prior. Along a direction the data determine with information $\lambda$ (in units
# of the prior's), it returns $\lambda/(1+\lambda)$, and that is the same number as
#
# $$ \text{recovery} = 1 - \left(\frac{\sigma_\text{post}}{\sigma_\text{prior}}\right)^2. $$
#
# **Exercise 5.** With a tight prior on $s_v$ (sd 0.01), drive straight for $n$ = 1, 2, 4, 8 steps,
# twice with the **same seed**: once with a 4% wheel-scale error, once without. The measured recovery is
# the difference of the two estimates divided by the injected shift. Compare it with the identity.

# %%
TIGHT = [0.01, 0.05, 0.02]
shift = 1 / 1.04 - 1
for n in (1, 2, 4, 8):
    rec, pred = [], []
    for seed in range(5):
        # BEGIN SOLUTION
        a = run(Body(wheel_scale=1.04), [(0.4, 0)] * n, seed, START, TIGHT)
        b = run(Body(), [(0.4, 0)] * n, seed, START, TIGHT)
        rec.append((a.mu[3] - b.mu[3]) / shift)
        pred.append(1 - (b.sd[3] / TIGHT[0]) ** 2)
        # END SOLUTION
    print(f"n = {n}:  recovered {np.mean(rec):.3f}   predicted from σ alone {np.mean(pred):.3f}")

# %% [markdown]
# **Lesson.** The same seed makes the two runs share every noise draw, so the difference sees only
# the injection: that pairing is what made the thesis's result readable. And the identity means an
# injection experiment is predictable from the posterior alone. Running both checks that the
# estimator is consistent with itself, rather than giving two independent facts.

# %% [markdown]
# ### A5. A body that drifts, and the prior that must not move
#
# Tyres wear and mounts loosen, so the thesis lets each parameter relax towards its nominal value
# with a drift time $T_d$ and regain prior variance as it does. Information arriving at rate $i$
# balances the drift at a steady variance $P$:
#
# $$ 0 = -\frac{2P}{T_d} + \frac{2\sigma_p^2}{T_d} - i\,P^2 . $$
#
# **Exercise 6.** Write $\sigma/\sigma_p$ at the steady state as a function of the single number
# $n = i\,\sigma_p^2 T_d$. How much tighter is a body tracked if it drifts 1000 times more slowly?

# %%
def steady_ratio(n):
    # BEGIN SOLUTION
    p = (np.sqrt(1 + 2 * n) - 1) / n          # P / σ_p², the positive root
    return np.sqrt(p)
    # END SOLUTION


for n in (1, 10, 100, 1e3, 1e4, 1e5):
    print(f"n = {n:8.0f}:  σ/σ_p = {steady_ratio(n):.4f}")
print(f"1000× slower drift: {steady_ratio(1e2) / steady_ratio(1e5):.2f}× tighter (the fourth root of 1000 is {1000 ** 0.25:.2f})")

# %% [markdown]
# A body that drifts a thousand times more slowly is tracked only about six times more tightly.
#
# When nothing excites a parameter, the drift model says it is becoming less known. Here is the
# learnt body, then a minute parked with $T_d$ = 30 s:

# %%
r_ = np.random.default_rng(5)
robot = Robot(learn_log[-1][0].copy(), TRUE_BODY, r_)
b, hist = learn_log[-1][2], []
for _ in range(120):
    b = localise(predict_body(b, robot.step(0, 0, 0.5), Td=30.0), lidar.scan(world, robot.pose, r_).points(), room)
    hist.append(np.r_[b.mu[3:], b.sd[3:]])
hist = np.array(hist)
fig, axs = plt.subplots(1, 2, figsize=(8, 2.6))
axs[0].plot(np.arange(120) * 0.5, hist[:, 3] / PRIOR_SD[0], label="s_v"); axs[0].plot(np.arange(120) * 0.5, hist[:, 5] / PRIOR_SD[2], label="b")
axs[0].set_ylabel("σ / σ_prior"); axs[0].set_xlabel("seconds parked"); axs[0].legend()
axs[1].plot(np.arange(120) * 0.5, hist[:, 0]); axs[1].axhline(TRUTH[0], color="k", ls="--", lw=0.8)
axs[1].set_title("s_v relaxes to nominal while unexcited"); axs[1].set_xlabel("seconds parked")
fig.tight_layout()

# %% [markdown]
# The parked robot forgets $s_v$ honestly: the mean returns to nominal **and the sd returns to the
# prior's**, so nothing downstream trusts a value no one is measuring. The bias, which a pause
# measures, stays sharp.
#
# The obvious way to stop forgetting is to re-centre the prior on the running estimate after each
# solve. The thesis rejected it.
#
# **Exercise 7.** Take one window of evidence about $s_v$ (the 8-step straight run's corrections are
# summarised below as one linear measurement). Solve it once. Then re-solve it five more times, each
# time using the previous posterior as the new prior, **with the same data**. What happens to the
# estimate and to its sd?

# %%
win = Gaussian([0.0], [[0.05 ** 2]])                 # prior on the deviation s_v − 1
J, R, y = np.array([[1.0]]), np.array([[0.004 ** 2]]), np.array([TRUTH[0] - 1 + 0.003])
# BEGIN SOLUTION
post = linear_gaussian_update(win, J, R, y)
history = [post]
for _ in range(5):
    post = linear_gaussian_update(post, J, R, y)     # the prior re-centred on its own estimate
    history.append(post)
# END SOLUTION
for k, h in enumerate(history):
    print(f"solve {k + 1}: estimate {h.mu[0]:+.4f}, sd {h.sd[0] * 1000:.2f} ×10⁻³")

# %% [markdown] solution
# *Answer.* The estimate barely moves, and the sd shrinks as $1/\sqrt{k}$ with no new data: the same
# evidence is counted $k$ times. That is the first failure. The second needs no repetition: a prior
# that follows the estimate exerts no pull on a parameter nothing is exciting, so each solve moves
# it a little by noise and the prior follows. The value performs a random walk, and every window
# still reports it as "not yet learnt", because each window's precision is computed from its own
# data. The thesis keeps the prior on the nominal value and accepts the forgetting: a parameter
# reported as unknown can be relearnt, a wrong value carried with confidence is never questioned.
# It proposes a second level (a slowly drifting mean, learnt from many sessions) to keep the memory
# without either failure.

# %% [markdown]
# **Question (for discussion).** Should the robot have a "calibrate yourself" action, such as
# spinning on the spot when a parameter goes stale?

# %% [markdown] solution
# *Answer.* The thesis computes it rather than guessing. An episode of length $T$ at rates $(v,
# \omega)$ carries information $\propto(\text{rate})^2 T$ per second about each scale, and $T$
# alone about the bias. A manoeuvre can only raise rate and duration, both capped by the platform.
# Forward driving is what navigation wants anyway, and a cluttered room forces the turns, so for
# this robot in this kind of room the ordinary task already spans the covariates. Where it does
# not, need and cost rise together: a route lacks rotation exactly because rotation costs the
# planner something. So the action should compete with idleness, never with a task. The only
# calibration action the analysis justifies on this platform is to stop while otherwise idle.

# %% [markdown]
# ## B. Where to look next
#
# Now the world's turn. The robot keeps an occupancy grid of the living room, 5 cm cells, each
# holding a log-odds $l$ of being occupied. A look lowers $l$ in the cells a beam passes through
# and raises it where the beam ends. The robot's pose is known here (Day 2 and Part A did that work).
#
# ### B1. What a look is worth
#
# A cell with occupancy probability $p = \sigma(l)$ will either return the beam (probability $p$)
# or not. Its expected entropy after the look is the average over the two outcomes:
#
# $$ \Delta H_\text{cell} = H(p) - \big[\,p\,H(\sigma(l + l_\text{hit})) + (1-p)\,H(\sigma(l - l_\text{miss}))\big]. $$
#
# **Exercise 8.** Write `cell_gain`. Plot it against $l$ from −8 to 8. How much does the first look
# at an unknown cell buy, and how much the third?

# %%
def cell_gain(l, l_hit=2.2, l_miss=2.2):
    # BEGIN SOLUTION
    p = sigmoid(l)
    return binary_entropy(p) - (p * binary_entropy(sigmoid(l + l_hit)) + (1 - p) * binary_entropy(sigmoid(l - l_miss)))
    # END SOLUTION


ls = np.linspace(-8, 8, 200)
plt.figure(figsize=(5, 2.8))
plt.plot(ls, cell_gain(ls)); plt.xlabel("log-odds l"); plt.ylabel("expected gain (nats)")
print(f"first look at an unknown cell: {cell_gain(0.0):.3f} nats;  after two misses: {cell_gain(-4.4):.3f} nats")

# %% [markdown]
# ### B2. The explorer
#
# A candidate viewpoint $v$ is scored by the entropy a look from it is expected to remove,
# **predicted through the model**: the room's walls and the grid's own beliefs, never the true room.
# `expected_gain` (in `toyworld.grid`) casts 180 predicted beams and sums `cell_gain` over the
# cells a real look would touch. The robot then chooses
#
# $$ v^* = \arg\min_v \; G(v) = \alpha + \beta\,\lVert v - x\rVert - \Delta H(v), $$
#
# where $\alpha$ is the price of one look (time), and $\beta$ the price of a metre. Both are
# preferences, priced in nats like everything else. It stops when no candidate has $G < 0$:
# when nothing left is worth its price.
#
# **Exercise 9.** Complete the decision in `explore`: score the candidates, pick the best, and stop
# when it is not worth it. Record the predicted and the realised gain of every look.

# %%
WALLS = segments_of(room)
ALPHA, BETA = 20.0, 2.0


def candidates(grid):
    """Lattice points 0.3 m from the walls whose neighbourhood the grid believes free."""
    pts = np.mgrid[0.25:6:0.25, 0.25:4:0.25].reshape(2, -1).T
    keep = []
    for c in pts:
        near = np.hypot(*(grid.centres - c).transpose(2, 0, 1)) < 0.3
        if distance_to_polygon(c[None], room)[0] >= 0.3 and (grid.p[near] < 0.5).all():
            keep.append(c)
    return np.array(keep)


def explore(seed, through_matter=False, max_looks=15):
    r = np.random.default_rng(seed)
    grid, x = Grid(), START.copy()
    in_room = grid.mask(room)
    grid.update(lidar.scan(world, x, r))
    looks = []
    for _ in range(max_looks):
        C = candidates(grid)
        gains = np.array([expected_gain(grid, c, WALLS, in_room, through_matter) for c in C])
        dist = np.hypot(*(C - x[:2]).T)
        # BEGIN SOLUTION
        G = ALPHA + BETA * dist - gains
        best = np.argmin(G)
        if G[best] >= 0:
            break
        # END SOLUTION
        before = grid.entropy(in_room)
        x = np.array([*C[best], 0.0])
        grid.update(lidar.scan(world, x, r))
        looks.append((C[best], gains[best], before - grid.entropy(in_room), dist[best]))
    return grid, looks


grid, looks = explore(0)
print(f"{'look':>4s} {'viewpoint':>12s} {'predicted':>10s} {'realised':>9s} {'travel':>7s}")
for k, (c, g, real, d) in enumerate(looks):
    print(f"{k + 1:4d} {str(c):>12s} {g:10.1f} {real:9.1f} {d:6.2f} m")
print(f"stopped after {len(looks)} looks;  entropy left in the room: {grid.entropy(grid.mask(room)):.0f} nats")

fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(grid.p.T, origin="lower", extent=(0, 6, 0, 4), cmap="gray_r", vmin=0, vmax=1)
ax.plot(*np.array([c for c, *_ in looks]).T, "o-", color="crimson", ms=4)
ax.set_title("occupancy after exploring; red: the viewpoints")

# %% [markdown]
# ### B3. The trap
#
# The first version of the thesis's explorer cast its predicted beams **through the room's polygon
# only**: the walls stopped them, but nothing the grid already knew did. Run the explorer that way.
#
# **Exercise 10.** Run `explore(seed, through_matter=True)` for seeds 0–3 next to the honest one.
# For each, report the number of looks, whether it stopped, and the total predicted and realised
# gain. Then look at where the entropy that is left sits.

# %%
table = world.things[0]
in_table = grid.mask(table.outline())
for seed in range(4):
    for tm in (False, True):
        # BEGIN SOLUTION
        g_, lk = explore(seed, through_matter=tm)
        pred_total, real_total = sum(l[1] for l in lk), sum(l[2] for l in lk)
        stopped = len(lk) < 15
        # END SOLUTION
        print(f"seed {seed}  {'through matter' if tm else 'honest        '}  looks {len(lk):2d}  "
              f"stopped {str(stopped):5s}  predicted {pred_total:6.0f}  realised {real_total:6.0f}  "
              f"left inside the table {g_.entropy(in_table):4.0f} nats")

# %% [markdown]
# **Questions.**
# - Why does the explorer that predicts through matter never stop?
# - Most of the entropy the honest explorer leaves behind sits inside the table. Is that a
#   failure?

# %% [markdown] solution
# *Answers.*
# - Its predicted beams pass through the table and promise the entropy of cells **inside** it. No
#   real beam ever reaches them, so the promise is never collected and is offered again at the
#   next decision, from whichever side of the table is nearest. Predicted gain stays high while
#   realised gain falls to the noise. In the thesis this made the robot shuttle between two
#   viewpoints, "each worth a hundred nats and neither collecting anything". A prediction must be
#   made through everything the model believes, or it is not an expectation.
# - No. A viewpoint is worth what it can **resolve**, not what it can see: those cells cannot be
#   resolved from anywhere, and stopping is the correct answer. It does mean a coverage number
#   such as "fraction of cells known" can never reach 100%, and that any metric scored over the
#   whole room is partly scoring what no one saw. Day 5 comes back to this.
#
# **Predicted against realised** is the most useful diagnostic an explorer has. The robot's
# explorer once agreed with nothing it predicted, because it consumed map-frame waypoints in the
# world frame and never drove its own plan (post-thesis, work in progress). The bug was found from
# a picture.
#
# ## C. Today's lesson
#
# Part A: the body is learnt only along the directions the motion excites; a pause teaches the
# bias; a constant spin cannot split scale from bias. Part B: the room is learnt only where the
# robot goes and looks, and only where a look can reach. In both, **what can be learnt is a
# property of the parameter and the action together.** The body gets its excitation for free from
# ordinary driving; the room does not, and the robot must go and fetch it.
