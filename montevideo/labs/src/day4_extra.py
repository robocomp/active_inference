# %% [markdown]
# # Day 4, extra — The explorer and the follower
#
# Two parts moved out of the main Day 4 lab, for anyone who wants the controller's side in code.
#
# - **A. The room.** An occupancy-grid explorer that knows when to stop, and the trap of predicting
#   through matter.
# - **B. Following a route.** A law with no tuned gains and no obstacle term.
#
# About 45 minutes. Independent of each other.

# %%
import sys
from pathlib import Path

LABS = Path.cwd() if (Path.cwd() / "toyworld").exists() else Path.cwd().parent
sys.path.insert(0, str(LABS))

import matplotlib.pyplot as plt
import numpy as np

from toyworld import Gaussian, Grid, Lidar, Robot, draw, laplace, se2, two_room_apartment
from toyworld.beliefs import binary_entropy, linear_gaussian_update, numeric_jacobian, sigmoid
from toyworld.geometry import distance_to_polygon, point_in_polygon, polygon_area, segments_of
from toyworld.grid import expected_gain
from toyworld.robot import Body, hidden_body, integrate

world = two_room_apartment()
lidar = Lidar(n_beams=360, sigma=0.02, max_range=12.0)
SIGMA = 0.02
room = world.room("living").polygon     # today the room is given: the operating regime, world frame
START = np.array([1.0, 0.8, 0.0])

# %% [markdown]
# ## A. The room: where to look next
#
# Now the world's turn. The robot keeps an occupancy grid of the living room, 5 cm cells, each
# holding a log-odds $l$ of being occupied. A look lowers $l$ in the cells a beam passes through
# and raises it where the beam ends. The robot's pose is known here (Day 4 did that work).
#
# ### A1. What a look is worth
#
# A cell with occupancy probability $p = \sigma(l)$ will either return the beam (probability $p$)
# or not. Its expected entropy after the look is the average over the two outcomes:
#
# $$ \Delta H_\text{cell} = H(p) - \big[\,p\,H(\sigma(l + l_\text{hit})) + (1-p)\,H(\sigma(l - l_\text{miss}))\big]. $$
#
# **Exercise 1.** Write `cell_gain`. Plot it against $l$ from −8 to 8. How much does the first look
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
# ### A2. The explorer
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
# **Exercise 2.** Complete the decision in `explore`: score the candidates, pick the best, and stop
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
# ### A3. The trap
#
# The first version of the thesis's explorer cast its predicted beams **through the room's polygon
# only**: the walls stopped them, but nothing the grid already knew did. Run the explorer that way.
#
# **Exercise 3.** Run `explore(seed, through_matter=True)` for seeds 0–3 next to the correct one.
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
#   whole room is partly scoring what no one saw.
#
# **Predicted against realised** is the most useful diagnostic an explorer has. The robot's
# explorer once agreed with nothing it predicted, because it consumed map-frame waypoints in the
# world frame and never drove its own plan (post-thesis, work in progress). The bug was found from
# a picture.
#

# %% [markdown]
# ## B. Following a route: pole assignment
#
# A chosen standpoint has to become motion. On the robot this is split in two: the **geometry** (a
# route optimised against the map and the live distance field) carries every obstacle, and a
# plain analytic law **follows** the curve. Here the robot's pose is known, so we can look at the
# control alone.
#
# Measure the error against the curve at the robot's foot point: the signed cross-track error $e$
# (positive to the left) and the heading error $\phi_e = \theta - \psi(s)$. For small errors
#
# $$ \dot e = v\sin\phi_e \approx v\,\phi_e, \qquad \dot\phi_e = \omega - v\,\kappa(s). $$
#
# ### B1. The law, and where its gains come from
#
# Cancel the curve with a feedforward $v\kappa$ and leave a regulator on the error:
#
# $$ \omega = v\,\kappa(s) - v\Big(\frac{2}{L}\,\phi_e + \frac{1}{L^2}\,e\Big). $$
#
# Substituting gives $\ddot e + \tfrac{2v}{L}\dot e + \tfrac{v^2}{L^2}e = 0$, whose characteristic
# polynomial is $(\sigma + v/L)^2$: **a double pole at $-v/L$**, critically damped by construction.
# Nobody tuned a gain. What was chosen is where the poles sit. In the arc length $s$ the speed cancels,
#
# $$ e(s) = (A + B\,s)\,e^{-s/L}, $$
#
# so the error decays over a **fixed distance**, not a fixed time. $L$ is a length you can read off
# the space the robot works in.
#
# **Exercise 4.** Write `tracking_law`.

# %%
from collections import deque

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


class Route:
    """A curve resampled by arc length, with its tangent angle and curvature."""

    def __init__(self, xy, ds=0.01):
        xy = np.asarray(xy, float)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.s = np.arange(0, s[-1], ds)
        self.xy = np.stack([np.interp(self.s, s, xy[:, 0]), np.interp(self.s, s, xy[:, 1])], 1)
        d = np.gradient(self.xy, ds, axis=0)
        self.psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.kappa = np.gradient(self.psi, ds)

    def foot(self, p, hint=0):
        """Nearest curve sample, searched only ahead of the last one (as the robot does)."""
        hi = min(len(self.s), hint + 200)
        return hint + int(np.argmin(np.hypot(*(self.xy[hint:hi] - p).T)))

    def errors(self, pose, i):
        d = pose[:2] - self.xy[i]
        n = np.array([-np.sin(self.psi[i]), np.cos(self.psi[i])])
        return float(n @ d), float(se2.wrap(pose[2] - self.psi[i]))


def tracking_law(v, kappa, e, phi, L):
    # BEGIN SOLUTION
    return v * kappa - v * (2 / L * phi + e / L ** 2)
    # END SOLUTION


CLEAN = Body(slip_v=0.0, slip_w=0.0, enc_sigma=0.0, gyro_sigma=0.0)


def follow(route, start, v, L=0.5, dt=0.02, feedforward=True, delay=0.0, fixed_speed=None, body=CLEAN, seed=0):
    """Drive the route. `fixed_speed`: evaluate the feedback at that speed, a gain not scheduled with v."""
    robot = Robot(np.asarray(start, float), body, np.random.default_rng(seed))
    i, log = 0, []
    pipe = deque([0.0] * int(round(delay / dt)))
    while i < len(route.s) - 30 and len(log) < 60000:
        i = route.foot(robot.pose[:2], i)
        e, phi = route.errors(robot.pose, i)
        vf = v if fixed_speed is None else fixed_speed                # the speed the feedback is evaluated at
        w = tracking_law(vf, (v / vf) * route.kappa[i] if feedforward else 0.0, e, phi, L)
        pipe.append(w)
        robot.step(v, pipe.popleft(), dt)
        log.append((route.s[i], e, phi))
    return np.array(log)


# %% [markdown]
# **Exercise 5.** On a straight route, release the robot $e_0 = 0.1$ m to one side, at
# $v$ = 0.15, 0.35 and 0.7 m/s, and plot $e$ against distance. Then release it on the route but
# aimed off by $\phi_0 = 0.2$ rad. Check against the prediction: $e_0(1+s/L)e^{-s/L}$ for the first,
# and a peak of exactly $\phi_0 L/e$ after one $L$ of travel for the second.

# %%
LINE = Route(np.array([[0.0, 0.0], [12.0, 0.0]]))
L = 0.5
decay_runs = {}
fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
for v in (0.15, 0.35, 0.7):
    # BEGIN SOLUTION
    a = follow(LINE, [0.0, 0.1, 0.0], v, L)
    b = follow(LINE, [0.0, 0.0, 0.2], v, L)
    # END SOLUTION
    decay_runs[v] = (a, b)
    axs[0].plot(a[:, 0] / L, a[:, 1] / 0.1, label=f"v = {v} m/s"); axs[1].plot(b[:, 0] / L, b[:, 1], label=f"v = {v} m/s")
    print(f"v = {v:4.2f}: displaced, e at s=L {np.interp(L, a[:, 0], a[:, 1]) / 0.1:.3f} (predict {2 / np.e:.3f})   "
          f"mis-aimed, peak {b[:, 1].max():.4f} m at s/L = {b[np.argmax(b[:, 1]), 0] / L:.2f} (predict {0.2 * L / np.e:.4f} at 1)")
xs = np.linspace(0, 6, 100)
axs[0].plot(xs, (1 + xs) * np.exp(-xs), "k:", label="$(1+s/L)e^{-s/L}$"); axs[0].legend(frameon=False, fontsize=8)
axs[0].set_xlabel("s / L"); axs[0].set_ylabel("e / e₀"); axs[1].set_xlabel("s / L"); axs[1].set_ylabel("e (m)"); fig.tight_layout()

# %% [markdown]
# ### B2. Break it three ways
#
# **Exercise 6.** Use `follow`'s options, on a circle of radius 2 m (curvature 0.5) at 0.35 m/s:
#
# 1. switch off the feedforward and read the steady cross-track error; predict it first from the
#    closed loop. (Set $\dot e=\dot\phi_e=0$.)
# 2. add a delay of 0.42, 0.7 and 1.0 s between command and motion, and report the overshoot of a 0.1 m
#    initial error (the largest error on the far side of zero);
# 3. on the straight line, freeze the feedback at `fixed_speed=0.35` while driving at 0.15 and 0.7 m/s.
#    What happened to "the error decays over a fixed distance"?

# %%
th = np.linspace(-np.pi / 2, 1.5 * np.pi, 1500)
CIRCLE = Route(np.stack([2 * np.cos(th), 2 + 2 * np.sin(th)], 1))
start_c = [0.0, 0.0, 0.0]
# BEGIN SOLUTION
no_ff = follow(CIRCLE, start_c, 0.35, L, feedforward=False)
steady = np.median(no_ff[len(no_ff) // 2:, 1])
print(f"no feedforward: steady e = {steady * 100:.1f} cm   (predict −κL² = {-0.5 * L ** 2 * 100:.1f} cm, on the outside of the bend)")
over = {}
for tau in (0.0, 0.42, 0.7, 1.0):
    o = follow(LINE, [0.0, 0.1, 0.0], 0.35, L, delay=tau)
    over[tau] = max(0.0, -o[:, 1].min()) / 0.1
    print(f"delay {tau:4.2f} s  (v·τ/L = {0.35 * tau / L:.2f}):  overshoot {over[tau] * 100:5.1f}% of e₀")
fixed = {}
for v in (0.15, 0.7):
    o = follow(LINE, [0.0, 0.1, 0.0], v, L, fixed_speed=0.35)
    fixed[v] = o
    print(f"gains frozen at 0.35 m/s, driving at {v}: e at s = L is {np.interp(L, o[:, 0], o[:, 1]) / 0.1:.3f} e₀  (scheduled: {2 / np.e:.3f})")
# END SOLUTION

# %% [markdown] solution
# *Answers.* (1) At rest on the curve, $\dot\phi_e = 0$ needs $\omega = v\kappa$. Without feedforward
# only the feedback can produce it, so it holds a standing error of $e = -\kappa L^2$, on the outside
# of the bend. The feedforward is what frees the feedback to reject error instead of producing the
# nominal turn. (2) Delay erodes the double pole's margin, and what matters is the dimensionless
# $v\tau/L$. (3) The gains must scale with the speed. The law does that by construction because every
# term is multiplied by $v$; freeze them and the poles move with the speed, and with them the distance.

# %% [markdown]
# ### B3. A follower with no obstacle term
#
# The law above never mentions an obstacle. Put one on the route and it will drive straight into
# it. The thesis's position is that this is correct, because **obstacles belong to the geometry**:
# the curve is deformed to keep clear, by minimising a functional over the route, and the follower is
# then handed a curve and nothing else. No avoidance mode, no arbitration between avoiding and
# tracking.
#
# $$ J[r] = w_c\big\langle u^2 + 4u^4\big\rangle + w_\kappa\big\langle (\rho\kappa)^2\big\rangle
#    + \lambda_a \lVert p - p_0\rVert^2 , \qquad u = \mathrm{clamp}\big(1 - d/d_c,\ 0,\ 1\big). $$
#
# $u$ is how far the robot's footprint encroaches on the comfort stand-off $d_c$; the quartic makes
# margin expensive near contact; $\rho\kappa$ measures curvature in units of the tightest circle the
# robot can take at full speed.
#
# **Exercise 7.** Complete the clearance term of `route_cost`, then deform the route and follow the
# result with **the same `follow`**. Report the minimum distance from the robot's body to the
# obstacle, before and after.

# %%
OBST, OBST_R, ROBOT_R, DC, RHO = np.array([3.0, 1.02]), 0.15, 0.25, 0.25, 0.8
xk = np.linspace(0.6, 5.4, 13)
P0 = np.stack([xk, 1.2 + 0.4 * np.sin(1.2 * xk)], 1)


def spline_points(P, n=240):
    t = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    cs = CubicSpline(t, P)
    u = np.linspace(0, t[-1], n)
    xy, d1, d2 = cs(u), cs(u, 1), cs(u, 2)
    return xy, (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.sum(d1 ** 2, 1), 1e-9) ** 1.5


def route_cost(q, wc=30.0, wk=1.0, la=0.5):
    P = P0.copy()
    P[1:-1] = q.reshape(-1, 2)
    xy, kappa = spline_points(P)
    d = np.hypot(*(xy - OBST).T) - OBST_R - ROBOT_R
    # BEGIN SOLUTION
    u = np.clip(1 - d / DC, 0.0, 1.0)
    clearance = np.mean(u ** 2 + 4 * u ** 4)
    # END SOLUTION
    return wc * clearance + wk * np.mean((RHO * kappa) ** 2) + la * np.sum((P[1:-1] - P0[1:-1]) ** 2)


res = minimize(route_cost, P0[1:-1].ravel(), method="L-BFGS-B", options=dict(maxiter=150))
P1 = P0.copy()
P1[1:-1] = res.x.reshape(-1, 2)
route0, route1 = Route(spline_points(P0, 800)[0]), Route(spline_points(P1, 800)[0])
start_r = [*P0[0], route0.psi[0]]


def body_path(route, seed=0, v=0.35):
    robot = Robot(np.array(start_r, float), Body(), np.random.default_rng(seed))
    i, path = 0, []
    while i < len(route.s) - 30 and len(path) < 20000:
        i = route.foot(robot.pose[:2], i)
        e, phi = route.errors(robot.pose, i)
        robot.step(v, tracking_law(v, route.kappa[i], e, phi, L), 0.02)
        path.append(robot.pose[:2].copy())
    return np.array(path)


path0, path1 = body_path(route0), body_path(route1)
clear = lambda path: (np.hypot(*(path - OBST).T) - OBST_R - ROBOT_R).min()
print(f"clearance of the body: straight through {clear(path0) * 100:+.1f} cm;  after deforming the route {clear(path1) * 100:+.1f} cm")
print(f"tracking rms on the deformed route: {np.sqrt(np.mean(follow(route1, start_r, 0.35, L, body=Body())[:, 1] ** 2)) * 1000:.1f} mm")

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(*route0.xy.T, ":", color="0.5", label="route before"); ax.plot(*path1.T, "-", color="crimson", label="robot, after")
ax.add_patch(plt.Circle(OBST, OBST_R, color="0.3")); ax.set_aspect("equal"); ax.legend(frameon=False)

# %% [markdown]
# **Question.** The arrangement turns an arbitration problem into a coverage requirement. What does
# the robot give up, and what must now be true of the route for the follower to be safe?

# %% [markdown] solution
# *Answer.* It gives up the follower's last-moment avoidance: an obstacle outside the region the live
# field covers, or one appearing inside the short prefix of the curve held fixed, is not accounted for
# by the curve, and the follower will not avoid it. So the curve must be re-solved fast enough, over a
# large enough window. The thesis is explicit that this window is justified by the division of labour
# it permits. No comparison with the window disabled has been run.

# %% [markdown]
# ## Today's lesson
#
# The room is learnt only where the robot goes and looks, and only where a look can reach:
# **what can be learnt depends on what the robot does.** Choosing between requests needed nothing but
# one currency, and following a curve needed nothing but where the poles sit.
