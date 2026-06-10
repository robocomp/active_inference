# The EFE-Gradient Arm Controller: Mathematics and Active-Inference Foundations

This document specifies, derives, and justifies the continuous-action controller
that drives the Kinova Gen3 7-DoF arm in `kinova_controller`. It covers the exact
math implemented in `src/efe_gradient.{h,cpp}`, the reasoning behind every term,
and how each piece maps onto Active Inference (AIF). It also records the design
history — the three controller iterations and the empirical evidence that drove
them — so future changes start from *why*, not just *what*.

> **Scope.** Everything here is in the **arm-base frame** (`+X` forward, `+Y`
> left, `+Z` up; `z=0` is the table surface). The control target is the
> end-effector (tool) frame `f(q) ∈ SE(3)`. The single per-cycle entry point is
> `efe_gradient_step()`, called every 20 ms from `SpecificWorker::compute()`.

---

## 1. Active Inference in one page

An active-inference agent carries a **generative model** `p(o, s)` over
observations `o` and hidden states `s`, and acts to minimise **free energy**.
Two quantities matter:

- **Variational free energy** `F` — minimised by *perception* (inferring `s`
  from `o`). Not the focus here: this agent's state estimate (the arm
  configuration) is read directly from the proxy, so perception is trivial.
- **Expected free energy** `G` — minimised by *action/planning*. Choosing what
  to do is choosing the policy `π` that minimises `G(π)`.

For a policy, `G` decomposes into

$$
G(\pi) \;=\; \underbrace{-\,\mathbb{E}_{q}\!\left[\ln \tilde p(o)\right]}_{\text{pragmatic / instrumental}}
\;\;\underbrace{-\,\mathbb{E}_{q}\!\left[D_{\mathrm{KL}}\big(q(s\mid o)\,\Vert\,q(s)\big)\right]}_{\text{epistemic / information gain}}.
$$

- The **pragmatic** term pulls observations toward the agent's *preferred*
  distribution `\tilde p(o)` (its goals, encoded as a prior over outcomes).
- The **epistemic** term rewards actions that reduce uncertainty.

**Action as inference.** In the continuous-time / generalized-coordinates
formulation of AIF (Friston's DEM / active inference for motor control), the
agent does not search a discrete policy tree. Instead the generative model
encodes **preferred dynamics** — a *flow* `\dot x^* = f^*(x)` toward goal states —
and the agent infers the action that makes its sensory dynamics match that flow.
With a Gaussian preference, the pragmatic term is a quadratic cost and **action
becomes gradient descent on that cost through the body's forward kinematics.**
That is exactly this controller.

**Why this matters for design.** Every term below is justified *first* as a piece
of a generative model / preference, and only *then* as a control law. When we
needed to change behaviour (e.g. "approach more gently"), the AIF-faithful move
was to **enrich the generative model** (add a preference over velocity), not to
bolt on a separate trajectory planner. See §9.

---

## 2. Notation and kinematics

| Symbol | Meaning |
|---|---|
| `q ∈ ℝ⁷` | arm joint angles (the 7 actuated DoF) |
| `f(q) ∈ SE(3)` | tool-frame pose; position `p(q) ∈ ℝ³`, rotation `R(q) ∈ SO(3)` |
| `J(q) ∈ ℝ^{6×7}` | geometric Jacobian of the tool frame |
| `J_lin = J[0:3,:]` | linear (positional) block, `∂p/∂q` |
| `J_ang = J[3:6,:]` | angular block, maps `\dot q` → tool angular velocity `ω` |
| `x* ∈ ℝ³` | preferred tool position (the standoff target) |
| `R_des ∈ SO(3)` | preferred tool orientation |
| `e = p(q) − x*` | position error |

**Pinocchio detail.** Forward kinematics and `J` come from a Pinocchio model of
`gen3_robotiq_2f_85-mod.urdf`. Four of the seven joints are *continuous*, encoded
with two configuration entries each (`cos, sin`), so `model.nq = 11` while
`model.nv = 7`. `Kinematics` converts between the 7-vector of angles the proxy
reports and Pinocchio's 11-vector. All control math operates on the 7 velocity
DoF, so `J ∈ ℝ^{6×7}` throughout.

**World frame.** All FK/Jacobian output is in the **world frame**: `Kinematics`'s
`base_tf_` is set at runtime to the arm base_link world pose (the Webots `P3Bot`
node pose composed with the fixed arm→body mount), so `p(q)`, `R(q)`, `J`, the
elbow position/Jacobian, the targets (bottle/table from `getObjectPose`) and the
obstacle geometry (mast at the arm-base xy, table top z) are all in one consistent
world frame. The mast is tilt/height-agnostic this way — re-reading the live arm
pose makes the controller track a re-mounted arm with no code change.

---

## 3. The generative model and its preferences

The agent's preference is a distribution over the **tool pose observation**
`o = f(q)`, factored into position and orientation:

$$
\tilde p(o) \;=\; \underbrace{\mathcal{N}\!\big(p;\,x^*,\,\Pi_p^{-1}\big)}_{\text{position}}
\;\times\;
\underbrace{\mathcal{N}_{SO(3)}\!\big(R;\,R_{des},\,\Pi_R^{-1}\big)}_{\text{orientation}} .
$$

- `Π_p = diag(C_pos)` is the **position precision** (inverse covariance). It is a
  diagonal anisotropic weighting; `C_pos = (1,1,1)` (isotropic) is used for the
  bottle approach (see §5.1 and §9 for why the earlier `(4,4,8)` was dropped).
- The orientation factor is a Gaussian on the rotation group, whose negative
  log-density is the squared **geodesic distance** `½‖log(R_des Rᵀ)‖²` (§5.2).

Crucially, the generative model is extended to a preference over **position
*and* velocity** — the attractor sits at `(x*, \dot x = 0)`. This is the
generalized-coordinates step that makes "arrive gently" a property of the model
rather than a tuning hack (§5.1, §9).

The **instrumental (pragmatic) cost** is the negative log preference:

$$
G_{\text{inst}}(q) \;=\; \tfrac12\,\big(p(q)-x^*\big)^\top \Pi_p \big(p(q)-x^*\big)
\;+\; \tfrac12\,\theta(R(q))^2 \, \pi_R \;+\;\text{const},
$$

where `θ` is the geodesic angle to `R_des` and `π_R` the (scalar) orientation
precision.

---

## 4. From EFE gradient to resolved-rate control

### 4.1 Bare gradient descent (the textbook starting point)

Action selection is gradient descent on `G_inst` through the kinematics. For the
position factor alone, the chain rule gives

$$
\nabla_q G_{\text{inst}} \;=\; J_{lin}^\top\,\Pi_p\,\big(p(q)-x^*\big),
\qquad
\dot q \;=\; -\,\alpha\,J_{lin}^\top\,\Pi_p\,\big(p(q)-x^*\big).
$$

This is the **original** controller (commit history: "first stage working"). It
works but has a structural flaw: `\dot q` magnitude scales with the error, so
far from the target it **saturates the joints** and the tool cruises at full
speed until the very end — overshooting and shoving the object. The fix is *not*
a speed clamp; it is a richer preference (§5.1).

### 4.2 Damped least squares = natural gradient with an effort prior

Steepest descent in *joint* space ignores the task-space geometry. The
**natural gradient** preconditions by the task metric `J J^\top`, giving the
resolved-rate (Gauss–Newton) form. With Tikhonov damping `λ`:

$$
\dot q \;=\; -\,J_{lin}^\top\big(J_{lin}J_{lin}^\top + \lambda^2 I\big)^{-1}\,\Pi_p\,e .
$$

`λ` plays a double role, both of which are AIF-meaningful:

1. **Regulariser** near kinematic singularities (bounds `\dot q` when `JJ^\top`
   loses rank) — a Levenberg–Marquardt trust region.
2. **Prior precision on action.** Damped least squares is the MAP solution of a
   model with a Gaussian prior `\mathcal N(\dot q; 0, \lambda^{-2}I)` on the
   command — i.e. a **preference for low-effort, smooth action**. This is Corke's
   "velocity-effort cost" and is itself a generative-model prior, not an ad-hoc
   addition. (`dls_lambda = 0.05`.)

### 4.3 The preferred-flow attractor (preference over velocity)

Extend the preference from position to a **preferred Cartesian flow**. Instead of
"be at `x*`", the model prefers a velocity field that *converges* to `x*` and
arrives at rest:

$$
\dot x^* \;=\; -\,v(\lVert e\rVert)\,\hat e, \qquad \hat e = \frac{\Pi_p\,e}{\lVert \Pi_p\,e\rVert},
$$

with a **constant-deceleration speed profile**

$$
\boxed{\;v(\lVert e\rVert) \;=\; \min\!\Big(v_{\text{app}},\;\sqrt{2\,a_{\text{app}}\,\lVert e\rVert}\Big)\;}
$$

#### Why the square root (and not `k·‖e‖`)
The `√` law is the velocity that decelerates to **zero exactly at the target**
under constant deceleration `a_app`. Setting `v² = 2 a d` (the kinematic
"stopping distance" relation) and solving for `v` gives `v = √(2 a d)`. It
reaches the goal in **finite time** with a smooth ease-in.

A naive *proportional* law `v = k‖e‖` (the first attempt) instead yields
`\dot d = -k d`, i.e. `d(t) = d₀ e^{-kt}` — an **exponential** that asymptotes
and never arrives. Empirically this produced the "very, very slow creep" near
the goal. The `√` profile fixed it. (`v_app = 0.15 m/s`, `a_app = 0.15 m/s²`;
the cruise→decel transition is at `d* = v_app²/(2 a_app) ≈ 7.5 cm`.)

The joint command realising this flow is the DLS resolved-rate
`\dot q = J_{lin}^\top(J_{lin}J_{lin}^\top+\lambda^2I)^{-1}\,\dot x^*`, so
`‖tool velocity‖ ≈ v(‖e‖)`.

**AIF reading.** This is the generalized-coordinates form: the generative model
specifies a *preferred trajectory* (a point attractor at `(x*, 0)`), and action
is inferred to realise it. The gentle, "planned-looking" arrival is the
**free-energy minimum of the goal**, not a separately clocked plan. This is the
distinction we deliberately preserved over an explicit screw-motion +
trapezoidal-profile planner, which would have been classical plan-then-track and
*outside* the AIF paradigm.

### 4.4 Orientation: the SO(3) geodesic flow

Orientation error lives on the rotation group. Given the current tool rotation
`R` and the desired `R_des`, the **shortest-path (geodesic) error** is

$$
R_e = R_{des}\,R^\top, \qquad \log(R_e) = \theta\,\hat a \quad(\text{axis–angle},\;\theta\in[0,\pi]),
$$

and the world-frame angular velocity that rotates `R → R_des` along that geodesic
is `ω = θ \hat a`. This is the Riemannian gradient flow of `½θ²` on `SO(3)`. We
bound its **speed** with the same saturated-ramp idea as position:

$$
\boxed{\;\omega^* \;=\; \min\!\big(\omega_{\max},\;\gamma\,\theta\big)\,\hat a\;}
$$

(`γ = gain_orient = 1.0` s⁻¹, `ω_max = 1.0` rad/s).

Two regimes select `R_des`:

- **Approach-axis only** (`gain_secondary ≤ 0`): pin only the tool `+Z` to the
  approach direction `z_des`, roll free (top-down grasps). Here
  `\hat a = (z_{\text{tool}}\times z_{des})/\lVert\cdot\rVert`,
  `θ = atan2(\lVert z_{\text{tool}}\times z_{des}\rVert, z_{\text{tool}}\!\cdot z_{des})`.
- **Full frame** (`gain_secondary > 0`): build the complete target frame
  `R_des = [\,x_\perp \;\; z_{des}\times x_\perp \;\; z_{des}\,]`, where `x_⊥` is
  the desired secondary (gripper open) axis re-orthogonalised against `z_des`.
  The geodesic of `R_des Rᵀ` is **one consistent shortest rotation**.
- **Yaw-free / pin tool +Y** (`align_tool_y = true`, used for the cylinder grasp):
  align only tool **+Y** to the bottle's long axis `y_des` (fingers stay ⟂ the
  bottle — a valid grasp) and leave the **yaw** about it free. Single-axis flow on
  +Y: `\hat a = (y_{\text{tool}}\times y_{des})/\lVert\cdot\rVert`. The freed yaw
  (approach azimuth) becomes a redundant DOF the arm spends on **column avoidance**
  — this is what let the full grasp run from the right side with the arm clear of
  the mast. Distinct from roll-free (which frees the wrong axis and tilts the
  fingers). The commit check then measures `angle(tool+Y, y_des)` only.

#### Why the geodesic and not two cross-products
An earlier version summed two independent per-axis alignment gradients
(`z_tool×z_des` *and* `x_tool×x_des`). Those are two attractors that **disagree
about the rotation axis** whenever the frame is far from aligned: they fight, and
on a large (~90°) "camera-up" roll the wrist flails and spins through a
singularity at the end. The single geodesic is the unique minimal rotation, so
it stays consistent for arbitrarily large errors.

### 4.5 The coordinated 6-DoF solve (the key fix)

The decisive correction. Position is a resolved-rate solve; orientation must be
solved **together with it**, not added on as a separate `J_ang^\top` gradient.

Stack the linear and angular tasks into one SE(3) twist and one geometric
Jacobian, and solve once:

$$
J_6 = \begin{bmatrix} J_{lin}\\ J_{ang}\end{bmatrix}\in\mathbb R^{6\times7},
\qquad
\xi^* = \begin{bmatrix}\dot x^*\\ \omega^*\end{bmatrix}\in\mathbb R^{6},
$$

$$
\boxed{\;\dot q \;=\; J_6^\top\big(J_6 J_6^\top + \lambda^2 I_6\big)^{-1}\,\xi^*\;}
$$

#### Why this is necessary (and the bug it fixed)
With a *separate* orientation gradient, the orientation joints move to fix `R`
**without regard to where they drag the tool position**. A 7-DoF arm doing a
6-DoF pose task has only **one** redundant DoF, so position and orientation
*cannot* be decoupled — they must be co-solved. In the broken version the
orientation term (geodesic angle up to `π`) demanded several rad/s of joint
velocity; the global velocity scaling (§4.7) then shrank the **entire** command,
throttling the position approach to ~25 % of commanded and letting orientation
hijack the direction (the tool flew *up* to reconfigure). Measured: 0.04 m/s vs
0.15 m/s commanded, ~27–30 s to converge, with a large initial excursion.

The coordinated solve makes both halves of the twist come out of the **same
operator**: they descend as one motion, neither dragging the other. Result:
**~1.5 s, straight descent, no fling — ~18× faster.** See §8.

**AIF reading.** This is simply the correct pragmatic term for a preference over
the **full pose** observation `f(q) ∈ SE(3)`, rather than treating position and
orientation as two independent preferences. The generative model has one
preferred outcome (a pose), so there is one coupled gradient.

### 4.6 Manipulability ascent in the null space (epistemic value)

The 7-DoF arm has one redundant DoF on a 6-DoF task. We spend it to keep the arm
**well-conditioned**, using the Yoshikawa manipulability
`μ(q) = √det(J Jᵀ)` (clamped at 0 against floating-point negatives):

$$
\dot q \mathrel{+}= \gamma_\mu\,\big(I - J_6^\top(J_6 J_6^\top+\lambda^2I)^{-1}J_6\big)\,\nabla_q \mu,
$$

with `∇_q μ` by central differences (14 extra Jacobian evals, ≈0.5 ms/cycle). The
soft projector `N = I − J_6^\top Q_6^{-1} J_6` confines this term to the
null space of the pose task, so it **cannot disturb the tool pose** — it only
moves the elbow.

**AIF reading — this is the epistemic / affordance-preserving term.** Staying in
high-manipulability configurations preserves the agent's *capacity to act*: it
keeps the arm away from singularities where its ability to realise future
preferred states collapses. In EFE language, it lowers *expected future* free
energy by maintaining controllability — a prior over configurations that
maximises future action affordance.

In practice the single redundant DoF is now spent on **explicit elbow placement**
(§4.6b) rather than manipulability — both cannot use the same DoF at once — so
`γ_μ = 0` while `γ_elbow > 0`. Manipulability remains available for postures where
no elbow preference applies.

### 4.6b Null-space elbow posture (redundancy as a posture prior)

We hold a human-like **"elbow-up"** posture: the elbow held high, the forearm
descending to the table. This is a preference biasing the elbow position `p_e(q)`
(joint_4 origin) along a unit direction `\hat b` (here **+Z, up**), realised through
the **same null-space projector** so it cannot disturb the tool pose:

$$
\dot q \mathrel{+}= \gamma_{\text{elbow}}\,\big(I - J_6^\top Q_6^{-1} J_6\big)\,
J_e^\top\,\hat b ,\qquad \hat b = \hat z .
$$

where `J_e` is the 3×7 elbow linear Jacobian (world frame). The projection
**self-limits**: when the elbow cannot rise further without moving the hand, the
projected term vanishes — i.e. "elbow as high as the redundancy allows." A constant
bias (not an error to a fixed point) avoids forcing an unreachable target and keeps
the elbow forward/high and naturally off the mast. **AIF reading:** a prior over the
redundant self-motion manifold — among all configurations realising the commanded
hand pose, prefer the elbow-up one. (An earlier *horizontal* "behind-the-mast"
target dragged the elbow toward the column and was replaced by this upward bias.)

### 4.6c Soft obstacle constraints (potential-field repulsions)

Two safety preferences keep the body clear of obstacles, each a **quadratic ramp**
that is exactly zero beyond a margin and grows as the obstacle is approached.
Added to `\dot q` **before** the velocity scaling, so they negotiate smoothly with
the task (a *soft* wall) and the command stays velocity-bounded.

*Whole arm ↔ column* — every movable joint `p_j(q)` for `j ∈ {3..7}` (skip the
shoulder/mount, which is always at the column) is repelled from the **finite**
column cylinder (axis `c_{xy}`, radius `r_c`, `z∈[z_lo,z_hi]`). With `c_j` the
closest point on the column segment to `p_j`, `d_j = ‖p_j-c_j‖`, clearance
`κ_j = d_j-r_c`:

$$
\dot q \mathrel{+}= \sum_{j}\gamma_{\text{col}}\Big(\tfrac{m-\kappa_j}{m}\Big)^2 J_j^\top\,\tfrac{p_j-c_j}{d_j},
\quad \kappa_j<m .
$$

Applied **directly** (not null-space): a single point on the elbow missed the
**wrist**, which was the link actually grazing the mast — so the repulsion now
covers the whole arm, and if the wrist nears the column the hand backs off.

*Hand ↔ table* (top at world `z_t`, clearance `d = z_{tool}-z_t`) — applied **directly**
(it *should* act on the hand, lifting it off the table):

$$
\dot q \mathrel{+}= \gamma_{\text{table}}\Big(\tfrac{m_t-d}{m_t}\Big)^2 J_{\text{lin}}^\top\,\hat z,
\quad d<m_t .
$$

Margins (`r_s=0.15` m, `m_t=0.06` m) are set below the working clearances (the
grasp sits ≈0.11 m above the table), so the constraints **do not fight a normal
grasp** — they only resist the elbow contacting the mast or the hand diving into
the table. **AIF reading:** soft barrier priors `\tilde p ∝ exp(−V_{obs})` whose
gradients are repulsive Cartesian velocities, mapped to joint space by `Jᵀ`.

### 4.7 Action bounds: velocity scaling and joint-limit repulsion

These are hard physical priors on the action and the configuration.

**Uniform velocity scaling** (preserves the inferred policy *direction*):

$$
s = \min_i \frac{\dot q_{\max,i}}{|\dot q_i|}\ \ (\text{only over } i:\,|\dot q_i|>\dot q_{\max,i}),
\qquad \dot q \leftarrow s\,\dot q .
$$

Per-joint clipping would distort the command direction (truncating the smaller
task when one joint saturates); uniform scaling keeps the **ratio** of position
to orientation intact. `\dot q_{max}` is the per-joint min of `max_joint_vel`
(0.87 rad/s, matching the Webots proto `maxVelocity = 0.8727`) and the URDF
limit. **AIF reading:** a prior on `‖\dot q‖_∞` (the actuator bound), applied so
as to preserve the MAP policy direction.

**Joint-limit repulsion** (a barrier prior over configuration), added *after*
scaling so the safety push acts at full strength:

$$
\rho_i(q_i) =
\begin{cases}
+g\big(\tfrac{m-(q_i-\underline q_i)}{m}\big)^2 & q_i-\underline q_i < m\\[4pt]
-g\big(\tfrac{m-(\overline q_i-q_i)}{m}\big)^2 & \overline q_i-q_i < m\\[4pt]
0 & \text{otherwise}
\end{cases}
$$

with margin `m = limit_margin = 0.10` rad and gain `g = limit_gain = 5.0`.
Continuous joints (`±∞` limits) contribute 0. **AIF reading:** a soft prior
preference `\tilde p(q) ∝ exp(−barrier)` keeping the configuration inside its
feasible set; its gradient is a repulsive velocity.

### 4.8 The full per-cycle law

$$
\dot q \;=\;
\underbrace{J_6^\top\big(J_6J_6^\top+\lambda^2 I_6\big)^{-1}
\begin{bmatrix}-v(\lVert e\rVert)\,\hat e\\[2pt] \min(\omega_{\max},\gamma\theta)\,\hat a\end{bmatrix}}_{\text{pragmatic: realise the preferred SE(3) flow}}
\;+\;
\underbrace{N\big(\gamma_\mu\nabla_q\mu + \gamma_{\text{elbow}}J_e^\top\hat b\big)}_{\text{redundancy: elbow-up posture / affordance prior}}
\;+\;
\underbrace{\gamma_{\text{mast}}(\cdot)J_e^\top\hat u + \gamma_{\text{table}}(\cdot)J_{\text{lin}}^\top\hat z}_{\text{soft obstacle repulsions}} ,
$$

then **scale** to `\dot q_{max}` (action bound) and **add** `ρ(q)` (joint-limit
barrier). The redundancy term and obstacle repulsions are added before scaling;
the joint-limit barrier after. This 7-vector of joint velocities is sent to the
`KinovaArm` proxy via `moveJointsWithSpeed`.

---

### 4.9 Migration to a constrained QP (proxQP) — in progress

The DLS solve (§4.5) **is** the unconstrained QP `min ½ q̇ᵀ(J₆ᵀJ₆+λ²I)q̇ − (J₆ᵀξ)ᵀq̇`,
so it generalises to Corke/Haviland's reactive QP (NEO) by adding hard inequality
constraints. Backend is `Controller.solver = "dls" | "qp"` (default `dls`;
`efe_gradient.cpp` keeps DLS as the fallback). Canonical NEO form:

$$
\min_{x=[\dot q;\,\delta]} \tfrac12 x^\top Q x + C^\top x
\quad\text{s.t.}\quad J\dot q = \nu+\delta,\; Ax\le b,\; x^-\le x\le x^+ .
$$

- `Q = diag(λ_q I, λ_δ I)` — **pure regularisers** (joint-velocity effort + slack).
- **`C = [−J_μ^\top;\,0]` — manipulability μ is the LINEAR term**, the *only*
  redundancy-shaping objective (`J_μ = ∂μ/∂q`). Our build keeps the pragmatic task in
  the objective as a least-squares term instead of a slacked equality — equivalent
  (the slack penalty folds into the task weight), and it reproduces DLS at step 1.
- **NEO has no elbow/posture term** — its redundancy resolution *is* μ-maximisation.
  Our "elbow into the back-right, off the mast" target is an application-specific
  addition; in the QP it is a **second linear term** `−η_e (∂φ_elbow/∂q)^\top \dot q`.
  We keep it because μ alone gives no control over *where* the elbow goes and this
  scene needs it clear of the mast — a deliberate deviation from canonical NEO.

Migration (each a separate, A/B-tested commit; DLS path kept as fallback) — **complete**:
1. ✅ **proxQP, objective only** → reproduces DLS (verified ~3e-9 over 200 trials).
2. ✅ **Joint-limit velocity dampers** as inequalities (Haviland & Corke): for a
   finite-limit joint inside the influence band `d_i`, `q̇_i ≤ η·(q^{max}_i−q_i)/d_i`
   (and lower), ramping to 0 at the wall — replaces the soft barrier (§4.7). Inactive
   outside the band ⇒ identical to DLS in normal operation. (Verified: far == DLS 3.6e-8;
   near-limit velocity hard-bounded 200/200.)
3. ✅ **Mast/table repulsions → collision velocity-damper inequalities**:
   `−n̂ᵀJ_point q̇ ≤ ξ·clr/d_i` (clamped ≥ 0 ⇒ q̇=0 always feasible: hard "no deeper
   penetration", not active retreat). Replaces the soft repulsions. (Verified: far ==
   DLS 4.4e-8; approach speed hard-bounded 200/200.)
4. ✅ **Full NEO form** — task as a slacked equality `J6 q̇ − δ = ξ`, `Q = diag(λ²I₇, I₆)`,
   **μ-ascent + elbow as the linear term** `g = −λ²·Σ gain·∇`, dampers as inequalities;
   the explicit null-space projector is dropped. Because `(J6ᵀJ6+λ²I)⁻¹ = (1/λ²)·N`, this
   weight makes the **free-space** solution exactly the old DLS-plus-null-space-projection
   (verified: NEO(g=0)==DLS 9.8e-10; NEO(g≠0)==DLS+gain·N·grad 1.1e-9), while the hard
   constraints now act on the **full** q̇ — so the step 2/3 guarantees are exact. The
   elbow term is *our* addition (canonical NEO has only μ); kept because μ alone gives no
   control over where the elbow goes and this scene needs it off the mast.

The live controller (`Controller.solver = "qp"`) is now this NEO QP; `"dls"` selects the
closed-form fallback. proxQP (`/opt/openrobots`, header-only) solves the 13-var QP each
20 ms cycle with no timing impact.

---

## 5. Parameters (with roles)

| Param | Symbol | Default | Role |
|---|---|---|---|
| `C_pos` | `Π_p` | `(1,1,1)` | position precision; isotropic ⇒ straight approach |
| `v_approach` | `v_app` | `0.15 m/s` | cruise speed cap of the position flow |
| `a_approach` | `a_app` | `0.15 m/s²` | constant deceleration; decel zone `= v²/(2a) ≈ 7.5 cm` |
| `gain_orient` | `γ` | `1.0 s⁻¹` | proportional gain of the angular speed ramp |
| `omega_max` | `ω_max` | `1.0 rad/s` | angular speed cap of the orientation flow |
| `gain_secondary` | — | `1.0` | >0 selects full-frame geodesic vs approach-axis only |
| `desired_approach` | `z_des` | — | preferred tool `+Z` (approach axis) |
| `desired_secondary` | `x_⊥` | — | preferred tool `+X` (gripper open axis) |
| `dls_lambda` | `λ` | `0.05` | DLS damping = singularity regulariser + effort prior |
| `gain_mu` | `γ_μ` | `0` | manipulability null-space ascent (epistemic); off while elbow-up bias is active |
| `gain_elbow` | `γ_elbow` | `1.5` | null-space elbow-posture bias gain (§4.6b) |
| `elbow_bias` | `b̂` | `(0,0,1)` | elbow push direction (null-space); +Z = elbow-up |
| `gain_mast` | `γ_mast` | `2.0` | elbow↔mast repulsion strength (§4.6c) |
| `mast_xy`, `mast_safe` | `c_xy`,`r_s` | arm-base xy, `0.15 m` | mast axis + activation radius |
| `gain_table` | `γ_table` | `2.0` | hand↔table repulsion strength (§4.6c) |
| `table_z`, `table_safe` | `z_t`,`m_t` | table top, `0.06 m` | table surface + activation margin |
| `max_joint_vel` | `\dot q_max` | `0.87 rad/s` | per-joint actuator bound |
| `limit_margin` | `m` | `0.10 rad` | joint-limit barrier activation distance |
| `limit_gain` | `g` | `5.0` | joint-limit barrier strength |

---

## 6. Where this sits relative to canonical Active Inference

**Faithful mappings**

- Pragmatic value ↔ Gaussian preference over the SE(3) tool-pose observation;
  the control law is gradient descent on its negative log (the instrumental EFE).
- Generalized coordinates ↔ the preferred-*flow* attractor (preference over
  position **and** velocity), which is why "arrive gently" is a model property,
  not a planner.
- Effort prior ↔ the DLS damping `λ` (Gaussian prior on action).
- Epistemic value ↔ manipulability ascent (preserve future affordance /
  controllability).
- Configuration priors ↔ the joint-limit barrier and the velocity bound.
- Hierarchy ↔ the outer phase machine (`SendingRestPose → Homing →
  WaitingForStart → ActiveEFE`) is a discrete policy layer selecting which
  preferred outcome the continuous layer descends — the standard discrete/continuous
  AIF stack. Switching the target (e.g. standoff → grasp) is just switching the
  prior the lower level minimises.

**Honest caveats (where it is AIF-*flavoured* engineering)**

- There is **no explicit policy posterior** `σ(−G)` over a set of policies and no
  Monte-Carlo rollout of `G`. This is a single-policy, continuous-time
  controller — the gradient-descent limit of EFE minimisation, not a tree search.
- The epistemic term is a **proxy** (manipulability), not a literally computed
  expected information gain integral.
- The DLS, uniform scaling, and barrier are standard robotics constructs; the AIF
  readings are principled *interpretations* that constrain how we extend the
  controller (see §9), not claims that the math was derived from a full variational
  bound.

This honesty is deliberate: the project's working rule is to keep extensions
inside the AIF paradigm (enrich the generative model) rather than reach for
plan-then-track control, *and* to be clear about which parts are exact and which
are motivated.

---

## 7. Why the two-stage approach target (standoff)

`compute_side_grasp_target()` aims the controller at a **standoff** point
`x* = c_body − z_des · d_standoff` (`d_standoff = 0.12 m`), where `c_body` is the
bottle's mid-body point. The approach axis `z_des` is the horizontal radial
direction to the bottle, projected perpendicular to the bottle's long axis so the
gripper stays normal to the body even when the bottle is tilted. The full grasp
frame is

$$
z_{\text{tool}} = z_{des}\ (\text{into the bottle}),\quad
x_{\text{tool}} = z_{bot}\times z_{des}\ (\text{horizontal finger axis}),\quad
y_{\text{tool}} = z_{bot}\ (\text{camera up}).
$$

Stopping at the standoff (not the bottle surface) means the preferred-flow
deceleration brings the tool to rest *beside* the bottle, ready for a separate
insertion/grasp sub-policy — the natural place for the planned grasp FSM
(Approach / Grasp / Lift) that will drive the already-wired `gripper_command_`.

---

## 8. Empirical validation

Tip trajectories were logged per cycle (`[tiplog] cycle,t,ee_xyz,tgt_xyz,dist,v_cmd,v_meas`,
enabled by `[Controller] tip_log` in `etc/config.toml`) and the run auto-armed via
`[Controller] auto_start`.

| Controller version | Time to arrive | Path | Tracked speed |
|---|---|---|---|
| Separate `J_ang^\top` orientation gradient | ~27–30 s | tool flies **up to z≈0.80 m**, then crawls down | ~0.04 m/s (throttled to ~25 % of command) |
| **Coordinated 6-DoF DLS** | **~1.5 s** | straight descent, **no fling** | brisk, monotonic, no overshoot |

The logs were decisive: they showed `v_cmd` pinned at 0.15 m/s while `v_meas`
sat at ~0.04 m/s, proving the slowness was the orientation term saturating the
joints and the uniform scaling throttling the whole command — **not** the
velocity profile. That ruled out "tune the profile" and pointed straight at the
coordinated solve.

Residuals to revisit: (i) measured tool speed peaks ~0.3–0.5 m/s early because
the angular task couples linear motion through the wrist lever arm — lower
`omega_max` (~0.5) for a gentler approach; (ii) small hover jitter at the goal —
candidate for a deadband.

---

## 9. Design history — the reasoning trail

1. **Bare EFE gradient** `\dot q = −α J_lin^\top Π_p e`. Works, but saturates and
   arrives at full speed → shoves the object. *Lesson: a preference over position
   alone has no opinion about arrival speed.*
2. **Preferred-flow attractor (position).** Added a preference over velocity
   (attractor at `(x*,0)`). First tried a **proportional** decel ramp → exponential
   crawl. Replaced with the **constant-deceleration `√` profile** → finite-time
   ease-in. Also dropped the `(4,4,8)` anisotropic `C_pos` to isotropic because
   the 2× `z`-weighting bent the path into an unnatural vertical-leg-first "L".
   *Lesson: enrich the model (velocity preference), don't bolt on a planner —
   this is the AIF-faithful path; an SE(3) screw + trapezoid would not be.*
3. **Coordinated 6-DoF solve.** Logging revealed the real bottleneck: the
   separate orientation gradient was saturating the joints and the uniform
   scaling throttled position to a crawl while the tool flew up to reconfigure.
   Stacking `[J_lin; J_ang]` and `[\dot x^*; ω^*]` into one DLS solve fixed it
   (27 s → 1.5 s). *Lesson: position+orientation on a redundant arm are one
   coupled pragmatic task, not two — there is one preferred pose, one gradient.*

---

## 10. File map

- `src/efe_gradient.{h,cpp}` — `EFEParams` and `efe_gradient_step()`; all of §3–§4.
- `src/specificworker.cpp` — `compute()` (the phase/cycle machine, per-cycle call,
  tip logging), `compute_side_grasp_target()` (§7).
- `src/kinematics.{h,cpp}` — Pinocchio FK/Jacobian wrapper (§2).
- `etc/config.toml` — `[Controller] auto_start`, `tip_log` diagnostics (§8).
