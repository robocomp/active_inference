# Thesis handoff — `room_concept`, 2026-08-12/13

**For:** the session maintaining Noé Zapata's thesis, chapter 8 `room-concept.tex`
(`/home/pbustos/drive/Tesis/Noe Zapata/tesis/`, files FLAT).
**From:** the `room_concept` implementation session.
**Commits:** `9eda7d4` (preintegration), `64146f6` (a correction to its own documentation),
`58f178a` + a follow-up (config arming). Live flag: `MotionPreintegration = true`.

Two pieces of work, one theoretical and one measurement:

1. **The motion factor's covariance is now derived by propagation instead of asserted.** This is the
   direct follow-through on a promise §8.4 (`sec:rc-motion`) already makes in prose — see §5 below,
   which is the single most important paragraph in this document for you.
2. **The SDF residual floor has been decomposed** into a systematic part and a sampling part, with the
   sampling part's $1/\sqrt{N}$ law verified. This is new quantitative content about the *information
   content of the observation model*, and it is thesis-worthy on its own.

There are also **three factual corrections to the existing chapter text** (§7). Please apply those
even if you use nothing else here.

---

## 0. Notation compliance — read this first

The 08-12 notation sweep is binding and I have written every equation below in the thesis dialect.
Symbols I need that are **already reserved**, and what I use instead:

| quantity | thesis symbol | never |
|---|---|---|
| pose (SE(2)) | `X` | `q` (belief only) |
| heading | `\phi` | `\theta` (concept params only) |
| objective / free energy | `\mathcal{L}` / `\mathcal{F}` | `L` (LiDAR only) |
| Hessian | `\mathcal{H}` | `H` (entropy only) |
| precision | `\Lambda` | `\lambda` (MPPI temperature only) |
| Huber knee | `\delta_H` | `\delta` (common-mode error) |
| room polygon | `\mathcal{R}` | `R` (rotation only) |
| window | `\mathcal{W}`, size `W` | — |
| measurement Jacobian | `J_z` | `G` (EFE only) |
| process-noise rate | `\varrho` | `q` |
| angular velocity | `\omega` | — |

**New symbols this work needs.** Please check each against `notation.tex` before use and add them to
the glossary; I have chosen them to avoid every collision in the audit, but I cannot see the glossary's
final state.

| new | meaning | why safe |
|---|---|---|
| `\varrho_{\mathrm{lat}}, \varrho_{\mathrm{lon}}, \varrho_\omega` | per-sample noise **densities** (m s$^{-1/2}$, rad s$^{-1/2}$) | `\varrho` is already the process-noise rate |
| `\mathbf{Q}` | diagonal density matrix built from the above | uppercase, unused |
| `A_i` | error-transport Jacobian of segment $i$ | unused |
| `B_i` | noise-injection Jacobian of segment $i$ | unused |
| `J_{s}` | scale Jacobian $\partial\Delta/\partial s$ | consistent with `J_z`; `J_{s_\omega}`, `J_{s_v}` |
| `s_\omega, s_v` | unknown multiplicative scale errors (dimensionless) | unused |
| `\boldsymbol{\Delta}` | preintegrated relative increment | ch.8 already writes `\Delta X^{\mathrm{odom}}` |
| `\mathbb{J}` | the SO(2) generator $\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$ | ⚠ **check** — `\mathbb{1}` is a known trap per the LaTeX conventions note; consider `\mathsf{J}` |

---

## 1. What the localiser is, stated as free-energy minimisation

§8.5 (`sec:rc-window`, `eq:rc-window`) already gives the objective. This section is the *derivation
that licenses calling it a free energy*, which the chapter currently asserts rather than shows. It is
short because the machinery is already in Appendix A.

### 1.1 The generative model

The window holds $W$ slots $\mathcal{W}=\{X_0,\dots,X_{W-1}\}$, $X_k=(x_k,y_k,\phi_k)\in SE(2)$,
newest last. Observations at slot $k$ are the scan $\mathcal{O}_k=\{\mathbf{p}_{k,i}\}_{i=1}^{N}$ in
the robot frame, plus any corner and object-anchor detections. The joint is a chain:

```latex
p(\mathcal{O}_{0:W-1}, \mathcal{W})
\;=\;
\underbrace{p(X_0)}_{\text{marginal of the past}}
\;\prod_{k=1}^{W-1} \underbrace{p(X_k \given X_{k-1})}_{\text{motion}}
\;\prod_{k=0}^{W-1} \underbrace{p(\mathcal{O}_k \given X_k)}_{\text{observation}} .
```

**The observation model is the conceptually interesting one and should be stated explicitly in the
chapter**, because it is not a standard measurement model. The model does not predict where a beam
will land; it asserts that *a returned point lies on a surface*, i.e. that the signed distance from
the point to the room boundary has expectation zero:

```latex
\mathrm{sdf}_{\mathcal{R}}\!\big(T(X_k)\,\mathbf{p}_{k,i}\big) \;=\; 0 \;+\; \epsilon_{k,i},
\qquad \epsilon_{k,i}\sim\mathcal{N}(0,\sigma_{\mathrm{sdf}}^2),
```

with $T(X_k)$ the rigid transform taking the robot frame to the room frame and
$\mathrm{sdf}_{\mathcal{R}}$ the signed distance to the polygon $\mathcal{R}$. The residual *is* the
prediction error, with no forward rendering step — which is exactly why the model is cheap and why the
Jacobian is available in closed form as $\nabla\mathrm{sdf}$.

Robustification: the residual passes through a Huber kernel with knee $\delta_H$, so

```latex
-\log p(\mathcal{O}_k \given X_k)
\;=\; \frac{1}{2\sigma_{\mathrm{sdf}}^2}\,\frac{1}{N}\sum_{i=1}^{N}
      \rho_{\delta_H}\!\Big(\mathrm{sdf}_{\mathcal{R}}\big(T(X_k)\mathbf{p}_{k,i}\big)\Big)
\;+\;\text{const}.
```

⚠ **Two implementation facts to state or to avoid mis-stating.** (i) The $1/N$ is real — the term is a
*mean* over points, not a sum, so the observation precision does not grow with scan density; that is a
deliberate choice and it interacts with §6 below. (ii) The corner/anchor factors use a kernel that is
**not** textbook Huber — it lacks the $-\delta_H^2/2$ offset, so in the saturated branch
$2\rho'(s)=u/2$ rather than $u$. This was a real bug (found by finite differences, not by any
convergence test) and the fix is in `room_obs_weights.h`. If the chapter writes a Huber, either write
*this* kernel or say the code's kernel is an unnormalised variant.

### 1.2 From variational free energy to the window objective

With a variational density $q(\mathcal{W})$ the free energy is the standard

```latex
\mathcal{F}[q] \;=\; \mathbb{E}_{q}\!\left[-\log p(\mathcal{O},\mathcal{W})\right] - \mathrm{H}[q].
```

Two reductions, and the chapter should name which one it is using:

- **Dirac family** $q=\delta(\mathcal{W}-\mathcal{W}^\star)$: the entropy term is constant and
  $\mathcal{F}$ collapses to $-\log p(\mathcal{O},\mathcal{W}^\star)$. Minimising $\mathcal{F}$ *is*
  MAP estimation. This is what `eq:rc-window` computes.
- **Gaussian family + Laplace**: expanding $-\log p$ to second order about the mode and optimising the
  covariance gives $\Sigma_q=\mathcal{H}^{-1}$ and the Laplace free energy. **Do not re-derive this** —
  Appendix A already has it as `eq:ai-laplace`, and the 08-12 conformance pass explicitly cut a
  duplicate derivation. Cite it.

So `eq:rc-window` is $\mathcal{F}$ under the Dirac reduction, and

```latex
\Lambda_{\text{post}} \;=\; \mathcal{H}\big|_{\mathcal{W}^\star}
\;=\; \sum_{\text{factors}} J^{\!\top}\Lambda\,J
```

is the Laplace posterior precision — which is what `compute_posterior_covariance` computes and what is
published to the graph and consumed by the controller's speed governor. **That identity is worth
stating plainly in the chapter**: the covariance every other agent trusts is the curvature of this
free energy at its minimum, not an independently maintained uncertainty.

### 1.3 Every term is a precision-weighted prediction error

This is the sentence that connects ch.8 to ch.4, and it is exact rather than rhetorical. Each factor
is $\tfrac12 r^{\!\top}\Lambda r$ with $r$ a prediction error and $\Lambda$ a precision, so

```latex
\frac{\partial\mathcal{F}}{\partial \mathcal{W}}
\;=\; \sum_{\text{factors}} J^{\!\top}\Lambda\,r ,
\qquad
\mathcal{H} \;=\; \sum_{\text{factors}} J^{\!\top}\Lambda\,J .
```

The Gauss–Newton/Levenberg backend solves $(\mathcal{H}+\lambda_{LM}\,\mathrm{diag}\,\mathcal{H})\,
\delta = -\,J^{\!\top}\Lambda r$, with $\lambda_{LM}$ updated by the Nielsen gain ratio. Robust kernels
enter as IRLS weights $u=2\rho'(r^2)$, which is the standard equivalence.

**A methodological point worth a paragraph**, because it is a genuine research-practice result rather
than an engineering note: the analytic backend was validated against the autograd one by *objective
identity* — both were made to evaluate the same $\mathcal{F}$ and agreed to $5.3\times10^{-6}$
relative — and the two bugs this uncovered were both **factor-of-two weight errors invisible to any
convergence test**, because scaling one factor's $\mathcal{H}$ and $b$ together leaves the step
unchanged while silently moving the minimum. A solver that converges beautifully to the wrong
stationary point is the failure mode; finite differences against the same loss is what catches it.

---

## 2. The motion factor before this work, and why its shape was wrong

§8.4 currently says the increment covariance carries three hand-wired modulations. Written out, the
implementation was

```latex
\Lambda_k^{-1} \;=\; \mathrm{diag}\big(\sigma_p^2,\ \sigma_p^2,\ \sigma_\phi^2\big),
```
```latex
\sigma_p = b_p + c_p\lVert\Delta\mathbf{p}\rVert \oplus c_{p\phi}|\Delta\phi|,
\qquad
\sigma_\phi = b_\phi + c_\phi|\Delta\phi| \oplus c_{\mathrm{slip}}|\Delta\phi|,
```

($\oplus$ = quadrature). Four defects follow from the *shape*, independent of how the constants are
tuned. These are the argument for the change and each is measurable:

1. **Diagonal and isotropic in $xy$.** A heading error at the start of an interval rotates *all* the
   translation that follows, so the true covariance has $\phi\!\leftrightarrow\!xy$ and
   $x\!\leftrightarrow\!y$ structure oriented along the direction of travel. The term $c_{p\phi}|\Delta\phi|$
   tries to carry that magnitude on the diagonal, where it cannot carry a *direction*.
2. **$\Delta t$ appears nowhere**, so every constant is update-rate dependent. This is not
   hypothetical: the same family of error had already cost a real failure, where a slip term was
   computed as $c\,(|\Delta\phi|/\Delta t)$ — an angular *rate* — and used as radians, inflating
   $\sigma$ by $1/\Delta t\approx16$ and the variance by $\approx260$ at a 62 ms interval, which
   drowned the measured prior at any $c$ and collapsed the fused prior onto the command channel.
3. **Random and systematic error are conflated.** $c_\phi|\Delta\phi|$ and $c_{\mathrm{slip}}|\Delta\phi|$
   both describe a *scale* error: fully correlated and sign-stable over the interval. Entering it as
   though it were per-sample noise mis-states how it grows with interval length — a correlated error
   grows like $T$, an independent one like $\sqrt{T}$ — and, more importantly, leaves nowhere for the
   estimator to *infer* it.
4. **A non-monotonic switch.** The floor is $\max$-selected by a stationary threshold that exceeds the
   moving base, so the model asserts *more* position uncertainty when barely moving than when moving
   fast. This is precisely the kind of hard gate ch.7/ch.8 argue against.

⚠ And the modulation §8.4 calls *online adaptation* is **switched off** — see §7.

---

## 3. Preintegration: the full derivation

The construction is Forster et al.'s on-manifold IMU preintegration specialised to SE(2) and to an
odometry stream. The problem it solves is the same: a high-rate stream between two states that must be
summarised into **one** factor, without either (a) instantiating a state per sample or (b) having to
re-integrate the raw stream every time a parameter estimate moves.

### 3.1 Setup

The interval $[t_a,t_b]$, of duration $T=\sum_i\Delta t_i$, is partitioned into $M$ segments. Segment
$i$ holds a constant body-frame velocity $(v_i^{\mathrm{lat}},v_i^{\mathrm{lon}},\omega_i)$ for
$\Delta t_i$. Let $\phi_a$ be the heading at $t_a$.

The accumulated increment after $i$ segments is $\boldsymbol{\Delta}_i=(\Delta x_i,\Delta y_i,\Delta\phi_i)$,
expressed in the **global (room) frame**. Segment $i$ contributes

```latex
\bar\phi_i \;=\; \phi_a + \Delta\phi_{i-1} + \tfrac12\,\omega_i\Delta t_i ,
\qquad
\mathbf{u}_i \;=\; \big(v_i^{\mathrm{lat}}\Delta t_i,\; v_i^{\mathrm{lon}}\Delta t_i\big),
```
```latex
\delta\mathbf{p}_i \;=\; R(\bar\phi_i)\,\mathbf{u}_i ,
\qquad
\delta\phi_i \;=\; \omega_i\Delta t_i ,
\qquad
\boldsymbol{\Delta}_i \;=\; \boldsymbol{\Delta}_{i-1} + (\delta\mathbf{p}_i,\ \delta\phi_i).
```

$\bar\phi_i$ is the **midpoint** heading; the mean is therefore the midpoint rule, which is what the
implementation already used before this work and which is left bit-for-bit unchanged.

⚠ **Be precise in the chapter about the parameterisation.** This is a *global-frame increment with
midpoint heading*, not a body-frame $SE(2)$ composition $\prod_i\exp(\Delta t_i\xi_i)$. The two agree
to second order in $\omega\Delta t$ and the choice was made so the new covariance is a drop-in for the
existing Euclidean motion residual $r_k=(X_k-X_{k-1})-\Delta X_k^{\mathrm{odom}}$. Writing the
manifold composition in the thesis while the code computes the midpoint rule would be a
misrepresentation; if you want the clean $SE(2)$ statement, mark it as the natural next step (§8).

### 3.2 Error transport, $A_i$

Let $\boldsymbol{\varepsilon}_{i-1}=(\boldsymbol{\varepsilon}_p,\varepsilon_\phi)$ be the error already
accumulated in $\boldsymbol{\Delta}_{i-1}$. The only channel by which it reaches segment $i$ is
$\bar\phi_i$, through $\Delta\phi_{i-1}$. With $\mathbb{J}=\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$
the SO(2) generator, so that $\partial_\phi R(\phi)=\mathbb{J}R(\phi)$,

```latex
\frac{\partial\,\delta\mathbf{p}_i}{\partial\varepsilon_\phi}
\;=\; \mathbb{J}\,R(\bar\phi_i)\,\mathbf{u}_i
\;=\; \mathbb{J}\,\delta\mathbf{p}_i
\;=\; \big(-\delta p_i^{y},\ \delta p_i^{x}\big)^{\!\top},
```

giving

```latex
A_i \;=\;
\begin{pmatrix}
1 & 0 & -\delta p_i^{y}\\
0 & 1 & \phantom{-}\delta p_i^{x}\\
0 & 0 & 1
\end{pmatrix}.
```

**This matrix is the whole argument of §2.1**: it is where every off-diagonal term comes from, and
setting $A_i=I$ is exactly what a diagonal model assumes.

### 3.3 Noise injection, $B_i$, and the density $\mathbf{Q}$

Model the stream as a Wiener process rather than as per-frame constants:

```latex
\mathbf{u}_i = (v_i\Delta t_i) + \mathbf{w}^p_i,\quad
\mathbf{w}^p_i\sim\mathcal{N}\!\big(0,\ \mathbf{Q}_p\,\Delta t_i\big),
\qquad
\delta\phi_i = \omega_i\Delta t_i + w^\phi_i,\quad
w^\phi_i\sim\mathcal{N}\!\big(0,\ \varrho_\omega^2\,\Delta t_i\big),
```

with $\mathbf{Q}=\mathrm{diag}(\varrho_{\mathrm{lat}}^2,\varrho_{\mathrm{lon}}^2,\varrho_\omega^2)$ a
matrix of **densities**: $[\varrho_p]=\mathrm{m\,s^{-1/2}}$, $[\varrho_\omega]=\mathrm{rad\,s^{-1/2}}$.

The yaw noise reaches the translation twice — directly through $\delta\phi_i$, and through $\bar\phi_i$
with coefficient $\tfrac12$ from the midpoint rule. Hence

```latex
B_i \;=\;
\begin{pmatrix}
\cos\bar\phi_i & -\sin\bar\phi_i & -\tfrac12\,\delta p_i^{y}\\
\sin\bar\phi_i & \phantom{-}\cos\bar\phi_i & \phantom{-}\tfrac12\,\delta p_i^{x}\\
0 & 0 & 1
\end{pmatrix},
```

and the propagation is

```latex
\boxed{\;\Sigma_i \;=\; A_i\,\Sigma_{i-1}\,A_i^{\!\top} \;+\; B_i\,\big(\mathbf{Q}\,\Delta t_i\big)\,B_i^{\!\top},
\qquad \Sigma_0=0. \;}
```

**Two consequences worth stating as results**, because they are exactly the defects of §2:

- $\Sigma_{\phi\phi}=\varrho_\omega^2 T$, so $\sigma_\phi=\varrho_\omega\sqrt{T}$: the random part grows
  like $\sqrt{T}$ and *shrinks* as the sample rate rises for fixed $T$. This is the honest statement of
  what a faster odometry stream buys — and it is the reason a two-rate cascade is not needed to exploit
  one.
- $\Delta t$ appears once, where the physics puts it. The units defect of §2.2 is not merely fixed but
  made **structurally impossible**, and the constants become rate-invariant properties of the sensor.

### 3.4 The scale term, and why $J_s$ is the interesting output

A scale error is a *bias*: one unknown constant, fully correlated across the interval. Entering it as
per-sample noise under-counts it by $\sqrt{M}$. Preintegration's answer — the same one Forster et al.
use for IMU bias — is to propagate its Jacobian.

Let $s_\omega$ scale the yaw channel, $\omega_i\mapsto(1+s_\omega)\omega_i$. Then
$\partial\delta\phi_i/\partial s_\omega=\delta\phi_i$ and
$\partial\bar\phi_i/\partial s_\omega=\partial\Delta\phi_{i-1}/\partial s_\omega+\tfrac12\delta\phi_i$, so

```latex
J_{s_\omega,i} \;=\; A_i\,J_{s_\omega,i-1} \;+\;
\begin{pmatrix}-\tfrac12\,\delta p_i^{y}\,\delta\phi_i\\[2pt] \phantom{-}\tfrac12\,\delta p_i^{x}\,\delta\phi_i\\[2pt] \delta\phi_i\end{pmatrix},
\qquad
J_{s_v,i} \;=\; A_i\,J_{s_v,i-1} \;+\;
\begin{pmatrix}\delta p_i^{x}\\ \delta p_i^{y}\\ 0\end{pmatrix},
```

the second for a scale $s_v$ on both translational channels. With independent priors
$s_\omega\sim\mathcal{N}(0,\varrho_{s_\omega}^2)$, $s_v\sim\mathcal{N}(0,\varrho_{s_v}^2)$, marginalising
them out to first order gives the interval covariance actually used:

```latex
\boxed{\;\Sigma \;=\; \underbrace{\Sigma_M}_{\text{random},\ \propto\sqrt{T}}
\;+\; \underbrace{\varrho_{s_\omega}^2\,J_{s_\omega}J_{s_\omega}^{\!\top}
     \;+\; \varrho_{s_v}^2\,J_{s_v}J_{s_v}^{\!\top}}_{\text{correlated},\ \propto\,T} \;}
```

This is exact for a constant unknown scale, and it generates the correct correlations for free because
$J_s$ is a *direction*, not a magnitude. Note $J_{s_\omega}$'s last component is $\Delta\phi$, so the
scale contributes $\varrho_{s_\omega}|\Delta\phi|$ to $\sigma_\phi$ — **linear in $T$, against
$\sqrt{T}$ for the random part.** That the two error classes scale differently is the content of §2.3
and it is now expressed in the model rather than folded into a shared constant.

**★ The point the chapter should make.** $J_s$ is also precisely what a scale *state* needs. Promoting
the scale from a covariance term to an inferred variable is: register $s$ in the solver's variable
index, give the motion factor a third variable slot with Jacobian $J_s$, and delete the outer product
from $\Sigma$. Nothing in the derivation changes. This is the concrete form of the argument §8.6
(`sec:rc-precision`) makes — that these modulations *ought to be inferred rather than tuned*.

### 3.5 Chaining, and the defect it repairs

The window is *strided*: a slot is admitted only after real motion, so one motion factor may span
several frames. For consecutive intervals $a$ then $b$ (with $b$ integrated from the heading $a$ ends
at), the combination is the same recursion at frame granularity:

```latex
\boldsymbol{\Delta}_{ab}=\boldsymbol{\Delta}_a+\boldsymbol{\Delta}_b,
\qquad
A=\begin{pmatrix}1&0&-\Delta y_b\\0&1&\Delta x_b\\0&0&1\end{pmatrix},
```
```latex
\Sigma_{ab}=A\,\Sigma_a A^{\!\top}+\Sigma_b,
\qquad
J_{s,ab}=A\,J_{s,a}+J_{s,b}.
```

The implementation previously accumulated $\Sigma_{ab}=\Sigma_a+\Sigma_b$ — i.e. $A=I$, described in
its own comment as "independent increments". That drops the transport term: an error in $a$'s heading
rotates the whole of $b$'s translation. **Measured** against one continuous integration, the chained
form is exact to $3.6\times10^{-7}$ while the additive sum is wrong by **7.8%** on the test trajectory
(0.4 rad over 0.5 s) and by **2.2–3.6%** at the stride limits actually configured (0.10 m / 0.15 rad),
reaching 8.8% if an interval spans twice the turn limit.

### 3.6 Relation to the marginalisation prior

$\mathcal{F}$'s first term, $p(X_0)$, is the FEJ+Schur marginal of everything dropped from the window:

```latex
\Lambda_{\mathrm{marg}} = \Omega - \Omega\,\Lambda_{00}^{-1}\,\Omega,
\qquad
\mu = X_1^{\star} - \Lambda_{\mathrm{marg}}^{-1} g_{\mathrm{marg}},
```

with Jacobians frozen at first estimates. **This is why the scale state is not yet implemented** and it
is a legitimate thesis point about fixed-lag consistency: a scale spans the whole window and beyond, so
marginalising the oldest pose would involve it, and freezing its Jacobian at a stale value is exactly
the mechanism that once made this prior ratchet monotonically and jail the estimate. The safe form is a
persistent variable kept *out* of the marginalisation with a random-walk prior.

---

## 4. The observation model's noise floor (the second result)

This is independent of preintegration and, in my judgement, the more quotable result.

### 4.1 The question and the design of the measurement

The early-exit decision variable is the mean absolute SDF at the odometry-predicted pose over the
current scan's $N$ sampled points:

```latex
m \;=\; \frac{1}{N}\sum_{i=1}^{N}\big|\,\mathrm{sdf}_{\mathcal{R}}(T(X^{\mathrm{pred}})\,\mathbf{p}_i)\,\big| .
```

With the robot stationary this quantity still fluctuates. The measurement design that settles the cause
is worth reporting because it needs no new instrumentation:

1. With zero odometry increment the predicted pose is **bit-identical** frame to frame — measured
   $\mathrm{std}=0.000000$, peak-to-peak $=0.000000$ over 2808 frames. So the pose cannot be the source.
2. The window's *older* slots hold point clouds captured once and never refreshed. Their SDF at the same
   frozen pose is **exactly constant**, digit for digit, for thousands of frames.
3. The current slot, with a fresh scan at the same frozen pose, fluctuates.

⟹ the fluctuation is entirely the new scan. Nothing in the estimator oscillates. The autocorrelation of
$m$ is $\le 0.02$ in magnitude at every lag 1–12, so it is white, not an oscillation.

### 4.2 The $1/\sqrt{N}$ law, verified

$m$ is a sample mean, so its sampling standard deviation should scale as $N^{-1/2}$. The window
provides two point budgets simultaneously — $N=405$ for the current slot, $N=200$ for older slots —
which makes the law testable for free:

| | $N$ | $\mathrm{std}(m)$ |
|---|---|---|
| current slot | 405 | 1.85 mm |
| older slots | 200 | 2.60 mm |

Observed ratio **1.41** against the predicted $\sqrt{405/200}=$ **1.42**. Back-solving,
$\sigma_{\mathrm{point}} = 1.85\,\mathrm{mm}\times\sqrt{405} \approx 37$ mm per point.

### 4.3 Decomposition of the residual floor

For a purely noise-driven residual with per-point $\sigma$, $\mathbb{E}|\epsilon|=\sigma\sqrt{2/\pi}
=0.798\,\sigma$. With $\sigma_{\mathrm{point}}=37$ mm that accounts for $\approx30$ mm of the observed
floor of $\approx87$ mm. The remaining $\approx57$ mm is **systematic model mismatch** — furniture,
doorway gaps, wall thickness that the polygon $\mathcal{R}$ does not represent — and no amount of
averaging removes it.

**This is a genuinely useful thing for the chapter to say**, because §8's gates are all expressed
relative to this floor: the floor is dominated by what the map omits, not by sensor noise, and the
*fluctuation* on top of it is a pure sampling effect with a known law. In the live configuration the
gate sits at $\approx10.5$ cm against a floor of 8.7 cm, so the sampling noise consumes $\approx1.9$ mm
of $\approx1.8$ cm of margin, and it can be shrunk predictably as $N^{-1/2}$ at the cost of more SDF
rows in the solve.

⚠ Two attributions I could **not** close and which should not be asserted: the split of the 37 mm
between (a) simulated per-beam range noise and (b) re-sampling of *different bearings* each frame,
because the index-stride subsample's index→bearing map shifts with the rotating head's phase and with
the height-band membership. The simulator's declared `noise 0.005` with `maxRange 100` does not by
itself land on 37 mm under either reading of that field, so it is a contributor of unknown weight.

---

## 5. ★ The paragraph in §8.4 this work discharges

§8.4 currently ends:

> *Each of these is a hand-wired instance of a higher-level quantity modulating a lower-level precision,
> and §\ref{sec:rc-precision} argues that this is exactly the pattern that ought to be inferred rather
> than tuned.*

That sentence now has a concrete follow-through, and the honest version of the story is a *partial*
one, which is more interesting than a triumphal one:

- Two of the three hand-wired modulations are **replaced by derivation**: the interval covariance now
  comes from a propagation whose only free parameters are per-sample noise densities with units and a
  measurement procedure, and the growth with rotation is generated by $A_i$ rather than asserted by a
  coefficient.
- The third — the scale — is **relocated but not yet inferred**: it moves from a hand-set coefficient to
  a rank-structured covariance contribution whose Jacobian is exactly what a state would need. The
  blockers are named and real (§3.6 marginalisation consistency; and it is unidentifiable in simulation,
  §6.4).
- ~8 configuration constants collapse into 5 densities and 2 scale priors, all of which are properties
  of the *sensor* rather than of the update rate.

**And the measured outcome is null**, which the chapter should say. That is the result.

---

## 6. Numbers, with provenance

Everything below is measured. Provenance is given so it can be re-checked; nothing here is estimated.

### 6.1 Validation of the covariance recursion (`tools/gn_selftest.cpp`)

Reference is **Monte Carlo**, deliberately: a wrong covariance still converges, it converges to a moved
minimum. So the noise the model claims is drawn, the nominal arithmetic is run with it, and the
empirical spread of $\boldsymbol{\Delta}$ is compared against the propagated $\Sigma$.

| check | result |
|---|---|
| $\Sigma$ vs 40 000-trial Monte Carlo | rel. Frobenius **0.0062** |
| — $\sigma_x$ | 0.0145 (analytic) vs 0.0146 (MC) |
| — $\sigma_\phi$ | 0.0354 vs 0.0354 |
| — $\rho(y,\phi)$ | **−0.189 vs −0.185** |
| off-diagonal is present at all | $|\rho(y,\phi)|=0.189$ (a diagonal model asserts 0) |
| $J_{s_\omega}$, $J_{s_v}$ vs central differences | 2.67e−4, 1.71e−4 |
| `chain()` vs one continuous integration | $\boldsymbol{\Delta}$ 2.7e−7, $\Sigma$ 3.6e−7, $J_s$ 6.7e−7 |
| — the additive (legacy) sum, same trajectory | **7.8% wrong** |
| rate invariance, same interval at 20 vs 100 Hz | $\sigma_\phi$ 0.03536 vs 0.03536; $\Sigma$ rel 1e−4 |

Pre-existing factor Jacobians, unchanged and still green: SDF 8.0e−3, motion 1.3e−5, corner 2.8e−5
(saturated 1.2e−4), anchor 2.8e−5 (saturated 1.2e−4), boundary 0, all factors 2.6e−4. Pose recovery
from 0.32 m / 0.12 rad → 0.2 mm in 10 LM iterations, 3.8 ms.

⚠ **A finite-difference gradient check is only valid in a band**, and this is a reusable methodological
result: at the optimum the relative error reads 4.3e−2 on *correct* Jacobians because both terms are
float noise; 5 mm/2 mrad off it reads 4.9e−3; 15 cm/60 mrad away it reads 5.9e−2 because a 1e−3 probe
re-assigns points to a different polygon segment. Three separate times in this work the *instrument*
was what needed fixing, not the solver.

### 6.2 Analytic covariance A/B at the live constants

Legacy / propagated, $\sigma$ in m and rad, measured-odometry channel. Strided rows compare against the
per-frame legacy covariance *summed* over the interval, which is what the implementation built.

| case | $\sigma_{xy}$ L / P | $\sigma_\phi$ L / P |
|---|---|---|
| parked, 50 ms | 0.0200 / 0.0200 | 0.0100 / 0.0100 |
| straight 0.5 m/s, 50 ms | 0.0120 / 0.0200 | 0.0100 / 0.0100 |
| pivot 1.0 rad/s, 50 ms | 0.0214 / 0.0200 | 0.0123 / 0.0105 |
| arc 0.5 m/s + 1.0 rad/s | 0.0142 / 0.0201 | 0.0123 / 0.0105 |
| strided arc, 0.5 s | 0.0447 / 0.0637 | 0.0388 / 0.0450 |
| strided straight, 0.5 s | 0.0379 / 0.0634 | 0.0316 / 0.0316 |

Two mechanisms explain the differences and both are the legacy shape being wrong: the legacy floor drops
discontinuously when $\lVert\Delta\mathbf{p}\rVert$ clears the stationary threshold (§2.4), and the
legacy rotation–position coupling inflates position even for a pivot in place, where a heading error has
no translation to rotate — the propagated form returns 0 there because the geometry is 0.

⚠ **The cross terms are nearly invisible at these constants**: $\rho(y,\phi)=-0.028$ on the strided arc,
against −0.189 in the self-test trajectory. The derived translation floor is 0.02 m per 50 ms $=0.089$
m s$^{-1/2}$ of position random walk *for a parked wheeled robot*, which is not a physical number — it
is a stabiliser raised to stop the motion term spiking — and it swamps the structure the change
introduces. **The algebra being right is not the same as the covariance being right.**

### 6.3 Controlled live A/B — the headline result is NULL

Same route, pose spans matched to 3 cm. Arm = propagated, 8124 frames; base = legacy, 3184 frames.
Wilson 95% CIs, so what is and is not resolvable is explicit.

| $\lvert\omega\rvert$ (rad/s) | legacy EE% | propagated EE% | $\Delta$ | resolvable? |
|---|---|---|---|---|
| <0.05 | 99.15 ±0.48 (n=1530) | 99.78 ±0.12 (n=5866) | **+0.63** | **yes** |
| 0.05–0.15 | 97.64 ±1.42 (n=467) | 97.45 ±1.47 (n=471) | −0.19 | no |
| 0.15–0.3 | 95.67 ±1.94 (n=439) | 96.09 ±1.71 (n=512) | +0.42 | no |
| 0.3–0.6 | 95.48 ±1.66 (n=619) | 96.27 ±1.17 (n=1020) | +0.80 | no |
| 0.6–0.9 | 94.57 ±4.06 (n=129) | 92.21 ±3.40 (n=244) | −2.36 | no |

Supporting: posterior $\Sigma_{\phi\phi}$ unchanged (parked 3.49e−4 → 3.30e−4; at 0.3–0.6 rad/s
5.66e−4 → 5.90e−4); marginalisation term max 5.76 vs 5.56, i.e. no ratchet either way; solve time
median 1.00 ms in both, p99 17.1 vs 12.6 ms.

**Proof the change was in force** (worth reporting as a verification method): the interval covariance
took the value $\varrho_p^2\,\Delta t$ *exactly*, for every observed frame interval —
2.48e−4 at 31 ms, 2.56e−4 at 32, 2.64e−4 at 33, 3.12e−4 at 39, 3.20e−4 at 40, 3.28e−4 at 41, 3.36e−4
at 42, 5.76e−4 at 72 ms. A per-frame constant *structurally cannot* produce a $\Delta t$-proportional
set of values.

### 6.4 What must not be claimed

- **Not** that preintegration improves localisation. Every moving band is inside its confidence
  interval. The one resolvable difference is +0.63 pp while parked, which is where the fused prior
  genuinely changed.
- **Not** that the cross terms do work at present constants ($\rho=-0.028$).
- **Not** anything about the motion term's cost: only 78 and 103 *optimised* frames exist across the
  two runs, ~10–30 per band, and the per-band medians are mutually contradictory. My own prediction for
  that quantity failed. Reading it needs a run with far more turning.
- **Not** any encoder scale calibration from simulation. The bridge publishes the supervisor's
  ground-truth yaw rate — there is no encoder in the simulated loop — and a prior bridge fix already
  took the odometry/posterior rotation ratio from 1.069 to 1.002. The residual scale that motivates the
  bias state is a *hardware* quantity.
- One transient on the legacy run (4 frames mid-turn, residual 0.205 m, self-corrected in ~200 ms) had
  no counterpart in the 2.5×-longer arm run, but **no recovery fired and no window flush occurred**;
  $n=1$, so it is not evidence.

---

## 7. Corrections to the existing chapter text

Please apply these regardless of what else you take.

1. **§8.4 says the motion-model parameters "are adapted online rather than fixed at configuration
   time". This is false as configured** — the learner is off, and it was measured and *rejected*:
   turning it on collapsed early exit while rotating from 91.6% to 68.5% and drove the fused weight on
   the encoder from 0.55 to 0.10. The reason is a good one for the thesis: its residual was a
   post-optimisation window-pair quantity that already contained the optimiser's own correction, so
   attributing it to odometry double-counted. Suggested replacement: describe the mechanism as
   *available but disabled*, and use the failure as the motivation for §3.4's scale-as-a-state — a
   variable whose credit assignment is done by the Hessian cannot double-count in that way.

2. **§8.3's paragraph "And a metric whose name lies" is only half right.** `sdf_mse` is *two different
   estimators depending on the code path*: a median absolute residual on optimised frames, and a **mean**
   absolute residual on early-exit frames — which are >98% of frames. For a half-normal the median is
   $0.674\sigma$ and the mean $0.798\sigma$, so comparing the statistic across the two paths carries an
   $\approx15\%$ systematic step, on top of the different sampling-noise coefficients
   ($\sqrt{\pi/2}$ for a median). The paragraph's warning should therefore be *stronger* and different:
   the name is wrong **and** the estimator is not stable across paths. (My §4 analysis is unaffected:
   both quantities I compared were means.)

3. **The motion residual in `eq:rc-window` is Euclidean**, $r_k=(X_k-X_{k-1})-\Delta X^{\mathrm{odom}}_k$
   with the angular component wrapped — which is what the chapter writes, correctly. Keep it that way,
   and if you introduce $SE(2)$ language in the new material, mark the manifold residual
   $\mathrm{Log}(\boldsymbol{\Delta}^{-1}X_{k-1}^{-1}X_k)$ explicitly as *not implemented* (§8).

---

## 8. Named future work (defensible as such, not as hand-waving)

1. **Calibrate the densities.** This is the load-bearing step, not a refinement: park the robot and take
   the sample variance of the odometry stream for $\varrho_p,\varrho_\omega$; one constant-velocity leg
   for $\varrho_{s_v},\varrho_{s_\omega}$. Until then the constants are a faithful translation of the old
   ones and the new structure is masked by a non-physical floor.
2. **The scale as a state**, with the marginalisation caveat of §3.6, on hardware.
3. **The manifold residual** $\mathrm{Log}(\boldsymbol{\Delta}^{-1}X_{k-1}^{-1}X_k)$, which must land in
   both optimiser backends simultaneously or the objective-identity check that validates the analytic
   backend stops measuring identity and starts measuring their disagreement.

---

## 9. Code and data pointers

| what | where |
|---|---|
| preintegration (header-only, fully commented derivation) | `room_concept/src/se2_preintegration.h` |
| wiring, both odometry channels | `room_concept/src/room_concept.cpp` (`integrate_odometry_over_window`, `integrate_velocity_over_window`, `compute_measured_odometry_prior`, `compute_odometry_prior`) |
| densities derived from the legacy constants | `room_concept/src/room_config.cpp` |
| the legacy model being replaced | `RoomConcept::compute_motion_covariance` |
| self-test (5 new checks) | `room_concept/tools/gn_selftest.cpp` → `make -C build gn_selftest && ../bin/gn_selftest` |
| solver, factor registry, objective | `room_concept/src/room_gn_solver.{h,cpp}` |
| config + the full A/B rationale | `room_concept/etc/config_apartamento.toml`, `MotionPreintegration` |
| **arm run** (propagated), 8124 frames | `room_concept/tmp/sdf_localizer/log_2026-08-13_12-05-32.csv` |
| **base run** (legacy), 3184 frames | `room_concept/tmp/sdf_localizer/log_2026-08-13_12-12-57.csv` |
| the stationary-noise analysis run | `room_concept/tmp/sdf_localizer/log_2026-08-13_10-30-47.csv` |

⚠ **Reading these logs.** Several columns are hardcoded zero on the early-exit path (`n_lidar` among
them), and the file has a history of ragged rows and header/writer order mismatches. Classify motion
with the `*_ingress` columns (`odom_rot_norm`, `odom_adv_norm`), which are populated on both paths.
And always count the motion population before treating a log as a baseline — one run I initially
offered as the legacy baseline turned out to be 24 434 of 24 458 frames *parked*, with zero frames
above 0.3 rad/s.
