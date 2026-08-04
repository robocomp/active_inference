# Path-tracking review — a 0.4 s-lagged differential drive on a known C² route

**Status: literature review, NOTHING IMPLEMENTED.** Commissioned 2026-08-04. Read §8 first if you
want the order of work; read §7 before quoting anything.

**Provenance.** Produced by a background literature agent against the measured plant below, then
partially verified against primary sources. Three classes of claim are mixed here and are marked
throughout: (a) **cited results** from peer-reviewed work, (b) **derivations** done for this review
(the loop-shaping numbers in §2 — assumptions stated, not independently checked), (c) **codebase /
documentation** claims about ROS stacks, which are not peer-reviewed. §7 lists everything thin.

**The plant it was asked about** (all measured, see [[pd-tracker-stage2]]):
differential drive, v_max 0.7 m/s, ω_max 0.8 rad/s, a_dec 1.0 m/s²; actuator identified from
(cmd_rot, meas_rot) as first-order lag τ = 0.213–0.236 s + pure delay 0.20 s, DC gain 0.89,
r² 0.94–0.95 on two independent laps ⇒ **total lag 0.41–0.44 s, lag-limited bandwidth 0.36–0.39 Hz**.
Control at ~20 Hz, data-driven off the lidar scan. Route: one C² arc-length-parameterised spline,
0.05 m spacing, analytic κ and heading, locally deformed by an elastic band against a live ESDF.
**Repeated laps of the same route.** Current tracker: pure pursuit on a 2.0 m carrot + Stanley-form
cross-track (gain 1.4, soft 0.30); cross-track rms 0.10–0.13 m. Speed = `v_max·cos^p(bearing)·dist`
→ EMA → Gaussian rotation brake → multiplicative safety gate.

---

## 0. Two findings that outrank the choice of controller

### (a) The carrot clip is a gain modulator, and it modulates the wrong way

`etc/config.toml` records `clip_carrot_to_reachable` binding on **75% of cycles**, achieved carrot
distance p50 **1.06 m**, under 0.5 m on **16%**. Pure pursuit's steering gain is `K = 2v/L_d²`, so
halving the lookahead **quadruples** the gain while halving the lead. Computed operating points at
v = 0.7 m/s (**derivation, §2 assumptions**):

| effective L_d | ω_c (rad/s) | phase margin |
|---|---|---|
| 2.0 m (the parameter) | 0.62 | ≈ 45° |
| 1.06 m (the measured p50) | 1.14 | ≈ 32° |
| 0.5 m (16% of cycles) | 2.23 | **≈ 6°** |

A safety-motivated carrot pull-in therefore raises loop gain exactly when the robot is in a tight
spot, and on one cycle in six the loop is effectively marginally stable. This would explain both why
the weave survived changing `CarrotLookahead` 2.0 → 1.1 (the clip was already setting it) and why it
survived replacing MPPI with the PD tracker (both steer at the same clipped carrot).
**Decoupling steering gain from geometric lookahead is worth more than any tracker swap.** → R3.

### (b) The dominant disturbance is measurement, not model

Pose jumps correlate +0.78 with the weave and lead `cmd_rot` by 0.2 s. They sit **outside** the loop's
usable band, so they cannot be rejected at any gain — only *not amplified*. Structural answer: close
the tracker on a continuous, drift-tolerant local frame (odom) rather than the jumping global one.
REP-105 (Meeussen 2010, <https://www.ros.org/reps/rep-0105.html> — *ROS standards doc, not
peer-reviewed*); quantitatively Nubert et al., "Holistic Fusion", [arXiv:2504.06479](https://arxiv.org/pdf/2504.06479),
which measures pose-jump-induced jerk and shows odom-frame estimates suppress jumps entirely.
Since the cross-track loop has no usable bandwidth above ~1 rad/s, low-passing the *correction* at
~1 Hz costs the controller nothing and removes the excitation.

---

## 1. Ranked recommendations

### R1 — Preview-shifted curvature feedforward with actuator inversion **(do this first)**

```
s_ff  = v · t_ff                        t_ff ≈ 0.40–0.45 s  (SWEEP IT, from 0 upward)
ω_ref = v · κ(s + s_ff)
ω_ff  = [ ω_ref + τ · dω_ref/dt ] / 0.89          τ ≈ 0.225 s
dω_ref/dt = v̇·κ + v²·κ'(s)              (analytic from the spline)
ω_cmd = ω_ff + feedback(e_y, e_θ)
```

**Why it fits.** Delay in a *feedforward* channel is not a bandwidth cost — it is a pure time shift,
and the route is known in advance, so you can shift non-causally for free. The unicycle is
differentially flat in (x, y), so `ω = v·κ` **is** the exact nominal input (Fliess, Lévine, Martin &
Rouchon, *Int. J. Control* 61(6):1327–1361, 1995, [DOI 10.1080/00207179508921959](https://doi.org/10.1080/00207179508921959));
using *future* reference values to cancel plant phase is ZPETC (Tomizuka, *ASME JDSMC* 109(1):65–68,
1987, [DOI 10.1115/1.3143822](https://asmedigitalcollection.asme.org/dynamicsystems/article-abstract/109/1/65/426747)).
The `1/0.89` and `τ·dω/dt` terms invert the model already identified at r² 0.94–0.95.

**Direct experimental precedent — a near-clone of this system.** Seiffer, Frey & Gauterin,
*Vehicles* 5(2):615–636, 2023, [DOI 10.3390/vehicles5020034](https://doi.org/10.3390/vehicles5020034):
same Stanley-form law with a `k_soft` denominator, same 0.20 s identified delay, RMS cross-track
0.106 m, same *outward-of-curve* bias. Sweeping `t_ff` in 0.1 s then 0.01 s steps, the optimum landed
at **t_ff = 0.20 s — exactly their measured delay**:

| | RMS cross-track | max |
|---|---|---|
| Stanley | 0.106 m | 0.262 m |
| preview-shifted FF | **0.033 m (−69%)** | **0.084 m (−68%)** |

Verified in the PDF: the enhanced and plain laws are **identical when `t_ff = 0` or when curvature is
constant**. The error therefore lives at curvature *transitions*, exactly where feedforward acts and
where extra feedback gain cannot.

**The `1/0.89` deserves its own test.** An 11% actuator gain deficit means the robot under-turns and
must build a standing cross-track error to command the extra rate — a steady-state **outward** offset
of order `0.062·L_d²/R` m (**derivation**), ≈ 6 cm at L_d = 2.0 m, R = 4 m. Seiffer et al. observed the
same signature: *"the position … follows the path with a deviation that is on the outside of the
curve."* **Test before fixing:** plot signed cross-track against κ. Consistently outward and scaling
with curvature ⇒ this is it, and the fix is one division.

**Cost** under a day. **Augments**, removes nothing.
**Failure mode:** over-shifting turns in early and cuts to the *inside* — the sign flips. Sweep from 0
and stop at the RMS minimum; do not set `t_ff` to the identified lag on faith. Also `v` and `θ_ss` are
assumed constant over `t_ff` (Seiffer et al. state this); 0.29 m of travel at 0.7 m/s, fine here,
wrong if speed is changing hard.

> ### ★★ R1(a) WAS BUILT AND IT FAILED — 2026-08-04. Do not rebuild it standalone.
>
> `ω_ff = −1.0·v·κ(s)` added to `cmd_rot`, κ sampled at the robot's own projection, sign taken from
> the data (`cmd_rot = −0.535·(v·κ)`). The term fired on **100%** of driving cycles, so this was a real
> test. Matched unsaturated bins (|v·κ| < 0.4), before → after:
>
> | | before | after |
> |---|---|---|
> | rms cross-track | 7.94 cm | **9.83 cm** |
> | slope a | +0.100 | **+0.123** |
> | cmd_rot / (v·κ) | −0.637 | **−0.349** |
>
> `mission_cost` 9.591; on the two-thirds comparable to earlier runs (smooth_lin + dev_norm) 5.901
> against a 4.49–5.86 baseline — worse than all five prior tours. p05 clearance 0.104, below every one.
>
> **★THE TELL.** Adding a feedforward of −1.0·v·κ should make the TOTAL command *bigger*. It got
> **smaller**. The feedback opposes the new term by more than one-for-one — the robot is being pushed
> off the path by the term meant to hold it on.
>
> **★CAUSE: pure pursuit ALREADY supplies this implicitly.** On a curve the carrot sits off-axis by
> exactly the angle that produces the right steering, so `Kp·carrot_angle` *is* a feedforward. An
> explicit `v·κ` on top DOUBLE-COUNTS it: the robot over-turns into the corner (slope a rises = further
> inside, which is what was measured) and the cross-track term then fights its own controller.
>
> Secondary defect, real but not the main one: the term used the PRE-brake, pre-gate speed. `max_adv`
> is correctly clamped by the route speed limit, but smoothing, the Gaussian brake and the safety gate
> cut speed further downstream, so on 4% of cycles `|ω_ff|` ALONE exceeded `max_rot` (peak 1.83 vs 0.8)
> and the clamp discarded the feedback entirely. Saturation 6.0% → 8.6%.
>
> **★★CONSEQUENCE FOR THE PLAN: there is no standalone R1(a).** A feedforward must REPLACE the
> carrot-bearing term, not stack on it — which is R3's law exactly (`ω = ω_ff + k_y·e_y + k_θ·e_θ`,
> geometric term gone, gains from the plant rather than the lookahead). The prerequisite framing was
> right; doing it as an addition to pure pursuit was wrong. ⚠n=1 per arm and the laps differed in
> length (131.6 s vs 118–121 s), but the direction is consistent across four independent measures and
> the causal story explains the sign.
>
> Code fully removed 2026-08-04 at the user's request — no dead flag, no dormant term. This note is the
> record.

### R2 — One speed profile, one optimisation, with the safety cut **inside** it

Delete the chain `cos^p(bearing) × dist → EMA → Gaussian brake → multiplicative gate`. Replace with a
single profile over a receding ~5 m window in the variable **`b(s) = v²(s)`**, in which every existing
limit becomes **linear**:

- lateral accel: `κ(s)·b ≤ a_lat,max`
- yaw rate: `b ≤ (ω_max/κ(s))²`
- accel/decel feasibility: `|b'(s)| ≤ 2·a_max`  ← *what the min-over-lookahead approximates and gets wrong*
- **clearance**, from the ESDF, as another limit in the same field: the DWA admissibility condition
  `v ≤ √(2·a_brake·d)` becomes `b ≤ 2·a_brake·d_clear(s)`

A forward–backward pass (or LP/SOCP) then gives the maximum feasible profile in O(n).

Verscheure et al., *IEEE TAC* 54(10):2318–2327, 2009, [DOI 10.1109/TAC.2009.2028959](https://doi.org/10.1109/TAC.2009.2028959);
cleanest exposition **Lipp & Boyd**, *Int. J. Control* 87(6):1297–1311, 2014,
[DOI 10.1080/00207179.2013.875224](https://doi.org/10.1080/00207179.2013.875224) ·
[PDF](https://web.stanford.edu/~boyd/papers/pdf/speed_opt.pdf). Two-pass reachability: TOPP-RA,
Pham & Pham, *IEEE T-RO* 34(3):645–659, 2018, [DOI 10.1109/TRO.2018.2819195](https://doi.org/10.1109/TRO.2018.2819195),
code <https://github.com/hungpham2511/toppra> (MIT) — single-digit ms for hundreds of grid points.
TOPP-RA does **not** support jerk (structural, not an implementation gap); for jerk in the program use
the LP of Shimizu, Horibe, Watanabe & Kato, ICRA 2022, [arXiv:2202.10029](https://arxiv.org/abs/2202.10029).

**Why the current chain produces exactly the measured roughness.** Three independent defects:

1. **A multiplicative cut applied after a smoother re-injects the discontinuity the smoother
   removed.** The gate being 79% of command roughness is the *expected outcome of the layering*, not a
   tuning failure. Stated in 1990: Dahl & Nielsen, *IEEE T-RA* 6(5):554–561,
   [DOI 10.1109/70.62047](https://doi.org/10.1109/70.62047) — when a limit binds, **rescale time along
   the path; do not multiply the command**, because multiplying breaks the path-following invariant.
2. **The gate is a quantised ladder with no hysteresis.** A one-cycle pulse cutting 0.1 m/s at 20 Hz
   commands |Δa| ≈ 2 m/s² — twice the stated `a_dec` — and jerk of order 40 m/s³, one to two orders
   above every comfort figure below. The 0.22 s actuator smears it at the wheels, which is why it does
   not *look* catastrophic; it means the actuator spends its bandwidth chasing pulses carrying no
   information.
3. **An EMA bounds nothing** — no `a_max`, no `j_max` in its definition — and since `v` depends on
   realised pose it sits *inside* the loop, costing `arctan(ω_c·T)` of phase margin. At α = 0.8 / 20 Hz
   its time constant is 0.22 s, comparable to the plant lag. Contrast Autoware, which low-passes
   steering at **3 Hz Butterworth**, ~20× above crossover: noise filtering, not command shaping.

**Replace the EMA** with a jerk-limited retarget: Ruckig (Berscheid & Kröger, RSS 2021,
[arXiv:2105.04830](https://arxiv.org/abs/2105.04830), MIT, microsecond solves) or an explicit S-curve.
**Comfort bounds — the literature disagrees:** de Winkel, Irmak, Happee & Shyrokau, *Applied
Ergonomics* 106:103881, 2023, [DOI 10.1016/j.apergo.2022.103881](https://doi.org/10.1016/j.apergo.2022.103881)
report a 50% comfort threshold near 0.6 m/s³ lateral jerk but *also* find higher jerk in shorter
pulses rated **more** comfortable, so a single scalar bound is a simplification. ISO 15622 caps
average negative jerk at 2.5 m/s³. **Design band 0.9–2.0 m/s³, `a_max ≤ 1.0 m/s²`.**

**Cost** 2–3 days (forward–backward), a week for LP/SOCP.
**Failure mode:** window seams — recomputing over a sliding window can jump at the boundary as the
band deforms. Fix is Admissible Velocity Propagation, which TOPP-RA gives naturally: carry the
reachable terminal-velocity *interval* into the next window as the initial condition.

### R3 — Decouple steering gain from lookahead; add an inner yaw-rate loop

```
ω = ω_ff(s + s_ff) + k_y·e_y + k_θ·e_θ + k_d,yaw·(ω_ref − ω_meas)
```
with `k_y, k_θ` set from a chosen (ζ, ω_n) against the identified plant — **not** from carrot geometry.
De Luca, Oriolo & Samson, in Laumond (ed.), *Robot Motion Planning and Control*, LNCIS 229, Springer
1998, pp. 171–253, [PDF](https://www.di.ens.fr/jean-paul.laumond/promotion/chap4.pdf); Kanayama,
Kimura, Miyazaki & Noguchi, ICRA 1990, pp. 384–389, [DOI 10.1109/ROBOT.1990.126006](https://doi.org/10.1109/ROBOT.1990.126006).

**Why.** Pure pursuit pins the lead contribution near 60° regardless of lookahead. Decoupling lets you
pick a ~4 s lead independently of gain, moving ω_c **0.62 → ~1.5 rad/s at the same 45° margin (2.4×)**.
An inner loop on measured yaw rate reduces the effective τ seen outside (0.225 → ~0.1 s) for another
~40%: **ω_c ≈ 2.1 rad/s**, at which point the 0.20 s dead time is the only thing left. That damping
term is Hoffmann et al.'s `k_d,yaw(ψ̇_ref − ψ̇)`, tuned to 0.150 s on hardware by Seiffer et al.
(Hoffmann, Tomlin, Montemerlo & Thrun, ACC 2007, pp. 2296–2301,
[PDF](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)). It also removes §0(a):
the ESDF carrot clip stops modulating loop gain.

**Correctness note on the current Stanley term.** Stanley is provably stable because cross-track is
measured at the **front axle**, co-located with the steered wheels — that collocation is the point
(Snider, CMU-RI-TR-09-08, 2009, [PDF](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
— *tech report*; Hoffmann et al. 2007). A differential drive has no front axle: from ω to lateral
position there is a **double integrator**, so the proof does not transfer, and a proportional-on-`e_y`
term with no lead consumes phase margin that comes entirely from the pure-pursuit lead. The
autocorrelation diagnostic used to justify raising `PdCrossTrackGain` detects *oscillation*, not
*margin erosion* — not the same thing until you are already unstable.

**Cost** ~2 days. **Failure mode:** a 4 s lead means the derivative term dominates ~6×, amplifying
heading/pose noise. **Contingent on §0(b)** — do the pose frame first or this makes the weave worse.

### R4 — Arc-length-indexed ILC on the feedforward channel, learned between laps

Keep the 20 Hz loop bit-identical; between laps update an additive feedforward `u_j(s)` indexed by
**arc length**, from the previous lap's error shifted upstream by the lag distance.

**Why this beats the bandwidth ceiling rather than respecting it.** ILC is non-causal *in the trial
domain* — it may use future values of the previous lap's error — so the delay becomes a **shift**, not
a constraint. 0.20 s at 20 Hz is exactly **4 samples**, so the optimal learning filter is `L = k·z^{+4}`.
With that shift the residual is a pure first-order lag whose phase asymptotes to −90° and never
reaches it, so monotonic convergence `|Q(1 − GSL)| < 1` holds at all frequencies for small k — whereas
*without* the shift a P-type ILC must be Q-filtered below ~0.65 Hz, barely better than feedback.
Oomen, "Learning for advanced motion control", IEEE AMC 2020, pp. 65–72,
[arXiv:2004.11017](https://arxiv.org/abs/2004.11017).

**Applied precedents.** Kapania & Gerdes, ACC 2015, pp. 2753–2758,
[arXiv:1902.00611](https://arxiv.org/abs/1902.00611) — multi-lap ILC on an Audi TTS, explicitly
**arc-length indexed** via an interpolated κ(s) table, PD-ILC (k_p 0.02, k_d 0.4 rad/m), norm-optimal
variant penalising lap-to-lap *change* (`S = 100·I`), **2 Hz zero-phase Q-filter**, RMS lateral
0.12 → 0.08–0.09 m in 2–3 laps at 0.8 g — a starting error in your band. Ostafew, Schoellig & Barfoot,
IROS 2013, pp. 176–181, [DOI 10.1109/IROS.2013.6696350](https://doi.org/10.1109/IROS.2013.6696350) —
"visual teach and repeat, repeat, repeat", >600 m on 50 kg and 160 kg field robots at up to 3× nominal
speed. Theory for learning in **spatial** coordinates: Consolini & Verrelli, *Automatica*
50(7):1867–1874, 2014, [DOI 10.1016/j.automatica.2014.05.002](https://doi.org/10.1016/j.automatica.2014.05.002).

**Why arc length, not time.** Lap time varies, so a temporal internal model is wrong. Position-domain
repetitive control exists (Mahawan & Luo, *Int. J. Control* 73(1):1–10, 2000) but requires
reformulating an LTI system in a spatial coordinate, which makes it nonlinear. **Indexing an ILC table
by `s` sidesteps this** — what Kapania & Gerdes actually did.

**Handling the elastic band.** A raw `u(s)` table is bound to one reference; deform the route and ILC
is *worse* than feedback alone. Fix: basis-function ILC — learn coefficients of (κ, κ̇, v·κ) rather
than a raw signal (Bolder & Oomen, *IEEE TCST* 23(2):722–729, 2015,
[DOI 10.1109/TCST.2014.2327578](https://doi.org/10.1109/TCST.2014.2327578); van Zundert, Bolder &
Oomen, *Automatica* 67:295–302, 2016, [DOI 10.1016/j.automatica.2016.01.026](https://doi.org/10.1016/j.automatica.2016.01.026))
plus a per-lap forgetting factor ~0.95 (Arimoto, Naniwa & Suzuki, ICRA 1991, pp. 728–733).

**Cost** 3–5 days. **Failure modes** (documented, and Kapania & Gerdes report the first two): bad
learning transients (asymptotic ≠ monotonic), amplification of *non-repeating* disturbances — your
lidar pose noise and ESDF flicker are exactly that class — and initial-condition sensitivity.
Mitigations: the `‖u_j − u_{j−1}‖` penalty, a spatial Q-filter (start ~1–1.5 m cutoff wavelength —
*inferred from their 2 Hz temporal filter, not published*), and a **freeze-on-regression guard**: if
lap RMS rises two laps running, stop learning.

**On "wary of in-loop adaptation" — this is the crux.** Nothing adapts inside the 50 ms cycle. The
health check is two scalars per lap: `‖e_j‖` non-increasing, `‖u_j − u_{j−1}‖` contracting
geometrically (that ratio *is* the empirical convergence rate). And it is **bit-for-bit reproducible
offline** — replay a logged lap on a laptop and recompute the next feedforward. No in-loop adaptive
scheme has that property.

### R5 — Delay-lifted TVLQR or Autoware-pattern linear MPC (only if R1–R4 fall short)

Frenet error state augmented with the actuator: `[e_y, e_θ, ω_act, u_{k−1..k−4}]` — 7–8 states, because
**0.20 s ÷ 0.05 s = exactly 4**, so the delay lifts into integer states. **Do not use Padé**: it exists
to embed delay in continuous time and injects RHP zeros that themselves limit bandwidth, buying
nothing when the delay is an integer number of samples. Since the route is fixed and repeated, the LTV
Riccati recursion can be solved **offline, once, backwards along the spline** and stored as `K(s)` —
zero online compute.

**Validating data point.** Autoware's production `mpc_lateral_controller` defaults are
`input_delay = 0.24 s` and `vehicle_model_steer_tau = 0.30 s` — a *larger* total lag than yours — and
compensates in three layers: the lag is a **model state**; the actuator state is predicted from the
buffered command history; `use_delayed_initial_state = true` replays buffered commands through the
discrete model to hand the QP a delay-free `x₀`. Plus curvature feedforward and a 3 Hz Butterworth
output filter. ([design doc](https://autowarefoundation.github.io/autoware_universe/main/control/autoware_mpc_lateral_controller/)
— *documentation + production code; no peer-reviewed evaluation of the delay compensation, its only
citation is Snider 2009*). **Your plant is strictly inside the envelope a shipped stack handles.**

**Cost** 1–2 weeks; compute is not the constraint (linear MPC at N = 15 measures 11.8–12.5 ms/QP on a
real diff-drive; acados SQP-RTI well under that for 3–4 states). Model fidelity is.
**Failure mode:** LQR's guaranteed 60° phase margin **evaporates** under output feedback — Doyle,
"Guaranteed margins for LQG regulators", *IEEE TAC* 23(4):756–757, 1978. Your ESDF/band/localisation
chain is effectively an estimator; do not bank the 60°.

---

## 2. What the measured 0.41 s rules out

**(Derivation — assumptions: small-angle Frenet error dynamics, `ω → e_y` as `v/s²`, pure pursuit
linearised as `(2v/L_d²)(1 + (L_d/v)s)`, v = 0.7, τ = 0.225, θ = 0.20. Check against a measured chirp
before betting on the exact numbers; the qualitative conclusions are robust to the details.)**

Plant `G(s) = 0.89·e^{−0.20s}/(0.225s + 1)`. The lateral channel adds `v/s²` (`ė_y = v·θ_e`,
`θ̇_e = ω − vκ`) — a **double integrator** contributing −180° by itself, so *all* phase margin must come
from lead, and lead is capped by the delay. Because pure pursuit's gain ∝ 1/L_d² while its lead time
∝ L_d, the lead contribution at crossover is pinned near 60° whatever you choose:

```
ω_c ≈ 1.78·v / L_d          PM ≈ 60° − atan(0.225·ω_c) − 11.46·ω_c     [ω_c in rad/s]
```

Every rad/s of bandwidth costs ~11.5° to dead time alone. Hence:

- **Pure pursuit can never exceed 60° phase margin on this plant, at any lookahead or gain.**
  45° PM ⇒ ω_c ≈ 0.62 rad/s (0.10 Hz); 30° PM ⇒ 1.25 rad/s. The 2.0 m configuration sits at
  0.62 rad/s / 45° — essentially the optimum. **Nothing left in tuning.**
- Hard wall from the delay alone: ω_c ≲ 2.5 rad/s (≈ 0.4 Hz), corroborated by the 60°-guaranteed-PM
  route `(π/3)/0.41 s = 2.55 rad/s`. Skogestad & Postlethwaite, *Multivariable Feedback Control*, 2nd
  ed., Wiley 2005, Ch. 5; Åström & Murray, *Feedback Systems*, 2nd ed., Princeton 2020, Ch. 11–12,
  [free PDF](http://www.cds.caltech.edu/~murray/books/AM05/pdf/fbs-limits_18Aug2019.pdf) — the
  `ω_c ≲ 1/θ` statement is a **rule of thumb**; no verbatim text was retrievable.
- **Cross-track settles with a ~1 s time constant (~0.7 m of travel).** Anything above ~1 rad/s —
  including the pose jumps — is unrejectable by feedback, full stop.
- Canonical primary source for geometric trackers *with loop delay*: Ollero & Heredia, IROS 1995,
  vol. 3, pp. 461–466, [DOI 10.1109/IROS.1995.525925](https://doi.org/10.1109/IROS.1995.525925),
  extended in Heredia & Ollero, *Advanced Robotics* 21:23–50, 2007,
  [DOI 10.1163/156855307779293715](https://doi.org/10.1163/156855307779293715) — analysed for straight
  and constant-curvature paths, exactly this spline decomposition. **Neither retrievable online**; the
  closed-form delay–lookahead–velocity stability condition is the single most relevant unretrieved
  result here and is worth a library request.

| Ruled out | Reason |
|---|---|
| **Sliding mode** | A relay loop with L = 0.41 s limit-cycles at ≈ 1/(4L) ≈ 0.6 Hz. Widening the boundary layer until it stops degrades SMC into a saturated P controller. Actively contraindicated. |
| **Feedback linearisation** (off-axis point) | Gain ∝ 1/b: tight tracking ⇒ crossover far above 2.5 rad/s. Heading becomes unregulated zero dynamics; dynamic FL singular at v = 0. Khalil et al., *Automatica* 32(9):1323–1327, 1996 prove recovery only *"if the actuator dynamics are sufficiently fast"* — at 0.41 s yours are **slower** than the target closed loop, so the hypothesis is violated, not marginally met. |
| **High-gain backstepping / aggressive Kanayama** | Cross-track eigenvalue `√(k_y·v)`; keeping it under 2.5 rad/s at v = 0.7 caps `k_y ≲ 9 m⁻²`, far below the 30–100 typical in papers. Backstepping into dynamics raises relative degree, costing phase you lack. |
| **MRAC** | Rohrs, Valavani, Athans & Stein, *IEEE TAC* 30(9):881–889, 1985, [DOI 10.1109/TAC.1985.1104071](https://doi.org/10.1109/TAC.1985.1104071) — two infinite-gain operators in the loop; output disturbances at *any* frequency including DC drive gain up without bound. Delay margin → 0 as adaptation gain grows. Plus Anderson's **bursting** (*Automatica* 21(3):247–258, 1985): identical laps is the archetypal loss-of-persistent-excitation case. Undebuggable. |
| **L1 adaptive** | Its robustness story is the low-pass filter, which must sit *below* control bandwidth. At 0.36 Hz there is no room. Independently contested: Ioannou, Annaswamy, Narendra et al., *IEEE TAC* 59(11):3075–3080, 2014 argue the L1 filter *deteriorates* robust-stability bounds vs MRAC; Ortega & Panteley (IFAC 2014; *Int. J. Control* 87(3):581–588) argue it converges to a linear PI. |
| **DOB / ADRC** | Gao's rule is ω_o ≈ 3–10× ω_c ⇒ 1.1–3.8 Hz observer, which 0.42 s of lag cannot support; the ESO would estimate its own delay-induced phase error as "disturbance". Zhao & Gao, *ISA Transactions* 53(4):882–888, 2014 confirm oscillation under large dead time and fix it by **adding a predictor**. |
| **Classical Smith predictor** | The cross-track channel is an integrator chain, where the plain Smith predictor fails. A *filtered* Smith predictor works, but MPC/TVLQR with a delay-replayed `x₀` is the same idea with constraints and preview included. |
| **More feedback gain anywhere** | See above. You are at the optimum. |

**Also worth knowing:** of widely deployed open-source mobile-robot trackers, **only Autoware's MPC
models a first-order actuator lag**, and only recent nav2 MPPI models transport delay
(`model_delay_vx`, default **0.0**, undocumented in any paper — version-dependent, verify in your
tree). nav2 RPP, DWB, TEB and Vector Pursuit have **no actuator model at all**. R1 + R5 puts you ahead
of the ROS field, not catching up.

---

## 3. Method-by-method verdicts

| Method | Fit to a 0.36 Hz lag-limited diff-drive on a C² route |
|---|---|
| **Pure pursuit** (Coulter, CMU-RI-TR-92-01, 1992, [PDF](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)) | **Good, and already at its ceiling.** Stability under delay comes entirely from preview (2.0 m / 0.7 m/s ≈ 7× the lag). Structural flaw: gain and lead coupled through one parameter, so a safety pull-in raises gain 16× (§0a). |
| **Adaptive lookahead** (Sukhil & Behl, [arXiv:2111.08873](https://arxiv.org/abs/2111.08873) — *preprint*) | Their version computes per-waypoint lookahead **offline and greedily** — an inspectable schedule, which suits the debuggability constraint. RL-tuned variants are in-loop, black-box, preprint-only: **do not put on hardware.** Hard floor either way: `L_d ≥ v·(θ+τ)` = 0.29 m at 0.7 m/s. |
| **Regulated Pure Pursuit** (Macenski, Singh, Martín & Ginés, *Autonomous Robots* 47:685–694, 2023, [DOI 10.1007/s10514-023-10097-6](https://doi.org/10.1007/s10514-023-10097-6)) | **Do not adopt — you already have it, and it is the problem.** RPP *is* this architecture: nominal pure-pursuit velocity then **multiplicative** curvature and proximity regulation. Your design is mainstream, not idiosyncratic; but the proximity heuristic is the discontinuous layer R2 moves inside the optimisation. |
| **Stanley** | Provably sound only at the **front axle**, co-located with steering. No such point on a diff drive (double integrator ω → lateral), so the proof does not transfer. **Keep the `k_soft` speed softening; reconsider the gain; add `k_d,yaw` damping instead.** |
| **LQR / TVLQR** | Vanilla Frenet LQR knows nothing about actuation and fails silently when tuned "for fast convergence". **Delay-lifted TVLQR scheduled offline along arc length** is the best capability-per-risk option in this family for a fixed repeated route → R5. |
| **Linear MPC (Autoware pattern)** | Best-validated fit. Copy the three-layer mechanism, not just the horizon. |
| **NMPC (acados/OSQP)** | Survives cleanly *only* with the predictor; without it the first horizon step is wrong by 0.41 s of motion — **0.29 m at 0.7 m/s, double the current RMS**. Compute is not the constraint at 20 Hz. |
| **nav2 MPPI** | Handles transport delay only if `model_delay_vx = 0.20` (default 0.0); **no τ model in any of its motion models.** |
| **Differential flatness FF + feedback** | **Best fit in the review, and you are 90% there.** The C² spline with analytic κ *is* the flat-output trajectory; `ω = v·κ` is the exact nominal input; delay in that channel is a pure shift. → R1. |
| **Frenet formulation** | A modelling choice, orthogonal to delay — but it yields analytic `A(s), B(s)` from spline curvature, which is what makes offline TVLQR scheduling possible. Adopt as the state representation regardless. |

**A framing worth having:** Aguiar, Hespanha & Kokotović, *Automatica* 44(3):598–610, 2008,
[DOI 10.1016/j.automatica.2007.10.008](https://doi.org/10.1016/j.automatica.2007.10.008) prove the
performance limitation from unstable zero dynamics applies to *reference tracking* but **does not
exist for path following** (geometric path, no assigned timing law). You are already on the better side
of that result. Keep the path/speed separation R2 formalises; do not drift toward time-parameterised
trajectory tracking.

---

## 4. Curvature feedforward and delay compensation

**Canonical structure is two feedforward terms plus feedback.** Peng & Tomizuka, *ASME JDSMC*
115(4):679–686, 1993, [DOI 10.1115/1.2899194](https://doi.org/10.1115/1.2899194): *"the optimal preview
control law consists of a feedback control term and two feedforward control terms."*

**Preview-control lineage** (all peer-reviewed): Sheridan, *IEEE Trans. Human Factors in Electronics*
HFE-7(2):91–102, 1966, [DOI 10.1109/THFE.1966.298341](https://doi.org/10.1109/THFE.1966.298341) — the
origin. Tomizuka, *IEEE TAC* 20(3):362–365, 1975. Katayama, Ohki, Inoue & Kato, *Int. J. Control*
41(3):677–699, 1985 — discrete-time previewable demand. MacAdam, *IEEE Trans. SMC* 11(6):393–399, 1981.
Sharp & Valtetsiotis, *Vehicle System Dynamics* 35(sup1):101–117, 2001 — LQ gains on multi-point road
geometry ahead of the vehicle.

**Delay compensation, increasing weight:**

1. **Time-shift the feedforward** (R1) — free, exact for the FF channel, the only option costing zero
   bandwidth. Seiffer et al. 2023 is the direct experimental validation.
2. **Inverse-model lead** — the `(1 + τs)/0.89` factor. Also feedforward, also free. Both are ZPETC.
3. **State prediction over the dead time** — roll `x` forward on buffered commands, hand the controller
   a delay-free `x₀`. Engineering consensus across MPC/NMPC/MPPI: **the delay is handled at `x₀`, not
   in the cost.** Rigorous: Artstein, *IEEE TAC* 27(4):869–879, 1982; nonlinear/robustness: Krstic,
   *Delay Compensation for Nonlinear, Adaptive, and PDE Systems*, Birkhäuser 2009, and
   Bekiaris-Liberis & Krstic, *Automatica* 49(6):1576–1583, 2013 (robustness to *small* delay
   mismatch — your ~10% τ spread is comfortably inside).
4. **Delay lifting** — exact here, 4 integer states → R5.
5. **Smith predictor** — subsumed by (3)+(4); needs the *filtered* variant on an integrating channel.
   A kinematic integration-free variant aimed at this regime: Festl & Stolz,
   [arXiv:2507.06935](https://arxiv.org/abs/2507.06935), 2025 — *preprint, simulation only*.

**How much preview.** Optimal preview gains decay, so preview beyond the plant's response horizon buys
nothing: `t_ff` ≈ total lag for the FF channel (sweep it), a few seconds of route lookahead for the
speed profile (5 m at 0.7 m/s = 7 s is generous).

**★ One measurement to make first.** The identification is `cmd_rot → measured_rot`, which **cannot
distinguish** delay in the *actuation* path from delay in the *measurement* path. If most of the 0.20 s
is measurement latency, an inner yaw-rate loop (R3) cannot help and the FF shift is the only lever.
Seiffer et al. decomposed theirs — localisation ~0.01 s, steering realisation 0.10 s, yaw-rate response
0.05 s, ~0.04 s comms — and that decomposition is what let them size `t_ff` confidently.

---

## 5. Self-adjusting control, per method

| Method | Stable at θ=0.20, τ=0.225? | Identified online | Hardware failure mode | Debuggable? |
|---|---|---|---|---|
| **ILC, arc-length, between laps** | **Yes — the delay is a 4-sample shift, not a constraint** | **Nothing** | Learning transients; non-repeating-disturbance amplification; reference change invalidates the learned signal | **Yes.** `‖e_j‖` non-increasing, `‖u_j−u_{j−1}‖` contracting. **Fully replayable offline.** |
| **Gain scheduling on (v, κ)** | **Yes** — verify each frozen point offline | Nothing | Gaps between schedule points; per-platform retune | **Best.** Three deterministic scalars. |
| Smith / nonlinear dead-time predictor | Yes | θ, τ — you have them | Delay mismatch | Good. Log `y − ŷ`: small, zero-mean. |
| Repetitive control (spatial) | Marginal — memory loop *inside* the 20 Hz loop | Nothing, but lap length must be exact | Period mismatch; divergence with **no freeze state** | Fair — live, not replayable. |
| DOB / ADRC | Degraded — filter bandwidth capped **below** ω_c | Nothing | ESO reads delay lag as "disturbance" | Fair; `d̂` ambiguous |
| Adaptive lookahead (offline/greedy) | Yes | Nothing | — | Good |
| RLS in-loop system ID | Depends | Slip/steering params | Bursting under no PE (identical laps ⇒ no excitation) | Poor |
| **L1 adaptive** | No usable margin at 0.36 Hz | Matched uncertainty, in-loop | Filter-bandwidth cliff | Poor |
| **MRAC** | **No** | Full plant params, in-loop | Rohrs instability; bursting | **No** |

**RC vs ILC, resolved.** RC embeds the periodic internal model *in the closed loop*, which must then be
stabilised by a q-filter inside the delay line — so 0.42 s **does** cap RC's bandwidth, unlike ILC.
Worse, the period is *spatial* and lap time varies, so a fixed-period temporal RC has the wrong
internal model outright. **RC gives nothing ILC doesn't and removes offline safety.** (Hara, Yamamoto,
Omata & Nakano, *IEEE TAC* 33(7):659–668, 1988; Francis & Wonham, *Automatica* 12(5):457–465, 1976.)

**The "online system ID" actually worth having, with no in-loop risk:** re-run the *existing* offline
identification once per lap on logged `(cmd_rot, measured_rot)`. You already have the estimator and know
it gives r² 0.94–0.95 repeatably. Log τ, θ, K per lap — that is a health monitor that never touches the
loop.

---

## 6. Smoothness and the safety layer, in principle

Part A is R2. On the safety layer specifically:

**The principled structure is a safety *filter*, not a gate.** The CBF-QP form
`u* = argmin ‖u − u_nom‖² s.t. L_f h + L_g h·u ≥ −α(h)` returns `u_nom` **unchanged** whenever it is
already safe, and otherwise the **nearest** safe command. The extended class-K function α is continuous
and zero only at `h = 0`, so as clearance shrinks the admissible set contracts *continuously* — no
threshold, no switch, no multiplier. This is exactly the "encode it as a continuous covariate, not a
hard gate" principle in CLAUDE.md, with `h` = ESDF clearance as the physical covariate. (Ames, Xu,
Grizzle & Tabuada, *IEEE TAC* 62(8):3861–3876, 2017,
[DOI 10.1109/TAC.2016.2638961](https://doi.org/10.1109/TAC.2016.2638961); survey: Ames, Coogan,
Egerstedt, Notomista, Sreenath & Tabuada, ECC 2019, [arXiv:1903.11199](https://arxiv.org/abs/1903.11199).)

**Expressiveness argument, independent of smoothness.** A multiplicative gate `u ← γ·u_nom`, γ ∈ [0,1],
can only *shrink* the command along its own direction. On a differential drive it therefore cannot
express "go slower **and** turn away" — usually the minimal safe correction. That is why the gate has
to cut hard where a QP would nudge.

**Four real caveats:**

1. **CBF-QP solutions are not automatically continuous** — they lose Lipschitz continuity where active
   constraint gradients become linearly dependent (LICQ failure): a small state perturbation flips the
   active set and the solution jumps. Morris, Powell & Ames, CDC 2015, pp. 151–158,
   [DOI 10.1109/CDC.2015.7402101](https://doi.org/10.1109/CDC.2015.7402101); Mestres, Allibhoy &
   Cortés, *European Journal of Control*, 2024, [arXiv:2311.13167](https://arxiv.org/abs/2311.13167).
   **For an ESDF this bites** as "which obstacle is closest" flips. Fix: log-sum-exp smoothing over
   nearby constraints instead of `min_i h_i`.
2. **CBF-QPs can create spurious asymptotically stable equilibria at the unsafe-set boundary** — the
   robot parks against an obstacle instead of going around — even for linear systems with one convex
   obstacle. Reis, Aguiar & Tabuada, *IEEE L-CSS* 5(2):731–736, 2021,
   [arXiv:2003.07819](https://arxiv.org/abs/2003.07819), with a fix.
3. **CBF under 0.4 s input delay is the thin part of the literature and the biggest risk here.**
   Sizing: at 0.7 m/s with 1.0 m/s² braking, braking distance is 0.245 m but the **0.44 s of open-loop
   travel before deceleration begins adds 0.308 m** — the delay term *dominates the braking term*. A
   margin that ignores delay is violated by more than the braking distance itself. Predictor feedback
   is not optional. Singletary, Chen & Ames, CDC 2020, pp. 804–809,
   [arXiv:2005.06418](https://arxiv.org/abs/2005.06418); Breeden, Garg & Panagou, *IEEE L-CSS*
   6:367–372, 2022, [arXiv:2103.03677](https://arxiv.org/abs/2103.03677). **Both simulation-only at
   this delay magnitude — no hardware validation found.**
4. **Pure safety filters are myopic** and can deadlock in narrow passages — live risk in a corridor.
   Remedy: safety as a constraint over a *horizon* (MPC-CBF: Zeng, Zhang & Sreenath, ACC 2021,
   pp. 3882–3889, [arXiv:2007.11718](https://arxiv.org/abs/2007.11718),
   <https://github.com/HybridRobotics/MPC-CBF>).

**Soft constraints, if you go the MPC route:** use an **ℓ₁ (linear)** slack penalty, not quadratic. A
linear penalty is an *exact* softening — enforced as if hard whenever feasible, relaxed only when
nothing feasible exists. A quadratic penalty always trades a little violation for a little cost
regardless of weight. Scokaert & Rawlings, *AIChE Journal* 45(8):1649–1659, 1999,
[DOI 10.1002/aic.690450805](https://doi.org/10.1002/aic.690450805); for a *computable* lower bound on
the weight rather than tuning, Kerrigan & Maciejowski, UKACC Control 2000,
[PDF](http://www-control.eng.cam.ac.uk/Homepage/papers/cued_control_53.pdf).

**Rate-limiting or low-passing a safety multiplier is ad hoc — no principled treatment was found.** The
nearest legitimate ancestor is sliding-mode chattering reduction (Tseng & Chen, *Asian Journal of
Control* 12(3):392–398, 2010; Lee & Utkin, *Annual Reviews in Control* 31(2):179–188, 2007), whose
consistent finding is that filtering the command reduces chattering **but breaks the guarantee unless
the filter is compensated in the model**. The principled version makes the rate bound a *constraint in
the same QP* (Filtered CBFs, [arXiv:2503.23267](https://arxiv.org/abs/2503.23267), 2025 — *preprint*).

**If you do not want a QP at all**, the lightest principled option is a **reference governor**: an
add-on that modifies the *reference* (not the command) to guarantee constraint satisfaction via a small
scalar optimisation each cycle. Continuous in the state, and because it acts on the reference side it
costs no phase margin. Garone, Di Cairano & Kolmanovsky, *Automatica* 75:306–328, 2017,
[DOI 10.1016/j.automatica.2016.08.013](https://doi.org/10.1016/j.automatica.2016.08.013). Here the
governed variable is the speed profile `b(s)` — which makes this and R2 the same change.

**Velocity obstacles and DWA are not the answer for smoothness.** Both select from an admissible set —
DWA by argmax over a sampled grid, ORCA by projection onto a polygon — so both are non-smooth in the
state *by construction*, structurally closer to the current gate than to a CBF. Their genuinely useful
contribution is DWA's admissibility condition `v ≤ √(2·d·a_brake)` (Fox, Burgard & Thrun, *IEEE
Robotics & Automation Magazine* 4(1):23–33, 1997, [DOI 10.1109/100.580977](https://doi.org/10.1109/100.580977))
— a continuous, physics-derived speed cap, exactly the clearance constraint to drop into R2's `b(s)`.

---

## 7. Where the evidence is thin or the literature disagrees

1. **The §0/§2 loop-shaping numbers are a derivation, not a citation.** Assumptions listed in §2. Check
   against a measured chirp. Qualitative conclusions (gain–lead coupling, 60° cap, clip-raises-gain)
   are robust; the exact rad/s values are not.
2. **The `PdCrossTrackGain` → ω mapping was not traced through this codebase**, so "the Stanley term
   consumes phase margin" is structural, not quantified for our implementation. Verify before acting.
3. **`ω_c ≲ 1/θ`** is a textbook rule of thumb; no verbatim source text was retrievable. Don't cite a
   page number.
4. **Ollero & Heredia (1995/2007) — the most on-point primary source — is inaccessible online.** Worth
   a library request.
5. **Autoware's delay compensation has no peer-reviewed evaluation.** Lineage is a 2009 CMU tech
   report. Cite as production design, not result.
6. **nav2 MPPI's `model_delay_vx` appears in no paper**, and rolling docs contradict Humble/Iron docs.
   Verify in your tree.
7. **CBF under ~0.4 s input delay: simulation only.** No hardware validation at this delay magnitude
   was found. **Weakest link in R2's safety layer.**
8. **Is jerk-constrained speed planning convex? Genuinely open.** Consolini & Locatelli, *Automatica*
   170:111879, 2024, [arXiv:2310.07583](https://arxiv.org/abs/2310.07583) — the title is literally a
   question; exactness of an SOCP relaxation is proved only under stated assumptions and generality is
   *conjectured*. Lowest-risk resolution: keep jerk out of the profile and impose it downstream
   (Ruckig).
9. **Comfortable jerk has no single number** — 0.6–2.9 m/s³ across the literature, and de Winkel et al.
   2023 show dependence on pulse shape and duration. Don't lean on ISO 2631 alone.
10. **No rigorous head-to-head smoothness benchmark of CBF vs DWA vs VO with jerk metrics exists.**
    nav2 comparisons are simulation-only and measure "control effort", not jerk.
11. **MRAC/L1 numeric delay margins** came from search snippets of NASA PDFs that could not be
    text-extracted — verify before quoting. The L1 dispute is unresolved.
12. **Comparative tracker studies almost never sweep delay.** PP/Stanley/LQR/MPC comparisons are
    simulation-dominated under *ideal actuation*, so their rankings say nothing about this question.
    Reported "24–96% improvement" spreads are not comparable across papers. Seiffer et al. is the
    exception, which is why it anchors this review.
13. **The spatial Q-filter cutoff (~1–1.5 m)** for R4 is a scaling of Kapania & Gerdes' 2 Hz temporal
    filter, not a published value. A/B it.

---

## 8. Suggested order of work

1. **Measure first** — plot signed cross-track vs κ and vs dκ/ds; confirm the outward-bias signature.
   Decompose the 0.20 s delay into actuation vs measurement (§4). *Hours, and it decides everything
   below.*
2. **Control on a jump-free local pose** (§0b). No tracker fixes this.
3. **R1** — preview-shifted curvature FF + `1/0.89` + `τ` lead; sweep `t_ff` from 0. *Under a day;
   expect the largest single gain.*
4. **R2** — collapse the speed chain into one `b(s)` profile with clearance as a constraint; delete the
   EMA and the post-hoc gate. *2–3 days; this is the smoothness fix.*
5. **R3** — decouple steering gain from lookahead; add `k_d,yaw`. *2 days.*
6. **R4** — arc-length ILC with basis functions and a freeze-on-regression guard. *3–5 days; the only
   thing that beats the bandwidth ceiling.*
7. **R5** only if 3–6 leave you short. *1–2 weeks.*

**If you make one change, make it R1. If you make one change to fix the smoothness complaint
specifically, make the "move the safety cut inside the profile computation" half of R2.**

---

## 8b. §8 STEP 1 EXECUTED — 2026-08-04, and it moves R1

Measured on a completed "complete tour" lap: 2339 cycles / 117 s / 38 m, Webots. Conventions grounded
from the data first, not assumed — `cmd_rot` is sign-flipped vs pose θ (the known negation),
sign(κ) = sign(dθ/dt), and the tracker's own `proj_robot.x > 0 ⇒ path to the right`. Together these make
**corr(e, κ) < 0 the outward signature**.

**1. The bias is INWARD, not outward — R1's motivating signature is absent.**
`corr(e, κ) = +0.757` over all driving, **+0.785** on curved stretches. Positive = the robot rides
*inside* the curve. It cuts corners; it does not bulge outward. Reproduces the 08-02 figure (+0.67), so
it is a stable property. **The `1/0.89` sub-claim of R1 loses its evidence** — not refuted (the gain
deficit is a measured plant property) but its predicted signature is not there, possibly masked by the
larger inward term. Test it alone, after the others, so they do not confound.

**2. The residual is NOT geometric corner-cutting either.** Pure pursuit predicts `e = κ·L²/2`, so the
slope should scale as L². Binned by achieved carrot distance:

| carrot L | 0.35 m | 0.72 m | 1.09 m | 1.72 m |
|---|---|---|---|---|
| slope a | +0.079 | +0.081 | +0.077 | +0.057 |
| a / L² | 0.635 | 0.156 | 0.065 | 0.019 |

`a` is essentially **constant at ~0.08 m per (1/m)** while `a/L²` varies 33×. So the residual is not set
by the lookahead geometry. What is left is the **steady-state error the cross-track loop must build in
order to generate the turn rate at all** — exactly what §2 predicts for a lag-limited loop, and exactly
what feedforward removes for free. ★**This STRENGTHENS R1(a), explicit `ω_ff = v·κ`**, which the
controller does not have today (pure pursuit supplies it only implicitly through carrot geometry).

**3. The error ANTICIPATES curvature; it does not lag it.** Peak of `corr(e(t), κ(t−lag))` at
**−550 ms** (+0.782) against +0.757 at zero lag, decreasing monotonically toward positive lag. A
delay-induced error would correlate with PAST curvature. ⚠The magnitude is weakly identified — κ's own
autocorrelation is +0.87 at 450 ms — so only the SIGN survives. **R1(b), the preview shift, is
contraindicated as a first move**: shifting the evaluation point further ahead pushes the wrong way.
Re-test it after R1(a), since removing the steady-state term may expose a lateness underneath.

**4. Delay decomposition, partial.** `cmd_rot → meas_rot` peaks at **+200 ms** (corr +0.932), matching
the 08-02 identification. `pose_stamp_age` p50 **84 ms**, p90 183 ms ⇒ roughly **40% of the loop delay
is measurement staleness**, not actuation. This data cannot split it further: `meas_rot` and the pose
come through the same DSR path. On the real robot the IMU gives the clean split — compare
`cmd_rot → IMU rate` against `cmd_rot → DSR-published rate`; the difference is transport. That decides
whether R3's inner yaw-rate loop can help at all.

**Revised ordering.** R1 splits: **(a) explicit `v·κ` feedforward — do first**, strengthened by finding
2, and a prerequisite for R3 anyway (`ω = ω_ff + k_y·e_y + k_θ·e_θ` needs an ω_ff to exist).
**(b) preview shift — hold**, contraindicated by finding 3. **(c) `1/0.89` + τ lead — test alone**,
unsupported by finding 1.

⚠**Experimental-design limit.** The route is 418 left-turn samples to 54 right, so κ is almost always
positive: intercept and slope are collinear (the fit says +3.2 cm constant offset while the
straight-only subset says −2.3 cm), and a left/right asymmetry would be invisible. **A constant lateral
offset is NOT established.** A mission with balanced turns would settle both.

## 9. What our own measurements already say about this

Independent corroboration from this codebase, measured 2026-08-04 (see [[controller-timing-and-wobble-2026-08-02]]):

- **Per-cycle decomposition of Δln(speed)** over a 576-cycle lap: safety gate **79.1%** of the variance
  (rms 0.191, worst ×0.31 in one 50 ms cycle), shaping 14.3%, rotation brake 6.5%. The gate fires in
  **33 episodes, every one exactly 1 cycle long** — a self-extinguishing loop, since
  `gate_horizon = v/a_dec + 0.15` shortens when the cut lands. Against a 0.41 s plant these pulses are
  too short to brake and too frequent to be smooth. **This is R2 defect (2), measured.**
- **Across 6 baseline laps, `lin_jerk_effort` vs `safety_guard_cycles` correlate at r = 0.995**
  (~0.78 jerk units per guard cycle), and guard cycles vs mean speed at r = −0.904. A completely
  separate metric set agrees: the gate is what makes the motion rough, and it costs lap time.
- `velocity_smoothing` (EMA, α = 0.60 ⇒ τ = 98 ms) is applied **before** the brake and the gate, so it
  cannot touch 85% of the roughness by construction, and it smooths the `rot` channel in the same
  vector — buying smoothness with exactly the phase margin R3 says you don't have. **R2 defect (3),
  measured.**
