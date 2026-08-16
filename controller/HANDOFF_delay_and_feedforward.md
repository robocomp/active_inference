# PLAIN: delay and feedback gains — and why the feedforward is neither

*Written 2026-08-16 from the thesis-side review; restructured the same day. The controller is
**PLAIN** (`controller/src/trackers/plain_tracker.cpp`, `ControlMode = "plain"`). Not pd, not MPPI —
those are earlier laws PLAIN replaced, and conclusions drawn about them do not transfer.*

**The two open issues are DELAY and FEEDBACK GAINS.** The curvature feedforward is *not* one of them:
it is built, exact and correctly structured, and it appears below only as a short do-not-re-open note,
because the same mis-attribution has now been made twice and cost real work each time.

## The law, so nothing is mis-attributed

```
omega = g_dc * [ v * kappa_bar(s + v*T_lag)  -  v_steer * ( (2/L) e_psi + (1/L^2) e_y ) ]
```

- The feedforward is the **kinematic relation** `omega = v*kappa`, not an approximation.
- It is **already previewed** by the identified actuator lag, `s + v*T_lag`.
- There is **no carrot and no lookahead point**. `kappa_avg` is a centred difference, so the preview
  is the one and only lookahead in the law.

---

## ISSUE 1 — delay: one half is compensated, the other is not

**The plant.** ~0.20 s transport delay on top of a lag of comparable size (tau 0.213–0.236 s), DC gain
0.89, loop bandwidth 0.36–0.39 Hz. **Roughly 40% of the loop delay is the age of the pose estimate at
the moment it is used** (`pose_stamp_age` p50 84 ms), not anything in the actuator.

★**RE-MEASURED 2026-08-16 evening, and the shape of this has changed — the issue stands, its headline
number does not.** Over two complete tours (`20260816-141531`, `20260816-202001`, n = 2176 and 2638
cycles, `mppi_diag.pose_stamp_age`): **p50 49–52 ms, p90 143–145 ms, max 226–237 ms.** The median has
roughly halved since 84 ms, so staleness is no longer ~40% of the budget at the typical cycle — but the
tail has not moved, and p90 ≈ 145 ms is most of a lag constant. **Re-frame the work: this is now a TAIL
problem, not a median one**, which changes what fixing it looks like. Forward-propagating the pose to the
instant of use (option 2 below) addresses a tail directly; shaving the median does much less than the
original 84 ms figure implied. Re-measure before committing to either — and see the hygiene note below
for why every number taken between 12:55 and 21:50 today is unusable for this.

**What the preview does and does not fix.** Previewing the feedforward at `s + v*T_lag` cancels
*actuator* lag exactly, because the curve ahead is known in closed form — the turn is commanded when
it will be needed rather than when it is observed. It does **nothing** for measurement staleness, and
cannot: a stale pose corrupts `e_y` and `e_psi`, which is to say it corrupts the **feedback** path.

**Why that is the half that matters.** It is the feedback that saturates, not the feedforward:
`(1/L^2)*e_y = 2.78` per metre, so 0.3 m off-route demands 0.83 rad/s on its own, while mean `|kappa|`
at saturating cycles was 0.65, i.e. `v*kappa = 0.44`. **The feedforward alone never saturates.** Stale
pose → wrong `e_y` → feedback demand → saturation → cannot converge → stays off-route. The delay lands
precisely on the loop with no headroom.

**What to do, in order.**
1. **Reduce the staleness itself** before touching any gain. 40% of the delay budget is sitting in how
   fresh the pose is at point of use. That is not a control problem and it is the cheapest
   improvement available.
2. **Or propagate the pose forward to the instant of use** (integrate odometry across the stamp age),
   so the feedback sees a current estimate rather than a recent one.
3. Only then consider anything predictive — see Issue 2 for why LQR is not the answer.

### ★WHERE THE STALENESS ACTUALLY COMES FROM (measured 2026-08-16) — most of it is a CHOICE

`ControllerWorldModel::read_robot_pose_in_room` (`controller_world_model.cpp:98-112`) queries the RT tree
at **`query_ts = last_lidar_timestamp_ms.value_or(timestamp_ms)`**. The control loop therefore asks for
the pose *as of the last LiDAR scan*, not as of now. The pose is not arriving late — a fresher one is
deliberately declined. Measured over 6000 cycles of `overlay_lag_eval.csv`:

| quantity | p50 | p90 | max |
|---|---|---|---|
| `pose_age_ms` — age of the pose the tracker used | 49 ms | 102 ms | 400 ms |
| `rt_lead_ms` — how much fresher the NEWEST RT block already is | **33 ms** | **68 ms** | 121 ms |
| twist-probe position residual over a ~49 ms horizon | **1.1 mm** | 2.3 mm | 37 mm |
| twist-probe heading residual over the same | **0.059°** | 0.146° | 1.30° |

Two independent levers, both already instrumented, neither needing new machinery:

**(A) Query at `now` rather than at the scan stamp.** That recovers `rt_lead_ms` — two thirds of the
median staleness — at *zero* modelling cost, because the fresher pose is a real RT block that already
exists, not an extrapolation. This is option 1 above, and it is a one-line change to which timestamp is
asked for.

**(B) Dead-reckon whatever age remains** (option 2). Its cost is already measured, not estimated: the
twist probe integrates the base twist across one LiDAR period and compares against the RT tree, and it
lands within **1.1 mm and 0.06°** at the median. Against an `e_y` term worth 2.78 rad/s per metre, a
millimetre of extrapolation error is 0.003 rad/s. It is free.

### IMPLEMENTED 2026-08-16 — `Controller.TrackerUsesLatestPose` (default **true**)

`ControllerWorldModel` now has two readers: `read_robot_pose_in_room` (scan-aligned, unchanged, all its
previous callers untouched) and `read_robot_pose_latest` (the newest block). The session resolves the
fresh one once per cycle and hands it to `path_controller.compute()` and to nothing else.

★**The "settle the ESDF coupling first" caveat this section used to carry was WRONG, and the reasoning
matters more than the conclusion.** The worry was that a fresher pose would advance the robot relative
to an obstacle field that had not advanced. It does not: the LiDAR buffer holds **room-frame** points,
registered per plane at their own stamps, and `read_lidar_points_robot` expresses them in the robot
frame by applying `pose.inverse()`. A room-frame cloud is a *map*; the pose that belongs in that
transform is the one for the instant you want the field to describe, which is NOW. Using the scan-time
pose displaces every obstacle by the robot's motion since the scan — 17 mm at 0.35 m/s and the median
age, 36 mm at p90 — and it displaces them **backwards relative to the body**, which is the unsafe
direction. So the fresh pose is not merely tolerable for the ESDF; it is the correct argument for the
whole call. Both halves of `compute()` want the same instant, and they now get it.

✅**Verified there is no hidden extrapolation.** `TrackerUsesLatestPose` returns a real RT block, not a
model: cortex's `bracketing_blocks` (`api/dsr_rt_api.cpp:104-119`) CLAMPS past the newest block —
`if (upper == blocks.end()) return {blocks.back(), blocks.back()}` — and the equal-endpoints branch then
returns that block verbatim. `Nearest` likewise resolves to a real block. Asking for a future timestamp
cannot invent a pose. This is what makes lever (A) free in a way lever (B) is not.

⚠**WHAT TO WATCH ON THE FIRST RUNS, because this is unvalidated on the robot.** A fresher pose is a LESS
SMOOTHED pose — the scan-aligned one was in effect delayed and thereby filtered. If the newest block is
jumpy, that arrives as `e_y` noise and therefore as rot chatter. **Compare `cross_track_rms_m` and
`rot_reversals` against a run with the flag false** before keeping it; the whole point of the flag is
that this is one config line and no rebuild. `overlay_lag_eval.csv` gained a `tracker_pose_lead_m`
column — the distance between the two poses, i.e. the correction actually being applied — so the size
of the change is visible per cycle rather than inferred. Multiply it by the `(1/L²) = 2.78` rad/s per
metre feedback gain to read it as the demand the loop is no longer making on stale evidence.

---

## ISSUE 2 — feedback gains: geometric, not derived from the plant

The gains are currently the geometric pair `(2/L, 1/L^2)` — a single length `L` setting both terms.
They are not derived from the identified plant, and this is the one genuinely open item left from the
bibliography review's R3 line. It is a **gains** question, not a structure question: the law's form is
already right.

**This is where the saturation lives.** `(1/L^2)*e_y = 2.78` per metre means a third of a metre
off-route asks for most of the actuator on the cross-track term alone, before heading error or the
nominal turn are added. Whatever replaces `L` has to be chosen against that, not against tracking
error in the small.

⚠**"Derive the gains" is not "raise the gains".** Gain and lookahead are coupled in the geometric
form, so the phase contribution is pinned near a fixed value whatever `L` is set to, and the deployed
configuration already sits at the resulting phase-margin limit. **The loop is not under-tuned.** A
sweep of `L` will not move it; only changing where the gains come from can.

⚠**Do not reach for LQR as the way to do that.** A PD-form law on `(e_y, e_psi)` already *has* LQR
structure, so LQR would change the gains, not the form — and its optimality guarantee assumes a
**delay-free** loop, which this is not. Under ~0.4 s of loop delay, recovering a guarantee needs state
augmentation or explicit prediction, i.e. a predictive controller; the review ranks that last (R5) for
exactly this reason, and MPPI already lost to a simpler law on measurement. If the gains are to be
derived, they must be derived against the *identified* plant including its delay, or the derivation
buys nothing the current pair does not already have.

★**Sequencing note.** Issues 1 and 2 are coupled and Issue 1 comes first. Stale pose corrupts `e_y`,
which is the input the gains multiply; deriving gains against a corrupted error signal fits the gains
to the staleness. Fix the freshness, then re-identify, then derive.

---

## SETTLED — the curvature feedforward. Do not re-open; three things to know

1. **The failed `omega_ff = -v*kappa` experiment does NOT apply to PLAIN.** It made everything worse
   (cross-track rms 7.94 → 9.83 cm, and the total command got *smaller* when it should have got
   bigger) on the **pure-pursuit / pd lineage**, where `Kp*carrot_angle` on a curve already *is* the
   nominal turn, so the explicit term double-counted it. That code was removed at the time. **PLAIN
   has no carrot to double-count**, and the recorded lesson — a feedforward must *replace* the
   geometric term, not stack on it — is satisfied by PLAIN's construction. It is not outstanding work.
2. **Do not add a second feedforward term.** If more feedforward authority is wanted, change what
   `kappa_bar` is evaluated on or how far ahead it looks. Introducing another term that also supplies
   part of the nominal turn reproduces the failure in 1.
3. **PLAIN's own feedforward risk is querying past the end of the route**, which is a different fault
   from double-counting. The preview reads `v*T_lag ≈ 0.29 m` ahead at full speed; on a curve about to
   finish, that ran off the end and `kappa_avg` differenced a real heading against a returned zero:
   `k_ff = remainder(0 - 1.15, 2pi) / 0.40 = -2.87 1/m` where the truth is about `1 1/m`. Omega
   saturated and the robot turned toward heading zero through its final 0.15 m, with `e_y` pinned at
   0.011 m and `e_psi` at 0.021 — **the feedback was asking for nothing at all**. Believed addressed
   2026-08-13. **Verify it is, and verify the guard covers the case where the preview window straddles
   the terminus.**

   ✅**VERIFIED 2026-08-16 — the terminus is covered.** `plain_tracker.cpp:134` clamps the preview to
   `sp.length() - 0.5f * p.plain_W`, and the reasoning is the right one: a CENTRED window needs W/2 of
   curve on both sides, so that — not `length()` — is the bound. The straddling case the note asks about
   is exactly what that half-window accounts for.

   ⚠**BUT THE MIRROR CASE AT THE START IS NOT GUARDED, and it is the same fault reflected.** The reason
   `heading_at` returns a bogus 0 rad past the end is that `position_at` CLAMPS its probes and the
   difference vector collapses (`route_spline.cpp:255-272`). It clamps at **both** ends — `max(0.f, ...)`
   as well as `min(length_, ...)` — so an argument below the start collapses identically, and 0 rad is
   just as much an ordinary heading there. `kappa_avg(q, W)` reads `heading_at(q - W/2)`, so with the
   shipped `PlainTrackerW = 0.40` and `route_spacing_m = 0.05` the lower probe collapses whenever
   `q <= W/2 - spacing = 0.15 m`, where `q = min(s + v*T_lag, length - 0.5*W)`.
   On a long route at speed the preview alone clears it (`v*T_lag ≈ 0.29 m`), which is why no mission
   has shown it. The reachable case is the upper clamp: **any route of length ≤ 0.35 m has `k_ff`
   differenced against a fake heading of zero for its whole length.** Mission routes are ~37 m; a short
   affordance hop to a standpoint is not, and those get a fitted spline too (`smooth_planned_path`).
   ★NOT FIXED HERE, deliberately, and read the fix note above before fixing it: clamping inside
   `heading_at` is the tempting one-liner and it was MEASURED to be a much larger change than it looks,
   because the bogus curvature was also acting as an accidental end-of-route brake. The same could be
   true at the start. Reproduce it on a short route first — the prediction is a saturated `omega` on the
   opening 0.35 m with `e_y` and `e_psi` both near zero, which is the signature that identified the
   terminus case.

---

## The one-line summary

Two open issues, in order: **delay** — the actuator half is already cancelled by the feedforward
preview, the measurement half (~40% of the budget) is uncompensated and lands on the feedback path;
and **feedback gains** — still geometric `(2/L, 1/L^2)` rather than derived from the identified plant,
which is where the saturation lives. Fix staleness first, because the gains multiply the error signal
that staleness corrupts. **The feedforward is finished and is not one of the issues.**

---

## ⚠MEASUREMENT HYGIENE — a window of today's data is unusable for any of this

Between **12:55 and ~21:50 on 2026-08-16** room_concept ran with `OdomInject*` armed in
`etc/config_apartamento.toml` (`951d4bd`) — a deliberate +10% velocity-scale, −6% rotation-scale and
injected noise, for a calibrator recovery test. It is off now. **Any run in that window measures the
injected system, not the plant.** Both issues here are identification problems — Issue 1 wants staleness
measured, Issue 2 wants gains derived against the *identified* plant — so numbers from that window fit
the model to a fault that has since been switched off. The `pose_stamp_age` figures above were taken at
14:15 and 20:20 and are therefore inside it; they are quoted because staleness is a pipeline property
and largely independent of localisation quality, but **re-take them before committing**. This is the
same trap the sequencing note names for Issue 2, one level further upstream: it is not enough for the
error signal to be un-corrupted by staleness if the estimator producing it was being degraded on purpose.

---

*Cross-references: thesis ch.11 §Route generation and optimisation and §Following (both updated
2026-08-16); memory `frenet-feedforward-tracker`, `tracker-bibliography-review`,
`controller-tracking-resume-2026-08-13`, `pose-sigma-throttle-deadlock-in-curves`.*
