# Door work queue items 1+2 — an opened door that never reads passable, and a passage that stops acting

Date 2026-11-05. Branch base: main = aff02fde. Scope: door_concept (P(passable)) and controller (door passage).
Evidence: warm59 and warm61 leg 3 (backward, room_2 → room_1 through DOOR_1, face `door_2`), both FAILED at 600 s.

## Why

After the relocaliser fix, the door is the only remaining cause of tour failures. Both failed legs end the same way, with the
robot standing still in front of an open door for nine minutes. They get there by two different mechanisms; neither is the
leaf-angle filter.

## What the logs show

### warm59 — P(passable) is pinned at 0.54 by the aperture WIDTH's reported σ, not by the leaf

- `open` was accepted (door_concept.log l.4020) and the leaf posterior followed it: ai2_log_phi.csv door_2 post_mean
  2.09 rad, post_sigma 0.012 rad, for the rest of the leg. The leaf channel is right.
- P(passable) (`door_open_prob`) rose with the commanded swing and then sat at **0.53–0.54 for nine minutes**, so `cross` (offer =
  P(passable) × P(exists), assert above 0.60) was never offered. The controller logged "'aff_door_2_cross' is not in the graph".
- P(passable) is `passable_logit(phi_curve_log, w, needed, σ_w)` (specificworker.cpp:2340–2360), with
  σ_w = sqrt(covariance_reported()(1,1)) = sqrt(Σ_ww + model_var_w_). Under each leaf angle it is
  Φ((w·(1−cos φ) − needed) / (σ_w·(1−cos φ))). With φ = 2.09, w = 1.00 and needed = 0.70, P = 0.54 requires **σ_w ≈ 5.3 m**.
  As σ_w → ∞ the expression tends to Φ(0⁺) = 0.5⁺ at **every** angle. That is the plateau: the leaf can be anywhere, and the
  door can never read better than a coin flip.
- `model_var_w_` (door_belief.cpp:174–206, Task E "honest uncertainty") is the squared Kalman move one mask frame's width
  residual would make against the template prior: m = Σ J r / (J²Σ + R), whose maximum over J is σ_prior·r/(2√R). With
  σ_prior = 0.06 m, R_w = 2·mb.var (the sides' pixel noise, ~1 px²) and r_w = hull width − mask width, a residual of ~170 px
  gives a 5 m "error". The hull it compares against (`predicted_column_extent`) includes the leaf's corners at the believed φ.
  When the door is open and the robot is 1.3 m away, the mask (a segmented doorway or leaf, possibly clipped by the image edge)
  and that hull disagree by hundreds of pixels for reasons that have nothing to do with the width.
- ⚠ Not yet measured directly: σ_w, r_w, J_w and R_w are not logged. The arithmetic match is exact (the plateau is the σ_w → ∞
  limit, and the only other input is a tight φ posterior), but step 0 below confirms it before anything is changed.

### warm61 — the passage deadlocks between "servo refused" and "route arrived"

- P(passable) 1.00, `cross` OFFERED and claimed (door_concept l.3711/3716), φ 1.39 → 2.09.
- The controller at the vantage: "not in front of an open door (… e −0.46 m of ±0.44 ✗ …) — the route keeps driving"
  (controller_session.cpp:3790). The anchor's perceived opening is 0.46 m lateral of the robot, against the room-frame door
  the route was built on (anchor σ_lat 0.37 m at hang time: model term 0.30 m).
- The route had already ARRIVED (its waypoints are on the room-frame door, which it believes it is aligned with), so nothing
  moves the body sideways; the servo will not start until |e| < w/2. No further event is logged for nine minutes. The "withdrawn
  by door_concept" in the why-line is stale (door_concept never withdrew it).
- This is queue item 2 ("between 'reads shut' and 'cross offered' it waits to timeout") in its concrete form. The stop is not a
  wait on the door's state but two rules that each hand the decision to the other.

### warm37 (queue item 1's other symptom) — set aside

"Shut door read open with ndet=0" (door_16 60→87°): ndet there is the existence channel's count, while the leaf channel had
700–1600 LiDAR rays per frame. The run predates the 11-02 leaf fixes. It is re-checked on the new code (a tour or a scan-tape
replay) before anything is built for it. No change is proposed for it now.

## Goals

1. An opened, clearly-seen door reads passable. P(passable) is limited by what is actually uncertain (the leaf angle, the
   width), and the width's uncertainty comes from the width.
2. At the vantage of an open door, the passage always has an action that reduces the reason it is not crossing. It never
   holds still with both sides waiting on each other.
3. warm59/61's leg 3 completes. The fwd + bwd three-room tour passes with no manual help (warm64+).

## Non-goals / anti-goals

- No new thresholds: no σ_w clamp, no `if (open) ignore model error`, no wider lateral tolerance. (CLAUDE.md: if one is
  unavoidable, flag it.)
- No change to the leaf channel (door_leaf_channel.h). It was right in both runs.
- No reverting Task E: a mask that really disagrees with the WIDTH must still lower P(passable).
- No new crossing chain: C1 (local policy drives the passage) stands. Fix the hand-off, don't add a stage.

## The design

### Step 0 — measure (cheap, first)

Log σ_w (Σ_ww and model_var_w_ separately), r_w, J_w, R_w, and whether either mask side touches the image border, both into
ai2_log.csv (new columns) and into the cross/approach precondition sentences. Confirm σ_w ≫ 1 m on a parked open-door case
before step 1. (Webots needed: a short parked recording, HOLD=1. Warn the peer session first.)

### Step 1 — the width's model error is attributed only to what can explain it (door_concept)

The Task E error is the mask's disagreement with the hull. That disagreement has several explanations (width, along-wall
offset, leaf angle, the image edge censoring a side), and today all of it is charged to w. The model-level fix, in order of
preference (step 0 says which applies):

- (a) **Censoring is part of the observation model.** A mask side at the image border says only "the edge is at or beyond
  here". The width residual of a censored side is not an observation of w, so no width error comes from it (the existing
  `hull_mismatch_var` doc already states this rule for s; `note_model_error` does not apply it to w).
- (b) **The residual is taken under the leaf posterior, not the leaf MAP.** The hull includes the leaf corners at the believed
  φ. The width move is computed on the part of the residual not explained by the leaf: the expected hull extent over the φ
  posterior and its variance go into the innovation covariance S (S = J²Σ + R + Var_φ[hull width]), so a leaf-dominated
  residual is absorbed by the leaf term, continuously.

**★ Revised while planning (code read, door_fit.cpp:776–815): (a) is ALREADY in place.** A side within 6 px of the border is
set to NaN and `note_model_error` returns on NaN. The real inconsistency is a third one:
- (c) **The width error must use the same generative model of a mask as the rest of the fit.** door_fit.cpp (task W) models a
  mask as "a contiguous PART of the hull, plus pixel noise", so a mask NARROWER than the hull is expected and is already
  charged to the midpoint (ΔW²/12). `note_model_error` charges the same shortfall to w a second time, as if the mask were the
  whole hull. Under the part-of-hull model, the width is contradicted only by a mask WIDER than the hull. The "move" is then
  computed from the part of r_w that the model cannot explain: zero for r_w ≥ 0 (mask inside the hull), and the full
  r_w < 0 (mask beyond the hull). That is a one-sided likelihood (the same shape as censoring), not a gate. Its innovation
  variance R_w must also include the midpoint's part-of-hull variance it already carries (var_common), not only 2·var.
  This also explains the rarity: the hull is the jambs plus the believed leaf's free edge, and only a wide-open leaf seen up
  close makes the hull hundreds of pixels wider than the doorway mask. Step 0 confirms r_w ≫ 0 (hull wider) in the plateau.
  (b) is dropped unless step 0 contradicts (c).
- ★★ (c) REJECTED while planning: it breaks test_door_look_vs_approach 1b, which requires that a door genuinely narrower than the
  template (consistent full masks, narrower than the hull) still raises σ_w. Under (c) that door could never report its
  error. The two readings of "mask narrower than hull" are only told apart across views, which is out of proportion here.
- (d) **THE CHOSEN FIX: the predicted hull contains only what the camera can SEE.** `predicted_column_extent` takes the
  jambs AND the believed leaf's free edge. In warm59 the leaf stands at 120° on the FAR side of the wall (opens out of this
  room) and the camera is 1.3 m in front. The free edge is then ~0.5 m behind the hinge along the wall and ~0.87 m beyond
  the wall plane, so the wall hides it from the camera. The hull predicts a mask over a hidden edge, and the shortfall is
  charged to w. Fix: a hull corner contributes to the predicted extent only where its line of sight from the camera passes
  through the aperture (or the corner is on the camera's side of the wall). A hidden corner is clipped to the aperture edge
  its ray is occluded by. This is geometry in the generative model, not a gate. A shut leaf or a leaf on the camera's side
  is unchanged, so 1b and the Task W tests keep their meaning. It also explains the rarity: a leaf >90° open on the far
  side, seen up close. The same visibility applies to `hull_mismatch_var` (it calls the same function).
  Step 0 confirms it: log r_w AND r_w with the leaf corners removed (jambs only). If the jambs-only residual is ≈ 0 while
  r_w ≫ 0, (d) is confirmed.

These are terms in the model, not gates. Expected effect: an open door seen from the vantage reports σ_w on the
order of Σ_ww (centimetres) and P(passable) → 1 under a φ posterior at 2.09.

### Step 2 — at the vantage the passage acts on the reason it is not crossing (controller)

The servo's start conditions (open belief, seen, near side, within, facing) are kept. Each failing check maps to the action
that reduces it, so there is never a state where nothing is commanded:

| failing check                | action (existing machinery)                                                                                   |
|------------------------------|---------------------------------------------------------------------------------------------------------------|
| within (|e| ≥ w/2)          | the local policy's Place goal = the point on the ANCHOR's axis at the vantage standoff (re-aim on the perceived opening) |
| facing                       | turn to the anchor's axis (Place goal heading)                                                                |
| seen / open belief           | unchanged: hold, `open` re-offered or `cross` withdrawn (warm24 rule "no mouth, no crossing")                 |
| near side                    | unchanged (servo keeps the hole once engaged)                                                                 |

This replaces "the route keeps driving" for the case where the route has already arrived. The re-aim uses the anchor's
opening, the same thing the servo will drive on, so success makes `within` true by construction. The re-aim's goal is
recomputed each cycle from the anchor; if the anchor and the room-frame door disagree by more than the corridor, that
disagreement is visible in the log line, not hidden in a hold. The why-line must also stop calling a claimed, still-offered
affordance "withdrawn".

### Validation

1. Bench/unit: door_concept test for step 1 (synthetic open-door mask with a censored side and with a leaf-dominated hull:
   σ_w stays ≈ Σ_ww; a real width disagreement still raises it). Controller test for step 2 (anchor e = −0.46, route
   arrived ⇒ a Place goal is commanded, never a stop).
2. Live: warm64 fwd + warm65 bwd (`LEGS="room_3 room_2 room_1"`), then two more each. Pass = all legs, no manual help, no
   9-minute stall; door_open_prob after an `open` ≥ 0.95 within 5 s of the leaf reaching its commanded angle.

## Calls made after the builder check (mine, not the user's — check them)

- **Step 0 is a hard gate.** Nothing in step 1 is built before it has run. Decision rule (revised twice): if the plateau
  frames show r_w ≫ 0 while the jambs-only residual is ≈ 0, build (d). Otherwise stop and come back to the user
  with the numbers. Step 2 does not depend on step 0 and may be built
  in parallel.
- **(b) as written is probably weak.** σ_φ is 0.012 rad in warm59, so Var_φ[hull width] is tiny. If step 0 points at the leaf,
  the likely real term is the leaf's VISIBILITY (a leaf swung behind the wall plane, away from the camera, is not in the
  mask, so its corners must not be in the predicted hull). That is a model fix to `predicted_column_extent`, still not a
  gate. The builder brings the measured case back before choosing.
- **The re-aim goal comes from the anchor's own filtered pose** (it is already a filter with σ_lat). It is not re-filtered,
  and no deadband is added. The goal is the anchor-axis point at the vantage standoff with heading = the axis. It owns the
  goal slot only while the route has arrived and the servo's start check fails on `within`/`facing`. It ends the cycle the
  servo starts (the existing branch already drops the route). The route is not rebuilt (the "built ONCE per leg" rule stands).
- **Facing** keeps the servo's existing check (|heading error| < 90°). The re-aim simply commands the axis heading.
- **The why-line** reads the affordance node's presence and its EXECUTING edge. It says "claimed, waiting for …" when the
  node exists, and "withdrawn" only when it is gone.
- **Validation split:** unit/bench tests offline; door crossing is live-only (the scan tapes carry no masks). The
  "≥ 0.95 within 5 s" criterion is timed from the φ posterior mean first coming within 0.1 rad of the commanded angle.

## Left to the builder

Column names; where the re-aim goal lives (controller_session.cpp around l.3790 or door_passage); the exact
(a)/(b) form once step 0 has measured which explanation dominates (both may be needed).

## Open

- warm37's shut-read-open: re-checked on the new code, as above.
- Queue items 3–5 are not part of this spec.
